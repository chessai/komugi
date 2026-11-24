"""One self-play worker: play games, record one row per move, write shards.

The loop of design D6, with the pieces it needs coming from three places --
the engine adapter (:mod:`komugi.selfplay.adapter`), the record schema and seed
derivation (:mod:`komugi.selfplay.records`), and whichever end of the inference
seam this process has (:mod:`komugi.selfplay.inference`).

What one move costs, and what it produces
-----------------------------------------
Per move the worker:

1. Draws **full vs fast** from ``SelfPlayConfig.prob_fast`` on the game's own
   random stream (design D6/F1) and records which it drew in ``search_mode``.
   ``DESIGN.html`` §3.1's reduced Gumbel scale for fast mode **has a knob** --
   :attr:`~komugi.search.gumbel.GumbelConfig.gumbel_scale`, shipped by
   KOM-029 -- and this worker now passes it, scheduled by
   :meth:`SelfPlayWorker._scale_for` from two
   :class:`~komugi.config.SelfPlayConfig` fields: ``fast_gumbel_scale`` while
   the search is fast, ``draft_gumbel_scale`` while **the mover** is drafting,
   composed multiplicatively (KOM-036 §2, §3). Both default to ``1.0``, so the
   composed scale is exactly ``1.0`` and every existing golden, replay and
   seed-discipline claim is bit-identical until an operator sets one.
2. Builds a **fresh** :class:`~komugi.search.gumbel.GumbelConfig` with
   ``seed=move_seed(game_seed, ply)``. Fresh per move because the seed is what
   changes: one config reused across a game would replay one Gumbel vector for
   every ply of the game (design F10d).
3. Runs the search, plays :attr:`~komugi.search.gumbel.SearchResult.action`,
   and buffers the row with everything except the two fields that are only
   knowable at the end. ``policy_kl`` (design D4/F2) is computed against
   :attr:`~komugi.search.gumbel.SearchResult.root_prior_logits`, i.e. the root
   prior the search itself used.

There used to be a fourth step, and deleting it is KOM-040 design review D9':
the worker evaluated the root **once more, before the search**, because the
search did not report the root prior it had used. That was a ``+1`` evaluation
on every move of every game (a few percent of a 32-simulation fast search, well
under one percent of a 256-simulation full one) spent re-asking a question that
had just been answered. The search now hands the prior back on its result, so
the call is gone; the other alternative -- storing the ``(A,)`` prior in the row
-- was and remains 150KB per sample, and is not what happened here.

Termination is the **engine's** decision and only the engine's, the engine's
turn cap included: there is no ply cap in this module (design D6/F15, Q3). When
``win_state()`` fires, the buffered rows are finalised -- ``wdl_target`` flipped
into each row's own mover's perspective, ``mlh_target`` counted back from the
final ply, ``final_control`` read off the final board once and flipped the same
way (KOM-010) -- and appended to the writer.

Two games at once (KOM-040 b1)
------------------------------
A worker spends a measured 57% of its wall blocked in
:meth:`~komugi.selfplay.inference.SlotClient.submit` -- 9.16 ms per call,
~87.6 calls per move -- sleeping while the server batches and forwards. With
``SelfPlayConfig.games_in_flight == 2`` the process plays two games in two
threads, so the sleep of one is the compute window of the other (``time.sleep``
releases the GIL). What makes that safe is that the lanes share **nothing**:
each has its own segment, its own :class:`~komugi.selfplay.inference.SlotClient`,
its own :class:`~komugi.train.shards.ShardWriter` over its own directory, and
its own game-index range (:func:`lane_game_index_base`). The seam sees two
ordinary clients, not one client used twice. Default 1, which is the code path
this module had before the knob existed, unchanged.

Waves (KOM-040 b2)
------------------
The other half of that 57%: a leaf-at-a-time search pays one shared-memory
round trip per leaf, and the b1 bench showed the shared batch already
synchronises the lanes, so what is left to cut is the *count* of round trips.
Both halves of the machinery exist below this module -- the transport
(:meth:`~komugi.selfplay.inference.SlotClient.submit_wave` behind
:func:`~komugi.selfplay.adapter.make_wave_evaluator`) and the schedule
(:class:`~komugi.search.gumbel.GumbelSearch` with a ``wave_evaluator``) -- and
this module is where they meet.

Waves are on **iff** ``InferenceConfig.max_wave > 1``, which is the convention
:func:`~komugi.selfplay.orchestrate.validate_wave_fit` already documents and
enforces (it refuses a ``max_wave`` below the search's ``m``, but only when
waves are enabled). There is deliberately no second knob: a run whose segments
are sized for one row per ticket cannot submit a wave, and a run whose segments
are sized for ``m`` rows has nothing to gain by not using them, so a config that
could say "wide slots, scalar submits" would only describe a way to waste the
memory. See :func:`_wave_evaluator_for`, which is the single place the decision
is made, for both lanes and for the single-game path.

``--serial-debug`` never waves (:func:`~komugi.selfplay.orchestrate.serial_debug`
builds a :class:`SelfPlayWorker` without one, and the parameter defaults to
``None``): that path predates this seam, holds an
:class:`~komugi.selfplay.inference.InProcessEvaluator` rather than a
:class:`~komugi.selfplay.inference.SlotClient` -- there is no ticket to widen --
and is the bit-identical reference the replay claim below rests on.

Reproducibility, honestly (design D11)
--------------------------------------
Everything random in a game derives from ``(run_seed, worker_id, incarnation,
game_index)``. In ``--serial-debug`` (one worker, in-process evaluator) two
runs of the same ``game_index`` produce **identical rows**, which is asserted
by the unit suite. In production the evaluator batches across games, so
floating-point results depend on batch composition and replay is promised only
up to evaluator tolerance -- not bit-exactly.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.game import TERMINATION_TURN_CAP, Game
from komugi.engine.observation import CONTROL_EMPTY, PLANE_MAP_VERSION
from komugi.modes import MODE_FAST as MODE_FAST
from komugi.modes import MODE_FULL as MODE_FULL
from komugi.search.gumbel import GumbelConfig, GumbelSearch
from komugi.search.state import AnyWaveEvaluator
from komugi.search.tactical import TacticalProvider
from komugi.selfplay.adapter import (
    ControlVector,
    GameState,
    MlhUtility,
    OutcomeVector,
    SearchContext,
    WaveSeam,
    make_evaluator,
    make_wave_evaluator,
    require_shared_context,
)
from komugi.selfplay.book import Book, BookEntry, load_book, select_book_entry
from komugi.selfplay.inference import (
    BatchEvaluator,
    InferenceConfig,
    SlotClient,
)
from komugi.selfplay.records import (
    MLH_SCALE,
    SCHEMA,
    SHORT_TERM_HORIZONS,
    draft_thresholds,
    game_seed,
    move_seed,
    num_actions,
)
from komugi.selfplay.seed_openings import (
    SeedChoice,
    SeedEntry,
    load_seed_openings,
    select_seed_entry,
)
from komugi.selfplay.tactical import GameMarshalCaptureProvider
from komugi.shapecheck import checked_seam
from komugi.terminations import TERMINATION_DECISIVE as TERMINATION_DECISIVE
from komugi.terminations import TERMINATION_DRAW as TERMINATION_DRAW
from komugi.terminations import TERMINATION_RESIGN as TERMINATION_RESIGN
from komugi.train.shards import ShardWriter

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
"""Where a seeded game announces itself (KOM-254 AM7).

One line per seeded game and nothing else: the mechanism records no column, so
the log is the only place a *running* worker says "this game started from a
harvested prefix", and the flip-on guard reads the seeded share off it.

Emitted at **WARNING**, which is a claim about this repository rather than
about the line's severity (impl review F1): nothing in ``src/`` or ``scripts/``
calls ``basicConfig``/``dictConfig``, so the effective level is the root
logger's default WARNING and ``logging.lastResort`` is WARNING too -- an INFO
record is dropped before any handler, and the guard would read 0% whether the
curriculum was live or silently seeding nothing. At the flip-on share the
volume is ~4% of a worker's games, which is a handful of lines per iteration."""

_SEED_SHA_PREFIX: Final[int] = 12
"""Hex characters of the seed file's SHA-256 carried on that line. Enough to
name one harvest window against another; the loader holds the whole digest."""

_TACTICAL_PROVIDER_ID: Final[str] = "game-marshal-capture-v1"
"""The one tactical generator self-play is allowed to discover with (KOM-220)."""

_DRAW_INDEX: Final[int] = 2
"""Draw slot in the engine's absolute ``[white, black, draw]`` outcome."""

_NO_LEADER: Final[int] = -1
""""Nobody is ahead" in KOM-245's cap adjudication -- not a colour.

A third value rather than a ``None`` beside a colour, and deliberately not
``0``: ``0`` is *white*, and a level cap game labelled as white's would be the
exact bug this constant exists to make unspellable."""

# MODE_FULL/MODE_FAST are RE-EXPORTED from :mod:`komugi.modes`, not defined
# here (KOM-036 §1.2a). The trainer needs the same two codes to weight a fast
# row, and importing them from this module would close a genuine cycle:
# `komugi.train.wiring -> komugi.selfplay.worker -> komugi.selfplay.records`,
# with `records` frozen at its own `from komugi.train.shards import ...` when
# the entry point was `import komugi.selfplay`. The leaf sits at the *package
# root* rather than under `komugi/selfplay/` because Python draws the edge at
# package granularity. `worker.MODE_FULL` keeps working, so every existing
# self-play reference -- this module's write site below, and the unit suite's
# `from komugi.selfplay.worker import MODE_FAST, MODE_FULL` -- is untouched.

# TERMINATION_DECISIVE/TERMINATION_DRAW/TERMINATION_RESIGN are RE-EXPORTED from
# :mod:`komugi.terminations` (KOM-160), not defined here, for exactly the reason
# the block above gives for the mode codes: the trainer keys a per-row WDL
# weight on the `termination` column and cannot import this module to learn its
# vocabulary. `worker.TERMINATION_DRAW` keeps working, so every existing
# reference -- `finalize_rows` below, `komugi.campaign.game_stats`,
# `scripts/analysis/cap_game_values.py`, the self-play unit suite -- is
# untouched, and there is still exactly one place the three codes are stated.

_DRAFT_PLANE_WHITE: Final[int] = 112
_DRAFT_PLANE_BLACK: Final[int] = 113
"""The encoder's drafting-rights planes -- "1.0 if <colour> is drafting".

The draft predicate is read from the observation this loop has already
computed, rather than from the engine, and that is forced rather than chosen:
:class:`komugi.engine.game.Game` -- the class this worker plays with -- exposes
no drafting accessor at all. ``komugi.engine.api.Game.in_draft()`` does
(``self._drafting["b"] or self._drafting["w"]``), but that is the JS-compatible
API class and not the one on this path.

These two planes are written from exactly the bits that predicate reads
(``observation.py`` fills 112 from ``board.draft_w`` and 113 from
``board.draft_b``), so the derived answer is the same answer, at zero extra
cost -- no second engine call on a loop whose whole point is to spend its time
in the search. A remap of the plane indices is the change this would not
survive, and it is precisely the change
:data:`~komugi.engine.observation.PLANE_MAP_VERSION` exists to make loud: every
row stamps the version it was encoded under and the reader refuses a
disagreement.
"""

_GAME_ID_STRIDE: Final[int] = 1_000_000
"""Digit room per component of a packed ``game_id`` (see :func:`game_id`)."""


def game_id(worker_id: int, incarnation: int, game_index: int) -> int:
    """Return the ``game_id`` recorded on every row of one game.

    ``(worker_id * S + incarnation) * S + game_index`` with ``S = 1e6``: a
    readable packing rather than a hash, so a row's provenance can be read off
    the number by eye and, more importantly, so ``game_id`` is a pure function
    of the same triple :func:`~komugi.selfplay.records.game_seed` derives from.
    Two runs of one ``game_index`` therefore agree on this field too, which is
    what lets the serial-debug replay check compare rows for exact equality.

    The packing is unique for the run sizes P0a can produce (design D9 sizes
    worker counts in the tens) and stays inside ``int64`` for worker ids up to
    ~9e6. The bounds are ENFORCED, not assumed: a component at or above the
    stride would silently collide with a neighbouring worker/incarnation's ids
    (wave-9b review F5).
    """
    for name, value in (
        ("worker_id", worker_id),
        ("incarnation", incarnation),
        ("game_index", game_index),
    ):
        if not 0 <= value < _GAME_ID_STRIDE:
            raise ValueError(
                f"{name} {value} is outside [0, {_GAME_ID_STRIDE}); the game_id "
                f"packing would collide with a neighbouring stream"
            )
    return (worker_id * _GAME_ID_STRIDE + incarnation) * _GAME_ID_STRIDE + game_index


def unpack_game_id(value: int) -> tuple[int, int, int]:
    """Return the ``(worker_id, incarnation, game_index)`` :func:`game_id` packed.

    The exact inverse, and it exists because the packing is *readable* rather
    than a hash (see :func:`game_id`): an offline audit holding only a shard's
    ``game_id`` column can recover the spawn key every seed of that game derives
    from, and therefore recompute what the game's own streams decided --
    KOM-014 §6.7, which is what makes "which book entry did this game use"
    answerable from artefacts rather than from a record field nobody wrote.

    Args:
        value: A ``game_id`` as :func:`game_id` produced it.

    Returns:
        The triple, each component in ``[0, 1e6)``.

    Raises:
        ValueError: On a negative value, which no packing can produce and which
            would otherwise invert into negative components that name a game
            that never ran.
    """
    if value < 0:
        raise ValueError(f"game_id must be >= 0, got {value!r}")
    rest, game_index = divmod(value, _GAME_ID_STRIDE)
    worker_id, incarnation = divmod(rest, _GAME_ID_STRIDE)
    return (worker_id, incarnation, game_index)


_LANE_INDEX_STRIDE: Final[int] = 100_000
"""Game-index room reserved for one interleaved lane (KOM-040 b1).

Lane ``k`` of an interleaved worker plays game indices ``(k + 1) * STRIDE + i``
rather than ``i``. That single offset is the whole of design D1' (C4)'s seed
obligation, because ``game_index`` is a component of BOTH
:func:`~komugi.selfplay.records.game_seed`'s spawn key and :func:`game_id`'s
packing: disjoint index ranges are disjoint seed streams and disjoint game ids,
derived rather than asserted.

The ``+ 1`` is the part that is easy to miss and load-bearing: it keeps lane 0
off index ``0`` too, so an interleaved run cannot replay the games a
``games_in_flight=1`` run of the same ``run_seed`` played. Without it lane 0
*is* the single-game worker, and merging the two runs' shards into one window
would duplicate positions while looking like twice the data.

100_000 rather than something larger: the lane bases plus a lane's budget must
stay inside :data:`_GAME_ID_STRIDE` for the ``game_id`` packing to remain
collision-free. An interleaved lane's budget is CHECKED against the stride
(:func:`lane_game_index_base`) rather than assumed small; the matching claim
about a ``games_in_flight=1`` run -- that its indices ``0..games-1`` never
reach a lane base -- holds for any per-worker budget under 100_000 games,
which is three orders of magnitude past the tens design D9 sizes.
"""


def lane_game_index_base(lane: int, games: int) -> int:
    """Return the first game index lane `lane` plays (KOM-040 b1, design D1'/C4).

    ``(lane + 1) * _LANE_INDEX_STRIDE`` -- see that constant for why the ranges
    are offset at all and why lane 0 is offset too. Called only on the
    interleaved path: a ``games_in_flight=1`` worker keeps base ``0`` and is
    therefore bit-identical to what it was before this knob existed.

    Args:
        lane: Zero-based lane index within the worker process.
        games: How many games that lane plays -- checked here, because a budget
            that overran the stride would put lane 0's last games on top of
            lane 1's first ones, silently replaying them.

    Raises:
        ValueError: On a negative lane, or a budget that would overrun the
            lane's reserved index range.
    """
    if lane < 0:
        raise ValueError(f"lane must be >= 0, got {lane!r}")
    if not 0 <= games <= _LANE_INDEX_STRIDE:
        raise ValueError(
            f"a lane's game budget must be in [0, {_LANE_INDEX_STRIDE}], got "
            f"{games!r}: a longer budget would run one lane's game indices into "
            f"the next lane's reserved range and replay its seeds"
        )
    return (lane + 1) * _LANE_INDEX_STRIDE


def lane_slot_id(worker_id: int, games_in_flight: int, lane: int) -> int:
    """Return the id the **server** knows one worker lane's segment by.

    ``worker_id * games_in_flight + lane``. The server keys its slot map by
    this number (:class:`~komugi.selfplay.inference.SlotHandle`), so every
    segment in a shard needs a distinct one and a worker's two lanes cannot
    both present ``worker_id``. It is a *slot* identity and nothing else --
    seeds, ``game_id`` s and shard directories all keep deriving from the real
    ``worker_id`` (:func:`lane_game_index_base` is what keeps the lanes apart
    there), so a run's data provenance does not change shape when the knob is
    flipped. With one game in flight it is the identity function, which is why
    a single-game worker's slot is still numbered by its worker id.

    It lives here, beside :class:`LaneSpec`, rather than in the orchestrator
    that hands the numbers out: the worker names its own lane 0 with it, and
    two copies of this arithmetic disagreeing would put a client's diagnostics
    on one slot while the server logged another.
    """
    return worker_id * games_in_flight + lane


def _masked_softmax(
    logits: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Return ``softmax(logits)`` over the legal actions, zero elsewhere.

    A local copy of the search's own private helper rather than an import of
    it: :func:`komugi.search.gumbel._masked_softmax` is private precisely so
    the search can change it, and the prior this file needs is fixed by design
    D4 ("masked-softmax root prior") independently of the search's internals.
    Illegal entries are dropped before the max, so a network's logits on
    illegal actions can be any finite value without shifting the result.
    """
    scores = np.asarray(logits, dtype=np.float64)
    out = np.zeros(scores.shape[0], dtype=np.float64)
    if not bool(legal.any()):
        return out
    shifted = np.where(legal, scores - float(scores[legal].max()), -np.inf)
    exp = np.where(legal, np.exp(shifted), 0.0)
    return np.asarray(exp / float(exp.sum()), dtype=np.float64)


def policy_kl(
    policy_target: npt.NDArray[np.float32],
    prior_logits: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
) -> float:
    """Return ``KL(policy_target || masked-softmax(prior_logits))``, in nats.

    ``DESIGN.html`` §4's surprise weight, computed at record time from the two
    distributions the worker already holds (design D4/F2): how far the search
    moved the network's own root prior. Zero means the search confirmed the
    prior; large means this position taught the network something.

    Computed in ``float64`` over the legal actions only -- the summand is zero
    wherever ``policy_target`` is, and the prior is zero exactly on the illegal
    actions, so restricting the sum is the definition rather than an
    approximation. The prior is floored at the smallest positive ``float64``
    before the logarithm: a legal action whose prior underflows to zero would
    otherwise give an infinite KL for a row that is merely very surprising.

    Args:
        policy_target: ``(A,)`` search target; nonzero on every legal action.
        prior_logits: ``(A,)`` raw root logits from the network.
        legal: ``(A,)`` legality mask at the root.

    Returns:
        The divergence in nats, ``>= 0`` up to floating-point noise.
    """
    target = np.asarray(policy_target, dtype=np.float64)
    prior = _masked_softmax(prior_logits, legal)
    support = legal & (target > 0.0)
    if not bool(support.any()):
        return 0.0
    tiny = float(np.finfo(np.float64).tiny)
    ratio = np.log(target[support]) - np.log(np.maximum(prior[support], tiny))
    return float(np.sum(target[support] * ratio))


def mover_relative_control(
    absolute: npt.NDArray[np.int8], to_play: int
) -> npt.NDArray[np.int8]:
    """Re-express an absolute control map for the player who moved in a row.

    ``absolute`` is :func:`komugi.engine.observation.final_control`'s frame --
    ``0`` white, ``1`` black, ``2`` empty. The result is the row's mover's:
    ``0`` where `to_play` controls the square, ``1`` where the opponent does,
    ``2`` where it is empty in both frames (an empty square belongs to nobody,
    so it is the one value the flip must leave alone -- mapping it through the
    colour comparison would label every empty square "the opponent's").

    Keyed off `to_play`, never off ply parity, for the reason
    :func:`finalize_rows` gives about ``wdl_target``: the draft lets one side
    move twice, so the movers of consecutive rows need not alternate (wave-9b
    review F7).

    Args:
        absolute: ``(81,)`` map in the engine's colour frame.
        to_play: ``0`` white or ``1`` black -- the row's own mover.

    Returns:
        A new ``(81,)`` ``int8`` map; `absolute` is not modified.
    """
    return np.where(
        absolute == CONTROL_EMPTY,
        np.int8(CONTROL_EMPTY),
        (absolute != np.int8(to_play)).astype(np.int8),
    ).astype(np.int8)


@dataclass(frozen=True, slots=True)
class PendingRow:
    """One recorded move, missing only what the game's end will decide.

    ``wdl_target``, ``mlh_target``, ``final_control``, ``termination``,
    ``short_term_value`` and ``draft_outcome`` are absent by construction: all
    six are functions of the final outcome, the final ply count, the final
    board or a row that had not been played yet, so a row cannot be complete
    until the game is. Everything else is fixed at the moment the move was
    chosen -- ``action`` included: it is the move the search returned and the
    loop then played (KOM-014 §3.1/B2), not something the game's end decides.

    ``in_draft`` is a field of the **buffer**, not of the schema: what reaches
    a shard is the masked ``draft_outcome`` vector, and this flag is what
    :func:`finalize_rows` gates it on.
    """

    observation: npt.NDArray[np.float32]
    policy_target: npt.NDArray[np.float32]
    policy_kl: float
    root_value: float
    search_mode: int
    to_play: int
    ply: int
    sims_used: int
    action: int
    in_draft: bool


CAP_ADJUDICATION_OFF_MODE: Final[str] = "off"
CAP_ADJUDICATION_MATERIAL_MODE: Final[str] = "material"
_CAP_ADJUDICATION_MODES: Final[frozenset[str]] = frozenset(
    (CAP_ADJUDICATION_OFF_MODE, CAP_ADJUDICATION_MATERIAL_MODE)
)


@dataclass(frozen=True, slots=True)
class CapAdjudication:
    """The three numbers KOM-245's soft cap label is built from.

    Frozen and tiny, and deliberately **not** a view onto
    :class:`~komugi.config.SelfPlayConfig`, for the reason
    :class:`~komugi.selfplay.adapter.MlhUtility` gives: the rule is arithmetic
    over three scalars and a unit test that wants to sweep ``p_max`` should not
    have to build a whole self-play config with a worker count in it.
    :meth:`from_selfplay` is the one place production reads the config.

    Attributes:
        mode: ``"off"`` -- a capped draw is labelled ``(0, 1, 0)``, which is
            every campaign before the switch -- or ``"material"``, the rule
            below.  Any other value is refused here rather than silently read
            as off.
        p_max: The ceiling of the ramp: the win probability a side ``d_full``
            pieces up is labelled with.  A mate is worth ``1.0``, so this is
            below it by design -- pressing to win must beat coasting to the cap.
        d_full: Where the ramp reaches ``p_max``, in pieces of TOTAL material
            (board at any tier plus hand).
    """

    mode: str
    p_max: float
    d_full: int

    def __post_init__(self) -> None:
        """Refuse a mode this module has no rule for.

        A typo'd mode is the failure this catches: read as "not material" it
        would silently keep the old labels for a campaign that was switched,
        and the switch is a pre-registered event whose whole value is that the
        rows changed.
        """
        if self.mode not in _CAP_ADJUDICATION_MODES:
            raise ValueError(
                f"cap adjudication mode must be one of "
                f"{sorted(_CAP_ADJUDICATION_MODES)}, got {self.mode!r}"
            )

    @classmethod
    def from_selfplay(cls, config: SelfPlayConfig) -> CapAdjudication:
        """Return the rule's parameters as this self-play config states them.

        The config validates the mode, ``p_max`` in ``(0, 1]`` and ``d_full``
        in ``[1, 25]`` in its own ``__post_init__`` (KOM-249), so nothing is
        re-bounded here: a second copy of those bounds would be a second thing
        to keep in step.
        """
        return cls(
            mode=str(config.cap_adjudication),
            p_max=float(config.cap_adjudication_p_max),
            d_full=int(config.cap_adjudication_d_full),
        )


CAP_ADJUDICATION_OFF: Final[CapAdjudication] = CapAdjudication(
    mode=CAP_ADJUDICATION_OFF_MODE, p_max=0.8, d_full=8
)
"""The default every caller that has not been told otherwise passes.

``p_max``/``d_full`` carry the pre-registered values rather than zeros so the
constant is a *switched-off rule* and not a differently-parameterised one: at
``mode == "off"`` neither number is read.
"""


@checked_seam()
def finalize_rows(
    rows: Sequence[PendingRow],
    outcome: OutcomeVector,
    final_control: ControlVector,
    total_plies: int,
    game_id_value: int,
    *,
    cap_adjudication: CapAdjudication = CAP_ADJUDICATION_OFF,
    final_material: tuple[int, int] | None = None,
    capped: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Turn buffered moves into schema-shaped samples, once the game is over.

    The record seam of design D7: ``@jaxtyped`` when ``KOMUGI_TYPECHECK=1``, so
    an ``outcome`` that is not three ``uint8`` fails here rather than being
    written into a shard.

    The six end-of-game fields:

    * ``wdl_target`` -- the **absolute** ``[white, black, draw]`` outcome
      re-expressed for the player who moved in that row -- keyed off the row's
      recorded ``to_play``, never off ply parity: during the draft the mover
      can repeat, so decisive rows need not strictly alternate ``[1,0,0]`` /
      ``[0,0,1]`` (wave-9b review F7). This per-row flip
      is the classic perspective bug (design F11) and is asserted at both
      parities by the unit suite. It is the *pure* result: ``draw_score`` is a
      search-time preference and never appears in a target (design D1/D4).

      **The one exception is KOM-245's cap adjudication**, and it is a change
      of this column alone: a game the engine stopped at its turn cap
      (``capped``, which is the engine's own ``termination_reason()`` and never
      a ply comparison -- review F3) keeps ``termination = TERMINATION_DRAW``
      but, under ``cap_adjudication.mode == "material"``, is labelled by the
      material on the final board instead of ``(0, 1, 0)``. With
      ``d = white - black`` and ``p = p_max * min(1, |d| / d_full)``, the
      leader's rows get ``(p, 1 - p, 0)`` and the trailer's ``(0, 1 - p, p)``,
      in the same per-row ``to_play`` frame everything else here is in;
      ``d == 0`` leaves ``(0, 1, 0)``. A draw reached *below* the cap is a rule
      draw (repetition, stalemate, insufficient material) and is untouched --
      neither side failed to convert there, the rules ended the game.

      **Every derived column follows from ``wdl`` and none of them is
      special-cased** (review F2): ``outcome_value`` is ``wdl[0] - wdl[2]``, so
      ``z`` is ``+-p``; ``short_term_value``'s past-the-end clamp is that same
      scalar; and ``draft_outcome`` is a copy of the row's ``wdl``. One flip,
      one derivation -- a second rule for any of the three would be a second
      place to disagree about what the game's result was.
    * ``mlh_target`` -- ``(total_plies - ply) * MLH_SCALE``, i.e. plies still to
      come, scaled once by the constant pinned in
      :mod:`komugi.selfplay.records`.
    * ``final_control`` -- the **absolute** top-of-tower ownership map of the
      final board (KOM-010), re-expressed for the row's own mover by
      :func:`mover_relative_control`. Computed once per game, off the same
      ``to_play`` ``wdl_target`` flips off, so a row's targets never disagree
      about whose point of view they are in. The map is a property of the
      *last* position only: it is the ownership head's label, not a per-ply
      board summary, and every row of one game therefore carries the same map
      up to its own flip.
    * ``termination`` -- ``1`` on a draw, ``0`` on a decisive result, derived
      from the same ``draw`` the ``wdl_target`` flip reads. Value ``2``
      (resign) is reserved and never written here: P0a/P0b have no resign, and
      writing it would claim a termination the worker cannot produce. It is a
      *game-level* value, so it is computed once and stamped on every row.
    * ``short_term_value`` -- for each horizon ``h`` of
      :data:`~komugi.selfplay.records.SHORT_TERM_HORIZONS`, the ``root_value``
      of the row ``h`` plies later, flipped when that row's mover differs from
      this one's, and clamped to the game's result ``z`` past the end.
      **Bounded by ``len(rows)``, never by ``total_plies``** -- see the
      ``total_plies`` argument below: on a booked game the two differ by the
      prefix length, so a ply-counter bound would index off the end of the
      buffer and raise on every booked game, and would fire the clamp ``L``
      rows too late in the cases that stayed in range. Nothing in that loop is
      a ply comparison. The lookup is *semantically* right under booking too,
      because ``rows`` is ply-contiguous after the prefix, so ``rows[i + h]``
      is still the row ``h`` plies later and "past the end of the buffer" is
      exactly "past the end of the game".

      **What this column carries at ``mlh_utility_lambda > 0``** (KOM-159
      design review F5): ``root_value`` is the search's ``v_mix``, which
      averages leaf utilities, and at a positive lambda those leaves carry the
      moves-left conversion term -- so this target, and every ``root_value``
      diagnostic read off the shard, carry it too. ``wdl_target``,
      ``mlh_target`` and ``termination`` do **not**: they are the game's
      result and its length, and no utility reaches them (design D9). The
      ``short_term_value`` head is staged off, so today this is a change of a
      recorded column's meaning and not of any gradient; a later bring-up of
      that head on campaign-6 shards has to decide whether to accept it.
    * ``draft_outcome`` -- the row's own ``wdl_target`` where the row's
      position was in the draft phase (:attr:`PendingRow.in_draft`, the
      drafting-rights predicate captured at record time), and all zeros
      elsewhere. One flip, one perspective, no second derivation: this column
      *is* ``wdl_target``, gated by a per-row boolean, so its label rule on
      resigned/adjudicated/truncated games is ``wdl_target``'s by construction.
      The zeros are the mask, and they cost no new seam.

    Args:
        rows: The game's buffered moves, in play order (``ply`` ascending).
        outcome: ``(3,)`` ``uint8`` ``win_state()`` of the final position.
        final_control: ``(81,)`` ``int8`` absolute control map of the final
            position, from
            :meth:`~komugi.selfplay.adapter.GameState.final_control`.
        total_plies: Plies played in the game, passed explicitly so the
            arithmetic is visible at the call site. It is ``len(rows)`` for an
            unbooked game and **is not** for a booked one: a book prefix is
            played but not recorded (KOM-014 §4.3), so the game's ply counter
            runs ahead of its row count by the prefix length ``L``. The
            ``mlh_target`` arithmetic below stays correct precisely because
            ``row.ply`` is absolute -- ``total_plies - row.ply`` is "plies still
            to come" either way, while ``len(rows) - row.ply`` would be a
            negative number for the first rows of a booked game.
        game_id_value: The ``game_id`` stamped on every row.
        cap_adjudication: KOM-245's rule and its two constants, defaulted to
            :data:`CAP_ADJUDICATION_OFF` so every existing caller -- and every
            campaign before the switch -- keeps today's labels exactly.
        final_material: ``(white_total, black_total)`` on the **final** board,
            from :meth:`~komugi.selfplay.adapter.GameState.final_material`.
            Read only when the rule fires; ``None`` is what a caller that does
            not adjudicate passes, and it is refused if the rule needs it.
        capped: Whether the engine stopped this game at its turn cap, i.e.
            ``state.game.termination_reason() == TERMINATION_TURN_CAP``. Passed
            in rather than re-derived from ``total_plies`` because the cap is
            the engine's rule and a ply comparison here would be a second
            spelling of it that reads the *class* constant -- one a test engine
            with a shortened cap silently falsifies (review F3).

    Returns:
        One dict per row, ready for :meth:`~komugi.train.shards.ShardWriter.append`.

    Raises:
        ValueError: If `outcome` names no result at all -- an unfinished game
            has no targets, and writing zeros would be a lie the trainer cannot
            detect. Or if the cap rule fires with no `final_material` to read:
            a labelling rule with no input must refuse rather than fall back to
            the label it was switched away from.
    """
    result = np.asarray(outcome, dtype=np.uint8)
    if not bool(result.any()):
        raise ValueError(
            "cannot finalize rows for a game with no outcome: win_state() is "
            "all zero, so the game is still in progress"
        )
    draw = bool(result[_DRAW_INDEX])
    # The final board is read ONCE, here, and every row is a flip of this one
    # array: re-deriving it per row would be 81 engine calls per ply for an
    # answer that cannot change after the last move.
    absolute_control = np.asarray(final_control, dtype=np.int8)
    # Game-level, computed once. `2` (resign) is reserved for resign v2 and is
    # unreachable from here on purpose: this worker cannot resign, and a
    # column that claimed otherwise would be worse than no column.
    termination = TERMINATION_DRAW if draw else TERMINATION_DECISIVE

    # KOM-245, computed ONCE per game like the control map above: the material
    # on the final board cannot change after the last move, and `p` is a
    # property of the game, not of a row. `leader` stays `_NO_LEADER` unless
    # the rule both fires and finds a lead, which is what leaves every other
    # game -- and a level cap game -- on exactly today's `(0, 1, 0)`.
    leader = _NO_LEADER
    cap_p = 0.0
    if draw and capped and cap_adjudication.mode == CAP_ADJUDICATION_MATERIAL_MODE:
        if final_material is None:
            raise ValueError(
                "cap adjudication is on and this game hit the engine's turn "
                "cap, but no final_material was passed: the label is a "
                "function of the final board and there is nothing to read it "
                "from"
            )
        white, black = int(final_material[0]), int(final_material[1])
        difference = white - black
        if difference != 0:
            cap_p = cap_adjudication.p_max * min(
                1.0, abs(difference) / cap_adjudication.d_full
            )
            # Colours as the engine indexes them, which is what `to_play` is in.
            leader = 0 if difference > 0 else 1

    samples: list[dict[str, torch.Tensor]] = []
    for index, row in enumerate(rows):
        wdl = np.zeros(3, dtype=np.float32)
        if draw:
            if leader == _NO_LEADER:
                wdl[1] = 1.0
            elif row.to_play == leader:
                # Keyed on the row's own mover, never on ply parity: the draft
                # lets one side move twice (the same discipline the decisive
                # branch below and `short_term_value` are under).
                wdl[0] = cap_p
                wdl[1] = 1.0 - cap_p
            else:
                wdl[1] = 1.0 - cap_p
                wdl[2] = cap_p
        elif bool(result[row.to_play]):
            wdl[0] = 1.0
        else:
            wdl[2] = 1.0
        # `z` is win-minus-loss in THIS row's perspective -- the same scalar
        # the uncertainty head's target is built from, so the two heads cannot
        # disagree about what "the result" is. Exactly +1 / 0 / -1, except on
        # an adjudicated cap game, where it is +-p by the same subtraction and
        # by no second rule (KOM-245 review F2).
        outcome_value = float(wdl[0] - wdl[2])
        short_term_value = np.empty(
            len(SHORT_TERM_HORIZONS), dtype=np.float32
        )
        for slot, horizon in enumerate(SHORT_TERM_HORIZONS):
            future = index + horizon
            # `len(rows)`, NOT `total_plies`: `rows` is a buffer and
            # `total_plies` is a ply counter, and a booked game's prefix makes
            # them differ. See the docstring.
            if future < len(rows):
                value = rows[future].root_value
                # Keyed off `to_play`, never off ply parity: the draft lets
                # one side move twice, so the movers of consecutive rows need
                # not alternate and a parity-keyed flip would silently
                # sign-invert every draft-adjacent target.
                short_term_value[slot] = (
                    value
                    if rows[future].to_play == row.to_play
                    else -value
                )
            else:
                short_term_value[slot] = outcome_value
        draft_outcome = wdl.copy() if row.in_draft else np.zeros(
            3, dtype=np.float32
        )
        samples.append(
            {
                "observation": torch.from_numpy(
                    np.ascontiguousarray(row.observation, dtype=np.float32)
                ),
                "policy_target": torch.from_numpy(
                    np.ascontiguousarray(row.policy_target, dtype=np.float32)
                ),
                "wdl_target": torch.from_numpy(wdl),
                "mlh_target": torch.tensor(
                    float(total_plies - row.ply) * MLH_SCALE, dtype=torch.float32
                ),
                "policy_kl": torch.tensor(row.policy_kl, dtype=torch.float32),
                "root_value": torch.tensor(row.root_value, dtype=torch.float32),
                "search_mode": torch.tensor(row.search_mode, dtype=torch.int64),
                "to_play": torch.tensor(row.to_play, dtype=torch.int64),
                "ply": torch.tensor(row.ply, dtype=torch.int64),
                "game_id": torch.tensor(game_id_value, dtype=torch.int64),
                "sims_used": torch.tensor(row.sims_used, dtype=torch.int64),
                # Stamped from the running encoder, never from a literal: the
                # row and the planes it holds must agree by construction.
                "plane_map_version": torch.tensor(
                    PLANE_MAP_VERSION, dtype=torch.int64
                ),
                # int64 because a shard holds float32 and int64 and nothing
                # else (design F4); the map itself is three-valued.
                "final_control": torch.from_numpy(
                    mover_relative_control(absolute_control, row.to_play).astype(
                        np.int64
                    )
                ),
                # The move that was actually played from this position (KOM-014
                # §3.1/B2). Carried through untouched from the search result the
                # loop advanced the state by, so a shard can be replayed without
                # re-searching it.
                "action": torch.tensor(row.action, dtype=torch.int64),
                "termination": torch.tensor(termination, dtype=torch.int64),
                "short_term_value": torch.from_numpy(short_term_value),
                "draft_outcome": torch.from_numpy(draft_outcome),
            }
        )
    return samples


@dataclass(frozen=True)
class WorkerConfig:
    """Which worker this is, and what it was asked to play.

    Separate from :class:`~komugi.config.SelfPlayConfig`, which describes *how*
    self-play is played and is shared by every worker in a run. These four
    integers are this process's identity, and three of them are the spawn key
    every seed in the game derives from (design D11).

    Args:
        worker_id: Index within the run. Also the id the inference server knows
            this worker by, and the ``worker_<k>`` in its shard path.
        incarnation: How many times this worker id has been (re)spawned,
            starting at 0. In the seed key so a respawn does not replay its
            predecessor's games (design F10c), and in the shard path so it
            cannot collide with a zombie's directory (design D10).
        run_seed: The run's entropy. One number, from the orchestrator.
        games: How many games :meth:`SelfPlayWorker.run` plays.
        shard_dir: Directory this worker owns exclusively --
          :class:`~komugi.train.shards.ShardWriter` enforces one writer per
          directory, and design D10 gives every worker its own.
        game_index_base: The first ``game_index``
          :meth:`SelfPlayWorker.run` plays. ``0`` -- one worker process, one
          game at a time -- unless this is one lane of an interleaved worker,
          in which case it is :func:`lane_game_index_base` and the lanes'
          index ranges (hence their seeds and their ``game_id`` s) are disjoint
          by construction (KOM-040 b1).
    """

    worker_id: int
    incarnation: int
    run_seed: int
    games: int
    shard_dir: str
    game_index_base: int = 0


class SelfPlayWorker:
    """Plays games with a Gumbel search and records one row per move.

    Owns no process and no segment: the evaluator and the writer are handed in,
    so the same object runs in a spawned worker (with a
    :class:`~komugi.selfplay.inference.SlotClient`) and in a single-process test
    (with a stub or an
    :class:`~komugi.selfplay.inference.InProcessEvaluator`) without a branch
    anywhere below :func:`~komugi.selfplay.adapter.make_evaluator`.

    Args:
        evaluator: The inference client. Anything satisfying
            :class:`~komugi.selfplay.inference.BatchEvaluator`, or the
            single-position :class:`~komugi.selfplay.inference.SlotClient`.
        selfplay: Search budgets, the full/fast mix, ``draw_score``, and the
            draft book (``book_path``/``book_fraction``, KOM-014 §4.1). The
            book is loaded **once**, here, and an unloadable one fails the
            worker rather than being skipped.
        config: This worker's identity and workload.
        writer: Where rows go. One writer, one directory, this worker's own.
        wave_evaluator: The batched seam, or ``None`` -- the default, and what
            every caller that predates KOM-040 b2 gets. ``None`` means the
            search evaluates leaves one at a time through `evaluator`, which is
            this class exactly as it was; given one, the search batches each
            halving round's leaves into waves and the worker's *rows* are
            unchanged (the search's own suite pins that the two schedules
            produce identical trees). Built by :func:`_wave_evaluator_for` for
            production lanes and left ``None`` by ``--serial-debug``, which is
            the bit-identical reference and must stay on the path that predates
            the seam (see the module docstring).
        search_context: The per-search moves-left reference (KOM-159), or
            ``None`` -- the default -- for "this worker owns its own". It is a
            parameter at all for one case: with a `wave_evaluator`, the two
            seams are built in two different places (``_run_lane`` builds the
            wave one over its lane's client, this constructor builds the scalar
            one) and they MUST centre on the same root, so the caller that
            builds both hands the same object to both. One per game slot: a
            worker plays its games one after another and a lane holds its own
            worker, so a single object per worker is a single object per game
            in flight.

            **Checked, not trusted** (KOM-171, KOM-159b review F2): when the
            moves-left term is live this constructor refuses a `wave_evaluator`
            that is centred on any other object, so the sharing is an invariant
            of the class rather than a property of its two production call
            sites. At ``mlh_utility_lambda == 0`` no check runs -- the
            reference is never read, and the disabled path must stay the path
            that predates the term (review F9).

    Raises:
        ValueError: If the moves-left term is live and `wave_evaluator` is not
            a :class:`~komugi.selfplay.adapter.WaveSeam` centred on
            `search_context`.
    """

    def __init__(
        self,
        evaluator: BatchEvaluator | SlotClient,
        selfplay: SelfPlayConfig,
        config: WorkerConfig,
        writer: ShardWriter,
        wave_evaluator: AnyWaveEvaluator | None = None,
        *,
        search_context: SearchContext | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._selfplay = selfplay
        # The KOM-220 tactical seam, built ONCE per worker and only when the
        # discovery layer is actually on: with `discovery_mode == "off"` the
        # provider is never constructed and never touched, so the off path is
        # the search this project has always run, to the line. The identity is
        # checked here rather than trusted, because it is what every search's
        # diagnostics -- and therefore the run's provenance -- will carry.
        self._tactical_provider: TacticalProvider | None = None
        if selfplay.discovery_mode != "off":
            provider = GameMarshalCaptureProvider()
            if provider.provider_id != _TACTICAL_PROVIDER_ID:
                raise ValueError(
                    f"tactical provider identity is "
                    f"{provider.provider_id!r}, expected "
                    f"{_TACTICAL_PROVIDER_ID!r}: the worker records the "
                    f"generator its games were discovered with, so an "
                    f"unrecognised one must not reach a shard"
                )
            self._tactical_provider = provider
        self._config = config
        self._writer = writer
        # Read off the config once, here, rather than per move: the four
        # numbers cannot change inside a run, and `make_evaluator` refuses a
        # positive lambda with no context at THIS line rather than at the
        # first leaf of the first game (KOM-159 R9).
        self._mlh = MlhUtility.from_selfplay(selfplay)
        self._search_context = (
            SearchContext() if search_context is None else search_context
        )
        # with_wdl: self-play is the one caller that asks the search to carry
        # the leaf (w, d, l) up the tree (KOM-173 §4). It changes no number the
        # search ranks on -- the vector rides beside the scalar and never feeds
        # selection -- and the wave seam is built with the same flag.
        self._search_evaluator = make_evaluator(
            evaluator,
            selfplay.draw_score,
            self._mlh,
            self._search_context,
            with_wdl=True,
        )
        # ... and the other half of the same guard: a LIVE wave seam must be
        # centred on the very object the scalar seam above just bound, because
        # the search asks the scalar seam about the root and the wave seam about
        # the leaves (KOM-159b review F2). At lam == 0 this is one comparison
        # that returns, so the bit-identical path stays bit-identical (F9).
        require_shared_context(wave_evaluator, self._mlh, self._search_context)
        self._wave_evaluator = wave_evaluator
        self._num_actions = num_actions()
        # ONCE per worker, not once per game (KOM-014 §7): a book is a ~300 KB
        # JSON file whose parse and whose guards (action space, GUNGI_MODE,
        # entry bounds) answer the same way for every game this process plays,
        # and re-reading it per game would put a file open on the critical path
        # of a loop whose whole point is to spend its time in the search.
        # Failure here is loud and kills the worker, which is the intent: a
        # worker that cannot load its book must not quietly play bookless
        # games and report a 10% booking rate (§6.3).
        self._book: Book | None = (
            load_book(selfplay.book_path) if selfplay.book_path else None
        )
        # The seed openings, on the same terms and for the same reasons
        # (KOM-254 D2): once per worker, and a file this process cannot read is
        # a dead worker rather than a generation that seeds 0% while reporting
        # `seed_openings_prob`. Empty tuple and empty digest when the knob is
        # off, so every read below is a `if self._selfplay.seed_openings_path`
        # away from the default path.
        self._seed_entries: tuple[SeedEntry, ...] = ()
        self._seed_sha256: str = ""
        if selfplay.seed_openings_path:
            self._seed_entries, self._seed_sha256 = load_seed_openings(
                selfplay.seed_openings_path
            )
            self._preflight_seed_openings()

    @property
    def config(self) -> WorkerConfig:
        """Return this worker's identity and workload."""
        return self._config

    def _scale_for(self, fast: bool, in_draft: bool) -> float:
        """Return the composed ``gumbel_scale`` for one move (KOM-036 §3.2).

        ``(draft_gumbel_scale if in_draft else 1.0) * (fast_gumbel_scale if
        fast else 1.0)`` -- the single site of the composition rule.

        **Multiplicative**, because the scale is a root temperature (§2.2) and
        temperatures compose on the log-odds scale: ``log s = log s_draft +
        log s_fast``. At the pre-registered values that puts a full draft ply
        at ``2.0``, a fast draft ply at ``2.0 * 0.5 == 1.0``, a full
        post-draft ply at ``1.0`` and a fast one at ``0.5``. The fast-draft
        cell is **intended**: at ``prob_fast = 0.75`` three of four draft plies
        are fast, so the alternative rule ("draft overrides fast") would put
        every draft ply at ``2.0``; the product instead guarantees that no
        draft ply is ever damped below the paper's default while the full
        draft searches -- the ones whose policy targets the P0b A/B weights
        most -- are hot.

        `in_draft` is **the mover's** bit, never the disjunction: see
        :meth:`komugi.engine.game.Game.in_draft`.

        The product is provably finite and ``> 0``:
        :class:`~komugi.config.SelfPlayConfig` bounds both factors to
        ``[1e-6, 1e6]``, so no accepted pair can compose to ``0.0`` or ``inf``
        and :meth:`~komugi.search.gumbel.GumbelConfig.__post_init__` can never
        fire from this path.
        """
        draft = self._selfplay.draft_gumbel_scale if in_draft else 1.0
        damped = self._selfplay.fast_gumbel_scale if fast else 1.0
        return draft * damped

    def _new_game(self) -> Game:
        """Return the engine game one self-play game is played on.

        A method rather than an inline ``Game()`` so the one thing this run's
        config decides about the referee -- which repetition key it counts
        occurrences of (KOM-141) -- can be asserted without standing up a
        worker rig, a shard writer or an inference server.

        The default is ``RepetitionRule.FULL``, Komugi's repaired 274-byte key,
        which is what every training row this project has produced was played
        under. Self-play has no reason to use gungi.js's cells-only key -- that
        is a REFEREE concern (see :class:`~komugi.eval.arena.ArenaConfig`) --
        and reading the field here rather than hardcoding it is only so a run
        that did configure one is played under the rule it declared instead of
        silently under the other.
        """
        return Game(repetition_rule=self._selfplay.repetition_rule)

    def _gumbel_config(
        self, fast: bool, in_draft: bool, seed: int
    ) -> GumbelConfig:
        """Return the search config for one move.

        Built fresh every move because ``seed`` changes every move (design
        F10d). ``m``/``n`` come straight from
        :class:`~komugi.config.SelfPlayConfig` -- 8/32 fast, 32/256 full --
        and ``gumbel_scale`` is the schedule :meth:`_scale_for` composes:
        §3.1's reduced scale for fast mode is KOM-029's knob, shipped, and
        KOM-036 is what finally passes it. At the shipped defaults the
        composed scale is exactly ``1.0``, which is the value this config
        carried before the field was threaded at all.

        The two search modes (KOM-223) are forwarded on **every** path -- fast
        and full, draft and not -- because they name the algorithm the run
        declared: a mode that reached three of four move classes would be a
        campaign whose recorded identity is true of only some of its rows. At
        the defaults both are ``"off"``, which is the config this method built
        before the fields existed.
        """
        if fast:
            m, n = self._selfplay.gumbel_m_fast, self._selfplay.gumbel_n_fast
        else:
            m, n = self._selfplay.gumbel_m_full, self._selfplay.gumbel_n_full
        return GumbelConfig(
            num_actions=self._num_actions,
            m=m,
            n=n,
            seed=seed,
            gumbel_scale=self._scale_for(fast, in_draft),
            proof_mode=self._selfplay.proof_mode,
            discovery_mode=self._selfplay.discovery_mode,
        )

    def _book_entry_for(self, game_index: int) -> BookEntry | None:
        """Return the prefix this game starts from, or ``None`` (KOM-014 §4.2).

        The decision is :func:`~komugi.selfplay.book.select_book_entry`'s and
        nothing here adds to it: a pure function of the same four integers that
        name the game, on its own seed stream, so the audit can recompute this
        answer from a shard's ``game_id`` and the book file alone (§6.7).
        """
        if self._book is None:
            return None
        choice = select_book_entry(
            self._config.run_seed,
            self._config.worker_id,
            self._config.incarnation,
            game_index,
            entries=len(self._book),
            fraction=self._selfplay.book_fraction,
        )
        if not choice.booked or choice.entry_index is None:
            return None
        return self._book.entries[choice.entry_index]

    def _replay_actions(
        self,
        state: GameState,
        actions: Sequence[int],
        *,
        source_kind: str,
        source_path: str,
        entry_id: str,
        terminal_tail: str,
        illegal_tail: str,
    ) -> int:
        """Play `actions` into `state`, returning the plies played.

        The one replay, shared by the book path (:meth:`_replay_prefix`) and
        the seed-openings path (:meth:`_replay_seed_prefix`). Shared rather than
        reused verbatim because the two prefixes come from different artefacts
        and a diagnostic that names the wrong one is worse than none: the book's
        messages interpolate ``book_path`` and ``book_fraction``, which are
        ``""`` and ``0.0`` in a bookless seeded run (KOM-254 AM5). The two
        tails, and the noun in front of the path, are therefore the caller's;
        the sentence around them -- and the book's two messages, byte for byte
        -- is here.

        Replayed, never searched (KOM-014 §4.3): no evaluator call, no Gumbel
        draw and no :class:`PendingRow` -- a prefixed game is an *initial
        state*, and its prefix is not a training example.

        **An illegal prefix fails here, loudly** (§6.3). Whoever wrote the file
        verified these actions against an engine, so a disagreement now means
        the shard, the engine or ``GUNGI_MODE`` changed under it. Catching it
        and starting from the empty board instead would convert a corrupt file
        into a generation that quietly prefixed 0% of its games while reporting
        the configured share.

        **Terminality is checked before each action AND after the last one**
        (KOM-254 AM1). The before-check is KOM-014's; the after-check is what a
        *harvested* prefix needs, because the last recorded action of a real
        game IS the move that ended it -- a cut at the end of one would leave
        the game loop's ``while not state.is_terminal()`` false on its first
        test, write zero rows, and report success (design review F1).

        Raises:
            ValueError: On a prefix action that is illegal at the position it
                is played into, or a prefix that reaches a terminal position at
                or before its end -- naming the artefact, the entry, the ply
                and the action.
        """
        for ply, action in enumerate(actions):
            if state.is_terminal():
                raise ValueError(
                    f"{source_kind} {source_path} entry {entry_id} reaches "
                    f"a terminal position after {ply} of its {len(actions)} "
                    f"plies: {terminal_tail}"
                )
            # The ENGINE's mask, not `state.legal_mask()`: a prefix is replayed,
            # never searched, so the question here is legality and not what
            # DRAFTMIN would have let a search consider (KOM-156). A book whose
            # prefix ends a draft below the threshold is a legal prefix played
            # before this run's curriculum existed; refusing it here would turn
            # a threshold into a crash on the first booked game.
            if not bool(np.asarray(state.game.valid_moves())[action]):
                raise ValueError(
                    f"{source_kind} {source_path} entry {entry_id} plays "
                    f"action {action} at prefix ply {ply}, which is illegal at "
                    f"the position that prefix reaches: {illegal_tail}"
                )
            # keep_root for the same reason the game loop's own play does: the
            # position a prefix reaches is the first searched ply's root.
            state.play(action, keep_root=True)
        if state.is_terminal():
            raise ValueError(
                f"{source_kind} {source_path} entry {entry_id} reaches "
                f"a terminal position after {len(actions)} of its "
                f"{len(actions)} plies: {terminal_tail}"
            )
        return len(actions)

    def _replay_prefix(self, state: GameState, entry: BookEntry) -> int:
        """Play a BOOK `entry`'s actions into `state` (KOM-014 §4.3).

        The book's call site of :meth:`_replay_actions`, and the only place the
        book's two messages are written -- byte-identical to what they were
        before the helper was shared, because ``load_book`` guarantees
        ``len(entry.actions) == entry.plies``.
        """
        return self._replay_actions(
            state,
            entry.actions,
            source_kind="book",
            source_path=self._selfplay.book_path,
            entry_id=entry.id,
            terminal_tail=(
                f"the book and this engine disagree, and a game "
                f"started from the empty board instead would silently book "
                f"0% while reporting {self._selfplay.book_fraction}"
            ),
            illegal_tail=(
                "the shard, the engine or GUNGI_MODE changed under this book"
            ),
        )

    def _replay_seed_prefix(
        self, state: GameState, entry: SeedEntry, cut: int
    ) -> int:
        """Play a SEED entry's first `cut` actions into `state` (KOM-254 D4).

        The seed path's call site of :meth:`_replay_actions`. Its two messages
        name the **seed file**, the entry and the ply and nothing else: a
        seeded run is routinely bookless, so a message built from ``book_path``
        would point an operator at an empty string (design review F5).
        """
        return self._replay_actions(
            state,
            entry.actions[:cut],
            source_kind="seed openings",
            source_path=self._selfplay.seed_openings_path,
            entry_id=entry.id,
            terminal_tail=(
                "the seed file and this engine disagree, and a seeded game "
                "whose prefix has already ended records no row at all while "
                "reporting success"
            ),
            illegal_tail=(
                "the harvest, the engine or GUNGI_MODE changed under this "
                "seed file"
            ),
        )

    def _preflight_seed_openings(self) -> None:
        """Replay the first entry's shortest prefix, once, before any game.

        The worker's loud failure alone is not enough here (design review F11):
        at the flip-on share of 4% a worker's first seeded game arrives ~25
        games in, so an engine, encoding or ``GUNGI_MODE`` drift would kill
        every worker after most of an iteration's GPU time had been spent.
        ``cut_min`` rather than the whole action list, because that is the
        shortest prefix any game can actually be handed and the point is a
        startup check, not a second harvest verification.
        """
        entry = self._seed_entries[0]
        self._replay_seed_prefix(
            GameState(self._new_game(), self._selfplay.draw_score),
            entry,
            entry.cut_min,
        )

    def _seed_choice_for(self, game_index: int) -> SeedChoice:
        """Return what this game's seed-openings draw decided (KOM-254 D4).

        The decision is :func:`~komugi.selfplay.seed_openings.select_seed_entry`'s
        and nothing here adds to it: a pure function of the same four integers
        that name the game, on its own seed stream, so a census can recompute
        it from a shard's ``game_id`` and the file alone
        (:func:`~komugi.selfplay.seed_openings.recompute_seed_choice`).
        """
        return select_seed_entry(
            self._config.run_seed,
            self._config.worker_id,
            self._config.incarnation,
            game_index,
            entries=self._seed_entries,
            prob=self._selfplay.seed_openings_prob,
        )

    def play_game(self, game_index: int) -> int:
        """Play one game to the engine's adjudication and write its rows.

        A booked game (KOM-014 §4.2) replays a book prefix first and records
        **no row** for it, so the returned count is the number of *searched*
        plies and the game's rows begin at ``ply == L``. A **seeded** game
        (KOM-254 D4) does the same thing with a harvested prefix and a drawn
        cut, and is never also booked.

        Args:
            game_index: Index within this worker's workload. With
                ``(run_seed, worker_id, incarnation)`` it names the game's
                entire random stream -- the book and seed draws included.

        Returns:
            The number of rows written -- one per searched ply.
        """
        seeds = game_seed(
            self._config.run_seed,
            self._config.worker_id,
            self._config.incarnation,
            game_index,
        )
        # The seed-openings draw comes FIRST, on its own domain, and a seeded
        # game skips the book consult entirely (KOM-254 D4): the two prefixes
        # are alternatives, not layers. Skipping the consult perturbs nothing,
        # because the book's stream is per game and leaving it undrawn changes
        # no other game's answer. No draw at all when the knob is off, which is
        # KOM-246's "no new code on the default path" doctrine rather than a
        # seed-discipline requirement -- the stream is dedicated, so drawing
        # from it would change nothing either way.
        seed_choice: SeedChoice | None = (
            self._seed_choice_for(game_index)
            if self._selfplay.seed_openings_path
            else None
        )
        seeded = seed_choice is not None and seed_choice.seeded
        # The book draw happens on its OWN stream, before `rng` exists and
        # without touching it (§4.2, §1.4): drawn from `rng` it would shift
        # every subsequent full/fast draw of every game, booked and unbooked
        # alike, and a bookless/booked comparison at fixed seed would be
        # comparing two different searches.
        entry = None if seeded else self._book_entry_for(game_index)
        # The game's own stream, used for the full/fast draw and nothing else.
        # Per-move search seeds come from the same SeedSequence by spawn-key
        # composition (records.move_seed), so the two uses never interleave and
        # neither depends on how often the other was drawn.
        rng = np.random.default_rng(seeds)
        # The per-colour DRAFTMIN draw (KOM-246), on its OWN domain
        # (records.DRAFT_THRESHOLD_DOMAIN) so that `rng` and every
        # `move_seed(seeds, ply)` below are untouched -- both are replay claims
        # (D11/F10). Skipping the draw when the knob is off is NOT what makes
        # the default path byte-identical (the stream is dedicated: drawing
        # from it changes nothing either way); it is the "no new code on the
        # default path" doctrine, which is the claim this branch serves.
        if self._selfplay.min_placements_max > 0:
            thresholds: int | tuple[int, int] = draft_thresholds(
                self._config.run_seed,
                self._config.worker_id,
                self._config.incarnation,
                game_index,
                self._selfplay.min_placements,
                self._selfplay.min_placements_max,
            )
        else:
            thresholds = self._selfplay.min_placements

        # `is_root=True` is the DRAFTMIN wiring (KOM-156), and this is the only
        # place in self-play that says it: `state` IS the root of every ply's
        # search -- the loop below searches it, plays the chosen action into it
        # and searches it again -- so the root view is a property of this one
        # object for the whole game, which is why its own moves are played
        # `keep_root=True` while the search's clones drop the view on their
        # first edge. `min_placements=0` (the default) leaves every line of this
        # method byte-identical to what it was: the mask is then the engine's
        # own array and no branch below it is even reached.
        state = GameState(
            self._new_game(),
            self._selfplay.draw_score,
            min_placements=thresholds,
            is_root=True,
        )
        rows: list[PendingRow] = []
        # Absolute from here on: a booked game's loop enters at `ply == L`, so
        # move_seed(seeds, ply) at its first searched move is move_seed(seeds,
        # L). A booked and an unbooked game of one seed do not share a Gumbel
        # vector at the same *position index*, which is correct -- they are
        # different games.
        if seeded:
            # Narrowed for the type checker as well as for the reader: `seeded`
            # is a bool and narrows nothing, and a seeded choice carries both an
            # entry index and a cut, neither of them optional once it is true.
            assert seed_choice is not None
            assert seed_choice.entry_index is not None
            assert seed_choice.cut is not None
            seed_entry = self._seed_entries[seed_choice.entry_index]
            ply = self._replay_seed_prefix(state, seed_entry, seed_choice.cut)
            # After the replay, so a prefix this engine cannot play is a raise
            # rather than a log line followed by one. The identity is the
            # game's, not the shard's, because this is what an operator greps
            # while a worker is running (KOM-254 AM7). WARNING and not INFO for
            # the reason `_LOGGER` states: at INFO nothing this repository
            # launches would ever see it.
            _LOGGER.warning(
                "seeded game worker=%d incarnation=%d game_index=%d "
                "entry=%s cut=%d seed_sha256=%s",
                self._config.worker_id,
                self._config.incarnation,
                game_index,
                seed_entry.id,
                seed_choice.cut,
                self._seed_sha256[:_SEED_SHA_PREFIX],
            )
        elif entry is not None:
            ply = self._replay_prefix(state, entry)
        else:
            ply = 0
        # No ply cap here, deliberately: the engine adjudicates every
        # termination, the turn-cap draw included (design D6/F15).
        while not state.is_terminal():
            fast = bool(rng.random() < self._selfplay.prob_fast)
            legal = state.legal_mask()
            observation = state.observation()
            to_play = state.to_play()
            # The TEMPERATURE SCHEDULE's draft read (KOM-036 §3.3), and a
            # different question from `in_draft` below: this one is
            # mover-relative (`GameState.in_draft()` asks the engine about
            # `to_play`), because a temperature is a property of the move being
            # searched. It is a READ, not a draw -- nothing here touches `rng`,
            # so the full/fast sequence above is scale-independent and every
            # replay claim survives. The `_gumbel_config` build moved below
            # these reads to consume it; `rng.random()` did not move and has
            # gained no sibling.
            mover_in_draft = state.in_draft()
            config = self._gumbel_config(
                fast, mover_in_draft, move_seed(seeds, ply)
            )
            # Read off `observation`, before the move is played, so the flag
            # describes the position the row records rather than the one after
            # it. True while EITHER side retains drafting rights (KOM-005b
            # §1.4.2): the draft sub-game is not over while one side can still
            # place, and the narrower "this row's mover is still drafting"
            # reading would split the phase mid-way and make the head's
            # training set depend on which side finished first. See
            # `_DRAFT_PLANE_WHITE` for why the planes rather than the engine.
            in_draft = bool(
                observation[_DRAFT_PLANE_WHITE, 0, 0] > 0.0
                or observation[_DRAFT_PLANE_BLACK, 0, 0] > 0.0
            )

            # The moves-left term is centred on THIS ply's root, so the
            # reference is dropped here, one line above the search that will
            # set it again (KOM-159 §3.3). Unconditional: at lambda = 0 the
            # context is never read by anything, so clearing it costs one
            # attribute store per move and removes the possibility of a
            # `lambda` turned on mid-run reading the previous ply's root.
            # This covers the forced-move path too, which goes through
            # `search()` and evaluates only the root.
            self._search_context.reset()
            # wave_evaluator=None is the scalar search this loop always ran;
            # a wave evaluator changes when the network is asked, never what
            # is asked or what the answers do to the tree.
            # The provider is passed only when discovery is on, so the off path
            # makes literally the call this loop has always made -- same
            # arguments, same search object (KOM-223). The search refuses the
            # two halves of the pair disagreeing, so a run can neither record a
            # discovery layer with nothing to consult nor carry a provider it
            # would never consult.
            if self._tactical_provider is None:
                result = GumbelSearch(
                    self._search_evaluator,
                    config,
                    wave_evaluator=self._wave_evaluator,
                ).search(state)
            else:
                result = GumbelSearch(
                    self._search_evaluator,
                    config,
                    wave_evaluator=self._wave_evaluator,
                    tactical_provider=self._tactical_provider,
                ).search(state)

            # KOM-173 (ii): `result` also carries `root_wdl`, `root_wdl_pi` and
            # `played_wdl` -- the searched (w, d, l) readouts, already in this
            # row's mover frame. They are NOT put on the row here: a row field
            # only reaches a shard through `SCHEMA`, and appending those three
            # columns is delegation (ii)'s fingerprint bump in `records.py`,
            # which this task does not touch. Until then the readouts are
            # produced and discarded, which is what keeps the shard bytes and
            # the golden fingerprint identical.
            rows.append(
                PendingRow(
                    observation=observation,
                    policy_target=result.policy_target,
                    policy_kl=policy_kl(
                        result.policy_target, result.root_prior_logits, legal
                    ),
                    root_value=result.root_value,
                    search_mode=MODE_FAST if fast else MODE_FULL,
                    to_play=to_play,
                    ply=ply,
                    sims_used=result.sims_used,
                    # Recorded here, one line above the play, so the row and the
                    # state cannot disagree about which move was made (KOM-014
                    # §3.1/B2).
                    action=int(result.action),
                    in_draft=in_draft,
                )
            )
            # keep_root: the position this move reaches is the NEXT ply's root,
            # not an interior node of the tree just searched (KOM-156). A bare
            # `play` would clear the root view here and switch DRAFTMIN off
            # after the first ply of every game.
            state.play(result.action, keep_root=True)
            ply += 1

        samples = finalize_rows(
            rows,
            state.absolute_outcome(),
            # The FINAL board, read after the loop and exactly once: this is
            # the position the ownership target describes (KOM-010).
            state.final_control(),
            ply,
            game_id(
                self._config.worker_id, self._config.incarnation, game_index
            ),
            cap_adjudication=CapAdjudication.from_selfplay(self._selfplay),
            # The same final board, read once beside the control map (KOM-245).
            final_material=state.final_material(),
            # The ENGINE's spelling of the cap, not a ply comparison here: the
            # cap is the engine's rule, `termination_reason` reports it first
            # and a `total_plies >= Game.max_turns()` here would read the class
            # constant -- which any engine with a shortened cap falsifies
            # silently (review F3).
            capped=state.game.termination_reason() == TERMINATION_TURN_CAP,
        )
        for sample in samples:
            self._writer.append(sample)
        return len(samples)

    def run(self) -> int:
        """Play ``config.games`` games, flushing after each one.

        The flush is the durability policy of design D10 made concrete: the
        writer's timed trigger only fires inside ``append``, so a worker that
        finishes a game and then spends minutes on the next one would leave its
        tail buffered. Sealing per game bounds what a kill costs to the game in
        progress -- and sealing an empty buffer is free.

        The indices played are ``config.game_index_base`` onwards, which is
        ``0`` onwards for every worker that is not one lane of an interleaved
        pair (KOM-040 b1).

        Returns:
            The number of rows written across every game -- what an interleaved
            worker's lane aggregation sums (design D1'/C8).
        """
        base = self._config.game_index_base
        rows = 0
        for game_index in range(base, base + self._config.games):
            rows += self.play_game(game_index)
            self._writer.flush()
        return rows


@dataclass(frozen=True)
class LaneSpec:
    """One interleaved game lane's *private* resources (KOM-040 b1).

    A lane is a thread playing its own games over its own segment with its own
    writer. Everything a lane touches is named here, and nothing here is shared
    with the other lane -- that is the property that makes interleaving a
    scheduling change rather than a protocol change: the seam still sees one
    client per segment with one request in flight, so every doorbell invariant
    (single writer, payload before counter, a fresh segment per incarnation)
    holds unchanged (design D1').

    Picklable by construction -- ints and strings only -- because the whole
    tuple crosses a ``spawn`` boundary as an argument of :func:`worker_main`.

    Args:
        lane: Zero-based index within the worker process. Decides the lane's
            game-index range via :func:`lane_game_index_base`, and nothing else.
        slot_id: The id the **server** knows this lane's segment by. Distinct
            from ``worker_id``: the server keys its slot map by this number, so
            two lanes of one worker must present two ids, while seeds and
            ``game_id`` s keep deriving from the real ``worker_id`` (see
            :func:`lane_game_index_base` for how the lanes stay disjoint there).
        shm_name: This lane's own segment, created by the orchestrator.
        shard_dir: This lane's own shard directory.
            :class:`~komugi.train.shards.ShardWriter` is documented not
            thread-safe and one-writer-per-directory, so the lanes cannot share
            one writer *or* one directory; the orchestrator's lane-suffixed
            names keep both readable and inside the reader's ``worker_*`` glob.
        games: How many games this lane plays.
    """

    lane: int
    slot_id: int
    shm_name: str
    shard_dir: str
    games: int


@dataclass
class _LaneOutcome:
    """What one lane thread produced, or what killed it.

    Mutable, and written by exactly one thread -- its own lane's. It is the
    only object a lane hands back, and the joining thread reads it strictly
    after ``join()``, which is the happens-before edge that makes the plain
    attributes enough (design D1'/C3: no shared mutable state between lanes
    except the final aggregation).
    """

    rows: int = 0
    games: int = 0
    error: BaseException | None = None


def _wave_evaluator_for(
    client: SlotClient,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
    context: SearchContext | None = None,
) -> WaveSeam | None:
    """Return this client's wave seam, or ``None`` if waves are off.

    The **one** place the enable decision is made (KOM-040 b2), so a lane of an
    interleaved worker and a single-game worker cannot end up on different
    schedules for the same config. The decision itself is one comparison:
    ``inference.max_wave > 1``, the convention
    :func:`~komugi.selfplay.orchestrate.validate_wave_fit` documents and
    enforces -- there is no separate "use waves" flag, because the segment is
    already sized by ``max_wave`` and the two could only ever disagree by
    wasting the memory (see the module docstring).

    Per **client**, not per worker: the closure holds one
    :class:`~komugi.selfplay.inference.SlotClient`, and a lane submitting on its
    sibling's segment would ring a doorbell it does not own. Both lanes call
    this with their own client, which is why interleaving needs nothing else
    from this ticket.

    Args:
        client: The lane's own client. Its ``max_wave`` is the same number
            checked here -- it and the segment come from `inference`.
        selfplay: For ``draw_score``, which the wave adapter applies exactly as
            the scalar one does (design D1), and for the four moves-left
            parameters, read here through
            :meth:`~komugi.selfplay.adapter.MlhUtility.from_selfplay` so that
            the wave seam and the scalar seam cannot be built from different
            readings of one config.
        inference: The seam config this run's segments were built from.
        context: The lane's :class:`~komugi.selfplay.adapter.SearchContext` --
            **the same object** the lane's :class:`SelfPlayWorker` is given, so
            the root the scalar seam records is the root the wave leaves are
            centred on. ``None`` (the default, and every pre-KOM-159 caller) is
            only legal at ``mlh_utility_lambda == 0``, which
            :func:`~komugi.selfplay.adapter.make_wave_evaluator` enforces. The
            *sameness* is enforced too, at the far end: the returned seam names
            its context and :class:`SelfPlayWorker` refuses one that is not its
            own (KOM-159b review F2).

    Returns:
        A :class:`~komugi.selfplay.adapter.WaveSeam` -- a wave evaluator that
        publishes `context` -- or ``None`` at ``max_wave == 1``, in which case
        the caller builds precisely the objects it built before waves existed.
    """
    if inference.max_wave <= 1:
        return None
    # with_wdl: the same value :class:`SelfPlayWorker` builds its scalar seam
    # with, and it has to be (KOM-173 §4). The search reads the answer shape off
    # the ROOT, which is always evaluated through the scalar seam, so a wave
    # that dropped the vector here would be refused at the first backup rather
    # than silently reported.
    return make_wave_evaluator(
        client, selfplay, MlhUtility.from_selfplay(selfplay), context, with_wdl=True
    )


def _lane_client(
    spec: LaneSpec, selfplay: SelfPlayConfig, inference: InferenceConfig
) -> SlotClient:
    """Build a lane's client with its diagnostics wired to the lane layout.

    The one argument that matters here is ``lanes=selfplay.games_in_flight``:
    without it the client's :func:`~komugi.selfplay.inference.slot_label` falls
    back to ``slot N`` and a two-lane run's timeouts stop naming the worker and
    lane an operator actually restarts (review F3 wiring, KOM-040f impl review
    R1 -- the label existed but no production caller passed ``lanes``).
    """
    return SlotClient(
        spec.shm_name, spec.slot_id, inference, lanes=selfplay.games_in_flight
    )


def _run_lane(
    spec: LaneSpec,
    worker_id: int,
    incarnation: int,
    run_seed: int,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
    outcome: _LaneOutcome,
) -> None:
    """Play one lane's games, recording the result (or the failure) in `outcome`.

    The body of a lane thread, and deliberately the *same* body a single-game
    worker runs: one :class:`~komugi.selfplay.inference.SlotClient` over one
    segment, one :class:`~komugi.train.shards.ShardWriter` over one directory,
    one :class:`SelfPlayWorker` -- and, when ``inference.max_wave > 1``, one
    wave evaluator over that one client (:func:`_wave_evaluator_for`). Nothing
    here knows there is another lane.

    No exception escapes: it is stored, because a thread that raises would
    print a traceback and let the process exit **zero**, which the orchestrator
    would read as a worker that finished its budget. :func:`_run_lanes` joins
    every lane and re-raises (design D1'/C6).
    """
    try:
        client = _lane_client(spec, selfplay, inference)
        try:
            writer = ShardWriter(spec.shard_dir, SCHEMA)
            try:
                # ONE context for this lane, shared by its two evaluator seams
                # and by nothing else: the lanes are threads, and a shared
                # reference would let one lane's root centre the other's
                # leaves (KOM-159; design D1'/C3's "no shared mutable state").
                context = SearchContext()
                outcome.rows = SelfPlayWorker(
                    client,
                    selfplay,
                    WorkerConfig(
                        worker_id=worker_id,
                        incarnation=incarnation,
                        run_seed=run_seed,
                        games=spec.games,
                        shard_dir=spec.shard_dir,
                        game_index_base=lane_game_index_base(spec.lane, spec.games),
                    ),
                    writer,
                    # This lane's own client, never the sibling's: a wave is a
                    # ticket, and a ticket belongs to one segment.
                    _wave_evaluator_for(client, selfplay, inference, context),
                    search_context=context,
                ).run()
                outcome.games = spec.games
            finally:
                writer.close()
        finally:
            client.close()
    except BaseException as exc:  # noqa: BLE001 -- re-raised by the joiner
        outcome.error = exc


def _run_lanes(
    lanes: Sequence[LaneSpec],
    worker_id: int,
    incarnation: int,
    run_seed: int,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> tuple[int, int]:
    """Run every lane concurrently, join them all, and report the totals.

    The interleaving itself is one sentence of mechanism: a lane blocked in
    :meth:`~komugi.selfplay.inference.SlotClient.submit` is sleeping in
    ``time.sleep``, which releases the GIL, so the other lane's engine and
    search code runs in that window. Threads rather than processes because the
    latency being hidden is the seam's, not the CPU's, and a second process
    would need a second copy of everything the worker holds.

    **A failure in either lane fails the whole process** (design D1'/C6): the
    timeout policy of :func:`worker_main` is unchanged, and a poisoned client
    is not something the sibling can repair. The sibling is joined first rather
    than abandoned -- an orphaned lane would keep ringing a doorbell into a
    segment the orchestrator is about to unlink -- and only then is the first
    error re-raised, with its own type, so ``InferenceTimeout`` still reads as
    ``InferenceTimeout`` in the exit path. The sibling is not *interrupted*:
    there is no cross-lane signal by design, so the process lives until the
    healthy lane finishes its budget or (the usual case, since a wedged server
    wedges both lanes) times out itself.

    Returns:
        ``(games, rows)`` summed over the lanes -- what a single-lane worker
        would have produced on its own.

    Raises:
        BaseException: The first error any lane recorded, re-raised unchanged.
    """
    outcomes = [_LaneOutcome() for _ in lanes]
    threads = [
        threading.Thread(
            target=_run_lane,
            args=(
                spec,
                worker_id,
                incarnation,
                run_seed,
                selfplay,
                inference,
                outcome,
            ),
            name=f"komugi-lane-{spec.lane}-w{worker_id}",
        )
        for spec, outcome in zip(lanes, outcomes, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [outcome.error for outcome in outcomes if outcome.error is not None]
    for extra in failed[1:]:
        # Not lost: the first error is the one that propagates (its type is the
        # process's exit story), and a second lane failing for its own reason
        # is diagnostic an operator would otherwise never see.
        warnings.warn(
            f"a second self-play lane of worker {worker_id} also failed: "
            f"{extra!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    if failed:
        raise failed[0]
    return (
        sum(outcome.games for outcome in outcomes),
        sum(outcome.rows for outcome in outcomes),
    )


def worker_main(
    worker_id: int,
    incarnation: int,
    run_seed: int,
    games: int,
    shard_dir: str,
    shm_name: str,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
    extra_lanes: Sequence[LaneSpec] = (),
) -> None:
    """Entry point for a spawned self-play worker (design D2/F18).

    Module-level, and taking only picklable arguments, because that is what a
    ``spawn`` context can target: a bound method or a closure over a live
    numpy view does not survive the process boundary. The orchestrator
    (KOM-023) creates the shared-memory segment and passes its *name*; this
    process only ever attaches (design D2/F9).

    **Timeout is fatal.** :class:`~komugi.selfplay.inference.InferenceTimeout`
    is deliberately not caught: it propagates out of this function, the
    interpreter exits nonzero, and the orchestrator restarts the worker against
    a *fresh* segment. Catching it to resubmit would re-ring a doorbell the
    server may still be answering, which is the stale-response race the fatal
    policy exists to delete. The ``finally`` blocks below still run, and that is
    correct rather than a hedge: rows are appended only at game end, so the
    writer's buffer holds whole games and flushing it saves completed work.

    **Interleaving is additive** (KOM-040 b1). With
    ``selfplay.games_in_flight == 1`` -- the default -- this function is the
    single-game path it always was: no lane object, no thread, one client and
    one writer inline in the process's main thread. With ``2`` the orchestrator
    also passes ``extra_lanes``, and *every* lane (this one included) becomes a
    thread with its own segment, client, writer and game-index range; see
    :func:`_run_lanes`. The seam protocol is untouched either way.

    **Waves are additive too** (KOM-040 b2). Every lane -- and the single-game
    path below -- asks :func:`_wave_evaluator_for` whether this run's
    ``inference.max_wave`` enables them, over *its own* client. At the default
    ``max_wave == 1`` the answer is ``None`` and this function constructs
    exactly the objects it constructed before waves existed; above 1 the search
    submits each halving round's leaves as one ticket instead of one each.

    Args:
        worker_id: This worker's index, as the server knows it. Also lane 0's
            ``slot_id``, and the ``worker_id`` every lane's seeds derive from.
        incarnation: Respawn counter, from the orchestrator.
        run_seed: The run's entropy.
        games: How many games to play before exiting cleanly -- lane 0's budget
            when interleaving (the orchestrator splits the worker's budget).
        shard_dir: This worker's own shard directory (lane 0's, when
            interleaving).
        shm_name: Name of this worker's slot segment (lane 0's, when
            interleaving).
        selfplay: Search budgets, the full/fast mix, and ``games_in_flight``.
        inference: Seam shapes and timings; must match the server's.
        extra_lanes: The lanes **after** lane 0, empty by default. Lane 0 is
            described by the arguments above rather than repeated here, so the
            single-game call is exactly the call it was.

    Raises:
        ValueError: If the lane count disagrees with
            ``selfplay.games_in_flight``. A worker that quietly played one lane
            when two were configured would halve the run's throughput and look
            healthy doing it.
    """
    if 1 + len(extra_lanes) != selfplay.games_in_flight:
        raise ValueError(
            f"worker {worker_id} was given {len(extra_lanes)} extra lane(s), i.e. "
            f"{1 + len(extra_lanes)} in total, but selfplay.games_in_flight="
            f"{selfplay.games_in_flight}: the lane count and the config must "
            f"agree, or the worker plays a budget nobody asked for"
        )
    if extra_lanes:
        _run_lanes(
            (
                LaneSpec(
                    lane=0,
                    slot_id=lane_slot_id(worker_id, selfplay.games_in_flight, 0),
                    shm_name=shm_name,
                    shard_dir=shard_dir,
                    games=games,
                ),
                *extra_lanes,
            ),
            worker_id,
            incarnation,
            run_seed,
            selfplay,
            inference,
        )
        return
    client = SlotClient(shm_name, worker_id, inference)
    try:
        writer = ShardWriter(shard_dir, SCHEMA)
        try:
            # The single-game path's one context, shared by both seams for the
            # reason `_run_lane` states (KOM-159).
            context = SearchContext()
            SelfPlayWorker(
                client,
                selfplay,
                WorkerConfig(
                    worker_id=worker_id,
                    incarnation=incarnation,
                    run_seed=run_seed,
                    games=games,
                    shard_dir=shard_dir,
                ),
                writer,
                _wave_evaluator_for(client, selfplay, inference, context),
                search_context=context,
            ).run()
        finally:
            writer.close()
    finally:
        client.close()
