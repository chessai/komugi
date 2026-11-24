"""What one self-play move becomes on disk, and the seeds that produced it.

Three things live here, and they are together because they are all "the shape
of the data a worker emits" rather than "the loop that emits it":

* :data:`SCHEMA` -- the sample record of design D4, declared once as a
  :class:`komugi.train.shards.Schema`. One row per move. Shards hold
  ``float32`` and ``int64`` and nothing else (design F4), so every provenance
  field is an ``int64`` even where a ``uint8`` would do.
* :class:`ConcatSource` -- the adapter that lets one replay-window iteration
  span several per-worker shard directories (design D10/F5).
* :func:`game_seed` / :func:`move_seed` -- the whole seed derivation (design
  D11/F10). No :func:`hash` anywhere: ``PYTHONHASHSEED`` randomises ``str``
  and ``bytes`` hashing per interpreter, so a hash-derived seed would silently
  differ between a worker and its own respawn.

Two constants are *pinned* here, meaning other modules cite this file rather
than restating the number: :data:`MLH_SCALE` (design F15/Q3) and the field set
of :data:`SCHEMA` itself, whose fingerprint is what makes a shard belong to a
data generation.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Final

import numpy as np
import torch

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.observation import (
    CONTROL_SIZE,
    MAX_TURNS,
    PLANE_MAP_VERSION,
    observation_size,
)
from komugi.train.shards import FieldSpec, Schema
from komugi.train.window import SampleSource

MLH_SCALE: Final[float] = 1.0 / MAX_TURNS
"""The one place the moves-left target's scale is pinned (design F15/Q3).

``mlh_target = (plies until adjudication) * MLH_SCALE``, so a label of ``1.0``
means "a full ``Game.max_turns()`` of play still to come". The scale is
``1 / MAX_TURNS`` -- derived from the engine's turn cap
(:data:`komugi.engine.observation.MAX_TURNS`) rather than restating it, which
makes the target live in ``[0, 1]`` for every game the engine can produce and
moves with the cap (KOM-165 raised it to 512).

"Until adjudication", not "until a decisive result": P0a has no resign, so a
game that reaches the cap is labelled with the plies until the *engine's*
turn-cap draw, which is what actually happened at the board. Resign (P0b)
revisits that definition, and doing so is a data-generation change -- the MLH
head and the trainer cite this constant instead of re-deriving it, so there is
exactly one line to change and one fingerprint to bump. ``termination`` is now
the companion that lets a reader tell which definition a shard was written
under.
"""

SHORT_TERM_HORIZONS: Final[tuple[int, int, int]] = (4, 16, 64)
"""The three horizons, in plies, of the short-term value target (KOM-005b §1.3).

``short_term_value[k]`` is the value of the position ``SHORT_TERM_HORIZONS[k]``
plies after the row, in the row's own mover's perspective, clamped to the
game's result past the end of the buffer. Pinned here beside :data:`MLH_SCALE`
and for the same reason: the writer and the head must not each hold their own
copy of a number that defines the target. The head's *width* is 3 because this
tuple is 3.

**Why 4 / 16 / 64.** Four plies is about one exchange (a tsuke and its
answer); 16 is a plan; 64 is a fraction of the engine's turn cap, the number
:data:`MLH_SCALE` is built from (a quarter of it when these horizons were
frozen at a 256-ply cap; KOM-165's 512 does not move them -- they are plies,
not cap fractions). Geometric rather than linear spacing so the
three are not three nearly-identical targets -- at ~140-ply games, horizons of
4/8/12 would correlate above 0.9 and cost three head outputs to learn one
function. Frozen before any data exists: a later change is a new
pre-registration and a fingerprint bump, not a re-analysis.
"""

_OBSERVATION_SHAPE: Final[tuple[int, ...]] = observation_size()
"""``(in_planes, 9, 9)``, read from the encoder rather than restated (128 today)."""

_WDL_SIZE: Final[int] = 3
"""Win, draw, loss -- the target's three components, mover-relative per row."""

_CONTROL_SIZE: Final[int] = CONTROL_SIZE
"""``81`` -- one square per cell of the final-control map, read from the
encoder (:data:`komugi.engine.observation.CONTROL_SIZE`) rather than restated,
so the field and the map it holds cannot disagree about the board's size."""

SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="observation", dtype="float32", shape=_OBSERVATION_SHAPE),
        FieldSpec(name="policy_target", dtype="float32", shape=(ACTION_SIZE,)),
        FieldSpec(name="wdl_target", dtype="float32", shape=(_WDL_SIZE,)),
        FieldSpec(name="mlh_target", dtype="float32", shape=()),
        FieldSpec(name="policy_kl", dtype="float32", shape=()),
        FieldSpec(name="root_value", dtype="float32", shape=()),
        FieldSpec(name="search_mode", dtype="int64", shape=()),
        FieldSpec(name="to_play", dtype="int64", shape=()),
        FieldSpec(name="ply", dtype="int64", shape=()),
        FieldSpec(name="game_id", dtype="int64", shape=()),
        FieldSpec(name="sims_used", dtype="int64", shape=()),
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
        FieldSpec(name="final_control", dtype="int64", shape=(_CONTROL_SIZE,)),
        FieldSpec(name="action", dtype="int64", shape=()),
        FieldSpec(name="termination", dtype="int64", shape=()),
        FieldSpec(
            name="short_term_value",
            dtype="float32",
            shape=(len(SHORT_TERM_HORIZONS),),
        ),
        FieldSpec(name="draft_outcome", dtype="float32", shape=(_WDL_SIZE,)),
    )
)
"""One self-play move, as design D4 specifies it. Field order is part of the
schema's fingerprint, so this tuple is not to be re-ordered for tidiness.

* ``observation`` -- the encoder's ``(in_planes, 9, 9)`` planes.
* ``policy_target`` -- :attr:`~komugi.search.gumbel.SearchResult.policy_target`
  verbatim; nonzero on every legal action.
* ``wdl_target`` -- the **pure** game outcome (win, draw, loss) in the
  perspective of the player who moved in *this* row. No contempt: design D1
  keeps ``draw_score`` out of the targets entirely.
* ``mlh_target`` -- plies to adjudication, scaled by :data:`MLH_SCALE`.
* ``policy_kl`` -- ``KL(policy_target || masked-softmax(root prior logits))``,
  the four bytes that buy ``DESIGN.html`` §4's surprise weighting without
  storing an ``(A,)`` prior vector per row (design D4/F2).
* ``root_value`` -- ``v_mix`` at the root. **Diagnostics only** (F14): the
  value target is ``wdl_target``.
* ``search_mode`` -- ``0`` full, ``1`` fast (design F1). The key the P0b
  down-weight-versus-discard A/B is stratified on, which is why every fast row
  is recorded rather than dropped now.
* ``to_play``, ``ply``, ``game_id``, ``sims_used`` -- provenance.
* ``plane_map_version`` -- :data:`komugi.engine.observation.PLANE_MAP_VERSION`
  at the moment the row was written (KOM-002 §4). **Appended, not inserted**,
  for the reason the paragraph above gives. Eight bytes against a 41,472-byte
  observation, and they buy the one break the fingerprint provably cannot see:
  the fingerprint is a function of names, dtypes and shapes, so a future ticket
  that fills one of the encoder's reserved planes 121-127 changes what a row
  MEANS without changing what it looks like. :meth:`ConcatSource.__getitem__`
  checks the value on the read side.
* ``final_control`` -- the top-of-tower ownership map of the game's **final**
  position (KOM-010), ``81`` squares indexed ``s = r * 9 + (8 - f)`` like a
  flattened observation plane, and **mover-relative per row**: ``0`` = the
  row's own mover controls that square at the end, ``1`` = the opponent does,
  ``2`` = the square is empty, ``3`` = **unlabelled**. The engine states it
  absolutely
  (:func:`komugi.engine.observation.final_control`, ``0`` white / ``1`` black)
  and :func:`komugi.selfplay.worker.finalize_rows` flips it per row off the
  row's recorded ``to_play``, exactly as it flips ``wdl_target`` -- so every
  target in a row shares one perspective, and the ownership head (KOM-005,
  staged for P0b) never has to know which colour played it. **Appended, not
  inserted**: field order is the fingerprint. Class ``3`` is reserved for
  resign v2 (KOM-005b D5) and means *the rest of the game was never played*, so
  the square is a claim about a board that does not exist: the ownership loss
  turns it into an all-zero target row and it contributes exactly nothing.
  Widening the value set is invisible to the fingerprint -- which is a feature
  here, since no existing shard contains the class, and the hazard
  ``termination`` below is the answer to. P0b's worker never writes it.
* ``action`` -- the action id the worker actually **played** on this row
  (KOM-014 §3.1/B2). Everything else in a row describes the *search*; this is
  the one field that describes the *game*, and without it a shard cannot be
  replayed at all: ``policy_target`` says what the search wanted, not what
  happened next, so the position a row's successor was recorded at is
  unreachable from the row. The arena already made this argument in its own
  words -- it keeps :attr:`~komugi.eval.arena.GameOutcome.moves` because *"a
  game is otherwise unreproducible except by replaying the whole search"*
  (:mod:`komugi.eval.arena`) -- and a draft book
  (:mod:`komugi.selfplay.book`) is precisely a demand to replay a *fragment*
  of a game without re-searching it. Eight bytes on a ~190 KB row.
  **Appended, not inserted**, for the reason the paragraph above gives:
  field order is the fingerprint, so every pre-existing field keeps its
  position and this change is a clean data-generation bump.
* ``termination`` -- what ended the game (KOM-005b §3.2). ``0`` decisive by the
  engine's adjudication, ``1`` draw (the engine's turn cap included), ``2``
  **reserved for resign v2 and never written by P0b's worker**. Game-level:
  every row of one game carries the same value. It is 8 bytes and it buys the
  auditable half of the staged heads' label rules -- "ownership is withheld on
  resigned games" becomes a query over a column rather than a claim about a
  worker version -- and it covers the same hazard ``plane_map_version`` covers
  for the planes: extending ``final_control``'s value set with class ``3``
  changes what a row MEANS while leaving names, dtypes and shapes untouched, so
  the fingerprint provably cannot see it and this column is how a post-hoc
  reader tells a pre-resign shard from a post-resign one. *Which* draw rule
  fired (turn cap, repetition, another engine rule) is deliberately not
  distinguished: no head branches on it and the engine does not expose the
  predicate. **Appended, not inserted.**
* ``short_term_value`` -- the value at each of
  :data:`SHORT_TERM_HORIZONS`, mover-relative to *this* row and clamped to the
  game's result past the end of the buffer (KOM-005b §1.3.2). Materialised on
  the write side because it provably cannot be computed at training time:
  batches are drawn one row at a time from a window, so row ``i + 4`` of the
  same game is not reachable from row ``i``. It needs no post-game pass either
  -- :func:`komugi.selfplay.worker.finalize_rows` holds the whole buffer.
  Past-the-end is a **label, not a gap**: the value of a position after the
  game ended is the game's result. **Appended, not inserted.**
* ``draft_outcome`` -- ``wdl_target`` on rows whose position was in the draft
  phase, all zeros elsewhere (KOM-005b §1.4.2). The zeros *are* the mask: a
  target with no mass contributes exactly zero through the cross-entropy's
  zero-mass guard, so the draft-outcome head needs no per-head sample weight
  and no branch in the assembler. Draft-ness is
  :meth:`komugi.engine.api.Game.in_draft` captured at record time -- true while
  **either** side retains drafting rights -- not re-derived in the trainer,
  which must not learn what a self-play row is. **Appended, not inserted.**

The last three land in **one** fingerprint bump, not three: a shard whose
fingerprint does not match the running ``SCHEMA`` is skipped with a warning
rather than refused, so each bump silently shortens the replay window, and
three bumps would cost that three times.

Deliberately absent: ``visit_counts`` (an ``(A,)`` ``int64`` is ~8x the
observation and no consumer wants it -- F17) and any soft-policy target (the
trainer derives it).
"""


def _check_plane_map_version(sample: dict[str, torch.Tensor]) -> None:
    """Raise if `sample` was encoded under a different plane map (KOM-002 §4).

    This is the read-side half of the reserved-plane mitigation, and it exists
    because :attr:`~komugi.train.shards.Schema.fingerprint` **cannot** catch the
    case: the fingerprint is a ``sha256`` over field names, dtypes and shapes,
    so filling one of the encoder's reserved planes 121-127 changes what a row
    means while leaving every one of those three unchanged. A shape change --
    ``(118, 9, 9)`` to ``(128, 9, 9)``, say -- is already refused by the reader,
    loudly and with both fingerprints named; this covers the other half.

    Checked on materialisation rather than at open time on purpose: a source is
    any :class:`~komugi.train.window.SampleSource`, including ones with no shard
    behind them, so the row itself is the only place the version is guaranteed
    to be readable. A row that predates the field entirely is refused too --
    absence is not agreement.
    """
    value = sample.get("plane_map_version")
    if value is None:
        raise ValueError(
            "sample carries no 'plane_map_version' field; it was written "
            f"before KOM-002 added it, under a plane map that is not this "
            f"process's (version {PLANE_MAP_VERSION}). Regenerate the shard."
        )
    version = int(value.item())
    if version != PLANE_MAP_VERSION:
        raise ValueError(
            f"sample was encoded under plane map version {version}, but this "
            f"process encodes version {PLANE_MAP_VERSION}; the observation's "
            "shape can be identical across a plane-map change (the reserved "
            "planes exist for exactly that), so the schema fingerprint cannot "
            "see this. Regenerate the shard."
        )


class ConcatSource:
    """Several :class:`~komugi.train.window.SampleSource` s read as one.

    Design D10 gives every worker its **own** shard directory, because
    :class:`~komugi.train.shards.ShardWriter` enforces one writer per
    directory; design F5 observes that
    :class:`~komugi.train.window.RecencyWindow` takes exactly one source per
    :class:`~komugi.train.window.IterationRef`. This class is the join: one
    iteration of a 16-worker run is one ``ConcatSource`` over 16
    :class:`~komugi.train.shards.ShardReader` s, and the window is none the
    wiser.

    Order is preserved -- source 0's samples first, then source 1's -- so a
    flat index is stable for as long as the source list is. Empty sources are
    allowed and simply contribute nothing.

    Nothing is read at construction: this holds references and prefix sums, so
    building one over a whole window costs no memory beyond the readers
    themselves.

    Args:
        sources: The sources to concatenate, in order.
    """

    def __init__(self, sources: Sequence[SampleSource]) -> None:
        self._sources = tuple(sources)
        offsets = [0]
        for source in self._sources:
            offsets.append(offsets[-1] + len(source))
        self._offsets = tuple(offsets)

    @property
    def sources(self) -> tuple[SampleSource, ...]:
        """Return the underlying sources, in order."""
        return self._sources

    def __len__(self) -> int:
        """Return the total number of samples across every source."""
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return sample `index`, counting across the sources in order.

        Negative indices count from the end, matching
        :class:`~komugi.train.shards.ShardReader`. Anything outside the range
        raises ``IndexError`` -- the window samples by index and a silent wrap
        would train on the wrong iteration's data.

        Raises ``ValueError`` if the materialised row's ``plane_map_version``
        is not the running encoder's -- see :func:`_check_plane_map_version`.
        """
        total = len(self)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"sample index {index} out of range for {total} samples")
        # Rightmost offset <= index. bisect_right rather than a scan so a
        # window over many workers stays O(log n) per draw; the duplicate
        # offsets an empty source produces resolve to the source *after* it,
        # which is the one that actually holds the sample.
        slot = bisect.bisect_right(self._offsets, index) - 1
        sample = self._sources[slot][index - self._offsets[slot]]
        _check_plane_map_version(sample)
        return sample


def game_seed(
    run_seed: int, worker_id: int, incarnation: int, game_index: int
) -> np.random.SeedSequence:
    """Return the seed sequence owning one game's randomness (design D11/F10).

    ``SeedSequence(run_seed, spawn_key=(worker_id, incarnation, game_index))``
    -- the entropy is the run's, and the position in the run is the spawn key,
    which is what makes two games in the same run independent by construction
    instead of by luck.

    ``incarnation`` is in the key for a specific failure (F10c): design D10
    respawns a worker that goes silent, and a respawned worker that re-derived
    from ``(worker_id, game_index)`` alone would replay the games its
    predecessor already played, quietly duplicating positions in the window.

    Nothing here hashes a string, and nothing reads the clock: this function is
    a pure function of four integers, so ``(run_seed, worker_id, incarnation,
    game_index)`` names a game forever. What that *buys* is bounded by design
    D11's honest replay claim -- ``--serial-debug`` games are bit-reproducible,
    production multi-worker games are reproducible only up to evaluator
    tolerance, because batch composition varies.
    """
    return np.random.SeedSequence(
        run_seed, spawn_key=(worker_id, incarnation, game_index)
    )


def move_seed(game_seed: np.random.SeedSequence, ply: int) -> int:
    """Return the Gumbel seed for one move of the game `game_seed` names.

    Per-move, not per-game, and design F10d says why: the search draws its
    Gumbel vector once per call from
    :attr:`~komugi.search.gumbel.GumbelConfig.seed`, so a whole game played on
    one seed would draw the *identical* noise vector at every ply. The same
    action ids would win the same top-``m`` tie-breaks all game long, which is
    a systematic exploration bias -- one lucky action id oversampled for a
    whole game -- rather than the extra noise it looks like.

    Derivation is by spawn-key composition, ``(*game_seed.spawn_key, ply)``,
    rather than by :meth:`numpy.random.SeedSequence.spawn`: ``spawn`` is
    *stateful* (it advances ``n_children_spawned``), so asking twice for "the
    seed of ply 3" would answer differently the second time and the
    serial-debug replay claim would be false. This is a pure function of its
    arguments.

    Args:
        game_seed: The game's sequence, from :func:`game_seed`.
        ply: Zero-based ply index within that game.

    Returns:
        A 64-bit seed for :attr:`~komugi.search.gumbel.GumbelConfig.seed`.
    """
    child = np.random.SeedSequence(
        entropy=game_seed.entropy, spawn_key=(*game_seed.spawn_key, ply)
    )
    return int(child.generate_state(1, dtype=np.uint64)[0])


BOOK_BUILD_DOMAIN: Final[int] = 0x9014_0001
"""Spawn-key domain for the draft-book **builder**'s generator (KOM-014 §2.3)."""

BOOK_SELECT_DOMAIN: Final[int] = 0x9014_0002
"""Spawn-key domain for a worker's per-game **book draw** (KOM-014 §4.2).

Two constants rather than one because the two streams answer different
questions -- "which prefixes exist this generation" and "does *this* game start
from one" -- and a single domain would make the second a continuation of the
first.
"""


_MAX_DRAFT_THRESHOLD: Final[int] = 25
"""Highest threshold a draft can reach -- a hand (``komugi.config``'s own
``_MAX_MIN_PLACEMENTS``). Restated rather than imported for the reason
:func:`book_seed` restates the seed construction: :mod:`komugi.config` is
configuration and this module is the seed discipline, and the edges this
function is handed are already validated there -- the check below is the
function's own guarantee as a *pure* callable the offline census invokes with
no ``SelfPlayConfig`` in sight."""

DRAFT_THRESHOLD_DOMAIN: Final[int] = 0x9246_0001
"""Spawn-key domain for a game's **draft-threshold draw** (KOM-246 §1).

Composed exactly as the book domains are, ``(*game_seed.spawn_key,
DRAFT_THRESHOLD_DOMAIN)``, and disjoint from every other stream a game owns for
two separate reasons: from :func:`move_seed`'s ``(*key, ply)`` because no ply
index the engine can reach comes near ``0x9246_0001`` (the engine adjudicates
at its turn cap), and from :data:`BOOK_BUILD_DOMAIN` /
:data:`BOOK_SELECT_DOMAIN` by value.

The whole scheme rests on **every key element being < 2**32**: numpy splits a
larger integer into two 32-bit words, so a key of ``(worker_id, incarnation,
(DOMAIN << 32) | game_index)`` would hash exactly like ``(worker_id,
incarnation, game_index, DOMAIN)`` and the "disjoint by composition" argument
would be false. :func:`komugi.selfplay.worker.game_id` already bounds each
component below ``1e6``, and this constant is itself below ``2**32``.
"""


def draft_thresholds(
    run_seed: int,
    worker_id: int,
    incarnation: int,
    game_index: int,
    lo: int,
    hi: int,
) -> tuple[int, int]:
    """Return one game's ``(N_w, N_b)`` draft thresholds -- WHITE FIRST.

    KOM-246: each side's DRAFTMIN threshold is drawn per game and per colour,
    ``Uniform{lo .. hi}`` inclusive and independent, so the population stops
    sitting at one draft length (campaign 6's white-closes-at-12 attractor).

    **Why a dedicated stream.** The game's ``rng`` (the full/fast draw) and
    :func:`move_seed` (the per-ply Gumbel seed) are a *replay claim* -- design
    D11/F10 -- so taking two integers out of either would shift every
    subsequent draw of every game and make a run at fixed seed unreproducible
    against the runs before it. This draw composes its own domain instead
    (:data:`DRAFT_THRESHOLD_DOMAIN`), the way the book draw does, and touches
    neither.

    **Why ``PCG64`` is named** rather than left to ``default_rng``: the pair of
    any recorded game is recomputed OFFLINE by the census, in another process
    -- ``game_id`` decodes to ``(worker_id, incarnation, game_index)`` via
    :func:`komugi.selfplay.worker.unpack_game_id` and the worker's ``run_seed``
    is :func:`komugi.campaign.loop.iteration_seed` of the campaign's -- and
    ``default_rng``'s algorithm is documented as an implementation choice
    (:data:`komugi.selfplay.book.BIT_GENERATOR`, R8). A pure function of six
    integers, with no state and no clock, is what makes that recomputation an
    answer rather than a guess.

    **The asymmetry** (design review F1): black's 終 ends BOTH drafts, white's
    ends only its own. So ``N_w`` is a floor on white's *own* close and nothing
    more -- whenever black closes first, white's draft ends at whatever it had
    placed, whatever ``N_w`` says. Only ``N_b`` bounds a full draft from both
    sides, and a census must read the pair conditionally, never as two
    independent floors.

    Args:
        run_seed: The run's entropy, as :func:`game_seed` takes it.
        worker_id: The worker within the run.
        incarnation: The worker's incarnation (design D10 respawns).
        game_index: Index within that worker's workload.
        lo: Low edge, inclusive -- ``SelfPlayConfig.min_placements``.
        hi: High edge, inclusive -- ``SelfPlayConfig.min_placements_max``.

    Returns:
        ``(N_w, N_b)``: white's threshold first, black's second, the same
        ``0 white / 1 black`` order
        :func:`komugi.engine.game.draft_root_mask` reads ``board.turn`` in.

    Raises:
        ValueError: Unless ``1 <= lo <= hi <= 25``. ``lo == 0`` is refused
            because 0 means the mask is OFF, and a range that includes it would
            switch the feature off for a share of seats rather than draw a
            threshold for them.
    """
    if not 1 <= lo <= hi <= _MAX_DRAFT_THRESHOLD:
        raise ValueError(
            f"draft threshold edges must satisfy 1 <= lo <= hi <= "
            f"{_MAX_DRAFT_THRESHOLD}, got lo={lo!r}, hi={hi!r}: 0 is the mask "
            f"switched off rather than a threshold, an empty range draws "
            f"nothing, and an edge above a hand is a threshold no draft can "
            f"reach"
        )
    seeds = game_seed(run_seed, worker_id, incarnation, game_index)
    gen = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                entropy=seeds.entropy,
                spawn_key=(*seeds.spawn_key, DRAFT_THRESHOLD_DOMAIN),
            )
        )
    )
    n_w, n_b = gen.integers(lo, hi + 1, size=2)
    return (int(n_w), int(n_b))


def book_seed(run_seed: int, domain: int, iteration: int | None = None) -> int:
    """Return a draft-book seed, derived the way every other campaign seed is.

    ``SeedSequence(run_seed, spawn_key=(domain[, iteration]))`` reduced to one
    32-bit word -- :func:`komugi.campaign.probe.probe_seed`'s construction
    verbatim, which is itself
    :func:`komugi.campaign.loop.iteration_seed`'s with a domain constant in
    front. Copied rather than imported because :mod:`komugi.campaign` already
    imports :mod:`komugi.selfplay`, and the reverse edge would be a cycle; the
    *convention* lives here, beside :func:`move_seed`, which is where the
    spawn-key discipline is documented.

    **Why a domain constant cannot collide with a ply.** :func:`move_seed`
    composes ``(*game_seed.spawn_key, ply)``, and the engine adjudicates at its
    turn cap (:meth:`komugi.selfplay.adapter.GameState.is_terminal`), so
    every reachable ``ply`` is far below ``0x9014_0001``. A book draw taken on
    a ``0x9014_xxxx`` key therefore cannot land on a per-move Gumbel stream, at
    any ply the engine can produce.

    Args:
        run_seed: The run's entropy -- for the builder this is the campaign's
            ``run_seed``; for a worker it is whatever seed that worker was
            handed (see KOM-014 §6.7 on the two-level derivation).
        domain: :data:`BOOK_BUILD_DOMAIN` or :data:`BOOK_SELECT_DOMAIN`.
        iteration: The generation the seed belongs to, or ``None`` for a
            campaign-lifetime stream.

    Returns:
        One ``uint32`` word, as an ``int``.
    """
    key = (domain,) if iteration is None else (domain, iteration)
    child = np.random.SeedSequence(run_seed, spawn_key=key)
    return int(child.generate_state(1, dtype=np.uint32)[0])


def num_actions() -> int:
    """Return the engine's action-space size ``A`` (37341 today).

    Read from :mod:`komugi.engine.action_encoding` rather than restated, and
    read from the *same* constant :data:`SCHEMA`'s ``policy_target`` is shaped
    by: a change to the action encoding is then one schema-fingerprint change
    and one :attr:`~komugi.search.gumbel.GumbelConfig.num_actions` change that
    cannot disagree with each other.
    """
    return ACTION_SIZE
