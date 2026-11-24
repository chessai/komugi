"""How a campaign's games *ended*, read off the shards it already wrote.

The previous architecture died of a draw spiral, and the campaign that
replaced it could not see one coming: :class:`komugi.telemetry.records.IterationMetrics`
carried ``draw_rate`` and ``avg_game_len`` as literal ``0.0`` ("the
orchestrator does not measure them", :mod:`komugi.campaign.loop`) and
``selfplay.jsonl`` carried games, rows, restarts and wall time. An operator
asking "what share of iteration 11's games were draws, and how long were they"
had to read the shard tree by hand -- so nobody asked, which is exactly how a
spiral gets a hundred iterations of head start.

Nothing new is measured here. Every number below is **already on disk**, in
four ``int64``/``float32`` columns every self-play row has carried since
KOM-005b:

* ``game_id`` -- the packed ``(worker, incarnation, game_index)`` triple
  (:func:`komugi.selfplay.worker.game_id`), unique across an iteration's whole
  shard tree, so grouping by it is grouping by game;
* ``ply`` -- the row's own zero-based ply;
* ``mlh_target`` -- ``(total_plies - ply) * MLH_SCALE``
  (:func:`komugi.selfplay.worker.finalize_rows`), which makes
  ``ply + mlh_target / MLH_SCALE`` **the game's total length, readable from any
  one of its rows**. Read at the row with the largest ply, where the remaining
  count is smallest and the ``float32`` therefore carries the most precision;
* ``termination`` -- ``0`` decisive, ``1`` draw (the engine's ply turn cap
  included), ``2`` reserved for resign v2 and never written by P0b. Game-level:
  every row of one game carries the same value.

The turn cap is not distinguishable from another draw rule by ``termination``
alone ("*which* draw rule fired is deliberately not distinguished",
:data:`komugi.selfplay.records.SCHEMA`) -- so it is read off the **length**
instead: a game whose total plies reach :data:`TURN_CAP_PLIES` is a game the
engine adjudicated because it ran out of turns, and that is the number that
separates "the players are shuffling" from "the cap is refereeing".

The draft, which is the other attractor (KOM-156b)
--------------------------------------------------
The four columns above cannot see the *draft*, and campaign 2 died in it: by
iteration 14 the median side placed **four** pieces and played 終, exactly the
attractor v1 met and answered with a draft-minimum curriculum ramped on its
own early-終 rate. Komugi measured none of it -- the diagnosis was made by hand
off the shards -- so the arithmetic is made permanent here, beside the game
shape, and stamped per iteration:

* a row is a **draft move by its mover** iff the mover's own drafting plane is
  set (``to_play`` 0 reads plane :data:`DRAFT_PLANE_WHITE`, 1 reads
  :data:`DRAFT_PLANE_BLACK`);
* a side's **placements** in a game are its draft rows there, the 終 row
  included -- a 終 places a piece and ends the draft in one action
  (:data:`~komugi.engine.action_encoding.SHU_BLOCK`);
* a game is an **early 終** iff at least one side's draft *ended* -- its last
  draft row's action is in that block, which is a 終 precisely because the
  mover was still drafting -- on fewer than :data:`EARLY_SHU_PLACEMENTS`
  placements.

**Booked games read low, and this is not fixable from here.** A game played
from a draft-book prefix records rows only from the ply the book left off at
(:func:`komugi.campaign.loop.book_for_iteration`), so the placements the book
made are not on disk and are not counted: the medians under-report by the
booked fraction, and an iteration that is entirely booked reports zero
placements for games that drafted fully. The rate to read alongside them is
``games`` and the book's own depth.

Cheap on purpose, except once
-----------------------------
A production shard row is ~190 KB, almost all of it ``observation`` and
``policy_target``, and an iteration is gigabytes. The game-shape half never
touches either: it goes through :func:`komugi.train.shards.read_columns`, the
projecting reader, so a whole iteration's census costs the four small columns
and nothing else. numpy throughout -- no torch tensor is ever materialised --
which is what makes it usable from a boundary that has a net on the GPU and
from a backfill script over a dead campaign's root alike.

The draft half **does** read ``observation``, because the drafting bit exists
nowhere else, and it reads it for every shard: there is no cheap filter that is
also exact (:func:`_draft_rows` says why both candidates fail). It is the one
expensive read here, it is bounded to one shard at a time, and it costs seconds
per iteration against a self-play run of minutes.

What it gives up, stated rather than hidden: a projection **cannot** verify a
shard's payload digest (:func:`~komugi.train.shards.read_columns` says why), so
this is a digest-unverified reader. Every shard it reads is digest-verified by
the training window built over the same tree in the same iteration, which is
the ground KOM-014 §2.4 already accepts for the book builder. A shard that
cannot be read at all is **skipped with a** :class:`~komugi.train.shards.ShardWarning`
rather than raised on: this census is a diagnostic, and a diagnostic that
aborts an iteration because one shard rotted is worse than one that reports the
other 999.
"""

from __future__ import annotations

import os
import warnings
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from komugi.engine.action_encoding import SHU_BLOCK
from komugi.selfplay.records import MLH_SCALE
from komugi.terminations import TERMINATION_DRAW
from komugi.train.shards import SHARD_GLOB, Schema, ShardWarning, read_columns

STAT_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "ply",
    "mlh_target",
    "termination",
)
"""The only four fields the *game-shape* half of this module reads. Four
``int64``/``float32`` scalars per row, against a ~190 KB row: the projection is
the whole point."""

DRAFT_COLUMNS: Final[tuple[str, ...]] = (
    "to_play",
    "action",
    "observation",
)
"""The three fields the **draft** half reads, and the one expensive read here.

``to_play`` and ``action`` are ``int64`` scalars; ``observation`` is the
encoder's ``(114, 9, 9)`` float32 block, ~37 KB per row, and it is read for
every shard (see :func:`_draft_rows` for why there is no cheap exact way to
skip it). It is reduced to **two booleans per row** the instant it is in hand
and dropped, so the peak cost is one shard's observations, not an iteration's.

A shard whose own embedded schema does not declare all three -- the four-column
census schema the unit tests write, or any future projection tree -- simply
contributes no draft rows. That is not damage and is not warned about.
"""

WDL_COLUMN: Final[str] = "wdl_target"
"""The label column the cap-adjudication half of this module reads (KOM-245).

Three ``float32`` per row, and read **per shard rather than per tree**: a shard
whose own embedded schema does not declare it -- the four-column census schema
the unit tests write, and every shard written before the column existed --
simply contributes no soft labels, exactly as :func:`_draft_rows` tolerates a
projection that carries no observations. That is not damage and is not warned
about; those games read as ``p = 0``, which is what an un-adjudicated cap game
*is*.
"""

CAP_P_THRESHOLD: Final[float] = 0.3
"""Where a cap game counts as *strongly* adjudicated, for K1's 41% share.

``p >= 0.3`` is ``|d| >= 3`` at the pre-registered ``p_max = 0.8`` /
``d_full = 8``, which is the lead the counterfactual census found in 41% of
campaign-6's cap games. Compared with a tolerance below, because a target is
stored as ``float32`` and ``0.3`` is not exact in either width.
"""

_CAP_P_TOLERANCE: Final[float] = 1e-6
"""Slack on :data:`CAP_P_THRESHOLD`, so a float32 ``0.3`` counts as ``0.3``."""

DRAFT_PLANE_WHITE: Final[int] = 112
DRAFT_PLANE_BLACK: Final[int] = 113
"""The encoder's drafting-rights planes: "1.0 iff <colour> is drafting"
(:mod:`komugi.engine.observation` fills 112 from ``board.draft_w`` and 113 from
``board.draft_b``).

The same two indices :mod:`komugi.selfplay.worker` reads the draft predicate
from, and for the same reason it gives: the engine class on this path exposes
no drafting accessor, and the planes are the bits that predicate would read
anyway. A remap of the plane map is the change this does not survive, and it is
what :data:`~komugi.engine.observation.PLANE_MAP_VERSION` exists to make loud.
"""

EARLY_SHU_PLACEMENTS: Final[int] = 10
"""Below how many placements a side's 終 counts as an *early* draft finish.

Ten because ten is v1's draft-minimum curriculum's first rung: v1 forced a
minimum of 10 placements, relaxed it to 7 and then to 4, and ramped each step
only while its early-終 rate stayed low. Campaign 2 reproduced the attractor
this measures -- by iteration 14 the median side placed **four** pieces and
ended the draft -- with no telemetry able to say so, because nothing here
counted placements. This is that count's threshold, stated once.
"""

TURN_CAP_PLIES: Final[int] = round(1.0 / MLH_SCALE)
"""The engine's turn cap in plies, derived rather than restated.

:data:`~komugi.selfplay.records.MLH_SCALE` is ``1 / cap`` *because* the cap is
what makes the moves-left target live in ``[0, 1]``, so the cap is recoverable
from it exactly. Deriving it here means a ticket that moves the cap (KOM-165
raised it 256 → 512) moves this census with it, instead of leaving a literal
behind in a second file to disagree with the first.
"""

_DAMAGE: Final[tuple[type[Exception], ...]] = (
    zipfile.BadZipFile,
    zlib.error,
    OSError,
    ValueError,
    KeyError,
    EOFError,
)
"""Everything a rotted shard can throw on the way back in.

The same set :mod:`komugi.train.shards` catches around its own reads and for
its reason: caught as a tuple rather than as bare ``Exception`` so a genuine
bug in this module still escapes as a traceback instead of being reported as
bit rot.
"""


@dataclass(frozen=True, slots=True)
class GameStats:
    """One shard tree's games, counted by how they ended and how long they ran.

    Attributes:
        games: Distinct ``game_id`` s in the tree.
        rows: Rows across every shard read, i.e. the sum of their ``__count__``
            sidecars. The cross-check against
            :attr:`komugi.campaign.telemetry.SelfPlayTelemetry.rows`: two
            numbers measured by different code over the same tree.
        draws: Games whose ``termination`` is
            :data:`~komugi.terminations.TERMINATION_DRAW` -- the turn cap
            included, because the engine adjudicates it as a draw.
        turn_cap_games: Games whose total plies reached the cap. A subset of
            the draws in P0b (a decisive result cannot be reached *by* running
            out of turns), and the number that says whether the draws are
            shuffling or the cap refereeing.
        mean_plies: Mean total plies over the games; ``0.0`` for no games.
        median_plies: Median total plies; ``0.0`` for no games. Carried beside
            the mean because a bimodal length distribution -- short decisive
            games and a pile of games at the engine's cap -- has a mean that describes
            neither mode.
        max_plies: The longest game's total plies; ``0`` for no games.
        median_placements_w: The median number of **draft rows white played**
            over the games, ``0.0`` for no games. A side's placements are its
            draft rows in that game -- the rows where its own drafting plane is
            set -- which **includes the 終 row**, because ending the draft is
            itself a placement (the 終 block places a hand piece *and* ends the
            draft, :data:`~komugi.engine.action_encoding.SHU_BLOCK`). A game
            with no draft rows at all (intro mode, or a booked prefix that
            swallowed the whole draft) contributes a ``0`` to the median rather
            than being excluded: the median is over *games*, and a campaign
            whose games have stopped drafting is exactly what this number has
            to be able to say.
        median_placements_b: The same for black.
        early_shu_games: Games in which **at least one side** ended its draft
            (its last draft row's action is in
            :data:`~komugi.engine.action_encoding.SHU_BLOCK`) having played
            fewer than :data:`EARLY_SHU_PLACEMENTS` placements. The draft
            attractor, counted: it is v1's ``p2_early_shu``, which is what its
            draft-minimum curriculum was ramped on. A side whose draft has
            simply not ended within the rows on disk is **not** counted -- an
            unfinished draft is not an early one.
        cap_adjudicated_games: Turn-capped games carrying a **soft** label,
            i.e. whose ``wdl_target`` has ``max(w, l) > 0`` (KOM-245). Under
            ``cap_adjudication = off`` -- and on every shard written before the
            column existed, or without it -- this is ``0``, because a plain cap
            draw is ``(0, 1, 0)``. A capped game whose sides were level is not
            counted either: ``d == 0`` produces no soft label.
        cap_p_mean: The mean ``p = max(w, l)`` over **all** turn-capped games,
            a plain ``(0, 1, 0)`` cap game counting ``0``; ``0.0`` for no cap
            games. Over all of them and not only the adjudicated ones, because
            the plan's pre-registered figure (0.25) is that denominator and a
            number that silently changed denominator would be unreadable.
        cap_p_ge_03_share: The share of turn-capped games whose ``p`` reaches
            :data:`CAP_P_THRESHOLD`; ``0.0`` for no cap games. **K1's gate**:
            the counterfactual census puts it at 41% on campaign 6.
    """

    games: int
    rows: int
    draws: int
    turn_cap_games: int
    mean_plies: float
    median_plies: float
    max_plies: int
    median_placements_w: float
    median_placements_b: float
    early_shu_games: int
    # Defaulted, so the census that predates KOM-245 -- and every caller and
    # test that builds a GameStats positionally -- is unchanged by the three
    # new numbers, and a tree with no label column reads them as the zeros they
    # honestly are.
    cap_adjudicated_games: int = 0
    cap_p_mean: float = 0.0
    cap_p_ge_03_share: float = 0.0

    @property
    def draw_rate(self) -> float:
        """Return the drawn share of the games, or ``0.0`` if there were none.

        An iteration that played nothing has no draw rate; reporting ``0.0``
        rather than raising is what lets a caller stamp this on a record
        unconditionally, and the ``games`` beside it is what keeps the zero
        from being read as "no draws".
        """
        return 0.0 if self.games == 0 else self.draws / self.games

    @property
    def cap_rate(self) -> float:
        """Return the turn-capped share of the games, ``0.0`` if there were none."""
        return 0.0 if self.games == 0 else self.turn_cap_games / self.games

    @property
    def early_shu_rate(self) -> float:
        """Return the early-終 share of the games, ``0.0`` if there were none.

        **The ramp gate.** A draft-minimum curriculum is relaxed a rung only
        while this stays low -- v1 held it below ``0.3`` -- so it is reported
        per iteration beside the count it is a rate of, for the reason
        :attr:`draw_rate` states: a rate over four games is not a measurement,
        and only the denominator says so.
        """
        return 0.0 if self.games == 0 else self.early_shu_games / self.games


EMPTY: Final[GameStats] = GameStats(
    games=0,
    rows=0,
    draws=0,
    turn_cap_games=0,
    mean_plies=0.0,
    median_plies=0.0,
    max_plies=0,
    median_placements_w=0.0,
    median_placements_b=0.0,
    early_shu_games=0,
    cap_adjudicated_games=0,
    cap_p_mean=0.0,
    cap_p_ge_03_share=0.0,
)
"""What a tree with no readable shard reads as: zeros, and no exception.

A campaign's iteration directory legitimately holds an empty worker directory
(a wedged incarnation created one and never filled it), and a backfill runs
over roots whose last iteration died before sealing anything. Both are
findings, not errors.
"""

_WHITE: Final[int] = 0
_BLACK: Final[int] = 1
"""The ``to_play`` codes, which are what decide *whose* draft plane a row's
drafting bit is read from (:mod:`komugi.selfplay.worker`)."""


@dataclass(frozen=True, slots=True)
class _DraftRows:
    """One shard's rows, reduced to the five columns the draft census needs.

    ``observation`` is **not** here: it is 37 KB a row and it is thrown away
    inside :func:`_draft_rows` the moment the two drafting bits are out of it.
    What survives per row is one bool.
    """

    game_id: npt.NDArray[Any]
    ply: npt.NDArray[Any]
    drafting: npt.NDArray[Any]
    to_play: npt.NDArray[Any]
    action: npt.NDArray[Any]


@dataclass(frozen=True, slots=True)
class _DraftCensus:
    """The three draft numbers, over a whole tree's games."""

    median_w: float
    median_b: float
    early_shu_games: int


_NO_DRAFT: Final[_DraftCensus] = _DraftCensus(
    median_w=0.0, median_b=0.0, early_shu_games=0
)
"""What a tree with no draft rows reads as: a campaign in intro mode plays no
draft at all, and zero placements is the true answer rather than a gap."""


def _draft_rows(
    path: Path, schema: Schema, columns: Mapping[str, npt.NDArray[Any]]
) -> _DraftRows | None:
    """Return `path`'s draft columns, or ``None`` if it carries none.

    **This is the expensive read of the module**, and it is unconditional by
    necessity rather than by choice. The cheap filters that suggest themselves
    are both unsound: a shard's ``action`` column holding a
    :data:`~komugi.engine.action_encoding.SHU_BLOCK` id does not imply a draft
    (block C is context-dependent -- the same id is a plain hand drop once the
    mover's draft bit is clear), and "rows at a low ply" is a guess at a draft
    length that the curriculum being measured is the thing that moves. Since
    the drafting predicate is *only* legible in the observation, every shard's
    observations are read. It costs seconds per iteration against a self-play
    run of minutes -- the same trade the advisor's by-hand census made -- and
    the block is reduced to two booleans per row here, so the memory high-water
    mark is one shard rather than an iteration.

    A shard whose own schema does not declare the three columns returns
    ``None`` **without a warning**: a projection tree (the unit suite's
    four-column census schema) is a legitimate shape, not damage. A shard that
    declares them and cannot produce them *is* damage and is warned about, the
    way an unreadable shard is -- and, as there, it costs its own rows and not
    the census.
    """
    if not set(DRAFT_COLUMNS).issubset(set(schema.names)):
        return None
    try:
        _schema, _count, draft = read_columns(path, DRAFT_COLUMNS)
    except _DAMAGE as error:
        warnings.warn(
            f"skipping unreadable draft columns of shard {path}: {error}",
            ShardWarning,
            stacklevel=3,
        )
        return None
    observation = draft["observation"]
    if observation.ndim < 2 or observation.shape[1] <= DRAFT_PLANE_BLACK:
        # An encoder with fewer planes than the draft map declares is not a
        # tree this reader can answer about; saying so beats reporting zero
        # placements as though the games had none.
        warnings.warn(
            f"shard {path} has {observation.shape[1:]!r} observations, which "
            f"do not reach draft plane {DRAFT_PLANE_BLACK}",
            ShardWarning,
            stacklevel=3,
        )
        return None
    planes = observation.reshape(observation.shape[0], observation.shape[1], -1)
    # The plane is filled with 1.0 across the whole board when the side is
    # drafting, so `any` over the square axis reads it however it is shaped.
    white: npt.NDArray[np.bool_] = np.any(planes[:, DRAFT_PLANE_WHITE] != 0.0, axis=1)
    black: npt.NDArray[np.bool_] = np.any(planes[:, DRAFT_PLANE_BLACK] != 0.0, axis=1)
    to_play = draft["to_play"]
    return _DraftRows(
        game_id=columns["game_id"],
        ply=columns["ply"],
        # A row is a draft move by **its own mover**: white's plane answers for
        # a white-to-play row and black's for a black one. The other side's
        # plane is set for as long as it is still drafting, which is exactly
        # the confusion that would make one side's placements the other's.
        drafting=np.where(to_play == _WHITE, white, black),
        to_play=to_play,
        action=draft["action"],
    )


def _label_p(path: Path, schema: Schema, count: int) -> npt.NDArray[np.float64]:
    """Return `path`'s per-row ``p = max(w, l)``, or all-zeros if it has none.

    KOM-245's stored label, reduced to the one number the cap census asks of
    it: the soft win/loss mass a row carries, which is ``0`` on a plain draw
    ``(0, 1, 0)`` and ``1`` on a decisive row. The *game*-level filtering (only
    turn-capped games count) happens in :func:`game_stats`, which is where the
    game lengths are.

    A shard whose schema does not declare :data:`WDL_COLUMN` -- the unit
    suite's four-column census schema, an older shard -- contributes zeros
    **without a warning**, for the reason :func:`_draft_rows` gives about its
    own columns: a projection is a legitimate shape, not damage. Zeros are also
    the honest answer there: no stored label is no adjudication.

    A shard that *declares* the column and cannot produce it is damage and is
    warned about, and costs its own rows rather than the census.
    """
    if WDL_COLUMN not in set(schema.names):
        return np.zeros(count, dtype=np.float64)
    try:
        _schema, _count, columns = read_columns(path, (WDL_COLUMN,))
    except _DAMAGE as error:
        warnings.warn(
            f"skipping unreadable label column of shard {path}: {error}",
            ShardWarning,
            stacklevel=3,
        )
        return np.zeros(count, dtype=np.float64)
    wdl = np.asarray(columns[WDL_COLUMN], dtype=np.float64)
    if wdl.ndim != 2 or wdl.shape[1] != 3:
        warnings.warn(
            f"shard {path} has {wdl.shape[1:]!r} labels, which are not the "
            "three-slot (win, draw, loss) target this census reads",
            ShardWarning,
            stacklevel=3,
        )
        return np.zeros(count, dtype=np.float64)
    # `max(w, l)`, not `1 - draw`: the two agree on every label this worker
    # writes, and the max is the one that stays right if a future label ever
    # fails to sum to one.
    return np.maximum(wdl[:, 0], wdl[:, 2])


def _draft_census(
    shards: Sequence[_DraftRows], games: npt.NDArray[Any]
) -> _DraftCensus:
    """Count placements and early 終 over `games`, in ascending game-id order.

    `games` is the census's own game list, so a game whose shards carried no
    draft columns -- and a game that simply never drafted -- contributes a
    ``0`` to each median rather than vanishing from the denominator.

    A game's rows are bucketed by ``game_id`` and ordered by ``ply``, so one
    side's **last** draft row is the last of its run: that row's action is what
    says whether the draft *ended* (a 終) or merely ran out of shard. A draft
    that never ended is not an early one, which is the case the ramp gate must
    not be allowed to fire on.
    """
    if not shards:
        return _NO_DRAFT
    index = np.searchsorted(games, np.concatenate([s.game_id for s in shards]))
    ply = np.concatenate([s.ply for s in shards])
    order = np.lexsort((ply, index))
    index = index[order]
    drafting = np.concatenate([s.drafting for s in shards])[order]
    to_play = np.concatenate([s.to_play for s in shards])[order]
    action = np.concatenate([s.action for s in shards])[order]

    shu = (action >= SHU_BLOCK[0]) & (action < SHU_BLOCK[1])
    early = np.zeros(games.size, dtype=bool)
    medians: list[float] = []
    for side in (_WHITE, _BLACK):
        rows = np.flatnonzero(drafting & (to_play == side))
        # Placements per game, the 終 row included: it places a piece too.
        placements = np.bincount(index[rows], minlength=int(games.size))
        medians.append(float(np.median(placements)))
        if rows.size == 0:
            continue
        of_game = index[rows]
        last = rows[
            np.concatenate(
                (of_game[1:] != of_game[:-1], np.ones(1, dtype=bool))
            )
        ]
        ended = shu[last]
        short = placements[index[last]] < EARLY_SHU_PLACEMENTS
        early[index[last][ended & short]] = True

    return _DraftCensus(
        median_w=medians[0],
        median_b=medians[1],
        early_shu_games=int(np.count_nonzero(early)),
    )


def game_stats(
    shard_dir: str | os.PathLike[str], *, cap_plies: int = TURN_CAP_PLIES
) -> GameStats:
    """Census the games under `shard_dir`, whatever level of the tree it names.

    Every ``shard-*.npz`` **at or below** `shard_dir` is read, so the same call
    works on one worker directory, on an iteration's ``shards`` root, or on the
    ``iter_<i>`` directory above it. Recursive because the layout between the
    iteration root and the worker directories is the loop's business, not this
    reader's (the rig's ``shard_dirs_under`` makes the same choice for the same
    reason).

    Args:
        shard_dir: A directory holding sealed shards, at any depth. One that
            does not exist reads as :data:`EMPTY`.
        cap_plies: The length at which a game counts as turn-capped, defaulting
            to the engine's own :data:`TURN_CAP_PLIES`. A parameter rather than
            a constant so a census of shards written under a *different* cap
            (a future ticket's, a synthetic tree's) can say so, and so the
            comparison is ``>=`` against a stated number rather than an
            equality against an assumed one.

    Returns:
        The :class:`GameStats` for those shards.
    """
    ids: list[npt.NDArray[Any]] = []
    plies: list[npt.NDArray[Any]] = []
    mlh: list[npt.NDArray[Any]] = []
    terminations: list[npt.NDArray[Any]] = []
    labels: list[npt.NDArray[np.float64]] = []
    draft: list[_DraftRows] = []
    rows = 0

    for path in sorted(Path(shard_dir).rglob(SHARD_GLOB)):
        try:
            schema, count, columns = read_columns(path, STAT_COLUMNS)
        except _DAMAGE as error:
            # Skipped, not raised: see the module docstring. The warning is
            # ShardWarning so a caller that would rather stop than census 90%
            # of a tree can escalate it, exactly as a trainer can.
            warnings.warn(
                f"skipping unreadable shard {path}: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        rows += count
        ids.append(columns["game_id"])
        plies.append(columns["ply"])
        mlh.append(columns["mlh_target"])
        terminations.append(columns["termination"])
        # Per shard and aligned with its own rows, so a tree that mixes a
        # labelled shard with a projection stays aligned (KOM-245).
        labels.append(_label_p(path, schema, count))
        shard_draft = _draft_rows(path, schema, columns)
        if shard_draft is not None:
            draft.append(shard_draft)

    if rows == 0:
        return EMPTY

    game_id = np.concatenate(ids)
    ply = np.concatenate(plies)
    remaining = np.concatenate(mlh).astype(np.float64)
    termination = np.concatenate(terminations)
    label_p = np.concatenate(labels)

    # Sort by (game_id, ply) so one game's rows are contiguous and its last row
    # -- the one with the largest ply -- is the last of its run. lexsort's last
    # key is the primary one.
    order = np.lexsort((ply, game_id))
    game_id = game_id[order]
    ply = ply[order]
    remaining = remaining[order]
    termination = termination[order]
    label_p = label_p[order]

    starts = np.concatenate(
        (np.ones(1, dtype=bool), game_id[1:] != game_id[:-1])
    )
    first = np.flatnonzero(starts)
    last = np.concatenate(
        (first[1:] - 1, np.array([game_id.size - 1], dtype=first.dtype))
    )

    # `ply + mlh / MLH_SCALE` at the deepest row of each game (the module
    # docstring's arithmetic). Rounded rather than truncated: mlh_target is a
    # float32 and int() of 59.999996 is 59.
    totals = np.rint(
        ply[last].astype(np.float64) + remaining[last] / MLH_SCALE
    ).astype(np.int64)
    # Termination is game-level, so any row of the game answers; the first is
    # taken because `first` is already in hand.
    drawn = termination[first] == TERMINATION_DRAW

    # The games, in ascending id order, are `game_id[first]` -- the key every
    # draft row is bucketed into below, so the two halves of this census count
    # over exactly the same set of games.
    census = _draft_census(draft, game_id[first])

    # KOM-245's cap labels. `p` is a GAME-level number -- every row of an
    # adjudicated game carries the same `max(w, l)`, the leader's as a win and
    # the trailer's as a loss -- so the first row of each game answers, the way
    # `termination` above does. A cap game with no soft label (mode off, a
    # level final board, a shard with no label column) reads `0` and stays in
    # the denominator, which is what makes `cap_p_mean` the plan's 0.25 rather
    # than the mean over the adjudicated subset.
    capped_games = totals >= cap_plies
    cap_p = label_p[first][capped_games]

    return GameStats(
        games=int(totals.size),
        rows=rows,
        draws=int(np.count_nonzero(drawn)),
        turn_cap_games=int(np.count_nonzero(totals >= cap_plies)),
        mean_plies=float(np.mean(totals)),
        median_plies=float(np.median(totals)),
        max_plies=int(totals.max()),
        median_placements_w=census.median_w,
        median_placements_b=census.median_b,
        early_shu_games=census.early_shu_games,
        cap_adjudicated_games=int(np.count_nonzero(cap_p > 0.0)),
        cap_p_mean=float(np.mean(cap_p)) if cap_p.size else 0.0,
        cap_p_ge_03_share=(
            float(
                np.count_nonzero(cap_p >= CAP_P_THRESHOLD - _CAP_P_TOLERANCE)
                / cap_p.size
            )
            if cap_p.size
            else 0.0
        ),
    )


def stats_for_iteration(
    root: str | os.PathLike[str],
    iteration: int,
    *,
    cap_plies: int = TURN_CAP_PLIES,
) -> GameStats:
    """Census iteration `iteration` of the campaign rooted at `root`.

    The path comes from :func:`komugi.campaign.loop.shard_root`, the loop's own
    helper, rather than from a second copy of ``iter_<i>/shards`` here: the
    layout is the loop's to state, and a reader holding its own copy of it is a
    reader that silently censuses an empty directory the day the loop moves.

    The import is deferred to call time because :mod:`komugi.campaign.loop`
    imports this module -- the boundary stamps these numbers on every
    iteration's record -- and a module-level import here would close that
    cycle.
    """
    from komugi.campaign.loop import shard_root

    return game_stats(shard_root(root, iteration), cap_plies=cap_plies)
