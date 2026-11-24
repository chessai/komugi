"""§D9's epoch driver: the one place every seam of the pool is composed.

``docs/design/kom-011d-ladder-pool.md`` §5's T4 row is this module: build the
entrants, call :func:`~komugi.eval.r5.assert_rated_run_permitted` **before any
game**, apply §D4.1's branch-3 roster refusal, play §D2.2's six-matchup
schedule (each matchup on its own seed via
:func:`~komugi.eval.ladder.matchup_seed`), pool the records, run
:func:`~komugi.eval.seats.check_paired_books` over the pooled set **before** the
fit, then ``pool_components`` + ``fit_elo`` **four times** + §D5.2's paired
bootstrap, evaluate both §D5 windows over ``D_no_external``, render §D9.1 and
write §D9.2.

The order is the design's and every step of it is a refusal placed *before* the
thing it protects:

* **R5 first, before a single game.** ``r5.py:750-764`` raises ``R5Missing`` on
  an absent artefact and ``RatedRunRefused`` on branch 3. Calling it after the
  games have been played would report the refusal against a GPU bill that was
  already paid, so it is the first statement of **both** entry points that
  play rated games -- :func:`run_epoch` and §D5.4's :func:`run_relink` -- and
  §6 node 17 asserts a game-counting spy sees **zero** calls when either
  raises. The rule is a property of the module, not of one function (T4 review
  F1): ``run_relink`` plays 2 x 128 rated games through the same seam, an
  epoch later, and the artefact can go stale in between.
* **Branch 3 changes the roster, not just the gauge.** All four v1 entrants
  leave the pool, the reference field drops from 7 to 3, and
  :func:`build_pool_roster` refuses to construct with a v1 member at all
  (§D4.1) rather than admitting them and discovering the refusal at the first
  game.
* **The paired-book audit runs before the fit.** A report-time audit is too
  late to stop a fit (§6 node 52): a ``GAME_COUNT`` violation means the record
  set is not the seat-balanced data ``elo.py:22-29``'s seat-free likelihood
  assumes, and a fit over it is a number nobody may publish.
* **Connectivity is checked, not hoped for.** A disconnected comparison graph
  still reports ``converged=True`` (``ab_endpoint.py:426-463``), so the fit
  cannot be its own detector, and the refusal is the **pool's own**
  (:func:`~komugi.eval.pool.require_connected_pool`) rather than KOM-036's
  cross-arm vocabulary.

**One code path for the observed statistic and its interval** (design §5 T4,
rev 4/F13; T4 review F4). The epoch's observed ``g_in`` / ``g_cross`` /
``g_cross_no_external`` / ``d`` / ``d_no_external`` are **read off the
:class:`~komugi.eval.divergence.DivergenceResult` the gate returned** -- the
object the verdict was computed from -- and are never recomputed here. The
merged T4 driver computed them a second time through its own
``gain_statistics`` seam and published that copy beside a verdict taken from
the other one; a one-character sign flip in the driver's own ``gain`` then put
a wrong-signed ``G_in`` in the record with a correctly-signed verdict beside
it, and the whole suite stayed green (the review's M2, executed and SURVIVED).
Two code paths is how a published ``d`` and the interval around it come to be
computed by different arithmetic, which is the class of defect §D5.1's *"never
merges the two record sets"* rule exists against.

**The games arrive through a seam.** :class:`MatchupPlayer` is what actually
plays a scheduled matchup, and :func:`seated_matchup_player` is the production
one -- ``arena.play_match`` when both sides are in-process
:class:`~komugi.eval.arena.ArenaPlayer` s and
``v1bridge.play_bridged_match`` when both are cross-engine
:class:`~komugi.eval.v1bridge.BridgeEntrant` s. The seam exists because the
External leg cannot be played at all from here (§D6.1: the site is reached only
through a browser) and because the driver's own refusals have to be testable
without a GPU, a checkpoint or the network.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from komugi.eval.anchor_aliases import resolve_pool_id
from komugi.eval.arena import (
    DEFAULT_GUMBEL_SCALE,
    DEFAULT_M,
    DEFAULT_N,
    EMPTY_BOOK,
    ArenaConfig,
    ArenaPlayer,
    play_match,
)
from komugi.eval.divergence import (
    DivergenceResult,
    Verdict,
    WindowKind,
    evaluate_window,
    window_for,
)
from komugi.eval.elo import EloResult, fit_elo
from komugi.eval.interval import (
    REPLICATES,
    BootstrapResult,
    Statistic,
    pair_bootstrap,
)
from komugi.eval.ladder import POOL_DRAW_SCORE, matchup_seed
from komugi.eval.pool import (
    KIND_EXTERNAL,
    EXTERNAL_FAMILY,
    EpochManifest,
    EpochSchedule,
    Matchup,
    PoolEntrant,
    PoolRoster,
    build_epoch_schedule,
    family_of,
    is_visits_variant,
    pair_namespace_for,
    require_connected_pool,
    require_connected_schedule,
)
from komugi.eval.pool_books import PoolBook
from komugi.eval.pool_report import (
    FLAG_BRACKET_SATURATED,
    FLAG_EXEMPT_CEILING,
    FLAG_FROZEN,
    FLAG_ONE_SIDED,
    FLAG_REFERENCE,
    FLAG_UNEQUAL_COMPUTE,
    ContemptMap,
    EpochReport,
    MatchupRow,
    ReportHeader,
    VisitsBlock,
    VisitsPoint,
    contempt_for,
    delta_max,
    rating_rows,
    render_epoch_report,
)
from komugi.eval.r5 import (
    PROCEED_ONE_SIDED,
    REFUSE,
    P4Branch,
    RatedRunRefused,
    assert_rated_run_permitted,
)
from komugi.eval.records import GameRecord, MatchSet
from komugi.eval.seats import PairViolation, check_paired_books, seat_matrix
from komugi.eval.v1bridge import (
    EVAL_CEILING,
    V1_FAMILY,
    BridgeEntrant,
    play_bridged_match,
)

# KOM-014's publish idiom, **called** rather than copied, exactly as
# ``pool_books.build_epoch_book`` calls it: a second copy of an atomicity
# sequence is a second thing to get subtly wrong (``book.py:696-714``).
from komugi.selfplay.book import _write_durably as write_durably

POOL_RECORD_VERSION: Final[int] = 1
"""§D9.2's schema version. Bumped by a change to the record's shape."""

RECORDS_FILENAME: Final[str] = "records.jsonl"
"""Every ``GameRecord`` of the pool so far, one JSON object per line.

§D9.2 property 1: the records are stored, not just the fit, so everything above
them is **recomputable** by someone who does not trust the code that wrote it --
the discipline ``R5Game.to_json`` follows when it keeps the raw per-ply eval
lists (``r5.py:324-327``). §6 node 23 is the round trip.
"""

RECORD_FILENAME: Final[str] = "pool.json"
"""§D9.2's per-epoch record, beside its ``records.jsonl``."""

REQUIRED_RECORD_BLOCKS: Final[tuple[str, ...]] = (
    "pool_record_version",
    "epoch",
    "created_utc",
    "divergence_gate",
    "provenance",
    "roster",
    "matchups",
    "fit",
    "bootstrap",
    "divergence",
    "visits_curve",
    "records",
)
"""§D9.2's top-level blocks, which :func:`load_epoch_record` requires.

A list rather than a shrug at ``KeyError``: the design says *"every field named
because §6 node 23 asserts the round trip"*, so a record missing a block is a
truncated or foreign artefact and saying which block is missing is the whole
difference between a refusal and a stack trace (T4 review F6).
"""

PINNED_BUDGET: Final[Mapping[str, float]] = {
    "m": float(DEFAULT_M),
    "n": float(DEFAULT_N),
    "gumbel_scale": float(DEFAULT_GUMBEL_SCALE),
    "ceiling": float(EVAL_CEILING),
}
"""§D7.1's pinned search budget, read by derivation rather than restated.

``m``, ``n`` and ``gumbel_scale`` come from :mod:`komugi.eval.arena`'s own
production defaults (which themselves derive from
:class:`~komugi.config.SelfPlayConfig`) and the ceiling from
``v1bridge.EVAL_CEILING``. A literal here would be a second place for the
numbers ``assert_pinned_budget`` turns into a build break.
"""

DIVERGENCE_GATE_FIRED: Final[str] = "FIRED"
"""§D5.4 step 1's stamp: *"every artefact carrying it is stamped ``FIRED``"*."""

DIVERGENCE_GATE_OK: Final[str] = "ok"
"""At least one window was evaluable and neither fired."""

DIVERGENCE_GATE_NOT_EVALUABLE: Final[str] = "not_evaluable"
"""Neither window could be evaluated, which §D5.3 says is **not** a pass."""

VISITS_GRID: Final[tuple[int, ...]] = (100, 400, 1600, 6400)
"""§D8.2's pre-registered Elo-vs-visits grid -- ``DESIGN.html:262``'s own set.

Checked rather than trusted (T4 review F13): a :class:`VisitsRun` carrying an
off-grid budget would publish a curve §D8.4's predictions were never registered
against, and the grid is no more a parameter than §D5's thresholds are.
"""

RELINK_NAMESPACE_SUFFIX: Final[str] = "R"
"""What §D5.4's re-link appends to the epoch's ``pair_namespace``."""

ANCHOR_ROSTER_NOTE: Final[str] = "gauge"
"""§D9.2's ``notes`` on the roster row that carries the pool's gauge."""

EXTERNAL_RULES_UNKNOWNS: Final[str] = "rules-audit.md:456-485"
"""§D9.2's ``rules_unknowns`` pointer for the site entrant, whose rules are
unverified (§0.2a, U4) -- carried in the record so the artefact says so."""

EXTERNAL_BATCH_INCOMPLETE: Final[str] = "batch_incomplete"
"""§D6.1: the **absence** of a measurement, which is not a 0-0-0 row."""

EXTERNAL_COMPLETED: Final[str] = "completed"
"""§D6.1: the site batch ran to its last game and wrote a summary."""

SITE_BOOK_ID: Final[str] = "site-start-advanced"
"""§D6.1(1)'s single constant ``book_id`` for an External game.

The same shape as ``v1bridge.bridged_book_id``'s ``empty-<mode>``, which exists
because a bare ``"empty"`` *"would then be a lie shared by both games of every
pair -- undetectable by the audit, which only checks that the two agree"*
(``v1bridge.py:969-980``). Restated here rather than imported because the
converter that mints it is an anchor-conversion script outside this
repository, not on this package's import path; the two must agree and §6 node
32's audit is what would notice if they stopped.
"""


class PairedBookViolations(ValueError):
    """The pooled record set failed :func:`check_paired_books` (§6 node 52).

    Its own type, because the repair is not the repair for any other refusal in
    this module: a ``GAME_COUNT`` violation is two matches pooled under one
    ``pair_namespace`` (§D2.3), ``SEATS_NOT_SWAPPED`` is a driver that stopped
    alternating seats, and ``BOOK_MISMATCH`` is a pair whose two games opened
    from different entries. None of them is fixed by refitting.
    """


# ---------------------------------------------------------------------------
# the seam: what actually plays a scheduled matchup
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayedMatchup:
    """One matchup after it has been played, in the pool's own terms.

    ``wall_clock_s`` is **measured**, because §D2.4 budgets the epoch and
    §D8.3's cap is enforced on the measured clock of the run rather than on the
    design's estimate (§6 node 46). Voids live beside the results and are never
    folded into them (``v1bridge.py:1975-1981``, §D9.2 property 2).
    """

    matchup: Matchup
    records: tuple[GameRecord, ...]
    decisive: int
    draws: int
    mean_plies: float
    wall_clock_s: float = 0.0
    void_rate: float = 0.0
    admissible: bool = True
    voids_by_bucket: Mapping[str, int] = field(default_factory=dict)
    voids_by_mode: Mapping[str, int] = field(default_factory=dict)
    """§D9.2's ``voids.by_mode``, beside ``by_bucket``: the bucket says how bad
    a void was and the mode says what it **was** (``v1bridge.VoidTally``)."""

    book_epoch: str = EMPTY_BOOK
    pair_namespace: str = ""


class MatchupPlayer(Protocol):
    """What :func:`run_epoch` calls to turn one scheduled matchup into records.

    A protocol rather than a concrete function so the driver's refusals are
    testable without a GPU, a checkpoint or the network: §6 node 17's
    game-counting spy is a :class:`MatchupPlayer` that counts and node 43's is
    one that records the :class:`~komugi.eval.arena.ArenaConfig` it was handed.
    """

    def __call__(self, matchup: Matchup, config: ArenaConfig) -> PlayedMatchup:
        """Play `matchup` under `config` and return what it came to."""
        ...


Seat = ArenaPlayer | BridgeEntrant
"""What a roster entry is seated as: an in-process rung or a cross-engine one."""


def seated_matchup_player(
    seats: Mapping[str, Seat], *, rated: bool = True
) -> MatchupPlayer:
    """Return the production :class:`MatchupPlayer` over `seats`.

    Two seams, one rule for choosing between them: a matchup whose two sides
    are both :class:`~komugi.eval.arena.ArenaPlayer` s is played by
    ``arena.play_match``, and one whose two sides are both
    :class:`~komugi.eval.v1bridge.BridgeEntrant` s by
    ``v1bridge.play_bridged_match``. A **mixed** matchup is refused rather than
    coerced: ``BridgeEntrant.arena_player`` exists and would seat one, but its
    evaluator factory raises ``WrongAltitude`` by design
    (``v1bridge.py:1766-1781``), so the coercion would fail inside the match
    loop after the pairing had already been set up. The komugi side of a
    cross-engine matchup is seated by
    :func:`komugi.eval.gumbel_mover.komugi_entrant`, which is what makes both
    sides ``BridgeEntrant`` s.

    Args:
        seats: entrant id -> the object that plays it.
        rated: passed to ``play_bridged_match``, which then asserts §D3.3's
            pinned budget and §D4's ``book_id`` invariant. ``True`` is the pool
            (§D6.2(1)); ``False`` exists for a smoke that is not anchor
            evidence.

    Returns:
        A player that raises ``ValueError`` on an unseated or mixed matchup and
        never on a scheduled one.
    """

    def play(matchup: Matchup, config: ArenaConfig) -> PlayedMatchup:
        """Seat both sides and play the matchup on the seam they share."""
        first, second = seats.get(matchup.a), seats.get(matchup.b)
        if first is None or second is None:
            missing = [
                entrant_id
                for entrant_id in (matchup.a, matchup.b)
                if entrant_id not in seats
            ]
            raise ValueError(
                f"matchup {matchup.a!r} x {matchup.b!r} cannot be played: "
                f"{missing} have no seat. Every scheduled entrant needs one "
                f"before the epoch starts, not at the matchup that reaches it"
            )
        started = time.perf_counter()
        if isinstance(first, ArenaPlayer) and isinstance(second, ArenaPlayer):
            outcome = play_match(first, second, config)
            return PlayedMatchup(
                matchup=matchup,
                records=outcome.records,
                decisive=outcome.summary.decisive,
                draws=outcome.summary.draws,
                mean_plies=outcome.summary.mean_plies,
                wall_clock_s=time.perf_counter() - started,
                book_epoch=config.book_id,
                pair_namespace=config.pair_namespace,
            )
        if isinstance(first, BridgeEntrant) and isinstance(second, BridgeEntrant):
            bridged = play_bridged_match(first, second, config, rated=rated)
            if bridged.outcome is None:
                raise ValueError(
                    f"matchup {matchup.a!r} x {matchup.b!r} did not complete: "
                    f"status {bridged.status!r}, reason {bridged.reason!r}. §D6 "
                    f"scores an aborted or match-voided run not at all -- its "
                    f"surviving pairs are exactly the biased sample the 2% gate "
                    f"exists to refuse"
                )
            return PlayedMatchup(
                matchup=matchup,
                records=bridged.outcome.records,
                decisive=bridged.outcome.summary.decisive,
                draws=bridged.outcome.summary.draws,
                mean_plies=bridged.outcome.summary.mean_plies,
                wall_clock_s=time.perf_counter() - started,
                void_rate=bridged.tally.void_rate,
                admissible=bridged.admissible,
                voids_by_bucket=bridged.tally.by_bucket(),
                voids_by_mode=bridged.tally.by_mode(),
                book_epoch=config.book_id,
                pair_namespace=config.pair_namespace,
            )
        raise ValueError(
            f"matchup {matchup.a!r} x {matchup.b!r} mixes an in-process seat "
            f"with a cross-engine one. Seat the komugi side as a BridgeEntrant "
            f"(gumbel_mover.komugi_entrant) when its opponent is one: "
            f"BridgeEntrant.arena_player's evaluator factory raises "
            f"WrongAltitude on purpose (v1bridge.py:1766-1781), so coercing "
            f"here would fail inside the game loop instead of here"
        )

    return play


@dataclass(frozen=True, slots=True)
class SiteLeg:
    """§D6.1's external-anchor batch, already converted by a script that is
    not part of this repository.

    The anchor is reached only through a browser and there is no request that
    plays a prefix into it, so this leg is **not** played through
    :class:`MatchupPlayer`: the caller runs the KOM-017 harness, converts its
    ``summary.json`` with that converter's own ``convert_summary``, and
    hands the result here.

    ``status`` is :data:`EXTERNAL_COMPLETED` or :data:`EXTERNAL_BATCH_INCOMPLETE`.
    An incomplete batch is the **absence** of a measurement -- not ``0 games``,
    not a void bucket, not a 0-0-0 row -- so its leg is simply absent from the
    fit for that epoch, and the graph stays connected without it because
    §D2.2(b)5's tie-in is what connects the component (§D6.1).
    """

    records: tuple[GameRecord, ...] = ()
    status: str = EXTERNAL_COMPLETED
    games_reached: int = 0
    void_rate: float = 0.0
    admissible: bool = True

    @property
    def present(self) -> bool:
        """Return whether this leg contributes records to the epoch's fit."""
        return self.status == EXTERNAL_COMPLETED and bool(self.records)


@dataclass(frozen=True, slots=True)
class VisitsRun:
    """§D8's Elo-vs-visits sub-pool, when the epoch ran one.

    Played off the rated cross-family path (a ``#n1600`` entrant breaches the
    257 ceiling by construction, §D8.2), on one milestone in three, and its
    games **are** in the published joint fit -- that is where the curve is read
    from. What "not merged with the rated rows" means is narrower and still
    binding: a ``#n<visits>`` entrant is its own row and is never averaged into
    the milestone's rated one.
    """

    milestone: str
    records: tuple[GameRecord, ...]
    n_per_matchup: Mapping[int, int]
    wall_clock_s: float

    def __post_init__(self) -> None:
        """Refuse a grid §D8.2 did not pre-register, before anything reads it."""
        off_grid = sorted(
            visits for visits in self.n_per_matchup if visits not in VISITS_GRID
        )
        if off_grid:
            raise ValueError(
                f"visits grid points {off_grid} are not in §D8.2's "
                f"pre-registered grid {list(VISITS_GRID)}: the grid is "
                f"DESIGN.html:262's own set, kept verbatim, and a curve read "
                f"over other budgets is not the curve §D8.4's predictions were "
                f"registered against. The grid is not a parameter"
            )

    def entrant_for(self, visits: int) -> str:
        """Return the grid point's entrant id: ``<milestone>#n<visits>``."""
        return f"{self.milestone}#n{visits}"


# ---------------------------------------------------------------------------
# the roster, under the R5 branch in force
# ---------------------------------------------------------------------------


def build_pool_roster(
    entrants: Sequence[PoolEntrant], *, anchor_id: str, branch: P4Branch
) -> PoolRoster:
    """Return the roster `branch` permits, refusing §D4.1's branch-3 shape.

    **Branch 3 changes the roster, not just the gauge** (§D4.1; rev 1 addressed
    only the gauge). ``r5.py:453-469`` forbids a rated v1 game outright, so all
    four v1 entrants leave the pool: the reference field drops from 7 to 3
    (``rawmcts`` only), §D2.2(a)'s frozen sub-pool becomes
    ``matchups(3, adjacency=2) = 3`` matchups rather than 11, the bracket
    chooses from three candidates and saturates sooner, and the anchor is the
    pre-registered fallback ``rawmcts/n800@<sha8>``. Connectivity survives
    unchanged, because every milestone still plays the anchor directly.

    The refusal is at **construction**, not at the first game, and it names
    every v1 member it found: admitting them and discovering the refusal
    mid-epoch would report a pre-registered decision against a GPU bill that
    was already paid.

    Args:
        entrants: the epoch's members.
        anchor_id: the gauge, minted by the pool and carried in the record.
        branch: the committed R5 result's branch
            (:func:`~komugi.eval.r5.assert_rated_run_permitted`).

    Returns:
        The :class:`~komugi.eval.pool.PoolRoster`.

    Raises:
        RatedRunRefused: on branch 3, if any v1 entrant is present.
        ValueError: on anything :class:`~komugi.eval.pool.PoolRoster` refuses.
    """
    if branch.branch == REFUSE:
        v1_members = sorted(
            entrant.id for entrant in entrants if entrant.family == V1_FAMILY
        )
        if v1_members:
            raise RatedRunRefused(
                f"R5 branch 3 forbids a rated v1 game at ceiling {EVAL_CEILING} "
                f"(r5.py:453-469), so these entrants may not be in the pool at "
                f"all: {v1_members}. Branch 3 is not 'the anchor moves' -- the "
                f"whole v1 family leaves, the reference field drops to rawmcts, "
                f"and the anchor is the pre-registered fallback rawmcts/n800 "
                f"(§D4.1). The report states that DESIGN.html:260's named v1 "
                f"members are absent by R5 branch 3, which is an announced hole "
                f"in the field rather than a silent one"
            )
    return PoolRoster(entrants=tuple(entrants), anchor_id=anchor_id)


# ---------------------------------------------------------------------------
# the record sets the four fits are taken over
# ---------------------------------------------------------------------------


def without_external(
    match_set: MatchSet, roster_map: Mapping[str, str]
) -> MatchSet:
    """Return `match_set` with every External record dropped (§D6.1(5)).

    Family is read from `roster_map`, never parsed from an id's prefix.
    """
    return MatchSet.of(
        record
        for record in match_set.records
        if EXTERNAL_FAMILY
        not in (family_of(roster_map, record.white), family_of(roster_map, record.black))
    )


def without_visits(match_set: MatchSet) -> MatchSet:
    """Return `match_set` with §D8's ``#n<visits>`` legs dropped (§D8.2).

    Mechanical, by the naming rule: ``#n256`` **is** the rated milestone row and
    carries no suffix, so only the off-grid budgets mint a suffixed id.
    """
    return MatchSet.of(
        record
        for record in match_set.records
        if not any(is_visits_variant(player) for player in record.seats)
    )


@dataclass(frozen=True, slots=True)
class EpochFits:
    """§D2.3's four fits over one epoch's records, plus their record sets.

    Four rather than one, and each for a stated reason: ``ratings`` is the
    published strength table; ``ratings_no_external`` prices the §D6.1(5)
    contamination a joint MLE propagates from a pendant whose rules are
    unverified; ``ratings_no_visits`` prices the §D8.2 one, from 32- and
    16-game unequal-compute links; and **every T-tier and certification number
    is read from ``ratings_certification``**, which drops both.

    Each fit's own record set is carried beside it, because a row's game count
    and score rate are properties of *that* set: reading them off the joint one
    would report a without-External rating beside a with-External game count.
    """

    ratings: EloResult
    ratings_no_external: EloResult
    ratings_no_visits: EloResult
    ratings_certification: EloResult
    records: MatchSet
    records_no_external: MatchSet
    records_no_visits: MatchSet
    records_certification: MatchSet

    @property
    def clamped(self) -> tuple[str, ...]:
        """Return the §D9.2 record keys of the fits that clamped.

        Named rather than counted, because the record publishes
        ``fit.converged`` for the joint fit alone: an epoch in which the
        certification fit clamps while the joint fit converges would otherwise
        put ``converged: true`` in the artefact and a ``NOT CONVERGED`` banner
        on the page with no way to tell which is the claim (T4 review F7).
        """
        return tuple(
            name
            for name, result in (
                ("ratings", self.ratings),
                ("ratings_no_external", self.ratings_no_external),
                ("ratings_no_visits", self.ratings_no_visits),
                ("ratings_certification", self.ratings_certification),
            )
            if not result.converged
        )

    @property
    def converged(self) -> bool:
        """Return whether all four fits are measurements rather than clamps."""
        return not self.clamped


def fit_epoch(
    match_set: MatchSet, *, roster_map: Mapping[str, str], anchor: str
) -> EpochFits:
    """Run §D2.3's four fits over `match_set`, all anchored on `anchor`.

    Each sub-fit is anchored on the pool's own gauge when it survives the
    subset and on the subset's first player when it does not, so a subset that
    happens to drop the anchor still produces a table rather than raising --
    the ratings are read as *differences* in every consumer here.
    """

    def fit(subset: MatchSet) -> EloResult:
        """Fit one subset, pinning the pool's anchor when it is in it."""
        players = subset.players()
        return fit_elo(subset, anchor=anchor if anchor in players else players[0])

    no_external = without_external(match_set, roster_map)
    no_visits = without_visits(match_set)
    certification = without_visits(no_external)
    return EpochFits(
        ratings=fit(match_set),
        ratings_no_external=fit(no_external),
        ratings_no_visits=fit(no_visits),
        ratings_certification=fit(certification),
        records=match_set,
        records_no_external=no_external,
        records_no_visits=no_visits,
        records_certification=certification,
    )


# ---------------------------------------------------------------------------
# the per-row intervals (the window gains belong to divergence.evaluate_window)
# ---------------------------------------------------------------------------


class _PoolFit:
    """One joint fit per replicate, served to every rating statistic.

    §D5.2 budgets the bootstrap's wall clock rather than assuming it cheap, and
    a rating interval per entrant would otherwise be one ``fit_elo`` call per
    entrant per replicate -- 40,000 fits on a 20-entrant pool at ``B = 2000``.
    The cache is keyed on the replicate object the bootstrap hands every
    statistic within one replicate, which is the same object for all of them;
    it is the idiom ``divergence._WindowFits`` already uses for the same reason.
    """

    def __init__(self, anchor: str) -> None:
        self._anchor = anchor
        self._cached_for: MatchSet | None = None
        self._cached: Mapping[str, float] | None = None

    def ratings(self, replicate: MatchSet) -> Mapping[str, float] | None:
        """Return the replicate's ratings, or ``None`` if it must be discarded."""
        if self._cached_for is not replicate:
            self._cached_for = replicate
            self._cached = None
            if len(replicate) == 0:
                return None
            players = replicate.players()
            pinned = self._anchor if self._anchor in players else players[0]
            result = fit_elo(replicate, anchor=pinned)
            self._cached = result.ratings if result.converged else None
        return self._cached


def rating_statistics(
    entrant_ids: Sequence[str], *, anchor: str
) -> tuple[Statistic, ...]:
    """Return one bootstrap statistic per entrant: its rating in the joint fit.

    The interval every §D9.1(2) row carries. ``EloResult`` has no dispersion
    (``elo.py:109-124``), so this is where a published rating gets one.
    """
    shared = _PoolFit(anchor)

    def statistic(entrant_id: str) -> Statistic:
        """Return the statistic reading `entrant_id`'s rating off the cache."""

        def compute(replicate: MatchSet) -> float | None:
            """Return this entrant's rating in the replicate's fit."""
            ratings = shared.ratings(replicate)
            if ratings is None:
                return None
            return ratings.get(entrant_id)

        return Statistic(name=entrant_id, compute=compute)

    return tuple(statistic(entrant_id) for entrant_id in entrant_ids)


def rating_intervals(result: BootstrapResult) -> dict[str, tuple[float, float]]:
    """Return each statistic's percentile interval as a ``(lo, hi)`` pair."""
    return {
        name: (interval.lo, interval.hi)
        for name, interval in result.intervals.items()
    }


# ---------------------------------------------------------------------------
# the audits that run before any fit
# ---------------------------------------------------------------------------


def require_paired_books(match_set: MatchSet) -> tuple[PairViolation, ...]:
    """Return no violations, or raise -- **before** anything fits these records.

    §6 node 52. ``elo.py:22-29`` fits no white-advantage term because the seat
    effect is designed out of the data by playing every opening twice; a
    ``GAME_COUNT`` violation says that design was not realised, and the fit
    over it is not a smaller measurement but a different quantity. A
    report-time audit is too late to stop a fit.

    Raises:
        PairedBookViolations: if the pooled set fails the audit.
    """
    violations = check_paired_books(match_set)
    if violations:
        listed = "; ".join(
            f"{violation.pair_id} [{violation.reason}: {violation.detail}]"
            for violation in violations[:8]
        )
        more = "" if len(violations) <= 8 else f" (+{len(violations) - 8} more)"
        raise PairedBookViolations(
            f"the pooled record set fails the paired-book audit with "
            f"{len(violations)} violations and is not fitted: {listed}{more}. "
            f"The Davidson fit carries no seat term because every opening is "
            f"played twice with the seats swapped (elo.py:22-29, seats.py:103-151); "
            f"a set that does not is a different quantity, not a noisier one. "
            f"GAME_COUNT usually means two matches of one pair of entrants were "
            f"pooled under one pair_namespace (§D2.3)"
        )
    return violations


# ---------------------------------------------------------------------------
# the epoch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """§D9.2's provenance block: everything copied in rather than referenced.

    A record that says *"see the R5 artefact"* is a record that stops being
    interpretable when the artefact moves (§D9.2 property 3), so the anchor,
    the branch, the book epoch and the contempt map are all values here.

    ``reference_book_epoch`` is §D2.2(a)'s consequence: the frozen sub-pool's
    re-used records keep the book era they were played in, forever, so from the
    second era on every joint fit mixes eras -- **permitted, declared and
    reported**, and this field is the declaration.
    """

    komugi_sha: str
    engine_rules_version: int
    gungi_mode: str
    action_size: int
    epoch_book_id: str
    source_book_sha256: str | None = None
    reference_book_epoch: str | None = None
    contempt: ContemptMap | None = None
    r5_artefact: str = ""


@dataclass(frozen=True, slots=True)
class EpochPlan:
    """Everything one epoch is, before a game of it has been played.

    Separate from :func:`run_epoch`'s body so an epoch is a *value* that can be
    written down, reviewed and replayed. The two seeds are distinct on purpose:
    ``run_seed`` names the games (through
    :func:`~komugi.eval.ladder.matchup_seed`) and ``bootstrap_seed`` names the
    resampling, and §D9.2 stores both.
    """

    epoch: int
    roster: PoolRoster
    milestone: str
    predicted_rating: float
    ratings: Mapping[str, float]
    provenance: Provenance
    r5_artefact: str | Path
    run_seed: int
    bootstrap_seed: int
    external: str | None = None
    prior_records: tuple[GameRecord, ...] = ()
    book: PoolBook | None = None
    replicates: int = REPLICATES
    site_leg: SiteLeg | None = None
    visits: VisitsRun | None = None
    relink_seed: int | None = None
    v1_absent_by_branch_three: bool = False
    """Set by a campaign that built §D4.1's v1-free, rawmcts-anchored pool.

    Carried rather than inferred: a first-epoch pool whose reference field is
    rawmcts-only looks identical from the inside, and the banner this raises
    (``pool_report.BRANCH_THREE_BANNER``) is a statement about *why* four named
    entrants are missing, which only the caller that read the R5 artefact knows.
    """

    def __post_init__(self) -> None:
        """Refuse a plan whose book and provenance disagree, before any game."""
        if self.book is not None and self.book.epoch_book_id != (
            self.provenance.epoch_book_id
        ):
            raise ValueError(
                f"the plan carries epoch book "
                f"{self.book.epoch_book_id!r} but its provenance declares "
                f"{self.provenance.epoch_book_id!r}; §D9.2 copies the book epoch "
                f"into the record rather than referencing it, so the two may not "
                f"disagree"
            )

    @property
    def pair_namespace(self) -> str:
        """Return this epoch's ``pair_namespace`` (§D2.3): ``E007`` at epoch 7."""
        return pair_namespace_for(self.epoch)

    def arena_config(self, matchup: Matchup, index: int) -> ArenaConfig:
        """Return the config matchup `index` of this epoch is played under.

        Every matchup gets **its own seed** through
        :func:`~komugi.eval.ladder.matchup_seed`: ``ArenaConfig``'s seed key
        contains no entrant identity (``arena.py:318-332``), so two matchups
        played at one seed draw identical per-ply Gumbel streams and their
        games are correlated.
        """
        return ArenaConfig(
            games=matchup.games,
            seed=matchup_seed(self.run_seed, index),
            # §D7.3: one contempt regime for the whole pool, and a constant
            # rather than a field for ``ladder.POOL_DRAW_SCORE``'s own reason --
            # a tolerance that can be passed in is a tolerance that will be.
            draw_score=POOL_DRAW_SCORE,
            book_id=(
                self.provenance.epoch_book_id if self.book is not None else EMPTY_BOOK
            ),
            pair_namespace=self.pair_namespace,
            book=self.book,
        )


@dataclass(frozen=True, slots=True)
class EpochResult:
    """One finished pool epoch: the games, the fits, the gate, and the page.

    ``record`` is §D9.2's JSON shape and ``report`` is §D9.1's text; both are
    derived here rather than by a caller, so the number in the table and the
    number in the record cannot be produced by two different arithmetics.
    """

    manifest: EpochManifest
    schedule: EpochSchedule
    played: tuple[PlayedMatchup, ...]
    match_set: MatchSet
    fits: EpochFits
    intervals: Mapping[str, tuple[float, float]]
    bootstrap: BootstrapResult
    divergence: tuple[DivergenceResult, ...]
    report: str
    record: Mapping[str, Any]
    branch: P4Branch
    visits: VisitsBlock | None = None
    wall_clock_s: float = 0.0

    @property
    def gate_fired(self) -> bool:
        """Return §D5's Q8 disjunction: **either** window firing fires the gate.

        The two windows are evaluated independently and neither touches the
        other's outcome, but the campaign's response (§D5.4 step 1) is to the
        *epoch*, not to a window: the headline is withheld and every artefact
        carrying it is stamped. A caller therefore needs one boolean, and the
        merged driver exposed none -- a ``FIRED`` verdict was visible only
        inside the fourteen-column divergence table (T4 review F2).
        """
        return any(row.verdict is Verdict.FIRED for row in self.divergence)

    def write(
        self, directory: str | os.PathLike[str], *, overwrite: bool = False
    ) -> Path:
        """Write §D9.2's ``pool.json`` and ``records.jsonl`` into `directory`.

        Returns:
            The path of the written ``pool.json``.

        Raises:
            FileExistsError: if `directory` already holds an epoch record and
                `overwrite` is ``False``.
        """
        return write_epoch_record(
            directory, self.record, self.match_set, overwrite=overwrite
        )


def _rating_flags(
    roster: PoolRoster,
    match_set: MatchSet,
    schedule: EpochSchedule,
    *,
    branch: P4Branch,
    roster_map: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Return §D9.1(2)'s per-row flags, every one of them read off the record.

    ``one-sided`` marks the rows whose **own** games include a v1 link, which
    is the narrower fact the per-row flag can carry; the column-level property
    branch 2 actually asserts is a banner (§D4.1, F7).
    """
    unequal: set[str] = set()
    for matchup in schedule.matchups:
        if matchup.unequal_compute:
            unequal.update((matchup.a, matchup.b))
    v1_linked: set[str] = set()
    if branch.branch == PROCEED_ONE_SIDED:
        for record in match_set.records:
            families = tuple(
                family_of(roster_map, player) for player in record.seats
            )
            if V1_FAMILY in families:
                v1_linked.update(record.seats)
    flags: dict[str, tuple[str, ...]] = {}
    for entrant in roster.entrants:
        row: list[str] = []
        if entrant.frozen:
            row.append(FLAG_FROZEN)
        if entrant.reference:
            row.append(FLAG_REFERENCE)
        if entrant.ceiling_exempt is True:
            row.append(FLAG_EXEMPT_CEILING)
        if entrant.id in unequal:
            row.append(FLAG_UNEQUAL_COMPUTE)
        if entrant.id in v1_linked:
            row.append(FLAG_ONE_SIDED)
        if schedule.bracket_saturated and entrant.id == schedule.milestone:
            row.append(FLAG_BRACKET_SATURATED)
        if row:
            flags[entrant.id] = tuple(row)
    return flags


def divergence_gate_state(rows: Sequence[DivergenceResult]) -> str:
    """Return the epoch-level gate stamp §D5.4 step 1 puts on every artefact.

    Three states rather than a boolean, for §D5.3's own reason: *"the
    instrument could not see"* and *"the instrument saw nothing"* are different
    states, and the only way they stay different is if the first one has a
    name. A window that fired dominates (Q8: **either** firing fires the gate);
    otherwise the epoch is ``ok`` if any window was evaluable at all, and
    ``not_evaluable`` if none was.
    """
    if any(row.verdict is Verdict.FIRED for row in rows):
        return DIVERGENCE_GATE_FIRED
    if any(row.verdict is Verdict.OK for row in rows):
        return DIVERGENCE_GATE_OK
    return DIVERGENCE_GATE_NOT_EVALUABLE


def _visits_block(
    plan: EpochPlan,
    fits: EpochFits,
    intervals: Mapping[str, tuple[float, float]],
    *,
    rated_wall_clock_s: float,
) -> VisitsBlock | None:
    """Return §D9.1(5)'s block, priced on the **measured** wall clocks.

    §D8.3 caps the harness at 25% of a pool epoch's GPU budget and enforces the
    cap *"on the measured wall clock of the first run, not on this estimate"*.
    The denominator is the epoch total with the visits harness inside it, which
    is §D8.3's own convention (``3.63/(7.05 + 3.63)``).
    """
    run = plan.visits
    if run is None:
        return None
    total = rated_wall_clock_s + run.wall_clock_s
    fraction = run.wall_clock_s / total if total > 0.0 else 0.0
    points: list[VisitsPoint] = []
    missing: list[int] = []
    for visits in sorted(run.n_per_matchup):
        entrant_id = run.entrant_for(visits)
        elo = fits.ratings.ratings.get(entrant_id)
        if elo is None:
            # Budgeted and unmeasured: carried, not dropped. Silently printing
            # three points where four were paid for makes an incomplete curve
            # indistinguishable from a complete one (T4 review F13).
            missing.append(visits)
            continue
        lo, hi = intervals.get(entrant_id, (elo, elo))
        points.append(
            VisitsPoint(
                visits=visits,
                entrant=entrant_id,
                elo=elo,
                lo=lo,
                hi=hi,
                n_per_matchup=run.n_per_matchup[visits],
            )
        )
    return VisitsBlock(
        milestone=run.milestone,
        points=tuple(points),
        wall_clock_s=run.wall_clock_s,
        epoch_budget_fraction=fraction,
        missing=tuple(missing),
    )


def run_epoch(plan: EpochPlan, play: MatchupPlayer) -> EpochResult:
    """Play one pool epoch and return everything §D9 publishes about it.

    The order below is the design's, and every step is a refusal placed before
    the thing it protects. Nothing after step 1 runs when R5 refuses, and
    nothing after step 5 runs when the pooled records are not properly paired.

    1. :func:`~komugi.eval.r5.assert_rated_run_permitted`, **before any game**.
    2. §D2.2's schedule, and the pool's own connectivity refusal over it.
    3. every matchup on its own seed, through `play`.
    4. the records pooled with every earlier epoch's (§D9.2 property 4).
    5. :func:`require_paired_books`, then
       :func:`~komugi.eval.pool.require_connected_pool` -- both before a fit.
    6. §D2.3's four fits, and §D5.2's bootstrap for every row's interval.
    7. §D5's gate on both windows, read over ``D_no_external``.
    8. §D9.1's report and §D9.2's record.

    Args:
        plan: what this epoch is.
        play: the seam that turns a scheduled matchup into records.

    Returns:
        The :class:`EpochResult`.

    Raises:
        R5Missing: if the committed R5 artefact is absent or unusable.
        RatedRunRefused: on §D3.6 branch 3.
        PairedBookViolations: if the pooled set fails the paired-book audit.
        ValueError: on a disconnected schedule or pool, or anything §D2.2's
            schedule builder refuses.
    """
    # 1. R5 first. A game-counting spy sees zero calls when this raises
    #    (§6 node 17): a rated run that discovers its own refusal after the
    #    games is a refusal reported against a bill already paid.
    branch = assert_rated_run_permitted(plan.r5_artefact)

    roster = plan.roster
    roster_map = roster.roster_map
    schedule = build_epoch_schedule(
        roster,
        epoch=plan.epoch,
        milestone=plan.milestone,
        predicted_rating=plan.predicted_rating,
        ratings=plan.ratings,
        external=plan.external,
    )
    require_connected_schedule(schedule.matchups, roster_map)

    manifest = EpochManifest.of(
        roster,
        schedule,
        epoch_book_id=plan.provenance.epoch_book_id,
        reference_book_epoch=plan.provenance.reference_book_epoch,
    )

    site_leg = plan.site_leg or SiteLeg(status=EXTERNAL_BATCH_INCOMPLETE)
    played: list[PlayedMatchup] = []
    for index, matchup in enumerate(schedule.matchups):
        if matchup.kind == KIND_EXTERNAL:
            # §D6.1: the site is reached only through a browser, so this leg is
            # converted rather than played. An incomplete batch is the absence
            # of a measurement and simply does not enter the fit; the graph
            # stays connected without it, because §D2.2(b)5's tie-in is what
            # connects the External component.
            played.append(
                PlayedMatchup(
                    matchup=matchup,
                    records=site_leg.records if site_leg.present else (),
                    decisive=sum(
                        1 for record in site_leg.records if not record.is_draw
                    )
                    if site_leg.present
                    else 0,
                    draws=sum(1 for record in site_leg.records if record.is_draw)
                    if site_leg.present
                    else 0,
                    mean_plies=0.0,
                    void_rate=site_leg.void_rate,
                    admissible=site_leg.admissible,
                    book_epoch=SITE_BOOK_ID,
                    pair_namespace=plan.pair_namespace,
                )
            )
            continue
        played.append(play(matchup, plan.arena_config(matchup, index)))

    rated_wall_clock_s = sum(outcome.wall_clock_s for outcome in played)
    visits_records = () if plan.visits is None else plan.visits.records
    # This epoch's own games, counted before the prior records are pooled in:
    # §D9.2 property 4 makes the stored set append-only, so `len(match_set)` is
    # every game the pool has ever played and is a different number (§D9.1(1),
    # T4 review F14).
    epoch_games = (
        sum(len(outcome.records) for outcome in played) + len(visits_records)
    )
    match_set = MatchSet.of(
        [
            *plan.prior_records,
            *(record for outcome in played for record in outcome.records),
            *visits_records,
        ]
    )

    # 5. Both audits before any fit: a fit cannot detect either failure.
    require_paired_books(match_set)
    components = require_connected_pool(match_set, roster_map)

    fits = fit_epoch(match_set, roster_map=roster_map, anchor=roster.anchor_id)
    bootstrap = pair_bootstrap(
        match_set,
        rating_statistics(match_set.players(), anchor=roster.anchor_id),
        replicates=plan.replicates,
        seed=plan.bootstrap_seed,
    )
    intervals = rating_intervals(bootstrap)

    rows = rating_rows(
        ratings=fits.ratings.ratings,
        intervals=intervals,
        roster_map=roster_map,
        match_set=match_set,
        anchor_id=roster.anchor_id,
        ratings_no_external=fits.ratings_no_external.ratings,
        ratings_no_visits=fits.ratings_no_visits.ratings,
        extra_flags=_rating_flags(
            roster, match_set, schedule, branch=branch, roster_map=roster_map
        ),
    )

    divergence_rows: list[DivergenceResult] = []
    for kind in (WindowKind.SINGLE, WindowKind.SUSTAINED):
        divergence_rows.append(
            evaluate_window(
                match_set,
                roster_map=roster_map,
                window=window_for(roster, epoch=plan.epoch, kind=kind),
                kind=kind,
                bootstrap_seed=plan.bootstrap_seed,
                replicates=plan.replicates,
            )
        )

    visits_block = _visits_block(
        plan, fits, intervals, rated_wall_clock_s=rated_wall_clock_s
    )

    matchup_rows = tuple(
        MatchupRow(
            a=outcome.matchup.a,
            b=outcome.matchup.b,
            games=len(outcome.records),
            decisive=outcome.decisive,
            draws=outcome.draws,
            mean_plies=outcome.mean_plies,
            kind=outcome.matchup.kind,
            book_epoch=outcome.book_epoch,
            pair_namespace=outcome.pair_namespace,
            unequal_compute=outcome.matchup.unequal_compute,
            void_rate=outcome.void_rate,
            admissible=outcome.admissible,
        )
        for outcome in played
    )

    header = ReportHeader(
        epoch=plan.epoch,
        epoch_book_id=plan.provenance.epoch_book_id,
        anchor_id=roster.anchor_id,
        contempt=plan.provenance.contempt
        or contempt_for(roster_map.values()),
        pinned_budget=PINNED_BUDGET,
        engine_rules_version=plan.provenance.engine_rules_version,
        r5_branch=branch.branch,
        entrants=len(roster.entrants),
        matchups=len(schedule.matchups),
        games=len(match_set),
        epoch_games=epoch_games,
        converged=fits.ratings.converged,
        converged_all=fits.converged,
        clamped_fits=fits.clamped,
        gate_fired=any(
            row.verdict is Verdict.FIRED for row in divergence_rows
        ),
        components=len(components),
        one_sided_only=branch.one_sided_only,
        v1_absent_by_branch_three=plan.v1_absent_by_branch_three,
        reference_book_epoch=plan.provenance.reference_book_epoch,
        external_games=len(site_leg.records),
        external_status=site_leg.status,
        external_delta_max=delta_max(rows, external=True),
        visits_delta_max=delta_max(rows, external=False),
        bracket_saturated=schedule.bracket_saturated,
    )
    report = render_epoch_report(
        EpochReport(
            header=header,
            ratings=rows,
            matchups=matchup_rows,
            match_set=match_set,
            violations=(),
            divergence=tuple(divergence_rows),
            visits=visits_block,
            relink_seed=plan.relink_seed,
        )
    )
    record = epoch_record(
        plan,
        branch=branch,
        manifest=manifest,
        played=played,
        fits=fits,
        intervals=intervals,
        bootstrap=bootstrap,
        divergence=tuple(divergence_rows),
        visits=visits_block,
        components=len(components),
        rows_external_delta_max=header.external_delta_max,
        rows_visits_delta_max=header.visits_delta_max,
    )
    return EpochResult(
        manifest=manifest,
        schedule=schedule,
        played=tuple(played),
        match_set=match_set,
        fits=fits,
        intervals=intervals,
        bootstrap=bootstrap,
        divergence=tuple(divergence_rows),
        report=report,
        record=record,
        branch=branch,
        visits=visits_block,
        wall_clock_s=rated_wall_clock_s,
    )


# ---------------------------------------------------------------------------
# §D5.4's re-link diagnostic
# ---------------------------------------------------------------------------


def relink_namespace_for(epoch: int) -> str:
    """Return the ``pair_namespace`` §D5.4's diagnostic plays under.

    Distinct from the epoch's own, because the diagnostic re-plays a matchup
    the epoch already played and a repeat collides on ``pair_id``
    (``arena.py:645-654``, §D2.3).
    """
    return f"{pair_namespace_for(epoch)}{RELINK_NAMESPACE_SUFFIX}"


@dataclass(frozen=True, slots=True)
class RelinkDiagnostic:
    """§D5.4 step 2a: an **independent** measurement of the same quantity."""

    relink_seed: int
    pair_namespace: str
    played: tuple[PlayedMatchup, ...]
    branch: P4Branch
    """The R5 branch these 256 rated games were played under.

    Carried rather than referenced, for §D9.2 property 3's reason: the
    diagnostic runs *"at the next epoch's expense"* (§D5.4 step 2), so time
    passes between the epoch's R5 read and this one, and an artefact that says
    *"see the epoch record"* stops being interpretable when the artefact moves.
    """

    @property
    def records(self) -> tuple[GameRecord, ...]:
        """Return every record the re-link produced."""
        return tuple(
            record for outcome in self.played for record in outcome.records
        )


def run_relink(
    plan: EpochPlan,
    window: tuple[str, str],
    play: MatchupPlayer,
    *,
    games: int = 128,
) -> RelinkDiagnostic:
    """Re-play the window's two anchor links as §D5.4's diagnostic.

    **Both** the ``pair_namespace`` and the ``ArenaConfig.seed`` change, and
    the seed is the load-bearing half (§6 node 43). ``pair_namespace`` changes
    only the ``pair_id``, while the games come from
    ``game_seeds(config, pair, seat)`` whose key contains **no entrant
    identity** (``arena.py:318-332``) -- so a re-link at the epoch's own seed
    would reproduce the original games move for move, a sampling accident would
    confirm itself, and §D5.4 step 3's *Not reproduced* branch would be
    unreachable.

    The seed is drawn once by the caller, stored in the epoch record **before**
    the re-play (``EpochPlan.relink_seed``), and printed beside ``D``.

    Args:
        plan: the epoch whose gate fired. Its ``relink_seed`` is the diagnostic's.
        window: the two endpoint milestones.
        play: the same seam :func:`run_epoch` uses.
        games: games per re-played matchup; §D5.4 pre-registers 128.

    Returns:
        The :class:`RelinkDiagnostic`.

    Raises:
        R5Missing: if the committed R5 artefact is absent or unusable.
        RatedRunRefused: on §D3.6 branch 3.
        ValueError: if no ``relink_seed`` was recorded, or if it equals the
            epoch's ``run_seed`` -- which would make the diagnostic a replay.
    """
    # R5 first here too, before the seed validation and long before a game.
    # This function is a **second entry point** that plays 2 x 128 rated games
    # through the same seam, and §5's T4 row states the rule over the code that
    # plays games rather than over one function: the epoch's R5 read was taken
    # an epoch earlier (§D5.4 step 2 runs "at the next epoch's expense"), so
    # the artefact may since have gone stale or away (T4 review F1).
    branch = assert_rated_run_permitted(plan.r5_artefact)
    if plan.relink_seed is None:
        raise ValueError(
            "§D5.4's re-link needs a fresh relink_seed, drawn once and stored "
            "in the epoch record before the re-play. Without one the "
            "diagnostic is not an independent measurement of the same quantity"
        )
    if plan.relink_seed == plan.run_seed:
        raise ValueError(
            f"relink_seed {plan.relink_seed!r} is the epoch's own run_seed. A "
            f"re-link that changes only the pair_namespace reproduces the "
            f"original games move for move -- game_seeds' key carries no "
            f"entrant identity (arena.py:318-332) -- so a sampling accident "
            f"would confirm itself and §D5.4 step 3's 'not reproduced' branch "
            f"would be unreachable"
        )
    namespace = relink_namespace_for(plan.epoch)
    anchor = plan.roster.anchor
    played: list[PlayedMatchup] = []
    for index, milestone in enumerate(window):
        matchup = Matchup(
            a=milestone,
            b=anchor.id,
            games=games,
            kind="relink",
            compute_a=plan.roster.entrant(milestone).compute,
            compute_b=anchor.compute,
        )
        config = ArenaConfig(
            games=games,
            seed=matchup_seed(plan.relink_seed, index),
            draw_score=POOL_DRAW_SCORE,
            book_id=(
                plan.provenance.epoch_book_id if plan.book is not None else EMPTY_BOOK
            ),
            pair_namespace=namespace,
            book=plan.book,
        )
        played.append(play(matchup, config))
    return RelinkDiagnostic(
        relink_seed=plan.relink_seed,
        pair_namespace=namespace,
        played=tuple(played),
        branch=branch,
    )


# ---------------------------------------------------------------------------
# §D9.2's record
# ---------------------------------------------------------------------------


def record_to_json(record: GameRecord) -> dict[str, Any]:
    """Return one :class:`~komugi.eval.records.GameRecord` as a JSON object."""
    return {
        "white": record.white,
        "black": record.black,
        "result": record.result,
        "book_id": record.book_id,
        "pair_id": record.pair_id,
    }


def record_from_json(
    payload: Mapping[str, Any], *, line_number: int | None = None
) -> GameRecord:
    """Return the :class:`~komugi.eval.records.GameRecord` `payload` describes.

    A missing key is a :class:`ValueError` naming the line, not a bare
    ``KeyError``: this is ``pool_books._entry_from_json``'s own review F4, in
    the same package for the same reason -- :func:`load_records` documents its
    refusals as ``ValueError``, and a caller that wrote ``except ValueError``
    around the load exactly as documented would otherwise watch a truncated
    line escape unhandled (T4 review F6).

    Args:
        payload: one line's parsed JSON object.
        line_number: its 1-based line number in :data:`RECORDS_FILENAME`, so
            the refusal names the line rather than the whole file.

    Raises:
        ValueError: if a field is missing or is not the type it must be.
    """
    where = "" if line_number is None else f" on line {line_number}"
    try:
        return GameRecord(
            # C1: a pool id stored under an old family name (before the
            # external-anchor rename) is mapped to its current name here, so
            # every downstream reader only ever sees the new one.
            white=resolve_pool_id(str(payload["white"])),
            black=resolve_pool_id(str(payload["black"])),
            result=float(payload["result"]),
            book_id=str(payload["book_id"]),
            pair_id=str(payload["pair_id"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"a stored game record{where} is missing field {exc}: every "
            f"{RECORDS_FILENAME} line carries all five of white, black, "
            f"result, book_id and pair_id, because §D9.2 property 1 makes the "
            f"fit recomputable from this file alone"
        ) from exc


def _ratings_block(
    result: EloResult,
    match_set: MatchSet,
    intervals: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Return one fit's ratings in §D9.2's per-entrant shape.

    `match_set` is **that fit's own** record set, so a without-External rating is
    reported beside a without-External game count. `intervals` is ``None`` for
    every fit but the published one: §D5.2's bootstrap is run over the joint
    record set, and quoting its interval beside a different fit's point
    estimate would be an interval for a quantity nobody estimated.
    """
    block: dict[str, Any] = {}
    for entrant_id, elo in sorted(result.ratings.items()):
        games = match_set.games_played_by(entrant_id)
        interval = None if intervals is None else intervals.get(entrant_id)
        block[entrant_id] = {
            "elo": elo,
            "lo": None if interval is None else interval[0],
            "hi": None if interval is None else interval[1],
            "games": games,
            "score_rate": (
                match_set.score_of(entrant_id) / games if games else 0.0
            ),
        }
    return block


def _seat_matrix_block(records: Sequence[GameRecord]) -> list[dict[str, Any]]:
    """Return §D9.2's ``seat_matrix`` for one matchup's own records.

    ``seats.seat_matrix``'s output, restricted to this matchup and rendered in
    its sorted key order, so the artefact carries what §D9.1(3) prints rather
    than only the live object being able to show it.
    """
    return [
        {
            "white": tally.white,
            "black": tally.black,
            "games": tally.games,
            "wins": tally.wins,
            "draws": tally.draws,
            "losses": tally.losses,
        }
        for tally in seat_matrix(MatchSet.of(records)).values()
    ]


def _divergence_entry(row: DivergenceResult) -> dict[str, Any]:
    """Return one window's §D9.2 entry, **entirely** off the gate's own row.

    Every field here -- the three gains, the two statistics, the intervals, the
    verdict, the discard rate and the failing condition -- comes from the
    :class:`~komugi.eval.divergence.DivergenceResult` that
    ``divergence.evaluate_window`` returned, because that is the object the
    verdict was computed from (T4 review F4). The driver publishing its own
    second computation of the gains is how a record comes to carry a
    wrong-signed ``G_in`` beside a correctly-signed verdict, which is what the
    review's M2 demonstrated on the merged code. ``gate_statistic`` is **named
    in the record, never inferred**.
    """
    return {
        "window_kind": str(row.window_kind.value),
        "window": None if row.window is None else list(row.window),
        "g_in": row.g_in,
        "g_cross": row.g_cross,
        "d": row.d,
        "lo": row.lo,
        "hi": row.hi,
        "g_cross_no_external": row.g_cross_no_external,
        "d_no_external": row.d_no_external,
        "lo_no_external": row.lo_no_external,
        "hi_no_external": row.hi_no_external,
        "gate_statistic": row.gate_statistic,
        "threshold": row.threshold,
        "verdict": str(row.verdict.value),
        "condition": row.condition,
        "external_drift": row.external_drift,
        "bracket_saturated": list(row.bracket_saturated),
        "discard_rate": row.discard_rate,
        "bootstrap_wall_clock_s": row.bootstrap_wall_clock_s,
        "relink_seed": None,
    }


def epoch_record(
    plan: EpochPlan,
    *,
    branch: P4Branch,
    manifest: EpochManifest,
    played: Sequence[PlayedMatchup],
    fits: EpochFits,
    intervals: Mapping[str, tuple[float, float]],
    bootstrap: BootstrapResult,
    divergence: Sequence[DivergenceResult],
    visits: VisitsBlock | None,
    components: int,
    rows_external_delta_max: float | None,
    rows_visits_delta_max: float | None,
) -> dict[str, Any]:
    """Return §D9.2's versioned JSON record for one epoch.

    Everything a reader needs is **copied in, not referenced** (property 3):
    the anchor id, the R5 branch, the book epoch, the reference field's own
    book epoch, and the contempt map with each family's scale named. The
    records themselves go to :data:`RECORDS_FILENAME` beside this file, so
    every number here is recomputable from them plus this provenance block by
    someone who does not trust the code that wrote it (property 1, §6 node 23).
    """
    roster = plan.roster
    roster_map = roster.roster_map
    provenance = plan.provenance
    return {
        "pool_record_version": POOL_RECORD_VERSION,
        "epoch": plan.epoch,
        # An epoch record that cannot say when it was written is weaker
        # evidence than one that can, and it is one line (T4 review F8).
        "created_utc": datetime.now(UTC).isoformat(),
        # §D5.4 step 1's stamp, at the **top level** beside the rows it is a
        # disjunction over: a reader deciding whether this epoch's headline may
        # be published must not have to scan a fourteen-column table for it
        # (T4 review F2).
        "divergence_gate": divergence_gate_state(divergence),
        "provenance": {
            "komugi_sha": provenance.komugi_sha,
            "engine_rules_version": provenance.engine_rules_version,
            "gungi_mode": provenance.gungi_mode,
            "action_size": provenance.action_size,
            "epoch_book_id": provenance.epoch_book_id,
            "source_book_sha256": provenance.source_book_sha256,
            "reference_book_epoch": provenance.reference_book_epoch,
            "contempt": dict(
                provenance.contempt or contempt_for(roster_map.values())
            ),
            "pinned_budget": dict(PINNED_BUDGET),
            "pair_namespace": manifest.pair_namespace,
            "r5": {
                "artefact": provenance.r5_artefact or os.fspath(plan.r5_artefact),
                "delta_mean": branch.delta_mean,
                "branch": branch.branch,
                "one_sided_only": branch.one_sided_only,
            },
        },
        "roster": [
            {
                "id": entrant.id,
                "family": entrant.family,
                "frozen": entrant.frozen,
                "reference": entrant.reference,
                "anchor": entrant.id == roster.anchor_id,
                "epoch": entrant.epoch,
                "ceiling_exempt": entrant.ceiling_exempt,
                "config_digest": entrant.config_digest,
                "compute": None
                if entrant.compute is None
                else dict(entrant.compute),
                # §D9.2's three remaining roster fields, emitted rather than
                # left to diverge from the schema (T4 review F8). The site's
                # games are paired **structurally** -- one synthetic pair_id
                # per two games -- but not by opening, and §D6.1(1) says the
                # manifest must record that so no reader mistakes one for the
                # other; its rules are unverified (U4) and the record points at
                # the audit that says which ones.
                "paired_books": entrant.family != EXTERNAL_FAMILY,
                "notes": (
                    ANCHOR_ROSTER_NOTE if entrant.id == roster.anchor_id else None
                ),
                "rules_unknowns": (
                    EXTERNAL_RULES_UNKNOWNS
                    if entrant.family == EXTERNAL_FAMILY
                    else None
                ),
            }
            for entrant in roster.entrants
        ],
        "matchups": [
            {
                "a": outcome.matchup.a,
                "b": outcome.matchup.b,
                "kind": outcome.matchup.kind,
                "games": len(outcome.records),
                "pairs": len(outcome.records) // 2,
                "decisive": outcome.decisive,
                "draws": outcome.draws,
                "mean_plies": outcome.mean_plies,
                "pair_namespace": outcome.pair_namespace,
                "book_epoch": outcome.book_epoch,
                "compute": {
                    "a": None
                    if outcome.matchup.compute_a is None
                    else dict(outcome.matchup.compute_a),
                    "b": None
                    if outcome.matchup.compute_b is None
                    else dict(outcome.matchup.compute_b),
                },
                "unequal_compute": outcome.matchup.unequal_compute,
                "exempt": list(outcome.matchup.exempt),
                # §D9.1(3) renders the per-seat matrices from the live
                # MatchSet, so without this the artefact did not carry them at
                # all -- and a seat-driven divergence has a signature the
                # aggregate rating cannot show (§D5.4 step 2b, review F8).
                "seat_matrix": _seat_matrix_block(outcome.records),
                "voids": {
                    "rate": outcome.void_rate,
                    "by_bucket": dict(outcome.voids_by_bucket),
                    "by_mode": dict(outcome.voids_by_mode),
                },
                "admissible": outcome.admissible,
                "wall_clock_s": outcome.wall_clock_s,
            }
            for outcome in played
        ],
        "fit": {
            "anchor": roster.anchor_id,
            # Two flags, because they answer two questions and the report's
            # banner reads the second: `converged` is the **published joint
            # fit's**, and `converged_all` is the conjunction over §D2.3's four
            # (T4 review F7). `clamped` names the failures rather than leaving
            # a reader to diff four rating tables.
            "converged": fits.ratings.converged,
            "converged_all": fits.converged,
            "clamped": list(fits.clamped),
            "iterations": fits.ratings.iterations,
            "draw_param": fits.ratings.draw_param,
            "components": components,
            "ratings": _ratings_block(fits.ratings, fits.records, intervals),
            "ratings_no_external": _ratings_block(
                fits.ratings_no_external, fits.records_no_external
            ),
            "ratings_no_visits": _ratings_block(
                fits.ratings_no_visits, fits.records_no_visits
            ),
            "ratings_certification": _ratings_block(
                fits.ratings_certification, fits.records_certification
            ),
            "external_delta_max": rows_external_delta_max,
            "visits_delta_max": rows_visits_delta_max,
        },
        "bootstrap": {
            "replicates": bootstrap.replicates,
            "unit": bootstrap.unit,
            "stratified_by": bootstrap.stratified_by,
            "discarded": bootstrap.discarded,
            "seed": bootstrap.seed,
            "wall_clock_s": bootstrap.wall_clock_s,
        },
        "divergence": [
            {**_divergence_entry(row), "relink_seed": plan.relink_seed}
            for row in divergence
        ],
        "visits_curve": None
        if visits is None
        else {
            "milestone": visits.milestone,
            "n_per_matchup": {
                str(point.visits): point.n_per_matchup for point in visits.points
            },
            "points": [
                {
                    "visits": point.visits,
                    "entrant": point.entrant,
                    "elo": point.elo,
                    "lo": point.lo,
                    "hi": point.hi,
                }
                for point in visits.points
            ],
            "missing": list(visits.missing),
            "monotone": visits.monotone,
            "span": visits.span,
            "wall_clock_s": visits.wall_clock_s,
            "epoch_budget_fraction": visits.epoch_budget_fraction,
            "over_cap": visits.over_cap,
        },
        "records": RECORDS_FILENAME,
    }


def write_epoch_record(
    directory: str | os.PathLike[str],
    record: Mapping[str, Any],
    match_set: MatchSet,
    *,
    overwrite: bool = False,
) -> Path:
    """Write §D9.2's ``pool.json`` and ``records.jsonl`` into `directory`.

    The record artefacts live where the campaign's results live --
    ``gungi-az-runs/pool/epoch_<NNN>/``, **not** the repository -- and the
    records are append-only across epochs: epoch ``E+1``'s ``records.jsonl``
    holds epoch ``E``'s verbatim plus the new ones, which the ``pair_namespace``
    is what makes safe (§D9.2 property 4).

    Two properties, both of them refusals (T4 review F5):

    * **Durable, through the repository's own publish idiom.**
      ``komugi.selfplay.book._write_durably`` (``book.py:696-714``) is called
      rather than copied -- dot-prefixed temp file, ``fsync`` the bytes,
      ``os.replace``, ``fsync`` the directory -- for the reason
      ``pool_books.build_epoch_book`` states at its own call site: *"a second
      copy of an atomicity sequence is a second thing to get subtly wrong"*.
      ``records.jsonl`` is written **first** and ``pool.json`` second, so the
      pointer never precedes its target.
    * **A committed epoch record is not silently replaced.** §D9.2 property 4
      makes the record set append-only and §6 node 10 establishes the refusal
      idiom for the epoch book; the pool record is the more consequential
      artefact of the two and had the weaker guarantee.

    Args:
        directory: where the epoch's artefacts go.
        record: §D9.2's JSON object, from :func:`epoch_record`.
        match_set: every record the epoch was fitted over.
        overwrite: replace an existing record rather than refusing. Explicit,
            and never the default: re-running an epoch over a committed
            directory is the accident the refusal exists for.

    Returns:
        The path of the written ``pool.json``.

    Raises:
        FileExistsError: if either target exists and `overwrite` is ``False``.
    """
    target = Path(directory)
    records_path = target / RECORDS_FILENAME
    record_path = target / RECORD_FILENAME
    if not overwrite:
        for existing in (records_path, record_path):
            if existing.exists():
                raise FileExistsError(
                    f"{existing} already holds a committed epoch record: "
                    f"§D9.2's record set is append-only across epochs -- epoch "
                    f"E+1's {RECORDS_FILENAME} carries epoch E's records "
                    f"verbatim plus the new ones, which is the mechanism, and "
                    f"the pair_namespace (§D2.3) is what makes it safe. "
                    f"Refusing to overwrite; write the next epoch at its own "
                    f"path, or pass overwrite=True if this record was never "
                    f"published"
                )
    target.mkdir(parents=True, exist_ok=True)
    # The records first: a reader that finds pool.json finds the games it
    # points at, never the other way round.
    write_durably(
        records_path,
        "".join(
            json.dumps(record_to_json(game), sort_keys=True) + "\n"
            for game in match_set.records
        ),
    )
    write_durably(
        record_path, json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record_path


def load_records(path: str | os.PathLike[str]) -> MatchSet:
    """Return the :class:`~komugi.eval.records.MatchSet` stored at `path`.

    One JSON object per line, in play order, which is what makes the fit
    reproducible from the file alone (``records.py:128-142``).

    Raises:
        ValueError: if a line is not a JSON object carrying all five fields.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return MatchSet.of(
        record_from_json(json.loads(line), line_number=number)
        for number, line in enumerate(lines, start=1)
        if line.strip()
    )


def load_epoch_record(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Return `directory`'s §D9.2 record, refusing one this code cannot read.

    :data:`POOL_RECORD_VERSION` was written by the emitter and read by nobody:
    a record from a future schema was either mis-read or died on a bare
    ``KeyError`` several frames below the load (T4 review F6). Both refusals
    are here, before any field is interpreted, in ``load_pool_book``'s own
    order -- the version **first**, then the blocks.

    Raises:
        ValueError: if the file is not a JSON object, if its
            ``pool_record_version`` is not :data:`POOL_RECORD_VERSION`, or if a
            top-level block §D9.2 names is absent.
    """
    path = Path(directory) / RECORD_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a §D9.2 epoch record object")
    version = payload.get("pool_record_version")
    if version != POOL_RECORD_VERSION:
        raise ValueError(
            f"{path} declares pool_record_version {version!r} and this code "
            f"reads {POOL_RECORD_VERSION}: the schema is versioned precisely "
            f"so a record written by another shape is refused rather than "
            f"partly understood (§D9.2). Read it with the code that wrote it"
        )
    missing = [block for block in REQUIRED_RECORD_BLOCKS if block not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing top-level §D9.2 block(s) {missing}: every "
            f"field is named in the schema because §6 node 23 asserts the "
            f"round trip, so an absent block is a truncated or foreign record "
            f"rather than a defaulted one"
        )
    return payload


def recompute_ratings(directory: str | os.PathLike[str]) -> dict[str, float]:
    """Refit an epoch's published ratings from its own artefacts alone.

    §D9.2 property 1's whole point: everything the record states is
    recomputable from ``records.jsonl`` plus the provenance block, **by someone
    who does not trust the code that wrote it**. This function reads only the
    stored records and the stored anchor -- it does not consult a roster, a
    schedule or a live driver, and it does not read the stored ratings at all
    -- and §6 node 23 asserts the result equals the stored ratings to ``1e-9``
    with that table zeroed on disk (T4 review F-23).

    Raises:
        ValueError: if the record is unreadable by this schema version
            (:func:`load_epoch_record`), or names an anchor that did not play.
    """
    target = Path(directory)
    payload = load_epoch_record(target)
    match_set = load_records(target / RECORDS_FILENAME)
    anchor = str(payload["fit"]["anchor"])
    return fit_elo(match_set, anchor=anchor).ratings
