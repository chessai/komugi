"""§D9.1's report: one pool epoch as the page a human actually reads.

``docs/design/kom-011d-ladder-pool.md`` §D9.1 fixes five blocks and their
order: header and provenance, the BayesElo table, the per-seat matrices, the
two divergence rows, and the Elo-vs-visits block when the epoch ran one. This
module renders exactly those, in the idiom
:meth:`komugi.eval.ladder.LadderResult.report` already establishes -- fixed
width, no plotting, returned as a string so the caller decides where it goes.

Four properties are load-bearing, and each is a refusal of an easier shape:

* **A banner sits above the tables, never below them.** ``converged=False``
  means the pool *"is not identified from these games -- the ratings below are
  clamps, not measurements"* (``ladder.py:271-278``, mechanism at
  ``elo.py:70-77``); ``pool_components() > 1`` means every rating across the
  split is an unidentified free parameter; R5 branch 2 means **every**
  rating in the table is a Komugi-minus-v1 margin, which a per-row flag cannot
  express (§D4.1); and a fired divergence gate means the epoch's headline is
  withheld altogether (§D5.4 step 1). Every one of them is a property of the
  whole column, so every one of them is a banner.
* **The anchor flag is read from the record, never re-derived.** ``0.0`` in a
  rating column does not identify the anchor: on a *balanced* pool every rating
  is exactly ``0.0``, and the 011c review's F1 was a report that scanned for the
  first ``0.0`` and named the wrong player (``ladder.py:218-241``,
  ``tickets/KOM-011.md:70-76``). :func:`rating_rows` takes the anchor id and
  flags the row that carries it.
* **A clean audit prints a line.** An absent per-seat section and a
  violation-free one must not look the same, so a pool with nothing to report
  prints :data:`CLEAN_AUDIT` explicitly (§D9.1(3)).
* **Every rating row carries an interval.** ``EloResult`` carries no dispersion
  (``elo.py:109-124``), so a bare rating in a published table is a point
  estimate wearing a measurement's clothes. The interval comes from §D5.2's
  paired bootstrap and the row cannot be built without one.

The fixed-width helper is a **copy** of ``komugi.eval.ladder._table``
(``ladder.py:470-487``) rather than an import of a private name from a module
that reaches torch (``ladder.py:58`` -> ``arena.py:81``). The copy is eight
lines and is cited where it sits; promoting the original would have put
``ladder.py`` in this task's blast radius for a formatting helper.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from komugi.eval.divergence import (
    GATE_STATISTIC,
    PUBLISHED_STATISTIC,
    DivergenceResult,
    Verdict,
)
from komugi.eval.records import WHITE_WIN, MatchSet
from komugi.eval.seats import PairViolation, seat_matrix

CLEAN_AUDIT: Final[str] = "paired-book audit: clean, 0 violations"
"""Printed when :func:`~komugi.eval.seats.check_paired_books` found nothing.

§D9.1(3): *"an empty list is printed as an explicit ... because an absent
section and a clean one must not look the same"*.
"""

NOT_CONVERGED_BANNER: Final[str] = (
    "!! NOT CONVERGED: this pool is not identified from these games -- "
    "the ratings below are clamps, not measurements"
)
"""``ladder.py:271-278``'s banner, restated for the pool's own fit."""

CLAMPED_FITS_SUFFIX: Final[str] = " (clamped: {fits})"
"""Appended to :data:`NOT_CONVERGED_BANNER` when the record can name the fits.

§D2.3 runs **four** fits and the record publishes ``fit.converged`` for the
joint one only, so an epoch whose certification fit clamps while the joint fit
converges printed a banner the artefact contradicted (T4 review F7). The
banner therefore says *which* fit failed, and the record carries
``fit.converged_all`` beside ``fit.converged`` so the two can be reconciled.
"""

DIVERGENCE_GATE_BANNER: Final[str] = (
    "!! DIVERGENCE GATE FIRED (§D5.4): the headline for this epoch's milestone "
    "is withheld -- no cross-family Elo may be published for it until the "
    "pre-registered diagnostic has run"
)
"""§D5.4 step 1, as a banner rather than a cell in the divergence table.

*"The headline is withheld immediately ... every artefact carrying it is
stamped ``divergence_gate: FIRED``"* -- a property of the whole published
column, exactly as R5 branch 2 is, and therefore a banner in the same idiom
(T4 review F2). Withholding the headline is a campaign action and stays out of
this renderer; saying so on the page is the renderer's job.
"""

COMPONENTS_BANNER: Final[str] = (
    "!! {components} COMPONENTS: the comparison graph is not connected, so the "
    "offset between its parts is an unidentified free parameter -- every rating "
    "compared across the split is meaningless"
)
"""§D2.2's identifiability failure, as a banner rather than a footnote."""

ONE_SIDED_BANNER: Final[str] = (
    "!! ONE-SIDED (R5 branch 2): every rating in this table is a lower bound; "
    "no row may be cited as a symmetric equal-compute number"
)
"""§D4.1 branch 2, which is a property of the **column**, not of a row.

With the pool anchored at ``v1/iter-0020 = 0`` every published rating is a
Komugi-minus-v1 margin, and ``r5.py:438-452`` forbids citing such a number
symmetrically *"in KOM-011 or anywhere else"*. The per-row ``one-sided`` flag
stays, marking the rows whose **own** games include a v1 link, which is a
narrower and still useful fact.
"""

BRANCH_THREE_BANNER: Final[str] = (
    "!! v1 ABSENT BY R5 BRANCH 3: DESIGN.html:260's v1/iter-0005, 0011, 0015 "
    "and 0020 are not in this pool -- an announced hole in the reference field"
)
"""§D4.1 branch 3: the hole in the field is announced, never silent."""

VISITS_BUDGET_CAP: Final[float] = 0.25
"""§D8.3's pre-registered cap on the Elo-vs-visits harness.

Enforced on the **measured** wall clock of the run, not on §D8.3's estimate
(§6 node 46): longer games at high visits push the clock up, and the estimate
would not notice.
"""

FLAG_ANCHOR: Final[str] = "anchor"
FLAG_FROZEN: Final[str] = "frozen"
FLAG_REFERENCE: Final[str] = "reference"
FLAG_ONE_SIDED: Final[str] = "one-sided"
FLAG_EXEMPT_CEILING: Final[str] = "exempt-ceiling"
FLAG_UNEQUAL_COMPUTE: Final[str] = "unequal-compute"
FLAG_BRACKET_SATURATED: Final[str] = "bracket-saturated"
"""§D9.1(2)'s flag column vocabulary, as constants so a reader and a test agree."""

_MISSING: Final[str] = "--"
"""What an absent number prints as: not ``0``, which is a measurement."""

UNDECLARED_REFERENCE_EPOCH: Final[str] = "(not declared)"
"""What an absent ``reference_book_epoch`` prints as: not this epoch's book id."""


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return `rows` as fixed-width text lines under `header`.

    **Copied from ``komugi.eval.ladder._table`` (``ladder.py:470-487``)**, not
    imported: importing it would drag ``komugi.eval.arena`` and torch into this
    module for a formatting helper, and promoting it to a shared home would put
    ``ladder.py`` into this task's blast radius (design §5's T4 row offers both
    and this is the cheaper one).

    First column left-aligned (it is an identifier), the rest right-aligned
    (they are numbers). No separator rule: the report is read in a terminal and
    in a commit message, and the widths already do the work.
    """
    columns = list(zip(header, *rows, strict=True))
    widths = [max(len(cell) for cell in column) for column in columns]

    def render(cells: Sequence[str]) -> str:
        """Pad one row's cells to the column widths."""
        return "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()

    return [render(header), *(render(row) for row in rows)]


def _elo(value: float | None) -> str:
    """Return an Elo number as the report prints it, or :data:`_MISSING`."""
    return _MISSING if value is None else f"{value:.1f}"


def _interval(lo: float | None, hi: float | None) -> str:
    """Return a 95% interval as ``[lo, hi]``, or :data:`_MISSING`."""
    if lo is None or hi is None:
        return _MISSING
    return f"[{lo:.1f}, {hi:.1f}]"


ContemptDeclaration = Mapping[str, float | str]
"""One family's contempt: its value, the scale it lives on, and that scale's neutral."""

ContemptMap = Mapping[str, ContemptDeclaration | None]
"""§D7.3's per-family contempt map. ``None`` means unknown, and is not ``0``."""

POOL_CONTEMPT: Final[ContemptMap] = {
    "komugi": {"value": 0.0, "scale": "[-1,1]", "neutral": 0.0},
    "v1": {"value": 0.5, "scale": "[0,1]", "neutral": 0.5},
    "rawmcts": {"value": 0.0, "scale": "[-1,1]", "neutral": 0.0},
    "external": None,
}
"""§D7.3's contempt declaration, per family, with each scale named.

A **map**, not a scalar, and the reason is that the families do not share a
scale. Komugi's ``draw_score`` lives on ``[-1, 1]`` with ``0.0`` neutral
(``arena.py:323-330``); v1's MCTS ``drawScore`` is pinned at ``0.5`` on v1's own
``[0, 1]`` convention (``tickets/KOM-023.md:92``) -- neutral, and therefore
*consistent* with ours, but not the same number. A single ``"contempt": 0.0``
would read as a claim about both. ``external`` is ``None`` because the site's
contempt is unknown (U4) and inheriting ours would be an invention.

``ab`` is absent because the family has no members yet (§D1.1); when KOM-031
lands it declares its own entry and :func:`contempt_for` is what refuses an
epoch that forgot to.
"""


def contempt_for(families: Iterable[str]) -> ContemptMap:
    """Return the contempt declarations for exactly `families`.

    Raises rather than defaulting, because a family playing at an undeclared
    contempt is the state §D7.3 exists to prevent: *"a number whose contempt is
    not on the page is not reproducible"*, and inheriting komugi's ``0.0`` for
    an engine on another scale would put a false statement in the manifest.

    Raises:
        ValueError: if a family has no pre-registered declaration.
    """
    selected: dict[str, ContemptDeclaration | None] = {}
    for family in sorted(set(families)):
        if family not in POOL_CONTEMPT:
            raise ValueError(
                f"family {family!r} has no pre-registered contempt declaration "
                f"(§D7.3); the pool's map covers {sorted(POOL_CONTEMPT)}. A "
                f"contempt is not inherited across scales -- declare this "
                f"family's value, its scale and that scale's neutral"
            )
        selected[family] = POOL_CONTEMPT[family]
    return selected


@dataclass(frozen=True, slots=True)
class RatingRow:
    """One row of §D9.1(2)'s BayesElo table.

    ``lo``/``hi`` are §D5.2's paired-bootstrap percentile interval and are
    **required**: a rating published without one is a point estimate wearing a
    measurement's clothes, and ``EloResult`` carries no dispersion of its own.
    ``delta_external`` and ``delta_visits`` are this row's rating minus its rating
    in the without-External (§D6.1(5)) and without-visits (§D8.2) fits -- the
    contamination as a number on the page rather than a claim in a design
    document.
    """

    id: str
    family: str
    elo: float
    lo: float
    hi: float
    games: int
    score_rate: float
    delta_external: float | None = None
    delta_visits: float | None = None
    flags: tuple[str, ...] = ()

    @property
    def is_anchor(self) -> bool:
        """Return whether this row carries the gauge, per the stored flag."""
        return FLAG_ANCHOR in self.flags


def rating_rows(
    *,
    ratings: Mapping[str, float],
    intervals: Mapping[str, tuple[float, float]],
    roster_map: Mapping[str, str],
    match_set: MatchSet,
    anchor_id: str,
    ratings_no_external: Mapping[str, float] | None = None,
    ratings_no_visits: Mapping[str, float] | None = None,
    extra_flags: Mapping[str, Sequence[str]] | None = None,
) -> tuple[RatingRow, ...]:
    """Return §D9.1(2)'s rows, in **family-then-strength** order.

    Family first because grouping by lineage is the thing ``DESIGN.html:260``
    says the pool exists to make visible; strength within a family because that
    is the only ordering a reader can scan for a gap. ``run_ladder`` sorts by
    *ladder* order for a different question and argues why
    (``ladder.py:243-259``); neither sorts by rating alone.

    The **anchor flag comes from `anchor_id`**, which the caller reads out of
    the stored record. It is never re-derived by looking for a ``0.0`` rating:
    on a balanced pool every rating is exactly ``0.0`` and that scan names an
    arbitrary player (``ladder.py:218-241``, §6 node 37).

    Family is read from `roster_map`, never parsed from an id's prefix
    (§D1.2 property 3).

    Args:
        ratings: the published joint fit's ratings.
        intervals: each id's 95% percentile interval; required for every id.
        roster_map: ``id -> family``.
        match_set: the pooled records, for games played and score rate.
        anchor_id: the gauge, carried in the record.
        ratings_no_external: §D6.1(5)'s fit, for ``delta_external``.
        ratings_no_visits: §D8.2's fit, for ``delta_visits``.
        extra_flags: further §D9.1(2) flags per id (``frozen``, ``reference``,
            ``one-sided``, ``exempt-ceiling``, ``unequal-compute``,
            ``bracket-saturated``).

    Raises:
        ValueError: if a rated id has no interval, or names no family.
    """
    flags_by_id = {} if extra_flags is None else extra_flags
    rows: list[RatingRow] = []
    for entrant_id, elo in ratings.items():
        interval = intervals.get(entrant_id)
        if interval is None:
            raise ValueError(
                f"rating row {entrant_id!r} carries no interval; EloResult has "
                f"no dispersion of its own (elo.py:109-124), so a row without "
                f"§D5.2's bootstrap interval would publish a point estimate as "
                f"though it were a measurement"
            )
        family = roster_map.get(entrant_id)
        if family is None:
            raise ValueError(
                f"rating row {entrant_id!r} is not in the roster map, so its "
                f"family is unknown; the prefix is not consulted (§D1.2)"
            )
        flags = list(flags_by_id.get(entrant_id, ()))
        if entrant_id == anchor_id and FLAG_ANCHOR not in flags:
            flags.insert(0, FLAG_ANCHOR)
        games = match_set.games_played_by(entrant_id)
        rows.append(
            RatingRow(
                id=entrant_id,
                family=family,
                elo=elo,
                lo=interval[0],
                hi=interval[1],
                games=games,
                score_rate=(
                    match_set.score_of(entrant_id) / games if games else 0.0
                ),
                delta_external=(
                    None
                    if ratings_no_external is None
                    or entrant_id not in ratings_no_external
                    else elo - ratings_no_external[entrant_id]
                ),
                delta_visits=(
                    None
                    if ratings_no_visits is None
                    or entrant_id not in ratings_no_visits
                    else elo - ratings_no_visits[entrant_id]
                ),
                flags=tuple(flags),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.family, -row.elo, row.id)))


def delta_max(rows: Sequence[RatingRow], *, external: bool) -> float | None:
    """Return the largest ``|Δ|`` across `rows`, or ``None`` if none carries one.

    §D9.1(1) prints both maxima in the header, because the per-row column
    answers *"which row moved"* and the header answers *"could any published
    number have moved enough to matter"* -- and only the second is readable at
    a glance.
    """
    deltas: list[float] = []
    for row in rows:
        delta = row.delta_external if external else row.delta_visits
        if delta is not None:
            deltas.append(abs(delta))
    return max(deltas) if deltas else None


@dataclass(frozen=True, slots=True)
class MatchupRow:
    """One scheduled matchup as §D9.1 and §D9.2 record it."""

    a: str
    b: str
    games: int
    decisive: int
    draws: int
    mean_plies: float
    kind: str
    book_epoch: str
    pair_namespace: str
    unequal_compute: bool
    void_rate: float = 0.0
    admissible: bool = True


@dataclass(frozen=True, slots=True)
class SeatAggregate:
    """One entrant's pool-level per-seat summary (§D9.1(3))."""

    entrant: str
    white_games: int
    white_score_rate: float
    black_games: int
    black_score_rate: float

    @property
    def difference(self) -> float:
        """Return white minus black score rate: the seat asymmetry itself."""
        return self.white_score_rate - self.black_score_rate


def seat_aggregates(match_set: MatchSet) -> tuple[SeatAggregate, ...]:
    """Return each entrant's white and black score rates over the whole pool.

    ``DESIGN.html:260`` demands the per-seat matrices for a stated reason --
    *"seat-swapping must expose seat weakness, not average it away"* -- and v1's
    most durable finding is a seat one (*"black: 0 draws in 29 straight anchor
    games"*, ``DESIGN.html:264``). The per-matchup matrices show it link by
    link; this is the column a reader scans first.
    """
    white_games: dict[str, int] = {}
    white_score: dict[str, float] = {}
    black_games: dict[str, int] = {}
    black_score: dict[str, float] = {}
    for record in match_set.records:
        white_games[record.white] = white_games.get(record.white, 0) + 1
        white_score[record.white] = white_score.get(record.white, 0.0) + record.result
        black_games[record.black] = black_games.get(record.black, 0) + 1
        black_score[record.black] = black_score.get(record.black, 0.0) + (
            WHITE_WIN - record.result
        )
    return tuple(
        SeatAggregate(
            entrant=entrant,
            white_games=white_games.get(entrant, 0),
            white_score_rate=(
                white_score.get(entrant, 0.0) / white_games[entrant]
                if white_games.get(entrant)
                else 0.0
            ),
            black_games=black_games.get(entrant, 0),
            black_score_rate=(
                black_score.get(entrant, 0.0) / black_games[entrant]
                if black_games.get(entrant)
                else 0.0
            ),
        )
        for entrant in match_set.players()
    )


@dataclass(frozen=True, slots=True)
class VisitsPoint:
    """One grid point of §D8's Elo-vs-visits curve."""

    visits: int
    entrant: str
    elo: float
    lo: float
    hi: float
    n_per_matchup: int


@dataclass(frozen=True, slots=True)
class VisitsBlock:
    """§D9.1(5)'s block: the curve, its counts, and its measured cost.

    ``epoch_budget_fraction`` is read off the **measured** wall clocks, which is
    what §D8.3 pre-registers the cap over -- *"the cap is enforced on the
    measured wall clock of the first run, not on this estimate"*. A block over
    the cap is printed as over cap; it is not withheld and it is not silently
    published as though it were under (§6 node 46).
    """

    milestone: str
    points: tuple[VisitsPoint, ...]
    wall_clock_s: float
    epoch_budget_fraction: float
    cap: float = VISITS_BUDGET_CAP
    missing: tuple[int, ...] = ()
    """Budgeted grid points that produced no rating, carried rather than dropped.

    A visits leg whose records never reached the match set used to fall out of
    the curve with no note, and the block then printed three points where four
    were budgeted while ``monotone`` and ``span`` were computed over the
    survivors as though the curve were complete (T4 review F13). A missing
    measurement and a measured curve must not look the same -- §D9.1(3)'s own
    argument about the clean audit, applied here.
    """

    @property
    def over_cap(self) -> bool:
        """Return whether the measured fraction breached §D8.3's cap."""
        return self.epoch_budget_fraction > self.cap

    @property
    def monotone(self) -> bool | None:
        """Return whether the curve rises at every step (§D8.4(1)).

        ``None`` when a budgeted point is missing: monotonicity over the
        survivors of an incomplete curve is not the claim §D8.4(1) makes.
        """
        if self.missing:
            return None
        return all(
            later.elo >= earlier.elo
            for earlier, later in zip(self.points, self.points[1:], strict=False)
        )

    @property
    def span(self) -> float | None:
        """Return ``r(top) - r(bottom)``; ``None`` on an incomplete curve.

        ``None`` on fewer than two points, and ``None`` when a budgeted point
        is :attr:`missing` -- the span of a curve whose top was never measured
        is not the span of the curve.
        """
        if self.missing or len(self.points) < 2:
            return None
        return self.points[-1].elo - self.points[0].elo

    @property
    def gains(self) -> tuple[float, ...]:
        """Return the per-step gains, which §D8.4(2) predicts are shrinking."""
        return tuple(
            later.elo - earlier.elo
            for earlier, later in zip(self.points, self.points[1:], strict=False)
        )


@dataclass(frozen=True, slots=True)
class ReportHeader:
    """§D9.1(1): everything a reader needs before the first number.

    ``reference_book_epoch`` is printed whenever the re-used reference x
    reference records carry a different era from this epoch's book (§D2.2(a):
    they keep the era they were played in, forever). Printing it is what keeps
    that reuse *declared* rather than discovered.
    """

    epoch: int
    epoch_book_id: str
    anchor_id: str
    contempt: ContemptMap
    pinned_budget: Mapping[str, float]
    engine_rules_version: int
    r5_branch: str
    entrants: int
    matchups: int
    games: int
    """Every game the pool has ever played (§D9.2 property 4: append-only)."""

    converged: bool
    """The **published joint** fit's, which is what the record's ``fit.converged``
    names. §D2.3's other three are reported through :attr:`converged_all`."""

    epoch_games: int = 0
    """This epoch's own games, printed beside :attr:`games`.

    Two counts with one label is what the merged header printed -- entrants and
    matchups scoped to the epoch, games scoped to the whole pool (T4 review
    F14). Both are printed, each named.
    """

    converged_all: bool = True
    """Whether all four §D2.3 fits converged; the banner's own condition."""

    clamped_fits: tuple[str, ...] = ()
    """Which fits clamped, by their record key, so the banner can name them."""

    gate_fired: bool = False
    """§D5.4 step 1: either window fired, so the headline is withheld."""

    components: int = 1
    one_sided_only: bool = False
    v1_absent_by_branch_three: bool = False
    reference_book_epoch: str | None = None
    external_games: int = 0
    external_status: str = ""
    external_delta_max: float | None = None
    visits_delta_max: float | None = None
    bracket_saturated: bool = False


@dataclass(frozen=True, slots=True)
class EpochReport:
    """Everything §D9.1 renders, gathered so the renderer stays a renderer.

    The driver builds this; :func:`render_epoch_report` turns it into text and
    decides nothing. That split is what lets §6 nodes 35-37 and 51 exercise the
    report on planted inputs without playing a game or writing a record.
    """

    header: ReportHeader
    ratings: tuple[RatingRow, ...]
    matchups: tuple[MatchupRow, ...] = ()
    match_set: MatchSet = field(default_factory=lambda: MatchSet.of(()))
    violations: tuple[PairViolation, ...] = ()
    divergence: tuple[DivergenceResult, ...] = ()
    visits: VisitsBlock | None = None
    relink_seed: int | None = None


def _header_lines(header: ReportHeader) -> list[str]:
    """Return §D9.1(1)'s provenance lines."""
    contempt = ", ".join(
        f"{family}={'unknown' if declared is None else declared['value']}"
        f"{'' if declared is None else ' on ' + str(declared['scale'])}"
        for family, declared in sorted(header.contempt.items())
    )
    budget = ", ".join(
        f"{name}={value:g}" for name, value in sorted(header.pinned_budget.items())
    )
    lines = [
        # Every count says which scope it is in. `games` is every game the pool
        # has ever played (§D9.2 property 4) while `entrants` and `matchups`
        # are this epoch's, and one sentence carrying two scopes under one
        # label is what the merged header printed (T4 review F14).
        f"pool epoch {header.epoch}: {header.entrants} entrants, "
        f"{header.matchups} matchups this epoch, "
        f"{header.epoch_games} games this epoch, "
        f"{header.games} games pooled",
        f"epoch_book_id: {header.epoch_book_id}",
        # **Not** the epoch's own book id when the field is absent: from era 2
        # the reference links keep era 1's openings forever (§D2.2(a)), so a
        # fallback would state a false equality between the reference field's
        # era and this epoch's -- the "declared, not discovered" property node
        # 49b exists for, inverted (T4 review F10).
        f"reference_book_epoch: "
        f"{header.reference_book_epoch or UNDECLARED_REFERENCE_EPOCH}",
        f"anchor: {header.anchor_id} at 0.0 Elo (carried, never re-derived)",
        f"contempt: {contempt}",
        f"pinned budget: {budget}; engine_rules_version="
        f"{header.engine_rules_version}",
        f"R5 branch: {header.r5_branch}, one_sided_only={header.one_sided_only}",
        f"max |Δ(external)|: {_elo(header.external_delta_max)}; "
        f"max |Δ(visits)|: {_elo(header.visits_delta_max)}",
    ]
    if header.external_status:
        lines.append(
            f"external leg: {header.external_games} games, {header.external_status}"
        )
    if header.bracket_saturated:
        lines.append(
            "bracket_saturated: the milestone outrates every reference entrant "
            "-- the pool has run out of opponents (§D2.2, Q5)"
        )
    return lines


def _banners(header: ReportHeader) -> list[str]:
    """Return the banners that must sit **above** the tables, in a fixed order."""
    banners: list[str] = []
    if header.gate_fired:
        banners.append(DIVERGENCE_GATE_BANNER)
    if not header.converged or not header.converged_all:
        # The condition is the field the record names: `fit.converged` is the
        # published joint fit's and `fit.converged_all` is the conjunction over
        # §D2.3's four, so an epoch whose certification fit clamps banners the
        # page *and* the artefact says which one (T4 review F7).
        banners.append(
            NOT_CONVERGED_BANNER
            if not header.clamped_fits
            else NOT_CONVERGED_BANNER
            + CLAMPED_FITS_SUFFIX.format(fits=", ".join(header.clamped_fits))
        )
    if header.components > 1:
        banners.append(COMPONENTS_BANNER.format(components=header.components))
    if header.one_sided_only:
        banners.append(ONE_SIDED_BANNER)
    if header.v1_absent_by_branch_three:
        banners.append(BRANCH_THREE_BANNER)
    return banners


def _rating_table(rows: Sequence[RatingRow]) -> list[str]:
    """Return §D9.1(2)'s table: no row prints a bare rating."""
    if not rows:
        return ["(no entrants)"]
    return _table(
        (
            "id",
            "family",
            "rating",
            "95% interval",
            "games",
            "score",
            "Δ(external)",
            "Δ(visits)",
            "flags",
        ),
        [
            (
                row.id,
                row.family,
                f"{row.elo:.1f}",
                _interval(row.lo, row.hi),
                str(row.games),
                f"{row.score_rate:.3f}",
                _elo(row.delta_external),
                _elo(row.delta_visits),
                " ".join(row.flags),
            )
            for row in rows
        ],
    )


def _matchup_table(rows: Sequence[MatchupRow]) -> list[str]:
    """Return the per-matchup summary, with each link's own book epoch."""
    if not rows:
        return ["(no matchups)"]
    return _table(
        (
            "matchup",
            "kind",
            "games",
            "decisive",
            "draws",
            "mean plies",
            "book epoch",
            "namespace",
            "void rate",
            "flags",
        ),
        [
            (
                f"{row.a} vs {row.b}",
                row.kind,
                str(row.games),
                str(row.decisive),
                str(row.draws),
                f"{row.mean_plies:.1f}",
                row.book_epoch,
                row.pair_namespace or "(none)",
                f"{row.void_rate:.3f}",
                " ".join(
                    flag
                    for flag, on in (
                        (FLAG_UNEQUAL_COMPUTE, row.unequal_compute),
                        ("inadmissible", not row.admissible),
                    )
                    if on
                ),
            )
            for row in rows
        ],
    )


def _seat_block(
    match_set: MatchSet, violations: Sequence[PairViolation]
) -> list[str]:
    """Return §D9.1(3): the per-seat matrices, the aggregate, and the audit."""
    matrix = seat_matrix(match_set)
    lines: list[str] = ["per-seat matrices (nothing symmetrised)"]
    if matrix:
        lines.extend(
            _table(
                ("white", "black", "games", "wins", "draws", "losses", "score"),
                [
                    (
                        tally.white,
                        tally.black,
                        str(tally.games),
                        str(tally.wins),
                        str(tally.draws),
                        str(tally.losses),
                        f"{tally.score_rate:.3f}",
                    )
                    for tally in matrix.values()
                ],
            )
        )
    else:
        lines.append("(no games)")
    aggregates = seat_aggregates(match_set)
    lines.extend(["", "per-seat aggregate"])
    if aggregates:
        lines.extend(
            _table(
                ("entrant", "white n", "white", "black n", "black", "white-black"),
                [
                    (
                        aggregate.entrant,
                        str(aggregate.white_games),
                        f"{aggregate.white_score_rate:.3f}",
                        str(aggregate.black_games),
                        f"{aggregate.black_score_rate:.3f}",
                        f"{aggregate.difference:+.3f}",
                    )
                    for aggregate in aggregates
                ],
            )
        )
    else:
        lines.append("(no games)")
    lines.append("")
    if violations:
        lines.append(f"paired-book audit: {len(violations)} violations")
        lines.extend(
            _table(
                ("pair_id", "reason", "detail"),
                [
                    (violation.pair_id, violation.reason, violation.detail)
                    for violation in violations
                ],
            )
        )
    else:
        lines.append(CLEAN_AUDIT)
    return lines


def _divergence_block(
    rows: Sequence[DivergenceResult], relink_seed: int | None
) -> list[str]:
    """Return §D9.1(4)'s two rows, with the gate's own statistic marked."""
    lines = ["divergence (§D5): one row per window, both evaluated every epoch"]
    if not rows:
        return [*lines, "(no windows)"]
    lines.extend(
        _table(
            (
                "window",
                "A -> B",
                "G_in",
                "G_cross",
                "d",
                "d 95%",
                "G_cross_no_external",
                "d_no_external",
                "d_no_external 95%",
                "T",
                "discard",
                "bootstrap s",
                "verdict",
                "flags",
            ),
            [
                (
                    str(row.window_kind.value),
                    "--" if row.window is None else f"{row.window[0]} -> {row.window[1]}",
                    _elo(row.g_in),
                    _elo(row.g_cross),
                    _elo(row.d),
                    _interval(row.lo, row.hi),
                    _elo(row.g_cross_no_external),
                    _elo(row.d_no_external),
                    _interval(row.lo_no_external, row.hi_no_external),
                    f"{row.threshold:.0f}",
                    _MISSING
                    if row.discard_rate is None
                    else f"{row.discard_rate:.1%}",
                    _MISSING
                    if row.bootstrap_wall_clock_s is None
                    else f"{row.bootstrap_wall_clock_s:.1f}",
                    row.message,
                    " ".join(
                        flag
                        for flag, on in (
                            ("external-drift", row.external_drift),
                            (
                                FLAG_BRACKET_SATURATED,
                                bool(row.bracket_saturated),
                            ),
                        )
                        if on
                    ),
                )
                for row in rows
            ],
        )
    )
    lines.append(
        f"gate statistic: {GATE_STATISTIC} -- "
        f"{PUBLISHED_STATISTIC} is published beside it and is never the predicate"
    )
    for row in rows:
        sign = row.sign
        if sign is not None and row.verdict is Verdict.FIRED:
            direction = (
                "in-family inflation" if sign > 0 else "in-family under-reading"
            )
            lines.append(
                f"{row.window_kind.value}: FIRED, sign {sign:+d} ({direction})"
            )
    if relink_seed is not None:
        lines.append(
            f"§D5.4 re-link diagnostic: relink_seed={relink_seed} "
            f"(a fresh, recorded seed -- a namespace change alone would replay "
            f"the original games move for move)"
        )
    return lines


def _visits_block(block: VisitsBlock) -> list[str]:
    """Return §D9.1(5)'s Elo-vs-visits block, cap verdict included."""
    lines = [f"Elo-vs-visits (§D8), milestone {block.milestone}"]
    if block.points:
        lines.extend(
            _table(
                ("visits", "entrant", "rating", "95% interval", "n per matchup", "gain"),
                [
                    (
                        str(point.visits),
                        point.entrant,
                        f"{point.elo:.1f}",
                        _interval(point.lo, point.hi),
                        str(point.n_per_matchup),
                        _MISSING
                        if index == 0
                        else f"{block.gains[index - 1]:+.1f}",
                    )
                    for index, point in enumerate(block.points)
                ],
            )
        )
    else:
        lines.append("(no grid points)")
    if block.missing:
        # Named, not dropped: three points where four were budgeted is a
        # different statement from a four-point curve (§D8.2, review F13).
        lines.append(
            f"MISSING grid points (budgeted, no rating): "
            f"{', '.join(str(visits) for visits in block.missing)} -- the "
            f"curve is incomplete, so monotone and span are not reported"
        )
    monotone = _MISSING if block.monotone is None else str(block.monotone)
    lines.append(
        f"monotone (§D8.4(1)): {monotone}; span: {_elo(block.span)} Elo"
    )
    verdict = "OVER CAP" if block.over_cap else "within cap"
    lines.append(
        f"measured wall clock: {block.wall_clock_s:.1f} s = "
        f"{block.epoch_budget_fraction:.1%} of the epoch, cap {block.cap:.0%} "
        f"-- {verdict}"
    )
    return lines


def render_epoch_report(report: EpochReport) -> str:
    """Return §D9.1's five blocks, in order, as one string.

    Banners come first, before any table, because ``converged=False``,
    ``components > 1``, R5 branch 2 and a fired divergence gate are each a
    property of the whole column that a reader must not be left to notice
    (§D9.1(2), §D4.1, §D5.4 step 1).
    """
    lines: list[str] = [
        *_header_lines(report.header),
        *_banners(report.header),
        "",
        "ratings (family, then strength)",
        *_rating_table(report.ratings),
        "",
        "matchups",
        *_matchup_table(report.matchups),
        "",
        *_seat_block(report.match_set, report.violations),
        "",
        *_divergence_block(report.divergence, report.relink_seed),
    ]
    if report.visits is not None:
        lines.extend(["", *_visits_block(report.visits)])
    lines.append("")
    return "\n".join(lines)
