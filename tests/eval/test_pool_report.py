"""§D9.1 nodes: the banners, the per-seat block, the rating rows, the contempt map.

Design: ``docs/design/kom-011d-ladder-pool.md`` (rev 4), §6 nodes 35, 36, 37
and 51.

Every fixture here is planted by hand and no game is played: the report is a
pure function of the record, which is exactly the property that lets these four
nodes run in milliseconds and lets a reviewer break the renderer and see them
go red. The one fixture that is *not* arbitrary is node 37's -- a **balanced**
pool in which every fitted rating is exactly ``0.0``, which is the case that
produced F1 on the in-family ladder (``ladder.py:218-241``,
``tickets/KOM-011.md:70-76``) and the only one on which "flag the anchor" and
"flag whichever row rates 0.0" disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from komugi.eval.divergence import (
    GATE_STATISTIC,
    THRESHOLD_SINGLE_ELO,
    DivergenceResult,
    Verdict,
    WindowKind,
)
from komugi.eval.elo import fit_elo
from komugi.eval.pool_report import (
    BRANCH_THREE_BANNER,
    CLEAN_AUDIT,
    COMPONENTS_BANNER,
    DIVERGENCE_GATE_BANNER,
    FLAG_ANCHOR,
    FLAG_REFERENCE,
    NOT_CONVERGED_BANNER,
    ONE_SIDED_BANNER,
    POOL_CONTEMPT,
    EpochReport,
    MatchupRow,
    RatingRow,
    ReportHeader,
    VisitsBlock,
    VisitsPoint,
    contempt_for,
    delta_max,
    rating_rows,
    render_epoch_report,
    seat_aggregates,
)
from komugi.eval.records import DRAW, WHITE_WIN, GameRecord, MatchSet
from komugi.eval.seats import (
    BOOK_MISMATCH,
    GAME_COUNT,
    PairViolation,
    check_paired_books,
    seat_matrix,
)

_ANCHOR: str = "v1/iter-0020@7eef51d0"
_A: str = "komugi/ckpt-3000@aaaa0001"
_B: str = "komugi/ckpt-6000@aaaa0002"
_REF: str = "rawmcts/n800@0f31ab77"

_ROSTER_MAP: dict[str, str] = {
    _ANCHOR: "v1",
    _A: "komugi",
    _B: "komugi",
    _REF: "rawmcts",
}

_BOOK: str = "E001-3f9c1a20"
_PINNED: dict[str, float] = {"m": 32.0, "n": 256.0, "gumbel_scale": 1.0, "ceiling": 257.0}


def _balanced_link(first: str, second: str, *, pairs: int = 2) -> list[GameRecord]:
    """Return `pairs` seat-swapped pairs in which **each side scores exactly half**.

    Pair ``k`` is one win for `first` and one for `second`, so every player's
    score rate is ``0.5`` in every matchup and :func:`fit_elo` returns exactly
    ``0.0`` for everyone. That is the balanced pool node 37 needs: on it, a
    report that identifies the anchor by scanning for a ``0.0`` rating names an
    arbitrary row and cannot be caught by any other fixture.
    """
    records: list[GameRecord] = []
    for index in range(pairs):
        pair_id = f"{first}:{second}:{index:05d}"
        records.append(
            GameRecord(
                white=first,
                black=second,
                result=WHITE_WIN,
                book_id=_BOOK,
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=second,
                black=first,
                result=WHITE_WIN,
                book_id=_BOOK,
                pair_id=pair_id,
            )
        )
    return records


def _balanced_pool() -> MatchSet:
    """Return a connected, perfectly balanced four-entrant pool."""
    return MatchSet.of(
        [
            *_balanced_link(_A, _B),
            *_balanced_link(_B, _ANCHOR),
            *_balanced_link(_A, _ANCHOR),
            *_balanced_link(_REF, _ANCHOR),
        ]
    )


def _intervals(match_set: MatchSet) -> dict[str, tuple[float, float]]:
    """Return a planted interval per player, so a row can be built at all."""
    return {player: (-12.5, 12.5) for player in match_set.players()}


def _header(
    *,
    converged: bool = True,
    converged_all: bool | None = None,
    clamped_fits: tuple[str, ...] = (),
    gate_fired: bool = False,
    components: int = 1,
    one_sided_only: bool = False,
    v1_absent_by_branch_three: bool = False,
    entrants: int = 4,
    matchups: int = 4,
    games: int = 16,
    epoch_games: int | None = None,
    reference_book_epoch: str | None = None,
    external_delta_max: float | None = None,
    visits_delta_max: float | None = None,
) -> ReportHeader:
    """Return a healthy §D9.1(1) header, with any part overridden."""
    return ReportHeader(
        epoch=7,
        epoch_book_id=_BOOK,
        anchor_id=_ANCHOR,
        contempt=POOL_CONTEMPT,
        pinned_budget=_PINNED,
        engine_rules_version=3,
        r5_branch="proceed",
        entrants=entrants,
        matchups=matchups,
        games=games,
        epoch_games=games if epoch_games is None else epoch_games,
        converged=converged,
        converged_all=converged if converged_all is None else converged_all,
        clamped_fits=clamped_fits,
        gate_fired=gate_fired,
        components=components,
        one_sided_only=one_sided_only,
        v1_absent_by_branch_three=v1_absent_by_branch_three,
        reference_book_epoch=reference_book_epoch,
        external_delta_max=external_delta_max,
        visits_delta_max=visits_delta_max,
    )


def _report(
    *,
    header: ReportHeader | None = None,
    match_set: MatchSet | None = None,
    rows: Sequence[RatingRow] | None = None,
    violations: Sequence[PairViolation] = (),
    divergence: Sequence[DivergenceResult] = (),
    visits: VisitsBlock | None = None,
) -> str:
    """Render a report over the balanced pool, with any part replaced."""
    pool = _balanced_pool() if match_set is None else match_set
    built = (
        rating_rows(
            ratings=fit_elo(pool, anchor=_ANCHOR).ratings,
            intervals=_intervals(pool),
            roster_map=_ROSTER_MAP,
            match_set=pool,
            anchor_id=_ANCHOR,
        )
        if rows is None
        else tuple(rows)
    )
    return render_epoch_report(
        EpochReport(
            header=_header() if header is None else header,
            ratings=built,
            matchups=(
                MatchupRow(
                    a=_A,
                    b=_B,
                    games=4,
                    decisive=4,
                    draws=0,
                    mean_plies=61.0,
                    kind="in-family",
                    book_epoch=_BOOK,
                    pair_namespace="E007",
                    unequal_compute=False,
                ),
            ),
            match_set=pool,
            violations=tuple(violations),
            divergence=tuple(divergence),
            visits=visits,
        )
    )


def _first_table_index(report: str) -> int:
    """Return the line index of the first table's header row."""
    lines = report.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("id ") and "95% interval" in line:
            return index
    raise AssertionError(f"no rating table found in report:\n{report}")


# --------------------------------------------------------------------------
# §6 node 35
# --------------------------------------------------------------------------
def _banner_index(report: str, banner: str) -> int:
    """Return the line index of `banner`, failing loudly when it is absent.

    A line index, not a substring offset: *"above the tables"* is a statement
    about rendered lines, and the report's own table header is what
    :func:`_first_table_index` locates.
    """
    lines = report.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(banner):
            return index
    raise AssertionError(f"banner not rendered:\n{banner}\n---\n{report}")


def test_a_non_converged_fit_gets_a_banner_above_the_tables() -> None:
    """§D9.1(2): a clamp is announced above the numbers, never under them.

    **Review F2** and the review's execution record for **M6**: moving
    ``_banners`` below ``_rating_table`` was caught by five other tests while
    this one -- the node whose whole subject is the ordering -- stayed green.
    Every banner this renderer emits is therefore checked here by line index
    against the first table's own header row, on a fixture where it actually
    renders.
    """
    # Red against a renderer that prints the convergence state only in the fit
    # line, or as a footnote after the table: `converged=False` means the pool
    # is not identified from these games and every number below is a clamp
    # (elo.py:70-77), which is the one property a reader must not be left to
    # notice for themselves. The same for a comparison graph in two pieces --
    # which still reports `converged=True` (ab_endpoint.py:426-463), so the
    # banner is the only thing that says it.
    healthy = _report()
    assert NOT_CONVERGED_BANNER not in healthy
    assert "COMPONENTS" not in healthy
    assert ONE_SIDED_BANNER not in healthy
    assert BRANCH_THREE_BANNER not in healthy
    assert DIVERGENCE_GATE_BANNER not in healthy

    clamped = _report(header=_header(converged=False))
    assert NOT_CONVERGED_BANNER in clamped
    assert clamped.index(NOT_CONVERGED_BANNER) < clamped.index("95% interval")
    assert _banner_index(clamped, NOT_CONVERGED_BANNER) < _first_table_index(
        clamped
    )

    split = _report(header=_header(components=2))
    expected = COMPONENTS_BANNER.format(components=2)
    assert expected in split
    assert _banner_index(split, expected) < _first_table_index(split)

    # R5 branch 2 is a property of the whole column and a per-row flag cannot
    # express it (§D4.1, F7), so it is a banner in the same idiom.
    one_sided = _report(header=_header(one_sided_only=True))
    assert ONE_SIDED_BANNER in one_sided
    assert _banner_index(one_sided, ONE_SIDED_BANNER) < _first_table_index(
        one_sided
    )

    absent_v1 = _report(header=_header(v1_absent_by_branch_three=True))
    assert BRANCH_THREE_BANNER in absent_v1
    assert _banner_index(absent_v1, BRANCH_THREE_BANNER) < _first_table_index(
        absent_v1
    )

    # §D5.4 step 1: a fired gate withholds the headline, which is a property of
    # the whole published column and therefore the same idiom again (F2).
    fired = _report(header=_header(gate_fired=True))
    assert DIVERGENCE_GATE_BANNER in fired
    assert _banner_index(fired, DIVERGENCE_GATE_BANNER) < _first_table_index(
        fired
    )


def test_the_banner_names_the_subfit_that_clamped_and_the_record_agrees() -> None:
    """§D2.3/§D9.2: the banner's condition is the field the record names.

    **Review F7.** The record publishes ``fit.converged`` for the **published
    joint** fit and the page banners on the conjunction over all four, so an
    epoch in which exactly one sub-fit clamps used to put ``converged: true``
    in the artefact and ``NOT CONVERGED`` on the page with nothing saying which
    was the claim. Red against a renderer whose banner reads the joint flag
    (no banner at all here) and against one that banners without naming the
    fit, which is the state that leaves the two readings irreconcilable.
    """
    header = _header(
        converged=True,
        converged_all=False,
        clamped_fits=("ratings_certification",),
    )
    # The fixture is the discriminating half: the joint fit converged, so a
    # banner keyed on `converged` alone renders nothing here.
    assert header.converged
    assert not header.converged_all

    report = _report(header=header)
    assert NOT_CONVERGED_BANNER in report
    banner = report.splitlines()[_banner_index(report, NOT_CONVERGED_BANNER)]
    assert "ratings_certification" in banner
    assert _banner_index(report, NOT_CONVERGED_BANNER) < _first_table_index(report)

    # And the plain banner still renders verbatim when no fit is named, which
    # is the joint-fit case node 35 asserts.
    plain = _report(header=_header(converged=False))
    assert NOT_CONVERGED_BANNER in plain.splitlines()


# --------------------------------------------------------------------------
# §6 node 36
# --------------------------------------------------------------------------
def test_the_per_seat_block_is_present_and_says_clean_when_it_is() -> None:
    """§D9.1(3): an absent section and a clean one must not look the same."""
    # Red against a renderer that omits the per-seat block when there is nothing
    # to report -- the exact case where a reader cannot tell "the audit passed"
    # from "the audit was never run", which is the whole reason the design
    # spells the clean line out.
    pool = _balanced_pool()
    assert check_paired_books(pool) == ()
    clean = _report(match_set=pool)
    assert CLEAN_AUDIT in clean
    assert "per-seat matrices" in clean
    assert "per-seat aggregate" in clean
    # Nothing is symmetrised: every ordered seating that occurred gets a line,
    # so a four-link pool prints eight of them and not four.
    lines = clean.splitlines()
    start = lines.index("per-seat matrices (nothing symmetrised)")
    end = lines.index("per-seat aggregate")
    seatings = seat_matrix(pool)
    assert len(seatings) == 8
    assert len(lines[start + 2 : end - 1]) == len(seatings)

    violated = _report(
        match_set=pool,
        violations=(
            PairViolation(
                pair_id=f"{_A}:{_B}:00000",
                reason=GAME_COUNT,
                detail="expected 2 games, found 1",
            ),
            PairViolation(
                pair_id=f"{_A}:{_B}:00001",
                reason=BOOK_MISMATCH,
                detail="'E001-a' then 'E001-b'",
            ),
        ),
    )
    assert CLEAN_AUDIT not in violated
    assert "paired-book audit: 2 violations" in violated
    assert GAME_COUNT in violated
    assert BOOK_MISMATCH in violated


def test_the_seat_aggregate_reports_each_seat_separately() -> None:
    """§D9.1(3): seat-swapping must expose seat weakness, not average it away."""
    # Red against an aggregate that pools the two seats into one score rate --
    # which is exactly the averaging DESIGN.html:260 forbids, and which would
    # report 0.5 for a player that scores 1.0 as white and 0.0 as black.
    seated = MatchSet.of(
        [
            GameRecord(
                white=_A, black=_B, result=WHITE_WIN, book_id=_BOOK, pair_id="p:0"
            ),
            GameRecord(
                white=_B, black=_A, result=WHITE_WIN, book_id=_BOOK, pair_id="p:0"
            ),
        ]
    )
    by_entrant = {
        aggregate.entrant: aggregate for aggregate in seat_aggregates(seated)
    }
    assert by_entrant[_A].white_score_rate == pytest.approx(1.0)
    assert by_entrant[_A].black_score_rate == pytest.approx(0.0)
    assert by_entrant[_A].difference == pytest.approx(1.0)
    assert by_entrant[_B].white_score_rate == pytest.approx(1.0)
    assert by_entrant[_B].black_score_rate == pytest.approx(0.0)


# --------------------------------------------------------------------------
# §6 node 37
# --------------------------------------------------------------------------
def test_every_rating_row_carries_an_interval_and_the_anchor_is_marked_from_the_record() -> None:
    """§D9.1(2): the anchor flag is read from the record, never re-derived."""
    # The fixture is the discriminating half. On a *balanced* pool every rating
    # is exactly 0.0 -- each player scores half its games, so elo.py's MM step
    # divides a number by itself -- and a report that names the anchor by
    # scanning for a 0.0 rating names an arbitrary row. That is F1 of the 011c
    # review, verbatim (ladder.py:218-241), and this node is red against any
    # re-derivation by rating.
    pool = _balanced_pool()
    fit = fit_elo(pool, anchor=_ANCHOR)
    assert fit.converged
    assert all(rating == 0.0 for rating in fit.ratings.values())

    rows = rating_rows(
        ratings=fit.ratings,
        intervals=_intervals(pool),
        roster_map=_ROSTER_MAP,
        match_set=pool,
        anchor_id=_ANCHOR,
        extra_flags={_REF: (FLAG_REFERENCE,)},
    )
    flagged = [row.id for row in rows if row.is_anchor]
    assert flagged == [_ANCHOR]
    assert all(row.elo == 0.0 for row in rows)

    # Not one row prints a bare rating: every one carries its interval.
    report = _report(match_set=pool, rows=rows)
    by_id: dict[str, str] = {}
    for row in rows:
        line = next(
            line for line in report.splitlines() if line.startswith(row.id + " ")
        )
        assert "[-12.5, 12.5]" in line
        by_id[row.id] = line.rstrip()
    assert by_id[_ANCHOR].endswith(FLAG_ANCHOR)
    assert by_id[_REF].endswith(FLAG_REFERENCE)
    assert not by_id[_A].endswith(FLAG_ANCHOR)
    assert not by_id[_B].endswith(FLAG_ANCHOR)
    # Family, then strength: the komugi rows are adjacent (DESIGN.html:260's
    # own reason for the ordering), not interleaved with the reference field.
    families = [row.family for row in rows]
    assert families == sorted(families)

    # A row whose interval is missing is refused rather than published bare.
    with pytest.raises(ValueError, match="carries no interval"):
        rating_rows(
            ratings=fit.ratings,
            intervals={},
            roster_map=_ROSTER_MAP,
            match_set=pool,
            anchor_id=_ANCHOR,
        )


def test_the_deltas_and_their_maxima_are_read_off_the_two_extra_fits() -> None:
    """§D9.1(1)-(2): Δ(external) and Δ(visits) per row, both maxima in the header."""
    # Red against a report that publishes one table and calls the contamination
    # argued-about rather than measured (§D6.1(5), §D8.2).
    pool = _balanced_pool()
    fit = fit_elo(pool, anchor=_ANCHOR)
    rows = rating_rows(
        ratings=fit.ratings,
        intervals=_intervals(pool),
        roster_map=_ROSTER_MAP,
        match_set=pool,
        anchor_id=_ANCHOR,
        ratings_no_external={player: -4.0 for player in fit.ratings},
        ratings_no_visits={player: 2.5 for player in fit.ratings},
    )
    assert all(row.delta_external == pytest.approx(4.0) for row in rows)
    assert all(row.delta_visits == pytest.approx(-2.5) for row in rows)
    assert delta_max(rows, external=True) == pytest.approx(4.0)
    assert delta_max(rows, external=False) == pytest.approx(2.5)
    assert delta_max((), external=True) is None

    report = _report(
        match_set=pool,
        rows=rows,
        header=_header(external_delta_max=4.0, visits_delta_max=2.5),
    )
    assert "Δ(external)" in report
    assert "Δ(visits)" in report
    assert "max |Δ(external)|: 4.0" in report
    assert "max |Δ(visits)|: 2.5" in report


# --------------------------------------------------------------------------
# §6 node 51
# --------------------------------------------------------------------------
def test_the_manifest_carries_contempt_per_family_with_its_scale() -> None:
    """§D7.3: contempt is a map, because the families do not share a scale."""
    # Red against a single scalar (F23). Komugi's draw_score lives on [-1, 1]
    # with 0.0 neutral (arena.py:323-330) and v1's MCTS drawScore is pinned at
    # 0.5 on v1's own [0, 1] convention (tickets/KOM-023.md:92) -- neutral, and
    # therefore consistent with ours, but not the same number. One number would
    # read as a claim about both.
    assert isinstance(POOL_CONTEMPT, Mapping)
    komugi = POOL_CONTEMPT["komugi"]
    assert komugi is not None
    assert komugi["value"] == 0.0
    assert komugi["scale"] == "[-1,1]"
    assert komugi["neutral"] == 0.0

    v1 = POOL_CONTEMPT["v1"]
    assert v1 is not None
    assert v1["value"] == 0.5
    assert v1["scale"] == "[0,1]"
    assert v1["neutral"] == 0.5
    assert v1["value"] != komugi["value"]

    rawmcts = POOL_CONTEMPT["rawmcts"]
    assert rawmcts is not None
    assert rawmcts["scale"] == "[-1,1]"

    # null, not 0: the site's contempt is unknown (U4) and inheriting ours
    # would be an invention.
    assert POOL_CONTEMPT["external"] is None

    selected = contempt_for(["komugi", "v1", "external"])
    assert set(selected) == {"komugi", "v1", "external"}
    assert selected["external"] is None

    # A family with no declaration is refused rather than defaulted: a number
    # whose contempt is not on the page is not reproducible.
    with pytest.raises(ValueError, match="no pre-registered contempt"):
        contempt_for(["komugi", "ab"])

    report = _report()
    assert "contempt:" in report
    assert "komugi=0.0 on [-1,1]" in report
    assert "v1=0.5 on [0,1]" in report
    assert "external=unknown" in report


def test_the_divergence_block_names_its_own_gate_statistic() -> None:
    """§D9.1(4): both statistics printed, and the gate's marked as such."""
    # Red against a block that prints one number and leaves a reader to infer
    # which one the verdict was read from -- the state §D5.1 forbids by naming
    # `gate_statistic` in the record rather than inferring it.
    row = DivergenceResult(
        window_kind=WindowKind.SINGLE,
        window=(_A, _B),
        verdict=Verdict.FIRED,
        threshold=THRESHOLD_SINGLE_ELO,
        g_in=120.0,
        g_cross=20.0,
        g_cross_no_external=15.0,
        d=100.0,
        d_no_external=105.0,
        lo=60.0,
        hi=140.0,
        lo_no_external=65.0,
        hi_no_external=145.0,
        discard_rate=0.02,
        bootstrap_wall_clock_s=1.5,
        external_drift=True,
    )
    report = _report(divergence=(row,))
    assert GATE_STATISTIC in report
    assert "d_no_external 95%" in report
    assert "[60.0, 140.0]" in report
    assert "[65.0, 145.0]" in report
    assert "external-drift" in report
    assert "FIRED, sign +1" in report
    assert "in-family inflation" in report


def test_the_visits_block_prints_its_measured_cost_against_the_cap() -> None:
    """§D9.1(5)/§D8.3: the cap is read off the measured wall clock."""
    # Red against a block that publishes a curve without the cost that bought
    # it, and against one that silently publishes an over-cap harness.
    points = tuple(
        VisitsPoint(
            visits=visits,
            entrant=f"{_B}#n{visits}",
            elo=elo,
            lo=elo - 30.0,
            hi=elo + 30.0,
            n_per_matchup=count,
        )
        for visits, elo, count in ((100, 0.0, 32), (400, 40.0, 32), (1600, 60.0, 16))
    )
    under = VisitsBlock(
        milestone=_B,
        points=points,
        wall_clock_s=100.0,
        epoch_budget_fraction=0.2,
    )
    assert under.monotone
    assert under.span == pytest.approx(60.0)
    assert under.gains == pytest.approx((40.0, 20.0))
    assert not under.over_cap
    assert "within cap" in _report(visits=under)

    over = VisitsBlock(
        milestone=_B,
        points=points,
        wall_clock_s=100.0,
        epoch_budget_fraction=0.44,
    )
    assert over.over_cap
    rendered = _report(visits=over)
    assert "OVER CAP" in rendered
    assert "44.0% of the epoch" in rendered


def test_a_report_over_no_games_still_renders_every_block() -> None:
    """A pool with nothing in it prints its sections, not a stack trace."""
    # Red against a renderer whose fixed-width helper divides by zero on an
    # empty table -- the state an aborted epoch leaves, and the one where a
    # readable page matters most.
    empty = MatchSet.of(())
    report = render_epoch_report(
        EpochReport(header=_header(games=0, entrants=0, matchups=0), ratings=())
    )
    assert "(no entrants)" in report
    assert "(no matchups)" in report
    assert CLEAN_AUDIT in report
    assert "(no windows)" in report
    assert seat_aggregates(empty) == ()


def test_a_draw_only_link_scores_a_half_on_both_seats() -> None:
    """The seat aggregate reads ``GameRecord.result`` from white's view, once."""
    # Red against an aggregate that credits a draw as a loss to black -- the
    # orientation bug records.py:10-14 fixes once, for everyone downstream.
    drawn = MatchSet.of(
        [
            GameRecord(white=_A, black=_B, result=DRAW, book_id=_BOOK, pair_id="q:0"),
            GameRecord(white=_B, black=_A, result=DRAW, book_id=_BOOK, pair_id="q:0"),
        ]
    )
    for aggregate in seat_aggregates(drawn):
        assert aggregate.white_score_rate == pytest.approx(0.5)
        assert aggregate.black_score_rate == pytest.approx(0.5)
        assert aggregate.difference == pytest.approx(0.0)
