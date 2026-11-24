"""Grading is exact, and it discriminates -- KOM-259's acceptance, as tests.

The property that matters here is not "the grader runs". It is that the
grader can *tell things apart*: the old soft mate tier scored a slow win and
a correct win the same, and scored a win against weak defence as a win, so
the number it produced could only go up. Every test below is built so that a
grader stuck on one verdict fails it.

Three move sources drive all of it, and they are the three interesting kinds
of player:

``oracle_source``
    plays the solver's own line, so it must grade ``KEPT_MATE`` at exactly
    the optimal distance on all 21 committed forced-mate fixtures.
``quiet_source``
    plays the first legal move that does *not* give check. Under tsume
    semantics the attacker only ever moves with check, so handing the
    defender a free tempo throws the win -- ``LOST_MATE`` on the same 21.
``illegal_source``
    returns a SAN ``api.Gungi`` will not play -- ``ILLEGAL_MOVE``.

The pair (oracle, quiet) on the same fixture is the vacuity guard: their
outcomes must differ, which no constant grader can arrange.
"""

from __future__ import annotations

from typing import Final

import pytest
from tsume_fixtures import (
    FRIEND,
    MARSHAL_CAPTURE_ROWS,
    MINED,
    NO_MATE_PLIES,
    NO_MATE_POSITIONS,
    PERTURBATIONS,
    checking_moves,
)

from komugi.engine.api import Gungi
from komugi.search.tsume import DEFAULT_NODE_BUDGET, solve_tsume
from komugi.search.tsume_grade import (
    GradeOutcome,
    MoveSource,
    PuzzleGrade,
    SuiteSummary,
    grade_puzzle,
    grade_suite,
)

#: ``(name, fen, max_plies)`` for every committed position with a forced mate
#: under the solver's semantics: the 14 composed puzzles plus the 6 mined
#: mate-in-1s the defender cannot refute by capturing the attacker's marshal
#: (7 until KOM-261 quarantined the already-in-check row ``iter_827-g200002``).
#: Loaded from the fixture modules, never re-embedded here.
FORCED_MATE_FIXTURES: Final = tuple(
    [(f.name, f.fen, f.tsume_plies) for f in FRIEND]
    + [
        (m.name, m.fen, m.mate_in)
        for m in MINED
        if m.name not in MARSHAL_CAPTURE_ROWS
    ]
)

#: Positions where no forced mate exists for the side to move: the quiet
#: no-mate set, plus the two puzzles with their mating resource deleted.
NO_MATE_FIXTURES: Final = tuple(
    [(name, fen, NO_MATE_PLIES) for name, fen in NO_MATE_POSITIONS]
    + [(p.name, p.fen, p.tsume_plies) for p in PERTURBATIONS]
)

#: The committed fixture used for the slower-than-optimal case. It is a mined
#: mate-in-1 that *also* has a longer forced mate: ``新弓(7-2-1)`` drops an
#: archer with check without mating, and the attacker still mates on its next
#: ply, for three plies against the optimum's one. Found by scanning every
#: checking non-mating move of every committed mate-in-1 (two fixtures have
#: the shape; this is the cheaper one), so no synthetic position is needed.
#: Pinned by fixture *name* so the FEN stays in the committed JSON.
SLOW_FIXTURE_NAME: Final = "iter_827-g2000000100001"
SLOW_FIRST_MOVE: Final = "新弓(7-2-1)"
#: The cap must exceed the optimum, or the slower line cannot be proved at
#: all and would grade LOST_MATE. Three plies is the smallest odd cap above
#: a mate-in-1.
SLOW_MAX_PLIES: Final = 3


class SourceCalled(Exception):
    """Raised by :func:`forbidden_source` to prove it was never consulted."""


def oracle_source(max_plies: int) -> MoveSource:
    """A perfect attacker: the first move of the solver's own line."""

    def source(fen: str) -> str:
        result = solve_tsume(fen, max_plies)
        if result.verdict != "MATE":
            raise AssertionError(f"oracle asked to move in {result.verdict}: {fen!r}")
        return result.line[0]

    return source


def deepening_oracle_source(max_plies: int) -> MoveSource:
    """A perfect attacker that does not need to be told the mate distance.

    :func:`oracle_source` is fixed at one depth, which is fine when every
    position it will be asked about mates within that depth. Over a *mixed*
    set it is not: the composed puzzles alone range from mate-in-1 to
    mate-in-7, and a depth-5 oracle asked to move in a mate-in-7 sees
    NO_MATE. This searches 1, 3, 5, ... instead and answers from the first
    depth that proves a mate, so the move it returns is the first move of a
    *shortest* line -- the same "perfect attacker" claim, stated in a way
    that does not depend on the caller knowing the distance in advance.
    """

    def source(fen: str) -> str:
        for plies in range(1, max_plies + 1, 2):
            result = solve_tsume(fen, plies)
            if result.verdict == "MATE":
                return result.line[0]
        raise AssertionError(
            f"oracle asked to move with no mate in {max_plies}: {fen!r}"
        )

    return source


def quiet_source(fen: str) -> str:
    """The first legal move that gives no check -- a wasted tempo."""
    checks = frozenset(checking_moves(fen))
    for san in Gungi(fen).moves():
        text = str(san)
        if text not in checks:
            return text
    raise AssertionError(f"no quiet move in {fen!r}")


def illegal_source(fen: str) -> str:
    """A SAN no Gungi position can play, whatever the position is."""
    return "not-a-move"


def forbidden_source(fen: str) -> str:
    raise SourceCalled(fen)


def dawdling_source(first_move: str, max_plies: int) -> MoveSource:
    """Plays *first_move* once, then the solver's line: slow, but still won."""
    oracle = oracle_source(max_plies)
    calls = 0

    def source(fen: str) -> str:
        nonlocal calls
        calls += 1
        return first_move if calls == 1 else oracle(fen)

    return source


def slow_fixture_fen() -> str:
    return next(m.fen for m in MINED if m.name == SLOW_FIXTURE_NAME)


# ---------------------------------------------------------------------------
# A3: the oracle keeps every mate, at exactly the optimal distance
# ---------------------------------------------------------------------------


def test_forced_mate_fixture_set_is_complete() -> None:
    """Every forced-mate fixture, or the tests below prove less than claimed.

    21 positions until KOM-261 quarantined the already-in-check mined row;
    20 now (14 composed + 9 mined - 3 marshal-capture). Derived from the
    three sets rather than transcribed, so a fixture change moves the number
    here only if it also fails to reach the parametrized tests below.
    """
    expected = len(FRIEND) + len(MINED) - len(MARSHAL_CAPTURE_ROWS)
    assert len(FORCED_MATE_FIXTURES) == expected
    assert len({name for name, _, _ in FORCED_MATE_FIXTURES}) == expected


@pytest.mark.parametrize(
    ("name", "fen", "plies"), FORCED_MATE_FIXTURES, ids=lambda v: str(v)
)
def test_oracle_keeps_the_mate(name: str, fen: str, plies: int) -> None:
    grade = grade_puzzle(fen, plies, oracle_source(plies))
    assert grade.outcome is GradeOutcome.KEPT_MATE, f"{name}: {grade.outcome}"
    assert grade.optimal_distance == plies
    assert grade.played_distance == plies
    assert grade.fen == fen


@pytest.mark.parametrize(
    ("name", "fen", "plies"), FORCED_MATE_FIXTURES, ids=lambda v: str(v)
)
def test_oracle_line_is_the_solver_line(name: str, fen: str, plies: int) -> None:
    """Both sides of the played line come from the solver, so it must match.

    The defender's replies are the solver's longest defence with the solver's
    tie-break, so an oracle attacker reproduces the principal variation move
    for move. A drift here means the grader's defender is not the solver's.
    """
    grade = grade_puzzle(fen, plies, oracle_source(plies))
    assert grade.played_line == grade.solver_line, name
    assert len(grade.played_line) == plies


# ---------------------------------------------------------------------------
# A3: a legal non-mating move loses the win, and an illegal one is caught
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fen", "plies"), FORCED_MATE_FIXTURES, ids=lambda v: str(v)
)
def test_quiet_move_throws_the_win(name: str, fen: str, plies: int) -> None:
    grade = grade_puzzle(fen, plies, quiet_source)
    assert grade.outcome is GradeOutcome.LOST_MATE, f"{name}: {grade.outcome}"
    assert grade.optimal_distance == plies, "the position's mate is still recorded"
    assert grade.played_distance is None, "a lost line never reaches mate"
    assert grade.played_line[0] not in checking_moves(fen)


@pytest.mark.parametrize(
    ("name", "fen", "plies"), FORCED_MATE_FIXTURES, ids=lambda v: str(v)
)
def test_oracle_and_quiet_source_disagree(name: str, fen: str, plies: int) -> None:
    """The vacuity guard: a grader with one answer cannot pass this.

    Same position, same budget, two sources -- if the outcomes are equal the
    grader is reading the position and ignoring the player.
    """
    kept = grade_puzzle(fen, plies, oracle_source(plies))
    lost = grade_puzzle(fen, plies, quiet_source)
    assert kept.outcome is not lost.outcome, name
    assert kept.optimal_distance == lost.optimal_distance


@pytest.mark.parametrize(
    ("name", "fen", "plies"), FORCED_MATE_FIXTURES, ids=lambda v: str(v)
)
def test_illegal_san_is_caught(name: str, fen: str, plies: int) -> None:
    """The rules engine's ValueError is graded, not propagated."""
    grade = grade_puzzle(fen, plies, illegal_source)
    assert grade.outcome is GradeOutcome.ILLEGAL_MOVE, f"{name}: {grade.outcome}"
    assert grade.played_line == ("not-a-move",), "the rejected SAN is recorded"
    assert grade.played_distance is None
    assert grade.optimal_distance == plies


# ---------------------------------------------------------------------------
# A3: no mate in the position at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fen", "plies"), NO_MATE_FIXTURES, ids=lambda v: str(v))
def test_no_forced_mate_grades_the_position_for_every_source(
    name: str, fen: str, plies: int
) -> None:
    for label, source in (
        ("quiet", quiet_source),
        ("illegal", illegal_source),
    ):
        grade = grade_puzzle(fen, plies, source)
        assert grade.outcome is GradeOutcome.NO_MATE_IN_POSITION, f"{name}/{label}"
        assert grade.optimal_distance is None
        assert grade.played_distance is None
        assert grade.played_line == ()
        assert grade.solver_line == ()


@pytest.mark.parametrize(("name", "fen", "plies"), NO_MATE_FIXTURES, ids=lambda v: str(v))
def test_no_forced_mate_never_asks_the_source(
    name: str, fen: str, plies: int
) -> None:
    """Step 1 of the algorithm: an unsolvable position grades without a move.

    Pinned with a source that raises, so "never asked" is observed rather
    than assumed -- a grader that asked and then discarded the answer would
    be charging a net for positions it cannot be scored on.
    """
    grade = grade_puzzle(fen, plies, forbidden_source)
    assert grade.outcome is GradeOutcome.NO_MATE_IN_POSITION, name


# ---------------------------------------------------------------------------
# A4: slower than optimal is its own outcome, not a success
# ---------------------------------------------------------------------------


def test_slower_than_optimal_is_detected() -> None:
    fen = slow_fixture_fen()
    grade = grade_puzzle(fen, SLOW_MAX_PLIES, dawdling_source(SLOW_FIRST_MOVE, SLOW_MAX_PLIES))
    assert grade.outcome is GradeOutcome.SLOWER_THAN_OPTIMAL
    assert grade.optimal_distance == 1
    assert grade.played_distance is not None
    assert grade.played_distance > grade.optimal_distance
    assert grade.played_line[0] == SLOW_FIRST_MOVE
    assert len(grade.played_line) == grade.played_distance


def test_slower_line_still_ends_in_mate_for_the_rules_engine() -> None:
    """The cross-engine referee: the slow line is real, just slow.

    SLOWER_THAN_OPTIMAL would be worthless if it could also mean "did not
    mate at all", so the whole played line is replayed through ``api.Gungi``
    -- which raises on an illegal move -- and the end position must be mate
    to the pure-Python rules engine too.
    """
    fen = slow_fixture_fen()
    grade = grade_puzzle(fen, SLOW_MAX_PLIES, dawdling_source(SLOW_FIRST_MOVE, SLOW_MAX_PLIES))
    game = Gungi(fen)
    for san in grade.played_line:
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
    assert bool(game.is_checkmate())


def test_the_same_fixture_is_kept_mate_when_played_optimally() -> None:
    """The A4 discrimination: one position, two sources, two outcomes."""
    fen = slow_fixture_fen()
    optimal = grade_puzzle(fen, SLOW_MAX_PLIES, oracle_source(SLOW_MAX_PLIES))
    assert optimal.outcome is GradeOutcome.KEPT_MATE
    assert optimal.played_distance == optimal.optimal_distance == 1


#: ``(fen, move)`` pairs where the move leaves the defender with no legal reply
#: but NOT in check -- a stalemate, a draw (KOM-281, fuzz-found in KOM-262
#: review 2). The checks-mode terminal() does not see it.
STALEMATE_REPRODUCERS: Final = (
    ("6m1J/8G/5J3/6M2/4S4/9/9/9/9 -/- w 3 - 1", "大(2-1-1)(2-5-1)"),
    ("8m/9/7N1/8T/8I/2M6/9/9/9 -/- w 3 - 1", "謀(4-1-1)(5-1-2)付"),
)


@pytest.mark.parametrize(("fen", "move"), STALEMATE_REPRODUCERS)
def test_grader_does_not_call_a_stalemate_a_kept_mate(fen: str, move: str) -> None:
    grade = grade_puzzle(fen, 1, lambda _f: move)
    assert grade.outcome == GradeOutcome.LOST_MATE


@pytest.mark.parametrize(("fen", "move"), STALEMATE_REPRODUCERS)
def test_the_stalemate_reproducers_really_stalemate(fen: str, move: str) -> None:
    """The referee agrees the reproducers are stalemates, not mates."""
    game = Gungi(fen)
    game.move(move)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
    assert bool(game.is_stalemate()) is True
    assert bool(game.is_checkmate()) is False


# ---------------------------------------------------------------------------
# A5: determinism and the node budget
# ---------------------------------------------------------------------------


def test_grading_is_deterministic() -> None:
    """Two runs over the whole set, compared field by field.

    Frozen dataclasses compare by value, so this covers the lines and node
    counts as well as the outcomes -- a grader whose defender wandered with
    dict ordering would differ in ``played_line`` long before ``outcome``.
    """
    puzzles = tuple((fen, plies) for _, fen, plies in FORCED_MATE_FIXTURES)
    first_grades, first_summary = grade_suite(puzzles, quiet_source)
    second_grades, second_summary = grade_suite(puzzles, quiet_source)
    assert first_grades == second_grades
    assert first_summary == second_summary


@pytest.mark.parametrize("budget", [1, 10, 200, DEFAULT_NODE_BUDGET])
def test_reported_nodes_never_exceed_the_budget(budget: int) -> None:
    """Summed over every solve a grade runs, not just the first one."""
    puzzles = tuple((fen, plies) for _, fen, plies in FORCED_MATE_FIXTURES)
    grades, _ = grade_suite(puzzles, oracle_source(7), node_budget=budget)
    for grade in grades:
        assert grade.nodes <= budget, grade


def test_a_starved_budget_does_not_report_a_kept_mate() -> None:
    """One node proves nothing, and the grade says so rather than passing."""
    puzzles = tuple((fen, plies) for _, fen, plies in FORCED_MATE_FIXTURES)
    grades, summary = grade_suite(puzzles, oracle_source(7), node_budget=1)
    assert summary.kept_mate == 0
    assert all(g.outcome is not GradeOutcome.KEPT_MATE for g in grades)


# ---------------------------------------------------------------------------
# The suite layer
# ---------------------------------------------------------------------------


def test_suite_summary_counts_the_grades_it_was_built_from() -> None:
    """Every fixture lands in exactly one bucket, and the buckets sum.

    The attacker is the deepening oracle, not ``oracle_source(NO_MATE_PLIES)``
    as this test was first written. That premise was wrong: it assumed one
    depth serves the whole mixed set, and the set contains a mate-in-7
    (F15), so the fixed depth-5 oracle was handed a position it could not
    prove a mate in and asserted. Nothing is weakened by the change -- the
    oracle still plays a shortest mate everywhere, and ``kept_mate`` still
    has to be every forced-mate fixture -- it is just no longer told the
    distance up front. The no-mate half never consults the source at all
    (``grade_puzzle`` returns ``NO_MATE_IN_POSITION`` before asking), so it
    is still graded at ``NO_MATE_PLIES`` through its own pairs.
    """
    mixed = tuple(
        (fen, plies) for _, fen, plies in FORCED_MATE_FIXTURES + NO_MATE_FIXTURES
    )
    deepest = max(plies for _, _, plies in FORCED_MATE_FIXTURES)
    grades, summary = grade_suite(mixed, deepening_oracle_source(deepest))
    assert summary.total == len(mixed) == len(grades)
    assert summary == SuiteSummary.from_grades(grades)
    counted = (
        summary.kept_mate
        + summary.slower_than_optimal
        + summary.lost_mate
        + summary.illegal_move
        + summary.no_mate_in_position
    )
    assert counted == summary.total
    assert summary.kept_mate == len(FORCED_MATE_FIXTURES)
    assert summary.no_mate_in_position == len(NO_MATE_FIXTURES)


def test_suite_composes_with_both_fixture_dataclasses() -> None:
    """The documented reason ``grade_suite`` takes pairs and not a protocol.

    ``FriendFixture.tsume_plies`` and ``MinedPuzzle.mate_in`` are different
    field names for the same thing, so the pair is what both compose into.

    Both suites are sized from the fixture sets they were built from -- the
    mined side was 7 rows before KOM-261 quarantined the already-in-check
    row -- because what is under test is that *every* row composes and loses
    its mate to a quiet move, not that some particular number of them does.
    """
    mined_clean = tuple(m for m in MINED if m.name not in MARSHAL_CAPTURE_ROWS)
    friend, friend_summary = grade_suite(
        tuple((f.fen, f.tsume_plies) for f in FRIEND), quiet_source
    )
    mined, mined_summary = grade_suite(
        tuple((m.fen, m.mate_in) for m in mined_clean),
        quiet_source,
    )
    assert friend_summary.lost_mate == len(friend) == len(FRIEND)
    assert mined_summary.lost_mate == len(mined) == len(mined_clean)


def test_grade_is_frozen() -> None:
    """A grade is a record, not a scratchpad: nothing may edit it after the fact."""
    name, fen, plies = FORCED_MATE_FIXTURES[0]
    grade = grade_puzzle(fen, plies, oracle_source(plies))
    assert isinstance(grade, PuzzleGrade)
    with pytest.raises(AttributeError):
        grade.outcome = GradeOutcome.LOST_MATE  # type: ignore[misc]


@pytest.mark.parametrize("max_plies", [0, 2, -1])
def test_even_or_nonpositive_depth_is_rejected(max_plies: int) -> None:
    """The solver's contract, unchanged by the grading layer on top of it."""
    _, fen, _ = FORCED_MATE_FIXTURES[0]
    with pytest.raises(ValueError):
        grade_puzzle(fen, max_plies, oracle_source(1))
