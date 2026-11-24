"""Exact, solver-graded scoring of mate puzzles (KOM-259).

**Why this exists.** The KOM-257 mate suite grades mate-in-1 exactly, by
enumeration, and everything deeper *softly*: the net plays both sides and the
puzzle counts as solved if a mate lands within the recorded distance. That
tier is unfalsifiable in both directions. A net that mates in five where a
three-ply mate exists scores "solved", so the measurement cannot see a player
getting slower; and a net whose own defence is weak scores "solved" on a line
the real defender refutes, so it cannot see a player getting *wrong*. Both
failures move the number in the flattering direction, which is the worst
property a benchmark can have.

This module replaces that tier with the merged df-pn solver
(:mod:`komugi.search.tsume`) as ground truth. Nothing in here changes solver
behaviour: it calls :func:`~komugi.search.tsume.solve_tsume` and reads
:class:`~komugi.search.tsume.CyTsumeProblem`'s two pure query methods
(``terminal`` / ``children``), never the search itself.

**What is measured.** The move source -- ``Callable[[str], str]``, a FEN to a
SAN, so the served net, a random player and a fixed script all grade
identically -- plays the *attacker* only. Every defender reply comes from the
solver's **longest defence** at that node, computed exactly the way
:mod:`komugi.search.tsume_core` computes it at an AND node: the first reply
that refutes the attack if there is one, otherwise the reply that drags the
mate out furthest, ties to the earliest in generation order. So a grade is a
statement about the attacker and nothing else, and re-running it gives the
same answer.

**The five outcomes** (:class:`GradeOutcome`) are exhaustive and disjoint.
``KEPT_MATE`` is the only success, and it requires the played line to be
*exactly* as long as the solver's optimum -- ``SLOWER_THAN_OPTIMAL`` is
deliberately not merged into it, because "kept the win but took longer" is
the thing the old soft tier could not say.

**Two semantics worth knowing before reading a grade:**

*Mate is the solver's mate*, not ``api.Gungi.is_checkmate``: an AND node with
no reply at all, where the reply set includes captures of the attacker's
marshal (the KOM-258 amendment-A semantics). That is strictly stronger than
``is_checkmate``, which counts a position as mate even when the defender can
answer by taking the attacker's marshal and ending the game. Grading against
the weaker test would hand out ``KEPT_MATE`` for the three mined rows
``tsume_fixtures.MARSHAL_CAPTURE_ROWS`` pins as refutable. ``api.Gungi`` is
still used here, as the cross-engine referee that decides whether the move
source's SAN is *legal* and applies it -- the same referee
``tests/search/tsume/test_tsume_fixtures.py`` replays solver lines through.

*A budget that runs out reads as failure.* Every solve in one grade draws on
one shared ``node_budget``, so the ``nodes`` a grade reports never exceeds the
budget it was given. The price is that an ``UNKNOWN`` -- a statement about the
search, never about the position -- is graded ``NO_MATE_IN_POSITION`` at the
root and ``LOST_MATE`` inside a line. Size the budget so that cannot happen;
the default is the solver's own, three orders of magnitude above what the
committed fixtures cost.

**Puzzle shape for** :func:`grade_suite`. ``(fen, max_plies)`` pairs rather
than a protocol, because the two fixture dataclasses this grades name their
depth field differently -- ``FriendFixture.tsume_plies`` and
``MinedPuzzle.mate_in`` -- so no structural type covers both. The pair is
what they have in common, and both compose into it in one comprehension::

    grade_suite([(f.fen, f.tsume_plies) for f in FRIEND], source)
    grade_suite([(m.fen, m.mate_in) for m in MINED], source)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from komugi.engine import cyboard, cyengine
from komugi.engine.api import Gungi
from komugi.search.tsume import DEFAULT_NODE_BUDGET, CyTsumeProblem, solve_tsume
from komugi.search.tsume_core import TsumeResult

__all__ = [
    "DEFAULT_NODE_BUDGET",
    "GradeOutcome",
    "MoveSource",
    "PuzzleGrade",
    "SuiteSummary",
    "grade_puzzle",
    "grade_suite",
]

#: A player, as far as grading is concerned: FEN in, SAN out. Deliberately
#: the narrowest possible interface -- it is what lets the served net, a
#: random mover and a scripted list be graded by the same code.
MoveSource = Callable[[str], str]


class GradeOutcome(StrEnum):
    """How one puzzle went. Exhaustive, disjoint, and only one is a pass."""

    #: The solver finds no forced mate for the side to move within
    #: ``max_plies`` (or the budget ran out at the root). No move is asked of
    #: the source: there is nothing to grade.
    NO_MATE_IN_POSITION = "NO_MATE_IN_POSITION"
    #: The source mated against the longest defence, in exactly the solver's
    #: optimal distance. The only success.
    KEPT_MATE = "KEPT_MATE"
    #: The source threw the win: after one of its moves and the defender's
    #: longest reply, no forced mate remains -- or its move stalemated the
    #: defender, which is a draw.
    LOST_MATE = "LOST_MATE"
    #: The source returned a SAN ``api.Gungi`` will not play in the position.
    ILLEGAL_MOVE = "ILLEGAL_MOVE"
    #: The source mated, but took strictly more plies than the optimum.
    SLOWER_THAN_OPTIMAL = "SLOWER_THAN_OPTIMAL"


@dataclass(frozen=True)
class PuzzleGrade:
    """One graded puzzle: the verdict and everything needed to re-derive it."""

    #: The root position, as handed to :func:`grade_puzzle`.
    fen: str
    outcome: GradeOutcome
    #: The solver's exact mate distance in plies (odd, attacker plies and
    #: defender plies both counted), or ``None`` when there is no mate to
    #: grade against.
    optimal_distance: int | None
    #: Plies the source actually took to mate, in the same unit as
    #: :attr:`optimal_distance` so the two compare directly. ``None`` whenever
    #: the line did not reach mate.
    played_distance: int | None
    #: Every SAN played, attacker and defender alternating from the attacker.
    #: For ``ILLEGAL_MOVE`` the last entry is the *rejected* SAN, which was
    #: never applied -- kept so the failure is reproducible from the grade
    #: alone.
    played_line: tuple[str, ...]
    #: The solver's own optimal line from the root, for comparison. Empty when
    #: there is no mate.
    solver_line: tuple[str, ...]
    #: Node expansions summed over every solve this grade ran. Never above the
    #: ``node_budget`` it was given.
    nodes: int


@dataclass(frozen=True)
class SuiteSummary:
    """Per-outcome counts over a graded suite. ``total`` pins the sum."""

    kept_mate: int
    slower_than_optimal: int
    lost_mate: int
    illegal_move: int
    no_mate_in_position: int
    total: int

    @classmethod
    def from_grades(cls, grades: Sequence[PuzzleGrade]) -> SuiteSummary:
        counts = {outcome: 0 for outcome in GradeOutcome}
        for grade in grades:
            counts[grade.outcome] += 1
        return cls(
            kept_mate=counts[GradeOutcome.KEPT_MATE],
            slower_than_optimal=counts[GradeOutcome.SLOWER_THAN_OPTIMAL],
            lost_mate=counts[GradeOutcome.LOST_MATE],
            illegal_move=counts[GradeOutcome.ILLEGAL_MOVE],
            no_mate_in_position=counts[GradeOutcome.NO_MATE_IN_POSITION],
            total=len(grades),
        )


@dataclass(frozen=True)
class _Defence:
    """The defender's chosen reply. ``result is None`` means it refutes."""

    san: str
    result: TsumeResult | None


class _Ledger:
    """The one node budget every solve in a single grade draws on.

    Without it, ``nodes`` would be a sum over an unbounded number of solves
    and the budget would bound none of them. With it, the remaining budget is
    handed to each solve in turn, so the reported total is exactly the budget
    at worst -- and a starved solve answers ``UNKNOWN``, which the caller
    grades as failure rather than crashing.
    """

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self.spent = 0

    def solve(self, fen: str, max_plies: int) -> TsumeResult:
        result = solve_tsume(fen, max_plies, max(0, self._budget - self.spent))
        self.spent += result.nodes
        return result


def grade_puzzle(
    fen: str,
    max_plies: int,
    move_source: MoveSource,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> PuzzleGrade:
    """Grade *move_source* as the attacker in *fen* against the solver.

    The algorithm, which is the whole contract:

    1. Solve the root. Not ``MATE`` -> ``NO_MATE_IN_POSITION``, and the source
       is never called.
    2. Ask the source for a SAN. Not legal to ``api.Gungi`` -> ``ILLEGAL_MOVE``.
    3. Apply it. If it mates -- the solver's mate: the defender is in check
       and has no reply at all -- the line is over. No reply *without* check
       is a stalemate, a draw -> ``LOST_MATE``.
    4. Otherwise the defender plays the solver's longest defence.
    5. If no forced mate survives that reply, the source threw the win ->
       ``LOST_MATE``. Else back to 2.
    6. At mate, compare the played distance with the optimum: equal is
       ``KEPT_MATE``, longer is ``SLOWER_THAN_OPTIMAL``.

    *max_plies* is the depth cap for every solve, so it also caps how slow a
    line can be and still be *seen* as slow: a mate that needs more than
    ``max_plies`` plies cannot be proved and grades ``LOST_MATE``. To measure
    slack, pass a cap above the optimum. It must be odd and ``>= 1``
    (:func:`~komugi.search.tsume.solve_tsume` raises ``ValueError`` otherwise).
    """
    ledger = _Ledger(node_budget)
    root_result = ledger.solve(fen, max_plies)
    if root_result.verdict != "MATE":
        return PuzzleGrade(
            fen=fen,
            outcome=GradeOutcome.NO_MATE_IN_POSITION,
            optimal_distance=None,
            played_distance=None,
            played_line=(),
            solver_line=(),
            nodes=ledger.spent,
        )

    optimal = len(root_result.line)
    solver_line = root_result.line
    problem = CyTsumeProblem(cyboard.board_from_fen(fen))
    game = Gungi(fen)
    played: list[str] = []
    remaining = max_plies

    def grade(
        outcome: GradeOutcome, played_distance: int | None = None
    ) -> PuzzleGrade:
        return PuzzleGrade(
            fen=fen,
            outcome=outcome,
            optimal_distance=optimal,
            played_distance=played_distance,
            played_line=tuple(played),
            solver_line=solver_line,
            nodes=ledger.spent,
        )

    while True:
        san = move_source(game.fen())
        try:
            game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        except ValueError:
            played.append(san)
            return grade(GradeOutcome.ILLEGAL_MOVE)
        played.append(san)

        board = cyboard.board_from_fen(game.fen())
        if problem.terminal(board) is not None:
            # The game ended without a mate -- the source captured the
            # defender's marshal, or exposed its own. Under tsume semantics
            # that is a failed attack, not a win.
            return grade(GradeOutcome.LOST_MATE)

        replies = problem.children(board)
        if not replies and not cyengine.in_check(board, problem.defender):
            # No reply but no check either: a stalemate, which is a draw in
            # gungi (rules-audit Q9/U9). The checks-mode terminal() does not
            # detect it, so without this it would grade as a mate.
            return grade(GradeOutcome.LOST_MATE)
        if not replies:
            distance = len(played)
            if distance < optimal:
                raise RuntimeError(
                    f"mated in {distance} plies against the longest defence "
                    f"but the solver called {optimal} optimal for {fen!r}; "
                    "the grader and the solver disagree, which is a bug"
                )
            outcome = (
                GradeOutcome.KEPT_MATE
                if distance == optimal
                else GradeOutcome.SLOWER_THAN_OPTIMAL
            )
            return grade(outcome, played_distance=distance)

        next_remaining = remaining - 2
        defence = _longest_defence(problem, ledger, replies, next_remaining)
        played.append(defence.san)
        if defence.result is None:
            # A refuting reply ends the grade, so it is recorded but never
            # applied -- which also keeps a marshal-capturing reply, a move no
            # mate line ever contains, off api.Gungi's untravelled path.
            return grade(GradeOutcome.LOST_MATE)
        game.move(defence.san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        remaining = next_remaining


def _longest_defence(
    problem: CyTsumeProblem,
    ledger: _Ledger,
    replies: Sequence[tuple[str, cyboard.Board]],
    next_remaining: int,
) -> _Defence:
    """The defender's best reply, by :mod:`komugi.search.tsume_core`'s rule.

    An AND node's value is its *minimum* over replies, so one surviving reply
    is the whole defence: the first refutation short-circuits, exactly as
    ``_DfPn._best_line`` breaks on the first child with no mate. Absent one,
    the defender drags the mate out as far as it goes, ties to the earliest
    reply -- the same tie-break, so grading and the solver's own line agree
    move for move when the attacker plays optimally.
    """
    best: _Defence | None = None
    best_distance = -1
    for san, child in replies:
        if problem.terminal(child) is not None or next_remaining < 1:
            # Game over (the attacker's marshal is gone), or the attacker has
            # run out of plies. Either way the attack does not mate here.
            return _Defence(san, None)
        result = ledger.solve(cyboard.board_to_fen(child), next_remaining)
        if result.verdict != "MATE":
            return _Defence(san, None)
        if len(result.line) > best_distance:
            best_distance = len(result.line)
            best = _Defence(san, result)
    if best is None:  # pragma: no cover - replies is non-empty by construction
        raise RuntimeError("longest defence asked of an empty reply set")
    return best


def grade_suite(
    puzzles: Sequence[tuple[str, int]],
    move_source: MoveSource,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> tuple[tuple[PuzzleGrade, ...], SuiteSummary]:
    """Grade every ``(fen, max_plies)`` puzzle and summarise the outcomes.

    *node_budget* is per puzzle, not for the suite: a suite total would make
    one expensive puzzle silently starve the rest, and the grades would stop
    being independent of the order they were listed in.
    """
    grades = tuple(
        grade_puzzle(fen, max_plies, move_source, node_budget=node_budget)
        for fen, max_plies in puzzles
    )
    return grades, SuiteSummary.from_grades(grades)
