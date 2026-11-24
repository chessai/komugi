"""Mine solver-verified tsume puzzles from a sequence of played positions.

KOM-260. The miner this replaces (an earlier one-off mining script)
looked at exactly one position per game: it replayed a decisive game,
confirmed the final position was checkmate, and kept the position before the
mating move. That finds mate-in-1 and only mate-in-1, only at the end of a
game the net *won*, and it is exhausted -- all 30 rows it produced are 1手.

This module does what lichess does with real games instead: run the df-pn
solver (:mod:`komugi.search.tsume`) over **every** position, keep the ones
that hold a forced mate, and record whether the mover actually took it. Two
things follow from scanning every position rather than the last one:

*Depth.* A forced mate in 3 or 5 is a puzzle wherever it appears, and it
appears in the middle of games, including games that were later drawn or
lost. The old miner's window could not see one:
``tests/search/tsume/test_tsume_mining.py`` pins a position the solver grades
mate-in-3 and shows the same position reading NO_MATE at ``max_plies=1``,
which is the entire blind spot in one assertion.

*Missed mates.* A **missed mate** is a position where the side to move had a
forced mate within ``max_plies`` and the move it actually played did not keep
one (``played_kept_mate`` False). That record is worth twice what an ordinary
puzzle is worth: it is a puzzle *and* a labelled failure of the net's search,
on in-distribution data, with the refutation already proven. KOM-232's
external-anchor leg had 7 of 7 non-wins reach a won evaluation without converting, so the
pre-registered bet is that these exist at a measurable rate.

**Engine-pure, by construction.** Nothing here reads a shard, replays a game,
loads a checkpoint or touches a GPU or the filesystem. The caller supplies
``(fen, played_san)`` pairs and game replay stays in the harness, which is
what makes the miner unit-testable against the committed fixtures.

**What "kept the mate" means.** ``played_san`` kept the mate when it is a move
the engine generates for this position and the position *after* it still holds
a forced mate for the same side within the remaining ``max_plies - 1`` plies.
That position is an AND node -- the defender is to move -- so the question is
not one ``solve_tsume`` call but one per defender reply: every reply must leave
a mate within ``max_plies - 2``. The defender's reply set is taken from
:class:`~komugi.search.tsume.CyTsumeProblem` itself (escapes plus replies that
capture the attacker's marshal) so that "kept" is measured under exactly the
semantics the root verdict was proven under. A move that mates immediately
leaves the defender no reply at all and counts.

The conservative direction is chosen deliberately: when a sub-solve comes back
UNKNOWN the position is reported as **kept**, never as missed. An unproven
"the net threw away a mate here" is the one claim this module refuses to make,
so every member of :func:`mine_missed_mates` is a proven miss, at the price of
losing the occasional real one to a budget.

**Budgets.** ``node_budget`` bounds each individual solver call. The optional
``total_budget`` bounds the cumulative node count of the whole scan, and it is
honoured *before the fact*: a solve is started only when its full per-call
budget still fits, so the scan never exceeds the total and no verdict is ever
produced by a half-funded search. When a scan stops this way it returns what
it has found with ``budget_exhausted`` True rather than silently truncating.

:func:`komugi.search.tsume._marshal_missing` is imported rather than
reimplemented: it is private to that module, but the "is this game already
over" test has to be the same statement here as the one ``solve_tsume``
rejects roots with (KOM-258 F1), and two copies of it would be two chances to
disagree.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from komugi.engine import cyboard, cyengine, cymovegen
from komugi.engine.cyboard import Board
from komugi.search.tsume import CyTsumeProblem, _marshal_missing, solve_tsume
from komugi.search.tsume_core import TsumeResult

__all__ = [
    "MinedPuzzle",
    "ScanResult",
    "dedupe_by_position",
    "mine_missed_mates",
    "scan_positions",
]


@dataclass(frozen=True)
class MinedPuzzle:
    """One position the solver proved holds a forced mate.

    Only MATE positions become records, so the verdict is not a field: it is
    ``"MATE"`` for every instance. NO_MATE and UNKNOWN positions are counted
    into :attr:`ScanResult.scanned` and nothing more.
    """

    #: The position, as it was handed to the scan.
    fen: str
    #: Mate distance in plies -- ``len(line)``, and therefore odd.
    distance: int
    #: The solver's principal variation in SAN: the attacker's shortest mate
    #: against the defender's longest defence.
    line: tuple[str, ...]
    #: The move actually played from this position, or ``None`` when the
    #: caller did not supply one (the last position of a game, say).
    played_san: str | None
    #: Whether :attr:`played_san` still holds a forced mate within
    #: ``max_plies - 1``. ``False`` when no move was supplied, when the move
    #: is not one the engine generates here, and when the mate is provably
    #: gone. See the module docstring for the UNKNOWN rule.
    played_kept_mate: bool
    #: Position of this record in the input sequence, counting skipped
    #: positions, so a record can be traced back to the game it came from.
    index: int
    #: Solver node expansions spent on this record: the root solve plus the
    #: ``played_kept_mate`` re-solves.
    nodes: int


@dataclass(frozen=True)
class ScanResult:
    """Everything one :func:`scan_positions` call found, and what it cost."""

    #: The MATE positions, in input order.
    puzzles: tuple[MinedPuzzle, ...]
    #: Positions whose root solve completed. A position abandoned part-way
    #: through its ``played_kept_mate`` check is counted here and *not*
    #: recorded, since its record would have to guess the flag.
    scanned: int
    #: Positions skipped without solving: in the draft, or with the game
    #: already over.
    skipped: int
    #: True exactly when the scan stopped early because ``total_budget`` could
    #: not fund another solve. ``scanned + skipped`` is then below the length
    #: of the input.
    budget_exhausted: bool
    #: Cumulative solver node expansions, never above ``total_budget``.
    nodes: int


#: Plies consumed by the played move itself plus the defender's reply, i.e.
#: how much shallower the ``played_kept_mate`` re-solves are than the root.
_KEPT_MATE_PLY_COST: Final = 2


class _Ledger:
    """The node accountant for one scan.

    Every solver call in the scan goes through :meth:`solve`, which is what
    makes ``total_budget`` a guarantee rather than a hope: the check happens
    before the call, so a call that would not fit is not made at all.
    """

    def __init__(self, node_budget: int, total_budget: int | None) -> None:
        self._node_budget = node_budget
        self._total_budget = total_budget
        #: Nodes spent so far.
        self.nodes = 0
        #: Set the first time a solve was refused for want of budget.
        self.exhausted = False

    def affordable(self) -> bool:
        """Whether a full ``node_budget`` solve still fits in the total."""
        if self._total_budget is None:
            return True
        return self.nodes + self._node_budget <= self._total_budget

    def solve(self, fen: str, max_plies: int) -> TsumeResult | None:
        """Solve *fen*, or return ``None`` and latch :attr:`exhausted`."""
        if not self.affordable():
            self.exhausted = True
            return None
        result = solve_tsume(fen, max_plies, self._node_budget)
        self.nodes += result.nodes
        return result


def scan_positions(
    pairs: Iterable[tuple[str, str | None]],
    *,
    max_plies: int,
    node_budget: int,
    total_budget: int | None = None,
) -> ScanResult:
    """Solve every position in *pairs* and keep the forced mates.

    *pairs* is ``(fen, played_san)``, in played order; ``played_san`` may be
    ``None`` where no move follows. Positions in the draft -- which
    ``solve_tsume`` raises on since KOM-282, and where mate is impossible
    anyway -- and positions whose game is already over -- a marshal off the
    board, the roots ``solve_tsume`` raises on since KOM-258 F1 -- are always
    counted into :attr:`ScanResult.skipped` and never reach the solver, so
    neither raises.

    A malformed FEN is *not* swallowed: ``cyboard.board_from_fen`` raises and
    the exception propagates, because that is a bug in the caller's replay and
    not a position to skip.

    Raises ``ValueError`` for a *max_plies* that is not odd and >= 1 (the
    ``solve_tsume`` rule, checked here so a scan that skips every position
    still rejects it) or a non-positive budget.
    """
    if max_plies < 1 or max_plies % 2 == 0:
        raise ValueError(
            f"max_plies must be odd and >= 1 (a mate lands on an attacker "
            f"ply), got {max_plies}"
        )
    if node_budget < 1:
        raise ValueError(f"node_budget must be >= 1, got {node_budget}")
    if total_budget is not None and total_budget < 1:
        raise ValueError(f"total_budget must be >= 1 when given, got {total_budget}")

    ledger = _Ledger(node_budget, total_budget)
    puzzles: list[MinedPuzzle] = []
    scanned = 0
    skipped = 0

    for index, (fen, played_san) in enumerate(pairs):
        root = cyboard.board_from_fen(fen)
        if int(root.draft_w) or int(root.draft_b):
            skipped += 1
            continue
        if _marshal_missing(root):
            skipped += 1
            continue

        before = ledger.nodes
        result = ledger.solve(fen, max_plies)
        if result is None:
            break
        scanned += 1
        if result.verdict != "MATE":
            continue

        kept = _played_kept_mate(ledger, root, played_san, max_plies)
        if kept is None:
            # The kept-mate check ran out of budget part-way through. Both
            # flags would be a guess, so the position is left unrecorded.
            break
        puzzles.append(
            MinedPuzzle(
                fen=fen,
                distance=len(result.line),
                line=result.line,
                played_san=played_san,
                played_kept_mate=kept,
                index=index,
                nodes=ledger.nodes - before,
            )
        )

    return ScanResult(
        puzzles=tuple(puzzles),
        scanned=scanned,
        skipped=skipped,
        budget_exhausted=ledger.exhausted,
        nodes=ledger.nodes,
    )


def mine_missed_mates(
    result_or_pairs: ScanResult | Iterable[tuple[str, str | None]],
    *,
    max_plies: int | None = None,
    node_budget: int | None = None,
    total_budget: int | None = None,
) -> tuple[MinedPuzzle, ...]:
    """The records whose mover had a forced mate and did not keep it.

    Filtering a :class:`ScanResult` is the primary form, and the one the
    harness uses: a scan is the expensive part, its full record set is worth
    keeping (the kept-mate puzzles are still puzzles), and the missed-mate
    subset is a view over it. Handed an iterable of ``(fen, played_san)``
    pairs instead, this runs the scan itself -- a convenience for callers that
    want nothing but the misses -- and then *max_plies* and *node_budget* are
    required. Passing them alongside an already-computed ``ScanResult`` is a
    ``ValueError`` rather than a silently ignored argument, since they cannot
    change what that scan already measured.
    """
    if isinstance(result_or_pairs, ScanResult):
        if max_plies is not None or node_budget is not None or total_budget is not None:
            raise ValueError(
                "budget arguments do not apply to an already-computed "
                "ScanResult; pass pairs to run a fresh scan"
            )
        result = result_or_pairs
    else:
        if max_plies is None or node_budget is None:
            raise ValueError(
                "scanning pairs requires max_plies and node_budget; pass a "
                "ScanResult to filter one that has already been scanned"
            )
        result = scan_positions(
            result_or_pairs,
            max_plies=max_plies,
            node_budget=node_budget,
            total_budget=total_budget,
        )
    return tuple(record for record in result.puzzles if not record.played_kept_mate)


def dedupe_by_position(records: Iterable[MinedPuzzle]) -> tuple[MinedPuzzle, ...]:
    """Collapse records of the same position, keeping the shortest mate.

    The identity is :func:`cyengine.repetition_key`, not the FEN string: the
    same position recurs across games and arrives with a different move
    number, and the key is the engine's exact serialization of what makes a
    position a position (cells, side to move, draft flags, hands) with the
    move counter left out.

    Shortest wins because the distance is the puzzle's grade and the shorter
    proof is the true one -- a scan run at ``max_plies=5`` can report 5 for a
    position another scan graded 3. Ties keep the first record seen, and the
    survivors come back in order of first appearance.
    """
    best: dict[bytes, MinedPuzzle] = {}
    for record in records:
        key = cyengine.repetition_key(cyboard.board_from_fen(record.fen))
        current = best.get(key)
        if current is None or record.distance < current.distance:
            best[key] = record
    return tuple(best.values())


def _played_kept_mate(
    ledger: _Ledger,
    root: Board,
    played_san: str | None,
    max_plies: int,
) -> bool | None:
    """Whether *played_san* keeps a forced mate; ``None`` if the budget ran out.

    Called only for a root the solver has already proven MATE, so the
    question is never "is there a mate" but "does this move still have one".
    """
    if played_san is None:
        return False
    move = _find_move(root, played_san)
    if move is None:
        # Not a move the engine generates here: it cannot have kept anything.
        return False

    after = root.copy()
    cyengine.apply_move(after, move)
    problem = CyTsumeProblem(root)
    if problem.terminal(after) is not None:
        # The played move ended the game by taking a marshal. Winning that
        # way is not delivering the mate the root held.
        return False

    # ``after`` has the defender to move, so ``problem.children`` dispatches
    # to the defender's reply set -- escapes plus marshal captures -- which is
    # what the root proof was measured against.
    replies = problem.children(after)
    if not replies:
        # No reply at all is mate, but only with the defender actually in
        # check: an AND node inside the search is reached by a checking move,
        # and a played move need not be one.
        return bool(cyengine.in_check(after, problem.defender))

    remaining = max_plies - _KEPT_MATE_PLY_COST
    if remaining < 1:
        # The move did not mate and there is no ply left to mate in.
        return False

    for _, child in replies:
        if problem.terminal(child) is not None:
            # The defender ends the game by capturing the attacker's marshal.
            return False
        sub = ledger.solve(cyboard.board_to_fen(child), remaining)
        if sub is None:
            return None
        if sub.verdict == "UNKNOWN":
            # Conservative: an unproven loss of the mate is not reported as a
            # miss. See the module docstring.
            return True
        if sub.verdict != "MATE":
            return False
    return True


def _find_move(board: Board, san: str) -> dict[str, Any] | None:
    """The generated move of *board* whose SAN is *san*, or ``None``.

    "Legal" here is the engine's own move list, which is pseudo-legal -- the
    same list :class:`~komugi.search.tsume.CyTsumeProblem` searches -- so that
    a move the solver would consider is never called illegal by this module.
    """
    for move in cymovegen.generate_all_moves(board):
        if str(move["san"]) == san:
            return move
    return None
