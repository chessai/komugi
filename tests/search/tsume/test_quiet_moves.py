"""KOM-262: the general forced-mate mode, ``attacker_moves="all"``.

The solver's default asks the tsume question -- is there a mate in which
*every* attacker move gives check. That is the right question for a composed
問題 and the wrong one for "did the net miss a forced win", because a quiet
move that takes away the last flight square mates just as forcibly and the
checks-only filter cannot see it. ``attacker_moves="all"`` asks the general
question instead.

The mode buys recall, and the only way it can be worse than the recall it
buys is by proving a mate that is not there. There is exactly one new way for
that to happen: a checks-only attacker guarantees the defender is in check at
every AND node, so "no reply" means mate; a quiet attacker move breaks that
guarantee, and "no reply with the defender not in check" is a *stalemate*.
So the tests below are weighted towards soundness rather than coverage:

* :func:`test_quiet_first_move_mate_is_found_only_under_all` and its vacuity
  guard -- the mode does something (A3).
* :func:`test_all_never_contradicts_checks_on_the_fixture_set` -- and it does
  not change a settled answer (A4).
* :func:`test_shared_table_is_not_fooled_by_the_other_mode` and its
  mode-blind twin -- the transposition table is partitioned by mode, pinned
  against a deliberately unpartitioned problem that gets the wrong answer
  (A5).
* :func:`test_starved_full_width_search_says_unknown` -- a spent budget is
  still UNKNOWN and never a verdict (A6).
* :func:`test_and_node_reply_set_widens_when_the_defender_is_not_in_check`
  and :func:`test_stalemate_is_not_scored_as_a_mate` -- the dangerous part,
  from both sides (A7).

**The two synthetic positions.** No committed fixture has either shape --
the friend set is composed tsume, where convention forbids a quiet move, and
the mined set is mate-in-1, where a quiet move cannot mate by definition --
so both are constructed here and spelled out in the constants below. Each was
found by generating sparse random boards (a black marshal in the corner
region, a white marshal parked at 9-9 so ``cyengine.is_checkmate`` is not
constant-False, two or three white pieces) and filtering for the wanted
shape; each is then verified move by move in the test rather than trusted,
and the mate line is replayed through ``api.Gungi`` -- the same second
referee the rest of the suite uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from tsume_fixtures import FRIEND, MINED, NO_MATE_POSITIONS, MinedPuzzle

from komugi.engine import cyboard, cyengine, cymovegen
from komugi.engine.api import Gungi
from komugi.engine.cyboard import Board
from komugi.search.tsume import (
    DEFAULT_NODE_BUDGET,
    CyTsumeProblem,
    _marshal_missing,
    solve_tsume,
)
from komugi.search.tsume_core import Outcome, TsumeResult, TsumeTable, solve_problem

# ---------------------------------------------------------------------------
# A3 / A7(i): the quiet-first-move mate
# ---------------------------------------------------------------------------

#: A forced mate in three whose first move gives no check.
#:
#: Five pieces, no hands, no drafting rights::
#:
#:     black:  marshal 1-3, soldier 5-4
#:     white:  marshal 9-9 (parked, only there to keep the board legal),
#:             general 5-6, archer 5-2
#:
#: The black marshal starts with five steps -- 1-2, 1-4, 2-2, 2-3, 2-4 -- and
#: no white check mates it, at depth 3 or at depth 5. The key is the quiet
#: ``大(5-6-1)(2-6-1)``: the general steps onto the second rank and takes 2-2,
#: 2-3 and 2-4 away at once without ever attacking the marshal. That is the
#: classic shape the mode exists for -- the move that removes the flight
#: squares is not itself a check. Black is left with two marshal steps and
#: two soldier pushes, and the archer mates against each of them (from 3-2
#: after 1-2, from 3-3 after any of the other three).
#:
#: The soldier is there for the reply-set test, not for the mate: it gives
#: black two replies that do not touch the marshal, so the reply set after
#: the quiet move is visibly *wider* than any evasion set. It does not
#: prevent a stalemate -- without it black still has its two marshal steps,
#: and ``"all"`` still mates in three (see :data:`SOLDIERLESS_QUIET_MATE_FEN`).
QUIET_MATE_FEN: Final = "6m2/9/9/9/3G1d1A1/9/9/9/M8 -/- w 3 - 1"

#: :data:`QUIET_MATE_FEN` with the black soldier removed. Pinned because an
#: earlier note claimed the soldier was what kept the quiet move from being a
#: stalemate; it is not, and this position is still a mate in three.
SOLDIERLESS_QUIET_MATE_FEN: Final = "6m2/9/9/9/3G3A1/9/9/9/M8 -/- w 3 - 1"

#: The quiet key move, and the whole principal variation against the
#: defender's longest defence.
QUIET_KEY_MOVE: Final = "大(5-6-1)(2-6-1)"
QUIET_MATE_LINE: Final = (
    QUIET_KEY_MOVE,
    "帥(1-3-1)(1-2-1)",
    "弓(5-2-1)(3-2-1)",
)

#: Black's four replies to the key move, in the order the binding generates
#: them. Pinned because A7(i) is about this list being *wider* than an
#: evasion set: black is not in check here, so none of these is an evasion,
#: and two of them do not touch the marshal at all.
QUIET_MATE_REPLIES: Final = (
    "帥(1-3-1)(1-2-1)",
    "帥(1-3-1)(1-4-1)",
    "兵(5-4-1)(6-4-1)",
    "兵(5-4-1)(4-4-1)",
)

#: Depth the quiet-mate position is searched to. The mate is in three.
QUIET_MATE_PLIES: Final = 3

# ---------------------------------------------------------------------------
# A7(ii): the stalemate
# ---------------------------------------------------------------------------

#: A position where the attacker has a move that *stalemates* the defender.
#:
#: Four pieces, no hands, no drafting rights::
#:
#:     black:  marshal 1-1
#:     white:  marshal 9-9, general 4-2, musketeer 5-4
#:
#: ``大(4-2-1)(3-2-1)`` leaves the black marshal with three pseudo-legal
#: steps, every one of them into check, and black has nothing else on the
#: board and nothing in hand. So the defender has no reply and is **not** in
#: check: ``api.Gungi.is_stalemate()`` agrees, and ``is_checkmate()`` does
#: not.
#:
#: This is the one position in the suite that can make the new mode *unsound*
#: rather than merely slow. The search's rule is "an AND node with no
#: children is checkmate"; here that rule is wrong, and the only thing
#: standing between it and a fabricated mate-in-1 is
#: :meth:`CyTsumeProblem.terminal` calling the node DISPROVED first. Delete
#: that branch and :func:`test_stalemate_is_not_scored_as_a_mate` goes red.
STALEMATE_FEN: Final = "8m/9/9/7G1/5K3/9/9/9/M8 -/- w 3 - 1"

#: The attacker move that produces the stalemate.
STALEMATE_MOVE: Final = "大(4-2-1)(3-2-1)"

# ---------------------------------------------------------------------------
# A4: the fixture sweep
# ---------------------------------------------------------------------------

#: Budget for the full-width half of the fixture sweep. Full width is one to
#: three orders of magnitude more expensive than checks-only (F09 corrected:
#: 8 node expansions against 375; F11: 98 against 18,776), so the sweep is
#: capped rather than run to exhaustion.
FULL_WIDTH_BUDGET: Final = 30_000

#: Per-row budgets for the deep friend rows whose full-width search does not
#: fit in :data:`FULL_WIDTH_BUDGET`. Measured need: F13 90,909 expansions,
#: F14 34,405. These are exactly the rows where "all" has the most room to
#: disagree with checks-only, so they are run to a verdict rather than let
#: off as UNKNOWN.
ROW_BUDGETS: Final[dict[str, int]] = {"F13-5move": 150_000, "F14-5move": 60_000}

#: Friend rows whose full-width search is still not run to a verdict in the
#: sweep. F15 at its 7-ply grade needs 844,542 expansions (four and a half
#: minutes), so it comes back UNKNOWN there -- a statement about the budget,
#: not a disagreement -- and its depth is covered instead by
#: :func:`test_f15_tail_is_mated_under_all`, which starts two plies in. The
#: set is pinned so that a *new* row falling off the end is a failure rather
#: than a silent loss of coverage.
KNOWN_UNRESOLVED: Final = frozenset({"F15-5move"})

#: Measured node expansions on the corrected F09, and the numbers the
#: :func:`solve_tsume` docstring states: checks-only at the fixture's grade,
#: then full width at depth 3 and at depth 5.
F09_NAME: Final = "F09-3move-corrected"
F09_CHECKS_NODES: Final = 8
F09_ALL_3_NODES: Final = 375
F09_ALL_5_NODES: Final = 9_447

#: The base-commit golden for the default mode: every row as
#: ``[name, plies, str(result) or str(ValueError)]``, captured at 02a6a03
#: (before ``attacker_moves`` existed) by review 2's ``identity.py``.
CHECKS_GOLDEN_PATH: Final = (
    Path(__file__).parent / "data" / "checks_golden_02a6a03.json"
)
CHECKS_GOLDEN_ROWS: Final = 63
CHECKS_GOLDEN_BUDGET: Final = 200_000


def _legal_replies(board: Board) -> list[str]:
    """Every move of the side to move that does not leave it in check.

    Computed here from ``cymovegen`` rather than asked of the solver, so that
    "the binding returns the legal replies" is a comparison between two
    implementations and not a tautology.
    """
    mover = int(board.turn)
    out: list[str] = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        if not cyengine.in_check(child, mover):
            out.append(str(move["san"]))
    return out


def _child_after(problem: CyTsumeProblem, node: Board, san: str) -> Board:
    """The child of *node* reached by *san*, or a failure naming what was there."""
    generated = problem.children(node)
    for label, child in generated:
        if label == san:
            return child
    raise AssertionError(f"{san!r} not generated; got {[s for s, _ in generated]}")


def _is_mate(problem: CyTsumeProblem, node: Board) -> bool:
    """Whether *node* is checkmate, by the engine's definition of the word.

    Not "the binding offers no children": under ``"all"`` that is also true
    of a stalemate, which is exactly the conflation the stalemate rule
    exists to prevent. ``cyengine.is_checkmate`` decides; the binding is
    cross-checked against it rather than trusted.
    """
    mate = bool(cyengine.is_checkmate(node))
    if mate:
        assert not problem.children(node), "a checkmated defender was given replies"
    return mate


def _replay(fen: str, line: tuple[str, ...]) -> Gungi:
    """Play *line* through the second engine, which raises on an illegal move."""
    game = Gungi(fen)
    for san in line:
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
    return game


# ---------------------------------------------------------------------------
# A3 -- the mode finds what checks-only cannot
# ---------------------------------------------------------------------------


def test_quiet_first_move_mate_is_found_only_under_all() -> None:
    """The constructed position: NO_MATE under ``"checks"``, MATE under ``"all"``.

    Both halves matter. The first is what makes the position a *quiet*-move
    mate rather than an ordinary one -- and it is asserted at depth 5 as well
    as at depth 3, so "no all-checks mate here" is not an artefact of the cap.
    The second is the mate itself, with the line pinned move for move; a mode
    that found *some* mate by a different route would still have to find one
    that opens with a quiet move, which the check assertion below enforces.
    """
    assert solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES).verdict == "NO_MATE"
    assert solve_tsume(QUIET_MATE_FEN, 5).verdict == "NO_MATE"

    result = solve_tsume(
        QUIET_MATE_FEN,
        QUIET_MATE_PLIES,
        attacker_moves="all",
        verify_checkmate=True,
    )
    assert result.verdict == "MATE", result
    assert result.line == QUIET_MATE_LINE, result.line
    assert len(result.line) == QUIET_MATE_PLIES


def test_the_key_move_really_gives_no_check() -> None:
    """The point of the fixture, stated separately so it cannot rot.

    If ``大(5-6-1)(2-6-1)`` ever became a checking move -- a movegen change, a
    transcription slip in the FEN -- the test above would still pass while
    testing nothing, because checks-only would then find the same mate.
    """
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    defender = 1 - int(board.turn)
    problem = CyTsumeProblem(board, attacker_moves="all")
    child = _child_after(problem, board, QUIET_KEY_MOVE)
    assert not cyengine.in_check(child, defender), (
        "the key move gives check, so the position is no longer a quiet-move mate"
    )
    assert QUIET_KEY_MOVE not in [
        san for san, _ in CyTsumeProblem(board).children(board)
    ], "the checks-only binding generated the quiet key move"


def test_the_two_modes_differ_somewhere() -> None:
    """The vacuity guard: a selector that silently did nothing must fail.

    Everything else in this file either compares the modes on positions where
    they are *supposed* to agree (A4) or inspects one mode's internals. This
    is the one assertion that a no-op ``attacker_moves`` cannot satisfy.
    """
    checks = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="checks")
    full = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all")
    assert checks.verdict != full.verdict, (
        f"both modes answered {checks.verdict} -- the selector is doing nothing"
    )


def test_the_mate_line_replays_through_the_api() -> None:
    """The cross-engine referee, as the rest of the suite uses it.

    ``Gungi.move`` raises on an illegal move, so this fails loudly if the
    solver's SAN or its move semantics have drifted; the per-ply assertions
    say which ply of the line is quiet and which delivers the mate.
    """
    result = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all")
    assert result.verdict == "MATE"

    game = Gungi(QUIET_MATE_FEN)
    attacker = str(game.turn())
    defender = "b" if attacker == "w" else "w"
    for ply, san in enumerate(result.line):
        assert str(game.turn()) == (attacker if ply % 2 == 0 else defender)
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        if ply == 0:
            assert not bool(game.in_check(defender)), "ply 1 was supposed to be quiet"
        elif ply % 2 == 0:
            assert bool(game.in_check(defender))
    assert bool(game.is_checkmate()), "the quiet-move line does not end in mate"


def test_every_defence_to_the_key_move_is_mated() -> None:
    """MATE means *every* reply loses, not just the one in the principal variation.

    The PV shows the defender's longest defence and nothing else, so on a
    three-ply mate it names one of four replies. This walks all four through
    ``api.Gungi`` and finds the mating move for each, which is the content of
    the word "forced".
    """
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    problem = CyTsumeProblem(board, attacker_moves="all")
    after_key = _child_after(problem, board, QUIET_KEY_MOVE)

    replies = [san for san, _ in problem.children(after_key)]
    assert tuple(replies) == QUIET_MATE_REPLIES, replies

    for reply in replies:
        mating = [
            san
            for san, grandchild in problem.children(
                _child_after(problem, after_key, reply)
            )
            if _is_mate(problem, grandchild)
        ]
        assert len(mating) == 1, f"{reply}: mating moves {mating}"
        game = _replay(QUIET_MATE_FEN, (QUIET_KEY_MOVE, reply, mating[0]))
        assert bool(game.is_checkmate()), f"{reply}: {mating[0]} is not mate"


def test_the_mate_predicate_rejects_a_stalemate() -> None:
    """The predicate above is False where "no children" would say mate.

    The node after :data:`STALEMATE_MOVE` has no reply and no check. A
    predicate that read "no children" as mate would call it one, and the
    defence test above would then be counting stalemating moves as mating
    ones.
    """
    board = cyboard.board_from_fen(STALEMATE_FEN)
    problem = CyTsumeProblem(board, attacker_moves="all")
    node = _child_after(problem, board, STALEMATE_MOVE)
    assert not problem.children(node), "the stalemate node has replies"
    assert not _is_mate(problem, node), "a stalemate was read as a mate"


def test_the_soldierless_quiet_mate_is_still_mate_in_three() -> None:
    """The soldier in :data:`QUIET_MATE_FEN` is not what averts a stalemate.

    Without it black still has two marshal steps after the quiet key move,
    so the attack is a mate, not a stalemate, and the full-width search
    finds it at the same distance.
    """
    result = solve_tsume(
        SOLDIERLESS_QUIET_MATE_FEN,
        QUIET_MATE_PLIES,
        attacker_moves="all",
        verify_checkmate=True,
    )
    assert result.verdict == "MATE", result
    assert len(result.line) == QUIET_MATE_PLIES, result.line


# ---------------------------------------------------------------------------
# A4 -- a superset, not a different answer
# ---------------------------------------------------------------------------

#: ``(name, fen, plies)`` for every committed row the sweep runs on: the
#: friend puzzles at their recorded grade, the mined mate-in-1s at depth 1.
#: Counts come from the loaders, so adding a row to either fixture file
#: extends the sweep rather than silently leaving the new row out.
SWEEP: Final[tuple[tuple[str, str, int], ...]] = tuple(
    [(f.name, f.fen, f.tsume_plies) for f in FRIEND]
    + [(m.name, m.fen, 1) for m in MINED]
)

SWEEP_PARAMS: Final = [pytest.param(*row, id=row[0]) for row in SWEEP]


def test_the_sweep_covers_every_friend_and_mined_row() -> None:
    """The sweep's parametrisation is the fixture set, not a copy of it."""
    assert len(SWEEP) == len(FRIEND) + len(MINED)
    assert {name for name, _, _ in SWEEP} == {f.name for f in FRIEND} | {
        m.name for m in MINED
    }
    assert KNOWN_UNRESOLVED <= {f.name for f in FRIEND}
    assert set(ROW_BUDGETS) <= {f.name for f in FRIEND}
    assert not set(ROW_BUDGETS) & KNOWN_UNRESOLVED


@pytest.mark.parametrize("name", sorted(ROW_BUDGETS))
def test_deep_friend_rows_are_mated_under_all(name: str) -> None:
    """F13 and F14 run to a verdict at full width, not let off as UNKNOWN.

    The sweep already asserts this through :data:`ROW_BUDGETS`; stated here
    on its own so that growing :data:`KNOWN_UNRESOLVED` back over these rows
    cannot quietly drop the deepest coverage the sweep has.
    """
    fixture = next(f for f in FRIEND if f.name == name)
    checks = solve_tsume(fixture.fen, fixture.tsume_plies)
    full = solve_tsume(
        fixture.fen, fixture.tsume_plies, ROW_BUDGETS[name], attacker_moves="all"
    )
    assert checks.verdict == "MATE", checks
    assert full.verdict == "MATE", full
    assert len(full.line) <= len(checks.line), (full.line, checks.line)


def test_f15_tail_is_mated_under_all() -> None:
    """F15's depth, reached two plies in rather than paid for at the root.

    F15 is a 7-ply puzzle and full width at the root needs some 845,000
    expansions. Two plies down its checks-only principal variation the
    remaining attack is a 5-ply one, which full width proves in about
    17,000 -- so the deepest friend row still gets a full-width verdict,
    and a mate no longer than the checks-only remainder.
    """
    fixture = next(f for f in FRIEND if f.name == "F15-5move")
    checks = solve_tsume(fixture.fen, fixture.tsume_plies)
    assert checks.verdict == "MATE", checks
    assert len(checks.line) == 7, checks.line

    fen = str(_replay(fixture.fen, checks.line[:2]).fen())
    tail = solve_tsume(fen, 5, 60_000, attacker_moves="all")
    assert tail.verdict == "MATE", tail
    assert len(tail.line) <= 5, tail.line


@pytest.mark.parametrize("name,fen,plies", SWEEP_PARAMS)
def test_all_never_contradicts_checks_on_the_fixture_set(
    name: str, fen: str, plies: int
) -> None:
    """``"all"`` is a superset of ``"checks"``, so it must not change these answers.

    The two groups are here for different reasons, and both are airtight
    rather than empirical:

    *Friend, at the recorded grade.* Checks-only says MATE, and every
    checks-only mate is a mate under the wider move set too -- the attacker
    is being given more moves, not fewer -- so ``"all"`` must also say MATE,
    at a distance no longer than the checks-only one.

    *Mined, at depth 1.* A mate in one ply is delivered by a move that gives
    check, so the two modes offer the same mating candidates and must agree
    exactly, NO_MATE rows included.

    UNKNOWN is admitted in one direction only: it is a statement about the
    budget, never a disagreement, and the rows it happens on are pinned in
    :data:`KNOWN_UNRESOLVED`. The deep rows that merely need more than the
    shared budget get their own, from :data:`ROW_BUDGETS`.
    """
    budget = ROW_BUDGETS.get(name, FULL_WIDTH_BUDGET)
    checks = solve_tsume(fen, plies, budget)
    full = solve_tsume(fen, plies, budget, attacker_moves="all")

    if full.verdict == "UNKNOWN":
        assert name in KNOWN_UNRESOLVED, (
            f"{name}: full width ran out of budget on a row that used to fit"
        )
        assert full.nodes <= budget
        return

    assert full.verdict == checks.verdict, f"{name}: {checks.verdict} -> {full.verdict}"
    if checks.verdict == "MATE":
        assert len(full.line) <= len(checks.line), (
            f"{name}: full width found a longer mate ({len(full.line)} plies) "
            f"than checks-only ({len(checks.line)})"
        )


def test_the_sweep_resolves_almost_every_row() -> None:
    """Coverage floor: the sweep must not decay into a parade of UNKNOWNs.

    The assertion above lets UNKNOWN through, so on its own it would pass on
    a solver that answered UNKNOWN everywhere. This re-runs the mined rows
    -- cheap, and the half where ``"all"`` and ``"checks"`` are provably the
    same question -- and requires every one of them to come back with a real
    verdict.
    """
    resolved = [
        m.name
        for m in MINED
        if solve_tsume(
            m.fen, 1, FULL_WIDTH_BUDGET, attacker_moves="all"
        ).verdict
        != "UNKNOWN"
    ]
    assert len(resolved) == len(MINED), resolved
    assert len(MINED) > 0


def test_the_documented_f09_node_counts() -> None:
    """The :func:`solve_tsume` docstring's measurements, pinned to the search.

    The docstring once quoted pre-correction numbers that nothing checked.
    These are the numbers it states now; if either the search or the
    fixture moves, this goes red and the docstring has to move with it.
    """
    fixture = next(f for f in FRIEND if f.name == F09_NAME)
    assert solve_tsume(fixture.fen, fixture.tsume_plies).nodes == F09_CHECKS_NODES
    assert solve_tsume(fixture.fen, 3, attacker_moves="all").nodes == F09_ALL_3_NODES
    assert solve_tsume(fixture.fen, 5, attacker_moves="all").nodes == F09_ALL_5_NODES


def test_full_width_f15_exceeds_the_default_budget() -> None:
    """The ``DEFAULT_NODE_BUDGET`` comment's warning, as a test.

    Under ``"all"`` F15 at its 7-ply grade needs far more than the default
    budget, so a caller who does not raise it gets UNKNOWN -- honestly, and
    within the budget -- rather than a verdict.
    """
    fixture = next(f for f in FRIEND if f.name == "F15-5move")
    result = solve_tsume(fixture.fen, 7, attacker_moves="all")
    assert result.verdict == "UNKNOWN", result
    assert result.nodes <= DEFAULT_NODE_BUDGET


# ---------------------------------------------------------------------------
# A5 -- the transposition table is mode-safe
# ---------------------------------------------------------------------------


def test_shared_table_is_not_fooled_by_the_other_mode() -> None:
    """Checks-only first, then full width, on one table: the answer is full width's.

    The hazard is exactly the one the remaining-depth field in the key was
    added for. A checks-only search writes "no mate from here within three
    plies" against this position; a full-width search asks a strictly weaker
    question of the same position, and if it reads that entry back it
    inherits a NO_MATE that was only ever true of the narrower move set. The
    mode is folded into :meth:`CyTsumeProblem.digest`, so the two never meet.
    """
    table = TsumeTable()

    checks = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, table=table)
    assert checks.verdict == "NO_MATE"
    assert table.entries, "the first solve wrote nothing to the shared table"

    full = solve_tsume(
        QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all", table=table
    )
    assert full.verdict == "MATE", full
    assert full.line == QUIET_MATE_LINE


def test_the_shared_table_is_really_being_read() -> None:
    """...because "not fooled" would also be satisfied by a table nobody uses.

    A repeat solve resolves at the root out of the table, so it expands no
    nodes at all. If that number is not zero the table is not being shared,
    and the test above proves nothing about sharing.
    """
    table = TsumeTable()
    first = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, table=table)
    again = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, table=table)
    assert first.nodes > 0
    assert again.verdict == first.verdict
    assert again.nodes == 0, again.nodes

    full_first = solve_tsume(
        QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all", table=table
    )
    full_again = solve_tsume(
        QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all", table=table
    )
    assert full_first.nodes > 0
    assert full_again.nodes == 0, full_again.nodes
    assert full_again.line == full_first.line


class _ModeBlindProblem(CyTsumeProblem):
    """The binding with the mode taken back out of the transposition key.

    Not a mistake anyone would write on purpose -- it is what
    :meth:`CyTsumeProblem.digest` looked like before KOM-262 -- and it exists
    so the mode tag can be pinned by the failure it prevents rather than by
    an assertion about its own implementation.
    """

    def digest(self, node: Board) -> bytes:
        return cyengine.repetition_key(node)


def test_a_mode_blind_digest_returns_the_stale_answer() -> None:
    """The bug the tag prevents, reproduced: same table, and the mode is lost.

    With the tag gone, the checks-only entry for the root satisfies the
    full-width query outright -- nought node expansions, and a NO_MATE for a
    position this file has already shown to be mate in three. That is the
    whole argument for tagging the digest, and it is red here on purpose.
    """
    root = cyboard.board_from_fen(QUIET_MATE_FEN)
    table = TsumeTable()

    checks: TsumeResult = solve_problem(
        _ModeBlindProblem(root), root, QUIET_MATE_PLIES, 100_000, table=table
    )
    assert checks.verdict == "NO_MATE"

    full: TsumeResult = solve_problem(
        _ModeBlindProblem(root, attacker_moves="all"),
        root,
        QUIET_MATE_PLIES,
        100_000,
        table=table,
    )
    assert full.verdict == "NO_MATE", (
        "the mode-blind problem got the right answer, so this test no longer "
        "reproduces the bug the mode tag exists to prevent"
    )
    assert full.nodes == 0

    # And the same pair with the tag in place gets it right.
    tagged = TsumeTable()
    assert (
        solve_problem(
            CyTsumeProblem(root), root, QUIET_MATE_PLIES, 100_000, table=tagged
        ).verdict
        == "NO_MATE"
    )
    assert (
        solve_problem(
            CyTsumeProblem(root, attacker_moves="all"),
            root,
            QUIET_MATE_PLIES,
            100_000,
            table=tagged,
        ).verdict
        == "MATE"
    )


# ---------------------------------------------------------------------------
# A6 -- budget honesty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 2, 5, 17, 100])
def test_starved_full_width_search_says_unknown(budget: int) -> None:
    """A spent budget is UNKNOWN, never a verdict, and never over-spent.

    The quiet-mate position needs 410 expansions at full width, so every
    budget here is a starvation. The wrong answers this rules out are both
    directions at once: NO_MATE (the position *is* mate in three) and MATE
    (the search never got far enough to know).
    """
    result = solve_tsume(
        QUIET_MATE_FEN, QUIET_MATE_PLIES, budget, attacker_moves="all"
    )
    assert result.verdict == "UNKNOWN", result
    assert result.line == ()
    assert result.nodes <= budget, result.nodes


def test_the_starvation_budgets_are_actually_starving() -> None:
    """The budgets above are below what the search needs, measured not assumed."""
    solved = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="all")
    assert solved.verdict == "MATE"
    assert solved.nodes > 100


# ---------------------------------------------------------------------------
# A7 -- the defender, when it is not in check
# ---------------------------------------------------------------------------


def test_and_node_reply_set_widens_when_the_defender_is_not_in_check() -> None:
    """A7(i): at a quiet AND node the replies are every legal move, not evasions.

    Under ``"checks"`` this node is unreachable: an AND node is entered only
    by a checking move, so the defender is always in check and its replies
    are always evasions. Here the defender is not in check, and the binding
    must offer it *everything* legal -- including the two soldier pushes,
    which answer no check because there is none to answer. A reply set
    narrowed to some notion of "evasions" would drop those two, and the
    defender would be confined to marshal moves it never had to make.
    """
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    defender = 1 - int(board.turn)
    problem = CyTsumeProblem(board, attacker_moves="all")
    node = _child_after(problem, board, QUIET_KEY_MOVE)

    assert not cyengine.in_check(node, defender), "the defender is in check here"
    assert not bool(cyengine.is_checkmate(node))
    assert problem.terminal(node) is None, "a node with replies is not terminal"

    replies = [san for san, _ in problem.children(node)]
    assert replies == _legal_replies(node), replies
    assert tuple(replies) == QUIET_MATE_REPLIES, replies

    non_marshal = [san for san in replies if not san.startswith("帥")]
    assert len(non_marshal) == 2, non_marshal
    assert set(non_marshal) == {"兵(5-4-1)(6-4-1)", "兵(5-4-1)(4-4-1)"}


def test_stalemate_is_not_scored_as_a_mate() -> None:
    """A7(ii): no reply and no check is a stalemate, and a failed attack.

    The dangerous case, end to end. ``大(4-2-1)(3-2-1)`` leaves the defender
    with nothing to play and no check to answer. The search's rule is that an
    AND node with no children is checkmate, so without
    :meth:`CyTsumeProblem.terminal` calling this node DISPROVED the solver
    would report a confident mate-in-1 on a position the rules engine calls
    a stalemate -- and a fabricated mate is strictly worse than the missed
    mates this whole mode exists to recover. The final assertion is the one
    that goes red if the stalemate branch is removed.
    """
    board = cyboard.board_from_fen(STALEMATE_FEN)
    defender = 1 - int(board.turn)
    problem = CyTsumeProblem(board, attacker_moves="all")

    assert not cyengine.in_check(board, defender), "not a legal tsume root"
    node = _child_after(problem, board, STALEMATE_MOVE)

    # The shape: no check, no reply -- but pseudo-legal moves do exist, so
    # this is a legality dead end rather than a defender with nothing on the
    # board.
    assert not cyengine.in_check(node, defender)
    assert not problem.children(node)
    assert not _legal_replies(node)
    assert len(cymovegen.generate_all_moves(node)) > 0
    assert not bool(cyengine.is_checkmate(node))

    # The rules engine's own reading of the same position.
    game = _replay(STALEMATE_FEN, (STALEMATE_MOVE,))
    assert bool(game.is_stalemate())
    assert not bool(game.is_checkmate())

    # What the binding tells the search, and what the search concludes.
    assert problem.terminal(node) is Outcome.DISPROVED
    assert solve_tsume(STALEMATE_FEN, 1, attacker_moves="all").verdict == "NO_MATE"
    assert (
        solve_tsume(
            STALEMATE_FEN, 3, FULL_WIDTH_BUDGET, attacker_moves="all"
        ).verdict
        == "NO_MATE"
    )


@pytest.mark.parametrize("plies", [1, 3])
def test_verify_checkmate_is_safe_at_stalemate_nodes(plies: int) -> None:
    """The armed escape-set assertion must not fire on a stalemate.

    ``verify_checkmate=True`` compares "no escaping reply" with
    ``cyengine.is_checkmate``, and that equivalence holds only for a
    defender *in check*. At the stalemate node the defender has no reply
    and is not mated, so an ungated assertion would raise here; the search
    must instead reach it quietly and answer NO_MATE.
    """
    result = solve_tsume(
        STALEMATE_FEN, plies, 30_000, attacker_moves="all", verify_checkmate=True
    )
    assert result.verdict == "NO_MATE", result


class _NoStalemateRuleProblem(CyTsumeProblem):
    """The binding with the stalemate branch taken back out of ``terminal``.

    This is exactly what :meth:`CyTsumeProblem.terminal` did before KOM-262 --
    game-over-by-captured-marshal and nothing else -- which was correct while
    every AND node was entered by a check. It is here so the stalemate rule
    can be pinned by the wrong answer it prevents.
    """

    def terminal(self, node: Board) -> Outcome | None:
        return Outcome.DISPROVED if _marshal_missing(node) else None


def test_without_the_stalemate_rule_the_search_invents_a_mate() -> None:
    """The bug the rule prevents, reproduced on the same position.

    Drop the branch and the stalemating move becomes an AND node with no
    children, which the df-pn search reads as checkmate: a confident MATE in
    one, with the stalemating move as its principal variation, on a position
    ``api.Gungi`` calls a stalemate. This is the failure mode that would make
    ``"all"`` worse than the checks-only default rather than better, so it is
    reproduced here rather than argued about.
    """
    root = cyboard.board_from_fen(STALEMATE_FEN)

    broken = solve_problem(
        _NoStalemateRuleProblem(root, attacker_moves="all"), root, 1, 100_000
    )
    assert broken.verdict == "MATE", (
        "the stalemate rule is no longer load-bearing here, so this test has "
        "stopped reproducing the bug it documents"
    )
    assert broken.line == (STALEMATE_MOVE,)

    fixed = solve_problem(
        CyTsumeProblem(root, attacker_moves="all"), root, 1, 100_000
    )
    assert fixed.verdict == "NO_MATE"


def test_checks_only_never_reaches_a_stalemate_node() -> None:
    """The default mode's freedom from all of the above, restated as a test.

    The claim in the module docstring is that stalemate needs no handling
    under ``"checks"`` because every AND node is entered by a checking move.
    Asserted here on the position built to break it: checks-only does not
    generate the stalemating move at all, and every AND node it does reach
    has the defender in check.
    """
    board = cyboard.board_from_fen(STALEMATE_FEN)
    defender = 1 - int(board.turn)
    problem = CyTsumeProblem(board)

    children = problem.children(board)
    assert STALEMATE_MOVE not in [san for san, _ in children]
    assert children, "the checks-only root has no moves, so this proves nothing"
    for _, child in children:
        assert cyengine.in_check(child, defender)


# ---------------------------------------------------------------------------
# The default, and the selector itself
# ---------------------------------------------------------------------------


def test_checks_is_the_default_everywhere() -> None:
    """Every existing caller keeps the narrow question without asking for it."""
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    assert CyTsumeProblem(board).attacker_moves == "checks"
    assert CyTsumeProblem(board, attacker_moves="checks").digest(
        board
    ) == CyTsumeProblem(board).digest(board)

    implicit = solve_tsume(QUIET_MATE_FEN, QUIET_MATE_PLIES)
    explicit = solve_tsume(
        QUIET_MATE_FEN, QUIET_MATE_PLIES, attacker_moves="checks"
    )
    assert implicit == explicit


def test_the_two_modes_have_different_digests() -> None:
    """The tag is on the digest, which is where the table key comes from."""
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    checks = CyTsumeProblem(board).digest(board)
    full = CyTsumeProblem(board, attacker_moves="all").digest(board)
    assert checks != full
    assert checks.endswith(cyengine.repetition_key(board))
    assert full.endswith(cyengine.repetition_key(board))


def test_an_unknown_mode_is_rejected() -> None:
    """A typo is a ValueError, not a silent fallback to one of the two modes."""
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    with pytest.raises(ValueError, match="attacker_moves"):
        CyTsumeProblem(board, attacker_moves="quiet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attacker_moves"):
        solve_tsume(QUIET_MATE_FEN, 1, attacker_moves="checks_only")  # type: ignore[arg-type]


def test_the_default_path_matches_the_base_golden() -> None:
    """A1 against the code *before* the selector, not against itself.

    The golden was captured at 02a6a03, before ``attacker_moves`` existed:
    every friend row at depths 1, 3 and its grade, every mined row and every
    no-mate position at 1 and 3. Comparing ``str(result)`` pins verdict,
    line and node count at once, so a change that merely made the default
    *slower* shows up too. Refusals are part of the contract and are
    compared by their message.
    """
    rows = json.loads(CHECKS_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(rows) == CHECKS_GOLDEN_ROWS

    fens: dict[str, str] = {f.name: f.fen for f in FRIEND}
    fens.update({m.name: m.fen for m in MINED})
    fens.update(dict(NO_MATE_POSITIONS))

    mismatches: list[tuple[str, int, str, str]] = []
    for name, plies, expected in rows:
        try:
            got = str(solve_tsume(fens[name], plies, CHECKS_GOLDEN_BUDGET))
        except ValueError as exc:
            got = str(exc)
        if got != expected:
            mismatches.append((name, plies, expected, got))
    assert not mismatches, mismatches


def test_the_reply_memo_cannot_be_mutated_through_children() -> None:
    """``children`` must not hand out the memo's own mutable object.

    Under ``"all"`` the defender's replies are memoised between
    ``terminal`` and ``children``. If the memo handed out its own list, a
    caller that cleared it would empty the cached reply set, and the next
    ``children`` of the same node would report a defender with no replies
    -- which the search reads as mate.
    """
    board = cyboard.board_from_fen(QUIET_MATE_FEN)
    problem = CyTsumeProblem(board, attacker_moves="all")
    node = _child_after(problem, board, QUIET_KEY_MOVE)

    first = problem.children(node)
    original = [san for san, _ in first]
    assert original, "no replies to mutate, so this proves nothing"
    if isinstance(first, list):
        first.clear()

    again = [san for san, _ in problem.children(node)]
    assert again, "the reply memo was emptied through children()"
    assert again == original


@pytest.mark.parametrize("puzzle", MINED[:3], ids=lambda m: m.name)
def test_mined_rows_agree_at_depth_one(puzzle: MinedPuzzle) -> None:
    """Depth 1 is the one depth where the two modes are the same question.

    A mate in one ply is delivered by a checking move, so widening the
    attacker's move list adds candidates that cannot mate. Anything other
    than agreement here would mean ``"all"`` had changed what *mate* means,
    not what moves are searched.
    """
    checks = solve_tsume(puzzle.fen, 1)
    full = solve_tsume(puzzle.fen, 1, attacker_moves="all")
    assert full.verdict == checks.verdict
    assert full.line == checks.line
