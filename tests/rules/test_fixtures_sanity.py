"""The fixtures, checked against the CURRENT engine -- unconditionally.

Everything else in ``tests/rules/`` is gated on KOM-027 and skips until the
repair lands.  This module is not: it exercises only the engine API that exists
today, so the suite has teeth in the worktree that authored it.  What it pins
is exactly the part of the fixtures that the repair does not change -- the
positions load from their FENs, the scripted lines from the rules audit are
legal move sequences, and each one restores the cells it claims to restore.

If any of these fail, the red-first tests next door are testing nothing: a
sequence that goes illegal halfway would leave the game in some position nobody
reasoned about, and an "aliasing" assertion on it would be meaningless.

The audit was a source read, never executed (``docs/design/rules-audit.md``
opening note), so this module is also the first execution of its scripted
lines.
"""

from __future__ import annotations

import pytest

pytest.importorskip("komugi.engine.cyengine")

from komugi.engine import cyengine  # noqa: E402
from komugi.engine.cyboard import Board, make_code  # noqa: E402

from fixtures import (  # noqa: E402
    BLACK_COLOR,
    PURE_SHUFFLE,
    SCENARIO_A_15PLY,
    SCENARIO_A_CYCLE,
    SCENARIO_B_LOOP,
    SOLDIER_INDEX,
    WHITE_COLOR,
    board_of,
    cells_bytes,
    hand_count,
    load_mate_position,
    load_position_p,
    load_position_q,
    play_sequence,
)

# Piece codes: type_index * 2 + color + 1 (cyboard's encoding).
_MARSHAL_W = make_code(0, WHITE_COLOR)
_MARSHAL_B = make_code(0, BLACK_COLOR)
_WARRIOR_W = make_code(4, WHITE_COLOR)
_LANCER_B = make_code(5, BLACK_COLOR)
_RIDER_B = make_code(6, BLACK_COLOR)


def _top(board: Board, rank: int, file: int) -> int:
    """Top piece code at 1-based ``(rank, file)`` -- ``Board.top`` is 0-based."""
    return int(board.top(rank - 1, file - 1))


# ---------------------------------------------------------------------------
# The positions load
# ---------------------------------------------------------------------------


def test_position_p_loads_from_its_fen() -> None:
    """P is the audit's Scenario A position: four pieces, empty hands, White."""
    game = load_position_p()
    board = board_of(game)

    assert board.turn == WHITE_COLOR
    assert board.mode == 3, "advanced mode -- FEN field 4"
    assert (board.draft_w, board.draft_b) == (0, 0), "both draft bits clear"

    assert _top(board, 9, 5) == _MARSHAL_W, "白 帥 on 9-5"
    assert _top(board, 5, 1) == _WARRIOR_W, "白 侍 on 5-1"
    assert _top(board, 1, 5) == _MARSHAL_B, "黒 帥 on 1-5"
    assert _top(board, 1, 1) == _LANCER_B, "黒 槍 on 1-1"

    assert int(board.hand_arr.sum()) == 0, "both hands empty"
    assert not cyengine.is_insufficient_material(board), (
        "each side keeps a non-marshal, so the insufficient-material draw "
        "cannot pre-empt the repetition rule"
    )


def test_position_q_loads_from_its_fen() -> None:
    """Q is P plus a black 馬 on 3-9 and two 兵 in White's hand."""
    game = load_position_q()
    board = board_of(game)

    assert board.turn == WHITE_COLOR
    assert board.mode == 3
    assert (board.draft_w, board.draft_b) == (0, 0)

    assert _top(board, 9, 5) == _MARSHAL_W
    assert _top(board, 5, 1) == _WARRIOR_W
    assert _top(board, 1, 5) == _MARSHAL_B
    assert _top(board, 1, 1) == _LANCER_B
    assert _top(board, 3, 9) == _RIDER_B, "黒 馬 on 3-9"

    assert hand_count(game, SOLDIER_INDEX, WHITE_COLOR) == 2, "白 hand 兵 x2"
    assert int(board.hand_arr.sum()) == 2, "and nothing else in either hand"


def test_the_two_positions_differ_only_where_the_audit_says() -> None:
    """Q's extra 馬 is the only cells difference; the hands are the rest."""
    p_board = board_of(load_position_p())
    q_board = board_of(load_position_q())

    assert cells_bytes(load_position_p()) != cells_bytes(load_position_q())
    assert _top(p_board, 3, 9) == 0
    assert _top(q_board, 3, 9) == _RIDER_B
    assert int(p_board.hand_arr.sum()) == 0
    assert int(q_board.hand_arr.sum()) == 2


def test_the_mate_fixture_really_is_checkmate_today() -> None:
    """RT-19's fixture is checkmate under the CURRENT engine.

    RT-19 itself is gated behind KOM-027, so without this test the mate
    construction would go unexercised until the repair lands and could be
    quietly wrong -- an F8 precedence pin that pins nothing.  Beginner mode
    (``mode == 1``) is load-bearing: it clears ``marshal_can_stack``, closing
    the 帥付(2-2) tier-2 escape that advanced mode would allow.
    """
    game = load_mate_position()
    board = board_of(game)

    assert board.mode == 1, "beginner -- see MATE_AND_REPETITION_FEN"
    assert board.turn == BLACK_COLOR, "Black is the mated side"
    assert cyengine.in_check(board, BLACK_COLOR)
    assert cyengine.is_checkmate(board)
    assert not cyengine.is_stalemate(board)
    assert not cyengine.is_insufficient_material(board)
    assert int(game.win_state()[WHITE_COLOR]) == 1, (
        "with no repetition in the history the mate simply wins for White"
    )


# ---------------------------------------------------------------------------
# The scripted sequences are legal
# ---------------------------------------------------------------------------


def test_scenario_a_cycle_is_legal_and_restores_the_cells() -> None:
    """Audit Scenario A, plies 1-5: cells return to P, side to move flips.

    This is the aliasing defect in its rawest form and it needs no repaired
    key to observe: on today's engine the cells are byte-identical while the
    game state plainly is not.
    """
    game = load_position_p()
    before = cells_bytes(game)

    play_sequence(game, SCENARIO_A_CYCLE)

    assert cells_bytes(game) == before, "5-ply cycle restores P's cells exactly"
    assert board_of(game).turn == BLACK_COLOR, (
        "w = 3 white plies vs b = 2 black plies flips the side to move -- "
        "the whole point of Scenario A"
    )
    assert int(board_of(game).hand_arr.sum()) == 0, "no drops, hands untouched"


def test_scenario_a_full_15_ply_line_is_legal() -> None:
    """All 15 plies are legal, and the cells equal P at plies 5, 10 and 15."""
    game = load_position_p()
    before = cells_bytes(game)

    for cycle_index in range(3):
        play_sequence(
            game,
            SCENARIO_A_15PLY[cycle_index * 5:(cycle_index + 1) * 5],
        )
        assert cells_bytes(game) == before, (
            f"cells must equal P after cycle {cycle_index + 1}"
        )

    assert board_of(game).turn == BLACK_COLOR, "odd number of cycles"


def test_scenario_b_loop_is_legal_and_spends_exactly_one_soldier() -> None:
    """Audit Scenario B, plies 1-8: same cells, same side to move, one 兵 gone.

    The 8-ply loop is ``w = b = 4``, so unlike Scenario A it does NOT flip the
    side to move -- what changed is invisible to the cells: White paid a 兵 and
    Black is a soldier up, permanently.
    """
    game = load_position_q()
    before = cells_bytes(game)
    soldiers_before = hand_count(game, SOLDIER_INDEX, WHITE_COLOR)

    play_sequence(game, SCENARIO_B_LOOP)

    assert cells_bytes(game) == before, "8-ply loop restores Q's cells exactly"
    assert board_of(game).turn == WHITE_COLOR, "matched ply counts, White again"

    soldiers_after = hand_count(game, SOLDIER_INDEX, WHITE_COLOR)
    assert soldiers_after == soldiers_before - 1, (
        "the dropped 兵 was captured and never returns: hands only decrease "
        "in this engine"
    )


def test_scenario_b_second_loop_and_shuffle_are_legal() -> None:
    """The full RT-09 line: two 8-ply loops, then the 4-ply pure shuffle."""
    game = load_position_q()
    before = cells_bytes(game)

    play_sequence(game, SCENARIO_B_LOOP)
    play_sequence(game, SCENARIO_B_LOOP)
    assert cells_bytes(game) == before
    assert hand_count(game, SOLDIER_INDEX, WHITE_COLOR) == 0, (
        "both 兵 spent -- two irreversible expenditures and two captures"
    )

    play_sequence(game, PURE_SHUFFLE)
    assert cells_bytes(game) == before
    assert board_of(game).turn == WHITE_COLOR


def test_pure_shuffle_is_a_genuine_cycle() -> None:
    """The 4-ply shuffle restores cells, side to move AND hands, three times.

    Three consecutive cycles is what RT-01 plays, so its legality on today's
    engine has to hold before RT-01 can mean anything.
    """
    game = load_position_p()
    before = cells_bytes(game)

    for cycle_index in range(3):
        play_sequence(game, PURE_SHUFFLE)
        assert cells_bytes(game) == before, f"cycle {cycle_index + 1}"
        assert board_of(game).turn == WHITE_COLOR
        assert int(board_of(game).hand_arr.sum()) == 0
