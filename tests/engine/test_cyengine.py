"""Tests for cyengine — Cython move application + terminal detection.

Four test categories:
  1. APPLY DIFFERENTIAL: for every ply in all four golden playouts, apply the
     chosen move via cyengine.apply_move and assert the resulting FEN matches
     the golden fen_after.
  2. VERDICT DIFFERENTIAL: for every ply, load board from fen_before and
     assert in_check/is_checkmate/is_stalemate/is_draw/is_game_over/turn/
     in_draft match the golden verdict.
  3. Cross-check against pure-Python gungi.api.Gungi on the same positions.
  4. CRAFTED terminals: checkmate, stalemate, insufficient-material
     (adjacent vs non-adjacent), and fourfold repetition.
"""

import numpy as np
import pytest

from conftest import load_golden
from komugi.engine.cyboard import board_from_fen, board_to_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.cyengine import (
    apply_move,
    in_check,
    is_checkmate,
    is_stalemate,
    is_insufficient_material,
    is_repetition_draw,
    is_draw,
    is_game_over,
    placement_key,
    repetition_key,
)
from komugi.engine import api as gungi_api


# ---------------------------------------------------------------------------
# Golden file list
# ---------------------------------------------------------------------------

_GOLDEN_FILES = [
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_move_by_san(moves, san):
    """Find a move record whose SAN matches *san* (stripping trailing =#)."""
    normalized = san.rstrip("=#")
    for mv in moves:
        if mv["san"] == normalized or mv["san"] == san:
            return mv
    return None


# ---------------------------------------------------------------------------
# 1. APPLY DIFFERENTIAL — golden playouts
# ---------------------------------------------------------------------------

class TestApplyDifferential:
    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_apply_move_matches_golden_fen_after(self, golden_file):
        """For every ply, apply the chosen move and assert FEN matches."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen_before = ply["fen_before"]
                chosen = ply["chosen"]
                expected_after = ply["fen_after"]

                board = board_from_fen(fen_before)
                moves = generate_all_moves(board)
                mv = _find_move_by_san(moves, chosen)
                assert mv is not None, (
                    f"{golden_file} game={gi} ply={pi}: "
                    f"chosen={chosen!r} not found in generated moves"
                )

                apply_move(board, mv)
                actual_after = board_to_fen(board)
                assert actual_after == expected_after, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"chosen={chosen}\n"
                    f"expected={expected_after}\n"
                    f"actual  ={actual_after}"
                )


# ---------------------------------------------------------------------------
# 2. VERDICT DIFFERENTIAL — golden playouts
# ---------------------------------------------------------------------------

class TestVerdictDifferential:
    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_verdict_matches_golden(self, golden_file):
        """For every ply, assert verdict fields match on fresh-loaded board."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                v = ply["verdict"]
                board = board_from_fen(fen)
                label = f"{golden_file} game={gi} ply={pi} fen={fen}"

                # turn: board.turn is 0=w, 1=b
                expected_turn = "w" if board.turn == 0 else "b"
                assert expected_turn == v["turn"], f"turn mismatch: {label}"

                # inDraft
                actual_in_draft = (board.draft_w != 0 or board.draft_b != 0)
                assert actual_in_draft == v["inDraft"], f"inDraft mismatch: {label}"

                # inCheck
                actual_in_check = in_check(board, board.turn)
                assert actual_in_check == v["inCheck"], f"inCheck mismatch: {label}"

                # isCheckmate
                actual_checkmate = is_checkmate(board)
                assert actual_checkmate == v["isCheckmate"], f"isCheckmate mismatch: {label}"

                # isStalemate
                actual_stalemate = is_stalemate(board)
                assert actual_stalemate == v["isStalemate"], f"isStalemate mismatch: {label}"

                # isDraw — fresh-loaded board has no history, so fourfold
                # is always False; matches golden since golden also fresh-loads.
                actual_draw = is_draw(board)
                assert actual_draw == v["isDraw"], f"isDraw mismatch: {label}"

                # isGameOver
                actual_game_over = is_game_over(board)
                assert actual_game_over == v["isGameOver"], f"isGameOver mismatch: {label}"


# ---------------------------------------------------------------------------
# 3. Cross-check against pure-Python oracle
# ---------------------------------------------------------------------------

class TestCrossCheckOracle:
    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_apply_matches_python_oracle(self, golden_file):
        """Apply each ply via both Cython and Python, compare after-FENs."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen_before = ply["fen_before"]
                chosen = ply["chosen"]

                # Python oracle
                g = gungi_api.Gungi(fen_before)
                g.move(chosen)
                py_after = g.fen()

                # Cython engine
                board = board_from_fen(fen_before)
                moves = generate_all_moves(board)
                mv = _find_move_by_san(moves, chosen)
                assert mv is not None
                apply_move(board, mv)
                cy_after = board_to_fen(board)

                assert cy_after == py_after, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"chosen={chosen}\n"
                    f"python ={py_after}\n"
                    f"cython ={cy_after}"
                )

    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_verdict_matches_python_oracle(self, golden_file):
        """For every ply, compare Cython verdicts against Python oracle."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                board = board_from_fen(fen)
                g = gungi_api.Gungi(fen)
                label = f"{golden_file} game={gi} ply={pi}"

                assert in_check(board, board.turn) == g.in_check(), (
                    f"in_check mismatch: {label}"
                )
                assert is_checkmate(board) == g.is_checkmate(), (
                    f"is_checkmate mismatch: {label}"
                )
                assert is_stalemate(board) == g.is_stalemate(), (
                    f"is_stalemate mismatch: {label}"
                )
                assert is_insufficient_material(board) == g.is_insufficient_material(), (
                    f"is_insufficient_material mismatch: {label}"
                )


# ---------------------------------------------------------------------------
# 4. CRAFTED terminals — ported from tests/test_api.py
# ---------------------------------------------------------------------------

class TestCraftedCheckmate:
    def test_detect_checkmate_when_in_check_no_escape(self):
        fen = (
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        board = board_from_fen(fen)
        assert is_checkmate(board) is True
        assert in_check(board, board.turn) is True
        assert is_game_over(board) is True

    def test_not_checkmate_when_in_check_with_escape(self):
        fen = (
            "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        board = board_from_fen(fen)
        assert is_checkmate(board) is False
        assert in_check(board, board.turn) is True

    def test_not_checkmate_when_not_in_check(self):
        fen = (
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        board = board_from_fen(fen)
        assert is_checkmate(board) is False
        assert in_check(board, board.turn) is False


class TestCraftedStalemate:
    def test_stalemate_not_in_check_all_moves_lead_to_check(self):
        fen = "8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1"
        board = board_from_fen(fen)
        assert in_check(board, board.turn) is False
        assert is_stalemate(board) is True
        assert is_draw(board) is True
        assert is_game_over(board) is True

    def test_not_stalemate_has_safe_moves(self):
        fen = (
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        board = board_from_fen(fen)
        assert is_stalemate(board) is False


class TestCraftedInsufficientMaterial:
    def test_two_non_adjacent_marshals(self):
        fen = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is True
        assert is_draw(board) is True
        assert is_game_over(board) is True

    def test_marshals_adjacent(self):
        fen = "mM7/9/9/9/9/9/9/9/9 -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is False

    def test_marshals_diagonally_adjacent(self):
        fen = "m8/1M7/9/9/9/9/9/9/9 -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is False

    def test_other_pieces_exist(self):
        fen = "m8/9/9/9/9/9/9/9/8M D1/d1 w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is False

    def test_only_one_marshal(self):
        fen = "m8/9/9/9/9/9/9/9/9 -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is False

    def test_three_or_more_marshals(self):
        fen = "mMm6/9/9/9/9/9/9/9/9 -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_insufficient_material(board) is False


class TestCraftedFourfoldRepetition:
    def test_placement_key_returns_the_243_byte_cells_block(self):
        """KOM-141: the cells-only key is real again, for referee parity only.

        It is gungi.js@3.4.11's ``isFourfoldRepetition`` key (FEN field 1), and
        it must be exactly the leading block of ``repetition_key`` so that one
        game can be scored under either rule. It used to raise, which was
        rules-audit Q13's guard; that guard is now
        ``tests/engine/test_repetition_rule_default.py``.
        """
        board = board_from_fen("4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1")
        key = placement_key(board)

        assert isinstance(key, bytes)
        assert len(key) == 243
        assert key == board.cells_arr.tobytes()
        assert key == repetition_key(board)[:243]

    def test_fourfold_repetition_via_apply_sequence(self):
        """Build a 4x repetition of the same repetition key via apply_move.

        The shuttle is a genuine cycle — same cells, same side to move, same
        (empty) hands every fourth ply — so the repaired key counts it exactly
        as the v1 cells-only key did, and it still draws at the 4th occurrence.
        """
        fen = "4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1"
        board = board_from_fen(fen)

        # Occurrence counter, seeded with the start position (as Game does).
        counts = {repetition_key(board): 1}

        # Shuttle generals back and forth to repeat the starting placement.
        # G 9-4 -> 8-4, g 1-4 -> 2-4, G 8-4 -> 9-4, g 2-4 -> 1-4 (2nd occ)
        # Repeat 2 more times for 3rd and 4th occurrences.
        shuttle_sans = [
            "\u5927(9-4-1)(8-4-1)",  # G 9-4 -> 8-4
            "\u5927(1-4-1)(2-4-1)",  # g 1-4 -> 2-4
            "\u5927(8-4-1)(9-4-1)",  # G 8-4 -> 9-4
            "\u5927(2-4-1)(1-4-1)",  # g 2-4 -> 1-4  -- 2nd occurrence
            "\u5927(9-4-1)(8-4-1)",
            "\u5927(1-4-1)(2-4-1)",
            "\u5927(8-4-1)(9-4-1)",
            "\u5927(2-4-1)(1-4-1)",  # 3rd occurrence
            "\u5927(9-4-1)(8-4-1)",
            "\u5927(1-4-1)(2-4-1)",
            "\u5927(8-4-1)(9-4-1)",
            "\u5927(2-4-1)(1-4-1)",  # 4th occurrence
        ]

        def current_count():
            return counts.get(repetition_key(board), 0)

        for san in shuttle_sans:
            assert is_repetition_draw(current_count()) is False, (
                f"Premature fourfold before move {san}"
            )
            moves = generate_all_moves(board)
            mv = _find_move_by_san(moves, san)
            assert mv is not None, f"Move {san!r} not found"
            apply_move(board, mv)
            key = repetition_key(board)
            counts[key] = counts.get(key, 0) + 1

        assert current_count() == 4, "the 4th occurrence of the start position"
        assert is_repetition_draw(current_count()) is True
        assert is_draw(board, current_count()) is True
        assert is_game_over(board, current_count()) is True

    def test_the_threshold_is_monotone_not_an_equality(self):
        """``>= 4``, never ``== 4`` (rules-audit F2)."""
        assert [is_repetition_draw(n) for n in (1, 2, 3)] == [False, False, False]
        assert all(is_repetition_draw(n) for n in (4, 5, 6, 9))


class TestDraftPhase:
    _DRAFT_FEN = (
        "9/9/9/9/9/9/9/9/9"
        " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
    )

    def test_checkmate_false_during_draft(self):
        board = board_from_fen(self._DRAFT_FEN)
        assert (board.draft_w != 0 or board.draft_b != 0) is True
        assert is_checkmate(board) is False

    def test_stalemate_false_during_draft(self):
        board = board_from_fen(self._DRAFT_FEN)
        assert is_stalemate(board) is False

    def test_insufficient_false_during_draft(self):
        board = board_from_fen(self._DRAFT_FEN)
        assert is_insufficient_material(board) is False

    def test_game_over_false_during_draft(self):
        board = board_from_fen(self._DRAFT_FEN)
        assert is_game_over(board) is False


class TestMarshalCaptured:
    _MC_FEN = (
        "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6"
        "/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164"
    )

    def test_game_over_marshal_captured(self):
        board = board_from_fen(self._MC_FEN)
        assert is_game_over(board) is True

    def test_not_checkmate_marshal_captured(self):
        board = board_from_fen(self._MC_FEN)
        assert is_checkmate(board) is False

    def test_not_stalemate_marshal_captured(self):
        board = board_from_fen(self._MC_FEN)
        assert is_stalemate(board) is False


class TestGameOverIntegration:
    def test_game_over_checkmate(self):
        fen = (
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        board = board_from_fen(fen)
        assert is_game_over(board) is True

    def test_game_over_insufficient(self):
        fen = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_game_over(board) is True

    def test_not_game_over_normal(self):
        fen = (
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        board = board_from_fen(fen)
        assert is_game_over(board) is False


class TestDrawDetection:
    def test_draw_insufficient(self):
        fen = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"
        board = board_from_fen(fen)
        assert is_draw(board) is True

    def test_not_draw_normal(self):
        fen = (
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        board = board_from_fen(fen)
        assert is_draw(board) is False

    def test_not_draw_checkmate(self):
        fen = (
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        board = board_from_fen(fen)
        assert is_draw(board) is False
