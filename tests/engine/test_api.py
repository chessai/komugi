"""Tests for gungi.api -- Gungi class game-ending detection, verdict/replay
parity against oracle goldens, and corner cases."""

import pytest

from conftest import load_golden
from komugi.engine.api import Gungi


# ---------------------------------------------------------------------------
# 1. PORT of reference/gungi.js/test/game-ending.test.ts  (verbatim FENs)
# ---------------------------------------------------------------------------

class TestCheckmate:
    def test_detect_checkmate_when_in_check_no_escape(self):
        g = Gungi(
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.is_checkmate() is True
        assert g.in_check() is True
        assert g.is_game_over() is True

    def test_not_checkmate_when_in_check_with_escape(self):
        g = Gungi(
            "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.is_checkmate() is False
        assert g.in_check() is True
        assert len(g.moves()) > 0

    def test_not_checkmate_when_not_in_check(self):
        g = Gungi(
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        assert g.is_checkmate() is False
        assert g.in_check() is False


class TestStalemate:
    def test_stalemate_not_in_check_all_moves_lead_to_check(self):
        g = Gungi("8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1")
        assert g.in_check() is False
        assert g.is_stalemate() is True
        assert g.is_draw() is True
        assert g.is_game_over() is True
        assert len(g.moves()) == 0

    def test_not_stalemate_has_safe_moves(self):
        g = Gungi(
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        assert g.is_stalemate() is False
        assert len(g.moves()) > 0

    def test_not_stalemate_when_in_check(self):
        g = Gungi(
            "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.in_check() is True
        assert g.is_stalemate() is False

    def test_not_stalemate_normal_position(self):
        g = Gungi(
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 1"
        )
        assert g.is_stalemate() is False


class TestInsufficientMaterial:
    def test_two_non_adjacent_marshals(self):
        g = Gungi("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1")
        assert g.is_insufficient_material() is True
        assert g.is_draw() is True
        assert g.is_game_over() is True

    def test_marshals_adjacent(self):
        g = Gungi("mM7/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        assert g.is_insufficient_material() is False

    def test_marshals_diagonally_adjacent(self):
        g = Gungi("m8/1M7/9/9/9/9/9/9/9 -/- w 3 - 1")
        assert g.is_insufficient_material() is False

    def test_other_pieces_exist(self):
        g = Gungi("m8/9/9/9/9/9/9/9/8M D1/d1 w 3 - 1")
        assert g.is_insufficient_material() is False

    def test_only_one_marshal(self):
        g = Gungi("m8/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        assert g.is_insufficient_material() is False

    def test_three_or_more_marshals(self):
        g = Gungi("mMm6/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        assert g.is_insufficient_material() is False


class TestDraftPhase:
    _DRAFT_FEN = (
        "9/9/9/9/9/9/9/9/9"
        " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
    )

    def test_checkmate_false_during_draft(self):
        g = Gungi(self._DRAFT_FEN)
        assert g.in_draft() is True
        assert g.is_checkmate() is False

    def test_stalemate_false_during_draft(self):
        g = Gungi(self._DRAFT_FEN)
        assert g.in_draft() is True
        assert g.is_stalemate() is False

    def test_insufficient_false_during_draft(self):
        g = Gungi(self._DRAFT_FEN)
        assert g.in_draft() is True
        assert g.is_insufficient_material() is False

    def test_game_over_false_during_draft(self):
        g = Gungi(self._DRAFT_FEN)
        assert g.in_draft() is True
        assert g.is_game_over() is False

    def test_only_draft_end_moves_when_one_piece_left(self):
        g = Gungi(
            "9/9/9/9/9/9/9/9/4M4"
            " D1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 w 1"
        )
        mvs = g.moves()
        assert len(mvs) > 0
        assert all(isinstance(s, str) and s.endswith("\u7d42") for s in mvs)

    def test_regular_and_draft_end_moves_when_two_plus_pieces(self):
        g = Gungi(
            "9/9/9/9/9/9/9/9/4M4"
            " D2/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 w 1"
        )
        mvs = g.moves()
        regular = [s for s in mvs if not s.endswith("\u7d42")]
        draft_end = [s for s in mvs if s.endswith("\u7d42")]
        assert len(regular) > 0
        assert len(draft_end) > 0


class TestMarshalCaptured:
    _MC_FEN = (
        "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6"
        "/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164"
    )

    def test_game_over_marshal_captured(self):
        g = Gungi(self._MC_FEN)
        assert g.is_game_over() is True

    def test_not_checkmate_marshal_captured(self):
        g = Gungi(self._MC_FEN)
        assert g.is_checkmate() is False

    def test_not_stalemate_marshal_captured(self):
        g = Gungi(self._MC_FEN)
        assert g.is_stalemate() is False


class TestGungiSpecificRules:
    def test_allow_moves_while_in_check(self):
        g = Gungi(
            "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.in_check() is True
        assert len(g.moves()) > 0

    def test_allow_moving_into_check(self):
        g = Gungi(
            "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert len(g.moves()) > 0


class TestGameOverIntegration:
    def test_game_over_checkmate(self):
        g = Gungi(
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.is_game_over() is True

    def test_game_over_insufficient(self):
        g = Gungi("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1")
        assert g.is_game_over() is True

    def test_not_game_over_normal(self):
        g = Gungi(
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        assert g.is_game_over() is False


class TestDrawDetection:
    def test_draw_insufficient(self):
        g = Gungi("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1")
        assert g.is_draw() is True

    def test_not_draw_normal(self):
        g = Gungi(
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        assert g.is_draw() is False

    def test_not_draw_checkmate(self):
        g = Gungi(
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.is_draw() is False


# ---------------------------------------------------------------------------
# 2. VERDICT + REPLAY parity against golden playouts
# ---------------------------------------------------------------------------

_GOLDEN_FILES = [
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
]


class TestGoldenVerdictReplay:
    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_verdict_parity(self, golden_file):
        """For every ply, load Gungi(fen_before) and assert verdict fields."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                v = ply["verdict"]
                g = Gungi(fen)
                label = f"{golden_file} game={gi} ply={pi} fen={fen}"

                assert g.turn() == v["turn"], f"turn mismatch: {label}"
                assert g.in_draft() == v["inDraft"], f"inDraft mismatch: {label}"
                assert g.in_check() == v["inCheck"], f"inCheck mismatch: {label}"
                assert g.is_checkmate() == v["isCheckmate"], f"isCheckmate mismatch: {label}"
                assert g.is_stalemate() == v["isStalemate"], f"isStalemate mismatch: {label}"
                assert g.is_draw() == v["isDraw"], f"isDraw mismatch: {label}"
                assert g.is_game_over() == v["isGameOver"], f"isGameOver mismatch: {label}"

    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_replay_parity(self, golden_file):
        """For every ply, apply chosen move and assert fen() == fen_after."""
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                chosen = ply["chosen"]
                expected_after = ply["fen_after"]
                g = Gungi(fen)
                g.move(chosen)
                actual_after = g.fen()
                assert actual_after == expected_after, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"chosen={chosen}\n"
                    f"expected={expected_after}\n"
                    f"actual={actual_after}"
                )


# ---------------------------------------------------------------------------
# 3. Corner cases
# ---------------------------------------------------------------------------

class TestCornerCases:
    def test_crafted_checkmate_is_game_over_not_draw(self):
        """Checkmate is game-over but NOT a draw."""
        g = Gungi(
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        assert g.is_checkmate() is True
        assert g.is_game_over() is True
        assert g.is_draw() is False

    def test_crafted_stalemate_is_draw_not_checkmate(self):
        """Stalemate IS a draw and game-over, but NOT checkmate."""
        g = Gungi("8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1")
        assert g.is_stalemate() is True
        assert g.is_draw() is True
        assert g.is_game_over() is True
        assert g.is_checkmate() is False

    def test_insufficient_marshals_adjacent_not_draw(self):
        """Two marshals that ARE adjacent: not insufficient material."""
        g = Gungi("mM7/9/9/9/9/9/9/9/9 -/- w 3 - 1")
        assert g.is_insufficient_material() is False
        # Adjacent marshals should not trigger insufficient-material draw
        assert g.is_game_over() is False

    def test_insufficient_marshals_non_adjacent_draw(self):
        """Two marshals NOT adjacent: insufficient material = draw."""
        g = Gungi("m8/9/9/9/9/9/9/9/8M -/- w 3 - 1")
        assert g.is_insufficient_material() is True
        assert g.is_draw() is True

    def test_fourfold_repetition_exactly_four(self):
        """Build a line where the same placement appears exactly 4 times."""
        # White marshal at 9-5, black marshal at 1-5.
        # Pieces: white general at 9-4 and black general at 1-4 to allow
        # back-and-forth moves.
        fen = "4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1"
        g = Gungi(fen)

        # The placement field of fen is "3gm4/9/9/9/9/9/9/9/4MG3".
        # Shuttle generals back and forth to create fourfold repetition
        # of this exact placement.
        shuttle = [
            # Move white general, then black general, then back
            "\u5927(9-4-1)(8-4-1)",  # G 9-4 -> 8-4
            "\u5927(1-4-1)(2-4-1)",  # g 1-4 -> 2-4
            "\u5927(8-4-1)(9-4-1)",  # G 8-4 -> 9-4
            "\u5927(2-4-1)(1-4-1)",  # g 2-4 -> 1-4  -- 2nd occurrence of start
            "\u5927(9-4-1)(8-4-1)",
            "\u5927(1-4-1)(2-4-1)",
            "\u5927(8-4-1)(9-4-1)",
            "\u5927(2-4-1)(1-4-1)",  # 3rd occurrence
            "\u5927(9-4-1)(8-4-1)",
            "\u5927(1-4-1)(2-4-1)",
            "\u5927(8-4-1)(9-4-1)",
            "\u5927(2-4-1)(1-4-1)",  # 4th occurrence
        ]

        for san in shuttle:
            assert g.is_fourfold_repetition() is False, (
                f"Premature fourfold before move {san}"
            )
            g.move(san)

        assert g.is_fourfold_repetition() is True
        assert g.is_draw() is True
        assert g.is_game_over() is True
