"""Tests for cymovegen — Cython move generation on the typed board.

Two test categories:
  1. Differential oracle: for every ply in all four golden playouts, compare
     the Cython-generated SAN set against BOTH the golden legal set AND the
     pure-Python gungi.api oracle.
  2. Targeted rule tests: specific scenarios ported from test_move_gen.py,
     run through the Cython path.
"""

import numpy as np
import pytest

from conftest import load_golden
from komugi.engine.cyboard import board_from_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine import api as gungi_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cy_sans(fen):
    """Return sorted SAN list from Cython move generation."""
    b = board_from_fen(fen)
    moves = generate_all_moves(b)
    return sorted(m["san"] for m in moves)


def _py_sans(fen):
    """Return sorted SAN list from pure-Python oracle."""
    g = gungi_api.Gungi(fen)
    return sorted(m["san"] for m in g.moves({"verbose": True}))


def _cy_moves(fen):
    """Return list of move dicts from Cython move generation."""
    b = board_from_fen(fen)
    return generate_all_moves(b)


# Piece kanji constants (for readability in targeted tests).
_MARSHAL = "\u5e25"
_GENERAL = "\u5927"
_LT_GENERAL = "\u4e2d"
_MJ_GENERAL = "\u5c0f"
_WARRIOR = "\u4f8d"
_LANCER = "\u69cd"
_RIDER = "\u99ac"
_SPY = "\u5fcd"
_FORTRESS = "\u7826"
_SOLDIER = "\u5175"
_CANNON = "\u7832"
_ARCHER = "\u5f13"
_MUSKETEER = "\u7b52"
_TACTICIAN = "\u8b00"


# ---------------------------------------------------------------------------
# 1. Differential oracle: golden playouts
# ---------------------------------------------------------------------------

_GOLDEN_FILES = [
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
]


class TestGoldenPlayoutsCython:
    """For every ply in all golden playouts, the Cython move gen must produce
    the exact same sorted SAN set as the golden legal set AND the pure-Python
    oracle."""

    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_legal_moves_match_golden_and_oracle(self, golden_file):
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                expected_legal = ply["legal"]

                cy_legal = _cy_sans(fen)
                py_legal = _py_sans(fen)

                assert cy_legal == expected_legal, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"fen={fen}\n"
                    f"extra={sorted(set(cy_legal) - set(expected_legal))}\n"
                    f"missing={sorted(set(expected_legal) - set(cy_legal))}"
                )
                assert cy_legal == py_legal, (
                    f"{golden_file} game={gi} ply={pi} "
                    f"Cython vs Python oracle mismatch\n"
                    f"fen={fen}\n"
                    f"cy_extra={sorted(set(cy_legal) - set(py_legal))}\n"
                    f"cy_missing={sorted(set(py_legal) - set(cy_legal))}"
                )


# ---------------------------------------------------------------------------
# 2. Targeted rule tests (ported from test_move_gen.py, Cython path)
# ---------------------------------------------------------------------------

# Helper to build a FEN from a minimal board description.
from komugi.engine.fen import encode_fen


def _create_empty_board():
    return [[[None] for _ in range(9)] for _ in range(9)]


def _place_tower(board, square, pieces):
    parts = square.split("-")
    rank, file_ = int(parts[0]), int(parts[1])
    board[rank - 1][9 - file_] = [
        {"square": square, "tier": i + 1, "type": p["type"], "color": p["color"]}
        for i, p in enumerate(pieces)
    ]


def _make_fen(board, hand=None, turn="w", mode="advanced", drafting=None, move_number=1):
    if hand is None:
        hand = []
    if drafting is None:
        drafting = {"b": False, "w": False}
    return encode_fen({
        "board": board,
        "hand": hand,
        "turn": turn,
        "mode": mode,
        "drafting": drafting,
        "moveNumber": move_number,
    })


def _cy_targets(fen, square):
    """Return list of target squares (rank-file, no tier) from Cython board moves."""
    b = board_from_fen(fen)
    parts = square.split("-")
    py, px = int(parts[0]), int(parts[1])
    moves = generate_all_moves(b)
    targets = []
    for m in moves:
        if m["from_r"] == py - 1 and m["from_f"] == px - 1:
            targets.append(f"{m['to_r'] + 1}-{m['to_f'] + 1}")
    return targets


class TestArcherWingBlockingCython:
    def test_blocks_diagonal_when_wing_has_higher_tower(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": _ARCHER, "color": "w"}])
        _place_tower(board, "4-6", [
            {"type": _SOLDIER, "color": "b"},
            {"type": _WARRIOR, "color": "b"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board)
        targets = _cy_targets(fen, "5-5")
        assert "3-4" in targets
        assert "3-6" not in targets

    def test_keeps_diagonal_legal_when_wing_not_higher(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": _ARCHER, "color": "w"}])
        _place_tower(board, "4-6", [{"type": _SOLDIER, "color": "b"}])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board)
        targets = _cy_targets(fen, "5-5")
        assert "3-6" in targets

    def test_black_archer_wing_blocking(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": _ARCHER, "color": "b"}])
        _place_tower(board, "6-4", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _WARRIOR, "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _cy_targets(fen, "5-5")
        assert "7-4" not in targets
        assert "7-6" in targets

    def test_black_archer_wing_not_higher_legal(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": _ARCHER, "color": "b"}])
        _place_tower(board, "6-4", [{"type": _SOLDIER, "color": "w"}])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _cy_targets(fen, "5-5")
        assert "7-4" in targets


class TestForwardOnlyLeapCython:
    def test_cannon_leaps_forward_not_sideways_backward(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _WARRIOR, "color": "w"},
            {"type": _CANNON, "color": "w"},
        ])
        _place_tower(board, "2-5", [{"type": _SOLDIER, "color": "b"}])
        _place_tower(board, "5-6", [{"type": _SOLDIER, "color": "b"}])
        _place_tower(board, "6-5", [{"type": _SOLDIER, "color": "b"}])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board)
        targets = _cy_targets(fen, "5-5")
        assert "1-5" in targets
        assert "5-7" not in targets
        assert "7-5" not in targets

    def test_black_cannon_leaps_forward_not_backward(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SOLDIER, "color": "b"},
            {"type": _WARRIOR, "color": "b"},
            {"type": _CANNON, "color": "b"},
        ])
        _place_tower(board, "8-5", [{"type": _SOLDIER, "color": "w"}])
        _place_tower(board, "4-5", [{"type": _SOLDIER, "color": "w"}])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _cy_targets(fen, "5-5")
        assert "9-5" in targets
        assert "3-5" not in targets


class TestBetrayalPowersetCython:
    def test_white_tactician_powerset_two_enemies(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SPY, "color": "b"},
            {"type": _SOLDIER, "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _TACTICIAN, "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": _SPY, "color": "w", "count": 1},
                {"type": _SOLDIER, "color": "w", "count": 1},
            ],
        )

        moves = _cy_moves(fen)
        betray_moves = [m for m in moves if m["move_type"] == 3]  # MT_BETRAY
        betray_sans = [m["san"] for m in betray_moves]

        assert "\u8b00(4-5-2)(5-5-3)\u8fd4\u5fcd" in betray_sans
        assert "\u8b00(4-5-2)(5-5-3)\u8fd4\u5fcd\u5175" in betray_sans
        assert "\u8b00(4-5-2)(5-5-3)\u8fd4\u5175" in betray_sans
        assert len(betray_sans) == 3

        # Verify recursive pre-order
        combo_types = []
        for m in betray_moves:
            combo_types.append([cap[0] for cap in m["captured"]])
        # spy=7, soldier=9
        assert combo_types == [[7], [7, 9], [9]]

    def test_affordability_boundary_excluded(self):
        """2 enemy soldiers but only 1 in hand → no betrayal."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SOLDIER, "color": "b"},
            {"type": _SOLDIER, "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _TACTICIAN, "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(
            board,
            hand=[{"type": _SOLDIER, "color": "w", "count": 1}],
        )

        moves = _cy_moves(fen)
        betray_moves = [m for m in moves if m["move_type"] == 3]
        assert len(betray_moves) == 0

    def test_affordability_boundary_included(self):
        """2 enemy soldiers and 2 in hand → 3 combos."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SOLDIER, "color": "b"},
            {"type": _SOLDIER, "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _TACTICIAN, "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(
            board,
            hand=[{"type": _SOLDIER, "color": "w", "count": 2}],
        )

        moves = _cy_moves(fen)
        betray_moves = [m for m in moves if m["move_type"] == 3]
        assert len(betray_moves) == 3


class TestDraftShuCython:
    def test_last_piece_draft_only_shu_variants(self):
        """Last hand piece → only 終 variants, and no conversion at all.

        KOM-170 changed the second half of this test.  It used to assert that
        dropping the LAST 謀 onto a tower holding an enemy 謀 produced
        新謀(8-5-3)返謀終, which is the underflow: the placement and the
        conversion are both paid out of hand[謀], so a hand of one pays two.
        Under the fix a conversion costs the dropped piece PLUS one more of the
        converted type, and a last-piece drop leaves the hand empty by
        definition — so a draft's last piece can never convert anything, for
        any type.  That is what is asserted now; the plain 終 drop is untouched.

        The pure-Python oracle still generates the old move
        (tests/engine/test_move_gen.py::test_last_piece_draft_betray_tactician_enemy),
        deliberately: it is the v1 reference transcription and gungi.js really
        does offer this move.  tests/rules/test_betrayal_hand_invariant.py
        carries the divergence argument.
        """
        board = _create_empty_board()
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])
        _place_tower(board, "8-5", [
            {"type": _TACTICIAN, "color": "b"},
            {"type": _SOLDIER, "color": "w"},
        ])

        fen = _make_fen(
            board,
            hand=[{"type": _TACTICIAN, "color": "w", "count": 1}],
            drafting={"b": False, "w": True},
        )

        moves = _cy_moves(fen)
        moves_85 = [m for m in moves if m["to_r"] == 7 and m["to_f"] == 4]

        for m in moves_85:
            assert m["san"].endswith("\u7d42"), (
                f"Expected \u7d42 suffix for last-piece draft, got: {m['san']}"
            )

        # The plain drop is still there ...
        sans_85 = [m["san"] for m in moves_85]
        assert "新謀(8-5-3)付終" in sans_85, sans_85

        # ... and it is the ONLY move at that square: converting the enemy 謀
        # would need a second 謀 in a hand that the drop has just emptied.
        betray_85 = [m for m in moves_85 if m["captured"] and len(m["captured"]) > 0]
        assert betray_85 == [], [m["san"] for m in betray_85]
        assert "新謀(8-5-3)返謀終" not in sans_85

    def test_non_last_piece_draft_has_both_variants(self):
        """Non-last piece → both 終 and non-終 variants.

        The enemy under the drop square is a 兵, NOT a 謀 (KOM-170): a dropped
        謀 converting an enemy 謀 would have to pay for the placement and the
        conversion out of the same hand slot, which one 謀 cannot do.  A 兵 is
        a different slot, so the 兵 in hand pays for it and the betray variants
        this test is about still exist.
        """
        board = _create_empty_board()
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])
        _place_tower(board, "8-5", [
            {"type": _SOLDIER, "color": "b"},
            {"type": _SOLDIER, "color": "w"},
        ])

        fen = _make_fen(
            board,
            hand=[
                {"type": _TACTICIAN, "color": "w", "count": 1},
                {"type": _SOLDIER, "color": "w", "count": 1},
            ],
            drafting={"b": False, "w": True},
        )

        moves = _cy_moves(fen)
        moves_85 = [m for m in moves if m["to_r"] == 7 and m["to_f"] == 4]

        betray_85 = [m for m in moves_85 if m["captured"] and len(m["captured"]) > 0]
        betray_sans = [m["san"] for m in betray_85]

        non_shu = [s for s in betray_sans if not s.endswith("\u7d42")]
        shu = [s for s in betray_sans if s.endswith("\u7d42")]
        assert len(non_shu) > 0
        assert len(shu) > 0


class TestGameOverHashCython:
    def test_capture_marshal_appends_hash(self):
        """Capture that takes a marshal → SAN ends with '#'."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "4-5", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(board)
        moves = _cy_moves(fen)
        capture_moves = [m for m in moves if m["move_type"] == 2]  # MT_CAPTURE
        assert len(capture_moves) >= 1
        for m in capture_moves:
            if m["to_r"] == 4 and m["to_f"] == 4:  # 5-5 in 0-based
                assert m["san"].endswith("#"), f"Expected '#', got: {m['san']}"

    def test_betray_marshal_appends_hash(self):
        """Betray that converts a marshal → SAN ends with '#'."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _MARSHAL, "color": "b"},
            {"type": _SOLDIER, "color": "w"},
        ])
        _place_tower(board, "4-5", [
            {"type": _SOLDIER, "color": "w"},
            {"type": _TACTICIAN, "color": "w"},
        ])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(
            board,
            hand=[{"type": _MARSHAL, "color": "w", "count": 1}],
        )

        moves = _cy_moves(fen)
        betray_moves = [m for m in moves if m["move_type"] == 3]
        assert len(betray_moves) == 1
        assert betray_moves[0]["san"].endswith("#")


class TestArataBetrayalCython:
    def test_white_tactician_arata_betrayal(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": _SPY, "color": "b"},
            {"type": _SOLDIER, "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": _MARSHAL, "color": "b"}])
        _place_tower(board, "9-9", [{"type": _MARSHAL, "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": _TACTICIAN, "color": "w", "count": 1},
                {"type": _SPY, "color": "w", "count": 1},
            ],
        )

        sans = _cy_sans(fen)
        assert "\u65b0\u8b00(5-5-3)\u8fd4\u5fcd" in sans  # 新謀(5-5-3)返忍
