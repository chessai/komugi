"""Tests for gungi.move_gen — oracle golden playout + targeted rule tests."""

import pytest

from conftest import load_golden
from komugi.engine.fen import encode_fen, parse_fen
from komugi.engine.move_gen import (
    generate_all_moves,
    generate_moves_for_square,
)
from komugi.engine.utils import piece


# ---------------------------------------------------------------------------
# Helpers (mirrors the TS test helpers in rules-update.test.ts)
# ---------------------------------------------------------------------------

def _create_empty_board():
    return [[[ None ] for _ in range(9)] for _ in range(9)]


def _place_tower(board, square, pieces):
    """Place a tower of pieces on the board.

    *pieces* is a list of {"type": kanji, "color": "w"|"b"} dicts.
    """
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


def _get_targets(fen, square):
    """Return list of target squares (rank-file, no tier) for board moves from *square*."""
    moves = generate_moves_for_square(square, fen)
    return [m["to"].rsplit("-", 1)[0] for m in moves]


def _get_san_moves(fen):
    """Return list of SAN strings for all moves in *fen*."""
    return [m["san"] for m in generate_all_moves(fen)]


# ---------------------------------------------------------------------------
# Oracle golden playout tests (all four modes, plus the betray corpus)
# ---------------------------------------------------------------------------

#: The four full-game playouts.  Each entry of their "games" arrays is a
#: contiguous playout: ply i's fen_after is ply i+1's fen_before.
_FULL_GAME_FILES = [
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
]

#: playouts_betray.json (KOM-101 G2) is a FILTERED corpus: its games hold only
#: the plies where a 返 move was legal, so consecutive plies are unrelated
#: positions and must never be chained.  Every test below reads only a single
#: ply's own fields (fen_before / legal / chosen / fen_after), which is what
#: makes it safe to include here; nothing in this file chains plies.
#: conftest.load_golden skips when the file is absent.
_GOLDEN_FILES = [
    *_FULL_GAME_FILES,
    "playouts_betray.json",
]


class TestGoldenPlayouts:
    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_legal_moves_match_golden(self, golden_file):
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                expected_legal = ply["legal"]
                generated = generate_all_moves(fen)
                actual_legal = sorted(m["san"] for m in generated)
                assert actual_legal == expected_legal, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"fen={fen}\n"
                    f"extra={sorted(set(actual_legal) - set(expected_legal))}\n"
                    f"missing={sorted(set(expected_legal) - set(actual_legal))}"
                )

    @pytest.mark.parametrize("golden_file", _GOLDEN_FILES)
    def test_make_move_after_fen(self, golden_file):
        data = load_golden(golden_file)
        for gi, game in enumerate(data["games"]):
            for pi, ply in enumerate(game):
                fen = ply["fen_before"]
                chosen_san = ply["chosen"]
                expected_after = ply["fen_after"]
                generated = generate_all_moves(fen)
                found = [m for m in generated if m["san"] == chosen_san]
                assert len(found) >= 1, (
                    f"{golden_file} game={gi} ply={pi}: "
                    f"chosen SAN '{chosen_san}' not found in generated moves"
                )
                actual_after = found[0]["after"]
                assert actual_after == expected_after, (
                    f"{golden_file} game={gi} ply={pi}\n"
                    f"chosen={chosen_san}\n"
                    f"expected_after={expected_after}\n"
                    f"actual_after={actual_after}"
                )


# ---------------------------------------------------------------------------
# Targeted tests (ported from rules-update.test.ts)
# ---------------------------------------------------------------------------


class TestDraftCompletionPriority:
    def test_ends_draft_immediately_when_black_chooses_shu(self):
        fen = (
            "9/9/9/9/9/9/9/9/4M4"
            " G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 b 3 wb 1"
        )
        san_moves = _get_san_moves(fen)
        black_done = [m for m in san_moves if m.startswith("新帥") and m.endswith("終")]
        assert len(black_done) > 0, "Expected black 終 move"

        # Play the first black 終 move
        chosen_san = black_done[0]
        move = [m for m in generate_all_moves(fen) if m["san"] == chosen_san][0]
        after = parse_fen(move["after"])
        assert after["drafting"] == {"b": False, "w": False}
        assert after["turn"] == "w"
        # white marshal at 9-5 should have moves
        assert len(generate_moves_for_square("9-5", move["after"])) > 0

    def test_keeps_draft_active_when_white_chooses_shu(self):
        fen = "4m4/9/9/9/9/9/9/9/4M4 D1/d2 w 3 wb 1"
        san_moves = _get_san_moves(fen)
        white_done = [m for m in san_moves if m.endswith("終")]
        assert len(white_done) > 0, "Expected white 終 move"

        chosen_san = white_done[0]
        move = [m for m in generate_all_moves(fen) if m["san"] == chosen_san][0]
        after = parse_fen(move["after"])
        # Draft still active: black still drafting
        assert after["drafting"] == {"b": True, "w": False}
        assert after["turn"] == "b"
        # Black should have non-終 moves
        black_sans = _get_san_moves(move["after"])
        assert any(not m.endswith("終") for m in black_sans)


class TestArcherWingBlocking:
    def test_blocks_diagonal_when_wing_has_higher_tower(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": piece["archer"], "color": "w"}])
        _place_tower(board, "4-6", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["warrior"], "color": "b"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board)
        targets = _get_targets(fen, "5-5")
        assert "3-4" in targets
        assert "3-6" not in targets

    def test_keeps_diagonal_legal_when_wing_not_higher(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": piece["archer"], "color": "w"}])
        _place_tower(board, "4-6", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board)
        targets = _get_targets(fen, "5-5")
        assert "3-6" in targets

    def test_black_archer_wing_blocking(self):
        """Black mirror: black archer at 5-5, higher tower at 6-4 blocks SE diagonal."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": piece["archer"], "color": "b"}])
        _place_tower(board, "6-4", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["warrior"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _get_targets(fen, "5-5")
        # For black, dirs are flipped. dirs[0]=[-1,1] becomes [1,-1] → toward 7-4.
        # Wing at (6,4) has tier 2 > archer tier 1 → direction toward 7-4 is blocked.
        # dirs[2]=[-1,-1] becomes [1,1] → toward 7-6. Wing at (6,6) is empty → open.
        assert "7-4" not in targets   # blocked by wing at 6-4 (tier 2 > tier 1)
        assert "7-6" in targets       # no wing blocker at 6-6

    def test_black_archer_wing_not_higher_legal(self):
        """Black mirror: equal-height wing does NOT block."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [{"type": piece["archer"], "color": "b"}])
        _place_tower(board, "6-4", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _get_targets(fen, "5-5")
        assert "7-4" in targets  # not blocked: wing at 6-4 has tier 1 = archer tier 1


class TestForwardOnlyLeap:
    def test_cannon_leaps_forward_not_sideways_backward(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["warrior"], "color": "w"},
            {"type": piece["cannon"], "color": "w"},
        ])
        _place_tower(board, "2-5", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "5-6", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "6-5", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board)
        targets = _get_targets(fen, "5-5")
        assert "1-5" in targets       # can leap forward past 2-5
        assert "5-7" not in targets   # sideways: blocked at 5-6, no leap
        assert "7-5" not in targets   # backward: blocked at 6-5, no leap

    def test_archer_leaps_forward_not_backward(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["warrior"], "color": "w"},
            {"type": piece["archer"], "color": "w"},
        ])
        _place_tower(board, "3-5", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "6-5", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board)
        targets = _get_targets(fen, "5-5")
        assert "1-5" in targets       # forward past 3-5
        assert "7-5" not in targets   # backward blocked

    def test_musketeer_leaps_forward_not_backward_diagonals(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["warrior"], "color": "w"},
            {"type": piece["musketeer"], "color": "w"},
        ])
        _place_tower(board, "3-5", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "6-6", [{"type": piece["soldier"], "color": "b"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board)
        targets = _get_targets(fen, "5-5")
        assert "1-5" in targets       # forward
        assert "7-7" not in targets   # backward diagonal blocked

    def test_black_cannon_leaps_forward_not_backward(self):
        """Black mirror: forward for black is increasing rank (dy > 0)."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["warrior"], "color": "b"},
            {"type": piece["cannon"], "color": "b"},
        ])
        _place_tower(board, "8-5", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "4-5", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _get_targets(fen, "5-5")
        assert "9-5" in targets       # forward for black (increasing rank)
        assert "3-5" not in targets   # backward for black (decreasing rank)

    def test_black_archer_leaps_forward_not_backward(self):
        """Black mirror: black archer forward = increasing rank."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["warrior"], "color": "b"},
            {"type": piece["archer"], "color": "b"},
        ])
        _place_tower(board, "7-5", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "4-5", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _get_targets(fen, "5-5")
        assert "9-5" in targets       # forward for black
        assert "3-5" not in targets   # backward for black

    def test_black_musketeer_leaps_forward_not_backward(self):
        """Black mirror: black musketeer forward = increasing rank."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["warrior"], "color": "b"},
            {"type": piece["musketeer"], "color": "b"},
        ])
        _place_tower(board, "7-5", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "4-4", [{"type": piece["soldier"], "color": "w"}])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, turn="b")
        targets = _get_targets(fen, "5-5")
        assert "9-5" in targets       # forward for black
        assert "3-3" not in targets   # backward diagonal blocked


class TestTacticianArataBetrayal:
    def test_white_tactician_arata_betrayal(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["warrior"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
                {"type": piece["soldier"], "color": "w", "count": 1},
            ],
        )
        san_moves = _get_san_moves(fen)
        assert "新謀(5-5-3)返兵" in san_moves

        # Play the betray move and verify tower state
        move = [m for m in generate_all_moves(fen) if m["san"] == "新謀(5-5-3)返兵"][0]
        after = parse_fen(move["after"])
        tower = after["board"][4][4]  # 5-5 → rank-1=4, 9-file=4
        assert [p["type"] for p in tower] == ["兵", "侍", "謀"]
        assert all(p["color"] == "w" for p in tower)

    def test_black_tactician_arata_betrayal(self):
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["warrior"], "color": "b"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            turn="b",
            hand=[
                {"type": piece["tactician"], "color": "b", "count": 1},
                {"type": piece["soldier"], "color": "b", "count": 1},
            ],
        )
        san_moves = _get_san_moves(fen)
        assert "新謀(5-5-3)返兵" in san_moves

        move = [m for m in generate_all_moves(fen) if m["san"] == "新謀(5-5-3)返兵"][0]
        after = parse_fen(move["after"])
        tower = after["board"][4][4]
        assert [p["type"] for p in tower] == ["兵", "侍", "謀"]
        assert all(p["color"] == "b" for p in tower)


class TestTacticianOnboardBetrayal:
    def test_tactician_tsuke_betrayal(self):
        fen = "4m4/9/9/3|s:D|5/4|f:w:T|4/9/9/9/4M4 D1S1T1/- w 3 - 1"
        san_moves = _get_san_moves(fen)
        assert "謀(5-5-3)(4-6-3)付" in san_moves
        assert "謀(5-5-3)(4-6-3)返忍" in san_moves

        move = [m for m in generate_all_moves(fen) if m["san"] == "謀(5-5-3)(4-6-3)返忍"][0]
        after = parse_fen(move["after"])
        # 4-6 → rank-1=3, 9-file=3
        tower = after["board"][3][3]
        assert [p["type"] for p in tower] == ["忍", "兵", "謀"]
        assert all(p["color"] == "w" for p in tower)


# ---------------------------------------------------------------------------
# Betrayal / convert corner-case tests
# ---------------------------------------------------------------------------


class TestBetrayalPowersetOrder:
    """Test 1: ON-BOARD multi-piece betrayal powerset + order."""

    def test_white_tactician_powerset_two_enemies(self):
        """White tactician stacks onto tower with 2 affordable enemy pieces.

        Tower at 5-5: spy(b, t1), soldier(b, t2) — both enemies, top at tier 2.
        In advanced mode max_tier=3, so tier 2 < 3 → tsuke allowed.
        Tactician at 4-5 (tier 1) reaches 5-5 via dirs[6]=[1,0].
        White hand: spy×1, soldier×1 → both types affordable.
        Enemies in tower order: [spy, soldier].
        Expected combos (pre-order): [spy], [spy,soldier], [soldier].
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["spy"], "color": "w", "count": 1},
                {"type": piece["soldier"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        betray_sans = [m["san"] for m in betray_moves]

        # Enemies are [spy, soldier] in tower order.
        # Powerset pre-order: [spy], [spy, soldier], [soldier]
        assert "謀(4-5-2)(5-5-3)返忍" in betray_sans
        assert "謀(4-5-2)(5-5-3)返忍兵" in betray_sans
        assert "謀(4-5-2)(5-5-3)返兵" in betray_sans
        assert len(betray_sans) == 3

        # Verify ORDER matches recursive pre-order
        combo_types = []
        for m in betray_moves:
            combo_types.append([p["type"] for p in m["captured"]])
        assert combo_types == [["忍"], ["忍", "兵"], ["兵"]]

    def test_black_tactician_powerset_two_enemies(self):
        """Black tactician stacks onto tower with 2 affordable enemy pieces.

        Tower at 5-5: spy(w, t1), soldier(w, t2) — both enemies.
        Black tactician at 6-5 (tier 1) — for black, dirs are flipped:
        dirs[6]=[1,0] → flipped to [-1,0] → (5-5).
        Black hand: spy×1, soldier×1 → both affordable.
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "w"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "6-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["tactician"], "color": "b"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            turn="b",
            hand=[
                {"type": piece["spy"], "color": "b", "count": 1},
                {"type": piece["soldier"], "color": "b", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        betray_sans = [m["san"] for m in betray_moves]

        assert "謀(6-5-2)(5-5-3)返忍" in betray_sans
        assert "謀(6-5-2)(5-5-3)返忍兵" in betray_sans
        assert "謀(6-5-2)(5-5-3)返兵" in betray_sans
        assert len(betray_sans) == 3

        combo_types = []
        for m in betray_moves:
            combo_types.append([p["type"] for p in m["captured"]])
        assert combo_types == [["忍"], ["忍", "兵"], ["兵"]]

    def test_white_tactician_converts_both_enemies(self):
        """Verify after-FEN: betraying both enemies converts them to white."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["spy"], "color": "w", "count": 1},
                {"type": piece["soldier"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        both_move = [m for m in moves if m["san"] == "謀(4-5-2)(5-5-3)返忍兵"][0]
        after = parse_fen(both_move["after"])
        tower = after["board"][4][4]  # 5-5 → rank-1=4, 9-file=4
        assert [p["type"] for p in tower] == ["忍", "兵", "謀"]
        assert all(p["color"] == "w" for p in tower)


class TestArataBetrayalPowerset:
    """Test 2: ARATA (drop-from-hand) multi-piece betrayal."""

    def test_white_arata_betrayal_powerset(self):
        """White tactician dropped from hand onto a mixed tower.

        Tower at 5-5: spy(b, t1), soldier(w, t2) — top is friendly soldier.
        Arata tactician lands at tier 3.
        White hand: tactician×1, spy×1.
        Enemy in tower: [spy]. Only 1 combo: [spy].
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
                {"type": piece["spy"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        arata_betray = [
            m for m in moves
            if m["type"] == "arata" and m.get("captured") and len(m["captured"]) > 0
            and m["to"].startswith("5-5")
        ]

        assert len(arata_betray) == 1
        assert arata_betray[0]["san"] == "新謀(5-5-3)返忍"

    def test_black_arata_betrayal_two_enemies(self):
        """Black tactician dropped from hand onto tower with 2 enemy pieces.

        Tower at 5-5: spy(w, t1), soldier(w, t2), warrior(b, t3) — nope, max tier 3 in advanced.
        Actually top is at tier 3, can't stack more. Use tier 2 max.
        Tower at 5-5: spy(w, t1), warrior(b, t2). Top is friendly warrior at tier 2.
        Tactician lands at tier 3 (advanced mode).
        Black hand: tactician×1, spy×1.
        Enemy: [spy]. Combo: [spy].
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "w"},
            {"type": piece["warrior"], "color": "b"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            turn="b",
            hand=[
                {"type": piece["tactician"], "color": "b", "count": 1},
                {"type": piece["spy"], "color": "b", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        arata_betray = [
            m for m in moves
            if m["type"] == "arata" and m.get("captured") and len(m["captured"]) > 0
            and m["to"].startswith("5-5")
        ]

        assert len(arata_betray) == 1
        assert arata_betray[0]["san"] == "新謀(5-5-3)返忍"

        # Verify conversion in after-FEN
        after = parse_fen(arata_betray[0]["after"])
        tower = after["board"][4][4]
        assert [p["type"] for p in tower] == ["忍", "侍", "謀"]
        assert all(p["color"] == "b" for p in tower)

    def test_arata_betrayal_converts_correctly(self):
        """Verify after-FEN: arata betray converts enemy and places tactician."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
                {"type": piece["spy"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        arata_betray = [
            m for m in moves
            if m["type"] == "arata" and m.get("captured") and len(m["captured"]) > 0
            and m["to"].startswith("5-5")
        ]

        assert len(arata_betray) == 1
        move = arata_betray[0]
        after = parse_fen(move["after"])
        tower = after["board"][4][4]
        # spy converted to white, soldier already white, tactician on top
        assert [p["type"] for p in tower] == ["忍", "兵", "謀"]
        assert all(p["color"] == "w" for p in tower)


class TestAffordabilityBoundary:
    """Test 3: AFFORDABILITY boundary — same-type enemy count vs hand count."""

    def test_two_same_type_enemies_hand_count_1_excluded(self):
        """Tower with 2 enemy soldiers; white hand has only 1 soldier.

        Tower at 5-5: soldier(b, t1), soldier(b, t2). Top at tier 2 < max_tier(3).
        Enemies = [soldier, soldier]. Same type, count=2.
        Hand has soldier×1 → 1 < 2 → type excluded → NO betrayal combos.
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["soldier"], "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["soldier"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        assert len(betray_moves) == 0

    def test_two_same_type_enemies_hand_count_2_both_offered(self):
        """Tower with 2 enemy soldiers; white hand has 2 soldiers.

        Both soldiers qualify → 3 combos: [s1], [s1,s2], [s2].
        """
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["soldier"], "color": "b"},
            {"type": piece["soldier"], "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["soldier"], "color": "w", "count": 2},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]

        # 2 items → 3 non-empty subsets: [s1], [s1,s2], [s2]
        assert len(betray_moves) == 3

        combo_types = [
            [p["type"] for p in m["captured"]] for m in betray_moves
        ]
        assert combo_types == [["兵"], ["兵", "兵"], ["兵"]]

        # SAN should reflect single or double soldier betrayal
        betray_sans = [m["san"] for m in betray_moves]
        assert "謀(4-5-2)(5-5-3)返兵" in betray_sans
        assert "謀(4-5-2)(5-5-3)返兵兵" in betray_sans

    def test_no_hand_pieces_no_betrayal(self):
        """Tower has enemy but hand is empty → no betrayal offered."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "b"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(board, hand=[])

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        assert len(betray_moves) == 0


class TestBetrayalSAN:
    """Test 4: SAN markers for betrayal — 返 suppresses 付."""

    def test_betray_san_has_kaeshi_no_tsuke(self):
        """A betray move landing above tier 1 shows 返 but NOT 付."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["spy"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        assert len(betray_moves) == 1

        san = betray_moves[0]["san"]
        # Should have 返 marker
        assert "返" in san
        # Should have the betrayed piece type
        assert "返忍" in san
        # Should NOT have 付 (betray suppresses tsuke marker)
        assert "付" not in san
        # The landing tier is 3 (above tier 1), so without suppression 付 would appear
        assert san == "謀(4-5-2)(5-5-3)返忍"

    def test_tsuke_without_betray_has_tsuke_marker(self):
        """Control: a plain tsuke (no betray) on the same tower DOES show 付."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["spy"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        tsuke_moves = [
            m for m in moves
            if m["type"] == "tsuke" and m["to"].startswith("5-5")
        ]
        assert len(tsuke_moves) == 1
        assert "付" in tsuke_moves[0]["san"]
        assert "返" not in tsuke_moves[0]["san"]

    def test_arata_betray_san_has_kaeshi(self):
        """Arata betray SAN: 新 prefix + 返 marker, no 付."""
        board = _create_empty_board()
        _place_tower(board, "5-5", [
            {"type": piece["spy"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
                {"type": piece["spy"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        arata_betray = [
            m for m in moves
            if m["type"] == "arata" and m.get("captured") and len(m["captured"]) > 0
            and m["to"].startswith("5-5")
        ]
        assert len(arata_betray) == 1
        san = arata_betray[0]["san"]
        assert san == "新謀(5-5-3)返忍"
        assert "付" not in san


class TestGameOverViaBetray:
    """Test 5: Betray/convert whose captured set includes a MARSHAL → game over '#'.

    NOTE: In normal play, the tsuke gate requires top.type != MARSHAL, so a
    marshal cannot be the top of a stack that a tactician tsuke's onto. However,
    a marshal CAN be below the top in a tower. If a marshal is an enemy piece in
    a tower and is "betrayed" (converted), the game_over logic in _make_move checks
    if any captured piece is a MARSHAL and appends '#'.

    But wait — the _make_move game_over check fires for betray moves too (line 536).
    The betrayed pieces in `captured` are converted (color flipped), not removed.
    For capture moves, the marshal is removed → game over makes sense.
    For betray moves, the marshal is converted → it's still on the board but now
    belongs to the mover. The code still appends '#' because it just checks
    `any(p["type"] == MARSHAL for p in move["captured"])`.

    Let's pin that behavior: if a marshal is in the betrayal combo, SAN gets '#'.
    """

    def test_betray_marshal_appends_game_over(self):
        """Betray that converts a marshal produces '#' in SAN."""
        board = _create_empty_board()
        # Tower at 5-5: marshal(b, t1), soldier(w, t2 top).
        # Tactician at 4-5 tsuke's to tier 3.
        # Top of target is soldier(w) at tier 2 — not marshal, tier < 3 → tsuke OK.
        # Enemies in tower: marshal(b). Hand needs marshal(w)×1.
        _place_tower(board, "5-5", [
            {"type": piece["marshal"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])
        _place_tower(board, "4-5", [
            {"type": piece["soldier"], "color": "w"},
            {"type": piece["tactician"], "color": "w"},
        ])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["marshal"], "color": "w", "count": 1},
            ],
        )

        moves = generate_all_moves(fen)
        betray_moves = [m for m in moves if m["type"] == "betray"]
        assert len(betray_moves) == 1

        san = betray_moves[0]["san"]
        assert san.endswith("#"), f"Expected '#' game-over suffix, got: {san}"
        assert "返帥" in san


class TestDraftBetrayalLastPiece:
    """Test 6: DRAFT interaction — betrayal-eligible arata with LAST hand piece.

    When drafting and the current hand count is 1 (last piece), only the 終
    (draft-finish) variants are generated; no continue-draft variant.
    """

    def test_last_piece_draft_betray_tactician_enemy(self):
        """Last hand piece (tactician) dropped onto tower containing enemy tactician.

        Hand = [tactician(w)×1]. Tower at 8-5 has [tactician(b,t1), soldier(w,t2)].
        enemy = tactician(b). Hand has tactician(w)×1 ≥ 1 → affordable.
        player_hand_count = 1 → is_last_piece.
        Only 終 variants should appear (no continue-draft).
        """
        board = _create_empty_board()
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])
        _place_tower(board, "8-5", [
            {"type": piece["tactician"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
            ],
            drafting={"b": False, "w": True},
        )

        moves = generate_all_moves(fen)
        # Filter to moves targeting 8-5
        moves_85 = [m for m in moves if m["to"].startswith("8-5")]

        # All moves for this square should be 終 variants (last piece)
        for m in moves_85:
            assert m["san"].endswith("終"), (
                f"Expected 終 suffix for last-piece draft, got: {m['san']}"
            )

        # There should be betray variants with 終
        betray_85 = [
            m for m in moves_85
            if m.get("captured") and len(m["captured"]) > 0
        ]
        assert len(betray_85) > 0, "Expected betrayal variants for last-piece draft"

        # The betray SAN should be: 新謀(8-5-3)返謀終
        betray_sans = [m["san"] for m in betray_85]
        assert "新謀(8-5-3)返謀終" in betray_sans

    def test_non_last_piece_draft_betray_has_both_variants(self):
        """Non-last piece: both 終 and non-終 betray variants exist."""
        board = _create_empty_board()
        _place_tower(board, "1-1", [{"type": piece["marshal"], "color": "b"}])
        _place_tower(board, "9-9", [{"type": piece["marshal"], "color": "w"}])
        _place_tower(board, "8-5", [
            {"type": piece["tactician"], "color": "b"},
            {"type": piece["soldier"], "color": "w"},
        ])

        fen = _make_fen(
            board,
            hand=[
                {"type": piece["tactician"], "color": "w", "count": 1},
                {"type": piece["soldier"], "color": "w", "count": 1},
            ],
            drafting={"b": False, "w": True},
        )

        moves = generate_all_moves(fen)
        moves_85 = [m for m in moves if m["to"].startswith("8-5")]

        betray_85 = [
            m for m in moves_85
            if m.get("captured") and len(m["captured"]) > 0
        ]
        betray_sans = [m["san"] for m in betray_85]

        # Should have both 終 and non-終 variants
        non_shu = [s for s in betray_sans if not s.endswith("終")]
        shu = [s for s in betray_sans if s.endswith("終")]
        assert len(non_shu) > 0, "Expected non-終 betray variant for non-last piece"
        assert len(shu) > 0, "Expected 終 betray variant for non-last piece"
