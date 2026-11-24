"""Tests for the Cython typed board and FEN boundary conversion."""

import numpy as np

from conftest import load_golden

from komugi.engine.cyboard import (
    Board,
    PIECE_INDEX,
    PIECE_TYPES,
    board_from_fen,
    board_to_fen,
    make_code,
    split_code,
)
from komugi.engine.api import Gungi
from komugi.engine.utils import (
    MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
    WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
    CANNON, ARCHER, MUSKETEER, TACTICIAN,
)


# ── 1. FEN round-trip differential ──────────────────────────────────────────

class TestFenRoundtripDifferential:
    """board_to_fen(board_from_fen(fen)) must equal the pure-Python canonical FEN."""

    def test_fen_roundtrip_goldens(self):
        entries = load_golden("fen_roundtrip.json")
        for entry in entries:
            inp = entry["input"]
            canonical = entry["canonical"]
            oracle_fen = Gungi(inp).fen()
            assert oracle_fen == canonical, (
                f"Oracle sanity: Gungi(inp).fen() != canonical for {inp!r}"
            )
            cy_fen = board_to_fen(board_from_fen(inp))
            assert cy_fen == canonical, (
                f"Cyboard round-trip mismatch:\n"
                f"  input:     {inp!r}\n"
                f"  expected:  {canonical!r}\n"
                f"  got:       {cy_fen!r}"
            )

    def test_playouts_intro(self):
        _check_playout_fens("playouts_intro.json")

    def test_playouts_beginner(self):
        _check_playout_fens("playouts_beginner.json")

    def test_playouts_intermediate(self):
        _check_playout_fens("playouts_intermediate.json")

    def test_playouts_advanced(self):
        _check_playout_fens("playouts_advanced.json")


def _check_playout_fens(golden_name):
    data = load_golden(golden_name)
    for game in data["games"]:
        for ply in game:
            fen = ply["fen_before"]
            oracle_fen = Gungi(fen).fen()
            cy_fen = board_to_fen(board_from_fen(fen))
            assert cy_fen == oracle_fen, (
                f"Playout FEN mismatch ({golden_name}):\n"
                f"  fen_before: {fen!r}\n"
                f"  oracle:     {oracle_fen!r}\n"
                f"  cyboard:    {cy_fen!r}"
            )


# ── 2. Structural tests ────────────────────────────────────────────────────

class TestStructural:
    """Verify cells, top(), height(), and code split for hand-built boards."""

    def test_single_piece(self):
        b = Board()
        code = make_code(0, 0)  # marshal, white
        b.cells_arr[0, 0, 0] = code
        assert b.top(0, 0) == code
        assert b.height(0, 0) == 1
        assert b.cells_arr[0, 0, 1] == 0
        assert b.cells_arr[0, 0, 2] == 0

    def test_tower_height_2(self):
        b = Board()
        c1 = make_code(9, 0)  # soldier, white
        c2 = make_code(4, 1)  # warrior, black
        b.cells_arr[3, 5, 0] = c1
        b.cells_arr[3, 5, 1] = c2
        assert b.height(3, 5) == 2
        assert b.top(3, 5) == c2
        assert b.cells_arr[3, 5, 0] == c1

    def test_tower_height_3(self):
        b = Board()
        c1 = make_code(8, 0)   # fortress, white
        c2 = make_code(9, 1)   # soldier, black
        c3 = make_code(0, 0)   # marshal, white
        b.cells_arr[8, 8, 0] = c1
        b.cells_arr[8, 8, 1] = c2
        b.cells_arr[8, 8, 2] = c3
        assert b.height(8, 8) == 3
        assert b.top(8, 8) == c3

    def test_empty_square(self):
        b = Board()
        assert b.top(4, 4) == 0
        assert b.height(4, 4) == 0

    def test_code_split_roundtrip_all_28(self):
        for type_idx in range(14):
            for color in range(2):
                code = make_code(type_idx, color)
                assert 1 <= code <= 28
                ti, col = split_code(code)
                assert ti == type_idx, f"type_index mismatch for code {code}"
                assert col == color, f"color mismatch for code {code}"

    def test_code_uniqueness(self):
        codes = set()
        for type_idx in range(14):
            for color in range(2):
                code = make_code(type_idx, color)
                assert code not in codes, f"duplicate code {code}"
                codes.add(code)
        assert len(codes) == 28


# ── 3. Copy independence ───────────────────────────────────────────────────

class TestCopyIndependence:
    """b2 = b.copy(); mutate b2 → b is unchanged."""

    def test_cells_independent(self):
        b = board_from_fen(
            "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3"
            " J2N2R2D1/j2n2r2d1 w 0 - 1"
        )
        b2 = b.copy()
        orig_val = int(b.cells_arr[0, 3, 0])
        b2.cells_arr[0, 3, 0] = 0
        assert b.cells_arr[0, 3, 0] == orig_val

    def test_hand_independent(self):
        b = board_from_fen(
            "9/9/9/9/9/9/9/9/9"
            " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/"
            "m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
        )
        b2 = b.copy()
        orig_val = int(b.hand_arr[0, 0])
        b2.hand_arr[0, 0] = 99
        assert b.hand_arr[0, 0] == orig_val

    def test_scalars_independent(self):
        b = Board()
        b.turn = 0
        b.mode = 1
        b.draft_w = 0
        b.draft_b = 0
        b.move_number = 5
        b2 = b.copy()
        b2.turn = 1
        b2.mode = 3
        b2.draft_w = 1
        b2.draft_b = 1
        b2.move_number = 99
        assert b.turn == 0
        assert b.mode == 1
        assert b.draft_w == 0
        assert b.draft_b == 0
        assert b.move_number == 5


# ── 4. PIECE_INDEX consistency ──────────────────────────────────────────────

class TestPieceIndexConsistency:
    """The 14-piece ordering must match gungi.utils and observation.PIECE_TYPES."""

    def test_matches_gungi_utils_order(self):
        """PIECE_TYPES kanji symbols are exactly the gungi.utils constants in order."""
        expected = [
            MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
            WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
            CANNON, ARCHER, MUSKETEER, TACTICIAN,
        ]
        assert list(PIECE_TYPES) == expected

    def test_matches_observation_piece_index(self):
        from komugi.engine.observation import (
            PIECE_INDEX as OBS_PIECE_INDEX,
            PIECE_TYPES as OBS_PIECE_TYPES,
        )
        assert list(PIECE_TYPES) == list(OBS_PIECE_TYPES)
        assert PIECE_INDEX == OBS_PIECE_INDEX

    def test_14_pieces(self):
        assert len(PIECE_TYPES) == 14
        assert len(PIECE_INDEX) == 14
