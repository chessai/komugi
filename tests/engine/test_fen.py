"""Tests for gungi.fen — FEN parser, encoder, and validator."""

from conftest import load_golden
from komugi.engine.fen import (
    ADVANCED_POSITION,
    BEGINNER_POSITION,
    INTERMEDIATE_POSITION,
    INTRO_POSITION,
    encode_fen,
    parse_fen,
    validate_fen,
)
from komugi.engine.utils import piece


# ---------------------------------------------------------------------------
# Round-trip golden oracle (2600+ FENs from gungi.js@3.4.11)
# ---------------------------------------------------------------------------


class TestRoundTripGolden:
    def test_fen_roundtrip_golden(self):
        entries = load_golden("fen_roundtrip.json")
        for entry in entries:
            inp = entry["input"]
            canonical = entry["canonical"]
            assert encode_fen(parse_fen(inp)) == canonical, (
                f"Round-trip mismatch for input: {inp}"
            )


# ---------------------------------------------------------------------------
# parseFEN — start positions + structural checks
# ---------------------------------------------------------------------------


class TestParseFEN:
    def test_intro_position(self):
        result = parse_fen(INTRO_POSITION)
        assert result["turn"] == "w"
        assert result["mode"] == "intro"
        assert result["drafting"] == {"b": False, "w": False}
        assert result["moveNumber"] == 1
        assert len(result["board"]) == 9
        for rank in result["board"]:
            assert len(rank) == 9
        # marshal at 1-5 (black)
        assert result["board"][0][4][0] == {
            "color": "b",
            "square": "1-5",
            "tier": 1,
            "type": piece["marshal"],
        }
        # general at 9-6 (white)
        assert result["board"][8][3][0] == {
            "color": "w",
            "square": "9-6",
            "tier": 1,
            "type": piece["general"],
        }

    def test_beginner_position(self):
        result = parse_fen(BEGINNER_POSITION)
        assert result["turn"] == "w"
        assert result["mode"] == "beginner"
        assert result["drafting"] == {"b": False, "w": False}
        assert result["moveNumber"] == 1
        assert len(result["board"]) == 9
        # archer at 2-7 (black)
        assert result["board"][1][2][0] == {
            "color": "b",
            "square": "2-7",
            "tier": 1,
            "type": piece["archer"],
        }
        # marshal at 9-5 (white)
        assert result["board"][8][4][0] == {
            "color": "w",
            "square": "9-5",
            "tier": 1,
            "type": piece["marshal"],
        }

    def test_intermediate_position(self):
        result = parse_fen(INTERMEDIATE_POSITION)
        assert result["turn"] == "w"
        assert result["mode"] == "intermediate"
        assert result["drafting"] == {"b": True, "w": True}
        assert result["moveNumber"] == 1
        for rank in result["board"]:
            for square in rank:
                assert square[0] is None

    def test_advanced_position(self):
        result = parse_fen(ADVANCED_POSITION)
        assert result["turn"] == "w"
        assert result["mode"] == "advanced"
        assert result["drafting"] == {"b": True, "w": True}
        assert result["moveNumber"] == 1
        for rank in result["board"]:
            for square in rank:
                assert square[0] is None

    def test_towers_and_empty_squares(self):
        fen = "3img3/1sa|a:g:s|1s3/9/9/9/9/9/9/9 M1N3/d2 b 3 b 2"
        result = parse_fen(fen)
        assert result["turn"] == "b"
        assert result["mode"] == "advanced"
        assert result["drafting"] == {"b": True, "w": False}
        assert result["moveNumber"] == 2

        tower = result["board"][1][3]
        assert len(tower) == 3
        p = {"color": "b", "square": "2-6", "tier": 1, "type": piece["archer"]}
        assert tower[0] == p
        assert tower[1] == {**p, "type": piece["general"], "tier": 2}
        assert tower[2] == {**p, "type": piece["spy"], "tier": 3}
        assert result["board"][8][8][0] is None

    def test_adjacent_towers(self):
        fen = "3img3/1sa|a:g:s|d:w|k:n|r:w|2/9/9/9/9/9/9/9 M1N3/d2 w 1 w 3"
        result = parse_fen(fen)
        assert result["turn"] == "w"
        assert result["mode"] == "beginner"
        assert result["drafting"] == {"b": False, "w": True}
        assert result["moveNumber"] == 3

        # Tower 1: archer/general/spy at 2-6
        tower1 = result["board"][1][3]
        p1 = {"color": "b", "square": "2-6", "tier": 1, "type": piece["archer"]}
        assert len(tower1) == 3
        assert tower1[0] == p1
        assert tower1[1] == {**p1, "type": piece["general"], "tier": 2}
        assert tower1[2] == {**p1, "type": piece["spy"], "tier": 3}

        # Tower 2: soldier/warrior at 2-5
        tower2 = result["board"][1][4]
        p2 = {"color": "b", "square": "2-5", "tier": 1, "type": piece["soldier"]}
        assert len(tower2) == 2
        assert tower2[0] == p2
        assert tower2[1] == {**p2, "type": piece["warrior"], "tier": 2}

        # Tower 3: musketeer/lancer at 2-4
        tower3 = result["board"][1][5]
        p3 = {"color": "b", "square": "2-4", "tier": 1, "type": piece["musketeer"]}
        assert len(tower3) == 2
        assert tower3[0] == p3
        assert tower3[1] == {**p3, "type": piece["lancer"], "tier": 2}

        # Tower 4: rider/warrior at 2-3
        tower4 = result["board"][1][6]
        p4 = {"color": "b", "square": "2-3", "tier": 1, "type": piece["rider"]}
        assert len(tower4) == 2
        assert tower4[0] == p4
        assert tower4[1] == {**p4, "type": piece["warrior"], "tier": 2}

        # Empty corners
        assert result["board"][0][0][0] is None
        assert result["board"][8][8][0] is None

    def test_hand_pieces(self):
        fen = (
            "3img3/1sa1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        result = parse_fen(fen)
        assert len(result["hand"]) == 10
        assert result["hand"][0] == {
            "type": piece["major_general"],
            "color": "w",
            "count": 2,
        }
        assert result["hand"][1] == {
            "type": piece["lancer"],
            "color": "w",
            "count": 2,
        }
        assert result["hand"][5] == {
            "type": piece["major_general"],
            "color": "b",
            "count": 2,
        }
        assert result["hand"][9] == {
            "type": piece["soldier"],
            "color": "b",
            "count": 1,
        }

    def test_empty_hands(self):
        fen = "9/9/9/9/9/9/9/9/9 -/- w 0 - 1"
        result = parse_fen(fen)
        assert len(result["hand"]) == 0
        assert result["turn"] == "w"
        assert result["mode"] == "intro"
        assert result["drafting"] == {"b": False, "w": False}
        assert result["moveNumber"] == 1

    def test_white_empty_hand(self):
        fen = "3img3/9/9/9/9/9/9/9/9 -/m1r3 b 1 - 1"
        result = parse_fen(fen)
        assert result["turn"] == "b"
        assert result["mode"] == "beginner"
        assert result["drafting"] == {"b": False, "w": False}
        assert result["moveNumber"] == 1
        assert len(result["board"]) == 9
        for rank in result["board"]:
            assert len(rank) == 9
        assert len(result["hand"]) == 2
        assert {"color": "b", "type": piece["marshal"], "count": 1} in result["hand"]
        assert {"color": "b", "type": piece["rider"], "count": 3} in result["hand"]

    def test_black_empty_hand(self):
        fen = "3img3/9/9/9/9/9/9/9/9 M1D3/- w 1 - 1"
        result = parse_fen(fen)
        assert result["turn"] == "w"
        assert result["mode"] == "beginner"
        assert result["drafting"] == {"b": False, "w": False}
        assert result["moveNumber"] == 1
        assert len(result["board"]) == 9
        for rank in result["board"]:
            assert len(rank) == 9
        assert len(result["hand"]) == 2
        assert {"color": "w", "type": piece["marshal"], "count": 1} in result["hand"]
        assert {"color": "w", "type": piece["soldier"], "count": 3} in result["hand"]

    def test_roundtrip_beginner(self):
        fen = (
            "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
            " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
        )
        assert encode_fen(parse_fen(fen)) == fen


# ---------------------------------------------------------------------------
# validateFen — positive & negative cases (ported from fen.test.ts)
# ---------------------------------------------------------------------------


class TestValidateFen:
    def test_valid_fen(self):
        assert validate_fen(BEGINNER_POSITION)["ok"] is True

    def test_incorrect_field_count(self):
        fen = "3img3/1ra1n1as1/d1fwdwf1d w 1 -"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected 6 fields" in result["error"]

    def test_invalid_piece(self):
        fen = (
            "3img3/1ra1n1xas1/d1fwdwf1d/9/9/9/9/9/9"
            " J2N2R1D1/j2n2r2d1 w 1 - 1"
        )
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "invalid piece" in result["error"]

    def test_invalid_rank_count(self):
        fen = (
            "3img3/1ra1n1xas1/d1fwdwf1d/9/9/9/9/9"
            " J2N2R1D1/j2n2r2d1 w 1 - 1"
        )
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected 9 ranks" in result["error"]

    def test_invalid_file_count(self):
        fen = (
            "3img3/1ra1n1a1/d1fwdwf1d/9/9/9/9/9/9"
            " J2N2R1D1/j2n2r2d1 w 1 - 1"
        )
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected 9 squares" in result["error"]

    def test_invalid_tower_size(self):
        fen = (
            "3img3/1ra1n1as1/d1fw|d:f:g:c|wf1d/9/9/9/9/9/9"
            " J2N2R1D1/j2n2r2d1 w 1 - 1"
        )
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "tower size" in result["error"]

    def test_invalid_hand_pieces(self):
        fen = "9/9/9/9/9/9/9/9/9 J2X2R1D1/j2n2r2d1 w 1 - 1"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "invalid piece" in result["error"]

    def test_invalid_active_player(self):
        fen = "9/9/9/9/9/9/9/9/9 -/- x 0 - 1"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected 'w' or 'b'" in result["error"]

    def test_invalid_setup_mode(self):
        fen = "9/9/9/9/9/9/9/9/9 -/- w 4 - 1"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected '0', '1', '2', or '3'" in result["error"]

    def test_invalid_drafting(self):
        fen = "9/9/9/9/9/9/9/9/9 -/- w 0 x 1"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "(drafting availability) is invalid" in result["error"]

    def test_invalid_move_number(self):
        fen = "9/9/9/9/9/9/9/9/9 -/- w 0 - x"
        result = validate_fen(fen)
        assert result["ok"] is False
        assert "expected an integer" in result["error"]
