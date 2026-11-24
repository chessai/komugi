"""Tests for komugi.engine.action_encoding."""

import numpy as np
import pytest

from conftest import load_golden
from komugi.engine import compat, cymovegen
from komugi.engine.api import Gungi
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import encode_fen
from komugi.engine.utils import (
    SQUARES,
    MARSHAL,
    GENERAL,
    LIEUTENANT_GENERAL,
    MAJOR_GENERAL,
    WARRIOR,
    LANCER,
    RIDER,
    SPY,
    FORTRESS,
    SOLDIER,
    CANNON,
    ARCHER,
    MUSKETEER,
    TACTICIAN,
)
from komugi.engine.action_encoding import (
    ACTION_SIZE,
    A0_START,
    A1_START,
    B_START,
    C_START,
    PIECE_INDEX,
    PIECE_TYPES,
    sq_index,
    sq_from_index,
    encode_move,
    decode_action,
    legal_action_ids,
    qualifying_mask,
)


# =========================================================================
# (1) Unit tests for constants and helpers
# =========================================================================


class TestConstants:
    def test_action_size(self):
        assert ACTION_SIZE == 37341
        assert ACTION_SIZE == 461 * 81  # 461 planes over a 9x9 destination grid

    def test_block_starts(self):
        assert A0_START == 0
        assert A1_START == 6561
        assert B_START == 32805
        assert C_START == 35073

    def test_block_sizes(self):
        # Planes per block: 81 from-squares, 81*4 (from, sel) pairs,
        # 14*2 (piece, bsel) pairs twice — 461 in total.
        assert A1_START - A0_START == 81 * 81           # 6561
        assert B_START - A1_START == 81 * 4 * 81        # 26244
        assert C_START - B_START == 14 * 2 * 81         # 2268
        assert ACTION_SIZE - C_START == 14 * 2 * 81     # 2268

    def test_block_starts_are_the_running_plane_sums(self):
        assert A0_START == 0
        assert A1_START == 81 * 81
        assert B_START == (81 + 81 * 4) * 81
        assert C_START == (81 + 81 * 4 + 14 * 2) * 81
        assert ACTION_SIZE == (81 + 81 * 4 + 14 * 2 + 14 * 2) * 81

    def test_piece_index_count(self):
        assert len(PIECE_INDEX) == 14
        assert set(PIECE_INDEX.values()) == set(range(14))

    def test_piece_index_all_types(self):
        all_kanji = [
            MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
            WARRIOR, LANCER, RIDER, SPY,
            FORTRESS, SOLDIER, CANNON, ARCHER,
            MUSKETEER, TACTICIAN,
        ]
        for k in all_kanji:
            assert k in PIECE_INDEX

    def test_sq_index_range(self):
        for i, sq in enumerate(SQUARES):
            assert sq_index(sq) == i

    def test_sq_index_strips_tier(self):
        """sq_index must handle rank-file-tier strings by stripping tier."""
        assert sq_index("6-1-1") == sq_index("6-1")
        assert sq_index("9-4-2") == sq_index("9-4")
        assert sq_index("1-9-3") == sq_index("1-9")

    def test_sq_from_index_roundtrip(self):
        for i in range(81):
            assert sq_index(sq_from_index(i)) == i

    def test_spot_indices(self):
        """Spot-check a few (from, sel, to) / (piece, bsel, sq) -> exact index."""
        # sq_index for "1-9" should be 0, "1-1" should be 8
        assert sq_index("1-9") == 0
        assert sq_index("1-1") == 8
        assert sq_index("9-1") == 80

        # A0: from=1-9(0), to=1-1(8) => 0*81+8 = 8
        assert A0_START + 0 * 81 + 8 == 8
        # A1: from=1-9(0), sel=0, to=1-1(8) => 6561 + (0*4+0)*81 + 8 = 6569
        assert A1_START + (0 * 4 + 0) * 81 + 8 == 6569
        # A1: the same square pair with sel=3 sits three planes further on
        assert A1_START + (0 * 4 + 3) * 81 + 8 == 6569 + 3 * 81

        # B: piece_idx=0 (marshal), bsel=0, to=1-9(0) => 32805
        assert B_START + (0 * 2 + 0) * 81 + 0 == 32805
        # C: piece_idx=0 (marshal), bsel=0, to=1-9(0) => 35073
        assert C_START + (0 * 2 + 0) * 81 + 0 == 35073

    def test_worked_example_from_the_design(self):
        """docs/design/betrayal-encoding.md, "Worked example".

        from_sq = 40, to_sq = 31, Q = 0b01: the tsuke and the single legal
        conversion are 81 apart, and the two illegal selectors are the ids
        directly above them.
        """
        from_sq, to_sq = 40, 31
        assert A1_START + (from_sq * 4 + 0) * 81 + to_sq == 19552
        assert A1_START + (from_sq * 4 + 1) * 81 + to_sq == 19633
        assert A1_START + (from_sq * 4 + 2) * 81 + to_sq == 19714
        assert A1_START + (from_sq * 4 + 3) * 81 + to_sq == 19795


# =========================================================================
# (2) Round-trip over all golden playouts
# =========================================================================


#: The four full-game playouts.  Each entry of their "games" arrays is a
#: contiguous playout: ply i's fen_after is ply i+1's fen_before.
FULL_GAME_FILES = [
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
GOLDEN_FILES = [
    *FULL_GAME_FILES,
    "playouts_betray.json",
]


class TestGoldenRoundTrip:
    @pytest.mark.parametrize("golden_file", GOLDEN_FILES)
    def test_roundtrip_encode_decode(self, golden_file):
        """For every ply in every golden game: encode chosen move, verify block,
        decode back, apply, and assert fen_after matches."""
        data = load_golden(golden_file)

        for gi, game_plies in enumerate(data["games"]):
            for pi, ply in enumerate(game_plies):
                fen_before = ply["fen_before"]
                chosen_san = ply["chosen"]
                expected_after = ply["fen_after"]

                g = Gungi(fen_before)
                verbose_moves = g.moves({"verbose": True})

                # Find the chosen move by SAN
                san_normalized = chosen_san.rstrip("=#")
                found = [
                    m for m in verbose_moves
                    if m["san"] == san_normalized or m["san"] == chosen_san
                ]
                assert found, (
                    f"golden {golden_file} game={gi} ply={pi}: "
                    f"chosen SAN '{chosen_san}' not in legal moves"
                )
                chosen_move = found[0]

                # Encode
                action_id = encode_move(chosen_move)
                assert 0 <= action_id < ACTION_SIZE, (
                    f"action_id {action_id} out of range"
                )

                # Verify block membership
                mtype = chosen_move["type"]
                if mtype == "arata":
                    if chosen_move.get("draftFinished"):
                        assert C_START <= action_id < ACTION_SIZE, (
                            f"arata+終 should be in block C"
                        )
                    else:
                        assert B_START <= action_id < C_START, (
                            f"arata should be in block B"
                        )
                elif mtype in ("route", "capture"):
                    assert A0_START <= action_id < A1_START, (
                        f"{mtype} should be in block A0"
                    )
                elif mtype in ("tsuke", "betray"):
                    assert A1_START <= action_id < B_START, (
                        f"{mtype} should be in block A1"
                    )

                # Decode back from a fresh game at same position
                g2 = Gungi(fen_before)
                decoded = decode_action(action_id, g2)
                assert decoded is not None

                # Apply decoded move and check fen_after
                g3 = Gungi(fen_before)
                g3.move(decoded["san"])
                assert g3.fen() == expected_after, (
                    f"golden {golden_file} game={gi} ply={pi}: "
                    f"fen mismatch after decode+apply\n"
                    f"chosen={chosen_san}\n"
                    f"decoded_san={decoded['san']}\n"
                    f"expected={expected_after}\n"
                    f"got={g3.fen()}"
                )


# =========================================================================
# (3) Bijection / mask test on golden FENs
# =========================================================================


class TestBijectionMask:
    @pytest.mark.parametrize("golden_file", GOLDEN_FILES)
    def test_legal_action_ids_match(self, golden_file):
        """On a sample of FENs, legal_action_ids == canonical encode set,
        and every id decodes to a legal move."""
        data = load_golden(golden_file)

        # Sample: first ply of each game + every 20th ply
        fens_seen = set()
        for game_plies in data["games"]:
            for pi, ply in enumerate(game_plies):
                if pi == 0 or pi % 20 == 0:
                    fen = ply["fen_before"]
                    if fen in fens_seen:
                        continue
                    fens_seen.add(fen)

                    g = Gungi(fen)
                    verbose = g.moves({"verbose": True})
                    encoded_set = {encode_move(m) for m in verbose}
                    legal_ids = legal_action_ids(g)

                    assert encoded_set == legal_ids

                    # Every id decodes to a legal move
                    for aid in legal_ids:
                        g2 = Gungi(fen)
                        decoded = decode_action(aid, g2)
                        assert decoded["san"] in [
                            m["san"] for m in verbose
                        ], f"decoded san not in legal moves: {decoded['san']}"


# =========================================================================
# (4) A0/A1 split test — capture vs tsuke at same (from_sq, to_sq)
# =========================================================================


class TestA0A1Split:
    def test_capture_vs_tsuke_different_ids(self):
        """Hand-build a position where a capture and a tsuke can occur at the
        same (from_sq, to_sq) pair. They must encode to different blocks."""
        # We need: a square with an enemy-topped tower that is also stackable.
        # Strategy: use a golden position and find such a pair, OR construct
        # a minimal FEN.
        #
        # Walk intro golden games looking for a ply where a piece has both
        # a route/capture and a tsuke to the same destination square.
        data = load_golden("playouts_intro.json")
        found_split = False

        for game_plies in data["games"]:
            for ply in game_plies:
                fen = ply["fen_before"]
                g = Gungi(fen)
                verbose = g.moves({"verbose": True})

                # Group board moves by (from_sq, to_sq)
                by_sq_pair: dict[tuple[int, int], list[dict]] = {}
                for m in verbose:
                    if m["type"] in ("route", "capture", "tsuke", "betray"):
                        key = (sq_index(m["from"]), sq_index(m["to"]))
                        by_sq_pair.setdefault(key, []).append(m)

                for key, moves in by_sq_pair.items():
                    a0_moves = [m for m in moves if m["type"] in ("route", "capture")]
                    a1_moves = [m for m in moves if m["type"] in ("tsuke", "betray")]
                    if a0_moves and a1_moves:
                        # Found! Verify different encoding
                        a0_id = encode_move(a0_moves[0])
                        a1_id = encode_move(a1_moves[0])
                        assert a0_id != a1_id
                        assert A0_START <= a0_id < A1_START
                        assert A1_START <= a1_id < B_START

                        # Decode back
                        g2 = Gungi(fen)
                        d0 = decode_action(a0_id, g2)
                        assert d0["type"] in ("route", "capture")

                        g3 = Gungi(fen)
                        d1 = decode_action(a1_id, g3)
                        assert d1["type"] in ("tsuke", "betray")

                        found_split = True
                        break
                if found_split:
                    break
            if found_split:
                break

        # If intro doesn't have it, try beginner
        if not found_split:
            for golden_file in GOLDEN_FILES[1:]:
                data = load_golden(golden_file)
                for game_plies in data["games"]:
                    for ply in game_plies:
                        fen = ply["fen_before"]
                        g = Gungi(fen)
                        verbose = g.moves({"verbose": True})

                        by_sq_pair: dict[tuple[int, int], list[dict]] = {}
                        for m in verbose:
                            if m["type"] in ("route", "capture", "tsuke", "betray"):
                                key = (sq_index(m["from"]), sq_index(m["to"]))
                                by_sq_pair.setdefault(key, []).append(m)

                        for key, moves in by_sq_pair.items():
                            a0_moves = [
                                m for m in moves if m["type"] in ("route", "capture")
                            ]
                            a1_moves = [
                                m for m in moves if m["type"] in ("tsuke", "betray")
                            ]
                            if a0_moves and a1_moves:
                                a0_id = encode_move(a0_moves[0])
                                a1_id = encode_move(a1_moves[0])
                                assert a0_id != a1_id
                                assert A0_START <= a0_id < A1_START
                                assert A1_START <= a1_id < B_START
                                found_split = True
                                break
                        if found_split:
                            break
                    if found_split:
                        break
                if found_split:
                    break

        assert found_split, (
            "Could not find a position with capture+tsuke at same (from,to) — "
            "test is inconclusive"
        )


# =========================================================================
# (5) The conversion selector — the inverses of the v1 canonicalization
#
# v1 collapsed every betrayal at one square pair (and the tsuke that declines
# it) onto a single id and decoded it to the MAXIMAL conversion.  Each test
# below is the inverse of the v1 test it replaces: the family now encodes
# distinctly, the plain tsuke is reachable alongside the betrays, an illegal
# selector raises, and arata conversions — which had no id at all in v1 —
# round-trip.  Positions are hand-built: these are exact selector lattices,
# pinned regardless of what playouts_betray.json happens to contain.
# =========================================================================


def _empty_board():
    return [[[None] for _ in range(9)] for _ in range(9)]


def _place(board, square, pieces):
    rank, file_ = (int(part) for part in square.split("-"))
    board[rank - 1][9 - file_] = [
        {"square": square, "tier": i + 1, "type": p["type"], "color": p["color"]}
        for i, p in enumerate(pieces)
    ]


def _fen(board, hand=None, turn="w", mode="advanced"):
    return encode_fen({
        "board": board,
        "hand": hand or [],
        "turn": turn,
        "mode": mode,
        "drafting": {"b": False, "w": False},
        "moveNumber": 1,
    })


def _tactician_fen(tower, hand):
    """White 謀 on 4-5 (tier 2) facing *tower* on 5-5, with *hand* in hand.

    The tactician must stand at tier 2 to reach a height-2 tower at all, so it
    sits on a friendly 兵; 5-5 is due south, one of 謀's three directions.
    """
    board = _empty_board()
    _place(board, "5-5", tower)
    _place(board, "4-5", [
        {"type": SOLDIER, "color": "w"},
        {"type": TACTICIAN, "color": "w"},
    ])
    _place(board, "1-1", [{"type": MARSHAL, "color": "b"}])
    _place(board, "9-9", [{"type": MARSHAL, "color": "w"}])
    return _fen(board, hand=hand)


#: Two enemies of two DIFFERENT types, only one of which the hand can pay for
#: — the design's worked example.  Q = 0b01.
MIXED_PAIR_FEN = _tactician_fen(
    tower=[
        {"type": SOLDIER, "color": "b"},   # tier 1 — affordable
        {"type": WARRIOR, "color": "b"},   # tier 2 — no 侍 in hand
    ],
    hand=[{"type": SOLDIER, "color": "w", "count": 1}],
)

#: Two enemies, both affordable — the full 4-element selector lattice.
POWERSET_FEN = _tactician_fen(
    tower=[
        {"type": SPY, "color": "b"},
        {"type": SOLDIER, "color": "b"},
    ],
    hand=[
        {"type": SPY, "color": "w", "count": 1},
        {"type": SOLDIER, "color": "w", "count": 1},
    ],
)

#: Two enemy 兵 against a single 兵 in hand: the over-strict hand test of
#: gungi.js disqualifies the type outright, singletons included.  Q = 0.
UNAFFORDABLE_PAIR_FEN = _tactician_fen(
    tower=[
        {"type": SOLDIER, "color": "b"},
        {"type": SOLDIER, "color": "b"},
    ],
    hand=[{"type": SOLDIER, "color": "w", "count": 1}],
)

_FROM_SQ = sq_index("4-5")
_TO_SQ = sq_index("5-5")


def _a1_id(sel):
    return A1_START + (_FROM_SQ * 4 + sel) * 81 + _TO_SQ


def _stack_family(game):
    """Every tsuke/betray from 4-5 to 5-5 in *game*."""
    return [
        m for m in game.moves({"verbose": True})
        if m["type"] in ("tsuke", "betray")
        and sq_index(m["from"]) == _FROM_SQ
        and sq_index(m["to"]) == _TO_SQ
    ]


def _tier_mask(move):
    return sum(1 << (p["tier"] - 1) for p in (move.get("captured") or []))


class TestConversionSelector:
    def test_betray_family_encodes_distinctly(self):
        """The v1 collapse, inverted: one id per combination, all four planes."""
        g = Gungi(POWERSET_FEN)
        family = _stack_family(g)
        # tsuke + {t1}, {t1,t2}, {t2}
        assert len(family) == 4
        assert sorted(_tier_mask(m) for m in family) == [0, 1, 2, 3]

        ids = {encode_move(m) for m in family}
        assert len(ids) == 4, f"betray combinations collapsed onto {ids}"
        assert ids == {_a1_id(sel) for sel in range(4)}
        for action_id in ids:
            assert A1_START <= action_id < B_START

    def test_every_selector_decodes_to_its_own_combination(self):
        """Decode is exact — no rounding up to the maximal conversion."""
        g = Gungi(POWERSET_FEN)
        for move in _stack_family(g):
            decoded = decode_action(encode_move(move), g)
            assert decoded["type"] == move["type"]
            assert _tier_mask(decoded) == _tier_mask(move)
            assert decoded["san"] == move["san"]

    def test_plain_tsuke_is_reachable_alongside_betrays(self):
        """Selector 0 means "convert nothing" even when betrayals exist."""
        g = Gungi(POWERSET_FEN)
        assert any(m["type"] == "betray" for m in _stack_family(g))

        decoded = decode_action(_a1_id(0), g)
        assert decoded["type"] == "tsuke"
        assert not decoded.get("captured")
        # ... and the betray ids still decode to betrays.
        for sel in (1, 2, 3):
            assert decode_action(_a1_id(sel), g)["type"] == "betray"

    def test_non_qualifying_selectors_are_rejected(self):
        """The worked example: sel 1 is legal, sel 2 and 3 are not."""
        g = Gungi(MIXED_PAIR_FEN)
        family = _stack_family(g)
        assert {_tier_mask(m) for m in family} == {0, 1}
        assert {encode_move(m) for m in family} == {_a1_id(0), _a1_id(1)}

        assert decode_action(_a1_id(1), g)["type"] == "betray"
        for sel in (2, 3):
            with pytest.raises(ValueError):
                decode_action(_a1_id(sel), g)

    def test_unaffordable_duplicate_type_leaves_only_the_tsuke(self):
        """Over-strict hand test: neither 兵 may be flipped, not even alone."""
        g = Gungi(UNAFFORDABLE_PAIR_FEN)
        family = _stack_family(g)
        assert [m["type"] for m in family] == ["tsuke"]
        assert encode_move(family[0]) == _a1_id(0)
        for sel in (1, 2, 3):
            with pytest.raises(ValueError):
                decode_action(_a1_id(sel), g)

    def test_arata_conversion_is_reachable(self):
        """v1 had NO id for an arata betrayal; here it is an ordinary action.

        White drops its 謀 onto its own 兵 at 5-5, flipping the black 忍
        underneath.  The plain drop and the converting drop differ in the
        block-B selector bit and in nothing else.
        """
        board = _empty_board()
        _place(board, "5-5", [
            {"type": SPY, "color": "b"},        # tier 1 — the conversion
            {"type": SOLDIER, "color": "w"},    # tier 2 — friendly top
        ])
        _place(board, "1-1", [{"type": MARSHAL, "color": "b"}])
        _place(board, "9-9", [{"type": MARSHAL, "color": "w"}])
        fen = _fen(board, hand=[
            {"type": TACTICIAN, "color": "w", "count": 1},
            {"type": SPY, "color": "w", "count": 1},
        ])

        g = Gungi(fen)
        drops = [
            m for m in g.moves({"verbose": True})
            if m["type"] == "arata"
            and m["piece"] == TACTICIAN
            and sq_index(m["to"]) == _TO_SQ
        ]
        assert len(drops) == 2, [m["san"] for m in drops]

        plain = [m for m in drops if not m.get("captured")][0]
        converting = [m for m in drops if m.get("captured")][0]
        assert _tier_mask(converting) == 0b01, "only tier 1 is ever convertible"

        piece_idx = PIECE_INDEX[TACTICIAN]
        assert encode_move(plain) == B_START + (piece_idx * 2 + 0) * 81 + _TO_SQ
        assert encode_move(converting) == (
            B_START + (piece_idx * 2 + 1) * 81 + _TO_SQ
        )

        assert decode_action(encode_move(plain), g)["san"] == plain["san"]
        assert (
            decode_action(encode_move(converting), g)["san"] == converting["san"]
        )


# =========================================================================
# (6) Wrong-encoding detection
# =========================================================================


class TestWrongEncodingDetection:
    def test_wrong_block_for_route(self):
        """A route move must NOT be in block A1."""
        data = load_golden("playouts_intro.json")
        ply = data["games"][0][0]
        g = Gungi(ply["fen_before"])
        verbose = g.moves({"verbose": True})
        routes = [m for m in verbose if m["type"] == "route"]
        if routes:
            aid = encode_move(routes[0])
            assert A0_START <= aid < A1_START  # must be A0, not A1

    def test_wrong_block_for_arata(self):
        """An arata without draftFinished must NOT be in block C."""
        data = load_golden("playouts_intro.json")
        # Find a draft ply with arata moves
        for game_plies in data["games"]:
            for ply in game_plies:
                g = Gungi(ply["fen_before"])
                verbose = g.moves({"verbose": True})
                aratas = [
                    m for m in verbose
                    if m["type"] == "arata" and not m.get("draftFinished")
                ]
                if aratas:
                    aid = encode_move(aratas[0])
                    assert B_START <= aid < C_START
                    return


# =========================================================================
# (7) The v1 -> v2 lift (komugi.engine.compat.rho) — the conversion is
#     TACTICIAN-ONLY.
#
# v1's decoder returned a conversion only where the rules engine had
# generated one, and only 謀 ever converts: betray records exist for a 謀
# mover, arata conversions for a 謀 drop.  Keying the lift on the target
# tower's qualifying mask alone (the shape rho shipped with) invents ids the
# legal mask never sets.  Each test below is red without its fix.
# =========================================================================


def _cy_board(fen):
    return board_from_fen(fen)


def _legal_mask(fen):
    """The native legal-action mask at *fen* — the ground truth rho must hit."""
    mask = np.zeros(ACTION_SIZE, dtype=np.uint8)
    cymovegen.fill_legal_mask(_cy_board(fen), mask)
    return mask


#: White drops onto its own 侍 at 7-5, which stands on a black 兵; white's
#: hand holds one 兵, so the tier-1 enemy qualifies and Q = 0b01.  Only the
#: 謀 drop may actually flip it.
DROP_TOWER_FEN = "4m4/9/9/9/9/9/4|d:W|4/9/4M4 T1D1/- w 3 - 1"

def _warrior_tsuke_fen():
    """A white 侍 on 4-5 facing a lone black 兵 tower on 5-5.

    5-5 is the 侍's one-step S — the same reachable square ``_tactician_fen``
    uses — and white's hand holds the matching 兵, so Q = 0b01 at the target.
    The 侍 can tsuke onto the tower, but it can never betray.
    """
    board = _empty_board()
    _place(board, "5-5", [{"type": SOLDIER, "color": "b"}])
    _place(board, "4-5", [{"type": WARRIOR, "color": "w"}])
    _place(board, "1-1", [{"type": MARSHAL, "color": "b"}])
    _place(board, "9-9", [{"type": MARSHAL, "color": "w"}])
    return _fen(board, hand=[{"type": SOLDIER, "color": "w", "count": 1}])


WARRIOR_TSUKE_FEN = _warrior_tsuke_fen()

_DROP_TO_SQ = sq_index("7-5")


def _v1_drop_id(piece_kanji):
    """The v1 block-B id of a plain drop of *piece_kanji* onto 7-5.

    Old layout arithmetic: block B started at 13122 and had ONE plane per
    piece type (no selector bit), so the id is
    ``V1_B_START + piece_idx * 81 + to_sq``.
    """
    return compat.V1_B_START + PIECE_INDEX[piece_kanji] * 81 + _DROP_TO_SQ


def _b_id(piece_kanji, bsel):
    return B_START + (PIECE_INDEX[piece_kanji] * 2 + bsel) * 81 + _DROP_TO_SQ


class TestCompatLiftIsTacticianOnly:
    def test_the_drop_tower_really_qualifies(self):
        """Both drop tests below are only meaningful while Q != 0 at 7-5."""
        assert qualifying_mask(_cy_board(DROP_TOWER_FEN), 7, 5, 0) == 0b01

    def test_pawn_drop_lifts_without_the_conversion_bit(self):
        """Finding 1: a 兵 drop is not a conversion, whatever the tower holds.

        rho used to key on the target square alone and hand back a bsel = 1
        id that fill_legal_mask never sets — an id outside the action space
        the agent may choose from.
        """
        v1_id = _v1_drop_id(SOLDIER)
        assert v1_id == 13122 + 9 * 81 + _DROP_TO_SQ  # 兵 is piece index 9

        lifted = compat.rho(DROP_TOWER_FEN, v1_id)
        assert lifted == _b_id(SOLDIER, 0)

        mask = _legal_mask(DROP_TOWER_FEN)
        assert mask[lifted], "the plain 兵 drop must be a legal action"
        assert not mask[_b_id(SOLDIER, 1)], (
            "a 兵 drop has no converting variant — the old lift invented one"
        )

    def test_tactician_drop_still_lifts_to_the_conversion(self):
        """Control: the conversion is preserved where v1 really converted."""
        lifted = compat.rho(DROP_TOWER_FEN, _v1_drop_id(TACTICIAN))
        assert lifted == _b_id(TACTICIAN, 1)
        assert _legal_mask(DROP_TOWER_FEN)[lifted]

    def test_non_tactician_tsuke_lifts_with_selector_zero(self):
        """Finding 2: only 謀 betrays, so every other stacker lifts to sel 0."""
        assert qualifying_mask(_cy_board(WARRIOR_TSUKE_FEN), 5, 5, 0) == 0b01

        v1_id = compat.V1_A1_START + _FROM_SQ * 81 + _TO_SQ
        lifted = compat.rho(WARRIOR_TSUKE_FEN, v1_id)
        assert lifted == _a1_id(0)

        mask = _legal_mask(WARRIOR_TSUKE_FEN)
        assert mask[_a1_id(0)], "the 侍 tsuke must be a legal action"
        assert not mask[_a1_id(1)], (
            "a 侍 cannot betray — the old lift pointed at a bit nobody sets"
        )

    @pytest.mark.parametrize("fen", [DROP_TOWER_FEN, WARRIOR_TSUKE_FEN])
    def test_pi_of_rho_is_the_identity(self, fen):
        """Finding 4's guard: the lift never leaves its own plane group.

        Every v1-reachable id is the projection of a legal v2 id, so pi over
        the legal mask enumerates exactly the v1 ids rho has to answer for.
        """
        mask = _legal_mask(fen)
        v1_ids = {compat.pi(int(i)) for i in np.nonzero(mask)[0]}
        assert v1_ids

        for v1_id in sorted(v1_ids):
            lifted = compat.rho(fen, v1_id)
            assert 0 <= lifted < ACTION_SIZE
            assert compat.pi(lifted) == v1_id, (
                f"lift of v1 id {v1_id} landed on {lifted}, which projects "
                f"back to {compat.pi(lifted)}"
            )
