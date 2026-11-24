"""Tests for komugi.engine.observation — AlphaZero observation encoding."""

import numpy as np
import pytest

from komugi.engine.api import Gungi
from komugi.engine.utils import (
    MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
    WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
    CANNON, ARCHER, MUSKETEER, TACTICIAN,
    WHITE, BLACK,
)
from komugi.engine.observation import (
    observation,
    observation_size,
    PIECE_INDEX,
    MAX_COUNT,
    COLOR_INDEX,
    MAX_TURNS,
)


# ---------------------------------------------------------------------------
# Helper FENs
# ---------------------------------------------------------------------------

# Intro start position (post-draft, pieces on board + some in hand).
INTRO_FEN = (
    "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3"
    " J2N2R2D1/j2n2r2d1 w 0 - 1"
)

# Intermediate position (all pieces in hand, both sides drafting).
INTERMEDIATE_FEN = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
)

# A position with a known 3-piece stacked tower.
# Tower at rank=1, file=9: |n:G:S| (lancer-black-t1, General-white-t2, Spy-white-t3)
# Marshals at rank=1 file=5 (black) and rank=9 file=5 (white).
STACKED_FEN = (
    "|n:G:S|3m4/9/9/9/9/9/9/9/4M4"
    " -/- w 0 - 5"
)


# =========================================================================
# 1. SHAPE / DTYPE
# =========================================================================

class TestShapeDtype:
    """observation(game).shape == observation_size() == (128,9,9) and dtype==float32."""

    @pytest.mark.parametrize("fen", [INTRO_FEN, INTERMEDIATE_FEN, STACKED_FEN])
    def test_shape_and_dtype(self, fen: str):
        g = Gungi(fen)
        obs = observation(g)
        assert obs.shape == (128, 9, 9)
        assert obs.shape == observation_size()
        assert obs.dtype == np.float32

    def test_observation_size_returns_tuple(self):
        s = observation_size()
        assert isinstance(s, tuple)
        assert len(s) == 3
        assert s == (128, 9, 9)


# =========================================================================
# 2. OCCUPANCY correctness
# =========================================================================

class TestOccupancy:
    """Verify that stacked towers light up exactly the right (type,tier,color) planes."""

    def test_stacked_tower_planes(self):
        """A |n:G:S| tower at rank=1,file=9 should light exactly 3 occupancy planes."""
        g = Gungi(STACKED_FEN)
        obs = observation(g)

        # Board coordinates for rank=1, file=9: board[0][9-9] = board[0][0]
        r, c = 0, 0

        # Expected planes for the 3 pieces in the tower:
        # lancer  (pi=5, tier=1, black ci=1): plane = 5*6 + 0*2 + 1 = 31
        # general (pi=1, tier=2, white ci=0): plane = 1*6 + 1*2 + 0 = 8
        # spy     (pi=7, tier=3, white ci=0): plane = 7*6 + 2*2 + 0 = 46
        expected_planes = {
            31: ("lancer", 1, "black"),
            8:  ("general", 2, "white"),
            46: ("spy", 3, "white"),
        }

        for plane_idx in range(84):  # all occupancy planes
            val = obs[plane_idx, r, c]
            if plane_idx in expected_planes:
                assert val == 1.0, (
                    f"Plane {plane_idx} ({expected_planes[plane_idx]}) should be 1.0 "
                    f"at ({r},{c}), got {val}"
                )
            else:
                assert val == 0.0, (
                    f"Plane {plane_idx} should be 0.0 at ({r},{c}), got {val}"
                )

    def test_single_piece_on_tier1(self):
        """Marshal at rank=1 file=5 (black) should light exactly one occupancy plane."""
        g = Gungi(STACKED_FEN)
        obs = observation(g)

        # rank=1, file=5 → board[0][9-5] = board[0][4]
        r, c = 0, 4

        # marshal (pi=0, tier=1, black ci=1): plane = 0*6 + 0*2 + 1 = 1
        for plane_idx in range(84):
            val = obs[plane_idx, r, c]
            if plane_idx == 1:
                assert val == 1.0, f"Plane 1 (marshal,t1,black) should be 1.0"
            else:
                assert val == 0.0, f"Plane {plane_idx} should be 0.0, got {val}"

    def test_empty_square_all_zero(self):
        """An empty square should have all occupancy planes == 0."""
        g = Gungi(STACKED_FEN)
        obs = observation(g)

        # rank=5, file=5 → board[4][4] — guaranteed empty in STACKED_FEN
        r, c = 4, 4
        assert np.all(obs[:84, r, c] == 0.0)

    def test_no_row_aliasing(self):
        """Verify planes are independent (no shared numpy buffers)."""
        g = Gungi(STACKED_FEN)
        obs = observation(g)

        # Modify one plane and verify others are unaffected
        obs_copy = obs.copy()
        obs_copy[0, 0, 0] = 99.0
        # Original should be unchanged (obs is a separate array)
        assert obs[0, 0, 0] != 99.0 or obs[0, 0, 0] == obs_copy[0, 0, 0]

        # Check that different planes at the same position are independent
        # At (0,0), planes 8, 31, 46 are 1.0 and others are 0.0
        assert obs[8, 0, 0] == 1.0
        assert obs[31, 0, 0] == 1.0
        assert obs[46, 0, 0] == 1.0
        assert obs[0, 0, 0] == 0.0  # different plane, same position


# =========================================================================
# 3. HAND normalization
# =========================================================================

class TestHand:
    """Hand planes should contain count/MAX_COUNT in [0,1] on the right planes."""

    def test_intermediate_hand_full(self):
        """In the intermediate position, all pieces are in hand at max count."""
        g = Gungi(INTERMEDIATE_FEN)
        obs = observation(g)

        # Every piece type at max count for both colors → all hand planes = 1.0
        for pi in range(14):
            for ci in range(2):
                plane = 84 + pi * 2 + ci
                expected = 1.0  # all at max count
                actual = obs[plane, 0, 0]
                assert actual == pytest.approx(expected), (
                    f"Hand plane {plane} (pi={pi}, ci={ci}) expected {expected}, got {actual}"
                )
                # Constant plane — all cells should be the same
                assert np.all(obs[plane] == actual), (
                    f"Hand plane {plane} is not constant"
                )

    def test_intro_hand_partial(self):
        """In the intro position, only some pieces are in hand with known counts."""
        g = Gungi(INTRO_FEN)
        obs = observation(g)

        # White hand: J2N2R2D1 = major_general(2), lancer(2), rider(2), soldier(1)
        # MAX: major_general=2, lancer=3, rider=2, soldier=4
        expected_white = {
            MAJOR_GENERAL: 2 / 2,   # 1.0
            LANCER: 2 / 3,          # 0.667
            RIDER: 2 / 2,           # 1.0
            SOLDIER: 1 / 4,         # 0.25
        }
        for ptype, expected_val in expected_white.items():
            pi = PIECE_INDEX[ptype]
            plane = 84 + pi * 2 + 0  # white
            actual = obs[plane, 0, 0]
            assert actual == pytest.approx(expected_val, abs=1e-6), (
                f"White hand plane for {ptype} expected {expected_val}, got {actual}"
            )

        # Black hand: j2n2r2d1 — same distribution
        expected_black = {
            MAJOR_GENERAL: 2 / 2,
            LANCER: 2 / 3,
            RIDER: 2 / 2,
            SOLDIER: 1 / 4,
        }
        for ptype, expected_val in expected_black.items():
            pi = PIECE_INDEX[ptype]
            plane = 84 + pi * 2 + 1  # black
            actual = obs[plane, 0, 0]
            assert actual == pytest.approx(expected_val, abs=1e-6), (
                f"Black hand plane for {ptype} expected {expected_val}, got {actual}"
            )

        # Pieces NOT in hand should have plane value 0.0
        not_in_hand_white = [MARSHAL, GENERAL, LIEUTENANT_GENERAL, WARRIOR,
                             SPY, FORTRESS, CANNON, ARCHER, MUSKETEER, TACTICIAN]
        for ptype in not_in_hand_white:
            pi = PIECE_INDEX[ptype]
            plane = 84 + pi * 2 + 0
            assert obs[plane, 0, 0] == 0.0, (
                f"White hand plane for {ptype} should be 0.0"
            )

    def test_empty_hand_all_zero(self):
        """When hand is empty, all hand planes should be 0."""
        g = Gungi(STACKED_FEN)
        obs = observation(g)

        for plane in range(84, 112):
            assert np.all(obs[plane] == 0.0), (
                f"Hand plane {plane} should be all zeros with empty hand"
            )

    def test_hand_planes_are_constant(self):
        """Each hand plane should be constant across all 81 cells."""
        g = Gungi(INTRO_FEN)
        obs = observation(g)

        for plane in range(84, 112):
            vals = obs[plane]
            assert np.all(vals == vals[0, 0]), (
                f"Hand plane {plane} is not constant"
            )


# ---------------------------------------------------------------------------
# Repetition helpers — driving a genuine out-and-back shuttle
# ---------------------------------------------------------------------------


def _cells_key(g: Gungi) -> str:
    """The placement field of the FEN — what occurrences are counted over."""
    return g.fen().split(" ")[0]


def _square_of(coord: str) -> str:
    """``"rank-file-tier"`` -> ``"rank-file"``."""
    rank, file, _tier = coord.split("-")
    return f"{rank}-{file}"


def _reversible_moves(g: Gungi) -> list[dict]:
    """Plain tier-1 relocations: a lone piece stepping onto an empty square.

    Only these undo ply-for-ply — a capture is irreversible, and a tsuke or
    betray rearranges the tower it lands on.
    """
    return [
        m
        for m in g.moves({"verbose": True})
        if m["type"] == "route"
        and m["from"].endswith("-1")
        and m["to"].endswith("-1")
    ]


def _reverse_of(g: Gungi, mv: dict) -> dict | None:
    """The legal move in *g* that puts *mv*'s piece back where it came from."""
    for cand in _reversible_moves(g):
        if (
            _square_of(cand["from"]) == _square_of(mv["to"])
            and _square_of(cand["to"]) == _square_of(mv["from"])
        ):
            return cand
    return None


def _find_shuttle(fen: str) -> list[str]:
    """SANs for a 4-ply cycle that returns the board to *fen*'s placement.

    White steps out, black steps out, then both step back — the out-and-back
    both sides play to repeat a position, discovered from the position itself
    rather than hardcoded.
    """
    start_key = _cells_key(Gungi(fen))
    for w_out in _reversible_moves(Gungi(fen))[:8]:
        opened = Gungi(fen)
        opened.move(w_out["san"])
        for b_out in _reversible_moves(opened)[:8]:
            probe = Gungi(fen)
            probe.move(w_out["san"])
            probe.move(b_out["san"])
            w_back = _reverse_of(probe, w_out)
            if w_back is None:
                continue
            probe.move(w_back["san"])
            b_back = _reverse_of(probe, b_out)
            if b_back is None:
                continue
            probe.move(b_back["san"])
            if _cells_key(probe) == start_key:
                return [w_out["san"], b_out["san"], w_back["san"], b_back["san"]]
    raise AssertionError(f"no out-and-back shuttle available from {fen}")


# =========================================================================
# 4. AUX planes
# =========================================================================

class TestAuxPlanes:
    """Drafting (112, 113), side to move (114), turn clock (115), repetition.

    Post-KOM-027 the tail of the plane map reads:
      115  min(plies, MAX_TURNS) / MAX_TURNS — PLIES, the unit the turn-cap
           draw is adjudicated in, so the plane reaches 1.0 exactly at the
           drawing ply.
      116  R2: 1.0 iff the current position has occurred >= 2 times.
      117  R3: 1.0 iff the current position has occurred >= 3 times.
    v1's single scalar plane 116 (min(occurrences, 4) / 4) is gone.
    """

    def test_drafting_planes_both_active(self):
        """Intermediate position has both sides drafting."""
        g = Gungi(INTERMEDIATE_FEN)
        obs = observation(g)

        assert np.all(obs[112] == 1.0), "White drafting plane should be 1.0"
        assert np.all(obs[113] == 1.0), "Black drafting plane should be 1.0"

    def test_drafting_planes_none_active(self):
        """Intro position has no one drafting."""
        g = Gungi(INTRO_FEN)
        obs = observation(g)

        assert np.all(obs[112] == 0.0), "White drafting plane should be 0.0"
        assert np.all(obs[113] == 0.0), "Black drafting plane should be 0.0"

    def test_side_to_move_white(self):
        """White to move → plane 114 all 0.0."""
        g = Gungi(INTRO_FEN)  # "w" to move
        obs = observation(g)
        assert np.all(obs[114] == 0.0)

    def test_side_to_move_black(self):
        """Black to move → plane 114 all 1.0."""
        fen_black = (
            "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3"
            " J2N2R2D1/j2n2r2d1 b 0 - 2"
        )
        g = Gungi(fen_black)
        obs = observation(g)
        assert np.all(obs[114] == 1.0)

    def test_turn_fraction(self):
        """Turn-clock plane = min(plies, MAX_TURNS)/MAX_TURNS — plies, not moves."""
        # INTRO_FEN is move 1 with white to move: zero plies elapsed.
        g = Gungi(INTRO_FEN)
        assert np.all(observation(g)[115] == 0.0)

        # One ply played → exactly one step of 1/MAX_TURNS.
        g.move(g.moves()[0])
        assert np.all(observation(g)[115] == pytest.approx(1.0 / MAX_TURNS))

        # STACKED_FEN is move 5 with white to move — 8 plies elapsed, not 5.
        obs2 = observation(Gungi(STACKED_FEN))
        assert np.all(obs2[115] == pytest.approx(8.0 / MAX_TURNS))

    def test_turn_fraction_clamped(self):
        """Move number beyond MAX_TURNS should clamp to 1.0."""
        fen = (
            "4m4/9/9/9/9/9/9/9/4M4"
            " -/- w 0 - 999"
        )
        g = Gungi(fen)
        obs = observation(g)
        assert np.all(obs[115] == pytest.approx(1.0))

    def test_repetition_plane_fresh_game(self):
        """A fresh game is occurrence 1 — neither threshold is met."""
        g = Gungi(INTRO_FEN)
        obs = observation(g)
        assert np.all(obs[116] == 0.0), "R2 needs >= 2 occurrences"
        assert np.all(obs[117] == 0.0), "R3 needs >= 3 occurrences"

    def test_repetition_plane_increments(self):
        """R2 lights at the 2nd occurrence, R3 at the 3rd — monotone thresholds.

        Driven by a genuine out-and-back shuttle rather than injected history,
        so the planes are read off positions the engine actually reached.
        """
        shuttle = _find_shuttle(INTRO_FEN)
        g = Gungi(INTRO_FEN)
        start_key = _cells_key(g)

        # Occurrence 1 — the starting position, both thresholds clear.
        obs1 = observation(g)
        assert np.all(obs1[116] == 0.0)
        assert np.all(obs1[117] == 0.0)

        # Occurrence 2 — R2 flips, R3 stays clear.
        for san in shuttle:
            g.move(san)
        assert _cells_key(g) == start_key, "the shuttle must return the position"
        assert not g.is_game_over(), "two occurrences is short of the draw"
        obs2 = observation(g)
        assert np.all(obs2[116] == 1.0)
        assert np.all(obs2[117] == 0.0)

        # Occurrence 3 — R3 flips too, and the game is still live.
        for san in shuttle:
            g.move(san)
        assert _cells_key(g) == start_key
        assert not g.is_game_over(), "three occurrences is short of the draw"
        obs3 = observation(g)
        assert np.all(obs3[116] == 1.0)
        assert np.all(obs3[117] == 1.0)

    def test_aux_planes_are_constant(self):
        """The constant aux planes (112-117, 120-127) are flat across 81 cells.

        Planes 118/119 are EXCLUDED, and widening the loop to cover them would
        be a trap rather than a fix: they are the last-move planes, which are
        spatial by construction, and this is the ORACLE path, which emits zeros
        for 118-127 permanently (KOM-002 §8 gap 4).  A zero plane is constant,
        so ``range(112, 128)`` would pass here while proving nothing about the
        two planes it appears to cover.  What a genuine constancy check on
        118/119 would need is a ``Game`` after a move, which this oracle-path
        test cannot build.
        """
        g = Gungi(INTRO_FEN)
        obs = observation(g)
        for plane in [*range(112, 118), *range(120, 128)]:
            vals = obs[plane]
            assert np.all(vals == vals[0, 0]), (
                f"Aux plane {plane} is not constant"
            )


# =========================================================================
# 5. DETERMINISM and independence
# =========================================================================

class TestDeterminism:
    """observation is identical on repeated calls; arrays are independent."""

    @pytest.mark.parametrize("fen", [INTRO_FEN, INTERMEDIATE_FEN, STACKED_FEN])
    def test_repeated_calls_identical(self, fen: str):
        """Two calls on the same state produce identical arrays."""
        g = Gungi(fen)
        obs1 = observation(g)
        obs2 = observation(g)
        np.testing.assert_array_equal(obs1, obs2)

    @pytest.mark.parametrize("fen", [INTRO_FEN, INTERMEDIATE_FEN, STACKED_FEN])
    def test_independent_arrays(self, fen: str):
        """Returned arrays do not share buffers — mutating one doesn't affect the other."""
        g = Gungi(fen)
        obs1 = observation(g)
        obs2 = observation(g)

        # Mutate obs1
        obs1[0, 0, 0] = 999.0

        # obs2 should be unaffected
        assert obs2[0, 0, 0] != 999.0 or obs2[0, 0, 0] == 0.0

    def test_piece_index_consistency(self):
        """PIECE_INDEX maps all 14 piece types to 0..13 with no collisions."""
        assert len(PIECE_INDEX) == 14
        assert set(PIECE_INDEX.values()) == set(range(14))

        # Verify all piece constants are present
        all_pieces = [
            MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
            WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
            CANNON, ARCHER, MUSKETEER, TACTICIAN,
        ]
        for p in all_pieces:
            assert p in PIECE_INDEX

    def test_max_count_covers_all_types(self):
        """MAX_COUNT has an entry for all 14 piece types, all positive."""
        assert len(MAX_COUNT) == 14
        for ptype in PIECE_INDEX:
            assert ptype in MAX_COUNT
            assert MAX_COUNT[ptype] > 0
