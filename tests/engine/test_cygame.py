"""Differential tests: Cython-backed Game vs pure-Python oracle.

The real correctness gate — every assertion compares the fast Cython path
against the pure-Python gungi.api.Gungi engine.
"""

import numpy as np
import pytest

from conftest import load_golden

from komugi.engine.game import Game
from komugi.engine import action_encoding, observation
from komugi.engine.cyboard import board_from_fen, board_to_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine import cyengine
from komugi.engine.api import Gungi


# ---------------------------------------------------------------------------
# Golden playout files
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

def _collect_golden_fens():
    """Collect every 2nd ply's fen_before across all four golden playouts."""
    fens = []
    for golden_file in _GOLDEN_FILES:
        data = load_golden(golden_file)
        games = data["games"]
        for game in games:
            for pi, ply in enumerate(game):
                fens.append(ply["fen_before"])
    return fens


# Pre-collect once at module load.
_GOLDEN_FENS = _collect_golden_fens()


def _pure_python_action_ids(fen):
    """Get canonical action ids from the pure-Python engine at *fen*."""
    engine = Gungi(fen)
    return action_encoding.legal_action_ids(engine)


def _cython_action_ids(fen):
    """Get canonical action ids from the Cython engine at *fen*."""
    board = board_from_fen(fen)
    return action_encoding.legal_cython_action_ids(board)


def _pure_python_observation(fen):
    """Get observation from the pure-Python engine at *fen*."""
    engine = Gungi(fen)
    return observation.observation(engine)


def _cython_observation(fen):
    """Get observation from the Cython board at *fen*."""
    board = board_from_fen(fen)
    return observation.observation_from_cyboard(board)


# ===========================================================================
# (1) VALID-MOVES DIFFERENTIAL — golden FEN sample
# ===========================================================================

class TestValidMovesDifferential:
    """Cython Game valid_moves must match pure-Python oracle at golden FENs."""

    def test_golden_sample_size(self):
        """Sanity: we have at least 1500 golden FENs."""
        assert len(_GOLDEN_FENS) >= 1500, (
            f"Expected >= 1500 golden FENs, got {len(_GOLDEN_FENS)}"
        )

    def test_valid_moves_match_oracle(self):
        """For every sampled golden FEN, cython action ids == pure-Python ids."""
        mismatches = []
        for i, fen in enumerate(_GOLDEN_FENS):
            cy_ids = _cython_action_ids(fen)
            py_ids = _pure_python_action_ids(fen)
            if cy_ids != py_ids:
                mismatches.append((i, fen, cy_ids - py_ids, py_ids - cy_ids))
                if len(mismatches) >= 5:
                    break
        assert not mismatches, (
            f"{len(mismatches)} mismatch(es); first: fen={mismatches[0][1]!r}, "
            f"cy_only={mismatches[0][2]}, py_only={mismatches[0][3]}"
        )


# ===========================================================================
# (2) OBSERVATION DIFFERENTIAL — golden FEN sample
# ===========================================================================

class TestObservationDifferential:
    """Cython observation must match pure-Python observation at golden FENs."""

    def test_observation_match_oracle(self):
        """For every sampled golden FEN, cython obs == pure-Python obs."""
        mismatches = []
        for i, fen in enumerate(_GOLDEN_FENS):
            cy_obs = _cython_observation(fen)
            py_obs = _pure_python_observation(fen)
            if not np.array_equal(cy_obs, py_obs):
                diff_planes = np.where(
                    np.any(cy_obs != py_obs, axis=(1, 2))
                )[0].tolist()
                mismatches.append((i, fen, diff_planes))
                if len(mismatches) >= 5:
                    break
        assert not mismatches, (
            f"{len(mismatches)} mismatch(es); first: fen={mismatches[0][1]!r}, "
            f"differing planes={mismatches[0][2]}"
        )


# ===========================================================================
# (3) FULL-GAME CONSISTENCY RUNS
# ===========================================================================

class TestFullGameConsistency:
    """Seeded full-game runs exercising the Cython Game end-to-end."""

    def test_seeded_games_no_crash(self):
        """Play 20 seeded random games to completion — no exceptions."""
        rng = np.random.RandomState(7777)
        for game_idx in range(20):
            g = Game()
            while True:
                ws = g.win_state()
                if ws.any():
                    assert ws.sum() == 1
                    break

                vm = g.valid_moves()
                assert vm.sum() > 0, (
                    f"Game {game_idx}: no valid moves but not terminal "
                    f"at turn {g.turns}"
                )

                legal = np.nonzero(vm)[0]
                action = int(rng.choice(legal))
                g.play_action(action)

                # Player tracks board turn
                assert g.player == g._board.turn, (
                    f"Game {game_idx}: player {g.player} != board.turn "
                    f"{g._board.turn} at turn {g.turns}"
                )

                assert g.turns <= Game.max_turns()

    def test_crafted_terminal_checkmate(self):
        """win_state fires correctly for a checkmate position."""
        fen = (
            "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
            " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
        )
        g = Game(board_from_fen(fen))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[0] == 1  # white wins

    def test_crafted_terminal_stalemate(self):
        """win_state fires correctly for a stalemate position."""
        fen = "8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1"
        g = Game(board_from_fen(fen))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1  # draw

    def test_crafted_terminal_marshal_captured(self):
        """win_state fires correctly when a marshal is captured."""
        fen = (
            "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6"
            "/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164"
        )
        g = Game(board_from_fen(fen))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[0] == 1  # white wins (black to move, marshal captured)

    def test_crafted_terminal_insufficient(self):
        """win_state fires correctly for insufficient material."""
        fen = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"
        g = Game(board_from_fen(fen))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1  # draw

    def test_observation_shape_during_play(self):
        """Observation shape stays (128,9,9) throughout a game."""
        rng = np.random.RandomState(42)
        g = Game()
        for _ in range(50):
            obs = g.observation()
            assert obs.shape == (128, 9, 9)
            assert obs.dtype == np.float32

            ws = g.win_state()
            if ws.any():
                break

            vm = g.valid_moves()
            if vm.sum() == 0:
                break
            action = int(rng.choice(np.nonzero(vm)[0]))
            g.play_action(action)
