"""Tests for the selectable start-mode feature of the AlphaZero Gungi Game."""

import numpy as np
import pytest

from komugi.engine.game import Game, _MODE_FENS
from komugi.engine.cyboard import board_from_fen, board_to_fen
from komugi.engine.fen import (
    INTRO_POSITION,
    BEGINNER_POSITION,
    INTERMEDIATE_POSITION,
    ADVANCED_POSITION,
)


_MODE_CONSTANTS = {
    "intro": INTRO_POSITION,
    "beginner": BEGINNER_POSITION,
    "intermediate": INTERMEDIATE_POSITION,
    "advanced": ADVANCED_POSITION,
}


@pytest.fixture(autouse=True)
def _restore_intro_mode():
    """Ensure every test leaves the class default back at intro."""
    yield
    Game.set_start_mode("intro")


# ---- (1) Each mode produces the correct FEN and has legal moves -----------

@pytest.mark.parametrize("mode", ["intro", "beginner", "intermediate", "advanced"])
def test_start_mode_fen_and_legal_moves(mode):
    Game.set_start_mode(mode)
    g = Game()
    actual_fen = board_to_fen(g._board)
    expected_fen = board_to_fen(board_from_fen(_MODE_CONSTANTS[mode]))
    assert actual_fen == expected_fen, f"FEN mismatch for mode {mode}"
    assert g.valid_moves().sum() > 0, f"No legal moves at start of {mode}"


# ---- (2) Advanced/intermediate start in draft phase ----------------------

@pytest.mark.parametrize("mode", ["advanced", "intermediate"])
def test_draft_phase_active(mode):
    Game.set_start_mode(mode)
    g = Game()
    assert g._board.draft_w or g._board.draft_b, (
        f"Expected draft phase active for {mode}"
    )
    # Legal moves at draft start should exist (placement/arata actions)
    assert g.valid_moves().sum() > 0


# ---- (3) Seeded random self-play in advanced mode -------------------------

def test_advanced_random_selfplay():
    Game.set_start_mode("advanced")
    g = Game()
    rng = np.random.RandomState(42)

    saw_draft = False
    for ply in range(40):
        if g._board.draft_w or g._board.draft_b:
            saw_draft = True

        ws = g.win_state()
        if ws.sum() > 0:
            break

        mask = g.valid_moves()
        assert mask.sum() > 0, f"No legal moves at ply {ply} but game not over"

        legal_actions = np.nonzero(mask)[0]
        action = rng.choice(legal_actions)

        expected_player = g._board.turn
        assert g._player == expected_player, (
            f"Player tracking mismatch at ply {ply}"
        )

        g.play_action(int(action))

    assert saw_draft, "Expected at least some plies in draft phase for advanced mode"


# ---- (4) Default restored to intro after tests ---------------------------

def test_default_restored_to_intro():
    # Set to a non-intro mode first
    Game.set_start_mode("advanced")
    # Autouse fixture will restore to intro after this test,
    # but we test the mechanism explicitly here:
    Game.set_start_mode("intro")
    g = Game()
    expected_fen = board_to_fen(board_from_fen(INTRO_POSITION))
    assert board_to_fen(g._board) == expected_fen
