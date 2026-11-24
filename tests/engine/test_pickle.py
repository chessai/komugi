"""Tests for pickle support of Board and Game (fixes silently-dropped result_queue items)."""

import multiprocessing as mp
import pickle
from pickle import dumps, loads

import numpy as np
import pytest

from komugi.engine import action_encoding
from komugi.engine.cyboard import Board, board_from_fen, board_to_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.game import Game
from komugi.engine.fen import (
    INTRO_POSITION,
    BEGINNER_POSITION,
    INTERMEDIATE_POSITION,
    ADVANCED_POSITION,
)


# Generals shuttling back and forth: four plies return the board to the start
# position, so the start key's occurrence count is 2 — mid-cycle, short of the
# fourfold draw, which is where a counter-serialization bug would matter.
FOURFOLD_FEN = "4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1"
HALF_SHUTTLE = [
    "大(9-4-1)(8-4-1)",
    "大(1-4-1)(2-4-1)",
    "大(8-4-1)(9-4-1)",
    "大(2-4-1)(1-4-1)",
]


def _play_san(game: Game, san: str) -> None:
    """Play a SAN move through ``Game.play_action`` (which owns the counter)."""
    for m in generate_all_moves(game._board):
        if m["san"] == san:
            game.play_action(action_encoding.encode_cython_move(m))
            return
    raise ValueError(f"No move with SAN {san!r}")


# ── 1. Board pickle round-trip ────────────────────────────────────────────────


class TestBoardPickle:
    def test_fresh_board(self):
        b = board_from_fen(INTRO_POSITION)
        original_fen = board_to_fen(b)
        b2 = loads(dumps(b))
        assert board_to_fen(b2) == original_fen

    def test_mid_game_board(self):
        """Apply a few moves via the cython engine, then pickle round-trip."""
        from komugi.engine import cyengine, cymovegen
        from komugi.engine import action_encoding

        b = board_from_fen(INTRO_POSITION)
        # Play a few moves to get a mid-game state.
        for _ in range(4):
            mask = np.zeros(action_encoding.ACTION_SIZE, dtype=np.uint8)
            cymovegen.fill_legal_mask(b, mask)
            actions = np.nonzero(mask)[0]
            if len(actions) == 0:
                break
            action = int(actions[0])
            decoded = action_encoding.decode_cython_action(action, b)
            cyengine.apply_move(b, decoded)

        original_fen = board_to_fen(b)
        b2 = loads(dumps(b))
        assert board_to_fen(b2) == original_fen

    def test_all_start_modes(self):
        """All four start-mode boards survive pickle round-trip."""
        for fen in (INTRO_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION, ADVANCED_POSITION):
            b = board_from_fen(fen)
            original_fen = board_to_fen(b)
            b2 = loads(dumps(b))
            assert board_to_fen(b2) == original_fen

    def test_all_protocols(self):
        """Board pickles correctly across all pickle protocols."""
        b = board_from_fen(INTRO_POSITION)
        original_fen = board_to_fen(b)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            b2 = loads(dumps(b, protocol))
            assert board_to_fen(b2) == original_fen, f"Failed at protocol {protocol}"


# ── 2. Game pickle round-trip ─────────────────────────────────────────────────


class TestGamePickle:
    def _make_mid_game(self) -> Game:
        game = Game(board_from_fen(INTRO_POSITION))
        for _ in range(3):
            moves = game.valid_moves()
            actions = np.nonzero(moves)[0]
            if len(actions) == 0:
                break
            game.play_action(int(actions[0]))
        return game

    def test_game_round_trip(self):
        game = self._make_mid_game()
        original_fen = board_to_fen(game._board)
        original_history = list(game._history)
        original_player = game._player
        original_turns = game._turns

        game2 = loads(dumps(game))
        assert board_to_fen(game2._board) == original_fen
        assert game2._history == original_history
        assert game2._player == original_player
        assert game2._turns == original_turns

    def test_repetition_count_survives_mid_cycle_round_trip(self):
        """Pin the COUNTER's serialization, not just the history's.

        Taken mid-repetition-cycle (count 2, short of the fourfold draw) so
        the assertion has something to lose: the previous round-trip test
        only compared histories, which a counter silently rebuilt from the
        history would satisfy identically.
        """
        game = Game(board_from_fen(FOURFOLD_FEN))
        for san in HALF_SHUTTLE:
            _play_san(game, san)
        assert game.repetition_count() == 2, "mid-cycle, not yet a draw"

        game2 = loads(dumps(game))
        assert game2.repetition_count() == 2
        assert game2._key_counts == game._key_counts
        assert game2._key == game._key
        assert game2.win_state().sum() == 0

        # And the unpickled game still reaches the draw on the 4th occurrence.
        for san in HALF_SHUTTLE * 2:
            _play_san(game2, san)
        assert game2.repetition_count() == 4
        assert game2.win_state()[2] == 1

    def test_pre_repair_pickle_shape_is_rejected(self):
        """A 243-byte-key history does NOT load (review F1).

        The history validation in ``Game.__init__`` refuses the v1 cells-only
        keys rather than loading a game whose ``repetition_count()`` is
        silently 0.

        The components are given at the FULL post-KOM-002 arity on purpose, so
        that the refusal this test pins is the history one.  The old *arity*
        has its own named refusal now (KOM-002 §5.2), pinned by
        ``tests/engine/test_gamestate.py::test_an_old_arity_pickle_is_refused``;
        calling with five components here would fire that instead and this test
        would stop saying anything about 243-byte keys.

        KOM-141 made the repetition rule a ninth REQUIRED component, so the
        call below passes it too, for the same reason: at eight components the
        KOM-141 arity refusal would fire first.  It passes ``FULL`` explicitly
        -- under ``PLACEMENT`` a 243-byte history is the CORRECT shape and
        would load, which is the whole point of the rule being a parameter.
        """
        from komugi.engine.game import RepetitionRule, _unpickle_game

        board = board_from_fen(FOURFOLD_FEN)
        legacy_history = [board.cells_arr.tobytes()] * 3
        assert len(legacy_history[0]) == 243

        with pytest.raises(ValueError, match="243 bytes"):
            _unpickle_game(
                board, legacy_history, 0, 4, None, None, None, 0, RepetitionRule.FULL
            )

    def test_clone_independence(self):
        """Mutating the unpickled game doesn't affect the original."""
        game = self._make_mid_game()
        original_fen = board_to_fen(game._board)

        game2 = loads(dumps(game))
        # Mutate the unpickled copy.
        moves = game2.valid_moves()
        actions = np.nonzero(moves)[0]
        if len(actions) > 0:
            game2.play_action(int(actions[0]))

        # Original must be unchanged.
        assert board_to_fen(game._board) == original_fen


# ── 3. Multiprocessing queue repro (the real failure path) ────────────────────


def _child_put(queue, fen):
    """Target for the child process: build a Game, clone it, push through queue."""
    game = Game(board_from_fen(fen))
    # Play a couple of moves so state is non-trivial.
    for _ in range(2):
        moves = game.valid_moves()
        actions = np.nonzero(moves)[0]
        if len(actions) == 0:
            break
        game.play_action(int(actions[0]))

    winstate = np.array([0, 0, 1], dtype=np.uint8)
    agent_id = 42
    # This is the exact payload SelfPlayAgent._finish_game puts on result_queue.
    queue.put((game.clone(), winstate, agent_id))


class TestMultiprocessingQueue:
    def test_result_queue_round_trip(self):
        """Reproduces the exact SelfPlayAgent._finish_game payload path.

        Before the fix, the feeder thread silently dropped the item because
        Board couldn't be pickled.  After the fix, the parent receives it.
        """
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        fen = INTRO_POSITION

        # Build the expected game state in the parent too.
        expected_game = Game(board_from_fen(fen))
        for _ in range(2):
            moves = expected_game.valid_moves()
            actions = np.nonzero(moves)[0]
            if len(actions) == 0:
                break
            expected_game.play_action(int(actions[0]))
        expected_fen = board_to_fen(expected_game._board)

        proc = ctx.Process(target=_child_put, args=(queue, fen))
        proc.start()

        try:
            result = queue.get(timeout=30)
        except Exception:
            proc.join(timeout=5)
            raise AssertionError("Queue.get timed out — item was silently dropped (the bug)")

        proc.join(timeout=10)
        assert proc.exitcode == 0

        received_game, winstate, agent_id = result
        assert board_to_fen(received_game._board) == expected_fen
        assert np.array_equal(winstate, np.array([0, 0, 1], dtype=np.uint8))
        assert agent_id == 42
