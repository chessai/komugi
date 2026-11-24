"""Tests for komugi.engine.game — GameState wrapper contract."""

import numpy as np
import pytest

from komugi.engine.game import (
    Game,
    _KEY_BYTES,
    _mirror_key,
    _mirror_sq,
    _mirror_action,
)
from komugi.engine import action_encoding
from komugi.engine import cyengine
from komugi.engine import observation
from komugi.engine.cyboard import board_from_fen, board_to_fen
from komugi.engine.cymovegen import generate_all_moves


# ---------------------------------------------------------------------------
# FEN fixtures
# ---------------------------------------------------------------------------

# Intro position (default) — white to move, no drafting
INTRO_FEN = (
    "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3"
    " J2N2R2D1/j2n2r2d1 w 0 - 1"
)

# Checkmate — black is in checkmate, white wins (black to move, no escape)
CHECKMATE_FEN = (
    "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3"
    " J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
)

# Stalemate — black to move, no legal moves, not in check
STALEMATE_FEN = "8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1"

# Insufficient material — two non-adjacent marshals
INSUFFICIENT_FEN = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"

# Marshal captured — game over, not checkmate or stalemate
MARSHAL_CAPTURED_FEN = (
    "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6"
    "/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164"
)

# Intermediate (drafting) — both sides have drafting rights
INTERMEDIATE_FEN = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
)

# Fourfold repetition setup — generals shuttle back and forth
FOURFOLD_FEN = "4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1"
FOURFOLD_SHUTTLE = [
    "\u5927(9-4-1)(8-4-1)",
    "\u5927(1-4-1)(2-4-1)",
    "\u5927(8-4-1)(9-4-1)",
    "\u5927(2-4-1)(1-4-1)",
    "\u5927(9-4-1)(8-4-1)",
    "\u5927(1-4-1)(2-4-1)",
    "\u5927(8-4-1)(9-4-1)",
    "\u5927(2-4-1)(1-4-1)",
    "\u5927(9-4-1)(8-4-1)",
    "\u5927(1-4-1)(2-4-1)",
    "\u5927(8-4-1)(9-4-1)",
    "\u5927(2-4-1)(1-4-1)",
]

# Beginner position — no drafting, more pieces than intro
BEGINNER_FEN = (
    "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
    " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_for_san(game, san):
    """Find the action id for a SAN string in the current position."""
    moves = generate_all_moves(game._board)
    for m in moves:
        if m["san"] == san:
            return action_encoding.encode_cython_move(m)
    raise ValueError(f"No move with SAN {san!r}")


def _play_san(game, san):
    """Play a SAN move through *game* (used for repetition setup).

    Goes through ``Game.play_action`` rather than poking the board directly,
    because the repetition counter is maintained there — a board mutated behind
    the Game's back would leave the counter stale.
    """
    game.play_action(_action_for_san(game, san))


# ===========================================================================
# (1) CONTRACT
# ===========================================================================

class TestContract:
    def test_valid_moves_dtype_and_length(self):
        g = Game()
        vm = g.valid_moves()
        assert vm.dtype == np.uint8
        assert len(vm) == Game.action_size()
        assert vm.sum() >= 1, "At least one legal move at start"

    def test_observation_shape_and_dtype(self):
        g = Game()
        obs = g.observation()
        assert obs.shape == Game.observation_size()
        assert obs.dtype == np.float32

    def test_num_players(self):
        assert Game.num_players() == 2

    def test_action_size(self):
        assert Game.action_size() == 37341  # 461 planes * 81 squares (KOM-028)

    def test_observation_size(self):
        assert Game.observation_size() == (128, 9, 9)

    def test_max_turns(self):
        # Derived from the one leaf (KOM-165), never a literal: the cap and the
        # denominator of observation planes 115/120 are the same number.
        assert Game.max_turns() == observation.MAX_TURNS

    def test_has_draw(self):
        assert Game.has_draw() is True

    def test_initial_player_is_zero(self):
        g = Game()
        assert g.player == 0  # white starts

    def test_initial_turns_is_zero(self):
        g = Game()
        assert g.turns == 0


# ===========================================================================
# (2) play_action advances state
# ===========================================================================

class TestPlayAction:
    def test_play_action_increments_turns(self):
        g = Game()
        vm = g.valid_moves()
        action = int(np.nonzero(vm)[0][0])
        g.play_action(action)
        assert g.turns == 1

    def test_play_action_sets_last_action(self):
        g = Game()
        vm = g.valid_moves()
        action = int(np.nonzero(vm)[0][0])
        g.play_action(action)
        assert g.last_action == action

    def test_player_tracks_engine_turn(self):
        g = Game()
        vm = g.valid_moves()
        action = int(np.nonzero(vm)[0][0])
        g.play_action(action)
        expected_player = g._board.turn  # integer attribute
        assert g.player == expected_player


# ===========================================================================
# (3) MASK == ENGINE
# ===========================================================================

class TestMaskEqualsEngine:
    def _assert_mask_matches(self, board):
        g = Game(board)
        vm = g.valid_moves()
        mask_ids = set(int(i) for i in np.nonzero(vm)[0])
        engine_ids = action_encoding.legal_cython_action_ids(board)
        assert mask_ids == engine_ids

    def test_mask_at_start(self):
        self._assert_mask_matches(board_from_fen(INTRO_FEN))

    def test_mask_beginner(self):
        self._assert_mask_matches(board_from_fen(BEGINNER_FEN))

    def test_mask_mid_draft(self):
        self._assert_mask_matches(board_from_fen(INTERMEDIATE_FEN))

    def test_every_set_bit_decodes(self):
        """Every set bit in valid_moves decodes to a legal move."""
        g = Game()
        vm = g.valid_moves()
        for aid in np.nonzero(vm)[0]:
            decoded = action_encoding.decode_cython_action(int(aid), g._board)
            assert decoded is not None

    def test_terminal_checkmate_all_zero(self):
        """A checkmate position has all-zero mask."""
        g = Game(board_from_fen(CHECKMATE_FEN))
        vm = g.valid_moves()
        assert vm.sum() == 0


# ===========================================================================
# (4) SYMMETRY
# ===========================================================================

class TestSymmetry:
    def test_symmetries_returns_exactly_two(self):
        g = Game()
        pi = np.random.RandomState(42).random(Game.action_size()).astype(np.float32)
        syms = g.symmetries(pi)
        assert len(syms) == 2

    def test_identity_is_clone(self):
        g = Game()
        pi = np.ones(Game.action_size(), dtype=np.float32)
        syms = g.symmetries(pi)
        identity_state, identity_pi = syms[0]
        assert identity_state == g
        np.testing.assert_array_equal(identity_pi, pi)

    def test_mirror_twice_is_identity_pi(self):
        """Applying mirror to the policy twice gives back the original."""
        rng = np.random.RandomState(123)
        pi = rng.random(Game.action_size()).astype(np.float32)
        mirrored_once = np.zeros_like(pi)
        for i in range(len(pi)):
            if pi[i] > 0:
                mirrored_once[_mirror_action(i)] = pi[i]
        mirrored_twice = np.zeros_like(pi)
        for i in range(len(mirrored_once)):
            if mirrored_once[i] > 0:
                mirrored_twice[_mirror_action(i)] = mirrored_once[i]
        np.testing.assert_allclose(mirrored_twice, pi, atol=1e-7)

    def test_mirror_sq_involution(self):
        """mirror(mirror(sq)) == sq for all squares."""
        for i in range(81):
            assert _mirror_sq(_mirror_sq(i)) == i

    def test_mirror_key_is_an_involution_on_274_byte_keys(self):
        """Mirroring twice is the identity, and the suffix never moves."""
        g = Game(board_from_fen(FOURFOLD_FEN))
        key = g._key
        assert len(key) == 274
        mirrored = _mirror_key(key)
        assert mirrored[243:] == key[243:], "turn/draft/hands pass through"
        assert mirrored[:243] != key[:243], "the cells really are mirrored"
        assert _mirror_key(mirrored) == key

    def test_symmetry_preserves_repetition_counts(self):
        """The mirror is a bijection on keys, so counts carry over exactly."""
        g = Game(board_from_fen(FOURFOLD_FEN))
        for san in FOURFOLD_SHUTTLE[:8]:
            _play_san(g, san)
        assert g.repetition_count() == 3

        pi = np.zeros(Game.action_size(), dtype=np.float32)
        _, (mirrored_game, _) = g.symmetries(pi)

        assert mirrored_game.repetition_count() == g.repetition_count()
        assert sorted(mirrored_game._key_counts.values()) == sorted(
            g._key_counts.values()
        )
        assert mirrored_game._key_counts == {
            _mirror_key(k): v for k, v in g._key_counts.items()
        }

    def test_no_rotation_or_vertical_flip(self):
        """Only identity + file-mirror; no rotation or vertical flip."""
        g = Game()
        pi = np.ones(Game.action_size(), dtype=np.float32)
        syms = g.symmetries(pi)
        assert len(syms) == 2, "Must be exactly 2 (identity + file-mirror)"


# ===========================================================================
# (5) WIN_STATE
# ===========================================================================

class TestWinState:
    def test_marshal_capture_win(self):
        """Marshal captured -> win for the side that captured."""
        g = Game(board_from_fen(MARSHAL_CAPTURED_FEN))
        ws = g.win_state()
        assert ws.sum() == 1
        # Marshal captured with black to move means white captured -> player 0 wins
        assert ws[0] == 1

    def test_checkmate_win_correct_player(self):
        """Checkmate -> win for the side NOT to move."""
        g = Game(board_from_fen(CHECKMATE_FEN))
        ws = g.win_state()
        assert ws.sum() == 1
        # Black to move and in checkmate -> white (player 0) wins
        assert ws[0] == 1

    def test_stalemate_draw(self):
        g = Game(board_from_fen(STALEMATE_FEN))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1  # draw slot

    def test_insufficient_draw(self):
        g = Game(board_from_fen(INSUFFICIENT_FEN))
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1

    def test_fourfold_draw(self):
        """The shuttle is a genuine cycle, so it still draws after the repair.

        Same cells, same side to move and same (empty) hands every fourth ply:
        the repaired 274-byte key splits none of these states apart, and the
        draw lands on the 4th occurrence of the start position.
        """
        g = Game(board_from_fen(FOURFOLD_FEN))
        assert g.repetition_count() == 1, "the start position is occurrence 1"
        for san in FOURFOLD_SHUTTLE:
            _play_san(g, san)
        assert g.repetition_count() == 4
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1

    def test_fourfold_does_not_fire_early(self):
        """Three occurrences is not a draw — the threshold is 4, not 3."""
        g = Game(board_from_fen(FOURFOLD_FEN))
        for san in FOURFOLD_SHUTTLE[:8]:
            _play_san(g, san)
        assert g.repetition_count() == 3
        assert g.win_state().sum() == 0
        assert g.valid_moves().sum() > 0

    def test_max_turns_draw(self):
        g = Game()
        g._turns = Game.max_turns()  # force max turns
        ws = g.win_state()
        assert ws.sum() == 1
        assert ws[2] == 1

    def test_mid_draft_all_zeros(self):
        g = Game(board_from_fen(INTERMEDIATE_FEN))
        ws = g.win_state()
        assert ws.sum() == 0

    def test_normal_position_all_zeros(self):
        g = Game()
        ws = g.win_state()
        assert ws.sum() == 0


# ===========================================================================
# (6) CLONE
# ===========================================================================

class TestClone:
    def test_clone_equals_parent(self):
        g = Game()
        c = g.clone()
        assert g == c

    def test_clone_independent_mutation(self):
        """Mutating clone does not affect parent."""
        g = Game()
        parent_fen = board_to_fen(g._board)
        parent_hist_len = len(g._history)
        c = g.clone()
        # Play a move on clone
        vm = c.valid_moves()
        action = int(np.nonzero(vm)[0][0])
        c.play_action(action)
        # Parent unchanged
        assert board_to_fen(g._board) == parent_fen
        assert len(g._history) == parent_hist_len
        assert g.turns == 0

    def test_clone_fourfold_independent(self):
        """Repetition on a clone doesn't leak to the parent.

        Covers the counter as well as the key list: the clone's dict must be a
        copy, not a shared reference.
        """
        g = Game(board_from_fen(FOURFOLD_FEN))
        # Play most of the shuttle on the original
        for san in FOURFOLD_SHUTTLE[:8]:
            _play_san(g, san)

        c = g.clone()
        assert c._key_counts is not g._key_counts, "no shared counter"
        # Finish the shuttle on the clone
        for san in FOURFOLD_SHUTTLE[8:]:
            _play_san(c, san)

        assert c.repetition_count() == 4
        assert c.win_state()[2] == 1
        # Parent should NOT be drawn
        assert g.repetition_count() == 3
        assert g.win_state().sum() == 0


# ===========================================================================
# (6b) CALLER-SUPPLIED HISTORY VALIDATION  (review F1)
# ===========================================================================

class TestHistoryValidation:
    """``Game(board, _history)`` must reject a history it cannot have played.

    The counter is the single source of truth for the repetition threshold, so
    a history that does not match the board is the one way the repaired
    invariants can be broken from outside.  Before this validation each case
    below "loaded" and then mis-reported ``repetition_count()``.
    """

    def test_pre_repair_243_byte_history_is_rejected(self):
        """A v1 cells-only history must raise, not silently count zero.

        This is exactly the shape of a pre-repair pickle (main's 5-tuple
        ``__reduce__``): 243-byte keys that can never equal a repaired
        274-byte key, so ``repetition_count()`` used to return 0 and every
        occurrence recorded before the pickle was silently forgotten.
        """
        board = board_from_fen(FOURFOLD_FEN)
        legacy_key = board.cells_arr.tobytes()
        assert len(legacy_key) == 243, "the v1 key was cells only"

        with pytest.raises(ValueError, match="243 bytes"):
            Game(board, [legacy_key] * 3)

    def test_history_ending_in_a_wrong_key_is_rejected(self):
        """Right length, wrong position — the late-adjudication case.

        A bridge handing over a history that does not end in the current
        position's key would count one occurrence too few and adjudicate the
        draw a ply late.
        """
        board = board_from_fen(FOURFOLD_FEN)
        foreign_key = cyengine.repetition_key(board_from_fen(INTRO_FEN))
        assert len(foreign_key) == _KEY_BYTES, "a valid key, just not this one"

        with pytest.raises(ValueError, match="current position"):
            Game(board, [foreign_key])

    def test_history_of_the_current_key_is_accepted(self):
        """The control: the bridge/fixture shape still constructs and counts.

        ``tests/rules/fixtures.game_with_repeated_history`` builds exactly
        this, so the validation must not close the legitimate door.
        """
        board = board_from_fen(FOURFOLD_FEN)
        key = cyengine.repetition_key(board)
        g = Game(board, [key] * 3)
        assert g.repetition_count() == 3
        assert g.win_state().sum() == 0

    def test_empty_history_is_accepted(self):
        """An empty history makes no claim, so there is nothing to contradict."""
        g = Game(board_from_fen(FOURFOLD_FEN), [])
        assert g.repetition_count() == 0

    def test_clone_round_trip_still_validates_clean(self):
        """clone() re-enters __init__ with its own history — mid-cycle too."""
        g = Game(board_from_fen(FOURFOLD_FEN))
        for san in FOURFOLD_SHUTTLE[:4]:
            _play_san(g, san)
        assert g.repetition_count() == 2

        c = g.clone()
        assert c.repetition_count() == 2
        assert c._history == g._history


def test_the_history_validation_class_still_carries_all_its_methods():
    """Tripwire: a dedent inside ``TestHistoryValidation`` must be LOUD.

    KOM-002a's ``#31`` node was inserted at module level *inside* this class,
    which ended the class body and re-parsed the two methods that followed as
    nested functions of it — uncollected, but still importing, still type
    checking, still green. Two KOM-027 history-validation regressions were
    silently gone and no gate could see it.

    So the membership is asserted against the class object rather than left to
    collection: any future edit that dedents a method out of the class fails
    here by name, in the same file, instead of quietly shrinking the suite.
    """
    expected = {
        "test_pre_repair_243_byte_history_is_rejected",
        "test_history_ending_in_a_wrong_key_is_rejected",
        "test_history_of_the_current_key_is_accepted",
        "test_empty_history_is_accepted",
        "test_clone_round_trip_still_validates_clean",
    }
    present = {
        name
        for name in vars(TestHistoryValidation)
        if name.startswith("test_") and callable(vars(TestHistoryValidation)[name])
    }
    assert expected <= present, (
        "these methods left TestHistoryValidation (a dedent swallows them into "
        f"a nested function and pytest stops collecting them): {expected - present}"
    )


def test_an_old_arity_pickle_is_refused():
    """KOM-002 §7 #31: the pre-KOM-002 component tuple RAISES, naming the ticket.

    ``_last_move`` and ``_progress_plies`` are REQUIRED components, not
    defaulted ones, and the asymmetry with ``counts`` is the whole argument:
    ``counts=None`` triggers a *correct rebuild* from the history, whereas
    ``_progress_plies = 0`` on a mid-game pickle is simply a WRONG value --
    in range, and indistinguishable from a game that has just captured. Such a
    pickle would load, feed a wrong plane 120 into training, and say nothing.

    Both older arities are pinned: the post-KOM-027 6-tuple (which is what an
    in-flight campaign's pickles actually are) and main's 5-tuple.

    KOM-141 added a NINTH component, the repetition rule, on the same terms --
    so the pre-KOM-141 8-tuple is refused too, naming that ticket instead.
    """
    from komugi.engine.game import _unpickle_game

    game = Game(board_from_fen(INTRO_FEN))
    game.play_action(int(np.nonzero(game.valid_moves())[0][0]))

    _, components = game.__reduce__()
    assert len(components) == 9, (
        "board, history, player, turns, action, counts, last move, progress, rule"
    )

    for arity in (6, 5):
        with pytest.raises(ValueError, match="KOM-002") as raised:
            _unpickle_game(*components[:arity])
        assert "_progress_plies" in str(raised.value), (
            f"arity {arity}: the refusal must name what is missing"
        )

    # KOM-141's own arity: last_move and progress are present, the rule is not.
    with pytest.raises(ValueError, match="KOM-141") as rule_raised:
        _unpickle_game(*components[:8])
    assert "_repetition_rule" in str(rule_raised.value), (
        "arity 8: the refusal must name what is missing"
    )

    # The control: at the full arity the round trip still restores all three.
    revived = _unpickle_game(*components)
    assert revived._last_move == game._last_move
    assert revived._progress_plies == game._progress_plies
    assert revived.repetition_rule is game.repetition_rule


# ===========================================================================
# (6c) REPETITION-KEY uint8 NARROWING GUARD  (review F6)
# ===========================================================================

class TestRepetitionKeyHandGuard:
    """``astype(np.uint8)`` wraps silently; the key must crash instead.

    Unreachable through play — hand counts start at <= 4 and only decrease —
    but ``Board.hand_arr`` is a writable public ndarray, so the guard has a
    real reproducing path and is not merely defensive documentation.
    """

    def test_hand_count_above_255_raises(self):
        board = board_from_fen(FOURFOLD_FEN)
        board.hand_arr[0, 0] = 256  # would wrap to 0 and alias with an empty slot
        with pytest.raises(ValueError, match="uint8"):
            cyengine.repetition_key(board)

    def test_negative_hand_count_raises(self):
        board = board_from_fen(FOURFOLD_FEN)
        board.hand_arr[0, 0] = -1  # would wrap to 255
        with pytest.raises(ValueError, match="uint8"):
            cyengine.repetition_key(board)

    def test_in_range_hand_counts_are_untouched(self):
        """The guard is a bounds check only — it must not change any key."""
        board = board_from_fen(BEGINNER_FEN)
        assert len(cyengine.repetition_key(board)) == _KEY_BYTES


# ===========================================================================
# (7) RANDOM GAMES
# ===========================================================================

class TestRandomGames:
    def test_random_games(self):
        """Play >=100 seeded random games to completion."""
        rng = np.random.RandomState(42)
        num_games = 100
        for game_idx in range(num_games):
            g = Game()
            while True:
                ws = g.win_state()
                if ws.any():
                    # Game terminated
                    assert ws.sum() == 1
                    break

                vm = g.valid_moves()
                assert vm.sum() > 0, (
                    f"Game {game_idx}: no legal moves but win_state is all zeros "
                    f"at turn {g.turns}"
                )

                legal_actions = np.nonzero(vm)[0]
                action = int(rng.choice(legal_actions))

                # Should not raise
                g.play_action(action)

                # Player tracks engine turn (integer attribute)
                expected = g._board.turn
                assert g.player == expected, (
                    f"Game {game_idx}: player {g.player} != expected {expected} "
                    f"at turn {g.turns}"
                )

                # Must terminate within max_turns
                assert g.turns <= Game.max_turns(), (
                    f"Game {game_idx}: exceeded max_turns at {g.turns}"
                )


# ===========================================================================
# (8) THE PER-SIDE DRAFT PREDICATE  (KOM-036 §3.3, acceptance #20)
# ===========================================================================
#
# Module-level, not class-scoped: the design pre-registers this node's id as a
# bare ``::test_...``, which does not resolve against a method.

_DRAFT_PATTERNS = ((1, 1), (1, 0), (0, 1), (0, 0))
"""Every ``(draft_w, draft_b)`` combination, the two asymmetric ones included.

The asymmetric rows are the whole point. ``Game.win_state`` gates on the
*disjunction* ``draft_w or draft_b``, which is correct for termination (a
whole-game property) and wrong for a temperature schedule (a property of the
move being searched). The two predicates agree on ``(1, 1)`` and ``(0, 0)`` and
differ for one side on each of the other two, so a test that only ever visited
the symmetric states could not tell them apart -- which is exactly what rev 1
of the design asserted, and what review M2 rejected.
"""


def _drafting_board(draft_w, draft_b):
    """Return an intermediate-mode board with the two draft bits set as asked.

    The bits are set on the board rather than reached by play because they are
    what the predicate reads: ``Board.draft_w``/``draft_b`` are public ints, and
    driving them directly is the tightest statement of "``in_draft(side)``
    returns *this side's* bit". Reaching the asymmetric states through random
    play would additionally depend on whether a random line ever happens to
    finish one side's draft, which is a fact about move generation and not
    about this predicate.
    """
    board = board_from_fen(INTERMEDIATE_FEN)
    board.draft_w = draft_w
    board.draft_b = draft_b
    return board


def test_in_draft_is_the_per_side_predicate_the_engine_already_spells():
    """``Game.in_draft(side)`` is ``draft_w if side == 0 else draft_b``.

    That is the expression ``game.py``'s placement-floor filter already spells
    for the side to move, and the schedule KOM-036 §3.2 composes is stated over
    it -- the **mover's** bit, never either side's. Three halves:

    * every ``(draft_w, draft_b)`` pattern, so the asymmetric ones are visited
      and the disjunction is shown to disagree there;
    * both start positions, drafting and past-draft;
    * every ply of a played game, for **both** sides, so the predicate is also
      checked against states the engine itself produced rather than only
      against states the test wrote.
    """
    for draft_w, draft_b in _DRAFT_PATTERNS:
        game = Game(_drafting_board(draft_w, draft_b))
        assert game.in_draft(0) is bool(draft_w)
        assert game.in_draft(1) is bool(draft_b)

        # The disjunction win_state uses cannot express these two states: on an
        # asymmetric pattern it answers True for the side that has finished.
        disjunction = bool(draft_w or draft_b)
        if draft_w != draft_b:
            assert (game.in_draft(0), game.in_draft(1)) != (
                disjunction,
                disjunction,
            ), "the per-side predicate collapsed into draft_w or draft_b"

    # Both sides are drafting at a drafting start position ...
    drafting = Game(board_from_fen(INTERMEDIATE_FEN))
    assert drafting.in_draft(0)
    assert drafting.in_draft(1)
    # ... and neither is at the intro start, which begins past the draft. The
    # start position is taken from a FEN rather than from ``Game()`` because
    # ``Game()`` follows GUNGI_MODE, and this node is about the predicate.
    intro = Game(board_from_fen(INTRO_FEN))
    assert not intro.in_draft(0)
    assert not intro.in_draft(1)

    # Every ply of a played game, for both sides.
    rng = np.random.RandomState(20260907)
    game = Game(board_from_fen(INTERMEDIATE_FEN))
    seen = set()
    plies = 0
    while plies < 400 and not game.win_state().any():
        board = game._board
        seen.add((int(board.draft_w), int(board.draft_b)))
        for side in (0, 1):
            expected = bool(board.draft_w if side == 0 else board.draft_b)
            assert game.in_draft(side) is expected, (
                f"ply {plies}, side {side}: in_draft disagrees with the "
                f"board's own bit"
            )
        legal = np.nonzero(game.valid_moves())[0]
        if legal.size == 0:
            break
        game.play_action(int(rng.choice(legal)))
        plies += 1

    assert plies > 0, "the played-game half never took a ply"
    assert (1, 1) in seen, "the game never started in the draft phase"
