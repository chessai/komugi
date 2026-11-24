"""The oracle's own tests: the fixtures, before anything is judged by them.

Nothing here imports the search. These are the tests that keep the rest of
the suite honest -- an oracle that is itself wrong does not fail loudly, it
quietly certifies whatever the implementation happens to do. So the games and
the solver are pinned against facts that hold independently of both: known
tic-tac-toe results (the empty board is a draw; a two-in-a-row with the third
cell free is a win in one; a single unanswered threat has to be blocked), and
payoffs that were written down by hand and can be minimaxed on paper.

The hand-written verdicts in ``TACTICAL_CASES`` are checked against the solver
here too. Two independent derivations of the same fact: if the analysis in the
comments and the negamax disagree, this file fails, and the oracle tests never
get to run on a standard nobody verified.
"""

from __future__ import annotations

import numpy as np
import pytest
from solver import (
    SolvableState,
    action_values,
    legal_actions,
    miscalibrated_evaluator,
    perfect_evaluator,
    solve,
)
from toygames import TACTICAL_CASES, BOARD_SIZE, TacticalCase, TicTacToe, TreeGame

#: A tree game small enough to minimax in your head:
#: row minima are -0.2, -1.0 and 0.2, so the root is worth 0.2 by playing 2.
HAND_TABLE: tuple[tuple[float, ...], ...] = (
    (0.5, -0.2, 0.1),
    (-1.0, 0.9, 0.3),
    (0.2, 0.2, 0.2),
)

X_WON = "XXX/OO./..."  # X took the top row; O is to move and has lost
DRAWN = "XOX/XOO/OXX"  # full board, no line


def _protocol_check(state: SolvableState) -> SolvableState:
    """Identity, but only type-checks if `state` duck-types the protocol."""
    return state


# --------------------------------------------------------------------------
# tic-tac-toe: the rules
# --------------------------------------------------------------------------


def test_both_games_satisfy_the_state_protocol() -> None:
    """The search is handed these objects; mypy is the assertion here."""
    assert _protocol_check(TicTacToe()).to_play() == 0
    assert _protocol_check(TreeGame(HAND_TABLE)).to_play() == 0


def test_the_empty_board_offers_every_cell() -> None:
    """Nine legal actions, player 0 to move, nothing decided."""
    state = TicTacToe()

    mask = state.legal_mask()

    assert mask.shape == (BOARD_SIZE,)
    assert mask.dtype == np.dtype(np.bool_)
    assert legal_actions(state) == tuple(range(BOARD_SIZE))
    assert state.to_play() == 0
    assert not state.is_terminal()
    assert state.winner() is None


def test_playing_writes_the_mark_and_passes_the_turn() -> None:
    """X's move leaves an X, hands the turn to O, and closes that cell."""
    state = TicTacToe()

    state.play(4)

    assert state.cells[4] == 1
    assert state.to_play() == 1
    assert legal_actions(state) == (0, 1, 2, 3, 5, 6, 7, 8)

    state.play(0)

    assert state.cells[0] == 2
    assert state.to_play() == 0


def test_illegal_moves_are_refused() -> None:
    """An occupied cell, or one off the board, is a caller bug and raises."""
    state = TicTacToe()
    state.play(4)

    with pytest.raises(ValueError):
        state.play(4)
    with pytest.raises(ValueError):
        state.play(9)
    with pytest.raises(ValueError):
        state.play(-1)


def test_a_decided_game_accepts_no_further_moves() -> None:
    """Once a line is complete the position is over, empty cells or not."""
    state = TicTacToe.from_string(X_WON)

    assert state.is_terminal()
    assert not state.legal_mask().any()
    with pytest.raises(ValueError):
        state.play(5)


def test_terminal_value_is_read_from_the_movers_seat() -> None:
    """The same won board is -1 to the loser on move and +1 to the winner."""
    lost = TicTacToe.from_string(X_WON)  # counts imply O to move
    won = TicTacToe.from_string(X_WON, to_play=0)

    assert lost.to_play() == 1
    assert lost.winner() == 0
    assert lost.terminal_value() == -1.0
    assert won.terminal_value() == 1.0


def test_a_full_board_without_a_line_is_a_draw() -> None:
    """The drawn board is terminal, has no winner, and is worth zero."""
    state = TicTacToe.from_string(DRAWN)

    assert state.is_terminal()
    assert state.winner() is None
    assert state.terminal_value() == 0.0


def test_terminal_value_refuses_a_live_position() -> None:
    """Asking a running game for its final value is a bug, not a zero."""
    with pytest.raises(ValueError):
        TicTacToe().terminal_value()


def test_clone_is_independent_of_the_original() -> None:
    """Playing on a clone must leave the original untouched, mask included."""
    state = TicTacToe()
    state.play(0)
    before_cells, before_to_play = state.cells, state.to_play()
    before_mask = state.legal_mask().copy()

    branch = state.clone()
    branch.play(4)
    branch.play(8)

    assert state.cells == before_cells
    assert state.to_play() == before_to_play
    assert np.array_equal(state.legal_mask(), before_mask)
    assert branch.cells[4] == 2
    assert branch.cells != state.cells


def test_from_string_infers_the_side_to_move_and_rejects_nonsense() -> None:
    """Mark counts pick the mover; counts no game could reach raise."""
    assert TicTacToe.from_string(".../.../...").to_play() == 0
    assert TicTacToe.from_string("X../.../...").to_play() == 1

    with pytest.raises(ValueError):
        TicTacToe.from_string("XXX/XX./...")  # five X, no O
    with pytest.raises(ValueError):
        TicTacToe.from_string("XX./OO./O.")  # eight cells
    with pytest.raises(ValueError):
        TicTacToe.from_string("XX./OO./O.Z")  # not a glyph


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------


def test_the_empty_board_is_a_draw_with_perfect_play() -> None:
    """The one tic-tac-toe fact everybody knows, and every opening holds it."""
    solution = solve(TicTacToe())

    assert solution.value == 0.0
    assert set(solution.best_actions) == set(range(BOARD_SIZE))
    assert all(value == 0.0 for value in action_values(TicTacToe()).values())


def test_a_terminal_state_has_its_own_value_and_no_moves() -> None:
    """The solver bottoms out where the game does."""
    solution = solve(TicTacToe.from_string(X_WON))

    assert solution.value == -1.0
    assert solution.best_actions == ()


def test_an_immediate_win_is_found_and_is_the_only_optimal_move() -> None:
    """Two in a row with the third cell free: value +1, one right answer."""
    state = TicTacToe.from_string("XX./OO./O.X")

    solution = solve(state)

    assert solution.value == 1.0
    assert solution.best_actions == (2,)
    assert action_values(state) == {2: 1.0, 5: -1.0, 7: -1.0}


def test_an_unanswered_threat_must_be_blocked() -> None:
    """Blocking holds the draw; the other two moves lose on the spot."""
    state = TicTacToe.from_string("OO./XXO/.X.")

    solution = solve(state)

    assert solution.value == 0.0
    assert solution.best_actions == (2,)
    assert action_values(state) == {2: 0.0, 6: -1.0, 8: -1.0}


def test_ties_are_reported_as_a_set_not_a_pick() -> None:
    """When several moves win, the solver names all of them."""
    solution = solve(TicTacToe.from_string("OXO/XX./O.."))

    assert solution.value == 1.0
    assert set(solution.best_actions) == {5, 7, 8}


@pytest.mark.parametrize(
    "case", TACTICAL_CASES, ids=[case.name for case in TACTICAL_CASES]
)
def test_the_hand_analysed_cases_match_the_solver(case: TacticalCase) -> None:
    """Every curated position's written-down verdict is checked, not trusted."""
    state = case.state()

    solution = solve(state)

    assert state.to_play() == case.to_play
    assert solution.value == case.value
    assert set(solution.best_actions) == set(case.optimal)
    assert set(case.optimal) <= set(legal_actions(state))


@pytest.mark.parametrize(
    "case", TACTICAL_CASES, ids=[case.name for case in TACTICAL_CASES]
)
def test_the_tactical_cases_are_small_enough_to_be_fully_searchable(
    case: TacticalCase,
) -> None:
    """At most four legal actions, so ``m >= 4`` covers every one of them.

    The oracle leans on this: with the whole legal set inside the candidate
    set, a search that returns a losing move cannot blame the Gumbel draw for
    having left the good move out.
    """
    assert len(legal_actions(case.state())) <= 4


def test_the_solver_is_repeatable_and_the_memo_does_not_leak_between_games() -> None:
    """Two tables, same move prefix, different truths -- and they stay apart."""
    state = TicTacToe.from_string("OO./XXO/.X.")

    assert solve(state) == solve(state.clone())

    other = tuple(tuple(-value for value in row) for row in HAND_TABLE)
    assert solve(TreeGame(HAND_TABLE)).value != solve(TreeGame(other)).value


# --------------------------------------------------------------------------
# the tree game
# --------------------------------------------------------------------------


def test_the_tree_game_plays_two_plies_and_hands_back_the_payoff() -> None:
    """to_play cycles 0, 1, 0 and the leaf reports its table entry."""
    state = TreeGame(HAND_TABLE)

    assert state.to_play() == 0
    assert state.num_actions == 3
    assert legal_actions(state) == (0, 1, 2)

    state.play(1)
    assert state.to_play() == 1
    assert not state.is_terminal()
    assert legal_actions(state) == (0, 1, 2)

    state.play(0)
    assert state.to_play() == 0
    assert state.is_terminal()
    assert not state.legal_mask().any()
    assert state.terminal_value() == pytest.approx(HAND_TABLE[1][0])


def test_the_tree_game_minimaxes_to_the_value_you_can_read_off_the_table() -> None:
    """max_a min_b of the hand table is 0.2, taken by action 2."""
    expected = max(min(row) for row in HAND_TABLE)

    solution = solve(TreeGame(HAND_TABLE))

    assert expected == pytest.approx(0.2)
    assert solution.value == pytest.approx(expected)
    assert solution.best_actions == (2,)


def test_tree_game_action_values_are_the_row_minima() -> None:
    """One negamax step down: playing `a` is worth min_b payoffs[a][b]."""
    values = action_values(TreeGame(HAND_TABLE))

    assert values == pytest.approx({0: -0.2, 1: -1.0, 2: 0.2})


def test_the_opponents_node_is_the_negated_row() -> None:
    """After the root plays 0 the mover wants max_b -payoffs[0][b] = 0.2."""
    state = TreeGame(HAND_TABLE)
    state.play(0)

    solution = solve(state)

    assert solution.value == pytest.approx(0.2)
    assert solution.best_actions == (1,)


def test_the_tree_game_clone_is_independent() -> None:
    """The root keeps its empty move list while a clone plays on."""
    state = TreeGame(HAND_TABLE)

    branch = state.clone()
    branch.play(2)

    assert state.moves == ()
    assert branch.moves == (2,)
    assert state.to_play() == 0
    assert branch.to_play() == 1


def test_the_tree_game_refuses_malformed_tables_and_moves() -> None:
    """Every payoff is a real value in range and the game is two plies deep."""
    with pytest.raises(ValueError):
        TreeGame(((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)))  # not square
    with pytest.raises(ValueError):
        TreeGame(((1.5, 0.0), (0.0, 0.0)))  # payoff out of [-1, 1]
    with pytest.raises(ValueError):
        TreeGame(())

    state = TreeGame(HAND_TABLE)
    state.play(0)
    state.play(0)
    with pytest.raises(ValueError):
        state.play(0)  # a third ply
    with pytest.raises(ValueError):
        TreeGame(HAND_TABLE).play(3)


# --------------------------------------------------------------------------
# the evaluators built on the solver
# --------------------------------------------------------------------------


def test_the_perfect_evaluator_is_uniform_in_policy_and_exact_in_value() -> None:
    """All-zero logits, and a value that already knows the whole game."""
    state = TicTacToe.from_string("XX./OO./O.X")

    logits, value = perfect_evaluator(state)

    assert logits.shape == (BOARD_SIZE,)
    assert logits.dtype == np.dtype(np.float32)
    assert not logits.any()
    assert value == 1.0
    assert perfect_evaluator(TicTacToe())[1] == 0.0
    assert perfect_evaluator(TicTacToe.from_string(X_WON))[1] == -1.0


def test_the_miscalibrated_evaluator_points_the_prior_at_the_losing_moves() -> None:
    """The bad prior is bad on purpose: its argmax loses, its value is right."""
    state = TicTacToe.from_string("OO./XXO/.X.")  # 2 draws, 6 and 8 lose

    evaluate = miscalibrated_evaluator()
    logits, value = evaluate(state)

    assert value == 0.0
    assert float(logits[2]) == pytest.approx(0.0)
    assert float(logits[6]) == pytest.approx(3.0)
    assert float(logits[8]) == pytest.approx(3.0)
    assert int(np.argmax(logits)) in (6, 8)


def test_the_miscalibrated_evaluator_survives_a_terminal_state() -> None:
    """No legal actions means no logits to bias, and the value still lands."""
    logits, value = miscalibrated_evaluator()(TicTacToe.from_string(X_WON))

    assert not logits.any()
    assert value == -1.0
