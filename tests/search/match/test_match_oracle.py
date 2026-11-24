"""A20 / T12 / T13: the oracle, and the committed tactical fixtures.

The evaluator here is the solver: exact values, uniform logits. That
combination removes every excuse. The prior says nothing, so whatever the
search reports came out of the search; the value says everything, so a losing
move was not a mistake about the position but a mistake about the search. It
is the same oracle standard the Gumbel side is already held to, which is the
point -- a second search that could not meet it would not be a match-play
configuration, it would be a bug.

T13's fixtures are **frozen literals**, produced by the rule
``planted/find_tactical_fixtures.py`` enumerates, and the test does not run the
enumeration. Every table below has:

* a **constant** best row, so every leaf under it backs up the same value and
  the winning edge's variance is exactly zero -- which makes the LCB penalty
  on it the floor term and nothing else;
* every other entry at least ``0.4`` below that value, so no other row's mean
  can reach it however the opponent spends its visits; and
* at least one **trap** row, which is as good as that bound allows except
  against one reply -- a table of flat rows would be won by a search that
  never looked past ply one.

That makes "the search returns the unique best move" a consequence of the
design's own constants rather than a prediction about visit distributions, and
it is why the count is twenty-four rather than the twenty the acceptance asks
for.
"""

from __future__ import annotations

import pytest

# `planted` before `solver`/`toygames`: importing it is what puts the oracle
# directory on ``sys.path``. See ``planted/__init__.py``.
from planted import MatchToyState, solver_wdl_evaluator
from solver import solve
from toygames import BOARD_SIZE, TACTICAL_CASES, TacticalCase, TicTacToe, TreeGame

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch

#: The budget both oracles run at, pinned as a literal.
PLAYOUTS = 64

#: ``(payoff table, the unique best root action)``. Frozen from the committed
#: enumeration; the verdict beside each is re-derived from the table by the
#: test itself, so a mistranscribed literal fails loudly instead of quietly
#: making the fixture easier.
TACTICAL_TABLES: tuple[tuple[tuple[tuple[float, ...], ...], int], ...] = (
    (((0.9, 0.9, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 0.0)), 0),
    (((0.5, 0.5, 0.0), (0.9, 0.9, 0.9), (0.5, 0.5, 0.0)), 1),
    (((0.5, 0.5, 0.0), (0.3, 0.3, 0.3), (0.9, 0.9, 0.9)), 2),
    (((0.8, 0.8, 0.8), (0.2, 0.2, 0.2), (0.4, 0.4, -0.1)), 0),
    (((0.4, 0.4, -0.1), (0.8, 0.8, 0.8), (0.4, 0.4, -0.1)), 1),
    (((0.4, 0.4, -0.1), (0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), 2),
    (((0.6, 0.6, 0.6), (0.0, 0.0, 0.0), (0.2, 0.2, -0.3)), 0),
    (((0.2, 0.2, -0.3), (0.6, 0.6, 0.6), (0.2, 0.2, -0.3)), 1),
    (((0.2, 0.2, -0.3), (0.0, 0.0, 0.0), (0.6, 0.6, 0.6)), 2),
    (((0.4, 0.4, 0.4), (-0.2, -0.2, -0.2), (0.0, 0.0, -0.5)), 0),
    (((0.0, 0.0, -0.5), (0.4, 0.4, 0.4), (0.0, 0.0, -0.5)), 1),
    (((0.0, 0.0, -0.5), (-0.2, -0.2, -0.2), (0.4, 0.4, 0.4)), 2),
    (((0.2, 0.2, 0.2), (-0.4, -0.4, -0.4), (-0.2, -0.2, -0.7)), 0),
    (((-0.2, -0.2, -0.7), (0.2, 0.2, 0.2), (-0.2, -0.2, -0.7)), 1),
    (((-0.2, -0.2, -0.7), (-0.4, -0.4, -0.4), (0.2, 0.2, 0.2)), 2),
    (((0.0, 0.0, 0.0), (-0.6, -0.6, -0.6), (-0.4, -0.4, -0.9)), 0),
    (((-0.4, -0.4, -0.9), (0.0, 0.0, 0.0), (-0.4, -0.4, -0.9)), 1),
    (((-0.4, -0.4, -0.9), (-0.6, -0.6, -0.6), (0.0, 0.0, 0.0)), 2),
    (((-0.2, -0.2, -0.2), (-0.8, -0.8, -0.8), (-0.6, -0.6, -1.0)), 0),
    (((-0.6, -0.6, -1.0), (-0.2, -0.2, -0.2), (-0.6, -0.6, -1.0)), 1),
    (((-0.6, -0.6, -1.0), (-0.8, -0.8, -0.8), (-0.2, -0.2, -0.2)), 2),
    (
        (
            (0.9, 0.9, 0.9, 0.9),
            (0.3, 0.3, 0.3, 0.3),
            (0.5, 0.5, 0.5, 0.0),
            (0.3, 0.3, 0.3, 0.3),
        ),
        0,
    ),
    (
        (
            (0.5, 0.5, 0.5, 0.0),
            (0.9, 0.9, 0.9, 0.9),
            (0.5, 0.5, 0.5, 0.0),
            (0.3, 0.3, 0.3, 0.3),
        ),
        1,
    ),
    (
        (
            (0.5, 0.5, 0.5, 0.0),
            (0.3, 0.3, 0.3, 0.3),
            (0.9, 0.9, 0.9, 0.9),
            (0.3, 0.3, 0.3, 0.3),
        ),
        2,
    ),
)

CASE_IDS = [case.name for case in TACTICAL_CASES]


def tictactoe_search() -> MatchSearch:
    """Return a match search over tic-tac-toe with the solver as its evaluator."""
    return MatchSearch(
        MatchConfig(num_actions=BOARD_SIZE, playouts=PLAYOUTS),
        solver_wdl_evaluator(num_actions=BOARD_SIZE),
    )


@pytest.mark.parametrize("case", TACTICAL_CASES, ids=CASE_IDS)
def test_the_search_plays_an_optimal_move_in_a_sharp_position(
    case: TacticalCase,
) -> None:
    """T12: from a curated position, the returned action is one the solver calls
    optimal.

    Never "close to optimal": these positions are won, drawn or lost by a full
    point, so an implementation that picks a loser is off by the largest gap
    the game has.
    """
    board = case.state()
    result = tictactoe_search().search(MatchToyState(board))
    assert result.action in case.optimal
    assert bool(board.legal_mask()[result.action])


def test_a_self_played_game_is_drawn() -> None:
    """T12: both sides searching from the empty board must draw.

    The per-game half of the oracle, and the one that catches a search which
    is right on average and wrong in the sharp positions: a single bad move
    anywhere in nine plies loses the game.
    """
    game = TicTacToe()
    search = tictactoe_search()
    played: list[int] = []
    while not game.is_terminal():
        result = search.search(MatchToyState(game), ply=len(played))
        assert bool(game.legal_mask()[result.action])
        game.play(result.action)
        search.observe(result.action)
        played.append(result.action)
    assert game.winner() is None, f"the game was decided: {played}"
    assert len(played) == BOARD_SIZE
    # Both sides are the same search with the same exact values, so a losing
    # move by either would be punished by the other -- which is why "drawn"
    # is the assertion and a per-ply optimality check would only be a more
    # brittle way of saying the same thing.
    assert solve(TicTacToe()).value == 0.0


def test_the_frozen_tactical_tables_have_the_verdicts_they_claim() -> None:
    """T13's provenance half: the literals are re-checked against the solver.

    Two independent derivations of the same fact. If the hand-frozen best move
    and the exact minimax disagree, this fails *before* any search is judged
    by either.
    """
    assert len(TACTICAL_TABLES) >= 20
    for table, best_action in TACTICAL_TABLES:
        minima = [min(row) for row in table]
        assert minima[best_action] == max(minima)
        assert sum(1 for value in minima if value == minima[best_action]) == 1
        assert solve(TreeGame(table)).best_actions == (best_action,)
        # The margin the fixture rule promises, which is what makes the search
        # assertion below a consequence rather than a hope.
        others = [value for index, value in enumerate(minima) if index != best_action]
        assert minima[best_action] - max(others) >= 0.4 - 1e-12


@pytest.mark.parametrize(
    ("table", "best_action"),
    TACTICAL_TABLES,
    ids=[f"table{index}" for index in range(len(TACTICAL_TABLES))],
)
def test_the_search_finds_the_unique_best_move(
    table: tuple[tuple[float, ...], ...], best_action: int
) -> None:
    """T13: at 64 playouts, the search returns the solver's unique best move."""
    width = len(table)
    search = MatchSearch(
        MatchConfig(num_actions=width, playouts=PLAYOUTS),
        solver_wdl_evaluator(num_actions=width),
    )
    result = search.search(MatchToyState(TreeGame(table)))
    assert result.action == best_action
