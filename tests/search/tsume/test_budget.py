"""The budget is a real limit, and running out of it is not a verdict.

``UNKNOWN`` is the only answer a budgeted searcher is allowed to invent. The
properties here say so precisely: a budget too small to prove anything
produces UNKNOWN *and* stops inside the budget; enlarging the budget turns
UNKNOWN into MATE and never into NO_MATE; and two identical calls return
identical results, so none of the above is being decided by hash order or
wall-clock.

The fixtures are chosen by their measured node counts, and the assertions
name the verdict expected at each budget -- a monotonicity test whose small
run already succeeded would be testing nothing at all.
"""

from __future__ import annotations

import pytest
from tsume_fixtures import FRIEND

from komugi.search.tsume import solve_tsume

#: Measured: 568 node expansions to prove. Any budget near ten is hopeless.
DEEP = next(f for f in FRIEND if f.name == "F14-5move")
#: Measured: 34 node expansions to prove, so it straddles a 5 / 50 pair.
STRADDLE = next(f for f in FRIEND if f.name == "F13-5move")

TINY_BUDGET = 10


def test_a_tiny_budget_gives_unknown_and_stops_inside_it() -> None:
    result = solve_tsume(DEEP.fen, DEEP.tsume_plies, node_budget=TINY_BUDGET)
    assert result.verdict == "UNKNOWN"
    assert result.nodes <= TINY_BUDGET
    assert result.line == ()


def test_unknown_is_not_a_verdict_flip() -> None:
    """The budgeted answer never contradicts the unbudgeted one."""
    starved = solve_tsume(DEEP.fen, DEEP.tsume_plies, node_budget=TINY_BUDGET)
    full = solve_tsume(DEEP.fen, DEEP.tsume_plies)
    assert starved.verdict == "UNKNOWN"
    assert full.verdict == "MATE"


def test_ten_times_the_budget_turns_unknown_into_mate() -> None:
    """The monotonicity pair: both ends asserted, so neither can be vacuous."""
    small = solve_tsume(STRADDLE.fen, STRADDLE.tsume_plies, node_budget=5)
    large = solve_tsume(STRADDLE.fen, STRADDLE.tsume_plies, node_budget=50)
    assert small.verdict == "UNKNOWN"
    assert small.nodes <= 5
    assert large.verdict == "MATE"
    assert len(large.line) == STRADDLE.tsume_plies


@pytest.mark.parametrize("budget", [1, 3, 7, 25])
def test_nodes_never_exceed_the_budget(budget: int) -> None:
    result = solve_tsume(DEEP.fen, DEEP.tsume_plies, node_budget=budget)
    assert result.nodes <= budget


def test_two_identical_runs_are_identical() -> None:
    """Determinism, on a puzzle deep enough for search order to matter."""
    first = solve_tsume(DEEP.fen, DEEP.tsume_plies)
    second = solve_tsume(DEEP.fen, DEEP.tsume_plies)
    assert first == second
    assert first.line == second.line
    assert first.nodes == second.nodes


def test_a_no_mate_verdict_also_costs_nodes() -> None:
    """NO_MATE is proved, not assumed: it has to expand the tree to say it."""
    result = solve_tsume(STRADDLE.fen, STRADDLE.tsume_plies - 2)
    assert result.verdict == "NO_MATE"
    assert result.nodes > 1
