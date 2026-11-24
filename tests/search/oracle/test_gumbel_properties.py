"""The invariants that hold for every position, budget and seed.

These are the properties the contract promises unconditionally, and each one
is here because its failure mode is silent:

* **full support** -- the improved policy must keep mass on every legal
  action, including the ones the root never sampled as candidates. A target
  that is zero outside the candidate set trains the network to believe the
  unvisited moves are unplayable; the entropy collapses over iterations and
  the loss curve never mentions it.
* **no root noise** -- Gumbel search draws its exploration from the Gumbel
  variables, and that is the entire point of the algorithm: no Dirichlet
  noise, no temperature schedule, no forced playouts. A config field offering
  any of those is a sign the AlphaZero machinery was carried over, so the
  absence is asserted structurally rather than left to a code review.
* **budget accounting** -- sequential halving that spends the wrong number of
  simulations still returns a plausible-looking answer. Off-by-one halving
  schedules are exactly the bug that hides behind "it plays fine".
* **determinism** -- same state, same config, same result, including when one
  search object is reused: an rng that carries over between calls makes every
  replay and every regression test unreproducible.
* **non-mutation** -- the caller's state has to come back untouched, or the
  game loop that called search plays on from a position the search invented.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

import numpy as np
import pytest
from solver import legal_actions, perfect_evaluator
from toygames import BOARD_SIZE, TicTacToe, TreeGame

gumbel = pytest.importorskip("komugi.search.gumbel")

#: Fields whose presence would mean AlphaZero's exploration machinery came
#: along for the ride. Matched as substrings, case-folded.
FORBIDDEN_FIELD_MARKERS: Final = ("dirichlet", "noise", "temperature", "forced")

#: The contract's field list; extra fields are allowed, missing ones are not.
REQUIRED_CONFIG_FIELDS: Final = frozenset(
    {"num_actions", "m", "n", "seed", "c_visit", "c_scale"}
)

#: X on 4 and 8, O on 0 and 2, X to move: five legal actions and four illegal
#: ones, enough of both to tell a "zero because illegal" apart from a "zero
#: because never sampled".
#:
#: Chosen for its *value spread* as much as its shape: nothing is won or lost
#: outright here, so the exact action values sit within a point of each other.
#: That matters for the full-support assertions -- sigma scales a Q-gap by
#: ``c_visit + max_visits``, so a two-point gap would push the loser's logit
#: past 130 nats below the leader and a correct implementation would round it
#: to a legitimate float32 zero. A test that reads underflow as entropy
#: collapse is a test that fails on correct code.
SPARSE = "O.O/.X./..X"

#: One empty cell and nothing decided yet: the early-return position.
ONE_MOVE_LEFT = "XOX/XOO/OX."

HAND_TABLE: Final[tuple[tuple[float, ...], ...]] = (
    (0.5, -0.2, 0.1),
    (-1.0, 0.9, 0.3),
    (0.2, 0.2, 0.2),
)


def _search(state: Any, *, actions: int, m: int, n: int, seed: int) -> Any:
    """Run one perfect-value search and return its result."""
    config = gumbel.GumbelConfig(num_actions=actions, m=m, n=n, seed=seed)
    return gumbel.GumbelSearch(perfect_evaluator, config).search(state)


# --------------------------------------------------------------------------
# full support
# --------------------------------------------------------------------------


def test_every_legal_action_keeps_mass_even_when_it_was_never_a_candidate() -> None:
    """Nine legal moves, two candidates: the other seven still get mass."""
    result = _search(TicTacToe(), actions=BOARD_SIZE, m=2, n=16, seed=0)
    target = result.policy_target

    assert target.shape == (BOARD_SIZE,)
    assert target.dtype == np.dtype(np.float32)
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-5)
    for action in range(BOARD_SIZE):
        assert float(target[action]) > 0.0, f"action {action} was zeroed out"


def test_the_target_is_zero_on_illegal_actions_and_only_there() -> None:
    """Exactly zero off the legal set, strictly positive on it."""
    state = TicTacToe.from_string(SPARSE)
    legal = set(legal_actions(state))
    assert legal == {1, 3, 5, 6, 7}

    result = _search(state, actions=BOARD_SIZE, m=2, n=16, seed=1)
    target = result.policy_target

    assert float(target.sum()) == pytest.approx(1.0, abs=1e-5)
    for action in range(BOARD_SIZE):
        if action in legal:
            assert float(target[action]) > 0.0, f"legal action {action} was zeroed"
        else:
            assert float(target[action]) == 0.0, f"illegal action {action} got mass"


@pytest.mark.parametrize("seed", (0, 2, 9))
@pytest.mark.parametrize("m", (2, 3, 9))
def test_the_target_is_a_distribution_at_every_candidate_budget(
    m: int, seed: int
) -> None:
    """Non-negative, normalised, and never a NaN sneaking through the softmax."""
    result = _search(TicTacToe(), actions=BOARD_SIZE, m=m, n=24, seed=seed)
    target = result.policy_target

    assert np.all(np.isfinite(target))
    assert np.all(target >= 0.0)
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= float(result.root_value) <= 1.0


# --------------------------------------------------------------------------
# no root noise
# --------------------------------------------------------------------------


def test_the_config_carries_no_root_noise_machinery() -> None:
    """Structural: the fields AlphaZero needs and Gumbel search must not have.

    ``gumbel_scale`` is the one sanctioned exception, mandated by DESIGN §3.1's
    reduced-scale fast mode (KOM-029; it evaded the lexical markers by name
    alone, per the KOM-029 review F2). It lives in the explicit allowlist below
    so the next noise-adjacent field has to argue its way in here rather than
    slip through on naming.

    **What it is, corrected.** This docstring used to call it "a
    noise-magnitude knob". It multiplies the noise array, but the distribution
    that induces is a root **temperature**: ``argsort_desc(s·γ + L) ==
    argsort_desc(γ + L/s)``, so the root samples ``m`` actions without
    replacement from ``softmax(L/s)`` (KOM-036 §2.2, which retires the same
    framing in ``gumbel.py``'s own module docstring). The exception stands and
    the allowlist mechanism below is untouched; only the description of what is
    being excepted changes.
    """
    names = {field.name for field in dataclasses.fields(gumbel.GumbelConfig)}
    sanctioned = {"gumbel_scale"}

    assert REQUIRED_CONFIG_FIELDS <= names
    for name in names - sanctioned:
        for marker in FORBIDDEN_FIELD_MARKERS:
            assert marker not in name.lower(), f"{name!r} looks like {marker} knob"


def test_the_config_defaults_are_the_paper_constants() -> None:
    """c_visit and c_scale default to the values the contract fixes."""
    config = gumbel.GumbelConfig(num_actions=BOARD_SIZE, m=4, n=16, seed=0)

    assert config.c_visit == 50.0
    assert config.c_scale == 1.0


def test_config_and_result_are_frozen() -> None:
    """Neither the settings nor the verdict can be edited after the fact."""
    config = gumbel.GumbelConfig(num_actions=BOARD_SIZE, m=4, n=16, seed=0)
    result = _search(TicTacToe(), actions=BOARD_SIZE, m=4, n=16, seed=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, "seed", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "action", 0)


# --------------------------------------------------------------------------
# budget accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", (8, 13, 16, 32, 50))
@pytest.mark.parametrize("m", (2, 3, 4, 5, 7))
def test_the_simulation_budget_is_spent_exactly(m: int, n: int) -> None:
    """Every (m, n) in the grid: n simulations, n visits, not one more.

    The awkward pairs are the point -- m that is not a power of two, and n
    that no halving schedule divides evenly. Sequential halving has to absorb
    the remainder somewhere, and "somewhere" must still add up to n.
    """
    result = _search(TicTacToe(), actions=BOARD_SIZE, m=m, n=n, seed=3)

    assert result.sims_used == n
    assert result.visit_counts.shape == (BOARD_SIZE,)
    assert result.visit_counts.dtype == np.dtype(np.int64)
    assert int(result.visit_counts.sum()) == n
    assert np.all(result.visit_counts >= 0)


def test_a_candidate_budget_larger_than_the_position_is_clipped() -> None:
    """m above the legal count spends the full budget on what is there."""
    state = TicTacToe.from_string(SPARSE)  # five legal actions, m asks for eight

    result = _search(state, actions=BOARD_SIZE, m=8, n=24, seed=0)

    assert result.sims_used == 24
    assert int(result.visit_counts.sum()) == 24
    assert result.action in legal_actions(state)
    for action in range(BOARD_SIZE):
        if action not in legal_actions(state):
            assert int(result.visit_counts[action]) == 0


def test_a_forced_move_returns_without_searching() -> None:
    """One legal action is not a search problem, and must not cost n sims."""
    state = TicTacToe.from_string(ONE_MOVE_LEFT)
    assert legal_actions(state) == (8,)

    result = _search(state, actions=BOARD_SIZE, m=4, n=32, seed=0)

    assert result.action == 8
    assert float(result.policy_target[8]) == pytest.approx(1.0, abs=1e-5)
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)
    for action in range(BOARD_SIZE):
        if action != 8:
            assert float(result.policy_target[action]) == 0.0
    assert result.sims_used <= 1
    assert int(result.visit_counts.sum()) <= result.sims_used


def test_a_three_action_game_is_searched_by_the_same_rules() -> None:
    """Nothing above is a nine-action coincidence."""
    result = _search(TreeGame(HAND_TABLE), actions=3, m=2, n=12, seed=0)

    assert result.sims_used == 12
    assert int(result.visit_counts.sum()) == 12
    assert result.policy_target.shape == (3,)
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert result.action in (0, 1, 2)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", (0, 4, 99))
def test_the_same_state_and_config_give_the_same_result(seed: int) -> None:
    """Bit-for-bit repeatable: the seed is the only source of randomness."""
    state = TicTacToe.from_string(SPARSE)

    first = _search(state, actions=BOARD_SIZE, m=3, n=24, seed=seed)
    second = _search(state.clone(), actions=BOARD_SIZE, m=3, n=24, seed=seed)

    assert first.action == second.action
    assert first.sims_used == second.sims_used
    assert first.root_value == second.root_value
    assert np.array_equal(first.policy_target, second.policy_target)
    assert np.array_equal(first.visit_counts, second.visit_counts)


def test_reusing_one_search_object_does_not_carry_the_rng_forward() -> None:
    """The second call on the same object must repeat the first, exactly.

    An rng constructed once and drawn from per search makes the same position
    answer differently on the second visit, which is unreproducible in a way
    that only shows up as an unreplayable self-play game months later.
    """
    config = gumbel.GumbelConfig(num_actions=BOARD_SIZE, m=3, n=24, seed=17)
    search = gumbel.GumbelSearch(perfect_evaluator, config)
    state = TicTacToe.from_string(SPARSE)

    first = search.search(state)
    second = search.search(state)

    assert first.action == second.action
    assert np.array_equal(first.policy_target, second.policy_target)
    assert np.array_equal(first.visit_counts, second.visit_counts)


def test_other_seeds_stay_inside_the_contract() -> None:
    """Seeds may disagree about the move; none may break the invariants.

    Deliberately no assertion that the actions *differ*: with exact values the
    Q-gaps can dominate every Gumbel draw, and demanding disagreement would be
    demanding noise the algorithm does not promise.
    """
    state = TicTacToe()
    legal = set(legal_actions(state))

    for seed in range(8):
        result = _search(state, actions=BOARD_SIZE, m=4, n=24, seed=seed)

        assert result.action in legal
        assert result.sims_used == 24
        assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# non-mutation
# --------------------------------------------------------------------------


def test_search_leaves_the_callers_board_alone() -> None:
    """Cells, side to move and legal mask all come back as they went in."""
    state = TicTacToe.from_string(SPARSE)
    cells, to_play = state.cells, state.to_play()
    mask = state.legal_mask().copy()

    _search(state, actions=BOARD_SIZE, m=3, n=32, seed=2)

    assert state.cells == cells
    assert state.to_play() == to_play
    assert np.array_equal(state.legal_mask(), mask)
    assert not state.is_terminal()


def test_search_leaves_the_callers_tree_game_alone() -> None:
    """Same promise for a state whose history is its identity."""
    state = TreeGame(HAND_TABLE)

    _search(state, actions=3, m=3, n=12, seed=2)

    assert state.moves == ()
    assert state.to_play() == 0
