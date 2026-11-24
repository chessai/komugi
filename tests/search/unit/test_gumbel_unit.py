"""Unit tests for the Gumbel search, on a toy state small enough to reason about.

The properties pinned here are the ones that are cheap to get subtly wrong and
expensive to notice later in a campaign:

* the simulation budget is spent *exactly*, so a "32 simulations" run is 32
  simulations and not 30 with two lost to integer division;
* the policy target has full legal support, because the failure this search
  exists to avoid is a target that is mostly zeros at branching ~229;
* the input state comes back untouched, because the self-play loop holds one
  state and searches from it move after move.

The toy is deliberately not a game: a fixed action set, a fixed depth, and a
terminal value that is a pure function of the move history. That makes every
assertion below a statement about the search rather than about a rules engine.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import (
    GumbelConfig,
    GumbelSearch,
    SearchResult,
    SearchState,
)
from komugi.search import gumbel


class ToyState:
    """A depth-limited toy position: fixed legal set, alternating movers.

    `constant_value` overrides the history-derived terminal value, which is
    how the negamax test pins a known sign at every leaf.
    """

    def __init__(
        self,
        *,
        num_actions: int = 6,
        legal: tuple[int, ...] = (0, 1, 2),
        max_depth: int = 2,
        constant_value: float | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.legal = legal
        self.max_depth = max_depth
        self.constant_value = constant_value
        self.history: tuple[int, ...] = ()

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set, or nothing at all once terminal."""
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        if not self.is_terminal():
            for action in self.legal:
                mask[action] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        if not bool(self.legal_mask()[action]):
            raise AssertionError(
                f"search played illegal action {action} at history {self.history}"
            )
        self.history = (*self.history, action)

    def clone(self) -> ToyState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        copy = ToyState(
            num_actions=self.num_actions,
            legal=self.legal,
            max_depth=self.max_depth,
            constant_value=self.constant_value,
        )
        copy.history = self.history
        return copy

    def is_terminal(self) -> bool:
        """Return whether `max_depth` plies have been played."""
        return len(self.history) >= self.max_depth

    def terminal_value(self) -> float:
        """Return a value in ``[-1, 1]`` for the player to move here."""
        if self.constant_value is not None:
            return self.constant_value
        return ((sum(self.history) % 5) - 2) / 2.0

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` vector whose ``w - l`` is :meth:`terminal_value`.

        ``w = (1 + v) / 2``, ``d = 0``, ``l = (1 - v) / 2`` -- a probability
        vector for every ``v`` in ``[-1, 1]``, a genuine **one-hot** at the ends
        (a mated mover, ``v = -1``, reads ``(0, 0, 1)``), and by construction
        consistent with the scalar the same leaf backs up. That consistency is
        what lets the KOM-173 identity ``root_wdl[0] - root_wdl[2] ==
        root_value`` be asserted on trees whose leaves are *terminal* rather
        than only on trees the stub evaluated: a terminal leaf that invented an
        unrelated vector would break the identity for a reason that says nothing
        about the search.
        """
        value = self.terminal_value()
        return np.array(
            [(1.0 + value) / 2.0, 0.0, (1.0 - value) / 2.0], dtype=np.float64
        )

    def to_play(self) -> int:
        """Return the mover, alternating every ply."""
        return len(self.history) % 2


class ToyEvaluator:
    """A deterministic stand-in for the network, plus a call log.

    Logits are a pure function of the history and are *not* uniform, so a
    prior-shaped target is distinguishable from a flat one. The value flips
    with the mover, which is what makes a sign error in the backup visible.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[tuple[int, ...]] = []

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, value)` for `state`, recording that it was asked."""
        assert isinstance(state, ToyState), "evaluator was handed a foreign state"
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        self.calls += 1
        self.seen.append(state.history)
        base = sum(state.history) + 1
        logits = np.array(
            [0.4 * ((action * 7 + base) % 5) for action in range(state.num_actions)],
            dtype=np.float32,
        )
        value = math.tanh(0.3 * base) * (1.0 if state.to_play() == 0 else -1.0)
        return logits, value


def _run(
    state: ToyState,
    *,
    m: int,
    n: int,
    seed: int = 7,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    evaluator: ToyEvaluator | None = None,
) -> SearchResult:
    """Search `state` with a fresh evaluator unless one is supplied."""
    search = GumbelSearch(
        evaluator if evaluator is not None else ToyEvaluator(),
        GumbelConfig(
            num_actions=state.num_actions,
            m=m,
            n=n,
            seed=seed,
            c_visit=c_visit,
            c_scale=c_scale,
        ),
    )
    return search.search(state)


def _copy_of(result: SearchResult) -> SearchResult:
    """Return a field-for-field copy of `result`, with its own arrays."""
    return SearchResult(
        action=result.action,
        policy_target=result.policy_target.copy(),
        root_prior_logits=result.root_prior_logits.copy(),
        root_value=result.root_value,
        visit_counts=result.visit_counts.copy(),
        sims_used=result.sims_used,
        root_wdl=result.root_wdl.copy(),
        root_wdl_pi=result.root_wdl_pi.copy(),
        played_wdl=result.played_wdl.copy(),
    )


@pytest.mark.parametrize(
    ("m", "n"),
    [
        (1, 1),
        (1, 9),
        (2, 2),
        (2, 7),
        (3, 3),
        (3, 4),
        (3, 16),
        (4, 5),
        (4, 1),
        (5, 17),
        (5, 32),
        (8, 3),
        (8, 32),
        (3, 0),
    ],
)
def test_the_budget_is_spent_exactly(m: int, n: int) -> None:
    """`sims_used` is `n` on the nose: halving leftovers are redistributed.

    The failure this catches is the natural one -- allocating
    ``n // (rounds * survivors)`` per candidate and quietly dropping the
    remainder, which turns a 32-simulation config into a 24-simulation search.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)

    result = _run(state, m=m, n=n)

    assert result.sims_used == n
    assert int(result.visit_counts.sum()) == n


def test_visit_counts_land_only_on_legal_actions() -> None:
    """No simulation is ever spent on an action outside the mask."""
    state = ToyState(num_actions=6, legal=(1, 3, 5), max_depth=2)

    result = _run(state, m=3, n=24)

    illegal = [a for a in range(state.num_actions) if a not in state.legal]
    assert all(result.visit_counts[a] == 0 for a in illegal)
    assert result.visit_counts.dtype == np.int64
    assert result.visit_counts.shape == (state.num_actions,)


def test_m_is_clipped_to_the_number_of_legal_actions() -> None:
    """A config sized for the opening still searches a three-move position.

    With ``m`` far above the legal count the search must fall back to "every
    legal action is a candidate" rather than sampling duplicates or running
    off the end of the ordering.
    """
    state = ToyState(num_actions=6, legal=(0, 2, 4), max_depth=2)

    result = _run(state, m=64, n=30)

    assert result.sims_used == 30
    assert all(result.visit_counts[a] > 0 for a in state.legal)


def test_a_candidate_set_smaller_than_the_legal_set_is_respected() -> None:
    """`m` really does bound the candidates: only they are ever simulated."""
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)

    result = _run(state, m=2, n=12)

    visited = [a for a in range(state.num_actions) if result.visit_counts[a] > 0]
    assert len(visited) <= 2


def test_policy_target_is_a_distribution_over_the_legal_actions() -> None:
    """Shape, dtype and normalisation, all of which the trainer relies on."""
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)

    result = _run(state, m=4, n=16)

    assert result.policy_target.shape == (state.num_actions,)
    assert result.policy_target.dtype == np.float32
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert bool((result.policy_target >= 0.0).all())


def test_illegal_actions_get_exactly_zero_target_mass() -> None:
    """Zero on the illegal actions, and only there."""
    state = ToyState(num_actions=6, legal=(1, 3, 5), max_depth=2)

    result = _run(state, m=3, n=16)

    for action in range(state.num_actions):
        if action in state.legal:
            assert result.policy_target[action] > 0.0
        else:
            assert result.policy_target[action] == 0.0


def test_legal_non_candidates_keep_nonzero_prior_mass() -> None:
    """The anti-collapse property: the target is not renormalised over candidates.

    A target built by normalising visits (or by softmaxing over the ``m``
    sampled actions alone) would put a hard zero on every legal action the
    root happened not to sample. At branching ~229 with ``m = 16`` that is
    most of the board, and it teaches the network that legal moves are
    impossible. Value completion keeps the prior tail alive instead.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)

    result = _run(state, m=2, n=7)

    unvisited = [a for a in state.legal if result.visit_counts[a] == 0]
    assert len(unvisited) >= 3
    for action in unvisited:
        assert result.policy_target[action] > 0.0


def test_the_returned_action_is_a_sampled_candidate_and_is_legal() -> None:
    """The move played is one the search actually looked at."""
    state = ToyState(num_actions=6, legal=(1, 3, 5), max_depth=2)

    result = _run(state, m=3, n=20)

    assert result.action in state.legal
    assert result.visit_counts[result.action] > 0


def test_identical_inputs_give_an_identical_result() -> None:
    """Same state, same config, same seed -- byte-for-byte the same search."""
    first = _run(ToyState(legal=(0, 1, 2), max_depth=2), m=3, n=17, seed=1234)
    second = _run(ToyState(legal=(0, 1, 2), max_depth=2), m=3, n=17, seed=1234)

    assert first == second
    assert np.array_equal(first.policy_target, second.policy_target)
    assert np.array_equal(first.visit_counts, second.visit_counts)
    assert first.root_value == second.root_value


def test_the_seed_actually_reaches_the_root_draw() -> None:
    """Different seeds must be able to produce different root samples.

    Run with ``c_scale = 0`` and no simulations, so the returned action is the
    Gumbel top-1 sample and nothing else. A search that ignored its seed --
    or drew from a global RNG once at import -- would return one action for
    every seed here.
    """
    actions = {
        _run(ToyState(legal=(0, 1, 2), max_depth=2), m=1, n=0, seed=s, c_scale=0.0).action
        for s in range(40)
    }

    assert len(actions) > 1


def test_search_does_not_mutate_the_state_it_is_given() -> None:
    """The self-play loop holds one state across a whole game."""
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3), max_depth=3)
    state.play(2)
    before = state.history

    result = _run(state, m=3, n=21)

    assert state.history == before
    assert state.history == (2,)
    assert not state.is_terminal()
    assert result.sims_used == 21


def test_a_forced_move_returns_without_searching() -> None:
    """One legal action: no simulations, a one-hot target, a real value."""
    state = ToyState(num_actions=6, legal=(4,), max_depth=2)
    evaluator = ToyEvaluator()

    result = _run(state, m=8, n=64, evaluator=evaluator)

    assert result.action == 4
    assert result.sims_used == 0
    assert int(result.visit_counts.sum()) == 0
    assert float(result.policy_target[4]) == pytest.approx(1.0)
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert evaluator.calls == 1
    assert state.history == ()


def test_a_state_with_no_legal_actions_is_an_error() -> None:
    """No move and no distribution exists; failing loudly beats inventing one."""
    state = ToyState(num_actions=6, legal=(0, 1), max_depth=0)

    with pytest.raises(ValueError, match="no legal actions"):
        _run(state, m=2, n=4)


def test_child_values_are_negated_on_the_way_up() -> None:
    """Negamax: leaves that are wins for the *child's* mover sink the root.

    Every child here is terminal with ``terminal_value = +1``, i.e. a win for
    the player to move at the child -- which is the opponent of the root's
    mover. A root value near ``-1`` is the only correct reading; a backup that
    forgot to negate would report ``+1``.
    """
    losing = ToyState(num_actions=4, legal=(0, 1, 2), max_depth=1, constant_value=1.0)
    winning = ToyState(num_actions=4, legal=(0, 1, 2), max_depth=1, constant_value=-1.0)

    assert _run(losing, m=3, n=32).root_value < -0.9
    assert _run(winning, m=3, n=32).root_value > 0.9


def test_root_value_is_the_network_value_when_nothing_was_simulated() -> None:
    """With no visits there is nothing to mix in, so ``v_mix`` is just ``v``."""
    state = ToyState(num_actions=6, legal=(0, 1, 2), max_depth=2)
    evaluator = ToyEvaluator()

    result = _run(state, m=3, n=0, evaluator=evaluator)

    # The toy's root value: base = 1, and the root's mover is player 0.
    assert result.root_value == pytest.approx(math.tanh(0.3), abs=1e-6)
    assert result.sims_used == 0
    assert evaluator.calls == 1


def test_zero_budget_still_yields_the_prior_as_a_target() -> None:
    """``n = 0`` degrades to the raw policy, not to a degenerate one-hot."""
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3), max_depth=2)

    result = _run(state, m=4, n=0)

    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)
    assert all(result.policy_target[a] > 0.0 for a in state.legal)
    assert int(result.visit_counts.sum()) == 0


def test_terminal_children_are_scored_by_the_game_not_the_network() -> None:
    """A depth-1 toy has only terminal children: one evaluation, at the root."""
    state = ToyState(num_actions=4, legal=(0, 1, 2), max_depth=1)
    evaluator = ToyEvaluator()

    result = _run(state, m=3, n=24, evaluator=evaluator)

    assert evaluator.calls == 1
    assert evaluator.seen == [()]
    assert result.sims_used == 24


def test_config_carries_no_noise_temperature_or_forced_playout_knob() -> None:
    """The absent AlphaZero fixtures are absent from the API, not defaulted off.

    A ``dirichlet_alpha = 0.0`` default would be a switch someone flips in six
    months; the point of pure Gumbel is that there is no switch.

    ``gumbel_scale`` is in the list and is not a counterexample, but **not for
    the reason this docstring used to give.** It used to say the field "cannot
    divide the logits the way a temperature would"; KOM-036 §2.2 shows that it
    does exactly that to the induced distribution -- ``argsort_desc(s·γ + L) ==
    argsort_desc(γ + L/s)``, so the root samples ``m`` actions without
    replacement from ``softmax(L/s)``. What makes it not a counterexample is
    that it is the **sanctioned** form of the concept rather than a second one:
    one field, applied at one site (the draw), provably unable to reach
    ``_improved_policy`` and therefore unable to change the recorded label's
    formula. There is still no ``root_temperature`` field, because there does
    not need to be.

    This node's assertion -- the exact field set -- is unchanged by that
    correction, and deliberately so: it is a tripwire against a *new* field,
    which is a different claim from anything §2.2 argues. ``proof_mode`` and
    ``discovery_mode`` are likewise in the list and likewise not counterexamples:
    they are the sanctioned, versioned switches of KOM-221 and KOM-220, validated
    as a pair by ``komugi.search.modes`` and recorded in the provenance identity,
    not a temperature and not a noise knob.
    """
    names = {field.name for field in dataclasses.fields(GumbelConfig)}

    assert names == {
        "num_actions",
        "m",
        "n",
        "seed",
        "c_visit",
        "c_scale",
        "gumbel_scale",
        "proof_mode",
        "discovery_mode",
    }


def test_nonsensical_configs_are_rejected() -> None:
    """A config that cannot describe a search raises where it is built."""
    with pytest.raises(ValueError):
        GumbelConfig(num_actions=0, m=1, n=1, seed=0)
    with pytest.raises(ValueError):
        GumbelConfig(num_actions=4, m=0, n=1, seed=0)
    with pytest.raises(ValueError):
        GumbelConfig(num_actions=4, m=1, n=-1, seed=0)
    with pytest.raises(ValueError):
        GumbelConfig(num_actions=4, m=1, n=1, seed=0, c_scale=math.inf)


def test_a_mask_of_the_wrong_width_is_an_error() -> None:
    """The action space is fixed; a mask that disagrees is a wiring bug."""
    state = ToyState(num_actions=5, legal=(0, 1, 2), max_depth=2)
    search = GumbelSearch(
        ToyEvaluator(), GumbelConfig(num_actions=6, m=2, n=4, seed=0)
    )

    with pytest.raises(ValueError, match="legal_mask"):
        search.search(state)


def test_results_compare_by_value_across_their_arrays() -> None:
    """`==` on two results answers rather than raising on the numpy fields."""
    result = _run(ToyState(legal=(0, 1, 2), max_depth=2), m=3, n=12)
    same = _copy_of(result)
    bumped = dataclasses.replace(
        _copy_of(result), visit_counts=result.visit_counts + 1
    )
    other_prior = dataclasses.replace(
        _copy_of(result), root_prior_logits=result.root_prior_logits + 1.0
    )
    other_wdl = dataclasses.replace(
        _copy_of(result),
        root_wdl=np.array([0.5, 0.25, 0.25], dtype=np.float32),
    )

    assert result == same
    assert result != bumped
    # Every array field participates, the newest one included: an ``__eq__``
    # that simply never learned about ``root_prior_logits`` would call two
    # results of two different priors equal.
    assert result != other_prior
    # The KOM-173 readouts too. This result's own are NaN (a two-tuple stub),
    # and ``same`` compares equal to it anyway: "no vector was carried" is a
    # state two identical searches agree on, which plain ``array_equal``
    # without ``equal_nan`` would call a difference.
    assert bool(np.isnan(result.root_wdl).all())
    assert result != other_wdl


def test_the_result_is_frozen() -> None:
    """A result cannot be edited after the search that justified it."""
    result = _run(ToyState(legal=(0, 1, 2), max_depth=2), m=3, n=8)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, "action", 0)


class PeakedEvaluator:
    """A near-one-hot policy over a value that says the position is lost.

    That combination is not exotic -- it is what a network reports in a
    position it has resolved -- and it is the regime in which the improved
    policy's score gap runs to hundreds of nats.
    """

    def __init__(
        self,
        *,
        num_actions: int,
        peak: int = 0,
        height: float = 120.0,
        value: float = -0.99,
    ) -> None:
        self.num_actions = num_actions
        self.peak = peak
        self.height = height
        self.value = value

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return the peaked logits and the fixed value, whatever `state` is."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        logits = np.zeros(self.num_actions, dtype=np.float32)
        logits[self.peak] = self.height
        return logits, self.value


def test_every_legal_action_keeps_mass_in_a_decided_position() -> None:
    """A float32 cast must not flush a legal action's target mass to zero.

    The engineered regime, on the *default* ``c_visit``/``c_scale``: every
    child is terminal and lost for its own mover, so the two sampled
    candidates back up ``q = +1`` while the four unsampled legal actions are
    completed with ``v_mix = (-0.99 + 32) / 33 ~ 0.94``. With ``max N = 24``
    the ``sigma`` scale is ``50 + 24 = 74``, and the peaked logit adds 120
    nats on top of that, so the improved policy's score gap between the peak
    and every other legal action is ~120-125 nats. ``exp(-120) ~ 8e-53`` is
    far below the smallest float32 subnormal ``1.4e-45``, so casting the
    float64 softmax straight down turns five legal actions into exact zeros
    -- and :attr:`SearchResult.policy_target` is zero *only* on illegals.
    """
    state = ToyState(
        num_actions=8, legal=(0, 1, 2, 3, 4, 5), max_depth=1, constant_value=-1.0
    )
    search = GumbelSearch(
        PeakedEvaluator(num_actions=8),
        GumbelConfig(num_actions=8, m=2, n=32, seed=3),
    )

    result = search.search(state)

    assert result.policy_target.dtype == np.float32
    for action in range(state.num_actions):
        if action in state.legal:
            assert result.policy_target[action] > 0.0, f"legal action {action} zeroed"
        else:
            assert result.policy_target[action] == 0.0
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)


class BrokenLogitEvaluator:
    """An evaluator that poisons exactly one entry of an otherwise flat policy."""

    def __init__(self, *, num_actions: int, action: int, bad_logit: float) -> None:
        self.num_actions = num_actions
        self.action = action
        self.bad_logit = bad_logit

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return flat logits with one poisoned entry, and a neutral value."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        logits = np.zeros(self.num_actions, dtype=np.float32)
        logits[self.action] = self.bad_logit
        return logits, 0.0


def test_a_nan_logit_is_rejected_where_it_entered() -> None:
    """A NaN on a legal action poisons the whole target, and does it silently.

    Unchecked, the masked softmax propagates it: every entry of
    ``policy_target`` comes back NaN, the search still returns a move, and the
    first thing that notices is the loss going to NaN some hours into a run.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2), max_depth=2)
    search = GumbelSearch(
        BrokenLogitEvaluator(num_actions=6, action=1, bad_logit=math.nan),
        GumbelConfig(num_actions=6, m=3, n=8, seed=0),
    )

    with pytest.raises(ValueError, match="non-finite logits"):
        search.search(state)


def test_a_minus_inf_logit_cannot_admit_an_illegal_action() -> None:
    """``-inf`` on a legal action ties it with the masked-out illegal ones.

    The candidate ordering is a stable ``argsort`` of ``gumbel + logits`` with
    the illegal entries set to ``-inf``. A legal ``-inf`` is indistinguishable
    from them, so here -- two legal actions, ``m = 2``, the poisoned one at
    index 5 -- the tie block hands the search index 1, which is illegal, and
    it tries to play it.
    """
    state = ToyState(num_actions=6, legal=(0, 5), max_depth=2)
    search = GumbelSearch(
        BrokenLogitEvaluator(num_actions=6, action=5, bad_logit=-math.inf),
        GumbelConfig(num_actions=6, m=2, n=8, seed=0),
    )

    with pytest.raises(ValueError, match="non-finite logits"):
        search.search(state)


class FlatEvaluator:
    """An evaluator with no opinion: uniform logits and a value of zero.

    The backup-sign tests need every sign they read at the root to have come
    from a terminal value, so any prior tilt or nonzero value would only blur
    what they are pinning.
    """

    def __init__(self, *, num_actions: int) -> None:
        self.num_actions = num_actions

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return uniform logits and a value of zero, whatever `state` is."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        return np.zeros(self.num_actions, dtype=np.float32), 0.0


class SignToyState:
    """A two-ply toy built so the root's Q-signs are hand-computable.

    Two knobs, each load-bearing for exactly one test:

    * `passes_first`: ply 0 is a *pass*, leaving the mover unchanged, so the
      root and its child share a ``to_play`` and the edge between them must
      not negate. Without it the mover alternates every ply as usual.
    * `leaf_values`: the terminal value at the leaf, keyed by the *root*
      action and read in the leaf mover's frame. Making it independent of the
      reply is what makes the optimal line something a reader can check.
    """

    def __init__(
        self,
        *,
        num_actions: int = 4,
        legal: tuple[int, ...] = (0, 1),
        passes_first: bool = False,
        leaf_values: dict[int, float] | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.legal = legal
        self.passes_first = passes_first
        self.leaf_values = (
            {0: -1.0, 1: -1.0} if leaf_values is None else dict(leaf_values)
        )
        self.history: tuple[int, ...] = ()

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set, or nothing at all once terminal."""
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        if not self.is_terminal():
            for action in self.legal:
                mask[action] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        if not bool(self.legal_mask()[action]):
            raise AssertionError(
                f"search played illegal action {action} at history {self.history}"
            )
        self.history = (*self.history, action)

    def clone(self) -> SignToyState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        copy = SignToyState(
            num_actions=self.num_actions,
            legal=self.legal,
            passes_first=self.passes_first,
            leaf_values=self.leaf_values,
        )
        copy.history = self.history
        return copy

    def is_terminal(self) -> bool:
        """Return whether both plies have been played."""
        return len(self.history) >= 2

    def terminal_value(self) -> float:
        """Return the leaf value the root action dictated, for the mover here."""
        return self.leaf_values[self.history[0]]

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the zero-draw vector whose ``w - l`` is :meth:`terminal_value`."""
        value = self.terminal_value()
        return np.array(
            [(1.0 + value) / 2.0, 0.0, (1.0 - value) / 2.0], dtype=np.float64
        )

    def to_play(self) -> int:
        """Return the mover; ply 0 leaves it alone when `passes_first` is set."""
        plies = len(self.history)
        if self.passes_first:
            plies = max(plies - 1, 0)
        return plies % 2


def test_a_pass_ply_is_not_negated_on_the_way_up() -> None:
    """Kills the "negate unconditionally, once per edge" mutant.

    Every other test in this file alternates the mover every ply, so
    ``below.to_play != parent.to_play`` is always true and the conditional is
    never seen declining to fire. Here the ``to_play`` schedule is ``0, 0, 1``:
    player 0 passes and then moves again. The leaf is a loss for the player to
    move there (player 1), so it must reach the root as ``+1`` for player 0 --
    one negation across two edges, not two. The mutant negates on the pass
    edge as well, cancels itself out, and reports ``root_value ~ -0.89``.
    """
    state = SignToyState(num_actions=4, legal=(0, 1), passes_first=True)
    search = GumbelSearch(
        FlatEvaluator(num_actions=4),
        GumbelConfig(num_actions=4, m=2, n=32, seed=5),
    )

    result = search.search(state)

    assert result.root_value > 0.5


def test_the_root_child_q_signs_follow_the_hand_computed_line() -> None:
    """Kills the "negate once per path" mutant (and its never-negate cousin).

    Movers alternate, so the schedule is ``0, 1, 0`` and both edges negate.
    Action 0 leads to a leaf worth ``+1`` to the leaf's mover, who is player 0
    -- the root's own mover -- so it must come back to the root as ``+1``;
    action 1 leads to ``-1`` and must come back as ``-1``. With ``max N = 24``
    the ``sigma`` gap between them is ``74 * (0.958 + 0.875) ~ 136`` nats, so
    the improved policy is all but a one-hot on action 0. A backup that
    negated once for the whole path instead of once per edge would swap the
    two signs and put the target on action 1.
    """
    state = SignToyState(num_actions=4, legal=(0, 1), leaf_values={0: 1.0, 1: -1.0})
    search = GumbelSearch(
        FlatEvaluator(num_actions=4),
        GumbelConfig(num_actions=4, m=2, n=32, seed=11),
    )

    result = search.search(state)

    assert result.action == 0
    assert float(result.policy_target[0]) > 0.99
    assert float(result.policy_target[1]) < 1e-6
    assert float(result.policy_target[1]) > 0.0
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# gumbel_scale: DESIGN 3.1's reduced-noise fast search
# --------------------------------------------------------------------------


class RampEvaluator:
    """Strictly ordered logits with a known gap, and no opinion on value.

    ``logits[a] = spacing * a``, so "the top ``m`` logits" is the top ``m``
    action indices and nothing else -- which is what makes a claim about the
    noise having stopped mattering checkable rather than approximate.
    """

    def __init__(self, *, num_actions: int, spacing: float = 1.0) -> None:
        self.num_actions = num_actions
        self.spacing = spacing

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return the ramp and a neutral value, whatever `state` is."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        logits = np.array(
            [self.spacing * action for action in range(self.num_actions)],
            dtype=np.float32,
        )
        return logits, 0.0


def _candidates(result: SearchResult) -> frozenset[int]:
    """Return the actions the search actually sampled, i.e. the candidate set."""
    return frozenset(int(a) for a in np.flatnonzero(result.visit_counts > 0))


@pytest.mark.parametrize("scale", [0.0, -1.0, -1e-12, math.inf, -math.inf, math.nan])
def test_a_gumbel_scale_that_is_not_strictly_positive_is_rejected(
    scale: float,
) -> None:
    """Zero is refused loudly rather than accepted as "the same search, quieter".

    At ``gumbel_scale = 0`` the root stops sampling ``m`` actions without
    replacement from ``softmax(logits)`` and starts taking the deterministic
    top-``m`` logits, which is a different algorithm with none of the paper's
    policy-improvement guarantee -- and one whose only outward symptom is
    self-play games that stop exploring. A caller who wants argmax play says
    so upstream, where a reader can see it.
    """
    with pytest.raises(ValueError, match="gumbel_scale"):
        GumbelConfig(num_actions=6, m=2, n=4, seed=0, gumbel_scale=scale)


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 5, 8, 13, 21))
@pytest.mark.parametrize("scale", (0.25, 1.0, 2.0, 7.5))
def test_the_variates_are_the_standard_draw_times_the_scale_exactly(
    scale: float, seed: int
) -> None:
    """The scaled variates are ``scale * g`` for the *same* seeded ``g``.

    With ``m = 1`` and ``n = 0`` the returned action is the argmax of
    ``scale * g + logits`` over the legal set and nothing else -- no
    simulations, no ``sigma``, no halving -- so recomputing that argmax here
    from a freshly seeded :func:`numpy.random.default_rng` pins the arithmetic
    at the SELECTION site. At ``scale = 2`` it holds only if the variates are
    exactly twice the ``scale = 1`` ones; a scale applied to ``g + logits``
    instead of to ``g``, or drawn from a differently-advanced stream, fails
    here. The halving and final-argmax uses of the same draw are pinned
    separately by ``test_the_scale_reaches_every_use_of_the_draw``.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    search = GumbelSearch(
        RampEvaluator(num_actions=6),
        GumbelConfig(num_actions=6, m=1, n=0, seed=seed, gumbel_scale=scale),
    )

    result = search.search(state)

    gumbel = np.random.default_rng(seed).gumbel(size=6)
    logits = np.arange(6, dtype=np.float64)
    draw = np.where(state.legal_mask(), scale * gumbel + logits, -np.inf)
    assert result.action == int(np.argmax(draw))


def test_a_tiny_scale_leaves_the_candidates_at_the_top_m_logits() -> None:
    """Damping the noise damps exactly the thing that makes candidates vary.

    The logit gap is 1 nat and the variates are scaled to ~1e-6, so the
    ordering is the logits' own: every seed samples the top three actions,
    ``{3, 4, 5}``. This is the limit fast mode moves *towards* -- and the
    reason the limit itself is not reachable by passing zero.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4, 5), max_depth=2)

    sampled = {
        _candidates(
            GumbelSearch(
                RampEvaluator(num_actions=6),
                GumbelConfig(
                    num_actions=6, m=3, n=30, seed=seed, gumbel_scale=1e-6
                ),
            ).search(state)
        )
        for seed in range(8)
    }

    assert sampled == {frozenset({3, 4, 5})}


def test_a_large_scale_lets_the_noise_outvote_the_logits() -> None:
    """The other direction, on the same evaluator: the scale really is live.

    Identical ramp, identical seeds, scale ~1e3 instead of ~1e-6: the 1-nat
    logit gaps are now negligible against the noise and the candidate set
    moves from seed to seed. A ``gumbel_scale`` that was accepted, validated
    and then never threaded into the draw would give the tiny-scale answer
    here too.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4, 5), max_depth=2)

    sampled = {
        _candidates(
            GumbelSearch(
                RampEvaluator(num_actions=6),
                GumbelConfig(
                    num_actions=6, m=3, n=30, seed=seed, gumbel_scale=1e3
                ),
            ).search(state)
        )
        for seed in range(8)
    }

    assert len(sampled) > 1


def test_the_default_config_is_the_explicit_scale_one_config() -> None:
    """Omitting the field is not merely close to ``gumbel_scale = 1.0``.

    The regression net for "the default path is untouched" is the rest of this
    file plus the oracle suite, all of which construct configs without the new
    field. This test pins the other half of that claim: the default and the
    explicit 1.0 are the same config and produce the same search, so those
    suites are testing the scaled code path rather than a bypass of it.
    """
    default = GumbelConfig(num_actions=6, m=3, n=17, seed=1234)
    explicit = GumbelConfig(num_actions=6, m=3, n=17, seed=1234, gumbel_scale=1.0)

    assert default.gumbel_scale == 1.0
    assert default == explicit

    from_default = GumbelSearch(ToyEvaluator(), default).search(
        ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    )
    from_explicit = GumbelSearch(ToyEvaluator(), explicit).search(
        ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    )

    assert from_default == from_explicit
    assert int(from_default.visit_counts.sum()) == 17


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 4, 5, 6, 7))
def test_the_scale_reaches_every_use_of_the_draw(seed: int) -> None:
    """Selection, halving and the final argmax all rank on the SAME scaled g.

    ``constant_value=0.0`` flattens every backed-up value, so ``sigma`` is
    identically zero and the entire search -- root candidate sampling, every
    sequential-halving cut, and the final action choice -- ranks on nothing
    but ``gumbel_scale * g + logits``. With real simulations in play
    (``m=3, n=30``), a scale applied at only SOME uses of the shared draw
    (say, selection but not the halving/final ranking) picks a different
    winner on most seeds; the KOM-029 adversarial review built exactly that
    mutant and it survived every other test in this file. This one kills it:
    verified failing under the mutant on 5 of these 8 seeds, passing here.
    """
    state = ToyState(
        num_actions=6, legal=(0, 1, 2, 3, 4, 5), max_depth=2, constant_value=0.0
    )
    scale = 0.25
    search = GumbelSearch(
        RampEvaluator(num_actions=6),
        GumbelConfig(num_actions=6, m=3, n=30, seed=seed, gumbel_scale=scale),
    )

    result = search.search(state)

    gumbel = np.random.default_rng(seed).gumbel(size=6)
    logits = np.arange(6, dtype=np.float64)
    assert result.action == int(np.argmax(scale * gumbel + logits))


_TEMPERATURE_ACTIONS: Final[int] = 10
"""Enough legal actions for the largest ``m`` this node sweeps."""

_TEMPERATURE_SCALES: Final[tuple[float, ...]] = (0.5, 2.0, 4.0)
"""KOM-036 §2.2's sweep -- and every one of them a power of two, deliberately.

``L / s`` must be the *exact* float32 ramp the tempered search is handed, or the
identity being checked would be approximate and a near-tie could flip an
ordering for a reason that has nothing to do with the claim. A ramp of integer
logits divided by a power of two is exact.
"""


@pytest.mark.parametrize("m", (1, 4, 8))
@pytest.mark.parametrize("scale", _TEMPERATURE_SCALES)
@pytest.mark.parametrize("seed", (0, 1, 2, 3, 5, 8, 13, 21))
def test_a_scaled_draw_is_a_tempered_logit_draw(
    m: int, scale: float, seed: int
) -> None:
    """KOM-036 D8: ``gumbel_scale = s`` **is** a root temperature ``T = s``.

    The claim, and the two lines that prove it. Root candidate selection takes
    the top ``m`` of ``s * g_a + L_a`` (``gumbel.py``'s ``draw``/``order``), and
    for ``s > 0`` dividing every entry by ``s`` preserves the ordering::

        argsort_desc(s * g + L)  ==  argsort_desc(g + L / s)

    By the Gumbel top-k trick the right-hand side is exactly a sample of ``m``
    actions **without replacement** from ``softmax(L / s)`` -- the definition of
    a temperature ``T = s``. So this node runs the same search twice: once at
    logits ``L`` with ``gumbel_scale = s``, once at logits ``L / s`` with the
    default scale, and requires the same ordered survivor list and the same
    played action.

    **Why ``m > 1`` matters.** Rev 1 of the design pinned only ``m = 1``, where
    the top-``m`` set collapses to an argmax and the claim actually being made
    -- a sample *without replacement* of ``m`` actions -- is untested. The
    ``m = 1`` case is kept in the sweep as the degenerate corner, not as the
    evidence.

    **Why ``n = 0``.** With no simulations every child is unvisited, so
    ``_completed_q`` returns the root's own value for every action, ``sigma`` is
    a constant vector, and the search's ranking is a function of
    ``s * g + L`` over the legal set and nothing else -- which makes the
    identity exact rather than distributional. It also keeps the two searches
    comparable: at ``n > 0`` the tempered search would evaluate its whole
    subtree at ``L / s`` and the two would be different searches by more than a
    root temperature.

    This is the node §3.1 leans on for the restated improvement guarantee: at
    scale ``s`` the played action improves on ``softmax(L / s)``, not on
    ``softmax(L)``, while the recorded label stays referenced to the untempered
    ``L``. That gap is the pre-registered cost of the draft schedule, and it is
    a checkable statement only because of this equality.
    """
    legal = tuple(range(_TEMPERATURE_ACTIONS))
    state = ToyState(
        num_actions=_TEMPERATURE_ACTIONS, legal=legal, max_depth=2
    )

    scaled = GumbelSearch(
        RampEvaluator(num_actions=_TEMPERATURE_ACTIONS, spacing=1.0),
        GumbelConfig(
            num_actions=_TEMPERATURE_ACTIONS,
            m=m,
            n=0,
            seed=seed,
            gumbel_scale=scale,
        ),
    ).search(state)

    tempered = GumbelSearch(
        # L / s, spelled as the ramp it is: L_a = a, so (L / s)_a = a / s.
        RampEvaluator(num_actions=_TEMPERATURE_ACTIONS, spacing=1.0 / scale),
        GumbelConfig(
            num_actions=_TEMPERATURE_ACTIONS,
            m=m,
            n=0,
            seed=seed,
            gumbel_scale=1.0,
        ),
    ).search(state)

    assert scaled.action == tempered.action, (
        f"a search at gumbel_scale={scale} played {scaled.action} where the "
        f"same search at logits L/{scale} and the default scale played "
        f"{tempered.action}; the scale is then not a root temperature"
    )
    assert scaled.sims_used == tempered.sims_used == 0

    # The ordered survivor list itself, which SearchResult does not expose:
    # recomputed from the same seeded draw, in the idiom
    # test_the_variates_are_the_standard_draw_times_the_scale_exactly already
    # uses at m = 1.
    gumbel = np.random.default_rng(seed).gumbel(size=_TEMPERATURE_ACTIONS)
    logits = np.arange(_TEMPERATURE_ACTIONS, dtype=np.float64)
    scaled_order = np.argsort(-(scale * gumbel + logits), kind="stable")[:m]
    tempered_order = np.argsort(-(gumbel + logits / scale), kind="stable")[:m]
    assert scaled_order.tolist() == tempered_order.tolist(), (
        "the two draws select different top-m survivors, or select them in a "
        "different order"
    )
    assert scaled.action in scaled_order.tolist()


def test_one_seeded_search_is_bit_frozen() -> None:
    """A float-golden: the exact policy-target bytes of one seeded search.

    Everything else in this suite is behavioral, which a distributionally-
    similar-but-not-bit-identical change could pass (KOM-029 review F3). This
    pins one full search bit-for-bit, so any future "arithmetically identical"
    claim -- like KOM-029's own gumbel_scale=1.0 default -- is checkable in
    CI instead of out-of-band. If this fails after an INTENDED numerical
    change, regenerate the constants with the recipe below and say so in the
    commit; if you did not intend one, you broke bit-identity.
    """
    import hashlib

    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    search = GumbelSearch(
        RampEvaluator(num_actions=6),
        GumbelConfig(num_actions=6, m=3, n=16, seed=42),
    )

    result = search.search(state)

    assert result.action == 4
    assert repr(float(result.root_value)) == "-0.03088111312021664"
    assert np.array_equal(
        result.root_prior_logits, np.arange(6, dtype=np.float32)
    ), "the ramp evaluator's root answer, reported verbatim"
    assert (
        hashlib.sha256(result.policy_target.tobytes()).hexdigest()
        == "f36632a133f002da7ff4edaa8ffebc9b898fcb48abf89a27db657abff507bd89"
    )

    # KOM-173 T1: the SAME golden, from an evaluator that also hands the search
    # each leaf's (w, d, l). The constants above are the pre-change ones, so this
    # is the bit-identity claim stated where it cannot rot: if carrying the
    # vector ever moves the search, the hash moves with it.
    with_vector = GumbelSearch(
        RampWdlEvaluator(num_actions=6),
        GumbelConfig(num_actions=6, m=3, n=16, seed=42),
    ).search(ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2))

    assert with_vector.action == 4
    assert repr(float(with_vector.root_value)) == "-0.03088111312021664"
    assert np.array_equal(with_vector.visit_counts, result.visit_counts)
    assert (
        hashlib.sha256(with_vector.policy_target.tobytes()).hexdigest()
        == "f36632a133f002da7ff4edaa8ffebc9b898fcb48abf89a27db657abff507bd89"
    )
    assert bool(np.isnan(result.root_wdl).all()), "the two-tuple run has no vector"
    assert _is_probability_vector(with_vector.root_wdl)


# --------------------------------------------------------------------------
# root_prior_logits: the prior the search actually used (KOM-040 D9')
# --------------------------------------------------------------------------


class CountingEvaluator:
    """A *different* logit vector on every call, plus the log of what it gave.

    Two properties the tests below need and :class:`ToyEvaluator` does not
    have. Every call answers differently, so "the reported prior is the
    **root's** evaluation" is checkable rather than merely consistent -- a
    result that handed back some child's logits, or the last evaluation of the
    search, fails instead of accidentally passing. And it does not refuse a
    terminal state, because the forced-move path now evaluates the root on both
    of its branches.
    """

    def __init__(self, *, num_actions: int, value: float = 0.25) -> None:
        self.num_actions = num_actions
        self.value = value
        self.calls = 0
        self.returned: list[npt.NDArray[np.float32]] = []
        self.seen: list[object] = []

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return a call-numbered ramp and a fixed value, recording both.

        ``seen`` records each state's ``history`` (where it has one) so a test
        can assert *which position* an evaluation was of -- the review's
        surviving mutant (D9' review F2/M6) evaluated the forced move's
        *successor* and no state-blind counter could tell.
        """
        self.seen.append(getattr(state, "history", None))
        self.calls += 1
        logits = np.array(
            [float(self.calls) + 0.25 * action for action in range(self.num_actions)],
            dtype=np.float32,
        )
        self.returned.append(logits)
        return logits, self.value


class ForcedTerminalState:
    """Terminal, and yet reporting exactly one legal action.

    A contradiction no rules engine should produce, which is precisely why
    :meth:`GumbelSearch._forced_move` has a branch for it: the value comes from
    the game rather than the network. The evaluation on that branch exists only
    to supply :attr:`SearchResult.root_prior_logits`.
    """

    def __init__(self, *, num_actions: int = 6, action: int = 2) -> None:
        self.num_actions = num_actions
        self.action = action

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the one-hot of the single "legal" action."""
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        mask[self.action] = True
        return mask

    def play(self, action: int) -> None:
        """Unreachable: a forced move is returned, never played here."""
        raise AssertionError("the forced-move path played a move")

    def clone(self) -> ForcedTerminalState:
        """Return an independent copy; there is no mutable state to share."""
        return ForcedTerminalState(num_actions=self.num_actions, action=self.action)

    def is_terminal(self) -> bool:
        """Return ``True`` -- that is the branch this state exists to reach."""
        return True

    def terminal_value(self) -> float:
        """Return a value nothing else in the search could have produced."""
        return -0.5

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the vector form of that value: ``(0.25, 0, 0.75)``."""
        return np.array([0.25, 0.0, 0.75], dtype=np.float64)

    def to_play(self) -> int:
        """Return the mover."""
        return 0


def test_the_result_reports_the_root_evaluation_the_search_used() -> None:
    """`root_prior_logits` is the root's own evaluation, verbatim.

    The whole point of the field: self-play's ``policy_kl`` needs the prior the
    search ranked its candidates against, and before KOM-040 D9' it bought that
    by evaluating the same position a second time. Every call here answers
    differently, so handing back a child's logits -- or the search's last
    evaluation rather than its first -- fails.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    evaluator = CountingEvaluator(num_actions=6)
    search = GumbelSearch(evaluator, GumbelConfig(num_actions=6, m=3, n=12, seed=5))

    result = search.search(state)

    assert evaluator.calls > 1, "the search evaluated children as well as the root"
    assert result.root_prior_logits.dtype == np.float32
    assert result.root_prior_logits.shape == (6,)
    assert np.array_equal(result.root_prior_logits, evaluator.returned[0])
    assert not np.array_equal(result.root_prior_logits, evaluator.returned[-1])


def test_a_forced_move_reports_the_prior_from_its_single_evaluation() -> None:
    """The forced path evaluates once and reports that evaluation's logits."""
    state = ToyState(num_actions=6, legal=(4,), max_depth=2)
    evaluator = CountingEvaluator(num_actions=6)
    search = GumbelSearch(evaluator, GumbelConfig(num_actions=6, m=8, n=64, seed=1))

    result = search.search(state)

    assert evaluator.calls == 1
    assert result.sims_used == 0
    assert np.array_equal(result.root_prior_logits, evaluator.returned[0])
    assert result.root_value == pytest.approx(evaluator.value)
    # The single evaluation must be of the ROOT, not the forced move's
    # successor: a successor eval carries the wrong position's logits AND a
    # wrong-perspective value, and the review's M6 mutant proved a
    # state-blind counter cannot see it (D9' review F2).
    assert evaluator.seen == [()], (
        f"the forced-path evaluation saw history {evaluator.seen}, not the "
        f"root's ()"
    )


# --------------------------------------------------------------------------
# KOM-159 T5: the root is the FIRST state handed to any evaluator
# --------------------------------------------------------------------------


class _EvaluationLog:
    """The histories every seam of one search was asked about, in call order.

    Shared by the scalar stub and the wave stub below, because the property
    under test spans both: ``GumbelSearch`` evaluates the root through
    ``evaluator`` and its leaves through ``wave_evaluator``, so "the root was
    first" is only checkable against a single ordered log of the two.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[int, ...]] = []

    def record(self, state: SearchState) -> None:
        """Note that `state` was evaluated, by its move history."""
        assert isinstance(state, ToyState), "a foreign state reached the seam"
        self.seen.append(state.history)


class _LoggingEvaluator:
    """The scalar seam, logging what it was asked and answering flatly."""

    def __init__(self, log: _EvaluationLog, *, num_actions: int) -> None:
        self._log = log
        self._num_actions = num_actions

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return a fixed prior and value, recording the position."""
        self._log.record(state)
        return np.zeros(self._num_actions, dtype=np.float32), 0.0


class _LoggingWaveEvaluator:
    """The wave seam, logging every state of every wave in submission order."""

    def __init__(self, log: _EvaluationLog, *, num_actions: int) -> None:
        self._log = log
        self._num_actions = num_actions

    def __call__(
        self, states: Sequence[SearchState]
    ) -> Sequence[tuple[npt.NDArray[np.float32], float]]:
        """Return one flat answer per state, recording each."""
        for state in states:
            self._log.record(state)
        return [
            (np.zeros(self._num_actions, dtype=np.float32), 0.0) for _ in states
        ]


@pytest.mark.parametrize("waves", [False, True], ids=["sequential", "wave"])
def test_the_root_is_the_first_state_any_evaluator_sees(waves: bool) -> None:
    """The invariant KOM-159's centred ``f(m)`` depends on (design R9).

    Self-play's moves-left term is centred on **the root's** predicted length,
    and the adapter learns which position that is by treating the first
    evaluation after a reset as the root
    (:class:`komugi.selfplay.adapter.SearchContext`). Nothing in the search's
    type signatures says the root comes first -- it is a property of the order
    of the statements in :meth:`~komugi.search.gumbel.GumbelSearch.search` --
    so a future scheduler that expanded a leaf before building the root node
    would silently centre every self-play search on a child. It fails here
    instead, on both schedules, because the wave path shares the root
    evaluation with the sequential one and only differs below it.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3), max_depth=3)
    log = _EvaluationLog()
    config = GumbelConfig(num_actions=6, m=4, n=16, seed=11)
    search = GumbelSearch(
        _LoggingEvaluator(log, num_actions=6),
        config,
        wave_evaluator=(
            _LoggingWaveEvaluator(log, num_actions=6) if waves else None
        ),
    )

    search.search(state)

    assert len(log.seen) > 1, "the search evaluated leaves as well as the root"
    assert log.seen[0] == (), (
        f"the first evaluation of the search was of history {log.seen[0]}, not "
        f"of the root's (); a centred leaf utility would be centred on that "
        f"position instead"
    )
    assert all(history != () for history in log.seen[1:]), (
        "the root was evaluated more than once, so 'the first call is the "
        "root' and 'a call on the root' are no longer the same statement"
    )


@pytest.mark.parametrize("waves", [False, True], ids=["sequential", "wave"])
def test_the_forced_move_path_evaluates_only_the_root(waves: bool) -> None:
    """One legal action: exactly one evaluation, and it is of the root.

    The forced path runs no simulation, so its single call is trivially the
    first -- and it must be of the position being searched rather than of the
    move's successor, which is the mutant KOM-040's D9' review found. A wave
    evaluator changes nothing here: there is no leaf to batch.
    """
    state = ToyState(num_actions=6, legal=(3,), max_depth=3)
    log = _EvaluationLog()
    search = GumbelSearch(
        _LoggingEvaluator(log, num_actions=6),
        GumbelConfig(num_actions=6, m=8, n=64, seed=3),
        wave_evaluator=(
            _LoggingWaveEvaluator(log, num_actions=6) if waves else None
        ),
    )

    result = search.search(state)

    assert result.sims_used == 0
    assert log.seen == [()]


def test_a_forced_terminal_root_is_evaluated_exactly_once_for_its_prior() -> None:
    """The branch that used to skip the network now asks it, once and only once.

    ``root_value`` still comes from the game -- the evaluator's ``0.25`` must
    not appear -- but the position is evaluated anyway, because otherwise
    ``root_prior_logits`` would have nothing honest to hold. That is not new
    cost: the self-play worker used to evaluate every root itself, this one
    included (KOM-040 D9').
    """
    state = ForcedTerminalState(num_actions=6, action=2)
    evaluator = CountingEvaluator(num_actions=6)
    search = GumbelSearch(evaluator, GumbelConfig(num_actions=6, m=4, n=32, seed=2))

    result = search.search(state)

    assert evaluator.calls == 1
    assert result.action == 2
    assert result.sims_used == 0
    assert result.root_value == pytest.approx(-0.5)
    assert np.array_equal(result.root_prior_logits, evaluator.returned[0])
    assert float(result.policy_target[2]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# KOM-173: the searched WDL vector (T1, T2, T4, T5, T11)
# --------------------------------------------------------------------------
#
# Every test below is about the *shadow*: an (A, 3) accumulator on the root
# node that carries each simulation's leaf ``(w, d, l)`` up the path, swapped
# once per change of mover. Two claims run through all of them --
#
#   (a) the scalar search does not move. The vector never reaches
#       ``_completed_q``, ``_improved_policy`` or ``_root_scores``, so a
#       three-tuple evaluator and a two-tuple one whose value agrees produce
#       bit-identical ``action`` / ``policy_target`` / ``root_value`` /
#       ``visit_counts`` / ``sims_used`` / ``root_prior_logits`` (T1);
#   (b) the readouts are probability vectors in the ROOT mover's frame, and
#       their utility ``w - l`` is the scalar the same tree backed up -- which
#       holds because ``w - l`` is odd under the frame swap, exactly where a
#       draw score's even ``s*d`` is not (T3, in
#       ``tests/search/unit/test_draw_utility_parity.py``).
#
# The stubs' vectors are **float64** deliberately (KOM-173 review F5): the
# production seam forms ``w - l`` in float32 and the accumulator sums ``w`` and
# ``l`` separately in float64, which is worth ~1e-8 in the field. A float64 stub
# removes that term from the measurement, so the identity below is pinned at
# 1e-12 rather than at the field's 1e-6.

_WDL_TOL: Final[float] = 1e-12
"""Identity tolerance with a float64 stub (review F5; the field's is 1e-6)."""

_SUM_TOL: Final[float] = 1e-9
"""How far a readout may be from summing to one, in float64 before the cast.

Design D4's number, and it is a claim about the accumulator's arithmetic: a
convex combination of probability vectors, summed in float64.
"""

_F32_SUM_TOL: Final[float] = 1e-6
"""The same claim after the ``float32`` cast the field performs.

Three ``float32`` entries each carry ~6e-8 of relative error, so their sum
cannot be pinned at 1e-9 -- that tolerance belongs to the float64 computation
(:data:`_SUM_TOL`), and asserting it on the cast field would be asserting
something about ``float32`` rather than about the search.
"""


def _zero_draw_wdl(value: float) -> npt.NDArray[np.float64]:
    """Return the ``d = 0`` probability vector whose ``w - l`` is `value`.

    The same construction :meth:`ToyState.terminal_wdl` uses, so a stub's leaf
    and a terminal leaf of the same toy speak the same language: whatever the
    search backs up as a scalar, the vector's utility is that scalar.
    """
    return np.array(
        [(1.0 + value) / 2.0, 0.0, (1.0 - value) / 2.0], dtype=np.float64
    )


class WdlToyEvaluator:
    """:class:`ToyEvaluator`'s answers, plus the vector they imply.

    The value is computed by the *same expression* as
    :meth:`ToyEvaluator.__call__`'s -- not merely a close one -- because T1's
    claim is bit-identity of the scalar path, and a stub that rounded the value
    differently would fail the claim for its own reason.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[tuple[int, ...]] = []

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return `(logits, value, wdl)` for `state`, recording that it was asked."""
        assert isinstance(state, ToyState), "evaluator was handed a foreign state"
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        self.calls += 1
        self.seen.append(state.history)
        base = sum(state.history) + 1
        logits = np.array(
            [0.4 * ((action * 7 + base) % 5) for action in range(state.num_actions)],
            dtype=np.float32,
        )
        value = math.tanh(0.3 * base) * (1.0 if state.to_play() == 0 else -1.0)
        return logits, value, _zero_draw_wdl(value)


class RampWdlEvaluator:
    """:class:`RampEvaluator`'s answers, plus the ``(0.5, 0, 0.5)`` they imply."""

    def __init__(self, *, num_actions: int, spacing: float = 1.0) -> None:
        self.num_actions = num_actions
        self.spacing = spacing

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return the ramp, a neutral value, and the vector that means."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        logits = np.array(
            [self.spacing * action for action in range(self.num_actions)],
            dtype=np.float32,
        )
        return logits, 0.0, _zero_draw_wdl(0.0)


class RandomWdlEvaluator:
    """A seeded random head: a genuine ``(w, d, l)`` with draw mass, and its utility.

    Unlike :func:`_zero_draw_wdl`'s vectors this one puts real mass on ``d``, so
    the sweep it drives covers the component the identity is *blind* to: ``d`` is
    swap-invariant, contributes nothing to ``w - l``, and is exactly what the
    whole ticket exists to record. The value is ``w - l`` in float64, which is
    what makes the identity assertable at 1e-12.

    Deterministic in the position: the per-leaf draw is seeded by a fold over
    the move history, so the search's own reproducibility guarantee survives.
    """

    def __init__(self, *, num_actions: int, seed: int) -> None:
        self.num_actions = num_actions
        self.seed = seed

    def _rng(self, history: tuple[int, ...]) -> np.random.Generator:
        """Return the generator this position's answer is drawn from."""
        folded = self.seed
        for action in history:
            folded = (folded * 1000003 + action + 1) % (2**32)
        return np.random.default_rng(folded)

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return `(logits, w - l, wdl)` for `state`."""
        assert not state.is_terminal(), "evaluator must not see a terminal state"
        history: tuple[int, ...] = getattr(state, "history", ())
        rng = self._rng(history)
        logits = np.asarray(
            rng.normal(scale=1.2, size=self.num_actions), dtype=np.float32
        )
        wdl = np.asarray(rng.dirichlet(np.ones(3)), dtype=np.float64)
        return logits, float(wdl[0] - wdl[2]), wdl


class _RootCapture:
    """The root node and improved policy of the last search, for the readouts.

    ``SearchResult`` exposes the three readouts but not the accumulator they
    were computed from, and T2/T5/T11 are claims about that accumulator. This
    installs itself over :func:`komugi.search.gumbel._wdl_readouts`, records its
    arguments and calls through, so the search under test is the real one and
    the test gets the tree it built. The same idiom
    ``tests/search/unit/test_draw_utility_parity.py`` already uses when it
    reaches for ``_Node`` and ``_select_interior``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root: object = None
        self.improved: npt.NDArray[np.float64] | None = None
        self.played: int | None = None
        inner = gumbel._wdl_readouts

        def spy(
            root: gumbel._Node, improved: npt.NDArray[np.float64], played: int
        ) -> tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
        ]:
            """Record the readouts' inputs, then produce them for real."""
            self.root = root
            self.improved = improved.copy()
            self.played = played
            return inner(root, improved, played)

        monkeypatch.setattr(gumbel, "_wdl_readouts", spy)

    @property
    def node(self) -> gumbel._Node:
        """Return the captured root, refusing to pretend when nothing ran."""
        assert isinstance(self.root, gumbel._Node), "no search reached the readouts"
        return self.root

    @property
    def sums(self) -> npt.NDArray[np.float64]:
        """Return the root's ``(A, 3)`` accumulator, refusing a missing one."""
        table = self.node.child_wdl_sums
        assert table is not None, "the root carried no searched-WDL accumulator"
        return table


def _interior_nodes(root: gumbel._Node) -> list[gumbel._Node]:
    """Return every node of `root`'s tree except `root` itself."""
    out: list[gumbel._Node] = []
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children.values())
    return out


def _is_probability_vector(vector: npt.NDArray[np.float32]) -> bool:
    """Return whether the ``float32`` `vector` is nonnegative and sums to one."""
    wide = vector.astype(np.float64)
    return bool((wide >= 0.0).all()) and abs(float(wide.sum()) - 1.0) <= _F32_SUM_TOL


@pytest.mark.parametrize(("m", "n"), [(8, 32), (32, 256)], ids=["fast", "full"])
def test_the_wdl_stub_leaves_the_scalar_search_bit_identical(m: int, n: int) -> None:
    """T1: carrying the vector changes nothing the search ranks on.

    The same toy, the same seed, the same values -- once through the two-tuple
    :class:`ToyEvaluator` and once through :class:`WdlToyEvaluator`, whose value
    is the same expression. Every scalar field is compared with ``==`` or
    ``array_equal``, not a tolerance: the vector is accumulated in a separate
    array that ``_completed_q``, ``_sigma``, ``_improved_policy``,
    ``_select_interior`` and ``_root_scores`` cannot reach, so "identical" is the
    only defensible claim. Both production budgets, because the two schedules
    allocate differently and a vector that leaked into selection would be
    likelier to show up at one of them.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=2)
    scalar_only = _run(state, m=m, n=n, seed=99, evaluator=ToyEvaluator())
    with_vector = GumbelSearch(
        WdlToyEvaluator(),
        GumbelConfig(num_actions=6, m=m, n=n, seed=99),
    ).search(state)

    assert with_vector.action == scalar_only.action
    assert with_vector.root_value == scalar_only.root_value
    assert with_vector.sims_used == scalar_only.sims_used
    assert np.array_equal(with_vector.policy_target, scalar_only.policy_target)
    assert np.array_equal(with_vector.visit_counts, scalar_only.visit_counts)
    assert np.array_equal(
        with_vector.root_prior_logits, scalar_only.root_prior_logits
    )

    # And the readouts the two-tuple run could not have.
    assert bool(np.isnan(scalar_only.root_wdl).all())
    assert _is_probability_vector(with_vector.root_wdl)
    assert abs(
        float(with_vector.root_wdl[0])
        - float(with_vector.root_wdl[2])
        - with_vector.root_value
    ) <= 1e-6


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 5, 8, 13, 21))
@pytest.mark.parametrize(("m", "n"), [(8, 32), (32, 256)], ids=["fast", "full"])
def test_the_root_wdl_utility_is_the_root_value(m: int, n: int, seed: int) -> None:
    """T1/T5: the reported ``root_wdl``'s utility is ``root_value``, and all
    three readouts are probability vectors.

    The sweep runs a fresh random head per seed -- real draw mass, values that
    are ``w - l`` exactly in float64 -- over both production budgets. The
    identity is not a coincidence of the stub: ``v_mix`` is an affine
    combination of the leaf scalars with weights that do not depend on them, the
    vector readout is the *same* combination component-wise, and ``w - l`` is
    odd under the ``(w, d, l) -> (l, d, w)`` swap, so the two agree wherever the
    leaf scalar is the vector's own utility. At ``draw_score != 0`` or
    ``lambda > 0`` the leaf scalar is not, and the gap is measured in
    ``test_draw_utility_parity.py`` instead of asserted away here.

    The tolerance here is the **field's** 1e-6: the readout is ``float32``, and
    three float32 components cannot reproduce a float64 difference more closely
    than that. The 1e-12 claim is about the accumulator and is asserted on it, in
    :func:`test_the_float64_identity_holds_to_1e_12_before_the_cast` (review F5).
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3)
    result = GumbelSearch(
        RandomWdlEvaluator(num_actions=6, seed=seed),
        GumbelConfig(num_actions=6, m=m, n=n, seed=seed),
    ).search(state)

    utility = float(result.root_wdl.astype(np.float64)[0]) - float(
        result.root_wdl.astype(np.float64)[2]
    )
    assert abs(utility - result.root_value) <= 1e-6, (
        "the float32 readout's utility is not the root value"
    )
    for readout in (result.root_wdl, result.root_wdl_pi, result.played_wdl):
        assert readout.dtype == np.float32
        assert readout.shape == (3,)
        assert _is_probability_vector(readout)


@pytest.mark.parametrize("seed", (0, 1, 2, 3, 5, 8, 13, 21))
def test_the_float64_identity_holds_to_1e_12_before_the_cast(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1 at its stated tolerance: the identity is 1e-12 in float64.

    :attr:`SearchResult.root_wdl` is ``float32``, which alone costs ~1e-8 of the
    identity -- so the 1e-12 claim (design D4 as amended by review F5) is a claim
    about the accumulator, and it is checked there: the readout is recomputed
    from the captured root in float64 and compared against ``root_value``.
    """
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3)
    result = GumbelSearch(
        RandomWdlEvaluator(num_actions=6, seed=seed),
        GumbelConfig(num_actions=6, m=8, n=32, seed=seed),
    ).search(state)

    wide = _expected_root_wdl(capture)
    assert abs(float(wide[0] - wide[2]) - result.root_value) <= _WDL_TOL
    assert abs(float(wide.sum()) - 1.0) <= _SUM_TOL
    np.testing.assert_allclose(
        result.root_wdl.astype(np.float64), wide, rtol=0.0, atol=1e-6
    )


def _expected_root_wdl(capture: _RootCapture) -> npt.NDArray[np.float64]:
    """Return the vector ``v_mix`` recomputed from a captured root, by hand.

    Deliberately a plain Python loop over the visited root edges rather than the
    vectorised expression the implementation uses: the point of a hand
    computation is to reach the same number by a different route.
    """
    root = capture.node
    sums = capture.sums
    visits = root.child_visits
    x_root = root.wdl_net
    assert x_root is not None
    total = int(visits.sum())
    if total == 0:
        return np.array(x_root, dtype=np.float64)
    weighted = np.zeros(3, dtype=np.float64)
    prior_mass = 0.0
    for action in range(visits.shape[0]):
        count = int(visits[action])
        if count == 0:
            continue
        prior = float(root.prior[action])
        prior_mass += prior
        weighted += prior * (sums[action] / count)
    if prior_mass <= 0.0:
        return np.array(x_root, dtype=np.float64)
    return (x_root + total * weighted / prior_mass) / (1.0 + total)


def test_root_wdl_pi_is_the_improved_policy_weighted_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: ``root_wdl_pi == sum_a pi'(a) Qtilde(a)``, completion included.

    ``Qtilde`` is ``S(a)/N(a)`` on visited root edges and ``root_wdl`` on the
    rest -- the vector form of ``_completed_q``'s rule, which is what keeps an
    unvisited action from being scored as a draw. Recomputed here in a loop from
    the captured accumulator and the captured improved policy, then compared to
    the reported field; the unvisited actions are what the test is really about,
    so the toy offers five legal actions to an ``m = 2`` search.
    """
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3)
    result = GumbelSearch(
        RandomWdlEvaluator(num_actions=6, seed=4242),
        GumbelConfig(num_actions=6, m=2, n=12, seed=4242),
    ).search(state)

    root = capture.node
    improved = capture.improved
    assert improved is not None
    unvisited = [
        action
        for action in np.flatnonzero(root.legal)
        if int(root.child_visits[action]) == 0
    ]
    assert len(unvisited) >= 3, "no completed action: the test would be vacuous"

    root_wdl = _expected_root_wdl(capture)
    expected = np.zeros(3, dtype=np.float64)
    for action in range(root.child_visits.shape[0]):
        weight = float(improved[action])
        if weight == 0.0:
            continue
        count = int(root.child_visits[action])
        mean = capture.sums[action] / count if count > 0 else root_wdl
        expected += weight * mean

    np.testing.assert_allclose(
        result.root_wdl_pi.astype(np.float64), expected, rtol=0.0, atol=1e-6
    )
    assert abs(float(expected.sum()) - 1.0) <= _SUM_TOL


def test_played_wdl_is_the_played_edge_mean_summed_by_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: ``played_wdl == S(a*) / N(a*)`` for the action the search returned."""
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3)
    result = GumbelSearch(
        RandomWdlEvaluator(num_actions=6, seed=77),
        GumbelConfig(num_actions=6, m=4, n=24, seed=77),
    ).search(state)

    visits = int(capture.node.child_visits[result.action])
    assert visits > 0, "the played action is a survivor, so it has visits"
    expected = capture.sums[result.action] / visits
    np.testing.assert_allclose(
        result.played_wdl.astype(np.float64), expected, rtol=0.0, atol=1e-6
    )
    assert capture.played == result.action


class _FixedWdlEvaluator:
    """One fixed ``(w, d, l)`` in the mover's frame, at every position.

    Which makes the frame arithmetic readable off the accumulator: a simulation
    that crossed one change of mover must have added the *swap* of this vector
    to the root's row, and one that crossed none must have added this vector
    itself.

    Does not refuse a terminal state, for the same reason
    :class:`CountingEvaluator` does not: the forced-terminal-root carve-out
    evaluates one, purely to supply ``root_prior_logits``, and the tests below
    exercise that branch.
    """

    def __init__(
        self, *, num_actions: int, wdl: tuple[float, float, float]
    ) -> None:
        self.num_actions = num_actions
        self.wdl = np.array(wdl, dtype=np.float64)

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return flat logits, the vector's own utility, and the vector."""
        return (
            np.zeros(self.num_actions, dtype=np.float32),
            float(self.wdl[0] - self.wdl[2]),
            self.wdl.copy(),
        )


_LEAF_WDL: Final[tuple[float, float, float]] = (0.7, 0.2, 0.1)
"""What every leaf answers, in its own mover's frame."""

_SWAPPED_LEAF_WDL: Final[tuple[float, float, float]] = (0.1, 0.2, 0.7)
"""The same vector one edge up, across a change of mover: ``d`` does not move."""


def test_a_leaf_vector_reaches_the_root_swapped_once_per_mover_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: ``(0.7, 0.2, 0.1)`` in the opponent's frame is ``(0.1, 0.2, 0.7)`` at
    the root.

    A hand-built two-ply line, pinned down to **one** simulation: ``m = 2`` with
    ``n = 1`` spends round 0 on nothing (``1 // 2 == 0``) and its single
    simulation on the finalist, so the whole tree is the root plus one depth-1
    leaf -- the opponent to move, answering ``(0.7, 0.2, 0.1)`` about itself. One
    change of mover on the way up, so the root's row for that edge is the
    *permutation* of it, and ``played_wdl`` is that row divided by its single
    visit. Note what does **not** happen: ``d`` stays at 0.2. A backup that
    negated the vector, or reversed it twice, or rotated the components, fails
    here.
    """
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=4, legal=(0, 1), max_depth=64)
    result = GumbelSearch(
        _FixedWdlEvaluator(num_actions=4, wdl=_LEAF_WDL),
        GumbelConfig(num_actions=4, m=2, n=1, seed=3),
    ).search(state)

    visited = [int(a) for a in np.flatnonzero(capture.node.child_visits > 0)]
    assert visited == [result.action], "the single simulation moved"
    assert int(capture.node.child_visits[result.action]) == 1
    np.testing.assert_allclose(
        capture.sums[result.action],
        np.array(_SWAPPED_LEAF_WDL),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.played_wdl.astype(np.float64),
        np.array(_SWAPPED_LEAF_WDL),
        rtol=0.0,
        atol=1e-7,
    )


def test_a_pass_ply_does_not_swap_the_vector_on_the_way_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2's other half -- the vector counterpart of
    :func:`test_a_pass_ply_is_not_negated_on_the_way_up`.

    ``passes_first`` makes the ``to_play`` schedule ``0, 0, 1``: the root's edge
    leads to a position with the **same** mover. The single simulation's depth-1
    leaf therefore answers ``(0.7, 0.2, 0.1)`` about a player who is also the
    root's mover, so the root's row must be that vector *unswapped*. A "swap once
    per edge" implementation reports ``(0.1, 0.2, 0.7)`` -- the opposite claim
    about who is winning -- and no shape or sum assertion would notice.
    """
    capture = _RootCapture(monkeypatch)
    state = SignToyState(num_actions=4, legal=(0, 1), passes_first=True)
    result = GumbelSearch(
        _FixedWdlEvaluator(num_actions=4, wdl=_LEAF_WDL),
        GumbelConfig(num_actions=4, m=2, n=1, seed=3),
    ).search(state)

    assert int(capture.node.child_visits[result.action]) == 1
    np.testing.assert_allclose(
        capture.sums[result.action], np.array(_LEAF_WDL), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        result.played_wdl.astype(np.float64),
        np.array(_LEAF_WDL),
        rtol=0.0,
        atol=1e-7,
    )


def test_the_draw_mass_survives_every_depth_class_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: ``d`` is swap-invariant, so it is the same at every depth -- a claim
    that needs no knowledge of the halving schedule.

    A 24-simulation search on a never-terminal toy mixes depth classes inside
    each root edge, and which mixture goes where is the schedule's business. But
    the swap is a *permutation*: whichever depth a leaf sat at, it contributed
    ``d = 0.2`` and ``w + l = 0.8`` to its root row. So every visited edge's mean
    reads ``d == 0.2`` exactly, and ``w`` and ``l`` are the only components the
    depth parity can move. This is precisely what a *negation* -- kom-021's
    scalar oscillation -- cannot do: it would alternate the draw term's sign.
    """
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3), max_depth=64)
    GumbelSearch(
        _FixedWdlEvaluator(num_actions=6, wdl=_LEAF_WDL),
        GumbelConfig(num_actions=6, m=4, n=24, seed=6),
    ).search(state)

    visited = [int(a) for a in np.flatnonzero(capture.node.child_visits > 0)]
    assert len(visited) >= 2, "too few visited edges for the claim to bite"
    deepest = max(int(capture.node.child_visits[a]) for a in visited)
    assert deepest > 1, "no edge was visited twice: no depth mixing to speak of"
    for action in visited:
        mean = capture.sums[action] / int(capture.node.child_visits[action])
        # 1e-12, not 0: the sum is float64 over up to two dozen addends, so it
        # carries a few ulps. It does not carry a depth-dependent SIGN, which is
        # the claim.
        assert float(mean[1]) == pytest.approx(_LEAF_WDL[1], abs=1e-12)
        assert float(mean[0] + mean[2]) == pytest.approx(0.8, abs=1e-12)


def test_a_mated_mover_enters_the_accumulator_as_a_one_hot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: a terminal leaf's vector comes from the game, as a one-hot.

    Every child of this depth-1 toy is terminal and lost for the player to move
    there, i.e. ``terminal_wdl() == (0, 0, 1)`` in the leaf mover's frame -- and
    that mover is the root's opponent, so the root's rows read ``(1, 0, 0)`` and
    ``root_wdl`` says the root mover is winning. The scalar half of this is
    :func:`test_child_values_are_negated_on_the_way_up`; here the *loss* slot
    has to move to the *win* slot, which a negation could not do.
    """
    capture = _RootCapture(monkeypatch)
    mated_child = ToyState(
        num_actions=4, legal=(0, 1, 2), max_depth=1, constant_value=-1.0
    )
    result = GumbelSearch(
        _FixedWdlEvaluator(num_actions=4, wdl=(0.5, 0.0, 0.5)),
        GumbelConfig(num_actions=4, m=3, n=32, seed=5),
    ).search(mated_child)

    mated = ToyState(
        num_actions=4, legal=(0, 1, 2), max_depth=1, constant_value=-1.0
    )
    mated.play(0)
    assert np.array_equal(
        mated.terminal_wdl(), np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ), "the toy's mated mover must read (0, 0, 1) for this test to mean anything"
    for action in np.flatnonzero(capture.node.child_visits > 0):
        count = int(capture.node.child_visits[action])
        np.testing.assert_allclose(
            capture.sums[action] / count,
            np.array([1.0, 0.0, 0.0]),
            rtol=0.0,
            atol=1e-15,
        )
    # The mated side is the root's OPPONENT, so the root reads itself as
    # winning: the loss slot became the win slot, which a negation of a scalar
    # can express but a negation of a vector cannot.
    assert float(result.root_wdl[0]) > 0.9
    assert result.root_value > 0.9
    assert _is_probability_vector(result.root_wdl)
    assert abs(
        float(result.root_wdl[0]) - float(result.root_wdl[2]) - result.root_value
    ) <= 1e-6


def test_a_forced_move_reports_the_root_vector_on_all_three_readouts() -> None:
    """T4: nothing was simulated, so all three readouts are the root's own vector."""
    state = ToyState(num_actions=6, legal=(4,), max_depth=2)
    evaluator = _FixedWdlEvaluator(num_actions=6, wdl=(0.6, 0.3, 0.1))
    result = GumbelSearch(
        evaluator, GumbelConfig(num_actions=6, m=8, n=64, seed=1)
    ).search(state)

    assert result.sims_used == 0
    assert int(result.visit_counts.sum()) == 0
    expected = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    for readout in (result.root_wdl, result.root_wdl_pi, result.played_wdl):
        np.testing.assert_allclose(readout, expected, rtol=0.0, atol=1e-7)
        assert _is_probability_vector(readout)
    # Three fields, three arrays: a caller may hold one without the others
    # changing under it.
    assert result.root_wdl is not result.root_wdl_pi
    assert result.root_wdl is not result.played_wdl


def test_a_forced_terminal_root_reports_the_games_one_hot() -> None:
    """T4: the adjudicated forced root takes its vector from the game, not the net.

    ``root_value`` already comes from :meth:`ForcedTerminalState.terminal_value`
    on this branch; the vector follows the same rule, so the evaluator's
    ``(0.6, 0.3, 0.1)`` must not appear. ``(0.25, 0, 0.75)`` is the game's own
    answer, and it is the vector form of the ``-0.5`` the scalar reports.
    """
    state = ForcedTerminalState(num_actions=6, action=2)
    result = GumbelSearch(
        _FixedWdlEvaluator(num_actions=6, wdl=(0.6, 0.3, 0.1)),
        GumbelConfig(num_actions=6, m=4, n=32, seed=2),
    ).search(state)

    assert result.root_value == pytest.approx(-0.5)
    expected = np.array([0.25, 0.0, 0.75], dtype=np.float32)
    for readout in (result.root_wdl, result.root_wdl_pi, result.played_wdl):
        np.testing.assert_allclose(readout, expected, rtol=0.0, atol=1e-7)
    assert (
        float(result.root_wdl[0]) - float(result.root_wdl[2])
        == pytest.approx(result.root_value, abs=1e-7)
    )


def test_a_two_tuple_evaluator_leaves_every_readout_nan() -> None:
    """The compatibility guarantee: no vector asked for, no vector invented.

    A legacy two-tuple evaluator -- the arena's, the gauntlet's, the ladder's,
    the anchor server's, and most of this suite's -- gets the search it always
    had, and the three readouts say so with ``NaN`` rather than with a uniform
    ``(1/3, 1/3, 1/3)`` that a reader could mistake for a measurement. The
    accumulator is not allocated at all, which is the other half of the claim
    (T11's memory invariant covers the live case).
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3)
    result = _run(state, m=8, n=32, seed=17)

    for readout in (result.root_wdl, result.root_wdl_pi, result.played_wdl):
        assert readout.dtype == np.float32
        assert readout.shape == (3,)
        assert bool(np.isnan(readout).all())
    assert result.sims_used == 32
    assert int(result.visit_counts.sum()) == 32


class _VectorWaveEvaluator:
    """A three-tuple wave seam -- the half of a mixed-arity pairing.

    Answers every leaf of every wave with ``(logits, value, wdl)`` while the
    scalar seam it is paired with answers with two-tuples, which is the
    construction KOM-173 (i) review F3 found the search accepted in silence.
    """

    def __init__(self, *, num_actions: int) -> None:
        self.num_actions = num_actions
        self.waves = 0

    def __call__(
        self, states: Sequence[SearchState]
    ) -> Sequence[
        tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]
    ]:
        """Return a flat prior, a neutral value and the vector that means."""
        self.waves += 1
        return [
            (
                np.zeros(self.num_actions, dtype=np.float32),
                0.0,
                _zero_draw_wdl(0.0),
            )
            for _ in states
        ]


def test_a_vector_leaf_under_a_scalar_root_is_refused() -> None:
    """The mixed-arity refusal is symmetric (KOM-210 F3, review finding 3).

    A two-tuple scalar seam with a three-tuple wave seam -- e.g.
    ``make_evaluator(client, s)`` paired with
    ``make_wave_evaluator(..., with_wdl=True)``, which nothing in the worker's
    construction prevents -- is the mirror image of the pairing
    :func:`~komugi.search.gumbel._backup` already refused. It used to be
    accepted in silence: the root's answer decides whether the accumulator
    exists, so there was nowhere for the wave's vectors to go, every leaf paid
    ``_as_wdl`` for nothing and all three readouts came back ``NaN`` -- a result
    indistinguishable from an honest two-tuple search
    (``test_a_two_tuple_evaluator_leaves_every_readout_nan``), which is exactly
    why no existing assertion could have caught it. Now it raises where the
    other direction raises, at the first backup, naming both seams.
    """
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3), max_depth=3)
    wave = _VectorWaveEvaluator(num_actions=6)
    search = GumbelSearch(
        ToyEvaluator(),
        GumbelConfig(num_actions=6, m=4, n=16, seed=11),
        wave_evaluator=wave,
    )

    with pytest.raises(ValueError, match="accumulator does not exist"):
        search.search(state)

    assert wave.waves > 0, (
        "no wave was ever submitted, so the mixed pairing was never exercised "
        "and this test would pass for the wrong reason"
    )


def test_only_the_root_holds_a_wdl_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T11: the memory invariant -- one ``(A, 3)`` table per search, on the root.

    Design D2's whole cost argument: a node is ~1.08 MB at ``A = 37341`` and a
    ``(32, 256)`` search creates up to 257 of them, so a per-node shadow would
    add 0.9 MB apiece (+83 %) to buy nothing -- every readout is a root
    quantity. The tree is walked here rather than sampled: *every* interior
    node's accumulator must be ``None``.
    """
    capture = _RootCapture(monkeypatch)
    state = ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=4)
    GumbelSearch(
        RandomWdlEvaluator(num_actions=6, seed=8),
        GumbelConfig(num_actions=6, m=32, n=256, seed=8),
    ).search(state)

    interior = _interior_nodes(capture.node)
    assert len(interior) > 10, "the tree is too small for the invariant to bite"
    assert capture.sums.shape == (6, 3)
    assert all(node.child_wdl_sums is None for node in interior), (
        "an interior node allocated a (A, 3) shadow: at A = 37341 that is "
        "0.9 MB per node rather than per search"
    )
    # The leaves still carry their own vector -- that is what the backup reads.
    assert any(node.wdl_net is not None for node in interior)
