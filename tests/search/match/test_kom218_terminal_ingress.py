"""KOM-218: terminal WDL is validated and owned at match-tree ingress."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch
from komugi.search.match.state import MatchState, WdlEval
from komugi.search.match.tree import WDL_SIMPLEX_TOL, MatchTree


class CountingEvaluator:
    """Return one valid nonterminal evaluation and count every call."""

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.calls = 0

    def __call__(self, state: MatchState) -> WdlEval:
        self.calls += 1
        assert not state.is_terminal(), "the terminal evaluator guard regressed"
        return WdlEval(
            np.zeros(self.num_actions, dtype=np.float32),
            np.asarray((0.25, 0.5, 0.25), dtype=np.float64),
        )


class Supplier:
    """A shared terminal-WDL supplier whose return object may be stateful."""

    def __init__(self, value: Any | Callable[[int], Any]) -> None:
        self.value = value
        self.calls = 0

    def read(self) -> Any:
        self.calls += 1
        if callable(self.value):
            return self.value(self.calls)
        return self.value


class IngressState:
    """A one-action root followed by a terminal supplied through `Supplier`."""

    def __init__(
        self,
        supplier: Supplier,
        *,
        ply: int = 0,
        terminal_root: bool = False,
        width: int = 1,
    ) -> None:
        self.supplier = supplier
        self.ply = ply
        self.terminal_root = terminal_root
        self.width = width

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        return np.full(self.width, not self.is_terminal(), dtype=np.bool_)

    def play(self, action: int) -> None:
        assert 0 <= action < self.width
        self.ply += 1

    def clone(self) -> IngressState:
        return IngressState(
            self.supplier,
            ply=self.ply,
            terminal_root=self.terminal_root,
            width=self.width,
        )

    def is_terminal(self) -> bool:
        return self.terminal_root or self.ply > 0

    def to_play(self) -> int:
        return self.ply % 2

    def observation(self) -> npt.NDArray[np.float32]:
        return np.zeros(1, dtype=np.float32)

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.supplier.read())


INVALID_TERMINAL_WDLS = (
    pytest.param(np.zeros((1, 3)), id="row"),
    pytest.param(np.zeros((3, 1)), id="column"),
    pytest.param(np.zeros(2), id="too-short"),
    pytest.param(np.zeros(4), id="too-long"),
    pytest.param((math.inf, 0.0, 0.0), id="positive-infinity"),
    pytest.param((-math.inf, 1.0, math.inf), id="negative-infinity"),
    pytest.param((-0.25, 0.5, 0.75), id="negative-sum-preserving"),
    pytest.param((0.0, 0.0, 0.0), id="zero-mass"),
    pytest.param((0.9, 0.9, 0.9), id="off-unit-mass"),
)


@pytest.mark.parametrize("wdl", INVALID_TERMINAL_WDLS)
def test_malformed_terminal_root_is_refused_before_installation(wdl: Any) -> None:
    supplier = Supplier(wdl)
    evaluator = CountingEvaluator(1)
    tree = MatchTree(MatchConfig(num_actions=1, playouts=1), evaluator)

    with pytest.raises(ValueError, match="terminal ingress"):
        tree.ensure_root(IngressState(supplier, terminal_root=True))

    assert supplier.calls == 1
    assert evaluator.calls == 0
    assert tree.root is None
    assert tree.node_count == 0
    assert tree.bias.keys() == ()


def _bias_snapshot(
    tree: MatchTree,
) -> dict[int, tuple[npt.NDArray[np.float64], float, int]]:
    snapshot = {}
    for key in tree.bias.keys():
        entry = tree.bias.entry(key)
        assert entry is not None
        snapshot[key] = (entry.err_sum.copy(), entry.weight_sum, entry.gen)
    return snapshot


def test_rejected_terminal_child_leaves_parent_and_bias_unchanged() -> None:
    supplier = Supplier((0.0, 0.0, 0.0))
    evaluator = CountingEvaluator(2)
    tree = MatchTree(MatchConfig(num_actions=2, playouts=1), evaluator)
    state = IngressState(supplier, width=2)
    tree.ensure_root(state)
    root = tree.root
    assert root is not None
    before_arrays = tuple(
        value.copy() for value in (root.visits, root.weight, root.wdl_sum, root.moment)
    )
    before_children = dict(root.children)
    before_bias = _bias_snapshot(tree)

    with pytest.raises(ValueError, match="terminal ingress"):
        tree.playout(state)

    assert supplier.calls == 1
    assert evaluator.calls == 1
    assert root.children == before_children
    for actual, expected in zip(
        (root.visits, root.weight, root.wdl_sum, root.moment),
        before_arrays,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    after_bias = _bias_snapshot(tree)
    assert after_bias.keys() == before_bias.keys()
    for key in before_bias:
        np.testing.assert_array_equal(after_bias[key][0], before_bias[key][0])
        assert after_bias[key][1:] == before_bias[key][1:]


def test_fractional_terminal_distribution_is_preserved_through_backup() -> None:
    accepted = np.asarray((0.5, 0.25, 0.25), dtype=np.float64)
    search = MatchSearch(
        MatchConfig(num_actions=2, playouts=2), CountingEvaluator(2)
    )
    result = search.search(IngressState(Supplier(accepted), width=2))
    tree = search.tree
    root = tree.root
    assert root is not None
    terminal = root.children[0]
    np.testing.assert_array_equal(terminal.wdl_net, accepted)
    np.testing.assert_array_equal(root.wdl_sum[0], (0.25, 0.25, 0.5))
    assert root.weight[0] == 1.0
    assert result.root_utility == pytest.approx(-0.25)


@pytest.mark.parametrize("sign", (-1.0, 1.0), ids=("below-one", "above-one"))
def test_absolute_simplex_tolerance_accepts_safe_inside_values(sign: float) -> None:
    delta = sign * (WDL_SIMPLEX_TOL / 2.0)
    supplied = np.asarray((1.0 + delta, 0.0, 0.0), dtype=np.float64)
    tree = MatchTree(MatchConfig(num_actions=1), CountingEvaluator(1))
    tree.ensure_root(IngressState(Supplier(supplied), terminal_root=True))
    root = tree.root
    assert root is not None
    np.testing.assert_array_equal(root.wdl_net, supplied)
    assert float(root.wdl_net.sum()) != 1.0, "ingress must not normalize"


@pytest.mark.parametrize("sign", (-1.0, 1.0), ids=("below-one", "above-one"))
def test_absolute_simplex_tolerance_rejects_safe_outside_values(sign: float) -> None:
    delta = sign * (WDL_SIMPLEX_TOL * 2.0)
    tree = MatchTree(MatchConfig(num_actions=1), CountingEvaluator(1))
    with pytest.raises(ValueError, match="terminal ingress"):
        tree.ensure_root(
            IngressState(Supplier((1.0 + delta, 0.0, 0.0)), terminal_root=True)
        )


def test_terminal_ingress_owns_a_shared_float64_view() -> None:
    backing = np.asarray((99.0, 0.5, 0.25, 0.25, 99.0), dtype=np.float64)
    shared_view = backing[1:4]
    supplier = Supplier(shared_view)
    evaluator = CountingEvaluator(1)
    tree = MatchTree(MatchConfig(num_actions=1), evaluator)
    state = IngressState(supplier)
    tree.ensure_root(state)
    tree.playout(state)
    root = tree.root
    assert root is not None
    terminal = root.children[0]
    assert not np.shares_memory(terminal.wdl_net, backing)
    np.testing.assert_array_equal(terminal.wdl_net, (0.5, 0.25, 0.25))

    shared_view[:] = (0.0, 1.0, 0.0)
    tree.playout(state)

    np.testing.assert_array_equal(terminal.wdl_net, (0.5, 0.25, 0.25))
    np.testing.assert_array_equal(root.wdl_sum[0], (0.5, 0.5, 1.0))
    assert root.weight[0] == 2.0
    assert supplier.calls == 1


def test_stateful_terminal_supplier_is_read_exactly_once() -> None:
    valid = np.asarray((0.5, 0.25, 0.25), dtype=np.float64)
    supplier = Supplier(
        lambda call: valid if call == 1 else np.asarray((0.0, 0.0, 0.0))
    )
    evaluator = CountingEvaluator(1)
    tree = MatchTree(MatchConfig(num_actions=1), evaluator)
    tree.ensure_root(IngressState(supplier, terminal_root=True))

    assert supplier.calls == 1
    assert evaluator.calls == 0
    assert tree.root is not None
    np.testing.assert_array_equal(tree.root.wdl_net, valid)


@pytest.mark.parametrize(
    "outcome", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
def test_ordinary_one_hot_terminal_outcomes_remain_valid(
    outcome: tuple[float, float, float]
) -> None:
    tree = MatchTree(MatchConfig(num_actions=1), CountingEvaluator(1))
    tree.ensure_root(IngressState(Supplier(outcome), terminal_root=True))
    assert tree.root is not None
    np.testing.assert_array_equal(tree.root.wdl_net, outcome)
