"""KOM-217: independent weighted sample ledgers and conservative readers."""

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest
from planted import DeepGame, MatchToyState, constant_wdl_evaluator, snapshot_node
from test_kom215_tactics import (
    SHIPPED_BIAS_FACTOR,
    assert_both_bias_terminals_were_reached,
    bias_search,
    bias_state,
)

from komugi.search.match import select
from komugi.search.match.bias import SubtreeValueBias
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode, bias_key_of
from komugi.search.match.select import (
    child_bias_correction,
    child_utilities,
    child_variances,
    corrected_child_wdl,
    lcb_scores,
)
from komugi.search.match.state import WdlEval
from komugi.search.match.tree import MatchTree


@pytest.mark.parametrize("budget", [64, 256])
def test_real_search_bias_nonvacuity(budget: int) -> None:
    """F1: real F2 visits respond, and both consumers read the pooled bias."""
    runs: list[npt.NDArray[np.float64]] = []
    for factor in (0.0, SHIPPED_BIAS_FACTOR):
        search, evaluator = bias_search(budget, subtree_bias_factor=factor)
        result = search.search(bias_state())
        assert_both_bias_terminals_were_reached(search, evaluator)
        assert result.action == 1
        root = search.tree.root
        assert root is not None
        assert float(root.weight.sum()) == budget
        runs.append(root.weight.copy())

    # Retain the actual shipped-factor search root for both score comparisons.
    # Each child has one direct neural sample. Exact loss/draw descendants give
    # parent-frame utility errors -2 and +1, independently of the reader.
    root = search.tree.root
    assert root is not None
    expected = SHIPPED_BIAS_FACTOR * np.array([-2.0, 1.0]) / root.weight
    config = search.tree.config
    off = replace(config, subtree_bias_factor=0.0)
    np.testing.assert_allclose(
        select.puct_scores(root, config, bias=search.tree.bias)
        - select.puct_scores(root, off, bias=search.tree.bias),
        expected, rtol=0, atol=5e-16,
        err_msg="PUCT must consume real pooled bias",
    )
    np.testing.assert_allclose(
        select.lcb_scores(root, config, bias=search.tree.bias)
        - select.lcb_scores(root, off, bias=search.tree.bias),
        expected, rtol=0, atol=5e-16,
        err_msg="LCB must consume real pooled bias",
    )
    assert not np.array_equal(runs[0], runs[1]), "real search visits must respond to bias"


class Weights:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls = 0

    def weight(self, evaluation: WdlEval) -> float:
        self.calls += 1
        return next(self.values)


def node(*, mover: int = 0, key: int = 0, terminal: bool = False) -> MatchNode:
    return MatchNode(
        to_play=mover,
        is_terminal=terminal,
        actions=np.array([] if terminal else [0, 1], dtype=np.int32),
        logits=np.zeros(0 if terminal else 2, dtype=np.float32),
        wdl_net=np.array([0.5, 0.375, 0.125]),
        bias_key=key,
    )


def evaluation(n: MatchNode) -> WdlEval:
    return WdlEval(policy_logits=n.logits.copy(), wdl=n.wdl_net.copy())


def tree_with(weights: list[float]) -> tuple[MatchTree, MatchNode, MatchNode, Weights]:
    weighter = Weights(weights)
    config = MatchConfig(num_actions=2, leaf_weighter=weighter,
                         bias_free_prop=0.5, bias_weight_exponent=1.0)
    tree = MatchTree(config, lambda state: evaluation(node()))
    root = node(key=bias_key_of(0, -1, -1, 2))
    child = node(key=bias_key_of(0, 0, -1, 2))
    root.children[0] = child
    tree._root = root
    return tree, root, child, weighter


def populated(key: int) -> SubtreeValueBias:
    table = SubtreeValueBias(free_prop=0.5, weight_floor=1e-6)
    table.update(key, contribution=np.zeros(3), weight=0.0, gen=0,
                 delta=np.array([-0.25, 0.125, 0.125]), new_weight=2.0)
    return table


@pytest.mark.parametrize("initial,descendants", [(2.0, (0.25, 0.5)),
                                                   (0.25, (2.0, 4.0))])
@pytest.mark.parametrize("mover", [0, 1])
def test_weighted_path_ledger(initial: float, descendants: tuple[float, float],
                              mover: int) -> None:
    tree, root, child, weighter = tree_with([initial, *descendants])
    child.to_play = mover
    leaf = node(mover=mover, terminal=True)
    leaf.wdl_net[:] = [0.125, 0.125, 0.75]
    child.children[0] = leaf
    # These are explicit parent-frame sample values, independent of swap().
    neural = np.array([0.5, 0.375, 0.125] if mover == 0 else [0.125, 0.375, 0.5])
    outcome = np.array([0.125, 0.125, 0.75] if mover == 0 else [0.75, 0.125, 0.125])
    tree._backup([(root, 0)], child, evaluation(child))
    table = populated(child.bias_key)
    config = MatchConfig(num_actions=2, subtree_bias_factor=0.5)
    expected_sum = initial * neural
    total = initial
    previous = np.inf
    for extra in (0.0, *descendants):
        if extra:
            tree._backup([(root, 0), (child, 0)], leaf, evaluation(leaf))
            expected_sum += extra * outcome
            total += extra
        assert root.weight[0] == total
        np.testing.assert_array_equal(root.wdl_sum[0], expected_sum)
        expected = expected_sum / total + 0.5 * np.array([-0.25, 0.125, 0.125]) * initial / total
        np.testing.assert_allclose(corrected_child_wdl(root, config, bias=table)[0],
                                   expected, rtol=0, atol=1e-15,
                                   err_msg="weighted ledger correction")
        magnitude = float(np.linalg.norm(child_bias_correction(root, config, bias=table)[0]))
        assert 0 < magnitude < previous
        previous = magnitude
    assert weighter.calls == 3  # Readers must never call a stateful weighter.
    if initial == 2:
        assert child.bias_weight == 0  # Outgoing 0.75 is below confidence 1.
    else:
        expected_delta = np.array([-0.375, -0.25, 0.625] if mover == 0
                                  else [0.625, -0.25, -0.375])
        np.testing.assert_array_equal(tree.bias.read(child.bias_key), expected_delta)


def test_terminal_shared_key_and_missing_provenance() -> None:
    tree, root, child, _ = tree_with([2.0])
    tree._backup([(root, 0)], child, evaluation(child))
    table = populated(child.bias_key)
    config = MatchConfig(num_actions=2, subtree_bias_factor=0.5)
    nonterminal = child_bias_correction(root, config, bias=table)[0]
    assert np.any(nonterminal != 0)
    terminal = node(terminal=True, key=child.bias_key)
    root.children[0] = terminal
    root.wdl_sum[0] = 2 * terminal.wdl_net
    np.testing.assert_array_equal(corrected_child_wdl(root, config, bias=table)[0],
                                  terminal.wdl_net, err_msg="terminal must stay raw")
    del root.children[0]
    np.testing.assert_array_equal(child_bias_correction(root, config, bias=table)[0],
                                  np.zeros(3), err_msg="missing child has no provenance")
    root.children[0] = child
    root.weight[0] = 0
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(child_bias_correction(root, config, bias=table),
                                      np.zeros((2, 3)))


def test_failed_initial_weight_has_zero_direct_mass() -> None:
    weighter = Weights([float("nan"), 2.0])
    config = MatchConfig(num_actions=1, leaf_weighter=weighter)
    state = MatchToyState(DeepGame(width=1, depth=5))
    tree = MatchTree(config, constant_wdl_evaluator([0.5, 0.375, 0.125], num_actions=1))
    tree.ensure_root(state)
    root = tree.root
    assert root is not None
    with pytest.raises(ValueError, match="leaf weight"):
        tree.playout(state)
    child = root.children[0]
    assert not child.is_terminal
    assert root.weight[0] == 0
    tree.playout(state)
    assert root.weight[0] == child.weight[0] == 2
    np.testing.assert_array_equal(
        child_bias_correction(root, config,
                              bias=populated(child.bias_key)), np.zeros((1, 3)))
    assert weighter.calls == 2


def test_first_expansion_below_one_uses_shared_table() -> None:
    weighter = Weights([0.25])
    config = MatchConfig(num_actions=1, leaf_weighter=weighter)
    state = MatchToyState(DeepGame(width=1, depth=5))
    tree = MatchTree(config, constant_wdl_evaluator([0.5, 0.375, 0.125], num_actions=1))
    tree.ensure_root(state)
    tree.playout(state)
    root = tree.root
    assert root is not None
    child = root.children[0]
    assert not child.is_terminal
    assert root.weight[0] == 0.25
    assert child.total_weight() == child.bias_weight == 0
    np.testing.assert_array_equal(child_bias_correction(
        root, config, bias=populated(child.bias_key))[0],
        0.45 * np.array([-0.25, 0.125, 0.125]))
    assert weighter.calls == 1


def test_rounding_fraction_is_bounded() -> None:
    tree, root, child, _ = tree_with([1.0])
    tree._backup([(root, 0)], child, evaluation(child))
    child.weight[0] = np.nextafter(1.0, 2.0)
    np.testing.assert_array_equal(
        child_bias_correction(root, MatchConfig(num_actions=2),
                              bias=populated(child.bias_key)), np.zeros((2, 3)),
        err_msg="rounding must not produce negative direct mass")


def test_interleaved_contributors_replacement_and_rehome() -> None:
    tree, root, first, _ = tree_with([2.0])
    tree._backup([(root, 0)], first, evaluation(first))
    second_parent = node(key=root.bias_key)
    second = node(mover=1, key=first.bias_key)
    second_parent.children[0] = second
    # Both nodes have successful direct backups with unequal masses.
    tree._config = replace(tree._config, leaf_weighter=Weights([0.25]))
    tree._backup([(second_parent, 0)], second, evaluation(second))
    a = np.array([-0.25, -0.125, 0.375])
    b = np.array([0.125, -0.25, 0.125])

    def contribute(n: MatchNode, parent: MatchNode, total: float,
                   own_mean: list[float]) -> None:
        leaf = node(mover=n.to_play, terminal=True)
        leaf.wdl_net[:] = own_mean
        n.children[0] = leaf
        extra = total - n.total_weight()
        tree.config = replace(tree.config, leaf_weighter=Weights([extra]))
        tree._backup([(parent, 0), (n, 0)], leaf, evaluation(leaf))
        assert n.total_weight() == total

    def check(weight: float, error: npt.NDArray[np.float64]) -> None:
        entry = tree.bias.entry(first.bias_key)
        assert entry is not None
        assert entry.weight_sum == weight
        np.testing.assert_array_equal(entry.err_sum, error)

    contribute(first, root, 2, [0.25, 0.25, 0.5])
    check(2, 2 * a)
    contribute(second, second_parent, 4, [0.625, 0.125, 0.25])
    check(1 + 4, a + 4 * b)
    contribute(first, root, 8, [0.25, 0.25, 0.5])
    check(2 + 8, 2 * b + 8 * a)
    # Rehome retains first as an interior node, so its incoming edge survives.
    # The other contributor is freed with its sibling subtree; its mass stays.
    ancestor = node()
    ancestor.children[0] = root
    ancestor.children[1] = second_parent
    tree._root = ancestor
    del second_parent, second
    tree.descend(0)
    del ancestor
    assert tree.root is root
    check(1 + 4, b + 4 * a)
    contribute(first, root, 16, [0.25, 0.25, 0.5])
    check(0.5 + 16, 0.5 * b + 16 * a)
    # Independently constructed target ledgers: D=2 or D=.25, descendants=4.
    for direct in (2.0, 0.25):
        target_tree, target, child, _ = tree_with([direct, 4.0])
        leaf = node(terminal=True)
        child.children[0] = leaf
        target_tree._backup([(target, 0)], child, evaluation(child))
        target_tree._backup([(target, 0), (child, 0)], leaf, evaluation(leaf))
        expected = 0.45 * (0.5 * b + 16 * a) / 16.5 * direct / (direct + 4)
        np.testing.assert_allclose(child_bias_correction(
            target, MatchConfig(num_actions=2), bias=tree.bias)[0], expected,
            rtol=0, atol=1e-15, err_msg="pooled retained mass correction")


def test_nonzero_draw_bias_contempt_is_read_only() -> None:
    tree, root, child, _ = tree_with([2.0])
    tree._backup([(root, 0)], child, evaluation(child))
    table = populated(child.bias_key)
    arrays: list[npt.NDArray[np.int32] | npt.NDArray[np.float64]] = [
        root.visits, root.weight, root.wdl_sum, root.moment,
        root.wdl_net, child.wdl_net, child.bias_contribution]
    entry = table.entry(child.bias_key)
    assert entry is not None
    arrays.append(entry.err_sum)
    before = [a.tobytes() for a in arrays]
    nodes_before = (snapshot_node(root), snapshot_node(child))
    table_before = (entry.gen, entry.weight_sum)
    config = MatchConfig(num_actions=2, subtree_bias_factor=0.5)
    # [.5,.375,.125] + .5*[-.25,.125,.125] = [.375,.4375,.1875].
    for contempt in (-0.5, 0.5):
        cfg = replace(config, contempt=contempt)
        expected = 0.1875 + contempt * 0.4375
        assert child_utilities(root, cfg, bias=table)[0] == expected
        # Single identical sample: raw variance remains the configured floor.
        assert child_variances(root, cfg)[0] == cfg.var_floor**2
        expected_lcb = expected - cfg.k_lcb * np.sqrt(cfg.var_floor**2 / 2)
        assert lcb_scores(root, cfg, bias=table)[0] == expected_lcb
    assert [a.tobytes() for a in arrays] == before
    assert (snapshot_node(root), snapshot_node(child)) == nodes_before
    assert (entry.gen, entry.weight_sum) == table_before


def test_selection_adjustments_are_not_normalized_or_clipped_wdl() -> None:
    tree, root, child, _ = tree_with([2.0])
    tree._backup([(root, 0)], child, evaluation(child))
    table = populated(child.bias_key)
    config = MatchConfig(num_actions=2, subtree_bias_factor=8.0)
    # A deliberately strong read-time adjustment yields [-1.5,1.375,1.125].
    expected = np.array([-1.5, 1.375, 1.125])
    before = snapshot_node(root)
    np.testing.assert_array_equal(corrected_child_wdl(root, config, bias=table)[0], expected)
    assert child_utilities(root, config, bias=table)[0] == -1.0
    assert lcb_scores(root, config, bias=table)[0] == (
        -2.625 - config.k_lcb * np.sqrt(config.var_floor**2 / 2))
    assert snapshot_node(root) == before
