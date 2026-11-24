"""KOM-221c review: proposed tests for the surviving mutants (M5, X7, X8, X9)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from komugi.search import GumbelConfig
from komugi.search import gumbel


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cov = _load("test_kom219_gumbel_coverage.py")
_reg = _load("test_gumbel_proof_regressions.py")
_W = 8


def _node(*, to_play: int, terminal: bool = False, value: float = 0.0, legal_n: int = 0) -> gumbel._Node:
    legal = np.zeros(_W, dtype=np.bool_)
    legal[:legal_n] = True
    return gumbel._Node(to_play=to_play, is_terminal=terminal, value=value, legal=legal,
                        logits=np.zeros(_W, dtype=np.float32))


def _config() -> GumbelConfig:
    return GumbelConfig(num_actions=_W, m=2, n=4, seed=0, proof_mode="interval_v1")


# F1 (kills M5): the readouts follow the proof target / the rescued edge exactly.
def test_readouts_follow_the_proof_target_and_the_rescued_edge_on_c3() -> None:
    m, n = _cov._FULL
    result, _ = _reg._search("interior", policy="uniform", m=m, n=n, seed=0, proof_mode="interval_v1")
    draw = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert result.action == 1
    assert np.array_equal(result.played_wdl, draw)
    # exact zero mass on the refuted edge => none of its (0.40, 0, 0.60) mean vector leaks in
    assert np.array_equal(result.root_wdl_pi, draw)


# F2 (kills X8): a proven win is the whole admissible set -- pure helper ...
def test_a_proven_win_is_the_whole_admissible_set() -> None:
    root = _node(to_play=0, legal_n=3)
    root.child_visits[:3] = 4
    root.children[0] = _node(to_play=1, terminal=True, value=-1.0)  # opponent lost there: root edge [1, 1]
    root.child_value_sums[0] = 4.0
    action, target, scope, excluded = gumbel._proof_decision(root, [1, 2, 0], 1, np.zeros(_W), _config())
    assert (action, scope, excluded) == (0, "original_candidates", 2)
    assert [float(target[a]) for a in (0, 1, 2)] == [1.0, 0.0, 0.0]


# ... and end to end on the C1 fixture, on every audited seed that samples the win.
def test_a_sampled_immediate_win_excludes_every_other_root_edge() -> None:
    qualifying = 0
    for m, n in (_cov._FAST, _cov._FULL):
        for seed in _cov._SEEDS:
            result, _ = _reg._search("root", policy="uniform", m=m, n=n, seed=seed, proof_mode="interval_v1")
            if int(result.visit_counts[_cov._ROOT_WIN]) == 0:
                continue
            qualifying += 1
            assert result.action == _cov._ROOT_WIN
            assert result.diagnostics is not None
            assert result.diagnostics.rescue_scope == "none"
            assert result.diagnostics.proof_excluded_actions == _cov._ROOT_LEGAL - 1
            assert float(result.policy_target[_cov._ROOT_WIN]) == 1.0
    assert qualifying > 0, "no seed sampled the win"


# F3 (kills X7): strict dominance drops a bounded edge below the best lower bound; an equal bound is kept.
def test_strict_dominance_drops_a_bounded_edge_below_the_best_lower_bound() -> None:
    root = _node(to_play=0, legal_n=3)
    root.child_visits[:3] = 4
    root.children[0] = _node(to_play=1, terminal=True, value=-0.5)  # root frame [0.5, 0.5]
    root.child_value_sums[0] = 2.0
    root.children[1] = _node(to_play=1, terminal=True, value=0.5)   # root frame [-0.5, -0.5]: not lost, but dominated
    root.child_value_sums[1] = -2.0
    action, target, scope, excluded = gumbel._proof_decision(root, [1, 2, 0], 1, np.zeros(_W), _config())
    assert excluded == 1
    assert float(target[1]) == 0.0
    assert float(target[0]) > 0.0 and float(target[2]) > 0.0  # unknown edge 2 (U = 1 >= 0.5) is retained
    assert (action, scope) == (0, "original_candidates")


def test_an_upper_bound_equal_to_the_best_lower_bound_is_retained() -> None:
    root = _node(to_play=0, legal_n=2)
    root.child_visits[:2] = 4
    root.children[0] = _node(to_play=1, terminal=True, value=0.0)  # [0, 0]
    opp = _node(to_play=1, legal_n=2)                               # [0, 1] in its frame -> root [-1, 0]
    opp.children[0] = _node(to_play=0, terminal=True, value=0.0)
    root.children[1] = opp
    _, target, _, excluded = gumbel._proof_decision(root, [0, 1], 0, np.zeros(_W), _config())
    assert excluded == 0
    assert float(target[1]) > 0.0


# F4 (kills X9): a solved edge is scored by its exact value, not by its backed-up mean.
def test_the_target_uses_the_exact_value_of_a_solved_edge_not_its_mean() -> None:
    root = _node(to_play=0, legal_n=2)
    opp = _node(to_play=1, legal_n=1, value=-1.0)
    opp.children[0] = _node(to_play=0, terminal=True, value=0.0)   # only reply is a draw: edge 0 is exactly [0, 0]
    opp.child_visits[0] = 1
    root.children[0] = opp
    root.child_visits[0] = 2
    root.child_value_sums[0] = -1.0   # mean -0.5: the evaluation, then the draw
    root.child_visits[1] = 2
    root.child_value_sums[1] = 0.0    # unknown edge, mean 0
    _, target, _, _ = gumbel._proof_decision(root, [0, 1], 0, np.zeros(_W), _config())
    assert float(target[0]) == pytest.approx(0.5, abs=1e-12)
    assert float(target[1]) == pytest.approx(0.5, abs=1e-12)


# (h) the layer mutates nothing on the root
def test_the_proof_layer_mutates_nothing_on_the_root() -> None:
    root = _reg._pure_root((0,))
    before = (root.child_visits.copy(), root.child_value_sums.copy(), root.logits.copy(),
              root.legal.copy(), root.prior.copy(), dict(root.children))
    gumbel._proof_decision(root, [0, 1, 2], 0, _reg._gumbel_vector(1), _reg._pure_config())
    assert np.array_equal(before[0], root.child_visits)
    assert np.array_equal(before[1], root.child_value_sums)
    assert np.array_equal(before[2], root.logits)
    assert np.array_equal(before[3], root.legal)
    assert np.array_equal(before[4], root.prior)
    assert before[5] == root.children
