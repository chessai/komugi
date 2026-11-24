"""Unit tests for the terminal-only interval proofs (KOM-221 unit U1b).

CPU only: no engine, no evaluator, no ``gumbel`` import. The nodes are a local
dataclass that satisfies :class:`komugi.search.proofs.ProofNode` structurally,
which is the same contract ``gumbel._Node`` meets.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search.proofs import (
    UNKNOWN,
    Interval,
    is_exact,
    known_maximal,
    node_bounds,
    root_edge_bounds,
)


@dataclass
class FakeNode:
    """A minimal node satisfying the ``ProofNode`` protocol."""

    to_play: int
    is_terminal: bool = False
    value: float = 0.0
    legal: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    children: dict[int, "FakeNode"] = field(default_factory=dict)


def _mask(size: int, *actions: int) -> npt.NDArray[np.bool_]:
    """Return a length-`size` legal mask with `actions` offered."""
    mask = np.zeros(size, dtype=bool)
    for action in actions:
        mask[action] = True
    return mask


def _terminal(to_play: int, value: float) -> FakeNode:
    """Return a terminal node whose exact game value is `value` in its frame."""
    return FakeNode(to_play=to_play, is_terminal=True, value=float(np.float64(value)))


def _draw(to_play: int) -> FakeNode:
    """Return a terminal draw for `to_play`."""
    return _terminal(to_play=to_play, value=0.0)


def _snapshot(node: FakeNode) -> object:
    """Return a hashable structural snapshot, for mutation checks."""
    children = tuple(
        (action, _snapshot(child)) for action, child in sorted(node.children.items())
    )
    return (
        node.to_play,
        node.is_terminal,
        node.value,
        tuple(bool(flag) for flag in node.legal),
        children,
    )


def test_unexpanded_legal_edge_stays_unknown() -> None:
    """T1: a legal edge with no attached child contributes [-1, 1]."""
    child = _terminal(to_play=1, value=1.0)
    root = FakeNode(to_play=0, legal=_mask(2, 0, 1), children={0: child})

    edges = root_edge_bounds(root)

    assert edges[0] == (-1.0, -1.0)
    assert edges[1] == UNKNOWN
    assert node_bounds(root) == (-1.0, 1.0)


def test_one_winning_edge_proves_the_node() -> None:
    """T2: an opponent child proved lost proves the parent exactly +1."""
    loser = _terminal(to_play=1, value=-1.0)
    root = FakeNode(to_play=0, legal=_mask(3, 0, 1, 2), children={1: loser})

    edges = root_edge_bounds(root)
    bounds: Interval = node_bounds(root)

    assert edges[1] == (1.0, 1.0)
    assert edges[0] == UNKNOWN
    assert edges[2] == UNKNOWN
    assert bounds == (1.0, 1.0)
    assert is_exact(bounds)
    assert known_maximal(bounds)


def test_same_mover_chain_uses_identity_mapping() -> None:
    """T3: a same-mover step (a draft) maps the child interval unchanged."""
    same_mover_win = _terminal(to_play=0, value=1.0)
    direct = FakeNode(to_play=0, legal=_mask(1, 0), children={0: same_mover_win})
    assert root_edge_bounds(direct)[0] == (1.0, 1.0)

    grandchild = _terminal(to_play=1, value=1.0)
    middle = FakeNode(to_play=0, legal=_mask(1, 0), children={0: grandchild})
    root = FakeNode(to_play=0, legal=_mask(1, 0), children={0: middle})

    assert root_edge_bounds(middle)[0] == (-1.0, -1.0)
    assert node_bounds(middle) == (-1.0, -1.0)
    assert root_edge_bounds(root)[0] == (-1.0, -1.0)
    assert node_bounds(root) == (-1.0, -1.0)


def test_draw_closure_needs_every_edge() -> None:
    """T4: all-draw closure is [0, 0]; one unknown alternative leaves [0, 1]."""
    closed = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={0: _draw(to_play=1), 1: _draw(to_play=1)},
    )
    assert node_bounds(closed) == (0.0, 0.0)

    partial = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={0: _draw(to_play=1)},
    )
    assert node_bounds(partial) == (0.0, 1.0)


def test_fractional_closure_takes_the_best_edge() -> None:
    """T5: exact fractional outcomes close the node at the maximum."""
    root = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={
            0: _terminal(to_play=1, value=-0.25),
            1: _terminal(to_play=1, value=-0.5),
        },
    )

    edges = root_edge_bounds(root)

    assert edges[0] == (0.25, 0.25)
    assert edges[1] == (0.5, 0.5)
    assert node_bounds(root) == (0.5, 0.5)


def test_float64_endpoints_are_never_rounded() -> None:
    """T6: a value one ulp inside the range is exact but not maximal."""
    just_under = float(np.float64(math.nextafter(1.0, 0.0)))
    bounds = node_bounds(_terminal(to_play=0, value=just_under))
    assert bounds == (just_under, just_under)
    assert is_exact(bounds)
    assert not known_maximal(bounds)

    just_over = float(np.float64(math.nextafter(-1.0, 0.0)))
    near_loss = node_bounds(_terminal(to_play=0, value=just_over))
    assert near_loss == (just_over, just_over)
    assert near_loss[0] > -1.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1.5, -1.5])
def test_out_of_range_terminal_value_is_rejected(value: float) -> None:
    """T7a: a terminal scalar outside the closed [-1, 1] range is an error."""
    with pytest.raises(ValueError):
        node_bounds(_terminal(to_play=0, value=value))


def test_nonterminal_with_no_legal_edge_is_rejected() -> None:
    """T7b: a non-terminal node must offer at least one legal edge."""
    root = FakeNode(to_play=0, legal=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="nonterminal zero-legal"):
        node_bounds(root)


def test_terminal_with_empty_legal_is_fine() -> None:
    """T7c: a terminal node legitimately offers no edges."""
    leaf = _terminal(to_play=1, value=-1.0)
    assert node_bounds(leaf) == (-1.0, -1.0)
    assert root_edge_bounds(leaf) == {}


def test_evidence_only_tightens_the_interval() -> None:
    """T8: intervals nest as evidence is attached, and exactness is stable."""
    stage1 = FakeNode(to_play=0, legal=_mask(2, 0, 1))
    stage2 = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={0: _draw(to_play=1)},
    )
    stage3 = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={0: _draw(to_play=1), 1: _draw(to_play=1)},
    )

    first = node_bounds(stage1)
    second = node_bounds(stage2)
    third = node_bounds(stage3)
    assert first == (-1.0, 1.0)
    assert second == (0.0, 1.0)
    assert third == (0.0, 0.0)
    assert first[0] <= second[0]
    assert second[1] <= first[1]
    assert second[0] <= third[0]
    assert third[1] <= second[1]

    proved = FakeNode(
        to_play=0,
        legal=_mask(2, 0, 1),
        children={0: _terminal(to_play=1, value=-1.0)},
    )
    assert node_bounds(proved) == (1.0, 1.0)
    proved.children[1] = _draw(to_play=1)
    assert node_bounds(proved) == (1.0, 1.0)


def test_deep_same_mover_chain_does_not_recurse() -> None:
    """T9: a chain far longer than the recursion limit still resolves."""
    depth = 3 * sys.getrecursionlimit()
    node = _terminal(to_play=0, value=1.0)
    for _ in range(depth):
        node = FakeNode(to_play=0, legal=_mask(1, 0), children={0: node})

    assert node_bounds(node) == (1.0, 1.0)


def test_root_edge_bounds_match_the_offered_edges_and_leave_nodes_intact() -> None:
    """T10: keys are the offered edges, node bounds are their max, no mutation."""
    root = FakeNode(
        to_play=0,
        legal=_mask(4, 0, 2, 3),
        children={
            0: _draw(to_play=1),
            2: FakeNode(
                to_play=1,
                legal=_mask(2, 0, 1),
                children={0: _terminal(to_play=0, value=1.0)},
            ),
        },
    )
    before = copy.deepcopy(root)

    edges = root_edge_bounds(root)
    bounds = node_bounds(root)

    assert set(edges) == {0, 2, 3}
    assert edges[0] == (0.0, 0.0)
    assert edges[2] == (-1.0, 1.0)
    assert edges[3] == UNKNOWN
    assert bounds == (
        max(lower for lower, _ in edges.values()),
        max(upper for _, upper in edges.values()),
    )
    assert _snapshot(root) == _snapshot(before)
