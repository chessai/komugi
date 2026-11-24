"""Terminal-only interval proofs over an attached search tree.

The semantics, in the KOM-221 adversarial review's words:

    For a state x, unresolved legal edges are [-1,1]. A terminal node is
    [v,v]. Map a child's interval to the parent using identity when movers
    match and [-U,-L] otherwise. Node bounds are [max L_a, max U_a] over all
    offered legal edges. A missing legal child contributes [-1,1]; forgetting
    it falsely closes the game. One +1 edge proves the node exactly +1
    irrespective of other edges. An opponent child proved +1 gives the parent
    edge exactly -1. A discovered draw gives [0,1] with unknown alternatives;
    full draw closure gives [0,0]. Adding legitimate evidence can only raise L
    and lower U.

Only terminal scalars and the legal-edge closure enter a bound: no evaluator
value, no visit count, no prior and no WDL vector is proof of anything, so
none of them is read here. The bounds are therefore sound at every stage of a
search and tighten monotonically as the tree grows.

The precedent is MCTS-Solver (Winands et al. 2008), which back-propagates
proven wins and losses through a Monte-Carlo tree, and lc0's certainty
propagation, which carries the same two-sided certainty through a
policy-guided search.

The node protocol is structural: :class:`komugi.search.gumbel._Node`
satisfies :class:`ProofNode` as-is, so the Gumbel integration can call this
module on its private nodes without adapters. Nothing here mutates a node or
caches across calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "UNKNOWN",
    "Interval",
    "ProofNode",
    "is_exact",
    "known_maximal",
    "node_bounds",
    "root_edge_bounds",
]


class ProofNode(Protocol):
    """Structural view of a search node that carries proof evidence."""

    to_play: int
    """The mover at this node (``0`` or ``1``)."""

    is_terminal: bool
    """Whether the state is a game-over state."""

    value: float
    """For a terminal node, the exact scalar game value in *this* node's mover
    frame. Ignored on a non-terminal node: an evaluator value is never proof."""

    legal: npt.NDArray[np.bool_]
    """The offered legal-edge mask at this node; a terminal node's is empty."""

    @property
    def children(self) -> Mapping[int, "ProofNode"]:
        """Attached children, keyed by the action (edge) that reaches them.

        Read-only in the protocol so that an implementer's concrete
        ``dict[int, <its own node type>]`` attribute satisfies it (a mutable
        protocol member would be invariant, and ``gumbel._Node.children`` is a
        ``dict`` of ``_Node``). This module never mutates it.
        """


Interval = tuple[float, float]
"""A closed ``(lower, upper)`` bound in float64, in the node's own mover frame."""

UNKNOWN: Final[Interval] = (-1.0, 1.0)
"""The bound of an edge with no evidence at all: the whole outcome range."""


def is_exact(bounds: Interval) -> bool:
    """Return whether `bounds` pins a single value (``L == U``)."""
    return bounds[0] == bounds[1]


def known_maximal(bounds: Interval) -> bool:
    """Return whether `bounds` proves the maximal outcome (``L == 1.0``)."""
    return bounds[0] == 1.0


def _terminal_interval(node: ProofNode) -> Interval:
    """Return ``[v, v]`` for a terminal `node`, validating the scalar.

    The comparison is exact float64: ``math.nextafter(1.0, 0.0)`` is a
    fractional outcome and not an exact win. No rounding, no epsilon.
    """
    value = float(node.value)
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError(
            f"terminal value must be finite and within [-1, 1], got {node.value!r}"
        )
    return (value, value)


def _map_to_parent(parent: ProofNode, child: ProofNode, bounds: Interval) -> Interval:
    """Map `bounds` from `child`'s frame into `parent`'s frame."""
    if child.to_play == parent.to_play:
        return bounds
    return (-bounds[1], -bounds[0])


def _offered_edges(node: ProofNode) -> list[int]:
    """Return the offered legal edges of `node`, in action order."""
    return [int(action) for action in np.flatnonzero(node.legal)]


def _edge_interval(
    node: ProofNode, action: int, resolved: Mapping[int, Interval]
) -> Interval:
    """Return the interval of `node`'s edge `action` in `node`'s own frame."""
    child = node.children.get(action)
    if child is None:
        return UNKNOWN
    return _map_to_parent(node, child, resolved[id(child)])


def _resolve(root: ProofNode) -> dict[int, Interval]:
    """Return ``{id(node): interval}`` for `root` and every attached descendant.

    The traversal is an explicit-stack postorder, never recursion: a same-mover
    chain longer than ``sys.getrecursionlimit()`` (a draft sequence, say) must
    resolve without blowing the Python stack. The memo is local to this call
    and keyed by identity; nothing is written back onto a node.
    """
    resolved: dict[int, Interval] = {}
    stack: list[tuple[ProofNode, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in resolved:
            continue
        if node.is_terminal:
            resolved[key] = _terminal_interval(node)
            continue
        edges = _offered_edges(node)
        if not edges:
            raise ValueError("nonterminal zero-legal")
        if not expanded:
            stack.append((node, True))
            for action in edges:
                child = node.children.get(action)
                if child is not None and id(child) not in resolved:
                    stack.append((child, False))
            continue
        lower = -1.0
        upper = -1.0
        for action in edges:
            edge_lower, edge_upper = _edge_interval(node, action, resolved)
            lower = max(lower, edge_lower)
            upper = max(upper, edge_upper)
        resolved[key] = (lower, upper)
    return resolved


def node_bounds(root: ProofNode) -> Interval:
    """Return the proof interval of `root`, in `root`'s own mover frame."""
    return _resolve(root)[id(root)]


def root_edge_bounds(root: ProofNode) -> dict[int, Interval]:
    """Return the interval of every offered legal edge of `root`.

    The keys are exactly the offered legal root edges, an edge with no attached
    child maps to :data:`UNKNOWN`, and :func:`node_bounds` on the same root is
    the elementwise maximum over the returned intervals.
    """
    resolved = _resolve(root)
    return {
        action: _edge_interval(root, action, resolved)
        for action in _offered_edges(root)
    }
