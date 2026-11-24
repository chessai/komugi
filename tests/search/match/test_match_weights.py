"""A5 / T7: the uncertainty seam is off, so weights are counts exactly.

The uncertainty head does not exist, so what ships is the **shape** that makes
wiring it later a change of one class rather than a change of the node schema:
every per-edge accumulator is a float weight, the backup multiplies all three
of them by ``leaf_weighter.weight(evaluation)``, and the default weighter
returns ``1.0``.

The invariant that makes that safe to ship unwired is the one below, and it is
asserted as **float equality** rather than with a tolerance: summing exact
``1.0`` values in float64 is exact up to ``2**53``, so a search that fails this
has a weight leak -- an edge whose weight was updated on a path where its visit
count was not, or the reverse -- and not a rounding problem.

The seam's other obligation is asserted here too: what it is handed is a
**copy**, so a weighter that normalises its input in place cannot write through
into design §S1 storage during the search (review F12).
"""

from __future__ import annotations

import numpy as np
from planted import (
    DRAW_WDL,
    DeepGame,
    MatchToyState,
    constant_wdl_evaluator,
    walk_nodes,
)

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch
from komugi.search.match.state import WdlEval

#: Pinned as a literal rather than read from the config, so a later budget
#: change cannot quietly weaken what this asserts.
PLAYOUTS = 512

#: Deeper than the budget, so no playout can reach a terminal position and
#: every playout takes the expand-and-back-up path.
DEPTH = 600

WIDTH = 3

#: What the evaluator answers at every non-terminal position of the seam test.
LEAF_WDL = (0.25, 0.5, 0.25)


class WdlEraser:
    """A :class:`LeafWeighter` that normalises its input **in place**.

    The hostile-but-plausible weighter of review F12: the obvious shape for a
    real one is "renormalise the triple, then read something off it", and an
    in-place ``/=`` is one keystroke from a ``[:] = 0``. It zeroes the WDL it
    is handed, so if the seam were passed a node's live ``wdl_net`` the node
    would be destroyed *during* the search -- where A10a's buffer snapshot,
    taken afterwards, could never see it move.

    ``policy_logits`` is left alone deliberately: on a terminal leaf it is the
    empty vector, and on an evaluated one it is the *evaluator's* full-width
    array rather than a node buffer (the node holds ``logits[actions]``, which
    fancy indexing has already copied), so there is no node state behind it to
    corrupt.
    """

    def __init__(self) -> None:
        self.calls = 0

    def weight(self, evaluation: WdlEval) -> float:
        """Zero the caller's WDL, then charge the usual unit weight."""
        self.calls += 1
        evaluation.wdl[:] = 0.0
        return 1.0


def test_weights_equal_visit_counts_exactly_under_unit_weight() -> None:
    """T7: ``W(a) == float(N(a))``, every edge, every node, after 512 playouts."""
    config = MatchConfig(num_actions=WIDTH, playouts=PLAYOUTS)
    search = MatchSearch(
        config,
        constant_wdl_evaluator([0.25, 0.5, 0.25], num_actions=WIDTH),
    )
    result = search.search(MatchToyState(DeepGame(width=WIDTH, depth=DEPTH)))
    assert result.playouts_used == PLAYOUTS

    nodes = walk_nodes(search.tree.root)
    assert len(nodes) == PLAYOUTS + 1
    checked = 0
    for node in nodes:
        for index in range(node.width):
            assert float(node.weight[index]) == float(int(node.visits[index]))
            checked += 1
    assert checked > 0
    np.testing.assert_array_equal(result.weights, result.visits.astype(np.float64))


def test_a_weighter_that_mutates_its_input_cannot_corrupt_a_node() -> None:
    """F12: the §A5 seam is handed copies, not the node's live buffers.

    ``_leaf_evaluation`` used to pass ``node.logits`` and ``node.wdl_net``
    straight through, and ``_expand`` used to return the very array the node
    stores as ``wdl_net``. Nothing in this package mutates either, so it was
    correct today and one third-party weighter away from not being: the write
    would land *during* the search, which is exactly the window A10a's
    snapshot cannot see into.

    Depth two, so both leaf paths run: the evaluated non-terminal leaf that
    ``_expand`` produces, and the terminal one that ``_leaf_evaluation`` does.
    Without the copies the first assertion below fires -- every node's
    ``wdl_net`` reads ``(0, 0, 0)`` -- and the backed-up mass goes with it.
    """
    eraser = WdlEraser()
    config = MatchConfig(num_actions=WIDTH, playouts=16, leaf_weighter=eraser)
    search = MatchSearch(
        config, constant_wdl_evaluator(LEAF_WDL, num_actions=WIDTH)
    )
    result = search.search(MatchToyState(DeepGame(width=WIDTH, depth=2)))

    nodes = walk_nodes(search.tree.root)
    terminals = [node for node in nodes if node.is_terminal]
    evaluated = [node for node in nodes if not node.is_terminal]
    # The fixture has to exercise both leaf paths, or half the fix is untested.
    assert terminals and len(evaluated) > 1
    for node in terminals:
        np.testing.assert_array_equal(node.wdl_net, DRAW_WDL)
    for node in evaluated:
        np.testing.assert_array_equal(node.wdl_net, np.asarray(LEAF_WDL))

    # The seam really ran, and really does mutate what it is given: without
    # both, this test would pass on a weighter that did nothing at all.
    assert eraser.calls == result.playouts_used > 0
    handed_over = WdlEval(
        np.zeros(0, dtype=np.float32), np.array(LEAF_WDL, dtype=np.float64)
    )
    eraser.weight(handed_over)
    np.testing.assert_array_equal(handed_over.wdl, np.zeros(3))

    # And the values that reached the tree are the real ones: a corrupted
    # `wdl_net` is backed up as zeros, so the root's mean would not sum to one.
    assert abs(float(result.root_wdl.sum()) - 1.0) < 1e-12


def test_the_root_weight_total_is_the_playout_count() -> None:
    """T7's corollary: no playout backed up twice, and none was dropped.

    Every playout passes through exactly one root edge, so the root's total
    weight is the budget. A double-counting backup would pass the per-edge
    equality above -- both accumulators would be wrong together -- and fails
    here.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=64)
    search = MatchSearch(
        config,
        constant_wdl_evaluator([0.25, 0.5, 0.25], num_actions=WIDTH),
    )
    search.search(MatchToyState(DeepGame(width=WIDTH, depth=DEPTH)))
    root = search.tree.root
    assert root is not None
    assert root.total_weight() == 64.0
    assert int(root.visits.sum()) == 64
