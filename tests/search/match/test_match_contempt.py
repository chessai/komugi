"""A10a / T4: one tree, two contempts, and not one byte moves.

The dynamic half of the contempt invariant. Its partner A10b is static and
lives in ``test_match_imports.py``; each catches a poisoning style the other
misses, and the pair is discriminating because an implementation that stores
contempt-poisoned values must write a contempt-derived expression at one of
exactly two times. Write it **during the search** and A10a's buffers never
move afterwards -- A10a passes, A10b fails on the syntax. Cache it **lazily on
first read** and A10b may see no static taint -- but the read-everything step
below forces the write *after* the snapshot, so the bytes change and A10a
fails.

That pairing has a seam, and review F2 found it: a value written during the
search is *already in* A10a's snapshot, so it never moves, and A10b sees only
what the syntax admits -- which mutant M6b launders away by building ``q(c)``
in a helper, by item assignment, a module away from the write. So this file
carries a second clause that closes the during-search window with bytes rather
than with syntax: :func:`test_two_forced_descent_searches_store_the_same_bytes_at_two_contempts`.

That clause looks like the thing an earlier revision of this acceptance
demanded and this one still rejects: "run the search at ``c1``, run it again at
``c2``, assert byte-equal buffers". **No correct implementation can pass that**
in general. Contempt drives selection -- it enters ``U(a)``, and FPU is derived
from it too -- so at a different contempt the playouts descend differently and
the visits, the weights, the sums, the moments *and the set of expanded nodes*
all differ; asserted over an ordinary fixture, that acceptance asserts the
negation of the design's own selection rule. The rejection turns entirely on
the descent being steerable, so it lapses on a fixture with **one legal action
at every position**, where there is nowhere else for a playout to go. Two
searches at two contempts are comparable buffer-for-buffer exactly there, and
nowhere else in this suite is the comparison made.
"""

from __future__ import annotations

import numpy as np

# `planted` before `toygames`: importing it is what puts the oracle directory
# on ``sys.path``. See ``planted/__init__.py``.
from planted import (
    CountingEvaluator,
    DeepGame,
    MatchToyState,
    constant_wdl_evaluator,
    snapshot_node,
    solver_wdl_evaluator,
    walk_nodes,
)
from toygames import BOARD_SIZE, TicTacToe

from komugi.search.match.bias import SubtreeValueBias
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.search import MatchSearch
from komugi.search.match.select import (
    child_mean_wdl,
    child_raw_utilities,
    child_utilities,
    child_variances,
    corrected_child_wdl,
    exploration_constant,
    first_play_urgency,
    lcb_eligible,
    lcb_scores,
    node_sigma,
    node_sum_of_squares,
    node_utility,
    puct_scores,
    select_action,
    select_child,
    variance_scale,
)
from komugi.search.match.tree import MatchTree

#: A position with three legal moves, one of which holds the draw. Drawn lines
#: are what put mass on the draw slot, and a fixture with no draw mass would
#: make clause (iii)'s ``0.30 * d`` assertion vacuously about zero.
FIXTURE_BOARD = "OO./XXO/.X."

PLAYOUTS = 64
SECOND_CONTEMPT = 0.30

#: The forced-descent fixture: one legal action at every position, so the
#: descent is the same walk whatever contempt says, and deeper than the playout
#: count, so no playout reaches a terminal and every one of them backs a
#: network value up the chain.
FORCED_PLAYOUTS = 24
FORCED_DEPTH = 600

#: The leaf value of that fixture. **Its draw mass is load-bearing**: every
#: contempt-derived quantity in this package is a multiple of the draw slot, so
#: a fixture whose leaves were pure win/loss would make two contempts agree by
#: arithmetic rather than by discipline, and the comparison below would pass on
#: an implementation that stored ``q(c) . x``.
FORCED_LEAF_WDL = (0.25, 0.5, 0.25)


def read_every_path(
    node: MatchNode, config: MatchConfig, bias: SubtreeValueBias | None = None
) -> None:
    """Exercise every read path of `node` under `config`.

    The point is not the values -- they are discarded -- but that a lazily
    cached utility, ``q`` vector or variance would have to be *written* by one
    of these calls, after the snapshot was taken.

    `bias` is the tree's own design §A2 table, threaded through the four calls
    that consult one. Passing it is what keeps this "every read path" rather
    than "every read path that does not touch the bias correction" -- the
    correction is a read like any other, and a caching bug inside it would
    write to a node just as visibly.
    """
    node_utility(node, config)
    if node.width == 0:
        return
    node_sum_of_squares(node, config)
    node_sigma(node, config)
    variance_scale(node, config)
    exploration_constant(node, config)
    first_play_urgency(node, config)
    child_mean_wdl(node)
    corrected_child_wdl(node, config, bias=bias)
    child_raw_utilities(node, config)
    child_utilities(node, config, bias=bias)
    child_variances(node, config)
    puct_scores(node, config, bias=bias)
    lcb_eligible(node, config)
    lcb_scores(node, config, bias=bias)
    select_child(node, config, bias=bias)
    select_action(node, config, bias=bias)


def test_a_contempt_flip_changes_utility_but_not_one_stored_byte() -> None:
    """T4: snapshot, install a second config, read everything, re-snapshot.

    ``subtree_bias_factor`` is pinned to ``0.0`` and that pin is load-bearing
    for clause (iii): with the default factor the difference would be
    ``0.30 * (d + lambda * b_d * direct_fraction)`` rather than ``0.30 * d``, and the assertion
    would fail on a *correct* implementation.
    """
    config = MatchConfig(
        num_actions=BOARD_SIZE,
        playouts=PLAYOUTS,
        contempt=0.0,
        subtree_bias_factor=0.0,
    )
    evaluator = CountingEvaluator(solver_wdl_evaluator(num_actions=BOARD_SIZE))
    search = MatchSearch(config, evaluator)
    state = MatchToyState(TicTacToe.from_string(FIXTURE_BOARD))
    first = search.search(state)

    root = search.tree.root
    assert root is not None
    nodes = walk_nodes(root)
    assert len(nodes) > 1
    before = [snapshot_node(node) for node in nodes]
    calls_before = evaluator.calls

    flipped = MatchConfig(
        num_actions=BOARD_SIZE,
        playouts=PLAYOUTS,
        contempt=SECOND_CONTEMPT,
        subtree_bias_factor=0.0,
    )
    # The same tree, read by a new config: no re-search, no descend, no new
    # evaluator. This *is* a mid-game contempt change.
    search.config = flipped
    for node in nodes:
        read_every_path(node, flipped, search.tree.bias)

    after = [snapshot_node(node) for node in walk_nodes(root)]

    # (i) every stored array is byte-identical to its snapshot.
    assert after == before
    # (ii) nothing was re-evaluated.
    assert evaluator.calls == calls_before
    # (iii) the utility moved by exactly the contempt times the draw mass.
    draw_mass = float(first.root_wdl[1])
    assert draw_mass > 0.0
    moved = node_utility(root, flipped) - node_utility(root, config)
    assert abs(moved - SECOND_CONTEMPT * draw_mass) <= 1e-12
    # (iv) the played move is *permitted* to change; both are recorded.
    second_choice = int(root.actions[select_action(root, flipped)])
    assert bool(state.legal_mask()[second_choice])
    assert bool(state.legal_mask()[first.action])


def _forced_descent_tree(contempt: float) -> tuple[MatchTree, list[MatchNode]]:
    """Search the width-one fixture at `contempt` and return the tree it built.

    Driven through :class:`MatchTree` rather than ``MatchSearch`` deliberately:
    ``MatchSearch.search`` runs **no** playouts when the root has a single legal
    action (``search.py:182``), which is right for a forced move in a game and
    useless here, where the whole point is to run a full playout loop whose
    descent no contempt can steer.
    """
    config = MatchConfig(
        num_actions=1,
        playouts=FORCED_PLAYOUTS,
        contempt=contempt,
        subtree_bias_factor=0.0,
    )
    tree = MatchTree(config, constant_wdl_evaluator(FORCED_LEAF_WDL, num_actions=1))
    state = MatchToyState(DeepGame(width=1, depth=FORCED_DEPTH))
    tree.ensure_root(state)
    for _ in range(FORCED_PLAYOUTS):
        tree.playout(state)
    root = tree.root
    assert root is not None
    return tree, walk_nodes(root)


def test_two_forced_descent_searches_store_the_same_bytes_at_two_contempts() -> None:
    """A10a, extended to the window the snapshot cannot see: **during** the search.

    The clause above catches a value cached at *read* time, after its snapshot.
    A value written into a design §S1 field during the playout loop is already
    in that snapshot, so the buffers never move afterwards and it passes -- the
    hole review F2 found in the A10 pair, and the one mutant M6b walks through
    (a ``q(c)``-derived vector stored on the node inside ``_backup``).

    What this closes it with is the two-search comparison the module docstring
    correctly rejects **in general**, run on the one fixture where it is sound:
    a game with a single legal action at every position. The rejection turns on
    contempt steering the descent -- at another contempt the playouts go
    elsewhere and nothing is comparable -- and here there is nowhere else to go.
    Every node is reached by the same forced walk, every leaf gets the same
    value, and design §S1 says every stored field is contempt-free, so the two
    searches must agree **byte for byte**. Nothing weaker would do: a checker
    over syntax can always be walked around by another indirection, and this
    one is over the bytes themselves.
    """
    quiet_tree, quiet_nodes = _forced_descent_tree(0.0)
    loud_tree, loud_nodes = _forced_descent_tree(SECOND_CONTEMPT)

    # The fixture really searched, and the descent really is forced: one node
    # per playout plus the root, every one of them a single-edge chain link.
    assert len(quiet_nodes) == FORCED_PLAYOUTS + 1
    assert len(loud_nodes) == len(quiet_nodes)
    assert all(node.width == 1 for node in quiet_nodes)
    quiet_root = quiet_nodes[0]
    assert quiet_root.total_weight() == float(FORCED_PLAYOUTS)
    assert float(quiet_root.wdl_sum[0][1]) > 0.0  # the draw mass really arrived

    # ...and the two contempts really are two: read under either config the
    # same tree reports a different utility, so a stored one would differ too.
    assert (
        abs(
            node_utility(quiet_root, loud_tree.config)
            - node_utility(quiet_root, quiet_tree.config)
        )
        > 1e-9
    )

    assert [snapshot_node(node) for node in quiet_nodes] == [
        snapshot_node(node) for node in loud_nodes
    ]


def test_the_stored_wdl_is_the_contempt_free_evidence() -> None:
    """The reason (i) is a byte comparison: the buffers are raw evidence.

    ``root_wdl`` is reported contempt-free and ``root_utility`` is reported
    with the contempt it was derived under, so a reader can always recompute
    one from the other -- and a stored utility would make that impossible the
    moment anybody changed their mind about a draw.
    """
    config = MatchConfig(
        num_actions=BOARD_SIZE, playouts=32, contempt=-0.4, subtree_bias_factor=0.0
    )
    search = MatchSearch(config, solver_wdl_evaluator(num_actions=BOARD_SIZE))
    result = search.search(MatchToyState(TicTacToe.from_string(FIXTURE_BOARD)))

    assert result.contempt == -0.4
    assert abs(float(result.root_wdl.sum()) - 1.0) < 1e-12
    recomputed = (
        float(result.root_wdl[0])
        - float(result.root_wdl[2])
        + result.contempt * float(result.root_wdl[1])
    )
    assert abs(recomputed - result.root_utility) <= 1e-12
    assert np.isfinite(result.lcb_scores[result.weights > 0.0]).all()
