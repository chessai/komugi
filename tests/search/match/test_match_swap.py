"""A14 / T-SWAP: the draw does not oscillate.

``adapter.py`` records a known P0a defect and names its fix: under scalar
negamax a constant mover-relative draw utility oscillates in sign up the tree
-- ``+c`` to this mover becomes ``-c`` to the opponent -- which is not the
symmetric draw-avoidance the design intends, and "the pre-registered P0b fix
is a WDL-carrying backup, not a patched scalar".

Two fixtures, because one is not enough. On a
forced-draw line every value that enters the tree is the one-hot draw
``(0, 1, 0)``, and the negamax step is a **swap** rather than a negation:
``swap((w, d, l)) = (l, d, w)`` leaves ``(0, 1, 0)`` fixed. So the root's mean
is the one-hot draw at every depth, and its utility is exactly the contempt in
force -- at odd and even parity alike. The scalar reference on the same line
alternates, and that difference is the whole content of the fix.

That fixture is, however, a **fixed point of the swap**, so it pins
WDL-vs-scalar and cannot see a swap that is missing altogether (review F9): a
backup with the permutation deleted passes it verbatim. The second fixture,
:func:`test_a_decided_line_exchanges_win_and_loss_at_every_mover_change`, runs
a *decided* line whose values are not fixed points, and it is the one that pins
the permutation itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from planted import (
    DRAW_WDL,
    DeepGame,
    MatchToyState,
    constant_wdl_evaluator,
    wdl_from_value,
)

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch
from komugi.search.match.tree import MatchTree, swap

WIDTH = 2
CONTEMPT = 0.30

#: The decided-line fixture's value, chosen so the triple it becomes --
#: ``(0.6, 0.4, 0.0)`` -- is **not** a fixed point of the swap and still
#: carries draw mass, which is what lets one line pin both halves of the
#: permutation at once (review F9).
DECIDED_VALUE = 0.6


def scalar_negamax_draw(depth: int, draw_score: float) -> float:
    """Return what the scalar path reports at the root of a forced-draw line.

    The reference implementation of the defect, written out rather than
    imported: a drawn terminal enters as the constant `draw_score` in the
    perspective of the mover *there*, and each step up negates it wholesale
    because a scalar has already folded the draw into a number it can only
    negate. The result therefore alternates with the parity of the line.
    """
    value = draw_score
    for _ in range(depth):
        value = -value
    return value


def test_the_swap_leaves_a_draw_where_it_is() -> None:
    """The permutation, in isolation: win/loss exchange, the draw does not."""
    np.testing.assert_array_equal(swap(DRAW_WDL), DRAW_WDL)
    np.testing.assert_array_equal(
        swap(np.array([0.7, 0.1, 0.2])), np.array([0.2, 0.1, 0.7])
    )
    # And the utility identity the design derives from it: the win/loss part
    # negates, the contempt term does not.
    triple = np.array([0.7, 0.1, 0.2])
    for contempt in (-0.5, 0.0, 0.3):
        ours = triple[0] - triple[2] + contempt * triple[1]
        swapped = swap(triple)
        theirs = swapped[0] - swapped[2] + contempt * swapped[1]
        assert abs(theirs - (-(triple[0] - triple[2]) + contempt * triple[1])) < 1e-15
        assert abs(theirs + ours - 2.0 * contempt * triple[1]) < 1e-15


@pytest.mark.parametrize("depth", [3, 4])
def test_a_forced_draw_line_reports_the_contempt_at_either_parity(
    depth: int,
) -> None:
    """T-SWAP: root utility ``== contempt`` at odd and even depth alike.

    Every leaf on this line is a draw and every interior evaluation is the
    one-hot draw too, so the *only* thing that could move the root's utility
    off the contempt is the backup's own arithmetic. A negating backup would
    report ``-contempt`` on the odd line and ``+contempt`` on the even one,
    which is exactly what the scalar reference below does.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=24, contempt=CONTEMPT)
    search = MatchSearch(
        config, constant_wdl_evaluator(DRAW_WDL, num_actions=WIDTH)
    )
    result = search.search(MatchToyState(DeepGame(width=WIDTH, depth=depth)))

    np.testing.assert_array_equal(result.root_wdl, DRAW_WDL)
    assert abs(result.root_utility - CONTEMPT) <= 1e-12
    assert result.contempt == CONTEMPT

    scalar = scalar_negamax_draw(depth, CONTEMPT)
    assert abs(scalar - (-1.0) ** depth * CONTEMPT) < 1e-15
    if depth % 2 == 1:
        # The line the scalar path gets wrong, and the WDL path does not.
        assert abs(scalar - result.root_utility) > 0.5


def test_a_decided_line_exchanges_win_and_loss_at_every_mover_change() -> None:
    """T-SWAP on a fixture that is **not** a fixed point of the swap.

    The parametrised test above is real evidence about the *draw*, and about
    nothing else: every value entering that tree is ``(0, 1, 0)``, and
    ``swap((0, 1, 0)) == (0, 1, 0)``, so it passes verbatim with the backup's
    swap deleted (review F9). Here every value entering the tree is
    ``(0.6, 0.4, 0.0)`` -- a won position for the mover *there*, with draw mass
    on it -- and the two halves of the permutation are separable.

    A forced line, one action wide and two plies deep, with the evaluator
    answering the same triple the terminal does, so the value entering the
    tree is the same at both depths and the arithmetic is the only variable.
    Two playouts, hand-derived:

    * playout 1 stops at the ply-1 child, whose mover is not the root's, so
      the root records **one** swap: ``(0.0, 0.4, 0.6)``;
    * playout 2 reaches the ply-2 terminal, whose mover is not the ply-1
      node's and *is* the root's, so the ply-1 edge records one swap and the
      root edge records a second, returning the win to the win slot:
      ``(0.6, 0.4, 0.0)``.

    **Under M3** -- the swap skipped for one side of the backup -- the ply-1
    node's edge assertion fires first: it holds ``(0.6, 0.4, 0.0)``, the
    leaf's own perspective, where ``(0.0, 0.4, 0.6)`` is asserted. The root's
    edge assertion fires too (``(0.0, 0.8, 1.2)`` for ``(0.6, 0.8, 0.6)``).
    The **draw** assertion does not fire under M3, and says so: a permutation
    that is skipped preserves the draw mass just as well as one that is
    applied, which is exactly why the draw alone was never evidence.
    """
    leaf_wdl = wdl_from_value(DECIDED_VALUE)
    np.testing.assert_allclose(leaf_wdl, [0.6, 0.4, 0.0], atol=1e-15)
    assert not np.array_equal(swap(leaf_wdl), leaf_wdl)

    config = MatchConfig(num_actions=1, contempt=CONTEMPT)
    tree = MatchTree(
        config, constant_wdl_evaluator(leaf_wdl, num_actions=1)
    )
    state = MatchToyState(DeepGame(width=1, depth=2, result=DECIDED_VALUE))
    tree.ensure_root(state)
    tree.playout(state)
    tree.playout(state)

    root = tree.root
    assert root is not None
    child = root.children[0]
    assert not child.is_terminal
    assert float(child.weight[0]) == 1.0
    assert float(root.weight[0]) == 2.0

    # One mover change: the leaf's win is a loss to the node above it.
    np.testing.assert_allclose(child.wdl_sum[0], [0.0, 0.4, 0.6], atol=1e-12)
    # Two mover changes: the win is back where it started, and the root has
    # one of each -- (0.0, 0.4, 0.6) from playout 1 and (0.6, 0.4, 0.0) from
    # playout 2.
    np.testing.assert_allclose(root.wdl_sum[0], [0.6, 0.8, 0.6], atol=1e-12)

    # The draw mass is conserved by every step, at both depths. This is the
    # half a negating backup gets wrong and a *skipped* swap gets right.
    assert abs(float(child.wdl_sum[0][1]) - 0.4) < 1e-12
    assert abs(float(root.wdl_sum[0][1]) - 0.8) < 1e-12
    assert abs(float(root.wdl_sum[0].sum()) - 2.0) < 1e-12


def test_the_same_line_is_neutral_at_zero_contempt() -> None:
    """The control: with no contempt a forced draw is worth exactly nothing.

    A backup that leaked a sign would still report ``0.0`` here, which is why
    this is the control and the parametrised test above is the evidence.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=16, contempt=0.0)
    search = MatchSearch(
        config, constant_wdl_evaluator(DRAW_WDL, num_actions=WIDTH)
    )
    result = search.search(MatchToyState(DeepGame(width=WIDTH, depth=3)))
    assert result.root_utility == 0.0
