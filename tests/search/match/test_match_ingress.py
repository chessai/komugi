"""KOM-179: what the match tree refuses at its two numeric seams.

Two holes from the review's numeric-ingress table (2026-09-12), both in
``tree.py`` and both with the same shape -- a number is taken from a seam and
written into design §S1 storage without being looked at:

* ``_expand`` checked the evaluator's ``(w, d, l)`` for **shape** and nothing
  else. A ``nan`` triple was stored as ``wdl_net``, backed up into ``wdl_sum``
  and ``moment``, and -- once an edge had unit weight -- written into design
  §A2's bias table, where a ``nan`` is permanent and poisons every later read of
  that key. The first complaint came from the table's own guard at the *second*
  playout, so the error named the bias table while the evaluator that produced
  it went unmentioned. That is why every assertion below matches on
  ``'evaluator'``: a bare ``pytest.raises(ValueError)`` is **green at playouts
  >= 2 without any fix at all**, which is precisely the trap this ticket exists
  to close.
* ``_backup`` multiplied all three accumulators by the §A5 leaf weight without
  checking it. A zero weight buys a visit and backs up nothing, breaking
  ``W(a) == float(N(a))``; a negative one subtracts probability mass from a
  distribution; a ``nan`` one destroys the subtree.

The weight is now checked **before** the walk up the path begins, so a refused
playout leaves every statistic exactly as it found it -- which is the property
the second half of this file asserts, on a tree that already has real numbers in
it rather than on an empty one where "unchanged" would mean "still zero".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from planted import DeepGame, MatchToyState, constant_wdl_evaluator

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch
from komugi.search.match.state import MatchState, WdlEval
from komugi.search.match.tree import MatchTree

WIDTH = 3

#: Deeper than any budget here, so no playout ends at a terminal position and
#: every leaf is one the evaluator answered for.
DEPTH = 64

#: A WDL that is a distribution, for the seam that is not under test.
GOOD_WDL = (0.25, 0.5, 0.25)

#: Playouts run before the weighter is armed, so "unchanged" is a statement
#: about real statistics and not about a tree of zeros.
SETTLED_PLAYOUTS = 6


def _state() -> MatchToyState:
    """Return a fresh position on the deep toy game."""
    return MatchToyState(DeepGame(width=WIDTH, depth=DEPTH))


class ScriptedWeight:
    """A §A5 leaf weighter that returns `bad` once the tree has settled.

    The first `good_calls` playouts are weighted 1.0, exactly as
    :class:`~komugi.search.match.config.UnitWeight` would, so the statistics the
    refusal must not disturb are genuinely non-zero.
    """

    def __init__(self, *, bad: float, good_calls: int) -> None:
        self.bad = bad
        self.good_calls = good_calls
        self.calls = 0

    def weight(self, evaluation: WdlEval) -> float:
        """Return the unit weight, then `bad` for every call after that."""
        self.calls += 1
        return 1.0 if self.calls <= self.good_calls else self.bad


class LateWdlEvaluator:
    """Answers `GOOD_WDL` for the first `good_calls` positions, then `bad`.

    Lets a test put the bad vector on a **child** expansion, with a healthy
    root above it: the ingress check has to sit at the evaluator seam rather
    than at the root, or a tree poisons itself on its second node.
    """

    def __init__(self, *, bad: tuple[float, float, float], good_calls: int) -> None:
        self.bad = bad
        self.good_calls = good_calls
        self.calls = 0

    def __call__(self, state: MatchState) -> WdlEval:
        """Return one evaluation, healthy or not depending on the call index."""
        self.calls += 1
        wdl = GOOD_WDL if self.calls <= self.good_calls else self.bad
        return WdlEval(
            np.zeros(WIDTH, dtype=np.float32), np.asarray(wdl, dtype=np.float64)
        )


# --------------------------------------------------------------------------
# _expand: the evaluator's (w, d, l) must be a distribution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("playouts", [0, 1, 8])
@pytest.mark.parametrize(
    ("wdl", "detail"),
    [
        ((math.nan, 0.5, 0.25), "non-finite"),
        ((math.inf, 0.5, 0.25), "non-finite"),
        ((-0.25, 0.5, 0.75), "negative entry"),
        ((0.75, 0.5, 0.25), "not a distribution"),
        ((0.1, 0.1, 0.1), "not a distribution"),
    ],
)
def test_match_nan_wdl_is_refused_at_ingress(
    playouts: int, wdl: tuple[float, float, float], detail: str
) -> None:
    """The error names the EVALUATOR, at every budget including zero.

    ``playouts=0`` still expands the root, so there is no budget at which a
    bad evaluation is quietly tolerated; ``playouts=8`` is past the point where
    the bias table used to raise first, and the ``match='evaluator'`` is what
    tells the two apart. ``(0.75, 0.5, 0.25)`` sums to 1.5 -- the unnormalised
    head -- and ``(0.1, 0.1, 0.1)`` is the same failure from below.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=playouts)
    search = MatchSearch(config, constant_wdl_evaluator(wdl, num_actions=WIDTH))

    with pytest.raises(ValueError, match="evaluator") as raised:
        search.search(_state())

    assert detail in str(raised.value)


def test_a_bad_wdl_from_a_child_expansion_is_refused_the_same_way() -> None:
    """The check is at the seam, not at the root: node two is checked too.

    The root answers with a distribution and the first child does not. Before
    KOM-179 that ``nan`` was stored, backed up, and then -- one playout later --
    reported as a bias-table error at a key the evaluator had never heard of.
    """
    evaluator = LateWdlEvaluator(bad=(math.nan, 0.5, 0.25), good_calls=1)
    config = MatchConfig(num_actions=WIDTH, playouts=4)
    search = MatchSearch(config, evaluator)

    with pytest.raises(ValueError, match="evaluator returned a non-finite wdl"):
        search.search(_state())

    assert evaluator.calls == 2, "the root evaluation was healthy and accepted"


def test_a_wdl_off_unit_mass_within_tolerance_is_accepted() -> None:
    """1e-7 of drift is a float32 softmax, not an unnormalised head.

    The tolerance's other half: without this the check would ban the very
    vectors the network produces.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=8)
    search = MatchSearch(
        config, constant_wdl_evaluator((0.25 + 1e-7, 0.5, 0.25), num_actions=WIDTH)
    )

    result = search.search(_state())

    assert result.playouts_used == 8
    assert bool(np.isfinite(result.root_wdl).all())


# --------------------------------------------------------------------------
# _backup: the §A5 weight must be finite and strictly positive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_a_nonpositive_leaf_weight_is_refused(bad: float) -> None:
    """A refused weight raises before a single statistic moves.

    All four §S1 accumulators are compared byte-for-byte across the failed
    playout. The check has to sit above the walk for that to hold: refusing
    halfway up would leave ``N``, ``W``, ``S`` and ``M`` disagreeing with each
    other on the edges already passed, and ``W(a) == float(N(a))`` -- the
    invariant the whole §A5 seam is checked by -- would be false with no
    exception left to explain it.
    """
    weighter = ScriptedWeight(bad=bad, good_calls=SETTLED_PLAYOUTS)
    config = MatchConfig(num_actions=WIDTH, playouts=32, leaf_weighter=weighter)
    tree = MatchTree(config, constant_wdl_evaluator(GOOD_WDL, num_actions=WIDTH))
    state = _state()
    tree.ensure_root(state)
    for _ in range(SETTLED_PLAYOUTS):
        tree.playout(state)

    root = tree.root
    assert root is not None
    assert int(root.visits.sum()) == SETTLED_PLAYOUTS, "the tree never settled"
    before = (
        root.visits.copy(),
        root.weight.copy(),
        root.wdl_sum.copy(),
        root.moment.copy(),
    )

    with pytest.raises(ValueError, match="leaf weighter") as raised:
        tree.playout(state)

    assert "ScriptedWeight" in str(raised.value), "the message must name the seam"
    np.testing.assert_array_equal(root.visits, before[0])
    np.testing.assert_array_equal(root.weight, before[1])
    np.testing.assert_array_equal(root.wdl_sum, before[2])
    np.testing.assert_array_equal(root.moment, before[3])


def test_a_positive_leaf_weight_still_backs_up() -> None:
    """The control: the guard refuses nothing the seam is allowed to return.

    A weighter returning a small positive number is a legitimate uncertainty
    weight, and a guard written as ``weight < 1`` or ``weight != 1`` would pass
    every test above and fail here.
    """
    weighter = ScriptedWeight(bad=0.25, good_calls=0)
    config = MatchConfig(num_actions=WIDTH, playouts=8, leaf_weighter=weighter)
    search = MatchSearch(config, constant_wdl_evaluator(GOOD_WDL, num_actions=WIDTH))

    result = search.search(_state())

    assert result.playouts_used == 8
    assert float(result.weights.sum()) == pytest.approx(8 * 0.25)
