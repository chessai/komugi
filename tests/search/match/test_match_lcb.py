"""A3: LCB move selection disagrees with max-visits, and agrees when it must.

PUCT selects during playouts; LCB selects the move to play, once, at the end.
The first is optimistic on purpose and the second pessimistic on purpose, and a
search that used either for the other's job would respectively never commit or
never explore. T2 is the test that the two are actually different rules, and
T-LCB-a is the smoke test that they coincide where they have to.

**How the fixtures are built.** Every test here is **planted**: the root is
expanded by the real search from a real ``TreeGame`` position and then its edge
statistics are written directly, through the same weighted accumulation and the
same ``pack_outer`` the backup uses, so the numbers are written down and each
of the mutants T2 names -- an LCB without the ``/sqrt(W)`` term, an inverted
eligibility test, a dropped ``k_lcb``, and the sign flip T-LCB-a cannot see --
dies to an exact arithmetic assertion nobody has to run a search to check.

Design §D2 additionally pre-registered T2 as a *searched* outcome, backed by
the deterministic sweep in ``planted/find_lcb_fixture.py``. **That sweep was
run against the real search (advisor, 2026-09-04) and found nothing**: on each
of the review's eight candidate tables, at 24 and at 48 playouts, the LCB pick
equalled the max-visits pick. It is structural rather than a gap in the
enumeration -- this
suite's evaluator is the solver, exact values do not spread, so every edge's
variance sits on ``var_floor`` and the LCB penalty becomes monotone in the
visit count alone. The finding is recorded in the sweep script and in
:func:`test_lcb_disagrees_with_max_visits_on_a_planted_root`, which carries T2
on a planted root instead; the frozen table an earlier revision claimed the
sweep had produced is gone rather than re-tuned.
"""

from __future__ import annotations

import math

import numpy as np
# `planted` before `toygames`: importing it is what puts the oracle directory
# on ``sys.path``. See ``planted/__init__.py``.
from planted import MatchToyState, constant_wdl_evaluator, plant_backup
from toygames import TreeGame

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.select import lcb_eligible, lcb_scores, select_action
from komugi.search.match.tree import MatchTree

#: T2's high-variance action: thirty wins and ten losses.
HIGH_VARIANCE = (*(1.0 for _ in range(30)), *(-1.0 for _ in range(10)))

#: T2's steady action: thirty backups of the same middling value.
STEADY = tuple(0.4 for _ in range(30))


def expanded_root(config: MatchConfig, width: int = 2) -> MatchNode:
    """Return a real, freshly expanded root with no statistics on it yet."""
    payoffs = tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
    state = MatchToyState(TreeGame(payoffs))
    tree = MatchTree(
        config,
        constant_wdl_evaluator([0.2, 0.6, 0.2], num_actions=width),
    )
    tree.ensure_root(state)
    root = tree.root
    assert root is not None
    assert root.width == width
    return root


def test_lcb_disagrees_with_max_visits_on_a_high_variance_edge() -> None:
    """T2: the most visited edge is not the one the search plays.

    Edge A carries thirty ``+1`` and ten ``-1`` backups: mean ``0.5``, second
    moment ``1.0``, so variance ``0.75``. Edge B carries thirty backups of
    ``0.4`` with no spread at all, so its variance is the floor. The lower
    bounds are then ``0.5 - 5*sqrt(0.75/40)`` and ``0.4 - 5*sqrt(0.0025/30)``,
    i.e. about ``-0.18`` against ``+0.35`` -- A is ahead on the mean and on
    visits, and behind where it matters.
    """
    config = MatchConfig(num_actions=2)
    root = expanded_root(config)
    plant_backup(root, 0, HIGH_VARIANCE)
    plant_backup(root, 1, STEADY)

    assert int(np.argmax(root.weight)) == 0
    assert float(root.weight[0]) == 40.0
    assert float(root.weight[1]) == 30.0

    bounds = lcb_scores(root, config)
    assert bool(lcb_eligible(root, config).all())
    expected_a = 0.5 - config.k_lcb * math.sqrt(0.75 / 40.0)
    expected_b = 0.4 - config.k_lcb * math.sqrt(config.var_floor**2 / 30.0)
    assert abs(float(bounds[0]) - expected_a) < 1e-12
    assert abs(float(bounds[1]) - expected_b) < 1e-12
    assert float(bounds[1]) > float(bounds[0])

    # The rule under test: the played move is the LCB argmax, not the visit
    # argmax, and the two genuinely differ here.
    assert select_action(root, config) == 1


def test_dropping_the_visit_term_would_flip_the_choice() -> None:
    """T2's mutant, made explicit: ``u - k*sigma`` picks the other move.

    Without the ``/sqrt(W)`` an edge's bound stops shrinking as evidence
    accumulates, and the whole point of a lower confidence bound is that it
    does. On this fixture the mutant scores ``0.5 - 5*sqrt(0.75) = -3.83``
    against ``0.4 - 5*0.05 = 0.15`` -- it happens to agree here, so the
    assertion is on the *numbers*, which differ by more than an order of
    magnitude from the correct ones.
    """
    config = MatchConfig(num_actions=2)
    root = expanded_root(config)
    plant_backup(root, 0, HIGH_VARIANCE)
    plant_backup(root, 1, STEADY)
    bounds = lcb_scores(root, config)
    mutant_a = 0.5 - config.k_lcb * math.sqrt(0.75)
    assert abs(float(bounds[0]) - mutant_a) > 1.0


def test_a_symmetric_node_agrees_with_max_visits() -> None:
    """T-LCB-a: identical children, so LCB, max-visits and the tie-break agree.

    Honestly labelled: this is a **smoke test**. It passes under a
    sign-flipped LCB and under any implementation that breaks ties by the
    lowest action id, so it is evidence about the tie-break and about nothing
    else. T2 above is what carries the sign.
    """
    config = MatchConfig(num_actions=3)
    root = expanded_root(config, width=3)
    for index in range(3):
        plant_backup(root, index, tuple(0.2 for _ in range(10)))

    bounds = lcb_scores(root, config)
    assert float(bounds[0]) == float(bounds[1]) == float(bounds[2])
    assert select_action(root, config) == int(np.argmax(root.weight)) == 0
    assert int(root.actions[0]) == 0


def test_lcb_never_selects_an_edge_with_no_weight() -> None:
    """A3's second invariant: an unvisited edge has no lower bound to win with.

    Falsifiable trivially, and it is the failure mode of an eligibility test
    written with the wrong comparison -- which would let the ``-inf`` an
    unvisited edge scores become the argmax under a flipped sign.
    """
    config = MatchConfig(num_actions=3)
    root = expanded_root(config, width=3)
    plant_backup(root, 1, tuple(-0.9 for _ in range(8)))
    plant_backup(root, 2, tuple(-0.8 for _ in range(8)))

    bounds = lcb_scores(root, config)
    assert bounds[0] == -np.inf
    assert not bool(lcb_eligible(root, config)[0])
    assert select_action(root, config) == 2


def test_with_nothing_eligible_the_choice_falls_back_to_max_weight() -> None:
    """A3's clause 3: at a very low budget, plain max-visits, lowest id first."""
    config = MatchConfig(num_actions=3)
    root = expanded_root(config, width=3)
    plant_backup(root, 2, (0.9,))
    # One visit is not a variance estimate, so nothing clears the 2.0 floor.
    assert not bool(lcb_eligible(root, config).any())
    assert select_action(root, config) == 2


#: T2's action ``A``: thirty-six ``+1`` backups and twelve ``-1``. Mean
#: ``+0.5``, second moment ``1.0`` (every value is ``+-1``), so ``var = 0.75``
#: -- and forty-eight visits, more than twice ``B``'s.
T2_HIGH_VARIANCE = (*(1.0 for _ in range(36)), *(-1.0 for _ in range(12)))

#: T2's action ``B``: eighteen backups of a steady ``0.3``. No spread at all,
#: so its variance is ``var_floor**2`` and its bound is its mean minus the
#: floor penalty.
T2_STEADY = tuple(0.3 for _ in range(18))

#: What the two edges' weights come to, named so the eligibility threshold
#: below is derivable rather than asserted from a magic number.
T2_WEIGHT_A = float(len(T2_HIGH_VARIANCE))
T2_WEIGHT_B = float(len(T2_STEADY))


def test_lcb_disagrees_with_max_visits_on_a_planted_root() -> None:
    """T2: the most-visited edge, ahead on the mean, is not the move played.

    **This is planted rather than searched, and that is a finding rather than
    a shortcut.** Design §D2 pre-registered T2 as a *search* outcome backed by
    the sweep in ``planted/find_lcb_fixture.py``. The advisor ran that sweep
    against the real search on 2026-09-04 -- the review's eight candidate
    tables, at 24 and at 48 playouts -- and **the LCB pick equalled the
    max-visits pick on every one of them**. An earlier revision of this file
    froze a table as though the sweep had succeeded; that literal asserted a
    disagreement the search does not produce and is gone.

    The reason is structural and is why no further sweeping would help. This
    suite's evaluator is the solver, so every leaf carries the position's
    *exact* value; an edge's variance is the spread of what is backed up
    through it, and exact values do not spread. Once the replies under an edge
    are resolved every playout through it returns the same number, the second
    moment equals the squared mean, and ``var(a)`` sits on ``var_floor`` for
    **every** edge. With a common floor the penalty ``k_lcb * sqrt(var/W)`` is
    monotone in ``W`` alone, so the most-visited edge takes both the best mean
    and the smallest penalty, and the two rules cannot come apart. Exhibiting
    the disagreement needs an evaluator whose *errors* differ between siblings
    -- a network -- which is not a fixture this suite can plant.

    So the statistics are written down instead, through the same weighted
    accumulation and the same ``pack_outer`` the backup uses, and every number
    below is derivable by hand from this docstring:

    * ``A``: 48 visits, mean ``0.5``, ``var = 1.0 - 0.5**2 = 0.75``, so
      ``lcb(A) = 0.5 - 5*sqrt(0.75/48) = 0.5 - 0.625 = -0.125``;
    * ``B``: 18 visits, mean ``0.3``, variance on the floor, so
      ``lcb(B) = 0.3 - 5*sqrt(0.05**2/18) = 0.3 - 0.0589 = 0.2411``.

    ``A`` leads on visits (48 to 18) and on the mean (0.5 to 0.3) and loses on
    the bound by more than a third of a point, which is the *sign* of the
    disagreement: the rule is pessimistic, and it is the spread that it
    punishes rather than the visit count that it rewards.

    *Kills:* LCB implemented as ``u - k*sigma`` without the ``/sqrt(W)`` term
    (asserted on the numbers by the test below it); a dropped ``k_lcb``, under
    which both bounds are the means and ``A`` wins; an inverted eligibility
    test, which would exclude ``B`` at ``0.15 * 48 = 7.2``; and the sign flip
    T-LCB-a cannot see, under which ``u + k*sqrt(var/W)`` puts ``A`` ahead
    again -- asserted explicitly, because a fixture where the mutant happens to
    agree proves nothing about the sign.
    """
    config = MatchConfig(num_actions=2)
    root = expanded_root(config)
    plant_backup(root, 0, T2_HIGH_VARIANCE)
    plant_backup(root, 1, T2_STEADY)

    # The premise: A is the most visited edge *and* the one ahead on the mean,
    # so neither visits nor the raw utility can explain the choice below.
    assert float(root.weight[0]) == T2_WEIGHT_A
    assert float(root.weight[1]) == T2_WEIGHT_B
    assert int(np.argmax(root.weight)) == 0
    assert T2_WEIGHT_A > 2.0 * T2_WEIGHT_B

    # Both edges clear `max(2.0, 0.15 * 48) = 7.2`, so the disagreement is
    # about the bounds and not about eligibility.
    assert float(root.weight[1]) > config.min_visit_prop_for_lcb * T2_WEIGHT_A
    assert bool(lcb_eligible(root, config).all())

    bounds = lcb_scores(root, config)
    expected_a = 0.5 - config.k_lcb * math.sqrt(0.75 / T2_WEIGHT_A)
    expected_b = 0.3 - config.k_lcb * math.sqrt(config.var_floor**2 / T2_WEIGHT_B)
    assert abs(expected_a - (-0.125)) < 1e-12
    assert abs(float(bounds[0]) - expected_a) < 1e-12
    assert abs(float(bounds[1]) - expected_b) < 1e-12

    # The sign of the disagreement, and its size: B is behind on every number
    # the search stores and ahead on the one it plays by.
    assert float(bounds[1]) - float(bounds[0]) > 0.3
    assert select_action(root, config) == 1
    assert select_action(root, config) != int(np.argmax(root.weight))

    # The sign-flip mutant, made explicit: an *upper* bound reverses the
    # verdict, so this fixture is evidence about the sign and not only about
    # the tie-break.
    upper_a = 0.5 + config.k_lcb * math.sqrt(0.75 / T2_WEIGHT_A)
    upper_b = 0.3 + config.k_lcb * math.sqrt(config.var_floor**2 / T2_WEIGHT_B)
    assert upper_a > upper_b
