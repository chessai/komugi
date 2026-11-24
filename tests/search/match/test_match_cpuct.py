"""A1: the variance machinery, and the two corrections rev 2 exists for.

Design §A1's ``SS(n) -> sigma(n) -> phi(n) -> c_puct(n)`` chain ships in
``select.py`` and, before this file, was called nowhere in the suite except
``test_match_contempt.py``'s ``read_every_path``, which discards every result.
Two mutants therefore survived the whole suite -- and they are precisely the
two defects rev 2 was written to correct (review F1, mutants M1 and M2):

* **M1** drops the **between-child** term from ``SS(n)``, summing only
  ``sum_a [q'M(a)q - W(a)u(a)**2]``. That is rev 1's within-edge-only form,
  and it inverts the feature: a node whose children maximally *disagree* --
  one backing ``+1`` every playout, one backing ``-1`` -- has zero within-edge
  spread on both edges, so the mutant reports ``SS = 0`` and scales exploration
  **down** at exactly the node that most needs it scaled up.
* **M2** replaces the ``sqrt(1 + W_tot(n))`` numerator with ``sqrt(W_tot(n))``.
  At ``W_tot = 0`` the exploration term is then zero for every child, and since
  FPU is child-independent every child scores exactly ``u_parent``: the first
  descent from a fresh node is decided by ``np.argmax``'s tie-break, i.e. the
  lowest legal action id -- an action-encoding artefact over a 37341-wide
  space, with the policy prior having no influence whatsoever.

The first two tests below are planted rather than searched: the statistics are
written down, so every number asserted is derivable by hand from this
docstring. The third is **T1**, which is a search outcome and therefore rests
on a swept fixture rather than on a hand-written table -- see its docstring for
the sweep and the run that found it.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# `planted` before `toygames`: importing it is what puts the oracle directory
# on ``sys.path``. See ``planted/__init__.py``.
from planted import (
    MatchToyState,
    constant_wdl_evaluator,
    plant_backup,
    solver_wdl_evaluator,
)
from toygames import TreeGame

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.search import MatchSearch
from komugi.search.match.select import (
    exploration_constant,
    node_sigma,
    node_sum_of_squares,
    node_utility,
    select_child,
    variance_scale,
)
from komugi.search.match.tree import MatchTree

#: Twenty ``+1`` backups on one edge and twenty ``-1`` on the other: the
#: design's own worked case for the between-child term (§A1, rev 2/B1).
BACKUPS = 20

#: Non-uniform logits for the first-descent test. Action 1 is the prior's
#: argmax and is deliberately **not** action 0, which is what makes "argmax of
#: the prior" and "lowest legal action id" distinguishable answers.
LOGITS = np.array([0.0, 1.5, 0.5], dtype=np.float32)


def expanded_root(
    config: MatchConfig,
    width: int,
    logits: npt.NDArray[np.float32] | None = None,
) -> MatchNode:
    """Return a real, freshly expanded root carrying no statistics yet.

    Built through :class:`MatchTree` rather than by calling
    :class:`MatchNode` directly, so that ``prior``, ``is_root`` and the array
    shapes are the ones the search itself produces.
    """
    payoffs = tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
    tree = MatchTree(
        config,
        constant_wdl_evaluator(
            [0.2, 0.6, 0.2], num_actions=width, logits=logits
        ),
    )
    tree.ensure_root(MatchToyState(TreeGame(payoffs)))
    root = tree.root
    assert root is not None
    assert root.width == width
    return root


def test_sigma_sees_between_child_disagreement() -> None:
    """M1's kill: ``SS(n)`` is about the **node's** mean, not each edge's.

    Twenty ``+1`` backups on edge 0 and twenty ``-1`` on edge 1. Every value
    is a one-hot, so ``q'M(0)q = 20`` and ``q'M(1)q = 20``; the node's own mean
    is ``(20 - 20)/40 = 0``; so

        SS = (20 + 20) - 40 * 0**2 = 40.

    The within-edge-only form of M1 computes ``(20 - 20*(+1)**2) +
    (20 - 20*(-1)**2) = 0`` instead: both edges are individually perfectly
    consistent and the disagreement between them is exactly what it throws
    away.

    **Under M1 the first assertion fires** -- ``node_sum_of_squares`` returns
    ``0.0`` where ``40.0`` is asserted -- and, were it loosened, so would the
    two after it: ``sigma`` would be ``sqrt(0.32/42) = 0.087`` rather than
    ``0.980``, and ``phi`` ``0.335`` rather than ``2.23``, i.e. the mutant
    scales exploration **down** by a factor of six at a node whose children
    could not disagree more.
    """
    config = MatchConfig(num_actions=2)
    root = expanded_root(config, width=2)
    plant_backup(root, 0, (1.0,) * BACKUPS)
    plant_backup(root, 1, (-1.0,) * BACKUPS)

    total = float(2 * BACKUPS)
    assert root.total_weight() == total
    # The node's own mean is zero, which is what makes the between-child term
    # the whole of SS here rather than merely a contributor to it.
    assert abs(node_utility(root, config)) < 1e-12

    assert abs(node_sum_of_squares(root, config) - 40.0) < 1e-12

    expected_sigma = math.sqrt(
        (config.sigma_prior_weight * config.sigma_prior**2 + 40.0)
        / (config.sigma_prior_weight + total)
    )
    assert abs(expected_sigma - 0.9797958971132712) < 1e-12
    assert abs(node_sigma(root, config) - expected_sigma) < 1e-12

    expected_phi = (1.0 - config.sigma_scale) + config.sigma_scale * (
        expected_sigma / config.sigma_prior
    )
    assert variance_scale(root, config) > 1.2
    assert abs(variance_scale(root, config) - expected_phi) < 1e-12

    # And the consequence the feature exists for: exploration at this node is
    # scaled *up*, not down. The mutant's `phi` is below one.
    growth = config.c_log * math.log(
        (total + config.c_base) / config.c_base
    )
    assert abs(
        exploration_constant(root, config)
        - (config.c_explore + growth) * expected_phi
    ) < 1e-12
    assert exploration_constant(root, config) > config.c_explore

    # The control: with the same evidence but no disagreement between the
    # edges, SS is zero and phi is *below* one. Without this, "SS = 40" could
    # be produced by any implementation that ignored the values entirely.
    agreeing = expanded_root(config, width=2)
    plant_backup(agreeing, 0, (1.0,) * BACKUPS)
    plant_backup(agreeing, 1, (1.0,) * BACKUPS)
    assert abs(node_sum_of_squares(agreeing, config)) < 1e-12
    assert variance_scale(agreeing, config) < 1.0


def test_first_descent_is_argmax_prior() -> None:
    """M2's kill: at ``W_tot = 0`` the descent follows the policy.

    A fresh root, no statistics anywhere. Every edge is unvisited, so every
    edge takes the same child-independent FPU value, and the exploration term
    is ``c_explore * P(a) * sqrt(1 + 0) / (1 + 0) = c_explore * P(a)``: the
    ordering is the prior's, exactly.

    **Under M2 the first assertion fires**: ``sqrt(W_tot)`` is ``0``, so every
    child scores exactly ``u_parent``, ``np.argmax`` returns the lowest index,
    and ``select_child`` yields ``0`` where ``argmax(prior) == 1`` is
    asserted. The ``phi == 1.0`` and ``c_puct == c_explore`` assertions below
    are exact rather than toleranced, because at ``W_tot = 0`` both reduce by
    an early return and ``log(1)`` rather than by arithmetic that happens to
    land close.
    """
    width = int(LOGITS.shape[0])
    config = MatchConfig(num_actions=width)
    root = expanded_root(config, width=width, logits=LOGITS)

    assert root.total_weight() == 0.0
    best = int(np.argmax(root.prior))
    assert best == 1, "the fixture's prior argmax must not be action 0"
    assert float(root.prior[best]) > float(root.prior[0])

    assert select_child(root, config) == best

    # The two exact identities design §A1 derives the claim from.
    assert node_sum_of_squares(root, config) == 0.0
    assert node_sigma(root, config) == config.sigma_prior
    assert variance_scale(root, config) == 1.0
    assert exploration_constant(root, config) == config.c_explore


#: T1's fixture, **frozen from the sweep** ``planted/find_cpuct_fixture.py``
#: (advisor run, 2026-09-04): the first candidate in that script's fixed
#: enumeration order for which all four of T1's clauses hold. The rows
#: minimax to ``+1`` and ``-1``, so the root's children maximally disagree and
#: the between-child term of ``SS(root)`` is large.
T1_TABLE = ((1.0, 1.0), (-1.0, -1.0))

#: ...and the prior points at action 1, the child the *values* do not support.
#: That is the whole lever: whether the node buys enough exploration to find
#: out is a decision the exploration constant makes. The mass is ``3.0`` and
#: not the ``1.0`` an earlier revision froze: at ``e**1 : 1`` the two arms
#: return the **same** action, so that candidate discriminates nothing and the
#: sweep's first clause fails on it (advisor probe, 2026-09-04).
T1_LOGITS = np.array([0.0, 3.0], dtype=np.float32)

#: The high-prior action, named rather than recomputed from the logits: the
#: sweep enumerated it alongside the table and the test asserts the two agree.
T1_HIGH_PRIOR_ACTION = 1

#: The sweep's playout budget, ``find_cpuct_fixture.PLAYOUTS``, pinned as a
#: literal here. A fixture found at one budget is not a fixture at another,
#: and the test must not follow a config default that moves.
T1_PLAYOUTS = 48


def t1_search(dynamic: bool) -> tuple[int, list[int], float]:
    """Run one arm of T1 and return ``(action, visits, phi(root))``.

    Both arms are the same search but for design §A1's variance scaling:
    ``dynamic_cpuct=False`` uses the fixed :attr:`MatchConfig.c_puct` and makes
    ``phi`` identically ``1.0``, which is what makes the fixed arm a control
    rather than a second dynamic run with different constants.

    ``subtree_bias_factor`` is pinned to ``0.0``, and the pin is
    load-bearing rather than defensive: this is a test about the *exploration
    constant*, and the probe that measured the fixture ran with the bias
    correction off, so ``lambda = 0`` is the configuration it was actually
    found under. It is also what keeps the fixture stable across changes to
    design §A2 -- the correction is multiplied by zero here, so neither arm
    reads the table at all. Whether the same table still separates the two arms
    at the shipped ``lambda = 0.45`` is a different question, and not one a
    frozen fixture can answer.
    """
    config = MatchConfig(
        num_actions=len(T1_TABLE),
        playouts=T1_PLAYOUTS,
        dynamic_cpuct=dynamic,
        c_puct=1.0,
        subtree_bias_factor=0.0,
    )
    search = MatchSearch(
        config,
        solver_wdl_evaluator(num_actions=len(T1_TABLE), logits=T1_LOGITS),
    )
    result = search.search(MatchToyState(TreeGame(T1_TABLE)), ply=0)
    root = search.tree.root
    assert root is not None
    phi = variance_scale(root, config)
    return result.action, [int(count) for count in result.visits], phi


def test_dynamic_cpuct_changes_the_pick_against_fixed_cpuct() -> None:
    """T1: the two arms return different actions, for the stated reason.

    *Provenance.* "The dynamic arm picks differently" is a **search outcome**,
    not something anybody can write down by hand, so design §D2 requires the
    fixture to come from a committed deterministic sweep
    (``tests/search/match/planted/find_cpuct_fixture.py``, fixed enumeration
    order, no RNG) and to be frozen here as a literal, with the test not
    running the sweep. This is that sweep's **third** candidate: payoffs
    ``((1.0, 1.0), (-1.0, -1.0))`` with logits ``(0.0, 3.0)`` and the high
    prior on action 1, at 48 playouts (advisor probe, 2026-09-04), measured as

        dynamic: action 1, visits ``[0, 48]``  /  fixed: action 0, visits
        ``[17, 31]``

    The **first** candidate -- the same table at logits ``(0.0, 1.0)``, which
    an earlier revision froze -- does not discriminate at this budget: the
    probe found its clauses no longer separating the two arms, so the test
    carrying it asserted nothing about the feature. A fixture that does not
    separate the arms is not a weaker fixture, it is not a fixture.

    The four clauses, which are exactly the sweep's:

    1. the arms return **different** actions -- and which is which is pinned,
       because "they differ" is also satisfied by the two arms swapping;
    2. the dynamic arm spends **more** on the high-prior child than the fixed
       arm does -- here everything it has;
    3. ``phi(root) != 1.0`` in the dynamic arm: the scaling is live rather than
       a constant the config never reaches;
    4. ``phi`` is identically ``1.0`` in the fixed arm, exactly.

    Clause 3 was ``phi(root) > 1.2`` when the design pre-registered it, and it
    is weakened here for a structural reason recorded in the sweep script:
    ``phi`` is read *after* the search, and the concentration that makes the
    dynamic arm choose differently is exactly what destroys the between-child
    spread ``phi`` is computed from -- with all 48 visits on one edge and every
    backup through it the same ``-1``, ``SS(root)`` is ``0`` and ``phi`` is
    about ``0.32``. No fixture can both diverge and end wide, so the clause
    that survives is that the two arms run genuinely different exploration
    constants.

    *Kills:* a ``phi(n)`` stuck at ``1.0`` (the variance scaling never wired
    up) fails clauses 1 and 3 together; and a ``sqrt(W_tot)`` exploration
    numerator, under which both arms descend prior-blind on the first visit and
    clause 2 fails.
    """
    dynamic_action, dynamic_visits, dynamic_phi = t1_search(True)
    fixed_action, fixed_visits, fixed_phi = t1_search(False)

    # The fixture is what the sweep says it is: the prior's argmax is the
    # action the payoffs do not support.
    assert int(np.argmax(T1_LOGITS)) == T1_HIGH_PRIOR_ACTION
    assert min(T1_TABLE[0]) > 0.0 > max(T1_TABLE[1])

    assert dynamic_action != fixed_action
    assert dynamic_action == T1_HIGH_PRIOR_ACTION
    assert fixed_action != T1_HIGH_PRIOR_ACTION
    assert (
        dynamic_visits[T1_HIGH_PRIOR_ACTION] > fixed_visits[T1_HIGH_PRIOR_ACTION]
    )
    assert dynamic_phi != 1.0
    assert fixed_phi == 1.0

    # Both arms spent the whole budget: the difference is where it went, not
    # how much of it there was.
    assert sum(dynamic_visits) == sum(fixed_visits) == T1_PLAYOUTS
