"""Selection: the PUCT descent (design §A1) and the LCB move choice (§A3).

Two rules, and confusing them is the mistake this module's split exists to
prevent. **PUCT selects during playouts**: it decides where the next
simulation goes, and it is deliberately optimistic. **LCB selects the move to
play**, once, at the end, and it is deliberately pessimistic -- a child whose
mean is high only because two of its three playouts got lucky is not a move to
stake a game on.

Everything here reads and nothing writes. Every scalar is derived from
contempt-free storage at call time, with the contempt taken from the `config`
argument on every call: no utility, no ``q`` and no variance is ever cached on
a node, which is what makes a retained tree survive a mid-game contempt change
(design §S2).

Design §A2's bias correction appears in the formulas below as
:func:`child_bias_correction`, and this revision **wires it**: the table
(:mod:`komugi.search.match.bias`) is passed in by the caller, per tree, and its
reading is added to each child's direct neural mass before the collapse. The
``lambda = 0`` bit-identity golden that acceptance A4's two-commit ordering
requires was recorded from the previous commit, where these call sites did not
exist yet, so it is a golden of a bias-free search rather than a recording of
the feature under test (T8, ``tests/search/match/test_match_bias_table.py``).

**The table is an argument, never a global** (design P-D3), and it is
*optional*: ``bias=None`` means "no table to consult" and makes the correction
identically zero. That is what a unit test calling one of these functions on a
hand-planted node passes, and it is why every such call site keeps working
unchanged. The search itself always passes its tree's table -- both on the
descent (:mod:`komugi.search.match.tree`) and at the LCB choice
(:mod:`komugi.search.match.search`), which are the two readers design §A2
names.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from komugi.search.match.bias import SubtreeValueBias
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import (
    WDL,
    MatchNode,
    bias_key_of,
    edge_action_of,
    quadratic_form,
    utility,
)


def _require_children(node: MatchNode, what: str) -> None:
    """Raise if `node` has no edges to compare, naming what was asked for."""
    if node.width == 0:
        raise ValueError(
            f"{what} on a node with no legal actions: a terminal position has "
            f"no children to compare, and returning an index into an empty "
            f"array would be a silent out-of-bounds"
        )


def node_utility(node: MatchNode, config: MatchConfig) -> float:
    """Return ``u_node(n)``: `node`'s own weighted mean utility.

    ``q(c) . (sum_a S(a)) / W_tot(n)`` once anything has been backed up, and
    the collapse of the node's own network WDL before that. Both forms are
    derived from contempt-free storage at read time, so neither can poison a
    reused tree.

    Args:
        node: The node to read.
        config: Supplies the contempt, read here and not stored.

    Returns:
        The mean utility in `node`'s own mover perspective.
    """
    total = node.total_weight()
    if total <= 0.0:
        return utility(node.wdl_net, config.contempt)
    return utility(node.subtree_wdl_sum(), config.contempt) / total


def node_sum_of_squares(node: MatchNode, config: MatchConfig) -> float:
    """Return ``SS(n)``: the sum of squares about **the node's own** mean.

    ``sum_a q'M(a)q - W_tot(n) * u_node(n)**2``, which decomposes as::

        SS = sum_a [ q'M(a)q - W(a)u(a)**2 ]        # within-child spread
           + [ sum_a W(a)u(a)**2 - W_tot u_node**2 ] # between-child disagreement

    The bracketed **between**-child term is "the children disagree about this
    position", and it is the term the whole feature exists to see: a node with
    one child backing up ``+1`` every playout and another backing up ``-1``
    has *zero* within-child spread on both edges while its true utility spread
    is maximal. Summing only the within-child part would scale exploration
    *down* at exactly that node -- the precise inversion of the feature's
    purpose.

    The result is clamped at zero. In exact arithmetic Cauchy-Schwarz makes it
    non-negative, but on a converged node the two terms are nearly equal large
    numbers and cancellation can leave the difference slightly negative;
    ``sqrt`` of that is ``nan``, which propagates silently into a move choice.
    """
    total = node.total_weight()
    if total <= 0.0:
        return 0.0
    mean = node_utility(node, config)
    squares = float(quadratic_form(node.moment, config.contempt).sum())
    return max(squares - total * mean * mean, 0.0)


def node_sigma(node: MatchNode, config: MatchConfig) -> float:
    """Return ``sigma(n)``: the prior-blended stdev of `node`'s utilities.

    ``sqrt((w0 * sigma_prior**2 + SS(n)) / (w0 + W_tot(n)))``, floored at
    ``config.sigma_floor`` so that an all-identical subtree cannot drive
    exploration to zero. At ``W_tot(n) == 0`` this is exactly
    ``sigma_prior``, which is what makes the variance scaling a no-op on a
    node's first descent by construction rather than by luck.
    """
    total = node.total_weight()
    weighted = (
        config.sigma_prior_weight * config.sigma_prior * config.sigma_prior
        + node_sum_of_squares(node, config)
    )
    sigma = math.sqrt(weighted / (config.sigma_prior_weight + total))
    return max(sigma, config.sigma_floor)


def variance_scale(node: MatchNode, config: MatchConfig) -> float:
    """Return ``phi(n) = (1 - s) + s * sigma(n) / sigma_prior``.

    Identically ``1.0`` when :attr:`MatchConfig.dynamic_cpuct` is off, which
    is what makes the fixed arm of design §D2's T1 a clean control rather than
    a second dynamic run with different constants.
    """
    if not config.dynamic_cpuct:
        return 1.0
    scale = config.sigma_scale
    return (1.0 - scale) + scale * node_sigma(node, config) / config.sigma_prior


def exploration_constant(node: MatchNode, config: MatchConfig) -> float:
    """Return ``c_puct(n)``, the exploration constant in force at `node`.

    ``[c_explore + c_log * ln((W_tot(n) + c_base) / c_base)] * phi(n)``, or
    the fixed :attr:`MatchConfig.c_puct` when dynamic cPUCT is off. At
    ``W_tot(n) == 0`` the logarithm is exactly zero and ``phi`` exactly one,
    so this reduces to ``c_explore``.
    """
    if not config.dynamic_cpuct:
        return config.c_puct
    total = node.total_weight()
    growth = config.c_log * math.log((total + config.c_base) / config.c_base)
    return (config.c_explore + growth) * variance_scale(node, config)


def child_bias_correction(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> npt.NDArray[np.float64]:
    """Return ``lambda * b(key(a)) * direct_neural_fraction`` per edge.

    The key of an edge is ``(node.to_play, the action, the action played into
    node)``, and the last of those is recovered from `node`'s own
    ``bias_key`` by :func:`~komugi.search.match.node.edge_action_of` -- so the
    key read here is *the same key* the playout loop files the edge's child
    under at expansion, at the root as well as at an interior node.

    **The reading is already in this node's frame.** The writer
    (:meth:`~komugi.search.match.tree.MatchTree._update_bias`) stores every
    contribution in the perspective of the mover design §A2's key names -- the
    mover at the *parent* of the edge, which is `node` here -- so the vector
    returned lines up slot for slot with ``S(a)/W(a)``, which design §S1 also
    stores in `node`'s perspective. Nothing is swapped on this side, and a
    correction swapped on neither side would enter every mover-changing edge
    with its win and loss slots transposed, i.e. with its sign reversed after
    the collapse.

    **The table is consulted only where the edge has weight.** An unvisited
    edge's mean WDL is itself a masked placeholder that no caller reads
    (:func:`child_mean_wdl`), so a correction applied to it could not reach a
    selection score, an LCB or anything else; skipping it keeps this off the
    hot loop's allocation path for the many legal actions a match position has
    and no playout has yet touched.

    With `bias` ``None`` the reading is all zeros, and with
    ``subtree_bias_factor == 0.0`` the whole term is zero whatever the table
    says -- which is the off-switch acceptance A4 pins against the golden.

    Args:
        node: The node whose edges are being corrected.
        config: Supplies ``lambda`` and the packing radix of the key.
        bias: The tree's table, or ``None`` for no correction at all.

    Returns:
        ``(L, 3)`` float64, the correction to add to each edge's mean.

    For an existing nonterminal child, incoming weight minus outgoing weight
    recovers the first successful neural backup's mass. Every later backup
    uses the same positive weight along the path, even with a stateful weighter.
    Missing children provide insufficient provenance; terminal children and
    zero-weight edges also receive zero. Clip the fraction for rounding.
    Float64 cancellation can lose tiny direct mass beside huge descendant mass;
    this does not solve total-statistic overflow. Future selective eviction,
    graph sharing or reevaluation would require explicit direct-mass storage.
    """
    table_reading = np.zeros((node.width, WDL), dtype=np.float64)
    if bias is not None:
        previous_action = edge_action_of(node.bias_key, config.num_actions)
        for index in range(node.width):
            weight = float(node.weight[index])
            child = node.children.get(index)
            if weight <= 0.0 or child is None or child.is_terminal:
                continue
            fraction = float(np.clip((weight - child.total_weight()) / weight, 0.0, 1.0))
            table_reading[index] = fraction * bias.read(
                bias_key_of(
                    node.to_play,
                    int(node.actions[index]),
                    previous_action,
                    config.num_actions,
                )
            )
    return np.asarray(
        config.subtree_bias_factor * table_reading, dtype=np.float64
    )


def child_mean_wdl(node: MatchNode) -> npt.NDArray[np.float64]:
    """Return ``x_bar(a) = S(a) / W(a)`` per edge, zeros where ``W(a) == 0``.

    An unvisited edge has no mean; the zero row is a placeholder that every
    caller masks off, never a claim that the edge is worth a draw.
    """
    weights = node.weight
    visited = weights > 0.0
    means = np.zeros((node.width, WDL), dtype=np.float64)
    np.divide(
        node.wdl_sum,
        weights.reshape(-1, 1),
        out=means,
        where=visited.reshape(-1, 1),
    )
    return means


def corrected_child_wdl(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> npt.NDArray[np.float64]:
    """Return raw mean plus the pooled direct-neural WDL adjustment per edge.

    The bias lives in WDL space rather than utility space precisely so that
    this addition is contempt-free; because the collapse is linear, a shift
    in WDL space still acts as a shift in utility space.

    The correction is applied at **read time only**: ``S(a)`` and ``M(a)`` are
    never written back with a bias in them, so turning the bias off mid-tree,
    or flipping contempt, leaves the raw statistics intact (design §A2).
    """
    return np.asarray(
        child_mean_wdl(node) + child_bias_correction(node, config, bias=bias),
        dtype=np.float64,
    )


def _collapse(
    triples: npt.NDArray[np.float64], contempt: float
) -> npt.NDArray[np.float64]:
    """Collapse ``(L, 3)`` triples to ``(L,)`` utilities under `contempt`."""
    return np.asarray(
        triples[:, 0] - triples[:, 2] + contempt * triples[:, 1], dtype=np.float64
    )


def child_raw_utilities(
    node: MatchNode, config: MatchConfig
) -> npt.NDArray[np.float64]:
    """Return ``u_raw(a) = q(c) . x_bar(a)``: the **uncorrected** utilities.

    The second moment was accumulated from uncorrected backups, so a variance
    formed against the corrected mean would be a variance about the wrong
    centre -- and would go negative whenever the bias was large. This is the
    mean the variance of design §A3 is formed against.
    """
    return _collapse(child_mean_wdl(node), config.contempt)


def child_utilities(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> npt.NDArray[np.float64]:
    """Return ``U(a)``: the selection utility of every edge of `node`.

    ``clip(q(c) . x_corr(a), -1, +1)`` on a visited edge, and design §A1's
    first-play-urgency value on an unvisited one. The clip is on the
    **utility**, after the collapse, and never on the WDL: clipping the triple
    would break the sum-to-one property the second-moment arithmetic relies
    on.
    """
    utilities = np.clip(
        _collapse(corrected_child_wdl(node, config, bias=bias), config.contempt),
        -1.0,
        1.0,
    )
    visited = node.weight > 0.0
    return np.asarray(
        np.where(visited, utilities, first_play_urgency(node, config)),
        dtype=np.float64,
    )


def first_play_urgency(node: MatchNode, config: MatchConfig) -> float:
    """Return the score design §A1 gives an edge of `node` that has no weight.

    ``u_parent(n) - r * sqrt(sum of P(b) over the explored edges b)``, with
    ``r`` the root or interior reduction depending on
    :attr:`MatchNode.is_root`. It is deliberately **child-independent** -- it
    is a property of the parent's estimate and of how much prior mass has
    already been spent, not of the action. That costs nothing, because the
    ``sqrt(1 + W_tot(n))`` numerator of the exploration term already supplies
    the per-child ordering at ``W_tot(n) == 0``: every child then scores
    ``u_parent - 0 + c_puct * P(a)``, so a fresh node's first descent is
    exactly ``argmax_a P(a)``.
    """
    reduction = config.fpu_reduction_root if node.is_root else config.fpu_reduction
    explored = float(node.prior[node.weight > 0.0].sum())
    return node_utility(node, config) - reduction * math.sqrt(max(explored, 0.0))


def puct_scores(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> npt.NDArray[np.float64]:
    """Return the full PUCT score of every edge of `node`.

    ``U(a) + c_puct(n) * P(a) * sqrt(1 + W_tot(n)) / (1 + W(a))``.

    **The numerator is ``sqrt(1 + W_tot(n))``, not ``sqrt(W_tot(n))``.** With
    the latter, a freshly expanded node scores every child at exactly
    ``u_parent`` -- the exploration term is zero for all of them and FPU is
    child-independent -- so the tie-break takes the numerically lowest legal
    action id, an action-encoding artefact uncorrelated with move quality over
    a 37341-wide space, with the policy prior having no influence at all.
    """
    _require_children(node, "puct_scores")
    total = node.total_weight()
    exploration = (
        exploration_constant(node, config)
        * node.prior.astype(np.float64)
        * math.sqrt(1.0 + total)
        / (1.0 + node.weight)
    )
    return np.asarray(
        child_utilities(node, config, bias=bias) + exploration, dtype=np.float64
    )


def select_child(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> int:
    """Return the index into `node`'s edges the next playout descends through.

    ``argmax`` of :func:`puct_scores`, with ties broken by the lowest action
    id -- which is exactly ``np.argmax`` semantics, because
    :attr:`MatchNode.actions` is ascending. The same tie-break Gumbel's
    interior selection documents, and no randomness anywhere.
    """
    return int(np.argmax(puct_scores(node, config, bias=bias)))


def child_variances(node: MatchNode, config: MatchConfig) -> npt.NDArray[np.float64]:
    """Return ``var(a)``, floored at ``var_floor**2``, for every edge.

    In the default ``"moment"`` mode this is the *empirical* variance of the
    utilities backed up through the edge, recovered exactly from the
    contempt-free second moment: ``q'M(a)q / W(a) - u_raw(a)**2``. In
    ``"plugin"`` mode it is instead the closed-form single-draw variance of
    the mean WDL, which stores nothing but measures distributional spread
    rather than disagreement across playouts (design §S3's memory escape
    hatch, and its semantic cost is stated there rather than discovered).

    The floor is not cosmetic: cancellation between two nearly equal large
    numbers can make the difference slightly negative on a converged edge, and
    ``sqrt`` of that is ``nan``, which would propagate silently into a move
    choice.
    """
    floor = config.var_floor * config.var_floor
    weights = node.weight
    visited = weights > 0.0
    if config.variance_mode == "plugin":
        # sum_k x_k q_k**2 - (sum_k x_k q_k)**2, with q = (1, c, -1) so that
        # q**2 = (1, c**2, 1): the variance of a single draw from the mean WDL.
        means = child_mean_wdl(node)
        squared_contempt = config.contempt * config.contempt
        squares = means[:, 0] + means[:, 2] + squared_contempt * means[:, 1]
        centre = _collapse(means, config.contempt)
        raw = np.asarray(squares - centre * centre, dtype=np.float64)
    else:
        raw = np.zeros(node.width, dtype=np.float64)
        np.divide(
            quadratic_form(node.moment, config.contempt),
            weights,
            out=raw,
            where=visited,
        )
        centre = child_raw_utilities(node, config)
        raw = raw - centre * centre
    return np.asarray(np.where(visited, np.maximum(raw, floor), floor), dtype=np.float64)


def lcb_eligible(node: MatchNode, config: MatchConfig) -> npt.NDArray[np.bool_]:
    """Return which edges are eligible for the LCB choice.

    ``W(a) >= max(2.0, min_visit_prop_for_lcb * max_b W(b))``. The ``2.0``
    floor is there because a one-visit variance estimate is not one, and it is
    also what makes "LCB never selects an edge with ``W(a) == 0``" true by
    construction rather than by hope.
    """
    weights = node.weight
    best = float(weights.max()) if node.width else 0.0
    threshold = max(2.0, config.min_visit_prop_for_lcb * best)
    return np.asarray(weights >= threshold, dtype=np.bool_)


def lcb_scores(
    node: MatchNode, config: MatchConfig, *, bias: SubtreeValueBias | None = None
) -> npt.NDArray[np.float64]:
    """Return ``lcb(a) = u(a) - k_lcb * sqrt(var(a) / W(a))`` per edge.

    ``u(a)`` is the design §A2-corrected utility, uncircumscribed by the
    selection clip; ``var(a)`` is formed about the **uncorrected** mean, for
    the reason :func:`child_raw_utilities` gives. An edge with no weight has
    no lower confidence bound at all and scores ``-inf``, so it can never win
    the comparison even if a caller ignores :func:`lcb_eligible`.
    """
    _require_children(node, "lcb_scores")
    weights = node.weight
    visited = weights > 0.0
    utilities = _collapse(
        corrected_child_wdl(node, config, bias=bias), config.contempt
    )
    spread = np.zeros(node.width, dtype=np.float64)
    np.divide(child_variances(node, config), weights, out=spread, where=visited)
    scores = utilities - config.k_lcb * np.sqrt(spread)
    return np.asarray(np.where(visited, scores, -np.inf), dtype=np.float64)


def select_action(
    node: MatchNode,
    config: MatchConfig,
    *,
    bias: SubtreeValueBias | None = None,
    allowed: npt.NDArray[np.bool_] | None = None,
) -> int:
    """Return the index of the edge to **play** -- design §A3's rule.

    The eligible edge with the greatest ``lcb(a)``; ties on ``lcb`` broken by
    greater ``W(a)``, then by lower action id. If no edge is eligible -- a
    very low playout count, or a forced move -- the fallback is plain
    max-weight, with the same lower-action-id tie-break.

    This is the rule the search returns a move by, and it is *not* the rule it
    descends by: the descent is optimistic (:func:`select_child`), the choice
    is pessimistic, and a search that used one for the other would either
    never explore or never commit.

    `allowed` is KOM-233's anti-decisive filter, and it **narrows**, never
    widens: an edge the mask excludes cannot be returned whatever its
    statistics say. The eligibility rule is applied first and then intersected
    with the mask, so a permitted edge that clears the LCB threshold still wins
    on ``lcb`` exactly as it always did. If the intersection is empty -- every
    permitted edge is below the visit threshold, which is the normal case when
    the filter leaves a handful of moves out of a wide root -- the choice falls
    back to the greatest ``lcb`` **among the permitted edges** rather than to
    plain max-weight: max-weight over an unrestricted root is what put the
    refuted move on the board in the first place, and an edge with no weight
    scores ``-inf`` here, so the fallback still prefers a searched permitted
    edge to an unsearched one. When *no* permitted edge was searched they all
    score ``-inf`` alike and the tie is broken by the **root prior**, greatest
    first: the network's own ranking of moves it was never given the budget to
    look at is the only information in the position, and the lowest action id
    is not information at all. The lowest action id remains the last tie-break,
    under an exact tie on both.

    Args:
        node: The root whose edge is being chosen.
        config: Supplies the LCB constants and the contempt.
        bias: The tree's design §A2 table, or ``None``.
        allowed: ``(L,)`` boolean over `node`'s edges, ``True`` where the edge
            may be played. ``None`` is the legacy rule verbatim.

    Raises:
        ValueError: If `allowed` is not ``(L,)``, or excludes every edge --
            "there is no move at all" is the caller's decision to make, not a
            mask this function may quietly ignore.
    """
    _require_children(node, "select_action")
    weights = node.weight
    eligible = lcb_eligible(node, config)
    if allowed is not None:
        permitted = np.asarray(allowed, dtype=np.bool_)
        if permitted.shape != (node.width,):
            raise ValueError(
                f"select_action: allowed must have shape ({node.width},), got "
                f"{permitted.shape!r}"
            )
        if not bool(permitted.any()):
            raise ValueError(
                "select_action: allowed permits no edge at all, which is not a "
                "move choice this function can make"
            )
        eligible = np.asarray(eligible & permitted, dtype=np.bool_)
        if not bool(eligible.any()):
            scores = lcb_scores(node, config, bias=bias)
            priors = np.asarray(node.prior, dtype=np.float64)
            best = int(np.flatnonzero(permitted)[0])
            for index in range(node.width):
                if not bool(permitted[index]):
                    continue
                if (float(scores[index]), float(priors[index])) > (
                    float(scores[best]),
                    float(priors[best]),
                ):
                    best = index
            return best
    if bool(eligible.any()):
        scores = lcb_scores(node, config, bias=bias)
        best = 0
        for index in range(node.width):
            if not bool(eligible[index]):
                continue
            if not bool(eligible[best]):
                best = index
                continue
            # Lower index wins every remaining tie, and `actions` is
            # ascending, so that is the lowest action id.
            if (float(scores[index]), float(weights[index])) > (
                float(scores[best]),
                float(weights[best]),
            ):
                best = index
        return best
    return int(np.argmax(weights))
