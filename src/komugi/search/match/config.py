"""The match search's configuration, and every constant of design §A.

One frozen dataclass. Every number below is *pre-registered* by the design:
the value is what ships, and the KataGo parameter name attached to it in
design §A1/§A2/§A3/§A5 is **provenance to be verified** (U-A1), not a citation
already made. No constant here may be changed because a run went badly; the
one amendment rule that exists -- for :attr:`MatchConfig.sigma_prior` -- is
written down in design §A1 *before* the measurement it consumes.

There is no ``seed``, no ``dirichlet_alpha`` and no ``root_temperature``, and
no ``draw_score`` on any node or state. Match search contains no randomness at
all (design §D1) and contempt is a property of the reader, read at every call
site rather than stored (design §S2). Adding any of the first three would be a
change of algorithm rather than of tuning -- the same posture ``GumbelConfig``
takes about its own absent knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

from komugi.search.match.state import LeafWeighter, WdlEval


@dataclass(frozen=True)
class UnitWeight:
    """The default :class:`~komugi.search.match.state.LeafWeighter`: ``1.0``.

    With the uncertainty head absent (design U-A4) there is nothing to weight
    a leaf by, and the invariant that makes the seam safe to ship unwired is
    that this class makes the search bit-identical to a weightless one:
    ``W(a) == float(N(a))`` exactly, for every edge of every node, after any
    number of playouts (design §A5, acceptance A5).
    """

    def weight(self, evaluation: WdlEval) -> float:
        """Return ``1.0``, whatever the evaluation says."""
        return 1.0


@dataclass(frozen=True)
class MatchConfig:
    """Everything one match search needs to know.

    Args:
        num_actions: Size ``A`` of the fixed action space. Every logit vector
            the search is handed must have shape ``(A,)``.
        playouts: ``P``, the per-move playout budget. One playout expands at
            most one node, so a fresh tree costs at most ``1 + P``
            evaluations -- 257 at the default (design §B1).
        contempt: What a draw is worth to the side to move, in ``[-1, 1]``.
            Read at **read** time at every call site, never stored on a node
            and never captured into a closure (design §S2). Changing it
            mid-game leaves a retained tree valid.
        tree_reuse: Whether a played move rehomes the tree onto the played
            child instead of clearing it. **Off by default**, because a
            retained subtree carries visits paid for out of an earlier move's
            budget, which ``kom-023`` §D3.2 clause 2 forbids on any rated,
            anchor or cross-engine path (design §B2).
        dynamic_cpuct: Whether ``c_puct`` is the variance-scaled quantity of
            design §A1. With it off the fixed `c_puct` below is used and the
            variance scaling is identically ``1.0``.
        c_puct: The fixed exploration constant used when `dynamic_cpuct` is
            off. Design §D2's T1 compares the two arms at this value.
        c_explore: Base exploration (KataGo ``cpuctExploration``). The rule it
            parameterises is **not** KataGo's: design §A1's numerator is
            ``sqrt(1 + W_tot(n))``, so the first descent from a fresh node is
            ``argmax_a P(a)`` rather than a tie broken by action id.
        c_log: Growth of exploration with weight (``cpuctExplorationLog``).
        c_base: The weight scale of that growth (``cpuctExplorationBase``).
        sigma_prior: Prior utility standard deviation
            (``cpuctUtilityStdevPrior``). The one constant with a
            pre-registered amendment rule (design §A1, U-A2): if a calibration
            run's median ``sigma(root)`` falls outside ``[0.20, 0.60]`` it is
            set to that median, once, and the measurement is committed.
        sigma_prior_weight: Pseudo-weight of that prior
            (``cpuctUtilityStdevPriorWeight``).
        sigma_scale: How much of the scaling factor is variance-driven
            (``cpuctUtilityStdevScale``).
        sigma_floor: Numerical floor on ``sigma(n)`` (ours), so that an
            all-identical subtree cannot drive exploration to zero.
        fpu_reduction: First-play-urgency reduction at an interior node
            (``fpuReductionMax``).
        fpu_reduction_root: The same at the root (``rootFpuReductionMax``).
        subtree_bias_factor: ``lambda``, the weight of the subtree-value-bias
            correction (``subtreeValueBiasFactor``). ``0.0`` must produce a
            search bit-identical to one with the bias code removed
            (acceptance A4), which is what makes the feature bisectable.
        bias_weight_exponent: ``beta`` in ``w_new = W_tot(n) ** beta``
            (``subtreeValueBiasWeightExponent``).
        bias_free_prop: ``rho``, the per-update decay of a table entry
            (``subtreeValueBiasFreeProp``). Value from KataGo, **rule
            differs**: design §A2's removal is generation-scaled, which
            upstream's undecayed removal does not do.
        bias_weight_floor: ``epsilon_w`` (ours). An entry whose weight is
            below it reads as empty rather than as a huge quotient.
        k_lcb: Standard deviations subtracted in the LCB (``lcbStdevs``).
        min_visit_prop_for_lcb: An edge is LCB-eligible only at this fraction
            of the best edge's weight (``minVisitPropForLCB``).
        var_floor: Floor on the per-edge standard deviation (ours), which is
            what stops catastrophic cancellation from producing a ``nan``
            variance that would propagate silently into a move choice.
        variance_mode: ``"moment"`` (default) uses the stored second-moment
            matrix; ``"plugin"`` uses the closed-form single-draw variance of
            the mean WDL and stores nothing, at the cost of measuring
            distributional spread rather than empirical disagreement across
            playouts. It is a memory escape hatch, not a free switch
            (design §S3).
        moment_dtype: ``"float64"`` (default) or ``"float32"``. Design §S3's
            single permitted response to an A17 memory overshoot.
        leaf_weighter: The design §A5 seam. Defaults to :class:`UnitWeight`;
            the uncertainty head is not shipped and no weighting arithmetic
            ships with it.
    """

    # The default is the engine's own action-space width, transcribed rather
    # than imported: design §M3's allowlist gives this package no way to name
    # `komugi.net.heads`, and acceptance A15 asserts the parity defaults
    # against a bare `MatchConfig()`, which needs every field to have one.
    num_actions: int = 37341  # net/heads.py:221
    playouts: int = 256
    contempt: float = 0.0

    tree_reuse: bool = False

    dynamic_cpuct: bool = True
    c_puct: float = 1.0
    c_explore: float = 1.0
    c_log: float = 0.45
    c_base: float = 500.0
    sigma_prior: float = 0.40
    sigma_prior_weight: float = 2.0
    sigma_scale: float = 0.85
    sigma_floor: float = 0.02
    fpu_reduction: float = 0.20
    fpu_reduction_root: float = 0.10

    subtree_bias_factor: float = 0.45
    bias_weight_exponent: float = 0.85
    bias_free_prop: float = 0.80
    bias_weight_floor: float = 1e-6

    k_lcb: float = 5.0
    min_visit_prop_for_lcb: float = 0.15
    var_floor: float = 0.05

    variance_mode: str = "moment"
    moment_dtype: str = "float64"
    leaf_weighter: LeafWeighter = UnitWeight()

    def __post_init__(self) -> None:
        """Reject a configuration the search could not honour.

        Raises:
            ValueError: If `num_actions` or `playouts` is negative; if
                `contempt` is outside ``[-1, 1]`` -- outside it a draw would be
                worth more than a win; if `sigma_prior` or `c_base` is not
                positive, each being a denominator of design §A1; or if
                `variance_mode` / `moment_dtype` names something design §S3
                does not define.
        """
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {self.num_actions}")
        if self.playouts < 0:
            raise ValueError(f"playouts must be >= 0, got {self.playouts}")
        if not -1.0 <= self.contempt <= 1.0:
            raise ValueError(
                f"contempt must lie in [-1, 1], got {self.contempt}: outside it a "
                f"draw is worth more than a win and the utility stops being an "
                f"ordering of results"
            )
        if self.sigma_prior <= 0.0:
            raise ValueError(
                f"sigma_prior must be > 0, got {self.sigma_prior}: it is the "
                f"denominator of design §A1's variance scaling"
            )
        if self.c_base <= 0.0:
            raise ValueError(f"c_base must be > 0, got {self.c_base}")
        if self.variance_mode not in ("moment", "plugin"):
            raise ValueError(
                f"variance_mode must be 'moment' or 'plugin', got "
                f"{self.variance_mode!r}"
            )
        if self.moment_dtype not in ("float64", "float32"):
            raise ValueError(
                f"moment_dtype must be 'float64' or 'float32', got "
                f"{self.moment_dtype!r}"
            )
