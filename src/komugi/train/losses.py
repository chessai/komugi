"""Multi-head loss assembly with KataGo-style policy surprise weighting.

Design §4 opens with the item this module exists for: "*Policy surprise
weighting* (KataGo: 'one of the larger improvements' between its runs) --
overweight samples where search disagreed with the prior. At branching 229 the
prior is wrong *often*; this is where that turns into learning speed."

Everything here operates on tensors. There is no net, no optimiser and no
device assumption: :class:`LossAssembler` is handed a dict of head outputs and
a dict of targets, and returns a scalar and a per-head breakdown. That keeps
it usable from a test with three hand-written numbers and from the real
trainer without a code path in between.

Heads
-----
``policy``
    Cross-entropy against a *full distribution* target -- the search's visit
    distribution, not an arg-max label. ``-sum(t * log_softmax(logits))``.
``soft_policy``
    A **second, independent** set of logits -- :data:`SOFT_POLICY_LOGITS`, the
    network's dedicated ``soft_policy_logits`` head -- against ``t **
    exponent``, renormalised (KataGo's soft policy target, exponent 0.25 by
    default). It is a separate key rather than the main policy's on purpose:
    scoring the sharpened target against ``policy_logits`` would train the
    *main* head against an auxiliary target it is not supposed to fit, while
    leaving ``SoftPolicyHead`` with no gradient at all and full forward cost
    (the wave-8 heads review, F1). Sharpening the *target's tail*
    upward makes the net keep mass on moves search looked at but did not
    choose, which is what stops the policy collapsing onto its own top move at
    high branching. Carried at its own weight because it is an auxiliary
    shaping term, not the objective.
``wdl``
    Three-way cross-entropy over (win, draw, loss) against a distribution
    target -- so a bootstrapped or smoothed value label is expressible without
    changing the loss.
``mlh``
    Moves-left regression, mean squared error against a scalar target.
    Accepts ``(batch,)`` or ``(batch, 1)`` predictions.
``uncertainty``
    Mean squared error of ``softplus(u)`` against the WDL head's own squared
    residual, ``r = stop_grad((v̂ - z) ** 2)`` with ``v̂ = p[0] - p[2]`` over
    ``softmax(wdl_logits)`` and ``z = wdl_target[0] - wdl_target[2]``. **No new
    target key**: the target is derived from two tensors already in hand, and
    it is **detached**, so the gradient reaches the uncertainty head only --
    without that, the WDL head acquires a second objective (*be predictable*)
    which it can serve by being confidently wrong in a consistent direction.
    ``v̂`` deliberately drops the draw component: a ``draw_score * p_draw`` term
    would smuggle search-time contempt into a target (design D1).
``ownership``
    Per-square 3-way cross-entropy of ``(N, 81, 3)`` logits against the
    ``final_control`` column, summed over the 81 squares and divided by 81.
    Class ``3`` in that column means *unlabelled* (reserved for resign v2): it
    is clamped to an all-zero target row, which the zero-mass guard in
    :func:`_distribution_cross_entropy` makes contribute exactly zero. The
    ``/ 81`` is a **constant, not a mask** -- an unlabelled row divides by 81
    too, and so contributes 0 rather than a ``0/0``.
``short_term_value``
    Mean squared error over three horizons against the ``short_term_value``
    column, averaged over the horizons and then over the batch.
``draft_outcome``
    The WDL head's loss verbatim, against the ``draft_outcome`` column -- which
    is ``wdl_target`` on draft rows and all zeros elsewhere. **The mask is the
    target, not a sample weight** (KOM-005b D7): a zero-mass row contributes
    exactly zero through the same guard, with no new seam and no branch.

``enabled`` and ``weight`` are separate knobs on purpose: ``enabled`` decides
whether a head is *computed and reported*, ``weight`` decides what it
contributes to the total. A head enabled at weight 0 is the staged-head case
from §2.4 -- measured every step, influencing nothing, so the moment it is
switched on there is already a curve to compare against.

**A weight of 0 is a skip, not a multiplication** (KOM-005b D17). The total
adds ``weight * loss`` only ``if weight > 0.0``; ``per_head`` records the
unweighted value unconditionally, exactly as before. ``0.0 * inf`` is ``NaN``
and ``0.0 * NaN`` is ``NaN``, so under the old unconditional add an
enabled-at-zero head whose loss went non-finite for *one* batch turned
``total`` into ``NaN`` and ``total.backward()`` wrote ``NaN`` into every
parameter of the net -- the trunk and all four P0a heads included.
"Influencing nothing" was true for finite losses only. The guard is on all
eight branches, so "which heads can poison the total" is not a property of
which ticket wrote the branch.

Per-head values in the returned dict are **unweighted**, so they are
comparable across config changes; the total is the weighted sum. Reporting the
weighted values instead would make a loss curve move when only a weight
changed, which is precisely the confusion this split avoids.

Surprise weighting
------------------
Each sample carries a weight::

    w = clip(1 + alpha * KL(search_target || prior), 1, max_weight)

where ``prior`` is ``softmax(prior_logits)`` and ``prior_logits`` were
**recorded at self-play time by the net that generated the sample** -- they
travel in the sample, not recomputed from the current net. That is the whole
point: the quantity of interest is how much search moved the prior *then*, a
fixed property of the sample, not a moving target that would shrink as the
trainer fits it and turn the weighting into a feedback loop.

``alpha = 0`` disables it, returning exactly ones -- and short-circuits before
touching ``prior_logits``, so samples without a recorded prior are legal as
long as the weighting is off. The clip's lower bound is 1, so surprise can
only ever *up*-weight; the upper bound is what stops one pathological sample
(a target concentrated where the prior put ~0 mass, KL in the tens) from
owning a batch.

Every head's loss is a weighted mean, ``sum(w * l) / sum(w)`` -- normalised by
the weight mass rather than the batch size, so turning the weighting on
changes the *mix* of a batch without changing the loss scale, and therefore
without silently changing the effective learning rate.

Masked logits
-------------
Policy logits and recorded priors may carry ``-inf`` at illegal moves, which
is how a legality mask is normally applied. Every sum over the action axis
here therefore takes its contributions only where the *target* has mass:
``0 * -inf`` is ``NaN``, and a NaN in one action of one sample would spread
through the head, the total and the gradients without ever raising. A ``-inf``
at a position the target *does* put mass on is not masking but contradiction,
and is deliberately left visible as an infinite loss (or a clipped-to-maximum
surprise weight) rather than swallowed.

The policy target is a distribution, and is checked
---------------------------------------------------
That same zero-mass guard is what let a **whole row** of zeros train silently:
the cross-entropy over an empty target is a finite ``0.0``, so a policy row
search never labelled reported a plausible loss while :func:`soft_policy_target`
divided ``0`` by ``0`` and ``backward()`` wrote NaN through the policy trunk
(review TS-01, 2026-09-12). :func:`validate_policy_target` draws the line where
it belongs: *within* a policy row, a zero is a mask; a whole policy row of
zeros, or a NaN one, or one carrying negative mass, is a broken sample and is
refused by row index before it becomes a gradient. Ownership, draft and
uncertainty targets are untouched -- their all-zero rows ARE masks.

What is deliberately **not** checked is exact normalisation; see
:func:`validate_policy_target` for the arithmetic that rules it out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as torch_functional

from komugi.engine.observation import CONTROL_SIZE

POLICY_LOGITS: Final[str] = "policy_logits"
SOFT_POLICY_LOGITS: Final[str] = "soft_policy_logits"
WDL_LOGITS: Final[str] = "wdl_logits"
MLH_PRED: Final[str] = "mlh"
UNCERTAINTY_PRED: Final[str] = "uncertainty"
OWNERSHIP_LOGITS: Final[str] = "ownership_logits"
SHORT_TERM_VALUE_PRED: Final[str] = "short_term_value"
DRAFT_OUTCOME_LOGITS: Final[str] = "draft_outcome_logits"
"""Keys the assembler reads out of the ``outputs`` mapping.

Seven of the eight are :class:`~komugi.net.model.NetOutput`'s own field names;
``MLH_PRED`` is ``"mlh"`` where the net calls the field ``moves_left``.
:func:`komugi.train.wiring.outputs_from_net` is the single place that
translates, so no caller is left matching the two vocabularies by hand. The
four staged keys are spelled *identically* to their ``NetOutput`` fields on
purpose: the translation stays a rename for exactly one legacy head.
"""

POLICY_TARGET: Final[str] = "policy_target"
WDL_TARGET: Final[str] = "wdl_target"
MLH_TARGET: Final[str] = "mlh_target"
PRIOR_LOGITS: Final[str] = "prior_logits"
FINAL_CONTROL_TARGET: Final[str] = "final_control"
SHORT_TERM_VALUE_TARGET: Final[str] = "short_term_value"
DRAFT_OUTCOME_TARGET: Final[str] = "draft_outcome"
"""Keys the assembler reads out of the ``targets`` mapping.

:data:`SHORT_TERM_VALUE_PRED` and :data:`SHORT_TERM_VALUE_TARGET` are the same
string in two different mappings, which is legal (``outputs`` and ``targets``
are separate arguments) and is the *point*: the short-term head's prediction
and target are the same quantity under two indexes, and the uncertainty head
has no target key at all because its target is derived. The collision has one
silent failure mode -- ``_require(outputs, SHORT_TERM_VALUE_TARGET,
"targets")``, one wrong word, ``mypy --strict``-clean, both tensors ``(N, 3)``
``float32``, scoring the head against **its own prediction** for a loss that is
identically ``0.0``. It is closed by a test that feeds distinct prediction and
target tensors and asserts the hand-computed MSE, not by renaming a key.
"""

HEAD_POLICY: Final[str] = "policy"
HEAD_SOFT_POLICY: Final[str] = "soft_policy"
HEAD_WDL: Final[str] = "wdl"
HEAD_MLH: Final[str] = "mlh"
HEAD_UNCERTAINTY: Final[str] = "uncertainty"
HEAD_OWNERSHIP: Final[str] = "ownership"
HEAD_SHORT_TERM_VALUE: Final[str] = "short_term_value"
HEAD_DRAFT_OUTCOME: Final[str] = "draft_outcome"
"""Keys of the per-head breakdown returned alongside the total."""

_OWNERSHIP_CLASSES: Final[int] = 3
"""Control classes the ownership head predicts: mover, opponent, empty."""

_OWNERSHIP_UNLABELLED: Final[int] = 3
"""``final_control`` value meaning *unlabelled* -- reserved for resign v2.

Clamped to class 2 and then zeroed out of the target, so an unlabelled square
contributes exactly nothing. P0b's worker never writes it (the column is
0/1/2 throughout), so the only place this value exists today is a synthesised
test row; it is implemented now because the label rule must exist before the
head lands, and because a widened value set is invisible to the schema
fingerprint.
"""

_OWNERSHIP_SQUARES: Final[int] = CONTROL_SIZE
"""Squares the per-square cross-entropy is averaged over. A constant, not a
count of labelled squares -- see the module docstring's ``ownership`` entry.

**Derived, not restated** (KOM-005b impl review F12). The divisor here, the
``final_control`` column's width and the ownership head's square axis are one
number, and the two other sites already read it from the encoder:
:data:`komugi.selfplay.records._CONTROL_SIZE` is ``CONTROL_SIZE`` and
:class:`~komugi.net.heads.OwnershipHead` reshapes the board it was handed. A
literal ``81`` here would have been the third independent copy -- the one that
survives a board-size change and silently divides a 121-square loss by 81. The
import is from :mod:`komugi.engine.observation` rather than from
:mod:`komugi.selfplay.records` because ``records`` imports
:mod:`komugi.train.shards`, so reading the constant from *there* would close a
``train -> selfplay -> train`` cycle; ``observation`` is the module ``records``
itself derives from, and :mod:`komugi.train.checkpoint` already imports it.
"""

_LOG_FLOOR: Final[float] = 1e-30
"""Floor under a target probability before taking its log.

Only ever multiplied by that same probability, so a zero entry contributes
``0 * log(1e-30) == 0`` exactly rather than ``0 * -inf == nan``. Chosen well
above float32's smallest normal (~1.2e-38) so the floored log is finite in
float32 too.
"""


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Per-head enable flags and weights, plus the surprise-weighting knobs.

    Defaults are the P0a staging from §2.4: policy and WDL on, soft policy and
    moves-left present but off, surprise weighting off (``alpha = 0``) so it is
    switched on as a deliberate, measurable act rather than inherited.

    The four staged heads default off at weight 0 too, and no new *kind* of
    validation comes with them -- deliberately, because a new kind of check is
    a new thing for a caller to be surprised by and every one of these knobs is
    the same kind of knob as ``mlh_weight``. Their bring-up path is
    ``enabled=True, weight=0.0`` first (a curve, no gradient), then a weight.

    **The contract on an all-zero-weight config, stated** (KOM-005b impl review
    F5): a config whose enabled heads *all* carry weight 0 is accepted here and
    produces a ``total`` with no ``grad_fn`` at all under D17's skip, so
    :meth:`komugi.train.step.TrainStep.run` raises ``RuntimeError`` from
    ``total.backward()`` rather than performing a silent zero-gradient step --
    intended, because an optimiser step that trains nothing is a configuration
    mistake worth a traceback, and it is pinned by a test.

    Every weight must additionally be **finite**: ``NaN < 0.0`` and ``NaN >
    0.0`` are both ``False``, so a non-finite weight would pass the
    non-negativity guard below and then be *skipped* by D17's ``weight > 0.0``
    -- a head that looks fully wired, reports a falling curve in ``per_head``,
    and receives no gradient at all.
    """

    policy_enabled: bool = True
    policy_weight: float = 1.0
    soft_policy_enabled: bool = False
    soft_policy_weight: float = 0.0
    soft_policy_exponent: float = 0.25
    wdl_enabled: bool = True
    wdl_weight: float = 1.0
    mlh_enabled: bool = False
    mlh_weight: float = 0.0
    uncertainty_enabled: bool = False
    uncertainty_weight: float = 0.0
    ownership_enabled: bool = False
    ownership_weight: float = 0.0
    short_term_value_enabled: bool = False
    short_term_value_weight: float = 0.0
    draft_outcome_enabled: bool = False
    draft_outcome_weight: float = 0.0
    surprise_alpha: float = 0.0
    surprise_max_weight: float = 5.0

    def __post_init__(self) -> None:
        """Reject configs that cannot produce a meaningful gradient."""
        if not (
            self.policy_enabled
            or self.soft_policy_enabled
            or self.wdl_enabled
            or self.mlh_enabled
            or self.uncertainty_enabled
            or self.ownership_enabled
            or self.short_term_value_enabled
            or self.draft_outcome_enabled
        ):
            raise ValueError("at least one head must be enabled")
        for name, weight in (
            ("policy_weight", self.policy_weight),
            ("soft_policy_weight", self.soft_policy_weight),
            ("wdl_weight", self.wdl_weight),
            ("mlh_weight", self.mlh_weight),
            ("uncertainty_weight", self.uncertainty_weight),
            ("ownership_weight", self.ownership_weight),
            ("short_term_value_weight", self.short_term_value_weight),
            ("draft_outcome_weight", self.draft_outcome_weight),
        ):
            # Finiteness FIRST, and it is not decoration: `NaN < 0.0` is
            # `False`, so a NaN weight passed the check below untouched, and
            # under D17 `NaN > 0.0` is `False` too -- the head would be
            # silently dropped from the total while `per_head` kept reporting
            # its curve. An `inf` weight is the mirror image: it is added, and
            # the first batch turns every gradient in the net non-finite.
            if not math.isfinite(weight):
                raise ValueError(f"{name} must be finite, got {weight}")
            if weight < 0.0:
                raise ValueError(f"{name} must be >= 0, got {weight}")
        if not 0.0 < self.soft_policy_exponent <= 1.0:
            raise ValueError(
                f"soft_policy_exponent must be in (0, 1], got {self.soft_policy_exponent}"
            )
        if self.surprise_alpha < 0.0:
            raise ValueError(
                f"surprise_alpha must be >= 0, got {self.surprise_alpha}"
            )
        if self.surprise_max_weight < 1.0:
            raise ValueError(
                "surprise_max_weight must be >= 1 (the clip's lower bound), "
                f"got {self.surprise_max_weight}"
            )


def _require(
    values: Mapping[str, torch.Tensor], key: str, role: str
) -> torch.Tensor:
    """Return ``values[key]``, raising ``KeyError`` naming what needed it."""
    if key not in values:
        raise KeyError(f"{role} is missing {key!r} (present: {sorted(values)})")
    return values[key]


def _batch_size(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> int | None:
    """Return the batch length implied by whatever tensors were supplied.

    Every head's tensor is batch-major, so any one of them answers the
    question; the first present wins. ``None`` means nothing batch-shaped was
    handed in, in which case there is no length to check a weight vector
    against and the missing-key errors below are the better complaint anyway.
    """
    for values, key in (
        (targets, POLICY_TARGET),
        (targets, WDL_TARGET),
        (targets, MLH_TARGET),
        (targets, FINAL_CONTROL_TARGET),
        (targets, SHORT_TERM_VALUE_TARGET),
        (targets, DRAFT_OUTCOME_TARGET),
        (outputs, POLICY_LOGITS),
        (outputs, SOFT_POLICY_LOGITS),
        (outputs, WDL_LOGITS),
        (outputs, MLH_PRED),
        (outputs, UNCERTAINTY_PRED),
        (outputs, OWNERSHIP_LOGITS),
        (outputs, SHORT_TERM_VALUE_PRED),
        (outputs, DRAFT_OUTCOME_LOGITS),
    ):
        tensor = values.get(key)
        if tensor is not None and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return ``sum(w * v) / sum(w)`` over the batch dimension."""
    return (values * weights).sum() / weights.sum()


def _distribution_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return per-sample ``-sum(target * log_softmax(logits))``.

    Full-distribution cross-entropy over the last axis, summing only where the
    target has mass. That guard is not cosmetic: masked policy logits carry
    ``-inf`` at illegal moves (the AlphaZero-standard encoding), ``log_softmax``
    is then ``-inf`` at exactly those entries, and ``0 * -inf`` is ``NaN`` --
    which would poison the head, the total and every gradient behind it, from
    the first batch, silently. Zero-target positions therefore contribute
    exactly zero whether their logit is ``-inf``, ``-1e30`` or ordinary.

    A ``-inf`` logit *where the target has mass* is a different animal: the
    sample says "search put probability here" and the net says "this is
    impossible". That is genuinely invalid, and it is left to produce ``+inf``
    rather than being hidden by the guard -- a loss that goes infinite is a
    bug report; a loss that quietly drops a term is not.
    """
    log_probs = torch_functional.log_softmax(logits, dim=-1)
    contributions = torch.where(
        target > 0, target * log_probs, torch.zeros_like(log_probs)
    )
    return -contributions.sum(dim=-1)


def uncertainty_target(
    wdl_logits: torch.Tensor, wdl_target: torch.Tensor
) -> torch.Tensor:
    """Return the **detached** ``(batch,)`` squared value residual in ``[0, 4]``.

    ``r = stop_grad((v̂ - z) ** 2)`` where ``v̂ = p[0] - p[2]`` over
    ``p = softmax(wdl_logits)`` and ``z = wdl_target[0] - wdl_target[2]``. No
    new column: both operands are already on the wire or already predicted.

    **``v̂`` deliberately excludes the draw probability.** ``draw_score`` is
    search-time contempt and design D1 keeps it out of every target, so a
    ``+ draw_score * p_draw`` term here would smuggle it back in. Dropping the
    draw component instead makes ``v̂`` the win-minus-loss margin, whose
    realised value ``z`` is exactly ``+1 / 0 / -1`` for the three one-hot
    ``wdl_target`` rows the self-play worker writes.

    **This is where the WDL index semantics become arithmetically
    load-bearing.** Index 0 and index 2 are subtracted from each other here;
    swap them and every residual flips sign -- invisibly, because ``r`` is
    squared. That is why the wire semantics (index 0 is *the row's mover won*)
    are pinned end to end, target through utility, and not only against this
    function.

    **The detach is the invariant.** Without it the uncertainty loss reaches
    the WDL head, which then has a second objective it can serve by being
    confidently wrong in a consistent direction.
    """
    probabilities = torch_functional.softmax(wdl_logits, dim=-1)
    predicted = probabilities[..., 0] - probabilities[..., 2]
    realised = wdl_target[..., 0] - wdl_target[..., 2]
    return ((predicted - realised) ** 2).detach()


def ownership_target(
    final_control: torch.Tensor, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return the ``(batch, 81, 3)`` one-hot control target, unlabelled zeroed.

    `final_control` is the schema's ``(81,)`` ``int64`` column, mover-relative
    already (the flip happened once per row in the self-play worker): ``0`` the
    row's mover, ``1`` the opponent, ``2`` empty, ``3`` **unlabelled**.

    An unlabelled square becomes an **all-zero row** rather than a fourth
    class, which the zero-mass guard of :func:`_distribution_cross_entropy`
    makes contribute exactly zero -- so there is no mask, no per-head sample
    weight and no new branch. The clamp before the one-hot is what keeps
    ``3`` in range; the multiply is what removes it again.

    Args:
        final_control: ``(batch, 81)`` ``int64`` column.
        dtype: Float dtype of the returned target. Defaults to torch's, and is
            passed by the assembler as the logits' own so that a ``float64``
            golden computation stays entirely in ``float64``.
    """
    labelled = final_control < _OWNERSHIP_UNLABELLED
    one_hot = torch_functional.one_hot(
        final_control.clamp(max=_OWNERSHIP_CLASSES - 1), _OWNERSHIP_CLASSES
    ).to(dtype=torch.get_default_dtype() if dtype is None else dtype)
    return one_hot * labelled.unsqueeze(-1)


POLICY_TARGET_MASS_DTYPE: Final[torch.dtype] = torch.float64
"""Accumulator dtype for a policy row's total mass.

The target is stored ``float32`` and :data:`~komugi.selfplay.records.ACTION_SIZE`
wide (``records.py``: ``FieldSpec(name='policy_target', dtype='float32',
shape=(ACTION_SIZE,))``), so summing a row in its own dtype measures the
summation as much as the row. Accumulating in ``float64`` costs one widening
pass over a tensor the forward has already touched and makes the reported mass
the row's actual mass: the residual is ``A * 2**-53``, about ``4e-12`` at this
width, which is below anything the checks below can care about.
"""


def validate_policy_target(
    target: torch.Tensor, *, legal: torch.Tensor | None = None
) -> None:
    """Refuse a policy target that is not a distribution, naming the batch row.

    The policy target is the one target in the assembler whose rows may not be
    masks. Ownership rows, draft rows and the illegal actions of a policy row
    are all *deliberately* all-zero -- the zero-mass guard in
    :func:`_distribution_cross_entropy` makes them contribute exactly zero --
    but an all-zero or NaN **policy** row is not a mask, it is a broken sample,
    and until review TS-01 (2026-09-12) it trained silently: the cross-entropy
    over an empty target is a finite ``0.0``, so the reported total stayed
    plausible while :func:`soft_policy_target`'s ``raised / raised.sum()``
    divided zero by zero, ``backward()`` wrote NaN through the policy trunk,
    and the iteration published a poisoned checkpoint and journalled a normal
    step. Three conditions, checked per row:

    * every entry finite,
    * every entry non-negative,
    * positive total mass, accumulated in :data:`POLICY_TARGET_MASS_DTYPE`.

    **Exact normalisation is deliberately NOT a fourth condition**, and the
    arithmetic is why. Storage is ``float32``: a row normalised in ``float64``
    and cast down loses at most ``2**-24`` of its mass in total (each entry
    rounds by at most that fraction *of itself*, and those errors sum to at
    most that fraction of the whole), i.e. ``6e-8`` -- independent of the width,
    and ``4e-12`` more once the row is re-accumulated in ``float64``. So the
    only tolerance this seam's dtype and width actually support is around
    ``1e-6``. Meanwhile the tree contains producers whose rows are not
    normalised **at all**: the campaign's book-generation fixture plants
    section 3.2's worked peaked restriction, ``(0.70, 0.10, 0.05) +
    (0.0115,) * 13``, whose mass is exactly ``0.9995`` -- ``5e-4`` short of one
    by construction rather than by rounding, and 500x outside any derived
    tolerance. Refusing it would be a regression, not a guard: a uniformly
    scaled row is harmless at both consumers, since
    :func:`soft_policy_target` renormalises and the hard policy cross-entropy
    is linear in the mass. What TS-01 found was *zero* and *NaN* mass, and the
    three conditions above are exactly the ones that catch it.

    Sited here rather than at the codec deliberately: shard fixtures write
    all-zero policy rows on purpose (the book audit's synthetic shards, the
    worker unit's), and a validator in ``ShardWriter`` would refuse data whose
    policy column nothing reads. This is the seam where a row becomes a
    gradient, and that is the seam where it has to be a distribution.

    The healthy path costs ONE device sync -- the reductions run on device and
    only the single ``all()`` crosses -- in the same budget the assembler's
    sample-weight check already spends. The per-row diagnosis below it runs
    only on the batch that is about to be refused.

    Args:
        target: ``(batch, actions)`` policy target. Extra leading axes are
            flattened, so the reported index is the row's position in the
            flattened batch.
        legal: Optional ``(batch, actions)`` boolean legality mask. When given,
            a row carrying *any* mass on an illegal action is refused --
            exactly zero, not a tolerance, because
            :func:`~komugi.search.gumbel._as_policy_target` writes exact
            ``0.0`` there and nothing downstream perturbs it.

    Raises:
        ValueError: Naming the first offending row and what is wrong with it.
    """
    if target.ndim == 0:
        raise ValueError("policy_target must have an action axis, got a scalar")
    rows = target.reshape(-1, target.shape[-1])
    if rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError(
            f"policy_target is empty (shape {tuple(target.shape)}): there is no "
            "distribution to train on"
        )
    finite_rows = torch.isfinite(rows).all(dim=-1)
    # A non-finite row's own sum is NaN, and NaN fails every comparison below
    # rather than failing the *right* one; zeroing it first makes such a row
    # fail on `finite_rows` alone and keeps the reductions well-defined.
    safe = torch.where(finite_rows.unsqueeze(-1), rows, torch.zeros_like(rows))
    mass = safe.sum(dim=-1, dtype=POLICY_TARGET_MASS_DTYPE)
    healthy = finite_rows & (safe.min(dim=-1).values >= 0.0) & (mass > 0.0)
    if legal is not None:
        if legal.shape != rows.shape:
            raise ValueError(
                f"legal has shape {tuple(legal.shape)}; it must match the "
                f"policy target's {tuple(rows.shape)}"
            )
        illegal_mass = torch.where(legal, torch.zeros_like(safe), safe).sum(
            dim=-1, dtype=POLICY_TARGET_MASS_DTYPE
        )
        healthy = healthy & (illegal_mass <= 0.0)
    if bool(healthy.all()):
        return

    flags: list[bool] = [bool(flag) for flag in healthy.tolist()]
    index = flags.index(False)
    row = rows[index]
    finite_entries = torch.isfinite(row)
    if not bool(finite_entries.all()):
        action = int((~finite_entries).nonzero()[0].item())
        raise ValueError(
            f"policy_target row {index} is not finite: action {action} is "
            f"{float(row[action].item())}. A non-finite target makes the policy "
            "cross-entropy and every gradient behind it NaN (review TS-01)"
        )
    smallest = float(row.min().item())
    if smallest < 0.0:
        raise ValueError(
            f"policy_target row {index} has a negative entry ({smallest}); a "
            "policy target is a distribution over the actions, not a score"
        )
    row_mass = float(row.sum(dtype=POLICY_TARGET_MASS_DTYPE).item())
    if row_mass <= 0.0:
        raise ValueError(
            f"policy_target row {index} has zero mass: an all-zero policy row "
            "is not a mask, it is a row search never labelled. It would train "
            "as a finite zero cross-entropy while the soft-policy target "
            "divided 0 by 0 (review TS-01)"
        )
    raise ValueError(
        f"policy_target row {index} puts mass on actions the legality mask "
        f"calls illegal; the row claims search visited a move that cannot be "
        f"played"
    )


def soft_policy_target(target: torch.Tensor, exponent: float) -> torch.Tensor:
    """Return ``target ** exponent``, renormalised over the last axis.

    KataGo's soft policy target. Raising to a power below 1 lifts small
    probabilities relative to large ones, so the renormalised distribution
    keeps mass on the moves search merely *considered*.
    """
    raised = target.clamp_min(0.0) ** exponent
    return raised / raised.sum(dim=-1, keepdim=True)


def policy_surprise_weight(
    policy_target: torch.Tensor,
    prior_logits: torch.Tensor,
    *,
    alpha: float,
    max_weight: float,
) -> torch.Tensor:
    """Return the per-sample surprise weight, shape ``(batch,)``.

    ``1 + alpha * KL(policy_target || softmax(prior_logits))``, clipped to
    ``[1, max_weight]``. The KL is clamped at 0 before use, so float error on a
    sample where search and prior agree exactly cannot produce a weight below
    1 -- the clip would catch it anyway, but a negative KL reaching the clip is
    the kind of thing that reads as a bug in a debugger.

    Returns exact ones at ``alpha == 0`` without reading `prior_logits`
    numerically, so an all-zero or garbage prior costs nothing while the
    weighting is off.

    Only positions where the target has mass contribute, for the same reason
    as in :func:`_distribution_cross_entropy`: a recorded prior masked with
    ``-inf`` at illegal moves makes ``log_target - log_prior`` ``+inf`` there,
    and ``0 * inf`` is ``NaN`` -- one such sample would NaN its own weight, the
    weighted mean it lands in, and therefore the whole batch's loss. A ``-inf``
    prior logit where the target *does* have mass means search chose a move the
    prior called impossible; the KL is then genuinely ``+inf`` and the clip
    turns it into ``max_weight``, which is the honest answer rather than a
    hidden one.
    """
    if alpha == 0.0:
        return torch.ones(
            policy_target.shape[0],
            dtype=policy_target.dtype,
            device=policy_target.device,
        )
    log_prior = torch_functional.log_softmax(prior_logits, dim=-1)
    log_target = torch.log(policy_target.clamp_min(_LOG_FLOOR))
    contributions = torch.where(
        policy_target > 0,
        policy_target * (log_target - log_prior),
        torch.zeros_like(log_prior),
    )
    kl = contributions.sum(dim=-1).clamp_min(0.0)
    return (1.0 + alpha * kl).clamp(min=1.0, max=max_weight)


class LossAssembler:
    """Turns head outputs and targets into a total loss and a per-head breakdown.

    Stateless apart from its frozen config, so one instance is safely shared
    across a training loop, an evaluation pass and several threads.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Bind `config` (defaults to :class:`LossConfig`'s staged defaults)."""
        self._config = config if config is not None else LossConfig()

    @property
    def config(self) -> LossConfig:
        """Return the frozen config in force."""
        return self._config

    def sample_weights(self, targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the surprise weight of every sample in the batch.

        Requires ``policy_target`` always, and ``prior_logits`` only when
        ``surprise_alpha > 0``.
        """
        policy_target = _require(targets, POLICY_TARGET, "targets")
        if self._config.surprise_alpha == 0.0:
            return torch.ones(
                policy_target.shape[0],
                dtype=policy_target.dtype,
                device=policy_target.device,
            )
        prior_logits = _require(targets, PRIOR_LOGITS, "targets")
        return policy_surprise_weight(
            policy_target,
            prior_logits,
            alpha=self._config.surprise_alpha,
            max_weight=self._config.surprise_max_weight,
        )

    def __call__(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        sample_weight: torch.Tensor | None = None,
        head_sample_weight: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total, per_head)`` for one batch.

        `sample_weight` overrides the computed surprise weights -- for a
        caller that already has them (they are worth logging) or that layers
        another per-sample weighting on top. It must be a ``(batch,)`` tensor
        of non-negative values with a finite non-zero sum, and every one of
        those words is checked: a wrong-length vector would broadcast, a
        negative one would extrapolate, an all-zero one would divide by zero.

        `head_sample_weight` weights ONE head differently from another, keyed
        by the ``HEAD_*`` constants. It is a **MULTIPLIER on the base vector,
        never a replacement for it** (KOM-036 §1.5): a head with a key uses
        ``weights * head_sample_weight[head]``, a head without one uses
        ``weights`` exactly as before -- identically to a multiplier of all
        ones, which is what makes every existing call site bit-identical and
        the current suites the regression for this seam. The replacement
        reading is the live defect it exists to avoid: on the path the trainer
        actually takes the base is resolved *inside* this method
        (:meth:`sample_weights`, or the ``sample_weight=`` a surprise-weighted
        step passed), so a caller composing its own base would have to
        reconstruct one it does not hold, and would silently drop the surprise
        weighting from exactly the heads it reweighted.

        Each supplied multiplier goes through the same shape and sign checks
        the base vector does, with the head named in the message. **A composed
        vector of zero mass is defined rather than refused**: an all-zero
        *base* is a caller bug and stays a hard error, but ``weights *
        multiplier`` summing to zero is reachable by design -- the P0b A/B's
        discard arm on a batch whose rows are all fast, probability
        ``0.75 ** batch``, which is ~1e-64 at batch 512 and ~0.32 at the batch
        of 4 a unit suite uses. That head is then **omitted from `per_head`**
        and contributes ``0.0`` to `total`: omitted rather than reported as
        ``0.0``, because a zero in a loss curve is a measurement and this is an
        absence, and :func:`~komugi.train.step.training_iteration` builds its
        journal extras from whatever keys the step returned.

        **The policy target is validated before either policy head reads it**
        (:func:`validate_policy_target`, review TS-01): a row that is not
        finite, non-negative and carrying positive mass raises ``ValueError``
        naming the row, here, where there is still no gradient to be poisoned.
        This is the one target checked that way, and the reason is that it is
        the one whose all-zero rows are not a mask -- ownership's unlabelled
        squares, draft's non-draft rows and a policy row's illegal actions are
        all meant to be zero, and they stay meant to be.

        Every returned per-head value is the *unweighted* head loss; ``total``
        is ``sum(config.<head>_weight * per_head[head])`` over the enabled
        heads. Both are scalars that carry gradient.

        Raises:
            ValueError: On a malformed `sample_weight`, a malformed
                `head_sample_weight` entry, or a ``policy_target`` that is not
                a distribution.
        """
        config = self._config
        weights = (
            self.sample_weights(targets) if sample_weight is None else sample_weight
        )
        if weights.ndim != 1:
            raise ValueError(
                f"sample_weight must be 1-D over the batch, got shape {tuple(weights.shape)}"
            )
        if weights.numel() == 0:
            raise ValueError("cannot assemble a loss over an empty batch")
        batch = _batch_size(outputs, targets)
        if batch is not None and weights.numel() != batch:
            raise ValueError(
                f"sample_weight has {weights.numel()} entries; it must carry "
                f"one per sample in the batch ({batch}). A shorter vector "
                "would broadcast against the per-sample losses and silently "
                "reweight the wrong thing."
            )
        # One device sync per batch, deliberately. A negative weight makes the
        # weighted mean an extrapolation, an all-zero one divides by zero, and
        # a NaN one poisons everything downstream -- all three are cheap here
        # and expensive to recognise from a loss curve two hours in.
        stats = torch.stack((weights.min(), weights.sum()))
        smallest = float(stats[0].item())
        mass = float(stats[1].item())
        if smallest < 0.0:
            raise ValueError(
                f"sample_weight must be non-negative, got a minimum of {smallest}"
            )
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"sample_weight must have a finite positive sum, got {mass}"
            )

        # The per-head multipliers, validated and composed once. `composed`
        # holds `weights * multiplier` for every head that was given one;
        # `omitted` holds the heads whose COMPOSED mass is zero, which is a
        # defined outcome rather than an error (see the docstring). A head
        # absent from both mappings resolves to `weights` and is therefore
        # bit-identical to the pre-seam code.
        composed: dict[str, torch.Tensor] = {}
        omitted: set[str] = set()
        if head_sample_weight:
            supplied = sorted(head_sample_weight)
            for head in supplied:
                multiplier = head_sample_weight[head]
                if multiplier.ndim != 1:
                    raise ValueError(
                        f"head_sample_weight[{head!r}] must be 1-D over the "
                        f"batch, got shape {tuple(multiplier.shape)}"
                    )
                if multiplier.numel() != weights.numel():
                    raise ValueError(
                        f"head_sample_weight[{head!r}] has "
                        f"{multiplier.numel()} entries; it must carry one per "
                        f"sample in the batch ({weights.numel()}). A shorter "
                        "vector would broadcast against the per-sample losses "
                        "and silently reweight the wrong thing."
                    )
                composed[head] = weights * multiplier
            # ONE further device sync for the whole mapping, not one per head:
            # the same budget the base vector's check above is spent under,
            # and for the same reason.
            head_stats: list[float] = torch.stack(
                [
                    statistic
                    for head in supplied
                    for statistic in (
                        head_sample_weight[head].min(),
                        composed[head].sum(),
                    )
                ]
            ).tolist()
            for index, head in enumerate(supplied):
                head_min = float(head_stats[2 * index])
                head_mass = float(head_stats[2 * index + 1])
                if head_min < 0.0:
                    raise ValueError(
                        f"head_sample_weight[{head!r}] must be non-negative, "
                        f"got a minimum of {head_min}"
                    )
                if not math.isfinite(head_mass):
                    raise ValueError(
                        f"head_sample_weight[{head!r}] composed with the base "
                        f"weights must have a finite sum, got {head_mass}"
                    )
                if head_mass <= 0.0:
                    # Reachable by design (the discard arm on an all-fast
                    # batch), so it is an omission, not a division by zero and
                    # not a reported 0.0.
                    omitted.add(head)

        # The policy target is validated ONCE, before either policy head reads
        # it: before `soft_policy_target` divides by its mass and before the
        # cross-entropy makes it a gradient (review TS-01). Only when a policy
        # head is actually going to consume it -- a recipe with both staged off
        # never touches the column, and a head omitted for zero weight mass is
        # not training on it either.
        policy_target_consumed = (
            config.policy_enabled and HEAD_POLICY not in omitted
        ) or (config.soft_policy_enabled and HEAD_SOFT_POLICY not in omitted)
        if policy_target_consumed:
            validate_policy_target(_require(targets, POLICY_TARGET, "targets"))

        per_head: dict[str, torch.Tensor] = {}
        total = torch.zeros((), dtype=weights.dtype, device=weights.device)

        # Each branch fetches what it needs. The two-line repetition is
        # deliberate: hoisting the lookups above the flags would make "which
        # tensors an enabled head requires" a fact about this function's
        # control flow rather than about each head, and mypy would have to
        # reason about it too. It also keeps the two policy branches honestly
        # separate now that they read different logits.
        #
        # Each branch also resolves its own weight vector, `composed.get(head,
        # weights)`: the base for a head with no multiplier, the composition
        # for a head with one. `head not in omitted` is the zero-mass rule --
        # the head drops out of `per_head` and out of `total` entirely.
        if config.policy_enabled and HEAD_POLICY not in omitted:
            policy_logits = _require(outputs, POLICY_LOGITS, "outputs")
            policy_target = _require(targets, POLICY_TARGET, "targets")
            loss = _weighted_mean(
                _distribution_cross_entropy(policy_logits, policy_target),
                composed.get(HEAD_POLICY, weights),
            )
            per_head[HEAD_POLICY] = loss
            # `if weight > 0` on every branch, never `total + 0.0 * loss`:
            # see the module docstring's D17 paragraph. `per_head` stays
            # unconditional, so a weight-0 head still produces its curve.
            if config.policy_weight > 0.0:
                total = total + config.policy_weight * loss

        if config.soft_policy_enabled and HEAD_SOFT_POLICY not in omitted:
            # SOFT_POLICY_LOGITS, not POLICY_LOGITS: the sharpened target
            # belongs to the net's own soft-policy head. The *target* is still
            # derived from POLICY_TARGET -- there is no separate soft target on
            # the wire (see komugi.selfplay.records.SCHEMA, "any soft-policy
            # target (the trainer derives it)"), only a separate prediction.
            soft_logits = _require(outputs, SOFT_POLICY_LOGITS, "outputs")
            policy_target = _require(targets, POLICY_TARGET, "targets")
            soft_target = soft_policy_target(
                policy_target, config.soft_policy_exponent
            )
            loss = _weighted_mean(
                _distribution_cross_entropy(soft_logits, soft_target),
                composed.get(HEAD_SOFT_POLICY, weights),
            )
            per_head[HEAD_SOFT_POLICY] = loss
            if config.soft_policy_weight > 0.0:
                total = total + config.soft_policy_weight * loss

        if config.wdl_enabled and HEAD_WDL not in omitted:
            wdl_logits = _require(outputs, WDL_LOGITS, "outputs")
            wdl_target = _require(targets, WDL_TARGET, "targets")
            loss = _weighted_mean(
                _distribution_cross_entropy(wdl_logits, wdl_target),
                composed.get(HEAD_WDL, weights),
            )
            per_head[HEAD_WDL] = loss
            if config.wdl_weight > 0.0:
                total = total + config.wdl_weight * loss

        if config.mlh_enabled and HEAD_MLH not in omitted:
            mlh_pred = _require(outputs, MLH_PRED, "outputs").reshape(-1)
            mlh_target = _require(targets, MLH_TARGET, "targets").reshape(-1)
            loss = _weighted_mean(
                (mlh_pred - mlh_target) ** 2, composed.get(HEAD_MLH, weights)
            )
            per_head[HEAD_MLH] = loss
            if config.mlh_weight > 0.0:
                total = total + config.mlh_weight * loss

        if config.uncertainty_enabled and HEAD_UNCERTAINTY not in omitted:
            # No target key: the target is derived from two tensors already
            # here, and it is DETACHED -- the gradient must reach the
            # uncertainty head only (see the module docstring).
            uncertainty_pred = _require(
                outputs, UNCERTAINTY_PRED, "outputs"
            ).reshape(-1)
            wdl_logits = _require(outputs, WDL_LOGITS, "outputs")
            wdl_target = _require(targets, WDL_TARGET, "targets")
            residual = uncertainty_target(wdl_logits, wdl_target)
            loss = _weighted_mean(
                (
                    torch_functional.softplus(uncertainty_pred) - residual
                )
                ** 2,
                composed.get(HEAD_UNCERTAINTY, weights),
            )
            per_head[HEAD_UNCERTAINTY] = loss
            if config.uncertainty_weight > 0.0:
                total = total + config.uncertainty_weight * loss

        if config.ownership_enabled and HEAD_OWNERSHIP not in omitted:
            ownership_logits = _require(outputs, OWNERSHIP_LOGITS, "outputs")
            final_control = _require(targets, FINAL_CONTROL_TARGET, "targets")
            per_square = _distribution_cross_entropy(
                ownership_logits,
                ownership_target(final_control, dtype=ownership_logits.dtype),
            )
            loss = _weighted_mean(
                per_square.sum(dim=-1) / _OWNERSHIP_SQUARES,
                composed.get(HEAD_OWNERSHIP, weights),
            )
            per_head[HEAD_OWNERSHIP] = loss
            if config.ownership_weight > 0.0:
                total = total + config.ownership_weight * loss

        if config.short_term_value_enabled and HEAD_SHORT_TERM_VALUE not in omitted:
            # SHORT_TERM_VALUE_PRED out of `outputs`, SHORT_TERM_VALUE_TARGET
            # out of `targets`: the two constants are the same string, so
            # naming the wrong mapping here scores the head against its own
            # prediction for an identically-zero loss. Spelled out rather
            # than hoisted, and pinned by a test.
            stv_pred = _require(outputs, SHORT_TERM_VALUE_PRED, "outputs")
            stv_target = _require(targets, SHORT_TERM_VALUE_TARGET, "targets")
            loss = _weighted_mean(
                ((stv_pred - stv_target) ** 2).mean(dim=-1),
                composed.get(HEAD_SHORT_TERM_VALUE, weights),
            )
            per_head[HEAD_SHORT_TERM_VALUE] = loss
            if config.short_term_value_weight > 0.0:
                total = total + config.short_term_value_weight * loss

        if config.draft_outcome_enabled and HEAD_DRAFT_OUTCOME not in omitted:
            draft_logits = _require(outputs, DRAFT_OUTCOME_LOGITS, "outputs")
            draft_target = _require(targets, DRAFT_OUTCOME_TARGET, "targets")
            loss = _weighted_mean(
                _distribution_cross_entropy(draft_logits, draft_target),
                composed.get(HEAD_DRAFT_OUTCOME, weights),
            )
            per_head[HEAD_DRAFT_OUTCOME] = loss
            if config.draft_outcome_weight > 0.0:
                total = total + config.draft_outcome_weight * loss

        return total, per_head
