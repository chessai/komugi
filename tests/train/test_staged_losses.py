"""The four staged loss branches: targets, masking, goldens and the zero guard.

Every head here lands at ``enabled=False, weight=0.0`` in the production
recipe, which means none of them is exercised by a single production training
step until its own gate opens. That is exactly why this module is
adversarial rather than confirmatory: an auxiliary that is mis-scaled by a
constant, scored against itself, or wired to the wrong index passes every
shape check, every finiteness check and every "the key appears in per_head"
check, and freezes the mistake into a net nobody can re-architect afterwards.

Four classes of node, in the order the failure modes matter:

* **hand-computed goldens in float64**, one per head, so a constant factor --
  ``/ 3`` instead of ``/ 81``, a sum where a mean belongs -- is red;
* **the masking primitives**, because ownership's class 3 and draft-outcome's
  all-zero rows are *the whole mask mechanism*: they buy KOM-005b out of
  needing a per-head sample-weight seam, and they are correct only because
  ``_distribution_cross_entropy`` contributes exactly zero where the target
  has none;
* **the uncertainty head's two invariants** -- the target is index 0 minus
  index 2, and it is detached;
* **the weight-0 guard (D17)**, which is red against the code as it stood
  before KOM-005b.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import pytest
import torch

from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay.records import SCHEMA
from komugi.train import losses
from komugi.train.step import StepConfig, TrainStep
from komugi.train.wiring import (
    MLH_RECIPE,
    P0B_RECIPE,
    outputs_from_net,
    targets_from_batch,
)

_SQUARES: Final[int] = 81
_CLASSES: Final[int] = 3
_UNLABELLED: Final[int] = 3
"""``final_control``'s fourth value: the square is unlabelled (resign v2)."""

_LN3: Final[float] = math.log(3.0)


def _softplus_inverse(value: float) -> float:
    """Return ``u`` with ``softplus(u) == value``, for ``value > 0``.

    ``softplus(log(exp(t) - 1)) == log(1 + exp(t) - 1) == t`` exactly in real
    arithmetic, so the goldens below can name the *activated* value -- which is
    what the loss is written in terms of -- rather than a raw logit whose
    softplus a reader would have to take on trust.
    """
    return math.log(math.exp(value) - 1.0)


def _ones(batch: int) -> torch.Tensor:
    """A float64 unit weight vector, so nothing is weighted by accident."""
    return torch.ones(batch, dtype=torch.float64)


_ONLY_UNCERTAINTY: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=False,
    soft_policy_enabled=False,
    wdl_enabled=False,
    uncertainty_enabled=True,
    uncertainty_weight=1.0,
)
_ONLY_OWNERSHIP: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=False,
    soft_policy_enabled=False,
    wdl_enabled=False,
    ownership_enabled=True,
    ownership_weight=1.0,
)
_ONLY_SHORT_TERM_VALUE: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=False,
    soft_policy_enabled=False,
    wdl_enabled=False,
    short_term_value_enabled=True,
    short_term_value_weight=1.0,
)
_ONLY_DRAFT_OUTCOME: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=False,
    soft_policy_enabled=False,
    wdl_enabled=False,
    draft_outcome_enabled=True,
    draft_outcome_weight=1.0,
)
"""One staged head on, every P0a head off -- so a loss is attributable.

Written out four times rather than built from a head name: a keyword-argument
dict would be untyped at the call, which is exactly the class of mistake
``mypy --strict`` is here to catch on a config whose fields are half bools and
half floats.
"""


# --------------------------------------------------------------------------
# Uncertainty: the index semantics and the detach
# --------------------------------------------------------------------------


def test_the_uncertainty_target_is_index_zero_minus_index_two() -> None:
    """``r = (v̂ - z)²`` with ``v̂ = p[0] - p[2]`` and ``z = t[0] - t[2]``.

    Three rows share one ``wdl_logits`` row and differ only in their one-hot
    ``wdl_target``, so the three residuals must be exactly ``(v̂ - 1)²``,
    ``v̂²`` and ``(v̂ + 1)²``. Two properties fall out and both are asserted:
    the draw row's residual is ``v̂²`` -- **no draw term anywhere**, which is
    what keeps ``draw_score`` out of a target -- and a loss-labelled row scored
    against logits that favour index 0 gives a strictly larger residual than a
    win-labelled one.

    **Necessary but not sufficient** for the wire-semantics obligation: an
    implementation that consistently writes ``v̂ = p[2] - p[0]`` *and*
    ``z = t[2] - t[0]`` produces bit-identical residuals and passes this node,
    because ``r`` is squared. This pins that ``v̂`` and ``z`` agree with **each
    other**; what index 0 means on the *wire* is pinned end to end in
    ``tests/selfplay/unit/test_staged_targets.py``. The pair discharges the
    obligation; neither does alone.
    """
    logits = torch.tensor(
        [[1.0, 0.0, -1.0]] * 3, dtype=torch.float64, requires_grad=False
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    residual = losses.uncertainty_target(logits, target)

    probabilities = torch.softmax(logits, dim=-1)
    predicted = float(probabilities[0, 0] - probabilities[0, 2])
    assert residual.shape == (3,)
    assert float(residual[0]) == pytest.approx((predicted - 1.0) ** 2, abs=1e-12)
    assert float(residual[1]) == pytest.approx(predicted**2, abs=1e-12)
    assert float(residual[2]) == pytest.approx((predicted + 1.0) ** 2, abs=1e-12)

    # Logits favour index 0, so "the mover lost" is the surprising label.
    assert float(residual[2]) > float(residual[0])

    # The draw row carries no draw-probability term: it is v̂² exactly, which
    # a `v̂ = p0 + draw_score * p1 - p2` implementation would not give.
    assert float(residual[1]) != pytest.approx(
        (predicted + 0.2 * float(probabilities[0, 1])) ** 2, abs=1e-9
    )


def test_the_uncertainty_loss_leaves_the_wdl_gradient_exactly_as_it_was() -> None:
    """The detach, asserted three ways at once.

    Without ``stop_grad`` on the target, the WDL head gains a second objective
    -- *be predictable* -- which it can serve by being confidently wrong in a
    consistent direction. So: the WDL parameters' gradients with the
    uncertainty head enabled at weight 1.0 must be ``torch.equal`` to their
    gradients with it disabled (not merely close -- a detached target
    contributes *nothing*, so the two are the same computation), while the
    uncertainty head's own parameters do receive gradient.
    """
    net = _tiny_net()
    # eval(), because this node runs three forward passes over one net and
    # compares their gradients to the bit: in train mode BatchNorm would
    # update its running statistics between them, so the second pass would
    # differ from the first for a reason that has nothing to do with the
    # detach.
    net.eval()
    observation = torch.randn(4, 4, 9, 9, generator=_generator(), dtype=torch.float32)
    targets = {
        losses.WDL_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        losses.POLICY_TARGET: torch.full((4, 81), 1.0 / 81.0),
    }

    def wdl_gradients(*, uncertainty: bool) -> list[torch.Tensor]:
        net.zero_grad(set_to_none=True)
        config = losses.LossConfig(
            policy_enabled=False,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
            uncertainty_enabled=uncertainty,
            uncertainty_weight=1.0 if uncertainty else 0.0,
        )
        total, _ = losses.LossAssembler(config)(
            outputs_from_net(net(observation)), targets
        )
        total.backward()  # type: ignore[no-untyped-call]
        return [
            parameter.grad.clone()
            for parameter in net.wdl.parameters()
            if parameter.grad is not None
        ]

    without = wdl_gradients(uncertainty=False)
    with_head = wdl_gradients(uncertainty=True)
    assert without, "the WDL head received no gradient at all"
    assert len(without) == len(with_head)
    for left, right in zip(without, with_head, strict=True):
        assert torch.equal(left, right), (
            "enabling the uncertainty head changed the WDL head's gradient: "
            "the target is not detached"
        )

    net.zero_grad(set_to_none=True)
    total, _ = losses.LossAssembler(
        losses.LossConfig(
            policy_enabled=False,
            soft_policy_enabled=False,
            wdl_enabled=False,
            uncertainty_enabled=True,
            uncertainty_weight=1.0,
        )
    )(outputs_from_net(net(observation)), targets)
    total.backward()  # type: ignore[no-untyped-call]
    moved = [
        parameter
        for parameter in net.uncertainty.parameters()
        if parameter.grad is not None
        and float(parameter.grad.abs().max().item()) > 0.0
    ]
    assert moved, "the uncertainty loss produced no gradient on its own head"


# --------------------------------------------------------------------------
# Masking: class 3, and the all-zero draft row
# --------------------------------------------------------------------------


def test_final_control_class_three_contributes_no_loss_and_no_gradient() -> None:
    """An unlabelled square costs exactly zero, and hides its own logits.

    **This synthesised row is the only place class 3 exists in P0b.** The
    worker writes 0/1/2 and nothing else until resign v2 lands, so no shard
    exercises this branch: it is specified and pinned now because the label
    rule must exist *before* the head does, and because a widened value set is
    invisible to the schema fingerprint.

    Two claims. An all-unlabelled row's ownership term is ``0.0`` **exactly**,
    not approximately -- which is the ``/ 81``-as-a-constant decision, since
    dividing by the labelled count would be ``0/0`` here. And a row with 40
    labelled squares gives the same loss whatever the other 41 squares' logits
    are, which is the claim that "unlabelled" really means "not trained on"
    rather than "trained towards class 2".
    """
    assembler = losses.LossAssembler(_ONLY_OWNERSHIP)
    weights = _ones(1)

    unlabelled = torch.full((1, _SQUARES), _UNLABELLED, dtype=torch.int64)
    logits = torch.randn(
        1, _SQUARES, _CLASSES, generator=_generator(), dtype=torch.float64
    )
    total, per_head = assembler(
        {losses.OWNERSHIP_LOGITS: logits},
        {losses.FINAL_CONTROL_TARGET: unlabelled},
        sample_weight=weights,
    )
    assert float(per_head[losses.HEAD_OWNERSHIP].item()) == 0.0
    assert float(total.item()) == 0.0

    # ... and no gradient reaches the logits of a fully unlabelled row.
    probe = logits.clone().requires_grad_(True)
    total, _ = assembler(
        {losses.OWNERSHIP_LOGITS: probe},
        {losses.FINAL_CONTROL_TARGET: unlabelled},
        sample_weight=weights,
    )
    total.backward()  # type: ignore[no-untyped-call]
    assert probe.grad is not None
    assert float(probe.grad.abs().max().item()) == 0.0

    # Half-labelled: perturbing the unlabelled squares' logits moves nothing.
    partial = torch.full((1, _SQUARES), _UNLABELLED, dtype=torch.int64)
    partial[0, :40] = torch.arange(40, dtype=torch.int64) % 3
    first, _ = assembler(
        {losses.OWNERSHIP_LOGITS: logits},
        {losses.FINAL_CONTROL_TARGET: partial},
        sample_weight=weights,
    )
    perturbed = logits.clone()
    perturbed[0, 40:] += 7.5
    second, _ = assembler(
        {losses.OWNERSHIP_LOGITS: perturbed},
        {losses.FINAL_CONTROL_TARGET: partial},
        sample_weight=weights,
    )
    assert float(first.item()) == pytest.approx(float(second.item()), abs=1e-12)


def test_a_zero_draft_outcome_target_contributes_no_gradient() -> None:
    """The mask **is** the target, so a non-draft row is inert.

    KOM-005b D7 buys out of a per-head sample-weight seam by giving non-draft
    rows an all-zero target instead. That is only correct because
    ``_distribution_cross_entropy`` contributes exactly zero where the target
    has no mass -- a guard written for ``-inf`` masked policy logits, used here
    as a general masking primitive. Perturbing a non-draft row's logits must
    therefore move neither the loss nor any gradient.
    """
    assembler = losses.LossAssembler(_ONLY_DRAFT_OUTCOME)
    target = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    logits = torch.tensor(
        [[0.5, -0.25, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    weights = _ones(2)

    baseline, _ = assembler(
        {losses.DRAFT_OUTCOME_LOGITS: logits},
        {losses.DRAFT_OUTCOME_TARGET: target},
        sample_weight=weights,
    )

    moved = logits.clone()
    moved[1] = torch.tensor([9.0, -9.0, 3.0], dtype=torch.float64)
    after, _ = assembler(
        {losses.DRAFT_OUTCOME_LOGITS: moved},
        {losses.DRAFT_OUTCOME_TARGET: target},
        sample_weight=weights,
    )
    assert float(baseline.item()) == pytest.approx(float(after.item()), abs=1e-12)

    probe = logits.clone().requires_grad_(True)
    total, _ = assembler(
        {losses.DRAFT_OUTCOME_LOGITS: probe},
        {losses.DRAFT_OUTCOME_TARGET: target},
        sample_weight=weights,
    )
    total.backward()  # type: ignore[no-untyped-call]
    assert probe.grad is not None
    assert float(probe.grad[1].abs().max().item()) == 0.0
    assert float(probe.grad[0].abs().max().item()) > 0.0


# --------------------------------------------------------------------------
# The collision the two short-term-value keys create
# --------------------------------------------------------------------------


def test_the_short_term_value_loss_is_nonzero_when_prediction_and_target_differ() -> (
    None
):
    """The head is scored against the target, not against itself.

    ``SHORT_TERM_VALUE_PRED`` and ``SHORT_TERM_VALUE_TARGET`` are the *same
    string* in two different mappings -- legal, and the honest encoding of "one
    quantity, two roles". The failure mode it opens is one wrong word:
    ``_require(outputs, SHORT_TERM_VALUE_TARGET, "targets")`` is
    ``mypy --strict``-clean, shape-clean, both tensors ``(N, 3)``, and scores
    the head against its **own prediction** for a loss that is identically
    ``0.0`` with an identically-zero gradient -- while every other node in this
    suite still passes, because the key does appear in ``per_head``, the total
    is unchanged at weight 0, and the head is mapped.

    Distinct tensors and a hand-computed MSE are what make that red.
    """
    prediction = torch.tensor([[0.5, -0.25, 0.75]], dtype=torch.float64)
    target = torch.tensor([[0.25, 0.25, -0.25]], dtype=torch.float64)
    # (0.25² + 0.5² + 1.0²) / 3 = (0.0625 + 0.25 + 1.0) / 3 = 1.3125 / 3
    expected = 1.3125 / 3.0

    total, per_head = losses.LossAssembler(_ONLY_SHORT_TERM_VALUE)(
        {losses.SHORT_TERM_VALUE_PRED: prediction},
        {losses.SHORT_TERM_VALUE_TARGET: target},
        sample_weight=_ones(1),
    )
    value = float(per_head[losses.HEAD_SHORT_TERM_VALUE].item())
    assert value == pytest.approx(expected, abs=1e-12)
    assert value > 0.0, "the head is being scored against itself"
    assert float(total.item()) == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# The goldens
# --------------------------------------------------------------------------


def test_golden_staged_losses_in_float64() -> None:
    """One hand-derived constant per new head, asserted to 1e-12.

    Without this class of node an implementation that mis-scales an auxiliary
    by a constant passes every other item in the suite and freezes the
    mis-scaling into a net that cannot be re-architected afterwards.

    **Uncertainty.** Two rows, ``wdl_logits = [0, 0, 0]`` so
    ``p = (1/3, 1/3, 1/3)`` and ``v̂ = 0`` exactly. Row 0 is a win
    (``z = 1``, ``r = 1``), row 1 a draw (``z = 0``, ``r = 0``). The raw
    outputs are chosen so ``softplus(u) = (0.5, 0.25)`` -- neither of them the
    ``softplus(0) = 0.693`` a freshly initialised head starts at, so the node
    distinguishes the head's *activation* from its *initialisation*. Loss =
    ``((0.5 - 1)² + (0.25 - 0)²) / 2 = (0.25 + 0.0625) / 2 = 5/32``.

    **Ownership.** One row, all logits zero so every class has probability
    1/3 and every labelled square costs ``ln 3``. Exactly 40 squares are
    labelled and 41 carry class 3. Loss = ``40 · ln3 / 81``. An implementation
    that divides by 3 (the class count) is off by 27x; one that divides by the
    *labelled* count gives ``ln 3`` and is off by 81/40.

    **Short-term value.** Two rows with three distinct per-horizon errors each
    and a non-uniform sample weight, so the horizon mean and the batch's
    weighted mean are both pinned. Row 0's errors are ``(1, 2, 3)`` -> mean
    ``14/3``; row 1's are ``(0, 1, 2)`` -> mean ``5/3``; at weights
    ``(1, 3)`` the weighted mean is ``(14/3 + 3·5/3) / 4 = 29/12``. Summing
    over horizons instead of averaging gives ``29/4``.

    **Draft outcome.** Two rows, all logits zero: one draft row with a one-hot
    target costing ``ln 3``, one non-draft row with an all-zero target costing
    ``0``. Loss = ``ln3 / 2``. The ``/2`` is §1.4.3's dilution factor made
    arithmetic -- ``_weighted_mean`` normalises by weight *mass*, not by the
    number of contributing rows, so the reported curve is the true draft-row
    mean scaled by the draft fraction of the batch. A "renormalise by the
    contributing rows" implementation reports ``ln 3`` and is red here.
    """
    # -- uncertainty ------------------------------------------------------
    activated = (0.5, 0.25)
    raw = torch.tensor(
        [_softplus_inverse(activated[0]), _softplus_inverse(activated[1])],
        dtype=torch.float64,
    )
    total, per_head = losses.LossAssembler(_ONLY_UNCERTAINTY)(
        {
            losses.UNCERTAINTY_PRED: raw,
            losses.WDL_LOGITS: torch.zeros(2, 3, dtype=torch.float64),
        },
        {
            losses.WDL_TARGET: torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            )
        },
        sample_weight=_ones(2),
    )
    assert float(per_head[losses.HEAD_UNCERTAINTY].item()) == pytest.approx(
        5.0 / 32.0, abs=1e-12
    )
    assert float(total.item()) == pytest.approx(5.0 / 32.0, abs=1e-12)

    # -- ownership --------------------------------------------------------
    control = torch.full((1, _SQUARES), _UNLABELLED, dtype=torch.int64)
    control[0, :40] = torch.arange(40, dtype=torch.int64) % 3
    _, per_head = losses.LossAssembler(_ONLY_OWNERSHIP)(
        {
            losses.OWNERSHIP_LOGITS: torch.zeros(
                1, _SQUARES, _CLASSES, dtype=torch.float64
            )
        },
        {losses.FINAL_CONTROL_TARGET: control},
        sample_weight=_ones(1),
    )
    assert float(per_head[losses.HEAD_OWNERSHIP].item()) == pytest.approx(
        40.0 * _LN3 / 81.0, abs=1e-12
    )

    # -- short-term value -------------------------------------------------
    prediction = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], dtype=torch.float64
    )
    _, per_head = losses.LossAssembler(_ONLY_SHORT_TERM_VALUE)(
        {losses.SHORT_TERM_VALUE_PRED: prediction},
        {losses.SHORT_TERM_VALUE_TARGET: torch.zeros(2, 3, dtype=torch.float64)},
        sample_weight=torch.tensor([1.0, 3.0], dtype=torch.float64),
    )
    assert float(per_head[losses.HEAD_SHORT_TERM_VALUE].item()) == pytest.approx(
        29.0 / 12.0, abs=1e-12
    )

    # -- draft outcome ----------------------------------------------------
    _, per_head = losses.LossAssembler(_ONLY_DRAFT_OUTCOME)(
        {losses.DRAFT_OUTCOME_LOGITS: torch.zeros(2, 3, dtype=torch.float64)},
        {
            losses.DRAFT_OUTCOME_TARGET: torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
            )
        },
        sample_weight=_ones(2),
    )
    assert float(per_head[losses.HEAD_DRAFT_OUTCOME].item()) == pytest.approx(
        _LN3 / 2.0, abs=1e-12
    )


# --------------------------------------------------------------------------
# Weight 0: measured, silent, and unable to poison the total
# --------------------------------------------------------------------------


def _staged_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Outputs and targets that satisfy all four staged branches at once."""
    outputs = {
        losses.POLICY_LOGITS: torch.zeros(2, 4, dtype=torch.float64),
        losses.WDL_LOGITS: torch.tensor(
            [[0.5, 0.0, -0.5], [0.0, 1.0, 0.0]], dtype=torch.float64
        ),
        losses.UNCERTAINTY_PRED: torch.tensor([0.25, -0.5], dtype=torch.float64),
        losses.OWNERSHIP_LOGITS: torch.zeros(
            2, _SQUARES, _CLASSES, dtype=torch.float64
        ),
        losses.SHORT_TERM_VALUE_PRED: torch.tensor(
            [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]], dtype=torch.float64
        ),
        losses.DRAFT_OUTCOME_LOGITS: torch.zeros(2, 3, dtype=torch.float64),
    }
    targets = {
        losses.POLICY_TARGET: torch.full((2, 4), 0.25, dtype=torch.float64),
        losses.WDL_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        ),
        losses.FINAL_CONTROL_TARGET: torch.zeros(
            2, _SQUARES, dtype=torch.int64
        ),
        losses.SHORT_TERM_VALUE_TARGET: torch.zeros(2, 3, dtype=torch.float64),
        losses.DRAFT_OUTCOME_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
    }
    return outputs, targets


_BASELINE: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=False,
    wdl_enabled=True,
    wdl_weight=1.0,
)
"""Policy and WDL only -- the arm every weight-0 comparison below is against."""

_AT_WEIGHT_ZERO: Final[tuple[tuple[str, losses.LossConfig], ...]] = (
    (
        losses.HEAD_UNCERTAINTY,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
            uncertainty_enabled=True,
            uncertainty_weight=0.0,
        ),
    ),
    (
        losses.HEAD_OWNERSHIP,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
            ownership_enabled=True,
            ownership_weight=0.0,
        ),
    ),
    (
        losses.HEAD_SHORT_TERM_VALUE,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
            short_term_value_enabled=True,
            short_term_value_weight=0.0,
        ),
    ),
    (
        losses.HEAD_DRAFT_OUTCOME,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
            draft_outcome_enabled=True,
            draft_outcome_weight=0.0,
        ),
    ),
)


@pytest.mark.parametrize(("key", "staged_config"), _AT_WEIGHT_ZERO)
def test_each_staged_head_at_weight_zero_is_measured_but_silent(
    key: str, staged_config: losses.LossConfig
) -> None:
    """The bring-up state: a curve on disk, no gradient in the net.

    ``enabled=True, weight=0.0`` is the first step of every staged head's
    ladder and is a mandated production configuration for one full campaign
    iteration. What it must mean is exactly two things: the head's *unweighted*
    value appears in ``per_head`` (so there is a curve to compare against the
    moment the weight goes up), and the total is unchanged to the bit.
    """
    outputs, targets = _staged_batch()
    weights = _ones(2)
    baseline, baseline_heads = losses.LossAssembler(_BASELINE)(
        outputs, targets, sample_weight=weights
    )
    total, per_head = losses.LossAssembler(staged_config)(
        outputs, targets, sample_weight=weights
    )

    assert key in per_head, "an enabled head must report its curve"
    assert set(per_head) == set(baseline_heads) | {key}
    assert float(total.item()) == pytest.approx(float(baseline.item()), abs=1e-12)


def _poisoned_batch(
    poisoned: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Outputs, targets and the leaf every output is a function of.

    One leaf so there is a single P0a-side gradient to check after
    ``backward()`` -- the stand-in for the trunk, which is what a poisoned
    total actually destroys. `poisoned` names the output key to drive
    non-finite **where the target has mass**, which is a contradiction rather
    than a mask and is therefore left to produce ``+inf`` by design.
    """
    leaf = torch.zeros((), dtype=torch.float64, requires_grad=True)

    def ramped(*shape: int) -> torch.Tensor:
        """``leaf`` spread over `shape` with a distinct factor per entry.

        A *uniform* dependence would be worse than useless: adding the same
        scalar to every logit leaves ``log_softmax`` unchanged, so the leaf's
        gradient through any cross-entropy head would be exactly zero and the
        non-vacuity clause below would fail on correct code.
        """
        count = 1
        for extent in shape:
            count *= extent
        ramp = (
            torch.arange(1, count + 1, dtype=torch.float64) / float(count)
        ).reshape(shape)
        return leaf * ramp

    outputs = {
        losses.POLICY_LOGITS: ramped(1, 4),
        losses.SOFT_POLICY_LOGITS: ramped(1, 4),
        losses.WDL_LOGITS: ramped(1, 3),
        losses.MLH_PRED: ramped(1),
        losses.OWNERSHIP_LOGITS: ramped(1, _SQUARES, _CLASSES),
    }
    hole = torch.zeros_like(outputs[poisoned])
    # A regression head goes non-finite by predicting `inf`; every
    # cross-entropy head does it with a `-inf` logit at an entry the target
    # puts mass on. Both give the head an infinite loss and neither is a mask.
    hole.reshape(-1)[0] = (
        float("inf") if poisoned == losses.MLH_PRED else float("-inf")
    )
    outputs[poisoned] = outputs[poisoned] + hole

    targets = {
        # Non-uniform, and mass on index 0 so the `-inf` there is a
        # contradiction rather than a mask. Uniform against uniform logits
        # would make `p - t` exactly zero and the live policy head's gradient
        # with it.
        losses.POLICY_TARGET: torch.tensor(
            [[0.7, 0.1, 0.1, 0.1]], dtype=torch.float64
        ),
        losses.WDL_TARGET: torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        losses.MLH_TARGET: torch.zeros(1, dtype=torch.float64),
        losses.FINAL_CONTROL_TARGET: torch.zeros(1, _SQUARES, dtype=torch.int64),
    }
    return outputs, targets, leaf


_POISONED_AT_WEIGHT_ZERO: Final[
    tuple[tuple[str, str, losses.LossConfig], ...]
] = (
    (
        losses.HEAD_POLICY,
        losses.POLICY_LOGITS,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=0.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=1.0,
        ),
    ),
    (
        losses.HEAD_SOFT_POLICY,
        losses.SOFT_POLICY_LOGITS,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=True,
            soft_policy_weight=0.0,
            wdl_enabled=False,
        ),
    ),
    (
        losses.HEAD_WDL,
        losses.WDL_LOGITS,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=True,
            wdl_weight=0.0,
        ),
    ),
    (
        losses.HEAD_MLH,
        losses.MLH_PRED,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=False,
            mlh_enabled=True,
            mlh_weight=0.0,
        ),
    ),
    (
        losses.HEAD_OWNERSHIP,
        losses.OWNERSHIP_LOGITS,
        losses.LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            soft_policy_enabled=False,
            wdl_enabled=False,
            ownership_enabled=True,
            ownership_weight=0.0,
        ),
    ),
)
"""The poisoned head, the output key that poisons it, and a config where it is
enabled at weight 0 beside one live head at weight 1.

**Four of the five are P0a branches** (impl review F4). D17 put the guard on
all eight adds for a stated reason -- "so that *which heads can poison the
total* is not a property of which ticket wrote the branch" -- and the node
below previously instantiated that claim on ``ownership`` alone, leaving the
four branches the guard was *newly* added to (they were unconditional adds
before KOM-005b) asserted by nothing. Deleting the guard from any one of them
is red here.
"""


@pytest.mark.parametrize(("key", "poisoned", "config"), _POISONED_AT_WEIGHT_ZERO)
def test_a_non_finite_staged_head_at_weight_zero_does_not_poison_the_total(
    key: str, poisoned: str, config: losses.LossConfig
) -> None:
    """D17: a weight of 0 is a **skip**, not a multiplication by zero.

    ``0.0 * inf`` is ``NaN`` and ``0.0 * NaN`` is ``NaN``. Under the
    unconditional ``total = total + weight * loss`` this branch replaced, an
    enabled-at-zero head whose loss went non-finite for **one** batch turned
    the total into ``NaN``, and ``total.backward()`` wrote ``NaN`` into every
    parameter of the net -- the trunk and all four P0a heads included. So
    "influencing nothing" was true for finite losses only, and P0b is what
    makes enabled-at-zero a mandated production configuration on a newly
    written head.

    The exposure is real per head and this node instantiates it on each of
    five: a ``-inf`` logit where the target has mass (or an ``inf`` regression
    prediction, for ``mlh``). That is not a masked-out entry but a
    contradiction -- the row says "the mover controlled this square" and the
    net says "impossible" -- which ``_distribution_cross_entropy`` deliberately
    leaves producing ``+inf`` rather than swallowing.

    **Four of the five heads here are P0a, not staged**, despite the node's
    name: the guard is on all eight branches by design, so the claim is only
    asserted if the four branches that were unconditional adds before KOM-005b
    are exercised too (impl review F4).

    Four assertions: the total stays finite, the head still *records* its
    non-finite value (the curve must not be silently repaired), the surviving
    gradient is finite after ``backward()``, and it is non-zero -- without that
    last clause a total that had quietly become a constant would pass.
    """
    outputs, targets, leaf = _poisoned_batch(poisoned)
    total, per_head = losses.LossAssembler(config)(
        outputs, targets, sample_weight=_ones(1)
    )

    assert bool(torch.isfinite(total).item()), (
        f"the weight-0 {key} head's non-finite loss reached the total: the "
        "guard is a multiplication by zero rather than a skip"
    )
    assert not bool(
        torch.isfinite(per_head[key]).item()
    ), "the head's own curve must record what actually happened"

    total.backward()  # type: ignore[no-untyped-call]
    assert leaf.grad is not None
    assert bool(torch.isfinite(leaf.grad).all().item()), (
        "backward() through a poisoned total wrote non-finite gradients into "
        "the rest of the net"
    )
    assert float(leaf.grad.abs().item()) > 0.0, (
        "the live head contributed no gradient at all, so the finiteness "
        "assertion above is vacuous"
    )


# --------------------------------------------------------------------------
# KOM-159 T2 -- the moves-left head's own weight-0 rung, against P0B_RECIPE
# --------------------------------------------------------------------------

_MLH_AT_WEIGHT_ZERO: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=8.0,
    soft_policy_exponent=0.25,
    wdl_enabled=True,
    wdl_weight=1.0,
    mlh_enabled=True,
    mlh_weight=0.0,
    uncertainty_enabled=False,
    uncertainty_weight=0.0,
    ownership_enabled=False,
    ownership_weight=0.0,
    short_term_value_enabled=False,
    short_term_value_weight=0.0,
    draft_outcome_enabled=False,
    draft_outcome_weight=0.0,
    surprise_alpha=0.0,
    surprise_max_weight=5.0,
)
""":data:`P0B_RECIPE` with the moves-left head enabled at weight 0.

The staging ladder's first rung for this head (``kom-005b-staged-heads.md``),
and deliberately **not** a constant in :mod:`komugi.train.wiring`: for MLH that
rung is a CI act rather than a campaign iteration (KOM-159 D1, the ``MLH_RECIPE``
docstring), so the configuration that instantiates it belongs to the test that
makes the claim and not to the production namespace, where it would be a recipe
no run is ever supposed to name.
"""


def _p0b_forward(
    config: losses.LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Score the full-schema batch under `config` on a fresh net, and backward.

    Returns the total, the per-head breakdown and every parameter gradient the
    backward left behind. A fresh :func:`_tiny_net` each call: ``KomugiNet`` is
    a pure function of its seeded config, so two calls start bit-identical, and
    a forward in ``train()`` mode advances BatchNorm's running estimates -- which
    is why the two arms may not share one net.

    The gradients, not the parameters, because there is no optimiser here:
    AdamW's update would fold its own epsilon into the comparison and make
    "bit-identical" a claim about the optimiser rather than about the loss.
    """
    batch = _full_schema_batch()
    net = _tiny_net()
    net.train()
    outputs = outputs_from_net(net(batch["observation"]))
    targets = targets_from_batch(batch)
    total, per_head = losses.LossAssembler(config)(outputs, targets)
    total.backward()  # type: ignore[no-untyped-call]
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in net.named_parameters()
        if parameter.grad is not None
    }
    assert gradients, "the backward left no gradient at all; nothing is pinned"
    return total.detach(), per_head, gradients


def test_the_mlh_head_at_weight_zero_is_bit_identical_to_p0b() -> None:
    """T2: the bring-up rung costs the P0b campaign nothing, to the bit.

    ``torch.equal``, not ``approx``: the whole content of the weight-0 rung is
    that a head can be *measured* for an iteration without perturbing the run
    it is measured inside, and "within 1e-9" is not that claim -- a total that
    moved by one ULP means the guard is ``total + 0.0 * loss`` (a real add of a
    real product) rather than the skip D17 specifies, and the day that product
    is ``0.0 * inf`` the run dies.

    Three assertions. The total is bit-identical; **every** parameter gradient
    is bit-identical, including the trunk's, which is what a poisoned or merely
    perturbed total would corrupt for all eight heads at once; and the head's
    own curve is nonetheless reported, because a rung that produced no number
    would be a no-op rather than a measurement.

    The moves-left head's own parameters are asserted *absent* from both
    gradient maps: at weight 0 the loss is computed and discarded, so
    ``mlh.*`` receives nothing to descend on. That is the difference between
    this rung and the next one.
    """
    baseline_total, baseline_heads, baseline_grads = _p0b_forward(P0B_RECIPE)
    total, per_head, gradients = _p0b_forward(_MLH_AT_WEIGHT_ZERO)

    assert torch.equal(total, baseline_total), (
        f"the weight-0 moves-left head moved the total from "
        f"{baseline_total.item()!r} to {total.item()!r}: the zero weight is a "
        f"multiplication rather than a skip"
    )
    assert set(gradients) == set(baseline_grads), (
        f"the two arms left gradients on different parameters: "
        f"{sorted(set(gradients) ^ set(baseline_grads))}"
    )
    for name, gradient in gradients.items():
        assert torch.equal(gradient, baseline_grads[name]), (
            f"the gradient on {name} differs between P0B_RECIPE and the same "
            f"recipe with the moves-left head enabled at weight 0"
        )
    assert not any(name.startswith("mlh.") for name in gradients), (
        "the moves-left head's own parameters received a gradient at weight 0"
    )

    assert losses.HEAD_MLH in per_head, (
        "the weight-0 rung reported no moves-left curve, so there is nothing "
        "to compare against when the weight goes up"
    )
    assert losses.HEAD_MLH not in baseline_heads
    assert math.isfinite(float(per_head[losses.HEAD_MLH].item()))


def test_a_non_finite_mlh_loss_at_weight_zero_does_not_poison_the_p0b_total() -> None:
    """T2's second half, on the production-shaped recipe rather than a fixture.

    ``_POISONED_AT_WEIGHT_ZERO`` above already instantiates the D17 guard on
    ``mlh`` -- against a two-head config built for that node. This one makes the
    same claim about the configuration campaign 6's bring-up rung actually is:
    P0b's four heads live, moves-left enabled at 0, and its prediction ``inf``
    where the target is finite. A single such batch under an unguarded
    ``total + 0.0 * loss`` turns the total into ``NaN`` and ``backward()``
    writes ``NaN`` into the trunk and all four P0a heads, which is a campaign
    lost to a head that was supposed to be inert.

    The total is compared against the *unpoisoned* P0b total by ``torch.equal``
    rather than merely asserted finite: a guard that skipped the head but
    reordered the remaining adds would be finite and wrong.
    """
    batch = _full_schema_batch()
    net = _tiny_net()
    net.train()
    outputs = outputs_from_net(net(batch["observation"]))
    targets = targets_from_batch(batch)

    poisoned = dict(outputs)
    hole = torch.zeros_like(poisoned[losses.MLH_PRED])
    hole.reshape(-1)[0] = float("inf")
    poisoned[losses.MLH_PRED] = poisoned[losses.MLH_PRED] + hole

    clean_total, _ = losses.LossAssembler(P0B_RECIPE)(outputs, targets)
    total, per_head = losses.LossAssembler(_MLH_AT_WEIGHT_ZERO)(poisoned, targets)

    assert bool(torch.isfinite(total).item()), (
        "an infinite moves-left loss at weight 0 reached the P0b total"
    )
    assert torch.equal(total.detach(), clean_total.detach()), (
        "the total moved even though it stayed finite: the weight-0 head is "
        "contributing something"
    )
    assert not bool(
        torch.isfinite(per_head[losses.HEAD_MLH]).item()
    ), "the head's own curve must record what actually happened"

    total.backward()  # type: ignore[no-untyped-call]
    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            continue
        assert bool(torch.isfinite(parameter.grad).all().item()), (
            f"backward() through the poisoned total wrote a non-finite "
            f"gradient into {name}"
        )


def test_the_mlh_recipe_is_p0b_plus_one_head_field_for_field() -> None:
    """The recipe is a new constant, and it differs in exactly two fields.

    ``wiring.py``'s standing rule is that a recipe is added, never edited: the
    curves campaigns 1-5 produced belong to :data:`P0B_RECIPE`, and a campaign-6
    change that reached into it would silently relabel them. Read off the
    dataclass fields rather than restated, so the claim survives a new head
    being appended to :class:`~komugi.train.losses.LossConfig`.
    """
    differing = {
        field.name
        for field in dataclasses.fields(P0B_RECIPE)
        if getattr(P0B_RECIPE, field.name) != getattr(MLH_RECIPE, field.name)
    }
    assert differing == {"mlh_enabled", "mlh_weight"}, (
        f"MLH_RECIPE differs from P0B_RECIPE in {sorted(differing)}; campaign "
        f"6 changes the moves-left head and nothing else"
    )
    assert MLH_RECIPE.mlh_enabled is True
    assert MLH_RECIPE.mlh_weight == 1.0
    assert P0B_RECIPE.mlh_enabled is False and P0B_RECIPE.mlh_weight == 0.0, (
        "P0B_RECIPE has been edited to turn the moves-left head on; every "
        "curve produced under that name now means something else"
    )


# --------------------------------------------------------------------------
# A pre-bump shard still trains the P0a heads
# --------------------------------------------------------------------------


def _generator() -> torch.Generator:
    """A local generator, so nothing here touches the global RNG."""
    return torch.Generator().manual_seed(20250906)


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net: big enough to wire, small enough to run."""
    return KomugiNet(
        NetConfig(
            trunk=TrunkConfig(
                in_planes=4, channels=16, blocks=1, gpool_every=3, seed=3
            ),
            heads=HeadsConfig(channels=16, num_actions=81, policy_planes=1, seed=3),
        )
    )


def _pre_bump_batch() -> dict[str, torch.Tensor]:
    """A batch carrying the pre-bump *target* names, plus the two strata.

    "Pre-bump" is about the **targets** the assembler asks for --
    ``short_term_value`` and ``draft_outcome`` are absent, which is the whole
    experiment below. The two provenance strata ``TrainStep.run`` requires
    unconditionally are present: ``search_mode`` (KOM-036) and ``termination``
    (KOM-160). Both refusals fire before any head is scored, so a batch without
    them could not reach the assembler at all and this node would be pinning
    the wrong ``KeyError``.
    """
    return {
        "observation": torch.randn(
            2, 4, 9, 9, generator=_generator(), dtype=torch.float32
        ),
        "policy_target": torch.full((2, 81), 1.0 / 81.0),
        "wdl_target": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        "mlh_target": torch.tensor([0.25, 0.5], dtype=torch.float32),
        "policy_kl": torch.zeros(2, dtype=torch.float32),
        # `search_mode` predates KOM-005b's append too -- it is one of the
        # original provenance columns -- and `TrainStep.run` requires it
        # unconditionally (KOM-036), so a pre-bump batch carries it.
        "search_mode": torch.zeros(2, dtype=torch.int64),
        # ... and `termination`, required on the same terms (KOM-160). All
        # decisive, so the cap-row multiplier is exactly ones here.
        "termination": torch.zeros(2, dtype=torch.int64),
        # ... and `ply`, which KOM-159 requires on the same terms again: with
        # `mlh_target` above it reconstructs the row's game length for the
        # moves-left cap-row weight. These rows are decisive, so that
        # multiplier is exactly ones whatever the plies say.
        "ply": torch.tensor([10, 11], dtype=torch.int64),
    }


def test_a_batch_without_the_new_columns_still_trains_under_p0b_recipe() -> None:
    """A pre-bump shard keeps training the P0a heads, and says so loudly if not.

    The schema bump skips a foreign shard rather than refusing it, so a
    replay window straddling the bump feeds the trainer rows that carry no
    ``short_term_value`` and no ``draft_outcome``. Under ``P0B_RECIPE`` -- all
    four staged heads off -- that must be a full, ordinary training step:
    ``targets_from_batch`` skips a field the batch does not carry rather than
    faking one, and no enabled head asks for it.

    The complement is the same batch with a staged head switched **on**, where
    the missing column must produce a ``KeyError`` naming the key that wanted
    it. A silently-skipped head would be the worst outcome available: a gate
    reading a curve for a head that was never trained.
    """
    net = _tiny_net()
    step = TrainStep(
        net,
        losses.LossAssembler(P0B_RECIPE),
        StepConfig(batch_size=2, lr=1e-3),
    )
    metrics = step.run(_pre_bump_batch())
    assert set(metrics) == {"total", "policy", "soft_policy", "wdl"}
    assert math.isfinite(metrics["total"])

    enabled = losses.LossConfig(
        policy_enabled=True,
        policy_weight=1.0,
        soft_policy_enabled=False,
        wdl_enabled=True,
        wdl_weight=1.0,
        short_term_value_enabled=True,
        short_term_value_weight=1.0,
    )
    hot = TrainStep(
        _tiny_net(),
        losses.LossAssembler(enabled),
        StepConfig(batch_size=2, lr=1e-3),
    )
    with pytest.raises(KeyError, match=losses.SHORT_TERM_VALUE_TARGET):
        hot.run(_pre_bump_batch())


# --------------------------------------------------------------------------
# The schema column -> assembler target path, end to end
# --------------------------------------------------------------------------


_ALL_EIGHT: Final[losses.LossConfig] = losses.LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=1.0,
    wdl_enabled=True,
    wdl_weight=1.0,
    mlh_enabled=True,
    mlh_weight=1.0,
    uncertainty_enabled=True,
    uncertainty_weight=1.0,
    ownership_enabled=True,
    ownership_weight=1.0,
    short_term_value_enabled=True,
    short_term_value_weight=1.0,
    draft_outcome_enabled=True,
    draft_outcome_weight=1.0,
)
"""Every head on at weight 1 -- the state the last staged gate leaves behind."""


def _full_schema_batch() -> dict[str, torch.Tensor]:
    """A batch of two rows carrying **every** :data:`SCHEMA` field name.

    Toy in *shape* (the tiny net above reads 4 planes and 81 actions, not the
    encoder's 128 and 37341) but exact in *names*, which is what this node is
    about: the path under test is schema column -> ``TARGET_KEYS`` ->
    ``targets_from_batch`` -> the assembler's ``_require``, and every step of
    that path is keyed by name alone.

    Built here rather than imported from ``tests/train/behavior/synthetic.py``:
    that module is reached by a sibling ``import synthetic``, which only
    resolves for tests collected from its own directory.
    """
    rows = 2
    generator = _generator()
    index = torch.arange(rows, dtype=torch.int64)
    return {
        "observation": torch.randn(
            rows, 4, 9, 9, generator=generator, dtype=torch.float32
        ),
        "policy_target": torch.full((rows, 81), 1.0 / 81.0),
        "wdl_target": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        "mlh_target": torch.tensor([0.25, 0.5], dtype=torch.float32),
        "policy_kl": torch.zeros(rows, dtype=torch.float32),
        "root_value": torch.tensor([0.3, -0.4], dtype=torch.float32),
        "search_mode": torch.zeros(rows, dtype=torch.int64),
        "to_play": index % 2,
        "ply": index,
        "game_id": torch.zeros(rows, dtype=torch.int64),
        "sims_used": torch.full((rows,), 32, dtype=torch.int64),
        "plane_map_version": torch.zeros(rows, dtype=torch.int64),
        # 0/1/2 across the board, so the ownership head's target is neither
        # all one class nor all unlabelled.
        "final_control": (
            torch.arange(rows * _SQUARES, dtype=torch.int64) % 3
        ).reshape(rows, _SQUARES),
        "action": index,
        "termination": torch.tensor([0, 1], dtype=torch.int64),
        "short_term_value": torch.tensor(
            [[0.5, -0.25, 0.125], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ),
        # One in-draft row and one masked row, so both halves of the
        # zero-target mask are exercised in the same batch.
        "draft_outcome": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        ),
    }


def test_every_schema_column_reaches_its_head_through_the_wiring() -> None:
    """The positive path: a full row in, eight head losses out.

    ``TARGET_KEYS`` gained three names with KOM-005b -- ``final_control``,
    ``short_term_value``, ``draft_outcome`` -- and **no test read them through
    the wiring** (impl review F3, mutant M3): every other staged-loss node in
    this module hands the assembler a ``targets`` dict it built by hand, and
    the one node that calls ``targets_from_batch`` feeds a pre-bump batch with
    all four staged heads off. Deleting the three names from ``TARGET_KEYS``
    was therefore green across 34 test files, and the failure -- a loud
    ``KeyError`` from ``_require`` -- would have landed on the first production
    campaign to open a gate.

    So: a batch with every schema field name, one ``TrainStep.run``, all eight
    heads enabled at weight 1. The assertion is the *metric key set*, exactly,
    because a head silently absent is the failure this whole ticket is about,
    and finiteness on every value, because a target that arrived transposed or
    mis-dtyped shows up there.

    The batch's key set is asserted against :data:`SCHEMA` itself, so a future
    column cannot be added to the wire and quietly left out of this node.
    """
    batch = _full_schema_batch()
    assert set(batch) == set(SCHEMA.names), (
        "the batch under test is no longer the whole schema; a new column has "
        "been appended without being wired to a head or excluded on purpose"
    )

    step = TrainStep(
        _tiny_net(),
        losses.LossAssembler(_ALL_EIGHT),
        StepConfig(batch_size=2, lr=1e-3),
    )
    metrics = step.run(batch)

    assert set(metrics) == {
        "total",
        losses.HEAD_POLICY,
        losses.HEAD_SOFT_POLICY,
        losses.HEAD_WDL,
        losses.HEAD_MLH,
        losses.HEAD_UNCERTAINTY,
        losses.HEAD_OWNERSHIP,
        losses.HEAD_SHORT_TERM_VALUE,
        losses.HEAD_DRAFT_OUTCOME,
    }
    for name, value in metrics.items():
        assert math.isfinite(value), f"{name} is not finite: {value}"
    assert metrics["total"] > 0.0


# --------------------------------------------------------------------------
# D17's new failure mode: every enabled head at weight 0
# --------------------------------------------------------------------------


def test_an_all_zero_weight_config_raises_rather_than_stepping_silently() -> None:
    """The contract on a config that weights nothing: a loud ``RuntimeError``.

    D17 changed what ``enabled=True, weight=0.0`` means for the *whole* config
    rather than for one head (impl review F5). Before it, ``total = total +
    0.0 * loss`` kept the autograd graph alive, so a config with every enabled
    head at weight 0 produced a step that was a legal zero-gradient no-op.
    After it, every branch is skipped, ``total`` is the ``torch.zeros(())`` it
    was initialised as, it has no ``grad_fn`` at all, and
    ``TrainStep.run``'s unconditional ``total.backward()`` raises.

    That is the intended behaviour and this node is what makes it a decision
    rather than an accident: an optimiser step that trains nothing is a
    configuration mistake, and the traceback is cheaper to diagnose than a flat
    loss curve. ``LossConfig`` deliberately still *accepts* the config -- it
    requires an enabled head, never a positive weight -- so the two halves are
    asserted together: construction succeeds, the assembled total carries no
    gradient, and the step raises.

    Not reachable from ``P0B_RECIPE``; reachable from an ablation sweep that
    zeroes the last weight it was walking down, which is exactly who will meet
    it.
    """
    all_zero = losses.LossConfig(
        policy_enabled=True,
        policy_weight=0.0,
        soft_policy_enabled=False,
        wdl_enabled=True,
        wdl_weight=0.0,
    )
    assert all_zero.policy_enabled and all_zero.policy_weight == 0.0

    outputs, targets = _staged_batch()
    total, per_head = losses.LossAssembler(all_zero)(
        outputs, targets, sample_weight=_ones(2)
    )
    # The curves are still there -- `enabled` is what reports, `weight` is what
    # contributes -- but nothing contributed, so there is no graph.
    assert set(per_head) == {losses.HEAD_POLICY, losses.HEAD_WDL}
    assert float(total.item()) == 0.0
    assert total.grad_fn is None, (
        "a config whose every enabled head carries weight 0 must produce a "
        "total with no graph; if it has one, D17's skip has become a "
        "multiplication by zero again"
    )

    step = TrainStep(
        _tiny_net(),
        losses.LossAssembler(all_zero),
        StepConfig(batch_size=2, lr=1e-3),
    )
    with pytest.raises(RuntimeError, match="does not require grad"):
        step.run(_pre_bump_batch())
