"""The loss is exactly the arithmetic it claims to be.

The centrepiece is a golden loss: a two-sample batch whose logits and targets
were chosen so that every head's value is a short closed form in ``ln 2`` and
``ln 3``, computed symbolically in the comments below and asserted as literal
constants. A refactor that changes a normalisation, a sign, or which axis a
softmax runs over cannot survive it.

The surprise-weighting tests pin the four claims that matter operationally:
disagreement raises a sample's weight strictly, agreement leaves it at exactly
1, ``alpha = 0`` is a true off switch (and does not even require a recorded
prior), and the clip binds before one pathological sample can own a batch.
"""

from __future__ import annotations

import math

import pytest
import torch

from komugi.selfplay.records import num_actions
from komugi.train.losses import (
    HEAD_MLH,
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    MLH_PRED,
    MLH_TARGET,
    POLICY_LOGITS,
    POLICY_TARGET,
    PRIOR_LOGITS,
    SOFT_POLICY_LOGITS,
    WDL_LOGITS,
    WDL_TARGET,
    LossAssembler,
    LossConfig,
    policy_surprise_weight,
    soft_policy_target,
    validate_policy_target,
)
from komugi.train.wiring import MLH_RECIPE, P0B_RECIPE

_LN3 = 1.0986122886681098

# --------------------------------------------------------------------------
# The golden batch, derived by hand.
#
# Two samples over three actions.
#
# policy_logits[0] = [0, ln2, 0]  -> softmax [1/4, 1/2, 1/4]
#                                 -> log p   [-ln4, -ln2, -ln4]
# policy_logits[1] = [0, 0, 0]    -> softmax [1/3, 1/3, 1/3]
#                                 -> log p   [-ln3, -ln3, -ln3]
#
# policy_target[0] = [1/17, 16/17, 0]
#   CE0 = (1/17)*ln4 + (16/17)*ln2 = (2*ln2 + 16*ln2)/17 = 18*ln2/17
# policy_target[1] = [0, 0, 1]
#   CE1 = ln3
#   policy = (CE0 + CE1) / 2                     (all sample weights are 1)
#
# soft target, exponent 1/4: t**0.25 for sample 0 is
#   [(1/17)**0.25, (16/17)**0.25, 0] = [1, 2, 0] / 17**0.25
#   -> renormalised exactly [1/3, 2/3, 0]        (the 17**0.25 cancels)
#   softCE0 = (1/3)*ln4 + (2/3)*ln2 = 2*ln2/3 + 2*ln2/3 = 4*ln2/3
#   sample 1's target is already one-hot, so softCE1 = ln3
#   soft_policy = (4*ln2/3 + ln3) / 2
#
# wdl_logits[0] = [0, 0, ln2] -> log p [-ln4, -ln4, -ln2], target [1,0,0]
#   -> ln4
# wdl_logits[1] = [0, 0, 0]   -> log p [-ln3]*3,           target [0,1,0]
#   -> ln3
#   wdl = (ln4 + ln3) / 2
#
# mlh pred [2, 5] vs target [3, 1] -> squared errors [1, 16] -> 8.5
# --------------------------------------------------------------------------

_GOLDEN_POLICY = 0.916266416395202  # (18*ln2/17 + ln3) / 2
_GOLDEN_SOFT_POLICY = 1.011404264707352  # (4*ln2/3 + ln3) / 2
_GOLDEN_WDL = 1.242453324894000  # (ln4 + ln3) / 2
_GOLDEN_MLH = 8.5  # (1 + 16) / 2

# weights: policy 1.0, soft 0.5, wdl 1.5, mlh 0.25
#   1.0*0.916266416395202 + 0.5*1.011404264707352
# + 1.5*1.242453324894000 + 0.25*8.5
_GOLDEN_TOTAL = 5.410648536089879

_GOLDEN_CONFIG = LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=0.5,
    soft_policy_exponent=0.25,
    wdl_enabled=True,
    wdl_weight=1.5,
    mlh_enabled=True,
    mlh_weight=0.25,
    surprise_alpha=0.0,
)


def _golden_batch(
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the (outputs, targets) of the hand-derived batch above.

    The soft-policy head is fed *the same logits* as the main policy head --
    the two are independent tensors in a real net (KOM-030 rewired the soft
    term onto its own :data:`SOFT_POLICY_LOGITS` key), but equal ones here keep
    the hand-derived ``softCE`` above a statement about the sharpened
    *target*, which is the arithmetic these goldens exist to pin. That the two
    keys are distinct, and that the soft term reads the soft one, is pinned in
    ``tests/train/test_netoutput_wiring.py`` against a real forward pass.
    """
    ln2 = math.log(2.0)
    policy_logits = torch.tensor([[0.0, ln2, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)
    outputs = {
        POLICY_LOGITS: policy_logits,
        SOFT_POLICY_LOGITS: policy_logits.clone(),
        WDL_LOGITS: torch.tensor([[0.0, 0.0, ln2], [0.0, 0.0, 0.0]], dtype=dtype),
        MLH_PRED: torch.tensor([2.0, 5.0], dtype=dtype),
    }
    targets = {
        POLICY_TARGET: torch.tensor(
            [[1.0 / 17.0, 16.0 / 17.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
        ),
        WDL_TARGET: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype),
        MLH_TARGET: torch.tensor([3.0, 1.0], dtype=dtype),
    }
    return outputs, targets


def test_golden_loss_in_float64() -> None:
    """Every head and the total match the hand-computed constants.

    Asserted in float64 to 1e-9 -- three orders tighter than the six decimals
    the ticket asks for, which is affordable because the closed forms above
    are exact and float64 carries ~15 digits.
    """
    outputs, targets = _golden_batch(torch.float64)
    total, per_head = LossAssembler(_GOLDEN_CONFIG)(outputs, targets)

    assert set(per_head) == {HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL, HEAD_MLH}
    assert per_head[HEAD_POLICY].item() == pytest.approx(_GOLDEN_POLICY, abs=1e-9)
    assert per_head[HEAD_SOFT_POLICY].item() == pytest.approx(
        _GOLDEN_SOFT_POLICY, abs=1e-9
    )
    assert per_head[HEAD_WDL].item() == pytest.approx(_GOLDEN_WDL, abs=1e-9)
    assert per_head[HEAD_MLH].item() == pytest.approx(_GOLDEN_MLH, abs=1e-9)
    assert total.item() == pytest.approx(_GOLDEN_TOTAL, abs=1e-9)


def test_golden_loss_in_float32() -> None:
    """The same batch in the trainer's working precision, to six decimals.

    Loosened to 1e-5 rather than 1e-6 only because float32 carries ~7
    significant digits and the total is ~5.4: 1e-6 there would be asserting
    on the last representable bit, which tests the FPU, not the loss.
    """
    outputs, targets = _golden_batch(torch.float32)
    total, per_head = LossAssembler(_GOLDEN_CONFIG)(outputs, targets)

    assert per_head[HEAD_POLICY].item() == pytest.approx(_GOLDEN_POLICY, abs=1e-5)
    assert per_head[HEAD_SOFT_POLICY].item() == pytest.approx(
        _GOLDEN_SOFT_POLICY, abs=1e-5
    )
    assert per_head[HEAD_WDL].item() == pytest.approx(_GOLDEN_WDL, abs=1e-5)
    assert per_head[HEAD_MLH].item() == pytest.approx(_GOLDEN_MLH, abs=1e-5)
    assert total.item() == pytest.approx(_GOLDEN_TOTAL, abs=1e-5)


def test_per_head_values_are_unweighted() -> None:
    """Halving a head's weight moves the total and leaves the head's value alone."""
    outputs, targets = _golden_batch(torch.float64)
    heavier = LossConfig(
        policy_enabled=True,
        policy_weight=1.0,
        wdl_enabled=True,
        wdl_weight=1.5,
    )
    lighter = LossConfig(
        policy_enabled=True,
        policy_weight=1.0,
        wdl_enabled=True,
        wdl_weight=0.5,
    )
    heavy_total, heavy_heads = LossAssembler(heavier)(outputs, targets)
    light_total, light_heads = LossAssembler(lighter)(outputs, targets)

    assert heavy_heads[HEAD_WDL].item() == pytest.approx(light_heads[HEAD_WDL].item())
    assert heavy_total.item() - light_total.item() == pytest.approx(
        1.0 * _GOLDEN_WDL, abs=1e-9
    )


def test_a_head_enabled_at_weight_zero_is_measured_but_silent() -> None:
    """The staged-head case: reported every step, contributing nothing."""
    outputs, targets = _golden_batch(torch.float64)
    config = LossConfig(
        policy_enabled=True,
        policy_weight=1.0,
        mlh_enabled=True,
        mlh_weight=0.0,
        wdl_enabled=False,
    )
    total, per_head = LossAssembler(config)(outputs, targets)
    assert per_head[HEAD_MLH].item() == pytest.approx(_GOLDEN_MLH, abs=1e-9)
    assert total.item() == pytest.approx(_GOLDEN_POLICY, abs=1e-9)


# --------------------------------------------------------------------------
# KOM-159 T1 -- what MLH_RECIPE adds to the production recipe, exactly
# --------------------------------------------------------------------------


def test_the_mlh_recipe_reports_the_mean_squared_error_and_adds_it_once() -> None:
    """T1: ``per_head['mlh']`` is the plain MSE, and the total gains ``1.0 x`` it.

    Two claims, and they fail differently.

    The **arithmetic** claim is that the moves-left head is what campaign 6
    thinks it is: an unweighted mean of ``(pred - target) ** 2`` over the batch,
    on the recorded ``MLH_SCALE`` scale, with no Huber clamp, no log and no
    normalisation by anything but the sample count. It is computed here in
    float64 from the golden batch's own numbers rather than read out of
    :mod:`komugi.train.losses`, so it is a golden and not a restatement -- and
    it is asserted to 1e-12, three orders inside what float64 carries at this
    magnitude, because the closed form ``(1 + 16) / 2`` is exact.

    The **composition** claim is that switching the recipe adds exactly that
    number, once, at weight 1.0, and changes nothing else. Asserted as a
    difference of two totals rather than against a constant: the P0b total is
    whatever P0b's four heads make it, and what KOM-159 D1 promises is that
    ``MLH_RECIPE`` is that plus ``w_mlh * loss_mlh`` and not, say, a recipe
    whose soft-policy weight moved while nobody was looking. The three shared
    heads are asserted equal for the same reason.
    """
    outputs, targets = _golden_batch(torch.float64)

    expected = float(
        (
            (outputs[MLH_PRED] - targets[MLH_TARGET]) ** 2
        ).mean()
    )
    assert expected == pytest.approx(_GOLDEN_MLH, abs=1e-12), (
        "the hand-computed MSE is not the golden batch's, so this node is "
        "measuring a batch it does not describe"
    )

    baseline, baseline_heads = LossAssembler(P0B_RECIPE)(outputs, targets)
    total, per_head = LossAssembler(MLH_RECIPE)(outputs, targets)

    assert HEAD_MLH not in baseline_heads, "P0B_RECIPE has the head staged off"
    assert per_head[HEAD_MLH].item() == pytest.approx(expected, abs=1e-12)
    assert set(per_head) == set(baseline_heads) | {HEAD_MLH}, (
        "MLH_RECIPE enables a head beyond the moves-left one, or disables one "
        "P0B_RECIPE had on: it is P0b plus this head and nothing else"
    )
    for head in baseline_heads:
        assert per_head[head].item() == pytest.approx(
            baseline_heads[head].item(), abs=1e-12
        ), f"the {head} head's value moved between the two recipes"

    assert MLH_RECIPE.mlh_weight == 1.0, (
        "the pre-registered bring-up weight is 1.0 (KOM-159 D1); this node's "
        "arithmetic below assumes it"
    )
    assert total.item() == pytest.approx(
        baseline.item() + MLH_RECIPE.mlh_weight * expected, abs=1e-12
    )
    assert total.item() > baseline.item(), (
        "the moves-left head contributed nothing to the total, so the "
        "difference above is vacuous and the recipe is P0b wearing a new name"
    )


def test_disabled_heads_are_absent_and_their_tensors_unneeded() -> None:
    """A disabled head is not computed, so its outputs need not be supplied."""
    outputs, targets = _golden_batch(torch.float64)
    del outputs[WDL_LOGITS]
    del outputs[MLH_PRED]
    config = LossConfig(policy_enabled=True, wdl_enabled=False, mlh_enabled=False)
    total, per_head = LossAssembler(config)(outputs, targets)
    assert set(per_head) == {HEAD_POLICY}
    assert total.item() == pytest.approx(_GOLDEN_POLICY, abs=1e-9)


def test_missing_tensors_for_an_enabled_head_are_named() -> None:
    """The error says which key was missing, not which line dereferenced it."""
    outputs, targets = _golden_batch(torch.float64)
    del outputs[WDL_LOGITS]
    with pytest.raises(KeyError, match=WDL_LOGITS):
        LossAssembler(LossConfig(wdl_enabled=True))(outputs, targets)


def test_explicit_sample_weights_are_a_weighted_mean() -> None:
    """Normalisation is by weight mass, not batch size.

    With weights [1, 3] the policy head must be
    ``(1*18*ln2/17 + 3*ln3) / 4 = 1.007439352531656``.
    """
    outputs, targets = _golden_batch(torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    config = LossConfig(policy_enabled=True, wdl_enabled=False)
    _, per_head = LossAssembler(config)(outputs, targets, sample_weight=weights)
    assert per_head[HEAD_POLICY].item() == pytest.approx(1.007439352531656, abs=1e-9)


def test_uniform_weights_leave_the_loss_scale_alone() -> None:
    """Doubling every weight changes nothing -- the mean is weight-normalised."""
    outputs, targets = _golden_batch(torch.float64)
    config = LossConfig(policy_enabled=True, wdl_enabled=True)
    plain, _ = LossAssembler(config)(outputs, targets)
    doubled, _ = LossAssembler(config)(
        outputs, targets, sample_weight=torch.full((2,), 2.0, dtype=torch.float64)
    )
    assert plain.item() == pytest.approx(doubled.item(), abs=1e-12)


def test_soft_policy_target_lifts_the_tail_and_stays_normalised() -> None:
    """t**0.25 renormalised: mass moves toward the moves search merely saw."""
    target = torch.tensor([[0.9, 0.09, 0.01]], dtype=torch.float64)
    soft = soft_policy_target(target, 0.25)
    assert soft.sum().item() == pytest.approx(1.0, abs=1e-12)
    assert soft[0, 0].item() < target[0, 0].item()
    assert soft[0, 2].item() > target[0, 2].item()
    # Exact case: [1/17, 16/17, 0] ** 0.25 renormalises to [1/3, 2/3, 0].
    exact = soft_policy_target(
        torch.tensor([[1.0 / 17.0, 16.0 / 17.0, 0.0]], dtype=torch.float64), 0.25
    )
    assert exact[0].tolist() == pytest.approx([1 / 3, 2 / 3, 0.0], abs=1e-12)


def test_surprise_weight_is_one_plus_alpha_times_kl() -> None:
    """The golden weight: KL([0,0,1] || uniform) = ln3, so w = 1 + alpha*ln3."""
    target = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    prior = torch.zeros((1, 3), dtype=torch.float64)
    weight = policy_surprise_weight(target, prior, alpha=0.5, max_weight=5.0)
    assert weight.shape == (1,)
    assert weight.item() == pytest.approx(1.0 + 0.5 * _LN3, abs=1e-12)
    assert weight.item() == pytest.approx(1.5493061443340549, abs=1e-12)


def test_disagreement_weighs_strictly_more_than_agreement() -> None:
    """A sample whose prior already knew the answer carries weight exactly 1."""
    target = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    # Sample 0's prior is effectively the target; sample 1's is uniform.
    prior = torch.tensor(
        [[-50.0, -50.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    weights = policy_surprise_weight(target, prior, alpha=1.0, max_weight=10.0)
    assert weights[0].item() == pytest.approx(1.0, abs=1e-9)
    assert weights[1].item() == pytest.approx(1.0 + _LN3, abs=1e-12)
    assert weights[1].item() > weights[0].item()


def test_alpha_zero_disables_the_weighting_entirely() -> None:
    """Exactly ones, and no recorded prior is required to get them."""
    outputs, targets = _golden_batch(torch.float64)
    assert PRIOR_LOGITS not in targets
    assembler = LossAssembler(LossConfig(surprise_alpha=0.0))
    weights = assembler.sample_weights(targets)
    assert torch.equal(weights, torch.ones(2, dtype=torch.float64))

    # Directly, too: a garbage prior costs nothing while the switch is off.
    off = policy_surprise_weight(
        targets[POLICY_TARGET],
        torch.full((2, 3), float("nan"), dtype=torch.float64),
        alpha=0.0,
        max_weight=5.0,
    )
    assert torch.equal(off, torch.ones(2, dtype=torch.float64))


def test_the_clip_binds_at_both_ends() -> None:
    """No sample drops below 1, and no sample runs away above max_weight."""
    target = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    prior = torch.zeros((1, 3), dtype=torch.float64)
    clipped = policy_surprise_weight(target, prior, alpha=100.0, max_weight=4.0)
    assert clipped.item() == pytest.approx(4.0, abs=1e-12)

    agreeing = policy_surprise_weight(
        target,
        torch.tensor([[-80.0, -80.0, 0.0]], dtype=torch.float64),
        alpha=100.0,
        max_weight=4.0,
    )
    assert agreeing.item() >= 1.0
    assert agreeing.item() == pytest.approx(1.0, abs=1e-9)


def test_surprise_weighting_actually_moves_the_loss() -> None:
    """Turning alpha up re-weights the batch toward the surprising sample.

    Sample 0's recorded prior is its own logits, so search barely moved it;
    sample 1's prior was uniform against a one-hot target, so search moved it
    a lot (KL = ln3). Sample 1 also carries the *larger* cross-entropy
    (ln3 = 1.0986 against 18*ln2/17 = 0.7339), so up-weighting it must push
    the policy loss **up**. A sign error in the KL would up-weight sample 0
    instead and move the loss the other way, which is what this pins.
    """
    outputs, targets = _golden_batch(torch.float64)
    targets[PRIOR_LOGITS] = torch.tensor(
        [[0.0, math.log(2.0), 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    base = LossConfig(policy_enabled=True, wdl_enabled=False, surprise_alpha=0.0)
    hot = LossConfig(policy_enabled=True, wdl_enabled=False, surprise_alpha=2.0)

    off, _ = LossAssembler(base)(outputs, targets)
    on, _ = LossAssembler(hot)(outputs, targets)
    weights = LossAssembler(hot).sample_weights(targets)

    assert weights[1].item() > weights[0].item()
    assert on.item() > off.item()
    assert off.item() == pytest.approx(_GOLDEN_POLICY, abs=1e-9)


def test_a_prior_is_required_once_the_weighting_is_on() -> None:
    """alpha > 0 without a recorded prior is an error, not a silent 1."""
    _, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(surprise_alpha=1.0))
    with pytest.raises(KeyError, match=PRIOR_LOGITS):
        assembler.sample_weights(targets)


def test_zero_probability_entries_do_not_poison_the_loss() -> None:
    """Targets with exact zeros contribute exactly zero, never NaN."""
    outputs = {
        POLICY_LOGITS: torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
    }
    targets = {
        POLICY_TARGET: torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        PRIOR_LOGITS: torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
    }
    config = LossConfig(policy_enabled=True, wdl_enabled=False, surprise_alpha=1.0)
    total, per_head = LossAssembler(config)(outputs, targets)
    assert math.isfinite(total.item())
    assert per_head[HEAD_POLICY].item() == pytest.approx(_LN3, abs=1e-12)


# --------------------------------------------------------------------------
# validate_policy_target (review TS-01, 2026-09-12)
# --------------------------------------------------------------------------


def test_a_healthy_policy_target_passes_validation() -> None:
    """The positive control: ordinary rows, including exact zeros at illegals."""
    validate_policy_target(
        torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25]], dtype=torch.float64)
    )
    wire = torch.rand(4, 81, dtype=torch.float32) + 0.05
    validate_policy_target(wire / wire.sum(dim=-1, keepdim=True))


def test_a_real_float32_policy_target_is_accepted() -> None:
    """A REAL row -- float32, :data:`~komugi.selfplay.records.ACTION_SIZE` wide.

    The regression for the defect that blocked the first attempt at this
    ticket. That validator carried a fourth condition, ``|mass - 1| <= 1e-4``,
    and a real two-iteration campaign died at iteration 0 on
    ``policy_target row 0 sums to 0.9994999766349792``.

    Two rows here, because the two halves of that failure are different facts:

    * **Row 0** is the shape :func:`~komugi.search.gumbel._as_policy_target`
      actually produces -- every legal entry floored at the smallest float32
      normal, renormalised in float64, cast to float32 -- at the real width. Its
      float64 mass is one to within ``2**-24``. Summing it in *float32* is what
      measures the summation instead of the row, which is why
      :data:`~komugi.train.losses.POLICY_TARGET_MASS_DTYPE` is float64.
    * **Row 1** is the campaign book-generation fixture's own planted target
      (``tests/campaign/behavior/test_book_generation.py``'s ``_SHAPE``,
      section 3.2's worked peaked restriction). ``0.70 + 0.10 + 0.05 +
      13 * 0.0115`` is exactly ``0.9995``, so that row is 5e-4 short of
      normalised *by construction* rather than by rounding -- which is the
      actual source of the blocking failure, and the reason exact normalisation
      is not a condition at all. It must be accepted.
    """
    # `num_actions()` is `records.ACTION_SIZE` through the accessor the rest of
    # the suite uses; the constant itself is an implicit re-export there.
    width = num_actions()
    tiny = float(torch.finfo(torch.float32).tiny)

    peaked = torch.full((width,), tiny, dtype=torch.float64)
    visits = torch.tensor(
        [512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0],
        dtype=torch.float64,
    )
    peaked[: visits.numel()] = visits
    produced = (peaked / peaked.sum()).to(torch.float32)

    planted = torch.zeros(width, dtype=torch.float32)
    for slot, mass in enumerate((0.70, 0.10, 0.05) + (0.0115,) * 13):
        planted[slot] = mass
    assert float(planted.sum(dtype=torch.float64).item()) == pytest.approx(
        0.9995, abs=1e-7
    ), "the fixture's planted shape is 0.9995 by construction, not by rounding"

    target = torch.stack([produced, planted])
    assert target.dtype is torch.float32
    assert target.shape == (2, width)
    validate_policy_target(target)


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ([0.0, 0.0, 0.0], "zero mass"),
        ([float("nan"), 0.5, 0.5], "not finite"),
        ([float("inf"), 0.5, 0.5], "not finite"),
        ([-0.25, 0.75, 0.5], "negative entry"),
    ],
)
def test_every_way_a_policy_row_stops_being_a_distribution_is_refused(
    row: list[float], reason: str
) -> None:
    """Three conditions, each named, each pointing at the offending row index."""
    target = torch.tensor(
        [[0.5, 0.25, 0.25], row, [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="row 1") as excinfo:
        validate_policy_target(target)
    assert reason in str(excinfo.value)


def test_an_unnormalised_but_positive_row_is_accepted() -> None:
    """Scale is not a condition, and the docstring's arithmetic is why.

    :func:`soft_policy_target` renormalises and the hard policy cross-entropy
    is linear in the mass, so a uniformly scaled row trains as itself scaled --
    while in-tree producers hand the assembler rows at ``0.9995``. What TS-01
    found was zero and NaN mass, and those are refused above.
    """
    validate_policy_target(
        torch.tensor([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]], dtype=torch.float64)
    )


def test_mass_on_an_illegal_action_is_refused_when_a_mask_is_given() -> None:
    """The optional arm: a row claiming search visited an unplayable move."""
    target = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64)
    legal = torch.tensor([[True, True, False], [True, True, False]])
    with pytest.raises(ValueError, match="row 1"):
        validate_policy_target(target, legal=legal)
    # Without the mask the same target is a perfectly good distribution.
    validate_policy_target(target)


def test_the_assembler_refuses_a_zero_row_before_the_soft_target_divides_by_it() -> (
    None
):
    """The siting: the guard fires in ``__call__``, not in the cross-entropy.

    Before TS-01 this batch produced a finite total -- the policy CE over an
    empty target is ``0.0`` and ``soft_policy_target`` returned NaN from
    ``0 / 0`` -- so the assertion is that it now raises at all, and that it
    names the row rather than the head.
    """
    outputs = {
        POLICY_LOGITS: torch.zeros((2, 3), dtype=torch.float64),
        SOFT_POLICY_LOGITS: torch.zeros((2, 3), dtype=torch.float64),
    }
    targets = {
        POLICY_TARGET: torch.tensor(
            [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
    }
    config = LossConfig(
        policy_enabled=True,
        soft_policy_enabled=True,
        soft_policy_weight=1.0,
        wdl_enabled=False,
    )
    with pytest.raises(ValueError, match="policy_target row 1 has zero mass"):
        LossAssembler(config)(outputs, targets)


def test_a_recipe_with_both_policy_heads_off_never_reads_the_column() -> None:
    """The guard is sited at the consumer, so a head that is off validates nothing.

    Which is the same rule that keeps the all-zero ownership and draft rows
    legal: an unread target is not a target.
    """
    outputs = {WDL_LOGITS: torch.zeros((2, 3), dtype=torch.float64)}
    targets = {
        POLICY_TARGET: torch.zeros((2, 3), dtype=torch.float64),
        WDL_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        ),
    }
    config = LossConfig(policy_enabled=False, wdl_enabled=True)
    total, per_head = LossAssembler(config)(outputs, targets)
    assert math.isfinite(total.item())
    assert set(per_head) == {HEAD_WDL}


def test_the_loss_carries_gradient() -> None:
    """Total is differentiable with respect to the logits it was built from."""
    logits = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    targets = {
        POLICY_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        ),
    }
    config = LossConfig(policy_enabled=True, wdl_enabled=False)
    total, _ = LossAssembler(config)({POLICY_LOGITS: logits}, targets)
    total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


def test_config_rejects_incoherent_settings() -> None:
    """Frozen config, validated at construction -- a typo is a TypeError."""
    with pytest.raises(ValueError, match="at least one head"):
        LossConfig(
            policy_enabled=False,
            soft_policy_enabled=False,
            wdl_enabled=False,
            mlh_enabled=False,
        )
    with pytest.raises(ValueError, match="policy_weight"):
        LossConfig(policy_weight=-1.0)
    with pytest.raises(ValueError, match="soft_policy_exponent"):
        LossConfig(soft_policy_exponent=0.0)
    with pytest.raises(ValueError, match="soft_policy_exponent"):
        LossConfig(soft_policy_exponent=1.5)
    with pytest.raises(ValueError, match="surprise_alpha"):
        LossConfig(surprise_alpha=-0.1)
    with pytest.raises(ValueError, match="surprise_max_weight"):
        LossConfig(surprise_max_weight=0.5)
    with pytest.raises(TypeError):
        LossConfig(policy_wieght=1.0)  # type: ignore[call-arg]


def test_config_rejects_negative_staged_weights() -> None:
    """The four staged weights join the same two guards, and nothing new.

    KOM-005b deliberately introduces no new *kind* of validation: each staged
    knob is the same kind of knob as ``mlh_weight``, so it joins the
    non-negativity loop and its ``enabled`` flag joins the "at least one head"
    disjunction. Both halves are asserted, because the second is the one an
    implementer forgets -- a staged head left out of the disjunction makes a
    config with only staged heads on raise, which is the opposite of what the
    ablation switches are for.
    """
    with pytest.raises(ValueError, match="uncertainty_weight"):
        LossConfig(uncertainty_weight=-1.0)
    with pytest.raises(ValueError, match="ownership_weight"):
        LossConfig(ownership_weight=-1.0)
    with pytest.raises(ValueError, match="short_term_value_weight"):
        LossConfig(short_term_value_weight=-1.0)
    with pytest.raises(ValueError, match="draft_outcome_weight"):
        LossConfig(draft_outcome_weight=-1.0)

    with pytest.raises(ValueError, match="at least one head"):
        LossConfig(
            policy_enabled=False,
            soft_policy_enabled=False,
            wdl_enabled=False,
            mlh_enabled=False,
            uncertainty_enabled=False,
            ownership_enabled=False,
            short_term_value_enabled=False,
            draft_outcome_enabled=False,
        )

    # ... and a staged head alone is a legal configuration: that is the whole
    # point of the per-head ablation switches.
    assert LossConfig(
        policy_enabled=False,
        soft_policy_enabled=False,
        wdl_enabled=False,
        uncertainty_enabled=True,
    ).uncertainty_enabled
    assert LossConfig(
        policy_enabled=False,
        soft_policy_enabled=False,
        wdl_enabled=False,
        ownership_enabled=True,
    ).ownership_enabled
    assert LossConfig(
        policy_enabled=False,
        soft_policy_enabled=False,
        wdl_enabled=False,
        short_term_value_enabled=True,
    ).short_term_value_enabled
    assert LossConfig(
        policy_enabled=False,
        soft_policy_enabled=False,
        wdl_enabled=False,
        draft_outcome_enabled=True,
    ).draft_outcome_enabled


def test_config_rejects_non_finite_weights() -> None:
    """A NaN weight is refused, and it has to be: under D17 it is now SILENT.

    ``float("nan") < 0.0`` is ``False``, so a NaN weight walked straight
    through the non-negativity guard. That used to be survivable in the loudest
    possible way -- the unconditional ``total + weight * loss`` turned the
    total into NaN on the first batch and every gradient with it. D17's
    ``if weight > 0.0`` inverts that: ``NaN > 0.0`` is ``False`` too, so the
    head is *skipped* from the total while ``per_head`` keeps reporting its
    curve. The result is a head that looks fully wired, publishes a falling
    loss to the journal, and receives no gradient at all -- which is exactly
    the failure the staged-head design exists to prevent, arriving through the
    door D17 opened (impl review F6).

    ``inf`` is the mirror image and refused by the same clause: it passes the
    ``>= 0`` check, passes the ``> 0.0`` guard, and poisons the total instead.

    Every weight field is covered, P0a and staged alike, because "which
    weights are validated" must not be a property of which ticket added the
    field. The message names the field in each case, since the whole cost of
    this class of bug is the hours between the symptom and the knob.
    """
    nan = float("nan")
    infinity = float("inf")

    with pytest.raises(ValueError, match="policy_weight must be finite"):
        LossConfig(policy_weight=nan)
    with pytest.raises(ValueError, match="soft_policy_weight must be finite"):
        LossConfig(soft_policy_weight=nan)
    with pytest.raises(ValueError, match="wdl_weight must be finite"):
        LossConfig(wdl_weight=nan)
    with pytest.raises(ValueError, match="mlh_weight must be finite"):
        LossConfig(mlh_weight=nan)
    with pytest.raises(ValueError, match="uncertainty_weight must be finite"):
        LossConfig(uncertainty_weight=nan)
    with pytest.raises(ValueError, match="ownership_weight must be finite"):
        LossConfig(ownership_weight=nan)
    with pytest.raises(ValueError, match="short_term_value_weight must be finite"):
        LossConfig(short_term_value_weight=nan)
    with pytest.raises(ValueError, match="draft_outcome_weight must be finite"):
        LossConfig(draft_outcome_weight=nan)

    with pytest.raises(ValueError, match="uncertainty_weight must be finite"):
        LossConfig(uncertainty_weight=infinity)
    with pytest.raises(ValueError, match="policy_weight must be finite"):
        LossConfig(policy_weight=-infinity)

    # The guard is finiteness, not "anything unusual": a large finite weight is
    # a legal, if aggressive, configuration and must still construct.
    assert LossConfig(uncertainty_weight=1e30).uncertainty_weight == 1e30


def test_a_malformed_sample_weight_is_rejected() -> None:
    """The weight vector must be 1-D over the batch and non-empty."""
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    with pytest.raises(ValueError, match="1-D"):
        assembler(outputs, targets, sample_weight=torch.ones((2, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="empty batch"):
        assembler(outputs, targets, sample_weight=torch.zeros(0, dtype=torch.float64))


# --------------------------------------------------------------------------
# Masked logits.
#
# A legality mask is applied by setting illegal actions to -inf, so both the
# policy logits and the *recorded prior* routinely arrive with -inf entries at
# exactly the positions where the search target is 0. Every test below has a
# finite twin -- the same batch with -1e3 instead of -inf, which softmaxes to
# the same distribution to well under float64 resolution -- so the assertion
# is not merely "finite" but "the number masking was supposed to produce".
# --------------------------------------------------------------------------

_NEG_INF = float("-inf")
_BIG_NEGATIVE = -1e3


def test_a_minus_inf_masked_prior_weighs_the_same_as_a_finite_mask() -> None:
    """0 * inf in the KL would NaN the weight, the batch and the loss."""
    target = torch.tensor([[0.0, 0.25, 0.75]], dtype=torch.float64)
    masked = torch.tensor([[_NEG_INF, 0.5, 2.0]], dtype=torch.float64)
    finite = torch.tensor([[_BIG_NEGATIVE, 0.5, 2.0]], dtype=torch.float64)

    from_mask = policy_surprise_weight(target, masked, alpha=1.0, max_weight=10.0)
    from_finite = policy_surprise_weight(target, finite, alpha=1.0, max_weight=10.0)

    assert math.isfinite(from_mask.item())
    assert from_mask.item() == pytest.approx(from_finite.item(), abs=1e-12)
    # And it is a real weight, not a degenerate 1: search moved this prior.
    assert from_mask.item() > 1.0


def test_a_minus_inf_prior_where_the_target_has_mass_saturates_the_clip() -> None:
    """Search chose a move the prior called impossible: infinite KL, clipped.

    The guard is on zero-target positions only, so this case stays visible --
    it is a contradiction in the sample, not a mask.
    """
    target = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    prior = torch.tensor([[0.0, 0.0, _NEG_INF]], dtype=torch.float64)
    weight = policy_surprise_weight(target, prior, alpha=1.0, max_weight=4.0)
    assert weight.item() == pytest.approx(4.0, abs=1e-12)


def test_minus_inf_policy_logits_at_zero_target_positions_stay_finite() -> None:
    """The policy CE analog: masked logits cost exactly nothing, not NaN."""
    ln2 = math.log(2.0)
    targets = {
        POLICY_TARGET: torch.tensor([[0.0, 1 / 3, 2 / 3]], dtype=torch.float64),
    }
    config = LossConfig(policy_enabled=True, wdl_enabled=False)

    masked_logits = torch.tensor([[_NEG_INF, 0.0, ln2]], dtype=torch.float64)
    finite_logits = torch.tensor([[_BIG_NEGATIVE, 0.0, ln2]], dtype=torch.float64)
    masked_total, masked_heads = LossAssembler(config)(
        {POLICY_LOGITS: masked_logits}, targets
    )
    finite_total, _ = LossAssembler(config)({POLICY_LOGITS: finite_logits}, targets)

    assert math.isfinite(masked_total.item())
    assert math.isfinite(masked_heads[HEAD_POLICY].item())
    assert masked_total.item() == pytest.approx(finite_total.item(), abs=1e-9)


def test_a_masked_batch_still_carries_finite_gradient() -> None:
    """A NaN loss would be visible; a NaN *gradient* is how a run dies quietly."""
    logits = torch.tensor(
        [[_NEG_INF, 0.0, math.log(2.0)]], dtype=torch.float64, requires_grad=True
    )
    targets = {POLICY_TARGET: torch.tensor([[0.0, 1 / 3, 2 / 3]], dtype=torch.float64)}
    config = LossConfig(policy_enabled=True, wdl_enabled=False)
    total, _ = LossAssembler(config)({POLICY_LOGITS: logits}, targets)
    total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()
    # The masked action gets no gradient: it is not a move the net may make.
    assert logits.grad[0, 0].item() == pytest.approx(0.0, abs=1e-12)


def test_a_minus_inf_logit_where_the_target_has_mass_is_not_hidden() -> None:
    """Documented behaviour: that is a contradiction, and it stays infinite."""
    outputs = {POLICY_LOGITS: torch.tensor([[_NEG_INF, 0.0, 0.0]], dtype=torch.float64)}
    targets = {POLICY_TARGET: torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)}
    total, _ = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))(
        outputs, targets
    )
    assert math.isinf(total.item())
    assert total.item() > 0.0


def test_the_whole_loss_survives_a_masked_prior_with_alpha_on() -> None:
    """End to end: every head and the total finite, weights finite and > 1."""
    outputs, targets = _golden_batch(torch.float64)
    # Recorded priors as self-play writes them: -inf at illegal moves, which
    # for this batch are exactly the zero-target positions.
    # Sample 0's prior softmaxes to [1/17, 16/17, 0] -- exactly its target.
    # Sample 1's is uniform over its two legal moves against a one-hot target,
    # so its KL is ln2 and its weight is 1 + 2*ln2.
    targets[PRIOR_LOGITS] = torch.tensor(
        [[0.0, math.log(16.0), _NEG_INF], [_NEG_INF, 0.0, 0.0]],
        dtype=torch.float64,
    )
    config = LossConfig(
        policy_enabled=True,
        soft_policy_enabled=True,
        soft_policy_weight=0.5,
        wdl_enabled=True,
        mlh_enabled=True,
        mlh_weight=0.25,
        surprise_alpha=2.0,
    )
    assembler = LossAssembler(config)
    weights = assembler.sample_weights(targets)
    total, per_head = assembler(outputs, targets)

    assert torch.isfinite(weights).all().item()
    assert math.isfinite(total.item())
    assert all(math.isfinite(value.item()) for value in per_head.values())
    assert weights[0].item() == pytest.approx(1.0, abs=1e-12)
    assert weights[1].item() == pytest.approx(1.0 + 2.0 * math.log(2.0), abs=1e-12)


def test_sample_weight_must_be_non_negative_and_carry_mass() -> None:
    """Its docstring's promises, checked rather than assumed."""
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    with pytest.raises(ValueError, match="non-negative"):
        assembler(
            outputs, targets, sample_weight=torch.tensor([-1.0, 2.0], dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="positive sum"):
        assembler(outputs, targets, sample_weight=torch.zeros(2, dtype=torch.float64))


def test_a_sample_weight_of_the_wrong_length_does_not_broadcast() -> None:
    """A (1,) vector used to broadcast over the batch and reweight nothing."""
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    with pytest.raises(ValueError, match="one per sample"):
        assembler(outputs, targets, sample_weight=torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="one per sample"):
        assembler(outputs, targets, sample_weight=torch.ones(3, dtype=torch.float64))


# --------------------------------------------------------------------------
# head_sample_weight: the per-head MULTIPLIER seam (KOM-036 §1.4, §1.5)
# --------------------------------------------------------------------------


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the per-sample full-distribution cross-entropy, computed here.

    Recomputed in the test rather than imported from the module under test:
    the nodes below assert that a head's loss is ``_weighted_mean(ce, base *
    multiplier)``, and importing the assembler's own arithmetic to check the
    assembler's arithmetic would make the composition claim circular.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    contributions = torch.where(
        target > 0, target * log_probs, torch.zeros_like(log_probs)
    )
    return -contributions.sum(dim=-1)


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Return ``sum(w*v) / sum(w)`` as a float."""
    return float(((values * weights).sum() / weights.sum()).item())


def test_a_head_sample_weight_multiplies_only_its_own_head() -> None:
    """A policy multiplier moves the policy head and leaves WDL bit-identical.

    Acceptance #1. The seam MULTIPLIES rather than overrides, so the name says
    so; what this node pins is the *scope*: an implementation that applied the
    mapping's vector to every head would move ``wdl`` too, and one that read
    the mapping for the wrong head would move nothing here.
    """
    outputs, targets = _golden_batch(torch.float64)
    config = LossConfig(policy_enabled=True, wdl_enabled=True)
    assembler = LossAssembler(config)

    _, plain = assembler(outputs, targets)
    _, weighted = assembler(
        outputs,
        targets,
        head_sample_weight={
            HEAD_POLICY: torch.tensor([1.0, 0.25], dtype=torch.float64)
        },
    )

    assert weighted[HEAD_POLICY].item() != plain[HEAD_POLICY].item()
    assert weighted[HEAD_WDL].item() == plain[HEAD_WDL].item()


def test_omitting_head_sample_weight_is_bit_identical_to_today() -> None:
    """Acceptance #2: ``None`` is the pre-seam call, to the bit.

    This is what makes the existing suites the regression for the whole
    change: every call site that predates the seam passes nothing, and nothing
    must be indistinguishable from the old code path rather than merely close
    to it.
    """
    outputs, targets = _golden_batch(torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    assembler = LossAssembler(_GOLDEN_CONFIG)

    without_total, without = assembler(outputs, targets, sample_weight=weights)
    with_none_total, with_none = assembler(
        outputs, targets, sample_weight=weights, head_sample_weight=None
    )

    assert with_none_total.item() == without_total.item()
    assert set(with_none) == set(without)
    for head, value in without.items():
        assert with_none[head].item() == value.item(), head


def test_an_all_ones_multiplier_is_bit_identical_to_omitting_it() -> None:
    """Arm C's algebra, at the assembler: ``weights * 1.0 == weights``.

    Not one of the numbered nodes on its own -- it is the assembler-level half
    of #13 -- and it is here because it is what licenses passing the mapping
    *unconditionally* from ``TrainStep.run``. If this were merely approximate,
    arm C would not be today's trainer and the A/B's reference arm would be a
    different experiment.
    """
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(_GOLDEN_CONFIG)
    ones = torch.ones(2, dtype=torch.float64)

    plain_total, plain = assembler(outputs, targets)
    seam_total, seam = assembler(
        outputs,
        targets,
        head_sample_weight={HEAD_POLICY: ones, HEAD_SOFT_POLICY: ones},
    )

    assert seam_total.item() == plain_total.item()
    for head, value in plain.items():
        assert seam[head].item() == value.item(), head


def test_a_per_head_weight_is_validated_like_the_base_vector() -> None:
    """Acceptance #3: same checks, with the head named in the message.

    A multiplier that broadcasts, extrapolates or arrives with the wrong rank
    is exactly as wrong as a base vector that does, and the message has to say
    *which head* -- a caller supplying two of them otherwise learns only that
    one of them was malformed.
    """
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=True))

    with pytest.raises(ValueError, match=HEAD_POLICY):
        assembler(
            outputs,
            targets,
            head_sample_weight={HEAD_POLICY: torch.ones(3, dtype=torch.float64)},
        )
    with pytest.raises(ValueError, match=HEAD_POLICY):
        assembler(
            outputs,
            targets,
            head_sample_weight={
                HEAD_POLICY: torch.tensor([-1.0, 1.0], dtype=torch.float64)
            },
        )
    with pytest.raises(ValueError, match=HEAD_POLICY):
        assembler(
            outputs,
            targets,
            head_sample_weight={HEAD_POLICY: torch.ones((2, 2), dtype=torch.float64)},
        )


def test_a_zero_mass_head_is_omitted_not_divided_by_zero() -> None:
    """Acceptance #4: a zero COMPOSED vector omits its head; nothing is ``nan``.

    The rule is stated over ``weights * multiplier``, so the case under test is
    a non-zero base with an all-zero multiplier -- the discard arm on a batch
    whose rows are all fast, which at ``prob_fast = 0.75`` is routine at the
    batch sizes a unit suite uses. Omitted rather than reported as ``0.0``,
    because a zero in a loss curve is a measurement and this is an absence.
    """
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(
        LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            wdl_enabled=True,
            wdl_weight=1.5,
        )
    )

    baseline_total, _ = assembler(outputs, targets)
    total, per_head = assembler(
        outputs,
        targets,
        head_sample_weight={HEAD_POLICY: torch.zeros(2, dtype=torch.float64)},
    )

    assert HEAD_POLICY not in per_head
    assert HEAD_WDL in per_head
    assert math.isfinite(total.item())
    assert math.isfinite(per_head[HEAD_WDL].item())
    # It contributed 0.0: the total is the baseline minus the policy term.
    assert total.item() == pytest.approx(
        baseline_total.item() - 1.0 * _GOLDEN_POLICY, abs=1e-9
    )


def test_a_head_multiplier_composes_with_the_assembler_base() -> None:
    """Acceptance #4a: ``base * v``, not ``v``, and not ones.

    The base here is the one the assembler resolves *for itself* --
    ``sample_weights(targets)`` under ``LossConfig(surprise_alpha=1.0)``, with
    no explicit ``sample_weight=``. That is the shape of the defect the
    multiplier exists to prevent: an implementation that replaced the base
    would silently drop surprise weighting from the reweighted heads while
    every other head kept it.

    This node works at the assembler level, where the test supplies its own
    ``targets`` mapping and can therefore put ``prior_logits`` in it. The
    trainer-level node (#13a) cannot be written this way, because
    ``targets_from_batch`` filters ``prior_logits`` out on purpose.
    """
    outputs, targets = _golden_batch(torch.float64)
    targets[PRIOR_LOGITS] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    config = LossConfig(
        policy_enabled=True,
        wdl_enabled=True,
        surprise_alpha=1.0,
        surprise_max_weight=5.0,
    )
    assembler = LossAssembler(config)
    base = assembler.sample_weights(targets)
    assert not torch.equal(base, torch.ones_like(base)), (
        "the surprise base is all ones, so this node could not distinguish "
        "`base * v` from `v`"
    )

    multiplier = torch.tensor([1.0, 0.125], dtype=torch.float64)
    _, per_head = assembler(
        outputs, targets, head_sample_weight={HEAD_POLICY: multiplier}
    )

    ce = _cross_entropy(outputs[POLICY_LOGITS], targets[POLICY_TARGET])
    assert per_head[HEAD_POLICY].item() == pytest.approx(
        _weighted(ce, base * multiplier), abs=1e-12
    )
    # Red against a replacement: `v` alone gives a different number here.
    assert per_head[HEAD_POLICY].item() != pytest.approx(
        _weighted(ce, multiplier), abs=1e-12
    )
    # ... and the head with no multiplier kept the base exactly.
    wdl_ce = _cross_entropy(outputs[WDL_LOGITS], targets[WDL_TARGET])
    assert per_head[HEAD_WDL].item() == pytest.approx(
        _weighted(wdl_ce, base), abs=1e-12
    )


def test_an_all_zero_base_weight_is_still_an_error() -> None:
    """Acceptance #5: the base's zero-sum refusal is unchanged by the seam.

    A composed zero is reachable by design; an all-zero *base* is a caller
    bug, and the two must not be conflated -- defining the second one away
    would swallow the very error the first one's definition relies on being
    kept.
    """
    outputs, targets = _golden_batch(torch.float64)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    with pytest.raises(ValueError, match="positive sum"):
        assembler(outputs, targets, sample_weight=torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="positive sum"):
        assembler(
            outputs,
            targets,
            sample_weight=torch.zeros(2, dtype=torch.float64),
            head_sample_weight={HEAD_POLICY: torch.ones(2, dtype=torch.float64)},
        )
