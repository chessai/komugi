"""The KOM-036 A/B's mechanism, at the trainer.

The whole experiment is one float: ``StepConfig.fast_policy_weight``, the
multiplier a fast row's policy and soft-policy loss is scaled by. Arm A is
``0.125`` (derived, ``gumbel_n_fast / gumbel_n_full = 32 / 256``), arm B is
``0.0`` (PCR-style discard), arm C is ``1.0`` (today).

What this file exists to make checkable, in order of how easy each is to get
wrong:

* **Arm B is not a row filter.** A filter would drop the row's *value* signal
  too, and the whole content of playout-cap randomisation is that a fast move
  still advances the game -- so its outcome-derived targets are exactly as good
  as a full move's. ``test_arm_b_still_trains_wdl_from_fast_rows`` is the node
  that distinguishes the two, and it is the reason the arm is a weight rather
  than a filter.
* **The soft-policy head moves with the policy head.** Its target is *derived
  from the same* ``policy_target`` at eight times the policy weight, so a
  "discard fast policy targets" arm that left it alone would be a no-op wearing
  a label.
* **The normaliser is the weight mass**, not the batch size, so the arms differ
  in the *composition* of the policy gradient and not in its scale. Without
  that, arm B would be a learning-rate ablation in disguise.
* **The multiplier composes with whatever base the assembler resolved** rather
  than replacing it -- pinned here at the *trainer* level, over
  ``StepConfig.surprise_alpha``, which is the surprise base a training step can
  actually reach.

Every node here runs the production ``TrainStep.run``. A step's update is a
deterministic function of (net state, batch), and the nets are built from a
seeded config, so "these two runs differ" is attributable to the arm and to
nothing else.
"""

from __future__ import annotations

import math
from typing import Final

import pytest
import torch

import synthetic as syn
from komugi.modes import MODE_FAST, MODE_FULL
from komugi.train.losses import (
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    POLICY_LOGITS,
    POLICY_TARGET,
    WDL_LOGITS,
    WDL_TARGET,
    LossAssembler,
    LossConfig,
)
from komugi.train.shards import stack_samples
from komugi.train.step import TOTAL_KEY, StepConfig, TrainStep
from komugi.train.wiring import (
    P0A_RECIPE,
    fast_row_weights,
    outputs_from_net,
    surprise_weights_from_kl,
    targets_from_batch,
)

_ROWS: Final[int] = 8
_LR: Final[float] = 0.05
"""Large enough that one step separates two different gradients well clear of
float32 noise, which is what the "these parameters differ" assertions need."""

_ARMS: Final[tuple[float, ...]] = (0.0, 0.125, 1.0)
"""Arm B, arm A, arm C. Every node that can run over all three does."""

_FAST_ROWS: Final[frozenset[int]] = frozenset({1, 3, 5, 7})
"""Which rows of the mixed batch are fast. Half, so neither stratum is a
rounding error and the weight-mass normaliser has real work to do."""


def _mixed_rows(
    *,
    policy_bump: int | None = None,
    wdl_bump: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Return `_ROWS` rows, half fast, with one row's target optionally moved.

    The perturbation is what makes "this row reached the gradient" observable:
    a row whose target changed and whose head's parameters did not is a row the
    head did not learn from.
    """
    rows = [
        syn.toy_row(
            index,
            policy_kl=0.25 * index,
            search_mode=MODE_FAST if index in _FAST_ROWS else MODE_FULL,
        )
        for index in range(_ROWS)
    ]
    if policy_bump is not None:
        target = rows[policy_bump][POLICY_TARGET].clone()
        # Move the whole distribution rather than nudging it: half the mass to
        # a different action, so a head that reads this row cannot miss it.
        moved = torch.zeros_like(target)
        moved[(int(target.argmax()) + 1) % target.numel()] = 0.5
        rows[policy_bump][POLICY_TARGET] = 0.5 * target + moved
    if wdl_bump is not None:
        # The other one-hot class: a different game result entirely.
        wdl = rows[wdl_bump][WDL_TARGET]
        rows[wdl_bump][WDL_TARGET] = torch.roll(wdl, 1)
    return rows


def _stepped(
    rows: list[dict[str, torch.Tensor]], fast_policy_weight: float
) -> dict[str, torch.Tensor]:
    """Run one production step over `rows` at `fast_policy_weight`.

    Returns the net's parameters afterwards. AdamW's update is a closed form of
    the gradient and its own (fresh) state, so equal parameters after one step
    from equal starting weights means equal gradients, and different parameters
    mean different gradients.
    """
    net = syn.tiny_net()
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=len(rows),
            lr=_LR,
            fast_policy_weight=fast_policy_weight,
            seed=0,
        ),
    )
    report = step.run(stack_samples(rows))
    assert math.isfinite(report[TOTAL_KEY]), report
    return syn.parameter_snapshot(net)


def _module_parameters(
    snapshot: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Return the entries of `snapshot` belonging to submodule `prefix`."""
    selected = {
        name: value
        for name, value in snapshot.items()
        if name.startswith(f"{prefix}.")
    }
    assert selected, f"no parameters under {prefix!r}; the filter is vacuous"
    return selected


def _same(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    """Return whether two parameter dicts are equal tensor for tensor."""
    assert set(left) == set(right)
    return all(torch.equal(left[name], right[name]) for name in left)


# --------------------------------------------------------------------------
# #9, #9a -- what the arm does and does not change
# --------------------------------------------------------------------------


def test_the_fast_weight_never_touches_a_full_row() -> None:
    """Acceptance #9: on an all-full batch the three arms are one step.

    The arm applies to fast rows and to nothing else, so a batch with no fast
    row must be bit-identical across arms. Worth having even though it cannot
    see the experiment's composition claim (an all-full batch's multiplier is
    all ones): it is red against a mapping keyed on the *wrong* mode, which
    would down-weight the full rows instead.
    """
    rows = [
        syn.toy_row(index, search_mode=MODE_FULL) for index in range(_ROWS)
    ]
    reference = _stepped(rows, 1.0)
    for arm in _ARMS:
        assert _same(_stepped(rows, arm), reference), (
            f"arm {arm} moved an all-full batch's parameters; the fast weight "
            f"is reaching rows whose search_mode is MODE_FULL"
        )


@pytest.mark.parametrize("arm", _ARMS)
def test_the_normaliser_is_the_weight_mass_on_a_mixed_batch(arm: float) -> None:
    """Acceptance #9a: the arm changes the mix, not the loss scale.

    The batch is built so that **every row's policy cross-entropy is the same
    constant** ``ln 3``: uniform logits over three actions, and any normalised
    target then gives ``-sum(t * log(1/3)) == ln 3``. A weighted mean of a
    constant is that constant *whatever the weights are* -- but only if the
    normaliser is the weight mass. An implementation dividing by the batch size
    or by a fixed row count reports ``ln3 * (mass / n)`` instead, which moves
    with the arm.

    This is the property §1.3 rests on: because the gradient magnitude does not
    change between arms, the comparison is a composition change rather than a
    learning-rate ablation in disguise.
    """
    modes = torch.tensor(
        [MODE_FAST if index in _FAST_ROWS else MODE_FULL for index in range(_ROWS)]
    )
    uniform = torch.zeros(_ROWS, 3, dtype=torch.float64)
    target = torch.zeros(_ROWS, 3, dtype=torch.float64)
    for index in range(_ROWS):
        target[index, index % 3] = 1.0

    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    fast = fast_row_weights(modes, arm)
    _, per_head = assembler(
        {POLICY_LOGITS: uniform},
        {POLICY_TARGET: target},
        head_sample_weight={HEAD_POLICY: fast},
    )

    assert per_head[HEAD_POLICY].item() == pytest.approx(math.log(3.0), abs=1e-12)


# --------------------------------------------------------------------------
# #10, #11, #12 -- which head a fast row reaches, at arm B
# --------------------------------------------------------------------------


def test_arm_b_gives_fast_rows_zero_policy_gradient() -> None:
    """Acceptance #10: at ``w = 0`` a fast row's policy target is inert.

    Perturbing a *fast* row's ``policy_target`` must change nothing -- the row
    contributes zero mass to the policy and soft-policy heads, and its WDL
    target is untouched here -- while perturbing a *full* row's must change the
    update. The pair is what makes this a statement about the stratum rather
    than about the perturbation being too small to see.
    """
    baseline = _stepped(_mixed_rows(), 0.0)
    fast_row = min(_FAST_ROWS)
    full_row = min(set(range(_ROWS)) - _FAST_ROWS)

    assert _same(_stepped(_mixed_rows(policy_bump=fast_row), 0.0), baseline), (
        "moving a fast row's policy target changed the update at arm B; the "
        "discard arm is not discarding"
    )
    assert not _same(_stepped(_mixed_rows(policy_bump=full_row), 0.0), baseline), (
        "moving a FULL row's policy target changed nothing either, so the "
        "assertion above is vacuous"
    )


def test_arm_b_still_trains_wdl_from_fast_rows() -> None:
    """Acceptance #11: the node that distinguishes the arm from a row filter.

    A fast move still advances the game, so its ``wdl_target`` is a function of
    the game's end rather than of the search -- exactly as good as a full
    move's. A filter that dropped fast rows before the batch was assembled
    would throw that away (and would break matched compute besides). The weight
    keeps it.
    """
    baseline = _stepped(_mixed_rows(), 0.0)
    fast_row = min(_FAST_ROWS)

    assert not _same(_stepped(_mixed_rows(wdl_bump=fast_row), 0.0), baseline), (
        "moving a fast row's WDL target changed nothing at arm B: the arm has "
        "become a row filter and is discarding the value signal too"
    )


def test_the_soft_policy_head_is_weighted_with_the_policy_head() -> None:
    """Acceptance #12: §1.4's 8x trap.

    ``P0A_RECIPE`` scores soft policy at **eight times** the policy head's
    weight, from a target derived from the same ``policy_target``. An arm that
    weighted only ``HEAD_POLICY`` would therefore still train on every fast
    policy target, at 8x, and arm B would be a no-op wearing a label. The
    assertion is on ``soft_policy``'s own parameters, so it is red against
    exactly that omission.
    """
    fast_row = min(_FAST_ROWS)
    baseline = _module_parameters(_stepped(_mixed_rows(), 0.0), "soft_policy")
    bumped = _module_parameters(
        _stepped(_mixed_rows(policy_bump=fast_row), 0.0), "soft_policy"
    )

    assert _same(bumped, baseline), (
        "moving a fast row's policy target moved the SOFT-policy head at arm "
        "B; the fast weight reaches HEAD_POLICY but not HEAD_SOFT_POLICY"
    )


# --------------------------------------------------------------------------
# #13, #13a -- arm C is today, and the multiplier composes
# --------------------------------------------------------------------------


def _pre_change_step(
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Reproduce ``TrainStep.run`` *without* the per-head seam, verbatim.

    Everything ``run`` does -- ``net.train()``, the forward, the wiring, the
    assembler, the pre-clear, ``backward``, ``step``, the post-clear -- with
    one difference: no ``head_sample_weight``. This is the code path that
    existed before KOM-036, spelled out here so "bit-identical to today" is a
    comparison against something rather than an assertion about a diff.
    """
    net = syn.tiny_net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=_LR, weight_decay=0.0)
    batch = stack_samples(rows)
    net.train()
    outputs = outputs_from_net(net(batch["observation"]))
    targets = targets_from_batch(batch)
    total, _ = LossAssembler(P0A_RECIPE)(outputs, targets, sample_weight=None)
    optimizer.zero_grad(set_to_none=True)
    total.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return syn.parameter_snapshot(net)


def test_weight_one_is_bit_identical_to_the_current_trainer() -> None:
    """Acceptance #13: arm C is today's trainer, to the bit, through the seam.

    The mapping is passed **unconditionally**, so ``w = 1.0`` runs an all-ones
    multiplier through ``weights * 1.0`` rather than bypassing the seam. That
    is what makes this node load-bearing: under a ``!= 1.0`` guard it would
    pin the bypass, and an incorrectly computed ``head_sample_weight`` would
    still pass it.
    """
    rows = _mixed_rows()
    assert _same(_stepped(rows, 1.0), _pre_change_step(rows)), (
        "arm C's step differs from the pre-seam step; `weights * ones` is not "
        "the identity here, so arm C is not the status quo it is reported as"
    )


def test_the_fast_weight_composes_with_a_surprise_weighted_base() -> None:
    """Acceptance #13a: ``kl_base * fast`` on policy, ``kl_base`` alone on WDL.

    The reachable surprise base is ``StepConfig.surprise_alpha``, not
    ``LossConfig``'s: ``run`` builds targets through ``targets_from_batch``,
    which projects only ``TARGET_KEYS``, and ``prior_logits`` is deliberately
    absent from both that tuple and the schema -- so a trainer step under
    ``LossConfig(surprise_alpha=1.0)`` raises ``KeyError`` inside
    ``sample_weights`` before any weighting happens. An implementer who sees
    that ``KeyError`` has hit that fact, not a bug in the seam.

    The pair of assertions is what discriminates. **Red** against an
    implementation that *replaces* the base on the policy head (it would report
    ``_weighted_mean(ce_policy, fast)``), and red against one that applies
    ``fast`` to every head (the WDL assertion fails). Green only against the
    multiplier, applied to policy and soft-policy alone.

    The second net is not a convenience: ``run``'s forward runs in train mode
    and updates BatchNorm's running estimates, so recomputing the logits on the
    *same* net afterwards would not reproduce them. Two nets from one seeded
    config start bit-identical, and a forward is a pure function of (weights,
    batch).
    """
    rows = _mixed_rows()
    batch = stack_samples(rows)
    alpha = 1.0
    max_weight = 5.0
    arm = 0.125

    trained = syn.tiny_net()
    step = TrainStep(
        trained,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=_ROWS,
            lr=_LR,
            surprise_alpha=alpha,
            surprise_max_weight=max_weight,
            fast_policy_weight=arm,
            seed=0,
        ),
    )
    metrics = step.run(batch)

    mirror = syn.tiny_net()
    mirror.train()
    outputs = outputs_from_net(mirror(batch["observation"]))

    kl_base = surprise_weights_from_kl(
        batch["policy_kl"], alpha=alpha, max_weight=max_weight
    )
    assert not torch.equal(kl_base, torch.ones_like(kl_base)), (
        "the surprise base is all ones, so this node could not tell "
        "`kl_base * fast` from `fast` alone"
    )
    fast = fast_row_weights(batch["search_mode"], arm)

    policy_ce = _cross_entropy(outputs[POLICY_LOGITS], batch[POLICY_TARGET])
    wdl_ce = _cross_entropy(outputs[WDL_LOGITS], batch[WDL_TARGET])

    assert metrics[HEAD_POLICY] == pytest.approx(
        _weighted(policy_ce, kl_base * fast), rel=1e-5
    )
    assert metrics[HEAD_POLICY] != pytest.approx(
        _weighted(policy_ce, fast), rel=1e-5
    )
    assert metrics[HEAD_WDL] == pytest.approx(_weighted(wdl_ce, kl_base), rel=1e-5)
    assert metrics[HEAD_WDL] != pytest.approx(
        _weighted(wdl_ce, kl_base * fast), rel=1e-5
    )
    assert HEAD_SOFT_POLICY in metrics


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-sample ``-sum(target * log_softmax(logits))``, computed here.

    Recomputed rather than imported from the module under test: checking the
    assembler's arithmetic with the assembler's arithmetic would make the
    composition claim circular.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    contributions = torch.where(
        target > 0, target * log_probs, torch.zeros_like(log_probs)
    )
    return -contributions.sum(dim=-1)


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Return ``sum(w*v) / sum(w)`` as a float."""
    return float(((values * weights).sum() / weights.sum()).item())
