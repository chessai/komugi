"""The P0a recipe, re-pinned here, and what the surprise weight actually does.

Two jobs, both about numbers nobody should be able to change quietly.

**The recipe.** ``P0A_RECIPE``'s weights are written out longhand below --
8.0 on soft policy, moves-left off, surprise off -- rather than compared
against whatever the wiring module happens to hold. That is the point: if the
trainer's own unit suite and this suite disagree about the production recipe,
the build should turn into a loud argument between two files, not into a
silently re-tuned run. §2.3's 8x soft-policy weight is the number that has to
survive; ``mlh`` is off because §2.4 stages heads in deliberately, and
``surprise_alpha`` is 0 because switching surprise weighting on is a
measurable act, not an inheritance.

**The weight.** ``surprise_weights_from_kl`` is pinned twice over: as
arithmetic (the four goldens -- zero, negative, mid-range, saturated) and as
*behaviour*. The behavioural half is the one that matters, and it exists
because of the KOM-020 follow-up gap: ``LossAssembler.sample_weights``
consumes an ``(A,)`` ``prior_logits`` vector that the D4 schema deliberately
does not store, so the only path from the stored ``policy_kl`` scalar to a
per-sample weight runs through the trainer passing ``sample_weight=``
explicitly. A ``TrainStep`` that reads ``surprise_alpha`` off its config,
computes nothing and hands the assembler no weights would satisfy every
arithmetic golden in this file. It would not survive
``test_surprise_alpha_changes_the_step``: two identical nets, one batch, one
row that search found surprising, and the only difference between the runs is
``alpha`` -- so if their parameters end up bit-identical, the weight reached
no gradient.
"""

from __future__ import annotations

from typing import Any, Final

import pytest
import torch

import synthetic as syn
from komugi.train import losses
from komugi.train.shards import stack_samples

wiring = pytest.importorskip("komugi.train.wiring")
if not hasattr(wiring, "P0A_RECIPE"):
    pytest.skip(
        "komugi.train.wiring is present but exports no P0A_RECIPE: the "
        "KOM-030 training-step contract has not landed yet",
        allow_module_level=True,
    )
step_module = pytest.importorskip("komugi.train.step")

_SOFT_POLICY_WEIGHT: Final[float] = 8.0
"""§2.3's production soft-policy weight. Written out, not imported."""

_SOFT_POLICY_EXPONENT: Final[float] = 0.25
"""KataGo's sharpening exponent, ``target ** 0.25`` renormalised."""

_ROWS: Final[int] = 8
_HOT_ROW: Final[int] = 0
_HOT_KL: Final[float] = 5.0
_STEP_LR: Final[float] = 0.05
_MAX_WEIGHT: Final[float] = 5.0


def _batch(hot_kl_row: int | None) -> dict[str, torch.Tensor]:
    """Return a stacked batch of :mod:`synthetic` rows, one optionally surprising."""
    return stack_samples(syn.toy_rows(_ROWS, hot_kl_row=hot_kl_row, hot_kl=_HOT_KL))


def _stepped_parameters(alpha: float) -> dict[str, torch.Tensor]:
    """Run exactly one training step at `alpha` and return the net's parameters.

    Everything but `alpha` is fixed: the net is built from a seeded config
    (equal configs give bit-identical ``state_dict``s), the batch is the same
    object graph both times, and the step's own seed is 0. So any difference
    between two calls is attributable to the surprise weighting and to nothing
    else.
    """
    net = syn.tiny_net()
    trainer: Any = step_module.TrainStep(
        net,
        losses.LossAssembler(wiring.P0A_RECIPE),
        step_module.StepConfig(
            batch_size=_ROWS,
            lr=_STEP_LR,
            surprise_alpha=alpha,
            surprise_max_weight=_MAX_WEIGHT,
            seed=0,
        ),
    )
    report: dict[str, float] = trainer.run(_batch(_HOT_ROW))
    assert report, "the step reported no metrics at all"
    return syn.parameter_snapshot(net)


def test_p0a_recipe_is_the_production_loss_recipe() -> None:
    """The recipe's numbers, restated independently of the wiring module."""
    recipe: losses.LossConfig = wiring.P0A_RECIPE
    assert isinstance(recipe, losses.LossConfig)

    assert recipe.policy_enabled is True
    assert recipe.policy_weight == 1.0

    assert recipe.soft_policy_enabled is True, (
        "soft policy is ON in P0a: it is the auxiliary target that keeps the "
        "policy from collapsing onto its own top move at branching ~229"
    )
    assert recipe.soft_policy_weight == _SOFT_POLICY_WEIGHT
    assert recipe.soft_policy_exponent == _SOFT_POLICY_EXPONENT

    assert recipe.wdl_enabled is True
    assert recipe.wdl_weight == 1.0

    assert recipe.mlh_enabled is False, "moves-left is a staged head (§2.4)"
    assert recipe.mlh_weight == 0.0

    assert recipe.surprise_alpha == 0.0, (
        "surprise weighting is off in the P0a recipe; turning it on is a "
        "deliberate, measured change"
    )
    assert recipe.surprise_max_weight >= 1.0


def test_p0b_recipe_lands_every_staged_head_off() -> None:
    """The P0b production recipe: P0a's eleven fields, plus four heads off.

    Two claims, and the second is the one that costs something to break.

    First, ``P0B_RECIPE`` agrees with ``P0A_RECIPE`` on every field that
    predates it -- ``soft_policy_weight == 8.0`` included, since a staged-head
    ticket silently halving the production auxiliary weight is exactly the
    class of change a field-by-field pin exists to catch.

    Second, ``P0A_RECIPE`` itself is **unedited**. A curve produced under a
    named recipe must stay reconstructible from that name, so the staged heads
    arrive as a new constant rather than as an amendment to the old one; the
    two tests either side of this one score ``P0A_RECIPE`` and keep passing
    verbatim. Note that they are therefore *blind* to ``P0B_RECIPE``: neither
    would catch a staged head switched on by accident. The node that catches
    that is the metrics-key assertion in
    ``tests/train/behavior/test_staged_off_identity.py``.
    """
    p0a: losses.LossConfig = wiring.P0A_RECIPE
    recipe: losses.LossConfig = wiring.P0B_RECIPE
    assert isinstance(recipe, losses.LossConfig)
    assert recipe is not p0a

    for field in (
        "policy_enabled",
        "policy_weight",
        "soft_policy_enabled",
        "soft_policy_weight",
        "soft_policy_exponent",
        "wdl_enabled",
        "wdl_weight",
        "mlh_enabled",
        "mlh_weight",
        "surprise_alpha",
        "surprise_max_weight",
    ):
        assert getattr(recipe, field) == getattr(p0a, field), (
            f"P0B_RECIPE disagrees with P0A_RECIPE on {field}: the staged "
            "heads are the only thing P0b changes about the recipe"
        )
    assert recipe.soft_policy_weight == _SOFT_POLICY_WEIGHT

    assert recipe.uncertainty_enabled is False
    assert recipe.uncertainty_weight == 0.0
    assert recipe.ownership_enabled is False
    assert recipe.ownership_weight == 0.0
    assert recipe.short_term_value_enabled is False
    assert recipe.short_term_value_weight == 0.0
    assert recipe.draft_outcome_enabled is False
    assert recipe.draft_outcome_weight == 0.0

    # P0A_RECIPE is not edited: it too carries the staged fields, at the same
    # OFF defaults, because they are dataclass defaults rather than a change
    # to the constant.
    assert p0a.uncertainty_enabled is False
    assert p0a.ownership_enabled is False
    assert p0a.short_term_value_enabled is False
    assert p0a.draft_outcome_enabled is False


def test_the_recipe_scores_exactly_three_heads() -> None:
    """The recipe as the assembler sees it: policy, soft policy, WDL -- no mlh."""
    net = syn.tiny_net()
    rows = syn.toy_rows(_ROWS)
    outputs: dict[str, torch.Tensor] = wiring.outputs_from_net(
        net(syn.observations(rows))
    )
    targets: dict[str, torch.Tensor] = wiring.targets_from_batch(_batch(None))

    total, per_head = losses.LossAssembler(wiring.P0A_RECIPE)(outputs, targets)

    assert set(per_head) == {
        losses.HEAD_POLICY,
        losses.HEAD_SOFT_POLICY,
        losses.HEAD_WDL,
    }
    assert bool(torch.isfinite(total).item())
    # The total is the weighted sum, and soft policy carries 8 of the 10 units
    # of weight in it -- so it cannot be a rounding error on the others.
    expected = (
        per_head[losses.HEAD_POLICY]
        + _SOFT_POLICY_WEIGHT * per_head[losses.HEAD_SOFT_POLICY]
        + per_head[losses.HEAD_WDL]
    )
    assert torch.allclose(total, expected)


def test_surprise_weight_goldens() -> None:
    """``clip(1 + alpha * clamp(kl, min=0), 1, max_weight)``, four values wide."""
    kl = torch.tensor([0.0, -2.5, 1.5, 40.0], dtype=torch.float32)

    weights: torch.Tensor = wiring.surprise_weights_from_kl(
        kl, alpha=2.0, max_weight=_MAX_WEIGHT
    )
    assert weights.shape == kl.shape
    assert weights.dtype == kl.dtype
    assert torch.allclose(
        weights, torch.tensor([1.0, 1.0, 4.0, _MAX_WEIGHT], dtype=torch.float32)
    ), (
        "expected kl=0 -> 1 (search agreed), kl<0 -> 1 (clamped, never a "
        "down-weight), kl=1.5 at alpha=2 -> 4, and a large kl saturated at "
        f"max_weight={_MAX_WEIGHT}"
    )

    # Exactly at the cap: 1 + 1 * 5 = 6, clipped to 5.
    at_cap: torch.Tensor = wiring.surprise_weights_from_kl(
        torch.tensor([_HOT_KL]), alpha=1.0, max_weight=_MAX_WEIGHT
    )
    assert float(at_cap.item()) == _MAX_WEIGHT

    off: torch.Tensor = wiring.surprise_weights_from_kl(
        kl, alpha=0.0, max_weight=_MAX_WEIGHT
    )
    assert torch.equal(off, torch.ones_like(kl)), (
        "alpha=0 must return exact ones for every kl, including the negative "
        "and the huge one"
    )


def test_surprise_alpha_changes_the_step() -> None:
    """One surprising row, one step: alpha=1 must land somewhere alpha=0 does not."""
    baseline = _stepped_parameters(0.0)
    control = _stepped_parameters(0.0)
    weighted = _stepped_parameters(1.0)

    assert set(baseline) == set(weighted)
    assert all(torch.equal(control[name], baseline[name]) for name in baseline), (
        "two steps at the same alpha disagreed: the step is not deterministic, "
        "so the comparison below cannot attribute anything to the weighting"
    )

    differing = [
        name for name in baseline if not torch.equal(baseline[name], weighted[name])
    ]
    assert differing, (
        f"row {_HOT_ROW} carries policy_kl={_HOT_KL} and every other row 0.0, "
        "yet a step at surprise_alpha=1 moved the net to exactly where "
        "surprise_alpha=0 did. The stored scalar never became a per-sample "
        "weight -- see KOM-020's 2026-08-23 follow-up: the assembler wants an "
        "(A,) prior_logits vector the D4 schema does not store, so the trainer "
        "must compute the weights itself and pass sample_weight= through"
    )
