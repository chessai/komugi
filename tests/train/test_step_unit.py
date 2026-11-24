"""The training step, the wiring under it, and the recipe it runs.

Four things are pinned here, in rising order of how quietly they would fail:

* :func:`surprise_weights_from_kl` -- the stored-scalar path the wave-9b review
  found "exists nowhere and no test pins it" (KOM-020, 2026-08-23). Goldens
  including both ends of the clip.
* :data:`P0A_RECIPE` -- field by field, so no run can inherit
  :class:`LossConfig`'s defaults while believing it ran the design's §2.3-2.4
  recipe.
* :class:`TrainStep` -- one step on a real (tiny) :class:`KomugiNet` moves the
  parameters and lowers the loss on the same batch. A step that silently
  trained nothing would otherwise look exactly like one that worked.
* The **WDL index pin** (the KOM-005a review's F5b obligation). ``WDLHead``'s
  docstring says its three logits are ``(win, draw, loss)`` -- index 0 win --
  and the loss's three-way cross-entropy takes that ordering on faith. Nothing
  crashes if the two disagree; the net simply learns to predict losses as wins.
  So the order is checked mechanically end to end: a win-target one-hot at
  index 0 must be *minimised* by logits favouring index 0, through the same
  ``net -> outputs_from_net -> LossAssembler`` path the trainer uses.

The net here is the real :class:`KomugiNet` at one block and 81 actions -- the
``_tiny_net`` pattern from ``tests/train/test_netoutput_wiring.py`` -- because
the wiring these tests exist for is precisely the seam between that class and
the assembler.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

from komugi.engine.observation import MAX_TURNS
from komugi.modes import MODE_FULL
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics
from komugi.terminations import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.train.checkpoint import load_checkpoint
from komugi.train.losses import (
    HEAD_MLH,
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    MLH_TARGET,
    POLICY_LOGITS,
    POLICY_TARGET,
    WDL_LOGITS,
    WDL_TARGET,
    LossAssembler,
    LossConfig,
    policy_surprise_weight,
)
from komugi.train.shards import stack_samples
from komugi.train.step import (
    DEFAULT_CAP_PLIES,
    TOTAL_KEY,
    NonFiniteUpdateError,
    StepConfig,
    TrainStep,
    training_iteration,
)
from komugi.train.wiring import (
    MLH_RECIPE,
    P0A_RECIPE,
    outputs_from_net,
    surprise_weights_from_kl,
    targets_from_batch,
)

_PLANES = 4
_ACTIONS = 81


def _tiny_net(seed: int = 7) -> KomugiNet:
    """A one-block, 81-action net: big enough to wire, small enough to run."""
    trunk = TrunkConfig(in_planes=_PLANES, channels=8, blocks=1, gpool_every=3, seed=seed)
    heads = HeadsConfig(channels=8, num_actions=_ACTIONS, policy_planes=1, seed=seed)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _sample(
    row: int,
    *,
    search_mode: int = MODE_FULL,
    termination: int = TERMINATION_DECISIVE,
) -> dict[str, torch.Tensor]:
    """One :data:`komugi.selfplay.records.SCHEMA`-shaped row, hand-built.

    Deterministic in `row` rather than random: a batch that changes between
    runs would make "the loss went down" a coin flip. ``ply`` is carried to
    prove that provenance columns do *not* reach the assembler.

    ``search_mode`` is carried because :meth:`TrainStep.run` requires it
    unconditionally (KOM-036 §1.5): the fast-row policy multiplier is applied
    at every arm, arm C (``1.0``) included, so that the three arms share one
    code path. It defaults to :data:`~komugi.modes.MODE_FULL`, where the
    multiplier is exactly ones.

    ``termination`` is carried on the same terms (KOM-160): the cap-row WDL
    multiplier is applied at every ``cap_row_value_weight``, ``1.0`` included.
    It defaults to :data:`~komugi.terminations.TERMINATION_DECISIVE`, where the
    multiplier is exactly ones and every node in this file that does not name
    the knob is unaffected by it.
    """
    generator = torch.Generator().manual_seed(1000 + row)
    policy = torch.rand(_ACTIONS, generator=generator) + 0.05
    wdl = torch.zeros(3)
    wdl[row % 3] = 1.0
    return {
        "observation": torch.rand(
            (_PLANES, 9, 9), generator=generator, dtype=torch.float32
        ),
        "policy_target": policy / policy.sum(),
        "wdl_target": wdl,
        "mlh_target": torch.tensor(0.25 * (row + 1), dtype=torch.float32),
        "policy_kl": torch.tensor(0.5 * row, dtype=torch.float32),
        "ply": torch.tensor(row, dtype=torch.int64),
        "search_mode": torch.tensor(search_mode, dtype=torch.int64),
        "termination": torch.tensor(termination, dtype=torch.int64),
    }


def _batch(rows: int = 2) -> dict[str, torch.Tensor]:
    """Stack `rows` hand-built samples into a batch, as the shard reader would.

    Through :func:`komugi.train.shards.stack_samples`, the same function the
    window uses, so the batch a step sees here is shaped by the production
    path rather than by this file's idea of it.
    """
    return stack_samples([_sample(row) for row in range(rows)])


# --------------------------------------------------------------------------
# targets_from_batch
# --------------------------------------------------------------------------


def test_targets_from_batch_passes_only_what_the_assembler_consumes() -> None:
    """The three target columns, verbatim, and nothing else from the row."""
    batch = _batch()
    targets = targets_from_batch(batch)

    assert set(targets) == {POLICY_TARGET, WDL_TARGET, MLH_TARGET}
    for key in targets:
        assert targets[key] is batch[key], f"{key} must be passed through, not rebuilt"
    # Provenance and the surprise scalar are not targets: a head name typo
    # must not find a plausible tensor to train on.
    assert "ply" not in targets
    assert "policy_kl" not in targets
    assert "observation" not in targets


def test_targets_from_batch_skips_a_column_the_batch_does_not_carry() -> None:
    """Absence is left for the assembler to complain about, by head name."""
    batch = _batch()
    del batch[MLH_TARGET]
    assert set(targets_from_batch(batch)) == {POLICY_TARGET, WDL_TARGET}


# --------------------------------------------------------------------------
# surprise_weights_from_kl -- the stored-scalar path
# --------------------------------------------------------------------------


def test_surprise_weights_are_one_plus_alpha_times_the_stored_kl() -> None:
    """The golden: w = 1 + alpha * kl, elementwise, in the input's dtype."""
    kl = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    weights = surprise_weights_from_kl(kl, alpha=0.5, max_weight=5.0)
    assert weights.dtype == torch.float64
    assert weights.tolist() == pytest.approx([1.0, 1.5, 2.0], abs=1e-12)


def test_the_clip_binds_at_both_ends_of_the_stored_path() -> None:
    """A negative KL cannot down-weight, and one huge one cannot own a batch."""
    kl = torch.tensor([-3.0, -1e-9, 0.0, 100.0], dtype=torch.float64)
    weights = surprise_weights_from_kl(kl, alpha=2.0, max_weight=4.0)
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 4.0], abs=1e-12)


def test_alpha_zero_is_an_exact_off_switch() -> None:
    """Ones, and the recorded KL is not read numerically to produce them."""
    kl = torch.tensor([float("nan"), -5.0, 7.0], dtype=torch.float32)
    weights = surprise_weights_from_kl(kl, alpha=0.0, max_weight=5.0)
    assert torch.equal(weights, torch.ones(3, dtype=torch.float32))


def test_the_stored_path_agrees_with_the_prior_vector_path() -> None:
    """Same clip, same arithmetic -- only the source of the KL differs.

    ``KL([0,0,1] || uniform over 3) = ln 3``, so the two functions must agree
    to the last bit on a sample whose prior vector is available.
    """
    target = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    prior = torch.zeros((1, 3), dtype=torch.float64)
    from_vector = policy_surprise_weight(target, prior, alpha=0.5, max_weight=5.0)
    from_scalar = surprise_weights_from_kl(
        torch.tensor([math.log(3.0)], dtype=torch.float64), alpha=0.5, max_weight=5.0
    )
    assert from_scalar.item() == pytest.approx(from_vector.item(), abs=1e-12)


def test_surprise_weights_reject_a_configuration_that_cannot_clip() -> None:
    """Validated at the call, not discovered as a silently reweighted batch."""
    kl = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="alpha"):
        surprise_weights_from_kl(kl, alpha=-0.5, max_weight=5.0)
    with pytest.raises(ValueError, match="max_weight"):
        surprise_weights_from_kl(kl, alpha=1.0, max_weight=0.5)
    with pytest.raises(ValueError, match="1-D"):
        surprise_weights_from_kl(
            torch.zeros((2, 1), dtype=torch.float32), alpha=1.0, max_weight=5.0
        )


def test_stored_weights_reweight_the_batch_through_the_assembler() -> None:
    """The end of the missing wire: the weights actually reach a loss.

    Sample 1 carries all the surprise, so up-weighting it must move the policy
    loss toward *its* cross-entropy and away from the unweighted mean.
    """
    outputs = {
        POLICY_LOGITS: torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
    }
    targets = {
        POLICY_TARGET: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
    }
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=False))
    weights = surprise_weights_from_kl(
        torch.tensor([0.0, 3.0], dtype=torch.float64), alpha=1.0, max_weight=5.0
    )
    assert weights.tolist() == pytest.approx([1.0, 4.0], abs=1e-12)

    _, unweighted = assembler(outputs, targets)
    _, weighted = assembler(outputs, targets, sample_weight=weights)
    # Both samples are uniform-logit against a one-hot target, so every
    # per-sample CE is ln3 and the *value* cannot move -- what this pins is
    # that a (batch,) weight vector from the stored scalar is accepted and
    # normalised, which is the wire, not the arithmetic.
    assert weighted[HEAD_POLICY].item() == pytest.approx(
        unweighted[HEAD_POLICY].item(), abs=1e-12
    )
    # And with genuinely different per-sample losses it does move.
    outputs[POLICY_LOGITS] = torch.tensor(
        [[9.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    _, plain = assembler(outputs, targets)
    _, tilted = assembler(outputs, targets, sample_weight=weights)
    assert tilted[HEAD_POLICY].item() > plain[HEAD_POLICY].item()


# --------------------------------------------------------------------------
# The production recipe
# --------------------------------------------------------------------------


def test_p0a_recipe_is_pinned_field_by_field() -> None:
    """§2.3-2.4's recipe, so a defaults change cannot quietly become the run."""
    assert isinstance(P0A_RECIPE, LossConfig)
    assert P0A_RECIPE.policy_enabled is True
    assert P0A_RECIPE.policy_weight == 1.0
    assert P0A_RECIPE.soft_policy_enabled is True
    assert P0A_RECIPE.soft_policy_weight == 8.0
    assert P0A_RECIPE.soft_policy_exponent == 0.25
    assert P0A_RECIPE.wdl_enabled is True
    assert P0A_RECIPE.wdl_weight == 1.0
    assert P0A_RECIPE.mlh_enabled is False
    assert P0A_RECIPE.mlh_weight == 0.0
    assert P0A_RECIPE.surprise_alpha == 0.0
    assert P0A_RECIPE.surprise_max_weight == 5.0


def test_the_recipe_is_not_the_defaults() -> None:
    """The point of naming it: it differs from what an omission would give."""
    assert P0A_RECIPE != LossConfig()
    assert LossConfig().soft_policy_enabled is False


# --------------------------------------------------------------------------
# StepConfig
# --------------------------------------------------------------------------


def test_step_config_rejects_what_cannot_train() -> None:
    """Every documented bound, checked at construction."""
    with pytest.raises(ValueError, match="batch_size"):
        StepConfig(batch_size=0, lr=1e-3)
    with pytest.raises(ValueError, match="lr"):
        StepConfig(batch_size=2, lr=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        StepConfig(batch_size=2, lr=1e-3, weight_decay=-1.0)
    with pytest.raises(ValueError, match="surprise_alpha"):
        StepConfig(batch_size=2, lr=1e-3, surprise_alpha=-0.1)
    with pytest.raises(ValueError, match="surprise_max_weight"):
        StepConfig(batch_size=2, lr=1e-3, surprise_max_weight=0.5)
    with pytest.raises(ValueError, match="seed"):
        StepConfig(batch_size=2, lr=1e-3, seed=-1)


@pytest.mark.parametrize("lr", [0.0, -0.0, -1e-3, -1.0])
def test_zero_lr_is_refused(lr: float) -> None:
    """KOM-022 acceptance #47: a guard this repo must not relax.

    ``lr = 0.0`` raises **at construction**, before a single step runs. The
    docstring beside the guard gives KOM-022's own failure mode as its reason:
    *a zero learning rate is a run that trains nothing while looking exactly
    like one that does*.

    This acceptance exists because rev 3 of KOM-022's design built its
    non-learning fixture out of exactly that -- and the standing temptation at
    implementation time is to delete two lines here rather than build the
    published-checkpoint-series fixture rev 4 specifies (see
    ``tests/train/test_ema_unit.py::test_a_non_learning_trainer_does_not_advance_the_digest``
    and ``tests/campaign/behavior/kom022_rig.py``). The predicate is
    ``not lr > 0``, not ``lr <= 0``, so a NaN cannot slip through either.
    """
    with pytest.raises(ValueError, match="lr"):
        StepConfig(batch_size=2, lr=lr)


# --------------------------------------------------------------------------
# TrainStep
# --------------------------------------------------------------------------


def _step(
    net: KomugiNet,
    *,
    surprise_alpha: float = 0.0,
    fast_policy_weight: float = 1.0,
) -> TrainStep:
    """A :class:`TrainStep` on `net` running the production recipe."""
    config = StepConfig(
        batch_size=2,
        lr=5e-3,
        surprise_alpha=surprise_alpha,
        fast_policy_weight=fast_policy_weight,
    )
    return TrainStep(net, LossAssembler(P0A_RECIPE), config)


def test_one_step_reports_every_enabled_head_and_a_finite_total() -> None:
    """The metrics dict is the recipe's enabled heads plus the weighted total."""
    net = _tiny_net()
    step = _step(net)
    metrics = step.run(_batch())

    assert set(metrics) == {TOTAL_KEY, HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL}
    assert HEAD_MLH not in metrics, "moves-left is staged off in P0A_RECIPE"
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(math.isfinite(value) for value in metrics.values())
    # The total is the weighted sum of the unweighted per-head values.
    assert metrics[TOTAL_KEY] == pytest.approx(
        1.0 * metrics[HEAD_POLICY]
        + 8.0 * metrics[HEAD_SOFT_POLICY]
        + 1.0 * metrics[HEAD_WDL],
        rel=1e-5,
    )


def test_a_step_moves_the_parameters_and_lowers_the_same_batch_loss() -> None:
    """Descent smoke: the step trains, and the second run proves it did.

    The second ``run`` scores the same batch *after* the first update, so the
    comparison isolates one optimiser step. Gradients are cleared by ``run``
    itself, so the second step cannot be riding the first one's.
    """
    net = _tiny_net()
    step = _step(net)
    batch = _batch()
    before = {name: tensor.clone() for name, tensor in net.state_dict().items()}

    first = step.run(batch)
    second = step.run(batch)

    moved = {
        name
        for name, tensor in net.state_dict().items()
        if not torch.equal(tensor, before[name])
    }
    # Named weights, not a ``startswith`` over the head: every head's
    # BatchNorm running statistics move in ``train()`` mode whether or not it
    # received a gradient, so a prefix test would pass on a head the loss
    # never touched -- which is exactly what the last assertion checks.
    assert "trunk.stem.weight" in moved
    assert "policy.conv.weight" in moved
    assert "soft_policy.conv.weight" in moved
    assert "wdl.out_linear.weight" in moved
    assert "mlh.out_linear.weight" not in moved, "moves-left is off in P0A_RECIPE"
    assert second[TOTAL_KEY] < first[TOTAL_KEY]
    # No gradient is left hanging on the parameters after a step.
    assert all(parameter.grad is None for parameter in net.parameters())


def test_a_step_without_an_observation_says_so() -> None:
    """The one batch key the step spells itself is checked by name."""
    net = _tiny_net()
    batch = _batch()
    del batch["observation"]
    with pytest.raises(KeyError, match="observation"):
        _step(net).run(batch)


def test_surprise_weighting_is_off_unless_the_step_config_asks() -> None:
    """At alpha 0 a batch with no ``policy_kl`` column still trains."""
    net = _tiny_net()
    batch = _batch()
    del batch["policy_kl"]
    metrics = _step(net).run(batch)
    assert math.isfinite(metrics[TOTAL_KEY])


def test_surprise_weighting_on_without_the_stored_column_is_loud() -> None:
    """Alpha above zero and no recorded KL is an error, not a silent 1."""
    net = _tiny_net()
    batch = _batch()
    del batch["policy_kl"]
    step = _step(net, surprise_alpha=1.0)
    with pytest.raises(KeyError, match="policy_kl"):
        step.run(batch)


@pytest.mark.parametrize("fast_policy_weight", (0.0, 0.125, 1.0))
def test_the_search_mode_column_is_required_at_every_weight(
    fast_policy_weight: float,
) -> None:
    """Acceptance #8: a batch with no ``search_mode`` raises, at every arm.

    Parametrised over all three arms because the fast-row multiplier is passed
    **unconditionally**: a node that fired only at ``w != 1.0`` would pin the
    bypass rather than the seam, and arm C would be a different code path --
    exactly the objection KOM-036 §1.2 raises against the rejected row filter.

    The ``KeyError`` naming the column is the ``policy_kl`` idiom: a missing
    provenance column is a malformed batch, not a reason to guess a stratum.
    """
    net = _tiny_net()
    batch = _batch()
    del batch["search_mode"]
    step = _step(net, fast_policy_weight=fast_policy_weight)
    with pytest.raises(KeyError, match="search_mode"):
        step.run(batch)


def test_step_config_refuses_an_arm_outside_zero_to_one() -> None:
    """The knob is a down-weight, so above the full-row weight is not an arm."""
    for value in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="fast_policy_weight"):
            StepConfig(batch_size=2, lr=1e-3, fast_policy_weight=value)


@pytest.mark.parametrize("cap_row_value_weight", (0.0, 0.25, 1.0))
def test_the_termination_column_is_required_at_every_weight(
    cap_row_value_weight: float,
) -> None:
    """KOM-160: a batch with no ``termination`` raises, at every arm.

    Parametrised over all three arms for the reason the ``search_mode`` node
    above is: the cap-row multiplier is passed **unconditionally**, so a node
    that fired only at ``w != 1.0`` would pin a bypass rather than the seam and
    would leave ``1.0`` on a different code path.

    ``1.0`` is in the sweep specifically. That is the default, so it is the
    value every campaign that has ever run used -- and an implementation that
    required the column only when the knob was engaged would be green on a
    ``0.25``-only node while shipping two trainers.
    """
    net = _tiny_net()
    batch = _batch()
    del batch["termination"]
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=2, lr=5e-3, cap_row_value_weight=cap_row_value_weight
        ),
    )
    with pytest.raises(KeyError, match="termination"):
        step.run(batch)


def test_step_config_refuses_a_cap_row_weight_outside_zero_to_one() -> None:
    """The knob down-weights draw rows; above 1 is a different experiment.

    NaN is in the sweep because the predicate is ``not 0.0 <= w <= 1.0``: the
    ``<``/``>`` spelling admits it, and a NaN multiplier is a WDL loss that is
    NaN from the first batch with nothing naming the field that caused it.
    """
    for value in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="cap_row_value_weight"):
            StepConfig(batch_size=2, lr=1e-3, cap_row_value_weight=value)


@pytest.mark.parametrize("mlh_cap_row_weight", (0.0, 0.25, 1.0))
@pytest.mark.parametrize("column", ("ply", "mlh_target"))
def test_the_moves_left_columns_are_required_at_every_weight(
    column: str, mlh_cap_row_weight: float
) -> None:
    """KOM-159 D4: no ``ply``/``mlh_target``, no step -- at every arm.

    The third pair of columns ``TrainStep.run`` requires unconditionally, and
    for the reason the two before them are required: the moves-left cap-row
    multiplier is passed at every ``mlh_cap_row_weight``, so ``1.0`` is the
    pre-KOM-159 arithmetic *through* the seam rather than around it. A node
    that fired only at ``0.0`` would pin a bypass.

    ``1.0`` is in the sweep for the sharper reason: it is not the default here
    (``0.0``, the censoring arm, is), so an implementation that required the
    columns only when censoring was engaged would ship two trainers and be
    green on a ``0.0``-only node.

    Both columns are swept because they answer different halves of the
    predicate -- where in the game the row sits, and how much of it was left --
    and an implementation reading only one would be green on the other's node.
    """
    net = _tiny_net()
    batch = _batch()
    del batch[column]
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=2, lr=5e-3, mlh_cap_row_weight=mlh_cap_row_weight
        ),
    )
    with pytest.raises(KeyError, match=column):
        step.run(batch)


def test_step_config_refuses_the_moves_left_knobs_it_cannot_use() -> None:
    """Every KOM-159 bound, checked at construction rather than at step 1.

    NaN is in both float sweeps because the predicates are ``not 0 <= w <= 1``:
    the ``<``/``>`` spelling admits it, and the failure would then be a
    moves-left head whose loss is NaN from the first batch, or -- for the
    lambda, which the step never reads -- a telemetry write that takes the
    campaign down at the *end* of an iteration that had already trained.

    ``cap_plies`` is refused below 1 because a cap of zero makes ``total >=
    cap`` true for every row, which censors a whole batch's moves-left head
    into a zero-mass omission: a silently missing curve rather than an error.
    """
    for value in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="mlh_cap_row_weight"):
            StepConfig(batch_size=2, lr=1e-3, mlh_cap_row_weight=value)
        with pytest.raises(ValueError, match="mlh_utility_lambda"):
            StepConfig(batch_size=2, lr=1e-3, mlh_utility_lambda=value)
    for cap in (0, -1):
        with pytest.raises(ValueError, match="cap_plies"):
            StepConfig(batch_size=2, lr=1e-3, cap_plies=cap)


def test_the_default_cap_is_the_engines_own_turn_cap() -> None:
    """``DEFAULT_CAP_PLIES`` is a literal, and this is what keeps it honest.

    The trainer may not import the engine (the cycle
    :mod:`komugi.train.wiring`'s docstring describes), so the fallback cap is
    stated rather than derived -- and a stated constant is one KOM-165's next
    successor would silently leave behind. It bumped the cap from 256 to 512
    once already; the censoring predicate evaluated against a stale cap would
    quietly stop finding cap rows, which looks exactly like an arm that is off.

    The *campaign* never relies on the default: it passes
    ``Game.max_turns()`` explicitly, which is pinned by
    ``test_the_campaign_passes_the_moves_left_arm_and_the_engines_cap_to_the_step``
    in ``tests/campaign/unit/test_loop_unit.py`` -- reading ``run_campaign``'s
    own AST, because the equality asserted here is exactly what makes a dropped
    keyword invisible to anything executed.
    """
    assert DEFAULT_CAP_PLIES == MAX_TURNS, (
        f"the trainer's fallback cap is {DEFAULT_CAP_PLIES} and the engine's "
        f"is {MAX_TURNS}: a shard written under the engine's cap would be "
        f"censored against the trainer's"
    )
    assert StepConfig(batch_size=2, lr=1e-3).cap_plies == MAX_TURNS


def test_the_moves_left_arm_reaches_the_reported_loss_and_leaves_the_rest_alone() -> None:
    """KOM-159's wire, end to end through the production ``run``, in one node.

    One decisive row and one row of a game that reached the cap, scored twice
    under a recipe with the head on. The moves-left value must move with the
    arm -- it is a weighted mean over two rows whose per-row squared errors
    differ -- while policy, soft policy and WDL must not: the censoring reaches
    ``HEAD_MLH`` and nothing else.

    The arithmetic itself lives in
    ``tests/train/behavior/test_mlh_row_weights.py``; what this node pins is
    that ``StepConfig`` -> ``mlh_row_weights`` -> the assembler is connected at
    all, beside the other ``StepConfig`` knobs' own nodes.
    """
    cap_row = _sample(1, termination=TERMINATION_DRAW)
    # Ply 12 of a game that ran to the cap: the recorder's own arithmetic, so
    # `round(ply + mlh_target * cap)` recovers exactly `cap`.
    cap_row["ply"] = torch.tensor(12, dtype=torch.int64)
    cap_row["mlh_target"] = torch.tensor(
        (MAX_TURNS - 12) / MAX_TURNS, dtype=torch.float32
    )
    batch = stack_samples([_sample(0), cap_row])

    def run(weight: float) -> dict[str, float]:
        """One step at moves-left arm `weight`, on its own net."""
        return TrainStep(
            _tiny_net(),
            LossAssembler(MLH_RECIPE),
            StepConfig(
                batch_size=2,
                lr=5e-3,
                mlh_cap_row_weight=weight,
                cap_plies=MAX_TURNS,
            ),
        ).run(batch)

    censored = run(0.0)
    plain = run(1.0)

    assert plain[HEAD_MLH] != pytest.approx(censored[HEAD_MLH], abs=1e-9), (
        "the arm did not change the reported moves-left loss, so "
        "mlh_cap_row_weight is not reaching the assembler at all"
    )
    for head in (HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL):
        assert censored[head] == pytest.approx(plain[head], abs=1e-12), (
            f"the moves-left arm moved the {head} head; it keys HEAD_MLH alone"
        )


def test_the_arm_reaches_the_reported_wdl_loss_and_leaves_policy_alone() -> None:
    """The wire, end to end through the production ``run``, in one node.

    One decisive row and one draw row, scored twice. The WDL value must move
    with the arm -- it is a weighted mean over two rows whose per-row losses
    differ -- while the policy and soft-policy values must not: the cap weight
    reaches ``HEAD_WDL`` and nothing else (design review F10). Two fresh nets
    from the same seed rather than one stepped twice, because ``run`` updates
    both the parameters and BatchNorm's running estimates.

    The arithmetic itself lives in
    ``tests/train/behavior/test_cap_row_value_weight.py``; what this node pins
    is that ``StepConfig`` -> ``cap_row_weights`` -> the assembler is connected
    at all, next to the other ``StepConfig`` knobs' own nodes.
    """
    batch = stack_samples(
        [_sample(0), _sample(1, termination=TERMINATION_DRAW)]
    )
    plain = TrainStep(
        _tiny_net(), LossAssembler(P0A_RECIPE), StepConfig(batch_size=2, lr=5e-3)
    ).run(batch)
    capped = TrainStep(
        _tiny_net(),
        LossAssembler(P0A_RECIPE),
        StepConfig(batch_size=2, lr=5e-3, cap_row_value_weight=0.25),
    ).run(batch)

    assert plain[HEAD_WDL] != pytest.approx(capped[HEAD_WDL], abs=1e-9), (
        "the arm did not change the reported WDL loss, so cap_row_value_weight "
        "is not reaching the assembler at all"
    )
    assert capped[HEAD_POLICY] == pytest.approx(plain[HEAD_POLICY], abs=1e-12)
    assert capped[HEAD_SOFT_POLICY] == pytest.approx(
        plain[HEAD_SOFT_POLICY], abs=1e-12
    ), "the cap weight reached a policy head; it keys HEAD_WDL alone"


def test_surprise_weighting_on_changes_the_reported_loss() -> None:
    """Row 1 carries KL 0.5 and row 0 carries none, so the batch tilts."""
    plain = _step(_tiny_net()).run(_batch())
    weighted = _step(_tiny_net(), surprise_alpha=2.0).run(_batch())
    assert plain[TOTAL_KEY] != pytest.approx(weighted[TOTAL_KEY], abs=1e-9)


def test_the_optimiser_is_an_internally_built_adamw() -> None:
    """P0a's narrowing, pinned: no optimiser enters from outside."""
    net = _tiny_net()
    step = _step(net)
    assert isinstance(step.optimizer, torch.optim.AdamW)
    group = step.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(5e-3)
    assert group["weight_decay"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The WDL index pin -- KOM-005a review F5b
# --------------------------------------------------------------------------


def _wdl_loss(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return the assembler's WDL head loss for hand-supplied logits."""
    assembler = LossAssembler(
        LossConfig(policy_enabled=False, wdl_enabled=True, wdl_weight=1.0)
    )
    _, per_head = assembler(
        {WDL_LOGITS: logits},
        {WDL_TARGET: target},
        # The surprise path wants a policy target it has no use for here, so
        # the weights are supplied directly: this test is about one head.
        sample_weight=torch.ones(logits.shape[0], dtype=logits.dtype),
    )
    return float(per_head[HEAD_WDL].item())


def test_a_win_target_is_minimised_by_logits_favouring_index_zero() -> None:
    """(win, draw, loss) survives net -> wiring -> loss, mechanically.

    ``komugi.net.heads.WDLHead``'s docstring fixes index 0 as *win*; the
    assembler's three-way cross-entropy just reads column 0 of whatever it is
    handed. If the two ever disagree the net learns to call losses wins and
    nothing raises -- so the ordering is checked by which prediction the loss
    actually rewards.
    """
    win_target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    favour_win = _wdl_loss(torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64), win_target)
    favour_draw = _wdl_loss(
        torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64), win_target
    )
    favour_loss = _wdl_loss(
        torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64), win_target
    )
    assert favour_win < favour_draw
    assert favour_win < favour_loss
    # The closed form, so this is a golden and not merely an ordering:
    # -log(softmax([3,0,0])[0]) = log(1 + 2*exp(-3)).
    assert favour_win == pytest.approx(math.log(1.0 + 2.0 * math.exp(-3.0)), abs=1e-12)


def test_the_index_pin_holds_through_a_real_net_forward() -> None:
    """The same claim, with the net's own logits as the baseline.

    ``outputs_from_net`` must hand the WDL column through unpermuted, so
    adding mass at index 0 of a real forward's logits has to *reduce* a
    win-target loss while adding it at index 2 raises one.
    """
    net = _tiny_net()
    out = net(torch.rand((2, _PLANES, 9, 9)))
    mapped = outputs_from_net(out)
    assert mapped[WDL_LOGITS] is out.wdl_logits
    assert mapped[WDL_LOGITS].shape == (2, 3)

    baseline_logits = out.wdl_logits.detach().to(torch.float64)
    win_target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    baseline = _wdl_loss(baseline_logits, win_target)

    bump_win = baseline_logits + torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    bump_loss = baseline_logits + torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    assert _wdl_loss(bump_win, win_target) < baseline
    assert _wdl_loss(bump_loss, win_target) > baseline


# --------------------------------------------------------------------------
# training_iteration
# --------------------------------------------------------------------------


class _ListSource:
    """An in-memory iteration: the smallest thing satisfying ``SampleSource``."""

    def __init__(self, size: int) -> None:
        self._samples = [_sample(row) for row in range(size)]

    def __len__(self) -> int:
        """Return the number of samples held."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample at `index`."""
        return self._samples[index]


def test_training_iteration_runs_every_step_and_returns_their_metrics() -> None:
    """One dict per step, in order, each the step's own report."""
    net = _tiny_net()
    step = _step(net)
    history = training_iteration(
        net, _ListSource(6), step, steps=3, rng_seed=11
    )
    assert len(history) == 3
    assert all(set(entry) == {TOTAL_KEY, HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL} for entry in history)
    assert all(math.isfinite(entry[TOTAL_KEY]) for entry in history)


def _run_iteration(source: _ListSource, rng_seed: int) -> list[dict[str, float]]:
    """Train a freshly built (deterministic) net over `source` at `rng_seed`."""
    net = _tiny_net(seed=0)
    return training_iteration(net, source, _step(net), steps=2, rng_seed=rng_seed)


def test_the_same_seed_draws_the_same_batches() -> None:
    """Determinism of the sampling stream, observed through the losses.

    Net construction is a pure function of its config and AdamW's update is a
    closed form, so two runs at one seed must agree step for step -- to the
    bit, not approximately. A different ``rng_seed`` draws different rows and
    lands somewhere else.
    """
    source = _ListSource(32)
    first = _run_iteration(source, rng_seed=5)
    second = _run_iteration(source, rng_seed=5)
    other = _run_iteration(source, rng_seed=6)

    assert [entry[TOTAL_KEY] for entry in first] == [
        entry[TOTAL_KEY] for entry in second
    ]
    assert other[0][TOTAL_KEY] != pytest.approx(first[0][TOTAL_KEY], abs=1e-9)


def test_an_empty_source_and_a_zero_step_count_are_refused() -> None:
    """Both are runs that look like training and are not."""
    net = _tiny_net()
    with pytest.raises(ValueError, match="steps"):
        training_iteration(net, _ListSource(4), _step(net), steps=0, rng_seed=1)
    with pytest.raises(ValueError, match="no samples"):
        training_iteration(net, _ListSource(0), _step(net), steps=1, rng_seed=1)


def test_the_iteration_checkpoints_and_journals_what_it_did(tmp_path: Path) -> None:
    """Both side effects land through the real APIs, and read back."""
    net = _tiny_net()
    step = _step(net)
    journal = tmp_path / "telemetry.jsonl"
    history = training_iteration(
        net,
        _ListSource(5),
        step,
        steps=2,
        rng_seed=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        journal_path=str(journal),
        iteration=4,
    )

    checkpoint = load_checkpoint(tmp_path / "ckpt")
    assert checkpoint is not None
    assert checkpoint.step == 2
    assert checkpoint.meta["iteration"] == 4
    assert checkpoint.meta["rng_seed"] == 3
    assert checkpoint.meta["loss_total"] == pytest.approx(history[-1][TOTAL_KEY])
    # The optimiser's moment estimates travel with the weights, or a resume
    # restarts Adam cold.
    assert checkpoint.optim_state["state"]
    assert set(checkpoint.model_state) == set(net.state_dict())

    read = read_records(journal, IterationMetrics)
    assert read.ok
    assert len(read.records) == 1
    record = read.records[0]
    assert record.iteration == 4
    # The honest mapping: this call played no games, so the game-shaped fields
    # are the schema's floor and the real numbers live in extras.
    assert record.games == 0
    assert record.draw_rate == 0.0
    assert record.extras["steps"] == 2.0
    assert record.extras["batch_size"] == 2.0
    assert record.extras["loss_total"] == pytest.approx(history[-1][TOTAL_KEY])
    assert record.extras["loss_policy"] == pytest.approx(history[-1][HEAD_POLICY])


def test_the_iteration_journals_and_stamps_the_cap_row_value_weight(
    tmp_path: Path,
) -> None:
    """KOM-160: the arm is in the journal **and** in the checkpoint meta.

    Both, not either. ``fast_policy_weight`` is checkpoint-only because it
    leaves every reported number's definition alone; this one does not. Under
    an arm below 1.0 the journalled ``loss_wdl`` is the cap-**weighted** mean,
    so a reader comparing two campaigns' WDL curves is comparing two different
    statistics unless the arm travels in the same append-only file the curve is
    in. A checkpoint directory can be pruned, copied or lost; the journal is
    the audit path the design names (§2, R9), and it is the one that would be
    silently missing if only the checkpoint were stamped.

    ``0.25`` rather than a round number so the assertion cannot pass against a
    hard-coded default, and the same value is asserted in both places so a
    stamp reading the wrong config field is red here too.
    """
    net = _tiny_net()
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(batch_size=2, lr=5e-3, cap_row_value_weight=0.25),
    )
    journal = tmp_path / "telemetry.jsonl"
    training_iteration(
        net,
        _ListSource(5),
        step,
        steps=2,
        rng_seed=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        journal_path=str(journal),
        iteration=1,
    )

    checkpoint = load_checkpoint(tmp_path / "ckpt")
    assert checkpoint is not None
    assert checkpoint.meta["cap_row_value_weight"] == 0.25, (
        "the trainer checkpoint does not name the arm that produced these "
        "weights; a served net could not be attributed to 4A or 4D after"
    )

    read = read_records(journal, IterationMetrics)
    assert read.ok
    extras = read.records[0].extras
    assert extras["cap_row_value_weight"] == 0.25, (
        "the journal does not carry the arm, so its `loss_wdl` series cannot "
        "be told from an unweighted one by anything reading the file"
    )
    # The pre-existing contract is untouched: this is a key ADDITION.
    assert extras["steps"] == 2.0
    assert extras["batch_size"] == 2.0


def test_the_iteration_journals_and_stamps_the_moves_left_knobs(
    tmp_path: Path,
) -> None:
    """KOM-159: the censoring arm and its cap -- and NOT the search-side lambda.

    Two keys, in the journal for two related reasons.

    ``mlh_cap_row_weight`` redefines ``loss_mlh`` exactly as
    ``cap_row_value_weight`` redefines ``loss_wdl``: below 1.0 the reported
    value is a mean over the *uncensored* rows only, which is not like-for-like
    with another campaign's. ``cap_plies`` rides beside it because the
    censoring predicate is meaningless without the cap it was evaluated
    against -- a series straddling a cap bump would be two statistics under one
    name.

    **``mlh_utility_lambda`` is absent, and the absence is the assertion**
    (KOM-171, KOM-159a finisher review F5). The switch still has to be readable
    off the journal (design review F10), but the campaign boundary is the single
    writer of that key: it is a property of the *games*, and a key stamped by
    both the trainer's record and the loop's is a key whose value a reader
    cannot attribute to either -- which is the trainer/boundary
    namespace-disjointness invariant
    ``tests/campaign/unit/test_loop_unit.py`` pins. ``StepConfig`` still
    carries the field (the loop hands it over, and it is range-validated
    there), so this node fails if a future edit re-adds the stamp.

    The arm is stamped on the checkpoint too, beside the other two arms, so a
    served net can be attributed to the recipe *and* the row weight that
    produced it; the lambda is not on the checkpoint either, for the same
    reason. Values are deliberately not round numbers, so no assertion here can
    pass against a hard-coded default.
    """
    net = _tiny_net()
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=2,
            lr=5e-3,
            mlh_cap_row_weight=0.25,
            cap_plies=384,
            mlh_utility_lambda=0.05,
        ),
    )
    journal = tmp_path / "telemetry.jsonl"
    training_iteration(
        net,
        _ListSource(5),
        step,
        steps=2,
        rng_seed=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        journal_path=str(journal),
        iteration=1,
    )

    checkpoint = load_checkpoint(tmp_path / "ckpt")
    assert checkpoint is not None
    assert checkpoint.meta["mlh_cap_row_weight"] == 0.25, (
        "the trainer checkpoint does not name the moves-left arm that "
        "produced these weights"
    )
    assert "mlh_utility_lambda" not in checkpoint.meta, (
        "the checkpoint META stamps the search-side lambda; it names the arms "
        "that produced these weights, and the search-side utility is a "
        "property of the games (KOM-171)"
    )

    read = read_records(journal, IterationMetrics)
    assert read.ok
    extras = read.records[0].extras
    assert extras["mlh_cap_row_weight"] == 0.25
    assert extras["cap_plies"] == 384.0
    assert "mlh_utility_lambda" not in extras, (
        f"the trainer's record carries mlh_utility_lambda="
        f"{extras.get('mlh_utility_lambda')!r}: the campaign boundary is the "
        f"single writer of that key (KOM-171), and two writers of one key make "
        f"the trainer/boundary namespaces overlap"
    )
    # The pre-existing contract is untouched: these are key ADDITIONS.
    assert extras["steps"] == 2.0
    assert extras["batch_size"] == 2.0
    assert extras["cap_row_value_weight"] == 1.0


def test_the_iteration_extras_carry_the_statistic_the_staging_gate_reads(
    tmp_path: Path,
) -> None:
    """The decile means and the step dispersion reach the journal (KOM-005b #30).

    Every staged head's gate has a condition 1 that reads two journalled
    numbers -- ``loss_<head>_first_decile`` and ``loss_<head>_last_decile`` --
    and compares them against a threshold pre-registered before the data
    exists. Short-term value's gate reads a third, ``loss_wdl_step_std``.
    **None of the three is recoverable from what ``extras`` carried before**:
    ``training_iteration`` accumulates the per-step dicts and keeps
    ``history[-1]`` only, so the journal held one value per loss scalar, and
    one value per iteration has no dispersion and no first decile. Without this
    node the gates' inputs are silent: nothing else in the suite reads them, so
    a refactor of ``training_iteration`` could drop them and every other test
    would stay green while the whole staging ladder became unopenable.

    The values are asserted, not merely the keys, against the same ``history``
    the call returns -- a key that is present but wrong is exactly as fatal to
    the gate as a key that is missing.

    **``steps`` is 25, and the 25 is the assertion** (impl review F7, mutant
    M5). At ``steps=10`` the decile is ``ceil(10 / 10) == 1``, so
    ``history[:1]`` and ``history[-1:]`` are the first and last *steps* --
    which makes the node bit-identical to one running against a hard-coded
    ``decile = 1``, and leaves the statistic's *definition* unpinned while its
    plumbing is fully asserted. A gate reading step 0 against step 9 instead of
    two decile means is precisely the noise sensitivity the design rejected
    monotonicity for. At 25 the decile is ``ceil(25 / 10) == 3``, so each slice
    is a genuine three-step mean and the two readings differ.
    """
    net = _tiny_net()
    step = _step(net)
    journal = tmp_path / "telemetry.jsonl"
    steps = 25
    history = training_iteration(
        net,
        _ListSource(8),
        step,
        steps=steps,
        rng_seed=11,
        journal_path=str(journal),
        iteration=2,
    )
    assert len(history) == steps

    read = read_records(journal, IterationMetrics)
    assert read.ok
    extras = read.records[0].extras

    # The pre-existing contract is untouched: this is a key ADDITION.
    assert extras["steps"] == float(steps)
    assert extras["batch_size"] == 2.0
    assert extras["loss_total"] == pytest.approx(history[-1][TOTAL_KEY])

    decile = math.ceil(steps / 10)
    assert decile == 3, (
        "the decile must span more than one step, or this node cannot tell "
        "`ceil(len(history) / 10)` from a hard-coded 1"
    )
    for name in history[-1]:
        series = [entry[name] for entry in history]
        first = statistics.fmean(series[:decile])
        last = statistics.fmean(series[-decile:])
        spread = statistics.pstdev(series)
        assert extras[f"loss_{name}_first_decile"] == pytest.approx(first)
        assert extras[f"loss_{name}_last_decile"] == pytest.approx(last)
        assert extras[f"loss_{name}_step_std"] == pytest.approx(spread)

    # The three-step means are not the single steps they would collapse to
    # under `decile = 1`, so a hard-coded decile is red on VALUE and not only
    # on the arithmetic above. Asserted on the total, which the dispersion
    # clause below independently shows is not a constant series.
    totals = [entry[TOTAL_KEY] for entry in history]
    assert extras[f"loss_{TOTAL_KEY}_first_decile"] != pytest.approx(
        totals[0]
    ) or extras[f"loss_{TOTAL_KEY}_last_decile"] != pytest.approx(totals[-1]), (
        "the journalled decile means coincide with the first and last steps, "
        "so this node cannot distinguish `ceil(len(history) / 10)` from 1"
    )

    # Written for EVERY loss scalar, not just one head: which head is in
    # bring-up is a property of the campaign, not of this function.
    for name in (TOTAL_KEY, HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL):
        assert f"loss_{name}_first_decile" in extras
        assert f"loss_{name}_last_decile" in extras
        assert f"loss_{name}_step_std" in extras

    # A ten-step run over a source of eight rows genuinely moves, so the
    # dispersion is not vacuously zero and the node can distinguish a real
    # standard deviation from a hard-coded 0.0.
    assert extras[f"loss_{TOTAL_KEY}_step_std"] > 0.0


def test_nothing_is_written_when_neither_sink_is_given(tmp_path: Path) -> None:
    """The default is a pure function of the net: no directory, no journal."""
    net = _tiny_net()
    training_iteration(net, _ListSource(4), _step(net), steps=1, rng_seed=2)
    assert list(tmp_path.iterdir()) == []


def test_training_iteration_rejects_a_net_the_step_does_not_train() -> None:
    """Identity, not equality: the wrong net checkpoints untrained weights.

    Every gradient update lands on ``step.net`` while the checkpoint records
    ``net`` -- pass a different object and the run trains one and persists the
    other, silently, surfacing only at resume (wave-10 review F1).
    """
    trained = _tiny_net(seed=3)
    other = _tiny_net(seed=3)  # equal weights, DIFFERENT object
    step = TrainStep(
        trained, LossAssembler(P0A_RECIPE), StepConfig(batch_size=2, lr=1e-3)
    )
    with pytest.raises(ValueError, match="not the one its"):
        training_iteration(other, [_sample(0), _sample(1)], step, steps=1, rng_seed=0)


def test_stale_gradients_outside_run_do_not_steer_the_update() -> None:
    """Grads accumulated between steps must not change the update direction.

    A gradient probe (or a net handed over with grads attached) previously
    added into the next ``backward()`` -- the reported loss stayed correct
    while the update direction silently shifted (wave-10 review F2). Two
    identical nets, same batch; one gets poisoned first; the post-step
    parameters must be bit-identical.
    """
    clean = _tiny_net(seed=11)
    poisoned = _tiny_net(seed=11)
    batch = _batch()

    # Poison: an out-of-band backward over the net's parameters.
    stray = torch.stack([p.sum() for p in poisoned.parameters()]).sum()
    stray.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
    assert any(p.grad is not None for p in poisoned.parameters())

    config = StepConfig(batch_size=2, lr=1e-3)
    TrainStep(clean, LossAssembler(P0A_RECIPE), config).run(batch)
    TrainStep(poisoned, LossAssembler(P0A_RECIPE), config).run(batch)

    for (name, a), (_, b) in zip(
        clean.named_parameters(), poisoned.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), (
            f"{name} diverged: the stale out-of-band gradient steered the update"
        )


# --------------------------------------------------------------------------
# Numerical integrity (review TS-01, 2026-09-12)
#
# The witness these three reproduce: an all-zero or NaN policy row produced a
# FINITE reported total (`0.9119...`) while `backward()` wrote NaN through the
# policy trunk, `training_iteration` published the poisoned checkpoint and the
# journal recorded an ordinary iteration. Each test here asserts the corrected
# behaviour -- a refusal that names the offender and leaves nothing behind.
# --------------------------------------------------------------------------


def _poisoned_sample(row: int, *, value: float) -> dict[str, torch.Tensor]:
    """A :func:`_sample` whose ``policy_target`` is the constant `value`.

    ``0.0`` is the all-zero row (search never labelled it); ``nan`` is the row
    a diverged search writes. Both are distributions in name only.
    """
    sample = _sample(row)
    sample[POLICY_TARGET] = torch.full_like(sample[POLICY_TARGET], value)
    return sample


class _PoisonedSource:
    """A :class:`_ListSource` whose every ``policy_target`` is all zeros.

    Built the same way rather than by wrapping, so what
    :func:`training_iteration` is handed here is the same ``SampleSource``
    shape the healthy nodes above hand it.
    """

    def __init__(self, size: int) -> None:
        self._samples = [_poisoned_sample(row, value=0.0) for row in range(size)]

    def __len__(self) -> int:
        """Return the number of samples held."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample at `index`."""
        return self._samples[index]


def _parameters_of(net: KomugiNet) -> dict[str, torch.Tensor]:
    """Snapshot every parameter by name, detached and cloned."""
    return {name: p.detach().clone() for name, p in net.named_parameters()}


def _assert_unchanged(net: KomugiNet, before: dict[str, torch.Tensor]) -> None:
    """Assert every parameter is bit-identical to its snapshot."""
    for name, parameter in net.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), (
            f"{name} moved: the refused step applied itself anyway"
        )


def test_a_zero_policy_row_is_refused_before_the_update(tmp_path: Path) -> None:
    """An all-zero policy row stops the step, by row index, with nothing written.

    The witness's exact input: one row of a real batch whose ``policy_target``
    is entirely zero. Before TS-01 it trained -- the cross-entropy over an empty
    target is a finite ``0.0`` and ``soft_policy_target`` divided ``0`` by
    ``0`` -- so the three assertions are the three things that must NOT have
    happened: no parameter moved, no checkpoint was published, no journal
    record was written.
    """
    net = _tiny_net()
    step = _step(net)
    before = _parameters_of(net)
    batch = stack_samples([_sample(0), _poisoned_sample(1, value=0.0)])

    with pytest.raises(ValueError, match="row 1") as excinfo:
        step.run(batch)
    assert "zero mass" in str(excinfo.value)
    _assert_unchanged(net, before)
    assert all(p.grad is None for p in net.parameters()), (
        "the refusal happened before any backward, so there is nothing to clear"
    )

    # The same refusal through the iteration that owns the two side effects.
    journal = tmp_path / "telemetry.jsonl"
    checkpoints = tmp_path / "ckpt"
    with pytest.raises(ValueError, match="policy_target"):
        training_iteration(
            net,
            _PoisonedSource(4),
            step,
            steps=1,
            rng_seed=3,
            checkpoint_dir=str(checkpoints),
            journal_path=str(journal),
            iteration=7,
        )
    assert load_checkpoint(checkpoints) is None, "a refused iteration checkpointed"
    assert not journal.exists(), "a refused iteration journalled a normal step"


def test_a_nan_policy_row_is_refused_naming_the_row() -> None:
    """The other half of the witness: a NaN row, named by index, not swallowed.

    ``_distribution_cross_entropy``'s ``target > 0`` mask is False at NaN, so
    the NaN row contributed exactly zero to the reported loss while poisoning
    the soft-policy target's renormalisation. The refusal names the row and the
    action, because "somewhere in this batch" is not a diagnosis.
    """
    net = _tiny_net()
    step = _step(net)
    before = _parameters_of(net)
    batch = stack_samples(
        [_sample(0), _sample(1), _poisoned_sample(2, value=float("nan"))]
    )

    with pytest.raises(ValueError, match="row 2") as excinfo:
        step.run(batch)
    assert "not finite" in str(excinfo.value)
    _assert_unchanged(net, before)


def test_a_nan_gradient_cannot_reach_optimizer_step() -> None:
    """A finite loss whose gradient is NaN is refused between backward and step.

    The target-side guards cannot catch this: the batch is healthy and the loss
    is finite, and the NaN appears only in the backward -- an overflowing
    activation, a division whose denominator went to zero, a hook like this
    one. What makes it fatal is ``optimizer.step()``: after it, the NaN is in
    the parameters AND in AdamW's moment estimates, and no later batch removes
    it. So the check sits before the update, and the update does not happen.
    """
    net = _tiny_net()
    step = _step(net)
    before = _parameters_of(net)
    poisoned_name, poisoned = next(iter(net.named_parameters()))
    # A tensor hook is the smallest honest injection: the batch is healthy and
    # the forward is the real one, and only the gradient of this one parameter
    # is replaced. Bound through a cast because ``Tensor.register_hook``'s
    # annotation has changed shape across torch releases (unannotated, then
    # returning ``Tensor``, then ``Tensor | None``) -- the same gap
    # ``Tensor.backward`` has all over this suite, spelled so that the pin does
    # not depend on which release is installed. The hook itself is fully typed.
    register = cast(
        "Callable[[Callable[[torch.Tensor], torch.Tensor]], object]",
        poisoned.register_hook,
    )
    register(lambda gradient: gradient * float("nan"))

    with pytest.raises(NonFiniteUpdateError) as excinfo:
        step.run(_batch())

    detail = str(excinfo.value)
    reported = float(detail.split("loss=", 1)[1].split(",", 1)[0])
    assert math.isfinite(reported), (
        "the precondition of this test is a FINITE loss: a non-finite one is "
        "the easy case and is caught by the same guard"
    )
    assert poisoned_name in detail, "the refusal must name the parameter"
    _assert_unchanged(net, before)
    assert step.optimizer.state_dict()["state"] == {}, (
        "AdamW built moment estimates: optimizer.step() ran on a NaN gradient"
    )
    assert all(p.grad is None for p in net.parameters()), (
        "the NaN gradient is still attached for the next step() to apply"
    )
