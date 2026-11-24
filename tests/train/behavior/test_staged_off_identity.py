"""OFF is bit-identical training, not merely close (KOM-005b D2).

The staged heads land in the net *now* -- all four of them, forward-evaluated
every step -- because the net freeze makes adding one later an act that
invalidates every checkpoint the production campaign has written. The entire
argument for doing that rests on a claim that has to be executed rather than
reasoned about: with the four losses off, training is **bit-identical** to what
the same code produced before they existed.

Two commands carry D2 and neither substitutes for the other.

``tests/train/behavior/test_golden_step.py`` is the absolute half: it fixes a
batch, a net and a recipe and asserts the losses and the AdamW update against
hand-derived constants. Those numbers must not move, and they are the only
assertion in the tree that can see a perturbation which hits *both* arms of a
two-arm test equally -- a two-arm comparison written under the post-change code
is structurally blind to that.

This module is the relative half, and it asks the question the golden cannot:
can a staged head's *parameter values* leak into a P0a parameter? It answers by
running the same seeded training twice, identical in everything except the
staged heads' initial weights, and requiring every P0a parameter to come out
``torch.equal``. A staged head that is accidentally summed into the total, that
shares a tensor with a P0a head, or whose BatchNorm writes into a shared buffer
is red here and green in the golden.
"""

from __future__ import annotations

import math
from typing import Final

import torch

from komugi.campaign.factory import SMALL_CONFIG
from komugi.modes import MODE_FULL
from komugi.net import KomugiNet
from komugi.train.losses import LossAssembler
from komugi.train.step import StepConfig, TrainStep
from komugi.train.wiring import P0B_RECIPE, outputs_from_net, targets_from_batch

_BATCH: Final[int] = 4
_STEPS: Final[int] = 3

_P0A_MODULES: Final[tuple[str, ...]] = (
    "trunk",
    "policy",
    "soft_policy",
    "wdl",
    "mlh",
)
"""The parameter sets that existed before KOM-005b and must not move."""

_STAGED_MODULES: Final[tuple[str, ...]] = (
    "uncertainty",
    "ownership",
    "short_term_value",
    "draft_outcome",
)
"""The four that arrive with it, off."""

_P0A_METRIC_KEYS: Final[set[str]] = {"total", "policy", "soft_policy", "wdl"}
"""Exactly what ``TrainStep.run`` reported before the staged heads existed."""


def _batch() -> dict[str, torch.Tensor]:
    """One fixed toy batch at the seam's real shapes.

    ``SMALL_CONFIG`` shrinks the trunk and keeps ``in_planes`` and
    ``num_actions`` at the production values, so this batch is the shape a
    real worker emits.
    """
    generator = torch.Generator().manual_seed(50_505)
    actions = SMALL_CONFIG.heads.num_actions
    planes = SMALL_CONFIG.trunk.in_planes

    policy_target = torch.zeros(_BATCH, actions, dtype=torch.float32)
    for row in range(_BATCH):
        policy_target[row, row * 7] = 0.5
        policy_target[row, row * 7 + 1] = 0.5

    wdl_target = torch.zeros(_BATCH, 3, dtype=torch.float32)
    for row in range(_BATCH):
        wdl_target[row, row % 3] = 1.0

    return {
        "observation": torch.randn(
            _BATCH, planes, 9, 9, generator=generator, dtype=torch.float32
        ),
        "policy_target": policy_target,
        "wdl_target": wdl_target,
        "mlh_target": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        "policy_kl": torch.zeros(_BATCH, dtype=torch.float32),
        # All full rows, so the fast-row weight this batch now has to carry
        # (KOM-036: `TrainStep.run` requires the column unconditionally) is
        # exactly ones and the staged-off identity claim is unaffected.
        "search_mode": torch.full((_BATCH,), MODE_FULL, dtype=torch.int64),
        # ... and the cap-row stratum, required on the same terms (KOM-160).
        # All decisive, so that multiplier is exactly ones too and the
        # staged-off identity claim is unaffected by it either.
        "termination": torch.zeros(_BATCH, dtype=torch.int64),
        # ... and the ply the row was recorded at, which the moves-left cap-row
        # weight reconstructs a game length from together with `mlh_target`
        # above -- required at every `mlh_cap_row_weight` (KOM-159 D4). All
        # rows are decisive, so that multiplier is exactly ones as well, and
        # under P0B_RECIPE the moves-left head is off, so nothing it composes
        # reaches a loss: the identity claim is unaffected a third time.
        "ply": torch.arange(_BATCH, dtype=torch.int64),
    }


def _module_parameters(net: KomugiNet, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Return ``{qualified name: parameter}`` for the named submodules."""
    collected: dict[str, torch.Tensor] = {}
    for name in names:
        module = getattr(net, name)
        for leaf, parameter in module.named_parameters():
            collected[f"{name}.{leaf}"] = parameter
    assert collected, f"no parameters found under {names}"
    return collected


def _snapshot(net: KomugiNet, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Detached clones of the named submodules' parameters."""
    return {
        key: value.detach().clone()
        for key, value in _module_parameters(net, names).items()
    }


def _randomise_staged(net: KomugiNet) -> None:
    """Overwrite every staged parameter, from a LOCAL generator.

    This is the whole experiment: the two arms differ in the staged heads'
    *values* and in nothing else. If a P0a parameter moves as a result, some
    staged quantity reached the total.
    """
    generator = torch.Generator().manual_seed(777_777)
    with torch.no_grad():
        for parameter in _module_parameters(net, _STAGED_MODULES).values():
            parameter.copy_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
            )


def _train(*, randomise: bool) -> tuple[KomugiNet, list[dict[str, float]]]:
    """Build a net from the same config and run `_STEPS` steps under P0B."""
    net = KomugiNet(SMALL_CONFIG)
    if randomise:
        _randomise_staged(net)
    step = TrainStep(
        net,
        LossAssembler(P0B_RECIPE),
        StepConfig(batch_size=_BATCH, lr=1e-3, seed=0),
    )
    batch = _batch()
    return net, [step.run(batch) for _ in range(_STEPS)]


def test_p0a_parameters_are_bit_identical_after_a_step_with_the_staged_heads_off() -> (
    None
):
    """Re-randomising every staged parameter moves no P0a parameter at all.

    ``torch.equal``, not ``allclose``: with the staged losses off, the staged
    parameters are not in the graph the backward walks, so the two runs are the
    *same* computation over the same inputs, not two computations that happen
    to agree. Anything short of bit-identity means a staged quantity is
    reaching the total -- and D2's whole point is that "close" is not the
    claim.

    Construction is a pure function of the config and leaves the global torch
    RNG untouched, so the two nets start from identical P0a weights without
    either arm having to reseed anything.
    """
    left, _ = _train(randomise=False)
    right, _ = _train(randomise=True)

    before = _snapshot(left, _P0A_MODULES)
    after = _snapshot(right, _P0A_MODULES)
    assert set(before) == set(after)

    moved = [name for name in before if not torch.equal(before[name], after[name])]
    assert moved == [], (
        "these P0a parameters differ between two runs that differ only in the "
        f"staged heads' initial values: {moved}"
    )

    # ... and the run genuinely trained, so bit-identity is not the identity of
    # two nets that never moved.
    fresh = _snapshot(KomugiNet(SMALL_CONFIG), _P0A_MODULES)
    assert any(
        not torch.equal(fresh[name], before[name]) for name in fresh
    ), "no P0a parameter moved at all: the comparison above is vacuous"

    # The staged parameters really were different between the two arms, or the
    # experiment had no independent variable.
    staged_left = _snapshot(left, _STAGED_MODULES)
    staged_right = _snapshot(right, _STAGED_MODULES)
    assert any(
        not torch.equal(staged_left[name], staged_right[name])
        for name in staged_left
    ), "the two arms' staged parameters are identical: nothing was varied"


def test_no_staged_parameter_moves_and_none_gets_a_gradient() -> None:
    """Off means untouched -- and the gradient half is checked where it is visible.

    The first clause is the easy one: after ``_STEPS`` steps under
    ``P0B_RECIPE``, every staged parameter is ``torch.equal`` to its initial
    value. A disabled head's parameters have no gradient, so AdamW -- which
    creates per-parameter state lazily -- never gives them a moment estimate
    and never updates them.

    The second clause must **not** be ``p.grad is None`` after
    ``TrainStep.run``. ``run`` ends with ``zero_grad(set_to_none=True)``
    *after* ``backward()`` and ``step()``, so at that point every parameter in
    the net has ``grad is None`` -- a P0a parameter's too, and a fully enabled
    staged head's too. The assertion could not fail, which makes it worse than
    no assertion. So the gradient is inspected where it exists: the assembler
    is called directly and ``backward()`` run on the total, with no
    ``zero_grad`` in between.
    """
    net = KomugiNet(SMALL_CONFIG)
    initial = _snapshot(net, _STAGED_MODULES)
    step = TrainStep(
        net,
        LossAssembler(P0B_RECIPE),
        StepConfig(batch_size=_BATCH, lr=1e-3, seed=0),
    )
    batch = _batch()
    for _ in range(_STEPS):
        step.run(batch)

    final = _snapshot(net, _STAGED_MODULES)
    moved = [name for name in initial if not torch.equal(initial[name], final[name])]
    assert moved == [], f"staged parameters moved while their losses were off: {moved}"

    # The gradient clause, inspected before any zero_grad.
    net.zero_grad(set_to_none=True)
    total, per_head = LossAssembler(P0B_RECIPE)(
        outputs_from_net(net(batch["observation"])), targets_from_batch(batch)
    )
    total.backward()  # type: ignore[no-untyped-call]

    with_gradient = [
        name
        for name, parameter in _module_parameters(net, _STAGED_MODULES).items()
        if parameter.grad is not None
    ]
    assert with_gradient == [], (
        "these staged parameters received a gradient with every staged loss "
        f"off: {with_gradient}"
    )
    assert set(per_head) == _P0A_METRIC_KEYS - {"total"}

    # The P0a half did get one, so "no gradient" is a claim about the staged
    # heads and not about a backward that did nothing.
    p0a_with_gradient = [
        name
        for name, parameter in _module_parameters(net, _P0A_MODULES).items()
        if parameter.grad is not None
    ]
    assert p0a_with_gradient, "the backward reached no P0a parameter either"


def test_the_step_metrics_keys_are_exactly_the_p0a_keys() -> None:
    """The metrics vocabulary does not move when the heads land.

    This is the node that catches a staged head switched on by accident: the
    recipe tests either side of it score ``P0A_RECIPE``, which KOM-005b
    deliberately does not edit, so they are blind to ``P0B_RECIPE`` entirely.
    A telemetry consumer keyed on this set is what makes the assertion worth
    having as an equality rather than a subset.
    """
    net = KomugiNet(SMALL_CONFIG)
    step = TrainStep(
        net,
        LossAssembler(P0B_RECIPE),
        StepConfig(batch_size=_BATCH, lr=1e-3, seed=0),
    )
    metrics = step.run(_batch())

    assert set(metrics) == _P0A_METRIC_KEYS, (
        "P0B_RECIPE reported a head P0a did not: a staged head is enabled"
    )
    assert all(
        isinstance(value, float) and math.isfinite(value)
        for value in metrics.values()
    )
