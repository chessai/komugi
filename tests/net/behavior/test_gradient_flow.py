"""Every parameter of the trunk must be reachable by a gradient (KOM-104).

A branch that is built, initialised, counted in ``parameters()``, and then
never used -- a residual branch whose result is dropped, a gpool head appended
to the module tree but not to the forward pass, a pooled vector taken under
``no_grad`` or through ``.detach()`` -- costs memory and optimizer state and
contributes nothing. Training runs; the loss falls; the dead weights stay at
their init values forever, and the net is quietly smaller than the config says.

``grad is None`` catches "not in the graph". ``grad == 0`` everywhere catches
the subtler "in the graph but detached from the loss", which is what a
``.detach()`` on a gpool path looks like from the outside.

This runs in train mode, the mode the failure would actually occur in: BatchNorm
affine parameters take their gradients through the batch-normalised activations
rather than through the running statistics, so ``weight``/``bias`` on every norm
are covered too.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import trunk_probes as probes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "Trunk"):
    pytest.skip("komugi.net exports no Trunk yet (KOM-004)", allow_module_level=True)

_IN_PLANES = 8
_CHANNELS = 32
_BATCH = 4


def _backward_through(trunk: Any, *, seed: int) -> None:
    """Sum-of-squares on the trunk output, backpropagated."""
    trunk.train()
    trunk.zero_grad(set_to_none=True)
    inputs = probes.noise((_BATCH, _IN_PLANES, probes.BOARD, probes.BOARD), seed=seed)
    out: torch.Tensor = trunk(inputs)
    loss = out.pow(2).sum()
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped


@pytest.mark.parametrize(
    ("blocks", "gpool_every"),
    [
        pytest.param(3, 3, id="gpool-on-the-last-block"),
        pytest.param(2, 1, id="gpool-on-every-block"),
    ],
)
def test_every_parameter_receives_a_live_gradient(blocks: int, gpool_every: int) -> None:
    """No parameter is left with a missing, dead, or non-finite gradient."""
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=blocks,
        gpool_every=gpool_every,
        seed=0,
    )
    _backward_through(trunk, seed=31)

    missing: list[str] = []
    dead: list[str] = []
    broken: list[str] = []
    total = 0
    for name, parameter in trunk.named_parameters():
        total += 1
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            broken.append(name)
        elif float(grad.abs().max().item()) == 0.0:
            dead.append(name)

    assert total > 0, "the trunk exposes no parameters at all"
    assert missing == [], f"parameters outside the autograd graph: {missing}"
    assert broken == [], f"parameters with NaN/Inf gradients: {broken}"
    assert dead == [], (
        f"parameters with an all-zero gradient: {dead} -- a dead branch, a "
        "detached gpool path, or a zero-initialised residual scale"
    )


def test_batchnorm_affine_parameters_are_trainable() -> None:
    """The one-batch-norm layout must carry learnable scale and shift.

    Called out separately because an affine-free BatchNorm still normalises,
    still trains, and still reports sensible statistics -- it just cannot undo
    the normalisation, which is a real capacity loss and an easy default to
    leave switched off.
    """
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=3,
        gpool_every=3,
        seed=0,
    )
    _backward_through(trunk, seed=32)

    norms = [module for module in trunk.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert norms, "no BatchNorm2d in the trunk"
    for index, norm in enumerate(norms):
        # Read the affine pair out of the parameter registry rather than off the
        # module: torch's _NormBase deliberately leaves `weight`/`bias`
        # undeclared, so attribute access falls through Module.__getattr__ and
        # types as `Tensor | Module`. named_parameters() is typed, and an
        # affine-free BatchNorm registers both as None -- so it simply omits
        # them here, which is exactly the failure this test is looking for.
        affine = dict(norm.named_parameters(recurse=False))
        for label in ("weight", "bias"):
            parameter = affine.get(label)
            assert parameter is not None, f"BatchNorm {index} has no affine {label}"
            grad = parameter.grad
            assert grad is not None, f"BatchNorm {index} {label} got no gradient"
            assert float(grad.abs().max().item()) > 0.0, (
                f"BatchNorm {index} {label} gradient is identically zero"
            )


def test_the_input_is_not_severed_from_the_graph() -> None:
    """Gradient reaches the trunk input, so the stem is genuinely in the path."""
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=2,
        gpool_every=3,
        seed=0,
    )
    trunk.train()
    inputs = probes.noise((_BATCH, _IN_PLANES, probes.BOARD, probes.BOARD), seed=33)
    inputs.requires_grad_(True)
    out: torch.Tensor = trunk(inputs)
    out.pow(2).sum().backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped

    grad = inputs.grad
    assert grad is not None, "no gradient reached the trunk input"
    assert float(grad.abs().max().item()) > 0.0, "the trunk input gradient is all zeros"
