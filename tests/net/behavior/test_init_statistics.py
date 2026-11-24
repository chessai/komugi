"""Init-time statistics and seed locality for the trunk (KOM-104).

Two failures live here, both invisible to a shape test:

*Unbounded residual growth.* A ten-block trunk whose branches are not scaled
adds ten unit-variance contributions to its stream, so activations leave the
range the value/policy heads were designed for and the first training steps
either saturate or explode. The probe is the ratio of the last block's input
std to the first's -- a stream that is merely carried forward keeps that ratio
near 1, a stream that accumulates unscaled branches grows like sqrt(depth).

*A trunk that reaches for the global RNG.* Init must be a function of
``config.seed`` alone, through a local generator. If it draws from the global
stream instead, two trunks built in a different order differ, a trainer that
seeds torch once gets different weights depending on what else ran first, and
every "same seed, same net" claim downstream is false. That is pinned from the
outside: the global stream must be byte-identical across a construction.

Everything runs in eval mode with freshly initialised BatchNorms, whose default
running statistics (mean 0, var 1) make eval normalisation the identity up to
the affine parameters. So these numbers are the trunk's own scaling, not
statistics accumulated from the probe batch.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import trunk_probes as probes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "Trunk"):
    pytest.skip("komugi.net exports no Trunk yet (KOM-004)", allow_module_level=True)

# The production shape, and the only large forward pass in this suite.
_IN_PLANES = 8
_B10C128 = {"in_planes": _IN_PLANES, "channels": 128, "blocks": 10, "gpool_every": 3}
_BATCH = 16

# Everything that does not need production depth uses this instead.
_SMALL = {"in_planes": _IN_PLANES, "channels": 32, "blocks": 4, "gpool_every": 3}


@pytest.fixture(scope="module")
def b10c128() -> Any:
    """A fresh b10c128 trunk at init, eval mode, built once for this module."""
    _, trunk = probes.build_trunk(net, seed=0, **_B10C128)
    return trunk


@pytest.fixture(scope="module")
def gaussian_batch() -> torch.Tensor:
    """Unit-variance Gaussian noise, the input distribution these bounds assume."""
    return probes.noise((_BATCH, _IN_PLANES, probes.BOARD, probes.BOARD), seed=1234)


def test_output_is_finite_and_bounded_at_init(
    b10c128: Any, gaussian_batch: torch.Tensor
) -> None:
    """Unit-variance in, same-order-of-magnitude out, no NaNs anywhere."""
    out = probes.forward_no_grad(b10c128, gaussian_batch)
    assert out.shape == (_BATCH, 128, probes.BOARD, probes.BOARD)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all()), "trunk output contains NaN or Inf at init"

    std = float(out.std().item())
    assert 0.2 <= std <= 5.0, f"init output std {std:.4f} outside [0.2, 5.0]"


def test_residual_growth_down_the_trunk_is_bounded(
    b10c128: Any, gaussian_batch: torch.Tensor
) -> None:
    """The last block is handed a stream no more than 2x the first block's.

    Missing branch scaling is exactly what this catches: ten unscaled residual
    additions grow the stream by roughly sqrt(10) ~ 3.2, which is over the bar,
    while a correctly scaled trunk sits near 1.
    """
    with probes.StdRecorder().attach(b10c128) as recorder:
        probes.forward_no_grad(b10c128, gaussian_batch)

    stds = recorder.trunk_width_stds(128)
    assert len(stds) >= 2, f"expected a per-block probe, recorded {recorder.channels}"
    assert all(math.isfinite(value) for value in stds), f"non-finite block statistics: {stds}"
    assert all(value > 0.0 for value in stds), f"a block was handed a dead stream: {stds}"

    ratio = stds[-1] / stds[0]
    assert ratio <= 2.0, (
        f"activation std grew {ratio:.2f}x from the first block to the last "
        f"({stds[0]:.4f} -> {stds[-1]:.4f}); residual branches look unscaled"
    )


def test_one_batchnorm_per_block() -> None:
    """The one-batch-norm layout, counted rather than assumed."""
    config, trunk = probes.build_trunk(net, seed=0, **_SMALL)
    norms = [m for m in trunk.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(norms) == config.blocks, (
        f"expected one BatchNorm per block ({config.blocks}), found {len(norms)}"
    )


def test_same_seed_reproduces_parameters() -> None:
    """Two trunks built from the same config are the same trunk."""
    _, first = probes.build_trunk(net, seed=17, **_SMALL)
    _, second = probes.build_trunk(net, seed=17, **_SMALL)
    differing = probes.differing_keys(
        probes.state_dict_of(first), probes.state_dict_of(second)
    )
    assert differing == [], f"same seed produced different tensors: {differing}"


def test_different_seed_changes_parameters() -> None:
    """...and a different seed is a different trunk, not a relabelled one."""
    _, first = probes.build_trunk(net, seed=17, **_SMALL)
    _, second = probes.build_trunk(net, seed=18, **_SMALL)
    differing = probes.differing_keys(
        probes.state_dict_of(first), probes.state_dict_of(second)
    )
    assert differing, "seeds 17 and 18 produced identical parameters"


def test_construction_does_not_touch_the_global_rng() -> None:
    """The local-generator promise, observed from outside the trunk.

    Seed the global stream, take two draws, and take them again with a trunk
    constructed in between. If init consumed the global stream the second pair
    diverges; if init *required* the global stream, seeding it identically
    would still leave this passing, which is why the seed-determinism tests
    above are the other half of the pair.
    """
    torch.manual_seed(20260104)
    control = [torch.randn(4), torch.randn(4)]

    torch.manual_seed(20260104)
    probes.build_trunk(net, seed=3, **_SMALL)
    observed = [torch.randn(4), torch.randn(4)]

    assert torch.equal(observed[0], control[0]), "trunk construction consumed the global RNG"
    assert torch.equal(observed[1], control[1]), "trunk construction consumed the global RNG"
