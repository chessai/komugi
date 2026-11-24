"""The trunk is config-driven, and the config is a contract (KOM-104).

The point of KOM-004's configuration object is that b18c256 is a config change
rather than a code change: the campaign scales the net by editing two integers,
and anything that hardcodes 10 blocks or 128 channels somewhere inside the
module tree turns that promise into a debugging session months later. So the
big config is constructed and run once here, and its parameter count is checked
against the small one -- width enters the count quadratically, depth linearly,
so the ratio is far above the linear-in-depth 1.8x a trunk that silently
ignored ``channels`` would show.

The input-plane count gets the same treatment from the other side: the encoder
feeds this trunk, its plane count moves as the encoding grows, and only the
stem may know about it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch

import trunk_probes as probes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "Trunk"):
    pytest.skip("komugi.net exports no Trunk yet (KOM-004)", allow_module_level=True)

_IN_PLANES = 8


def _forward_one(trunk: Any, in_planes: int) -> torch.Tensor:
    return probes.forward_no_grad(
        trunk, probes.noise((1, in_planes, probes.BOARD, probes.BOARD), seed=41)
    )


def test_b18c256_constructs_and_forwards() -> None:
    """The scaled-up configuration is a config change and nothing else."""
    config, trunk = probes.build_trunk(
        net, in_planes=_IN_PLANES, channels=256, blocks=18, gpool_every=3, seed=0
    )
    assert (config.channels, config.blocks) == (256, 18)

    out = _forward_one(trunk, _IN_PLANES)
    assert out.shape == (1, 256, probes.BOARD, probes.BOARD)
    assert bool(torch.isfinite(out).all()), "b18c256 produced NaN or Inf at init"


def test_parameter_count_grows_superlinearly_with_the_config() -> None:
    """b18c256 is much more than 1.8x b10c128: width counts quadratically."""
    _, small = probes.build_trunk(
        net, in_planes=_IN_PLANES, channels=128, blocks=10, gpool_every=3, seed=0
    )
    _, large = probes.build_trunk(
        net, in_planes=_IN_PLANES, channels=256, blocks=18, gpool_every=3, seed=0
    )
    small_count = probes.parameter_count(small)
    large_count = probes.parameter_count(large)

    assert small_count > 0 and large_count > 0
    ratio = large_count / small_count
    assert ratio > 3.0, (
        f"b18c256 has {large_count} parameters against b10c128's {small_count} "
        f"(ratio {ratio:.2f}); the config is not driving both depth and width"
    )


@pytest.mark.parametrize("in_planes", [8, 128])
def test_in_planes_is_respected(in_planes: int) -> None:
    """Both a narrow and an encoder-sized input stack feed the same trunk."""
    config, trunk = probes.build_trunk(
        net, in_planes=in_planes, channels=32, blocks=2, gpool_every=3, seed=0
    )
    assert config.in_planes == in_planes
    out = _forward_one(trunk, in_planes)
    assert out.shape == (1, 32, probes.BOARD, probes.BOARD)


def test_a_mismatched_input_stack_is_rejected() -> None:
    """``in_planes`` is load-bearing, not decorative: the wrong width must fail."""
    _, trunk = probes.build_trunk(
        net, in_planes=_IN_PLANES, channels=32, blocks=2, gpool_every=3, seed=0
    )
    # RuntimeError is what a Conv2d channel mismatch raises; ValueError covers
    # an implementation that checks the input stack itself before the stem.
    with pytest.raises((RuntimeError, ValueError)):
        _forward_one(trunk, _IN_PLANES + 1)


def test_the_trunk_keeps_its_config() -> None:
    """``trunk.config`` is the config it was built from, and it is frozen."""
    config, trunk = probes.build_trunk(
        net, in_planes=_IN_PLANES, channels=32, blocks=2, gpool_every=3, seed=7
    )
    assert trunk.config == config
    assert trunk.config.seed == 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        trunk.config.seed = 8


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"in_planes": 0}, id="in_planes-zero"),
        pytest.param({"in_planes": -1}, id="in_planes-negative"),
        pytest.param({"channels": 0}, id="channels-zero"),
        pytest.param({"channels": 33}, id="channels-odd"),
        pytest.param({"blocks": 0}, id="blocks-zero"),
        pytest.param({"gpool_every": 0}, id="gpool_every-zero"),
        pytest.param({"gpool_every": -3}, id="gpool_every-negative"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, int]) -> None:
    """Validation happens at config time, not at the first NaN.

    Odd channel counts are in the list because the nested-bottleneck block
    halves the trunk width for its inner branch; an odd width either truncates
    silently or fails deep inside the forward pass, and neither is a diagnosis
    anyone wants from a training run.
    """
    fields: dict[str, int] = {
        "in_planes": _IN_PLANES,
        "channels": 32,
        "blocks": 2,
        "gpool_every": 3,
        "seed": 0,
    }
    fields.update(kwargs)
    with pytest.raises(ValueError):
        net.TrunkConfig(**fields)


def test_the_default_configuration_is_b10c128() -> None:
    """Only ``in_planes`` is required; the defaults are the production trunk."""
    config = net.TrunkConfig(in_planes=_IN_PLANES)
    assert (config.channels, config.blocks, config.gpool_every, config.seed) == (
        128,
        10,
        3,
        0,
    )
