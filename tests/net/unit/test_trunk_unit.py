"""Unit tests for the trunk: its contract, not its behaviour.

Shapes, dtypes, configuration validation, and the two promises construction
makes -- that it is a pure function of the config, and that it is invisible to
the global RNG. The deeper behavioural properties (global-pooling reach,
initialisation statistics, batch independence in eval, gradient flow) are the
companion oracle suite's business and are deliberately not duplicated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from komugi.net import Trunk, TrunkConfig

# Small enough that a forward pass costs nothing, wide enough to be a real
# trunk: even channels, several blocks, and at least one pooling block.
_SMALL = TrunkConfig(in_planes=7, channels=16, blocks=4, gpool_every=2)


def _state_dicts_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    """Return whether two state dicts have the same keys and equal tensors."""
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_forward_maps_planes_to_trunk_features() -> None:
    """``(N, in_planes, 9, 9)`` float32 in, ``(N, channels, 9, 9)`` out."""
    trunk = Trunk(_SMALL)
    x = torch.zeros(3, _SMALL.in_planes, 9, 9, dtype=torch.float32)

    out = trunk(x)

    assert out.shape == (3, _SMALL.channels, 9, 9)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


def test_forward_at_init_is_finite_on_noise() -> None:
    """Unit-variance noise through an untrained trunk stays finite.

    The quantitative version of this -- the output standard deviation lands in
    a sane band, and does not grow monotonically with depth -- belongs to the
    companion suite. Here it is only a smoke test that the fixed-variance
    initialisation is wired up at all.
    """
    trunk = Trunk(_SMALL)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(4, _SMALL.in_planes, 9, 9, generator=generator)

    out = trunk(x)

    assert bool(torch.isfinite(out).all())


def test_the_stored_config_is_the_one_passed_in() -> None:
    """:attr:`Trunk.config` is the config, unmodified."""
    trunk = Trunk(_SMALL)

    assert trunk.config == _SMALL


def test_config_rejects_shapes_that_cannot_describe_a_trunk() -> None:
    """Every structural field must be positive, and the width must be even."""
    with pytest.raises(ValueError, match="in_planes"):
        TrunkConfig(in_planes=0)
    with pytest.raises(ValueError, match="in_planes"):
        TrunkConfig(in_planes=-1)
    with pytest.raises(ValueError, match="channels"):
        TrunkConfig(in_planes=8, channels=0)
    with pytest.raises(ValueError, match="even"):
        TrunkConfig(in_planes=8, channels=127)
    with pytest.raises(ValueError, match="blocks"):
        TrunkConfig(in_planes=8, blocks=0)
    with pytest.raises(ValueError, match="gpool_every"):
        TrunkConfig(in_planes=8, gpool_every=0)
    with pytest.raises(ValueError, match="seed"):
        TrunkConfig(in_planes=8, seed=-1)


def test_config_defaults_are_b10c128_every_third() -> None:
    """The default config is the v2.0 trunk the design document specifies."""
    config = TrunkConfig(in_planes=128)

    assert (config.channels, config.blocks, config.gpool_every) == (128, 10, 3)
    assert config.bottleneck_channels == 64
    gpool = [i for i in range(1, config.blocks + 1) if config.is_gpool_block(i)]
    assert gpool == [3, 6, 9]


def test_pooling_blocks_are_built_where_the_config_says() -> None:
    """Only blocks 3, 6 and 9 of the default shape own a pooling linear.

    Read off the parameter names, which is the cheapest way to see the
    structure without reaching into private classes. The block index in the
    name is zero-based, so blocks 3/6/9 are entries 2/5/8.
    """
    trunk = Trunk(TrunkConfig(in_planes=4, channels=8, blocks=10))

    pooled = sorted(
        int(name.split(".")[1])
        for name in trunk.state_dict()
        if name.endswith("pool.linear.weight")
    )

    assert pooled == [2, 5, 8]


def test_the_same_config_builds_the_same_parameters() -> None:
    """Construction is a pure function of the config."""
    left = Trunk(_SMALL)
    right = Trunk(_SMALL)

    assert _state_dicts_equal(left.state_dict(), right.state_dict())


def test_a_different_seed_builds_different_parameters() -> None:
    """The seed is load-bearing: it is the only thing that differs here."""
    left = Trunk(_SMALL)
    right = Trunk(
        TrunkConfig(
            in_planes=_SMALL.in_planes,
            channels=_SMALL.channels,
            blocks=_SMALL.blocks,
            gpool_every=_SMALL.gpool_every,
            seed=_SMALL.seed + 1,
        )
    )

    assert not _state_dicts_equal(left.state_dict(), right.state_dict())


def test_construction_leaves_the_global_rng_untouched() -> None:
    """Building a trunk neither reads nor advances the global generator.

    Both halves matter: the state is unchanged afterwards (so a caller's
    seeded stream is not silently perturbed), and the parameters are the same
    whatever the state was beforehand (so a trunk is reproducible without
    reproducing the whole process).
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    first = Trunk(_SMALL)
    after = torch.get_rng_state()

    assert torch.equal(before, after)

    torch.manual_seed(4321)
    second = Trunk(_SMALL)

    assert _state_dicts_equal(first.state_dict(), second.state_dict())


def test_state_dict_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """Saved and reloaded parameters are bit-identical, and still run."""
    original = Trunk(_SMALL)
    path = tmp_path / "trunk.pt"
    torch.save(original.state_dict(), path)

    restored = Trunk(
        TrunkConfig(
            in_planes=_SMALL.in_planes,
            channels=_SMALL.channels,
            blocks=_SMALL.blocks,
            gpool_every=_SMALL.gpool_every,
            seed=_SMALL.seed + 7,
        )
    )
    assert not _state_dicts_equal(original.state_dict(), restored.state_dict())

    restored.load_state_dict(torch.load(path, weights_only=True))

    assert _state_dicts_equal(original.state_dict(), restored.state_dict())

    original.eval()
    restored.eval()
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(2, _SMALL.in_planes, 9, 9, generator=generator)
    with torch.no_grad():
        assert torch.equal(original(x), restored(x))


def test_b18c256_is_a_config_change() -> None:
    """The v2.1 scale target builds and runs with no code change at all."""
    config = TrunkConfig(in_planes=128, channels=256, blocks=18, gpool_every=3)
    trunk = Trunk(config)
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(2, config.in_planes, 9, 9, generator=generator)

    with torch.no_grad():
        out = trunk(x)

    assert out.shape == (2, 256, 9, 9)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    # An all-zero output would satisfy every assertion above while meaning the
    # deeper half of the trunk had quietly annihilated the signal.
    assert float(out.std()) > 0.0
