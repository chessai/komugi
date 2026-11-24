"""Sanity suite for the trunk probes themselves (KOM-104).

Unlike its five siblings this module has no ``importorskip`` guard: it depends
only on torch, so it runs today, in a worktree where ``komugi.net`` does not
exist. The point is that the trunk suite's sharpest assertion -- "a gpool-free
trunk provably cannot move these squares" -- rests on a receptive-field
argument, and an argument is not a test. Here the argument is checked against
real convolutions, so when the trunk lands the control test is known to be
measuring what it claims.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import trunk_probes as probes


def _conv_stack(depth: int) -> torch.nn.Sequential:
    """``depth`` stacked 3x3 same-padding convolutions with constant weights.

    Constant positive weights make the spread exact rather than probabilistic:
    every square within the receptive field is reached with a strictly positive
    coefficient, so "unchanged" means "unreachable" and nothing else.
    """
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            torch.nn.init.constant_(conv.weight, 0.1)
        layers.append(conv)
    return torch.nn.Sequential(*layers)


def test_receptive_radius_counts_every_three_by_three() -> None:
    """One square of reach per 3x3 convolution, stem included.

    The default is two 3x3s per block, counted off the block KOM-004 merged --
    so one block reaches 3, not the 5 the pre-review default claimed.
    """
    assert probes.receptive_radius(0) == 1
    assert probes.receptive_radius(1) == 3
    assert probes.receptive_radius(3) == 7
    assert probes.receptive_radius(2, convs_per_block=2, stem_convs=0) == 4
    # Explicit counts still compose the same way -- the arithmetic is general,
    # only the default changed with the review (union review F1).
    assert probes.receptive_radius(1, convs_per_block=4) == 5
    assert probes.receptive_radius(3, convs_per_block=4) == 13


def test_receptive_radius_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        probes.receptive_radius(-1)


def test_far_mask_matches_a_brute_force_walk() -> None:
    """The vectorised mask agrees with the definition, square by square."""
    radius = 3
    mask = probes.far_mask(4, 2, radius)
    assert mask.shape == (probes.BOARD, probes.BOARD)
    for row in range(probes.BOARD):
        for col in range(probes.BOARD):
            distance = max(abs(row - 4), abs(col - 2))
            assert bool(mask[row, col]) is (distance > radius)


def test_far_mask_of_a_board_sized_radius_is_empty() -> None:
    """No square is out of reach once the radius covers the board."""
    assert not bool(probes.far_mask(0, 0, probes.BOARD - 1).any())


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_conv_stack_spread_is_exactly_the_chebyshev_radius(depth: int) -> None:
    """The bound the control test relies on, measured on real convolutions.

    Beyond the radius the outputs come back *bit*-identical, and at the radius
    they do not: the bound is both sound and tight, which is what makes the
    no-gpool control able to fail a trunk that leaks information globally.
    """
    stack = _conv_stack(depth)
    stack.eval()
    base = probes.noise((1, 1, probes.BOARD, probes.BOARD), seed=7)
    bumped = probes.perturbed(base, sample=0, row=0, col=0, delta=1.0)

    with torch.no_grad():
        left: torch.Tensor = stack(base)
        right: torch.Tensor = stack(bumped)
    delta = probes.position_delta(left, right)

    outside = probes.far_mask(0, 0, depth)
    assert torch.equal(delta[outside], torch.zeros_like(delta[outside]))
    # Tightness: the ring at exactly `depth` moved, so the radius is not slack.
    ring = probes.far_mask(0, 0, depth - 1) & ~outside
    assert bool((delta[ring] > 0).all())


def test_position_delta_reduces_over_batch_and_channel() -> None:
    left = torch.zeros(2, 3, probes.BOARD, probes.BOARD)
    right = left.clone()
    right[1, 2, 4, 5] = -0.25
    delta = probes.position_delta(left, right)
    assert delta.shape == (probes.BOARD, probes.BOARD)
    assert float(delta[4, 5].item()) == pytest.approx(0.25)
    assert float(delta.sum().item()) == pytest.approx(0.25)


def test_position_delta_rejects_mismatched_or_unshaped_input() -> None:
    with pytest.raises(ValueError):
        probes.position_delta(torch.zeros(1, 1, 9, 9), torch.zeros(1, 2, 9, 9))
    with pytest.raises(ValueError):
        probes.position_delta(torch.zeros(9, 9), torch.zeros(9, 9))


def test_noise_is_seed_determined_and_leaves_the_global_rng_alone() -> None:
    """Fixtures draw from a local generator, exactly as the trunk must."""
    assert torch.equal(probes.noise((2, 3), seed=1), probes.noise((2, 3), seed=1))
    assert not torch.equal(probes.noise((2, 3), seed=1), probes.noise((2, 3), seed=2))

    torch.manual_seed(4242)
    control = torch.randn(3)
    torch.manual_seed(4242)
    probes.noise((16, 8, probes.BOARD, probes.BOARD), seed=99)
    assert torch.equal(torch.randn(3), control)


def test_perturbed_touches_one_square_of_one_sample() -> None:
    base = torch.zeros(2, 4, probes.BOARD, probes.BOARD)
    bumped = probes.perturbed(base, sample=1, row=3, col=6, delta=2.0)
    assert torch.equal(base, torch.zeros_like(base))  # the original is untouched
    assert float(bumped.sum().item()) == pytest.approx(8.0)  # 4 planes x 2.0
    assert bool((bumped[1, :, 3, 6] == 2.0).all())
    assert torch.equal(bumped[0], base[0])


def test_differing_keys_reports_value_and_shape_disagreement() -> None:
    left = {"a": torch.zeros(2), "b": torch.ones(2)}
    right = {"a": torch.zeros(2), "b": torch.ones(2)}
    assert probes.differing_keys(left, right) == []

    right["b"] = torch.full((2,), 1.0001)
    assert probes.differing_keys(left, right) == ["b"]

    del right["a"]
    assert probes.differing_keys(left, right) == ["a", "b"]


def test_state_dict_and_parameter_count_agree_with_torch() -> None:
    module = torch.nn.Conv2d(3, 5, kernel_size=3, padding=1)
    assert probes.parameter_count(module) == 3 * 5 * 3 * 3 + 5
    assert sorted(probes.state_dict_of(module)) == ["bias", "weight"]


class _TwoNormToy(torch.nn.Module):
    """Two BatchNorms at different widths, to exercise the width filter."""

    def __init__(self) -> None:
        super().__init__()
        self.wide = torch.nn.BatchNorm2d(8)
        self.narrow_proj = torch.nn.Conv2d(8, 4, kernel_size=1, bias=False)
        self.narrow = torch.nn.BatchNorm2d(4)
        self.wide_again = torch.nn.BatchNorm2d(8)
        self.restore = torch.nn.Conv2d(4, 8, kernel_size=1, bias=False)
        # Constant projections keep the growth this toy exhibits deterministic
        # rather than a draw from the default init.
        with torch.no_grad():
            torch.nn.init.constant_(self.narrow_proj.weight, 0.5)
            torch.nn.init.constant_(self.restore.weight, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.wide(x)
        out = self.narrow(self.narrow_proj(out))
        result: torch.Tensor = self.wide_again(self.restore(out) * 3.0)
        return result


def test_std_recorder_walks_batchnorm_inputs_in_forward_order() -> None:
    toy = _TwoNormToy()
    toy.eval()
    inputs = probes.noise((4, 8, probes.BOARD, probes.BOARD), seed=11)

    with probes.StdRecorder().attach(toy) as recorder:
        with torch.no_grad():
            toy(inputs)

    assert recorder.channels == [8, 4, 8]
    assert recorder.stds[0] == pytest.approx(float(inputs.std().item()), rel=1e-5)
    # The width filter keeps the full-width stream and drops the bottleneck.
    assert recorder.trunk_width_stds(8) == [recorder.stds[0], recorder.stds[2]]
    # ...and falls back to everything when nothing matches the asked-for width.
    assert recorder.trunk_width_stds(64) == recorder.stds

    # The x3.0 before the last norm is visible as growth, so the ratio the
    # init-statistics test computes really does track the residual stream.
    assert recorder.stds[2] / recorder.stds[0] > 2.0


def test_std_recorder_detaches_after_use() -> None:
    toy = _TwoNormToy()
    toy.eval()
    recorder = probes.StdRecorder().attach(toy)
    recorder.remove()
    with torch.no_grad():
        toy(probes.noise((1, 8, probes.BOARD, probes.BOARD), seed=3))
    assert recorder.stds == []


def test_build_trunk_uses_the_module_it_is_handed() -> None:
    """``build_trunk`` is plumbing: config kwargs through, eval mode out.

    Checked against a stand-in module so the plumbing is covered before the
    real ``komugi.net`` exists.
    """

    class _FakeConfig:
        def __init__(self, *, in_planes: int, channels: int) -> None:
            self.in_planes = in_planes
            self.channels = channels

    class _FakeNet:
        TrunkConfig = _FakeConfig

        @staticmethod
        def Trunk(config: Any) -> torch.nn.Module:  # noqa: N802 - mirrors the real name
            module = torch.nn.Conv2d(config.in_planes, config.channels, kernel_size=1)
            module.train()
            return module

    config, trunk = probes.build_trunk(_FakeNet(), in_planes=8, channels=32)
    assert (config.in_planes, config.channels) == (8, 32)
    assert trunk.training is False

    out = probes.forward_no_grad(trunk, torch.zeros(1, 8, probes.BOARD, probes.BOARD))
    assert out.shape == (1, 32, probes.BOARD, probes.BOARD)
    assert out.requires_grad is False
