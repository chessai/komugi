"""The structural test: global pooling must actually be global (KOM-104).

A global-pooling bias is the one component of the trunk that cannot be checked
by shape or by loss curve. A trunk whose "gpool" is really a per-position bias,
or whose pooled vector is broadcast from a stale buffer, trains to a perfectly
respectable loss and quietly cannot see a ladder to the other side of the
board -- which is the entire reason KataGo-style nets carry the block.

So the test is differential rather than numeric. Perturb one square of one
sample and ask which output squares moved:

* with a gpool block present, *every* square must move, including squares far
  outside any convolutional receptive field;
* with no gpool block (``gpool_every > blocks``), influence must stop dead at
  the Chebyshev radius the convolution stack can reach -- squares beyond it
  come back bit-identical.

The control is what gives the first assertion teeth: on a 9x9 board a deep
enough conv stack reaches everywhere anyway, so "everything moved" only means
something next to a configuration where it provably should not. The radius
arithmetic itself is pinned against real convolutions in
``test_trunk_probes.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import trunk_probes as probes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "Trunk"):
    pytest.skip("komugi.net exports no Trunk yet (KOM-004)", allow_module_level=True)

# Deliberately tiny: the properties here are structural, not statistical, and a
# one-block trunk keeps the receptive-field bound well inside the board.
_IN_PLANES = 8
_CHANNELS = 32
_ROW = 0
_COL = 0
_DELTA = 1.0


def _delta_map(trunk: Any, *, batch: int = 2, seed: int = 5) -> torch.Tensor:
    """Per-square output movement when ``(_ROW, _COL)`` of sample 0 is bumped."""
    base = probes.noise((batch, _IN_PLANES, probes.BOARD, probes.BOARD), seed=seed)
    bumped = probes.perturbed(base, sample=0, row=_ROW, col=_COL, delta=_DELTA)
    left = probes.forward_no_grad(trunk, base)
    right = probes.forward_no_grad(trunk, bumped)
    assert left.shape == (batch, _CHANNELS, probes.BOARD, probes.BOARD)
    return probes.position_delta(left[:1], right[:1])


def test_a_gpool_trunk_moves_every_square_on_the_board() -> None:
    """One perturbed square, and nothing on the board is left untouched."""
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=3,
        gpool_every=3,
        seed=0,
    )
    delta = _delta_map(trunk)
    unmoved = [
        (row, col)
        for row in range(probes.BOARD)
        for col in range(probes.BOARD)
        if float(delta[row, col].item()) == 0.0
    ]
    assert unmoved == [], f"squares untouched by the perturbation: {unmoved}"


def test_gpool_carries_influence_past_the_convolutional_receptive_field() -> None:
    """The kill for a fake gpool: reach beyond what convolutions can explain.

    One block -- one stem 3x3 plus the block's two 3x3s -- so the conv stack
    reaches a Chebyshev radius of 3 from the corner and most of the board is
    out of its range. With the block's global-pooling bias in play those far
    squares must move anyway.
    """
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=1,
        gpool_every=1,
        seed=0,
    )
    delta = _delta_map(trunk)
    outside = probes.far_mask(_ROW, _COL, probes.receptive_radius(1))
    assert bool(outside.any()), "the control region must be non-empty to mean anything"
    far = delta[outside]
    assert bool((far > 0).all()), (
        "squares beyond the convolutional receptive field did not move: "
        f"{int((far == 0).sum().item())} of {far.numel()} unchanged -- the gpool "
        "bias is not global"
    )


def test_without_gpool_influence_stops_at_the_receptive_field() -> None:
    """CONTROL: ``gpool_every > blocks`` leaves a pure convolution stack.

    Bit-identical, not merely close: the far squares are computed from windows
    the perturbation never entered, so the same weights over the same numbers
    must produce the same bits. Any leak -- a genuine gpool that ignored the
    config, a batch-statistic dependence, a flatten-and-project shortcut --
    shows up here as a non-zero difference.
    """
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=1,
        gpool_every=2,  # strictly greater than `blocks`: no block carries gpool
        seed=0,
    )
    delta = _delta_map(trunk)
    # Radius 3 = one stem 3x3 + the block's two 3x3s, counted off the merged
    # block rather than assumed. Tightened from 5 by the union review (F1): at
    # 5 this covered 45 squares and a trunk that grew its per-block reach to
    # radius <= 5 slipped through; at 3 it covers 65 and that growth fails here.
    radius = probes.receptive_radius(1)
    outside = probes.far_mask(_ROW, _COL, radius)

    far = delta[outside]
    assert torch.equal(far, torch.zeros_like(far)), (
        f"a gpool-free trunk moved {int((far != 0).sum().item())} squares beyond "
        f"Chebyshev radius {radius}; max movement {float(far.max().item()):.3e}"
    )
    # And the perturbation was live: the squares inside the radius did move.
    inside = delta[~outside]
    assert float(inside.max().item()) > 0.0


def test_gpool_pools_within_a_sample_not_across_the_batch() -> None:
    """Perturbing sample 0 must leave sample 1's output alone.

    A pooled statistic taken over ``dim=(0, 2, 3)`` instead of ``dim=(2, 3)``
    passes every globality assertion above while silently making the trunk's
    output depend on its batch-mates.
    """
    _, trunk = probes.build_trunk(
        net,
        in_planes=_IN_PLANES,
        channels=_CHANNELS,
        blocks=3,
        gpool_every=3,
        seed=0,
    )
    base = probes.noise((2, _IN_PLANES, probes.BOARD, probes.BOARD), seed=5)
    bumped = probes.perturbed(base, sample=0, row=_ROW, col=_COL, delta=_DELTA)
    left = probes.forward_no_grad(trunk, base)
    right = probes.forward_no_grad(trunk, bumped)

    neighbour = probes.position_delta(left[1:], right[1:])
    assert float(neighbour.max().item()) < 1e-6, (
        "the untouched sample moved when its batch-mate was perturbed: "
        f"max {float(neighbour.max().item()):.3e} -- pooling crosses the batch"
    )
