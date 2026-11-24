"""The policy flatten IS the engine's action id (KOM-105).

THE test of this suite.  Everything else about a policy head -- its width, its
finiteness, its gradients -- can be right while this one thing is wrong, and if
it is wrong nothing downstream can notice: training would optimise a
consistently permuted target, MCTS would expand a consistently permuted child,
and the net would learn a coherent, fluent, wrong game.  The engine's mask
would still be the right length; the softmax would still sum to one.

So the correspondence is pinned STRUCTURALLY rather than statistically, by
driving the head's final 1x1 convolution to a known constant and reading off
where the value lands:

* **plane -> id range.**  Zero every weight and bias of the final conv, then
  set ``bias[plane] = 1.0``.  Whatever precedes the conv is now irrelevant --
  a zeroed kernel annihilates its input -- so the output is the bias, spread
  over all 81 squares of that plane and nothing else.  If the flatten is
  plane-major, exactly the ids ``[plane * 81, (plane + 1) * 81)`` are 1.0.  A
  square-major flatten puts those ones 461 apart instead, and every pinned id
  fails.
* **square -> offset.**  Zero the conv again, set a single kernel tap so the
  chosen output plane copies input channel 0, and hand the conv a
  single-hot ``(1, C, 9, 9)`` input.  The one non-zero logit must be at
  ``plane * 81 + sq_index``.  Square 1 and square 9 are both pinned precisely
  because they are the pair a column-major flatten would swap.

The conv's input is substituted with a ``forward_pre_hook`` rather than fed
through the head, because the layers before it (a 1x1 mixer, a norm) may spread
one hot square over a neighbourhood -- which would make "the hot logit" a
statement about their kernels rather than about the flatten.  The hook is
plain torch, touches no attribute of the implementation, and is removed in a
``finally``.

The final conv is located by SHAPE -- the unique ``Conv2d`` in the head whose
``out_channels`` is 461 -- so the test survives any naming the implementer
picks, and says something intelligible if the head is not built that way at all.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import action_ids
from action_ids import BOARD, NUM_ACTIONS, PINS, POLICY_PLANES, SQUARES

net = pytest.importorskip("komugi.net")

if not hasattr(net, "KomugiNet"):
    pytest.skip(
        "komugi.net exports no KomugiNet yet (KOM-005a)", allow_module_level=True
    )

#: Deliberately narrow: this suite measures indexing, not capacity.
_CHANNELS = 16

_HEAD_NAMES = ("PolicyHead", "SoftPolicyHead")


def _policy_head(head_name: str) -> Any:
    """Build one of the two policy heads at ``_CHANNELS``, in eval mode."""
    config = net.HeadsConfig(channels=_CHANNELS)
    head = getattr(net, head_name)(config)
    head.eval()
    return head


def _final_conv(head: Any) -> torch.nn.Conv2d:
    """The head's 461-plane convolution, found by shape rather than by name."""
    named: list[tuple[str, torch.nn.Conv2d]] = [
        (name, module)
        for name, module in head.named_modules()
        if isinstance(module, torch.nn.Conv2d) and module.out_channels == POLICY_PLANES
    ]
    assert len(named) == 1, (
        f"expected exactly one Conv2d with out_channels == {POLICY_PLANES} in the "
        f"policy head, found {[name for name, _ in named]} -- the flatten "
        "contract says the head projects to 461 planes over 9x9 and flattens"
    )
    return named[0][1]


def _zeroed(tensor: torch.Tensor) -> torch.Tensor:
    """Zero a parameter in place (under ``no_grad``) and hand it back."""
    with torch.no_grad():
        tensor.zero_()
    return tensor


def _bias_of(conv: torch.nn.Conv2d) -> torch.Tensor:
    """The conv's bias, registering a zero one if it was built without.

    A bias-free final conv is a legitimate implementation choice; adding one
    here changes nothing about the head's own behaviour (it is zeroed anyway)
    and keeps the plane probe available.
    """
    if conv.bias is None:
        conv.bias = torch.nn.Parameter(torch.zeros(conv.out_channels))
    bias = conv.bias
    assert bias is not None
    return bias


def _forward(head: Any, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out: torch.Tensor = head(inputs)
    return out


def _dummy_input(batch: int = 1) -> torch.Tensor:
    """A head-shaped input. Its VALUES never matter in this module."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(105)
    return torch.randn(batch, _CHANNELS, BOARD, BOARD, generator=generator)


_PIN_IDS = [f"{pin.block}-{pin.action_id}" for pin in PINS]


# ---------------------------------------------------------------------------
# plane -> the 81 ids it owns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_name", _HEAD_NAMES)
@pytest.mark.parametrize("pin", PINS, ids=_PIN_IDS)
def test_a_planes_bias_lights_exactly_that_planes_81_ids(
    head_name: str, pin: action_ids.Pin
) -> None:
    """``bias[plane] = 1`` must show up at ids ``[plane * 81, (plane + 1) * 81)``.

    Including the pinned id itself, which is the whole point: id 19553 is
    plane 241 square 32 by hand-derived arithmetic off the engine's block
    table, so a head whose plane 241 does not own that id is mis-indexed
    however plausible its outputs look.
    """
    head = _policy_head(head_name)
    conv = _final_conv(head)
    _zeroed(conv.weight)
    bias = _zeroed(_bias_of(conv))
    with torch.no_grad():
        bias[pin.plane] = 1.0

    logits = _forward(head, _dummy_input(batch=2))
    assert logits.shape == (2, NUM_ACTIONS), (
        f"{head_name} must emit (N, {NUM_ACTIONS}) raw logits"
    )
    assert float(logits[0, pin.action_id].item()) == 1.0, (
        f"{head_name}: id {pin.action_id} ({pin.meaning}) is not on plane "
        f"{pin.plane}"
    )

    own = logits[:, pin.plane * SQUARES : (pin.plane + 1) * SQUARES]
    assert bool(torch.equal(own, torch.ones_like(own))), (
        f"{head_name}: plane {pin.plane} does not own 81 CONTIGUOUS ids -- the "
        "flatten is not plane-major"
    )

    elsewhere = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    elsewhere[pin.plane * SQUARES : (pin.plane + 1) * SQUARES] = False
    rest = logits[:, elsewhere]
    assert bool(torch.equal(rest, torch.zeros_like(rest))), (
        f"{head_name}: plane {pin.plane}'s bias leaked outside its own 81 ids"
    )


# ---------------------------------------------------------------------------
# square -> the offset inside the plane
# ---------------------------------------------------------------------------

#: ``(plane, square)`` probes.  Squares 1 and 9 are the row-major/column-major
#: discriminator -- ``sq_index`` runs along a rank first, so square 1 is
#: ``(row 0, col 1)`` and square 9 is ``(row 1, col 0)``; a column-major
#: flatten swaps exactly these two.  The rest walk the corners and the
#: worked example's plane.
_SQUARE_PROBES: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (0, 9),
    (0, 80),
    (241, 32),
    (241, 40),
    (POLICY_PLANES - 1, 80),
]


@pytest.mark.parametrize("head_name", _HEAD_NAMES)
@pytest.mark.parametrize(
    ("plane", "square"),
    _SQUARE_PROBES,
    ids=[f"p{plane}-sq{square}" for plane, square in _SQUARE_PROBES],
)
def test_a_hot_square_lands_at_plane_times_81_plus_its_square_index(
    head_name: str, plane: int, square: int
) -> None:
    """One hot board square -> exactly one hot logit, at ``plane * 81 + sq``."""
    head = _policy_head(head_name)
    conv = _final_conv(head)
    kernel_h, kernel_w = conv.kernel_size
    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, (
        f"even kernel {conv.kernel_size} has no centre tap to probe with"
    )

    _zeroed(conv.weight)
    _zeroed(_bias_of(conv))
    with torch.no_grad():
        conv.weight[plane, 0, kernel_h // 2, kernel_w // 2] = 1.0

    row, col = action_ids.spatial_of_sq(square)
    crafted = torch.zeros(1, conv.in_channels, BOARD, BOARD)
    crafted[0, 0, row, col] = 1.0

    def substitute(
        module: torch.nn.Module, args: tuple[Any, ...]
    ) -> tuple[torch.Tensor]:
        """Feed the final conv ``crafted`` whatever the head handed it."""
        return (crafted,)

    handle = conv.register_forward_pre_hook(substitute)
    try:
        logits = _forward(head, _dummy_input())
    finally:
        handle.remove()

    assert logits.shape == (1, NUM_ACTIONS)
    expected = plane * SQUARES + square
    assert expected == action_ids.plane_and_square_to_id(plane, square)

    hot = torch.nonzero(logits[0], as_tuple=False).flatten().tolist()
    assert hot == [expected], (
        f"{head_name}: a hot square at (row {row}, col {col}) of plane {plane} "
        f"should be logit {expected} (= {plane} * 81 + sq_index {square}); the "
        f"non-zero logits are {hot}"
    )
    assert float(logits[0, expected].item()) == 1.0


@pytest.mark.parametrize("head_name", _HEAD_NAMES)
def test_the_whole_plane_maps_square_for_square(head_name: str) -> None:
    """All 81 squares of one plane at once: the map is the identity, not a shuffle.

    The probes above pin the discriminating squares; this pins the entire
    permutation of one plane in a single forward, so a flatten that is
    row-major *except* for some transposed sub-block cannot hide between them.
    """
    head = _policy_head(head_name)
    conv = _final_conv(head)
    kernel_h, kernel_w = conv.kernel_size
    _zeroed(conv.weight)
    _zeroed(_bias_of(conv))
    plane = 241
    with torch.no_grad():
        conv.weight[plane, 0, kernel_h // 2, kernel_w // 2] = 1.0

    # A distinct value per square: value(sq) = sq + 1, so the logit vector of
    # the plane must come back as 1..81 in square order.
    crafted = torch.zeros(1, conv.in_channels, BOARD, BOARD)
    for square in range(SQUARES):
        row, col = action_ids.spatial_of_sq(square)
        crafted[0, 0, row, col] = float(square + 1)

    def substitute(
        module: torch.nn.Module, args: tuple[Any, ...]
    ) -> tuple[torch.Tensor]:
        return (crafted,)

    handle = conv.register_forward_pre_hook(substitute)
    try:
        logits = _forward(head, _dummy_input())
    finally:
        handle.remove()

    got = logits[0, plane * SQUARES : (plane + 1) * SQUARES]
    expected = torch.arange(1, SQUARES + 1, dtype=got.dtype)
    assert bool(torch.equal(got, expected)), (
        f"{head_name}: plane {plane}'s 81 logits are not its 81 squares in "
        f"sq_index order -- got {got.tolist()[:12]}..."
    )
