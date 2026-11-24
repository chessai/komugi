"""Torch-level probes shared by the trunk behaviour suite (KOM-104).

Nothing here imports ``komugi.net``: the helpers are pure torch, so they are
exercised unconditionally by ``test_trunk_probes.py`` even in a worktree where
the trunk (KOM-004) does not exist yet. That is deliberate -- the behaviour
tests are only as trustworthy as the arithmetic they lean on, and the two
pieces of arithmetic that can silently rot are

* the Chebyshev receptive-field bound used by the no-gpool control test, and
* the "which position changed" reduction used by the globality test.

Both are pinned empirically against real convolutions in the sanity suite.
"""

from __future__ import annotations

from typing import Any

import torch

# The board the trunk is defined over. The contract fixes the spatial shape of
# both the input and the output at 9x9, so this is not a tunable.
BOARD = 9


def receptive_radius(blocks: int, *, convs_per_block: int = 2, stem_convs: int = 1) -> int:
    """Chebyshev radius one input square can reach through a gpool-free trunk.

    Every 3x3 convolution with padding 1 widens a value's influence by exactly
    one square in each of the eight directions, and Chebyshev distance is the
    metric that composes additively under that widening. A trunk with no global
    pooling block is a pure stack of such convolutions -- ``stem_convs`` in the
    stem plus ``convs_per_block`` per residual block -- so the reachable set is
    a square of this radius around the perturbed square, and nothing outside it
    can move.

    The default ``convs_per_block=2`` counts the 3x3s in the block KOM-004
    actually merged: a plain bottleneck, ``BN -> Mish -> 1x1 reduce -> Mish ->
    3x3 -> Mish -> 3x3 -> [gpool bias] -> Mish -> 1x1 expand -> residual add``.
    There are no inner residual branches; the two 1x1s change width, not reach,
    so they contribute nothing to the radius.

    Review-driven tightening (union review F1): this used to default to 4,
    describing KataGo's nested block rather than ours, which made the no-gpool
    control assert bit-identity only beyond radius 5 (45 squares) instead of
    beyond radius 3 (65) -- a trunk that quietly grew its per-block reach to
    radius <= 5 would have passed. The constant is *counted from the
    implementation's 3x3s*, so it is deliberately coupled to the block: if the
    block ever gains a convolution, this default must grow with it, and until
    it does the widened reach fails the control. Over-estimating is the
    fail-safe direction (a more forgiving control), never a false red.
    """
    if blocks < 0 or convs_per_block < 0 or stem_convs < 0:
        raise ValueError("receptive_radius takes non-negative counts")
    return stem_convs + convs_per_block * blocks


def far_mask(row: int, col: int, radius: int, *, size: int = BOARD) -> torch.Tensor:
    """Boolean ``(size, size)`` mask: True where Chebyshev distance > ``radius``.

    These are the squares a gpool-free trunk provably cannot reach from
    ``(row, col)``, and therefore the squares whose outputs must come back
    bit-identical when only ``(row, col)`` is perturbed.
    """
    coords = torch.arange(size)
    rows = (coords - row).abs().unsqueeze(1).expand(size, size)
    cols = (coords - col).abs().unsqueeze(0).expand(size, size)
    distance = torch.maximum(rows, cols)
    return distance > radius


def position_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-square maximum absolute difference of two ``(N, C, 9, 9)`` tensors.

    Reducing over batch and channel answers the only question the globality
    test asks -- did *anything* at this square move -- without caring which
    channel carried the news.
    """
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    if left.dim() != 4:
        raise ValueError(f"expected (N, C, H, W), got {tuple(left.shape)}")
    return (left - right).abs().amax(dim=(0, 1))


def noise(
    shape: tuple[int, ...], *, seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Unit-variance Gaussian noise from a LOCAL generator.

    Tests must not perturb the global torch RNG, because one of the properties
    under test is that constructing a trunk leaves that RNG alone; a fixture
    that consumed it would mask the very failure it is meant to catch.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=dtype)


def perturbed(tensor: torch.Tensor, *, sample: int, row: int, col: int, delta: float) -> torch.Tensor:
    """Copy of ``tensor`` with one square of one sample shifted by ``delta``.

    Every input plane at that square moves together: the trunk's stem mixes
    planes, so shifting a single plane risks landing in the null space of the
    stem's first convolution and reading as "no perturbation".
    """
    out = tensor.clone()
    out[sample, :, row, col] += delta
    return out


def build_trunk(net: Any, **config_kwargs: Any) -> tuple[Any, Any]:
    """Construct ``(config, trunk)`` from ``komugi.net``, trunk left in eval mode.

    ``net`` is the ``komugi.net`` module (``ModuleType``, hence ``Any``-typed
    attributes); the sanity suite hands it a stand-in with the same two names.

    Eval mode is the default posture for every behaviour test here: it is the
    mode with defined per-sample semantics, and at init a fresh BatchNorm has
    running mean 0 / running var 1, so eval normalisation is the identity up to
    the affine parameters -- which keeps the init-statistics assertions about
    the trunk's own scaling rather than about accumulated batch statistics.
    """
    config = net.TrunkConfig(**config_kwargs)
    trunk = net.Trunk(config)
    trunk.eval()
    return config, trunk


def forward_no_grad(trunk: Any, inputs: torch.Tensor) -> torch.Tensor:
    """Run ``trunk`` under ``no_grad`` and hand back a typed tensor."""
    with torch.no_grad():
        out: torch.Tensor = trunk(inputs)
    return out


def state_dict_of(module: Any) -> dict[str, torch.Tensor]:
    """``state_dict()`` narrowed to the mapping the comparisons want."""
    raw: dict[str, torch.Tensor] = dict(module.state_dict())
    return raw


def differing_keys(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> list[str]:
    """Keys whose tensors are not bit-identical (missing keys count as differing)."""
    keys = sorted(set(left) | set(right))
    out: list[str] = []
    for key in keys:
        if key not in left or key not in right:
            out.append(key)
        elif not torch.equal(left[key], right[key]):
            out.append(key)
    return out


class StdRecorder:
    """Records the std of every BatchNorm2d input, in forward order.

    The contract guarantees one BatchNorm per residual block but pins no
    submodule names, so the BatchNorms are the only structure-independent
    probe into the residual stream. Their inputs are what the block is handed,
    which is exactly the quantity whose growth the branch scaling is supposed
    to bound.
    """

    def __init__(self) -> None:
        self.stds: list[float] = []
        self.channels: list[int] = []
        self._handles: list[Any] = []

    def attach(self, module: Any) -> StdRecorder:
        """Hook every BatchNorm2d under ``module``; returns self for chaining."""
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.BatchNorm2d):
                self._handles.append(submodule.register_forward_pre_hook(self._hook))
        return self

    def _hook(self, module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        tensor = inputs[0]
        if not isinstance(tensor, torch.Tensor):
            return
        self.stds.append(float(tensor.detach().float().std().item()))
        self.channels.append(int(tensor.shape[1]))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> StdRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    def trunk_width_stds(self, channels: int) -> list[float]:
        """Recorded stds for tensors at full trunk width.

        A bottleneck block may normalise a narrowed tensor; comparing those to
        full-width ones would measure the bottleneck ratio rather than residual
        growth. If no recording is at trunk width the caller gets everything
        back, which is still an ordered walk down the same stream.
        """
        widths = [std for std, width in zip(self.stds, self.channels) if width == channels]
        return widths if len(widths) >= 2 else list(self.stds)


def parameter_count(module: Any) -> int:
    """Total number of parameter elements, trainable or not."""
    return sum(int(p.numel()) for p in module.parameters())
