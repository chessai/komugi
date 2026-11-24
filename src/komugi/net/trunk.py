"""The trunk: a KataGo-generation nested-bottleneck ResNet, config-driven.

This module is the representation half of the network. It maps an encoded
position ``(N, in_planes, 9, 9)`` to a feature stack ``(N, channels, 9, 9)``
and stops there; the policy and value heads are a separate concern and a
separate module, which is what lets a transformer trunk be slotted in later
without touching them.

Four KataGo-generation choices are implemented here, and each one is a
decision rather than a default:

**Nested bottleneck blocks.** A plain residual block spends its parameters on
two full-width 3x3 convolutions. A nested bottleneck halves the width first,
does *two* 3x3s down there, and expands back -- roughly the same parameter
count buys twice the spatial mixing per block. KataGo measured this as
matching much heavier plain nets at better speed.

**Global-pooling bias every N-th block.** A stack of 3x3 convolutions grows
its receptive field one square per convolution; on a 9x9 board that is enough
to see the whole board eventually, but only through a deep, heavily
attenuated path. A pooling bias hands every square the board-wide mean and
maximum of each feature in one step. See :class:`_GlobalPoolingBias`.

**One batch norm per block.** Not one per convolution. The single
normalisation sits at the block entry, before the activation, in the
pre-activation arrangement -- so the residual stream itself is never
normalised and stays an unobstructed identity path from the stem to the
output.

**Fixed-variance initialisation.** Every weight is drawn from a zero-mean
normal whose standard deviation is fixed by fan-in, and the last layer of
each residual branch is damped by ``1/sqrt(blocks)``. See
`Initialisation`_ below.

Exact block ordering
--------------------
For a block with input ``x`` of shape ``(N, C, H, W)``, writing ``M = C // 2``
for the bottleneck width::

    h = Mish(BatchNorm2d(x))            # the one and only normalisation
    h = Conv1x1(C -> M)(h)
    h = Mish(h)
    h = Conv3x3(M -> M)(h)              # nested pair, first
    h = Mish(h)
    h = Conv3x3(M -> M)(h)              # nested pair, second
    h = h + gpool_bias(h)               # designated blocks only
    h = Mish(h)
    h = Conv1x1(M -> C)(h)
    out = x + h                         # residual add, no activation after

No convolution carries a bias term: the block's only additive freedom is the
batch-norm's own ``beta`` (plus the pooling bias where present), which is
what "one batch norm" means in practice.

.. _Initialisation:

Initialisation
--------------
Every convolution and linear weight is drawn ``Normal(0, sigma)`` with

    ``sigma = gain / sqrt(fan_in)``

and ``fan_in = in_channels * kh * kw`` for a convolution, ``in_features`` for
a linear. Biases are zero; batch-norm ``gamma`` is 1 and ``beta`` is 0.

The gains, and where each comes from:

* **After a Mish** (the three interior convolutions and the expand):
  ``gain = 1 / sqrt(E[Mish(z)^2])`` for ``z ~ N(0, 1)``, which is the exact
  He construction with ReLU's second moment replaced by Mish's. Numerical
  quadrature puts ``E[Mish(z)^2]`` at ``~0.4526`` (ReLU's is ``0.5``), so the
  gain is ``~1.486`` where He/ReLU would use ``sqrt(2) ~ 1.414``. Only the
  order of magnitude matters -- the property being bought is that the second
  moment neither collapses nor explodes over depth, not any exact constant.
* **The stem**: ``gain = 1``. Its input is the encoded position, which is
  already roughly unit second moment and is emphatically not the output of an
  activation, so the Mish correction would be a factor applied to nothing.
* **The pooling linear**: a small gain, see :class:`_GlobalPoolingBias`.
* **The expand convolution** additionally carries ``1 / sqrt(blocks)``. It is
  the last layer of the residual branch, so this scales the whole branch:
  each of the ``B`` blocks then contributes about ``1/B`` of the incoming
  variance to the residual stream, and the stream's second moment stays
  ``O(1)`` at any depth instead of compounding. A ``b10c128`` and a
  ``b18c256`` trunk come out of the box with the same output scale.

Everything above is a statement about *initialisation*. Training moves the
weights wherever the gradient says; the point is only that the first forward
pass through an untrained trunk of any configured depth produces activations
of a sane magnitude, so the first gradients are informative rather than
saturated or vanished.

Device and RNG
--------------
Nothing here calls ``.cuda()`` or names a device: a trunk is built on CPU and
the caller moves it. Construction seeds every draw from a local
:class:`torch.Generator` and restores the global torch RNG state on the way
out, so building a trunk is invisible to -- and unaffected by -- whatever
seeding the surrounding process is doing.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch
from torch import nn

from komugi.net.config import TrunkConfig

# E[Mish(z)^2] for z ~ N(0, 1), by numerical quadrature. ReLU's is 0.5, so
# Mish is very slightly the more attenuating of the two.
_MISH_SECOND_MOMENT = 0.4526

# He-style gain for a layer whose input is a Mish activation: undo exactly the
# attenuation the activation applied.
_MISH_GAIN = 1.0 / math.sqrt(_MISH_SECOND_MOMENT)

# The stem reads the encoded position, not an activation.
_STEM_GAIN = 1.0

# Second moment of the pooled statistics relative to the features they are
# pooled from. The mean over 81 squares is small (the squares partially
# cancel), but the *maximum* over 81 roughly-standard-normal squares sits near
# 2.2, so the concatenated (mean, max) vector averages well above the unit
# second moment a bare 1/sqrt(fan_in) would assume.
_POOLED_SECOND_MOMENT = 2.5

# How much variance the pooling bias is meant to add to the features it
# biases. A quarter: enough that the global summary genuinely moves every
# square, small enough that it modulates the local features rather than
# drowning them.
_GPOOL_BIAS_FRACTION = 0.25

_GPOOL_GAIN = math.sqrt(_GPOOL_BIAS_FRACTION / _POOLED_SECOND_MOMENT)


@contextlib.contextmanager
def _undisturbed_global_rng() -> Iterator[None]:
    """Run the body, then put the global torch RNG back where it was.

    ``nn.Conv2d(...)`` and friends draw their default initialisation from the
    global generator. Those draws are thrown away here -- every parameter is
    overwritten from a config-seeded local generator -- but the *advance* they
    cause would still be visible to the caller, which would make constructing
    a trunk silently change the next sample drawn anywhere else in the
    process. Snapshot and restore, and it does not.
    """
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _fill_parameter(module: nn.Module, name: str, value: float) -> None:
    """Set `module`'s `name` parameter to `value` if it has one.

    ``getattr`` rather than attribute access because the parameters this is
    used for -- a convolution's optional bias, a batch norm's affine pair --
    are deliberately left undeclared by torch (see the comment on
    ``_NormBase``), so they are absent rather than ``None`` in the cases where
    the module was built without them.
    """
    parameter = getattr(module, name, None)
    if isinstance(parameter, torch.Tensor):
        with torch.no_grad():
            parameter.fill_(value)


def _init_fixed_variance(
    layer: nn.Conv2d | nn.Linear, *, gain: float, generator: torch.Generator
) -> None:
    """Draw `layer`'s weight ``Normal(0, gain / sqrt(fan_in))``, zero its bias.

    ``layer.weight[0].numel()`` is the fan-in of one output unit for both
    kinds of layer: ``in_channels * kh * kw`` for a convolution's
    ``(out, in, kh, kw)`` weight, ``in_features`` for a linear's
    ``(out, in)``.
    """
    fan_in = layer.weight[0].numel()
    sigma = gain / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.normal_(0.0, sigma, generator=generator)
    _fill_parameter(layer, "bias", 0.0)


class _GlobalPoolingBias(nn.Module):
    """Board-wide statistics turned into a per-channel bias.

    Takes the block's mid-features ``(N, M, H, W)``, reduces each channel over
    the board to its mean and its maximum, concatenates those into a ``2M``
    vector per sample, and passes that through a single linear layer to get
    one bias value per channel. The bias is returned shaped ``(N, M, 1, 1)``
    so that adding it broadcasts the same value over every square.

    Why both statistics: the mean answers "how much of this feature is on the
    board at all" (material, tempo, total pressure) and the maximum answers
    "is there at least one square where it is extreme" (a mate threat, a
    single unstoppable tower). Neither summarises the other, and a 3x3
    convolution stack can express neither cheaply.

    The consequence that makes this structure worth its parameters: after a
    pooling block, *every* output square depends on *every* input square.
    A blocks-worth of 3x3 convolutions can only reach two squares out.

    The linear is initialised with a deliberately small gain --
    ``sqrt(_GPOOL_BIAS_FRACTION / _POOLED_SECOND_MOMENT)``, about 0.32 --
    because its input is not a unit-second-moment activation: the max
    statistic alone is around 2.2 in magnitude. At a gain of 1 the bias would
    arrive with more variance than the features it biases, so at
    initialisation the trunk would be mostly reading a global summary and
    barely reading the board.
    """

    channels: int
    linear: nn.Linear

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(2 * channels, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the ``(N, M, 1, 1)`` bias implied by `features`."""
        flat = features.flatten(2)
        stats = torch.cat((flat.mean(dim=2), flat.amax(dim=2)), dim=1)
        bias: torch.Tensor = self.linear(stats)
        return bias.unsqueeze(-1).unsqueeze(-1)


class _NestedBottleneckBlock(nn.Module):
    """One pre-activation nested-bottleneck residual block.

    The exact layer ordering is in the module docstring and is the contract;
    the names here (``reduce``, ``inner_a``, ``inner_b``, ``expand``) are not.

    Args:
        channels: Block width ``C``, in and out.
        gpool: Whether this block carries the global-pooling bias.
    """

    norm: nn.BatchNorm2d
    act: nn.Mish
    reduce: nn.Conv2d
    inner_a: nn.Conv2d
    inner_b: nn.Conv2d
    expand: nn.Conv2d
    pool: _GlobalPoolingBias | None

    def __init__(self, channels: int, *, gpool: bool) -> None:
        super().__init__()
        mid = channels // 2
        self.norm = nn.BatchNorm2d(channels)
        # Mish is stateless, so one instance serves all four activation sites.
        self.act = nn.Mish()
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.inner_a = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.inner_b = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.expand = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.pool = _GlobalPoolingBias(mid) if gpool else None

    def initialise_parameters(
        self, *, residual_scale: float, generator: torch.Generator
    ) -> None:
        """Draw every parameter of this block from `generator`.

        `residual_scale` is the ``1/sqrt(blocks)`` damping applied to the last
        layer of the branch -- the expand convolution -- and hence to the
        block's whole contribution to the residual stream.

        The draw order is fixed, so the parameters a block receives depend
        only on the configuration and on how many blocks were initialised
        before it.
        """
        _init_fixed_variance(self.reduce, gain=_MISH_GAIN, generator=generator)
        _init_fixed_variance(self.inner_a, gain=_MISH_GAIN, generator=generator)
        _init_fixed_variance(self.inner_b, gain=_MISH_GAIN, generator=generator)
        if self.pool is not None:
            _init_fixed_variance(
                self.pool.linear, gain=_GPOOL_GAIN, generator=generator
            )
        _init_fixed_variance(
            self.expand, gain=_MISH_GAIN * residual_scale, generator=generator
        )
        # Identity affine. Already the constructor's default; stated here so
        # that "the initialisation is all in one place" stays true.
        _fill_parameter(self.norm, "weight", 1.0)
        _fill_parameter(self.norm, "bias", 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x + branch(x)``, both shaped ``(N, C, H, W)``."""
        h: torch.Tensor = self.act(self.norm(x))
        h = self.act(self.reduce(h))
        h = self.act(self.inner_a(h))
        h = self.inner_b(h)
        if self.pool is not None:
            # Before the expand, so the global summary is mixed back into the
            # full-width features by a learned projection rather than added
            # raw to the residual stream.
            h = h + self.pool(h)
        h = self.expand(self.act(h))
        return x + h


class Trunk(nn.Module):
    """The b10c128 (by default) nested-bottleneck trunk.

    ``forward`` maps ``(N, in_planes, 9, 9)`` float32 to
    ``(N, channels, 9, 9)`` float32. The output is the raw residual stream:
    no final normalisation and no final activation, because the heads that
    consume it each want their own, and normalising here would throw away the
    scale information the value head is entitled to see.

    Args:
        config: The shape to build. Stored unchanged as :attr:`config`.
    """

    config: TrunkConfig
    stem: nn.Conv2d
    blocks: nn.ModuleList

    def __init__(self, config: TrunkConfig) -> None:
        super().__init__()
        self.config = config

        with _undisturbed_global_rng():
            generator = torch.Generator()
            generator.manual_seed(config.seed)

            self.stem = nn.Conv2d(
                config.in_planes,
                config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            _init_fixed_variance(self.stem, gain=_STEM_GAIN, generator=generator)

            # Built into a plain list first so that each block can be reached
            # at its own type for initialisation; nn.ModuleList hands its
            # members back as bare nn.Module.
            residual_scale = 1.0 / math.sqrt(config.blocks)
            built: list[_NestedBottleneckBlock] = []
            for index in range(1, config.blocks + 1):
                block = _NestedBottleneckBlock(
                    config.channels, gpool=config.is_gpool_block(index)
                )
                block.initialise_parameters(
                    residual_scale=residual_scale, generator=generator
                )
                built.append(block)
            self.blocks = nn.ModuleList(built)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the trunk features for the batch of encoded positions `x`."""
        out: torch.Tensor = self.stem(x)
        for block in self.blocks:
            out = block(out)
        return out
