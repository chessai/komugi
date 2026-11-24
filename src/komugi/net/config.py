"""Configuration for the network trunk: its shape, and nothing else.

The point of :class:`TrunkConfig` is that the v2.1 scale target -- b18c256,
the ~40M-parameter shape KataGo found matches much heavier plain nets -- is a
*config change* and not a code change. Nothing in :mod:`komugi.net.trunk`
hardcodes 10 blocks, 128 channels, or the 128 input planes the observation
encoder happens to emit today.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrunkConfig:
    """The shape of a :class:`~komugi.net.trunk.Trunk`.

    Args:
        in_planes: Number of input planes ``C_in`` the trunk's stem consumes.
            The observation encoder currently emits 128 of them, but that is
            the encoder's business: this field has no default precisely so
            that a change over there cannot silently disagree with a stale
            constant over here.
        channels: Trunk width ``C``. Every residual block reads and writes
            ``C`` channels and works internally at ``C // 2``, so ``C`` must
            be even.
        blocks: Number of nested-bottleneck residual blocks. Also the depth
            the residual-branch initialisation is damped by (``1/sqrt(blocks)``,
            see :mod:`komugi.net.trunk`), which is why the trunk needs it as a
            number and not merely as a loop bound.
        gpool_every: Every ``gpool_every``-th block, counting blocks from 1,
            carries the global-pooling bias structure -- blocks 3, 6 and 9 of
            a 10-block trunk at the default. A value larger than
            :attr:`blocks` is legal and means "no global pooling at all",
            which is the configuration the receptive field of a plain
            convolutional trunk is measured against.
        seed: Seed for parameter initialisation. Construction is a pure
            function of the config: two trunks built from equal configs have
            bit-identical ``state_dict``s, and building one neither reads nor
            advances the global torch RNG.
    """

    in_planes: int
    channels: int = 128
    blocks: int = 10
    gpool_every: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject configurations that cannot describe a trunk."""
        if self.in_planes < 1:
            raise ValueError(f"in_planes must be >= 1, got {self.in_planes!r}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels!r}")
        if self.channels % 2 != 0:
            raise ValueError(
                f"channels must be even -- the nested bottleneck works at "
                f"channels // 2 -- got {self.channels!r}"
            )
        if self.blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {self.blocks!r}")
        if self.gpool_every < 1:
            raise ValueError(f"gpool_every must be >= 1, got {self.gpool_every!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed!r}")

    @property
    def bottleneck_channels(self) -> int:
        """Width ``C // 2`` of the block interior, where the 3x3s live."""
        return self.channels // 2

    def is_gpool_block(self, index: int) -> bool:
        """Return whether the `index`-th block (**1-based**) pools globally.

        One-based because "every third block" reads as blocks 3, 6, 9 -- the
        last block of each group of three, so a group's local features are
        computed before that group's global summary biases them.
        """
        return index % self.gpool_every == 0
