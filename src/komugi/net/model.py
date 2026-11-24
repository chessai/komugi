"""The assembled network: one trunk, eight heads, one forward pass.

:class:`KomugiNet` is the whole of the network and almost none of its
substance -- the representation lives in :mod:`komugi.net.trunk` and the
readouts in :mod:`komugi.net.heads`. What this module contributes is the two
things that only exist once the pieces are put together: the *agreement*
between trunk width and head width, checked at config time rather than
discovered as a shape error in a forward pass, and a named output tuple so
that callers unpack eight predictions by name instead of by position.

The trunk is evaluated exactly once per batch and its features are handed to
all eight heads. That is the entire point of the architecture: the expensive
part is shared, and adding a head costs one small readout rather than one more
network.

Four of the eight are the P0a core and four are the staged heads, present and
forward-evaluated from the first P0b net with their *losses* off -- see
`StagingRule`_ in :mod:`komugi.net.heads` for why the net freeze forces that.

Nothing here is device-aware; a net is built on CPU and the caller moves it.
Construction is a pure function of :class:`NetConfig` and leaves the global
torch RNG exactly as it found it -- both properties inherited from the trunk
and the heads, which each seed their own local generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from komugi.net.config import TrunkConfig
from komugi.net.heads import (
    DraftOutcomeHead,
    HeadsConfig,
    MLHHead,
    OwnershipHead,
    PolicyHead,
    ShortTermValueHead,
    SoftPolicyHead,
    UncertaintyHead,
    WDLHead,
)
from komugi.net.trunk import Trunk


@dataclass(frozen=True)
class NetConfig:
    """The shape of a :class:`KomugiNet`: a trunk shape and a head shape.

    The two are separate objects rather than one flat config because they are
    separately meaningful -- a policy-head A/B (§2.3) varies the heads at a
    fixed trunk, and the b18c256 scale-up varies the trunk at fixed heads --
    and they are validated against each other here because exactly one field
    is shared between them.

    Args:
        trunk: Shape of the representation half.
        heads: Shape of the readout half. Its ``channels`` must equal the
            trunk's: the heads read the trunk's output, and a mismatch is a
            configuration error, not a resizing request.
    """

    trunk: TrunkConfig
    heads: HeadsConfig

    def __post_init__(self) -> None:
        """Reject a head shape that does not fit the trunk it reads."""
        if self.heads.channels != self.trunk.channels:
            raise ValueError(
                f"heads.channels must equal trunk.channels -- the heads read "
                f"the trunk's output -- got heads.channels="
                f"{self.heads.channels!r} and trunk.channels="
                f"{self.trunk.channels!r}"
            )


class NetOutput(NamedTuple):
    """Everything one forward pass predicts, for a batch of ``N`` positions.

    A :class:`~typing.NamedTuple` rather than a dict: the set of heads is
    fixed by the net freeze and known statically, so the type checker can see a
    misspelled field, and a caller can bind a prediction by name.

    Every field is raw. No softmax, no legality mask, no utility scaling; see
    the individual heads for why each one stops where it does.

    **Eight required fields, and the four staged ones take no defaults**
    (KOM-005b D18). This is a required 8-tuple: an exact-arity unpack such as
    ``_, _, wdl, _ = net(x)`` raises ``ValueError``, and constructing one with
    four keyword arguments raises ``TypeError``. Both are deliberate. A default
    would let a fake net silently omit a head and hand
    :func:`komugi.train.wiring.outputs_from_net` a ``None``-shaped hole,
    converting a loud failure at the seam into an untrained head nobody
    notices. The inference seam is unaffected for a different reason:
    ``komugi.selfplay.inference._NetOutputLike`` is a structural ``Protocol``
    over three read-only properties, which an eight-field output satisfies
    unchanged.

    **Appended, never inserted.** Field order is the documented contract and
    is asserted by two suites; the staged fields go at the end, in
    ``DESIGN.html`` §2.4's normative staging order (uncertainty ->
    final-control -> short-term value -> draft-outcome).

    Attributes:
        policy_logits: ``(N, num_actions)``. Index is the engine's action id
            exactly -- see the policy plane ordering section of
            :mod:`komugi.net.heads`.
        soft_policy_logits: ``(N, num_actions)``, same indexing, independent
            parameters. Trained against KataGo's sharpened auxiliary target.
        wdl_logits: ``(N, 3)`` in (win, draw, loss) order, from the
            side-to-move's perspective.
        moves_left: ``(N,)`` on the raw moves-left scale.
        uncertainty: ``(N,)`` raw, pre-``softplus``. Not the predicted squared
            error; the loss applies the activation.
        ownership_logits: ``(N, 81, 3)`` per-square control logits, square
            index ``row * 9 + col`` and classes (mover, opponent, empty).
        short_term_value: ``(N, 3)`` raw values, one per horizon of
            :data:`komugi.selfplay.records.SHORT_TERM_HORIZONS`.
        draft_outcome_logits: ``(N, 3)`` in the same (win, draw, loss) order
            as `wdl_logits`.
    """

    policy_logits: torch.Tensor
    soft_policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor
    uncertainty: torch.Tensor
    ownership_logits: torch.Tensor
    short_term_value: torch.Tensor
    draft_outcome_logits: torch.Tensor


class KomugiNet(nn.Module):
    """Trunk plus the four P0a core heads and the four staged heads.

    ``forward`` maps ``(N, in_planes, 9, 9)`` float32 to a :class:`NetOutput`.

    The submodule names -- ``trunk``, ``policy``, ``soft_policy``, ``wdl``,
    ``mlh``, ``uncertainty``, ``ownership``, ``short_term_value``,
    ``draft_outcome`` -- are part of the ``state_dict`` and so are load-bearing
    for checkpoints; the order in which they are constructed is not, because
    each seeds its own generator. The four staged names are why they are all
    constructed now: see `StagingRule`_ in :mod:`komugi.net.heads`.

    Args:
        config: The shape to build. Stored unchanged as :attr:`config`.
    """

    config: NetConfig
    trunk: Trunk
    policy: PolicyHead
    soft_policy: SoftPolicyHead
    wdl: WDLHead
    mlh: MLHHead
    uncertainty: UncertaintyHead
    ownership: OwnershipHead
    short_term_value: ShortTermValueHead
    draft_outcome: DraftOutcomeHead

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = Trunk(config.trunk)
        self.policy = PolicyHead(config.heads)
        self.soft_policy = SoftPolicyHead(config.heads)
        self.wdl = WDLHead(config.heads)
        self.mlh = MLHHead(config.heads)
        self.uncertainty = UncertaintyHead(config.heads)
        self.ownership = OwnershipHead(config.heads)
        self.short_term_value = ShortTermValueHead(config.heads)
        self.draft_outcome = DraftOutcomeHead(config.heads)

    def forward(self, x: torch.Tensor) -> NetOutput:
        """Return every head's prediction for the batch of positions `x`."""
        features: torch.Tensor = self.trunk(x)
        return NetOutput(
            policy_logits=self.policy(features),
            soft_policy_logits=self.soft_policy(features),
            wdl_logits=self.wdl(features),
            moves_left=self.mlh(features),
            uncertainty=self.uncertainty(features),
            ownership_logits=self.ownership(features),
            short_term_value=self.short_term_value(features),
            draft_outcome_logits=self.draft_outcome(features),
        )
