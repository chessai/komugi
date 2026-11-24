"""The komugi network: a KataGo-generation convolutional trunk, plus heads.

Three layers, each a module of its own and each usable without the others:
:class:`TrunkConfig`/:class:`Trunk` are the representation, the eight heads
(the P0a core :class:`PolicyHead`, :class:`SoftPolicyHead`, :class:`WDLHead`
and :class:`MLHHead`, plus the staged :class:`UncertaintyHead`,
:class:`OwnershipHead`, :class:`ShortTermValueHead` and
:class:`DraftOutcomeHead`, all shaped by :class:`HeadsConfig`) are the
readouts, and :class:`NetConfig`/:class:`KomugiNet` assemble one of each into
the network that self-play actually evaluates, returning a :class:`NetOutput`.

The seam between trunk and heads is the ``(N, channels, 9, 9)`` feature stack
and nothing else, which is what keeps a future transformer trunk a drop-in
replacement rather than a rewrite. The staged heads of ``docs/DESIGN.html``
§2.4 -- uncertainty, final-control, short-term value, draft-outcome -- are
**present and forward-evaluated, with their losses off**: the net freeze makes
adding a head later an act that invalidates every earlier checkpoint, so what
stages is each head's loss weight, not its parameters. See `StagingRule`_ in
:mod:`komugi.net.heads`.

Nothing here is device-aware: modules are built on CPU and the caller moves
them.
"""

from __future__ import annotations

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
from komugi.net.model import KomugiNet, NetConfig, NetOutput
from komugi.net.trunk import Trunk

__all__ = [
    "DraftOutcomeHead",
    "HeadsConfig",
    "KomugiNet",
    "MLHHead",
    "NetConfig",
    "NetOutput",
    "OwnershipHead",
    "PolicyHead",
    "ShortTermValueHead",
    "SoftPolicyHead",
    "Trunk",
    "TrunkConfig",
    "UncertaintyHead",
    "WDLHead",
]
