"""The serving-net factory the orchestrator is allowed to be handed.

:func:`~komugi.selfplay.orchestrate.run_selfplay` does not take a model. It
takes a **module-level, no-argument callable** and calls it inside the server
process it spawned (design D2/F18,
:func:`~komugi.selfplay.orchestrate.require_module_level_factory`): spawn
pickles the factory by qualified name, so a closure over "this iteration's
checkpoint" or a :func:`functools.partial` binding a path is exactly what
cannot cross. That is not an inconvenience to route around -- it is the reason
a model never travels as pickled bytes -- so the question the campaign loop has
to answer is how a factory with **no parameters** learns which checkpoint this
iteration serves.

The environment is the channel, and it is not a novelty here
--------------------------------------------------------------
The loop sets :data:`CHECKPOINT_ENV` (and :data:`DEVICE_ENV`,
:data:`NET_ENV`) in its own process before calling ``run_selfplay``;
``spawn`` hands the child a copy of ``os.environ``, so every server and worker
this iteration starts reads the same values, and the factory below is a pure
function of them. The precedent is the one the self-play suite already proved
across a spawn boundary -- the counter-file rig in ``tests/selfplay`` passes
its path the same way -- and the D7 shape-check seam arms itself from the
environment at child import time for the same reason.

Consequences worth stating, because an environment variable is a global:

* The values are read **at call time, inside the child**, never captured. A
  loop that changes :data:`CHECKPOINT_ENV` between iterations gets a different
  net next iteration with no other coordination.
* Unset or empty :data:`CHECKPOINT_ENV` means *fresh initialisation*, which is
  exactly iteration 0. There is no "missing checkpoint" fallback: a
  :data:`CHECKPOINT_ENV` that names a directory holding nothing loadable
  raises, because a run that silently served random weights for an hour is the
  failure this project spends its checkpoint digests to avoid.
* One factory, fully configured by the environment. A second entry point
  (``serving_net_small``, ``serving_net_cuda``, ...) per axis would multiply,
  and the orchestrator pins the *callable*, so every axis has to fit through
  this one door anyway. :func:`small_serving_net` exists as its own name only
  because a suite wants to spell "the tiny shape" directly; it is the same
  function with :data:`NET_ENV` forced, and :func:`serving_net` delegates to it
  when the environment asks.

Device placement is a *net* decision (mirroring the inference server's "no
device literals" rule and :meth:`komugi.train.step.TrainStep.run`): this module
is the one place that reads :data:`DEVICE_ENV` and calls ``.to``. Default cpu,
because CI has no GPU; the desktop run sets ``KOMUGI_CAMPAIGN_DEVICE=cuda``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch

from komugi.engine.observation import observation_size
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.train.checkpoint import (
    load_checkpoint,
    load_checkpoint_file,
    require_matching_plane_count,
)

CHECKPOINT_ENV: Final[str] = "KOMUGI_CAMPAIGN_CKPT"
"""Environment variable naming the checkpoint this iteration serves.

Either a checkpoint **file** as
:func:`~komugi.train.checkpoint.save_checkpoint` published it -- served
*exactly*, never upgraded to a newer sibling, so pinning an older checkpoint
(a rollback) means what it says -- or the **directory** holding them, which
serves the newest one that survives digest verification. Verification is
:mod:`komugi.train.checkpoint`'s either way; it is the only code allowed to
decide what "loadable" means. Empty or unset means "freshly initialised
weights" -- iteration 0.
"""

DEVICE_ENV: Final[str] = "KOMUGI_CAMPAIGN_DEVICE"
"""Environment variable choosing the serving device: ``cuda``, or CPU."""

NET_ENV: Final[str] = "KOMUGI_CAMPAIGN_NET"
"""Environment variable choosing the net shape: :data:`SMALL_NET`, or b10c128."""

SMALL_NET: Final[str] = "small"
"""Value of :data:`NET_ENV` selecting the tiny shape (:data:`SMALL_CONFIG`)."""

CUDA: Final[str] = "cuda"
"""Value of :data:`DEVICE_ENV` selecting the GPU. Anything else is the CPU."""

IN_PLANES: Final[int] = observation_size()[0]
"""The net's input plane count, READ FROM THE ENCODER, never restated.

It was a literal until KOM-002, which is how it came to be the only hardcoded
plane count in ``src/``: a number that has to be edited in lockstep with the
encoder is a number that eventually is not.
"""

NUM_ACTIONS: Final[int] = 37341
POLICY_PLANES: Final[int] = 461

CONFIG: Final[NetConfig] = NetConfig(
    trunk=TrunkConfig(
        in_planes=IN_PLANES, channels=128, blocks=10, gpool_every=3, seed=0
    ),
    heads=HeadsConfig(
        channels=128,
        num_actions=NUM_ACTIONS,
        policy_planes=POLICY_PLANES,
        seed=0,
    ),
)
"""The P0a campaign net: b10c128, the design §2.1 shape the first run trains."""

SMALL_CONFIG: Final[NetConfig] = NetConfig(
    trunk=TrunkConfig(
        in_planes=IN_PLANES, channels=16, blocks=2, gpool_every=3, seed=0
    ),
    heads=HeadsConfig(
        channels=16,
        num_actions=NUM_ACTIONS,
        policy_planes=POLICY_PLANES,
        seed=0,
    ),
)
"""A b2c16 net for tests: the *trunk* shrinks, the seam does not.

``in_planes`` and ``num_actions`` stay at the production values on purpose --
they are the inference seam's shapes
(:class:`~komugi.selfplay.inference.InferenceConfig`), and a factory that
narrowed them would produce a net the real workers cannot talk to, which is
the one property a behavioural test of the campaign loop must not fake.
"""


def net_config() -> NetConfig:
    """Return the shape :data:`NET_ENV` selects, without building anything."""
    return SMALL_CONFIG if os.environ.get(NET_ENV, "") == SMALL_NET else CONFIG


def new_net() -> KomugiNet:
    """Return a freshly initialised net of the environment's shape, on the CPU.

    No checkpoint, no ``eval()``, no device move: the *training* half of the
    campaign wants exactly this and then loads its own state, and sharing the
    constructor with :func:`serving_net` is what keeps the trainer's shape and
    the servers' shape one decision rather than two that agree today.
    """
    return KomugiNet(net_config())


def _load_model_state(net: KomugiNet, target: str) -> None:
    """Load the checkpoint `target` names into `net`.

    `target` may name a directory -- the newest verified checkpoint in it is
    served -- or a specific checkpoint file, which is served *exactly*: a
    pinned file must never be silently upgraded to a newer sibling, or
    rollback-by-pinning (the stated point of coexisting per-iteration
    checkpoints) is impossible (wave-11 union review F4). Verification is
    delegated either way, because a second implementation of "is this file
    whole" -- digest, format version, step agreement -- is a second thing to
    keep in step with the writer.

    Raises:
        FileNotFoundError: If what `target` names does not verify. Serving
            random weights because a checkpoint was unreadable is a silently
            wasted iteration; this is the loud version.
        ValueError: If the checkpoint was written for a different plane count
            (KOM-002). Checked HERE, on the returned object, rather than inside
            the loader -- see
            :func:`~komugi.train.checkpoint.require_matching_plane_count`.
    """
    path = Path(target)
    if path.is_dir():
        checkpoint = load_checkpoint(path)
    else:
        checkpoint = load_checkpoint_file(path)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{CHECKPOINT_ENV}={target!r} but it names no checkpoint "
            f"that verifies; refusing to serve freshly initialised weights "
            f"under a name that promises trained ones"
        )
    require_matching_plane_count(checkpoint, net.config.trunk.in_planes)
    net.load_state_dict(checkpoint.model_state)


def serving_net() -> torch.nn.Module:
    """Build the network this iteration's self-play servers evaluate.

    The module-level, no-argument callable
    :func:`~komugi.selfplay.orchestrate.run_selfplay` requires, configured
    entirely by the environment (see the module docstring):

    1. the shape from :data:`NET_ENV` -- :data:`SMALL_NET` delegates to
       :func:`small_serving_net`, anything else is b10c128;
    2. the weights from :data:`CHECKPOINT_ENV` when it is set and non-empty,
       otherwise a fresh initialisation;
    3. ``eval()``, always -- self-play is inference, and a net left in
       training mode would update its BatchNorm running statistics from
       whatever batch composition the server happened to assemble, making
       every worker's evaluations depend on its neighbours';
    4. the device from :data:`DEVICE_ENV`.
    """
    if os.environ.get(NET_ENV, "") == SMALL_NET:
        return small_serving_net()
    return _finish(KomugiNet(CONFIG))


def small_serving_net() -> torch.nn.Module:
    """Build the b2c16 serving net: :func:`serving_net`'s shape forced small.

    Also module-level and also no-argument, so it is usable as a factory in
    its own right -- a test that wants the tiny net without setting
    :data:`NET_ENV` names this instead. Checkpoint loading, ``eval()`` and
    device placement are identical, because a test rig that skipped any of the
    three would be exercising a different code path than the run it stands in
    for.
    """
    return _finish(KomugiNet(SMALL_CONFIG))


def _finish(net: KomugiNet) -> torch.nn.Module:
    """Apply the checkpoint, eval mode and device placement to a built `net`."""
    checkpoint = os.environ.get(CHECKPOINT_ENV, "")
    if checkpoint:
        _load_model_state(net, checkpoint)
    net.eval()
    if os.environ.get(DEVICE_ENV, "") == CUDA:
        net.to(CUDA)
    return net
