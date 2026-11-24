"""Rig for KOM-022's boundary suite: planted roots, no trainer, no games.

Everything KOM-022's liveness section gates reads **published checkpoints**.
``canonical_digest`` is applied to a ``model_state`` loaded back from a file --
which is why the parameter/buffer split is by name and works on a state dict as
well as on a ``Module`` -- and the run-length gate walks the fingerprint series
those files produce. So a root can be *planted*: the checkpoints, the EMA
series and the journal built directly, and then the same boundary evaluation
the campaign runs at each iteration boundary invoked over it.

That is not a shortcut, it is the design's own construction (§3a, rev 4). The
acceptances that make the hash gate falsifiable (#27, #41, #43) need a
**non-learning trainer's output**, and rev 3's way of getting one -- a trainer
at ``lr = 0.0`` -- is refused at construction by both ``StepConfig`` and
``CampaignConfig``, whose guards this ticket must not relax (acceptance #47).
The output is exactly characterised without the trainer:

* the parameters are **bit-identical** from iteration to iteration, because the
  optimiser is their only writer and both of AdamW's parameter-writing terms
  scale by ``lr``;
* the BatchNorm ``running_mean`` / ``running_var`` and ``num_batches_tracked``
  are **not**, because ``net.train()`` advances them in the *forward* pass.

:func:`flat_parameter_states` builds that series and
:func:`advancing_states` builds the healthy one it must be told apart from.
Neither constructs a ``StepConfig``, an optimiser or a learning rate, so no
guard is relaxed, monkeypatched or reached around.

**What a planted root does not exercise**, stated rather than glossed: that a
live non-learning trainer really publishes such a series. That link rests on
the trainer's own contract (*"BatchNorm is the one piece of state a step
mutates besides the parameters"*) and is argued in the design, not tested here
-- testing it would mean deleting the ``lr > 0`` guard, and the guard is worth
more than the test. What keeps the fixture honest as the net changes is
acceptance #42, which asserts the digest's key set *equals* the live
optimiser's parameter set: a fixture that perturbed a tensor the optimiser owns,
or froze one it does not, fails there.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from komugi.campaign import boundary
from komugi.campaign.loop import checkpoint_dir, ema_dir, journal_path
from komugi.campaign.pause import Mute
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import NetFingerprint
from komugi.train.checkpoint import list_checkpoints, save_checkpoint
from komugi.train.digest import parameter_keys
from komugi.train.ema import reconcile_ema

STEPS_PER_ITERATION: Final[int] = 8
"""The step stride between planted iterations.

Only ever a *stride*: nothing in this rig runs an optimiser step. It has to be
positive and consistent, because iteration ``i`` files at
``(i + 1) * steps_per_iteration`` and the EMA is published at its source's
step.
"""

BUFFER_DRIFT: Final[float] = 1e-3
"""How far each iteration moves every BatchNorm running statistic.

Large enough to survive float32 at the running statistics' own scale (0 and 1
at initialisation), small enough that it is obviously a *statistic* moving and
not a parameter.
"""

PARAMETER_DRIFT: Final[float] = 1e-2
"""How far each iteration moves every parameter, in the *healthy* series."""


@dataclass(frozen=True, slots=True)
class PlantedIteration:
    """One planted boundary: what it served, and what it fingerprinted."""

    iteration: int
    served: str
    fingerprints: boundary.Fingerprints


def _snapshot(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of `net`'s whole state dict."""
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def _series(
    net: torch.nn.Module, iterations: int, parameter_drift: float
) -> list[dict[str, torch.Tensor]]:
    """Return `iterations` published states, drifting parameters by `parameter_drift`.

    Buffers always drift: a float buffer by :data:`BUFFER_DRIFT` per iteration
    and an integer one by :data:`STEPS_PER_ITERATION`, because a train-mode
    forward advances both whatever the optimiser does. Parameters drift by
    `parameter_drift`, which is ``0.0`` for the non-learning series -- and then
    the tensors are literally the same object, so "byte for byte identical" is
    a property of the construction rather than of float arithmetic.
    """
    snapshot = _snapshot(net)
    params = set(parameter_keys(net))
    states: list[dict[str, torch.Tensor]] = []
    for index in range(iterations):
        state: dict[str, torch.Tensor] = {}
        for key, value in snapshot.items():
            if key in params:
                state[key] = (
                    value
                    if parameter_drift == 0.0
                    else value + float(index) * parameter_drift
                )
            elif value.is_floating_point():
                state[key] = value + float(index) * BUFFER_DRIFT
            else:
                state[key] = value + index * STEPS_PER_ITERATION
        states.append(state)
    return states


def flat_parameter_states(
    net: torch.nn.Module, iterations: int
) -> list[dict[str, torch.Tensor]]:
    """Return what a **non-learning** trainer publishes: flat parameters, moving BN."""
    return _series(net, iterations, 0.0)


def advancing_states(
    net: torch.nn.Module, iterations: int
) -> list[dict[str, torch.Tensor]]:
    """Return what a **healthy** trainer publishes: everything moves."""
    return _series(net, iterations, PARAMETER_DRIFT)


def plant(
    root: str | os.PathLike[str],
    states: Sequence[Mapping[str, torch.Tensor]],
    param_keys: Sequence[str],
) -> list[PlantedIteration]:
    """Plant a root's checkpoints, EMA series and fingerprint journal.

    One pass per iteration, in the loop's own order and for the loop's own
    reasons:

    1. read the **serving** checkpoint -- the newest EMA, i.e. ``EMA_{i-1}`` --
       *before* anything for this iteration is published. Empty at iteration 0,
       where ``CHECKPOINT_ENV`` is ``""`` and the workers initialise freshly;
    2. publish ``theta_i`` into ``<root>/ckpt``;
    3. reconcile the EMA, which publishes ``EMA_i`` at ``theta_i``'s step. One
       reconcile per publication, because §2's resume rule *refuses* an empty
       EMA directory beside more than one train checkpoint -- that is a
       pre-KOM-022 root, and backfilling it would manufacture a liveness record
       nothing observed;
    4. write the iteration's three (or, at iteration 0, two) fingerprints,
       threading the previous iteration's BatchNorm digest so the separate
       ``bn_digest_advanced`` series has something to compare against.

    No gate is evaluated here: planting and judging are separate so a test can
    assert on the *series* before deciding what the gate should have said about
    it.
    """
    checkpoints = checkpoint_dir(root)
    ema_directory = ema_dir(root)
    journal = journal_path(root)
    Path(root).mkdir(parents=True, exist_ok=True)

    planted: list[PlantedIteration] = []
    previous_bn: float | None = None
    for index, state in enumerate(states):
        serving = list_checkpoints(ema_directory)
        served = str(serving[0].path) if serving else ""
        save_checkpoint(
            checkpoints,
            (index + 1) * STEPS_PER_ITERATION,
            state,
            {},
            {"iteration": index, "role": "train"},
        )
        reconcile_ema(
            checkpoints,
            ema_directory,
            param_keys,
            steps_per_iteration=STEPS_PER_ITERATION,
        )
        fingerprints = boundary.record_fingerprints(
            journal,
            index,
            checkpoints,
            ema_directory,
            served,
            param_keys,
            previous_bn,
        )
        previous_bn = fingerprints.bn_lo
        planted.append(
            PlantedIteration(
                iteration=index, served=served, fingerprints=fingerprints
            )
        )
    return planted


def recorded_fingerprints(
    root: str | os.PathLike[str],
) -> tuple[NetFingerprint, ...]:
    """Return every fingerprint this root's journal holds, in write order."""
    return read_records(
        journal_path(root), NetFingerprint, missing_ok=True
    ).records


def digests_for(
    fingerprints: Sequence[NetFingerprint], role: str
) -> list[str]:
    """Return one role's digests, ordered by iteration."""
    series = sorted(
        (f for f in fingerprints if f.role == role), key=lambda f: f.iteration
    )
    return [f.sha256 for f in series]


def gate(
    root: str | os.PathLike[str],
    iteration: int,
    *,
    mutes: Mapping[str, Mute] | None = None,
) -> boundary.BoundaryOutcome:
    """Run the boundary's structural gates over a planted root, and enforce.

    Exactly the call the campaign makes at each iteration boundary: the three
    unmutable gates (completeness, run-length per role, cross-role) over the
    journal's fingerprints, then :func:`~komugi.campaign.boundary.enforce`,
    which writes ``<root>/PAUSED`` **before** raising so that a watcher finds
    the marker rather than racing the exception.

    Returns the outcome when nothing armed a tripwire; raises
    :class:`~komugi.campaign.pause.CampaignPaused` when something did.
    """
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_liveness(
        recorded_fingerprints(root), iteration, outcome, mutes or {}
    )
    boundary.enforce(root, outcome, iteration, ())
    return outcome
