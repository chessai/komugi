"""The served net: an exponential moving average over *published* checkpoints.

KOM-022 §2. ``EMA_i = d * EMA_{i-1} + (1 - d) * theta_i``, one update per
campaign iteration, published through :func:`~komugi.train.checkpoint.save_checkpoint`
into a sibling directory so that the thing self-play serves is an ordinary
checkpoint with an ordinary digest-bearing name.

Three decisions here are load-bearing and each has a failure mode behind it.

**Which tensors are averaged: every floating-point entry of the state dict.**
The parameters *and* the BatchNorm ``running_mean`` / ``running_var``. Averaging
the parameters while leaving the buffers stale is the classic EMA bug -- a net
whose convolutions are an average of four generations, normalised by one
generation's statistics, is not any of the four. The integer
``num_batches_tracked`` counters are copied verbatim from ``theta_i``: an
exponentially-averaged counter is not a counter.

**This is deliberately not the digest's partition** (§3a,
:mod:`komugi.train.digest`). Tying them together is elegant and is exactly how
the BatchNorm buffers ended up inside the liveness digest twice: the running
statistics *must* be averaged here, because they describe the net, and *must
not* be digested there, because they advance without learning. Two rules, two
reasons, and a test on each set.

**The arithmetic runs on CPU tensors read back from files.** Not on the
in-memory training net, which the loop has already moved to
``campaign.device`` and which production runs on ``cuda``.
:func:`~komugi.train.checkpoint.load_checkpoint` maps to the CPU, so the
update is the same arithmetic over the same bytes whatever the campaign's
device is -- which is what makes replay-after-crash reproduce the steady
state's bytes on the machine the campaign actually runs on.

Surviving ``kill -9``: the EMA is **not** made atomic with the train
checkpoint, it is made **recomputable**. ``EMA_i`` is a pure function of
``(EMA_{i-1}, theta_i, d)`` and every ``theta`` coexists on disk, so
:func:`reconcile_ema` rebuilds whatever the crash cost -- and rebuilds an EMA
that was *corrupted* rather than merely missing, since an unverifiable file is
skipped by the loader, reads as "behind", and is replayed over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from komugi.train.checkpoint import (
    Checkpoint,
    CheckpointRef,
    list_checkpoints,
    load_checkpoint_file,
    save_checkpoint,
)
from komugi.train.digest import canonical_digest

EMA_DECAY: Final[float] = 0.75
"""The pre-registered smoothing constant.

Horizon ``1 / (1 - d) = 4`` iterations, chosen to match the recency window's
:class:`~komugi.train.window.WindowConfig` size of 5: the serving net's
smoothing horizon and the training data's horizon are then one decision rather
than two numbers that happen to agree. A tie to an existing constant, not a
tuned value -- KOM-022 §10 Claim 1 is the pre-registered refutation.
"""

EMA_ROLE: Final[str] = "ema"
"""What the published file's ``meta['role']`` says it is."""


class EmaError(RuntimeError):
    """The EMA directory is in a state no amount of replay can repair."""


def ema_state(
    previous: Mapping[str, torch.Tensor],
    theta: Mapping[str, torch.Tensor],
    decay: float = EMA_DECAY,
) -> dict[str, torch.Tensor]:
    """Return one EMA update: ``decay * previous + (1 - decay) * theta``.

    Floating-point entries are interpolated in float32 on CPU tensors;
    everything else -- ``num_batches_tracked`` and any future integer buffer --
    is **copied from `theta`**, because averaging a counter produces a number
    that counts nothing.

    Both mappings must carry exactly the same keys. A silent key union would
    publish an EMA whose shape is neither net's, and ``load_state_dict`` is
    strict at every consumer, so the failure would surface in a spawned child
    rather than here.

    Raises:
        ValueError: On a key mismatch, or a `decay` outside ``[0, 1)``.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay!r}")
    if set(previous) != set(theta):
        only_previous = sorted(set(previous) - set(theta))[:4]
        only_theta = sorted(set(theta) - set(previous))[:4]
        raise ValueError(
            f"EMA and train state dicts disagree on their keys: "
            f"{only_previous} only in the EMA, {only_theta} only in the "
            f"trainer's. Refusing to publish an average of two different nets"
        )
    updated: dict[str, torch.Tensor] = {}
    for key, new in theta.items():
        old = previous[key]
        if not new.is_floating_point():
            updated[key] = new.detach().to("cpu").clone()
            continue
        # `lerp` on a fresh CPU float32 copy, never in place on an input: an
        # in-place update on `previous` would alias the caller's tensors, and
        # an EMA that aliases the trainer's net is the exact bug the design's
        # KL == 0 guard exists to catch.
        base = old.detach().to(device="cpu", dtype=torch.float32).clone()
        target = new.detach().to(device="cpu", dtype=torch.float32)
        updated[key] = base.lerp_(target, 1.0 - decay).to(dtype=new.dtype)
    return updated


def ema_meta(
    iteration: int,
    source_step: int,
    train_sha256: str,
    ema_sha256: str,
    decay: float = EMA_DECAY,
    fast_policy_weight: float | None = None,
) -> dict[str, Any]:
    """Return the ``meta`` an EMA checkpoint carries.

    All JSON-serialisable, as :func:`~komugi.train.checkpoint.save_checkpoint`
    requires. The two digests are §3a's **parameter** digests: the buffer half
    of the file is covered by the filename digest over the whole file bytes,
    and per iteration by the separate ``bn_digest`` series.

    `fast_policy_weight` is KOM-036's arm, and the EMA is the artefact most
    likely to be copied out of a run directory and handed to an evaluator, so
    it must be able to name the arm that produced it. **Absent means the key is
    omitted, never stamped as ``None``**: an unlabelled EMA checkpoint stays
    unlabelled rather than claiming an arm it does not know, and a reader
    therefore distinguishes "no arm recorded" from "an arm recorded as
    nothing". It has a default because the parameters here are explicit and a
    required argument would break every existing call site for a field that
    predates none of them.
    """
    meta: dict[str, Any] = {
        "iteration": iteration,
        "role": EMA_ROLE,
        "decay": decay,
        "source_step": source_step,
        "train_sha256": train_sha256,
        "ema_sha256": ema_sha256,
    }
    if fast_policy_weight is not None:
        meta["fast_policy_weight"] = fast_policy_weight
    return meta


def publish_ema(
    directory: str | Path,
    step: int,
    state: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
) -> Path:
    """Publish `state` as an EMA checkpoint at `step`, and return its path.

    ``optim_state={}``: an EMA has no optimiser and nothing resumes from it.
    The empty mapping round-trips through the checkpoint codec unchanged, and
    the loop's own resume path is wired to the *train* directory precisely so
    that this file never reaches ``Optimizer.load_state_dict``.
    """
    return save_checkpoint(directory, step, state, {}, meta)


def _load_or_raise(ref: CheckpointRef) -> Checkpoint:
    """Load exactly `ref`, or raise -- a replay may not skip an input."""
    checkpoint = load_checkpoint_file(ref.path)
    if checkpoint is None:
        raise EmaError(
            f"{ref.path} does not verify, so the EMA series cannot be "
            f"rebuilt through it: every update is a function of its "
            f"predecessor and this one is missing"
        )
    return checkpoint


def _arm_of(checkpoint: Checkpoint) -> float | None:
    """Return the KOM-036 arm `checkpoint` records, or ``None`` if it records none.

    Read off the **source trainer checkpoint's own meta**, the same way
    :func:`_iteration_of` reads its iteration, and for the same reason: this is
    the EMA writer's only production caller and it has no ``StepConfig``, no
    ``CampaignConfig`` and no arm otherwise in scope. Sourcing it here rather
    than threading a second copy down makes the trainer's stamp and the EMA's
    stamp provably the *same value* instead of two independent claims that
    could disagree -- which is the misattribution the arm stamp exists to
    prevent.

    A checkpoint written before the stamp existed carries no key, and gets no
    arm: absent stays absent.
    """
    recorded = checkpoint.meta.get("fast_policy_weight")
    if isinstance(recorded, bool):
        return None
    if isinstance(recorded, (int, float)):
        return float(recorded)
    return None


def _iteration_of(checkpoint: Checkpoint, steps_per_iteration: int) -> int:
    """Return the iteration `checkpoint` records, falling back to its step."""
    recorded = checkpoint.meta.get("iteration")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded >= 0:
        return recorded
    if steps_per_iteration > 0:
        return max(checkpoint.step // steps_per_iteration - 1, 0)
    return 0


def reconcile_ema(
    checkpoints: str | Path,
    ema_directory: str | Path,
    param_keys: Sequence[str],
    *,
    decay: float = EMA_DECAY,
    steps_per_iteration: int = 0,
) -> tuple[int, ...]:
    """Bring the EMA directory level with the train directory, and say what it did.

    The resume rule of §2, in order:

    * equal newest steps -- nothing to do;
    * EMA behind by ``k`` -- replay ``k`` updates in **ascending step order**
      from the train checkpoints, which are all present;
    * EMA directory empty -- ``EMA_0 := theta_0``, then replay;
    * EMA **ahead** -- refuse, loudly. It means the ordering invariant (train
      first, EMA second) was violated, or two campaigns share one root.

    Each replayed update is itself a :func:`publish_ema`, i.e. individually
    atomic, and same-step supersede makes a replay idempotent: running this
    twice publishes the same bytes under the same name.

    **Fresh roots only.** An empty EMA directory beside *more than one* train
    checkpoint is a pre-KOM-022 root, and backfilling it would manufacture an
    EMA history that never served a single game -- and, if fingerprints were
    backfilled with it, a fabricated liveness record, the one thing this
    ticket must never produce. That case raises; migrating an existing root is
    a successor ticket, not a silent startup behaviour.

    Returns:
        The steps published by this call, ascending. Empty means the EMA was
        already level.

    Raises:
        EmaError: EMA ahead of the trainer, an unmigratable root, or an input
            checkpoint that does not verify.
    """
    train_refs = tuple(reversed(list_checkpoints(checkpoints)))
    ema_refs = tuple(reversed(list_checkpoints(ema_directory)))
    if not train_refs:
        if ema_refs:
            raise EmaError(
                f"{ema_directory} holds {len(ema_refs)} EMA checkpoint(s) but "
                f"{checkpoints} holds none: the EMA is ahead of the trainer, "
                f"which the publish ordering makes impossible on one campaign. "
                f"Two campaigns sharing one root is the likely cause"
            )
        return ()
    if ema_refs and ema_refs[-1].step > train_refs[-1].step:
        raise EmaError(
            f"EMA is ahead of the trainer: newest EMA step "
            f"{ema_refs[-1].step} > newest train step {train_refs[-1].step}. "
            f"The EMA is published *after* its source, so this cannot happen "
            f"in one campaign; refusing to start rather than averaging over an "
            f"unknown history"
        )
    if not ema_refs and len(train_refs) > 1:
        raise EmaError(
            f"{checkpoints} holds {len(train_refs)} checkpoints and "
            f"{ema_directory} is empty: this is a pre-KOM-022 root. Backfilling "
            f"would publish an EMA history that never served a game (and a "
            f"liveness record nothing observed). Migration is out of scope; "
            f"start a fresh root"
        )

    behind = ema_refs[-1].step if ema_refs else -1
    pending = [ref for ref in train_refs if ref.step > behind]
    if not pending:
        return ()

    previous: dict[str, torch.Tensor] | None = None
    if ema_refs:
        previous = _load_or_raise(ema_refs[-1]).model_state

    published: list[int] = []
    for ref in pending:
        theta = _load_or_raise(ref)
        if previous is None:
            # EMA_0 := theta_0, exactly. Iteration 1 then serves bit-identical
            # weights to the pre-EMA campaign's, and smoothing first bites at
            # iteration 2.
            state = {
                key: value.detach().to("cpu").clone()
                for key, value in theta.model_state.items()
            }
        else:
            state = ema_state(previous, theta.model_state, decay)
        publish_ema(
            ema_directory,
            ref.step,
            state,
            ema_meta(
                iteration=_iteration_of(theta, steps_per_iteration),
                source_step=ref.step,
                train_sha256=canonical_digest(theta.model_state, param_keys).sha256,
                ema_sha256=canonical_digest(state, param_keys).sha256,
                decay=decay,
                fast_policy_weight=_arm_of(theta),
            ),
        )
        published.append(ref.step)
        previous = state
    return tuple(published)
