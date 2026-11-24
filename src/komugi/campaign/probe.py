"""The frozen probe set, and the policy KL measured on it.

KOM-022 §3b/§7. Two nets are compared by the mean ``KL(pi_ema || pi_train)``
over the **legal** actions of a fixed set of positions. Three properties of
that sentence are decisions:

**Legal, not all.** ``pi`` is a masked softmax over the position's legal
actions. The unmasked distribution is dominated by 37,000 logits nothing will
ever play, so a KL over it measures agreement about moves that do not exist.
The mask is derived from ``policy_target > 0``, which the shard schema
documents as *"nonzero on every legal action"*. Stated plainly (§11.15): that
gives one direction, and this construction needs the biconditional. The shard
docstring's missing half is a one-line change in
``komugi.selfplay.records``, outside this ticket's file allowlist, so this
module *validates what it can* -- a row whose mask is empty is refused rather
than silently contributing a zero -- and the gap is recorded rather than
papered over.

**Frozen, not this iteration's.** The band gate's reference distribution must
not move under the gate, or a band exit could not be attributed to the nets
rather than to the data. The probe set is sampled once, from iteration 0's
shards, and never resampled.

**And the drift that follows, stated in advance.** Iteration 0 plays with
fresh weights, so these are positions reached by random play; as the campaign
strengthens, both nets move into a distribution the probe set does not cover.
The band gate ``r`` is substantially protected -- its numerator and
denominator are both probe-set quantities at the same iteration, so a common
drift factor cancels -- and §7's ``ood_ratio`` is **not**. It is expected to
trend upward over a healthy campaign for reasons that have nothing to do with
OOD, and §10 Claim 5 records that as a pre-registered expectation rather than
leaving a successor ticket to threshold a drifting quantity.

Accumulation is float64 on CPU and every result is checked finite before it
reaches a record: :class:`~komugi.telemetry.records.IterationMetrics` rejects
non-finite extras, so an unguarded ``inf`` takes the campaign down *while it
is writing telemetry* -- and the blanket pause handler then wedges it on a
diagnostic that feeds no control law.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

from komugi.campaign.pause import write_json_durably
from komugi.train.window import SampleSource

PROBE_DIRNAME: Final[str] = "probe"
PROBE_FILENAME: Final[str] = "probe.npz"
PROBE_SIZE: Final[int] = 512
"""Positions in the probe set.

At the production shapes the file is ~19.6 MB of observations plus ~19.1 MB of
boolean legal mask, i.e. **~39 MB** uncompressed -- immaterial to the design,
material to anyone budgeting the root.
"""

PROBE_DOMAIN: Final[int] = 0x9022_0001
OOD_DOMAIN: Final[int] = 0x9022_0002
"""Spawn-key domains, so the probe and OOD samplers cannot collide with the
value :func:`~komugi.campaign.loop.iteration_seed` already uses for the window
and the batch sampler."""

_OBSERVATION: Final[str] = "observation"
_LEGAL: Final[str] = "legal"


class ProbeError(RuntimeError):
    """The probe set could not be sampled, read, or trusted."""


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """A batch of positions and their legal-action masks."""

    observation: torch.Tensor
    legal: torch.Tensor

    def __len__(self) -> int:
        """Return how many positions the set holds."""
        return int(self.observation.shape[0])


def probe_path(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/probe/probe.npz``, whether or not it exists."""
    return Path(root) / PROBE_DIRNAME / PROBE_FILENAME


def probe_seed(run_seed: int, domain: int, iteration: int | None = None) -> int:
    """Return a sampler seed, derived the way every other campaign seed is.

    ``SeedSequence(run_seed, spawn_key=(domain, ...))`` reduced to one 32-bit
    word -- :func:`~komugi.campaign.loop.iteration_seed`'s construction with a
    domain constant in front, so the probe sampler's stream cannot coincide
    with the window's or the batch sampler's.
    """
    key = (domain,) if iteration is None else (domain, iteration)
    child = np.random.SeedSequence(run_seed, spawn_key=key)
    return int(child.generate_state(1, dtype=np.uint32)[0])


def sample_probe(source: SampleSource, size: int, seed: int) -> ProbeSet:
    """Draw `size` positions from `source` without replacement, as a probe set.

    Rows are drawn by a seeded permutation, so the same source and seed give
    the same set on any machine and after any resume. A source smaller than
    `size` contributes every row it has -- a short probe set is weaker, and a
    campaign that refused to start over it would be refusing over the least
    interesting cause.

    Raises:
        ProbeError: If `source` is empty, or if a drawn row's ``policy_target``
            is zero everywhere -- a row with no legal action is either a
            terminal position that should never have been written or a shard
            whose schema is not what this expects, and either way a masked
            softmax over it is undefined.
    """
    rows = len(source)
    if rows == 0:
        raise ProbeError(
            "cannot sample a probe set from an empty source: the band gate's "
            "reference distribution would have no positions"
        )
    generator = np.random.default_rng(seed)
    order = generator.permutation(rows)[: min(size, rows)]
    observations: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for index in sorted(int(value) for value in order):
        sample = source[index]
        target = sample["policy_target"]
        mask = target > 0
        if not bool(mask.any()):
            raise ProbeError(
                f"row {index} has a policy target that is zero on every "
                f"action, so it names no legal move: refusing to build a "
                f"masked softmax over it"
            )
        observations.append(sample[_OBSERVATION].to(torch.float32))
        masks.append(mask)
    return ProbeSet(
        observation=torch.stack(observations), legal=torch.stack(masks)
    )


def write_probe(path: str | os.PathLike[str], probe: ProbeSet) -> str:
    """Write `probe` durably to `path` and return the sha256 of the bytes.

    Temp -> ``fsync`` -> ``os.replace`` -> ``fsync`` the directory, **before**
    the digest is computed and recorded. A torn write would otherwise pin the
    journal to the digest of a truncated file and make every later comparison
    agree with the wrong thing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp-{os.getpid()}"
    payload = {
        _OBSERVATION: probe.observation.numpy(),
        _LEGAL: probe.legal.numpy(),
    }
    with tmp.open("wb") as handle:
        # A file object, not a path, for the reason the shard writer gives:
        # numpy appends '.npz' to any path that does not already end in it,
        # which would rename the tmp file out from under the replace below.
        np.savez(handle, **payload)  # type: ignore[arg-type]  # numpy stubs mistype savez's **kwargs
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return probe_digest(target)


def probe_digest(path: str | os.PathLike[str]) -> str:
    """Return sha256 over the probe file's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_probe(path: str | os.PathLike[str]) -> ProbeSet:
    """Read a probe set back, or raise :class:`ProbeError`."""
    try:
        with np.load(Path(path)) as archive:
            observation = np.asarray(archive[_OBSERVATION])
            legal = np.asarray(archive[_LEGAL])
    except (OSError, KeyError, ValueError) as exc:
        raise ProbeError(f"{path} is not a readable probe set: {exc}") from exc
    return ProbeSet(
        observation=torch.from_numpy(observation.astype(np.float32)),
        legal=torch.from_numpy(legal.astype(np.bool_)),
    )


def _masked_log_softmax(
    logits: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Return log-softmax over the legal actions only, in float64.

    Illegal entries are pushed to ``-inf`` *before* the normalisation, so they
    contribute nothing to the partition function; the result is then finite on
    the legal entries and ``-inf`` elsewhere, and the KL below only ever reads
    the legal ones.
    """
    masked = logits.to(torch.float64).masked_fill(~legal, float("-inf"))
    return masked - torch.logsumexp(masked, dim=-1, keepdim=True)


def kl_rows(
    left_logits: torch.Tensor, right_logits: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Return per-position ``KL(left || right)`` over the legal actions.

    Direction matters and is fixed by the design: the **first** argument is
    the net that generates the data. ``KL(pi_selfplay || pi_train)`` weights
    the regions the generating net actually visits, which is the quantity that
    describes the data the trainer will receive.
    """
    left = _masked_log_softmax(left_logits, legal)
    right = _masked_log_softmax(right_logits, legal)
    probability = left.exp()
    contribution = torch.where(
        legal, probability * (left - right), torch.zeros_like(probability)
    )
    return contribution.sum(dim=-1)


def policy_kl(
    generator: torch.nn.Module,
    reference: torch.nn.Module,
    probe: ProbeSet,
    *,
    batch_size: int = 64,
) -> tuple[float, int]:
    """Return the mean ``KL(generator || reference)`` on `probe`, and its zero count.

    Both nets are evaluated in ``eval()`` mode -- a train-mode forward here
    would update BatchNorm running statistics from the probe set, which is a
    diagnostic silently editing the thing it measures. The mean is accumulated
    in float64 and asserted finite by the caller before it reaches a record.

    The second element is how many probe positions produced **exactly** zero
    KL. An all-zero count is the aliasing bug -- the EMA update binding the
    trainer's tensor by reference instead of copying it -- and that check needs
    no model at all, which makes it the sharpest thing in §3b.

    ``no_grad`` as a ``with`` block rather than as a decorator: this repo spells
    it that way everywhere, and the decorator form is untyped under strict
    mypy.
    """
    generator.eval()
    reference.eval()
    # The probe set is held on the CPU (it is read back from a file), while
    # both nets sit on the campaign's device -- `cuda` in production. Moving
    # the batch rather than the nets keeps the measurement one device move per
    # batch instead of two net copies per iteration, and makes the KL the same
    # arithmetic on either device.
    device = next(generator.parameters()).device
    total = 0.0
    zeros = 0
    with torch.no_grad():
        for start in range(0, len(probe), batch_size):
            observations = probe.observation[start : start + batch_size].to(device)
            legal = probe.legal[start : start + batch_size].to(device)
            left = generator(observations).policy_logits
            right = reference(observations).policy_logits
            values = kl_rows(left, right, legal)
            total += float(values.sum().item())
            zeros += int((values == 0.0).sum().item())
    return total / max(len(probe), 1), zeros


def observation_batch(rows: Sequence[dict[str, torch.Tensor]]) -> ProbeSet:
    """Return `rows` as a probe set, for the OOD signal's fresh sample.

    The same shape as the frozen probe set so one KL routine serves both; the
    difference is only which positions are in it.
    """
    observations = [row[_OBSERVATION].to(torch.float32) for row in rows]
    masks = [row["policy_target"] > 0 for row in rows]
    if not observations:
        raise ProbeError("no fresh rows to measure the OOD signal on")
    return ProbeSet(observation=torch.stack(observations), legal=torch.stack(masks))


def record_probe_digest(root: str | os.PathLike[str], digest: str) -> None:
    """Stamp the probe set's digest beside it, durably.

    The journal carries ``probe_sha256_lo`` per iteration; this is the full
    digest, kept next to the file it describes so a swapped probe set is
    detectable by inspection as well as by a series comparison.
    """
    write_json_durably(
        Path(root) / PROBE_DIRNAME / "probe.sha256.json", {"sha256": digest}
    )


