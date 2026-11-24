"""The victim process ``test_crash_safety`` SIGKILLs, and the constants it checks.

Not a test module. ``test_crash_safety.py`` both *imports* this (for the
schema, the seal cadence and the expected checkpoint contents) and *runs* it as
a separate interpreter -- the same rig pattern as
``tests/selfplay/behavior/worker_rig.py``. It has to be a real process because
the property under test is what a ``SIGKILL`` leaves behind, and a mocked
exception cannot produce that: an exception unwinds, runs ``finally`` blocks and
flushes buffers, which is precisely the forgiving path the shard writer and the
checkpoint writer are built to survive *without*.

Two modes, one per half of the claim.

``shards`` -- the preemption budget, made countable
    Appends rows forever, sealing on the writer's *periodic* trigger only
    (``samples_per_shard`` is set far out of reach). The clock is the
    injectable one, ticking a fixed 1.0s per call against a 10.0s flush
    interval, so the seal cadence is exactly :data:`SEAL_EVERY` appends rather
    than a function of how fast the machine happens to be. Whole numbers on
    purpose: the tick is applied by repeated addition, and ten additions of
    ``0.1`` come to ``0.9999999999999999``, which is *below* a 1.0s interval --
    the first shard would seal one append late and every count assertion in the
    parent would be off by one. Integers are exact in binary floating point, so
    ``now - last_seal`` is exactly ``SEAL_EVERY`` ticks. That is what turns
    "bounded loss" from a hope into an arithmetic assertion: whatever instant
    the kill lands on, the rows on disk are a whole number of ten-row shards
    and the buffer that died held fewer than ten.

    Progress is published *after* each accepted append, through a rename, so
    the parent reads a complete number and never a half-written one. The count
    it reads may lag the truth by exactly one append -- the one that had been
    accepted when the kill landed but whose progress file had not been
    replaced yet -- and the parent's bounds allow for that explicitly.

``checkpoint`` -- the save that dies mid-flight
    Publishes two small checkpoints and then saves a deliberately large one
    (:data:`BLOB_FLOATS` floats, ~16 MiB) over and over. The blob is built
    *before* the step is announced, so the window between the announcement the
    parent is waiting for and the bytes hitting the disk is the save itself and
    almost nothing else: the kill lands inside
    :func:`~komugi.train.checkpoint.save_checkpoint` with very high
    probability. If it somehow does not, the child writes :data:`FINISHED` and
    the parent fails loudly rather than passing on a scenario that never ran.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Final

import torch

from komugi.train.checkpoint import save_checkpoint
from komugi.train.shards import FieldSpec, Schema, ShardWriter

SCHEMA: Final[Schema] = Schema(
    fields=(FieldSpec(name="row", dtype="int64", shape=()),)
)
"""One int64 per sample: its own row number, so the survivors are checkable."""

SAMPLES_PER_SHARD: Final[int] = 1_000_000
"""Out of reach on purpose -- only the periodic trigger may seal here."""

FLUSH_EVERY_SECONDS: Final[float] = 10.0
CLOCK_TICK: Final[float] = 1.0
SEAL_EVERY: Final[int] = 10
"""``FLUSH_EVERY_SECONDS / CLOCK_TICK``: the writer checks its clock once per
append, so with a clock that advances a tenth of the interval per call a shard
seals on exactly every tenth append."""

APPEND_PAUSE_S: Final[float] = 0.002
"""Real sleep between appends. It does not touch the seal cadence (that runs
off the injected clock) -- it only widens the wall-clock window the parent has
to land its kill in, so the test is not a race against a fast machine."""

MAX_APPENDS: Final[int] = 200_000
"""A backstop, not a budget: the parent kills long before this."""

BLOB_FLOATS: Final[int] = 4_000_000
"""~16 MiB per checkpoint, so a save is slow enough to be interrupted."""

SMALL_STEPS: Final[int] = 2
"""Checkpoints published before the big ones start: the "old valid checkpoint"
the atomicity claim says must survive a kill during the next save."""

MAX_STEP: Final[int] = 6
FINISHED: Final[str] = "finished"
"""Written to the progress file if the child runs out of work uninterrupted."""


class TickClock:
    """A monotonic clock that advances a fixed amount per reading.

    The writer reads its clock exactly once per :meth:`ShardWriter.append` (and
    once at construction), so a fixed tick makes the flush policy a function of
    the append count and nothing else -- no sleeping, and no dependence on how
    long a shard takes to compress on this particular machine.
    """

    def __init__(self, tick: float = CLOCK_TICK) -> None:
        """Start at zero, advancing `tick` per call."""
        self._now = 0.0
        self._tick = tick

    def __call__(self) -> float:
        """Return the current time, then advance."""
        now = self._now
        self._now += self._tick
        return now


def weights_for(step: int) -> dict[str, torch.Tensor]:
    """Return the deterministic model state this child saves at `step`.

    Imported by the parent, which uses it to check that what survived a kill is
    a *correct* checkpoint for the step its name claims -- not merely a file
    whose bytes agree with its own digest.
    """
    return {"weight": torch.full((4,), float(step), dtype=torch.float32)}


def publish(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, so a reader never sees half of it.

    The same rename trick the modules under test use, for the same reason: this
    file is read by another process while this one may be killed at any
    instruction, and a torn progress count would be indistinguishable from a
    real one.
    """
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_shards(directory: Path, progress: Path) -> None:
    """Append rows until killed, sealing every :data:`SEAL_EVERY` appends."""
    writer = ShardWriter(
        directory,
        SCHEMA,
        samples_per_shard=SAMPLES_PER_SHARD,
        flush_every_seconds=FLUSH_EVERY_SECONDS,
        clock=TickClock(),
    )
    for row in range(MAX_APPENDS):
        writer.append({"row": torch.tensor(row, dtype=torch.int64)})
        publish(progress, str(row + 1))
        time.sleep(APPEND_PAUSE_S)
    publish(progress, FINISHED)


def run_checkpoints(directory: Path, progress: Path) -> None:
    """Publish two small checkpoints, then large ones until killed."""
    for step in range(1, SMALL_STEPS + 1):
        publish(progress, str(step))
        save_checkpoint(directory, step, weights_for(step), {}, {"step": step})

    # Built before anything is announced: the parent's kill must land inside a
    # save, not inside the allocation of the thing being saved.
    blob = torch.full((BLOB_FLOATS,), 1.5, dtype=torch.float32)
    for step in range(SMALL_STEPS + 1, MAX_STEP + 1):
        state = weights_for(step)
        state["blob"] = blob
        publish(progress, str(step))
        save_checkpoint(directory, step, state, {}, {"step": step})
    publish(progress, FINISHED)


def main(argv: list[str]) -> int:
    """Dispatch ``<mode> <directory> <progress-file>``."""
    if len(argv) != 3:
        print(f"usage: {__file__} <shards|checkpoint> <dir> <progress>", file=sys.stderr)
        return 2
    mode, directory, progress = argv[0], Path(argv[1]), Path(argv[2])
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "shards":
        run_shards(directory, progress)
    elif mode == "checkpoint":
        run_checkpoints(directory, progress)
    else:
        print(f"unknown mode {mode!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover -- exercised as a subprocess
    sys.exit(main(sys.argv[1:]))
