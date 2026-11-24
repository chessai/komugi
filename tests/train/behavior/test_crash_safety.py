"""A real process, a real ``kill -9``, and what the trainer keeps.

``tests/train/test_shards.py`` and ``tests/train/test_checkpoint.py`` stage
their crashes in-process: a writer that is dropped without closing, a file
truncated by hand. Those pin the *recovery* logic, and they are the right shape
for it. They cannot pin the thing a preemption actually does, because an
in-process simulation always dies somewhere the code chose. This module hands
the choice to the kernel.

The victim is :mod:`crash_child`, launched as a separate interpreter and
stopped with ``SIGKILL`` -- not ``terminate()``, which would run ``atexit``
handlers, buffered-file finalisers and ``ShardWriter.close``, i.e. the whole
forgiving path the two invariants below exist to be true *without*. Both tests
assert the victim died by signal 9 before asserting anything about the disk, so
a scenario that quietly failed to run cannot pass.

**(a) Bounded loss.** ``ShardWriter``'s periodic trigger promises that a
preemption forfeits appends, not an iteration. The child seals on exactly every
tenth append (the injected clock makes the cadence arithmetic rather than a
race), so the bound is a number and not an adjective: whatever instruction the
kill lands on, the rows on disk are a whole number of ten-row shards, the
surviving rows are the first ``n`` in order, and fewer than ten accepted
appends are missing.

**(b) Atomicity.** The child is killed while a ~16 MiB checkpoint is being
written. What is on disk afterwards must be the old valid checkpoint or the new
valid one -- never a file that ``load_checkpoint_file`` accepts and that is not
a whole checkpoint. That is asserted three ways: every published file's bytes
are re-hashed against the digest in its own name, every published file is
loaded through the real loader and its tensors compared with what the child
meant to save at that step, and whatever the kill left half-written is checked
to be invisible to the loader entirely.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import torch

import crash_child as child
import komugi
from komugi.train.checkpoint import (
    CHECKPOINT_GLOB,
    CheckpointWarning,
    list_checkpoints,
    load_checkpoint,
    load_checkpoint_file,
    save_checkpoint,
)
from komugi.train.shards import SHARD_GLOB, ShardReader

_CHILD_SCRIPT: Final[Path] = Path(__file__).with_name("crash_child.py")

_MIN_APPENDS: Final[int] = 25
"""Kill only once at least two shards have sealed, so "what survived" is a
statement about the flush policy rather than about an empty directory."""

_WARMUP_WALL_S: Final[float] = 180.0
"""Generous: the child pays a fresh interpreter's torch import first."""

_JOIN_WALL_S: Final[float] = 60.0
_POLL_S: Final[float] = 0.001


@dataclass(frozen=True, slots=True)
class _Victim:
    """A spawned child, the directory it writes into, and where it reports."""

    process: subprocess.Popen[bytes]
    directory: Path
    progress: Path
    stderr: Path


def _child_environment() -> dict[str, str]:
    """Return the child's environment, with this ``komugi`` on its import path.

    The suite runs from a source checkout with nothing pip-installed (see
    ``ci/check.sh``), so a fresh interpreter finds the package only through
    ``PYTHONPATH``. Deriving the directory from the already-imported package
    rather than trusting the inherited variable means the child imports the
    *same* copy this test is asserting about.
    """
    package = komugi.__file__
    assert package is not None, "komugi is a namespace package with no __file__"
    source_root = str(Path(package).resolve().parent.parent)
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else f"{source_root}{os.pathsep}{existing}"
    )
    return environment


def _spawn(mode: str, tmp_path: Path) -> _Victim:
    """Launch :mod:`crash_child` in `mode` and return the handle to kill."""
    directory = tmp_path / mode
    progress = tmp_path / f"{mode}-progress"
    stderr = tmp_path / f"{mode}-stderr"
    handle = stderr.open("wb")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(_CHILD_SCRIPT),
                mode,
                str(directory),
                str(progress),
            ],
            stdout=subprocess.DEVNULL,
            stderr=handle,
            env=_child_environment(),
        )
    finally:
        handle.close()
    return _Victim(
        process=process, directory=directory, progress=progress, stderr=stderr
    )


def _read_progress(victim: _Victim) -> int | None:
    """Return the child's published progress, or ``None`` if it has published none.

    ``crash_child.FINISHED`` means the child ran out of work before it could be
    killed, which makes every assertion downstream vacuous -- so it fails here,
    where the reason is still legible.
    """
    try:
        text = victim.progress.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if text == child.FINISHED:
        pytest.fail(
            "the child completed all of its work before it could be killed, so "
            "nothing below would have tested a crash"
        )
    return int(text)


def _diagnostics(victim: _Victim) -> str:
    """Return whatever the child said on stderr, for a failure message."""
    try:
        return victim.stderr.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:  # pragma: no cover -- only on a vanished tmp dir
        return "<stderr unavailable>"


def _await_progress(victim: _Victim, minimum: int) -> int:
    """Block until the child has published at least `minimum`, or fail."""
    deadline = time.monotonic() + _WARMUP_WALL_S
    while time.monotonic() < deadline:
        published = _read_progress(victim)
        if published is not None and published >= minimum:
            return published
        if victim.process.poll() is not None:
            pytest.fail(
                f"the child exited on its own ({victim.process.returncode}) "
                f"before reaching {minimum}: {_diagnostics(victim)}"
            )
        time.sleep(_POLL_S)
    pytest.fail(
        f"the child never reached progress {minimum} within {_WARMUP_WALL_S}s: "
        f"{_diagnostics(victim)}"
    )


def _sigkill(victim: _Victim) -> int:
    """Send ``SIGKILL``, join, and return the exit status."""
    if victim.process.poll() is not None:
        pytest.fail(
            f"the child was already dead ({victim.process.returncode}) when the "
            f"kill was due: {_diagnostics(victim)}"
        )
    os.kill(victim.process.pid, signal.SIGKILL)
    return victim.process.wait(timeout=_JOIN_WALL_S)


def _reap(victim: _Victim) -> None:
    """Make sure nothing survives this test, whatever it asserted."""
    if victim.process.poll() is None:  # pragma: no cover -- only after a failure
        os.kill(victim.process.pid, signal.SIGKILL)
        victim.process.wait(timeout=_JOIN_WALL_S)


def test_a_sigkilled_writer_loses_less_than_one_flush_interval(
    tmp_path: Path,
) -> None:
    """kill -9 mid-iteration: whole shards survive, and the loss has a number.

    The bound asserted is the one the module docstring promises -- the buffer
    never holds more than one interval's worth of appends -- expressed in the
    only unit that is checkable from outside: appends the child had confirmed,
    minus rows the reader can serve.
    """
    victim = _spawn("shards", tmp_path)
    try:
        _await_progress(victim, _MIN_APPENDS)
        status = _sigkill(victim)
    finally:
        _reap(victim)

    assert status == -signal.SIGKILL, (
        f"the child exited {status}, not by SIGKILL: the scenario did not run "
        f"({_diagnostics(victim)})"
    )
    recorded = _read_progress(victim)
    assert recorded is not None and recorded >= _MIN_APPENDS

    reader = ShardReader(victim.directory, schema=child.SCHEMA)
    assert reader.skipped == (), (
        "a kill left a shard the reader had to skip -- publication is supposed "
        f"to be an atomic rename, so no partial shard can be visible: "
        f"{reader.skipped}"
    )
    recovered = len(reader)

    assert recovered % child.SEAL_EVERY == 0, (
        f"{recovered} rows survived, which is not a whole number of "
        f"{child.SEAL_EVERY}-row shards: a shard was published short, or the "
        "periodic trigger did not fire on the cadence the clock dictates"
    )
    assert [info.count for info in reader.shards] == [child.SEAL_EVERY] * (
        recovered // child.SEAL_EVERY
    )
    assert recovered >= 2 * child.SEAL_EVERY, (
        f"only {recovered} rows survived a kill after {recorded} confirmed "
        "appends: the periodic flush is not sealing"
    )

    # The bound, both ways round. The child publishes its progress *after* each
    # accepted append, so the reader may legitimately be one row ahead of the
    # count (the kill landed between the append and the rename) and may be up
    # to one unsealed buffer behind it.
    assert recorded - recovered < child.SEAL_EVERY, (
        f"{recorded - recovered} confirmed appends were lost, more than the "
        f"{child.SEAL_EVERY - 1} a single unsealed buffer can hold"
    )
    assert recovered <= recorded + 1, (
        f"the reader serves {recovered} rows but the child only ever confirmed "
        f"{recorded} appends"
    )

    # What survived is the *prefix*, in order: a kill costs the tail and
    # nothing in the middle.
    assert [int(reader[index]["row"].item()) for index in range(recovered)] == list(
        range(recovered)
    )

    # And the only thing a kill may leave besides published shards is the
    # dot-prefixed temporary the reader's glob cannot see.
    for path in sorted(victim.directory.iterdir()):
        assert fnmatch.fnmatch(path.name, SHARD_GLOB) or path.name.startswith("."), (
            f"{path.name} is neither a published shard nor invisible to the "
            "reader"
        )


def test_a_sigkilled_save_leaves_only_whole_checkpoints(tmp_path: Path) -> None:
    """kill -9 mid-save: the old checkpoint or the new one, never a torn one.

    Atomicity is asserted rather than inferred from the absence of a crash. The
    kill lands inside a multi-megabyte
    :func:`~komugi.train.checkpoint.save_checkpoint`; afterwards every file the
    loader can see is re-hashed independently, loaded, and compared against the
    tensors the child meant to write at that step.
    """
    victim = _spawn("checkpoint", tmp_path)
    try:
        announced = _await_progress(victim, child.SMALL_STEPS + 1)
        status = _sigkill(victim)
    finally:
        _reap(victim)

    assert status == -signal.SIGKILL, (
        f"the child exited {status}, not by SIGKILL: the scenario did not run "
        f"({_diagnostics(victim)})"
    )
    final = _read_progress(victim)
    announced = final if final is not None else announced
    assert announced > child.SMALL_STEPS, (
        "the kill landed before the first large save even started, so nothing "
        "was interrupted"
    )

    refs = list_checkpoints(victim.directory)
    assert refs, "the kill destroyed every checkpoint the child had published"

    for ref in refs:
        data = ref.path.read_bytes()
        assert hashlib.sha256(data).hexdigest()[:16] == ref.digest, (
            f"{ref.path.name} is visible under a checkpoint name and its bytes "
            "do not hash to the digest that name carries -- a torn write "
            "became loadable"
        )
        loaded = load_checkpoint_file(ref.path)
        assert loaded is not None, (
            f"{ref.path.name} is published and does not load: a kill made a "
            "published checkpoint unusable"
        )
        assert loaded.step == ref.step
        assert torch.equal(
            loaded.model_state["weight"], child.weights_for(ref.step)["weight"]
        ), (
            f"the checkpoint published at step {ref.step} holds tensors the "
            "child never meant to write at that step"
        )

    steps = sorted(ref.step for ref in refs)
    assert steps == list(range(1, len(steps) + 1)), (
        f"the published steps are not a prefix of what the child ran: {steps}"
    )

    newest = load_checkpoint(victim.directory)
    assert newest is not None
    assert newest.step in {announced - 1, announced}, (
        f"the newest loadable checkpoint is step {newest.step} while the child "
        f"had announced step {announced}: what survives a kill must be either "
        "the save that had completed or the one that was in flight, and "
        "nothing older"
    )

    # Whatever the kill left half-written is outside the loader's glob, so no
    # entry point can reach it: not the directory walk, not the pinned-file
    # loader.
    residue = [
        path
        for path in sorted(victim.directory.iterdir())
        if not fnmatch.fnmatch(path.name, CHECKPOINT_GLOB)
    ]
    published = {ref.path for ref in refs}
    for path in residue:
        assert path.name.startswith(".") and ".tmp-" in path.name, (
            f"{path.name} was left in the checkpoint directory by a kill and is "
            "neither a published checkpoint nor a dot-prefixed temporary"
        )
        assert path not in published
        assert load_checkpoint_file(path) is None, (
            f"{path.name} is a temporary the kill interrupted and the loader "
            "accepted it"
        )


def test_the_pinned_file_loader_refuses_a_truncated_checkpoint(
    tmp_path: Path,
) -> None:
    """The other half of atomicity: visibility alone would not be enough.

    The test above asserts that what a kill leaves half-written is invisible to
    the loader. That is only a defence if the loader would *also* reject such a
    file were it visible -- and the pinned-file entry point is the one a
    rollback uses, where silently substituting a healthy sibling is exactly
    what must not happen. So: publish two checkpoints, truncate the newer one
    under its own name, and require
    :func:`~komugi.train.checkpoint.load_checkpoint_file` to return ``None``
    rather than the older one and rather than half a tensor.

    In-process on purpose -- the bytes are ordinary
    :func:`~komugi.train.checkpoint.save_checkpoint` output, and what is under
    test is the acceptance predicate, not the kill.
    """
    directory = tmp_path / "pinned"
    older = save_checkpoint(directory, 1, child.weights_for(1), {}, {"step": 1})
    newer = save_checkpoint(directory, 2, child.weights_for(2), {}, {"step": 2})

    assert load_checkpoint_file(newer) is not None, (
        "the checkpoint does not load before it is damaged, so truncating it "
        "would prove nothing"
    )
    data = newer.read_bytes()
    newer.write_bytes(data[: len(data) // 2])

    with pytest.warns(CheckpointWarning, match="digest mismatch"):
        assert load_checkpoint_file(newer) is None, (
            "a truncated file was accepted under the name of the checkpoint a "
            "caller pinned"
        )
    # The pin is not silently redirected either: the healthy sibling is still
    # there, and still only reachable by asking for it.
    pinned_older = load_checkpoint_file(older)
    assert pinned_older is not None and pinned_older.step == 1
    with pytest.warns(CheckpointWarning, match="digest mismatch"):
        fallback = load_checkpoint(directory)
    assert fallback is not None and fallback.step == 1
