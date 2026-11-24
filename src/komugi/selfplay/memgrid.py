"""The desktop memory grid: the evidence the node-pool deferral rests on.

Design D9 defers ``DESIGN.html`` §7.3/§8's node pools, which is an explicit
DESIGN deviation. What makes the deferral honest rather than optimistic is a
*model*: a ``_Node`` costs ~1.08MB, a search holds at most ``n + 1`` of them, so
a shard costs roughly ``workers * (n_full + 1) * 1.1MB`` at its worst-case
all-full moment. Design F12 makes that model falsifiable by measuring it -- a
``(workers, n_full)`` grid against the cgroup's own high-water mark -- and
pre-registers the outcome: if the measurements do not track the model, the node
pool ticket re-opens automatically.

This module is the harness for that measurement, and nothing else. It fits
tables, it does not; the fitted model is committed with the run.

Why every point runs in a fresh subprocess
------------------------------------------
``memory.peak`` is a **high-water mark**: it only ever goes up over the life of
a cgroup. The orchestrator process itself has already imported torch, built
whatever the caller built, and -- crucially -- run every *previous* grid point,
so measuring in-process would give each point a floor equal to the largest
point measured so far. The grid would come out monotone by construction and
would agree with any model at all.

So each point is one spawned child that runs a short
:func:`~komugi.selfplay.orchestrate.run_selfplay` and exits; its memory is
returned to the kernel at exit while the mark it set stays readable. The mark
is also **reset** before each point, which is what makes the numbers comparable
rather than merely non-decreasing.

The reset is per-file-descriptor (KOM-055)
------------------------------------------
``memory.peak`` has been writable since Linux 6.7, but since Linux 6.11 a write
resets the watermark **only for the file description it was written through**:
the cgroup keeps its lifetime mark, and every other opener still sees it. So a
reset written through one fd and a reading ``open()`` of the same path are two
different counters, and the second one is the cgroup's lifetime high-water mark
-- in KOM-055's gate run, 5.4 GB, byte-for-byte the daemon service's own peak,
reported for a toy point. :func:`measure_grid` therefore keeps the fd it reset
open across the point and reads back through it with ``pread``; a fresh open is
never a measurement.

What a point reports is consequently a *growth*: the per-fd watermark after the
child exited, minus ``memory.current`` at the moment of the reset. That is the
allocation the point is responsible for, plus whatever any cohabitant of the
cgroup allocated meanwhile -- which is why :func:`cgroup_is_shared` is a
precondition and not a footnote.
"""

from __future__ import annotations

import os

from collections.abc import Callable, Sequence
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Final, NamedTuple

import torch

from komugi.config import SelfPlayConfig
from komugi.selfplay.inference import InferenceConfig
from komugi.selfplay.orchestrate import (
    OrchestratorConfig,
    require_module_level_factory,
    run_selfplay,
)

_CGROUP_ROOT: Final[Path] = Path("/sys/fs/cgroup")
"""Where the cgroup v2 hierarchy is mounted on every system this runs on."""

_PROC_CGROUP: Final[Path] = Path("/proc/self/cgroup")
"""This process's cgroup membership. The v2 line is the one starting ``0::``."""

_PEAK_FILE: Final[str] = "memory.peak"
"""The controller file holding the cgroup's high-water mark, in bytes."""

_CURRENT_FILE: Final[str] = "memory.current"
"""The controller file holding the cgroup's *instantaneous* usage, in bytes.

A sibling of :data:`_PEAK_FILE` in the same cgroup directory. It is the
subtrahend of every reported point: a per-fd watermark reset does not zero the
memory that is already charged to the cgroup, so the reset watermark starts at
whatever ``memory.current`` reads at that instant.
"""


class GridPoint(NamedTuple):
    """One measured cell of the design F12 grid.

    Attributes:
        workers: Worker processes the point ran with.
        n_full: ``gumbel_n_full`` the point ran with -- design Q5's correction,
            that the memory variable is the simulation count and not the
            branching factor, is exactly what this axis exists to test.
        peak_bytes: The cgroup's memory **growth over this point**, in bytes:
            the per-fd high-water mark read after the child exited, minus the
            cgroup's ``memory.current`` at the instant the mark was reset. It
            is *not* the cgroup's lifetime peak (that is
            :func:`cgroup_memory_peak`, and it would be dominated by whatever
            the cgroup did before the grid started). It is attributable to the
            point plus whatever cohabitants of the cgroup allocated while the
            point ran, which is what :func:`cgroup_is_shared` exists to rule
            out.
    """

    workers: int
    n_full: int
    peak_bytes: int


def _peak_path() -> Path | None:
    """Return this process's ``memory.peak`` file, or ``None`` if there is none.

    ``/proc/self/cgroup``'s ``0::<path>`` line names the cgroup v2 path this
    process belongs to, relative to the hierarchy root; the controller files
    live at ``/sys/fs/cgroup<path>/``. Both halves can be absent for reasons
    that are not failures: a cgroup v1-only system has no ``0::`` line at all,
    and a cgroup with the memory controller disabled has the directory but not
    the file. The root cgroup is tried as a fallback, which is what a container
    with its own cgroup namespace usually presents.
    """
    try:
        text = _PROC_CGROUP.read_text()
    except OSError:
        return None
    relative: str | None = None
    for line in text.splitlines():
        # "hierarchy-id:controllers:path"; the unified (v2) hierarchy is the
        # one with an empty controller field, i.e. the line starting "0::".
        if line.startswith("0::"):
            relative = line[3:].strip()
            break
    candidates: list[Path] = []
    if relative is not None:
        candidates.append(_CGROUP_ROOT / relative.lstrip("/") / _PEAK_FILE)
    candidates.append(_CGROUP_ROOT / _PEAK_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def cgroup_memory_peak() -> int | None:
    """Return this cgroup's memory high-water mark in bytes, or ``None``.

    ``None`` means *unavailable*, not *zero*, and it has two ordinary causes:
    the system is not running cgroup v2 (no ``0::`` line in
    ``/proc/self/cgroup``), or the memory controller is not enabled for this
    cgroup (no ``memory.peak`` file). A developer laptop can be either; callers
    and tests are expected to skip rather than to fail, which is why this
    returns ``None`` instead of raising.

    The value is monotone for the life of the cgroup -- it is a *peak*, not a
    usage -- so it is only meaningful next to a statement of what ran since it
    was last reset. This reads through a **fresh open**, so what it returns is
    the cgroup's LIFETIME high-water mark and never what :func:`measure_grid`
    reports: a reset written elsewhere applies only to that file description
    (Linux >= 6.11), so this function is an availability probe, not a
    measurement.
    """
    path = _peak_path()
    if path is None:
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _open_peak_reset() -> int | None:
    """Reset the high-water mark and return the fd the reset belongs to.

    Opens ``memory.peak`` read-write, writes ``0`` to it, and hands back the
    still-open descriptor. The descriptor is the point of the function: since
    Linux 6.11 the write resets the watermark only for *this* file description,
    so the reset is worthless to anyone who reads the path through a different
    ``open``. The caller keeps the fd for the whole measurement, reads it back
    with :func:`_read_peak`, and closes it.

    Returns ``None`` if there is nothing to reset -- no controller file
    (:func:`_peak_path`), or the open/write is refused, as it is when the cgroup
    filesystem is mounted read-only for a sandboxed run. A ``None`` means the
    point cannot be measured at all; it is not a degraded reading.
    """
    path = _peak_path()
    if path is None:
        return None
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDWR)
        os.write(fd, b"0")
    except OSError:
        if fd is not None:
            os.close(fd)
        return None
    return fd


def _read_peak(fd: int) -> int:
    """Read the high-water mark back through the fd that reset it.

    Uses ``pread`` at offset 0 so the read does not depend on -- or disturb --
    the file offset the reset write left behind, and so that the *same* file
    description is used. Reopening the path here would silently substitute the
    cgroup's lifetime mark for the point's, which is the KOM-055 defect.

    Args:
        fd: A descriptor returned by :func:`_open_peak_reset`, still open.

    Returns:
        The watermark in bytes, as this file description has seen it since the
        reset.

    Raises:
        OSError: If the descriptor cannot be read.
        ValueError: If the controller file's contents are not an integer.
    """
    return int(os.pread(fd, 64, 0).decode().strip())


def _read_current() -> int | None:
    """Return the cgroup's instantaneous memory usage in bytes, or ``None``.

    ``memory.current`` sits next to ``memory.peak`` in the same cgroup
    directory. It is the floor a freshly reset watermark starts from -- the
    reset does not un-charge memory that is already charged -- so subtracting it
    is what turns the mark into the growth a grid point caused.

    ``None`` means unavailable for the same ordinary reasons
    :func:`cgroup_memory_peak` returns it (no unified hierarchy, no memory
    controller), plus a kernel that exposes ``memory.peak`` without
    ``memory.current``; callers decide whether that is a skip or a failure.
    """
    path = _peak_path()
    if path is None:
        return None
    try:
        return int((path.parent / _CURRENT_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def memory_peak_is_resettable() -> bool:
    """Whether this process can actually reset ``memory.peak``.

    A ``False`` is the third way a grid becomes unmeasurable, next to a missing
    controller (:func:`cgroup_memory_peak`) and a cohabited cgroup
    (:func:`cgroup_is_shared`), and it is the one neither of those catches: the
    cgroup filesystem is mounted read-only, as it is for a sandboxed gate run
    inside a systemd **service** cgroup. The controller file is there and
    readable, nobody foreign is in the cgroup -- but the write is refused, so
    the number read after a point is the cgroup's LIFETIME high-water mark
    (the service manager's, accumulated since boot) rather than the child's.
    KOM-055 saw exactly that: a toy point "peaking" at 5.4 GB, byte-for-byte
    the daemon service's own ``memory.peak``.

    This performs one real reset attempt and closes the descriptor again, so
    calling it does clear the mark for a file description nobody keeps -- which
    is harmless, because :func:`measure_grid` opens and resets its own fd
    immediately before every point it measures.
    """
    fd = _open_peak_reset()
    if fd is None:
        return False
    os.close(fd)
    return True


def _grid_child(
    model_factory: Callable[[], torch.nn.Module],
    orchestrator: OrchestratorConfig,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> None:
    """Run one grid point's self-play. The spawned child's entry point.

    Module-level and taking only picklable arguments, for the same reason
    :func:`~komugi.selfplay.worker.worker_main` is (design D2/F18) -- and it
    returns nothing on purpose: the measurement is read from the cgroup by the
    parent after this process is gone, so there is no result to send back and
    no queue to keep alive while the peak is being taken.
    """
    run_selfplay(model_factory, orchestrator, selfplay, inference)


def measure_grid(
    model_factory: Callable[[], torch.nn.Module],
    points: Sequence[tuple[int, int]],
    base_orchestrator: OrchestratorConfig,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> list[GridPoint]:
    """Measure peak memory across a ``(workers, n_full)`` grid (design F12).

    Each point runs a **short** self-play -- ``base_orchestrator``'s
    ``games_per_worker``, which the caller keeps small -- in a fresh spawned
    child, under its own shard root so the points cannot collide on a writer's
    single-writer directory. The child's peak is read from the cgroup after it
    exits; see the module docstring for why a subprocess is the only way to get
    a number that is not floored by the parent.

    Each point resets ``memory.peak`` through an fd it then holds open for the
    whole point and reads back with ``pread``, because the reset applies to
    that file description alone (KOM-055; module docstring). What lands in
    ``peak_bytes`` is that mark minus ``memory.current`` at the reset, i.e. the
    cgroup's growth over the point. A refused reset is no longer a degraded
    reading that gets reported anyway -- it is unmeasurable, and raises.

    Args:
        model_factory: Module-level factory (design F18); it is called in the
            grandchild server process of each point, never here.
        points: ``(workers, n_full)`` pairs, measured in the order given.
        base_orchestrator: Everything a point does not vary -- seed, games per
            worker, timeouts, and the shard root the per-point roots hang off.
        selfplay: The search config; each point overrides ``workers`` and
            ``gumbel_n_full`` on a copy.
        inference: The seam config, unchanged across points.

    Returns:
        One :class:`GridPoint` per input pair, in order.

    Raises:
        TypeError: If `model_factory` is not module-level.
        ValueError: On a non-positive ``workers`` or ``n_full``.
        RuntimeError: If ``memory.peak`` is unavailable (see
            :func:`cgroup_memory_peak` -- the caller should skip rather than
            fail), if it cannot be reset or read back through the fd that reset
            it (KOM-055 -- callers that can skip test
            :func:`memory_peak_is_resettable` first, as the behaviour suite's
            fixture does), or if a point's child exits nonzero.
    """
    require_module_level_factory(model_factory)
    if cgroup_memory_peak() is None:
        raise RuntimeError(
            "cgroup v2 memory.peak is unavailable on this machine (no '0::' "
            "line in /proc/self/cgroup, or no memory controller for this "
            "cgroup), so there is nothing to measure the design F12 grid "
            "against. Run the grid on a cgroup v2 host, or skip it -- a grid "
            "measured without a peak counter would be a fabricated number"
        )

    ctx = get_context("spawn")
    root = Path(base_orchestrator.shard_root)
    measured: list[GridPoint] = []
    for workers, n_full in points:
        if workers < 1 or n_full < 1:
            raise ValueError(
                f"grid point ({workers}, {n_full}) is not runnable: both "
                f"workers and n_full must be >= 1"
            )
        orchestrator = replace(
            base_orchestrator,
            workers=workers,
            shard_root=str(root / f"w{workers}_n{n_full}"),
        )
        point_selfplay = replace(selfplay, workers=workers, gumbel_n_full=n_full)

        # Reset first, so the mark this point reports is this point's -- and
        # hold the fd, because on Linux >= 6.11 the reset is that fd's alone
        # (KOM-055). Everything up to the close is one measurement.
        fd = _open_peak_reset()
        if fd is None:
            raise RuntimeError(
                f"KOM-055: memory.peak could not be reset before point "
                f"(workers={workers}, n_full={n_full}), so this point cannot "
                f"be measured -- the only number available would be the "
                f"cgroup's LIFETIME high-water mark, which belongs to whatever "
                f"ran in this cgroup before the grid did. The usual cause is a "
                f"read-only cgroup filesystem, as inside a sandboxed run. Call "
                f"memory_peak_is_resettable() first and skip on False"
            )
        try:
            baseline = _read_current()
            if baseline is None:
                raise RuntimeError(
                    f"KOM-055: memory.current is unreadable next to a writable "
                    f"memory.peak while measuring point (workers={workers}, "
                    f"n_full={n_full}); without the usage at the reset there is "
                    f"no baseline to subtract, and the mark alone is not this "
                    f"point's growth"
                )
            child = ctx.Process(
                target=_grid_child,
                args=(model_factory, orchestrator, point_selfplay, inference),
                name=f"komugi-grid-w{workers}-n{n_full}",
            )
            child.start()
            child.join()
            if child.exitcode != 0:
                raise RuntimeError(
                    f"grid point (workers={workers}, n_full={n_full}) exited "
                    f"with code {child.exitcode}; its peak would describe a "
                    f"failed run"
                )
            # After the child is gone: its memory is back with the kernel, but
            # the mark it set is still ours to read -- through this fd only.
            try:
                peak = _read_peak(fd)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"cgroup memory.peak became unreadable while measuring "
                    f"point (workers={workers}, n_full={n_full}): {exc}"
                ) from exc
        finally:
            os.close(fd)
        measured.append(
            GridPoint(
                workers=workers,
                n_full=n_full,
                # Growth, not level: the reset does not un-charge the memory
                # the cgroup already held. Clamped, because a cohabitant
                # freeing memory during the point can put current above the
                # mark and a negative "peak" is not a thing.
                peak_bytes=max(0, peak - baseline),
            )
        )
    return measured


def cgroup_is_shared() -> bool | None:
    """Whether foreign processes cohabit this cgroup, or ``None`` if unknowable.

    On a desktop session the process usually lands in an interactive app scope
    (a browser's, a terminal's), and ``memory.peak`` there is the HIGH-WATER
    MARK OF EVERYTHING in the scope -- a grid measured in one is cohabitant-
    dominated and meaningless (wave-9c review F7; observed: a toy point
    "peaking" at 46 GiB of browser). Callers must treat ``True`` as
    unmeasurable, exactly like a missing controller, and run the committed
    grid under an isolated scope instead::

        systemd-run --user --scope -- python -m ...

    A pid in ``cgroup.procs`` counts as foreign unless it is this process,
    one of its ancestors (the shells and runners that launched it), or a
    descendant (our spawned children).
    """
    peak = _peak_path()
    if peak is None:
        return None
    try:
        pids = [
            int(line)
            for line in (peak.parent / "cgroup.procs").read_text().split()
        ]
    except (OSError, ValueError):
        return None

    def parent_of(pid: int) -> int | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        # comm is parenthesised and may contain spaces: split after the last ')'.
        try:
            return int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            return None

    me = os.getpid()
    ancestors: set[int] = set()
    cursor: int | None = me
    while cursor is not None and cursor > 1 and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = parent_of(cursor)

    def is_descendant(pid: int) -> bool:
        seen: set[int] = set()
        walker: int | None = pid
        while walker is not None and walker > 1 and walker not in seen:
            if walker == me:
                return True
            seen.add(walker)
            walker = parent_of(walker)
        return False

    return any(pid not in ancestors and not is_descendant(pid) for pid in pids)
