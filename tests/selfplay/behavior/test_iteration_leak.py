"""Successive self-play iterations must return their resources, every time.

Audit gap G5 (``docs/design/kom-101-121-test-debt-audit.md`` section 4): the
suite proves that *one* run cleans up after itself
(``test_e2e_loop.test_the_run_left_no_shared_memory_segment_behind``,
``test_failures_are_loud``'s per-raise ``/dev/shm`` sweeps) and never that
*repeating* a run does.  A campaign is a loop
(:mod:`komugi.campaign.loop`): iteration ``i + 1``'s self-play runs in the same
supervising interpreter that ran iteration ``i``'s, so anything the
orchestrator forgets to give back is paid for again at every iteration and the
failure appears fifty iterations into an overnight campaign rather than in CI.

This module runs :data:`_ITERATIONS` successive bounded runs in ONE process and
asserts that three resources come back to where they started:

* **memory** -- the cgroup's ``memory.peak``, which is what design F12 already
  measures against (:func:`komugi.selfplay.cgroup_memory_peak`);
* **shared memory** -- the ``psm_*`` segments in ``/dev/shm``, by NAME;
* **file descriptors** -- the entries in ``/proc/self/fd``, by count.

Three properties of the harness are what make the numbers mean anything, and
each is a deliberate choice rather than an accident of how it was written.

**Every iteration does identical work.**  Same ``run_seed``, same worker count,
same game budget, same net.  A campaign varies the seed per iteration
(:func:`komugi.campaign.loop.iteration_seed`) and this deliberately does not:
the whole assertion is a *delta* between iterations, so any difference in the
work would show up as a difference in the resources and the test would be
measuring the games rather than the leak.

**Iteration 1 is excluded from the memory bound, and is the baseline for the
descriptors.**  A first iteration legitimately allocates things that are then
kept for the life of the process: torch's lazy imports and allocator arenas,
and -- this is the one that matters for descriptors -- the
:mod:`multiprocessing` ``resource_tracker``, a child born on the first use of
shared memory that lives forever and holds a pipe in this process.  Warm-up is
not a leak; a *second* copy of the same allocation is.

**Each iteration's shard tree is deleted once it has been measured.**
``memory.peak`` is a high-water mark over everything charged to the cgroup, and
that includes the page cache of the shards the run just wrote -- one row is
~188 KB of observation plus policy target, so a game is tens of megabytes, and
on a ``tmpfs`` ``/tmp`` those bytes are resident memory rather than reclaimable
cache.  Left on disk they would accumulate across iterations and the peak would
climb for a reason that has nothing to do with the orchestrator.  Unlinking a
file drops its page cache, so removing the tree returns the charge and leaves
each iteration's peak describing that iteration.

What this module does NOT claim: that a leak of a given size is *absent*, only
that none of the three resources grows past a pre-registered bound over
:data:`_ITERATIONS` iterations.  A leak smaller than the noise of re-importing
torch in two fresh children is invisible to the memory assertion by
construction -- which is exactly why the segment and descriptor assertions are
exact-set and exact-count rather than bounded: those two are the ones a small
leak shows up in first.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_ITERATIONS: Final[int] = 4
"""How many successive runs to drive.

Four, not three.  Iteration 1 is warm-up and iteration 2 is the baseline, so
three iterations would leave exactly ONE measured delta -- and a single delta
cannot tell a step (one iteration paid a one-off cost) from a slope (every
iteration pays it).  Four gives two, and a leak is the shape that grows in both.
"""

_BASELINE: Final[int] = 2
"""The iteration whose readings the later ones are held to (1-based).

Not iteration 1: see the module docstring on warm-up.  The descriptor baseline
is iteration 1 rather than 2, because a descriptor is a counted thing with no
allocator noise -- once the tracker's pipe exists, the count is exact, and
holding iterations 2..N to iteration 1 is the stronger claim.
"""

_WORKERS: Final[int] = 1
_GAMES_PER_WORKER: Final[int] = 1
"""The smallest run the rig supports: one worker, one game, one server.

The subject is the *iteration boundary*, not the fan-out -- a second worker
would double the wall-clock and add nothing, since every resource this module
watches is created and destroyed once per launch whatever the worker count is.
"""

_RUN_SEED: Final[int] = 5050505
"""One seed for every iteration, on purpose -- see the module docstring."""

_ITERATION_WALL_S: Final[float] = 300.0
"""Wall on one iteration.  It bounds a *hang*; it is not a latency assertion.

``rig.E2E_WALL_S`` is 900s for a 2-worker x 2-game run; this is a quarter of
that run driven four times over, so 300s per iteration is proportionally the
more generous of the two while keeping a wedged gate bounded by 20 minutes
rather than an hour.  A healthy pass is expected to be far under it: the target
for the whole module is well inside two minutes.
"""

_PEAK_DELTA_BOUND_BYTES: Final[int] = 128 * 1024**2
"""Pre-registered: how much ``memory.peak`` may rise PER ITERATION past the
baseline.

**This number was chosen before this test was run for the first time**, from
the arithmetic below and not from an observed reading -- which is the only
thing that makes it evidence rather than a transcription of whatever the
machine happened to do.

The reasoning it was chosen by.  Every iteration performs byte-identical work
and deletes its own output before the next one starts, so the *expected* delta
is zero: the two children are spawned fresh and their torch footprint is a
high-water mark iteration 2 has already set.  What is left is noise -- a
different allocator arena layout, a different page-fault pattern, a
differently-sized transient in a fresh interpreter -- which is bounded by a
fraction of one torch import and nowhere near 128 MiB.  On the other side, the
allocations this test exists to catch are orders of magnitude LARGER than the
bound: design D9's ``_Node`` is ~1.08 MB and a search holds ``n + 1`` of them,
so one retained search per ply is a gigabyte inside a single game.  The gap
between "noise" and "a leak worth having a test for" is wide enough that the
bound's exact value is not load-bearing; what is load-bearing is that it was
fixed in advance.

It is a per-iteration allowance, multiplied by the span, because a leak is paid
once per iteration: holding iteration 4 to the same total as iteration 3 would
make the test's strictness depend on :data:`_ITERATIONS`.
"""


@dataclass(frozen=True)
class Iteration:
    """Everything one iteration is judged on, read while it was fresh.

    Attributes:
        index: 1-based iteration number, as the assertion messages spell it.
        games_completed: ``RunSummary.games_completed`` for this iteration.
        rows_written: ``RunSummary.rows_written`` for this iteration.  Non-zero
            is what makes the three leak assertions non-vacuous: a run that
            played nothing would leak nothing either.
        peak_bytes: ``cgroup_memory_peak()`` after the run, or ``None`` where
            this machine has no readable ``memory.peak``.
        fd_count: How many entries ``/proc/self/fd`` held after the run.
        fd_targets: ``fd -> readlink target``, kept only so a failure can say
            *what* leaked rather than merely that something did.
        shm_survivors: ``psm_*`` segments that appeared during the run and were
            still in ``/dev/shm`` afterwards.
    """

    index: int
    games_completed: int
    rows_written: int
    peak_bytes: int | None
    fd_count: int
    fd_targets: dict[str, str]
    shm_survivors: frozenset[str]


@dataclass(frozen=True)
class LeakRun:
    """The whole sequence, plus why memory may not be judgeable here.

    Attributes:
        iterations: One :class:`Iteration` per run, in order.
        memory_unmeasurable: ``None`` when ``memory.peak`` is a number this
            machine's readings can be trusted to describe; otherwise the loud
            reason the memory assertion skips with.  The segment and descriptor
            assertions do not consult it -- they need no cgroup at all.
    """

    iterations: tuple[Iteration, ...]
    memory_unmeasurable: str | None


_FD_DIR: Final[Path] = Path("/proc/self/fd")
"""This process's open descriptors, as the kernel lists them."""


def _open_fds() -> dict[str, str]:
    """Return this process's open descriptors, ``fd -> target``.

    ``os.listdir`` opens a directory descriptor of its own, and that descriptor
    is in the listing it returns and closed by the time this function reads the
    links -- so exactly one entry resolves to ``<gone>`` in **every** snapshot.
    That is a constant, not noise: it cancels in a comparison of two snapshots,
    which is the only way this result is ever used.  Leaving it in is cheaper
    than filtering and cannot hide a real descriptor, because a leaked one is
    still open and still resolves.
    """
    found: dict[str, str] = {}
    for entry in sorted(os.listdir(_FD_DIR), key=int):
        try:
            found[entry] = os.readlink(_FD_DIR / entry)
        except OSError:
            found[entry] = "<gone>"
    return found


def _memory_unmeasurable() -> str | None:
    """Return why ``memory.peak`` cannot be trusted here, or ``None``.

    The same two gates ``test_memory_grid`` applies, and its wording, because
    they are the same two facts about the machine.  One deliberate difference:
    a ``cgroup_is_shared()`` of ``None`` -- ``cgroup.procs`` unreadable, so
    cohabitation is *unknowable* -- is treated as unmeasurable here where the
    grid treats it as fine.  The grid asserts a 2 GiB ceiling, which survives a
    cohabitant; this module asserts a delta of a few megabytes between two
    moments, which a browser tab allocating in the same scope would blow apart.
    A test that cannot tell a leak from a cohabitant must say so, not guess.
    """
    peak = sp.cgroup_memory_peak()
    if peak is None:
        return (
            "cgroup_memory_peak() is None: this machine exposes no cgroup v2 "
            "memory.peak (no unified hierarchy, or the memory controller is "
            "not delegated to this cgroup), so there is no high-water mark to "
            "take a between-iterations delta of. The segment and descriptor "
            "halves of this module still ran."
        )
    if not isinstance(peak, int) or peak <= 0:
        return (
            f"cgroup_memory_peak() returned {peak!r}: a non-None reading is a "
            f"byte count and must be a positive integer, so this machine's "
            f"readings cannot be differenced."
        )
    shared = sp.cgroup_is_shared()
    if shared is None:
        return (
            "cgroup_is_shared() is None: this cgroup's cgroup.procs is "
            "unreadable, so whether foreign processes cohabit it -- and "
            "therefore whether memory.peak moved because of THIS run -- cannot "
            "be established. An unattributable delta is not evidence. The "
            "segment and descriptor halves of this module still ran."
        )
    if shared:
        return (
            "this process cohabits its cgroup with foreign processes (a "
            "desktop app scope), so memory.peak is the high-water mark of "
            "everything in it -- a toy grid point 'peaked' at 46 GiB of "
            "browser here (wave-9c review F7). An iteration-to-iteration delta "
            "measured in one would be the cohabitant's allocations, not this "
            "run's. Run under an isolated scope: systemd-run --user --scope."
        )
    return None


def _run_once(index: int, root: Path) -> Any:
    """Drive one bounded self-play iteration into `root` and return its summary.

    Its own shard root per iteration, because
    :class:`~komugi.train.shards.ShardWriter` allows one writer per directory
    and every iteration launches incarnation 0 -- a shared root would put two
    iterations' writers in the same ``worker_0_r0``.  It is also what a
    campaign does (``<root>/selfplay/iter_<i>``), so the layout under test is
    the production one.
    """
    return rig.walled_run_selfplay(
        rig.tiny_factory,
        rig.orchestrator_config(
            root,
            run_seed=_RUN_SEED,
            workers=_WORKERS,
            games_per_worker=_GAMES_PER_WORKER,
        ),
        rig.tiny_selfplay_config(workers=_WORKERS),
        rig.inference_config(max_batch=_WORKERS),
        wall_s=_ITERATION_WALL_S,
        label=f"leak-iter-{index}",
    ).unwrap()


@pytest.fixture(scope="module")
def leak_run(tmp_path_factory: pytest.TempPathFactory) -> LeakRun:
    """Run the iterations once and record what each of them gave back.

    One fixture for the whole module because every assertion below is a
    different view of the same sequence, and the sequence costs real seconds.
    """
    base = Path(tmp_path_factory.mktemp("iteration-leak"))
    unmeasurable = _memory_unmeasurable()

    # Taken ONCE, before the first iteration: `new_shm_survivors` is then asked
    # the same question after every iteration -- "what psm_* segment appeared
    # since the module started and is still there?" -- which is strictly
    # stronger than comparing each iteration against iteration 1's leftovers,
    # and coincides with it exactly when iteration 1 leaked nothing, which is
    # what the iteration-1 case of the assertion proves.
    shm_before = rig.shm_snapshot()

    measured: list[Iteration] = []
    for index in range(1, _ITERATIONS + 1):
        root = base / f"iter_{index}" / "shards"
        root.mkdir(parents=True)
        summary = _run_once(index, root)

        # Descriptors first: it is the cheapest reading and the one most easily
        # perturbed by anything else this fixture might do. Then segments (the
        # rig re-checks survivors across a grace, so this can cost seconds --
        # but only when something actually survived). Then the peak, which is a
        # high-water mark and so cannot be lowered by reading it late.
        fds = _open_fds()
        survivors = rig.new_shm_survivors(shm_before)
        peak = sp.cgroup_memory_peak()

        measured.append(
            Iteration(
                index=index,
                games_completed=int(summary.games_completed),
                rows_written=int(summary.rows_written),
                peak_bytes=None if peak is None else int(peak),
                fd_count=len(fds),
                fd_targets=fds,
                shm_survivors=frozenset(survivors),
            )
        )

        # Both of these are cleanup, and both happen strictly AFTER the
        # readings above so that neither can launder a leak: killing a
        # straggler closes the descriptors it held, and deleting the tree
        # returns its page-cache charge (module docstring). Cleaning up before
        # measuring would be a test that repairs what it is looking for.
        rig.kill_stragglers()
        shutil.rmtree(base / f"iter_{index}")

    return LeakRun(iterations=tuple(measured), memory_unmeasurable=unmeasurable)


def _describe_iterations(run: LeakRun) -> str:
    """One line per iteration, for an assertion that has to name the trend."""
    return "; ".join(
        f"iter {entry.index}: peak={entry.peak_bytes} fds={entry.fd_count} "
        f"rows={entry.rows_written}"
        for entry in run.iterations
    )


def test_every_iteration_actually_played_its_games(leak_run: LeakRun) -> None:
    """Non-vacuity: a run that did nothing would leak nothing either.

    The three assertions below are all "X did not grow".  Each of them passes
    trivially against a sequence of runs that never spawned a server, never
    created a segment and never wrote a row, so the sequence has to be shown to
    have happened first -- per iteration, not in total, because a leak test
    whose fourth iteration silently did no work is a test with one measured
    delta pretending to have two.
    """
    assert len(leak_run.iterations) == _ITERATIONS, (
        f"expected {_ITERATIONS} iterations, recorded "
        f"{len(leak_run.iterations)}: {_describe_iterations(leak_run)}"
    )
    for entry in leak_run.iterations:
        assert entry.games_completed >= _WORKERS * _GAMES_PER_WORKER, (
            f"iteration {entry.index} completed {entry.games_completed} games, "
            f"asked for {_WORKERS} worker(s) x {_GAMES_PER_WORKER} game(s)"
        )
        assert entry.rows_written > 0, (
            f"iteration {entry.index} wrote no rows: it created no segments and "
            f"ran no search, so it cannot be evidence about leaking either"
        )


def test_peak_memory_does_not_climb_across_iterations(leak_run: LeakRun) -> None:
    """(a) ``memory.peak`` after iteration N is within the pre-registered bound.

    Iteration 1 is excluded (warm-up is legitimate) and iteration
    :data:`_BASELINE` is the reference; ``memory.peak`` is monotone by
    construction, so the delta can only be non-negative and the question is
    solely whether it stayed inside :data:`_PEAK_DELTA_BOUND_BYTES` per
    iteration.  It climbs when an iteration keeps something a previous one
    also kept: an ``_Incarnation`` retained past its teardown, a segment's
    numpy views left alive, a per-iteration cache in the orchestrator.

    Where the peak is unreadable or the cgroup is shared this SKIPS, loudly and
    with the reason -- the same discipline ``test_memory_grid`` uses, and for
    the same reason: a harness that silently reported "no growth" on a machine
    it could not measure would turn "we never looked" into "we looked and it
    was fine".
    """
    if leak_run.memory_unmeasurable is not None:
        pytest.skip(leak_run.memory_unmeasurable)

    baseline = leak_run.iterations[_BASELINE - 1]
    assert baseline.peak_bytes is not None, (
        f"memory.peak was readable before the run and None after iteration "
        f"{baseline.index}: {_describe_iterations(leak_run)}"
    )
    for later in leak_run.iterations[_BASELINE:]:
        assert later.peak_bytes is not None, (
            f"memory.peak became unreadable at iteration {later.index}: "
            f"{_describe_iterations(leak_run)}"
        )
        span = later.index - baseline.index
        allowance = _PEAK_DELTA_BOUND_BYTES * span
        delta = later.peak_bytes - baseline.peak_bytes
        assert delta <= allowance, (
            f"memory.peak rose {delta / 1024**2:.1f} MiB between iteration "
            f"{baseline.index} and iteration {later.index} ({span} iteration(s) "
            f"of identical work), over the pre-registered allowance of "
            f"{allowance / 1024**2:.0f} MiB "
            f"({_PEAK_DELTA_BOUND_BYTES / 1024**2:.0f} MiB per iteration). "
            f"Every iteration ran the same seed, the same games and deleted its "
            f"own shards, so a rising high-water mark is memory one iteration "
            f"kept. Readings: {_describe_iterations(leak_run)}"
        )


def test_no_shared_memory_segment_survives_any_iteration(leak_run: LeakRun) -> None:
    """(b) ``/dev/shm`` holds no ``psm_*`` segment this module created.

    By name, not by count: :func:`orchestrate_rig.new_shm_survivors` returns
    the segments that appeared since the module's opening snapshot and are
    still there, so a failure names the leaked segment rather than reporting an
    arithmetic discrepancy that could as easily be a foreign process's.

    Design D2/F9 puts every segment's lifetime in the orchestrator's
    ``finally`` (:func:`~komugi.selfplay.orchestrate._teardown`), and Python
    3.12 has no ``SharedMemory(track=...)`` (bpo-38119), so that ``finally`` is
    the entire mechanism -- there is no tracker backstop to catch what it
    misses.  Asserted after **every** iteration including the first, which is
    also what makes the "iteration-1 baseline" this is nominally compared
    against the empty set.
    """
    for entry in leak_run.iterations:
        assert not entry.shm_survivors, (
            f"iteration {entry.index} left {len(entry.shm_survivors)} "
            f"shared-memory segment(s) in /dev/shm: "
            f"{sorted(entry.shm_survivors)}. Each is "
            f"workers x games_in_flight segments' worth of pressure that every "
            f"later iteration of a campaign inherits"
        )


def test_open_descriptors_return_to_the_first_iterations_count(
    leak_run: LeakRun,
) -> None:
    """(c) ``/proc/self/fd`` is the same size after iteration N as after 1.

    Exactly the same size, in both directions.  The orchestrator's descriptors
    are all per-incarnation and all closed on the way out -- each created
    segment holds one (``SharedMemory`` keeps its ``shm_open`` descriptor until
    :meth:`~multiprocessing.shared_memory.SharedMemory.close`, which
    :meth:`komugi.selfplay.Slots.close` calls), and each spawned child holds a
    sentinel pipe until it is joined -- so a correct iteration ends where it
    started and a count that grew names which of the two was skipped.

    Iteration 1 is the baseline rather than iteration 2 because the descriptors
    a first iteration legitimately adds are one-off and permanent (the
    :mod:`multiprocessing` resource tracker's pipe), so they are inside the
    baseline rather than inside the delta.  Nothing about this reading needs a
    cgroup, so it runs on every machine.
    """
    baseline = leak_run.iterations[0]
    for later in leak_run.iterations[1:]:
        if later.fd_count == baseline.fd_count:
            continue
        opened = {
            fd: target
            for fd, target in later.fd_targets.items()
            if fd not in baseline.fd_targets
        }
        closed = {
            fd: target
            for fd, target in baseline.fd_targets.items()
            if fd not in later.fd_targets
        }
        raise AssertionError(
            f"iteration {later.index} ended with {later.fd_count} open "
            f"descriptors, iteration {baseline.index} with "
            f"{baseline.fd_count}. Descriptors not open at the baseline: "
            f"{opened or '<none>'}; descriptors the baseline had and this "
            f"iteration does not: {closed or '<none>'}. Every iteration runs "
            f"the same work, so the difference is a descriptor an iteration "
            f"opened and did not give back. Readings: "
            f"{_describe_iterations(leak_run)}"
        )
