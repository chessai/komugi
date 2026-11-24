"""The orchestrator: who owns the processes, the segments, and the restarts.

Design ``docs/design/kom-021-selfplay.md`` D2 (as amended by KOM-021b's doorbell
protocol), D10 and D11 give this module three jobs, and the wave-9a/9b reviews
turned each of them into an obligation with a name. They are listed here and
cited again at the line that enforces them, because every one of them is a rule
whose violation is silent:

* **The orchestrator owns every SharedMemory segment.** It creates them
  (:func:`~komugi.selfplay.inference.create_slots`), hands out *names*, and
  unlinks them in a ``finally``. Workers and servers only ever attach, through
  :func:`~komugi.selfplay.inference.attach_untracked`. Before unlinking, this
  module **re-registers** the segment with the resource tracker
  (:func:`reregister_and_unlink`) -- wave-9a review F4. There are ``workers *
  SelfPlayConfig.games_in_flight`` of them (KOM-040 b1): an interleaved worker
  plays each of its concurrent games over its own segment, so the rule reads
  "one segment per lane" and is otherwise word for word what it was. Their
  *names* say who owns them (:func:`slot_shm_name`, KOM-054), because
  ``/dev/shm`` is machine-global and the ownership rule is otherwise
  unauditable from outside the process.
* **A segment is never reused across worker incarnations**, and *any* worker
  death restarts the **whole shard**: the server's client list is fixed at
  ``serve`` start, so a respawned worker's fresh segment can never reach a
  running :class:`~komugi.selfplay.inference.ShardServer` (doorbell review
  F4/F5). Every child is SIGKILLed and **joined -- confirmed dead** -- before
  the successor incarnation starts, because a zombie worker still holding its
  old shard directory would collide with its successor on
  :class:`~komugi.train.shards.ShardWriter`'s single-writer seal (design F6).
* **Server death fails the run loudly** (design D2/Q4), with the server's exit
  code in the message. Per-shard respawn with generation counters is the
  pre-registered P1 ticket, not a thing to improvise here.

What a restart costs, stated once
---------------------------------
A restarted shard replays its **full** ``games_per_worker`` budget under
incarnation ``i + 1``. That is not a bug and not a duplicate: ``incarnation`` is
in :func:`~komugi.selfplay.records.game_seed`'s spawn key (design F10c), so the
successor's game 0 is a *different game* from its predecessor's game 0, and its
``game_id`` differs too (:func:`~komugi.selfplay.worker.game_id` packs the same
triple). The consequence is that a run with restarts produces **more** than
``workers * games_per_worker`` games; P0a accepts that and
:class:`RunSummary` reports what actually happened rather than what was asked
for.

Sticky worker -> shard
----------------------
P0a runs ``InferenceConfig.shards == 1``. Assignment of workers to shards is
sticky by construction here -- every worker's slot is scanned by exactly one
server -- because two shards scanning one slot would break ``response_seq``'s
single-writer discipline (design D2/D3). ``shards > 1`` raises
:class:`NotImplementedError`: the multi-shard layout wants the dynamic
worker registration that design D2/Q4 pre-registers as the P1 ticket.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import warnings
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from multiprocessing import get_context, resource_tracker, synchronize
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final, NamedTuple
from uuid import uuid4

import numpy as np
import torch

from komugi.config import SelfPlayConfig
from komugi.selfplay.inference import (
    InferenceConfig,
    InProcessEvaluator,
    Slots,
    SlotHandle,
    create_slots,
    serve_shard,
)
from komugi.selfplay.records import SCHEMA
from komugi.selfplay.worker import (
    LaneSpec,
    SelfPlayWorker,
    WorkerConfig,
    lane_slot_id,
    worker_main,
)
from komugi.telemetry.liveness import log_freshness
from komugi.train.shards import COUNT_KEY, SHARD_GLOB, ShardWriter

_POLL_S: Final[float] = 2.0
"""How long the monitor sleeps between two passes over the shard's children.

Two seconds, because every signal it reads is coarse: a process exit code, and
a directory mtime whose grain is one *game*. Polling faster would only spend
CPU to learn the same thing, and the liveness gate is specified in minutes.
"""

_KILL_JOIN_S: Final[float] = 30.0
"""Wall on the ``join()`` that follows a SIGKILL.

SIGKILL cannot be blocked, so this join returns in milliseconds in every case
that is not a kernel-level wedge (uninterruptible I/O). It is a wall on a
*confirmation*, and the confirmation is mandatory: design F6 restarts a shard
only after every predecessor is reaped, because an unreaped worker still owns
its shard directory and would make its successor's first seal raise
``FileExistsError``.
"""

_CORRUPT_SHARD_ERRORS: Final[tuple[type[Exception], ...]] = (
    OSError,
    ValueError,
    KeyError,
    zipfile.BadZipFile,
)
"""What a half-written or damaged shard raises when its count is read.

A killed worker leaves at most one such file behind, which is the expected
shape of a preemption rather than an error: :func:`count_rows` skips it and
counts the rest, exactly as :class:`~komugi.train.shards.ShardReader` does.
"""


# ---------------------------------------------------------------------------
# segment naming (KOM-054)
#
# ``/dev/shm`` is MACHINE-global and ``SharedMemory``'s default name is
# ``psm_<random hex>``, which says nothing about who made it. That is fine for
# the protocol -- only the orchestrator creates and only its children attach,
# both by name -- and it is exactly wrong for anyone auditing the invariant
# from outside: "did this run leak a segment?" can only be answered by diffing
# a global directory, so a concurrent orchestrator anywhere on the box (another
# suite, a worker gate, an experiment) reads as this run's leak. KOM-053's
# investigation root-caused an overnight gate flake to precisely that.
#
# So a segment carries who made it. Two components, because the two questions
# are different:
#
# * the SCOPE identifies this process tree, and is what an auditor in the
#   process that started the run can recompute without being told anything;
# * the RUN TOKEN identifies one ``run_selfplay`` call within that tree, so a
#   human reading ``/dev/shm`` can tell two concurrent runs of the same tree
#   apart, and so nothing can collide.
# ---------------------------------------------------------------------------

SHM_SCOPE_ENV: Final[str] = "KOMUGI_SHM_SCOPE"
"""Environment variable carrying this process TREE's segment-name scope.

The environment rather than a module global, because a scope that stopped at
the process boundary would be useless to the one caller that needs it most:
:func:`~komugi.selfplay.memgrid.measure_grid` runs each grid point's
``run_selfplay`` in a *spawned child*, and the auditor watching ``/dev/shm``
is the parent. A spawned child inherits ``os.environ``, so publishing the
scope there at import time makes every descendant that runs an orchestrator
share the scope of the process that first imported this module -- which is the
process tree the auditor is asking about.

Set it yourself only to *join* an existing tree's scope; leaving it unset is
the normal case and mints a fresh one.
"""

_SHM_FAMILY: Final[str] = "psm_"
"""``SharedMemory``'s own prefix, kept deliberately.

Every tool, runbook and docstring in this repository that talks about leaked
segments talks about ``psm_*``, and a POSIX segment named anything else would
be invisible to all of them. The scope goes *after* the family prefix, not
instead of it.
"""


def _resolve_shm_scope() -> str:
    """Return this process tree's scope, minting and publishing one if needed.

    ``psm_k<pid hex><4 hex of entropy>``. The pid makes it unique among live
    processes; the entropy makes it unique against a *dead* process's leaked
    segment whose pid has since been recycled, which is the only way two scopes
    could ever collide and the only way a name could hit ``FileExistsError``.

    The whole name budget is spent with room to spare. A segment name is
    ``<scope>r<run>_i<incarnation>s<slot>``: at most 15 characters of scope
    (``psm_k`` plus a six-hex pid at Linux's default ``pid_max`` plus four of
    entropy) plus a handful more, against Linux's 255-byte ``NAME_MAX`` and
    even against macOS's much tighter 31-byte ``PSHMNAMLEN`` (the leading ``/``
    that ``shm_open`` prepends is counted).
    """
    inherited = os.environ.get(SHM_SCOPE_ENV)
    if inherited:
        return inherited
    scope = f"{_SHM_FAMILY}k{os.getpid():x}{uuid4().hex[:4]}"
    os.environ[SHM_SCOPE_ENV] = scope
    return scope


_SHM_SCOPE: Final[str] = _resolve_shm_scope()
"""This process tree's scope, fixed at import (see :func:`shm_scope`)."""

_RUN_SERIAL_LOCK: Final[threading.Lock] = threading.Lock()
"""Guards :data:`_run_serial`.

``run_selfplay`` is routinely called on a helper thread (the behavioural suite
walls every call that way), so two runs of one process can mint concurrently
and a read-modify-write of a plain integer is not something to reason about
per-interpreter.
"""

_run_serial: int = 0
"""How many run tokens this process has minted."""


def shm_scope() -> str:
    """Return the name prefix every segment this process TREE creates carries.

    The auditor's entry point (KOM-054): a leak check is a diff of
    machine-global ``/dev/shm``, and this is what makes that diff answerable --
    a name that does not start with this string was not created by any
    orchestrator in this process tree, whatever else it looks like.

    Constant for the life of the process, and shared with every descendant
    through :data:`SHM_SCOPE_ENV`.
    """
    return _SHM_SCOPE


def mint_run_token() -> str:
    """Return a fresh token identifying ONE run's segments.

    Called once per :func:`run_selfplay` and threaded down into every
    :func:`~komugi.selfplay.inference.create_slots` call the run makes, so that
    each of a run's segments -- across every incarnation a restart produces --
    is identifiable as *that run's* by name alone.

    Always starts with :func:`shm_scope`, so filtering by scope catches every
    run of this tree and filtering by a token catches exactly one of them.

    The trailing ``_`` is load-bearing, not decoration: a token is used as a
    *prefix filter*, and without a terminator run ``1``'s token would be a
    prefix of run ``16``'s (``r1`` against ``r10``), so auditing one run would
    silently police another.
    """
    global _run_serial
    with _RUN_SERIAL_LOCK:
        serial = _run_serial
        _run_serial += 1
    return f"{_SHM_SCOPE}r{serial:x}_"


def slot_shm_name(run_token: str, incarnation: int, slot_id: int) -> str:
    """Return the ``/dev/shm`` name for one slot of one incarnation of a run.

    The incarnation is in the name for the same reason it is in
    :func:`worker_shard_dir`'s: a restart tears the predecessor's segments down
    and builds fresh ones, and if the successor reused a name it would race the
    predecessor's ``unlink`` for it. Fresh names make that collision impossible
    rather than unlikely, and they keep a leaked segment legible -- the name
    says which incarnation of which run of which process tree dropped it.

    Args:
        run_token: From :func:`mint_run_token`.
        incarnation: The (re)spawn counter this segment belongs to.
        slot_id: The slot's index in ``workers * games_in_flight`` order --
            :func:`~komugi.selfplay.worker.lane_slot_id`'s numbering.
    """
    return f"{run_token}i{incarnation:x}s{slot_id:x}"


@dataclass(frozen=True)
class OrchestratorConfig:
    """What a self-play run is asked to produce, and how patient it is.

    Separate from :class:`~komugi.config.SelfPlayConfig` (how a game is played)
    and :class:`~komugi.selfplay.inference.InferenceConfig` (how the seam is
    shaped): these are the fields the *process supervisor* reads, and none of
    them reaches a worker except ``run_seed`` and ``games_per_worker``.

    Args:
        run_seed: The run's entropy, and the first component of every game's
            seed key (design D11). Non-negative, because
            :class:`numpy.random.SeedSequence` rejects negative entropy and the
            failure would otherwise surface inside a spawned worker.
        workers: How many worker processes to run. This is the orchestrator's
            number; ``SelfPlayConfig.workers`` describes the same fan-out for
            the parts of the system that never spawn anything, and this one
            wins if they disagree.
        games_per_worker: Games each worker plays before exiting cleanly. A
            restarted shard replays the whole budget under a new incarnation --
            see the module docstring.
        shard_root: Directory the per-worker shard directories live under. Each
            incarnation gets fresh ones (``worker_<k>_r<i>``), because
            :class:`~komugi.train.shards.ShardWriter` allows one writer per
            directory and design F6/F10c will not let a successor share a
            predecessor's (see :func:`worker_shard_dir`).
        liveness_timeout_s: How long a worker's shard directory may go without
            being written to before the shard is restarted (design D10). The
            gate's grain is one game: rows are appended when a game ends, so a
            healthy worker is silent for as long as its longest game. A run
            whose games can exceed this bound must raise it: the worst case is
            the engine's turn cap in plies x a full search x the per-evaluation
            latency, across both of a worker's lanes.
        drain_timeout_s: How long to wait for the server to finish and exit
            after the last worker has. Drain runs only once **every** worker
            has exited 0, so no game is in flight during it and this bound is
            not sized to a game (KOM-165 design review F5 corrected the older
            claim that it was); it covers the server's own shutdown, and 900s
            is generous for that. The bound a long game actually tests is
            ``liveness_timeout_s`` above.
        first_shard_grace_s: How long a worker may exist with an EMPTY shard
            directory before the liveness gate declares it never-started.
            Deliberately its own knob (wave-9c review F3): tying it to
            ``drain_timeout_s`` handed anyone tightening ``liveness_timeout_s``
            a hidden 900s gate on the canonical wedge (server stuck from the
            first evaluation, no game ever seals). Size it to your slowest
            first game.
        max_shard_restarts: How many whole-shard restarts the run tolerates
            before failing. Exceeding it raises with the diagnostics of every
            restart, because a run that restarts forever is a broken run
            pretending to make progress.

    Raises:
        ValueError: On a negative seed, a non-positive count or duration, a
            negative restart budget, or an empty ``shard_root``.
    """

    run_seed: int
    workers: int
    games_per_worker: int
    shard_root: str
    liveness_timeout_s: float = 120.0
    drain_timeout_s: float = 900.0
    first_shard_grace_s: float = 300.0
    max_shard_restarts: int = 2

    def __post_init__(self) -> None:
        """Reject a configuration whose failure would appear in a child."""
        if self.run_seed < 0:
            raise ValueError(f"run_seed must be >= 0, got {self.run_seed!r}")
        for field, count in (
            ("workers", self.workers),
            ("games_per_worker", self.games_per_worker),
        ):
            if count < 1:
                raise ValueError(f"{field} must be >= 1, got {count!r}")
        for field, duration in (
            ("liveness_timeout_s", self.liveness_timeout_s),
            ("drain_timeout_s", self.drain_timeout_s),
            ("first_shard_grace_s", self.first_shard_grace_s),
        ):
            # `not > 0` rather than `<= 0` so a NaN is rejected rather than
            # silently becoming an infinite patience.
            if not duration > 0.0:
                raise ValueError(f"{field} must be > 0, got {duration!r}")
        if self.max_shard_restarts < 0:
            raise ValueError(
                f"max_shard_restarts must be >= 0, got {self.max_shard_restarts!r}"
            )
        if not self.shard_root:
            raise ValueError("shard_root must be a non-empty path")


class RunSummary(NamedTuple):
    """What a finished run actually produced.

    Actually, not nominally: with restarts the totals exceed
    ``workers * games_per_worker`` (see the module docstring), and reporting the
    request back would hide exactly the thing an operator needs to see.

    Attributes:
        games_completed: Games finished by workers that exited cleanly, summed
            over every incarnation. A worker killed mid-game contributes the
            games it had already finished only in ``rows_written``, never here:
            this counter is exit-code bookkeeping, and a killed worker's exit
            code says nothing about how far it got.
        rows_written: Rows readable in the shard directories at the end of the
            run -- one row per move, counted from the shards themselves
            (:func:`count_rows`) rather than predicted.
        shard_dirs: Every directory holding at least one sealed shard, in
            creation order. With ``games_in_flight > 1`` that is one entry per
            productive *lane* (``worker_<k>_r<i>_l<lane>``), not per worker:
            each lane has its own writer, and every consumer of this field
            reads a directory at a time, so listing them separately is what
            makes both lanes' rows reachable. Including restarted incarnations'
            directories
            when they sealed games before the restart (real data, belongs in
            the window), and EXCLUDING directories that produced nothing (a
            wedged incarnation's empty dir would break any reader-per-dir
            consumer; wave-9c review F2 adjudication).
        restarts: How many whole-shard restarts happened.
        worker_incarnations: The incarnation each worker id finished under, in
            worker-id order. All equal in P0a: a restart is always
            shard-wide, because a running server cannot re-attach one
            respawned worker's fresh segment (doorbell review F4/F5).
    """

    games_completed: int
    rows_written: int
    shard_dirs: tuple[str, ...]
    restarts: int
    worker_incarnations: tuple[int, ...]


def worker_shard_dir(
    shard_root: str, worker_id: int, incarnation: int, lane: int | None = None
) -> str:
    """Return the directory worker `worker_id` writes to in `incarnation`.

    ``<shard_root>/worker_<k>_r<i>``. The incarnation is in the *name*, not
    merely in the contents, and design D10/F6 is why: a worker that went silent
    is SIGKILLed, and while the kernel is still tearing it down its
    :class:`~komugi.train.shards.ShardWriter` may yet seal a shard. If the
    successor shared the directory, the two would pick the same next index and
    one of them would raise ``FileExistsError`` -- the single-writer guard
    firing against us. A fresh directory per incarnation makes the collision
    impossible instead of unlikely.

    With ``games_in_flight > 1`` (KOM-040 b1) the same rule applies one level
    down and for the same reason: ``ShardWriter`` is one writer per directory
    and is not thread-safe, so each lane gets ``worker_<k>_r<i>_l<lane>`` and
    its own writer. The suffix goes on the END of the existing name on purpose
    -- the campaign reader globs ``worker_*`` under the shard root
    (:func:`komugi.campaign.loop.iteration_source`), so lane directories are
    picked up as ordinary per-worker directories with no reader change at all,
    and :class:`~komugi.selfplay.records.ConcatSource` concatenates them like
    any others.

    Args:
        shard_root: Directory the per-worker directories live under.
        worker_id: The worker's index in the run.
        incarnation: The (re)spawn counter this directory belongs to.
        lane: The interleaved lane, or ``None`` for a worker that plays one
            game at a time -- whose name is then exactly what it always was.
    """
    name = f"worker_{worker_id}_r{incarnation}"
    if lane is not None:
        name = f"{name}_l{lane}"
    return str(Path(shard_root) / name)


def lane_game_budget(games_per_worker: int, games_in_flight: int, lane: int) -> int:
    """Return how many games lane `lane` of a worker plays (KOM-040 b1).

    The worker's budget split as evenly as it goes, earlier lanes taking the
    remainder: ``games_per_worker=5`` over two lanes is ``3 + 2``. Split rather
    than duplicated -- a worker still plays ``games_per_worker`` games in total,
    so :class:`RunSummary`'s accounting and the run's nominal size are exactly
    what they were; only the wall-clock changes.

    A lane may draw **zero** games (one game over two lanes is ``1 + 0``). That
    lane attaches its segment, writes nothing and exits, which is honest: its
    empty directory is dropped from ``shard_dirs`` by the same rule that drops
    a wedged incarnation's.
    """
    if games_in_flight < 1:
        raise ValueError(f"games_in_flight must be >= 1, got {games_in_flight!r}")
    if not 0 <= lane < games_in_flight:
        raise ValueError(
            f"lane {lane!r} is outside [0, {games_in_flight}) for "
            f"games_in_flight={games_in_flight}"
        )
    return (games_per_worker + games_in_flight - 1 - lane) // games_in_flight


def reregister_and_unlink(shm: SharedMemory) -> None:
    """Re-register `shm` with the resource tracker, then unlink it.

    The obligation from wave-9a review F4, at its enforcement site. The
    resource tracker keys registrations by NAME in one tracker process, so the
    first :func:`~komugi.selfplay.inference.attach_untracked` from *any* child
    deletes the creator's registration as well. Two consequences, and this
    function is the answer to both:

    * A bare ``unlink()`` here would make ``SharedMemory.unlink`` unregister a
      name the tracker no longer holds, and the tracker daemon answers a
      missing key with a ``KeyError`` traceback on stderr -- noise that looks
      exactly like a real leak in a CI log.
    * Between the children's detach and this call the tracker is **not** a
      backstop: an orchestrator that dies badly leaks the segment into
      ``/dev/shm``. Re-registering restores that backstop for the moments when
      it can still act, which is why this is not merely cosmetic.

    Registering a name the tracker still holds is harmless (its cache is a
    set), so this is safe whether or not a child ever attached.

    Args:
        shm: The segment this process created. Its views must already be
            released (:meth:`~komugi.selfplay.inference.Slots.close`).
    """
    # SharedMemory registers (and unlinks under) the slash-prefixed name while
    # reporting `.name` without it, matching attach_untracked's unregister.
    tracked_name = shm.name if shm.name.startswith("/") else f"/{shm.name}"
    registered = True
    try:
        resource_tracker.register(tracked_name, "shared_memory")
    except OSError:
        # The tracker is unreachable (it dies with the process that started
        # it). There is nothing to hand ownership back to, and the unlink below
        # is the part that must happen regardless: an unregistered tracker is a
        # noisy log, an unlinked segment is a leak that outlives the process.
        registered = False
    try:
        # On success this unregisters the name again, which is what makes the
        # pair balanced: register here, unregister there, one of each.
        shm.unlink()
    except FileNotFoundError:
        # Someone (a mis-behaving child, a previous cleanup) got there first.
        # The postcondition -- the segment is gone -- holds, but the
        # registration just made must not outlive it, or the tracker reports a
        # phantom leak at interpreter shutdown.
        if registered:
            with contextlib.suppress(OSError):
                resource_tracker.unregister(tracked_name, "shared_memory")


def count_rows(shard_dir: str | Path) -> int:
    """Return how many rows the sealed shards in `shard_dir` hold.

    The cheap way, deliberately: each shard is an ``npz`` whose ``__count__``
    member is a single integer, and ``np.load`` decompresses members lazily, so
    this reads a few bytes per shard instead of the ``(128, 9, 9)``
    observations :class:`~komugi.train.shards.ShardReader` would inflate. A
    16-worker iteration is thousands of shards; the reader's full validation is
    the trainer's job at consumption time, not the orchestrator's at shutdown.

    Damaged or half-written shards are skipped rather than raised on: exactly
    one such file is the expected residue of the SIGKILL this module performs
    by design.
    """
    total = 0
    for path in sorted(Path(shard_dir).glob(SHARD_GLOB)):
        try:
            loaded = np.load(path, allow_pickle=False)
            try:
                total += int(loaded[COUNT_KEY].item())
            finally:
                loaded.close()
        except _CORRUPT_SHARD_ERRORS:
            continue
    return total


def require_module_level_factory(factory: Callable[[], torch.nn.Module]) -> None:
    """Reject a model factory a spawned process could not import (design F18).

    A model never crosses a process boundary as pickled bytes (design D2): the
    server receives a *factory* and calls it inside its own interpreter. Spawn
    pickles that factory by qualified name, so a closure or a lambda fails --
    but it fails *inside* ``ShardServer.serve``, in a child, as a pickling
    error whose traceback names neither this run's model nor this rule. Checking
    here turns that into one sentence at the call site.

    The rule is enforced on the in-process path too
    (:func:`serial_debug`), and that is deliberate rather than pedantic:
    ``--serial-debug`` exists to reproduce a *production* game, so a factory
    that only works in serial debug would be a trap wearing a reproducibility
    claim.

    A callable with no ``__qualname__`` at all -- a
    :class:`functools.partial`, or an instance of a module-level class with
    ``__call__`` -- is accepted: those are picklable by their own rules, and
    the spawn's pickler is the honest arbiter for them.

    Raises:
        TypeError: If the factory is a lambda, a closure, or has no module.
    """
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", "")
    if not module or "<locals>" in qualname or qualname.endswith("<lambda>"):
        raise TypeError(
            f"model factory {qualname or factory!r} (module {module!r}) is not "
            f"module-level: a spawned server pickles the factory by qualified "
            f"name, so a lambda or a closure cannot reach it (design D2/F18). "
            f"Pass a module-level function that loads the checkpoint inside the "
            f"server process."
        )


@dataclass(eq=False)
class _Incarnation:
    """One live shard: its processes, its segments, its directories.

    ``eq=False`` because this is an identity -- it owns operating-system
    resources, and two of them are never "equal".

    ``dirs`` is indexed by worker id and holds that worker's directories: one
    per lane, so a single-game run's entry is a 1-tuple and nothing about the
    shape depends on the knob (KOM-040 b1). ``segments`` is flat and holds one
    entry per *slot* -- ``workers * games_in_flight`` of them -- because
    unlinking does not care who owned what.
    """

    index: int
    dirs: tuple[tuple[Path, ...], ...]
    segments: list[Slots]
    server: BaseProcess
    workers: list[BaseProcess]
    stop_event: synchronize.Event
    started_at: float

    @property
    def children(self) -> tuple[BaseProcess, ...]:
        """Every process this incarnation started, workers first."""
        return (*self.workers, self.server)

    @property
    def all_dirs(self) -> tuple[Path, ...]:
        """Every directory of every lane of every worker, in worker order."""
        return tuple(directory for lanes in self.dirs for directory in lanes)


class _Poll(NamedTuple):
    """One monitor pass: is the shard done, and does it need restarting?"""

    finished: bool
    restart_reason: str | None


def _newest_mtime(directory: Path) -> float | None:
    """Return the newest file mtime in `directory`, or ``None`` if it has none.

    Files only, and the directory's own mtime is deliberately ignored: the
    orchestrator creates the directory, so its mtime would report the
    orchestrator's liveness rather than the worker's -- a gate that passes
    because the thing it watches has not started yet.
    """
    newest: float | None = None
    try:
        entries = list(directory.iterdir())
    except OSError:
        return None
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            mtime = entry.stat().st_mtime
        except OSError:
            # A shard being renamed into place under us. Not evidence of
            # anything except activity, which the next pass will see.
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def _silence_reason(
    directories: Sequence[Path],
    worker_id: int,
    since_start_s: float,
    config: OrchestratorConfig,
) -> str | None:
    """Return why worker `worker_id` looks dead, or ``None`` if it looks alive.

    Design D10's liveness gate: no new journal record type, just the mtime of
    what the worker already writes, judged by
    :func:`~komugi.telemetry.liveness.log_freshness` so the decision boundary
    stays the one the telemetry suite pins.

    A worker with **no file yet** is exempt, but not indefinitely: rows are
    appended only when a game ends, so nothing exists until the first game is
    sealed. That grace is its OWN named knob, ``first_shard_grace_s`` -- not a
    silent reuse of ``drain_timeout_s``: an operator tightening
    ``liveness_timeout_s`` must not inherit a hidden 900s gate on the
    never-started case (wave-9c review F3). A worker that has produced nothing
    after the grace is not slow, it is stuck.

    **What this gate does NOT see with two lanes** (KOM-040 b1, design D1'/C5),
    stated here rather than discovered later: `directories` is the worker's
    lane directories and the judgement is on the NEWEST file across all of
    them, because either lane writing means the worker process is alive and
    producing. The cost is exact and accepted: one lane wedged while its
    sibling keeps sealing games is **not** detected -- the mtime stays fresh.
    Per-lane liveness is out of scope; what still bounds the damage is that a
    lane's wedge is almost always the seam's (a poisoned client fails the whole
    process, C6) and that a whole-worker stall still restarts the whole shard.

    Args:
        directories: The worker's shard directories -- one per lane, so a
            single-game worker's is a 1-tuple and the judgement is identical to
            what it was before lanes existed.
        worker_id: The worker being judged, for the diagnostic.
        since_start_s: How long this incarnation has been running.
        config: The patience knobs.
    """
    mtimes = [
        mtime
        for mtime in (_newest_mtime(directory) for directory in directories)
        if mtime is not None
    ]
    if not mtimes:
        shown = ", ".join(str(directory) for directory in directories)
        if since_start_s > config.first_shard_grace_s:
            return (
                f"worker {worker_id} has written nothing to {shown} in "
                f"{since_start_s:.1f}s, past first_shard_grace_s="
                f"{config.first_shard_grace_s:.1f}s; it never got started"
            )
        return None
    verdict = log_freshness(time.time() - max(mtimes), config.liveness_timeout_s)
    if verdict.ok:
        return None
    shown = ", ".join(str(directory) for directory in directories)
    return f"worker {worker_id} in {shown}: {verdict.reason}"


def _poll_shard(incarnation: _Incarnation, config: OrchestratorConfig) -> _Poll:
    """Run one monitor pass over a live shard.

    The three questions of design D2/D10, in the order their answers matter:

    (a) Is the server gone? Then the run **fails loudly** -- with the exit code,
        because "the server died" and "the server was OOM-killed" are different
        operator actions. P0a does not respawn a shard server (design D2/Q4).
    (b) Did any worker exit nonzero, or go silent past the liveness bound? Then
        the **whole shard** restarts. Not the one worker: the server's client
        list is fixed at ``serve`` start, so a respawned worker's fresh segment
        would be rung into a slot nobody scans (doorbell review F4/F5).
    (c) Have all workers exited 0? Then the shard is finished and the caller
        drains the server.

    Raises:
        RuntimeError: If the server process is gone.
    """
    server_exit = incarnation.server.exitcode
    if server_exit is not None:
        raise RuntimeError(
            f"inference server for incarnation {incarnation.index} exited with "
            f"code {server_exit} while workers were still running; a self-play "
            f"run does not survive its server (design D2/Q4 -- per-shard "
            f"respawn with generation counters is the pre-registered P1 ticket)"
        )

    codes = [worker.exitcode for worker in incarnation.workers]
    failed = [
        f"worker {worker_id} exited {code}"
        for worker_id, code in enumerate(codes)
        if code is not None and code != 0
    ]
    if failed:
        return _Poll(finished=False, restart_reason="; ".join(failed))
    if all(code == 0 for code in codes):
        return _Poll(finished=True, restart_reason=None)

    since_start = time.monotonic() - incarnation.started_at
    for worker_id, code in enumerate(codes):
        if code is not None:
            # Exited cleanly; it is not writing any more and must not be
            # judged for it while its siblings finish.
            continue
        reason = _silence_reason(
            incarnation.dirs[worker_id], worker_id, since_start, config
        )
        # `dirs[worker_id]` is every lane's directory: with one game in flight
        # that is the single directory this gate always judged.
        if reason is not None:
            return _Poll(finished=False, restart_reason=reason)
    return _Poll(finished=False, restart_reason=None)


def validate_wave_fit(selfplay: SelfPlayConfig, inference: InferenceConfig) -> None:
    """Check the wave width against both configs, at the one point they meet.

    :class:`~komugi.selfplay.inference.InferenceConfig` already refuses
    ``max_wave < 1`` and ``max_wave > max_batch`` on its own. What it cannot see
    is the *search's* ``m``: a wave carries one root's sampled actions, so a
    ``max_wave`` below ``gumbel_m_full`` would force the search to split a wave
    into several tickets and give back exactly the round trips KOM-040 b2 exists
    to delete. The orchestrator is the only place both configs are in scope, so
    the check lives here (W10).

    Enforced **only when waves are enabled** (``max_wave > 1``), because P0a
    ships ``max_wave == 1``: at that width nothing submits a wave at all, and
    demanding ``max_wave >= 32`` of a run that does not use them would refuse
    every default config in the repository.

    Raises:
        ValueError: Naming both fields, so the reader knows which to move.
    """
    m = max(selfplay.gumbel_m_full, selfplay.gumbel_m_fast)
    if inference.max_wave > 1 and inference.max_wave < m:
        raise ValueError(
            f"InferenceConfig.max_wave={inference.max_wave} is below the "
            f"search's m={m} (SelfPlayConfig.gumbel_m_full="
            f"{selfplay.gumbel_m_full}, gumbel_m_fast={selfplay.gumbel_m_fast}): "
            f"a wave that cannot carry one root's actions splits back into the "
            f"round trips waves exist to delete. Raise max_wave (and max_batch "
            f"with it) or leave max_wave at 1"
        )


def _launch(
    model_factory: Callable[[], torch.nn.Module],
    index: int,
    config: OrchestratorConfig,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
    run_token: str,
) -> _Incarnation:
    """Create incarnation `index`: fresh directories, fresh segments, processes.

    Order matters and is the ownership rule (design D2/F9): segments are created
    **here**, by the orchestrator, and only their names travel. The server is
    started before the workers so that a worker's very first doorbell has
    somebody scanning for it -- the client's patience is
    ``request_timeout_s`` and timeout is fatal, so this is not a race worth
    running.

    Anything that fails midway tears down what it already built, because a
    half-launched incarnation still holds segments and this function's caller
    has nothing to clean up on its behalf yet.

    With ``selfplay.games_in_flight == 2`` (KOM-040 b1) every count in the
    paragraph above doubles and NOTHING else changes: ``workers *
    games_in_flight`` **fresh** segments are created here, by this process,
    registered and unlinked by the same ``finally``; the server's client list
    is one :class:`~komugi.selfplay.inference.SlotHandle` per segment, still
    fixed at ``serve`` start; and each worker is handed its own lanes' names.
    A lane is a slot, and a slot is owned exactly like the one slot a
    single-game worker gets.

    `run_token` names the segments (KOM-054, :func:`slot_shm_name`) and changes
    nothing else: the orchestrator still hands out names and only names, and
    the children still attach to whatever they are given. What it buys is
    outside the protocol -- a segment in machine-global ``/dev/shm`` says which
    run created it, so an auditor diffing that directory can tell this run's
    leak from a concurrent orchestrator's live segment.
    """
    validate_wave_fit(selfplay, inference)
    ctx = get_context("spawn")
    lanes = selfplay.games_in_flight
    dirs = tuple(
        tuple(
            Path(
                worker_shard_dir(
                    config.shard_root,
                    worker_id,
                    index,
                    lane=None if lanes == 1 else lane,
                )
            )
            for lane in range(lanes)
        )
        for worker_id in range(config.workers)
    )
    segments: list[Slots] = []
    workers: list[BaseProcess] = []
    started: list[BaseProcess] = []
    stop_event = ctx.Event()
    try:
        for worker_dirs in dirs:
            for directory in worker_dirs:
                directory.mkdir(parents=True, exist_ok=True)
        # Appended one at a time rather than built as a comprehension: a
        # failure halfway through must leave the segments it already created
        # reachable by the cleanup below, not stranded in a discarded list.
        for slot_id in range(config.workers * lanes):
            segments.append(
                create_slots(inference, slot_shm_name(run_token, index, slot_id))
            )
        # Slot order is (worker 0 lane 0, worker 0 lane 1, worker 1 lane 0,
        # ...), which is exactly `lane_slot_id`'s numbering -- the server's
        # slot map and this list are the same order by construction.
        handles = [
            SlotHandle(shm_name=slots.shm.name, worker_id=slot_id)
            for slot_id, slots in enumerate(segments)
        ]
        server = ctx.Process(
            target=serve_shard,
            args=(model_factory, inference, handles, stop_event),
            name=f"komugi-server-r{index}",
        )
        server.start()
        started.append(server)
        for worker_id in range(config.workers):
            extra_lanes = tuple(
                LaneSpec(
                    lane=lane,
                    slot_id=lane_slot_id(worker_id, lanes, lane),
                    shm_name=handles[lane_slot_id(worker_id, lanes, lane)].shm_name,
                    shard_dir=str(dirs[worker_id][lane]),
                    games=lane_game_budget(config.games_per_worker, lanes, lane),
                )
                for lane in range(1, lanes)
            )
            process: BaseProcess = ctx.Process(
                target=worker_main,
                args=(
                    worker_id,
                    index,
                    config.run_seed,
                    lane_game_budget(config.games_per_worker, lanes, 0),
                    str(dirs[worker_id][0]),
                    handles[lane_slot_id(worker_id, lanes, 0)].shm_name,
                    selfplay,
                    inference,
                    extra_lanes,
                ),
                name=f"komugi-worker-{worker_id}-r{index}",
            )
            process.start()
            workers.append(process)
            started.append(process)
        return _Incarnation(
            index=index,
            dirs=dirs,
            segments=segments,
            server=server,
            workers=workers,
            stop_event=stop_event,
            started_at=time.monotonic(),
        )
    except BaseException:
        # This IS the teardown for an incarnation that never reached
        # ``started``: there is no :class:`_Incarnation` for the caller to hand
        # :func:`_teardown`, so the same obligation has to be discharged here,
        # and to the same standard (KOM-053). Two halves, and the segments are
        # the half that must not depend on the other: stopping the children can
        # itself raise -- a ``kill()`` on a process the OS has already reaped
        # races into ``ProcessLookupError``, a wedged ``join()`` past
        # :data:`_KILL_JOIN_S` -- and a segment left in /dev/shm outlives this
        # process while a surviving child does not. Hence the ``finally``, and
        # hence the guarded sweep rather than a bare loop.
        try:
            stop_event.set()
            for process in started:
                if process.is_alive():
                    process.kill()
                process.join(_KILL_JOIN_S)
        finally:
            _release_all(segments)
        raise


def _release(slots: Slots) -> None:
    """Close one segment's views and unlink it, swallowing nothing important."""
    slots.close()
    reregister_and_unlink(slots.shm)


def _release_all(segments: Sequence[Slots]) -> None:
    """Release every segment in `segments`, then re-raise the first failure.

    Per-item guard (wave-9c review F8): one segment's release raising -- an
    un-dropped view's ``BufferError``, an unlink ``PermissionError`` -- must not
    leak every segment after it. The sweep finishes first; the first error is
    re-raised afterwards, so nothing is swallowed and nothing is stranded.

    One function rather than two loops (KOM-053), because there are *two*
    places that own segments and both owe the same guarantee: :func:`_teardown`
    for an incarnation that reached ``started``, and :func:`_launch`'s failure
    path for one that never did. The second used to sweep unguarded, so a
    segment whose release raised leaked every segment after it -- with the
    caller none the wiser, since the exception it raised looked like the
    original launch failure.

    Raises:
        BaseException: The first error the sweep hit, after the sweep is done.
    """
    first_error: BaseException | None = None
    for slots in segments:
        try:
            _release(slots)
        except BaseException as exc:  # noqa: BLE001 -- sweep must finish
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _teardown(
    incarnation: _Incarnation, config: OrchestratorConfig, *, drain: bool
) -> None:
    """Stop every child of `incarnation`, confirm it, and unlink its segments.

    ``drain=True`` is the graceful path of design D10: the workers have already
    exited on their own (they flush after every game), so all that is left is to
    let the server finish whatever it is forwarding and exit within
    ``drain_timeout_s``.

    Then, either way, **SIGKILL and join**. The join is the load-bearing half
    (design F6): a shard restart may not begin until every predecessor is
    reaped, or the zombie's :class:`~komugi.train.shards.ShardWriter` and its
    successor's would race for the same shard index. A child still alive after
    the wall raises rather than being ignored -- respawning past it is the
    collision this rule exists to prevent.

    Segments are released in a ``finally``, because a leaked ``/dev/shm``
    segment outlives the process that leaked it (design D2/F9).

    Raises:
        RuntimeError: If a child survives SIGKILL plus :data:`_KILL_JOIN_S`.
    """
    try:
        incarnation.stop_event.set()
        if drain:
            incarnation.server.join(config.drain_timeout_s)
            if incarnation.server.is_alive():
                warnings.warn(
                    f"inference server did not exit within "
                    f"{config.drain_timeout_s}s of the stop signal; killing it "
                    f"(design D10: hard-kill after the drain wall, and record it)",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for process in incarnation.children:
            if process.is_alive():
                process.kill()
            process.join(_KILL_JOIN_S)
        stuck = [
            process.name for process in incarnation.children if process.is_alive()
        ]
        if stuck:
            raise RuntimeError(
                f"children of incarnation {incarnation.index} survived SIGKILL "
                f"and {_KILL_JOIN_S}s of join(): {stuck}. Refusing to continue: "
                f"an unreaped worker still owns its shard directory and would "
                f"collide with its successor's writer (design D10/F6)"
            )
    finally:
        _release_all(incarnation.segments)


def run_selfplay(
    model_factory: Callable[[], torch.nn.Module],
    orchestrator: OrchestratorConfig,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> RunSummary:
    """Run one self-play shard to completion and report what it produced.

    The loop of design D2/D10, in full: create fresh directories and fresh
    segments, spawn one :func:`~komugi.selfplay.inference.serve_shard` process
    and ``workers`` :func:`~komugi.selfplay.worker.worker_main` processes,
    monitor them every :data:`_POLL_S` seconds, and restart the **whole shard**
    on any worker death -- the module docstring lists the obligations and
    :func:`_poll_shard` enforces them.

    ``selfplay.games_in_flight`` decides how many games each worker plays at
    once (KOM-040 b1), and the only thing that changes up here is arithmetic:
    the shard has ``workers * games_in_flight`` slots (one segment, one
    :class:`~komugi.selfplay.inference.SlotHandle` and one shard directory per
    lane), and each worker's ``games_per_worker`` budget is SPLIT across its
    lanes rather than multiplied (:func:`lane_game_budget`). A run's nominal
    size, its restart policy and this summary's accounting are therefore
    unchanged; ``games_completed`` is still ``games_per_worker`` per
    cleanly-exited worker, which is now the sum over that worker's lanes, and
    ``rows_written`` sums the lane directories exactly as it summed the worker
    directories.

    Everything is ``spawn``: the production process model is spawned workers
    (design D2), fork would share torch state a spawned child does not, and the
    shape-check seam arms from the environment at import time in the child
    (design D7).

    Every segment this call creates -- across every incarnation, including the
    ones a restart throws away -- is named from one token minted here
    (:func:`mint_run_token`, KOM-054), so machine-global ``/dev/shm`` can be
    audited per run instead of per box. Nothing about the protocol changes:
    names are still opaque to whoever receives them.

    Args:
        model_factory: Module-level callable, invoked **inside** the server
            process to build the network. Never a model, never a closure --
            see :func:`require_module_level_factory`.
        orchestrator: Fan-out, budgets and patience.
        selfplay: How a game is played; passed through to every worker.
        inference: The seam's shapes and timings; identical in server and
            workers by construction, since this is the only place it is passed.

    Returns:
        A :class:`RunSummary` counted from the shards and the exit codes, not
        from the request. With restarts it exceeds ``workers *
        games_per_worker``; that is the documented P0a semantics.

    Raises:
        TypeError: If `model_factory` is not module-level (design F18).
        NotImplementedError: If ``inference.shards > 1``.
        RuntimeError: If the server dies, if a restart budget is exhausted, or
            if a child survives SIGKILL.
    """
    require_module_level_factory(model_factory)
    if inference.shards > 1:
        raise NotImplementedError(
            f"shards={inference.shards}: P0a runs a single server shard. Serving "
            f"more than one shard needs the dynamic worker registration with "
            f"generation counters that design D2/Q4 pre-registers as the P1 "
            f"ticket -- without it a restarted worker cannot reach any running "
            f"server, and a sticky worker->shard map would still be required to "
            f"keep response_seq single-writer (design D3)"
        )

    shard_dirs: list[str] = []
    diagnostics: list[str] = []
    games_completed = 0
    restarts = 0
    index = 0
    # Minted ONCE, here, and threaded through every incarnation this run
    # launches (KOM-054): a restart must not change the answer to "which run
    # created this segment", or the run-scoped leak check would go blind
    # exactly on the restart paths it most needs to watch.
    run_token = mint_run_token()

    while True:
        incarnation = _launch(
            model_factory, index, orchestrator, selfplay, inference, run_token
        )
        shard_dirs.extend(str(directory) for directory in incarnation.all_dirs)
        reason: str | None = None
        try:
            while True:
                poll = _poll_shard(incarnation, orchestrator)
                if poll.finished:
                    break
                if poll.restart_reason is not None:
                    reason = poll.restart_reason
                    break
                time.sleep(_POLL_S)
        finally:
            # A worker that exited 0 played its whole budget, whether or not a
            # sibling then failed the shard: its rows are on disk and its games
            # happened. Read after the joins so the codes are final.
            _teardown(incarnation, orchestrator, drain=reason is None)
            games_completed += orchestrator.games_per_worker * sum(
                1 for worker in incarnation.workers if worker.exitcode == 0
            )

        if reason is None:
            break

        diagnostics.append(f"incarnation {index}: {reason}")
        if restarts >= orchestrator.max_shard_restarts:
            raise RuntimeError(
                f"self-play shard exceeded max_shard_restarts="
                f"{orchestrator.max_shard_restarts}. Restart history:\n  "
                + "\n  ".join(diagnostics)
            )
        restarts += 1
        index += 1

    # Adjudicated (wave-9c review F2): the summary names only directories that
    # hold at least one sealed shard. A wedged incarnation that produced
    # nothing is bookkeeping, not data -- and every consumer recipe this
    # summary feeds (reader-per-dir into ConcatSource) breaks on an empty
    # directory. A restarted-but-productive incarnation's dirs stay: its
    # sealed games are real training data.
    productive = [d for d in shard_dirs if count_rows(d) > 0]
    return RunSummary(
        games_completed=games_completed,
        rows_written=sum(count_rows(directory) for directory in productive),
        shard_dirs=tuple(productive),
        restarts=restarts,
        worker_incarnations=tuple(index for _ in range(orchestrator.workers)),
    )


def serial_debug(
    model_factory: Callable[[], torch.nn.Module],
    run_seed: int,
    worker_id: int,
    incarnation: int,
    game_index: int,
    shard_dir: str,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> int:
    """Replay exactly one game in this process, and return the rows it wrote.

    Design D11's bit-reproducible path, and the only claim about replay this
    project makes: one worker, one
    :class:`~komugi.selfplay.inference.InProcessEvaluator`, no shared memory and
    no processes, so nothing about the result depends on batch composition.
    Two calls with the same ``(run_seed, worker_id, incarnation, game_index)``
    and the same factory write **byte-identical** rows. Production games batch
    across workers and are promised only up to evaluator tolerance -- the
    KOM-104 GEMM lesson -- and this function exists so that there is somewhere
    the stronger claim is true.

    The seed derivation is not re-implemented here: this constructs the same
    :class:`~komugi.selfplay.worker.SelfPlayWorker` with the same
    :class:`~komugi.selfplay.worker.WorkerConfig` a spawned worker would, and
    calls :meth:`~komugi.selfplay.worker.SelfPlayWorker.play_game`. A
    re-derivation would be a second copy of design D11 that could drift from
    the first, which is the one way this function could lie.

    ``selfplay.games_in_flight`` is **ignored here** and always will be
    (KOM-040 b1, design D1'/C7): this path is one game, in this thread, with no
    shared memory and no lanes, because that is what makes the bit-exact claim
    checkable. A ``games_in_flight=2`` config replays exactly the game a
    ``games_in_flight=1`` config would -- the lane offset of
    :func:`~komugi.selfplay.worker.lane_game_index_base` lives in the worker's
    lane construction, not in the seed derivation, so ``game_index`` here means
    what it has always meant.

    Args:
        model_factory: Module-level callable returning the network. The rule is
            enforced here too, on purpose -- see
            :func:`require_module_level_factory`.
        run_seed: The run's entropy.
        worker_id: The worker id whose stream to replay.
        incarnation: The incarnation whose stream to replay. Restarts change
            it, and it is in the spawn key (design F10c), so replaying the
            wrong one silently replays a different game.
        game_index: Which game of that worker's workload to play.
        shard_dir: Where the rows go. Its own directory, like every writer's.
        selfplay: Search budgets and the full/fast mix.
        inference: Seam shapes; used to validate what the model is handed.
            ``forward_graph`` is **forced off** on the copy handed down
            (KOM-048): this path never captures a CUDA graph, whatever the
            production config it is replaying says.

    Returns:
        The number of rows written -- one per ply of the game.

    Raises:
        TypeError: If `model_factory` is not module-level (design F18).
    """
    require_module_level_factory(model_factory)
    # KOM-048: forced off, on the config this path passes down, so that a
    # production config with forward_graph="on" still replays here. Two
    # reasons, and the second is the one that makes this a pinned property
    # rather than a tidy-up: (a) --serial-debug runs wherever the operator is,
    # which is usually a CPU box, and a graphs-on evaluator refuses a
    # host-resident model by design; (b) this path's whole claim is
    # bit-identical replay, and a captured forward is exactly the kind of thing
    # that could make the claim depend on the hardware it was captured on.
    # InProcessEvaluator never captures anyway -- this says so out loud, on the
    # config, where a reader can check it.
    evaluator = InProcessEvaluator(
        model_factory(), replace(inference, forward_graph="off")
    )
    writer = ShardWriter(shard_dir, SCHEMA)
    try:
        worker = SelfPlayWorker(
            evaluator,
            selfplay,
            WorkerConfig(
                worker_id=worker_id,
                incarnation=incarnation,
                run_seed=run_seed,
                # The workload this game belongs to. `run()` is never called
                # here -- only game `game_index` is played -- but the field is
                # part of the worker's identity and a wrong value would be a
                # confusing thing to read back out of a debug session.
                games=game_index + 1,
                shard_dir=shard_dir,
            ),
            writer,
        )
        rows = worker.play_game(game_index)
        # The writer seals on triggers checked inside `append`, so the tail of
        # a single game stays buffered without this: a debug replay that wrote
        # nothing to disk would be a poor reproduction of a worker.
        writer.flush()
    finally:
        writer.close()
    return rows
