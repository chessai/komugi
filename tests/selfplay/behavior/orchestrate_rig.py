"""Shared rig for the KOM-123 orchestrator behavioural suite.

The orchestrator (KOM-023: ``selfplay/orchestrate.py``) is being implemented
concurrently by a different worker, so everything here is written against the
pinned contract and nothing else::

    OrchestratorConfig(run_seed, workers, games_per_worker, shard_root,
                       liveness_timeout_s=120.0, drain_timeout_s=900.0,
                       max_shard_restarts=2)
    RunSummary(games_completed, rows_written, shard_dirs, restarts,
               worker_incarnations)
    run_selfplay(model_factory, orchestrator, selfplay, inference) -> RunSummary
    serial_debug(model_factory, run_seed, worker_id, incarnation, game_index,
                 shard_dir, selfplay, inference) -> int
    GridPoint(workers, n_full, peak_bytes)
    cgroup_memory_peak() -> int | None
    measure_grid(...)

Shard directories are ``shard_root/worker_<k>_r<incarnation>``; any worker
death or wedge restarts the WHOLE shard (fresh segments, incarnation + 1, fresh
directories); server death fails the run loudly; the model factory must be
module-level.

Rules this rig obeys, and why each is a rule
--------------------------------------------

* **Never import :mod:`komugi.net`.**  The orchestrator's contract is "a
  module-level callable returning something KomugiNet-*shaped*"
  (``policy_logits`` / ``soft_policy_logits`` / ``wdl_logits`` /
  ``moves_left``), so :class:`TinyNet` below is that shape and nothing more.
  A real net would couple this suite to KOM-005a and pay seconds per forward
  for nothing.
* **Every stub model is module-level.**  Design D2 forbids a pickled model
  across the process boundary: the factory is imported and *called* inside the
  server process.  A closure or a lambda cannot make that trip, which is
  exactly what ``test_failures_are_loud`` asserts the orchestrator says out
  loud.
* **Real engine shapes.**  ``A`` is the engine's 37341 and the observation is
  its 128 planes, both read off the engine rather than restated, because the
  end-to-end loop plays the real ``Game``.
* **Tiny budgets everywhere.**  ``gumbel_n_full=4`` / ``gumbel_n_fast=2`` /
  ``prob_fast=0.5`` per the KOM-123 spec.  ``prob_fast`` is one half rather
  than the production 0.75 so that a single run of a few hundred moves is
  certain to record BOTH ``search_mode`` values.
* **Every wall is explicit.**  ``pytest-timeout`` is not in the dev shell (see
  ``flake.nix``), and this suite's whole subject matter is processes that hang.
  So no test calls ``run_selfplay`` directly: :func:`walled_run` runs it on a
  daemon helper thread and turns "still running at the wall" into a *failure in
  bounded time* rather than a wedged ``ci/check.sh``.  Killing a thread is not
  possible in Python, so the wall additionally kills every child process the
  orchestrator left behind, which is what lets the stuck thread unwind.
* **Nothing here carries a skip gate.**  The five test modules next to this one
  each gate on ``pytest.importorskip`` plus ``hasattr(sp, "run_selfplay")``;
  this module must stay importable from a *spawned child*, where a
  module-level ``pytest.skip`` would raise ``Skipped`` in an interpreter with
  no session to catch it and turn the child's exit code into a lie.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, NamedTuple

from multiprocessing import resource_tracker

import torch
from torch import nn

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.observation import observation_size
from komugi.train.shards import ShardReader

# ---------------------------------------------------------------------------
# shapes, budgets and constants
# ---------------------------------------------------------------------------

IN_PLANES, BOARD_H, BOARD_W = observation_size()
"""The real encoder's observation shape -- (128, 9, 9) today."""

NUM_ACTIONS: int = ACTION_SIZE
"""The real engine's flat action space, ``A``."""

DRAW_SCORE: float = 0.20
"""DESIGN.html section 3.3's contempt, and ``SelfPlayConfig``'s default."""

SHARD_DIR_RE: Final[str] = r"worker_(\d+)_r(\d+)"
"""Design D10's per-worker-per-incarnation directory name, as a pattern.

Written once here because three of the five test modules assert on it: the
incarnation suffix is what keeps a respawned worker from colliding with a
zombie predecessor on ``ShardWriter``'s single-writer guard.
"""

SHARD_GLOB: Final[str] = "shard-*.npz"
"""What a *sealed* shard is called on disk (``komugi.train.shards``)."""

COUNTER_ENV: Final[str] = "KOMUGI_TEST_WEDGE_COUNTER"
"""Environment variable naming the wedge test's construction counter file.

The test sets it before calling ``run_selfplay``; spawned children inherit the
parent's environment, so the *server* process reads the same file the test
wrote the path of.  This is the whole mechanism that makes
``test_wedge_restart`` deterministic rather than timing-dependent: incarnation
0's model hangs because it read a 0, incarnation 1's behaves because it read a
1, and neither decision involves a clock.
"""

_GAME_ID_STRIDE: Final[int] = 1_000_000
"""``worker.game_id``'s packing radix, mirrored so a row's provenance can be
decoded back out here without importing a private detail."""

# ---------------------------------------------------------------------------
# walls.  Generous on purpose: they bound *hangs*, they are not latency
# assertions.  A spawned child imports torch from scratch (seconds on a cold
# page cache) and then plays real games.
# ---------------------------------------------------------------------------

E2E_WALL_S: Final[float] = 900.0
"""Wall for the 2-worker x 2-game end-to-end loop."""

RESTART_WALL_S: Final[float] = 600.0
"""Wall for a run that must detect one wedge and restart the shard once."""

LOUD_WALL_S: Final[float] = 300.0
"""Wall for a run that is *supposed* to raise."""

KILL_GRACE_S: Final[float] = 10.0
"""How long a straggler child gets to die after ``kill()``."""


def selfplay() -> ModuleType:
    """The module under test, imported in whichever interpreter asks.

    ``importlib`` rather than a top-level import because the orchestrator half
    of this package does not exist yet (this suite is red-first) and because
    spawned children re-import it in their own interpreter anyway.
    """
    return importlib.import_module("komugi.selfplay")


def has_orchestrator() -> bool:
    """Whether KOM-023 has landed, for a child process that must not skip."""
    try:
        return hasattr(selfplay(), "run_selfplay")
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# stub models -- module-level classes and module-level factories, always
# ---------------------------------------------------------------------------


class StubOutput(NamedTuple):
    """The three fields the **inference seam** reads, owned by this suite.

    Deliberately *not* a mirror of ``komugi.net.NetOutput``, which carries
    eight fields since the staged heads landed: the serving path reaches this
    object through ``inference._NetOutputLike``, a structural ``Protocol`` over
    ``policy_logits``, ``wdl_logits`` and ``moves_left``, which a four-field
    stub satisfies unchanged.
    """

    policy_logits: torch.Tensor
    soft_policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class TinyNet(nn.Module):
    """A net at the engine's real shapes that costs almost nothing to run.

    One parameter, so ``next(model.parameters())`` gives the server a device to
    follow (:class:`~komugi.selfplay.InProcessEvaluator` places tensors on the
    model's own device and names none itself).

    **Elementwise only, and deterministic in ``x`` alone.**  Both properties
    are load-bearing for ``test_serial_vs_production``: a row's output must not
    depend on how many other rows shared its batch, so there is no cross-row
    reduction anywhere (the per-row ``mean`` reduces within one row), and there
    is no counter, clock or global RNG to make two runs of one game differ.
    The policy is a ramp plus a pooled function of the position -- non-constant,
    so a stale or transposed buffer would show, and one broadcast rather than a
    37341-wide matmul.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x: torch.Tensor) -> StubOutput:
        """Return a KomugiNet-shaped output for a ``(N, 128, 9, 9)`` batch."""
        rows = x.shape[0]
        pooled = x.reshape(rows, -1).mean(dim=1, keepdim=True)
        ramp = torch.arange(NUM_ACTIONS, dtype=x.dtype, device=x.device) * 1e-4
        policy = ramp.unsqueeze(0) + pooled * self.scale
        wdl = torch.cat([pooled * 0.1, pooled * 0.0, pooled * -0.1], dim=1)
        return StubOutput(policy, policy.clone(), wdl, pooled.reshape(-1))


class HangingNet(nn.Module):
    """A net whose forward never returns.

    The wedge these tests are about, reduced to its smallest form: the server
    process is alive and healthy, its workers' doorbells are rung, and no
    response will ever be written.  Design D10's liveness gate is the only
    thing that can notice, which is precisely what makes this a test of the
    gate rather than of the inference timeout.

    ``time.sleep`` in a loop rather than a spin, so a wedged run costs a core
    nothing while the gate counts down, and so SIGKILL lands promptly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x: torch.Tensor) -> StubOutput:
        """Never return.  A response for `x` is never written."""
        del x
        while True:
            time.sleep(0.05)


def tiny_factory() -> TinyNet:
    """The well-behaved model factory (module-level == importable == spawnable)."""
    model = TinyNet()
    model.eval()
    return model


def exploding_factory() -> TinyNet:
    """A factory that dies where the *server* builds its model (design Q4).

    Raised at construction, inside the server process, which is the one failure
    design D2 says must fail the whole run loudly rather than be restarted:
    a running server cannot re-attach fresh segments in P0a, so there is no
    recovery to attempt.
    """
    raise RuntimeError(
        "stub model factory refuses to build (KOM-123: simulated server death)"
    )


def always_hanging_factory() -> HangingNet:
    """A factory whose every incarnation wedges -- restarts can never converge."""
    model = HangingNet()
    model.eval()
    return model


def bump_counter() -> int:
    """Read the wedge counter, increment it, and return the value *read*.

    A file rather than shared state because the reader is a *spawned server
    process*: the path arrives through :data:`COUNTER_ENV` (children inherit the
    environment), and the file is the only thing that survives the process
    boundary.  Locked with ``O_EXCL``-free ``flock`` so that a future
    multi-shard run cannot interleave two increments, even though P0a builds one
    server per incarnation and therefore never races.

    Returns:
        The counter value *before* this call: 0 for the first construction in a
        run, 1 for the second, and so on.

    Raises:
        RuntimeError: If the environment variable is unset, which would mean the
            factory is running somewhere the test did not intend.
    """
    import fcntl

    path = os.environ.get(COUNTER_ENV)
    if not path:
        raise RuntimeError(
            f"{COUNTER_ENV} is unset: the wedge factory only makes sense when a "
            f"test has pointed it at a counter file"
        )
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        raw = os.read(handle, 64).decode("ascii").strip()
        value = int(raw) if raw else 0
        payload = f"{value + 1}".encode("ascii")
        os.lseek(handle, 0, os.SEEK_SET)
        os.write(handle, payload)
        os.ftruncate(handle, len(payload))
    finally:
        os.close(handle)  # closing releases the flock
    return value


def read_counter(path: Path) -> int:
    """Return how many models the wedge factory has built (0 if it never ran)."""
    try:
        raw = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return 0
    return int(raw) if raw else 0


def wedge_once_factory() -> nn.Module:
    """Hang on incarnation 0, behave from incarnation 1 on.

    The determinism the KOM-123 spec asks for: which model gets built is decided
    by :func:`bump_counter`, not by a race.  Incarnation 0's server hangs every
    evaluation, so its workers starve and go silent; the liveness gate declares
    the wedge and restarts the whole shard; the restarted server reads a 1 and
    plays the run out.
    """
    if bump_counter() == 0:
        return always_hanging_factory()
    return tiny_factory()


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def tiny_selfplay_config(workers: int = 1, **overrides: Any) -> SelfPlayConfig:
    """``SelfPlayConfig`` at the KOM-123 budgets.

    ``workers`` lives here as well as on ``OrchestratorConfig`` (it is a field
    of the frozen config, not of the orchestrator's own record), so callers pass
    the same number to both and an orchestrator that reads either one agrees.
    """
    fields: dict[str, Any] = {
        "workers": workers,
        "process_batch_size": 1,
        "gumbel_m_full": 2,
        "gumbel_n_full": 4,
        "gumbel_m_fast": 2,
        "gumbel_n_fast": 2,
        "prob_fast": 0.5,
        "draw_score": DRAW_SCORE,
    }
    fields.update(overrides)
    return SelfPlayConfig(**fields)


def inference_config(**overrides: Any) -> Any:
    """``InferenceConfig`` at the engine's real shapes.

    ``request_timeout_s`` defaults *high* on purpose.  The wedge tests want the
    orchestrator's liveness gate to be what notices a hung server; a worker-side
    :class:`~komugi.selfplay.InferenceTimeout` firing first would also end in a
    whole-shard restart (design D2) but would be a different mechanism, and a
    test that cannot tell them apart pins neither.
    """
    fields: dict[str, Any] = {
        "in_planes": IN_PLANES,
        "num_actions": NUM_ACTIONS,
        "max_batch": 4,
        "batch_timeout_ms": 1.0,
        "request_timeout_s": 240.0,
    }
    fields.update(overrides)
    return selfplay().InferenceConfig(**fields)


def orchestrator_config(shard_root: Path, **overrides: Any) -> Any:
    """``OrchestratorConfig`` over `shard_root`, at this suite's tiny sizes.

    ``shard_root`` crosses as a ``str``: it is handed to spawned workers, and
    ``WorkerConfig.shard_dir`` is spelled ``str`` in the pinned worker contract.
    """
    fields: dict[str, Any] = {
        "run_seed": 20260823,
        "workers": 1,
        "games_per_worker": 1,
        "shard_root": str(shard_root),
    }
    fields.update(overrides)
    return selfplay().OrchestratorConfig(**fields)


# ---------------------------------------------------------------------------
# walls: running the orchestrator so that a hang is a bounded failure
# ---------------------------------------------------------------------------


class WallExpired(AssertionError):
    """``run_selfplay`` was still running when its wall ran out."""


@dataclass(frozen=True)
class WalledRun:
    """What one walled ``run_selfplay`` call produced.

    Exactly one of `summary` and `error` is set: the orchestrator either
    returned a ``RunSummary`` or raised.  Both are captured rather than
    re-raised, because half of this suite is *about* the exception -- its type,
    its message, and the state of ``/dev/shm`` after it.
    """

    summary: Any | None
    error: BaseException | None
    elapsed: float

    def unwrap(self) -> Any:
        """Return the summary, failing the test with the traceback if it raised."""
        if self.error is not None:
            raise AssertionError(
                f"run_selfplay raised {type(self.error).__name__}: {self.error}"
            ) from self.error
        assert self.summary is not None, "run_selfplay returned None, want a RunSummary"
        return self.summary

    def raised(self) -> BaseException:
        """Return the exception, failing the test if the call actually succeeded."""
        if self.error is None:
            raise AssertionError(
                f"run_selfplay returned {self.summary!r} instead of raising"
            )
        return self.error


def walled(call: Callable[[], Any], wall_s: float, label: str) -> WalledRun:
    """Run `call` on a daemon thread and hold it to `wall_s`.

    A thread rather than a subprocess because the orchestrator must run *in this
    interpreter* for the child-process bookkeeping :func:`surviving_children` to
    see anything, and because ``tmp_path``/fixtures are here.  A thread cannot be
    killed, so the wall does the next best thing and kills the orchestrator's
    children: that unblocks whatever the stuck thread was waiting on, and the
    thread is a daemon so a truly unkillable one cannot stop the interpreter
    exiting.  Either way the *test* fails now instead of the gate hanging.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        started = time.monotonic()
        try:
            box["summary"] = call()
        except BaseException as exc:  # noqa: BLE001 -- reported, not swallowed
            box["error"] = exc
        finally:
            box["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=target, name=f"walled-{label}", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(wall_s)
    if thread.is_alive():
        killed = kill_stragglers()
        raise WallExpired(
            f"{label}: run_selfplay was still running after {wall_s}s. "
            f"Killed {len(killed)} orphaned child process(es): {killed}"
        )
    return WalledRun(
        summary=box.get("summary"),
        error=box.get("error"),
        elapsed=float(box.get("elapsed", time.monotonic() - started)),
    )


def walled_run_selfplay(
    factory: Any,
    orchestrator: Any,
    selfplay_config: Any,
    inference: Any,
    *,
    wall_s: float,
    label: str,
) -> WalledRun:
    """``run_selfplay(model_factory, orchestrator, selfplay, inference)``, walled.

    Positional, in the pinned order, so this suite pins the *contract* rather
    than a set of parameter names the spec never fixed.
    """
    sp = selfplay()

    def call() -> Any:
        return sp.run_selfplay(factory, orchestrator, selfplay_config, inference)

    return walled(call, wall_s, label)


# ---------------------------------------------------------------------------
# leftovers: child processes and shared-memory segments
# ---------------------------------------------------------------------------


def child_pids() -> dict[int, str]:
    """Return this process's live children, ``pid -> "comm state"``, from /proc.

    ``psutil`` is not in the dev shell, so this is a plain scan: for every
    numeric entry in ``/proc``, read ``stat`` and keep the ones whose parent is
    us.  ``stat``'s second field is the comm in parentheses (which may itself
    contain spaces), so the split is done from the closing paren rightwards --
    field 1 after it is the state, field 2 is the ppid.

    Deliberately *not* :func:`multiprocessing.active_children`: that call
    ``join``s finished children as a side effect, which would reap exactly the
    unjoined corpse design D10 requires the orchestrator to have reaped itself.
    A ``Z`` state in the result is that failure, visible.
    """
    # The multiprocessing resource tracker is a child born lazily on first
    # shm/semaphore use and it lives forever. Force it into existence NOW so
    # a before/after diff can never blame run_selfplay for birthing it -- in a
    # standalone module run it would otherwise appear as a "leaked" child
    # (wave-9c review F5).
    resource_tracker.ensure_running()
    found: dict[int, str] = {}
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        close = stat.rfind(")")
        if close < 0:
            continue
        rest = stat[close + 1 :].split()
        if len(rest) < 2:
            continue
        if int(rest[1]) != me:
            continue
        comm = stat[stat.find("(") + 1 : close]
        found[int(entry)] = f"{comm} [{rest[0]}]"
    return found


def kill_stragglers() -> list[str]:
    """Kill and join every child process this interpreter still owns.

    Used by the wall and by teardown.  ``multiprocessing.active_children`` sees
    the orchestrator's children because ``run_selfplay`` runs in *this* process
    (on the wall's helper thread) and ``multiprocessing``'s child registry is a
    module global shared by every context.
    """
    killed: list[str] = []
    for proc in multiprocessing.active_children():
        killed.append(f"{proc.name} (pid {proc.pid})")
        proc.kill()
        proc.join(KILL_GRACE_S)
    return killed


SHM_DIR: Final[Path] = Path("/dev/shm")
"""Where POSIX shared memory shows up on Linux.  ``SharedMemory`` names its
segments ``psm_<hex>`` unless told otherwise, so a survivor is visible by name.
"""

_FALLBACK_SHM_SCOPE: Final[str] = "psm_"
"""What :func:`shm_run_scope` answers when the orchestrator has no scope.

This rig is red-first by construction and must stay importable before (and
inside a spawned child of) an implementation that has not landed yet, so the
scope is read through ``getattr`` and degrades to the machine-global ``psm_``
prefix -- which is exactly the filter this rig used before KOM-054.
"""


def shm_run_scope() -> str:
    """Return the prefix every segment THIS process tree's runs are named with.

    ``komugi.selfplay.orchestrate.shm_scope()`` (KOM-054), imported lazily so
    that this module keeps its no-skip, always-importable contract.

    The scope is inherited by spawned descendants through an environment
    variable, so it covers the runs this process makes *and* the runs a child
    it spawned makes -- which is what keeps ``test_memory_grid`` (whose
    ``measure_grid`` runs each point's ``run_selfplay`` in a spawned child)
    checking something.
    """
    try:
        orchestrate = importlib.import_module("komugi.selfplay.orchestrate")
    except ImportError:
        return _FALLBACK_SHM_SCOPE
    scope = getattr(orchestrate, "shm_scope", None)
    if scope is None:
        return _FALLBACK_SHM_SCOPE
    return str(scope())


def shm_snapshot() -> set[str]:
    """Return the set of shared-memory segment names currently on disk."""
    try:
        return set(os.listdir(SHM_DIR))
    except (FileNotFoundError, PermissionError):
        return set()


def new_shm_survivors(before: set[str], scope: str | None = None) -> set[str]:
    """Return THIS run's segments that appeared since `before` and are still there.

    Design D2 puts segment lifetime in the orchestrator's ``finally``: every
    segment it created is unlinked whatever happened, including a run that
    raised and a run that restarted a shard.  A survivor is a leak that the next
    run would inherit as pressure on ``/dev/shm``, so it is asserted against
    after *every* call in this suite, successful or not.

    **The serial-execution assumption is retired (KOM-054).**  This used to
    filter on the bare ``psm_`` prefix, which is the prefix of every POSIX
    segment on the machine: any concurrent orchestrator -- another suite, a
    worker gate, an experiment -- holding a segment across our window read as
    our leak, and KOM-053 traced an overnight gate flake to exactly that.  The
    filter is now the orchestrator's own run scope
    (:func:`shm_run_scope`), so a foreign segment is not merely tolerated, it
    is *invisible*: this rig no longer needs the box to itself.

    The filter cannot blind the check to a real leak, and that is not an
    argument, it is a test -- ``test_shm_run_scope`` runs a launch that
    genuinely fails to release one of its own segments and requires this
    function to name it.

    Args:
        before: A :func:`shm_snapshot` taken before the run under audit.
        scope: Name prefix to police, defaulting to this process tree's
            (:func:`shm_run_scope`).  Pass a narrower one -- a single run's
            token from ``mint_run_token`` -- to police one run of several.
    """
    prefix = shm_run_scope() if scope is None else scope
    # Even within one scope /dev/shm is written concurrently: a sibling run on
    # another thread of this process is in scope and may be mid-teardown.
    # Re-check twice with a grace so only PERSISTENT new segments -- the actual
    # leak signature -- fail the suite (wave-9c review F4).
    survivors = {name for name in shm_snapshot() - before if name.startswith(prefix)}
    for _ in range(2):
        if not survivors:
            break
        time.sleep(1.0)
        survivors &= shm_snapshot()
    return survivors


# ---------------------------------------------------------------------------
# reading a run back off disk
# ---------------------------------------------------------------------------


def shard_dirs_on_disk(root: Path) -> list[Path]:
    """Every ``worker_<k>_r<i>`` directory under `root`, sorted by name."""
    import re

    pattern = re.compile(SHARD_DIR_RE)
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and pattern.fullmatch(path.name)),
        key=lambda path: path.name,
    )


def parse_shard_dir(path: Path) -> tuple[int, int]:
    """Return ``(worker_id, incarnation)`` parsed out of a shard directory name."""
    import re

    match = re.fullmatch(SHARD_DIR_RE, path.name)
    assert match is not None, (
        f"{path.name!r} is not a design-D10 shard directory (want worker_<k>_r<i>)"
    )
    return int(match.group(1)), int(match.group(2))


def sealed_shards(directory: Path) -> list[Path]:
    """The sealed shard files in `directory`, oldest first."""
    return sorted(directory.glob(SHARD_GLOB))


def summary_dirs(summary: Any) -> list[Path]:
    """``RunSummary.shard_dirs`` as ``Path``s, whatever it spells them as."""
    return [Path(os.fspath(entry)) for entry in summary.shard_dirs]


def summary_incarnations(summary: Any) -> list[int]:
    """``RunSummary.worker_incarnations`` as a plain list of integers.

    The pinned contract fixes the *field*, not whether it is a mapping keyed by
    worker id or a sequence indexed by it.  Both spellings answer the question
    this suite asks -- "which incarnation finished the run, per worker" -- so
    both are accepted and the values are what get asserted on.
    """
    value = summary.worker_incarnations
    if isinstance(value, Mapping):
        return [int(entry) for entry in value.values()]
    assert isinstance(value, Iterable), (
        f"worker_incarnations is {value!r}, want a mapping or a sequence"
    )
    return [int(entry) for entry in value]


def reader_for(directory: Path) -> ShardReader:
    """A schema-checked reader over one worker directory."""
    reader = ShardReader(directory, schema=selfplay().SCHEMA)
    assert not reader.skipped, (
        f"shard directory {directory} has unreadable shards: "
        f"{[(entry.path.name, entry.reason) for entry in reader.skipped]}"
    )
    return reader


def read_rows(directory: Path) -> list[dict[str, torch.Tensor]]:
    """Every row in a shard directory, in write order, schema-validated."""
    return list(reader_for(directory))


def as_int(row: Mapping[str, torch.Tensor], name: str) -> int:
    """One scalar int64 field of a row."""
    return int(row[name].item())


def worker_of(game_id: int) -> int:
    """The worker id packed into a ``game_id`` (``worker.game_id``'s radix)."""
    return game_id // (_GAME_ID_STRIDE * _GAME_ID_STRIDE)


def float_fields() -> tuple[str, ...]:
    """The schema's float32 field names, in schema order."""
    return tuple(
        field.name
        for field in selfplay().SCHEMA.fields
        if field.dtype == "float32"
    )


def int_fields() -> tuple[str, ...]:
    """The schema's **scalar** int64 field names, in schema order.

    Scalar because every caller reads these through :func:`as_int`, which is
    ``.item()``: an int64 field with a shape -- ``final_control``'s ``(81,)``
    map (KOM-010) -- is a column of 81 numbers, not one, and belongs to a
    whole-array comparison rather than to this per-column one.
    """
    return tuple(
        field.name
        for field in selfplay().SCHEMA.fields
        if field.dtype == "int64" and field.shape == ()
    )


def bind_kwargs(
    function: Callable[..., Any], candidates: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the subset of `candidates` that `function` actually accepts.

    Used only by ``test_memory_grid``.  The pinned contract spells the memory
    grid's entry point ``measure_grid(...)`` -- its two mandatory arguments are
    fixed (the model factory and the grid points) but its optional knobs are
    not, so the suite passes those by *name* and drops the ones this
    implementation does not have.  A function taking ``**kwargs`` gets
    everything.
    """
    import inspect

    parameters = inspect.signature(function).parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(candidates)
    return {name: value for name, value in candidates.items() if name in parameters}


def describe(paths: Sequence[Path]) -> str:
    """Render a directory list for an assertion message."""
    return ", ".join(sorted(path.name for path in paths)) or "<none>"


SENTINEL_ENV: Final[str] = "KOMUGI_TEST_STALL_SENTINEL"
"""Path of the stall sentinel for :func:`stall_on_sentinel_factory`."""


class StallOnSentinelNet(nn.Module):
    """A net that behaves until a sentinel file appears, then never returns.

    The wrote-then-wedged scenario the wave-9c review (F3) found untested at
    process level: a worker that sealed real shards and THEN went silent is
    detected by the mtime-staleness branch of the liveness gate
    (``log_freshness``), not by the never-started grace. The test creates the
    sentinel only after it has observed a shard file on disk, so the stall
    point is deterministic relative to the data, not to a clock.
    """

    def __init__(self, sentinel: str) -> None:
        super().__init__()
        self._sentinel = sentinel
        self._inner = TinyNet()

    def forward(self, x: torch.Tensor) -> Any:
        while os.path.exists(self._sentinel):
            time.sleep(0.05)
        return self._inner(x)


def stall_on_sentinel_factory() -> nn.Module:
    """Stall-on-sentinel for incarnation 0; plain :class:`TinyNet` afterwards.

    The restarted server must NOT re-read the sentinel (it still exists), or
    the run could never converge -- :func:`bump_counter` decides, exactly as
    :func:`wedge_once_factory` does.
    """
    sentinel = os.environ.get(SENTINEL_ENV)
    if not sentinel:
        raise RuntimeError(f"{SENTINEL_ENV} is unset")
    if bump_counter() == 0:
        model: nn.Module = StallOnSentinelNet(sentinel)
    else:
        model = TinyNet()
    model.eval()
    return model
