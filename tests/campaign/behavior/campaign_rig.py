"""Shared rig for the KOM-131 campaign-loop behavioural suite.

The campaign loop (KOM-031: ``komugi/campaign/loop.py`` and
``komugi/campaign/factory.py``) is being implemented concurrently by a
different worker, so everything here is written against the pinned contract and
nothing else::

    komugi.campaign.factory
        CHECKPOINT_ENV = "KOMUGI_CAMPAIGN_CKPT"
        serving_net() -> nn.Module        # module-level, env-configured
        # KOMUGI_CAMPAIGN_DEVICE: "cpu" | "cuda" (default cpu)
        # KOMUGI_CAMPAIGN_NET:    "small" selects channels=16 / blocks=2

    komugi.campaign.loop
        CampaignConfig(iterations, run_seed, root, steps_per_iteration,
                       batch_size, lr, workers, games_per_worker, device="cpu")
        IterationResult(iteration, games, rows, first_loss, last_loss,
                        checkpoint_step)
        run_campaign(campaign, selfplay, orchestrator_overrides=None,
                     inference=None) -> list[IterationResult]

    komugi.train.step.training_iteration(..., *, global_step_base=0)

What this suite is *for*
------------------------
The pieces on either side of the campaign loop are already pinned by their own
behavioural suites: ``tests/selfplay/behavior`` owns "self-play produces
shards", ``tests/train/behavior`` owns "a training iteration consumes shards,
writes a checkpoint and a journal record". What nobody owns is the *seam
between two iterations*, and every claim this suite makes lives there:

* iteration 1 happens **after** iteration 0 and does not overwrite it
  (two checkpoints, strictly monotone steps -- the campaign-level shape of the
  wave-10 F5 same-step-supersede finding);
* iteration 1's self-play is played by **iteration 0's weights** -- the link
  between "a checkpoint was written" and "a checkpoint was served", which is
  otherwise a comment in a docstring;
* a crash costs at most one iteration (resume does not replay iteration 0);
* the design-§4 recency window finally indexes rows from more than one
  iteration.

Rules this rig obeys, and why each is a rule
--------------------------------------------

* **Every ``run_campaign`` call is wall-bounded.** ``pytest-timeout`` is not in
  the dev shell (see ``flake.nix``), and a campaign owns a self-play
  orchestrator, which owns spawned processes, which is exactly the kind of
  thing that hangs. So no test calls ``run_campaign`` directly:
  :func:`walled` runs it on a daemon helper thread and turns "still running at
  the wall" into a *failure in bounded time* rather than a wedged
  ``ci/check.sh``. A thread cannot be killed, so the wall additionally kills
  every child process the campaign left behind, which is what lets the stuck
  thread unwind. This is the discipline of
  ``tests/selfplay/behavior/orchestrate_rig.py``, deliberately restated rather
  than imported: that module sits under a *different* rootdir-inserted
  directory, so importing it would make this suite's importability depend on
  pytest having already collected another one (and ``tests/campaign`` sorts
  first).
* **The walls are generous and the liveness knobs more so.** A wall bounds a
  hang; it is not a latency assertion. The orchestrator's liveness /
  first-shard / drain timeouts are pushed to ten minutes for the same reason
  in reverse: these games take seconds, so the only thing those knobs could do
  here is turn a slow CI box into a spurious shard restart. Only the walls
  bound these tests.
* **One campaign, shared.** A two-iteration campaign plays real games with the
  real net and runs real optimiser steps; it is the most expensive thing in
  the suite by an order of magnitude. :func:`shared_campaign` runs it at most
  once per interpreter and hands the same result to every module that asks
  (each through its own module-scoped fixture). A separate run is paid for
  only where the scenario demands it -- ``test_kill_and_resume`` needs a
  campaign that *stopped*, which by definition cannot be the one that ran to
  completion.
* **Failure is cached too.** If the shared campaign raises or hits its wall,
  the outcome is remembered and re-raised, so three modules do not pay three
  times to watch the same hang.
* **Nothing here carries a skip gate.** Each test module gates on
  ``pytest.importorskip("komugi.campaign.loop")`` plus a ``hasattr`` check for
  ``run_campaign``; this module stays importable regardless, so that a helper
  can be called from anywhere without a module-level ``Skipped`` escaping.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import multiprocessing
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import torch

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.observation import observation_size
from komugi.selfplay.inference import InferenceConfig
from komugi.selfplay.records import SCHEMA, ConcatSource
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics
from komugi.train.checkpoint import (
    CHECKPOINT_GLOB,
    CheckpointRef,
    list_checkpoints,
    load_checkpoint_file,
)
from komugi.train.shards import SHARD_GLOB, ShardReader

# ---------------------------------------------------------------------------
# the pinned environment contract
# ---------------------------------------------------------------------------

CHECKPOINT_ENV: Final[str] = "KOMUGI_CAMPAIGN_CKPT"
"""What ``komugi.campaign.factory.CHECKPOINT_ENV`` must be, spelled here too.

Restated rather than imported so that ``test_weights_are_actually_served`` can
assert the two agree: the constant is the whole interface between the trainer
(which writes a checkpoint) and a *spawned server process* (which must load
it), and a rename that broke it would otherwise be invisible -- self-play would
keep working, with random weights, forever.
"""

NET_ENV: Final[str] = "KOMUGI_CAMPAIGN_NET"
"""Selects the net shape the factory builds. ``"small"`` is this suite's."""

DEVICE_ENV: Final[str] = "KOMUGI_CAMPAIGN_DEVICE"
"""``"cpu"`` or ``"cuda"``; everything here is CPU."""

SMALL_NET: Final[str] = "small"
"""The tiny channels=16 / blocks=2 config. A real b18c256 forward pass in a
2-worker self-play run would make this suite cost minutes per iteration and
would pin nothing extra: every claim here is about *plumbing*."""

CPU_DEVICE: Final[str] = "cpu"

# ---------------------------------------------------------------------------
# shapes and budgets
# ---------------------------------------------------------------------------

IN_PLANES, BOARD_H, BOARD_W = observation_size()
"""The real encoder's observation shape -- (128, 9, 9) today."""

NUM_ACTIONS: int = ACTION_SIZE
"""The real engine's flat action space, ``A``."""

ITERATIONS: Final[int] = 2
"""Iterations in the shared campaign. Two is the smallest number that has a
*seam*, and the seam is this suite's entire subject."""

RUN_SEED: Final[int] = 20260823

WORKERS: Final[int] = 2
"""Two, not one: the design-F5 ``ConcatSource`` path (several worker
directories per iteration) is only exercised above one."""

GAMES_PER_WORKER: Final[int] = 1
STEPS_PER_ITERATION: Final[int] = 8
BATCH_SIZE: Final[int] = 16
LR: Final[float] = 1e-3

LIVENESS_S: Final[float] = 600.0
"""Liveness / first-shard / drain patience handed to the orchestrator.

Ten minutes for games that take seconds. These knobs exist to notice a *wedge*;
at this scale the only thing a tight setting could produce is a shard restart
caused by a loaded CI box, which would corrupt every count this suite asserts
on. The walls below are what bound the tests."""

CAMPAIGN_WALL_S: Final[float] = 1800.0
"""Wall for the shared two-iteration campaign."""

RESUME_WALL_S: Final[float] = 1200.0
"""Wall for a single-iteration campaign, and for the resumed one."""

KILL_GRACE_S: Final[float] = 10.0
"""How long a straggler child gets to die after ``kill()``."""

ITER_DIR_FMT: Final[str] = "iter_{iteration}"
"""How the loop lays iterations out under ``CampaignConfig.root``.

Part of the pinned contract (``root/iter_0``, ``root/iter_1``): the layout is
what makes an iteration's data addressable after the fact -- by the recency
window, by a resumed run, and by an operator reading a dead campaign off disk.
"""

SHARD_DIR_RE: Final[str] = r"worker_(\d+)_r(\d+)"
"""Design D10's per-worker-per-incarnation directory name, as a pattern."""


# ---------------------------------------------------------------------------
# the modules under test, imported lazily
# ---------------------------------------------------------------------------


def campaign_loop() -> ModuleType:
    """Return :mod:`komugi.campaign.loop`, imported in whichever interpreter asks."""
    return importlib.import_module("komugi.campaign.loop")


def campaign_factory() -> ModuleType:
    """Return :mod:`komugi.campaign.factory`."""
    return importlib.import_module("komugi.campaign.factory")


def has_run_campaign() -> bool:
    """Whether KOM-031's loop has landed."""
    try:
        return hasattr(campaign_loop(), "run_campaign")
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def tiny_selfplay_config(**overrides: Any) -> SelfPlayConfig:
    """``SelfPlayConfig`` at the KOM-131 budgets.

    ``prob_fast`` is one half rather than the production 0.75 so a short run
    still records both ``search_mode`` values; the Gumbel budgets are the
    spec's smallest that still runs the real search.
    """
    fields: dict[str, Any] = {
        "workers": WORKERS,
        "process_batch_size": 1,
        "gumbel_n_full": 4,
        "gumbel_m_full": 2,
        "gumbel_n_fast": 2,
        "gumbel_m_fast": 2,
        "prob_fast": 0.5,
    }
    fields.update(overrides)
    return SelfPlayConfig(**fields)


def inference_config(**overrides: Any) -> InferenceConfig:
    """``InferenceConfig`` at the engine's real shapes.

    Passed explicitly rather than left to the loop's default, because the seam
    shapes are the one thing a campaign cannot guess wrong quietly: a mismatch
    between the served net and the slot geometry surfaces inside a spawned
    child. ``request_timeout_s`` is high for the reason
    :data:`LIVENESS_S` is high.
    """
    fields: dict[str, Any] = {
        "in_planes": IN_PLANES,
        "num_actions": NUM_ACTIONS,
        "max_batch": 4,
        "batch_timeout_ms": 1.0,
        "request_timeout_s": 240.0,
    }
    fields.update(overrides)
    return InferenceConfig(**fields)


def orchestrator_overrides(**overrides: Any) -> dict[str, Any]:
    """The liveness knobs every campaign in this suite runs with."""
    fields: dict[str, Any] = {
        "liveness_timeout_s": LIVENESS_S,
        "first_shard_grace_s": LIVENESS_S,
        "drain_timeout_s": LIVENESS_S,
    }
    fields.update(overrides)
    return fields


def campaign_config(root: Path, iterations: int = ITERATIONS, **overrides: Any) -> Any:
    """``CampaignConfig`` over `root`, at this suite's tiny sizes."""
    fields: dict[str, Any] = {
        "iterations": iterations,
        "run_seed": RUN_SEED,
        "root": str(root),
        "steps_per_iteration": STEPS_PER_ITERATION,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "workers": WORKERS,
        "games_per_worker": GAMES_PER_WORKER,
        "device": CPU_DEVICE,
    }
    fields.update(overrides)
    return campaign_loop().CampaignConfig(**fields)


@contextlib.contextmanager
def small_net_env(**extra: str) -> Iterator[None]:
    """Point the module-level factory at the tiny CPU net, and restore after.

    Plain ``os.environ`` rather than ``monkeypatch`` because the shared
    campaign is built outside any fixture's scope, and because the value has to
    be visible to *spawned children*: they inherit this process's environment,
    and the server that builds the served net is one of them.
    """
    wanted: dict[str, str] = {NET_ENV: SMALL_NET, DEVICE_ENV: CPU_DEVICE}
    wanted.update(extra)
    previous = {name: os.environ.get(name) for name in wanted}
    os.environ.update(wanted)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# walls
# ---------------------------------------------------------------------------


class WallExpired(AssertionError):
    """``run_campaign`` was still running when its wall ran out."""


@dataclass(frozen=True)
class WalledRun:
    """What one walled ``run_campaign`` call produced.

    Exactly one of `results` and `error` is set. Both are captured rather than
    re-raised at the call site so that a failure can be reported once, with its
    traceback, by whichever test unwraps it.
    """

    results: Any | None
    error: BaseException | None
    elapsed: float

    def unwrap(self) -> list[Any]:
        """Return the ``IterationResult`` list, failing the test if it raised."""
        if self.error is not None:
            raise AssertionError(
                f"run_campaign raised {type(self.error).__name__}: {self.error}"
            ) from self.error
        assert self.results is not None, (
            "run_campaign returned None, want a list of IterationResults"
        )
        return list(self.results)


def kill_stragglers() -> list[str]:
    """Kill and join every child process this interpreter still owns.

    ``multiprocessing.active_children`` sees the orchestrator's children
    because ``run_campaign`` runs in *this* process (on the wall's helper
    thread) and the child registry is a module global shared by every context.
    """
    killed: list[str] = []
    for proc in multiprocessing.active_children():
        killed.append(f"{proc.name} (pid {proc.pid})")
        proc.kill()
        proc.join(KILL_GRACE_S)
    return killed


def walled(call: Callable[[], Any], wall_s: float, label: str) -> WalledRun:
    """Run `call` on a daemon thread and hold it to `wall_s`.

    A thread rather than a subprocess, because the campaign must run *in this
    interpreter* for its spawned grandchildren to be reachable by
    :func:`kill_stragglers` and for the temporary directories to be ours. A
    thread cannot be killed, so the wall does the next best thing and kills the
    campaign's children: that unblocks whatever the stuck thread was waiting
    on, and the thread is a daemon so a truly unkillable one cannot stop the
    interpreter exiting. Either way the *test* fails now instead of the gate
    hanging.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        started = time.monotonic()
        try:
            box["results"] = call()
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
            f"{label}: run_campaign was still running after {wall_s}s. "
            f"Killed {len(killed)} orphaned child process(es): {killed}"
        )
    return WalledRun(
        results=box.get("results"),
        error=box.get("error"),
        elapsed=float(box.get("elapsed", time.monotonic() - started)),
    )


def walled_run_campaign(
    root: Path,
    *,
    iterations: int = ITERATIONS,
    wall_s: float = CAMPAIGN_WALL_S,
    label: str = "campaign",
) -> WalledRun:
    """``run_campaign(campaign, selfplay, orchestrator_overrides=, inference=)``.

    The two mandatory arguments travel positionally, in the pinned order, so
    this suite pins the *contract* rather than two parameter names the spec
    never fixed; the optional pair travel by keyword, which the spec did fix.
    """

    def call() -> Any:
        with small_net_env():
            return campaign_loop().run_campaign(
                campaign_config(root, iterations),
                tiny_selfplay_config(),
                orchestrator_overrides=orchestrator_overrides(),
                inference=inference_config(),
            )

    return walled(call, wall_s, label)


# ---------------------------------------------------------------------------
# the one shared campaign
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedCampaign:
    """The two-iteration campaign every module but the resume test reads."""

    root: Path
    run: WalledRun

    def results(self) -> list[Any]:
        """The ``IterationResult`` list, or an assertion carrying the failure."""
        return self.run.unwrap()


_SHARED: dict[str, SharedCampaign] = {}
"""Memo for :func:`shared_campaign`. A dict rather than a module-level
``SharedCampaign | None`` so that "not built yet" and "built, and it failed"
are different states -- the second must not be rebuilt three times."""


def shared_campaign() -> SharedCampaign:
    """Run the two-iteration campaign once per interpreter and return it.

    The root is a plain ``mkdtemp`` rather than pytest's ``tmp_path_factory``
    because this runs outside any one module's fixture scope -- it is shared
    *between* modules, and the first one to ask pays for it. The directory is
    deliberately not removed: when an assertion about a checkpoint or a shard
    tree fails, the tree it failed about is the first thing worth looking at.
    """
    cached = _SHARED.get("run")
    if cached is not None:
        return cached
    root = Path(tempfile.mkdtemp(prefix="komugi-campaign-shared-"))
    built = SharedCampaign(
        root=root,
        run=walled_run_campaign(
            root, iterations=ITERATIONS, wall_s=CAMPAIGN_WALL_S, label="shared"
        ),
    )
    _SHARED["run"] = built
    return built


# ---------------------------------------------------------------------------
# reading a campaign back off disk
# ---------------------------------------------------------------------------


def iteration_dir(root: Path, iteration: int) -> Path:
    """Return ``root/iter_<n>``, whether or not it exists."""
    return root / ITER_DIR_FMT.format(iteration=iteration)


def shard_dirs_under(directory: Path) -> list[Path]:
    """Every ``worker_<k>_r<i>`` directory at or below `directory`, sorted.

    Recursive because the contract pins ``root/iter_<n>`` and design D10 pins
    ``worker_<k>_r<i>``, but says nothing about whether the loop puts anything
    between them (a ``shards/`` level is a reasonable choice this suite must
    not forbid).
    """
    import re

    pattern = re.compile(SHARD_DIR_RE)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_dir() and pattern.fullmatch(path.name)
        ),
        key=lambda path: str(path),
    )


def sealed_shards(directory: Path) -> list[Path]:
    """Every sealed shard at or below `directory`, sorted by path."""
    return sorted(directory.rglob(SHARD_GLOB))


def readers_for(dirs: Sequence[Path]) -> list[ShardReader]:
    """One schema-checked :class:`ShardReader` per non-empty worker directory."""
    readers: list[ShardReader] = []
    for directory in dirs:
        reader = ShardReader(directory, schema=SCHEMA)
        assert not reader.skipped, (
            f"shard directory {directory} has unreadable shards: "
            f"{[(entry.path.name, entry.reason) for entry in reader.skipped]}"
        )
        if len(reader) > 0:
            readers.append(reader)
    return readers


def concat_source(dirs: Sequence[Path]) -> ConcatSource:
    """One iteration's worker directories as a single ``SampleSource``.

    Design F5's answer to "``RecencyWindow`` takes one source per iteration but
    an iteration is several directories", which is exactly the wiring
    ``test_window_spans_iterations`` reconstructs.
    """
    return ConcatSource(readers_for(dirs))


def checkpoint_refs(root: Path) -> list[CheckpointRef]:
    """Every checkpoint anywhere under `root`, oldest first.

    Found by glob rather than by an assumed directory, because the contract
    pins the *step numbering* (``IterationResult.checkpoint_step``) and the
    atomic naming (``komugi.train.checkpoint``) but not where the loop keeps
    them. Sorted ascending by step, then digest, so the order is total.
    """
    directories = sorted({path.parent for path in root.rglob(CHECKPOINT_GLOB)})
    refs = [ref for directory in directories for ref in list_checkpoints(directory)]
    return sorted(refs, key=lambda ref: (ref.step, ref.digest))


EMA_ROLE: Final[str] = "ema"
"""What an EMA checkpoint's ``meta['role']`` says it is (KOM-022 §2)."""


def is_ema_checkpoint(ref: CheckpointRef) -> bool:
    """Whether `ref` is a published EMA rather than a trainer's checkpoint.

    Decided by what the file *says it is* -- ``meta['role'] == 'ema'``, written
    by :func:`komugi.train.ema.ema_meta` -- rather than by which directory it
    sits in, for the same reason :func:`checkpoint_refs` discovers checkpoints
    by glob: the campaign's layout is not in the pinned contract, and a suite
    that hard-coded ``<root>/ema`` would pass vacuously the day the loop moved
    it. A file that does not verify is *not* claimed to be an EMA: it is left
    in the trainer's listing, where the caller's own assertion reports it.
    """
    checkpoint = load_checkpoint_file(ref.path)
    if checkpoint is None:
        return False
    return str(checkpoint.meta.get("role", "")) == EMA_ROLE


def train_checkpoint_refs(root: Path) -> list[CheckpointRef]:
    """Every **trainer** checkpoint under `root`, oldest first.

    KOM-022 §2 (decision W1) publishes a *second* checkpoint series into the
    same root: the EMA self-play serves, written through the unchanged
    :func:`~komugi.train.checkpoint.save_checkpoint` into ``<root>/ema`` so
    that the thing self-play serves is an ordinary checkpoint with an ordinary
    digest-bearing name. So ``<root>`` now holds two files per iteration, and
    :func:`checkpoint_refs` -- which finds every checkpoint anywhere under the
    root, deliberately -- returns both series.

    The claims about the *trainer's* checkpoints are claims about this subset:
    one file per iteration, at strictly increasing steps, carrying AdamW's
    moments. The EMA carries ``optim_state={}`` by design and would answer
    "the checkpoint at the newest step" with a file that never held an
    optimiser.

    Copies of one checkpoint count once. §4's anchor is *the same file* under
    ``<root>/anchors/gen_<g>/``, deliberately keeping its published basename so
    the loader can read it back, and a frozen opponent is not a second
    iteration of the trainer. Two files agreeing on step **and** digest are the
    same bytes: the name carries 64 bits of sha256 over the file.
    """
    seen: set[tuple[int, str]] = set()
    trained: list[CheckpointRef] = []
    for ref in checkpoint_refs(root):
        if is_ema_checkpoint(ref) or (ref.step, ref.digest) in seen:
            continue
        seen.add((ref.step, ref.digest))
        trained.append(ref)
    return trained


def journal_candidates(root: Path) -> list[Path]:
    """Files under `root` that could be the campaign's telemetry journal."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".jsonl", ".json", ".log"} or "journal" in path.name)
    )


def journal_records(root: Path) -> list[IterationMetrics]:
    """Every :class:`IterationMetrics` written anywhere under `root`.

    The journal's path is not in the pinned contract, so it is discovered.
    Records are returned in file-then-line order, which for a single journal is
    write order -- the thing an operator reconstructs a dead campaign from.
    """
    records: list[IterationMetrics] = []
    for path in journal_candidates(root):
        read = read_records(path, IterationMetrics, missing_ok=True)
        records.extend(read.records)
    return records


def digest_of(path: Path) -> str:
    """Return ``sha256`` over a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FileStamp:
    """A file's identity for a before/after comparison: bytes and mtime.

    Both, because they answer different questions. The digest catches a rewrite
    that happens to produce the same length; the mtime catches a rewrite that
    produces the same *bytes* -- which is precisely what an iteration-0 replay
    would look like if the campaign is deterministic, and precisely the thing
    ``test_kill_and_resume`` must be able to see.
    """

    digest: str
    mtime_ns: int
    size: int


def stamp_tree(directory: Path) -> dict[str, FileStamp]:
    """Stamp every file at or below `directory`, keyed by relative path."""
    stamps: dict[str, FileStamp] = {}
    if not directory.is_dir():
        return stamps
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        info = path.stat()
        stamps[str(path.relative_to(directory))] = FileStamp(
            digest=digest_of(path), mtime_ns=info.st_mtime_ns, size=info.st_size
        )
    return stamps


def state_dicts_match(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    """Whether two state dicts hold the same keys and bit-identical tensors.

    ``torch.equal`` rather than ``allclose``: the claim under test is that the
    *same bytes* were served, not that something similar was. A tolerance here
    would pass for a net that had been trained one extra step, which is the
    interesting way for the serving path to be wrong.
    """
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key].cpu(), right[key].cpu()) for key in left)


def describe(paths: Sequence[Path]) -> str:
    """Render a path list for an assertion message."""
    return ", ".join(sorted(str(path) for path in paths)) or "<none>"
