"""Three ways to break a run, and the noise each one has to make.

Design D2/Q4 draws a hard line through the failure space.  A *worker* dying is
recoverable: the whole shard restarts with fresh segments and a bumped
incarnation.  A *server* dying is not, in P0a -- a running server cannot
re-attach fresh segments (dynamic registration with generations is the
pre-registered P1 ticket) -- so the run fails loudly.  And a wedge that keeps
recurring past ``max_shard_restarts`` is the third case: restarting forever
would turn a permanently broken model into a run that never ends and never
says why.

What "loudly" has to mean, in all three cases:

* an exception reaches the caller -- not a summary reporting zero games, and not
  a log line;
* its message names the thing that broke, so the operator does not have to
  reconstruct a five-process failure from timestamps;
* it happens in **bounded time** (every call here is walled: a failure mode that
  hangs is the same outage as a hang);
* and ``/dev/shm`` is clean afterwards.  Design D2/F9 puts every segment's
  ``unlink`` in the orchestrator's ``finally`` precisely so that the failure
  path is the *tested* path; Python 3.12 has no ``SharedMemory(track=...)`` and
  the resource tracker is not a backstop for a badly-dying owner (bpo-38119).

The third case doubles as the only test in this suite that watches the restart
budget being spent: ``max_shard_restarts=1`` with a factory that hangs on
*every* incarnation means two wedges, one restart, and then a stop.

Two more scenarios (KOM-053) split that third case along the seam where the
``/dev/shm`` obligation is easiest to drop, and both replace the *process
targets* rather than the model, so neither depends on how loaded the box is:

* **never started** -- every worker blocks before its first game, so the
  restart is declared by the ``first_shard_grace_s`` branch of the liveness
  gate ("it never got started") rather than by an exit code or a stale mtime.
  The budget is spent on that branch alone, and both incarnations' segments
  still have to be gone at the end.
* **half launched** -- an incarnation that never reaches ``started`` at all:
  ``_launch`` creates the segments, then spawning a worker fails.  There is no
  ``_Incarnation`` for the orchestrator to hand ``_teardown``, so ``_launch``
  owes the cleanup itself, and it owes it in a ``finally`` with the same
  per-item guard: one segment's release raising must not leak the rest.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_orchestrate: ModuleType = importlib.import_module("komugi.selfplay.orchestrate")
"""The implementation module, for the two scenarios that swap process targets.

``run_selfplay`` reads ``serve_shard`` and ``worker_main`` out of this module's
globals at spawn time, so patching them here is what lets a scenario pin a
*supervision* branch without also paying for a real server, a real model and a
real game -- which is exactly what made the never-started branch reproducible
only under load before (KOM-053).
"""

_RUN_SEED = 5150
_LIVENESS_TIMEOUT_S = 10.0
"""Small, and safe to keep small here: in the always-hanging scenario NO
incarnation can ever produce liveness evidence, so there is no healthy worker
for a tight gate to shoot by mistake (contrast ``test_wedge_restart``)."""

_STARTUP_GRACE_S = 1.0
"""``first_shard_grace_s`` for the never-started scenario.

Below the orchestrator's own two-second poll interval on purpose: the *first*
pass sees an incarnation zero seconds old and lets it be, the second sees one
past the grace with an empty shard directory, and there is no third answer.
That is what makes the branch deterministic rather than a race between the
grace and however long a spawned child takes to import torch on a busy box.
"""


@dataclass(frozen=True)
class Failure:
    """A run that was supposed to raise, and what it left behind."""

    error: BaseException
    elapsed: float
    shm_leaked: set[str]
    children_leaked: dict[int, str]

    @property
    def message(self) -> str:
        """The exception's message, lowercased, for substring assertions."""
        return str(self.error).lower()


def _expect_failure(
    factory: Any,
    root: Path,
    *,
    label: str,
    wall_s: float,
    **orchestrator_overrides: Any,
) -> Failure:
    """Run the orchestrator inside a wall and require that it raised."""
    shm_before = rig.shm_snapshot()
    children_before = set(rig.child_pids())

    overrides: dict[str, Any] = {
        "run_seed": _RUN_SEED,
        "workers": 1,
        "games_per_worker": 1,
    }
    overrides.update(orchestrator_overrides)
    outcome = rig.walled_run_selfplay(
        factory,
        rig.orchestrator_config(root, **overrides),
        rig.tiny_selfplay_config(workers=int(overrides["workers"])),
        rig.inference_config(max_batch=1),
        wall_s=wall_s,
        label=label,
    )
    error = outcome.raised()

    leaked_shm = rig.new_shm_survivors(shm_before)
    leaked_children = {
        pid: name
        for pid, name in rig.child_pids().items()
        if pid not in children_before
    }
    rig.kill_stragglers()
    return Failure(
        error=error,
        elapsed=outcome.elapsed,
        shm_leaked=leaked_shm,
        children_leaked=leaked_children,
    )


@pytest.fixture(scope="module")
def server_death(tmp_path_factory: pytest.TempPathFactory) -> Failure:
    """A model factory that raises where the server builds its model."""
    root = Path(tmp_path_factory.mktemp("server-death")) / "shards"
    root.mkdir()
    return _expect_failure(
        rig.exploding_factory, root, label="server-death", wall_s=rig.LOUD_WALL_S
    )


@pytest.fixture(scope="module")
def bad_factory(tmp_path_factory: pytest.TempPathFactory) -> Failure:
    """A lambda where design D2 requires a module-level factory."""
    root = Path(tmp_path_factory.mktemp("bad-factory")) / "shards"
    root.mkdir()
    return _expect_failure(
        lambda: rig.tiny_factory(),
        root,
        label="bad-factory",
        wall_s=rig.LOUD_WALL_S,
    )


@pytest.fixture(scope="module")
def exhausted_restarts(tmp_path_factory: pytest.TempPathFactory) -> Failure:
    """Every incarnation wedges, and the restart budget is one."""
    root = Path(tmp_path_factory.mktemp("exhausted")) / "shards"
    root.mkdir()
    return _expect_failure(
        rig.always_hanging_factory,
        root,
        label="exhausted-restarts",
        wall_s=rig.LOUD_WALL_S,
        liveness_timeout_s=_LIVENESS_TIMEOUT_S,
        drain_timeout_s=30.0,
        # The never-started grace is its own knob since the wave-9c review
        # split it out of drain_timeout_s (F3) -- an always-empty shard dir is
        # detected by THIS timer, so it must be the small one here.
        first_shard_grace_s=30.0,
        max_shard_restarts=1,
    )


# ---------------------------------------------------------------------------
# process targets for the two KOM-053 scenarios
#
# Module-level, because ``spawn`` pickles a process target by qualified name --
# the same rule the model factory obeys.  A spawned child re-imports THIS
# module, which is safe only because the guard at the top cannot skip once
# ``run_selfplay`` exists (a module-level skip in a child would raise
# ``Skipped`` in an interpreter with no session and turn its exit code into a
# lie -- see ``orchestrate_rig``'s docstring).
# ---------------------------------------------------------------------------


def idle_server_main(
    model_factory: Any, inference: Any, handles: Any, stop_event: Any
) -> None:
    """A shard server that starts, answers nothing, and waits to be stopped.

    ``serve_shard``'s signature and none of its behaviour.  It deliberately
    never attaches to a segment: the scenarios it serves are about an
    incarnation whose *workers* never got going, and a server that answered
    would only add a second reason for the run to end.

    Alive until ``stop_event`` -- so ``_poll_shard``'s first question ("is the
    server gone?") keeps answering no, and the run reaches the liveness gate.
    """
    del model_factory, inference, handles
    stop_event.wait()


def blocked_worker_main(
    worker_id: Any,
    incarnation: Any,
    run_seed: Any,
    games: Any,
    shard_dir: Any,
    shm_name: Any,
    selfplay_config: Any,
    inference: Any,
    extra_lanes: Any = (),
) -> None:
    """A worker that starts and then blocks before it can seal anything.

    ``worker_main``'s signature, arity for arity, and the smallest possible
    never-started worker: it is alive (so the exit-code branch cannot fire) and
    its shard directory stays empty forever (so the ``log_freshness`` branch has
    no mtime to judge).  ``first_shard_grace_s`` is then the ONLY thing that can
    end the run, which is the whole point of the scenario.

    ``sleep`` in a loop rather than a spin, so the wedge costs a core nothing
    and SIGKILL lands promptly.
    """
    del (
        worker_id,
        incarnation,
        run_seed,
        games,
        shard_dir,
        shm_name,
        selfplay_config,
        inference,
        extra_lanes,
    )
    while True:
        time.sleep(0.05)


def unspawnable_worker_target() -> Any:
    """Return a worker target ``spawn`` cannot pickle, so ``start()`` raises.

    A *local* function: ``multiprocessing`` pickles the process object in the
    parent, before any child exists, so this fails deterministically inside
    ``_launch`` -- after the segments are created and the server is started, and
    before the incarnation reaches ``started``.  That is the exact window the
    half-launched scenario is about, and it is reached without racing anything.
    """

    def target(*args: Any) -> None:
        raise AssertionError("this target can never reach a child process")

    return target


@pytest.fixture(scope="module")
def never_started(tmp_path_factory: pytest.TempPathFactory) -> Failure:
    """Every incarnation's worker blocks before its first game (KOM-053).

    The startup-timeout twin of ``exhausted_restarts``: same budget, same
    ending, different branch of the gate -- and no model, no server and no game
    anywhere in it, so which branch fires is decided by arithmetic on
    ``first_shard_grace_s`` rather than by how fast the box is.
    """
    root = Path(tmp_path_factory.mktemp("never-started")) / "shards"
    root.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_orchestrate, "serve_shard", idle_server_main)
        patch.setattr(_orchestrate, "worker_main", blocked_worker_main)
        return _expect_failure(
            rig.tiny_factory,
            root,
            label="never-started",
            wall_s=rig.LOUD_WALL_S,
            liveness_timeout_s=_LIVENESS_TIMEOUT_S,
            drain_timeout_s=10.0,
            first_shard_grace_s=_STARTUP_GRACE_S,
            max_shard_restarts=1,
        )


@pytest.fixture(scope="module")
def half_launched(tmp_path_factory: pytest.TempPathFactory) -> Failure:
    """``_launch`` fails after creating segments, and one release raises.

    Two faults, because one alone proves nothing: spawning a worker fails (so
    the incarnation never reaches ``started`` and ``_teardown`` is never
    called), and the first segment's release raises *after* doing its job (so
    the sweep has to survive an error to reach the second segment).  With
    ``workers=2`` there are two segments and the second one is the evidence: a
    cleanup that stops at the first exception leaves it in ``/dev/shm``.

    ``_release`` is patched rather than faked away -- the real release still
    runs -- so the only thing this scenario changes is whether the sweep
    *continues*, which is the property under test.
    """
    root = Path(tmp_path_factory.mktemp("half-launched")) / "shards"
    root.mkdir()
    real_release = _orchestrate._release
    released: list[int] = []

    def failing_release(slots: Any) -> None:
        real_release(slots)
        released.append(1)
        if len(released) == 1:
            raise RuntimeError("KOM-053: simulated failure releasing segment 0")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_orchestrate, "serve_shard", idle_server_main)
        patch.setattr(_orchestrate, "worker_main", unspawnable_worker_target())
        patch.setattr(_orchestrate, "_release", failing_release)
        return _expect_failure(
            rig.tiny_factory,
            root,
            label="half-launched",
            wall_s=rig.LOUD_WALL_S,
            workers=2,
        )


# ---------------------------------------------------------------------------
# (a) server death
# ---------------------------------------------------------------------------


def test_a_dead_server_fails_the_run_and_names_the_server(
    server_death: Failure,
) -> None:
    """Design Q4: server death is fatal to the run, not a restart trigger.

    The factory raises *inside* the server process, which is where design D2
    says a model is built (never pickled across the boundary), so the
    orchestrator sees a server that exited before it ever answered anything. It
    must say so: "worker 0 timed out" would send an operator looking at the
    wrong process entirely.
    """
    assert isinstance(server_death.error, Exception), (
        f"run_selfplay raised {server_death.error!r}, want a normal exception"
    )
    assert "server" in server_death.message, (
        f"the failure message does not mention the server: {server_death.error}"
    )


def test_a_dead_server_does_not_hang_the_run(server_death: Failure) -> None:
    """It raises promptly rather than waiting out a drain or a liveness gate.

    The wall already bounds this; the tighter assertion is what distinguishes
    "noticed the server exit" from "eventually timed out waiting for rows that
    were never coming", which is the same outage with a different message.
    """
    assert server_death.elapsed < rig.LOUD_WALL_S / 2.0, (
        f"took {server_death.elapsed:.1f}s to notice a server that died at "
        f"construction time"
    )


def test_a_dead_server_leaves_no_segments_and_no_children(
    server_death: Failure,
) -> None:
    """The ``finally`` runs on the failure path too (design D2/F9)."""
    assert not server_death.shm_leaked, (
        f"segments still in /dev/shm: {sorted(server_death.shm_leaked)}"
    )
    assert not server_death.children_leaked, (
        f"child processes surviving the failure: {server_death.children_leaked}"
    )


# ---------------------------------------------------------------------------
# (b) a factory that cannot cross a process boundary
# ---------------------------------------------------------------------------


_FACTORY_MARKERS = ("module-level", "module level", "picklable", "pickle", "lambda")
"""Any of these in the message means the operator was told the real rule.

Design D2 permits a checkpoint path or a **module-level** factory and nothing
else, because ``spawn`` pickles the target's arguments: a lambda, a closure or a
bound method dies in the child with ``Can't pickle local object``, several
frames deep in ``multiprocessing`` and long after the useful context is gone.
The orchestrator is required to reject it in the parent, where it can explain.
"""


def test_a_lambda_factory_is_rejected_with_the_documented_error(
    bad_factory: Failure,
) -> None:
    """A non-module-level factory fails at the API, saying which rule it broke."""
    assert isinstance(bad_factory.error, Exception), (
        f"run_selfplay raised {bad_factory.error!r}, want a normal exception"
    )
    assert any(marker in bad_factory.message for marker in _FACTORY_MARKERS), (
        f"the failure message explains nothing about the module-level factory "
        f"rule (design D2): {bad_factory.error}"
    )


def test_a_lambda_factory_is_rejected_before_anything_is_spawned(
    bad_factory: Failure,
) -> None:
    """Cheap and immediate: no processes, no segments, no half-started run."""
    assert bad_factory.elapsed < rig.LOUD_WALL_S / 2.0, (
        f"took {bad_factory.elapsed:.1f}s to reject an unspawnable factory"
    )
    assert not bad_factory.shm_leaked, (
        f"segments still in /dev/shm: {sorted(bad_factory.shm_leaked)}"
    )
    assert not bad_factory.children_leaked, (
        f"child processes surviving the rejection: {bad_factory.children_leaked}"
    )


# ---------------------------------------------------------------------------
# (c) the restart budget runs out
# ---------------------------------------------------------------------------


_EXHAUSTION_MARKERS = ("max_shard_restarts", "liveness", "wedge", "silent", "timeout")
"""Diagnostics that tell an operator *why* the budget was spent, not just that
it was."""


def test_restarts_past_the_budget_raise_with_diagnostics(
    exhausted_restarts: Failure,
) -> None:
    """A permanently wedging model ends the run instead of restarting forever.

    ``max_shard_restarts=1`` against a factory that hangs on every incarnation:
    the gate fires, the shard restarts once, the replacement wedges too, and
    there is no budget left.  The message has to carry both halves -- that the
    restart budget is what ended the run, and what kept triggering it -- because
    the two point at completely different fixes (raise the budget vs. the model
    never answers).
    """
    assert isinstance(exhausted_restarts.error, Exception), (
        f"run_selfplay raised {exhausted_restarts.error!r}, want a normal exception"
    )
    message = exhausted_restarts.message
    assert "restart" in message, (
        f"the failure message never mentions restarts: {exhausted_restarts.error}"
    )
    assert any(marker in message for marker in _EXHAUSTION_MARKERS), (
        f"the failure message carries no diagnostic about what kept wedging: "
        f"{exhausted_restarts.error}"
    )


def test_restart_exhaustion_is_bounded_by_the_budget_and_the_gate(
    exhausted_restarts: Failure,
) -> None:
    """Two wedges at a ten-second gate, not an unbounded retry loop.

    The ceiling is deliberately loose (a spawned child imports torch from
    scratch twice over here); what it excludes is a third incarnation, which is
    what ``max_shard_restarts`` exists to prevent and what an off-by-one in the
    budget would produce.
    """
    assert exhausted_restarts.elapsed < rig.LOUD_WALL_S * 0.8, (
        f"restart exhaustion took {exhausted_restarts.elapsed:.1f}s with a "
        f"{_LIVENESS_TIMEOUT_S}s liveness gate and a budget of one restart"
    )


def test_restart_exhaustion_leaves_no_segments_and_no_children(
    exhausted_restarts: Failure,
) -> None:
    """The hardest cleanup case: two shards' worth of segments, both wedged.

    Every process involved has to be SIGKILLed and joined, and every segment
    either incarnation created has to be unlinked, on a path that ends in an
    exception.  This is the case where a ``finally`` that only covers the happy
    path leaks two segments per worker and leaves a hung server behind.
    """
    assert not exhausted_restarts.shm_leaked, (
        f"segments still in /dev/shm: {sorted(exhausted_restarts.shm_leaked)}"
    )
    assert not exhausted_restarts.children_leaked, (
        f"child processes surviving the failure: "
        f"{exhausted_restarts.children_leaked}"
    )


# ---------------------------------------------------------------------------
# (d) the budget runs out on the STARTUP-TIMEOUT branch (KOM-053)
# ---------------------------------------------------------------------------


def test_a_never_started_shard_spends_the_budget_on_the_startup_gate(
    never_started: Failure,
) -> None:
    """The run ends on the never-started branch, and says which branch it was.

    ``exhausted_restarts`` reaches the same ending through a hung *model*; this
    one reaches it through a worker that never produces a first shard at all,
    and the two want different operator actions (the model never answers vs.
    the worker never got going).  So the diagnostic has to name the branch, not
    merely the budget -- and the wording it names is the gate's own
    ``it never got started``, which is what ``first_shard_grace_s`` documents.
    """
    assert isinstance(never_started.error, Exception), (
        f"run_selfplay raised {never_started.error!r}, want a normal exception"
    )
    message = never_started.message
    assert "max_shard_restarts" in message, (
        f"the failure message never names the budget that ended the run: "
        f"{never_started.error}"
    )
    assert "never got started" in message, (
        f"the run ended without saying it was the startup grace that fired -- "
        f"an operator cannot tell this from a wedged model: {never_started.error}"
    )


def test_a_never_started_shard_leaves_no_segments_and_no_children(
    never_started: Failure,
) -> None:
    """The KOM-053 regression: this branch owes the same clean ``/dev/shm``.

    Segments are created in ``_launch`` *before* anything is spawned, so an
    incarnation whose workers never ran still owns a full set of them -- one
    per lane, twice over here, since the budget buys two incarnations.  Design
    D2/F9 puts their unlink in a ``finally``; a run that ends on this branch is
    one of the paths that ``finally`` exists for, and the failure this test was
    written against carried two live ``psm_*`` segments out of it.
    """
    assert never_started.shm_leaked == set(), (
        f"segments still in /dev/shm after a never-started shard exhausted its "
        f"restart budget: {sorted(never_started.shm_leaked)}"
    )
    assert never_started.children_leaked == {}, (
        f"child processes surviving the failure: {never_started.children_leaked}"
    )


# ---------------------------------------------------------------------------
# (e) an incarnation that never reached `started` (KOM-053)
# ---------------------------------------------------------------------------


def test_a_half_launched_incarnation_releases_every_segment(
    half_launched: Failure,
) -> None:
    """One segment's release raising must not strand the segments after it.

    ``_teardown`` has swept per-item since the wave-9c review (F8), but the
    other owner of segments -- ``_launch``'s failure path, for an incarnation
    that never reached ``started`` and therefore never gets a ``_teardown`` --
    swept in a bare loop.  With two segments and a first release that raises,
    that leaks the second one into ``/dev/shm``, and the exception the caller
    sees looks like an ordinary launch failure, so nothing says so.
    """
    assert half_launched.shm_leaked == set(), (
        f"segments still in /dev/shm after a half-launched incarnation: "
        f"{sorted(half_launched.shm_leaked)}"
    )
    assert half_launched.children_leaked == {}, (
        f"child processes surviving the failed launch: "
        f"{half_launched.children_leaked}"
    )


def test_a_half_launched_incarnation_still_raises(half_launched: Failure) -> None:
    """It fails loudly rather than silently returning a summary of nothing.

    The sweep re-raises the first error it hit rather than swallowing it, so a
    cleanup that had to work around a broken release is still an exception at
    the caller -- which is the whole contract of this module.
    """
    assert isinstance(half_launched.error, Exception), (
        f"run_selfplay raised {half_launched.error!r}, want a normal exception"
    )
    assert half_launched.elapsed < rig.LOUD_WALL_S / 2.0, (
        f"took {half_launched.elapsed:.1f}s to fail a launch that cannot spawn "
        f"its first worker"
    )
