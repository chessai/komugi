"""A wedged shard is detected, killed, and restarted -- deterministically.

Design D10 gives the orchestrator a liveness gate: a worker that has gone silent
for longer than ``liveness_timeout_s`` is SIGKILLed, joined *before* anything is
respawned (a zombie would collide with its successor on ``ShardWriter``'s
single-writer guard), and -- in P0a, per design D2 -- the whole shard restarts:
fresh segments, ``incarnation + 1``, fresh ``worker_<k>_r<i>`` directories,
because a running server cannot re-attach a fresh segment.

Making that testable without a race
-----------------------------------
A wedge produced by "sleep for a while and hope the gate fires first" is a
timing test wearing a behaviour test's clothes.  So the wedge here is decided by
a **counter file**, not by a clock:

* the test writes an environment variable naming the counter file, and spawned
  children inherit the environment;
* :func:`orchestrate_rig.wedge_once_factory` -- module-level, as design D2
  requires of every model factory -- increments that counter each time it is
  called and returns a model whose forward **sleeps forever** iff the value it
  read was ``0``;
* the factory is called once per server, inside the server process, so
  incarnation 0's server hangs every evaluation and incarnation 1's does not.

Incarnation 0's workers therefore starve: they ring their doorbells and no
response is ever written, they produce no rows, they go silent.  Only the
liveness gate can notice, and the run can only complete if it restarts the
shard.  ``request_timeout_s`` is left far above ``liveness_timeout_s`` on
purpose (see :func:`orchestrate_rig.inference_config`): a worker-side
``InferenceTimeout`` would *also* end in a whole-shard restart, and a test that
cannot tell the two mechanisms apart pins neither.

On the size of ``liveness_timeout_s``
-------------------------------------
It is small -- the wedge must be declared in seconds, and the whole test is
walled -- but it is **not** the ~5s a spec sketch would suggest.  A spawned
child imports torch from scratch and then plays a whole game before it seals its
first shard, which is the mtime evidence design D10's gate reads; a five-second
gate would race the *healthy* incarnation and shoot it too, turning this test
into a flake that fails for the opposite of the reason it exists.  Thirty
seconds is comfortably above a cold start plus a tiny-budget game and
comfortably below the wall.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_RUN_SEED = 90909
_LIVENESS_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Run:
    """The wedged-then-restarted run, and the leftovers it did or did not leave."""

    summary: Any
    root: Path
    elapsed: float
    constructions: int
    shm_leaked: set[str]
    children_leaked: dict[int, str]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One run whose first incarnation cannot answer a single evaluation."""
    base = Path(tmp_path_factory.mktemp("wedge"))
    root = base / "shards"
    root.mkdir()
    counter = base / "constructions"

    shm_before = rig.shm_snapshot()
    children_before = set(rig.child_pids())

    with pytest.MonkeyPatch.context() as patch:
        # Set BEFORE run_selfplay, so every process the orchestrator spawns
        # inherits it -- the server is the process that actually reads it.
        patch.setenv(rig.COUNTER_ENV, str(counter))
        outcome = rig.walled_run_selfplay(
            rig.wedge_once_factory,
            rig.orchestrator_config(
                root,
                run_seed=_RUN_SEED,
                workers=1,
                games_per_worker=1,
                liveness_timeout_s=_LIVENESS_TIMEOUT_S,
                # The canonical wedge leaves the shard dir EMPTY forever, so
                # the never-started grace -- its own knob since the wave-9c
                # review split it out of drain_timeout_s (F3) -- is the gate
                # that must fire. Kept equal to the liveness timeout.
                first_shard_grace_s=_LIVENESS_TIMEOUT_S,
                max_shard_restarts=2,
            ),
            rig.tiny_selfplay_config(workers=1),
            # request_timeout_s far above the WALL, not merely above the
            # liveness timeout: the wave-9c review (F3) showed 240s could
            # still fire first via the worker's own InferenceTimeout, making
            # the test green through the wrong mechanism. At 3600s the
            # liveness gate is the only detector that can act within the wall.
            rig.inference_config(max_batch=1, request_timeout_s=3600.0),
            wall_s=rig.RESTART_WALL_S,
            label="wedge",
        )
    summary = outcome.unwrap()

    leaked_shm = rig.new_shm_survivors(shm_before)
    leaked_children = {
        pid: name
        for pid, name in rig.child_pids().items()
        if pid not in children_before
    }
    rig.kill_stragglers()
    return Run(
        summary=summary,
        root=root,
        elapsed=outcome.elapsed,
        constructions=rig.read_counter(counter),
        shm_leaked=leaked_shm,
        children_leaked=leaked_children,
    )


def test_the_wedge_was_real_and_happened_exactly_once(run: Run) -> None:
    """Two model constructions: the hung incarnation and its replacement.

    This is the premise every other assertion rests on.  One construction would
    mean the run never restarted (so nothing was wedged); three would mean the
    gate fired on a *healthy* incarnation as well, which is the flake this
    test's timeout sizing is chosen to avoid.
    """
    assert run.constructions == 2, (
        f"the wedge factory built {run.constructions} model(s); want exactly 2 "
        f"(incarnation 0 hangs, incarnation 1 behaves)"
    )


def test_exactly_one_whole_shard_restart_was_recorded(run: Run) -> None:
    """``restarts == 1`` and every worker finished on incarnation 1.

    Design D2's P0a rule is that a worker death or wedge restarts the WHOLE
    shard, not the worker: a running server cannot re-attach a fresh segment, so
    there is no partial recovery to record.  With one worker configured the two
    readings coincide, and the incarnation is what distinguishes them from a
    restart that was never counted.
    """
    assert run.summary.restarts == 1, (
        f"one wedge should be one whole-shard restart, summary says "
        f"{run.summary.restarts}"
    )
    incarnations = rig.summary_incarnations(run.summary)
    assert incarnations == [1], (
        f"worker_incarnations {incarnations}, want every worker on its second "
        f"incarnation after exactly one restart"
    )


def test_the_run_finished_its_games_after_the_restart(run: Run) -> None:
    """The point of restarting: the work still gets done."""
    assert run.summary.games_completed >= 1, (
        "the restarted incarnation played no game, so the restart bought nothing"
    )
    assert run.summary.rows_written > 0


def test_the_completed_shard_directories_are_the_second_incarnations(
    run: Run,
) -> None:
    """Design D10: a respawn writes to a FRESH ``worker_<k>_r1`` directory."""
    reported = rig.summary_dirs(run.summary)
    assert reported, "the summary names no shard directory"
    for path in reported:
        worker, incarnation = rig.parse_shard_dir(path)
        assert incarnation == 1, (
            f"summary names {path.name!r} (worker {worker}, incarnation "
            f"{incarnation}); the completed data lives in the _r1 directory"
        )
        assert rig.sealed_shards(path), f"{path.name} sealed no shard"


def test_the_wedged_incarnations_directory_contributes_nothing(run: Run) -> None:
    """``*_r0`` holds no sealed shard and is absent from the summary.

    Pinned as a conjunction because in *this* scenario both halves are
    determined rather than a matter of policy: incarnation 0's server never
    answered an evaluation, so its worker never finished a game, so it can never
    have sealed a shard -- rows are appended at game end (design D10's per-game
    flush).  Whether the orchestrator deletes the empty directory or leaves it
    as evidence is its call; what may not happen is a half-written first
    incarnation being reported as part of the run's data.
    """
    on_disk = rig.shard_dirs_on_disk(run.root)
    reported = {path.resolve() for path in rig.summary_dirs(run.summary)}
    for path in on_disk:
        _, incarnation = rig.parse_shard_dir(path)
        if incarnation != 0:
            continue
        assert not rig.sealed_shards(path), (
            f"{path.name} sealed a shard, but its server never answered an "
            f"evaluation -- no game can have finished there"
        )
        assert path.resolve() not in reported, (
            f"the summary reports the wedged incarnation's directory {path.name}"
        )


def test_the_detection_and_restart_stayed_inside_a_bounded_wall(run: Run) -> None:
    """A liveness gate that never fires is indistinguishable from a hang.

    The wall in :func:`orchestrate_rig.walled` already turns a non-firing gate
    into a bounded failure; this asserts the *positive* bound, so a gate that
    fires only after some unrelated multi-minute timeout expires is a failure
    too rather than a slow pass.
    """
    ceiling = _LIVENESS_TIMEOUT_S + rig.RESTART_WALL_S / 2.0
    assert run.elapsed < ceiling, (
        f"the run took {run.elapsed:.1f}s with a {_LIVENESS_TIMEOUT_S}s liveness "
        f"gate; the wedge was not what ended incarnation 0"
    )


def test_no_child_process_of_ours_survived_the_restart(run: Run) -> None:
    """Design D10's join-before-respawn, observed from outside.

    The children are counted by scanning ``/proc`` for processes whose parent is
    this interpreter (``psutil`` is not in the dev shell), snapshotted before the
    run so that nothing another test owns can be blamed on this one.  The scan
    runs *before* anything in this suite calls
    ``multiprocessing.active_children``, which would reap an unjoined corpse as a
    side effect and hide exactly the failure being looked for: a ``Z``-state
    child here is a SIGKILLed worker the orchestrator never joined.
    """
    assert not run.children_leaked, (
        f"child processes surviving the run: {run.children_leaked}"
    )


def test_no_shared_memory_segment_survived_the_restart(run: Run) -> None:
    """Both incarnations' segments are unlinked (design D2/F9).

    A restart is the case where leaking is easiest: the first incarnation's
    segments belong to a shard that is being torn down mid-flight, and the
    ``finally`` that unlinks them is the only backstop Python 3.12 offers
    (no ``SharedMemory(track=...)``, bpo-38119).
    """
    assert not run.shm_leaked, (
        f"segments still in /dev/shm after the restart: {sorted(run.shm_leaked)}"
    )


def test_a_worker_that_wrote_then_wedged_is_caught_by_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mtime-staleness branch of the liveness gate, end to end.

    Every other wedge in this module leaves the shard directory empty, so it
    exercises only the never-started grace; the wave-9c review (F3) found the
    ``log_freshness`` branch -- the actual D10 gate -- reached by no process-
    level test. Here the worker seals real games first, then the server stalls
    (a sentinel the TEST creates only after observing a shard file on disk),
    the directory's mtime goes stale, and freshness is the only detector that
    can fire: the never-started grace is disabled (1h) and so is the worker's
    own InferenceTimeout (1h).
    """
    root = tmp_path / "shards"
    root.mkdir()
    counter = tmp_path / "constructions"
    sentinel = tmp_path / "stall"
    monkeypatch.setenv(rig.COUNTER_ENV, str(counter))
    monkeypatch.setenv(rig.SENTINEL_ENV, str(sentinel))

    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["summary"] = sp.run_selfplay(
                rig.stall_on_sentinel_factory,
                rig.orchestrator_config(
                    root,
                    run_seed=_RUN_SEED + 1,
                    workers=1,
                    games_per_worker=3,
                    liveness_timeout_s=15.0,
                    first_shard_grace_s=3600.0,
                    max_shard_restarts=2,
                ),
                rig.tiny_selfplay_config(workers=1),
                rig.inference_config(max_batch=1, request_timeout_s=3600.0),
            )
        except BaseException as exc:  # noqa: BLE001 -- reported by the test body
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + rig.RESTART_WALL_S
        sealed = False
        while time.monotonic() < deadline:
            if any(entry.is_file() for entry in root.rglob("*")):
                sealed = True
                break
            time.sleep(0.2)
        assert sealed, "no shard file ever appeared; cannot stage the stall"
        sentinel.write_text("stall")
        thread.join(rig.RESTART_WALL_S)
    finally:
        rig.kill_stragglers()

    assert not thread.is_alive(), "run_selfplay outlived its wall after the stall"
    assert "error" not in box, f"run raised: {box.get('error')!r}"
    summary = box["summary"]
    assert summary.restarts == 1, (
        f"expected exactly one freshness-triggered restart, got "
        f"{summary.restarts} (dirs: {summary.shard_dirs})"
    )
    # Incarnation 0 sealed real games before the stall, so its directory stays
    # in the summary (the F2 adjudication's inclusive half); incarnation 1
    # completed the budget.
    incarnations = {Path(d).name.rsplit("_r", 1)[1] for d in summary.shard_dirs}
    assert incarnations == {"0", "1"}, summary.shard_dirs
    assert rig.read_counter(counter) == 2, "restarted server must be the plain net"
