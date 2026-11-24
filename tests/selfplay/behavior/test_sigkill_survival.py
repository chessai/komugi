"""SIGKILL one worker mid-traffic; the server and the other workers live on.

Design F9's attack, restated as a test.  Python 3.12 has no
``SharedMemory(track=...)`` (that lands in 3.13), so every process that
*attaches* a segment registers it with its ``resource_tracker`` unless the
implementation explicitly unregisters -- bpo-38119.  The design's answer is
that the orchestrator creates and unlinks every segment and everyone else
attaches and immediately unregisters.  Two ways to get that wrong, both
invisible until a worker dies hard:

* a worker (or the server) unlinks a segment it merely attached, or leaves it
  registered with a tracker that outlives it -- and the *other* workers' slots
  vanish from ``/dev/shm`` underneath a running run;
* the server treats a client that stopped answering as a fatal condition and
  tears the whole shard down, taking the healthy workers with it.

SIGKILL specifically, not ``terminate()``: SIGTERM runs the child's atexit and
finalizer machinery, which is the *forgiving* path.  SIGKILL leaves shared
memory and every synchronisation primitive in whatever state they were in
mid-request, which is what actually happens when the liveness gate (design
D10) kills a wedged worker, or when the OOM killer picks one.

With the lock-free doorbell protocol (KOM-021b) that is now a *deterministic*
property rather than a lucky one: a worker SIGKILLed at ANY instruction cannot
be holding a lock anyone else needs, because the worker side of the seam takes
no lock at all -- it writes its slot, stores an aligned 8-byte ``request_seq``,
and polls ``response_seq``.  That is the property this test pins.  Under the
previous shared ``mp.Queue`` plus per-worker ``mp.Event`` signalling it did not
hold: a victim killed inside the queue feeder's write lock, or inside
``Event.clear()``, orphaned a semaphore the SURVIVORS then blocked on until
``InferenceTimeout`` -- which is exactly how this test was observed to fail
intermittently.

The survivors then have to complete at least twenty further round trips each
-- enough that "it happened to have one answer already in flight" cannot pass
-- and every segment, the victim's included, must still be attachable
afterwards.  The victim's is the strict form of the F9 pin: nobody but the
orchestrator may unlink, so a dead worker's segment is still there until the
``finally`` in :func:`inference_rig.slots` removes it.
"""

from __future__ import annotations

import os
import signal
import time
from typing import Any

import pytest

import inference_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "ShardServer"):
    pytest.skip(
        "komugi.selfplay exports no ShardServer yet (KOM-021a)",
        allow_module_level=True,
    )

_WORKERS = 3
_VICTIM = 0
_WARMUP_ROUNDS = 3
_POST_KILL_ROUNDS = 25
_REQUIRED_POST_KILL_ROUNDS = 20

#: Short enough that a survivor blocked on a dead server fails the test in
#: seconds rather than at the join wall, long enough not to fire under load.
_REQUEST_TIMEOUT_S = 20.0

_WARM_WALL_S = 180.0


def _worker_main(
    config: Any,
    handle: Any,
    warm: Any,
    go_event: Any,
    errors: Any,
    dones: Any,
    victim: bool,
) -> None:
    """Warm up, announce, then either spin forever (victim) or measure."""
    worker_id = handle.worker_id
    code = worker_id + 1
    try:
        client = rig.make_client(config, handle)
        obs = rig.encode_obs(code)
        legal = rig.all_legal()

        def round_trip() -> str | None:
            return rig.check_response(client.submit(obs, legal), code)

        for _ in range(_WARMUP_ROUNDS):
            problem = round_trip()
            if problem is not None:
                errors.put(f"worker {worker_id} warmup: {problem}")
                return
        warm.put(worker_id)

        if victim:
            # Keep requests in flight so the kill lands mid-transaction: the
            # interesting states are "doorbell rung, slot half-read by the
            # server" and "server writing a response nobody will ever read".
            # The 2ms pause keeps this from monopolising the batch window, and
            # polling ``response_seq`` is still "mid-request", so the attack
            # keeps its teeth.  Unlike the queue-based rev 1, there is now no
            # instant in this loop at which the victim holds a lock a survivor
            # could block on -- which is the point of the whole test.
            while True:
                client.submit(obs, legal)
                time.sleep(0.002)

        if not go_event.wait(_WARM_WALL_S):
            errors.put(f"worker {worker_id} never saw the post-kill go signal")
            return
        done = 0
        for _ in range(_POST_KILL_ROUNDS):
            problem = round_trip()
            if problem is not None:
                errors.put(f"worker {worker_id} post-kill round {done}: {problem}")
                return
            done += 1
        dones.put((worker_id, done))
    except BaseException as exc:
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def test_server_survives_a_sigkilled_worker() -> None:
    """One of three workers dies hard; the run continues without it."""
    config = rig.make_config(max_batch=16, request_timeout_s=_REQUEST_TIMEOUT_S)
    worker_ids = list(range(_WORKERS))

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        go_event = rig.CTX.Event()
        warm = rig.CTX.Queue()
        errors = rig.CTX.Queue()
        dones = rig.CTX.Queue()

        server = rig.spawn(
            rig.server_main,
            rig.encoder_factory,
            config,
            handles,
            stop_event,
            name="shard-server",
        )
        workers = [
            rig.spawn(
                _worker_main,
                config,
                handle,
                warm,
                go_event,
                errors,
                dones,
                handle.worker_id == _VICTIM,
                name=f"worker-{handle.worker_id}",
            )
            for handle in handles
        ]
        victim = workers[_VICTIM]
        survivors = [proc for i, proc in enumerate(workers) if i != _VICTIM]

        try:
            warmed = sorted(rig.collect(warm, _WORKERS, _WARM_WALL_S))
            if warmed != worker_ids:
                rig.join_all(workers + [server], rig.JOIN_GRACE_S)
                pytest.fail(
                    f"only workers {warmed} reached the traffic phase "
                    f"(errors: {rig.drain(errors)}) -- the kill never happened, "
                    f"so nothing below would have been tested"
                )

            victim_pid = victim.pid
            assert victim_pid is not None
            if not victim.is_alive():
                pytest.fail(
                    f"the victim exited on its own ({victim.exitcode}) before it "
                    f"could be killed (errors: {rig.drain(errors)})"
                )
            os.kill(victim_pid, signal.SIGKILL)
            victim.join(rig.JOIN_GRACE_S)
            victim_exit = victim.exitcode
            go_event.set()

            failures = rig.join_all(survivors, rig.SPAWN_WALL_S, errors)
            # collect, not drain: a process is joinable the instant its queue
            # feeder has flushed, which is not quite the instant the reader can
            # see the bytes.  The wall only costs anything when a report is
            # genuinely missing.
            reported = dict(rig.collect(dones, len(survivors), rig.JOIN_GRACE_S))
            server_alive = server.is_alive()

            # Read every segment back through the implementation's own
            # untracked-attach helper: if the dying worker's resource_tracker
            # unlinked anything, this is where it shows up as FileNotFoundError.
            missing: list[str] = []
            for handle in handles:
                try:
                    segment = sp.attach_untracked(handle.shm_name)
                except FileNotFoundError:
                    missing.append(handle.shm_name)
                else:
                    assert len(segment.buf) >= sp.slot_nbytes(config)
                    segment.close()
        finally:
            failures_stop = rig.stop_server(server, stop_event)
            rig.join_all(workers, rig.JOIN_GRACE_S)

    assert victim_exit == -signal.SIGKILL, (
        f"the victim exited {victim_exit}, not by SIGKILL -- the scenario did "
        f"not run"
    )
    assert missing == [], (
        f"segments unlinked by a dying process (bpo-38119 / design F9): "
        f"{missing}"
    )
    assert failures == [], failures
    assert server_alive, "the server died with the worker it was serving"
    assert failures_stop == [], failures_stop

    assert sorted(reported) == [wid for wid in worker_ids if wid != _VICTIM]
    for worker_id, rounds in reported.items():
        assert rounds >= _REQUIRED_POST_KILL_ROUNDS, (
            f"worker {worker_id} completed only {rounds} round trips after the "
            f"kill, want >= {_REQUIRED_POST_KILL_ROUNDS}"
        )
