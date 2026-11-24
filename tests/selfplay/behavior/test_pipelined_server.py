"""The pipelined shard server, tested through processes rather than mocks.

KOM-032 replaces ``ShardServer.serve``'s scan/forward/scatter loop with a
pipelined one: the collect for batch N+1 runs *while* forward N is in flight,
behind an :class:`AsyncEvaluator` seam whose ``launch(obs, legal)`` returns a
``ForwardHandle`` (``query() -> bool``, ``result() -> NetEval``).  Dispatch is
unconditional -- ``serve`` runs the pipelined loop in production, and
``serve_once`` survives only as the serial reference -- so every existing
behaviour suite already covers the new path.  What none of them cover is what
the *pipelining itself* introduced, and that is this file:

**I1, the in-flight exclusion set (design review D1, BLOCKER) -- gated by the
unit suite, not by this file.**  Batch N's workers keep ``request_seq >
response_seq`` for the entire duration of forward N, because ``response_seq``
is published only at scatter.  A collect that reads nothing but that
comparison re-admits every one of them into batch N+1.  The deterministic
held-open-mock regression in ``test_pipelined_unit.py`` is the I1 gate.  This
file's soak does NOT reliably detect an exclusion-free collect on the CPU
hosts that run the merge gate: the CPU ``launch`` returns an already-ready
handle, so the in-flight window is effectively empty, and the union review ran
the exclusion-deleted mutant through the soak four times -- it passed every
time (review F1).  What the soak genuinely pins is exactly-once accounting and
answer *routing* under real multi-process concurrency (it kills the
scatter-row-swap mutant); on a GPU host, where forwards are multi-millisecond,
its pressure against torn reads/writes becomes real -- a bonus there, never
the CI claim.

**Drain-on-stop (design review D2/D5, shape 2).**  There are two
unanswered-at-exit windows, not one: the launched-not-scattered batch and the
collected-not-launched pending set.  A ``serve`` that returns without answering
both leaves a worker polling ``response_seq`` for the whole
``request_timeout_s`` (60 s in production) and then dying of
:class:`InferenceTimeout`.  Nothing in the existing suite catches it, because
every other rig stops the server only after its workers have finished
submitting.  :func:`test_stop_mid_submit_drains` stops it in the middle
instead.

**I6, bit-identity on CPU.**  The pipeline changes *when* a forward is
launched, never what it computes.  :func:`test_pipelined_matches_serial_corpus`
pins that at the byte level against direct ``serve_once`` passes.  Note the
scope carefully: batch *composition* is pinned equal (``max_batch=1`` on both
sides) because two different compositions may legitimately differ in the last
ulp -- the KOM-104 GEMM lesson, which is why ``test_equivalence_corpus`` gates
cross-composition agreement at 1e-5 rather than at equality.  Bit-identity is
the right claim only where composition is held fixed, and that is the claim
made here.

**Non-regression, not a benchmark.**
:func:`test_throughput_smoke_is_not_worse` runs one workload through a spawned
pipelined ``serve`` and the same workload through a spawned serial
``serve_once`` loop and allows the pipelined leg 1.5x the serial wall.  That
bound is deliberately far too loose to measure a win with; it exists to catch a
pipeline that accidentally *serialises* -- an extra sleep in the collect, a
handle that is waited on before the next collect starts.  The real numbers come
from the advisor's bench on the GPU box, where the overlap this code exists for
is observable at all; on CPU the forward is synchronous and the pipeline
degenerates by design.

Suite discipline, inherited from :mod:`inference_rig` and not re-derived here:
spawn context only, every spawned process joined inside an explicit wall (a
hang fails in bounded time instead of wedging ``ci/check.sh``), tiny stub nets
on CPU, and the module under test reached lazily so this file stays green on a
tree where KOM-032 has not landed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inference_rig as rig

inf = pytest.importorskip("komugi.selfplay.inference")

if not hasattr(inf, "AsyncEvaluator"):
    pytest.skip(
        "komugi.selfplay.inference exports no AsyncEvaluator yet (KOM-032): the "
        "pipelined loop and its ForwardHandle seam are not in this tree, so "
        "every assertion below would be about a server that does not exist",
        allow_module_level=True,
    )


def _read_counters(config: Any, handles: Any) -> dict[int, tuple[int, int]]:
    """Final ``(request_seq, response_seq)`` per worker, straight from /dev/shm.

    The doorbell counters are the server's ledger, readable by anyone holding
    the segment name: ``request_seq`` is the number of round trips a worker
    *started*, ``response_seq`` is the ticket the server last published back.
    Reading them in the parent after the workers have exited turns "every
    request was answered exactly once" into an arithmetic check that does not
    depend on any worker's report.
    """
    counters: dict[int, tuple[int, int]] = {}
    for handle in handles:
        entry = inf.attach_slots(handle.shm_name, config)
        try:
            counters[handle.worker_id] = (
                int(entry.request_seq[0]),
                int(entry.response_seq[0]),
            )
        finally:
            entry.close()
    return counters


# ---------------------------------------------------------------------------
# (1) concurrency soak
# ---------------------------------------------------------------------------

_SOAK_WORKERS = 8
_SOAK_PER_WORKER = 250

#: Codes are ``worker_id * stride + k + 1``, so no two submissions anywhere in
#: the run share one and none of them is 0 (a zero code is indistinguishable
#: from an unwritten slot).  The stride exceeds the per-worker count, so the
#: ranges cannot overlap however the run is scheduled.  The largest code here
#: (2350) puts ``10 * code * OBS_CELLS`` at 7.6e6, comfortably below 2**24 and
#: therefore an exactly representable float32 integer -- which is what lets
#: ``inference_rig.check_response`` decode with a few-ulp tolerance instead of
#: a real one.
_SOAK_CODE_STRIDE = 300

#: Well above any plausible per-request latency at this batch size, and small
#: enough that a wedged worker fails the test in half a minute rather than at
#: the spawn wall.
_SOAK_REQUEST_TIMEOUT_S = 30.0


def _soak_codes(worker_id: int) -> list[int]:
    """This worker's payload tags, distinct across the whole run."""
    return [worker_id * _SOAK_CODE_STRIDE + k + 1 for k in range(_SOAK_PER_WORKER)]


def _soak_worker_main(
    config: Any,
    handle: Any,
    barrier: Any,
    errors: Any,
    dones: Any,
) -> None:
    """Submit a few hundred distinct tags; check every answer decodes to mine."""
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        legal = rig.all_legal()
        # Co-arrival: without it the workers trickle in and the server never
        # has a reason to hold a batch in flight while collecting the next.
        barrier.wait(rig.BARRIER_WALL_S)
        done = 0
        for code in _soak_codes(worker_id):
            problem = rig.check_response(
                client.submit(rig.encode_obs(code), legal), code
            )
            if problem is not None:
                errors.put(
                    f"worker {worker_id} request {done} (code {code}): {problem}"
                )
                return
            done += 1
        dones.put((worker_id, done))
    except BaseException as exc:  # reported, not raised: the parent asserts
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def test_concurrent_soak_answers_every_request_exactly_once() -> None:
    """Eight workers, two thousand tagged requests, no crossed or double wires.

    The process-level pin of exactly-once accounting and answer ROUTING under
    genuine concurrency: every answer decodes back to the tag that asked for
    it (a scatter that hands worker A worker B's row -- the row-swap mutant --
    fails here), and the counter ledger closes the count: a server that
    quietly stopped answering, or published a ticket never asked for, shows up
    as ``request_seq != response_seq`` even if no single response was
    malformed.

    What this soak does NOT claim (union review F1): detecting an
    exclusion-free in-flight collect (I1) on the CPU gate host -- there the
    handle is ready before the collect ever runs, the exclusion-deleted mutant
    passed this soak repeatedly, and I1's gate is the deterministic mock-handle
    regression in the unit suite.  See the module docstring.
    """
    config = rig.make_config(
        max_batch=16, batch_timeout_ms=0.5, request_timeout_s=_SOAK_REQUEST_TIMEOUT_S
    )
    worker_ids = list(range(_SOAK_WORKERS))

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        barrier = rig.CTX.Barrier(_SOAK_WORKERS)
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
                _soak_worker_main,
                config,
                handle,
                barrier,
                errors,
                dones,
                name=f"soak-{handle.worker_id}",
            )
            for handle in handles
        ]

        failures = rig.join_all(workers, rig.SPAWN_WALL_S, errors)
        # collect rather than drain: joinable and readable are not the same
        # instant for a queue the child wrote to just before exiting.
        reported = dict(rig.collect(dones, _SOAK_WORKERS, rig.JOIN_GRACE_S))
        server_alive = server.is_alive()
        # Every worker has exited, so nothing is writing these segments; the
        # server is still up but has nothing left to answer.
        counters = _read_counters(config, handles)
        failures.extend(rig.stop_server(server, stop_event))

    assert failures == [], failures
    assert server_alive, "the server died while its workers were still submitting"
    # Anti-vacuity: a server that answered nothing, or a worker that skipped its
    # loop, would otherwise leave a green with no round trips behind it.
    assert reported == {wid: _SOAK_PER_WORKER for wid in worker_ids}, (
        f"not every worker completed its {_SOAK_PER_WORKER} requests: {reported}"
    )
    want = {wid: (_SOAK_PER_WORKER, _SOAK_PER_WORKER) for wid in worker_ids}
    assert counters == want, (
        f"doorbell ledger disagrees with the workers' reports: {counters} != "
        f"{want} -- request_seq counts submits started and response_seq is the "
        f"last ticket published, so a mismatch means a request went unanswered "
        f"or an answer was published for a ticket nobody asked for"
    )


# ---------------------------------------------------------------------------
# (2) drain on stop
# ---------------------------------------------------------------------------

_DRAIN_WORKERS = 4
_DRAIN_WARMUP = 5

#: The failure this test exists for costs exactly one of these per wedged
#: worker, so it must be long enough to be unambiguous (no correct run comes
#: near it) and short enough that a red run is over in about a minute. 60 s
#: rather than 20: on the first full-suite gate run, 20 s of CPU starvation
#: under a saturated 32-thread box produced 4/4 false InferenceTimeouts
#: (green in isolation). The discriminator is drained-vs-dropped, which holds
#: at any timeout; the length only buys immunity to load.
_DRAIN_REQUEST_TIMEOUT_S = 60.0

#: How long after ``stop_event`` every worker must have finished its
#: outstanding submit and exited.  A quarter of the timeout: an omitted drain
#: takes the full timeout, a correct drain takes milliseconds plus process
#: teardown, and nothing legitimate lands in between.
_DRAIN_WALL_S = _DRAIN_REQUEST_TIMEOUT_S * 0.25

#: Joining generously *past* the timeout on purpose: a wedged worker must be
#: allowed to reach its InferenceTimeout and report it, because "worker 2
#: raised InferenceTimeout" is a far better failure message than "still alive
#: at the wall (killed)".
_DRAIN_JOIN_WALL_S = _DRAIN_REQUEST_TIMEOUT_S * 3.0

#: The submit loop's own escape hatch, in case ``stop_event`` never arrives.
_DRAIN_SUBMIT_WALL_S = 120.0

#: How long the parent watches the segments for the moment at which every
#: worker has an unanswered doorbell.  Each worker spends nearly all of its
#: cycle inside ``submit``, so this is normally satisfied on the first poll.
_DRAIN_OUTSTANDING_WALL_S = 15.0


def _drain_worker_main(
    config: Any,
    handle: Any,
    ready: Any,
    stop_event: Any,
    errors: Any,
    dones: Any,
) -> None:
    """Submit continuously until the shared stop event says to stop."""
    worker_id = handle.worker_id
    code = worker_id + 1
    try:
        client = rig.make_client(config, handle)
        obs = rig.encode_obs(code)
        legal = rig.all_legal()

        for _ in range(_DRAIN_WARMUP):
            problem = rig.check_response(client.submit(obs, legal), code)
            if problem is not None:
                errors.put(f"worker {worker_id} warmup: {problem}")
                return
        ready.put(worker_id)

        # The same event the server stops on, deliberately: a worker cannot be
        # told to stop *before* the server without the scenario evaporating
        # (nothing would be in flight at the stop), and it cannot be told after
        # without the server being gone before the worker's last submit.  The
        # residual race -- a doorbell rung in the microseconds after the check
        # and never scanned -- is what makes the review's "final serial pass"
        # drain shape the obviously-correct one; a drain that only scatters
        # what it already holds is a strictly narrower promise.
        submits = 0
        started = time.monotonic()
        while not stop_event.is_set():
            problem = rig.check_response(client.submit(obs, legal), code)
            if problem is not None:
                errors.put(f"worker {worker_id} submit {submits}: {problem}")
                return
            submits += 1
            if time.monotonic() - started > _DRAIN_SUBMIT_WALL_S:
                errors.put(
                    f"worker {worker_id} never saw stop_event after "
                    f"{_DRAIN_SUBMIT_WALL_S}s and {submits} submits"
                )
                return
        dones.put((worker_id, submits))
    except BaseException as exc:  # InferenceTimeout lands here -- that is the bug
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def _wait_all_outstanding(config: Any, handles: Any, wall_s: float) -> bool:
    """Watch the segments until every worker has an unanswered doorbell.

    Setting ``stop_event`` on a timer would leave it unknowable whether any
    request was actually in flight when the server saw it, which is the entire
    scenario.  Reading the counters instead makes the ordering explicit: when
    this returns True, every worker's ``request_seq`` exceeds its
    ``response_seq``, so every worker rang a doorbell strictly before the stop
    that follows.
    """
    entries = [inf.attach_slots(handle.shm_name, config) for handle in handles]
    try:
        deadline = time.monotonic() + wall_s
        while time.monotonic() < deadline:
            if all(
                int(entry.request_seq[0]) > int(entry.response_seq[0])
                for entry in entries
            ):
                return True
            time.sleep(0.0002)
        return False
    finally:
        for entry in entries:
            entry.close()


def test_stop_mid_submit_drains() -> None:
    """Stop the server mid-traffic: every outstanding submit still completes.

    Design review D5, shape 2.  The server is stopped while all four workers
    have a request in flight -- verified through the doorbell counters, not
    assumed from a sleep -- and the drain obligation is that ``serve`` answers
    both cohorts (the launched-but-not-scattered batch and the
    collected-but-not-launched pending set) before it returns.

    Omit the drain and every one of those workers polls ``response_seq`` until
    ``request_timeout_s``, then dies of :class:`InferenceTimeout`.  That is
    visible here three ways at once: as an error report from the worker, as the
    missing ``dones`` entry, and as a post-stop wall four times over budget.
    """
    config = rig.make_config(
        # max_batch >= the worker count on purpose: the drain contract covers
        # what the server has admitted, and a ceiling below the worker count
        # would leave requests legitimately uncollected at stop, testing the
        # rig's patience rather than the server's drain.
        max_batch=16,
        batch_timeout_ms=0.5,
        request_timeout_s=_DRAIN_REQUEST_TIMEOUT_S,
    )
    worker_ids = list(range(_DRAIN_WORKERS))

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        ready = rig.CTX.Queue()
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
                _drain_worker_main,
                config,
                handle,
                ready,
                stop_event,
                errors,
                dones,
                name=f"drain-{handle.worker_id}",
            )
            for handle in handles
        ]

        try:
            warmed = sorted(rig.collect(ready, _DRAIN_WORKERS, rig.SPAWN_WALL_S))
            if warmed != worker_ids:
                rig.join_all(workers, rig.JOIN_GRACE_S)
                pytest.fail(
                    f"only workers {warmed} reached the traffic phase "
                    f"(errors: {rig.drain(errors)}) -- nothing was in flight, so "
                    f"the stop below would have tested nothing"
                )
            in_flight = _wait_all_outstanding(
                config, handles, _DRAIN_OUTSTANDING_WALL_S
            )

            stopped_at = time.monotonic()
            stop_event.set()
            failures = rig.join_all(workers, _DRAIN_JOIN_WALL_S, errors)
            drain_wall = time.monotonic() - stopped_at
            reported = dict(rig.collect(dones, _DRAIN_WORKERS, rig.JOIN_GRACE_S))
        finally:
            failures_stop = rig.stop_server(server, stop_event)

    assert in_flight, (
        f"never observed all {_DRAIN_WORKERS} workers with an unanswered "
        f"doorbell within {_DRAIN_OUTSTANDING_WALL_S}s -- the scenario needs "
        f"requests in flight at the stop, so this run proved nothing"
    )
    assert failures == [], (
        f"{failures} -- an InferenceTimeout here is the omitted drain: the "
        f"server returned while these workers' tickets were unanswered "
        f"(design review D2/D5)"
    )
    assert sorted(reported) == worker_ids, (
        f"workers {sorted(reported)} finished their outstanding submit and "
        f"exited the loop; wanted {worker_ids}"
    )
    for worker_id, submits in reported.items():
        assert submits >= 1, f"worker {worker_id} completed no timed submits"
    assert drain_wall <= _DRAIN_WALL_S, (
        f"the last worker exited {drain_wall:.3f}s after stop_event, budget is "
        f"{_DRAIN_WALL_S:.3f}s -- a drained server answers in milliseconds, an "
        f"undrained one costs request_timeout_s "
        f"({_DRAIN_REQUEST_TIMEOUT_S}s) per wedged worker"
    )
    assert failures_stop == [], failures_stop


# ---------------------------------------------------------------------------
# (3) bit-identity against the serial reference
# ---------------------------------------------------------------------------

#: One row per corpus position: policy ``(N, A)``, wdl ``(N, 3)``, moves_left
#: ``(N,)``, all float32 and all compared byte for byte.
_Answers = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]

_CORPUS_SIZE = 64
_CORPUS_SEED = 20260823
_CORPUS_WORKERS = 2


def _corpus() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Deterministic ``(obs, legal)`` requests -- same in every process."""
    rng = np.random.default_rng(np.random.SeedSequence(_CORPUS_SEED))
    obs = rng.standard_normal(
        (_CORPUS_SIZE, rig.IN_PLANES, rig.BOARD, rig.BOARD)
    ).astype(np.float32)
    legal = rng.random((_CORPUS_SIZE, rig.NUM_ACTIONS)) < 0.35
    # A position with no legal move never crosses the seam, and it would let a
    # masking implementation return all -inf and "agree" with anything.
    legal[:, 0] = True
    return obs, legal.astype(np.bool_)


def _serve_once_reference(
    config: Any, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> _Answers:
    """The corpus through direct ``serve_once`` passes, in this process.

    ``serve_once`` is the serial reference the pipelined loop must reproduce,
    so the reference drives it exactly as the old ``serve`` did -- one attached
    slot, one doorbell rung by hand, one scan/forward/scatter step, one
    response read back -- rather than short-circuiting to
    :class:`InProcessEvaluator`.  What that buys is that the gather, the row
    packing and the scatter are on *both* sides of the comparison, so the only
    difference left between the two paths is the pipelining itself.
    """
    policies: list[npt.NDArray[np.float32]] = []
    wdls: list[npt.NDArray[np.float32]] = []
    moves_left: list[npt.NDArray[np.float32]] = []

    with rig.slots(config, [0]) as handles:
        entry = inf.attach_slots(handles[0].shm_name, config)
        try:
            evaluator = inf.InProcessEvaluator(rig.random_factory(), config)
            slots_map = {0: entry}
            for index in range(obs.shape[0]):
                inf.pack_request(entry, obs[index], legal[index])
                # Payload first, doorbell second -- the worker's half of the
                # protocol, open-coded because there is no worker here.
                entry.request_seq[0] = int(entry.request_seq[0]) + 1
                served = inf.serve_once(slots_map, evaluator, config)
                assert served == 1, f"serve_once served {served} rows, want 1"
                answer = inf.read_response(entry)
                policies.append(np.asarray(answer.policy_logits, dtype=np.float32))
                wdls.append(np.asarray(answer.wdl, dtype=np.float32))
                moves_left.append(
                    np.asarray(answer.moves_left, dtype=np.float32).reshape(-1)
                )
        finally:
            entry.close()

    return (
        np.stack(policies),
        np.stack(wdls),
        np.concatenate(moves_left),
    )


def _corpus_worker_main(
    config: Any,
    handle: Any,
    obs: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
    barrier: Any,
    errors: Any,
    answers: Any,
) -> None:
    """Submit the whole corpus through the served pipeline; ship the answers."""
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        policies: list[npt.NDArray[np.float32]] = []
        wdls: list[npt.NDArray[np.float32]] = []
        moves_left: list[npt.NDArray[np.float32]] = []
        barrier.wait(rig.BARRIER_WALL_S)
        for index in range(obs.shape[0]):
            response = client.submit(obs[index], legal[index])
            policies.append(np.asarray(response.policy_logits, dtype=np.float32))
            wdls.append(np.asarray(response.wdl, dtype=np.float32))
            moves_left.append(
                np.asarray(response.moves_left, dtype=np.float32).reshape(-1)
            )
        # Stacked, not per-row: one small message per worker keeps this well
        # inside the queue's pipe buffer, where a few hundred pickled arrays
        # would not be.
        answers.put(
            (
                worker_id,
                np.stack(policies),
                np.stack(wdls),
                np.concatenate(moves_left),
            )
        )
    except BaseException as exc:
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def _first_mismatch(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> int | None:
    """Index of the first row that differs by a single bit, or None."""
    if got.shape != want.shape:
        return 0
    differing = np.nonzero(np.any((got != want).reshape(got.shape[0], -1), axis=1))[0]
    return int(differing[0]) if differing.size else None


def test_pipelined_matches_serial_corpus() -> None:
    """Same requests, same weights, same bytes out -- invariant I6.

    Both paths run batch-of-one forwards (``max_batch=1``, and the reference's
    slot map holds a single slot), so batch composition is pinned equal on the
    two sides and equality is the honest assertion: the pipelined loop reorders
    *when* work is launched, never what the network is asked to compute.
    Cross-composition agreement is a different claim with a different tolerance
    and it already has its own gate in ``test_equivalence_corpus``.

    Two workers rather than one, so the served leg still exercises the
    pipeline's cross-worker routing while every forward stays a single row.
    """
    obs, legal = _corpus()
    config = rig.make_config(
        max_batch=1, batch_timeout_ms=0.5, request_timeout_s=60.0
    )
    reference = _serve_once_reference(config, obs, legal)

    worker_ids = list(range(_CORPUS_WORKERS))
    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        barrier = rig.CTX.Barrier(_CORPUS_WORKERS)
        errors = rig.CTX.Queue()
        answers = rig.CTX.Queue()

        server = rig.spawn(
            rig.server_main,
            rig.random_factory,
            config,
            handles,
            stop_event,
            name="shard-server",
        )
        workers = [
            rig.spawn(
                _corpus_worker_main,
                config,
                handle,
                obs,
                legal,
                barrier,
                errors,
                answers,
                name=f"corpus-{handle.worker_id}",
            )
            for handle in handles
        ]

        failures = rig.join_all(workers, rig.SPAWN_WALL_S, errors)
        # Joinable before readable, as everywhere else in this suite; the
        # stacked per-worker message is ~22 KB, comfortably inside the queue's
        # pipe buffer, so no child blocks at exit waiting for a reader.
        collected = rig.collect(answers, _CORPUS_WORKERS, rig.JOIN_GRACE_S)
        server_alive = server.is_alive()
        failures.extend(rig.stop_server(server, stop_event))

    assert failures == [], failures
    assert server_alive, "the server died before the corpus was answered"
    assert len(collected) == _CORPUS_WORKERS, (
        f"{len(collected)} of {_CORPUS_WORKERS} workers shipped their answers"
    )

    want_policy, want_wdl, want_mlh = reference
    # Anti-vacuity: a reference that answered zeros everywhere would make every
    # equality below true and mean nothing.
    assert float(np.max(np.abs(want_policy))) > 1e-3, (
        f"the serve_once reference policy logits are ~0 "
        f"({float(np.max(np.abs(want_policy))):g})"
    )

    for worker_id, got_policy, got_wdl, got_mlh in collected:
        for field, got, want in (
            ("policy_logits", got_policy, want_policy),
            ("wdl", got_wdl, want_wdl),
            ("moves_left", got_mlh, want_mlh),
        ):
            row = _first_mismatch(got, want)
            assert row is None, (
                f"worker {worker_id}: served {field} differs from the serve_once "
                f"reference at corpus row {row} (shapes {got.shape} vs "
                f"{want.shape}) -- the pipelined loop must be bit-identical to "
                f"the serial path at equal batch composition (invariant I6)"
            )


# ---------------------------------------------------------------------------
# (4) throughput smoke
# ---------------------------------------------------------------------------

_TIMED_WORKERS = 4
_TIMED_PER_WORKER = 250
_TIMED_WARMUP = 10

#: Generous by construction.  Anything tighter would be a benchmark run on
#: whatever core CI felt like giving us; anything looser would not notice a
#: pipeline that serialised itself.
_TIMED_RATIO = 1.5

#: How many times each leg is run.  The ratio is taken between the two legs'
#: *minima*, not between two single runs -- see ``_best_of_walls``.
_TIMED_REPEATS = 3

#: The serial reference loop must sleep exactly as the pre-KOM-032 ``serve``
#: did, or this compares the pipeline against a busy-spin instead of against
#: the code it replaced.  Read off the module rather than copied, so the
#: baseline cannot silently drift away from the idle poll the seam documents;
#: the literal is the value that constant has held since KOM-021b.
_SERIAL_IDLE_POLL_S = float(getattr(inf, "_SERVER_IDLE_POLL_S", 0.001))


def _serial_server_main(
    factory: Any,
    config: Any,
    handles: Any,
    stop_event: Any,
) -> None:
    """The pre-KOM-032 serial loop, kept here as the timing baseline.

    ``ShardServer.serve`` now dispatches the pipelined loop unconditionally
    (design review D4), which is the right production choice and leaves the
    serial arrangement with no entry point of its own.  This is that entry
    point, assembled from the seam's own public pieces -- attach, then
    ``serve_once`` until stopped, sleeping only when a scan found nothing.
    """
    evaluator = inf.InProcessEvaluator(factory(), config)
    slots_map = {
        handle.worker_id: inf.attach_slots(handle.shm_name, config)
        for handle in handles
    }
    try:
        while not stop_event.is_set():
            if inf.serve_once(slots_map, evaluator, config) == 0:
                time.sleep(_SERIAL_IDLE_POLL_S)
    finally:
        for entry in slots_map.values():
            entry.close()


def _timed_worker_main(
    config: Any,
    handle: Any,
    barrier: Any,
    errors: Any,
    dones: Any,
) -> None:
    """Warm up (and check correctness once), then time a fixed request count."""
    worker_id = handle.worker_id
    code = worker_id + 1
    try:
        client = rig.make_client(config, handle)
        obs = rig.encode_obs(code)
        legal = rig.all_legal()
        for _ in range(_TIMED_WARMUP):
            problem = rig.check_response(client.submit(obs, legal), code)
            if problem is not None:
                errors.put(f"worker {worker_id} warmup: {problem}")
                return
        # The barrier is inside the measured region's fence, not the process
        # spawn: torch imports and model construction are excluded from the
        # wall, so the two legs are compared on traffic alone.
        barrier.wait(rig.BARRIER_WALL_S)
        started = time.monotonic()
        for _ in range(_TIMED_PER_WORKER):
            # Deliberately unchecked: decoding costs the same on both legs, but
            # it is worker-side CPU that dilutes the thing being measured.
            client.submit(obs, legal)
        dones.put((worker_id, time.monotonic() - started))
    except BaseException as exc:
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def _timed_run(server_target: Any, config: Any) -> tuple[float, list[str]]:
    """Run the workload against `server_target`; return (wall, failures)."""
    worker_ids = list(range(_TIMED_WORKERS))
    failures: list[str] = []

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        barrier = rig.CTX.Barrier(_TIMED_WORKERS)
        errors = rig.CTX.Queue()
        dones = rig.CTX.Queue()

        server = rig.spawn(
            server_target,
            rig.encoder_factory,
            config,
            handles,
            stop_event,
            name="timing-server",
        )
        workers = [
            rig.spawn(
                _timed_worker_main,
                config,
                handle,
                barrier,
                errors,
                dones,
                name=f"timed-{handle.worker_id}",
            )
            for handle in handles
        ]

        failures.extend(rig.join_all(workers, rig.SPAWN_WALL_S, errors))
        reported = rig.collect(dones, _TIMED_WORKERS, rig.JOIN_GRACE_S)
        if not server.is_alive():
            failures.append("the server died before the workload finished")
        failures.extend(rig.stop_server(server, stop_event))

    if len(reported) != _TIMED_WORKERS:
        failures.append(
            f"only {len(reported)} of {_TIMED_WORKERS} workers reported a wall"
        )
        return float("inf"), failures
    # The workload's wall is the last worker to finish, not the mean: the
    # workers are released together and the server is the shared resource.
    return max(float(elapsed) for _, elapsed in reported), failures


def _best_of_walls(
    server_target: Any, config: Any, repeats: int
) -> tuple[float, list[str]]:
    """Run the workload `repeats` times; return (min wall, all failures).

    *Why the minimum.*  A measured wall is ``true_time + interference``, and
    interference from a shared box -- other tests, a gate, a self-play
    campaign -- is non-negative: contention can only ever add.  So the
    smallest wall over several repeats is the estimator closest to the leg's
    true time, and taking it for *both* legs makes their ratio a property of
    the two server loops rather than of whatever the box happened to be doing.
    A mean, or a single sample, mixes in that one-sided noise, and on a loaded
    box it does so unequally between the legs -- which is exactly how this
    test reported a 1.53x ratio against unmodified code (KOM-176).

    Failures are concatenated across repeats rather than reduced: a leg that
    failed on any repeat has not produced a comparable wall, and the caller
    asserts the list is empty.

    Note on ordering: this helper runs its own repeats back-to-back, so the
    test does *not* call it per leg.  It drives one repeat of each leg at a
    time (``repeats=1``) so that the two legs interleave; see
    ``test_throughput_smoke_is_not_worse``.
    """
    walls: list[float] = []
    failures: list[str] = []
    for _ in range(repeats):
        wall, run_failures = _timed_run(server_target, config)
        walls.append(wall)
        failures.extend(run_failures)
    return min(walls), failures


def test_throughput_smoke_is_not_worse() -> None:
    """The pipelined server is not dramatically slower than the serial one.

    Both legs are the *same* workload against the *same* transport: four
    spawned workers, a fixed request count each, one spawned server process.
    Only the server's loop differs -- ``ShardServer.serve`` (pipelined,
    unconditional) against a serial ``serve_once`` loop assembled here.  That
    is what makes the ratio meaningful; timing a served path against an
    in-process one would measure the shared-memory round trip and call it a
    regression.

    This is a smoke test and nothing more.  On CPU the forward is synchronous,
    so the pipeline has almost nothing to overlap and the honest expectation is
    "about the same"; the 1.5x allowance absorbs CI scheduling noise while
    still failing a loop that waits on its handle before collecting again, or
    that sleeps once per iteration where the serial loop slept only when idle.
    The GPU numbers this design exists for come from the advisor's bench.

    *Load robustness (KOM-176).*  Timing one leg once each made this a load
    instrument rather than a regression instrument: on a shared box running a
    self-play campaign plus CI gates (1-minute loadavg 40-60 on 32 cores) the
    single-run form reported 1.53x against *unmodified* code and failed 3/3.
    Two changes fix that without touching the budget, which stays at 1.5x.
    First, each leg is run ``_TIMED_REPEATS`` times and compared on its minimum
    wall -- interference is non-negative, so the minimum is the estimator
    closest to the leg's true time (see ``_best_of_walls``).  Second, the legs
    are *interleaved* -- serial, pipelined, serial, pipelined, ... -- rather
    than run as a block of serial followed by a block of pipelined, so a load
    transient lands on both legs instead of penalising whichever leg it
    overlapped.  Within each pair the serial leg still goes first, for the
    page-cache reason below.

    Deliberately *not* done: no ``pytest.skip`` on ``os.getloadavg()`` (that
    makes the smoke vacuous on precisely the busy box that runs it) and no
    CPU-time ratio (CPU time cannot see overlap, which is the whole point of
    pipelining).
    """
    config = rig.make_config(
        max_batch=16, batch_timeout_ms=0.5, request_timeout_s=60.0
    )

    serial_walls: list[float] = []
    pipelined_walls: list[float] = []
    serial_failures: list[str] = []
    pipelined_failures: list[str] = []

    for _ in range(_TIMED_REPEATS):
        # Serial first within each pair: whatever page-cache warmth the first
        # leg pays for, the second leg inherits, and the bound is one-sided in
        # the pipeline's favour.
        wall, run_failures = _best_of_walls(_serial_server_main, config, 1)
        serial_walls.append(wall)
        serial_failures.extend(run_failures)

        wall, run_failures = _best_of_walls(rig.server_main, config, 1)
        pipelined_walls.append(wall)
        pipelined_failures.extend(run_failures)

    serial_wall = min(serial_walls)
    pipelined_wall = min(pipelined_walls)

    assert serial_failures == [], f"serial leg: {serial_failures}"
    assert pipelined_failures == [], f"pipelined leg: {pipelined_failures}"
    assert serial_wall > 0.0, "the serial leg reported a zero wall"
    assert pipelined_wall <= serial_wall * _TIMED_RATIO, (
        f"pipelined wall {pipelined_wall:.3f}s vs serial {serial_wall:.3f}s "
        f"({pipelined_wall / serial_wall:.2f}x) over "
        f"{_TIMED_WORKERS * _TIMED_PER_WORKER} requests (best of "
        f"{_TIMED_REPEATS} interleaved runs per leg), budget is "
        f"{_TIMED_RATIO}x -- the pipelined loop must not serialise itself with "
        f"a wait-then-collect ordering or a per-iteration sleep"
    )
