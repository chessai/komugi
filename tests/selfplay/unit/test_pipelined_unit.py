"""Unit tests for the pipelined shard server (KOM-032).

The pipelined loop's interesting behaviour is all *interleaving*: what the
collect for batch N+1 may look at while batch N is on the GPU, what happens to
each of the two unanswered cohorts when ``stop_event`` arrives mid-cycle, and
whether any of it changes a single output byte. None of that is testable
against a real GPU in this gate (there isn't one), and against a real CPU
forward it would be timing-dependent -- the forward finishes before the collect
can observe anything.

So the seam is driven by a **manually-completed** :class:`ForwardHandle`: a
handle whose readiness the test flips by hand, in the main thread, at the exact
point the interleaving under test requires. The server runs in one thread (its
loop blocks by design, so something else has to be the test), the handle stays
in flight for as long as the test says, and every assertion below is a
consequence of program order rather than of a sleep winning a race.

The one test that does *not* use the mock is the equivalence pin: it runs the
real :class:`AsyncEvaluator` -- which on CPU is the serial evaluator with an
already-finished handle around it -- and compares its bytes to
:func:`serve_once`'s, exactly (invariant I6).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from multiprocessing import synchronize
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt

from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay import inference
from komugi.selfplay.inference import (
    AsyncEvaluator,
    BatchEvaluator,
    ForwardHandle,
    ForwardLauncher,
    InferenceConfig,
    InProcessEvaluator,
    NetEval,
    SlotClient,
    Slots,
)

IN_PLANES = 4
NUM_ACTIONS = 81

_SETTLE_S = 0.05
"""Long enough for the server thread to run ~50 collect scans.

Used only where the assertion is that something did NOT happen (a re-admitted
worker, an invisible payload). A too-short sleep would make those tests pass
vacuously, never spuriously.
"""


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net -- the same one the other unit suite uses."""
    trunk = TrunkConfig(in_planes=IN_PLANES, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=NUM_ACTIONS, policy_planes=1, seed=7)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _config(
    max_batch: int = 4, batch_timeout_ms: float = 0.0, max_wave: int = 1
) -> InferenceConfig:
    """A config shaped for the tiny net; the window defaults to "do not wait".

    ``max_wave`` defaults to 1 -- the single-row protocol every test above the
    wave section is written against -- so widening it is an opt-in per test.
    """
    return InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=max_batch,
        batch_timeout_ms=batch_timeout_ms,
        max_wave=max_wave,
    )


def _worker_request(
    worker_id: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A request that *names its owner*: every cell holds ``worker_id + 1``.

    That makes a batch's composition recoverable from the observations alone
    (:func:`_batch_ids`), which is how these tests assert which workers went
    into which forward without reaching inside the server's state.
    """
    obs = np.full((IN_PLANES, 9, 9), float(worker_id + 1), dtype=np.float32)
    return obs, np.ones(NUM_ACTIONS, dtype=np.bool_)


def _batch_ids(obs: npt.NDArray[np.float32]) -> list[int]:
    """Recover the worker ids of a stacked batch built by :func:`_worker_request`."""
    return [int(round(float(row.reshape(-1)[0]))) - 1 for row in obs]


def _wave_request(
    worker_id: int, rows: int, round_index: int = 0
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """An ``R``-row request whose every row *names itself* (KOM-040 b2).

    :func:`_worker_request`'s trick, one level finer: a ticket now carries
    several rows, so the tag has to identify the row rather than the worker.
    ``worker_id * 100 + round_index * 10 + row`` is unique across every worker,
    round and row these tests submit, which is what makes both wave claims
    checkable from the outside -- the batch compositions
    (:func:`_row_tags`) say which rows shared a forward, and the answers say
    which row each one came back in (:class:`_TagEchoEvaluator`).
    """
    tags = [float(worker_id * 100 + round_index * 10 + row) for row in range(rows)]
    obs = np.stack(
        [np.full((IN_PLANES, 9, 9), tag, dtype=np.float32) for tag in tags]
    )
    legal = np.zeros((rows, NUM_ACTIONS), dtype=np.bool_)
    # Two legal actions per row: enough that no row is the K=0 case the client
    # refuses before the doorbell, and few enough to be obviously deliberate.
    legal[:, :2] = True
    return obs, legal


def _wave_tags(
    request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]],
) -> list[float]:
    """The tags :func:`_wave_request` wrote, in submission order."""
    return _row_tags(request[0])


def _row_tags(obs: npt.NDArray[np.float32]) -> list[float]:
    """Recover the per-row tags of a stacked batch of :func:`_wave_request` rows."""
    return [float(row.reshape(-1)[0]) for row in obs]


class _TagEchoEvaluator:
    """A :class:`BatchEvaluator` that answers each row with its own tag.

    The tiny net answers correctly but anonymously: its outputs cannot say
    which *row* of which ticket they were computed from, and comparing them
    back is batch-composition dependent (see the drain test above). Echoing the
    tag through ``moves_left`` makes routing directly observable -- a client
    that got its own rows back, in its own order, reads its own tags -- which is
    the claim the wave scatter (one ``response_seq`` covering ``R`` rows) has to
    earn.
    """

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one tag-carrying evaluation per row of `obs`."""
        tags = np.asarray(_row_tags(obs), dtype=np.float32)
        return NetEval(
            policy_logits=np.asarray(
                np.repeat(tags[:, None], legal.shape[1], axis=1), dtype=np.float32
            ),
            wdl=np.asarray(np.repeat(tags[:, None], 3, axis=1), dtype=np.float32),
            moves_left=tags.copy(),
        )


def _random_request(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A distinguishable observation and a non-trivial legality mask."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((IN_PLANES, 9, 9)).astype(np.float32)
    legal = rng.random(NUM_ACTIONS) < 0.3
    return obs, np.asarray(legal, dtype=np.bool_)


def _pack(
    slots: Slots, request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]
) -> None:
    """Write the payload and *not* the doorbell -- half a submit, on purpose."""
    inference.pack_request(slots, request[0], request[1])


def _ring(
    slots: Slots,
    request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]] | None = None,
) -> int:
    """Do what :meth:`SlotClient.submit` does up to the doorbell; return the ticket.

    Payload first, counter second -- if this helper ever gets that backwards the
    tests below stop meaning what they say.
    """
    if request is not None:
        _pack(slots, request)
    ticket = int(slots.request_seq[0]) + 1
    slots.request_seq[0] = ticket
    return ticket


def _ring_wave(
    slots: Slots, request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]
) -> int:
    """:func:`_ring` for a whole wave: ``R`` rows, then ONE doorbell.

    The multi-row half of the protocol, and the same ordering rule -- the rows
    and the ``row_count`` header are payload, and the counter that announces
    them comes strictly last.
    """
    inference.pack_wave_request(slots, request[0], request[1])
    return _ring(slots)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 20.0) -> bool:
    """Poll `predicate` until it holds or the wall expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


@contextlib.contextmanager
def _slot_map(
    config: InferenceConfig, worker_ids: Sequence[int]
) -> Iterator[dict[int, Slots]]:
    """Create one real segment per worker; close and unlink them afterwards.

    The creator's own views stand in for the server's attached ones, which
    keeps the resource tracker balanced (see ``create_slots``' docstring).
    """
    slots_map: dict[int, Slots] = {}
    segments: list[SharedMemory] = []
    try:
        for worker_id in worker_ids:
            created = inference.create_slots(config)
            segments.append(created.shm)
            slots_map[worker_id] = created
        yield slots_map
    finally:
        for entry in slots_map.values():
            entry.close()
        for segment in segments:
            segment.unlink()


class _MockHandle:
    """A :class:`ForwardHandle` whose readiness the test flips explicitly.

    Nothing here is asynchronous: the "forward" already happened when the
    handle was built (the launcher evaluates eagerly, exactly as a real gather
    reads the slots eagerly). What the handle models is the *gap* -- the window
    in which the server has launched, owes those workers an answer, and must
    not touch their slots. :meth:`query` is false until the test says
    otherwise, and :meth:`result` blocks until then, which is precisely what a
    real CUDA event does.
    """

    def __init__(self, evaluation: NetEval, ready: bool = False) -> None:
        self._evaluation = evaluation
        self._ready = threading.Event()
        if ready:
            self._ready.set()

    def complete(self) -> None:
        """The forward has finished. Called from the test, never the server."""
        self._ready.set()

    def query(self) -> bool:
        return self._ready.is_set()

    def result(self) -> NetEval:
        if not self._ready.wait(timeout=30.0):
            raise AssertionError("blocked on a handle the test never completed")
        return self._evaluation


class _MockLauncher:
    """A :class:`ForwardLauncher` handing out manually-completed handles.

    Records every batch it was asked to launch, by worker id
    (:func:`_batch_ids`), so a test can assert what went into each forward --
    including the thing invariant I1 is about: that no worker went into two.

    ``tags`` is the same record read the other way, per ROW
    (:func:`_row_tags`), for the wave tests: at ``max_wave > 1`` a batch's
    worker ids no longer say how many rows each ticket contributed, and "no
    wave was split across forwards" is a statement about rows.
    """

    def __init__(self, evaluator: BatchEvaluator) -> None:
        self._evaluator = evaluator
        self.handles: list[_MockHandle] = []
        self.batches: list[list[int]] = []
        self.tags: list[list[float]] = []
        self.auto_complete = False

    def launch(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> ForwardHandle:
        handle = _MockHandle(self._evaluator.evaluate(obs, legal), self.auto_complete)
        self.batches.append(_batch_ids(obs))
        self.tags.append(_row_tags(obs))
        self.handles.append(handle)
        return handle

    def release_all(self) -> None:
        """Unblock everything, so a test that failed mid-flight still tears down."""
        self.auto_complete = True
        for handle in self.handles:
            handle.complete()


@contextlib.contextmanager
def _serving(
    slots_map: dict[int, Slots],
    launcher: ForwardLauncher,
    config: InferenceConfig,
    release: Callable[[], None] = lambda: None,
) -> Iterator[synchronize.Event]:
    """Run :func:`serve_pipelined` in a thread; yield its stop event.

    A thread, not a process: these tests hold the other end of every doorbell
    themselves. ``release`` unblocks any handle the body left in flight so that
    a failing assertion produces a failure rather than a hang.
    """
    stop = multiprocessing.Event()
    thread = threading.Thread(
        target=inference.serve_pipelined,
        args=(slots_map, launcher, config, stop),
        daemon=True,
    )
    thread.start()
    try:
        yield stop
    finally:
        stop.set()
        release()
        thread.join(30.0)
    assert not thread.is_alive(), "serve_pipelined never returned after stop_event"


def _assert_answer(
    slots: Slots,
    evaluator: InProcessEvaluator,
    request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]],
) -> None:
    """The bytes in `slots` are this request's answer, to the bit."""
    want = evaluator.evaluate(request[0][None], request[1][None])
    got = inference.read_response(slots)
    np.testing.assert_array_equal(got.policy_logits, want.policy_logits[0])
    np.testing.assert_array_equal(got.wdl, want.wdl[0])
    np.testing.assert_array_equal(got.moves_left, want.moves_left[0])


# --------------------------------------------------------------------------
# I1 -- the in-flight exclusion set (design review D1, the BLOCKER)
# --------------------------------------------------------------------------


def test_scan_pending_skips_the_excluded_in_flight_ids() -> None:
    """The mechanism, in isolation: an excluded id is not collected.

    An in-flight worker reads ``request_seq > response_seq`` for the whole
    forward -- the server publishes ``response_seq`` at scatter -- so the
    exclusion set is the *only* thing standing between the pipeline and a
    second serve of a request that is still being answered.
    """
    config = _config()
    with _slot_map(config, (0, 1)) as slots_map:
        for worker_id in (0, 1):
            _ring(slots_map[worker_id], _worker_request(worker_id))

        pending: dict[int, int] = {}
        inference._scan_pending(slots_map, pending, exclude=frozenset({0}))
        assert pending == {1: 1}, "an in-flight worker was re-admitted to the scan"
        # Worker 0 is still ringing; only the exclusion was hiding it.
        assert int(slots_map[0].request_seq[0]) > int(slots_map[0].response_seq[0])
        inference._scan_pending(slots_map, pending)
        assert pending == {0: 1, 1: 1}


def test_an_in_flight_worker_is_never_readmitted_and_is_answered_once() -> None:
    """I1 end to end: hold batch N open, and worker W must not enter batch N+1.

    Worker 0's request is in flight and its doorbell is still ringing for the
    entire duration -- which is exactly the state in which an exclusion-free
    scan re-admits it. If it were re-admitted, the loop would launch a *second*
    forward containing worker 0 the moment the first one scattered: a duplicate
    serve, a ``read_request`` of a slot the worker owns again, and a
    ``write_response`` over bytes the worker may be mid-read of. The assertion
    is therefore about the batch list, not about a timing window.
    """
    config = _config(max_batch=4)
    request = _worker_request(0)
    with _slot_map(config, (0, 1)) as slots_map:
        evaluator = InProcessEvaluator(_tiny_net(), config)
        launcher = _MockLauncher(evaluator)
        with _serving(slots_map, launcher, config, launcher.release_all):
            ticket = _ring(slots_map[0], request)
            assert _wait_until(lambda: len(launcher.handles) == 1)
            assert launcher.batches == [[0]]

            # In flight, unanswered, doorbell still ringing: ~50 collect scans
            # run against this state.
            time.sleep(_SETTLE_S)
            assert int(slots_map[0].response_seq[0]) == 0
            assert len(launcher.batches) == 1

            launcher.handles[0].complete()
            assert _wait_until(lambda: int(slots_map[0].response_seq[0]) == ticket)
            # ... and nothing follows it into a second forward.
            time.sleep(_SETTLE_S)
            assert launcher.batches == [[0]], "worker 0 was served twice (I1 violated)"
            assert int(slots_map[0].request_seq[0]) == ticket
            _assert_answer(slots_map[0], evaluator, request)


# --------------------------------------------------------------------------
# Ordering pins, mirroring test_inference_unit.py's serve_once section
# --------------------------------------------------------------------------


def test_a_payload_without_a_doorbell_is_invisible_to_the_pipelined_collect() -> None:
    """Half a submit is not a submit -- in the idle scan and in the busy one.

    The pipelined collect runs in two places (idle, and during a forward) and
    both must treat "bytes present, counter not advanced" as nothing at all.
    That is what makes the worker's payload-before-counter ordering safe rather
    than merely tidy.
    """
    config = _config(max_batch=4)
    with _slot_map(config, (0, 1)) as slots_map:
        launcher = _MockLauncher(InProcessEvaluator(_tiny_net(), config))
        with _serving(slots_map, launcher, config, launcher.release_all):
            # Worker 1 writes its observation and never rings.
            _pack(slots_map[1], _worker_request(1))
            time.sleep(_SETTLE_S)
            assert launcher.batches == [], "a payload with no doorbell was collected"

            # Worker 0 rings; worker 1 is still silent while 0 is in flight.
            _ring(slots_map[0], _worker_request(0))
            assert _wait_until(lambda: len(launcher.handles) == 1)
            time.sleep(_SETTLE_S)
            assert launcher.batches == [[0]]

            launcher.handles[0].complete()
            assert _wait_until(lambda: int(slots_map[0].response_seq[0]) == 1)
            time.sleep(_SETTLE_S)
            assert launcher.batches == [[0]]
            assert int(slots_map[1].response_seq[0]) == 0


def test_two_pipelined_cycles_answer_every_scanned_request_exactly_once() -> None:
    """Collect during the forward, scatter, launch the next batch -- once each.

    Worker 1 rings *while* worker 0's batch is in flight, so it is collected by
    the overlapping scan and launched the instant batch 0 scatters (no idle
    sleep, no second window). Both workers are answered, each in exactly one
    forward, each with its own bytes.
    """
    config = _config(max_batch=4)
    requests = {worker_id: _worker_request(worker_id) for worker_id in (0, 1)}
    with _slot_map(config, (0, 1)) as slots_map:
        evaluator = InProcessEvaluator(_tiny_net(), config)
        launcher = _MockLauncher(evaluator)
        with _serving(slots_map, launcher, config, launcher.release_all):
            ticket0 = _ring(slots_map[0], requests[0])
            assert _wait_until(lambda: len(launcher.handles) == 1)

            ticket1 = _ring(slots_map[1], requests[1])
            time.sleep(_SETTLE_S)  # collected, but batch 0 is still in flight
            assert launcher.batches == [[0]], "batch N+1 launched before N scattered"

            launcher.handles[0].complete()
            assert _wait_until(lambda: len(launcher.handles) == 2)
            assert int(slots_map[0].response_seq[0]) == ticket0
            launcher.handles[1].complete()
            assert _wait_until(lambda: int(slots_map[1].response_seq[0]) == ticket1)

            time.sleep(_SETTLE_S)
            assert launcher.batches == [[0], [1]]
            served = [worker_id for batch in launcher.batches for worker_id in batch]
            assert sorted(served) == [0, 1]
            assert len(served) == len(set(served)), "a request was served twice"
            for worker_id in (0, 1):
                _assert_answer(slots_map[worker_id], evaluator, requests[worker_id])


# --------------------------------------------------------------------------
# I4 -- drain on stop covers BOTH unanswered windows (design review D2/D5)
# --------------------------------------------------------------------------


def test_the_in_flight_collect_outlives_the_batching_window() -> None:
    """I3 pinned directly: a ringer arriving long after the window still joins.

    The union review (F2) showed I3's mutant -- an in-flight collect that
    honours ``batch_timeout_ms`` -- dying only by accident of the drain test's
    ``batch_timeout_ms=0.0`` config. This is the direct pin: with a *large*
    window (250 ms) and a ringer arriving well after it (400 ms into the
    forward), the correct loop has already collected the ringer when the
    forward completes, and the loop-top ``if not pending`` skips the idle
    window entirely -- the next launch is immediate. The mutant's early-closed
    collect misses the ringer, falls to the idle path after scatter, and the
    idle window then waits its full 250 ms below the ceiling. The 125 ms
    assertion sits between ~3 ms (correct) and >=250 ms (mutant) with 40x/2x
    margins, so the I3 win cannot silently evaporate again.
    """
    config = _config(max_batch=4, batch_timeout_ms=250.0)
    with _slot_map(config, (0, 1)) as slots_map:
        evaluator = InProcessEvaluator(_tiny_net(), config)
        launcher = _MockLauncher(evaluator)
        with _serving(slots_map, launcher, config, launcher.release_all):
            _ring(slots_map[0], _worker_request(0))
            assert _wait_until(lambda: len(launcher.handles) == 1)
            assert launcher.batches == [[0]]

            # Well past any honoured in-flight deadline, with the forward
            # still open: the window-ignoring collect must pick this up.
            time.sleep(0.4)
            ticket = _ring(slots_map[1], _worker_request(1))
            time.sleep(0.05)  # a few 1 ms collect scans

            launcher.handles[0].complete()
            start = time.monotonic()
            assert _wait_until(lambda: len(launcher.batches) == 2)
            elapsed = time.monotonic() - start
            assert launcher.batches[1] == [1]
            assert elapsed < 0.125, (
                f"the ringer was launched {elapsed * 1e3:.0f} ms after the "
                f"forward completed: it was not collected in flight (I3), and "
                f"the idle window was paid for it instead"
            )
            launcher.handles[1].complete()
            assert _wait_until(lambda: int(slots_map[1].response_seq[0]) == ticket)


def test_stop_drains_the_in_flight_batch_and_the_collected_pending() -> None:
    """Both cohorts are answered before ``serve_pipelined`` returns.

    Stop is set while worker 0's forward is unresolved (window a: launched, not
    scattered) and worker 1 has been collected but not launched (window b). A
    server that returned here would leave both workers polling
    ``response_seq`` for the full ``request_timeout_s`` and then dying of
    :class:`InferenceTimeout` -- a behaviour regression against
    ``serve_once``, which answers everything it ever scans before returning.
    """
    config = _config(max_batch=4)
    requests = {worker_id: _worker_request(worker_id) for worker_id in (0, 1)}
    with _slot_map(config, (0, 1)) as slots_map:
        evaluator = InProcessEvaluator(_tiny_net(), config)
        launcher = _MockLauncher(evaluator)
        with _serving(slots_map, launcher, config, launcher.release_all) as stop:
            ticket0 = _ring(slots_map[0], requests[0])
            assert _wait_until(lambda: len(launcher.handles) == 1)
            ticket1 = _ring(slots_map[1], requests[1])
            time.sleep(_SETTLE_S)  # worker 1 is now collected, not launched

            # The drain's own forward must be able to finish; only the handle
            # that is ALREADY in flight stays manual.
            launcher.auto_complete = True
            stop.set()
            time.sleep(_SETTLE_S)
            # Still nothing published: the drain is blocked on the unresolved
            # forward, which is the state this test exists to reach.
            assert int(slots_map[0].response_seq[0]) == 0
            assert int(slots_map[1].response_seq[0]) == 0

            launcher.handles[0].complete()

        # _serving joined the thread: serve_pipelined has RETURNED, and both
        # cohorts were published before it did.
        assert int(slots_map[0].response_seq[0]) == ticket0
        assert int(slots_map[1].response_seq[0]) == ticket1
        assert launcher.batches == [[0], [1]]
        for worker_id in (0, 1):
            _assert_answer(slots_map[worker_id], evaluator, requests[worker_id])


# --------------------------------------------------------------------------
# I6 -- on CPU the pipeline is the serial loop, byte for byte
# --------------------------------------------------------------------------


def test_a_ring_the_loop_never_scanned_is_still_answered_at_stop() -> None:
    """The drain's final scan: rung-before-stop must never mean wedged.

    The gate's stop-mid-submit behavior test caught the narrow drain twice:
    a stop landing between a worker's ring and the loop's next scan left the
    request in no cohort -- neither launched nor collected -- and the drain,
    which deliberately ran no scan, dropped it; the worker wedged for the
    full ``request_timeout_s`` and died of :class:`InferenceTimeout` (4/4
    workers, both gate runs, green in isolation -- scheduling skew, not
    load). The deterministic distillation: doorbells ring, stop is already
    set, the loop body never runs a scan at all. The drain's one final scan
    is what answers them; without it this test wedges on the workers' side
    of the story and fails immediately on the counters.
    """
    config = _config(max_batch=4)
    requests = {0: _worker_request(0), 1: _worker_request(1)}
    with _slot_map(config, (0, 1)) as slots_map:
        evaluator = InProcessEvaluator(_tiny_net(), config)
        launcher = _MockLauncher(evaluator)
        launcher.auto_complete = True
        tickets = {
            worker_id: _ring(slots_map[worker_id], request)
            for worker_id, request in requests.items()
        }
        stop = multiprocessing.Event()
        stop.set()

        inference.serve_pipelined(slots_map, launcher, config, stop)

        # Answered is the claim; payload routing is the row-swap mutant's test.
        # (No bit-compare here: the drain serves both rings as one batch of 2,
        # and a batch-of-1 recomputation differs in conv reduction order.)
        for worker_id, ticket in tickets.items():
            assert int(slots_map[worker_id].response_seq[0]) == ticket, (
                f"worker {worker_id} rang before the stop and was never "
                f"answered: the drain dropped a request the loop had not "
                f"scanned yet"
            )
            answer = inference.read_response(slots_map[worker_id])
            assert np.isfinite(answer.policy_logits).all()


# --------------------------------------------------------------------------
# Waves through the pipelined loop (KOM-040 b2; b2 wave-1 review F3)
#
# Everything above runs at ``max_wave == 1``, where a ticket is a row and the
# loop's row arithmetic is indistinguishable from ticket arithmetic. These two
# run the same loop at width: the drain over tickets no single forward can
# carry, and a steady state with clients of three different widths.
# --------------------------------------------------------------------------


def test_the_drain_runs_one_forward_per_wave_when_a_wave_fills_the_ceiling() -> None:
    """I4, window (b), at width: three 4-row waves cost three forwards.

    The row ceiling here is 4 (``max_batch``), and each of the three pending
    tickets is 4 rows wide, so the no-split admission rule can take exactly one
    per pass: the drain's ``while pending`` loop is the only reason all three
    are answered. A single-forward drain -- the shape the loop had before a
    ticket could be worth more than one row (mutant M10) -- serves the rank-1
    wave, finds two tickets it cannot fit, and returns, leaving two workers to
    wedge for the whole ``request_timeout_s`` and die of
    :class:`~komugi.selfplay.inference.InferenceTimeout`.

    ``stop`` is set before the call, so the loop body never runs and every
    ticket goes through the drain, which is the window under test.
    """
    config = _config(max_batch=4, max_wave=4)
    worker_ids = (0, 1, 2)
    requests = {
        worker_id: _wave_request(worker_id, rows=4) for worker_id in worker_ids
    }
    with _slot_map(config, worker_ids) as slots_map:
        launcher = _MockLauncher(_TagEchoEvaluator())
        launcher.auto_complete = True
        tickets = {
            worker_id: _ring_wave(slots_map[worker_id], requests[worker_id])
            for worker_id in worker_ids
        }
        stop = multiprocessing.Event()
        stop.set()

        inference.serve_pipelined(slots_map, launcher, config, stop)

        assert len(launcher.tags) == 3, (
            f"the drain ran {len(launcher.tags)} forward(s) for three waves that "
            f"cannot share one: {launcher.tags}"
        )
        # Fairness order at equal tickets is worker id, so the forwards are the
        # waves in slot order -- each whole, and each alone in its batch.
        assert launcher.tags == [_wave_tags(requests[w]) for w in worker_ids]
        for worker_id in worker_ids:
            slots = slots_map[worker_id]
            assert int(slots.response_seq[0]) == tickets[worker_id], (
                f"worker {worker_id}'s wave was never answered: the drain "
                f"stopped after the batches it could fill in one forward"
            )
            # The width the client checks its own submission against (W5).
            assert int(slots.row_count_echo[0]) == 4
            answers = inference.read_wave_response(slots, requests[worker_id][1], config)
            assert [float(answer.moves_left) for answer in answers] == _wave_tags(
                requests[worker_id]
            )


def test_a_pipelined_steady_state_routes_every_row_and_splits_no_wave() -> None:
    """The loop under mixed-width load: rows land where they were asked from.

    Three real :class:`~komugi.selfplay.inference.SlotClient` s -- 4, 1 and 2
    rows per ticket -- submit five waves each against the running pipelined
    loop, resubmitting the instant they are answered. That is the arrival
    pattern production has and the tests above do not: every forward's collect
    overlaps clients that are already ringing again.

    Three claims, all read off the observable batch compositions and the
    clients' own answers:

    * **Routing.** Every client gets back its own rows, in its own order --
      each row's tag (:func:`_wave_request`) echoed by the evaluator through
      ``moves_left``. A wave scatter publishes ``R`` rows under ONE
      ``response_seq``, so a mis-set cursor would hand a client its sibling's
      rows with a counter that says they are its own.
    * **No wave is split.** Each ticket's rows appear together, contiguously,
      in exactly one forward. Half a wave in a forward cannot be published --
      there is one counter per ticket -- so a split is a wedge, not a slow path.
    * **Row composition under mixed widths.** (The ROWS-not-tickets window
      bound is pinned by test_the_idle_window_bound_counts_rows_not_tickets
      below -- this test's parameters cannot see it.) Three tickets are
      almost always pending and the ceiling is 4 rows, so a ticket-counting
      bound has nothing to stop at; the row-counting one does, and it shows in
      the compositions: no forward carries more than 4 rows, and a forward that
      carries worker 0's 4-row wave carries *nothing else*, because the wave
      alone saturates the ceiling.
    """
    config = _config(max_batch=4, batch_timeout_ms=0.2, max_wave=4)
    widths = {0: 4, 1: 1, 2: 2}
    rounds = 5
    answers: dict[int, list[list[float]]] = {}
    failures: dict[int, BaseException] = {}

    with _slot_map(config, tuple(widths)) as slots_map:
        launcher = _MockLauncher(_TagEchoEvaluator())
        launcher.auto_complete = True

        def play(worker_id: int) -> None:
            """Submit `rounds` waves back to back, recording what came back."""
            client = SlotClient(slots_map[worker_id].shm.name, worker_id, config)
            try:
                got: list[list[float]] = []
                for round_index in range(rounds):
                    obs, legal = _wave_request(
                        worker_id, widths[worker_id], round_index
                    )
                    got.append(
                        [
                            float(answer.moves_left)
                            for answer in client.submit_wave(obs, legal)
                        ]
                    )
                answers[worker_id] = got
            except BaseException as error:  # noqa: BLE001 -- re-raised below
                failures[worker_id] = error
            finally:
                client.close()

        with _serving(slots_map, launcher, config, launcher.release_all):
            threads = [
                threading.Thread(target=play, args=(worker_id,), daemon=True)
                for worker_id in widths
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60.0)
                assert not thread.is_alive(), "a wave client never got its answer"

    assert not failures, f"a client failed: {failures}"

    for worker_id, width in widths.items():
        assert len(answers[worker_id]) == rounds
        for round_index, got in enumerate(answers[worker_id]):
            want = _wave_tags(_wave_request(worker_id, width, round_index))
            assert got == want, (
                f"worker {worker_id} round {round_index} was answered with "
                f"{got}, not its own rows {want}"
            )

    ceiling = min(config.max_batch, config.max_wave * len(widths))
    for batch in launcher.tags:
        assert len(batch) <= ceiling, (
            f"a forward carried {len(batch)} rows over a {ceiling}-row ceiling: "
            f"the collect bound is counting tickets, not rows"
        )
    for worker_id, width in widths.items():
        for round_index in range(rounds):
            want = _wave_tags(_wave_request(worker_id, width, round_index))
            homes = [batch for batch in launcher.tags if want[0] in batch]
            assert len(homes) == 1, (
                f"worker {worker_id}'s round-{round_index} wave appears in "
                f"{len(homes)} forwards; a ticket is served exactly once"
            )
            start = homes[0].index(want[0])
            assert homes[0][start : start + width] == want, (
                f"worker {worker_id}'s round-{round_index} wave was split across "
                f"forwards or reordered within one: {homes[0]}"
            )
    saturating = [batch for batch in launcher.tags if len(batch) == ceiling]
    assert saturating, "no forward ever reached the row ceiling"
    for batch in saturating:
        assert batch == _wave_tags(_wave_request(0, 4, int(batch[0]) // 10)), (
            f"a {ceiling}-row forward mixed worker 0's full wave with other "
            f"rows: {batch}"
        )


def test_pipelined_output_is_bit_identical_to_serve_once_on_cpu() -> None:
    """The same requests, the same weights, the same bytes -- ``assert_array_equal``.

    Not ``allclose``: invariant I6 is *bit*-identical, which is what makes the
    equivalence corpus and the serial-debug golden path unaffected by KOM-032.
    It holds by construction -- on CPU :class:`AsyncEvaluator` calls the very
    :class:`InProcessEvaluator` the serial loop calls -- and this pins that the
    construction survives, including the batch composition: all three doorbells
    ring before either loop starts, so both build one batch of three in slot
    order.
    """
    net = _tiny_net()
    config = _config(max_batch=4)
    worker_ids = (0, 1, 2)
    requests = {
        worker_id: _random_request(seed=40 + worker_id) for worker_id in worker_ids
    }

    with _slot_map(config, worker_ids) as slots_map:
        for worker_id in worker_ids:
            _ring(slots_map[worker_id], requests[worker_id])
        serial_evaluator = InProcessEvaluator(net, config)
        assert inference.serve_once(slots_map, serial_evaluator, config) == 3
        serial = {
            worker_id: inference.read_response(slots_map[worker_id])
            for worker_id in worker_ids
        }

    with _slot_map(config, worker_ids) as slots_map:
        for worker_id in worker_ids:
            _ring(slots_map[worker_id], requests[worker_id])
        with _serving(slots_map, AsyncEvaluator(net, config), config):
            assert _wait_until(
                lambda: all(
                    int(slots_map[worker_id].response_seq[0]) == 1
                    for worker_id in worker_ids
                )
            )
        pipelined = {
            worker_id: inference.read_response(slots_map[worker_id])
            for worker_id in worker_ids
        }

    for worker_id in worker_ids:
        np.testing.assert_array_equal(
            pipelined[worker_id].policy_logits, serial[worker_id].policy_logits
        )
        np.testing.assert_array_equal(pipelined[worker_id].wdl, serial[worker_id].wdl)
        np.testing.assert_array_equal(
            pipelined[worker_id].moves_left, serial[worker_id].moves_left
        )


def test_the_idle_window_bound_counts_rows_not_tickets() -> None:
    """The wave-1 review's F3 debt, paid: kill M11 (_collected_rows -> tickets).

    Authored and validated by the wave-2 union reviewer (their probe passed
    pristine in 1.3 s and failed under the re-applied mutant at a 3 s window
    hold); adopted verbatim at the union.

    Three clients ring BEFORE the server starts, with 4+1+1 = 6 pending rows
    over a 4-row ceiling and a deliberately huge 3000 ms window.  A
    rows-counting window bound sees 6 >= 4 and closes immediately; a
    ticket-counting one sees 3 < 4 tickets and holds the window for the full
    3 s before launching anything.  The elapsed bound is what makes the
    mutant visible: latency, which is the only thing the bound governs.
    """
    config = _config(max_batch=4, batch_timeout_ms=3000.0, max_wave=4)
    worker_ids = (0, 1, 2)
    with _slot_map(config, worker_ids) as slots_map:
        launcher = _MockLauncher(_TagEchoEvaluator())
        launcher.auto_complete = True
        requests = {
            w: _wave_request(w, rows=4 if w == 0 else 1) for w in worker_ids
        }
        tickets = {
            w: _ring_wave(slots_map[w], requests[w]) for w in worker_ids
        }
        stop = multiprocessing.Event()
        server = threading.Thread(
            target=inference.serve_pipelined,
            args=(slots_map, launcher, config, stop),
            daemon=True,
        )
        start = time.monotonic()
        server.start()
        try:
            answered = _wait_until(
                lambda: all(
                    int(slots_map[w].response_seq[0]) == tickets[w]
                    for w in worker_ids
                ),
                timeout_s=20.0,
            )
            elapsed = time.monotonic() - start
        finally:
            stop.set()
            server.join(20.0)
        assert answered, "the waves were never answered"
        assert elapsed < 2.0, (
            f"6 pending rows over a 4-row ceiling held the window for "
            f"{elapsed:.2f}s: the window bound is counting tickets, not rows"
        )
