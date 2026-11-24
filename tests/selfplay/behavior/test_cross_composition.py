"""Cross-composition equivalence: a REAL net, real concurrency, real batches.

Audit gap G6 (``docs/design/kom-101-121-test-debt-audit.md`` section 4). Both
numeric equivalence gates that exist pin ``max_batch=1`` -- the corpus gate next
door submits one position at a time through a one-row server, and
:mod:`test_serial_vs_production` compares serial against production with the
batch ceiling nailed shut -- while the one suite that *is* concurrent
(:mod:`test_batch_permutation`, four workers at ``max_batch=16``) drives a stub
net whose every output is a pure function of its own row. Put together: **no
test in this repository has ever checked a real network's numbers coming out of
a multi-row batch.**

That is a hole with a shape. Every defect in the class "a row's answer depends
on which other rows shared its forward, or on where in the batch it sat" is
invisible to all three: a gather/scatter cursor that drifts by a row once a
batch holds more than one ticket, a transposed batch dimension, a stale or
zero-padded row read back as an answer (the KOM-048 capture pads to
``max_batch`` and slices; the slice is the only thing between a real answer and
the pad), a normalisation that mixes rows. The stub suite cannot see them
numerically -- its net encodes the *submitter's* code, so it catches crossed
wiring but agrees with itself about arithmetic -- and the batch-of-1 gates
cannot see them at all, because a batch of one row has no cross-row behaviour
to get wrong.

So this file runs the corpus gate's own comparison (per-logit, 1e-5, against a
batch-of-1 :class:`InProcessEvaluator` reference) through a server that is
actually batching: four spawned clients, released together onto one
:class:`ShardServer`, ``max_batch=8`` and a 50 ms window so the server has both
the room and the patience to assemble multi-row batches. The net is
``inference_rig.random_factory`` -- the same real net the equivalence corpus
uses, weights fixed by a numpy seed so the spawned shard and the in-process
reference build bit-identical parameters without touching torch's global RNG.

**The non-vacuity guard is the other half of the test.** A server that batched
nothing -- one row per forward, 96 times -- would pass the equivalence
assertion perfectly, and the green would mean nothing at all; that is precisely
how the existing concurrent suite ends up unable to claim anything about
batching ("the pinned seam exposes no hook for it",
``test_batch_permutation.py:30-31``). KOM-051 adds that hook --
``komugi.selfplay.inference.forward_row_histogram()``, a monotone per-process
``rows -> forwards`` counter written at :meth:`AsyncEvaluator.launch` -- and
the shard process reports its histogram on the way out. This suite asserts that
at least one executed forward carried **two or more rows**, so a run where
batching never happened FAILS here instead of passing quietly, and it asserts
that the histogram's total rows equal the submissions, so the counter cannot
claim work that was never done.

Two things this deliberately does not claim, so nobody reads more into a green:

* ``rig.RandomNet`` has no :class:`torch.nn.BatchNorm2d`, so the specific
  cross-row leak of a net left in *train* mode is out of reach here -- the net
  is the equivalence corpus's, on purpose, and pinning that failure needs a
  normalising net (a separate gap, worth its own ticket). What is pinned is
  every leak that goes through row *composition* and *placement*, which is
  where the seam's own code lives.
* Design F16 caps a batch at ``min(max_batch, max_wave * clients)``, so with
  four one-row clients the effective ceiling is 4, not 8. ``max_batch=8`` is
  therefore deliberately non-binding: the batching this test observes comes
  from the window and the co-arrival, and the ceiling is not what is being
  measured.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

import inference_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "ShardServer"):
    pytest.skip(
        "komugi.selfplay exports no ShardServer yet (KOM-021a)",
        allow_module_level=True,
    )

#: Four clients, per the G6 remedy: enough that a batch is a composition and
#: not a coincidence, and the same width the permutation suite uses.
_WORKERS = 4
_PER_WORKER = 24
_CORPUS_SIZE = _WORKERS * _PER_WORKER

#: ``max_batch=8`` leaves the F16 ceiling (4 rows here) doing the bounding, and
#: the wide window is what makes multi-row batches near-certain rather than a
#: race the CI loses -- the same reasoning, and the same 50 ms, as
#: ``test_batch_permutation``'s wide leg (wave-9a review F6).
_MAX_BATCH = 8
_WINDOW_MS = 50.0

#: The corpus gate's pre-registered tolerance (design D3), reused unchanged:
#: this is that comparison, moved onto a batching server.
_TOLERANCE = 1e-5

#: Any fixed value; written out so the corpus is reconstructable from the file.
_CORPUS_SEED = 20260830


def _seam() -> Any:
    """:mod:`komugi.selfplay.inference` itself, in whichever process asks.

    Direct rather than through :func:`inference_rig.selfplay`, because the
    KOM-051 histogram is deliberately *not* re-exported from the package: it is
    per-process observability, not part of what crosses the seam.
    """
    return importlib.import_module("komugi.selfplay.inference")


def _corpus() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """96 (obs, legal) pairs from a SeedSequence -- same in every process."""
    rng = np.random.default_rng(np.random.SeedSequence(_CORPUS_SEED))
    obs = rng.standard_normal(
        (_CORPUS_SIZE, rig.IN_PLANES, rig.BOARD, rig.BOARD)
    ).astype(np.float32)
    legal = rng.random((_CORPUS_SIZE, rig.NUM_ACTIONS)) < 0.35
    # A position with no legal move is never evaluated -- the client refuses it
    # before the doorbell -- and it would let a masking implementation return
    # all -inf and "agree".
    legal[:, 0] = True
    return obs, legal.astype(np.bool_)


def _serial_reference(
    obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
    """Batch-of-1 through a bare InProcessEvaluator, in this process.

    One row per forward, deliberately: the reference must be the answer a
    position gets when it shares its forward with *nothing*, or the comparison
    below could not detect a batch that changed it.
    """
    evaluator = sp.InProcessEvaluator(rig.random_factory(), rig.make_config())
    rows: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]] = []
    for index in range(obs.shape[0]):
        out = evaluator.evaluate(obs[index : index + 1], legal[index : index + 1])
        rows.append(
            (
                np.asarray(out.policy_logits, dtype=np.float64).reshape(-1),
                np.asarray(out.wdl, dtype=np.float64).reshape(-1),
                rig.as_scalar(out.moves_left),
            )
        )
    return rows


def _shard_main(
    factory: Any,
    config: Any,
    handles: Any,
    stop_event: Any,
    report: Any,
) -> None:
    """Serve, then report this process's forward-row histogram (KOM-051).

    Not :func:`inference_rig.server_main`, and for exactly one reason: the
    histogram is per-process module state living in the *shard*, so the only
    way the parent can learn what widths were actually forwarded is for the
    shard to say so before it exits. The report goes in a ``finally`` so a
    server that dies still tells us what it managed to batch -- that is the
    diagnostic that separates "the equivalence assertion never ran" from "it
    ran and nothing was ever batched".
    """
    inference = _seam()
    try:
        rig.selfplay().ShardServer(factory, config, list(handles)).serve(stop_event)
    finally:
        report.put(inference.forward_row_histogram())


def _worker_main(
    config: Any,
    handle: Any,
    obs: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
    barrier: Any,
    errors: Any,
    answers: Any,
) -> None:
    """Submit this client's slice of the corpus and ship every answer back.

    The comparison happens in the parent, which holds the serial reference; a
    worker's job is to be one of four concurrent submitters and to report what
    it was told.
    """
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        # Co-arrival: without it the clients trickle in one at a time and the
        # server never has a reason to batch across them (the permutation
        # suite's F6 lesson, and here it is load-bearing for the guard).
        barrier.wait(rig.BARRIER_WALL_S)
        rows: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]] = []
        for index in range(obs.shape[0]):
            response = client.submit(obs[index], legal[index])
            rows.append(
                (
                    np.asarray(response.policy_logits, dtype=np.float32),
                    np.asarray(response.wdl, dtype=np.float32),
                    rig.as_scalar(response.moves_left),
                )
            )
        answers.put((worker_id, rows))
    except BaseException as exc:  # reported, not raised: the parent asserts
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def test_multi_row_batches_match_the_batch_of_one_reference() -> None:
    """Four concurrent clients, real net, real batches, 1e-5 per logit."""
    assert not torch.backends.cudnn.benchmark, (
        "cudnn.benchmark must stay False for the equivalence claim (D3): "
        "autotuned kernels can differ between the two paths"
    )

    obs, legal = _corpus()
    reference = _serial_reference(obs, legal)

    config = rig.make_config(
        max_batch=_MAX_BATCH,
        batch_timeout_ms=_WINDOW_MS,
        request_timeout_s=60.0,
    )
    worker_ids = list(range(_WORKERS))

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        barrier = rig.CTX.Barrier(_WORKERS)
        errors = rig.CTX.Queue()
        answers = rig.CTX.Queue()
        report = rig.CTX.Queue()

        server = rig.spawn(
            _shard_main,
            rig.random_factory,
            config,
            handles,
            stop_event,
            report,
            name="shard-server",
        )
        workers = [
            rig.spawn(
                _worker_main,
                config,
                handle,
                obs[index * _PER_WORKER : (index + 1) * _PER_WORKER],
                legal[index * _PER_WORKER : (index + 1) * _PER_WORKER],
                barrier,
                errors,
                answers,
                name=f"worker-{handle.worker_id}",
            )
            for index, handle in enumerate(handles)
        ]

        failures = rig.join_all(workers, rig.SPAWN_WALL_S, errors)
        # collect rather than drain: joinable and readable are not the same
        # instant for a queue the child wrote to just before exiting.
        collected = dict(rig.collect(answers, _WORKERS, rig.JOIN_GRACE_S))
        server_alive = server.is_alive()
        stop_event.set()
        # Read the shard's histogram BEFORE joining it: the report is written
        # on the way out, and joining first is the classic way to turn a
        # reported result into a hang.
        reports = rig.collect(report, 1, rig.JOIN_GRACE_S * 3)
        failures.extend(rig.join_all([server], rig.JOIN_GRACE_S * 3))

    assert failures == [], failures
    assert server_alive, "the server died before the clients finished"
    assert sorted(collected) == worker_ids, (
        f"clients {sorted(collected)} reported, expected {worker_ids}"
    )

    worst = {"policy_logits": 0.0, "wdl": 0.0, "moves_left": 0.0}
    submissions = 0
    for worker_id, rows in collected.items():
        assert len(rows) == _PER_WORKER, (
            f"client {worker_id} reported {len(rows)} answers, "
            f"expected {_PER_WORKER}"
        )
        for offset, (policy, wdl, moves_left) in enumerate(rows):
            index = worker_id * _PER_WORKER + offset
            want_policy, want_wdl, want_mlh = reference[index]
            got_policy = np.asarray(policy, dtype=np.float64).reshape(-1)
            got_wdl = np.asarray(wdl, dtype=np.float64).reshape(-1)
            assert got_policy.shape == want_policy.shape
            assert got_wdl.shape == want_wdl.shape
            worst["policy_logits"] = max(
                worst["policy_logits"],
                float(np.max(np.abs(got_policy - want_policy))),
            )
            worst["wdl"] = max(
                worst["wdl"], float(np.max(np.abs(got_wdl - want_wdl)))
            )
            worst["moves_left"] = max(
                worst["moves_left"], abs(moves_left - want_mlh)
            )
            submissions += 1

    for field, delta in worst.items():
        assert delta <= _TOLERANCE, (
            f"max |delta {field}| = {delta:g} over {submissions} positions "
            f"served in multi-row batches, gate is {_TOLERANCE:g} "
            f"(docs/design/kom-021-selfplay.md D3, audit gap G6)"
        )
    # Anti-vacuity on the reference itself: a net that returned zeros
    # everywhere would pass the tolerance above with room to spare.
    spread = float(np.max(np.abs(np.stack([row[0] for row in reference]))))
    assert spread > 1e-3, f"the reference policy logits are ~0 ({spread:g})"

    # The guard the whole file exists for (KOM-051 seam): a server that served
    # every request alone would have satisfied every assertion above.
    assert reports, (
        "the shard never reported a forward-row histogram: without it the "
        "equivalence assertion above cannot be distinguished from 96 batches "
        "of one, which is the vacuous pass this test exists to prevent"
    )
    histogram = {int(rows): int(count) for rows, count in reports[0].items()}
    batched = sum(count for rows, count in histogram.items() if rows >= 2)
    assert batched >= 1, (
        f"no executed forward carried two or more rows: {histogram!r}. Every "
        f"answer above was then a batch of one, so this run proves nothing "
        f"about cross-row composition -- which is audit gap G6 exactly. Raise "
        f"batch_timeout_ms or check that the server is assembling batches"
    )
    assert sum(rows * count for rows, count in histogram.items()) == submissions, (
        f"the shard forwarded {sum(r * c for r, c in histogram.items())} rows "
        f"for {submissions} submissions ({histogram!r}): the histogram and the "
        f"answers describe different work"
    )


def test_the_forward_row_histogram_counts_rows_once_per_forward() -> None:
    """The seam's own premise, pinned in-process (KOM-051).

    The guard above reads a number out of another process and calls it "rows in
    a forward". This is where that meaning is checked: launch batches of known
    widths through the production launcher and confirm the histogram gained
    exactly one entry per forward, at the forward's own width -- not one per
    row, not one per client, and not two for the CPU path's launch-then-serial
    delegation. It also pins the two properties the guard's phrasing rests on:
    the counter is monotone, and what it hands back is a snapshot.
    """
    inference = _seam()
    config = rig.make_config(max_batch=_MAX_BATCH)
    evaluator = inference.AsyncEvaluator(rig.random_factory(), config)
    obs, legal = _corpus()

    before = inference.forward_row_histogram()
    for start, stop in ((0, 3), (3, 4), (4, 7)):
        evaluator.launch(obs[start:stop], legal[start:stop]).result()
    after = inference.forward_row_histogram()

    assert after.get(3, 0) - before.get(3, 0) == 2, (
        f"two three-row forwards should be two entries at width 3: "
        f"{before!r} -> {after!r}"
    )
    assert after.get(1, 0) - before.get(1, 0) == 1, (
        f"one one-row forward should be one entry at width 1: "
        f"{before!r} -> {after!r}"
    )
    # Monotone: this is a counter, not a gauge, and the guard above reads it
    # without ever resetting it.
    assert all(after.get(rows, 0) >= count for rows, count in before.items()), (
        f"the histogram lost counts: {before!r} -> {after!r}"
    )
    # A snapshot, not a live view: mutating what a reader was handed must not
    # reach the counter, or one test could poison another's guard.
    after[10_000] = 1
    assert 10_000 not in inference.forward_row_histogram()
