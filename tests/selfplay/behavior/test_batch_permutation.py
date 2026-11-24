"""Batch assembly must return each worker its OWN answer (KOM-121).

This is the fidelity test the KOM-021 design names first in its P0a acceptance
surface: "batch-assembly permutation test (stub net encodes input in output)".
It is the one test in this suite a scatter/gather bug cannot survive.

Why it needs a real multi-process rig rather than a unit test of the batching
helper: the failure mode is a *pairing* bug -- requests gathered in doorbell
scan order, responses written back in tensor-row order, and the two orders
drifting apart the moment a batch is assembled from more than one worker.
Every piece of that (the per-slot request/response counters, the shared-memory
slots, the row packing) has to be live for the pairing to be exercised at all.
A stub net
whose output is a hash of its input turns the pairing into an assertion each
worker can make locally: I submitted `code`, so every channel of the response
I read must decode back to `code`.

Three independent channels carry the identity (see
``inference_rig.EncoderStub``): the policy row, ``moves_left``, and -- through
the server's softmax -- the ``wdl`` argmax.  A transposed batch, a stale
buffer, an off-by-one in the slot offsets and a partially-written response all
land somewhere in that triple.

Run twice: ``max_batch=16`` (the server may assemble multi-worker batches) and
``max_batch=1`` (strict serialisation).  Same assertions both times -- the seam
promises the same answers regardless of how requests are grouped, which is the
per-logit half of DESIGN.html section 8's "multi-consumer inference == serial
inference" claim, stated here as exact identity because the stub net is exact.

Note on what is NOT asserted: the *actual* batch size the server assembled.
The pinned seam exposes no hook for it, and design F16 caps the effective batch
at ``min(max_batch, live workers)`` anyway, so at four workers the two
configurations differ by "up to 4 per batch" vs "exactly 1".  What is asserted
instead is that every submission completed -- see the done-count check, which
is what keeps a server that quietly answered nothing from passing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import inference_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "ShardServer"):
    pytest.skip(
        "komugi.selfplay exports no ShardServer yet (KOM-021a)",
        allow_module_level=True,
    )

_WORKERS = 4
_PER_WORKER = 8

#: Codes are ``worker_id * 16 + k + 1`` so no two submissions in the run share
#: one, and 1 is the smallest (a zero code would be indistinguishable from an
#: unwritten slot -- the bug most worth catching).
_CODE_STRIDE = 16


def _codes(worker_id: int) -> list[int]:
    return [worker_id * _CODE_STRIDE + k + 1 for k in range(_PER_WORKER)]


def _worker_main(
    config: Any,
    handle: Any,
    barrier: Any,
    errors: Any,
    dones: Any,
) -> None:
    """Submit this worker's codes and check every response decodes to it."""
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        legal = rig.all_legal()
        # Co-arrival: without it the workers trickle in one at a time and the
        # server never has a reason to batch across them.
        barrier.wait(rig.BARRIER_WALL_S)
        done = 0
        for code in _codes(worker_id):
            response = client.submit(rig.encode_obs(code), legal)
            problem = rig.check_response(response, code)
            if problem is not None:
                errors.put(f"worker {worker_id} submitted code {code}: {problem}")
                return
            done += 1
        dones.put((worker_id, done))
    except BaseException as exc:  # reported, not raised: the parent asserts
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("max_batch", [16, 1])
def test_every_worker_gets_its_own_response(max_batch: int) -> None:
    """Four workers, eight distinct observations each, no crossed wires."""
    # A wide batch window on the max_batch=16 leg makes genuinely multi-row
    # batches near-certain with barrier-released workers; without it, a loaded
    # CI can degenerate this leg into 32 batches of one and prove nothing
    # about scatter (wave-9a review F6). The max_batch=1 leg stays tight.
    timeout_ms = 50.0 if max_batch > 1 else 2.0
    config = rig.make_config(
        max_batch=max_batch, batch_timeout_ms=timeout_ms, request_timeout_s=60.0
    )
    worker_ids = list(range(_WORKERS))

    with rig.slots(config, worker_ids) as handles:
        stop_event = rig.CTX.Event()
        barrier = rig.CTX.Barrier(_WORKERS)
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
                barrier,
                errors,
                dones,
                name=f"worker-{handle.worker_id}",
            )
            for handle in handles
        ]

        failures = rig.join_all(workers, rig.SPAWN_WALL_S, errors)
        # collect rather than drain: joinable and readable are not the same
        # instant for a queue the child wrote to just before exiting.
        reported = dict(rig.collect(dones, _WORKERS, rig.JOIN_GRACE_S))
        server_alive = server.is_alive()
        failures.extend(rig.stop_server(server, stop_event))

    assert failures == [], f"max_batch={max_batch}: {failures}"
    # Anti-vacuity: a server that answered nothing, or a client that skipped
    # its loop, would otherwise leave a green with zero round trips behind it.
    assert reported == {wid: _PER_WORKER for wid in worker_ids}
    assert server_alive, "the server died before the workers finished"


def test_encoder_stub_is_actually_injective() -> None:
    """The rig's own premise: distinct codes give distinguishable responses.

    If the stub ever stopped encoding its input -- a refactor that drops the
    ramp, a dtype that saturates -- the permutation test above would pass
    vacuously.  This pins the stub locally, in-process, against the same
    decoder the workers use.
    """
    model = rig.encoder_factory()
    codes = [1, 2, 3, 17, 56]
    batch = torch.from_numpy(
        np.stack([rig.encode_obs(code) for code in codes]).astype(np.float32)
    )
    out = model(batch)

    for row, code in enumerate(codes):
        response = sp.NetEval(
            policy_logits=out.policy_logits[row].numpy(),
            wdl=torch.softmax(out.wdl_logits[row], dim=-1).numpy(),
            moves_left=out.moves_left[row].numpy(),
        )
        assert rig.check_response(response, code) is None

    # ... and a mismatched pairing is detected, which is the property the
    # spawned assertions rely on.
    crossed = sp.NetEval(
        policy_logits=out.policy_logits[0].numpy(),
        wdl=torch.softmax(out.wdl_logits[0], dim=-1).numpy(),
        moves_left=out.moves_left[0].numpy(),
    )
    assert rig.check_response(crossed, codes[1]) is not None
