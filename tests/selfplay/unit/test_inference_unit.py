"""Unit tests for the inference seam: shapes, arithmetic, and the byte layout.

Everything here runs in one process. The seam's *process-level* behaviour --
batch assembly across workers, timeout-vs-max-batch, a SIGKILLed worker leaving
its siblings' segments readable -- belongs to the KOM-121 behavioural suite; a
unit test that spawned processes would be a slow, flaky copy of it.

What is left is exactly what a unit test is good at: the field order the seam
promises, the config's refusals, the utility arithmetic done by hand, one real
(tiny) network through the evaluator, a round trip through the shared-memory
layout using the same pack/read helpers the client and server use, and -- since
KOM-021b factored :func:`~komugi.selfplay.inference.serve_once` out of the
serve loop -- the doorbell protocol itself, driven by a pump *thread* rather
than a server process.
"""

from __future__ import annotations

import threading
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay import inference
from komugi.selfplay.inference import (
    InferenceConfig,
    InProcessEvaluator,
    NetEval,
    pack_request,
    read_request,
    read_response,
    slot_nbytes,
    wdl_utility,
    write_response,
)

IN_PLANES = 4
NUM_ACTIONS = 81


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net -- mirrors tests/train/test_netoutput_wiring.py."""
    trunk = TrunkConfig(in_planes=IN_PLANES, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=NUM_ACTIONS, policy_planes=1, seed=7)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _config() -> InferenceConfig:
    """An :class:`InferenceConfig` shaped for the tiny net."""
    return InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)


# --------------------------------------------------------------------------
# The seam's shape
# --------------------------------------------------------------------------


def test_neteval_field_order_is_pinned() -> None:
    """Callers unpack positionally, so the field order is part of the contract."""
    assert NetEval._fields == ("policy_logits", "wdl", "moves_left")


def test_neteval_carries_no_scalar_value() -> None:
    """Design D1/F3: wdl travels, utility is derived by the reader."""
    assert "value" not in NetEval._fields
    # ... and the training-only auxiliary head does not cross the seam either.
    assert "soft_policy_logits" not in NetEval._fields


# --------------------------------------------------------------------------
# InferenceConfig
# --------------------------------------------------------------------------


def test_non_positive_counts_are_rejected() -> None:
    """A zero or negative count is a configuration error, not a resizing request."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="in_planes"):
            InferenceConfig(in_planes=bad, num_actions=NUM_ACTIONS)
        with pytest.raises(ValueError, match="num_actions"):
            InferenceConfig(in_planes=IN_PLANES, num_actions=bad)
        with pytest.raises(ValueError, match="max_batch"):
            InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS, max_batch=bad)
        with pytest.raises(ValueError, match="shards"):
            InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS, shards=bad)


def test_negative_durations_are_rejected() -> None:
    """Negative patience is meaningless; zero is a legitimate 'do not wait'."""
    with pytest.raises(ValueError, match="batch_timeout_ms"):
        InferenceConfig(
            in_planes=IN_PLANES, num_actions=NUM_ACTIONS, batch_timeout_ms=-1.0
        )
    with pytest.raises(ValueError, match="request_timeout_s"):
        InferenceConfig(
            in_planes=IN_PLANES, num_actions=NUM_ACTIONS, request_timeout_s=-1.0
        )
    assert InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        batch_timeout_ms=0.0,
        request_timeout_s=0.0,
    )


def test_fp16_policy_transfer_is_a_deferred_knob() -> None:
    """Design D8: the seam carries the knob, P0a refuses it, citing the ticket."""
    with pytest.raises(NotImplementedError, match="KOM-021"):
        InferenceConfig(
            in_planes=IN_PLANES, num_actions=NUM_ACTIONS, policy_dtype="f16"
        )


def test_sparse_policy_is_no_longer_a_deferred_knob() -> None:
    """Design D8's other half, implemented by KOM-038: it constructs, it sticks.

    The pair above and here is the point. ``policy_dtype='f16'`` is still
    reserved and still refused; ``sparse_policy`` is not -- the sparse transport
    exists, so a config that asks for it must be accepted rather than raise.
    (Accepted, not enabled: the default stays ``False`` until the on-hardware
    bitwise comparison of the device-side pack, which the CPU gate structurally
    cannot reach.)
    """
    config = InferenceConfig(
        in_planes=IN_PLANES, num_actions=NUM_ACTIONS, sparse_policy=True
    )
    assert config.sparse_policy is True


def test_default_config_is_the_documented_one() -> None:
    """P0a defaults: one shard, batch 16, 0.5ms collection window.

    KOM-032 moved the window from 2.0 to 0.5: with the pipelined server the
    collect that matters overlaps the forward and ignores this window entirely,
    so all it still buys is a straggler or two at the start of a burst.
    """
    config = _config()
    assert (config.max_batch, config.shards, config.batch_timeout_ms) == (16, 1, 0.5)
    assert (config.policy_dtype, config.sparse_policy) == ("f32", False)


def test_the_poll_intervals_are_the_bench_adopted_ones() -> None:
    """Both poll constants are bench-adopted values, pinned so a change is a re-bench.

    KOM-040(a)'s poll matrix (2026-08-25) measured seven cells against a doubled
    control that pinned run-to-run noise below 0.1%:

    * the **server's** idle poll at 0.2 ms produced 18,018 games/day against
      17,496 at the 1 ms this started at (+3.0%), replicated at 18,017 on a
      confirmation run. The mechanism is the one the KOM-032 and KOM-038 reviews
      flagged from the code: a request ringing just after an empty scan waits out
      the whole sleep, ~0.5 ms of completion-detection latency per cycle;
    * the **client's** poll refused the same treatment -- 0.1 ms measured -0.7%
      and 0.05 ms -2.8%, because 28 worker-side pollers spend on wakeups the CPU
      the workers need to generate the next request. 0.5 ms stays, deliberately.

    So neither number is free to drift: this test exists so that changing one is
    a decision to re-measure, not an edit that passes quietly.
    """
    assert inference._SERVER_IDLE_POLL_S == 0.0002
    assert inference._CLIENT_POLL_S == 0.0005


# --------------------------------------------------------------------------
# wdl -> utility (design D1, DESIGN.html §3.3)
# --------------------------------------------------------------------------


def test_wdl_utility_is_hand_computable() -> None:
    """``w*1 + d*draw_score + l*(-1)``, three rows, by hand."""
    wdl = np.array(
        [
            [1.0, 0.0, 0.0],  # certain win        -> 1.0
            [0.0, 1.0, 0.0],  # certain draw       -> draw_score
            [0.25, 0.5, 0.25],  # 0.25 - 0.25 + 0.5*0.2 -> 0.1
        ],
        dtype=np.float32,
    )
    utility = wdl_utility(wdl, draw_score=0.2)
    assert utility.shape == (3,)
    assert utility.dtype == np.float32
    np.testing.assert_allclose(utility, [1.0, 0.2, 0.1], rtol=0, atol=1e-6)


def test_wdl_utility_follows_the_sign_of_draw_score() -> None:
    """A negative draw score makes a certain draw a negative utility."""
    wdl = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(
        wdl_utility(wdl, draw_score=-0.5), [-0.5, -1.0], rtol=0, atol=1e-6
    )


def test_wdl_utility_rejects_a_non_wdl_shape() -> None:
    """The (N, 4) mis-shape the spawned-worker shape gate hunts for (design D7)."""
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        wdl_utility(np.zeros((2, 4), dtype=np.float32), draw_score=0.2)


# --------------------------------------------------------------------------
# InProcessEvaluator
# --------------------------------------------------------------------------


def test_evaluator_returns_probabilities_and_the_right_shapes() -> None:
    """One forward of a real (tiny) net, straight through the seam."""
    config = _config()
    evaluator = InProcessEvaluator(_tiny_net(), config)
    obs = np.zeros((3, IN_PLANES, 9, 9), dtype=np.float32)
    legal = np.ones((3, NUM_ACTIONS), dtype=np.bool_)

    result = evaluator.evaluate(obs, legal)

    assert result.policy_logits.shape == (3, NUM_ACTIONS)
    assert result.wdl.shape == (3, 3)
    assert result.moves_left.shape == (3,)
    for array in result:
        assert array.dtype == np.float32
    # Probabilities, not logits: every reader must agree what a draw
    # probability is before draw_score is applied.
    np.testing.assert_allclose(result.wdl.sum(axis=1), np.ones(3), rtol=0, atol=1e-6)
    assert bool((result.wdl >= 0.0).all())


def test_evaluator_puts_the_model_in_eval_mode() -> None:
    """A forward in train mode would corrupt the run via batchnorm statistics."""
    net = _tiny_net()
    net.train()
    InProcessEvaluator(net, _config())
    assert not net.training


def test_evaluator_does_not_build_a_graph() -> None:
    """Inference only: no autograd graph escapes the seam."""
    net = _tiny_net()
    evaluator = InProcessEvaluator(net, _config())
    evaluator.evaluate(
        np.zeros((1, IN_PLANES, 9, 9), dtype=np.float32),
        np.ones((1, NUM_ACTIONS), dtype=np.bool_),
    )
    assert all(parameter.grad is None for parameter in net.parameters())


@pytest.mark.parametrize(
    ("obs_shape", "legal_shape"),
    [
        ((2, IN_PLANES + 1, 9, 9), (2, NUM_ACTIONS)),  # wrong plane count
        ((2, IN_PLANES, 9, 9), (2, NUM_ACTIONS + 1)),  # wrong action count
        ((2, IN_PLANES, 9, 9), (3, NUM_ACTIONS)),  # mask/batch disagreement
    ],
)
def test_evaluator_rejects_mis_shaped_input(
    obs_shape: tuple[int, ...], legal_shape: tuple[int, ...]
) -> None:
    """Shape errors surface here, named, rather than deep inside torch."""
    evaluator = InProcessEvaluator(_tiny_net(), _config())
    with pytest.raises(ValueError):
        evaluator.evaluate(
            np.zeros(obs_shape, dtype=np.float32), np.ones(legal_shape, dtype=np.bool_)
        )


def test_evaluator_matches_a_direct_forward() -> None:
    """The seam is a transport, not a transformation (bar the wdl softmax)."""
    net = _tiny_net()
    net.eval()
    obs = np.arange(2 * IN_PLANES * 81, dtype=np.float32).reshape(2, IN_PLANES, 9, 9)
    obs /= float(obs.size)
    with torch.no_grad():
        direct = net(torch.from_numpy(obs))

    result = InProcessEvaluator(net, _config()).evaluate(
        obs, np.ones((2, NUM_ACTIONS), dtype=np.bool_)
    )

    np.testing.assert_allclose(
        result.policy_logits, direct.policy_logits.numpy(), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result.wdl,
        torch.softmax(direct.wdl_logits, dim=-1).numpy(),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.moves_left, direct.moves_left.numpy(), rtol=0, atol=1e-6
    )


# --------------------------------------------------------------------------
# The shared-memory slot layout
# --------------------------------------------------------------------------


def test_slot_nbytes_arithmetic_is_golden() -> None:
    """Doorbells + row_count + request + row_count_echo + counts + response.

    KOM-038 added the ``policy_count`` u32 between the alignment pad and the
    response floats. It is there **unconditionally** -- both transport modes
    size the segment identically, which is what makes ``sparse_policy`` a
    configuration change rather than a re-plumbing.

    KOM-040 b2 (layout v2) adds two more unconditional u32 headers, for the
    same reason and with the same consequence: a ``row_count`` on the request
    side and a ``row_count_echo`` on the response side, so a ticket can carry a
    whole wave and the client can check the width it got back. Both are present
    at every ``max_wave``, including the default 1, so **every slot is exactly
    eight bytes wider than it was in v1** -- and the goldens below are re-derived
    from the layout rather than bumped by eight, which is the KOM-038 lesson.
    """
    # tiny net, max_wave=1: 8 request_seq + 8 response_seq, 4 row_count,
    # 1*4*4*81 = 1296 obs, 1*81 legal -> padded 84, 4 row_count_echo,
    # 4*1 policy_count, 4*1*81 = 324 policy, 4*1*3 wdl, 4*1 moves-left.
    assert slot_nbytes(_config()) == 16 + 4 + 1296 + 84 + 4 + 4 + 324 + 12 + 4 == 1748
    # production shapes: 128 planes, the real action space -- 16 doorbells,
    # 4 row_count, 4*128*81 = 41472 obs, 37341 legal -> padded 37344,
    # 4 row_count_echo, 4 policy_count, 4*37341 = 149364 policy, 12 wdl,
    # 4 moves-left.
    assert (
        slot_nbytes(InferenceConfig(in_planes=128, num_actions=37341))
        == 16 + 4 + 41472 + 37344 + 4 + 4 + 149364 + 12 + 4
        == 228224
    )


def test_slot_offsets_do_not_overlap_and_stay_aligned() -> None:
    """Doorbells 8-byte aligned, every float32 view 4-byte; nothing overlaps."""
    config = InferenceConfig(in_planes=128, num_actions=37341)
    layout = inference._layout(config)
    assert layout.request_seq == 0
    assert layout.response_seq == 8
    # The request header sits between the doorbells and the observation, so the
    # worker owns one contiguous run of bytes and the server owns the other.
    assert layout.row_count == 16
    assert layout.obs == 20
    assert layout.legal == 20 + 4 * 128 * 81
    assert layout.legal + config.num_actions <= layout.row_count_echo
    assert layout.policy_count == layout.row_count_echo + 4
    assert layout.policy == layout.policy_count + 4
    assert layout.policy + 4 * config.num_actions == layout.wdl
    assert layout.wdl + 12 == layout.moves_left
    assert layout.moves_left + 4 == layout.nbytes
    for offset in (
        layout.row_count,
        layout.obs,
        layout.row_count_echo,
        layout.policy_count,
        layout.policy,
        layout.wdl,
        layout.moves_left,
    ):
        assert offset % 4 == 0
    # The doorbells' 8-byte alignment is the safety argument, not a detail: an
    # aligned 8-byte store is atomic on x86-64, so a torn counter -- the one way
    # a stale response could pass for a fresh one -- cannot be observed.
    for offset in (layout.request_seq, layout.response_seq):
        assert offset % 8 == 0


def test_doorbells_start_at_zero_on_a_fresh_segment() -> None:
    """A worker's first ticket is 1, so a zeroed response_seq can never match."""
    config = _config()
    created = inference.create_slots(config)
    segment = created.shm
    try:
        assert int(created.request_seq[0]) == 0
        assert int(created.response_seq[0]) == 0
        assert created.request_seq.dtype == np.uint64
        assert created.response_seq.dtype == np.uint64
    finally:
        created.close()
        # unlink() through the creator's own handle: it unregisters the name
        # this process registered at create, which keeps the resource tracker's
        # books balanced (see create_slots' docstring on bpo-38119).
        segment.unlink()


def _random_request(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A distinguishable observation and a non-trivial legality mask."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((IN_PLANES, 9, 9)).astype(np.float32)
    legal = rng.random(NUM_ACTIONS) < 0.3
    return obs, np.asarray(legal, dtype=np.bool_)


def test_request_round_trips_through_the_layout() -> None:
    """The client's write path and the server's gather path agree, byte for byte.

    A plain :class:`SharedMemory` stands in for a segment the orchestrator would
    own; no process is spawned, and the helpers under test are the same ones
    :meth:`SlotClient.submit` and :meth:`ShardServer.serve` call.
    """
    config = _config()
    shm = SharedMemory(create=True, size=slot_nbytes(config))
    try:
        slots = inference._views(shm, config)
        obs, legal = _random_request(seed=11)

        pack_request(slots, obs, legal)
        read_obs, read_legal = read_request(slots)

        np.testing.assert_array_equal(read_obs, obs)
        np.testing.assert_array_equal(read_legal, legal)
        # The server's copy must survive the worker overwriting its slot.
        pack_request(slots, np.zeros_like(obs), np.zeros_like(legal))
        np.testing.assert_array_equal(read_obs, obs)
        slots.close()
    finally:
        shm.unlink()


def test_response_round_trips_with_the_batch_dimension_stripped() -> None:
    """The server scatters row `i`; the worker reads one position's worth."""
    config = _config()
    shm = SharedMemory(create=True, size=slot_nbytes(config))
    try:
        slots = inference._views(shm, config)
        batched = NetEval(
            policy_logits=np.arange(2 * NUM_ACTIONS, dtype=np.float32).reshape(
                2, NUM_ACTIONS
            ),
            wdl=np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=np.float32),
            moves_left=np.array([7.0, 42.0], dtype=np.float32),
        )

        write_response(slots, batched, row=1)
        result = read_response(slots)

        assert result.policy_logits.shape == (NUM_ACTIONS,)
        assert result.wdl.shape == (3,)
        assert result.moves_left.shape == ()
        np.testing.assert_array_equal(result.policy_logits, batched.policy_logits[1])
        np.testing.assert_array_equal(result.wdl, batched.wdl[1])
        assert float(result.moves_left) == 42.0
        # Copies, not views: the next response must not mutate this one.
        write_response(slots, batched, row=0)
        np.testing.assert_array_equal(result.wdl, batched.wdl[1])
        slots.close()
    finally:
        shm.unlink()


def test_attach_slots_sees_the_creators_bytes() -> None:
    """An attached view aliases the same segment (and is not tracker-owned)."""
    config = _config()
    created = inference.create_slots(config)
    try:
        obs, legal = _random_request(seed=3)
        pack_request(created, obs, legal)

        attached = inference.attach_slots(created.shm.name, config)
        try:
            attached_obs, attached_legal = read_request(attached)
            np.testing.assert_array_equal(attached_obs, obs)
            np.testing.assert_array_equal(attached_legal, legal)
            # ... and the alias is live in both directions.
            attached.policy[:] = 1.5
            # .flat[0]: the policy view is (max_wave, A) under layout v2, and
            # this assertion is about aliasing, not shape.
            assert float(created.policy.flat[0]) == 1.5
        finally:
            attached.close()
    finally:
        name = created.shm.name
        created.close()
        SharedMemory(name=name).unlink()


def test_pack_request_rejects_a_mis_shaped_observation() -> None:
    """Slot writes are shape-checked before they corrupt a neighbouring field."""
    config = _config()
    shm = SharedMemory(create=True, size=slot_nbytes(config))
    try:
        slots = inference._views(shm, config)
        with pytest.raises(ValueError, match="obs"):
            pack_request(
                slots,
                np.zeros((IN_PLANES + 1, 9, 9), dtype=np.float32),
                np.zeros(NUM_ACTIONS, dtype=np.bool_),
            )
        with pytest.raises(ValueError, match="legal"):
            pack_request(
                slots,
                np.zeros((IN_PLANES, 9, 9), dtype=np.float32),
                np.zeros(NUM_ACTIONS + 1, dtype=np.bool_),
            )
        slots.close()
    finally:
        shm.unlink()


def test_views_reject_a_segment_that_is_too_small() -> None:
    """A size/config mismatch is caught at attach, not at first write."""
    config = _config()
    shm = SharedMemory(create=True, size=slot_nbytes(config) - 4)
    try:
        with pytest.raises(ValueError, match="bytes"):
            inference._views(shm, config)
    finally:
        shm.close()
        shm.unlink()


# --------------------------------------------------------------------------
# ShardServer's effective batch ceiling (design D2/F16) -- no processes needed
# --------------------------------------------------------------------------


def test_effective_max_batch_cannot_exceed_the_worker_count() -> None:
    """Each worker has one request in flight, so more batch than workers is dead."""
    config = InferenceConfig(
        in_planes=IN_PLANES, num_actions=NUM_ACTIONS, max_batch=16
    )
    # A handle is a name and an id, nothing more: no event to fake, and nothing
    # here has to build a multiprocessing context to construct one.
    server = inference.ShardServer(_tiny_net, config, [inference.SlotHandle("nope", 0)])
    assert server.max_batch == 1
    assert (
        inference.ShardServer(
            _tiny_net,
            InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS, max_batch=2),
            [inference.SlotHandle(f"nope{i}", i) for i in range(4)],
        ).max_batch
        == 2
    )


# --------------------------------------------------------------------------
# The doorbell protocol (KOM-021b) -- still no processes, only a pump thread
# --------------------------------------------------------------------------


def _pump_one(
    slots_map: dict[int, inference.Slots],
    evaluator: InProcessEvaluator,
    config: InferenceConfig,
    wall_s: float = 30.0,
) -> None:
    """Run :func:`serve_once` until it serves one request, or the wall expires.

    A thread, not a process: ``submit`` blocks by design, so something else has
    to be the server. The thread is the *only* concession this file makes to
    concurrency, and it runs the same ``serve_once`` the real loop runs.
    """
    deadline = time.monotonic() + wall_s
    while time.monotonic() < deadline:
        if inference.serve_once(slots_map, evaluator, config) > 0:
            return
        time.sleep(0.0005)


def test_submit_writes_the_payload_then_rings_the_doorbell() -> None:
    """request_seq increments exactly once per submit; response_seq mirrors it.

    The two halves of the ordering rule, pinned separately:

    * a payload written WITHOUT a doorbell is invisible -- ``serve_once``
      returns 0, which is what makes "counter strictly after payload" safe
      rather than merely tidy;
    * one submit advances ``request_seq`` by exactly one, and the server
      publishes that same value as ``response_seq`` after the response bytes,
      so ``response_seq == request_seq`` identifies the answer to *that*
      request.
    """
    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=4,
        batch_timeout_ms=0.0,
        request_timeout_s=30.0,
    )
    # The creator's own views stand in for the server's attached ones -- one
    # attach fewer, which keeps the resource tracker balanced against the
    # client's attach_untracked (create_slots' docstring, bpo-38119).
    server_slots = inference.create_slots(config)
    name = server_slots.shm.name
    try:
        client = inference.SlotClient(name, 0, config)
        try:
            evaluator = InProcessEvaluator(_tiny_net(), config)
            slots_map = {0: server_slots}
            obs, legal = _random_request(seed=5)

            # A payload with no doorbell is not a request.
            pack_request(server_slots, obs, legal)
            assert inference.serve_once(slots_map, evaluator, config) == 0
            assert int(server_slots.request_seq[0]) == 0

            for expected_ticket in (1, 2):
                pump = threading.Thread(
                    target=_pump_one, args=(slots_map, evaluator, config), daemon=True
                )
                pump.start()
                response = client.submit(obs, legal)
                pump.join(60.0)

                assert not pump.is_alive(), "serve_once never picked the request up"
                assert int(server_slots.request_seq[0]) == expected_ticket
                assert int(server_slots.response_seq[0]) == expected_ticket
                # ... and the bytes the counter announced are the right ones.
                want = evaluator.evaluate(obs[None], legal[None])
                np.testing.assert_allclose(
                    response.policy_logits, want.policy_logits[0], rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(response.wdl, want.wdl[0], rtol=0, atol=1e-6)
        finally:
            client.close()
            server_slots.close()
    finally:
        SharedMemory(name=name).unlink()


def test_serve_once_answers_every_pending_slot_in_one_forward() -> None:
    """Two rung doorbells, one batch, each answer landing in its own slot."""
    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=4,
        batch_timeout_ms=0.0,
    )
    slots_map: dict[int, inference.Slots] = {}
    segments: list[SharedMemory] = []
    try:
        for worker_id in (0, 1):
            created = inference.create_slots(config)
            segments.append(created.shm)
            slots_map[worker_id] = created

        requests = {
            worker_id: _random_request(seed=20 + worker_id) for worker_id in (0, 1)
        }
        for worker_id, (obs, legal) in requests.items():
            pack_request(slots_map[worker_id], obs, legal)
            slots_map[worker_id].request_seq[0] = 1

        evaluator = InProcessEvaluator(_tiny_net(), config)
        assert inference.serve_once(slots_map, evaluator, config) == 2
        # Nothing pending any more: response_seq caught request_seq up, so a
        # second scan finds no work rather than re-serving the same rows.
        assert inference.serve_once(slots_map, evaluator, config) == 0

        for worker_id, (obs, legal) in requests.items():
            assert int(slots_map[worker_id].response_seq[0]) == 1
            want = evaluator.evaluate(obs[None], legal[None])
            got = read_response(slots_map[worker_id])
            np.testing.assert_allclose(
                got.policy_logits, want.policy_logits[0], rtol=0, atol=1e-6
            )
    finally:
        for entry in slots_map.values():
            entry.close()
        for segment in segments:
            segment.unlink()


def test_a_full_batch_serves_the_least_served_workers_first() -> None:
    """More pending workers than the ceiling: nobody can be starved.

    The shared request queue this protocol replaced was FIFO, so fairness came
    for free; a doorbell scan has a fixed slot order, and taking the first
    ``max_batch`` hits every iteration would let low-numbered workers starve
    their siblings all the way to ``InferenceTimeout``. ``request_seq`` doubles
    as the fairness key: a worker passed over keeps its ticket while the served
    ones climb, so it strictly rises and must eventually win.
    """
    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=1,
        batch_timeout_ms=0.0,
    )
    slots_map: dict[int, inference.Slots] = {}
    segments: list[SharedMemory] = []
    try:
        obs, legal = _random_request(seed=31)
        # Worker 0 has already been served twice, worker 1 once, worker 2 never
        # -- and all three are pending. Scan order would pick 0 every time.
        for worker_id, already_served in ((0, 2), (1, 1), (2, 0)):
            created = inference.create_slots(config)
            segments.append(created.shm)
            pack_request(created, obs, legal)
            created.response_seq[0] = already_served
            created.request_seq[0] = already_served + 1
            slots_map[worker_id] = created

        evaluator = InProcessEvaluator(_tiny_net(), config)
        # Ceiling is 1 (max_batch=1), so one worker per forward, least served
        # first: 2, then 1, then 0.
        for expected in (2, 1, 0):
            assert inference.serve_once(slots_map, evaluator, config) == 1
            entry = slots_map[expected]
            assert int(entry.response_seq[0]) == int(entry.request_seq[0]), (
                f"worker {expected} was not the one served"
            )
        # Everyone answered, nothing left pending.
        assert inference.serve_once(slots_map, evaluator, config) == 0
    finally:
        for entry in slots_map.values():
            entry.close()
        for segment in segments:
            segment.unlink()


def test_submit_times_out_and_poisons_the_client_with_no_server() -> None:
    """Nobody moves response_seq, so the poll runs out and the client stays dead.

    The in-process half of the KOM-121 timeout-policy behaviour: the wall here
    is only that the second submit must refuse *without* waiting out another
    window, which is what makes a stale response unreachable (design D2/F2).
    """
    config = InferenceConfig(
        in_planes=IN_PLANES, num_actions=NUM_ACTIONS, request_timeout_s=0.2
    )
    created = inference.create_slots(config)
    name = created.shm.name
    created.close()
    try:
        client = inference.SlotClient(name, 0, config)
        try:
            obs, legal = _random_request(seed=9)
            started = time.monotonic()
            with pytest.raises(inference.InferenceTimeout):
                client.submit(obs, legal)
            assert time.monotonic() - started >= 0.2

            started = time.monotonic()
            with pytest.raises(inference.InferenceTimeout, match="poisoned"):
                client.submit(obs, legal)
            assert time.monotonic() - started < 0.1
        finally:
            client.close()
    finally:
        SharedMemory(name=name).unlink()


def test_a_lane_client_names_its_worker_and_lane_not_the_bare_slot() -> None:
    """Under lanes, slot 3 is worker 1 lane 1 -- and must say so (review F3).

    The orchestrator numbers segments ``worker * games_in_flight + lane``, so a
    diagnostic that interpolated the slot id after the word "worker" sent an
    operator to the wrong process's output: at two lanes, "worker 3" of a
    two-worker run names a worker that does not exist. Both of this client's
    failure messages -- the timeout and the refusal that follows it -- are
    driven here, because the incident starts with whichever one is read first.
    """
    config = InferenceConfig(
        in_planes=IN_PLANES, num_actions=NUM_ACTIONS, request_timeout_s=0.05
    )
    created = inference.create_slots(config)
    name = created.shm.name
    created.close()
    try:
        # Slot 3 of a games_in_flight=2 run: worker 1's second lane.
        client = inference.SlotClient(name, 3, config, lanes=2)
        try:
            obs, legal = _random_request(seed=11)
            with pytest.raises(inference.InferenceTimeout) as timed_out:
                client.submit(obs, legal)
            with pytest.raises(inference.InferenceTimeout) as refused:
                client.submit(obs, legal)
            for message in (str(timed_out.value), str(refused.value)):
                assert "worker 1 lane 1" in message, message
                assert "slot 3" in message, message
                assert "worker 3" not in message, message
        finally:
            client.close()
    finally:
        SharedMemory(name=name).unlink()


@pytest.mark.parametrize(
    ("slot_id", "lanes", "expected"),
    [
        (0, 1, "slot 0"),
        (3, 1, "slot 3"),
        (0, 2, "worker 0 lane 0 (slot 0)"),
        (3, 2, "worker 1 lane 1 (slot 3)"),
        (4, 2, "worker 2 lane 0 (slot 4)"),
    ],
)
def test_slot_label_never_calls_a_slot_a_worker_without_the_geometry(
    slot_id: int, lanes: int, expected: str
) -> None:
    """The label is honest in both directions: both identities, or neither.

    At ``lanes == 1`` the slot id happens to equal the worker id, but a caller
    that was never told the geometry cannot distinguish that run from an
    interleaved one -- so the number is named for what it certainly is.
    """
    assert inference.slot_label(slot_id, lanes) == expected


def test_the_doorbell_rings_strictly_after_the_payload_is_packed() -> None:
    """``pack_request`` must observe the PRE-increment ``request_seq``.

    The payload-before-counter ordering is the protocol (module docstring),
    but line order is its only enforcement; the KOM-021b review (F1) showed a
    swapped order passes every other test with overwhelming probability
    because the race window is ~1000x smaller than any poll. This pins it
    deterministically: intercept ``pack_request`` and record the counter AT
    CALL TIME -- if the doorbell ever rings first, the recorded value equals
    the new ticket and this fails on every run, no race required.
    """
    from unittest import mock

    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=4,
        batch_timeout_ms=0.0,
        request_timeout_s=30.0,
    )
    server_slots = inference.create_slots(config)
    name = server_slots.shm.name
    try:
        client = inference.SlotClient(name, 0, config)
        try:
            evaluator = InProcessEvaluator(_tiny_net(), config)
            slots_map = {0: server_slots}
            obs, legal = _random_request(seed=11)

            seen: list[int] = []
            real_pack = inference.pack_wave_request

            def recording_pack(
                slots: inference.Slots,
                packed_obs: npt.NDArray[np.float32],
                packed_legal: npt.NDArray[np.bool_],
            ) -> int:
                seen.append(int(slots.request_seq[0]))
                return real_pack(slots, packed_obs, packed_legal)

            # KOM-040 b2: the one request writer is `pack_wave_request`, and
            # `pack_request` is its single-row degenerate case -- so the
            # ordering pin intercepts the wave writer, which is the function
            # every submit (wave or scalar) actually goes through.
            with mock.patch.object(inference, "pack_wave_request", recording_pack):
                pump = threading.Thread(
                    target=_pump_one, args=(slots_map, evaluator, config), daemon=True
                )
                pump.start()
                client.submit(obs, legal)
                pump.join(60.0)
            assert not pump.is_alive(), "serve_once never picked the request up"
            assert seen == [0], (
                f"pack_request observed request_seq {seen}; a value equal to the "
                f"new ticket means the doorbell rang BEFORE the payload write"
            )
        finally:
            client.close()
            server_slots.close()
    finally:
        SharedMemory(name=name).unlink()


def test_the_response_is_published_strictly_after_its_bytes() -> None:
    """``write_response`` must observe the PRE-publish ``response_seq``.

    The server-side mirror of the ordering pin above (KOM-021b review F1):
    the response bytes go into the slot first, ``response_seq = ticket``
    second. Intercepting ``write_response`` and reading the counter at call
    time makes a swapped publish fail deterministically.
    """
    from unittest import mock

    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=4,
        batch_timeout_ms=0.0,
        request_timeout_s=30.0,
    )
    server_slots = inference.create_slots(config)
    name = server_slots.shm.name
    try:
        client = inference.SlotClient(name, 0, config)
        try:
            evaluator = InProcessEvaluator(_tiny_net(), config)
            slots_map = {0: server_slots}
            obs, legal = _random_request(seed=13)

            seen: list[int] = []
            real_write = inference.write_response

            def recording_write(
                slots: inference.Slots,
                evaluation: NetEval,
                row: int,
                slot_row: int = 0,
            ) -> None:
                seen.append(int(slots.response_seq[0]))
                real_write(slots, evaluation, row, slot_row)

            with mock.patch.object(inference, "write_response", recording_write):
                pump = threading.Thread(
                    target=_pump_one, args=(slots_map, evaluator, config), daemon=True
                )
                pump.start()
                client.submit(obs, legal)
                pump.join(60.0)
            assert not pump.is_alive(), "serve_once never picked the request up"
            assert seen == [0], (
                f"write_response observed response_seq {seen}; a value equal to "
                f"the ticket means the answer was published BEFORE its bytes"
            )
            assert int(server_slots.response_seq[0]) == 1
        finally:
            client.close()
            server_slots.close()
    finally:
        SharedMemory(name=name).unlink()
