"""Unit tests for the sparse-policy transport (KOM-038).

``sparse_policy`` puts only the LEGAL actions' logits on the wire: ``K`` float32
values in ascending action-id order, packed from offset zero of the policy
region, with ``K`` in the slot's ``policy_count`` field and **no index array**
-- the worker rebuilds the positions from the very mask it submitted. That is a
protocol change on both sides, so what this file pins is the protocol rather
than any one function:

* the **codec** -- ascending order against a hand-written reference scatter,
  random masks, and both edges (``K == 1``, ``K == A``);
* the **refusals** -- ``K == 0`` is unreachable by construction (the search
  refuses such a position before it ever asks for an evaluation) and is refused
  loudly anyway, everywhere it could enter;
* the **mode-mismatch tripwire**, in *both* directions. The slot layout is
  byte-identical in both modes on purpose, so a dense server answering a sparse
  worker (or the reverse) would otherwise pass every size check and hand the
  search finite, plausible, wrong priors. ``policy_count`` is the mode tag and
  both readers check it;
* the **equivalence claim** -- a real CPU serve round trip under
  ``sparse_policy=True`` produces, on the legal support, the exact bytes the
  dense serve produces. On the legal support and nowhere else: the illegal
  lanes are fill by design, and the fill is a finite ``0.0`` because the search
  validates the whole vector for finiteness before it masks anything;
* the **dense path**, unchanged and still saying so -- the sentinel is written,
  and the layout ``_views`` computes is identical in both modes.

Everything here runs in one process. Where a server is needed it is a pump
*thread* running the same :func:`~komugi.selfplay.inference.serve_once` or
:func:`~komugi.selfplay.inference.serve_pipelined` production runs, exactly as
the sibling unit suites do -- ``submit`` blocks by design, so something else has
to be the server.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt
import pytest

from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay import inference
from komugi.selfplay.inference import (
    DENSE_POLICY_COUNT,
    AsyncEvaluator,
    InferenceConfig,
    InferenceProtocolError,
    InProcessEvaluator,
    NetEval,
    PackedPolicy,
    Slots,
    pack_sparse_batch,
    read_response,
    read_sparse_response,
    slot_nbytes,
    unpack_sparse_policy,
    write_response,
    write_sparse_response,
)

IN_PLANES = 4
NUM_ACTIONS = 81


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net -- the same one the sibling unit suites use."""
    trunk = TrunkConfig(in_planes=IN_PLANES, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=NUM_ACTIONS, policy_planes=1, seed=7)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _config(
    *,
    sparse: bool,
    max_batch: int = 16,
    batch_timeout_ms: float = 0.5,
    request_timeout_s: float = 60.0,
) -> InferenceConfig:
    """A config shaped for the tiny net, in the requested transport mode."""
    return InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=max_batch,
        batch_timeout_ms=batch_timeout_ms,
        request_timeout_s=request_timeout_s,
        sparse_policy=sparse,
    )


def _random_request(
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A distinguishable observation and a non-trivial legality mask."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((IN_PLANES, 9, 9)).astype(np.float32)
    legal = rng.random(NUM_ACTIONS) < 0.3
    # A mask with no legal move is not a position anyone evaluates; keep the
    # generator honest rather than letting a seed produce an unreachable case.
    legal[seed % NUM_ACTIONS] = True
    return obs, np.asarray(legal, dtype=np.bool_)


def _reference_pack(
    logits: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """The wire format, spelled out: ascending action id, legal only.

    A Python loop rather than ``logits[legal]``, deliberately: the property
    under test is the *order*, and boolean indexing would be the same numpy call
    the implementation makes rather than an independent statement of it.
    """
    return np.array(
        [logits[action] for action in range(logits.shape[0]) if bool(legal[action])],
        dtype=np.float32,
    )


def _reference_scatter(
    packed: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """The reconstruction, spelled out: consume the values in ascending order."""
    dense = np.zeros(legal.shape[0], dtype=np.float32)
    cursor = 0
    for action in range(legal.shape[0]):
        if bool(legal[action]):
            dense[action] = packed[cursor]
            cursor += 1
    assert cursor == packed.shape[0], "the reference consumed the wrong count"
    return dense


def _row_eval(logits: npt.NDArray[np.float32]) -> NetEval:
    """Wrap ``(N, A)`` logits in a :class:`NetEval` with filler wdl/mlh."""
    rows = logits.shape[0]
    wdl = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32), (rows, 1))
    return NetEval(
        policy_logits=logits,
        wdl=wdl,
        moves_left=np.arange(rows, dtype=np.float32),
    )


@contextlib.contextmanager
def _one_slot(config: InferenceConfig) -> Iterator[Slots]:
    """One created segment's views; closed and unlinked afterwards."""
    created = inference.create_slots(config)
    name = created.shm.name
    try:
        yield created
    finally:
        created.close()
        SharedMemory(name=name).unlink()


@contextlib.contextmanager
def _slot_map(
    config: InferenceConfig, worker_ids: Sequence[int]
) -> Iterator[dict[int, Slots]]:
    """One real segment per worker; close and unlink them afterwards.

    The creator's own views stand in for the server's attached ones, which keeps
    the resource tracker balanced (see ``create_slots``' docstring).
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


def _ring(
    slots: Slots, request: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]
) -> int:
    """Do what ``SlotClient.submit`` does up to the doorbell; return the ticket.

    Payload first, counter second -- if this helper ever gets that backwards the
    tests below stop meaning what they say.
    """
    inference.pack_request(slots, request[0], request[1])
    ticket = int(slots.request_seq[0]) + 1
    slots.request_seq[0] = ticket
    return ticket


def _pump_one(
    slots_map: dict[int, Slots],
    evaluator: InProcessEvaluator,
    config: InferenceConfig,
    wall_s: float = 30.0,
) -> None:
    """Run :func:`serve_once` until it serves one request, or the wall expires."""
    deadline = time.monotonic() + wall_s
    while time.monotonic() < deadline:
        if inference.serve_once(slots_map, evaluator, config) > 0:
            return
        time.sleep(0.0005)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 20.0) -> bool:
    """Poll `predicate` until it holds or the wall expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


# --------------------------------------------------------------------------
# The codec: ascending order, round trip, both edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
def test_packing_is_legal_only_in_ascending_action_id_order(seed: int) -> None:
    """C2: the wire carries the legal logits, ascending, and nothing else.

    Checked against :func:`_reference_pack`, which is an explicit ascending
    loop -- so a packing that reversed the order, used ``~legal``, or emitted
    ``(index, value)`` pairs fails here rather than surviving to be caught by a
    round trip that would undo its own mistake.
    """
    rng = np.random.default_rng(seed)
    legal = np.asarray(rng.random((3, NUM_ACTIONS)) < 0.4, dtype=np.bool_)
    legal[:, seed % NUM_ACTIONS] = True
    logits = rng.standard_normal((3, NUM_ACTIONS)).astype(np.float32)

    packed = pack_sparse_batch(_row_eval(logits), legal)

    assert packed.counts.tolist() == [int(row.sum()) for row in legal]
    for row in range(3):
        count = int(packed.counts[row])
        np.testing.assert_array_equal(
            packed.values[row, :count], _reference_pack(logits[row], legal[row])
        )
        # Ascending is a claim about ids, not only about a reference array.
        ids = [action for action in range(NUM_ACTIONS) if bool(legal[row][action])]
        assert ids == sorted(ids)
        np.testing.assert_array_equal(
            packed.values[row, :count], logits[row][np.array(ids, dtype=np.int64)]
        )


@pytest.mark.parametrize("seed", [4, 5, 6, 23, 101])
def test_the_codec_round_trips_and_fills_illegal_lanes_with_finite_zero(
    seed: int,
) -> None:
    """C5: legal lanes are bit-identical; illegal lanes are exactly ``0.0``.

    Not ``-inf``, and not ``NaN``: the search validates the *whole* vector for
    finiteness before it masks anything (design review D1), so a non-finite
    fill would crash on the first leaf of the first search.
    """
    rng = np.random.default_rng(seed)
    legal = np.asarray(rng.random(NUM_ACTIONS) < 0.35, dtype=np.bool_)
    legal[seed % NUM_ACTIONS] = True
    logits = rng.standard_normal(NUM_ACTIONS).astype(np.float32)

    packed = pack_sparse_batch(_row_eval(logits[None]), legal[None])
    count = int(packed.counts[0])
    dense = unpack_sparse_policy(packed.values[0, :count], legal)

    assert dense.shape == (NUM_ACTIONS,)
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense, _reference_scatter(packed.values[0], legal))
    np.testing.assert_array_equal(dense[legal], logits[legal])
    np.testing.assert_array_equal(
        dense[~legal], np.zeros(int((~legal).sum()), dtype=np.float32)
    )
    assert bool(np.isfinite(dense).all())


def test_k_of_one_is_a_single_value_on_the_wire() -> None:
    """The narrowest legal mask: one value, at its own action id, zeros around."""
    logits = np.arange(NUM_ACTIONS, dtype=np.float32) * -1.5
    for action in (0, 37, NUM_ACTIONS - 1):
        legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        legal[action] = True

        packed = pack_sparse_batch(_row_eval(logits[None]), legal[None])

        assert int(packed.counts[0]) == 1
        assert packed.values.shape == (1, 1)
        assert float(packed.values[0, 0]) == float(logits[action])
        dense = unpack_sparse_policy(packed.values[0, :1], legal)
        assert float(dense[action]) == float(logits[action])
        assert float(dense.sum()) == float(logits[action])


def test_k_equal_to_the_action_space_is_the_identity_packing() -> None:
    """C8: an all-legal mask packs to the dense vector itself, and fits exactly.

    The edge the dedicated ``policy_count`` field exists for (design review D2):
    a count carried as a prefix *inside* the policy region would need
    ``4 + 4A`` bytes in a region of ``4A``, so the fully-legal position -- the
    opening of every game -- could not be transported at all.
    """
    logits = np.linspace(-3.0, 3.0, NUM_ACTIONS, dtype=np.float32)
    legal = np.ones(NUM_ACTIONS, dtype=np.bool_)

    packed = pack_sparse_batch(_row_eval(logits[None]), legal[None])

    assert int(packed.counts[0]) == NUM_ACTIONS
    np.testing.assert_array_equal(packed.values[0], logits)
    np.testing.assert_array_equal(unpack_sparse_policy(packed.values[0], legal), logits)

    with _one_slot(_config(sparse=True)) as slots:
        write_sparse_response(slots, packed, row=0)
        assert int(slots.policy_count[0]) == NUM_ACTIONS
        # The identity packing occupies the region exactly -- no truncation, no
        # spill into `wdl`.
        np.testing.assert_array_equal(np.asarray(slots.policy).reshape(-1), logits)
        np.testing.assert_array_equal(
            read_sparse_response(slots, legal).policy_logits, logits
        )
        np.testing.assert_array_equal(np.asarray(slots.wdl).reshape(-1), packed.wdl[0])


def test_k_of_zero_is_refused_everywhere_it_could_enter() -> None:
    """C8: a position with no legal move is never evaluated -- say so, loudly.

    Unreachable in practice (the search refuses such a position before it asks
    for an evaluation), which is precisely why it is worth refusing here: a
    zero-length packed row is indistinguishable from a response that was never
    written at all.
    """
    logits = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
    empty = np.zeros((1, NUM_ACTIONS), dtype=np.bool_)

    with pytest.raises(ValueError, match="K=0"):
        pack_sparse_batch(_row_eval(logits), empty)
    with pytest.raises(ValueError, match="K=0"):
        inference._sparse_indices(empty)
    with pytest.raises(ValueError, match="K=0"):
        unpack_sparse_policy(np.zeros(0, dtype=np.float32), empty[0])

    # ... and on the writing side, where the count arrives pre-computed and the
    # mask is no longer in scope to be re-derived from.
    packed = PackedPolicy(
        values=np.zeros((1, 1), dtype=np.float32),
        counts=np.zeros(1, dtype=np.int64),
        wdl=np.zeros((1, 3), dtype=np.float32),
        moves_left=np.zeros(1, dtype=np.float32),
    )
    with _one_slot(_config(sparse=True)) as slots:
        with pytest.raises(ValueError, match="K=0"):
            write_sparse_response(slots, packed, row=0)

    # A mixed batch is refused too: one empty row poisons the pack, rather than
    # being quietly packed as a row nobody can decode.
    mixed = np.zeros((2, NUM_ACTIONS), dtype=np.bool_)
    mixed[0, 3] = True
    with pytest.raises(ValueError, match=r"rows \[1\]"):
        pack_sparse_batch(_row_eval(np.zeros((2, NUM_ACTIONS), np.float32)), mixed)


def test_the_codec_rejects_a_payload_of_the_wrong_length() -> None:
    """A count that does not match the mask is a bug, not a resizing request."""
    legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    legal[[2, 5, 9]] = True
    for size in (2, 4):
        with pytest.raises(ValueError, match=r"\(3,\)"):
            unpack_sparse_policy(np.zeros(size, dtype=np.float32), legal)
    with pytest.raises(ValueError, match="legal"):
        unpack_sparse_policy(np.zeros(3, dtype=np.float32), legal[None])
    with pytest.raises(ValueError, match="same shape"):
        pack_sparse_batch(
            _row_eval(np.zeros((1, NUM_ACTIONS + 1), np.float32)), legal[None]
        )


def test_the_writer_refuses_a_row_wider_than_the_policy_region() -> None:
    """A count above ``A`` would spill into ``wdl``; it is caught, not clipped."""
    packed = PackedPolicy(
        values=np.zeros((1, NUM_ACTIONS + 1), dtype=np.float32),
        counts=np.full(1, NUM_ACTIONS + 1, dtype=np.int64),
        wdl=np.zeros((1, 3), dtype=np.float32),
        moves_left=np.zeros(1, dtype=np.float32),
    )
    with _one_slot(_config(sparse=True)) as slots:
        with pytest.raises(ValueError, match="policy region holds"):
            write_sparse_response(slots, packed, row=0)


# --------------------------------------------------------------------------
# The layout: policy_count is unconditional, and the dense path is unchanged
# --------------------------------------------------------------------------


def test_policy_count_sits_between_the_pad_and_the_policy_region() -> None:
    """C3: a dedicated, 4-aligned, 4-byte field -- in BOTH modes, identically.

    Sizing both modes the same costs four bytes and buys a layout that does not
    depend on a config flag: nobody has to know the mode to attach a segment,
    and ``_views``' size check keeps meaning what it meant.
    """
    dense = InferenceConfig(in_planes=128, num_actions=37341)
    sparse = InferenceConfig(in_planes=128, num_actions=37341, sparse_policy=True)

    assert inference._layout(dense) == inference._layout(sparse)
    assert slot_nbytes(dense) == slot_nbytes(sparse)

    layout = inference._layout(dense)
    assert layout.legal + dense.num_actions <= layout.policy_count
    assert layout.policy_count % 4 == 0
    assert layout.policy == layout.policy_count + 4
    assert layout.policy + 4 * dense.num_actions == layout.wdl
    # The golden arithmetic, with the new field: 2 u64 doorbells + obs f32 +
    # legal bool (padded) + the u32 count + policy/wdl/mlh f32.
    # Layout v2 (KOM-040 b2): + u32 row_count after the doorbells and a u32
    # row_count_echo after the padded legal region (8 bytes total at max_wave=1).
    assert slot_nbytes(_config(sparse=False)) == 16 + 4 + 1296 + 84 + 4 + 4 + 324 + 12 + 4
    # The same arithmetic at the production shape, recomputed for KOM-002's
    # 128 observation planes (it read 224984 while the encoder emitted 118):
    #   16 doorbells + 4 row_count
    # + 4 * 128 * 81 = 41472 observation f32
    # + 37341 legal bool padded up to 4 bytes = 37344
    # + 4 row_count_echo + 4 policy_count
    # + 4 * 37341 = 149364 policy f32 + 4 * 3 = 12 wdl + 4 mlh
    # = 228224.  The ten planes KOM-002 appended are the whole of the delta:
    # 10 * 81 * 4 = 3240 = 228224 - 224984.
    assert slot_nbytes(dense) == 16 + 4 + 41472 + 37344 + 4 + 4 + 149364 + 12 + 4
    assert slot_nbytes(dense) == 228224


def test_the_count_view_is_a_live_uint32_alias_of_those_four_bytes() -> None:
    """``_views`` maps the field, in both modes, and it aliases shared memory."""
    for sparse in (False, True):
        config = _config(sparse=sparse)
        with _one_slot(config) as created:
            assert created.policy_count.dtype == np.uint32
            assert created.policy_count.shape == (1,)
            attached = inference.attach_slots(created.shm.name, config)
            try:
                created.policy_count[0] = DENSE_POLICY_COUNT
                assert int(attached.policy_count[0]) == DENSE_POLICY_COUNT
            finally:
                attached.close()


def test_views_size_check_covers_the_count_field() -> None:
    """A segment four bytes short of the layout is refused, in either mode."""
    for sparse in (False, True):
        config = _config(sparse=sparse)
        shm = SharedMemory(create=True, size=slot_nbytes(config) - 4)
        try:
            with pytest.raises(ValueError, match="bytes"):
                inference._views(shm, config)
        finally:
            shm.close()
            shm.unlink()


def test_the_dense_writer_stamps_the_sentinel() -> None:
    """C4: dense responses tag themselves, so a sparse reader can tell.

    The dense round trip is otherwise untouched -- same bytes, same shapes --
    which is the regression this half of the test pins.
    """
    with _one_slot(_config(sparse=False)) as slots:
        assert int(slots.policy_count[0]) == 0, "a fresh segment claims nothing"
        batched = _row_eval(
            np.arange(2 * NUM_ACTIONS, dtype=np.float32).reshape(2, NUM_ACTIONS)
        )

        write_response(slots, batched, row=1)

        assert int(slots.policy_count[0]) == DENSE_POLICY_COUNT
        got = read_response(slots)
        np.testing.assert_array_equal(got.policy_logits, batched.policy_logits[1])
        np.testing.assert_array_equal(got.wdl, batched.wdl[1])
        assert float(got.moves_left) == float(batched.moves_left[1])


# --------------------------------------------------------------------------
# C4 -- the mode-mismatch tripwire, in both directions
# --------------------------------------------------------------------------


def test_a_dense_reader_refuses_a_sparse_response() -> None:
    """Server sparse, worker dense: caught by the missing sentinel.

    Without this check the worker reads ``K`` packed logits sitting at action
    ids ``0..K-1`` plus whatever the tail of the region last held -- all finite
    floats, so nothing downstream fires and the priors are simply wrong.
    """
    legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    legal[[1, 4, 8, 20]] = True
    logits = np.linspace(-1.0, 1.0, NUM_ACTIONS, dtype=np.float32)
    packed = pack_sparse_batch(_row_eval(logits[None]), legal[None])
    with _one_slot(_config(sparse=True)) as slots:
        write_sparse_response(slots, packed, row=0)

        with pytest.raises(InferenceProtocolError, match="dense sentinel"):
            read_response(slots)


def test_a_sparse_reader_refuses_a_dense_response() -> None:
    """Server dense, worker sparse: caught by the sentinel *being* there."""
    legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    legal[[1, 4, 8, 20]] = True
    dense = _row_eval(np.zeros((1, NUM_ACTIONS), dtype=np.float32))
    with _one_slot(_config(sparse=False)) as slots:
        write_response(slots, dense, row=0)

        with pytest.raises(InferenceProtocolError, match="sparse_policy=False"):
            read_sparse_response(slots, legal)


def test_a_sparse_reader_refuses_a_count_that_disagrees_with_its_mask() -> None:
    """The tripwire is ``count == mask.sum()``, not merely ``count != sentinel``.

    A codec bug that packed the wrong number of values -- an off-by-one in the
    index build, a row read from the wrong slot -- lands here, and it is caught
    even though the response looks perfectly sparse.
    """
    legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    legal[[1, 4, 8, 20]] = True
    logits = np.linspace(-1.0, 1.0, NUM_ACTIONS, dtype=np.float32)
    packed = pack_sparse_batch(_row_eval(logits[None]), legal[None])
    with _one_slot(_config(sparse=True)) as slots:
        write_sparse_response(slots, packed, row=0)
        assert int(slots.policy_count[0]) == 4

        wider = legal.copy()
        wider[30] = True
        with pytest.raises(InferenceProtocolError, match="5 legal actions"):
            read_sparse_response(slots, wider)

        slots.policy_count[0] = 3
        with pytest.raises(InferenceProtocolError, match="different number"):
            read_sparse_response(slots, legal)


def test_a_server_in_the_other_mode_poisons_the_client() -> None:
    """C4 end to end, both directions, through a real submit.

    The client and the pump thread hold configs that disagree about
    ``sparse_policy`` while sharing one segment -- which they can, because the
    layout does not depend on the mode. The submit must fail loudly and the
    client must stay dead afterwards: two ends that disagree about the wire
    format will disagree about every subsequent answer too.
    """
    for server_sparse, client_sparse in ((True, False), (False, True)):
        server_config = _config(
            sparse=server_sparse,
            max_batch=4,
            batch_timeout_ms=0.0,
            request_timeout_s=30.0,
        )
        client_config = _config(
            sparse=client_sparse,
            max_batch=4,
            batch_timeout_ms=0.0,
            request_timeout_s=30.0,
        )
        server_slots = inference.create_slots(server_config)
        name = server_slots.shm.name
        try:
            client = inference.SlotClient(name, 0, client_config)
            try:
                evaluator = InProcessEvaluator(_tiny_net(), server_config)
                obs, legal = _random_request(seed=77)
                pump = threading.Thread(
                    target=_pump_one,
                    args=({0: server_slots}, evaluator, server_config),
                    daemon=True,
                )
                pump.start()
                with pytest.raises(InferenceProtocolError):
                    client.submit(obs, legal)
                pump.join(60.0)
                assert not pump.is_alive(), "serve_once never picked the request up"

                # Poisoned -- and by the protocol error rather than by a
                # timeout, so the refusal is immediate.
                started = time.monotonic()
                with pytest.raises(inference.InferenceTimeout, match="poisoned"):
                    client.submit(obs, legal)
                assert time.monotonic() - started < 0.1
            finally:
                client.close()
                server_slots.close()
        finally:
            SharedMemory(name=name).unlink()


# --------------------------------------------------------------------------
# The equivalence claim: a real CPU serve round trip, on the legal support
# --------------------------------------------------------------------------


def test_sparse_serve_is_bit_identical_to_dense_on_the_legal_support() -> None:
    """C9: the same weights, the same requests, the same bytes where they exist.

    ``assert_array_equal``, not ``allclose``: this is a transport change, not an
    approximation -- which is the property fp16 cannot offer, and the reason
    this ticket is the wire-format prerequisite for KOM-040.

    The comparison is restricted to the legal support because that is the only
    place the two paths *claim* to agree: the dense path leaves a raw logit in
    an illegal lane, the sparse one leaves the fill. Both halves are asserted --
    equality on ``legal``, exact zero off it.

    The sparse side runs the production loop (:func:`serve_pipelined` over a
    real :class:`AsyncEvaluator`), so the packing goes through the CPU handle's
    own ``packed_result`` rather than through a helper this file called
    directly. All three doorbells ring before either loop starts, so both build
    one batch of three in slot order -- the batch composition has to match, or
    the conv reduction order would differ and "bit-identical" would be a lie.
    """
    net = _tiny_net()
    dense_config = _config(sparse=False, max_batch=4, batch_timeout_ms=0.0)
    sparse_config = _config(sparse=True, max_batch=4, batch_timeout_ms=0.0)
    worker_ids = (0, 1, 2)
    requests = {
        worker_id: _random_request(seed=140 + worker_id) for worker_id in worker_ids
    }

    with _slot_map(dense_config, worker_ids) as slots_map:
        for worker_id in worker_ids:
            _ring(slots_map[worker_id], requests[worker_id])
        serial = InProcessEvaluator(net, dense_config)
        assert inference.serve_once(slots_map, serial, dense_config) == 3
        dense = {
            worker_id: read_response(slots_map[worker_id]) for worker_id in worker_ids
        }

    with _slot_map(sparse_config, worker_ids) as slots_map:
        for worker_id in worker_ids:
            _ring(slots_map[worker_id], requests[worker_id])
        stop = multiprocessing.Event()
        thread = threading.Thread(
            target=inference.serve_pipelined,
            args=(slots_map, AsyncEvaluator(net, sparse_config), sparse_config, stop),
            daemon=True,
        )
        thread.start()
        try:
            assert _wait_until(
                lambda: all(
                    int(slots_map[worker_id].response_seq[0]) == 1
                    for worker_id in worker_ids
                )
            )
        finally:
            stop.set()
            thread.join(30.0)
        assert not thread.is_alive(), "serve_pipelined never returned after stop_event"

        for worker_id in worker_ids:
            legal = requests[worker_id][1]
            assert int(slots_map[worker_id].policy_count[0]) == int(legal.sum())
            sparse = read_sparse_response(slots_map[worker_id], legal)
            np.testing.assert_array_equal(
                sparse.policy_logits[legal], dense[worker_id].policy_logits[legal]
            )
            np.testing.assert_array_equal(
                sparse.policy_logits[~legal],
                np.zeros(int((~legal).sum()), dtype=np.float32),
            )
            assert bool(np.isfinite(sparse.policy_logits).all())
            # wdl and moves-left are untouched by this ticket and must stay so.
            np.testing.assert_array_equal(sparse.wdl, dense[worker_id].wdl)
            np.testing.assert_array_equal(
                sparse.moves_left, dense[worker_id].moves_left
            )


def test_a_sparse_client_gets_a_dense_vector_back() -> None:
    """C5/D6: reconstruction lives in the client, and nobody above it can tell.

    ``submit`` returns the same shapes it always did, in both modes; the caller
    -- the search's evaluator adapter -- cannot see which transport ran. What it
    must never receive is a short vector, an index array, or a ``-inf``.
    """
    config = _config(
        sparse=True, max_batch=4, batch_timeout_ms=0.0, request_timeout_s=30.0
    )
    server_slots = inference.create_slots(config)
    name = server_slots.shm.name
    try:
        client = inference.SlotClient(name, 0, config)
        try:
            evaluator = InProcessEvaluator(_tiny_net(), config)
            obs, legal = _random_request(seed=61)
            pump = threading.Thread(
                target=_pump_one,
                args=({0: server_slots}, evaluator, config),
                daemon=True,
            )
            pump.start()
            answer = client.submit(obs, legal)
            pump.join(60.0)
            assert not pump.is_alive(), "serve_once never picked the request up"

            assert answer.policy_logits.shape == (NUM_ACTIONS,)
            assert answer.wdl.shape == (3,)
            assert answer.moves_left.shape == ()
            assert bool(np.isfinite(answer.policy_logits).all())

            want = evaluator.evaluate(obs[None], legal[None])
            np.testing.assert_array_equal(
                answer.policy_logits[legal], want.policy_logits[0][legal]
            )
            np.testing.assert_array_equal(
                answer.policy_logits[~legal],
                np.zeros(int((~legal).sum()), dtype=np.float32),
            )
            # Only K floats crossed the seam: the count says so, and it is the
            # transport saving stated as a fact rather than as an intention.
            assert int(server_slots.policy_count[0]) == int(legal.sum())
            assert int(legal.sum()) < NUM_ACTIONS
        finally:
            client.close()
            server_slots.close()
    finally:
        SharedMemory(name=name).unlink()


# --------------------------------------------------------------------------
# The publish path nobody could falsify (union-38 review F1) and the dtype
# hole (F3)
# --------------------------------------------------------------------------


class _DenseOnlyHandle:
    """A ready ForwardHandle that answers dense and cannot pack itself.

    Deliberately NOT a :class:`~komugi.selfplay.inference._PacksPolicy`: it is
    the "anything else" case :func:`~komugi.selfplay.inference._packed_result`
    documents -- the handle whose rows the server must pack itself, from the
    legal batch it launched the forward with.
    """

    def __init__(self, evaluation: NetEval) -> None:
        self._evaluation = evaluation

    def query(self) -> bool:
        return True

    def result(self) -> NetEval:
        return self._evaluation


class _DenseOnlyLauncher:
    """A ForwardLauncher handing out :class:`_DenseOnlyHandle`s, kept."""

    def __init__(self, evaluator: InProcessEvaluator) -> None:
        self._evaluator = evaluator
        self.handles: list[_DenseOnlyHandle] = []

    def launch(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> _DenseOnlyHandle:
        handle = _DenseOnlyHandle(self._evaluator.evaluate(obs, legal))
        self.handles.append(handle)
        return handle


def test_pipelined_sparse_serves_a_handle_that_cannot_pack_itself() -> None:
    """F1 (union-38 review): the CPU fallback pack, finally exercised.

    Before this test, :func:`~komugi.selfplay.inference._packed_result`'s
    fallback branch and the ``in_flight_legal`` plumbing that feeds it were
    shipped control flow nothing could falsify: replacing the fallback with
    ``raise`` (mutant M9) and zeroing the legal batch handed to ``_publish``
    (mutant M10) both passed every suite. Here :func:`serve_pipelined` runs
    sparse over a launcher whose handles do NOT pack themselves, so the server
    must pack from ``in_flight_legal`` -- M9's raise fires immediately, and
    M10's zeroed masks either refuse (K=0) or transport the wrong support,
    failing the equality below either way.
    """
    net = _tiny_net()
    sparse_config = _config(sparse=True, max_batch=4, batch_timeout_ms=0.0)
    worker_ids = (0, 1, 2)
    requests = {
        worker_id: _random_request(seed=520 + worker_id) for worker_id in worker_ids
    }
    launcher = _DenseOnlyLauncher(
        InProcessEvaluator(net, _config(sparse=False, max_batch=4))
    )

    with _slot_map(sparse_config, worker_ids) as slots_map:
        for worker_id in worker_ids:
            _ring(slots_map[worker_id], requests[worker_id])
        stop = multiprocessing.Event()
        thread = threading.Thread(
            target=inference.serve_pipelined,
            args=(slots_map, launcher, sparse_config, stop),
            daemon=True,
        )
        thread.start()
        try:
            assert _wait_until(
                lambda: all(
                    int(slots_map[worker_id].response_seq[0]) == 1
                    for worker_id in worker_ids
                )
            )
        finally:
            stop.set()
            thread.join(30.0)
        assert not thread.is_alive(), "serve_pipelined never returned after stop_event"

        # All three doorbells rang before the loop started and the timeout is
        # zero, so the first forward took all three rows in slot order.
        dense = launcher.handles[0].result()
        for row, worker_id in enumerate(worker_ids):
            legal = requests[worker_id][1]
            assert int(slots_map[worker_id].policy_count[0]) == int(legal.sum())
            sparse = read_sparse_response(slots_map[worker_id], legal)
            np.testing.assert_array_equal(
                sparse.policy_logits[legal], dense.policy_logits[row][legal]
            )
            np.testing.assert_array_equal(
                sparse.policy_logits[~legal],
                np.zeros(int((~legal).sum()), dtype=np.float32),
            )


def test_reconstruction_refuses_a_non_bool_mask() -> None:
    """F3 (union-38 review): an integer mask must be refused, not fancy-indexed.

    ``dense[legal] = packed`` with an integer array is index assignment, not a
    boolean scatter: an all-ones ``int64`` mask at K=A would silently write
    every packed value into lane 1 -- finite, plausible, wrong, and past the
    ``policy_count`` tripwire because the count still matches. The codec must
    make the dtype load-bearing.
    """
    logits = np.arange(8, dtype=np.float32)
    int_mask = np.ones(8, dtype=np.int64)
    with pytest.raises(ValueError, match="bool"):
        inference.unpack_sparse_policy(logits, int_mask)  # type: ignore[arg-type]
