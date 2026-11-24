"""Unit tests for the wave transport: layout v2, the wave codec, admission.

KOM-040 b2 (design rev2). The b1 interleaving bench established the mechanism
this ticket acts on: the shared server batch already **synchronises** the lanes,
so the lever left is the *count* of round trips, not their width. A wave is one
ticket carrying up to ``max_wave`` rows, announced by one doorbell and answered
under one counter.

What is pinned here, in the order the protocol is built:

* **Layout v2 is unconditional.** Every region is sized for ``max_wave`` rows
  and both u32 row-count headers are always present, so at the default
  ``max_wave == 1`` a slot is exactly eight bytes wider than v1 and byte-for-byte
  the same shape otherwise. Nobody has to know the wave width to attach a
  segment; both ends compute it from one config.
* **The codec round-trips**, at one row, at ``k`` rows and at the full
  ``max_wave``, in both transport modes, with the sparse rows still in ascending
  action-id order per row.
* **The refusals happen before the doorbell.** Zero rows and any all-false mask
  are caller bugs, refused at the client -- including on the plain scalar
  ``submit``, which is KOM-038's flip-gate item 3 finally shipped.
* **The two header tripwires fire**, one per direction: the server refuses a
  ``row_count`` outside ``1..max_wave`` and poisons only that client; the client
  refuses a ``row_count_echo`` that is not the width it submitted.
* **Admission is fairness-ordered and never splits a wave.** A ticket is served
  whole or skipped, lower ranks may fill the space a skipped wave left, and an
  unserved wave's rank strictly rises -- so it is admitted no later than the
  forward after it reaches rank 1.

Everything runs in one process. The one concession to concurrency is a pump
*thread* running the same ``serve_once`` the real loop runs, exactly as in
``test_inference_unit.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt
import pytest

from komugi.config import SelfPlayConfig
from komugi.selfplay import inference
from komugi.selfplay.inference import (
    DENSE_POLICY_COUNT,
    InferenceConfig,
    InferenceProtocolError,
    NetEval,
    SlotClient,
    Slots,
    pack_sparse_batch,
    pack_wave_request,
    read_wave_request,
    read_wave_response,
    slot_nbytes,
)
from komugi.selfplay.orchestrate import validate_wave_fit

IN_PLANES = 4
NUM_ACTIONS = 81


def _config(
    *,
    max_batch: int = 4,
    max_wave: int = 1,
    batch_timeout_ms: float = 0.0,
    request_timeout_s: float = 60.0,
    sparse_policy: bool = False,
) -> InferenceConfig:
    """A tiny-net config; ``max_wave`` defaults to 1, the shipped width."""
    return InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=max_batch,
        batch_timeout_ms=batch_timeout_ms,
        request_timeout_s=request_timeout_s,
        sparse_policy=sparse_policy,
        max_wave=max_wave,
    )


@contextmanager
def _one_slot(config: InferenceConfig) -> Iterator[Slots]:
    """Create one segment, yield its views, and unlink it afterwards."""
    created = inference.create_slots(config)
    name = created.shm.name
    try:
        yield created
    finally:
        created.close()
        SharedMemory(name=name).unlink()


@contextmanager
def _slot_map(config: InferenceConfig, worker_ids: list[int]) -> Iterator[
    dict[int, Slots]
]:
    """Create one segment per id and yield the server's slot map."""
    slots_map: dict[int, Slots] = {}
    names: list[str] = []
    try:
        for worker_id in worker_ids:
            created = inference.create_slots(config)
            names.append(created.shm.name)
            slots_map[worker_id] = created
        yield slots_map
    finally:
        for entry in slots_map.values():
            entry.close()
        for name in names:
            SharedMemory(name=name).unlink()


def _wave(
    tags: list[float], num_actions: int = NUM_ACTIONS
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A wave whose rows are individually identifiable by their tag.

    Every cell of row ``r``'s observation is ``tags[r]``, so a recording
    evaluator can say which slot a row came from and a reader can say which row
    an answer belongs to -- which is what the no-split and per-row-routing
    assertions below actually check.
    """
    rows = len(tags)
    obs = np.stack(
        [np.full((IN_PLANES, 9, 9), tag, dtype=np.float32) for tag in tags]
    )
    legal = np.zeros((rows, num_actions), dtype=np.bool_)
    for row in range(rows):
        # A different, non-trivial support per row: ascending ids, never empty,
        # and never the same set twice.
        legal[row, row % 3 :: 3 + row] = True
        legal[row, row] = True
    return obs, legal


class _TaggedEvaluator:
    """A :class:`~komugi.selfplay.inference.BatchEvaluator` that echoes its tags.

    Row ``r`` of the answer is filled with ``obs[r, 0, 0, 0]``, so every value a
    client reads back can be traced to the row that asked for it. It also
    records each forward's tags, which is how the no-split rule is observed:
    a wave that was split would appear in two batches.
    """

    def __init__(self) -> None:
        self.batches: list[list[float]] = []

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        tags = np.asarray(obs[:, 0, 0, 0], dtype=np.float32)
        self.batches.append([float(tag) for tag in tags])
        rows = int(obs.shape[0])
        return NetEval(
            policy_logits=np.repeat(
                tags[:, None], legal.shape[1], axis=1
            ).astype(np.float32),
            wdl=np.repeat(tags[:, None], 3, axis=1).astype(np.float32),
            moves_left=tags.reshape(rows),
        )


# --------------------------------------------------------------------------
# W1/W2 -- the config knob and the unconditional layout
# --------------------------------------------------------------------------


def test_max_wave_defaults_to_one_and_must_be_positive() -> None:
    """The shipped width is 1; zero rows is not a configuration, it is a bug."""
    assert _config().max_wave == 1
    with pytest.raises(ValueError, match="max_wave must be >= 1"):
        _config(max_wave=0)


def test_a_wave_may_never_be_wider_than_a_batch() -> None:
    """W1: the foundation of the no-split rule, refused loudly at construction.

    A wave wider than ``max_batch`` could not be admitted at *any* rank -- the
    batch starts empty and still would not hold it -- so it would be skipped
    forever and its client would die on ``request_timeout_s``.
    """
    with pytest.raises(ValueError, match="max_wave=5 exceeds max_batch=4"):
        _config(max_wave=5, max_batch=4)
    # Equal is fine, and is exactly the "always fits an empty batch" edge.
    assert _config(max_wave=4, max_batch=4).max_wave == 4


def test_layout_v2_costs_exactly_eight_bytes_at_a_wave_of_one() -> None:
    """The two u32 headers, and nothing else, at the default width.

    Re-derived from the v1 arithmetic rather than asserted against a bumped
    constant: 16 doorbells + obs + padded legal + policy_count + response, plus
    ``row_count`` and ``row_count_echo``.
    """
    v1_tiny = 16 + 1296 + 84 + 4 + 324 + 12 + 4
    assert slot_nbytes(_config()) == v1_tiny + 8 == 1748
    production = InferenceConfig(in_planes=128, num_actions=37341)
    # The observation term is 4 * in_planes * 81, so KOM-002's move from 118 to
    # 128 planes moves it from 38232 to 41472 -- 3240 bytes, which is the whole
    # difference between this figure and the 224984 it read at 118 planes.
    v1_production = 16 + 41472 + 37344 + 4 + 149364 + 12 + 4
    assert slot_nbytes(production) == v1_production + 8 == 228224


def test_every_region_is_sized_for_max_wave_rows() -> None:
    """W2(a): obs, legal, policy_count, policy, wdl and mlh all scale together."""
    single = _config()
    wide = _config(max_wave=4, max_batch=4)
    layout = inference._layout(wide)

    assert layout.obs == 20
    assert layout.legal == layout.obs + 4 * 4 * IN_PLANES * 81
    assert layout.legal + 4 * NUM_ACTIONS <= layout.row_count_echo
    assert layout.policy_count == layout.row_count_echo + 4
    assert layout.policy == layout.policy_count + 4 * 4
    assert layout.wdl == layout.policy + 4 * 4 * NUM_ACTIONS
    assert layout.moves_left == layout.wdl + 4 * 4 * 3
    assert layout.nbytes == layout.moves_left + 4 * 4

    # The headers are fixed-size; everything else quadruples, so the growth is
    # four times the per-row payload and not a byte more.
    per_row = 4 * IN_PLANES * 81 + NUM_ACTIONS + 4 + 4 * NUM_ACTIONS + 12 + 4
    grown = slot_nbytes(wide) - slot_nbytes(single)
    assert abs(grown - 3 * per_row) <= 4, "only alignment may differ"


def test_the_views_have_a_row_axis_that_degenerates_at_width_one() -> None:
    """W2: a ``max_wave==1`` segment carries exactly one row of everything."""
    with _one_slot(_config()) as slots:
        assert slots.max_wave == 1
        assert slots.obs.shape == (1, IN_PLANES, 9, 9)
        assert slots.legal.shape == (1, NUM_ACTIONS)
        assert slots.policy_count.shape == (1,)
        assert slots.policy.shape == (1, NUM_ACTIONS)
        assert slots.wdl.shape == (1, 3)
        assert slots.moves_left.shape == (1,)
        assert slots.row_count.dtype == np.uint32
        assert slots.row_count_echo.dtype == np.uint32
        # A fresh segment claims no rows in either direction.
        assert int(slots.row_count[0]) == 0
        assert int(slots.row_count_echo[0]) == 0


# --------------------------------------------------------------------------
# W3/W6 -- the codec round trips, dense and sparse
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_the_dense_wave_codec_round_trips(rows: int) -> None:
    """One, some, and all ``max_wave`` rows through write -> read, per row."""
    config = _config(max_wave=5, max_batch=5)
    obs, legal = _wave([1.0 * (r + 1) for r in range(rows)])
    evaluator = _TaggedEvaluator()

    with _one_slot(config) as slots:
        assert pack_wave_request(slots, obs, legal) == rows
        assert int(slots.row_count[0]) == rows

        server_obs, server_legal = read_wave_request(slots, config)
        np.testing.assert_array_equal(server_obs, obs)
        np.testing.assert_array_equal(server_legal, legal)

        evaluation = evaluator.evaluate(server_obs, server_legal)
        inference._scatter(
            {0: slots}, evaluation, [inference._Admission(0, 7, rows)]
        )

        assert int(slots.row_count_echo[0]) == rows
        assert int(slots.response_seq[0]) == 7
        answers = read_wave_response(slots, legal, config)
        assert len(answers) == rows
        for row, answer in enumerate(answers):
            assert answer.policy_logits.shape == (NUM_ACTIONS,)
            assert answer.wdl.shape == (3,)
            assert answer.moves_left.shape == ()
            # Row r's answer carries row r's tag: no cross-row smearing.
            assert float(answer.moves_left) == float(row + 1)
            np.testing.assert_array_equal(
                answer.policy_logits,
                np.full(NUM_ACTIONS, float(row + 1), dtype=np.float32),
            )


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_the_sparse_wave_codec_round_trips_in_ascending_action_id_order(
    rows: int,
) -> None:
    """W6: per-row ``K``, per-row packing, and the ids stay ascending."""
    config = _config(max_wave=5, max_batch=5, sparse_policy=True)
    obs, legal = _wave([1.0 * (r + 1) for r in range(rows)])
    # A policy whose value IS the action id, so "ascending action-id order" is
    # checkable by looking at the packed bytes alone.
    logits = np.repeat(
        np.arange(NUM_ACTIONS, dtype=np.float32)[None], rows, axis=0
    )
    evaluation = NetEval(
        policy_logits=logits,
        wdl=np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32), (rows, 1)),
        moves_left=np.arange(rows, dtype=np.float32),
    )

    with _one_slot(config) as slots:
        pack_wave_request(slots, obs, legal)
        packed = pack_sparse_batch(evaluation, legal)
        inference._scatter_sparse(
            {0: slots}, packed, [inference._Admission(0, 3, rows)]
        )

        assert int(slots.row_count_echo[0]) == rows
        for row in range(rows):
            ids = np.flatnonzero(legal[row])
            count = int(slots.policy_count[row])
            assert count == ids.size
            np.testing.assert_array_equal(
                np.array(slots.policy[row, :count]), ids.astype(np.float32)
            )

        answers = read_wave_response(slots, legal, config)
        for row, answer in enumerate(answers):
            ids = np.flatnonzero(legal[row])
            np.testing.assert_array_equal(
                answer.policy_logits[legal[row]], ids.astype(np.float32)
            )
            # Illegal lanes are a finite 0.0, never -inf and never NaN.
            np.testing.assert_array_equal(
                answer.policy_logits[~legal[row]],
                np.zeros(NUM_ACTIONS - ids.size, dtype=np.float32),
            )


def test_the_dense_sentinel_lands_in_every_used_row_and_nowhere_else() -> None:
    """W6: per-row mode tags; entries past ``row_count`` stay untouched."""
    config = _config(max_wave=4, max_batch=4)
    obs, legal = _wave([1.0, 2.0, 3.0])
    evaluation = _TaggedEvaluator().evaluate(obs, legal)

    with _one_slot(config) as slots:
        pack_wave_request(slots, obs, legal)
        inference._scatter({0: slots}, evaluation, [inference._Admission(0, 1, 3)])

        for row in range(3):
            assert int(slots.policy_count[row]) == DENSE_POLICY_COUNT
        # The fourth entry is beyond this ticket's row_count: unobservable by
        # contract, and in fact never written.
        assert int(slots.policy_count[3]) == 0


def test_a_mode_mismatch_is_caught_per_row() -> None:
    """W6: the KOM-038 tripwire applies to every row of a wave, not just row 0."""
    dense = _config(max_wave=3, max_batch=3)
    sparse = _config(max_wave=3, max_batch=3, sparse_policy=True)
    obs, legal = _wave([1.0, 2.0, 3.0])
    evaluation = _TaggedEvaluator().evaluate(obs, legal)

    with _one_slot(dense) as slots:
        pack_wave_request(slots, obs, legal)
        inference._scatter({0: slots}, evaluation, [inference._Admission(0, 1, 3)])
        # A sparse reader over a dense answer: the very first row already
        # disagrees, and it says so rather than unpacking 4 billion values.
        with pytest.raises(InferenceProtocolError, match="policy_count"):
            read_wave_response(slots, legal, sparse)


# --------------------------------------------------------------------------
# W3 -- refusals, strictly before the doorbell
# --------------------------------------------------------------------------


def test_a_zero_row_wave_is_refused_before_it_can_be_rung() -> None:
    """A ticket describing no rows is unanswerable; say so at the writer."""
    config = _config(max_wave=4, max_batch=4)
    with _one_slot(config) as slots:
        empty_obs = np.zeros((0, IN_PLANES, 9, 9), dtype=np.float32)
        empty_legal = np.zeros((0, NUM_ACTIONS), dtype=np.bool_)
        with pytest.raises(ValueError, match="at least one row"):
            pack_wave_request(slots, empty_obs, empty_legal)
        assert int(slots.request_seq[0]) == 0, "nothing may have been rung"

        client = SlotClient(slots.shm.name, 0, config)
        try:
            with pytest.raises(ValueError, match="at least one row"):
                client.submit_wave(empty_obs, empty_legal)
            assert int(slots.request_seq[0]) == 0
        finally:
            client.close()


def test_a_wave_wider_than_the_slot_is_refused() -> None:
    """The client cannot ask for more rows than the segment was sized for."""
    config = _config(max_wave=2, max_batch=4)
    obs, legal = _wave([1.0, 2.0, 3.0])
    with _one_slot(config) as slots:
        with pytest.raises(ValueError, match="max_wave=2"):
            pack_wave_request(slots, obs, legal)
        assert int(slots.request_seq[0]) == 0


def test_an_all_false_mask_is_refused_by_the_client_in_both_shapes() -> None:
    """W3 / KOM-038 flip-gate item 3: K=0 is refused BEFORE the doorbell.

    And on the plain scalar ``submit`` too, which is the half that was missing:
    ``_sparse_indices`` has refused K=0 on the server since KOM-038, but only
    under ``sparse_policy``, so with the dense default an all-false mask
    travelled happily and came back as ``A`` meaningless logits.
    """
    config = _config(max_wave=3, max_batch=3)
    obs, legal = _wave([1.0, 2.0, 3.0])
    legal[1, :] = False

    with _one_slot(config) as slots:
        client = SlotClient(slots.shm.name, 0, config)
        try:
            with pytest.raises(ValueError, match=r"rows \[1\] have no legal"):
                client.submit_wave(obs, legal)
            assert int(slots.request_seq[0]) == 0, "the doorbell must not ring"

            # The same refusal, reached through the scalar seam. The config is
            # dense -- this is the mode production runs.
            assert config.sparse_policy is False
            with pytest.raises(ValueError, match=r"rows \[0\] have no legal"):
                client.submit(obs[1], legal[1])
            assert int(slots.request_seq[0]) == 0
        finally:
            client.close()


# --------------------------------------------------------------------------
# W5 -- the two header tripwires
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 5, 2**32 - 1])
def test_the_server_refuses_a_row_count_outside_the_wave(bad: int) -> None:
    """W5, server side: a header the layout cannot describe is not read past."""
    config = _config(max_wave=4, max_batch=4)
    with _one_slot(config) as slots:
        slots.row_count[0] = bad
        with pytest.raises(InferenceProtocolError, match="row_count is"):
            read_wave_request(slots, config)


def test_a_poisoned_header_costs_that_client_and_not_the_shard() -> None:
    """W5: the offender is dropped; every other worker is still served."""
    config = _config(max_wave=4, max_batch=4)
    obs, legal = _wave([9.0])
    with _slot_map(config, [0, 1]) as slots_map:
        pack_wave_request(slots_map[0], obs, legal)
        # Worker 0 rings with a header nobody can honour.
        slots_map[0].row_count[0] = 0
        slots_map[0].request_seq[0] = 1

        pack_wave_request(slots_map[1], obs, legal)
        slots_map[1].request_seq[0] = 1

        evaluator = _TaggedEvaluator()
        # No exception escapes: one bad client is not a reason to lose a run.
        assert inference.serve_once(slots_map, evaluator, config) == 1
        assert int(slots_map[1].response_seq[0]) == 1
        assert int(slots_map[0].response_seq[0]) == 0, "the offender is unanswered"
        assert evaluator.batches == [[9.0]]


def test_the_client_refuses_an_echo_that_is_not_what_it_submitted() -> None:
    """W5, client side: a width disagreement is loud, not silently short."""
    config = _config(max_wave=4, max_batch=4)
    obs, legal = _wave([1.0, 2.0])
    evaluation = _TaggedEvaluator().evaluate(obs, legal)

    with _one_slot(config) as slots:
        pack_wave_request(slots, obs, legal)
        inference._scatter({0: slots}, evaluation, [inference._Admission(0, 1, 2)])
        assert len(read_wave_response(slots, legal, config)) == 2

        slots.row_count_echo[0] = 1
        with pytest.raises(InferenceProtocolError, match="row_count_echo is 1"):
            read_wave_response(slots, legal, config)


def test_a_bad_echo_poisons_the_client_through_submit() -> None:
    """The tripwire is reachable from the blocking seam, and it is fatal.

    A hand-rolled server thread that answers the right rows but publishes the
    wrong width: the client must refuse the answer and refuse every later
    submit, exactly as it does after a timeout.
    """
    config = _config(max_wave=4, max_batch=4, request_timeout_s=30.0)
    obs, legal = _wave([1.0, 2.0])
    evaluation = _TaggedEvaluator().evaluate(obs, legal)

    with _one_slot(config) as slots:
        def lying_server() -> None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                ticket = int(slots.request_seq[0])
                if ticket > int(slots.response_seq[0]):
                    # Hand-rolled rather than through `_scatter`, because the
                    # lie has to be in place BEFORE the counter that publishes
                    # it -- otherwise the client could read a correct echo and
                    # the tripwire would never be reached.
                    for row in range(2):
                        inference.write_response(slots, evaluation, row, row)
                    slots.row_count_echo[0] = 1
                    slots.response_seq[0] = ticket
                    return
                time.sleep(0.0005)

        client = SlotClient(slots.shm.name, 0, config)
        server = threading.Thread(target=lying_server, daemon=True)
        try:
            server.start()
            with pytest.raises(InferenceProtocolError, match="row_count_echo"):
                client.submit_wave(obs, legal)
            server.join(30.0)
            assert not server.is_alive()
            with pytest.raises(inference.InferenceTimeout, match="poisoned"):
                client.submit_wave(obs, legal)
        finally:
            client.close()


# --------------------------------------------------------------------------
# W4 -- row-aware ceilings, no-split admission, the starvation bound
# --------------------------------------------------------------------------


def test_the_batch_ceiling_counts_rows_not_tickets() -> None:
    """W4: ``min(max_batch, max_wave * slots)``, which is v1 at ``max_wave==1``."""
    single = _config(max_batch=16)
    assert inference._row_ceiling(single, 4) == 4
    assert inference._row_ceiling(single, 32) == 16
    wide = _config(max_batch=16, max_wave=4)
    # Two clients can only have 8 rows in flight, however big max_batch is.
    assert inference._row_ceiling(wide, 2) == 8
    assert inference._row_ceiling(wide, 8) == 16
    # The rank-1 guarantee: the ceiling is never below one whole wave.
    assert inference._row_ceiling(wide, 1) >= wide.max_wave
    assert inference._row_ceiling(wide, 0) >= wide.max_wave


def test_a_wave_is_skipped_whole_and_lower_ranks_fill_the_space() -> None:
    """W4: no split, and no wasted rows behind a skip.

    Four pending tickets and a four-row ceiling. Rank 1 takes two rows; rank 2
    wants four and cannot have them, so it is skipped **entire** -- a
    half-answered ticket cannot publish under one counter. Ranks 3 and 4 are
    one row each and fill the remainder, because leaving the batch short to
    preserve a strict order would waste rows the forward pays for anyway.
    """
    config = _config(max_wave=4, max_batch=4)
    widths = {0: 2, 1: 4, 2: 1, 3: 1}
    with _slot_map(config, [0, 1, 2, 3]) as slots_map:
        for worker_id, width in widths.items():
            obs, legal = _wave([10.0 * worker_id + row for row in range(width)])
            pack_wave_request(slots_map[worker_id], obs, legal)
            # request_seq is the fairness key: worker id order IS rank order.
            slots_map[worker_id].request_seq[0] = worker_id + 1

        evaluator = _TaggedEvaluator()
        assert inference.serve_once(slots_map, evaluator, config) == 4
        assert evaluator.batches == [[0.0, 1.0, 20.0, 30.0]]
        assert int(slots_map[1].response_seq[0]) == 0, "rank 2 was skipped whole"
        for worker_id in (0, 2, 3):
            entry = slots_map[worker_id]
            assert int(entry.response_seq[0]) == int(entry.request_seq[0])
            assert int(entry.row_count_echo[0]) == widths[worker_id]


def test_a_starved_wide_wave_is_served_on_the_forward_after_it_reaches_rank_one(
) -> None:
    """W4's starvation bound, pinned end to end.

    An oversized wave sits behind a stream of small ones. It loses the first
    forward -- four one-row tickets with lower tickets fill the ceiling exactly
    -- and that is the whole risk the no-split rule takes on. The bound that
    makes it safe is arithmetic, not luck: the wave keeps its ticket while the
    served ones climb, so its fairness rank strictly rises, and at rank 1 it is
    admitted whole because the batch starts empty and a wave always fits one.
    """
    config = _config(max_wave=4, max_batch=4)
    with _slot_map(config, [0, 1, 2, 3, 4]) as slots_map:
        big_obs, big_legal = _wave([100.0, 101.0, 102.0, 103.0])
        pack_wave_request(slots_map[0], big_obs, big_legal)
        # Worker 0 has been served four times already, so it ranks LAST.
        slots_map[0].response_seq[0] = 4
        slots_map[0].request_seq[0] = 5

        for worker_id in (1, 2, 3, 4):
            small_obs, small_legal = _wave([float(worker_id)])
            pack_wave_request(slots_map[worker_id], small_obs, small_legal)
            slots_map[worker_id].request_seq[0] = 1

        evaluator = _TaggedEvaluator()

        # Forward 1: the four small waves fill the ceiling; the wide one is
        # skipped, and not one of its rows leaks into this batch.
        assert inference.serve_once(slots_map, evaluator, config) == 4
        assert evaluator.batches[0] == [1.0, 2.0, 3.0, 4.0]
        assert int(slots_map[0].response_seq[0]) == 4, "still unanswered"

        # Forward 2: the small waves are answered and no longer pending, so the
        # wide one is rank 1 -- and it is served WHOLE, in one forward.
        assert inference.serve_once(slots_map, evaluator, config) == 4
        assert evaluator.batches[1] == [100.0, 101.0, 102.0, 103.0]
        assert int(slots_map[0].response_seq[0]) == 5
        assert int(slots_map[0].row_count_echo[0]) == 4
        answers = read_wave_response(slots_map[0], big_legal, config)
        assert [float(answer.moves_left) for answer in answers] == [
            100.0,
            101.0,
            102.0,
            103.0,
        ]
        assert len(evaluator.batches) == 2, "no wave was split across forwards"


def test_a_wave_of_one_is_byte_for_byte_the_single_row_protocol() -> None:
    """The degenerate case really is the same path, not a parallel one."""
    config = _config(max_wave=1, max_batch=4, request_timeout_s=30.0)
    obs, legal = _wave([5.0])

    with _one_slot(config) as slots:
        client = SlotClient(slots.shm.name, 0, config)
        evaluator = _TaggedEvaluator()

        def pump() -> None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if inference.serve_once({0: slots}, evaluator, config) > 0:
                    return
                time.sleep(0.0005)

        server = threading.Thread(target=pump, daemon=True)
        try:
            server.start()
            scalar = client.submit(obs[0], legal[0])
            server.join(60.0)
            assert not server.is_alive()
            assert int(slots.row_count[0]) == 1
            assert int(slots.row_count_echo[0]) == 1
            assert scalar.policy_logits.shape == (NUM_ACTIONS,)
            assert float(scalar.moves_left) == 5.0
        finally:
            client.close()


# --------------------------------------------------------------------------
# W9/W10 -- the adapter seam and the orchestrator's cross-config check
# --------------------------------------------------------------------------


class _StubState:
    """The minimum an :class:`EvaluableState` owes the wave evaluator."""

    def __init__(self, tag: float) -> None:
        self._tag = tag

    def observation(self) -> npt.NDArray[np.float32]:
        return np.full((IN_PLANES, 9, 9), self._tag, dtype=np.float32)

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        mask[[0, 3, 7]] = True
        return mask


@pytest.mark.parametrize("with_wdl", [False, True], ids=["scalar", "wdl"])
def test_make_wave_evaluator_submits_one_ticket_for_the_whole_wave(
    with_wdl: bool,
) -> None:
    """W9: N states in, N answers out, one round trip -- in both shapes.

    Parametrised over ``with_wdl`` by KOM-173 (i) review F6: the three-tuple
    wave seam is the one the self-play worker runs, and before this it was
    exercised only end to end, where a *dropped* vector raises at the search's
    first backup but a **mis-aligned** one would not be noticed. At ``True``
    each answer is a three-tuple whose third element is THAT leaf's own client
    row -- ``_TaggedEvaluator`` fills row ``r``'s wdl with row ``r``'s tag, so
    a seam that handed every row the first row's vector fails here and nowhere
    else.
    """
    from komugi.selfplay.adapter import make_wave_evaluator

    config = _config(max_wave=3, max_batch=3, request_timeout_s=30.0)
    selfplay = SelfPlayConfig(workers=1, process_batch_size=1, draw_score=0.25)

    with _one_slot(config) as slots:
        client = SlotClient(slots.shm.name, 0, config)
        evaluator = _TaggedEvaluator()

        def pump() -> None:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if inference.serve_once({0: slots}, evaluator, config) > 0:
                    return
                time.sleep(0.0005)

        server = threading.Thread(target=pump, daemon=True)
        try:
            server.start()
            wave_evaluator = make_wave_evaluator(
                client, selfplay, with_wdl=with_wdl
            )
            states = [_StubState(2.0), _StubState(4.0), _StubState(6.0)]
            results = wave_evaluator(states)  # type: ignore[arg-type]
            server.join(60.0)
            assert not server.is_alive()

            assert len(results) == 3
            # ONE forward for three leaves: the count of round trips this whole
            # ticket exists to cut. One ticket in BOTH shapes -- the vector
            # rides the same round trip rather than asking for another.
            assert evaluator.batches == [[2.0, 4.0, 6.0]]
            for answer, tag in zip(results, (2.0, 4.0, 6.0), strict=True):
                # The seam's static type is the union of both shapes (KOM-173
                # §4), so unpack by index after pinning the arity the flag asks
                # for: 2 at with_wdl=False, 3 at True.
                assert len(answer) == (3 if with_wdl else 2), answer
                policy, value = answer[0], answer[1]
                assert policy.shape == (NUM_ACTIONS,)
                np.testing.assert_array_equal(
                    policy, np.full(NUM_ACTIONS, tag, dtype=np.float32)
                )
                # wdl is (tag, tag, tag), so w - l + d*draw_score == tag * 0.25.
                assert value == pytest.approx(tag * 0.25, rel=1e-6)
                if with_wdl:
                    # THAT leaf's own row, not the wave's first: the client
                    # answers row r with (tag_r, tag_r, tag_r), so a seam that
                    # broadcast rows[0].wdl over the wave reads (2, 2, 2) here
                    # for the 4.0 and 6.0 leaves (review F6's mutant).
                    # Indexed through a widened alias for the same reason
                    # `komugi.search.gumbel._unpack_evaluation` does: mypy
                    # cannot narrow the two-shape union on `len`, and the arity
                    # is already pinned above.
                    fields: tuple[object, ...] = answer
                    vector = fields[2]
                    assert isinstance(vector, np.ndarray)
                    wdl = np.asarray(vector, dtype=np.float32)
                    assert wdl.shape == (3,)
                    np.testing.assert_array_equal(
                        wdl, np.full(3, tag, dtype=np.float32)
                    )
        finally:
            client.close()


def test_make_wave_evaluator_refuses_an_empty_wave() -> None:
    """No states is not a question; the seam says so instead of stacking zero."""
    from komugi.selfplay.adapter import make_wave_evaluator

    config = _config(max_wave=3, max_batch=3)
    selfplay = SelfPlayConfig(workers=1, process_batch_size=1)
    with _one_slot(config) as slots:
        client = SlotClient(slots.shm.name, 0, config)
        try:
            with pytest.raises(ValueError, match="at least one state"):
                make_wave_evaluator(client, selfplay)([])
        finally:
            client.close()


def test_validate_wave_fit_names_both_configs_when_the_wave_is_too_narrow() -> None:
    """W10: the orchestrator is where the two configs meet, so it checks.

    Only once waves are actually enabled: at ``max_wave == 1`` nothing submits a
    wave, and demanding ``max_wave >= gumbel_m_full`` of the default config
    would refuse every run in the repository.
    """
    selfplay = SelfPlayConfig(
        workers=1, process_batch_size=1, gumbel_m_full=32, gumbel_m_fast=8
    )
    # Waves off: nothing to check, whatever m is.
    validate_wave_fit(selfplay, _config(max_wave=1))
    # Waves on and wide enough: fine.
    validate_wave_fit(selfplay, _config(max_wave=32, max_batch=64))
    # Waves on and too narrow: refused, naming both fields.
    with pytest.raises(ValueError, match="max_wave=16.*gumbel_m_full=32"):
        validate_wave_fit(selfplay, _config(max_wave=16, max_batch=64))


def test_admission_counts_rows_not_tickets(  # b2 union review F2 / mutant M5
) -> None:
    """W4: the ceiling bounds ROWS admitted, not tickets.

    Three two-row waves against a four-row ceiling: exactly two waves (four
    rows) go into the first forward, the third is served whole on the next.
    The review's surviving mutant M5 made ``_admit`` count tickets instead of
    rows -- it admitted a SIX-row batch under a four-row ceiling while every
    other admission test stayed green, because their widths never made the
    accumulator's unit distinguishable. ``max_batch`` is the GPU bound the
    whole config rests on; this test makes its unit unmistakable.
    """
    config = _config(max_wave=2, max_batch=4)
    with _slot_map(config, [0, 1, 2]) as slots_map:
        for worker_id in (0, 1, 2):
            obs, legal = _wave([10.0 * worker_id, 10.0 * worker_id + 1.0])
            pack_wave_request(slots_map[worker_id], obs, legal)
            slots_map[worker_id].request_seq[0] = worker_id + 1

        evaluator = _TaggedEvaluator()

        assert inference.serve_once(slots_map, evaluator, config) == 4
        assert evaluator.batches[0] == [0.0, 1.0, 10.0, 11.0], (
            "the first forward must hold exactly the first two waves' four "
            "rows -- six rows here means the accumulator counted tickets"
        )
        assert int(slots_map[2].response_seq[0]) == 0, "third wave waits whole"

        assert inference.serve_once(slots_map, evaluator, config) == 2
        assert evaluator.batches[1] == [20.0, 21.0]
        assert int(slots_map[2].row_count_echo[0]) == 2
