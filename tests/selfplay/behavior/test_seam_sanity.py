"""The small, exact promises of the inference seam (KOM-121).

Everything here is in-process and cheap; it exists because the expensive tests
next door all assume it.  Four groups:

* **``wdl_utility``** -- design D1 keeps the scalar value OUT of the seam: wdl
  travels, utility is derived at read time as ``w*1 + d*draw_score + l*(-1)``.
  That derivation is the *only* place DESIGN.html section 3.3's contempt
  (draw_score 0.20) enters a value, so it gets goldens including the sign
  behaviour at 0.2 -- a drawish position must be worth *more* than zero to the
  mover, which is the whole point, and a sign slip here is a silently
  draw-seeking engine.
* **field order** -- ``NetEval`` is a NamedTuple so callers may unpack it
  positionally; renaming or reordering its fields is an API break that no
  shape check would catch.
* **D8's deferred knob** -- ``policy_dtype='f16'`` is reserved in the seam and
  must raise ``NotImplementedError``.  A knob that is silently ignored is worse
  than one that is missing: the caller believes it got fp16 transfers.  Its
  twin ``sparse_policy=True`` is no longer deferred -- KOM-038 implemented the
  sparse transport, so that one must be *accepted*, and the test below pins the
  accepting.
* **``slot_nbytes``** -- the segment holds ONE request/response pair, not a
  batch and not a double buffer.  See the layout arithmetic below.
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

_DRAW_SCORE = 0.20


def test_neteval_field_order_is_the_documented_one() -> None:
    """``policy_logits``, ``wdl``, ``moves_left`` -- in that order (D1)."""
    assert sp.NetEval._fields == ("policy_logits", "wdl", "moves_left")
    # And no scalar value field: D1/F3 removed it deliberately, because a
    # value in the seam is a contempt convention frozen server-side.
    assert "value" not in sp.NetEval._fields


@pytest.mark.parametrize(
    ("wdl", "draw_score", "want"),
    [
        ((1.0, 0.0, 0.0), _DRAW_SCORE, 1.0),
        ((0.0, 0.0, 1.0), _DRAW_SCORE, -1.0),
        ((0.0, 1.0, 0.0), _DRAW_SCORE, _DRAW_SCORE),
        ((0.0, 1.0, 0.0), 0.0, 0.0),
        ((0.0, 1.0, 0.0), -_DRAW_SCORE, -_DRAW_SCORE),
        ((0.5, 0.5, 0.0), _DRAW_SCORE, 0.5 + 0.5 * _DRAW_SCORE),
        ((0.25, 0.5, 0.25), _DRAW_SCORE, 0.5 * _DRAW_SCORE),
        ((0.2, 0.3, 0.5), _DRAW_SCORE, 0.2 - 0.5 + 0.3 * _DRAW_SCORE),
    ],
)
def test_wdl_utility_goldens(
    wdl: tuple[float, float, float], draw_score: float, want: float
) -> None:
    """Hand-computed utilities, including the contempt sign at draw_score 0.2."""
    got = sp.wdl_utility(np.asarray([wdl], dtype=np.float32), draw_score)
    values = np.asarray(got, dtype=np.float64).reshape(-1)
    assert values.shape == (1,)
    assert values[0] == pytest.approx(want, abs=1e-6)


def test_wdl_utility_is_vectorised_and_float32() -> None:
    """(N, 3) in, (N,) f32 out -- the batch form the adapter actually calls."""
    wdl = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    got = sp.wdl_utility(wdl, _DRAW_SCORE)
    assert np.asarray(got).shape == (3,)
    assert np.asarray(got).dtype == np.float32
    assert np.allclose(np.asarray(got), [1.0, _DRAW_SCORE, -1.0], atol=1e-6)


def test_a_pure_draw_is_worth_more_than_a_pure_loss_and_less_than_a_win() -> None:
    """The ordering contempt exists to create (DESIGN.html section 3.3)."""
    wdl = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    win, draw, loss = np.asarray(sp.wdl_utility(wdl, _DRAW_SCORE)).tolist()
    assert loss < 0.0 < draw < win
    assert draw == pytest.approx(_DRAW_SCORE, abs=1e-6)


def _tiny_model() -> Any:
    return rig.random_factory()


def test_fp16_policy_transfer_raises_not_implemented() -> None:
    """D8: the knob is reserved in the seam, not honoured in P0a."""
    with pytest.raises(NotImplementedError):
        # Either site is acceptable -- the config may reject it outright, or
        # the evaluator may on construction.  What is not acceptable is
        # accepting it and quietly transferring float32.
        sp.InProcessEvaluator(_tiny_model(), rig.make_config(policy_dtype="f16"))


def test_sparse_policy_return_is_honoured_not_refused() -> None:
    """D8's other knob, opposite rule since KOM-038: it is honoured, not refused.

    Neither site may raise any more -- not the config, not the evaluator on
    construction.  Refusing here would be the mirror of the failure the fp16
    test guards: a knob the seam documents as implemented, rejected at the one
    place a caller can reach it.
    """
    config = rig.make_config(sparse_policy=True)
    evaluator = sp.InProcessEvaluator(_tiny_model(), config)
    assert evaluator.config.sparse_policy is True


def test_default_config_leaves_the_deferred_knobs_off() -> None:
    """The defaults are the supported path, so nothing raises by accident."""
    config = rig.make_config()
    assert config.policy_dtype == "f32"
    assert config.sparse_policy is False
    assert config.shards == 1
    assert config.max_batch == 16
    assert config.batch_timeout_ms == pytest.approx(0.5)
    assert config.request_timeout_s == pytest.approx(60.0)
    # KOM-040 b2: the wave transport ships with a width of one, which is the
    # single-row protocol this seam has always run. Enabling waves is a config
    # change, and the layout is sized for it either way.
    assert config.max_wave == 1


# ---------------------------------------------------------------------------
# slot layout
# ---------------------------------------------------------------------------

#: One slot, hand-computed at in_planes=4 / num_actions=81 / max_wave=1:
#:
#:   doorbells request_seq   1 uint64 =              8 B
#:             response_seq  1 uint64 =              8 B
#:   request   row_count     1 uint32 =              4 B
#:             obs      4*9*9 float32 = 324 * 4 = 1296 B
#:             legal    81    bool    =            81 B
#:   response  row_count_echo 1 uint32 =             4 B
#:             policy_count  1 uint32 =              4 B
#:             policy   81    float32 =           324 B
#:             wdl      3     float32 =            12 B
#:             mlh      1     float32 =             4 B
#:                                      ------------------
#:                                                1745 B
#:
#: The two counters are the KOM-021b doorbells: 16 bytes, aligned, one written
#: only by the worker and one only by the server.  They are the ENTIRE
#: signalling channel, which is why they belong in the packed size rather than
#: in the padding slack -- a slot that did not carry them would have to carry a
#: lock instead.
#:
#: The three u32 fields are payload, not signalling.  ``policy_count`` is
#: KOM-038's mode tag (one entry per wave row); ``row_count`` and
#: ``row_count_echo`` are KOM-040 b2's layout-v2 wave headers -- the request's
#: width and the server's echo of it -- and they are present unconditionally, at
#: every ``max_wave`` including the default 1, which is why they are re-derived
#: into the packed size here rather than absorbed into the padding slack.
_DOORBELL_NBYTES = 2 * 8
_HEADER_NBYTES = 4 + 4 + 4  # row_count, row_count_echo, policy_count
_PACKED_NBYTES = (
    _DOORBELL_NBYTES
    + _HEADER_NBYTES
    + (4 * 9 * 9 * 4)
    + 81
    + (81 * 4)
    + (3 * 4)
    + 4
)

#: Alignment slack the layout may spend on padding between the five regions
#: (8-byte alignment on five boundaries cannot cost more than this).  It is
#: deliberately far too small to hide a second copy of any region: doubling
#: the response alone would cost 340 B.
_PADDING_SLACK = 64


def test_slot_nbytes_matches_the_hand_computed_layout() -> None:
    """One request + one response, and nothing else, in each segment."""
    config = rig.make_config()
    nbytes = sp.slot_nbytes(config)
    assert nbytes >= _PACKED_NBYTES, (
        f"slot_nbytes {nbytes} cannot hold the {_PACKED_NBYTES} B a slot needs "
        f"(doorbells 16 + row_count 4 + obs 1296 + legal 81 + row_count_echo 4 "
        f"+ policy_count 4 + policy 324 + wdl 12 + mlh 4)"
    )
    assert nbytes <= _PACKED_NBYTES + _PADDING_SLACK, (
        f"slot_nbytes {nbytes} exceeds the packed layout ({_PACKED_NBYTES} B) by "
        f"more than alignment can explain -- a batch-sized or double-buffered "
        f"slot changes the ownership story in design D2"
    )


@pytest.mark.parametrize(
    ("field", "value", "extra"),
    [
        # +81 actions costs one more legal byte and one more policy float.
        ("num_actions", 2 * rig.NUM_ACTIONS, rig.NUM_ACTIONS * 5),
        # +4 planes costs 4*81 observation floats.
        ("in_planes", 2 * rig.IN_PLANES, rig.IN_PLANES * rig.BOARD * rig.BOARD * 4),
    ],
)
def test_slot_nbytes_scales_with_the_shape_it_carries(
    field: str, value: int, extra: int
) -> None:
    """The marginal byte cost is the layout's, not a rounded-up block size."""
    base = sp.slot_nbytes(rig.make_config())
    grown = sp.slot_nbytes(rig.make_config(**{field: value}))
    # Alignment slack can SHRINK as well as grow when a dimension doubles
    # (base pads 3 bytes, the grown layout may pad fewer), so the lower bound
    # concedes the padding, not just the upper (wave-9a review F3).
    assert extra - _PADDING_SLACK <= grown - base <= extra + _PADDING_SLACK, (
        f"doubling {field} moved slot_nbytes by {grown - base} B, want ~{extra} B"
    )


def test_created_segment_is_at_least_one_slot_and_readable() -> None:
    """``create_slots`` really does back each handle with a segment."""
    config = rig.make_config()
    with rig.slots(config, [3]) as handles:
        (handle,) = handles
        assert handle.worker_id == 3
        assert isinstance(handle.shm_name, str) and handle.shm_name
        # The handle carries a NAME and nothing else: with the doorbell
        # protocol the signalling counters live inside the segment, so there is
        # no event (and no lock a SIGKILLed worker could orphan) to hand over.
        assert not hasattr(handle, "response_ready")

        segment = sp.attach_untracked(handle.shm_name)
        try:
            assert len(segment.buf) >= sp.slot_nbytes(config)
        finally:
            segment.close()

        # Both doorbells must start at 0 -- a nonzero response_seq would let
        # the very first submit read an unwritten slot, which is the same
        # failure a pre-set response event used to be able to cause.
        attached = sp.attach_slots(handle.shm_name, config)
        try:
            assert int(attached.request_seq[0]) == 0
            assert int(attached.response_seq[0]) == 0
        finally:
            attached.close()


# ---------------------------------------------------------------------------
# the in-process evaluator, which the equivalence corpus uses as its reference
# ---------------------------------------------------------------------------


def test_in_process_evaluator_returns_a_well_formed_neteval() -> None:
    """Shapes, dtypes, and wdl actually being a distribution."""
    config = rig.make_config()
    evaluator = sp.InProcessEvaluator(_tiny_model(), config)
    rows = 3
    obs = np.zeros((rows, rig.IN_PLANES, rig.BOARD, rig.BOARD), dtype=np.float32)
    legal = np.ones((rows, rig.NUM_ACTIONS), dtype=np.bool_)

    out = evaluator.evaluate(obs, legal)
    assert isinstance(out, sp.NetEval)

    policy = np.asarray(out.policy_logits)
    wdl = np.asarray(out.wdl)
    moves_left = np.asarray(out.moves_left)
    assert policy.shape == (rows, rig.NUM_ACTIONS)
    assert wdl.shape == (rows, 3)
    assert moves_left.shape == (rows,)
    for array in (policy, wdl, moves_left):
        assert array.dtype == np.float32
    assert np.all(np.isfinite(policy))
    # Probabilities, not logits: D1 says the seam carries the softmax so every
    # consumer derives utility the same way.
    assert np.all(wdl >= 0.0)
    assert np.allclose(wdl.sum(axis=1), 1.0, atol=1e-5)


def test_in_process_evaluator_is_batch_independent() -> None:
    """Row i of a batch equals the same row evaluated alone.

    The server is allowed to group requests however it likes; that is only
    safe because the evaluator's rows do not talk to each other.  If this ever
    fails, the equivalence corpus next door is measuring the wrong thing.
    """
    config = rig.make_config()
    evaluator = sp.InProcessEvaluator(_tiny_model(), config)
    rng = np.random.default_rng(11)
    obs = rng.standard_normal((4, rig.IN_PLANES, rig.BOARD, rig.BOARD)).astype(
        np.float32
    )
    legal = np.ones((4, rig.NUM_ACTIONS), dtype=np.bool_)

    batched = evaluator.evaluate(obs, legal)
    for index in range(4):
        alone = evaluator.evaluate(obs[index : index + 1], legal[index : index + 1])
        assert np.allclose(
            np.asarray(batched.policy_logits)[index],
            np.asarray(alone.policy_logits)[0],
            atol=1e-5,
        )
        assert np.allclose(
            np.asarray(batched.wdl)[index], np.asarray(alone.wdl)[0], atol=1e-5
        )


def test_no_grad_leaks_through_the_seam() -> None:
    """The server forwards under ``no_grad`` (D2); nothing returned is a graph.

    A NetEval carrying live autograd tensors would keep the whole forward graph
    alive per in-flight request -- the memory model in design D9 assumes it
    does not, and numpy conversion at the shared-memory boundary assumes it
    cannot happen either.
    """
    config = rig.make_config()
    evaluator = sp.InProcessEvaluator(_tiny_model(), config)
    obs = np.zeros((1, rig.IN_PLANES, rig.BOARD, rig.BOARD), dtype=np.float32)
    legal = np.ones((1, rig.NUM_ACTIONS), dtype=np.bool_)

    out = evaluator.evaluate(obs, legal)
    for field in sp.NetEval._fields:
        value = getattr(out, field)
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad, f"{field} came back attached to a graph"
