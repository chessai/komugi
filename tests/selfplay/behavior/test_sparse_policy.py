"""The sparse-policy transport (KOM-038), tested through real server processes.

KOM-038 changes what the server puts on the wire, on **both** sides of the
seam: with ``InferenceConfig.sparse_policy=True`` the response carries only the
legal moves' logits -- ascending action id, float32, packed from the policy
region's offset 0 -- and the client rebuilds the dense ``(A,)`` row from the
mask it submitted. There is no index array on the wire, so the *only* thing
tying a transported value back to an action id is agreement between the two
ends about that mask and about the ordering. Every test here exists because
that agreement can break silently.

What each test catches, in the vocabulary of the pinned contract (design rev2):

**(1) Equivalence on the legal support (C9, C5).** The transport is not an
approximation: for every legal action the sparse round trip must deliver the
*same bits* the dense round trip delivers, and every illegal lane must be
exactly ``0.0`` -- a finite fill, never ``-inf`` and never NaN, because a
downstream ``exp`` over a poisoned lane produces a plausible-looking policy
rather than a crash. :func:`test_sparse_matches_dense_on_legal_support` runs
one corpus through a spawned dense server and the same corpus through a spawned
sparse server and demands ``sparse == where(legal, dense, 0.0)`` byte for byte.
Batch composition is pinned equal on both legs (``max_batch=1``, so every
forward is a single row) for the KOM-104 GEMM reason the neighbouring
``test_pipelined_matches_serial_corpus`` documents: cross-composition agreement
is a 1e-5 claim, and only equal composition earns equality.

**(2) Mode mismatch is loud in both directions (C3, C4).** The dangerous
failure is not a crash, it is a *quiet* one: a dense reader handed a sparse
payload finds K floats followed by whatever the previous response left behind,
and every value it reads is finite and plausible. The u32 ``policy_count``
field exists to make that unreadable rather than misreadable -- sparse writes
K, dense writes the ``0xFFFFFFFF`` sentinel, and each reader rejects the other's
value. :func:`test_mode_mismatch_poisons_the_client` reaches this at the
process level with nothing but public configuration: the spawned server gets
one config and the spawned client another. That is legal precisely *because* of
C3 -- ``policy_count`` is in the layout in both modes, so the two configs
describe the same bytes and a mismatched pair can actually meet. The test
asserts that equality first; if it ever fails, the layout became
mode-dependent, which is itself the contract breaking.

**(3) Sparse survives real traffic (soak).** Six spawned workers, hundreds of
round trips, every request carrying a *different* mask. Masks that differ per
request are the point: a server that packed K from the wrong row's mask, or a
client that reconstructed against a stale one, produces a self-consistent
answer for a *fixed* mask and garbage here. Every response is checked in full
against the stub net's closed form, and the doorbell ledger closes the count so
a server that quietly stopped answering cannot pass.

**(4) The ends of the K range (C2).** K=1 and K=ACTION_SIZE are the two rows
where an off-by-one in the pack (or a K that is one short) is either invisible
or fatal, and a row whose only legal actions are the first and last action ids
is where a descending pack looks exactly like an ascending one until you read
the values. The stub net's policy is ``base + action_id``, so the transported
value *encodes* the id it belongs to: any permutation of the pack order, and
any reconstruction that walks the mask the wrong way, lands a value at an index
whose closed form it does not match.

Suite discipline is inherited from :mod:`inference_rig` and not re-derived:
spawn context only, every spawned process joined inside an explicit wall (a
hang fails in bounded time instead of wedging ``ci/check.sh``), tiny stub nets
on CPU, per-test segments created and unlinked by the rig so nothing here can
collide with another module's shm names, and the module under test reached
lazily so this file stays green on a tree where KOM-038 has not landed.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inference_rig as rig

inf = pytest.importorskip("komugi.selfplay.inference")


def _sparse_is_implemented() -> bool:
    """Is the KOM-038 transport actually in this tree?

    ``hasattr(InferenceConfig, "sparse_policy")`` is NOT the probe: the field
    has been on the config since KOM-021 as a *reserved* knob whose
    ``__post_init__`` raises :class:`NotImplementedError` when it is set, so the
    attribute is present on a base that cannot serve a single sparse request.
    The field check therefore only guards against a tree where the seam knob
    itself is gone; constructing the config is what distinguishes "reserved"
    from "implemented".
    """
    fields = {field.name for field in dataclasses.fields(inf.InferenceConfig)}
    if "sparse_policy" not in fields:
        return False
    try:
        rig.make_config(sparse_policy=True)
    except NotImplementedError:
        return False
    return True


if not _sparse_is_implemented():
    pytest.skip(
        "InferenceConfig.sparse_policy is still the reserved P0a knob that "
        "raises NotImplementedError (KOM-038 has not landed in this tree): "
        "every assertion below is about a wire format no server here can "
        "write, so they would all be about code that does not exist",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# closed-form expectations for the encoder stub
# ---------------------------------------------------------------------------


def _encoder_dense(code: int) -> npt.NDArray[np.float64]:
    """What :class:`inference_rig.EncoderStub` computes for `code`, densely.

    ``10 * code * OBS_CELLS + action_id``. Every value is an exactly
    representable float32 integer at the codes this module uses (the largest is
    under 4e6, well below 2**24), so comparisons against it are *equality*, not
    tolerance -- which is what lets a single wrong lane be an unambiguous
    failure rather than a judgement call.

    The ``+ action_id`` ramp is the load-bearing half: the value transported
    for an action encodes which action it is, so a pack that walked the mask in
    descending order, or a reconstruction that filled the legal positions out of
    order, cannot produce a row that matches this.
    """
    ramp = np.arange(rig.NUM_ACTIONS, dtype=np.float64)
    base = 10.0 * float(code) * float(rig.OBS_CELLS)
    return np.asarray(ramp + base, dtype=np.float64)


def _sparse_problem(
    response: Any, code: int, legal: npt.NDArray[np.bool_]
) -> str | None:
    """Describe how a sparse response for `code`/`legal` is wrong, or None.

    A string rather than an exception because the caller is usually a spawned
    worker reporting through a queue, exactly as
    :func:`inference_rig.check_response` (whose de-ramp decode this replaces:
    that helper assumes a dense row and a sparse one is deliberately not one).
    """
    policy = np.asarray(response.policy_logits, dtype=np.float64)
    if policy.shape != (rig.NUM_ACTIONS,):
        return f"policy_logits shape {policy.shape}, want {(rig.NUM_ACTIONS,)}"
    if not bool(np.all(np.isfinite(policy))):
        return (
            f"policy_logits has non-finite lanes at "
            f"{np.nonzero(~np.isfinite(policy))[0][:4].tolist()} -- the illegal "
            f"fill is exactly 0.0, never -inf or NaN (contract C5)"
        )

    want = np.where(legal, _encoder_dense(code), 0.0)
    wrong = np.nonzero(policy != want)[0]
    if wrong.size:
        index = int(wrong[0])
        return (
            f"policy_logits[{index}] == {policy[index]!r}, want {want[index]!r} "
            f"(legal={bool(legal[index])}, K={int(legal.sum())}, code={code}, "
            f"{wrong.size} lanes wrong) -- legal lanes carry the dense value "
            f"bit for bit and illegal lanes are exactly 0.0 (C2/C5/C9)"
        )

    moves_left = rig.as_scalar(response.moves_left)
    if abs(moves_left - code) > 1e-2:
        return f"moves_left {moves_left}, submitted code {code}"

    wdl = np.asarray(response.wdl, dtype=np.float64)
    if wdl.shape != (inf.WDL_SIZE,):
        return f"wdl shape {wdl.shape}, want {(inf.WDL_SIZE,)}"
    if abs(float(wdl.sum()) - 1.0) > 1e-4:
        return f"wdl does not sum to 1 (got {wdl.tolist()}) -- softmax missing?"
    if int(np.argmax(wdl)) != code % 3:
        return f"wdl argmax {int(np.argmax(wdl))}, want {code % 3} for code {code}"
    return None


# ---------------------------------------------------------------------------
# (1) dense/sparse equivalence on the legal support
# ---------------------------------------------------------------------------

#: Each worker ships one message: policy ``(N, A)``, wdl ``(N, 3)``,
#: moves_left ``(N,)``, all float32 and all compared byte for byte.
_CORPUS_SIZE = 48
_CORPUS_SEED = 20260824
_CORPUS_WORKERS = 2

#: Corpus rows with hand-set masks, so the equivalence leg covers the same K
#: extremes the dedicated edge test covers -- through a *different* net (the
#: random conv, whose values are not integers) and a *different* oracle (the
#: dense run rather than a closed form).
_K1_ROW = 0
_K1_ACTION = 7
_KFULL_ROW = 1


def _corpus() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Deterministic ``(obs, legal)`` requests -- same in every process."""
    rng = np.random.default_rng(np.random.SeedSequence(_CORPUS_SEED))
    obs = rng.standard_normal(
        (_CORPUS_SIZE, rig.IN_PLANES, rig.BOARD, rig.BOARD)
    ).astype(np.float32)
    legal = np.asarray(
        rng.random((_CORPUS_SIZE, rig.NUM_ACTIONS)) < 0.35, dtype=np.bool_
    )
    # A position with no legal move never crosses the seam, and K=0 is outside
    # the contract (there would be nothing to pack).
    legal[:, 0] = True
    legal[_K1_ROW, :] = False
    legal[_K1_ROW, _K1_ACTION] = True
    legal[_KFULL_ROW, :] = True
    return obs, legal


def _answer_worker_main(
    config: Any,
    handle: Any,
    obs: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
    barrier: Any,
    errors: Any,
    answers: Any,
) -> None:
    """Submit the whole corpus through the served path; ship the answers."""
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
        # Stacked, not per-row: one ~16 KB message per worker stays well inside
        # the queue's pipe buffer, where a few hundred pickled arrays would not.
        answers.put(
            (
                worker_id,
                np.stack(policies),
                np.stack(wdls),
                np.concatenate(moves_left),
            )
        )
    except BaseException as exc:  # reported, not raised: the parent asserts
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def _run_corpus(
    config: Any, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> tuple[list[Any], list[str]]:
    """Serve the corpus to `_CORPUS_WORKERS` spawned clients under `config`."""
    worker_ids = list(range(_CORPUS_WORKERS))
    failures: list[str] = []

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
            name="sparse-corpus-server",
        )
        workers = [
            rig.spawn(
                _answer_worker_main,
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

        failures.extend(rig.join_all(workers, rig.SPAWN_WALL_S, errors))
        # collect rather than drain: joinable and readable are not the same
        # instant for a queue the child wrote to just before exiting.
        collected = rig.collect(answers, _CORPUS_WORKERS, rig.JOIN_GRACE_S)
        if not server.is_alive():
            failures.append("the server died before the corpus was answered")
        failures.extend(rig.stop_server(server, stop_event))

    if len(collected) != _CORPUS_WORKERS:
        failures.append(
            f"only {len(collected)} of {_CORPUS_WORKERS} workers shipped answers"
        )
    return collected, failures


def _first_mismatch(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> int | None:
    """Index of the first row that differs by a single bit, or None."""
    if got.shape != want.shape:
        return 0
    differing = np.nonzero(np.any((got != want).reshape(got.shape[0], -1), axis=1))[0]
    return int(differing[0]) if differing.size else None


def test_sparse_matches_dense_on_legal_support() -> None:
    """One corpus, two spawned servers, identical bits where it is defined.

    The contract's central claim (C9): sparse policy return is a *transport*
    change, so for every legal action the sparse answer equals the dense answer
    exactly, and the only permitted difference anywhere in the row is that the
    illegal lanes read exactly ``0.0`` instead of the network's unmasked value
    (C5). Both halves are asserted as one array equality against
    ``where(legal, dense, 0.0)``, so a sparse row that dropped the last legal
    action, shifted the pack by one, filled illegal lanes with ``-inf``, or
    quietly returned the dense row untouched all fail on the first offending
    lane -- with that lane's index, its K and its legality in the message.

    Values and aux outputs are compared unmasked and bit for bit: nothing about
    the policy wire format may perturb wdl or moves_left, and a server that
    reused a response buffer across modes would show up there first.

    Batch composition is held equal (``max_batch=1``: every forward is one row
    on both legs) because equality is only the honest assertion at equal
    composition -- the KOM-104 GEMM lesson that ``test_equivalence_corpus``
    gates at 1e-5 and ``test_pipelined_matches_serial_corpus`` documents.
    """
    obs, legal = _corpus()
    dense_config = rig.make_config(
        max_batch=1, batch_timeout_ms=0.5, request_timeout_s=60.0
    )
    sparse_config = rig.make_config(
        max_batch=1, batch_timeout_ms=0.5, request_timeout_s=60.0, sparse_policy=True
    )

    dense, dense_failures = _run_corpus(dense_config, obs, legal)
    sparse, sparse_failures = _run_corpus(sparse_config, obs, legal)

    assert dense_failures == [], f"dense leg: {dense_failures}"
    assert sparse_failures == [], f"sparse leg: {sparse_failures}"

    reference = {int(item[0]): item for item in dense}
    ref_policy = reference[0][1]
    ref_wdl = reference[0][2]
    ref_mlh = reference[0][3]

    # Anti-vacuity, three ways: a reference of zeros would make every equality
    # below true; a reference whose illegal lanes were already 0.0 would make
    # the "exactly 0.0" half of the claim free; and a corpus with no illegal
    # lanes at all would make the masking untested.
    assert float(np.max(np.abs(ref_policy))) > 1e-3, (
        f"the dense reference policy logits are ~0 "
        f"({float(np.max(np.abs(ref_policy)))!r})"
    )
    illegal = ~legal
    assert bool(np.any(illegal)), "the corpus has no illegal lanes to mask"
    assert bool(np.any(ref_policy[illegal] != 0.0)), (
        "every illegal lane of the DENSE reference is already 0.0, so the "
        "sparse fill assertion below would pass without transporting anything"
    )
    for worker_id, got_policy, _wdl, _mlh in dense:
        row = _first_mismatch(got_policy, ref_policy)
        assert row is None, (
            f"the two dense workers disagree at corpus row {row} (worker "
            f"{worker_id}) -- the reference below is not well defined, so the "
            f"sparse comparison would be meaningless"
        )

    want_policy = np.where(legal, ref_policy, np.float32(0.0)).astype(np.float32)
    for worker_id, got_policy, got_wdl, got_mlh in sparse:
        assert got_policy.shape == (_CORPUS_SIZE, rig.NUM_ACTIONS), (
            f"sparse worker {worker_id} reconstructed {got_policy.shape}, want "
            f"{(_CORPUS_SIZE, rig.NUM_ACTIONS)} -- the client rebuilds the dense "
            f"row from the mask it submitted (C2/C5)"
        )
        assert bool(np.all(np.isfinite(got_policy))), (
            f"sparse worker {worker_id} returned non-finite policy lanes -- the "
            f"illegal fill is exactly 0.0, never -inf or NaN (C5)"
        )
        row = _first_mismatch(got_policy, want_policy)
        if row is not None:
            bad = np.nonzero(got_policy[row] != want_policy[row])[0]
            index = int(bad[0])
            pytest.fail(
                f"sparse worker {worker_id}: corpus row {row} lane {index} is "
                f"{got_policy[row][index]!r}, want {want_policy[row][index]!r} "
                f"(legal={bool(legal[row][index])}, K={int(legal[row].sum())}, "
                f"{bad.size} lanes wrong) -- the sparse round trip must deliver "
                f"the dense bits on the legal support and exactly 0.0 elsewhere "
                f"(contract C5/C9)"
            )
        for field, got, want in (
            ("wdl", got_wdl, ref_wdl),
            ("moves_left", got_mlh, ref_mlh),
        ):
            aux_row = _first_mismatch(got, want)
            assert aux_row is None, (
                f"sparse worker {worker_id}: {field} differs from the dense run "
                f"at corpus row {aux_row} -- sparse policy return is a policy "
                f"transport change and must not touch the value or aux outputs"
            )


# ---------------------------------------------------------------------------
# (2) mode mismatch, both directions
# ---------------------------------------------------------------------------

#: The mismatch must be caught when the response is READ, not waited out. A
#: client that failed to notice and then blocked would eventually raise
#: InferenceTimeout, which is an exception too -- so the test discriminates on
#: promptness as well as on the raise, and the two walls below are what make
#: "detected" and "waited out" different observations.
_MISMATCH_TIMEOUT_S = 60.0

#: The first submit races the server's own startup: a spawned child imports
#: torch from scratch (seconds on a cold page cache) and there is no warmup
#: round trip available here, because in this scenario *every* submit fails.
#: 20 s is far above any plausible cold start and far below the 60 s an
#: undetected mismatch costs.
_MISMATCH_FIRST_WALL_S = 20.0

#: By the second submit the server is warm and idle, so a client that still
#: goes to the wire is answered in milliseconds. 10 s is pure load allowance.
_MISMATCH_SECOND_WALL_S = 10.0

_MISMATCH_CODE = 5


def _mismatch_mask() -> npt.NDArray[np.bool_]:
    """A mask whose popcount can never be confused with the dense sentinel.

    Every other action, so K is 41: unequal to ``0xFFFFFFFF`` (which is what a
    sparse reader must reject) and unequal to A (which would let a dense
    reader's stale-tail read look accidentally complete).
    """
    mask = np.zeros(rig.NUM_ACTIONS, dtype=np.bool_)
    mask[::2] = True
    return mask


def _mismatch_worker_main(
    config: Any,
    handle: Any,
    reports: Any,
) -> None:
    """Submit twice under a config the server does not share; report both.

    Two submits, not one: C4 says a violation *poisons* the client, so the
    second call must fail as well. A client that raised once and then carried
    on would be a client whose next submit reads a slot the server is free to
    be rewriting.
    """
    obs = rig.encode_obs(_MISMATCH_CODE)
    legal = _mismatch_mask()
    try:
        client = rig.make_client(config, handle)
        for attempt in ("first", "second"):
            started = time.monotonic()
            try:
                response = client.submit(obs, legal)
            except BaseException as exc:
                reports.put(
                    (
                        attempt,
                        "raised",
                        f"{type(exc).__name__}: {exc}"[:400],
                        time.monotonic() - started,
                    )
                )
            else:
                head = np.asarray(response.policy_logits, dtype=np.float64)[:4]
                reports.put(
                    (
                        attempt,
                        "returned",
                        f"a NetEval whose policy head is {head.tolist()}",
                        time.monotonic() - started,
                    )
                )
    except BaseException as exc:
        reports.put(("setup", "crashed", f"{type(exc).__name__}: {exc}"[:400], 0.0))


def _run_mismatch(
    server_config: Any, client_config: Any
) -> tuple[list[Any], list[str]]:
    """One spawned server and one spawned client that disagree about the mode."""
    failures: list[str] = []
    with rig.slots(server_config, [0]) as handles:
        stop_event = rig.CTX.Event()
        reports = rig.CTX.Queue()

        server = rig.spawn(
            rig.server_main,
            rig.encoder_factory,
            server_config,
            handles,
            stop_event,
            name="mismatch-server",
        )
        worker = rig.spawn(
            _mismatch_worker_main,
            client_config,
            handles[0],
            reports,
            name="mismatch-client",
        )

        failures.extend(rig.join_all([worker], rig.SPAWN_WALL_S))
        collected = rig.collect(reports, 2, rig.JOIN_GRACE_S)
        if not server.is_alive():
            failures.append(
                "the server died before the client could read a response -- the "
                "mismatch must poison the CLIENT, not take the shard down"
            )
        failures.extend(rig.stop_server(server, stop_event))
    return collected, failures


def _assert_loud_mismatch(
    direction: str, reports: list[Any], failures: list[str]
) -> None:
    """Both submits raised, promptly, and no NetEval ever came back."""
    assert failures == [], f"{direction}: {failures}"
    by_attempt = {str(item[0]): item for item in reports}
    assert "setup" not in by_attempt, (
        f"{direction}: the client never reached a submit ({by_attempt['setup']}) "
        f"-- the mismatch is a READ-time contract violation, so building the "
        f"client must still work"
    )
    assert sorted(by_attempt) == ["first", "second"], (
        f"{direction}: reports {sorted(by_attempt)}, want both submits -- a "
        f"missing report means the client wedged or died without saying how"
    )

    _, outcome, detail, elapsed = by_attempt["first"]
    assert outcome == "raised", (
        f"{direction}: the first submit returned {detail} instead of failing. "
        f"That is the exact failure C3/C4 exist to delete: a reader handed the "
        f"other mode's payload finds finite, plausible floats and cannot tell "
        f"they are the wrong ones. The u32 policy_count (K when sparse, the "
        f"0xFFFFFFFF sentinel when dense) must be checked against the mask the "
        f"client submitted and the client poisoned on disagreement"
    )
    assert detail.strip(), f"{direction}: the client failed with an empty message"
    assert float(elapsed) < _MISMATCH_FIRST_WALL_S, (
        f"{direction}: the client took {float(elapsed):.3f}s to fail ({detail}), "
        f"budget {_MISMATCH_FIRST_WALL_S:.3f}s of a {_MISMATCH_TIMEOUT_S}s "
        f"request timeout -- a mode mismatch is detected when the response is "
        f"read, so a wall this long means the client waited the request out "
        f"instead of checking policy_count"
    )

    _, second_outcome, second_detail, second_elapsed = by_attempt["second"]
    assert second_outcome == "raised", (
        f"{direction}: the second submit returned {second_detail} -- the client "
        f"must not carry on past a mode violation (C4's poison). A client that "
        f"keeps going re-rings its doorbell while the server may still be "
        f"writing the slot it is about to read"
    )
    assert float(second_elapsed) < _MISMATCH_SECOND_WALL_S, (
        f"{direction}: the second submit took {float(second_elapsed):.3f}s to "
        f"fail ({second_detail}), budget {_MISMATCH_SECOND_WALL_S:.3f}s against "
        f"a warm server -- a poisoned client fails immediately and even a client "
        f"that re-detects on the wire is answered in milliseconds here"
    )


def test_mode_mismatch_poisons_the_client() -> None:
    """Sparse reader vs dense server, and dense reader vs sparse server.

    Both directions are reachable through public configuration alone, and the
    reason is itself part of the contract: C3 puts ``policy_count`` in the slot
    layout in **both** modes, so a dense config and a sparse config describe the
    same bytes and a mismatched pair can be pointed at one segment at all. The
    test asserts that byte-size equality first -- if it fails, the layout became
    mode-dependent and the mismatch this test exists for turned into a segment
    sizing bug instead.

    So no private seam is needed: the spawned server gets one config, the
    spawned client gets the other, and everything else is the ordinary rig.
    What is asserted is the whole of C4 -- the error surfaces (never a NetEval,
    however finite and plausible its floats look), it surfaces at read time
    rather than after ``request_timeout_s``, it carries a message, the client
    does not carry on (the next submit fails too, and immediately -- a poisoned
    client and one that merely re-detects are indistinguishable from out here,
    and both satisfy the property that matters: no submit after a violation
    ever returns a NetEval), and the *server* survives, because a misconfigured
    worker must not take its shard down with it.
    """
    dense_config = rig.make_config(
        max_batch=1, batch_timeout_ms=0.5, request_timeout_s=_MISMATCH_TIMEOUT_S
    )
    sparse_config = rig.make_config(
        max_batch=1,
        batch_timeout_ms=0.5,
        request_timeout_s=_MISMATCH_TIMEOUT_S,
        sparse_policy=True,
    )

    assert inf.slot_nbytes(dense_config) == inf.slot_nbytes(sparse_config), (
        f"dense slots are {inf.slot_nbytes(dense_config)} bytes and sparse ones "
        f"{inf.slot_nbytes(sparse_config)} -- C3 puts policy_count in the layout "
        f"in BOTH modes, so the layout must not depend on the mode at all"
    )

    _assert_loud_mismatch(
        "sparse client against a dense server",
        *_run_mismatch(dense_config, sparse_config),
    )
    _assert_loud_mismatch(
        "dense client against a sparse server",
        *_run_mismatch(sparse_config, dense_config),
    )


# ---------------------------------------------------------------------------
# (3) sparse soak
# ---------------------------------------------------------------------------

_SOAK_WORKERS = 6
_SOAK_PER_WORKER = 120
_SOAK_SEED = 20260825

#: Codes are ``worker_id * stride + k + 1``: distinct across the whole run and
#: never 0 (a zero code is indistinguishable from an unwritten slot). The
#: stride exceeds the per-worker count, so no two ranges can overlap however
#: the run is scheduled. The largest code here (worker 5's last, 1120) puts
#: ``10 * code * OBS_CELLS`` at 3.6e6, comfortably
#: below 2**24 and therefore an exactly representable float32 integer -- which
#: is what lets every response be checked by equality rather than tolerance.
_SOAK_CODE_STRIDE = 200

#: Well above any plausible per-request latency at this batch size, and small
#: enough that a wedged worker fails the test in half a minute rather than at
#: the spawn wall.
_SOAK_REQUEST_TIMEOUT_S = 30.0


def _soak_masks(worker_id: int) -> npt.NDArray[np.bool_]:
    """This worker's per-request masks -- a different one every submit.

    Varying K per request is the whole point of the soak. A server that packed
    K from the wrong row of the batch's mask, or a client that reconstructed
    against a stale mask, still answers a *fixed*-mask workload consistently;
    it cannot answer this one. The first two rows are pinned at the extremes so
    every worker exercises K=1 and K=A under real batching pressure.
    """
    rng = np.random.default_rng(_SOAK_SEED + worker_id)
    masks = np.asarray(rng.random((_SOAK_PER_WORKER, rig.NUM_ACTIONS)) < 0.3)
    masks[:, 0] = True
    masks[0, :] = False
    masks[0, 3] = True
    masks[1, :] = True
    return masks.astype(np.bool_)


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
    """Submit a few hundred (tag, mask) pairs; check every answer in full."""
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        masks = _soak_masks(worker_id)
        # Co-arrival: without it the workers trickle in and the server never
        # batches, so the mixed-K rows never share a forward.
        barrier.wait(rig.BARRIER_WALL_S)
        done = 0
        for index, code in enumerate(_soak_codes(worker_id)):
            legal = masks[index]
            problem = _sparse_problem(
                client.submit(rig.encode_obs(code), legal), code, legal
            )
            if problem is not None:
                errors.put(
                    f"worker {worker_id} request {done} (code {code}, "
                    f"K={int(legal.sum())}): {problem}"
                )
                return
            done += 1
        dones.put((worker_id, done))
    except BaseException as exc:  # reported, not raised: the parent asserts
        errors.put(f"worker {worker_id} raised {type(exc).__name__}: {exc}")


def _read_counters(config: Any, handles: Any) -> dict[int, tuple[int, int]]:
    """Final ``(request_seq, response_seq)`` per worker, straight from /dev/shm.

    The doorbell counters are the server's ledger: ``request_seq`` is the number
    of round trips a worker started, ``response_seq`` is the ticket the server
    last published back. Reading them in the parent after every worker has
    exited turns "every request was answered exactly once" into an arithmetic
    check that does not depend on any worker's own report.
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


def test_sparse_soak_under_real_batching() -> None:
    """Six workers, 720 mixed-K sparse round trips, every answer checked.

    The sustained-traffic gate for the sparse transport: real batching (a
    ``max_batch`` of 16 over six workers, so rows with different K genuinely
    share a forward), a different mask on every request, and a full closed-form
    check of every response rather than a spot check -- the stub net's
    ``base + action_id`` policy means each legal lane must equal a value that
    encodes both the submitter's tag and the action id it is filed under.

    What that catches beyond the equivalence test: a pack that used the batch's
    first row's K for every row, a scatter that wrote row N's K floats into row
    M's slot, a reconstruction that reused the previous submit's mask, and any
    drift that only appears after a few hundred cycles. No timeout and no
    poison is asserted by the empty failure list (an
    :class:`InferenceTimeout` or a mode-mismatch poison would arrive as a
    worker report), and the doorbell ledger closes the count so a server that
    quietly stopped answering cannot pass a green.
    """
    config = rig.make_config(
        max_batch=16,
        batch_timeout_ms=0.5,
        request_timeout_s=_SOAK_REQUEST_TIMEOUT_S,
        sparse_policy=True,
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
            name="sparse-soak-server",
        )
        workers = [
            rig.spawn(
                _soak_worker_main,
                config,
                handle,
                barrier,
                errors,
                dones,
                name=f"sparse-soak-{handle.worker_id}",
            )
            for handle in handles
        ]

        failures = rig.join_all(workers, rig.SPAWN_WALL_S, errors)
        reported = dict(rig.collect(dones, _SOAK_WORKERS, rig.JOIN_GRACE_S))
        server_alive = server.is_alive()
        # Every worker has exited, so nothing is writing these segments; the
        # server is still up but has nothing left to answer.
        counters = _read_counters(config, handles)
        failures.extend(rig.stop_server(server, stop_event))

    assert failures == [], failures
    assert server_alive, "the server died while its workers were still submitting"
    # Anti-vacuity: a worker that skipped its loop would otherwise leave a green
    # with no sparse round trips behind it.
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
# (4) the ends of the K range
# ---------------------------------------------------------------------------

#: ``(code, legal)`` pairs, each naming the edge it pins. K=1 and
#: K=ACTION_SIZE are the ends of the range; the both-ends row is where an
#: ascending pack and a descending one differ in exactly two lanes and nowhere
#: else, so the ordering half of C2 has a case that isolates it.
_EDGE_K1_ACTION = 7
_EDGE_CODES = (11, 22, 33)


def _edge_cases() -> list[tuple[int, npt.NDArray[np.bool_]]]:
    """One request per K edge, all through the same public ``submit``."""
    single = np.zeros(rig.NUM_ACTIONS, dtype=np.bool_)
    single[_EDGE_K1_ACTION] = True

    ends = np.zeros(rig.NUM_ACTIONS, dtype=np.bool_)
    ends[0] = True
    ends[rig.NUM_ACTIONS - 1] = True

    return [
        (_EDGE_CODES[0], single),
        (_EDGE_CODES[1], ends),
        (_EDGE_CODES[2], rig.all_legal()),
    ]


def _edge_worker_main(config: Any, handle: Any, reports: Any) -> None:
    """Submit each edge case; ship the reconstructed row for the parent."""
    worker_id = handle.worker_id
    try:
        client = rig.make_client(config, handle)
        for code, legal in _edge_cases():
            response = client.submit(rig.encode_obs(code), legal)
            reports.put(
                (
                    code,
                    _sparse_problem(response, code, legal),
                    np.asarray(response.policy_logits, dtype=np.float64),
                )
            )
    except BaseException as exc:
        reports.put(
            (-1, f"worker {worker_id} raised {type(exc).__name__}: {exc}", None)
        )


def test_sparse_edges_k_one_and_k_action_size() -> None:
    """K=1, both-ends K=2 and K=ACTION_SIZE, through the real served path.

    The ends of the range are where the pack's arithmetic is least constrained.
    At K=1 a length bug is invisible in the payload (one float is one float
    however you count it) and only the reconstruction's *position* can be wrong;
    at K=ACTION_SIZE the sparse payload exactly fills the dense policy region,
    so a pack that writes K+1 values, or a reader that stops one short, runs
    into the region's edge rather than into slack. Both are constructible
    through the public surface -- the mask is a ``submit`` argument -- so both
    are here.

    The stub net makes each transported value encode the action it belongs to
    (``10 * code * OBS_CELLS + action_id``), which is what turns the both-ends
    row into an ordering test: with only actions 0 and A-1 legal, a descending
    pack produces a row that is finite, correctly K-sized and wrong in exactly
    two lanes, and ascending order (C2) is the only thing that distinguishes
    them.

    The K=1 row also carries the anti-vacuity for the zero fill in the opposite
    direction from the equivalence test: exactly one lane is nonzero, so an
    implementation that returned an all-zero row -- which would satisfy "0.0 at
    every illegal position" perfectly -- fails here.
    """
    config = rig.make_config(
        max_batch=1, batch_timeout_ms=0.5, request_timeout_s=30.0, sparse_policy=True
    )
    cases = _edge_cases()

    with rig.slots(config, [0]) as handles:
        stop_event = rig.CTX.Event()
        reports = rig.CTX.Queue()

        server = rig.spawn(
            rig.server_main,
            rig.encoder_factory,
            config,
            handles,
            stop_event,
            name="sparse-edge-server",
        )
        worker = rig.spawn(
            _edge_worker_main, config, handles[0], reports, name="sparse-edge-client"
        )

        failures = rig.join_all([worker], rig.SPAWN_WALL_S)
        collected = rig.collect(reports, len(cases), rig.JOIN_GRACE_S)
        if not server.is_alive():
            failures.append("the server died before the edge cases were answered")
        failures.extend(rig.stop_server(server, stop_event))

    assert failures == [], failures
    assert len(collected) == len(cases), (
        f"{len(collected)} of {len(cases)} edge cases came back: {collected}"
    )

    rows = {int(item[0]): item for item in collected}
    for code, legal in cases:
        assert code in rows, (
            f"no answer came back for the K={int(legal.sum())} case (code {code})"
        )
        _, problem, policy = rows[code]
        assert problem is None, f"K={int(legal.sum())} (code {code}): {problem}"

        nonzero = np.nonzero(policy != 0.0)[0]
        assert nonzero.tolist() == np.nonzero(legal)[0].tolist(), (
            f"K={int(legal.sum())} (code {code}): nonzero lanes {nonzero.tolist()} "
            f"but legal lanes {np.nonzero(legal)[0].tolist()} -- every legal lane "
            f"of this stub is nonzero, so the two sets coincide exactly; an "
            f"all-zero row would satisfy the illegal-fill rule and nothing else"
        )
