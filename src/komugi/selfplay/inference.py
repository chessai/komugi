"""The batching inference seam: one GPU forward serving many self-play workers.

``docs/DESIGN.html`` §7.3 asks for multi-consumer inference because the serial
in-process loop tops out around 100-160 evaluations/second; the reviewed design
``docs/design/kom-021-selfplay.md`` (D1, D2, D3, D8) turns that into the concrete
protocol this module implements. Three pieces, each usable without the others:

* **The seam itself** -- :class:`NetEval`, :class:`BatchEvaluator`,
  :func:`wdl_utility`. Numpy crosses the seam, torch never does. There is no
  scalar ``value`` field (design D1/F3): win-draw-loss *probabilities* travel and
  the scalar utility is a read-time derivation, which is what lets the draw score
  of §3.3 be applied by the reader, in the perspective of the side to move at the
  node where the value enters the tree.
* **The shared-memory slot protocol** -- :func:`create_slots`,
  :func:`attach_slots`, :func:`slot_nbytes`, :func:`attach_untracked`. One
  segment per worker, holding that worker's request and its response. The byte
  layout is written down once, in :func:`_layout`, and nothing else may know it.
* **The two ends** -- :class:`SlotClient` (worker side) and :class:`ShardServer`
  (server side), plus :class:`InProcessEvaluator`, which is both the server's
  internal forward and the whole evaluator in ``--serial-debug`` runs.

Constraints this module is built under, all from the reviewed design:

* **Never a pickled model across a process boundary.** A model enters either as
  the ``model`` argument of :class:`InProcessEvaluator` or as the
  ``model_factory`` of :class:`ShardServer`, which is called *inside* the server
  process (design D2). :func:`serve_shard` is a module-level function so a
  spawned process has something importable to target.
* **The orchestrator owns segment lifetime.** KOM-023 creates and unlinks every
  segment in a ``finally``; workers and the server only ever *attach*, and they
  do it through :func:`attach_untracked` (design D2/F9, bpo-38119).
* **Timeout is fatal.** :class:`SlotClient` raises :class:`InferenceTimeout` and
  the caller is expected to die rather than resubmit; a fresh incarnation gets a
  fresh segment, which is what makes stale responses and torn reads impossible
  by construction rather than by locking (design D2/Q1).
* **No device literals.** Nothing here calls ``.cuda()`` or names a device; the
  caller places the model and the evaluator follows it. KOM-039's fp16 serving
  forms do not weaken that: they *read* the model's device type to decide
  whether a reduced-precision forward is possible at all -- and the autocast
  form names ``"cuda"`` to torch's autocast API, which takes a device *type*
  and not a device -- but nothing here ever moves a model. KOM-048's captured
  forward reads the model's device the same way and allocates its static
  buffer *there*; what it adds is the opposite constraint -- once a graph is
  captured the model may no longer move at all, and every launch says so
  loudly rather than replaying stale addresses (see :class:`_GraphRunner`).

Doorbell protocol
-----------------

Signalling is **lock-free**. Rev 1 of this module signalled with a shared
``multiprocessing.Queue`` of worker ids plus one ``mp.Event`` per worker; both
are built on POSIX semaphores, and both take an internal lock a worker holds
for a few instructions on every submit (the queue's feeder thread holds the
write lock while it pickles and writes; ``Event.clear()`` takes the condition's
lock). Design D10 has the orchestrator's liveness gate **SIGKILL** wedged
workers, and SIGKILL cannot be deferred: a worker killed inside either critical
section leaves that lock held by a process that no longer exists, and every
*surviving* worker then blocks on it until :class:`InferenceTimeout`. The
KOM-121 SIGKILL-survival test caught exactly this, intermittently -- the
failure is in the signalling primitives, not in the shared memory.

So the signalling moved into the segment itself. Each slot carries two
8-byte-aligned ``uint64`` counters, both starting at 0:

``request_seq``
    Written **only** by that worker. Incremented once per submit, *after* the
    request payload is in the slot. That store is the doorbell.
``response_seq``
    Written **only** by the server. Set to the ``request_seq`` value the server
    gathered, *after* that worker's response bytes are in the slot.

The invariant is a single line: ``request_seq > response_seq`` means "this
worker has a request pending", and ``response_seq == request_seq`` means "the
answer for that request is in the slot, complete". Both sides poll; neither
takes a lock, opens a file descriptor, or touches a semaphore.

Two properties make that safe without any locking at all:

* **Single writer per counter.** Nobody ever read-modify-writes a counter
  another process also writes, so there is no increment to lose and no CAS to
  need. The two counters are separate 8-byte fields precisely so that the two
  writers never share a cache line's worth of *ownership*, whatever they share
  of the line itself.
* **Aligned 8-byte stores are atomic on x86-64.** A naturally aligned 64-bit
  load or store is a single indivisible memory operation on this
  architecture, so a reader can never observe a half-written counter -- a torn
  counter would be the one way a stale response could be mistaken for a fresh
  one. :func:`_layout` therefore places both counters at offsets 0 and 8, and
  keeps the alignment explicit rather than incidental.

Ordering is the rest of the protocol, and it runs the same way in both
directions: **payload first, counter second**. A reader that sees the new
counter value is guaranteed the payload it describes was written before it.
Program order becomes cross-process visibility order through x86-TSO: stores
are not reordered with other stores, and numpy's C calls cannot be reordered
across bytecode boundaries by any compiler in play.

Honesty about what that argument rests on (KOM-021b review F2/F3) -- three
dependencies that hold today and are named so a change to any of them is
recognized as a protocol event, not a refactor:

* C guarantees nothing about a plain store being one instruction; the real
  guarantee is that numpy's setitem on an ALIGNED, native-dtype uint64 view
  takes the scalar fast path (``*(npy_uint64*)ptr = v``), which every
  mainstream x86-64 compiler emits as a single ``mov``. Verified at runtime
  for these views (base page-aligned, offsets 0/8, ALIGNED flag set).
* x86-TSO's store ordering excludes *non-temporal* stores, which glibc's
  memcpy uses only above a multi-megabyte threshold; every payload copy here
  is tens of kilobytes at most, far below it (and glibc fences its NT paths
  anyway). **KOM-040 b2 makes this dependency explicit rather than incidental.**
  A slot now carries up to ``max_wave`` rows, so the natural temptation is to
  scatter a whole wave with one bulk ``slots.policy[:rows] = ...`` copy -- and a
  wave-sized copy is exactly what could cross the NT threshold at production
  shapes. It is therefore forbidden: every gather and every scatter on this
  path copies **one row at a time** (see :func:`_scatter`,
  :func:`_scatter_sparse`, :func:`write_response`), and one row is at most
  ``4 * num_actions`` bytes -- ~150 KB at the P0a action space, two orders of
  magnitude below any NT path glibc takes. Introducing a multi-row bulk copy
  here is a **protocol event**, not a refactor: it would put the
  payload-before-counter argument back in the hands of memcpy's size heuristic.
* The whole argument is **x86-64-specific.** On ARMv8 the aligned stores stay
  atomic but store->store ordering is NOT architectural and Python offers no
  fence -- payload-before-counter would silently stop being a visibility
  guarantee. Production is x86-64; porting this module to another
  architecture requires revisiting the protocol, not just recompiling.

What this buys, stated as the property the SIGKILL test now pins: a worker
killed at *any* instruction -- mid-``pack_request``, between the payload and
the doorbell, while polling -- holds nothing anyone else needs. The worst it
can leave behind is its own slot with ``request_seq > response_seq``, which the
server answers once, into a segment nobody reads, and then forgets.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import resource_tracker, synchronize
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch

BOARD_SIZE = 9
"""Board edge; observations are ``(in_planes, 9, 9)`` throughout komugi."""

PLANE_CELLS = BOARD_SIZE * BOARD_SIZE
"""Cells in one observation plane."""

WDL_SIZE = 3
"""Win, draw, loss -- the three components of the value head's output."""

_F32_BYTES = 4

_U32_BYTES = 4
"""Width of the ``policy_count`` field (KOM-038). Four bytes, 4-aligned, so it
slots in between the request's alignment pad and the response floats without
disturbing anyone's alignment."""

DENSE_POLICY_COUNT = 0xFFFFFFFF
"""What a **dense** response writes into ``policy_count`` (KOM-038, design D4).

The slot layout is byte-identical in both modes -- deliberately, so that
enabling ``sparse_policy`` is a configuration change and not a re-plumbing --
which means a worker attached with one mode against a server running the other
would pass the :func:`_views` size check and then read finite garbage: packed
logits at the wrong action ids, plus whatever the tail of the region last held.
Nothing downstream would fire, because every byte of it is a valid float.

So the count field is also the mode tag, and **both** readers check it:
:func:`read_response` requires this sentinel, :func:`read_sparse_response`
requires ``count == mask.sum()``. A mismatch in either direction is a loud
:class:`InferenceProtocolError` that poisons the client.

It cannot collide with a real ``K``: ``K <= num_actions``, and a 4-billion-wide
action space would have overflowed the segment long before it reached here.
"""

_U64_BYTES = 8
"""Width of a doorbell counter. Also its required alignment: an aligned 8-byte
store is what makes a torn read impossible on x86-64 (see the module docstring).
"""

_CLIENT_POLL_S = 0.0005
"""How long :meth:`SlotClient.submit` sleeps between two reads of
``response_seq``.

Half a millisecond is the whole latency tax the doorbell charges over the
event-wakeup it replaces, and it buys the property that a worker waiting for an
answer holds no cross-process lock (design D2 as amended by KOM-021b). It is
also the granularity of ``request_timeout_s``, which is measured against
:func:`time.monotonic` rather than counted in polls.

Half a millisecond **deliberately**, and re-measured rather than inherited: the
KOM-040(a) poll matrix (2026-08-25) shortened this poll too and the seam went
backwards -- 0.1 ms measured -0.7% and 0.05 ms -2.8% against the same control
that :data:`_SERVER_IDLE_POLL_S` gained +3.0% on -- because there is one poller
per worker here and 28 of them, so a finer client poll spends on wakeups
exactly the CPU the workers need to generate the next request.
"""

_SERVER_IDLE_POLL_S = 0.0002
"""How long :meth:`ShardServer.serve` sleeps after a scan that found nothing.

An idle shard must not busy-spin a core (design D2), and with no queue left to
block on this sleep is the entire mechanism that keeps that promise. It also
bounds ``stop_event`` shutdown latency and the added latency of a request that
arrives just after an empty scan.

**0.2 ms, not the 1 ms this started at, because that latency was measured.**
The KOM-032 and KOM-038 reviews both flagged the same mechanism from the code
alone: a request that rings just after an empty scan waits out the whole sleep
before anyone looks again, costing ~0.5 ms of completion-detection latency per
cycle on a path each worker traverses ~87.6 times a move. The KOM-040(a) poll
matrix (2026-08-25 -- seven cells, with a doubled control pinning run-to-run
noise below 0.1%) measured **18,018 games/day at 0.2 ms against 17,496 at
1 ms, +3.0%**, replicated at 18,017 on a confirmation run. The core this
shortens the sleep on is one server process's, not the 28 workers'; the same
experiment run on :data:`_CLIENT_POLL_S` came back negative, for the reason
recorded there. A future change to this value is a re-bench, not a guess.

It is also the granularity of :func:`serve_pipelined`'s collect-while-busy
poll, which is the one place the server loops without an idle sleep -- and it
does so only while a forward is actually in flight, so the idle promise above
is untouched (design review D8, invariant I3).
"""


class NetEval(NamedTuple):
    """One network evaluation per row, as it crosses the inference seam.

    Field order is part of the seam: callers unpack positionally
    (``policy, wdl, mlh = evaluator.evaluate(...)``) and the shared-memory
    layout below is written in this order too.

    There is deliberately **no scalar value field** (design D1, review F3). A
    scalar collapses win/draw/loss before the reader has said what a draw is
    worth to it; §3.3's contempt is a property of the reader, not of the
    network. Use :func:`wdl_utility` at the point where the value enters the
    tree.

    Attributes:
        policy_logits: ``(N, A)`` float32. Index is the engine's action id.
            Masking is the search's job -- the seam carries ``legal`` so that
            the server can pack the transport (D8/KOM-038), not so that the
            evaluator can apply it.

            **Raw at legal positions; fill elsewhere under
            ``sparse_policy``.** With the sparse transport only the legal
            actions' logits cross the seam, and :class:`SlotClient` rebuilds
            the full vector with a finite ``0.0`` in every illegal lane
            (design review D1). Consumers must not read a raw logit out of an
            illegal lane -- and none does: every one of them either indexes
            legal-only or routes through a mask whose illegal lane contributes
            exact ``0.0`` identically either way, which is what makes the
            sparse path bit-identical on the legal support and on every number
            derived from it. The fill is finite rather than ``-inf`` because
            the search validates the *whole* vector for finiteness before it
            masks anything.
        wdl: ``(N, 3)`` float32 *probabilities* in (win, draw, loss) order,
            summing to 1 along the last axis, from the perspective of the side
            to move. Logits are softmaxed server-side so every reader agrees.
        moves_left: ``(N,)`` float32 on the head's raw scale.
    """

    policy_logits: npt.NDArray[np.float32]
    wdl: npt.NDArray[np.float32]
    moves_left: npt.NDArray[np.float32]


class BatchEvaluator(Protocol):
    """Anything that evaluates a batch of positions.

    The single seam between self-play and the network: an in-process torch
    module (:class:`InProcessEvaluator`), a shared-memory round trip to a server
    shard (:class:`SlotClient`, batch of one), or a test stub all satisfy it.
    """

    def evaluate(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> NetEval:
        """Evaluate ``obs`` ``(N, in_planes, 9, 9)`` with mask ``legal`` ``(N, A)``."""
        ...


def wdl_utility(
    wdl: npt.NDArray[np.float32], draw_score: float
) -> npt.NDArray[np.float32]:
    """Collapse win/draw/loss probabilities to a scalar utility in ``[-1, 1]``.

    ``w * 1 + d * draw_score + l * (-1)``, per ``DESIGN.html`` §3.3 and design
    D1. The caller supplies ``draw_score`` in the perspective of the side to
    move at *this* node; the known negamax sign-oscillation of a constant draw
    utility is an accepted, documented P0a approximation (design D1/F3b), and
    the recorded ``wdl_target`` never sees contempt at all.

    Args:
        wdl: ``(N, 3)`` probabilities, (win, draw, loss).
        draw_score: What a draw is worth to the side to move.

    Returns:
        ``(N,)`` float32 utilities.

    Raises:
        ValueError: If `wdl` is not ``(N, 3)``.
    """
    if wdl.ndim != 2 or wdl.shape[1] != WDL_SIZE:
        raise ValueError(f"wdl must be (N, {WDL_SIZE}), got {wdl.shape!r}")
    win = wdl[:, 0]
    draw = wdl[:, 1]
    loss = wdl[:, 2]
    return np.asarray(win - loss + draw * np.float32(draw_score), dtype=np.float32)


class InferenceTimeout(RuntimeError):
    """A worker waited out ``request_timeout_s`` for a response.

    **Fatal by policy** (design D2/Q1): the worker exits nonzero and the
    orchestrator respawns it against a *fresh* segment. Nothing may catch this
    and resubmit -- a retry against the same slot is exactly the stale-response
    race the fresh-segment rule exists to delete.
    """


class InferenceProtocolError(RuntimeError):
    """The bytes in a slot do not describe the protocol both ends agreed on.

    Raised by the two response readers when ``policy_count`` disagrees with the
    mode they are reading in (KOM-038, design review D4): a sparse reader whose
    count is not ``mask.sum()``, or a dense reader whose count is not
    :data:`DENSE_POLICY_COUNT`. Both are configuration errors -- one end has
    ``sparse_policy`` set and the other does not, or a codec bug packed the
    wrong number of values -- and both are **fatal** in the same way a timeout
    is: :class:`SlotClient` poisons itself and refuses to submit again. There is
    no correct way to continue, because the only alternative is to carry on with
    priors that are finite, plausible and wrong.
    """


FORWARD_DTYPES = ("f32", "f16", "f16_autocast")
"""The three compute forms :attr:`InferenceConfig.forward_dtype` accepts.

``"f32"`` is today's forward, unchanged to the bit. ``"f16"`` is the primary
serving form (KOM-039): the weights are converted **once**, at evaluator
construction, and only on CUDA. ``"f16_autocast"`` is the measurement form --
torch's own mixed precision, per call, around the model forward and nothing
else -- kept so the two can be benched against each other on the same run
rather than argued about from first principles.
"""

FORWARD_GRAPH_MODES = ("off", "on")
"""The two forms :attr:`InferenceConfig.forward_graph` accepts (KOM-048).

``"off"`` is today's launch, byte-identical, and is what the whole suite runs
under. ``"on"`` captures the model forward **once**, at a fixed shape, as a
:class:`torch.cuda.CUDAGraph` and replays it per batch -- which is a different
thing from making the forward faster: it deletes the per-launch kernel-launch
CPU cost, which is what a 9x9 net's short kernels are actually bounded by.

Two values and no third on purpose. There is no ``"auto"``: a mode that decides
for itself whether it captured would make "was this bench graphed?" a question
nobody could answer from the config, and every refusal below exists precisely so
that a run which believes it is graphing never silently measures eager.
"""


@dataclass(frozen=True)
class InferenceConfig:
    """Shape and timing of the inference seam.

    Args:
        in_planes: Observation planes per position (128 in P0a).
        num_actions: Size of the action space, ``A``.
        max_batch: Server-side batch ceiling, in **rows**. The *effective*
            ceiling is ``min(max_batch, max_wave * len(clients))``
            (:func:`_row_ceiling`): a batch can never exceed what the live
            workers can have in flight, because each worker has at most one
            ticket pending and a ticket carries at most ``max_wave`` rows
            (design D2/F16, KOM-040 b2).
        batch_timeout_ms: How long the server keeps collecting after the first
            pending request, in the **idle** state only. The latency knob.
            KOM-032 lowered the default from 2.0 to 0.5, the measured optimum:
            with :func:`serve_pipelined` the collect for batch N+1 overlaps
            forward N and is bounded by the forward, not by this window (I3),
            so the window's only remaining job is to let a couple of stragglers
            join the batch that starts a burst. At 2.0 it was pure serial
            latency on every idle-to-busy transition.
        shards: Number of server processes (design D3). P0a default 1;
            worker->shard assignment is sticky and lives in the orchestrator.
        request_timeout_s: Worker-side patience before :class:`InferenceTimeout`.
        policy_dtype: The **wire** dtype of the policy: what the policy bytes
            *are* where they cross the shared-memory seam, still float32-only
            and reserved for the fp16 policy transfer of design D8. Its
            compute-side twin is ``forward_dtype``, and the two are
            independent -- an fp16 forward still writes float32 into the slot.
        sparse_policy: Transport only the LEGAL actions' logits across the seam
            (design D8, KOM-038). The response's policy region then carries
            ``K`` float32 values in ascending action-id order and nothing else
            -- no index array, because the worker rebuilds the positions from
            the very mask bytes it submitted -- and ``policy_count`` carries
            ``K``. The dense ``(A,)`` vector is reconstructed by
            :class:`SlotClient` with a finite ``0.0`` in every illegal lane, so
            every number the search derives is bit-identical (design review D1).

            **Both ends must agree**, and this is not checkable from here: the
            server and every worker it serves must be built from the same
            config. The count field is the tripwire that makes a disagreement
            loud rather than silently wrong -- see :data:`DENSE_POLICY_COUNT`.

            Default OFF at merge. The flip to ON is gated on a one-time
            on-hardware bitwise sparse-vs-dense comparison of the legal support:
            the CPU suite proves the codec but structurally cannot reach the
            device-side pack (design review D3/7).
        max_wave: How many rows one ticket may carry (KOM-040 b2, design rev2).
            ``1`` is today's protocol: one position per submit, one doorbell per
            position. Above 1 a search may submit a whole *wave* of leaves as a
            single ticket -- :meth:`SlotClient.submit_wave` -- which is the
            lever the b1 bench identified: the shared batch already synchronises
            the lanes, so the thing left to cut is the **count** of round trips,
            not their width.

            The layout is sized for ``max_wave`` rows **unconditionally** (see
            :func:`_layout`), exactly as ``policy_count`` is present in both
            transport modes: both ends compute the segment size from this one
            config, so a disagreement is a size mismatch caught at attach rather
            than a silent misread.

            ``max_wave <= max_batch`` is enforced here and is the foundation of
            the server's no-split admission rule (:func:`_admit`): a wave is
            never split across two forwards, so the highest-priority pending
            wave must always fit a batch that starts empty. Without this
            invariant a wave wider than the batch could never be served at all.
        forward_dtype: The **compute** dtype of the server-side forward
            (KOM-039), one of :data:`FORWARD_DTYPES`: what the model's own
            arithmetic runs in, which is exactly what ``policy_dtype`` is not
            -- the wire stays float32 in every form here.

            ``"f16"`` converts the weights once, at evaluator construction,
            and only when the model is on CUDA
            (:func:`_half_with_f32_bn`, which keeps every ``BatchNorm2d`` in
            float32 -- eval-mode BN is the numerically fragile op and
            everything else runs in half). ``"f16_autocast"`` leaves the
            weights alone and wraps the forward -- and nothing else -- in
            :func:`torch.autocast` per call.

            **On CPU both forms run the f32 forward, bit-identically**: the
            conversion is never applied and autocast is never enabled off
            CUDA, so ``--serial-debug`` and the CPU-only gate see exactly what
            ``"f32"`` produces. That is a pinned property, not an
            implementation accident.

            **Under ``"f16"`` the model must stay on the device it was
            constructed on**: the weight conversion happens once, gated on the
            construction-time device, while the per-call input cast follows
            the current device -- moving the model afterwards fails loudly
            with a dtype mismatch on the next forward (impl review F4). The
            other two forms keep the device-following contract.
        forward_graph: Serve the forward from a **captured CUDA graph**
            (KOM-048), one of :data:`FORWARD_GRAPH_MODES`. ``"off"`` is
            today's launch to the byte. ``"on"`` captures the model forward
            once, at the fixed shape ``(max_batch, in_planes, 9, 9)``, and
            every launch becomes: copy the rows into the captured static
            input, zero the pad, replay, then the same eager tail (softmax,
            KOM-038 gather, event) this module always ran.

            The knob is honoured by :class:`AsyncEvaluator` **only**, and only
            on CUDA; :class:`InProcessEvaluator` never captures anything (one
            capture site, so there is one thing to reason about). It is also
            refused rather than degraded in three cases, all at evaluator
            construction and all by name: on a ROCm torch build, with a
            host-resident model, and together with
            ``forward_dtype="f16_autocast"``. See
            :func:`_refuse_ungraphable` -- a bench that believes it is
            replaying a graph must never quietly measure the eager path.

            **Under ``"on"`` the model is pinned**, harder than
            ``forward_dtype="f16"`` pins it: the capture bakes in the
            parameters' *addresses*, so the model may not move device, and its
            weights may only be updated by in-place copy (``load_state_dict``
            with the default ``assign=False``). Rebinding ``param.data``,
            ``load_state_dict(..., assign=True)`` or swapping a submodule
            leaves the graph replaying the old memory, silently.

    Raises:
        ValueError: On a non-positive count, a negative duration, a
            ``max_wave`` above ``max_batch``, a ``forward_dtype`` outside
            :data:`FORWARD_DTYPES`, or a ``forward_graph`` outside
            :data:`FORWARD_GRAPH_MODES`.
        NotImplementedError: If ``policy_dtype`` is set. The seam *carries* the
            knob so that enabling it later is configuration rather than
            re-plumbing, but P0a transfers float32.
    """

    in_planes: int
    num_actions: int
    max_batch: int = 16
    batch_timeout_ms: float = 0.5
    shards: int = 1
    request_timeout_s: float = 60.0
    policy_dtype: str = "f32"
    sparse_policy: bool = False
    max_wave: int = 1
    forward_dtype: str = "f32"
    forward_graph: str = "off"

    def __post_init__(self) -> None:
        """Reject impossible shapes, an unknown precision, and the deferred knob."""
        for field, value in (
            ("in_planes", self.in_planes),
            ("num_actions", self.num_actions),
            ("max_batch", self.max_batch),
            ("shards", self.shards),
            ("max_wave", self.max_wave),
        ):
            if value < 1:
                raise ValueError(f"{field} must be >= 1, got {value!r}")
        for field, duration in (
            ("batch_timeout_ms", self.batch_timeout_ms),
            ("request_timeout_s", self.request_timeout_s),
        ):
            if not duration >= 0.0:  # also rejects NaN
                raise ValueError(f"{field} must be >= 0, got {duration!r}")
        if self.max_wave > self.max_batch:
            # Loud at construction, because the alternative is a wave that no
            # forward can ever admit: the server never splits a ticket across
            # two batches, so a wave wider than the batch ceiling would be
            # skipped at every rank, forever, until the client timed out.
            raise ValueError(
                f"max_wave={self.max_wave} exceeds max_batch={self.max_batch}: a "
                f"wave is never split across forwards, so it must always fit a "
                f"batch that starts empty (KOM-040 b2, admission rule W4)"
            )
        if self.forward_dtype not in FORWARD_DTYPES:
            raise ValueError(
                f"forward_dtype must be one of {list(FORWARD_DTYPES)!r}, got "
                f"{self.forward_dtype!r}"
            )
        if self.forward_graph not in FORWARD_GRAPH_MODES:
            raise ValueError(
                f"forward_graph must be one of {list(FORWARD_GRAPH_MODES)!r}, "
                f"got {self.forward_graph!r}"
            )
        if self.policy_dtype != "f32":
            raise NotImplementedError(
                f"policy_dtype={self.policy_dtype!r}: fp16 policy transfer is a "
                f"reserved seam knob, enabled by KOM-021's P1 scope (design D8); "
                f"P0a transfers float32"
            )


class _Layout(NamedTuple):
    """Byte offsets into one worker's slot segment, plus its total size."""

    request_seq: int
    response_seq: int
    row_count: int
    obs: int
    legal: int
    row_count_echo: int
    policy_count: int
    policy: int
    wdl: int
    moves_left: int
    nbytes: int


def _layout(config: InferenceConfig) -> _Layout:
    """The one and only description of a slot segment's bytes.

    One :class:`~multiprocessing.shared_memory.SharedMemory` block per worker
    holds that worker's doorbells, its request *and* its response -- the request
    is dead by the time the response is written, and one block per worker means
    one name to create, attach and unlink. ``W`` below is ``config.max_wave``::

        offset                       bytes                   field
        ---------------------------------------------------------------------
        0                            8                       request_seq   u64
        8                            8                       response_seq  u64
        16                           4                       row_count     u32
        20                           4 * W * in_planes * 81  request obs   f32
        obs_end                      W * num_actions         request legal bool
        (padded to 4 bytes)          0..3                    alignment pad
        pad_end                      4                       row_count_echo u32
        echo_end                     4 * W                   policy_count  u32
        count_end                    4 * W * num_actions     response policy f32
        policy_end                   4 * W * 3               response wdl   f32
        wdl_end                      4 * W * 1               response mlh   f32

    **Layout v2 (KOM-040 b2) is unconditional**, in the same sense and for the
    same reason ``policy_count`` is: every region is sized for ``max_wave``
    rows and the two row-count headers are always present, so the size of a
    segment is a function of the config both ends were built from and nothing
    else. At ``max_wave == 1`` the shapes degenerate to exactly the v1
    semantics -- one observation, one mask, one scalar ``policy_count``, one
    answer -- and the slot is **exactly 8 bytes wider** than v1: 4 for
    ``row_count``, 4 for ``row_count_echo``.

    The two headers sit at the ends of their own writer's region, which is the
    single-writer discipline made geometric: the worker owns every byte from
    ``row_count`` through the end of ``legal``, the server owns every byte from
    ``row_count_echo`` to the end of the segment. Both are ordinary **payload**
    under the one-counter rule -- ``row_count`` is written before
    ``request_seq``, ``row_count_echo`` before ``response_seq`` -- and neither
    is a second doorbell. There is no new fence, because there is no new
    counter (design rev2, blocker D3').

    ``policy_count`` is present **unconditionally**, in both transport modes
    (KOM-038, design review D2), and in v2 it is a ``(W,)`` vector: one entry
    per row. Sparse responses write each row's ``K``, the number of legal
    actions in that row's mask, and fill that row's policy region's first ``K``
    floats; dense ones write :data:`DENSE_POLICY_COUNT` into every **used**
    row's entry and fill all ``A``. Entries at index ``>= row_count`` are
    **unobservable by contract**: the server does not write them and no reader
    may look at them, because the ticket only ever described ``row_count`` rows.
    Giving the count a region of its own -- rather than a prefix inside the
    policy region -- is what lets the ``K == A`` edge still fit, and keeps the
    region's offset arithmetic in this one function.

    The two counters come first, at offsets 0 and 8, so their 8-byte alignment
    is a property of the segment's own base address (POSIX shared memory is
    page-aligned) rather than of whatever happened to precede them. That
    alignment is load-bearing, not cosmetic: it is what makes each counter's
    store a single indivisible operation on x86-64, and therefore what makes
    the lock-free doorbell protocol in the module docstring safe.

    Their combined 16 bytes are a multiple of 4, and ``row_count`` adds four
    more, so every float32 view further down keeps the 4-byte alignment it had
    before either existed. The remaining pad exists so that ``row_count_echo``
    and the response floats behind it start 4-byte aligned; the bool block is
    the only one whose length is not already a multiple of 4.
    """
    rows = config.max_wave
    request_seq = 0
    response_seq = request_seq + _U64_BYTES
    row_count = response_seq + _U64_BYTES
    obs = row_count + _U32_BYTES
    legal = obs + _F32_BYTES * rows * config.in_planes * PLANE_CELLS
    row_count_echo = _align4(legal + rows * config.num_actions)
    policy_count = row_count_echo + _U32_BYTES
    policy = policy_count + _U32_BYTES * rows
    wdl = policy + _F32_BYTES * rows * config.num_actions
    moves_left = wdl + _F32_BYTES * rows * WDL_SIZE
    return _Layout(
        request_seq=request_seq,
        response_seq=response_seq,
        row_count=row_count,
        obs=obs,
        legal=legal,
        row_count_echo=row_count_echo,
        policy_count=policy_count,
        policy=policy,
        wdl=wdl,
        moves_left=moves_left,
        nbytes=moves_left + _F32_BYTES * rows,
    )


def _align4(offset: int) -> int:
    """Round `offset` up to the next multiple of 4."""
    return (offset + 3) & ~3


def slot_nbytes(config: InferenceConfig) -> int:
    """Return the size in bytes of one worker's slot segment."""
    return _layout(config).nbytes


@dataclass(eq=False)
class Slots:
    """Numpy views onto one worker's slot segment.

    ``eq=False``: a generated ``__eq__`` would compare numpy arrays and raise
    on the ambiguous truth value of the result. Slots are identities, not
    values.

    The arrays alias shared memory: writing to :attr:`obs` *is* the IPC. They
    are plain views, deliberately -- no locks, no double-buffering. What makes
    that safe is the protocol, not the container.

    **Single-writer discipline.** Every field has exactly one writer, and which
    process that is never changes:

    * ``request_seq`` -- written only by the WORKER, read by the server. The
      doorbell that says a request is ready.
    * ``row_count``, ``obs``, ``legal`` -- written only by the WORKER, read by
      the server.
    * ``row_count_echo``, ``policy_count``, ``policy``, ``wdl``, ``moves_left``
      -- written only by the SERVER, read by the worker.
    * ``response_seq`` -- written only by the SERVER, read by the worker. The
      doorbell that says the answer is ready.

    Because no counter is ever written by two processes, there is no increment
    to lose; and because :attr:`request_seq` and :attr:`response_seq` are
    8-byte aligned ``uint64`` fields (:func:`_layout` places them at offsets 0
    and 8), each store is a single indivisible memory operation on x86-64, so a
    reader can never see a torn counter. That -- plus writing the payload
    strictly *before* the counter that announces it -- is the entire safety
    argument for the doorbell protocol; see the module docstring. There is no
    lock anywhere in it, which is what lets a worker be SIGKILLed at any
    instruction without wedging its siblings (design D2 as amended by KOM-021b).

    The payload fields inherit their own single-writer discipline from the
    counters: the worker owns ``obs``/``legal`` until it rings the doorbell and
    does not touch them again until the answer arrives; the server owns the
    response fields for exactly the window between reading a request and
    publishing ``response_seq``. A segment is also never reused across worker
    incarnations (design D2), which is what keeps a timed-out worker's stale
    request from ever meeting a fresh one.

    Attributes:
    ``W`` below is ``config.max_wave``; at the default ``W == 1`` every shape
    degenerates to the single-row protocol this started as.

    Attributes:
        shm: The attached (or created) segment.
        request_seq: ``(1,)`` uint64 doorbell, worker-written, starts 0.
        response_seq: ``(1,)`` uint64 doorbell, server-written, starts 0.
        row_count: ``(1,)`` uint32 REQUEST header -- how many of the ``W`` rows
            this ticket actually carries, in ``1..W``. Worker-written, ordinary
            payload: it goes in before ``request_seq``, like the observations
            beside it, and is not a second doorbell.
        obs: ``(W, in_planes, 9, 9)`` float32 request. Rows at index
            ``>= row_count`` are stale and unobservable by contract.
        legal: ``(W, A)`` bool request, same row discipline.
        row_count_echo: ``(1,)`` uint32 RESPONSE header -- how many rows the
            server answered. Server-written payload, published before
            ``response_seq``; the client compares it to what it submitted, which
            is the tripwire that turns a ticket/answer width disagreement into a
            loud :class:`InferenceProtocolError` instead of a silently short
            wave (KOM-040 b2, W5).
        policy_count: ``(W,)`` uint32 response, one entry per row -- that row's
            ``K`` under ``sparse_policy``, :data:`DENSE_POLICY_COUNT` otherwise.
            Payload, not a doorbell: written inside the same
            payload-before-``response_seq`` window as the floats beside it, in
            no particular order relative to them, and the worker reads it only
            after its ticket comes back (design review D8). Do not fence it
            against the values; do not treat it as a second counter. One counter
            is the protocol. Entries at index ``>= row_count`` are never written
            and never read.
        policy: ``(W, A)`` float32 response. Under ``sparse_policy`` only row
            ``r``'s first ``policy_count[r]`` entries are written or meaningful.
        wdl: ``(W, 3)`` float32 response, probabilities.
        moves_left: ``(W,)`` float32 response.
    """

    shm: SharedMemory
    request_seq: npt.NDArray[np.uint64]
    response_seq: npt.NDArray[np.uint64]
    row_count: npt.NDArray[np.uint32]
    obs: npt.NDArray[np.float32]
    legal: npt.NDArray[np.bool_]
    row_count_echo: npt.NDArray[np.uint32]
    policy_count: npt.NDArray[np.uint32]
    policy: npt.NDArray[np.float32]
    wdl: npt.NDArray[np.float32]
    moves_left: npt.NDArray[np.float32]

    @property
    def max_wave(self) -> int:
        """How many rows this segment was sized for (``config.max_wave``)."""
        return int(self.obs.shape[0])

    def close(self) -> None:
        """Drop the views, then unmap the segment. Never unlinks (design D2)."""
        # The views export pointers into the mapping; SharedMemory.close()
        # raises BufferError while any of them is alive, so they go first.
        self.request_seq = np.empty(0, dtype=np.uint64)
        self.response_seq = np.empty(0, dtype=np.uint64)
        self.row_count = np.empty(0, dtype=np.uint32)
        self.obs = np.empty((0, 0, 0, 0), dtype=np.float32)
        self.legal = np.empty((0, 0), dtype=np.bool_)
        self.row_count_echo = np.empty(0, dtype=np.uint32)
        self.policy_count = np.empty(0, dtype=np.uint32)
        self.policy = np.empty((0, 0), dtype=np.float32)
        self.wdl = np.empty((0, 0), dtype=np.float32)
        self.moves_left = np.empty(0, dtype=np.float32)
        self.shm.close()


def _views(shm: SharedMemory, config: InferenceConfig) -> Slots:
    """Map `shm`'s buffer to typed arrays per :func:`_layout`."""
    layout = _layout(config)
    if shm.size < layout.nbytes:
        raise ValueError(
            f"segment {shm.name!r} is {shm.size} bytes; this config needs "
            f"{layout.nbytes}"
        )
    buffer = shm.buf
    rows = config.max_wave
    return Slots(
        shm=shm,
        request_seq=np.ndarray(
            (1,), dtype=np.uint64, buffer=buffer, offset=layout.request_seq
        ),
        response_seq=np.ndarray(
            (1,), dtype=np.uint64, buffer=buffer, offset=layout.response_seq
        ),
        row_count=np.ndarray(
            (1,), dtype=np.uint32, buffer=buffer, offset=layout.row_count
        ),
        obs=np.ndarray(
            (rows, config.in_planes, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
            buffer=buffer,
            offset=layout.obs,
        ),
        legal=np.ndarray(
            (rows, config.num_actions),
            dtype=np.bool_,
            buffer=buffer,
            offset=layout.legal,
        ),
        row_count_echo=np.ndarray(
            (1,), dtype=np.uint32, buffer=buffer, offset=layout.row_count_echo
        ),
        policy_count=np.ndarray(
            (rows,), dtype=np.uint32, buffer=buffer, offset=layout.policy_count
        ),
        policy=np.ndarray(
            (rows, config.num_actions),
            dtype=np.float32,
            buffer=buffer,
            offset=layout.policy,
        ),
        wdl=np.ndarray(
            (rows, WDL_SIZE), dtype=np.float32, buffer=buffer, offset=layout.wdl
        ),
        moves_left=np.ndarray(
            (rows,), dtype=np.float32, buffer=buffer, offset=layout.moves_left
        ),
    )


def attach_untracked(name: str) -> SharedMemory:
    """Attach an existing segment *without* registering it with the tracker.

    Python 3.12's :mod:`multiprocessing.shared_memory` registers every segment
    it touches -- including ones it merely attaches to -- with the
    ``resource_tracker``, which then unlinks them when the attaching process
    dies and warns about "leaked shared_memory objects" (bpo-38119; the
    ``track=`` parameter that fixes this properly arrives in 3.13). Left alone,
    one worker's death would unlink the segments its *siblings* are still using
    -- exactly the failure the KOM-121 SIGKILL test hunts for.

    So: attach, then immediately unregister. Ownership is unambiguous and lives
    elsewhere -- the orchestrator (KOM-023) creates every segment and unlinks it
    in a ``finally`` (design D2/F9). :class:`SlotClient` and :class:`ShardServer`
    attach *only* through this function.

    Args:
        name: Segment name, with or without the leading slash.

    Returns:
        The attached segment. The caller closes it; nobody but the orchestrator
        unlinks it.
    """
    shm = SharedMemory(name=name)
    # The tracker keys POSIX shared memory by its slash-prefixed name, which is
    # what SharedMemory itself registered (it prepends the slash on Linux while
    # reporting `.name` without it).
    tracked_name = name if name.startswith("/") else f"/{name}"
    resource_tracker.unregister(tracked_name, "shared_memory")
    return shm


def create_slots(config: InferenceConfig, name: str | None = None) -> Slots:
    """Create a new slot segment sized for `config` and return its views.

    For the **orchestrator only** (design D2): whoever calls this owns the
    segment and must ``shm.unlink()`` it in a ``finally``. It stays registered
    with the resource tracker at creation, but be honest about what that buys
    (wave-9a review F4): the tracker keys registrations by NAME in one shared
    tracker process, so the first ``attach_untracked`` from any worker or
    server deletes the creator's registration too -- from that point the
    tracker is NOT a backstop (an orchestrator that dies badly leaks the
    segment in /dev/shm), and the creator's own ``unlink()`` will provoke a
    harmless-but-noisy ``KeyError`` traceback from the tracker daemon. The
    orchestrator (KOM-023) must own cleanup explicitly and may re-register
    before unlink to silence the tracker.

    Args:
        config: Fixes the segment's size and the shape of every view.
        name: Segment name; ``None`` lets the OS pick one, readable afterwards
            as ``slots.shm.name``.
    """
    shm = SharedMemory(create=True, size=slot_nbytes(config), name=name)
    slots = _views(shm, config)
    # A fresh POSIX segment is already zero-filled, but "both doorbells start
    # at 0" is a protocol precondition, not a platform detail to be inherited:
    # a nonzero response_seq would make the worker's first submit read an
    # unwritten slot.
    slots.request_seq[0] = 0
    slots.response_seq[0] = 0
    # Same reasoning for the two v2 headers: "an unrung slot claims no rows" is
    # a protocol precondition, not a platform detail to be inherited.
    slots.row_count[0] = 0
    slots.row_count_echo[0] = 0
    return slots


def attach_slots(shm_name: str, config: InferenceConfig) -> Slots:
    """Attach to an existing slot segment (via :func:`attach_untracked`)."""
    return _views(attach_untracked(shm_name), config)


def pack_wave_request(
    slots: Slots,
    obs_rows: npt.NDArray[np.float32],
    legal_rows: npt.NDArray[np.bool_],
) -> int:
    """Write a whole wave's request into `slots` and return its row count.

    The **one** request writer (KOM-040 b2): :func:`pack_request` is this
    function with a single row, not a second code path. Rows go in one at a
    time -- never as one wave-sized bulk copy -- for the reason the module
    docstring gives under the TSO argument (W7).

    ``row_count`` is written **last among the payload**, but it is still
    payload: the caller rings ``request_seq`` afterwards, and that store is the
    only doorbell. Writing the header after the rows is belt-and-braces
    ordering within a window no reader can observe, not a second fence.

    Args:
        slots: This worker's segment views.
        obs_rows: ``(R, in_planes, 9, 9)`` float32, ``1 <= R <= max_wave``.
        legal_rows: ``(R, A)`` bool, the same ``R``.

    Returns:
        ``R``, the number of rows written -- what the caller must later see
        echoed in ``row_count_echo``.

    Raises:
        ValueError: On zero rows, more rows than ``max_wave``, or a row shape
            that does not match this segment's.
    """
    capacity = slots.max_wave
    if obs_rows.ndim != 4 or obs_rows.shape[1:] != slots.obs.shape[1:]:
        raise ValueError(
            f"obs must be (R, *{slots.obs.shape[1:]!r}), got {obs_rows.shape!r}"
        )
    rows = int(obs_rows.shape[0])
    if legal_rows.shape != (rows, slots.legal.shape[1]):
        raise ValueError(
            f"legal must be {(rows, slots.legal.shape[1])!r}, "
            f"got {legal_rows.shape!r}"
        )
    if rows < 1:
        raise ValueError(
            "a wave must carry at least one row: a zero-row ticket would ring "
            "the doorbell for an answer nobody could write"
        )
    if rows > capacity:
        raise ValueError(
            f"a wave of {rows} rows does not fit a slot sized for "
            f"max_wave={capacity}"
        )
    for row in range(rows):
        slots.obs[row] = obs_rows[row]
        slots.legal[row] = legal_rows[row]
    slots.row_count[0] = rows
    return rows


def pack_request(
    slots: Slots, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> None:
    """Write one position's request into `slots` (worker side).

    The degenerate case of :func:`pack_wave_request`: a wave of exactly one
    row, which is also what ``row_count`` says.
    """
    if obs.shape != slots.obs.shape[1:]:
        raise ValueError(f"obs must be {slots.obs.shape[1:]!r}, got {obs.shape!r}")
    if legal.shape != slots.legal.shape[1:]:
        raise ValueError(
            f"legal must be {slots.legal.shape[1:]!r}, got {legal.shape!r}"
        )
    pack_wave_request(slots, obs[None], legal[None])


def read_wave_request(
    slots: Slots, config: InferenceConfig
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Copy this ticket's whole wave out of `slots` (server side).

    Copies rather than views, row by row, per :func:`read_request`: the batch
    outlives the read, and the worker is free to overwrite its slot the moment
    its response lands.

    Returns:
        ``(obs (R, in_planes, 9, 9), legal (R, A))`` where ``R`` is the ticket's
        ``row_count``.

    Raises:
        InferenceProtocolError: If ``row_count`` is 0 or above
            ``config.max_wave`` (tripwire W5). The offending client is poisoned
            by the caller; the shard keeps serving everybody else, because one
            worker's bad header is not a reason to lose a run.
    """
    rows = _wave_rows(slots, config)
    return (
        np.array(slots.obs[:rows], dtype=np.float32),
        np.array(slots.legal[:rows], dtype=np.bool_),
    )


def _wave_rows(slots: Slots, config: InferenceConfig) -> int:
    """Read and validate a ticket's ``row_count`` (server side, tripwire W5)."""
    rows = int(slots.row_count[0])
    if rows < 1 or rows > config.max_wave:
        raise InferenceProtocolError(
            f"row_count is {rows}, which is not in 1..{config.max_wave}: this "
            f"ticket claims a wave the layout cannot describe, so either the "
            f"two ends were built from different configs or the header was "
            f"never written. The bytes behind it cannot be trusted"
        )
    return rows


def read_request(slots: Slots) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Copy row 0's request out of `slots`, batch dim stripped (server side).

    The single-row reader, kept for the serial reference and for callers that
    know a ticket carries one position. Copies rather than views: the batch
    outlives the read, and the worker is free to overwrite its slot the moment
    its response lands.
    """
    return (
        np.array(slots.obs[0], dtype=np.float32),
        np.array(slots.legal[0], dtype=np.bool_),
    )


class PackedPolicy(NamedTuple):
    """A batch's answer with the policy already packed to its legal support.

    The internal shape of the server's response between
    :meth:`ForwardHandle.result` and :func:`write_sparse_response` under
    ``sparse_policy`` (design review D3). It is deliberately *not* a
    :class:`NetEval`: the packed rows are ragged, and ``NetEval``'s field set is
    part of the public seam. Nothing outside this module's server path sees it
    -- the worker's end of the seam still receives a dense ``NetEval``.

    Attributes:
        values: ``(N, Kmax)`` float32. Row ``r``'s first ``counts[r]`` entries
            are that position's legal logits in **ascending action-id order**;
            the lanes past ``counts[r]`` are padding and are never transported.
            A rectangle rather than a ragged list because that is what one
            device-side ``gather`` produces (and one ``take_along_axis`` on the
            CPU), which is the whole point of packing before the D2H.
        counts: ``(N,)`` int64, the per-row ``K``.
        wdl: ``(N, 3)`` float32 probabilities, exactly as in :class:`NetEval`.
        moves_left: ``(N,)`` float32, exactly as in :class:`NetEval`.
    """

    values: npt.NDArray[np.float32]
    counts: npt.NDArray[np.int64]
    wdl: npt.NDArray[np.float32]
    moves_left: npt.NDArray[np.float32]


def _sparse_indices(
    legal: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per-row legal action ids padded to a rectangle, plus the per-row counts.

    The gather index, computed **on the CPU** from the mask the server already
    holds as numpy (design review D3/5). That is the point: ``masked_select``
    would have to size its output on the device, and ``nonzero`` on ROCm is
    historically slow, whereas here the index is free -- the server was going to
    touch the mask anyway -- and the device does nothing but one gather.

    Rows shorter than ``Kmax`` are padded with action id ``0``. The padding
    lanes are read by the gather and then never transported: only the first
    ``counts[r]`` values of row ``r`` ever leave the server.

    Raises:
        ValueError: If any row has no legal actions (``K == 0``). Unreachable
            in practice -- the search refuses such a position before it ever
            asks for an evaluation -- and refused here anyway, loudly, because a
            zero-length packed row is indistinguishable from a response that was
            never written (design review D7).
    """
    if legal.ndim != 2:
        raise ValueError(f"legal must be (N, A), got {legal.shape!r}")
    counts: npt.NDArray[np.int64] = np.count_nonzero(legal, axis=1).astype(np.int64)
    if not bool((counts > 0).all()):
        empty = [int(row) for row in np.flatnonzero(counts == 0)]
        raise ValueError(
            f"sparse policy refuses K=0: rows {empty} have no legal actions. "
            f"A position with no legal move is never evaluated -- the search "
            f"refuses it first -- so this is a caller bug, not an edge case"
        )
    width = int(counts.max())
    index = np.zeros((legal.shape[0], width), dtype=np.int64)
    for row in range(legal.shape[0]):
        ids = np.flatnonzero(legal[row])
        index[row, : ids.size] = ids
    return index, counts


def pack_sparse_batch(evaluation: NetEval, legal: npt.NDArray[np.bool_]) -> PackedPolicy:
    """Pack a dense batch down to its legal support, on the CPU.

    The reference implementation of the wire format, and the *only* one the gate
    can run: the device-side pack in :meth:`AsyncEvaluator.launch` computes the
    same thing with the same index array, but through ``torch.gather`` on a GPU
    this repository's CI does not have. That asymmetry is why flipping
    ``sparse_policy`` on by default is gated on a one-time on-hardware bitwise
    comparison rather than on this suite alone (design review D3).

    Args:
        evaluation: A dense batch, ``policy_logits`` ``(N, A)``.
        legal: ``(N, A)`` bool, the masks those rows were evaluated with.

    Returns:
        The same batch with the policy packed; see :class:`PackedPolicy`.

    Raises:
        ValueError: On a shape disagreement, or on a row with no legal actions.
    """
    if evaluation.policy_logits.shape != legal.shape:
        raise ValueError(
            f"policy_logits {evaluation.policy_logits.shape!r} and legal "
            f"{legal.shape!r} must have the same shape"
        )
    index, counts = _sparse_indices(legal)
    return PackedPolicy(
        values=np.asarray(
            np.take_along_axis(evaluation.policy_logits, index, axis=1),
            dtype=np.float32,
        ),
        counts=counts,
        wdl=evaluation.wdl,
        moves_left=evaluation.moves_left,
    )


def unpack_sparse_policy(
    packed: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Scatter ``K`` packed logits back into a full ``(A,)`` vector.

    The worker's half of the codec. There is no index array on the wire: the
    positions come from `legal`, which is the array this worker itself wrote
    into its slot and which the server read out of it. Those are the same bytes
    by construction -- single writer, payload before doorbell, one request in
    flight, a fresh segment per incarnation -- so the reconstruction cannot
    disagree with the packing about *which* actions the values belong to.

    Illegal lanes get a finite ``0.0``, never ``-inf`` and never ``NaN``: the
    search validates the whole vector for finiteness before it masks anything,
    and every consumer of an illegal lane either discards it through a mask
    (where an exact ``0.0`` contributes identically to the raw logit's
    discarded lane) or never indexes it at all (design review D1).

    Args:
        packed: ``(K,)`` float32, ascending action-id order.
        legal: ``(A,)`` bool, with exactly ``K`` true entries.

    Returns:
        ``(A,)`` float32.

    Raises:
        ValueError: If `legal` is not one-dimensional, if it has no legal
            actions (``K == 0``, refused for the reason in
            :func:`_sparse_indices`), or if `packed` does not hold exactly
            ``K`` values.
    """
    if legal.ndim != 1:
        raise ValueError(f"legal must be (A,), got {legal.shape!r}")
    if legal.dtype != np.bool_:
        raise ValueError(
            f"legal must be a bool mask, got dtype {legal.dtype}: the scatter "
            f"below is boolean indexing, and an integer mask would "
            f"fancy-index instead -- silently placing values at the wrong "
            f"actions (union-38 review F3)"
        )
    count = int(np.count_nonzero(legal))
    if count == 0:
        raise ValueError(
            "sparse policy refuses K=0: this mask has no legal actions, and a "
            "position with no legal move is never evaluated"
        )
    if packed.shape != (count,):
        raise ValueError(
            f"packed policy must be ({count},) for a mask with {count} legal "
            f"actions, got {packed.shape!r}"
        )
    dense = np.zeros(legal.shape[0], dtype=np.float32)
    dense[legal] = packed
    return dense


def write_response(
    slots: Slots, evaluation: NetEval, row: int, slot_row: int = 0
) -> None:
    """Write batch row `row` into slot row `slot_row` of `slots` (server side).

    The **dense** writer: all ``A`` logits, plus :data:`DENSE_POLICY_COUNT` in
    that row's ``policy_count`` entry so that a worker configured for the sparse
    transport discovers the mismatch instead of reading these logits as a packed
    row. Every **used** row's entry gets the sentinel; entries beyond the
    ticket's ``row_count`` are neither written nor readable by contract (W6).

    ``slot_row`` defaults to 0, which is the single-row protocol exactly as it
    was: a caller that never heard of waves keeps writing the one row its
    ticket described.

    One row per call, deliberately -- see the module docstring's TSO argument
    (W7): a wave-sized bulk copy is a protocol event, not a refactor.

    The caller publishes ``response_seq`` *after* this returns: the bytes must
    be in place before the counter that says they are. That ordering is the
    protocol, not an optimisation -- see the module docstring. Within this
    function the order is immaterial: the count is payload like everything else
    here, and no reader can observe an intermediate state (design review D8).
    """
    slots.policy[slot_row] = evaluation.policy_logits[row]
    slots.wdl[slot_row] = evaluation.wdl[row]
    slots.moves_left[slot_row] = evaluation.moves_left[row]
    slots.policy_count[slot_row] = DENSE_POLICY_COUNT


def write_sparse_response(
    slots: Slots, packed: PackedPolicy, row: int, slot_row: int = 0
) -> None:
    """Write batch row `row` into slot row `slot_row` (server side, sparse).

    That slot row's policy region receives ``K`` float32 values from its own
    offset zero and nothing else; the tail keeps whatever it last held and is
    never read. Same ordering contract, same one-row-per-call rule and same
    ``slot_row`` default as :func:`write_response`.

    Raises:
        ValueError: If the row's ``K`` is zero, or wider than the region.
    """
    count = int(packed.counts[row])
    if count <= 0:
        raise ValueError(
            f"sparse policy refuses K={count} for row {row}: a position with no "
            f"legal move is never evaluated"
        )
    if count > slots.policy.shape[1]:
        raise ValueError(
            f"row {row} packs {count} logits but the policy region holds "
            f"{slots.policy.shape[1]}"
        )
    slots.policy[slot_row, :count] = packed.values[row, :count]
    slots.wdl[slot_row] = packed.wdl[row]
    slots.moves_left[slot_row] = packed.moves_left[row]
    slots.policy_count[slot_row] = count


def read_response(slots: Slots, slot_row: int = 0) -> NetEval:
    """Copy one position's dense response out of `slots`, batch dim stripped.

    Shapes are ``(A,)``, ``(3,)`` and ``()`` -- one position's worth, not a
    batch of one, because the caller asked about one position. ``slot_row``
    selects the row within the wave and defaults to 0, the single-row protocol.

    Raises:
        InferenceProtocolError: If that row's ``policy_count`` is not
            :data:`DENSE_POLICY_COUNT` -- i.e. the server packed a sparse
            response into a slot this end is reading dense. Half of the
            mode-mismatch tripwire; the other half lives in
            :func:`read_sparse_response`. The check is **per row** (W6): a wave
            is answered in one mode, and any row that disagrees is the same
            configuration error.
    """
    count = int(slots.policy_count[slot_row])
    if count != DENSE_POLICY_COUNT:
        raise InferenceProtocolError(
            f"policy_count is {count}, not the dense sentinel "
            f"{DENSE_POLICY_COUNT}: this response carries {count} packed legal "
            f"logits, so the server is running sparse_policy=True while this "
            f"end is reading dense. Both ends must be built from one config"
        )
    return NetEval(
        policy_logits=np.array(slots.policy[slot_row], dtype=np.float32),
        wdl=np.array(slots.wdl[slot_row], dtype=np.float32),
        moves_left=np.array(slots.moves_left[slot_row], dtype=np.float32),
    )


def read_sparse_response(
    slots: Slots, legal: npt.NDArray[np.bool_], slot_row: int = 0
) -> NetEval:
    """Copy and reconstruct one position's sparse response out of `slots`.

    `legal` is the mask the caller submitted, held locally rather than re-read
    from the slot: same bytes either way (the server never writes that region),
    but the local array is cheaper and says out loud where the positions come
    from (design review D6).

    Returns a dense :class:`NetEval` -- shapes ``(A,)``, ``(3,)`` and ``()`` --
    with a finite ``0.0`` in every illegal lane. See
    :func:`unpack_sparse_policy`.

    Raises:
        InferenceProtocolError: If ``policy_count`` disagrees with the number of
            legal actions in `legal`. That is a mode mismatch (a dense server
            leaves the sentinel here) or a codec bug; either way the priors that
            would follow are finite, plausible and wrong, so nothing downstream
            would catch it.
        ValueError: If `legal` is the wrong shape for this slot.
    """
    if legal.shape != slots.legal.shape[1:]:
        raise ValueError(
            f"legal must be {slots.legal.shape[1:]!r}, got {legal.shape!r}"
        )
    count = int(slots.policy_count[slot_row])
    expected = int(np.count_nonzero(legal))
    if count != expected:
        detail = (
            "policy_count holds the dense sentinel, so the server is running "
            "sparse_policy=False while this end is reading sparse. Both ends "
            "must be built from one config"
            if count == DENSE_POLICY_COUNT
            else "the server packed a different number of values than this "
            "worker's mask has legal actions"
        )
        raise InferenceProtocolError(
            f"policy_count is {count} but the mask this worker submitted has "
            f"{expected} legal actions: {detail}"
        )
    packed = np.array(slots.policy[slot_row, :count], dtype=np.float32)
    return NetEval(
        policy_logits=unpack_sparse_policy(packed, legal),
        wdl=np.array(slots.wdl[slot_row], dtype=np.float32),
        moves_left=np.array(slots.moves_left[slot_row], dtype=np.float32),
    )


def read_wave_response(
    slots: Slots,
    legal_rows: npt.NDArray[np.bool_],
    config: InferenceConfig,
) -> list[NetEval]:
    """Decode a whole wave's answer out of `slots` (worker side).

    The width tripwire (W5) lives here, and it is the reason the response
    carries an echo at all: ``row_count_echo`` must equal the number of rows
    this client submitted. A server that answered a different number of rows
    -- a codec bug, a config skew, a ticket that raced a resize -- would
    otherwise hand the search a wave whose rows silently belong to a different
    question, and every one of those values is finite and plausible.

    Args:
        slots: This worker's segment views.
        legal_rows: ``(R, A)`` -- the masks this client submitted, held locally
            rather than re-read from the slot (design review D6). ``R`` is
            therefore what the echo is checked against.
        config: This client's config; ``sparse_policy`` chooses the row decoder.

    Returns:
        One :class:`NetEval` per row, in submission order, each with shapes
        ``(A,)``, ``(3,)`` and ``()``. Always dense, in both transport modes.

    Raises:
        InferenceProtocolError: On a ``row_count_echo`` that is not ``R``, or
            from a row's own ``policy_count`` mode check.
    """
    rows = int(legal_rows.shape[0])
    echo = int(slots.row_count_echo[0])
    if echo != rows:
        raise InferenceProtocolError(
            f"row_count_echo is {echo} but this worker submitted {rows} rows: "
            f"the server answered a different wave than the one this ticket "
            f"described, so the rows below belong to a question nobody asked. "
            f"Every value in them would be finite, plausible and wrong"
        )
    if config.sparse_policy:
        return [
            read_sparse_response(slots, legal_rows[row], row) for row in range(rows)
        ]
    return [read_response(slots, row) for row in range(rows)]


def _refuse_empty_masks(legal_rows: npt.NDArray[np.bool_]) -> None:
    """Refuse a wave in which any row has no legal action (``K == 0``).

    The **client-side** half of the K=0 rule (KOM-038 flip-gate item 3, carried
    into KOM-040 b2 as W3). ``_sparse_indices`` has refused this on the server
    since KOM-038, but only under ``sparse_policy`` -- so with the default dense
    transport an all-false mask travelled happily, was answered with ``A``
    meaningless logits, and the bug surfaced only when the sparse flag was
    flipped. Refusing it at the submitting end makes it loud in the mode
    production actually runs, and it costs one ``count_nonzero`` per wave.

    Raises:
        ValueError: Naming the offending rows, before any doorbell is rung.
    """
    counts = np.count_nonzero(legal_rows, axis=1)
    if not bool((counts > 0).all()):
        empty = [int(row) for row in np.flatnonzero(counts == 0)]
        raise ValueError(
            f"rows {empty} have no legal actions: a position with no legal move "
            f"is never evaluated -- the search refuses it first -- so this is a "
            f"caller bug, not an edge case. Refused before the doorbell, in "
            f"both transport modes"
        )


def _model_device(model: torch.nn.Module) -> torch.device:
    """Where `model`'s parameters live -- the device this module *follows*.

    Read per use rather than cached: nothing here names a device, the caller's
    placement decides, and a caller is entitled to move a model between calls.
    A parameterless model (a test stub) answers CPU, which is where an empty
    tensor is built.
    """
    return next(model.parameters(), torch.empty(0)).device


def _weights_are_half(config: InferenceConfig, device: torch.device) -> bool:
    """Does this (config, device) pair serve from **converted fp16 weights**?

    True only for ``forward_dtype == "f16"`` on CUDA (KOM-039). The device half
    of that test is the CPU fallback: with either fp16 form configured but the
    model on the host, the forward runs f32 exactly as it always has, so
    ``--serial-debug`` and the CPU-only gate stay bit-identical to
    ``forward_dtype="f32"``. It is the same rule at the three sites that need
    it -- the one conversion, and the two input casts that must agree with it
    -- which is why it is a function and not three conditions.
    """
    return config.forward_dtype == "f16" and device.type == "cuda"


def _bn_upcast_input(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Forward-pre-hook: hand a float32 BatchNorm a float32 input."""
    return tuple(item.float() for item in inputs)


def _bn_downcast_output(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """Forward-hook: return the float32 BatchNorm's answer to the half net."""
    return output.half()


def _half_with_f32_bn(model: torch.nn.Module) -> torch.nn.Module:
    """Convert `model` to fp16 **except** its BatchNorms, in place (KOM-039).

    The primary serving form. Everything the trunk and heads spend their time
    in -- convolutions and linears -- runs in half; every
    :class:`torch.nn.BatchNorm2d` keeps float32 parameters and buffers and is
    fed float32, because eval-mode BN is the one numerically fragile op in this
    network: it divides by ``sqrt(running_var + eps)``, and a running variance
    that underflows to zero in half turns a normalisation into a division by
    ``eps`` -- finite, plausible, and wrong in every evaluation for the rest of
    the run.

    The upcast/downcast is a pair of **hooks** rather than a replaced
    ``forward``: assigning over a bound method is what a type checker refuses
    and a reader has to reverse-engineer, while a pre-hook that returns
    replacement inputs and a hook that returns a replacement output are torch's
    own documented way to say exactly this, and they leave the module's own
    ``forward`` -- and its ``repr``, and its scripting -- untouched.

    The BatchNorm state is snapshotted in float32 before ``model.half()`` and
    copied back after, so the norms keep their exact trained values -- not a
    float16 round-trip of them (impl review F3: the round-trip flushes a tiny
    running variance to zero and quantizes everything else by ~5e-4).

    Applied **once**, at evaluator construction, and only on CUDA: see
    :func:`_weights_are_half`. Converting per forward would pay the conversion
    on every batch, and converting on the host would break the CPU fallback's
    bit-identity.

    Args:
        model: The network, already placed by the caller. Mutated in place.

    Returns:
        The same object, for the convenience of a caller that wants to say what
        it did on one line.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # Snapshot every BatchNorm's float32 state BEFORE the conversion:
    # ``model.half()`` quantizes the norms' parameters and buffers through
    # float16, and ``module.float()`` would restore only their *dtype* -- a
    # running variance of 1e-9 comes back as exactly 0.0, the very failure
    # this helper exists to prevent (impl review 2026-08-30, F3).
    saved = [
        {
            name: tensor.detach().clone()
            for name, tensor in [
                *module.named_parameters(recurse=False),
                *module.named_buffers(recurse=False),
            ]
        }
        for module in norms
    ]
    model.half()
    with torch.no_grad():
        for module, state in zip(norms, saved, strict=True):
            module.float()
            for name, tensor in [
                *module.named_parameters(recurse=False),
                *module.named_buffers(recurse=False),
            ]:
                tensor.copy_(state[name])
            module.register_forward_pre_hook(_bn_upcast_input)
            module.register_forward_hook(_bn_downcast_output)
    return model


class _ForwardContext(Protocol):
    """The two methods a ``with`` block needs from an autocast region.

    Structural for the reason :class:`_CudaEventLike` is: ``torch.autocast``
    is annotated where it is constructed but not where it is *entered*, so
    under ``--strict`` a direct ``with torch.autocast(...)`` is a
    ``no-untyped-call`` error. Naming the context by the protocol types the
    call site without an ignore whose correctness depends on which torch
    release is installed, and it lets :func:`contextlib.nullcontext` stand in
    for "no autocast here" as the same static type.
    """

    def __enter__(self) -> object:
        """Enter the region."""
        ...

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object, /
    ) -> object:
        """Leave it, restoring whatever was saved."""
        ...


_new_autocast: Callable[..., _ForwardContext] = torch.autocast
"""``torch.autocast``, seen through :class:`_ForwardContext`.

An attribute lookup, not a CUDA call: importing this module on a host with no
GPU is unaffected, and the region is only ever *constructed* on the CUDA path.
"""


def _forward_autocast(config: InferenceConfig, device: torch.device) -> _ForwardContext:
    """The autocast region the model forward runs in, for this config.

    A real ``torch.autocast("cuda", dtype=torch.float16)`` for
    ``forward_dtype == "f16_autocast"`` on CUDA, and a
    :func:`contextlib.nullcontext` -- autocast disabled, which is what
    ``enabled=False`` means -- for every other combination. Constructing
    nothing off the CUDA path is deliberate: it keeps the CPU fallback from
    touching torch's autocast state at all, which is what makes the
    bit-identity of the f32 and fp16 CPU forwards a property rather than a
    hope.

    The region wraps **the model forward and nothing else** (KOM-039 design
    item 3). The softmax, the casts on the way out and the KOM-038 sparse
    gather all stay outside it, so the seam's float32 contract is decided by
    this module rather than by autocast's op table.
    """
    if config.forward_dtype == "f16_autocast" and device.type == "cuda":
        return _new_autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _refuse_nonfinite_wdl(wdl: torch.Tensor, forward_dtype: str) -> None:
    """Refuse a non-finite wdl, in **every** forward dtype.

    The KOM-039 tripwire, and the reason it is worth its cost: an fp16 forward
    that overflows produces ``inf`` logits, a softmax over them produces
    ``NaN``, and a ``NaN`` prior is not a crash -- it is a search that quietly
    stops preferring anything, in a run whose only symptom is a strength curve
    that goes flat. This turns that into a loud failure at the batch that
    caused it.

    **It runs on the ``"f32"`` path too (review TS-01, 2026-09-12), which
    reverses KOM-039's decision that it should not.** The original argument was
    that f32 does not overflow, so the precision that needs no tripwire should
    not pay for one. That is an argument about *overflow* and this is a check
    for *non-finiteness*, which reaches f32 by other roads entirely: a weight
    file that was already NaN when it was loaded, a net poisoned by an earlier
    training step, a corrupted transfer. In every one of those a NaN prior is
    the same silent flat run it is in fp16, and the cost of noticing is one
    ``isfinite`` over ``(N, 3)`` values -- nothing beside the forward that
    produced them, and nothing at all beside the days of self-play the check
    saves. `forward_dtype` is now diagnosis rather than control: it names the
    form in the message so a fp16 overflow still reads as one.

    Args:
        wdl: The post-softmax ``(N, 3)`` tensor, on whichever device it was
            computed. Reading it is a synchronisation on CUDA, so the CUDA
            callers do this at their sync point, never in a launch.
        forward_dtype: The config's knob, named in the message.

    Raises:
        RuntimeError: If any element is non-finite.
    """
    if not bool(torch.isfinite(wdl).all()):
        raise RuntimeError(
            f"forward_dtype={forward_dtype!r} produced a non-finite wdl after "
            f"the softmax: the priors behind it would be NaN rather than "
            f"wrong-but-finite, which is a search that stops preferring "
            f"anything without ever failing (KOM-039; extended to the f32 path "
            f"by review TS-01, where the cause is a poisoned net rather than an "
            f"overflow). Re-run this shard from a checkpoint whose weights are "
            f"finite, or at forward_dtype='f32' if this was a reduced-precision "
            f"forward"
        )


def _refuse_ungraphable(
    config: InferenceConfig, device: torch.device, hip_version: str | None
) -> None:
    """Refuse ``forward_graph="on"`` where capture must not be attempted.

    All three refusals are KOM-048's, they all fire at evaluator construction,
    and none of them has a fallback: a run that asked for a captured forward
    and got an eager one is a *measurement* that lies, and the only thing worse
    than a slow bench is a fast-looking one that was never graphed. So the knob
    never degrades quietly -- it either captures or it raises by name.

    The three, in the order they are checked:

    1. **``forward_dtype="f16_autocast"``** (checked first, because it is the
       one refusal decidable from the config alone -- it does not depend on
       which box this is, which makes it the same refusal everywhere). Autocast
       is a per-call, thread-local, cache-carrying region: capture would bake
       one particular state of its cast cache into every replay. The two knobs
       are individually supported and their combination is forbidden; the
       measurable fp16 form under graphs is ``"f16"``, whose cast is one
       captured device op.
    2. **A ROCm torch build** (design finding D1). ROCm's torch reports
       ``device.type == "cuda"``, so a device check cannot see this at all --
       ``torch.version.hip`` is the only honest test. hipGraph capture on the
       development box's gfx1034 (a W6400) core-dumped every probe, and this
       card's protection is a refusal rather than a measurement: there is
       nothing to measure on hardware that dies during capture.
    3. **A host-resident model.** There is no such thing as a CPU graph here,
       and a bench that thinks graphs are on must not silently time the eager
       CPU forward.

    Args:
        config: The evaluator's config. Only ``forward_dtype`` is read; the
            caller has already decided that ``forward_graph == "on"``.
        device: Where the model's parameters live (:func:`_model_device`).
        hip_version: ``torch.version.hip`` -- ``None`` on a CUDA build, the
            HIP version string on a ROCm one. Passed in rather than read here
            so that both branches are reachable in a test on either build.

    Raises:
        RuntimeError: On any of the three, naming KOM-048 and the reason.
    """
    if config.forward_dtype == "f16_autocast":
        raise RuntimeError(
            "forward_graph='on' with forward_dtype='f16_autocast' is refused "
            "(KOM-048): an autocast region is capture-hostile -- it is a "
            "per-call thread-local state with its own cast cache, and a "
            "capture would freeze one instant of that cache into every "
            "replay. Use forward_dtype='f16', whose conversion is a single "
            "captured device op, or forward_graph='off'"
        )
    if hip_version is not None:
        raise RuntimeError(
            f"forward_graph='on' is refused on a ROCm torch build "
            f"(torch.version.hip={hip_version!r}) -- KOM-048 finding D1. ROCm "
            f"reports device.type == 'cuda', so a device check cannot tell "
            f"this apart from CUDA; hipGraph capture core-dumped every gfx1034 "
            f"probe, and refusing is this card's protection rather than a "
            f"measurement. Run graphs on a CUDA device, or forward_graph='off'"
        )
    if device.type != "cuda":
        raise RuntimeError(
            f"forward_graph='on' needs a CUDA-resident model, but this model's "
            f"parameters are on {device!r} (KOM-048). There is no captured "
            f"forward on the host, and falling back to the eager one would "
            f"make a bench that believes it is graphing measure something "
            f"else. Place the model on CUDA, or set forward_graph='off' -- "
            f"which is what --serial-debug does for you"
        )


class _NetOutputLike(Protocol):
    """The subset of :class:`komugi.net.NetOutput` the seam consumes.

    Structural on purpose: a test stub is a legal model here, and
    ``soft_policy_logits`` is deliberately absent -- the soft-policy head is a
    training-time auxiliary and does not cross the inference seam.
    """

    @property
    def policy_logits(self) -> torch.Tensor: ...

    @property
    def wdl_logits(self) -> torch.Tensor: ...

    @property
    def moves_left(self) -> torch.Tensor: ...


class InProcessEvaluator:
    """A :class:`BatchEvaluator` that calls a torch module directly.

    Both the server's internal forward and the entire evaluator in
    ``--serial-debug`` runs. Torch stops here: numpy comes in, one
    :func:`torch.no_grad` forward happens, numpy goes out (design D1).

    The model arrives as an object, never as pickled bytes crossing a process
    boundary (design D2) -- in the server it is built inside the server process
    by ``model_factory``.

    Nothing here names a device: tensors are built on the model's own parameter
    device, so placing the model is the caller's decision and this code follows
    it.

    Args:
        model: The network. Put into eval mode here rather than trusted to be
            in it -- a forward in train mode would silently corrupt every
            evaluation in the run via batchnorm's running statistics. Under
            ``forward_dtype="f16"`` on CUDA it is also **converted here, once**
            (:func:`_half_with_f32_bn`): the caller's object is mutated, which
            is what makes the conversion a construction-time cost rather than a
            per-batch one, and what makes :class:`AsyncEvaluator` -- which
            builds one of these -- serve from the same half weights.
        config: Shapes the evaluator validates against, and the precision it
            forwards in (``forward_dtype``). ``forward_graph`` is **not read
            here and never will be** (KOM-048): CUDA graph capture has exactly
            one site, :class:`AsyncEvaluator`, and this class is the serial
            reference -- including the instance ``AsyncEvaluator`` itself holds
            as its CPU fallback, which is why a second capture site here would
            mean two captures of one model. Callers that must not graph say so
            on the config anyway (:func:`~komugi.selfplay.orchestrate.serial_debug`
            forces it off), so the value is checkable rather than merely inert.
    """

    def __init__(self, model: torch.nn.Module, config: InferenceConfig) -> None:
        self._model = model
        self._config = config
        self._model.eval()
        if _weights_are_half(config, _model_device(model)):
            _half_with_f32_bn(self._model)

    @property
    def config(self) -> InferenceConfig:
        """The config this evaluator validates against."""
        return self._config

    def evaluate(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> NetEval:
        """Evaluate a batch. See :class:`BatchEvaluator`.

        ``legal`` is validated and otherwise unused in P0a: the mask crosses the
        seam so that the sparse-return knob of design D8 is a configuration
        change later, and masking the policy stays the search's job.

        Raises:
            RuntimeError: If a reduced-precision forward produced a non-finite
                wdl (:func:`_refuse_nonfinite_wdl`), in **every**
                ``forward_dtype`` including ``"f32"`` (review TS-01).
        """
        _validate_batch(obs, legal, self._config)

        device = _model_device(self._model)
        batch = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(device)
        with torch.no_grad():
            if _weights_are_half(self._config, device):
                batch = batch.half()
            with _forward_autocast(self._config, device):
                out: _NetOutputLike = self._model(batch)
            # Probabilities, not logits, cross the seam: every reader must agree
            # on what "draw probability" means before draw_score is applied
            # (design D1). The .float() is the load-bearing cast of KOM-039: an
            # fp16 softmax saturates, and it is exactly this tensor the tripwire
            # below then checks.
            wdl = torch.softmax(out.wdl_logits.float(), dim=-1)
            # The seam's numpy contract is float32 in every forward_dtype; on
            # the f32 path both of these are the tensor they were given.
            policy = out.policy_logits.float()
            moves_left = out.moves_left.float()
        _refuse_nonfinite_wdl(wdl, self._config.forward_dtype)
        return NetEval(
            policy_logits=_to_numpy(policy),
            wdl=_to_numpy(wdl),
            moves_left=_to_numpy(moves_left),
        )


def _to_numpy(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """Detach `tensor` to a contiguous float32 numpy array on the host.

    ``.cpu()`` is a transfer to the host, not a device choice: numpy lives
    there and nowhere else.
    """
    # The .copy() is load-bearing: for a contiguous float32 CPU tensor,
    # ascontiguousarray over .numpy() is a zero-copy VIEW into torch's buffer;
    # a model reusing an internal output buffer would silently corrupt any
    # retained NetEval (wave-9a review F9).
    return np.ascontiguousarray(
        tensor.detach().cpu().numpy(), dtype=np.float32
    ).copy()


def _validate_batch(
    obs: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
    config: InferenceConfig,
) -> None:
    """Reject a mis-shaped batch here, by name, rather than deep inside torch.

    Shared by :class:`InProcessEvaluator` and :class:`AsyncEvaluator` so that a
    forward launched asynchronously refuses exactly what the serial one does --
    the two paths must be indistinguishable to a caller (I6).
    """
    expected = (config.in_planes, BOARD_SIZE, BOARD_SIZE)
    if obs.ndim != 4 or obs.shape[1:] != expected:
        raise ValueError(f"obs must be (N, *{expected!r}), got {obs.shape!r}")
    if legal.shape != (obs.shape[0], config.num_actions):
        raise ValueError(
            f"legal must be {(obs.shape[0], config.num_actions)!r}, "
            f"got {legal.shape!r}"
        )


class ForwardHandle(Protocol):
    """A forward that has been *started* and may not have finished yet.

    The seam the pipelined server (KOM-032) is built on: the server launches
    batch N, collects batch N+1 while the GPU is still working on N, and only
    then blocks for N's rows. The split between the two methods is the whole
    point -- :meth:`query` is called from the collect loop and must never
    block; :meth:`result` is the sync point and must.

    A handle is single-use and belongs to exactly one batch. What it retains
    until :meth:`result` returns is part of the contract, not an implementation
    detail: the pinned staging tensor whose bytes are still being DMA'd, and
    the device-side outputs the sync will read (design review D6/I5).
    """

    def query(self) -> bool:
        """Has the forward finished? Never blocks."""
        ...

    def result(self) -> NetEval:
        """Block until the forward has finished; return its float32 rows."""
        ...


@runtime_checkable
class _PacksPolicy(Protocol):
    """A :class:`ForwardHandle` that can also return its rows already packed.

    Deliberately a *second*, optional protocol rather than a method on
    :class:`ForwardHandle`: the pipelined suite drives the server with
    hand-completed handles, and widening the handle contract would make every
    one of those fakes structurally invalid for a mode they do not exercise.
    :func:`_packed_result` asks with ``isinstance`` and falls back to packing
    the dense answer on the CPU, which is the right answer for a fake and the
    slow answer for nobody -- the two production handles both implement this.
    """

    def packed_result(self) -> PackedPolicy:
        """Block until the forward finished; return its rows, policy packed."""
        ...


class ForwardLauncher(Protocol):
    """Anything that starts a batch forward without waiting for it.

    :class:`AsyncEvaluator` is the production implementation.
    :func:`serve_pipelined` is written against this protocol rather than
    against the class so that the unit suite can drive every interleaving with
    a hand-completed :class:`ForwardHandle` and no GPU -- which is what makes
    the in-flight exclusion (I1) and the two-window drain (I4) deterministic
    tests rather than hardware anecdotes.
    """

    def launch(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> ForwardHandle:
        """Start evaluating ``obs`` ``(N, in_planes, 9, 9)``, mask ``legal``."""
        ...


class _ReadyHandle:
    """A :class:`ForwardHandle` that was already finished when it was created.

    The CPU path's handle: there is no stream to overlap against, so
    :meth:`AsyncEvaluator.launch` runs the forward synchronously and hands back
    its answer wrapped in this. The pipelined loop then degenerates, exactly,
    to the serial one -- which is what invariant I6 (bit-identical CPU output)
    rests on: the bytes come from the same :class:`InProcessEvaluator` call the
    serial path makes.
    """

    __slots__ = ("_evaluation", "_legal")

    def __init__(self, evaluation: NetEval, legal: npt.NDArray[np.bool_]) -> None:
        self._evaluation = evaluation
        self._legal = legal

    def query(self) -> bool:
        """Always ``True``: the forward finished before the handle existed."""
        return True

    def result(self) -> NetEval:
        """Never blocks."""
        return self._evaluation

    def packed_result(self) -> PackedPolicy:
        """The same rows, policy packed -- on the CPU, after the fact.

        There is no D2H to get in front of here, so "pack before the transfer"
        degenerates to "pack before the shm writes", which is the CPU path's
        entire share of the win. It also makes the sparse wire format something
        the gate can run end to end (design review D3).
        """
        return pack_sparse_batch(self._evaluation, self._legal)


class _CudaEventLike(Protocol):
    """The two methods the pipeline needs from a CUDA event.

    Structural rather than a bare ``torch.cuda.Event`` annotation because
    torch's CUDA event surface is a thin Python wrapper over a C extension
    type, and that wrapper is not annotated: under ``--strict`` every call into
    it is a ``no-untyped-call`` error, exactly as ``Tensor.backward`` is in
    ``train/step.py``. A protocol types the call sites without an ignore whose
    correctness would depend on which torch release is installed, and it says
    out loud how little of the event API this module actually uses.
    """

    def record(self) -> None:
        """Enqueue the event on the current stream."""
        ...

    def query(self) -> bool:
        """Have all preceding kernels retired? Never blocks."""
        ...


_new_cuda_event: Callable[[], _CudaEventLike] = torch.cuda.Event
"""``torch.cuda.Event``, seen through :class:`_CudaEventLike`.

An attribute lookup, not a CUDA call: importing this module on a host with no
GPU is unaffected, and the event is only ever *constructed* on the CUDA path.
"""


_new_cuda_graph: Callable[..., Any] = torch.cuda.CUDAGraph
"""``torch.cuda.CUDAGraph``, behind an alias (KOM-048)."""

_cuda_graph_capture: Callable[..., Any] = torch.cuda.graph
"""``torch.cuda.graph`` -- the capture region -- behind an alias."""

_new_cuda_stream: Callable[..., Any] = torch.cuda.Stream
"""``torch.cuda.Stream``, behind an alias."""

_cuda_stream_scope: Callable[..., Any] = torch.cuda.stream
"""``torch.cuda.stream`` -- the "run this block on that stream" region."""

_current_cuda_stream: Callable[..., Any] = torch.cuda.current_stream
"""``torch.cuda.current_stream``, behind an alias.

These five are attribute lookups, not CUDA calls: importing this module on a
host with no GPU is unaffected, and every one of them is only ever *called* on
the CUDA path, from :class:`_GraphRunner`'s constructor.

They are typed ``Any`` rather than through a protocol, which is the one place
this module departs from the :class:`_CudaEventLike` idiom next door, and the
reason is worth stating: the event protocol works because ``record``/``query``
are two annotated methods on a wrapper. The capture API is not that -- it is a
mix of thin wrappers over C types (``CUDAGraph``) and unannotated context
managers (``graph``, ``stream``) whose signatures move between torch releases,
so a structural protocol here would pin *torch's* annotations rather than this
module's usage, and would break on an upgrade that changed nothing we call.
The surface actually used is five names and four calls, all in one constructor.
"""

_GRAPH_WARMUP_ITERS = 3
"""Forwards run before capture, on a side stream (KOM-048's capture recipe).

Not a superstition and not a tuning knob: cuDNN/MIOpen pick their algorithms
on first call, the caching allocator has to reach steady state, and lazy module
initialisation has to happen -- all of which must occur *before* the capture,
because a capture records the kernels that ran, and a first-call heuristic
baked into a graph is a heuristic chosen for a batch shape nobody will ever
send again. Three is torch's own documented recipe.

The side stream matters as much as the count. Warming up on the **legacy
default stream** is what torch's capture documentation forbids: the default
stream carries implicit synchronisation with every other stream in the
process, which is exactly the thing a capture cannot record.
"""


class _GraphOutputs(NamedTuple):
    """The three heads a replay hands the eager tail, already sliced to ``N``.

    Structurally a :class:`_NetOutputLike`, so the launch's tail cannot tell a
    replayed forward from an eager one -- which is the entire point: the
    softmax, the KOM-038 gather and the outbound casts are the same code on
    both paths, and neither of them was captured.

    These are **views into the graph's static output buffers**, not copies. They
    are valid exactly until the next replay, which is what the handshake in
    :class:`_ReplayToken` exists to enforce.
    """

    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class _ReplayToken:
    """The graphs-mode exclusivity handshake: one unread replay at a time.

    A captured graph writes into **one** set of static output buffers. Replay
    N+1 overwrites replay N's answer the moment it is *enqueued* -- not when it
    finishes -- so the pipeline's central promise ("launch batch N+1 while N is
    still in flight") is precisely what graphs mode cannot have. The rule is
    therefore: a launch may happen only after the previous launch's outputs
    have been read off the device.

    A CUDA event cannot express that, and it is worth saying why, because the
    event is the obvious tool and it is the wrong one: ``event.query()``
    answers "have N's kernels retired?", which is true long before anybody has
    *read* N's outputs. The dangerous case -- a caller that holds a handle and
    launches again without ever calling ``result()`` -- is exactly the case an
    event reports as safe. So the handshake is bookkeeping, not hardware: the
    launch marks the token outstanding and :meth:`_CudaHandle.result` (or
    :meth:`~_CudaHandle.packed_result`) clears it after its synchronous D2H.

    :func:`serve_pipelined` already satisfies this, and it is worth saying so
    rather than leaving it to be rediscovered: its loop is launch N -> collect
    N+1 -> ``_publish`` N (which calls ``result()``) -> launch N+1, so exactly
    one handle is ever unread. What it overlaps is the *collect* with the
    forward, not two forwards -- which is invariant I1, from long before this
    ticket. Graphs mode therefore costs that loop nothing today; a future
    launcher that tried to hold two handles would meet this refusal, which is
    the intended outcome.

    Deliberately a tiny class with no CUDA in it, so the protocol is testable
    on a machine with no GPU (which is the machine the gate runs on).
    """

    __slots__ = ("_outstanding",)

    def __init__(self) -> None:
        self._outstanding = False

    @property
    def outstanding(self) -> bool:
        """Is a replay's answer still sitting unread in the static buffers?"""
        return self._outstanding

    def mark(self) -> None:
        """Claim the static buffers for a new replay.

        Raises:
            RuntimeError: If the previous replay's answer was never consumed.
                Loud rather than silent, because the silent version is a batch
                of evaluations that belong to a different set of positions --
                finite, plausible, and wrong in every one of them. Note that a
                prior exception AT the sync point (the KOM-039 tripwire
                raising inside ``result()``/``packed_result()``) also leaves
                the token outstanding, so a caller that catches and relaunches
                sees this refusal, not the original failure -- read the first
                error (impl review G6).
        """
        if self._outstanding:
            raise RuntimeError(
                "forward_graph='on': a replay was launched while the previous "
                "launch's answer was still unread (KOM-048). A captured graph "
                "has one set of static output buffers, so enqueueing this "
                "replay would overwrite rows nobody has collected -- the "
                "pipelined overlap of KOM-032 is the one thing graphs mode "
                "cannot have. Call result()/packed_result() on the outstanding "
                "handle before launching again, or set forward_graph='off'"
            )
        self._outstanding = True

    def consume(self) -> None:
        """Release the buffers: the D2H that read them has completed.

        Idempotent, because a handle being read twice is a caller oddity and
        not a protocol violation -- the second read returns the same bytes the
        first one copied to the host.
        """
        self._outstanding = False


class _GraphRunner:
    """The model forward, captured once at a fixed shape and replayed per batch.

    Owned by :class:`AsyncEvaluator`'s CUDA path and by nothing else (KOM-048).
    :class:`InProcessEvaluator` never captures -- including the one
    :class:`AsyncEvaluator` holds as its serial fallback -- because two runners
    over one model would capture the same forward twice, into two private
    memory pools, for a path only one of them is ever asked to serve.

    What is captured is **the model forward and nothing else**: the same
    boundary KOM-039 drew around its autocast region. The wdl softmax, the
    KOM-038 sparse gather and the outbound float casts stay eager, so the
    seam's float32 contract keeps being decided by this module rather than by
    what happened to be recorded.

    The capture recipe, in the order the constructor runs it:

    * **The static input is allocated outside ``inference_mode``.** Every
      launch mutates it in place (``copy_`` the rows, ``zero_`` the pad), and
      an in-place update to an *inference* tensor from outside inference mode
      is refused by torch. Allocating it here, in ordinary mode, is what makes
      the per-launch mutation legal; it costs one buffer of
      ``max_batch`` rows for the process's life.
    * **The static input stays float32 in every ``forward_dtype``.** Under
      ``"f16"`` the captured region is ``model(static_input.half())``, so the
      conversion is one device op *inside* the graph. The alternative -- a
      half static input -- would make every launch's H2D a dtype-converting
      copy, which is a synchronous staging step, which is the pipelining this
      module exists to keep.
    * **Eval mode is verified, not assumed.** A train-mode capture would bake
      BatchNorm's cross-row batch statistics into every replay: rows would
      silently influence each other's evaluations forever, and nothing
      downstream would notice.
    * **Warmup runs on a side stream**, three times, at the full capture shape.
      See :data:`_GRAPH_WARMUP_ITERS`.
    * **All four heads become static outputs.** The net has four
      (``policy_logits``, ``soft_policy_logits``, ``wdl_logits``,
      ``moves_left``) and serving reads three: the soft-policy head is a
      training-time auxiliary that does not cross the seam. Its output buffer
      is captured dead weight -- ``max_batch x num_actions`` float32, ~19 MB at
      the P0a action space and a 16-row batch -- and it is accepted rather than
      engineered away, because the alternative is a serving-only forward path
      that could drift from the trained one. The whole output object is
      retained (not just the three tensors) so that the fourth buffer's
      lifetime is not left to the allocator.

    **The model is pinned once this exists.** A capture records parameter
    *addresses*, so:

    * moving the model to another device invalidates every replay --
      :meth:`require_device` checks the parameter device on every launch and
      refuses, *before* the launch's own host fallback could swallow it;
    * updating weights is legal **only** in place, which is what
      ``load_state_dict`` does by default. ``assign=True``, rebinding
      ``param.data``, or replacing a submodule all leave the graph reading the
      memory it captured, and no replay would fail -- it would just keep
      serving the old network.
    """

    __slots__ = (
        "_model",
        "_device",
        "_half",
        "_token",
        "_static_in",
        "_graph",
        "_out",
    )

    def __init__(
        self, model: torch.nn.Module, config: InferenceConfig, device: torch.device
    ) -> None:
        """Capture `model`'s forward at ``(max_batch, in_planes, 9, 9)``.

        The caller has already run :func:`_refuse_ungraphable`, so this is only
        ever reached on a CUDA-resident model on a non-ROCm build.

        Raises:
            RuntimeError: If `model` is in train mode.
        """
        self._model = model
        self._device = device
        # Read once, here: the weights were converted (or not) before this
        # runner existed, and under graphs the model can no longer move, so
        # there is nothing left for a per-launch re-read to follow.
        self._half = _weights_are_half(config, device)
        self._token = _ReplayToken()
        # OUTSIDE inference_mode, deliberately -- see the class docstring.
        # zeros, not empty: warmup then forwards defined values instead of
        # allocator garbage (harmless either way, but under f16 garbage can
        # push inf through warmup activations -- impl review G9).
        self._static_in = torch.zeros(
            (config.max_batch, config.in_planes, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float32,
            device=device,
        )
        if model.training:
            # The design says "assert"; this raises instead, because an
            # assertion is exactly what `python -O` removes and this is the one
            # capture-time mistake whose consequence is silent for the rest of
            # the run.
            raise RuntimeError(
                "forward_graph='on' refuses to capture a model in train mode "
                "(KOM-048): the capture would record BatchNorm's *batch* "
                "statistics path, so every replay would mix rows into each "
                "other's evaluations -- and every value it produced would "
                "still be finite and plausible. Call model.eval() first"
            )
        side = _new_cuda_stream()
        side.wait_stream(_current_cuda_stream())
        with _cuda_stream_scope(side):
            for _ in range(_GRAPH_WARMUP_ITERS):
                with torch.inference_mode():
                    self._model(self._capture_input())
        _current_cuda_stream().wait_stream(side)
        graph = _new_cuda_graph()
        with torch.inference_mode(), _cuda_graph_capture(graph):
            out: _NetOutputLike = self._model(self._capture_input())
        self._graph = graph
        self._out = out

    @property
    def token(self) -> _ReplayToken:
        """The exclusivity handshake this runner's launches mark and hand on."""
        return self._token

    @property
    def max_batch(self) -> int:
        """Rows the captured shape holds -- the hard ceiling on a launch.

        The server cannot exceed it: :func:`_row_ceiling` bounds every batch by
        ``min(max_batch, max_wave * len(clients))``, and this is captured at
        ``max_batch``. The refusal in :meth:`run` is therefore for the direct
        callers of this seam -- benches, tests, a future launcher -- not for
        :func:`serve_pipelined`.
        """
        return int(self._static_in.shape[0])

    def _capture_input(self) -> torch.Tensor:
        """What the captured region consumes, warmup and capture alike.

        The ``.half()`` is *inside* what gets captured under
        ``forward_dtype="f16"``, which is the whole reason the static input is
        float32; the warmup takes the identical expression so that what is
        warmed and what is captured cannot drift.
        """
        if self._half:
            return self._static_in.half()
        return self._static_in

    def require_device(self, device: torch.device) -> None:
        """Refuse a model that has moved since its forward was captured.

        Called by :meth:`AsyncEvaluator.launch` **before** its
        ``device.type != "cuda"`` fallback, which is the whole point of it
        being separate from :meth:`run`: a model moved back to the host would
        otherwise take that fallback and be served, correctly but eagerly, by
        the serial evaluator -- a graphs-on run silently ceasing to be one,
        which is the exact failure :func:`_refuse_ungraphable` exists to
        prevent at construction.

        Raises:
            RuntimeError: If `device` is not the captured device.
        """
        if device != self._device:
            raise RuntimeError(
                f"forward_graph='on': the model moved from {self._device!r} to "
                f"{device!r} after its forward was captured (KOM-048). A "
                f"capture records parameter addresses, so a replay after the "
                f"move would read the old device's memory -- and a fallback to "
                f"the eager path would leave a graphs-on run quietly not "
                f"graphing. The model is pinned for the life of the evaluator"
            )

    def run(self, staged: torch.Tensor, rows: int) -> None:
        """Fill the static input from `staged` and replay, on the current stream.

        Three device operations and no synchronisation: the H2D copy of `rows`
        rows, a ``zero_`` over the pad, and the replay. All of them are
        enqueued and none of them is waited on, exactly as the eager launch's
        forward is -- the caller records its event afterwards and reads the
        outputs at its own sync point.

        **The pad is zeroed every launch**, and that is not defensive
        programming: the static input keeps whatever the *previous*, possibly
        wider, batch left in it. The network is row-independent in eval mode
        (:func:`_refuse_ungraphable` guarantees eval mode at capture), so stale
        pad rows could not corrupt a real row's answer, and the pad's own
        outputs are sliced away by :meth:`outputs` before anything -- the
        KOM-039 tripwire included -- reads them. The zero is determinism
        defense-in-depth (design rev 2 sect. H): pad outputs stay a fixed
        function of nothing, so a full-buffer debug scan never sees
        batch-to-batch ghosts (impl review G4 pinned this white-box).

        The caller has already run :meth:`require_device`. Both refusals below
        happen **before** the token is marked, so a launch this method rejects
        leaves the handshake exactly as it found it -- the outstanding handle,
        if there is one, is still the one to read.

        Args:
            staged: The pinned host batch, ``(rows, in_planes, 9, 9)`` float32.
            rows: ``N``. The capture is fixed-shape; short batches ride in the
                first ``N`` rows and the rest is pad.

        Raises:
            ValueError: If `rows` exceeds the captured shape.
            RuntimeError: If the previous launch's answer was never consumed
                (:meth:`_ReplayToken.mark`).
        """
        if rows > self.max_batch:
            raise ValueError(
                f"forward_graph='on' was given a batch of {rows} rows, but the "
                f"graph was captured at a fixed shape of {self.max_batch} rows "
                f"(config.max_batch), and a capture cannot be re-shaped "
                f"(KOM-048). Raise max_batch and rebuild the evaluator, or set "
                f"forward_graph='off'"
            )
        self._token.mark()
        self._static_in[:rows].copy_(staged, non_blocking=True)
        self._static_in[rows:].zero_()
        # The replay runs INSIDE inference_mode: the captured region ran under
        # it, so the graph output buffers are inference tensors, and the
        # replay in-place update of them is refused outside the mode
        # ("Inplace update to inference tensor outside InferenceMode") -- the
        # first real CUDA replay on the 2026-08-30 pod died on exactly this.
        # The copy_/zero_ above stay OUTSIDE: the static input is an ordinary
        # tensor (allocated outside the mode, see __init__) and needs nothing.
        with torch.inference_mode():
            self._graph.replay()

    def outputs(self, rows: int) -> _NetOutputLike:
        """The replayed heads, sliced to the ``rows`` the caller asked about.

        Views into the static output buffers, valid until the next
        :meth:`run` -- which the handshake makes unreachable before this
        launch's handle has been read. The mirror of the sentence on
        :meth:`_CudaHandle.result`: that method's synchronous D2H is what turns
        these views into a batch's own answer, and it is load-bearing here
        rather than merely convenient.
        """
        return _GraphOutputs(
            policy_logits=self._out.policy_logits[:rows],
            wdl_logits=self._out.wdl_logits[:rows],
            moves_left=self._out.moves_left[:rows],
        )


class _CudaHandle:
    """An in-flight CUDA forward, plus everything that must outlive it.

    Every field is retained deliberately (design review D6, invariant I5):

    * ``staged`` is the **pinned host** tensor that is the *source* of an
      in-flight async H2D copy. Torch's caching host allocator only
      event-tracks a pinned block for as long as something references it;
      dropping this reference while the DMA is running would let the next
      ``pin_memory()`` hand the same page to a later batch, and the copy would
      read whatever that batch wrote.
    * the three output tensors are device-side, so the caching allocator keeps
      them stream-ordered -- but they are also what :meth:`result` reads, so
      they are held for the ordinary reason as well.

    ``event`` is recorded after the forward is enqueued, so ``event.query()``
    answers "have all of this batch's kernels retired?" without a sync.

    Under ``sparse_policy`` two more fields join them and the meaning of
    ``policy`` changes (KOM-038):

    * ``policy`` is the **packed** ``(N, Kmax)`` tensor the launch's
      ``torch.gather`` produced, not the dense ``(N, A)`` head output. That is
      the entire point -- the D2H below moves kilobytes instead of megabytes --
      and it is why :meth:`result` refuses to run in this mode: a dense
      :class:`NetEval` cannot be built from these bytes without undoing the
      saving on the server's own critical path.
    * ``index`` is the **pinned host** index tensor, retained for exactly the
      reason ``staged`` is (I5): it is the source of an in-flight async H2D
      copy, and torch's caching host allocator only event-tracks a pinned block
      while something references it.
    * ``counts`` is the per-row ``K``, computed on the CPU at launch, which is
      what makes the packed rectangle interpretable without a second transfer.

    ``forward_dtype`` is carried for one reason: the KOM-039 tripwire reads the
    wdl tensor, and reading a device tensor is a synchronisation. It therefore
    fires at the **sync point** -- :meth:`result` and :meth:`packed_result`,
    which are already blocking by contract -- and never in the launch, where a
    single ``bool()`` would collapse the overlap the whole pipeline exists for.

    ``token`` is present only under ``forward_graph="on"`` (KOM-048), and it
    turns this handle into the second half of a handshake: the three tensors
    above are then **views into the graph's static output buffers**, not this
    batch's own allocations, and they stop being this batch's answer the moment
    the next replay is enqueued. Reading them here is what releases the
    buffers -- see :class:`_ReplayToken`.
    """

    __slots__ = (
        "_event",
        "_policy",
        "_wdl",
        "_moves_left",
        "_staged",
        "_counts",
        "_index",
        "_forward_dtype",
        "_token",
    )

    def __init__(
        self,
        event: _CudaEventLike,
        policy: torch.Tensor,
        wdl: torch.Tensor,
        moves_left: torch.Tensor,
        staged: torch.Tensor,
        counts: npt.NDArray[np.int64] | None = None,
        index: torch.Tensor | None = None,
        forward_dtype: str = "f32",
        token: _ReplayToken | None = None,
    ) -> None:
        self._event = event
        self._policy = policy
        self._wdl = wdl
        self._moves_left = moves_left
        self._staged = staged
        self._counts = counts
        self._index = index
        self._forward_dtype = forward_dtype
        self._token = token

    def query(self) -> bool:
        """Non-blocking readiness: has every kernel up to the event retired?"""
        return self._event.query()

    def result(self) -> NetEval:
        """The sync point -- ``.cpu()`` inside :func:`_to_numpy` waits for it.

        The device->host copy is what synchronises; no explicit
        ``event.synchronize()`` is needed, and none is issued, because the copy
        is stream-ordered behind the forward that produced these tensors. Each
        ``.cpu()`` lands in a fresh host tensor allocated *outside*
        ``inference_mode``, which is why the numpy conversion below is legal on
        outputs that were produced inside it.

        **That synchronous D2H is load-bearing under ``forward_graph="on"``**
        (KOM-048): the tensors it copies are the captured graph's static output
        buffers, which stop holding this batch's answer the instant the next
        replay is *enqueued*. Copying them here, before this call returns, is
        what makes the answer this batch's own -- and completing that copy is
        what releases the buffers for the next launch. Making this method lazy,
        or deferring the copy behind the event, would silently serve one
        batch's evaluations for another's positions.

        Raises:
            ValueError: If this handle was launched under ``sparse_policy``.
                Its policy tensor is packed, so there is no dense row to return
                and no way to invent one that would not cost more than the
                packing saved. Sparse callers use :meth:`packed_result`.
            RuntimeError: If the forward produced a non-finite wdl
                (:func:`_refuse_nonfinite_wdl`, every ``forward_dtype``
                including ``"f32"``). Checked here, at the sync point, rather
                than in the launch that produced it.
        """
        if self._counts is not None:
            raise ValueError(
                "this forward was launched with sparse_policy=True: its policy "
                "tensor holds packed legal logits, not a dense (N, A) row. Use "
                "packed_result()"
            )
        _refuse_nonfinite_wdl(self._wdl, self._forward_dtype)
        evaluation = NetEval(
            policy_logits=_to_numpy(self._policy),
            wdl=_to_numpy(self._wdl),
            moves_left=_to_numpy(self._moves_left),
        )
        # After the D2H, never before: the handshake releases the graph's
        # static buffers only once their bytes are on the host (KOM-048).
        if self._token is not None:
            self._token.consume()
        return evaluation

    def packed_result(self) -> PackedPolicy:
        """The sync point for the sparse path: D2H the packed rows.

        Raises:
            ValueError: If this handle was launched dense. Packing here would
                mean D2H-ing the full ``(N, A)`` policy first, which is exactly
                the transfer sparse mode exists to avoid.
            RuntimeError: If the forward produced a non-finite wdl, exactly as
                in :meth:`result`.
        """
        if self._counts is None:
            raise ValueError(
                "this forward was launched with sparse_policy=False: its policy "
                "tensor is dense and packing it here would cost the D2H the "
                "sparse transport exists to avoid. Use result()"
            )
        _refuse_nonfinite_wdl(self._wdl, self._forward_dtype)
        packed = PackedPolicy(
            values=_to_numpy(self._policy),
            counts=self._counts,
            wdl=_to_numpy(self._wdl),
            moves_left=_to_numpy(self._moves_left),
        )
        # Same rule as result(): the release follows the copy (KOM-048).
        if self._token is not None:
            self._token.consume()
        return packed


_FORWARD_ROWS: dict[int, int] = {}
"""How many forwards of each width this PROCESS has executed (KOM-051).

``rows -> count``, and **monotone**: an entry only ever appears and only ever
grows, nothing here resets it, and it is written at exactly one site --
:meth:`AsyncEvaluator.launch`, which is the server's forward path
(:meth:`ShardServer.serve` builds one of these and :func:`serve_pipelined`
launches every batch through it).

It exists because the batch a shard actually assembled was, until now,
unobservable from outside the shard. The KOM-121 permutation suite says so in
as many words -- "the pinned seam exposes no hook for it" -- so a concurrency
test could assert that every answer was *correct* but never that more than one
row was ever in a batch, and a server that quietly served every request alone
passed such a test vacuously (audit gap G6,
``docs/design/kom-101-121-test-debt-audit.md`` section 4). This is the hook,
and its only job is to make that guard possible.

**Observability, never protocol.** Nothing in this module reads it, no
behaviour branches on it, and it carries no invariant: a reader that finds it
empty learns that this process ran no forward through
:class:`AsyncEvaluator`, and nothing else. A launcher stub -- which is what the
pipelined unit suite drives :func:`serve_pipelined` with -- is therefore
correctly invisible here.

Deliberately per-process module state, and deliberately not re-exported from
:mod:`komugi.selfplay`: a shard runs in a *spawned* process (design D2), so
this dict describes that process's forwards and nobody else's. A reader outside
it has to ask the shard, which is exactly honest about where the batching
happened.

The cost when nobody reads it is one dict lookup and one store per batch --
nanoseconds beside a forward measured in milliseconds -- and there is
deliberately no flag to switch it off, because a knob would give an empty
histogram two indistinguishable meanings ("nothing batched" and "nobody was
counting"). Both operations are single bytecodes on a builtin dict, so the GIL
makes them atomic; a shard serves from one thread anyway.
"""


def _record_forward_rows(rows: int) -> None:
    """Count one forward of `rows` rows. See :data:`_FORWARD_ROWS`."""
    _FORWARD_ROWS[rows] = _FORWARD_ROWS.get(rows, 0) + 1


def forward_row_histogram() -> dict[int, int]:
    """This process's :data:`_FORWARD_ROWS`: ``rows -> forwards of that width``.

    A **copy**, so a reader cannot mutate the counter it is reading, and a
    **snapshot**, so what it returns is a fact about the forwards that have
    already happened rather than a live view a running server would change
    underneath it.
    """
    return dict(_FORWARD_ROWS)


class AsyncEvaluator:
    """:class:`InProcessEvaluator`'s forward, split into launch and collect.

    Same inputs, same outputs, same device-follows-the-model rule; the only
    difference is *when* the answer is available. On CUDA a launch stages the
    observations through pinned host memory, copies them
    ``non_blocking=True``, enqueues the forward under
    :func:`torch.inference_mode`, records a :class:`torch.cuda.Event`, and
    returns -- the caller gets a handle while the GPU is still working, which
    is the whole point of KOM-032.

    On any other device there is nothing to overlap against, so ``launch``
    runs the forward *synchronously*, through :class:`InProcessEvaluator`, and
    returns an already-finished handle. That is not a fallback stub: it is what
    makes the pipelined loop's CPU output **bit-identical** to the serial
    loop's (invariant I6), and therefore what lets the gate -- which has no GPU
    -- verify the pipelining logic against the serial path it replaces.

    Under ``forward_dtype`` the two paths stay indistinguishable in the same
    way (KOM-039): the CUDA launch casts its staged batch to half inside the
    device-work region and the CPU path does not cast at all, because the
    weights it forwards through were never converted. Neither is a decision
    made here -- both read :func:`_weights_are_half`, and the conversion itself
    happened once in the :class:`InProcessEvaluator` this wraps, on the very
    model object this class forwards through. There is deliberately no second
    conversion site: two would mean two chances to disagree about the device.

    Under ``forward_graph="on"`` (KOM-048) this class is also the **only**
    thing in the module that captures a CUDA graph. The capture happens here,
    once, at construction, *after* the :class:`InProcessEvaluator` below has
    put the model in eval mode and applied any fp16 conversion -- so what is
    captured is the network as it will actually serve. Two properties of the
    knob-on path are worth stating where the class is, because both are
    weakenings of promises this class otherwise makes:

    * **The model is pinned.** The device-follows-the-model rule survives, but
      only in its construction-time form: a capture bakes in parameter
      addresses, so the model may not move afterwards and every launch checks.
    * **The pipeline's overlap is gone.** One captured graph has one set of
      static output buffers, so a launch may not be issued until the previous
      launch's handle has been read (:class:`_ReplayToken`). That is a real
      cost, paid knowingly: what graphs mode buys is the per-launch CPU cost of
      enqueueing several dozen short kernels, which is what a 9x9 net's serving
      forward is bounded by.

    Args:
        model: The network, placed by the caller. Put into eval mode -- and,
            under ``forward_dtype="f16"`` on CUDA, converted to half weights
            with float32 BatchNorms -- by the :class:`InProcessEvaluator` this
            wraps.
        config: Shapes the launcher validates against, the precision it
            forwards in, and whether that forward is captured.

    Raises:
        RuntimeError: If ``forward_graph="on"`` on a ROCm build, a
            host-resident model, or beside ``forward_dtype="f16_autocast"``
            (:func:`_refuse_ungraphable`), or if the model is in train mode at
            capture.
    """

    def __init__(self, model: torch.nn.Module, config: InferenceConfig) -> None:
        self._model = model
        self._config = config
        # Refuse before anything is mutated or captured: an evaluator that
        # cannot graph must not leave a half-converted model behind it.
        if config.forward_graph == "on":
            _refuse_ungraphable(config, _model_device(model), torch.version.hip)
        self._serial = InProcessEvaluator(model, config)
        # After the serial evaluator, never before: it is what puts the model in
        # eval mode and converts the weights under forward_dtype="f16", and the
        # capture must record the network as it will serve (KOM-048).
        self._graph: _GraphRunner | None = (
            _GraphRunner(model, config, _model_device(model))
            if config.forward_graph == "on"
            else None
        )

    @property
    def config(self) -> InferenceConfig:
        """The config this evaluator validates against."""
        return self._config

    def launch(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> ForwardHandle:
        """Start one batch forward. See :class:`ForwardLauncher`.

        Under ``sparse_policy`` `legal` is what the pack is computed from and
        the handle returned is a packing one; otherwise it is validated and
        otherwise unused, exactly as in :meth:`InProcessEvaluator.evaluate`.

        The caller may move the model between launches -- **except** under
        ``forward_graph="on"``, where that permission is revoked (KOM-048): the
        captured graph holds parameter addresses, and a launch after a move
        raises rather than replaying the old device's memory.

        Raises:
            ValueError: Under ``forward_graph="on"``, if the batch is wider
                than the captured shape (``config.max_batch``).
            RuntimeError: Under ``forward_graph="on"``, if the model has moved
                since capture, or if the previous launch's handle was never
                read (:class:`_ReplayToken`).
        """
        _validate_batch(obs, legal, self._config)
        # KOM-051's observability seam (:data:`_FORWARD_ROWS`), at the one site
        # every server-side forward passes through. After the shape check, so a
        # mis-shaped batch that never becomes a forward is not counted, and
        # before the device work, so there is exactly one record per launch --
        # the CPU path below delegates to `self._serial`, which does not count,
        # precisely so that one forward is one entry on both paths. The three
        # refusals after this point (a moved model, a batch wider than the
        # capture, an unread replay) are all fatal by design, so a counted batch
        # either ran or ended the run.
        _record_forward_rows(int(obs.shape[0]))
        # The device is read per launch, not cached: nothing here names a
        # device, the model's placement decides, and a caller is entitled to
        # move the model between launches. Under forward_graph="on" that last
        # clause no longer holds -- `_GraphRunner.require_device` refuses a move
        # rather than following it -- and this read is what detects it.
        device = _model_device(self._model)
        runner = self._graph
        if runner is not None:
            # Before the CPU fallback below, deliberately: a model moved back
            # to the host must raise, not be quietly served eagerly.
            runner.require_device(device)
        if device.type != "cuda":
            return _ReadyHandle(self._serial.evaluate(obs, legal), legal)

        # Pinned staging is what makes the copy asynchronous at all: a pageable
        # source forces torch to stage it synchronously, and the launch would
        # not return until the bytes were on the device.
        staged = torch.from_numpy(
            np.ascontiguousarray(obs, dtype=np.float32)
        ).pin_memory()
        # The eager path's H2D is enqueued HERE, before the sparse-index CPU
        # work below, so the copy overlaps it (design review D3/5; hoisted
        # back per impl review G2 -- the graphs path instead lands its copy
        # in the captured static input inside _GraphRunner.run).
        batch: torch.Tensor | None = (
            None if runner is not None else staged.to(device, non_blocking=True)
        )
        counts: npt.NDArray[np.int64] | None = None
        index_staged: torch.Tensor | None = None
        index_device: torch.Tensor | None = None
        if self._config.sparse_policy:
            # The index is CPU work on a mask the server already holds, and it
            # rides the same pinned/non_blocking path the observations do, so
            # nothing here waits on the device (design review D3/5).
            ids, counts = _sparse_indices(legal)
            index_staged = torch.from_numpy(ids).pin_memory()
            index_device = index_staged.to(device, non_blocking=True)
        if runner is not None:
            # KOM-048: the H2D lands directly in the captured static input and
            # the forward is one replay. Outside inference_mode, deliberately:
            # the static input is an ordinary tensor and these are in-place
            # mutations of it (see _GraphRunner). Nothing is waited on here.
            runner.run(staged, int(obs.shape[0]))
        with torch.inference_mode():
            if runner is None:
                assert batch is not None
                out: _NetOutputLike = self._forward_eager(batch, device)
            else:
                out = runner.outputs(int(obs.shape[0]))
            # Probabilities, not logits, cross the seam -- the same rule as the
            # serial path, applied at the same point (design D1). The .float()
            # before the softmax is KOM-039's load-bearing cast: an fp16 softmax
            # saturates, and this is the tensor every prior is derived from.
            wdl = torch.softmax(out.wdl_logits.float(), dim=-1)
            policy = out.policy_logits
            moves_left = out.moves_left.float()
            if index_device is not None:
                # KOM-038: pack on the device, INSIDE the forward's stream and
                # BEFORE the D2H, which is the whole win -- one gather over
                # (N, A) replaces a (N, A) transfer with a (N, Kmax) one.
                # `gather` and not `masked_select`: the latter sizes its output
                # from a device-side count, and nonzero on ROCm is slow enough
                # to eat the saving (design review D3).
                policy = torch.gather(policy, 1, index_device)
            # Float32 on the way out, because that is the seam's numpy contract
            # in every forward_dtype (the wire dtype is policy_dtype's business,
            # not this knob's). Last, so that under forward_dtype="f16" the
            # gather above still runs on half-width rows; the identity on f32.
            policy = policy.float()
        event = _new_cuda_event()
        event.record()
        # I5: the handle -- not this frame -- owns `staged` (and the pinned
        # index, for the identical reason) until the sync.
        return _CudaHandle(
            event,
            policy,
            wdl,
            moves_left,
            staged,
            counts,
            index_staged,
            self._config.forward_dtype,
            None if runner is None else runner.token,
        )

    def _forward_eager(
        self, batch: torch.Tensor, device: torch.device
    ) -> _NetOutputLike:
        """The un-captured forward: today's launch, moved into one method.

        Called from inside :meth:`launch`'s ``inference_mode`` region and from
        nowhere else. It exists so that the knob's two paths are two branches
        rather than two interleaved blocks -- the tail below them (softmax,
        KOM-038 gather, outbound casts, event) is then provably the same code
        on both. The H2D is NOT here: :meth:`launch` enqueues it right after
        pinning, before the sparse-index CPU work, exactly where the
        pre-KOM-048 launch did (impl review G2 restored that overlap).

        Args:
            batch: The device batch, its async H2D already enqueued.
            device: The model's parameter device, already known to be CUDA.
        """
        if _weights_are_half(self._config, device):
            # KOM-039: the cast lives INSIDE the device-work region, beside the
            # forward and the gather, so the half batch is produced on the same
            # stream that consumes it and the launch still returns without
            # waiting for anything.
            batch = batch.half()
        with _forward_autocast(self._config, device):
            out: _NetOutputLike = self._model(batch)
        return out

    def evaluate(
        self,
        obs: npt.NDArray[np.float32],
        legal: npt.NDArray[np.bool_],
    ) -> NetEval:
        """Launch and wait: the :class:`BatchEvaluator` shape of this seam.

        For callers with nothing to overlap -- and any test that wants one
        forward's worth of answer.

        Raises:
            ValueError: On CUDA under ``sparse_policy``, where the launched
                forward's policy tensor is packed and no dense row exists to
                return. That combination has no caller: the server publishes
                through :func:`_publish`, which asks the handle for its packed
                rows. On CPU it stays dense and this keeps working, because the
                CPU handle holds the serial evaluator's own dense answer.
        """
        return self.launch(obs, legal).result()


def slot_label(slot_id: int, lanes: int = 1) -> str:
    """Name a slot the way an operator can act on it (KOM-040 b1 review F3).

    Since interleaving, a slot id is **not** a worker id: the orchestrator
    numbers segments ``worker * lanes + lane``, so under
    ``games_in_flight=2`` slot 3 is worker 1's lane 1. Every diagnostic that
    printed "worker 3" for that slot sent the operator reading the wrong
    process's output during an incident, which is the first minute of every
    incident. The thread name carried the truth; the exception did not.

    Two honest forms, and no third:

    * ``lanes > 1`` -- ``"worker 1 lane 1 (slot 3)"``. Both identities, with
      the raw slot kept because that is what the server's own bookkeeping and
      the segment names are keyed by.
    * otherwise -- ``"slot 3"``. When the lane geometry is not known here, the
      number is named for what it certainly is rather than for what it is only
      sometimes: at ``lanes == 1`` the slot id *equals* the worker id, but a
      client that was never told its geometry cannot tell that case from the
      interleaved one, and guessing "worker" is exactly the mislabel this
      exists to delete.

    Args:
        slot_id: The segment's index in the run, as the server keys it.
        lanes: ``games_in_flight`` for the run, when the caller knows it.
            Default ``1`` means "not known here" and yields the ``slot N``
            form.

    Returns:
        A label for interpolation into a diagnostic, without a trailing period.
    """
    if lanes > 1:
        worker, lane = divmod(slot_id, lanes)
        return f"worker {worker} lane {lane} (slot {slot_id})"
    return f"slot {slot_id}"


class SlotClient:
    """The worker's end of the seam: one request in flight, blocking.

    Satisfies nothing formally -- ``submit`` takes a single position, not a
    batch -- because that is what a tree search needs: the worker adapts this to
    the search's scalar ``Evaluator`` one leaf at a time, and *the server* is
    where batching happens, across games (design D1).

    The submission order is load-bearing (design D2, KOM-021b)::

        write the slots  ->  increment request_seq  ->  poll response_seq

    The increment *is* the doorbell, and it comes strictly after the payload:
    the server treats ``request_seq > response_seq`` as "a complete request is
    in this slot", so a counter published before the bytes would hand the
    server a half-written observation. Nothing in this path takes a lock, opens
    a file descriptor or touches a semaphore -- see the module docstring for
    why that, and not the shared memory, was the thing SIGKILL used to break.

    Args:
        shm_name: Name of *this worker's* segment, created by the orchestrator.
        worker_id: This client's **slot** id, as the server knows it -- the
            name is the protocol's, from before a worker could hold more than
            one slot. Identification only: the request signal is the segment's
            own ``request_seq``, so nothing about a submit leaves the segment.
        config: Must be the same config the server holds. Under KOM-038 that is
            no longer only a convention: ``sparse_policy`` decides how the
            response bytes are read, and a disagreement is caught by the
            ``policy_count`` tripwire (:data:`DENSE_POLICY_COUNT`) rather than
            silently believed.
        lanes: ``games_in_flight``, when the constructing worker knows it. Used
            for **diagnostics only** (:func:`slot_label`, review F3): it is
            what turns "slot 3" into "worker 1 lane 1 (slot 3)" in this
            client's two failure messages. Nothing about the protocol reads it,
            and the default ``1`` -- "geometry not stated" -- leaves every
            message naming the slot, which is true of any run.
    """

    def __init__(
        self,
        shm_name: str,
        worker_id: int,
        config: InferenceConfig,
        lanes: int = 1,
    ) -> None:
        self._worker_id = worker_id
        self._lanes = lanes
        self._config = config
        self._slots = attach_slots(shm_name, config)
        self._poison: str | None = None

    @property
    def worker_id(self) -> int:
        """This client's slot id, as the server sees it."""
        return self._worker_id

    @property
    def label(self) -> str:
        """How this client names itself in a diagnostic (:func:`slot_label`)."""
        return slot_label(self._worker_id, self._lanes)

    @property
    def config(self) -> InferenceConfig:
        """The config this client's segment and codec were built from."""
        return self._config

    @property
    def max_wave(self) -> int:
        """The widest wave one ticket may carry (``config.max_wave``)."""
        return self._config.max_wave

    def submit(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate one position, blocking until the server answers.

        The degenerate case of :meth:`submit_wave` -- a wave of one row -- and
        implemented as exactly that, so there is one submission path and not
        two. Every refusal :meth:`submit_wave` makes applies here, including the
        all-false mask refusal (KOM-038's flip-gate item 3).

        Args:
            obs: ``(in_planes, 9, 9)`` float32.
            legal: ``(A,)`` bool.

        Returns:
            A :class:`NetEval` for this one position: shapes ``(A,)``, ``(3,)``
            and ``()``, with no batch dimension. Always dense, in both transport
            modes -- under ``sparse_policy`` the full vector is rebuilt here,
            from `legal`, with a finite ``0.0`` in the illegal lanes (design
            review D1/D6). Nothing above this method can tell which transport
            ran.

        Raises:
            ValueError: If `legal` has no legal actions. See
                :meth:`submit_wave`; the refusal happens before the doorbell.
            InferenceTimeout: After ``config.request_timeout_s``, or on any
                later submit once this client has been poisoned. Fatal -- see
                the exception's docstring; do not catch it to retry.
            InferenceProtocolError: If the response's ``policy_count`` does not
                match the mode this client reads in, or its ``row_count_echo``
                is not 1. Also poisons the client: the two ends disagree about
                the wire format, and every further answer would be wrong in the
                same silent way.
        """
        return self.submit_wave(obs[None], legal[None])[0]

    def submit_wave(
        self,
        obs_rows: npt.NDArray[np.float32],
        legal_rows: npt.NDArray[np.bool_],
    ) -> list[NetEval]:
        """Evaluate ``1..max_wave`` positions as ONE ticket, blocking.

        The KOM-040 b2 transport (design rev2). The b1 bench showed the shared
        batch already synchronises the lanes, so what is left to cut is the
        **count** of round trips: a wave writes every row, rings the doorbell
        **once**, and gets one answer back under one counter. Everything about
        the doorbell protocol is unchanged -- payload first, counter second, one
        ticket in flight, a fresh segment per incarnation.

        Two refusals happen **before** the doorbell rings, because a ticket that
        has been announced cannot be taken back:

        * **Zero rows.** A ticket that describes no rows is unanswerable: the
          server would have nothing to gather and the client nothing to read.
        * **A row whose mask is all-false.** ``K == 0`` is refused at the
          client, not merely deep in the sparse codec (KOM-038's flip-gate item
          3): a position with no legal move is never evaluated -- the search
          refuses it first -- so this is a caller bug in *both* transport
          modes, and finding it here means finding it in the dense default
          rather than only when ``sparse_policy`` is flipped on.

        Args:
            obs_rows: ``(R, in_planes, 9, 9)`` float32, ``1 <= R <= max_wave``.
            legal_rows: ``(R, A)`` bool.

        Returns:
            ``R`` :class:`NetEval` values in submission order, each shaped as
            :meth:`submit`'s.

        Raises:
            ValueError: On zero rows, more rows than ``max_wave``, a shape
                mismatch, or any row with no legal actions -- all before the
                doorbell.
            InferenceTimeout: As :meth:`submit`. One deadline covers the whole
                wave (W8): a ticket is one request, so it gets one patience.
            InferenceProtocolError: On a bad ``row_count_echo`` or a per-row
                mode mismatch. Poisons the client.
        """
        if self._poison is not None:
            raise InferenceTimeout(
                f"the client for {self.label} is poisoned: "
                f"{self._poison}. A resubmit would ring this slot's doorbell "
                f"again while the server may still be answering the abandoned "
                f"request, re-creating the torn-read race the fatal policy "
                f"exists to delete (wave-9a review F2)."
            )
        # Shapes first (pack_wave_request owns them), then the K=0 refusal --
        # and both strictly before the doorbell. Writing the payload of a
        # request that is then refused costs nothing and is observable by
        # nobody: the slot is this worker's own until request_seq moves.
        rows = pack_wave_request(self._slots, obs_rows, legal_rows)
        _refuse_empty_masks(legal_rows[:rows])
        # The payload is in the slot; NOW ring the doorbell. The ordering is
        # the protocol -- the server reads the payload only because this store
        # happened, so this store must not happen first. Nobody else writes
        # request_seq, so the read-increment-write below cannot race. ONE ring
        # for the whole wave: that is the entire point of the ticket.
        ticket = int(self._slots.request_seq[0]) + 1
        self._slots.request_seq[0] = ticket

        # One deadline per submission, not per row (W8): the wave is one
        # request, answered under one counter, so it is one wait.
        deadline = time.monotonic() + self._config.request_timeout_s
        while True:
            # response_seq is published after the response bytes, so seeing the
            # ticket here means the whole answer -- every row of it -- is
            # already in the slot.
            if int(self._slots.response_seq[0]) == ticket:
                return self._read_answer(legal_rows[:rows])
            if time.monotonic() >= deadline:
                self._poison = (
                    f"it waited out {self._config.request_timeout_s}s for a "
                    f"response, and timeout is fatal by design (D2), never "
                    f"retried"
                )
                raise InferenceTimeout(
                    f"{self.label} got no inference response within "
                    f"{self._config.request_timeout_s}s; this is fatal by design "
                    f"(D2) -- the worker exits and the orchestrator respawns it "
                    f"against a fresh segment"
                )
            time.sleep(_CLIENT_POLL_S)

    def _read_answer(self, legal_rows: npt.NDArray[np.bool_]) -> list[NetEval]:
        """Decode the response now sitting in the slot, in this client's mode.

        A protocol error here poisons the client for the same reason a timeout
        does: the two ends disagree about the bytes, so every subsequent answer
        is wrong in the same undetectable way, and continuing would fill the
        training data with plausible garbage.
        """
        try:
            return read_wave_response(self._slots, legal_rows, self._config)
        except InferenceProtocolError as error:
            self._poison = str(error)
            raise

    def close(self) -> None:
        """Unmap this worker's segment. Never unlinks (the orchestrator does)."""
        self._slots.close()


@dataclass(frozen=True)
class SlotHandle:
    """What the server needs to know about one client, built by the orchestrator.

    Deliberately not a :class:`SlotClient`: the server attaches to the segment
    itself, and a segment *name* survives pickling into a spawned process while
    a live numpy view does not.

    Two fields, and no synchronisation primitive among them: with the doorbell
    protocol the counters live inside the segment, so the name is the whole
    channel. That is why nothing here needs to be inheritable, and why a
    respawned worker's handle is nothing but a new name (design D2/Q4).

    Attributes:
        shm_name: The worker's segment name -- the entire signalling channel.
        worker_id: That client's **slot** id, used to key the server's slot map
            and to name it in diagnostics. Since KOM-040 b1 this is a slot
            identity and not a worker identity: the orchestrator hands out
            ``worker * games_in_flight + lane``, so at two lanes slot 3 belongs
            to worker 1. The field keeps its protocol name; diagnostics that
            print the number go through :func:`slot_label`, which never calls
            it a worker unless it has been told the lane geometry (review F3).
    """

    shm_name: str
    worker_id: int


class ShardServer:
    """One server shard: scan doorbells, one forward, scatter, publish.

    The loop, per iteration (design D2/D3, doorbells per KOM-021b) -- stated
    here in its serial form, which is :func:`serve_once` and which the
    production loop :func:`serve_pipelined` overlaps rather than changes:

    1. Scan every client's slot for ``request_seq > response_seq``. No queue to
       block on, so an idle iteration finds nothing and sleeps
       ``_SERVER_IDLE_POLL_S``; an idle shard must not spin a core.
    2. Keep re-scanning until the batch reaches :func:`_row_ceiling` **rows**
       -- ``min(max_batch, max_wave * len(clients))``, because each worker has
       at most one ticket in flight and a ticket carries at most one wave
       (design D2/F16, KOM-040 b2) -- or until ``batch_timeout_ms`` since the
       first pending request was discovered. If more rows are pending than the
       ceiling admits, the least-served tickets go first and the rest are
       served next iteration, whole (see :func:`_admit`); the FIFO queue this
       replaced was fair for free, and a bare scan order would not be.
    3. One :class:`InProcessEvaluator` forward for the whole batch.
    4. Write each response, then publish that worker's ``response_seq`` as the
       ``request_seq`` value that was gathered -- bytes first, counter second.

    :meth:`serve` runs :func:`serve_pipelined`, unconditionally and with no
    config flag (design review D4): step 1's collect for the *next* batch
    happens during step 3's forward. There is no serial production mode to
    select, which is the point -- every behaviour suite reaches the server
    through :meth:`serve`, so the path they cover is the path production runs
    (L19). :func:`serve_once` remains as the serial reference, as the drain
    pass's shape, and as what the unit ordering-pins drive directly.

    A dead worker's pending request is served harmlessly: its slot is still
    readable (the orchestrator holds the segment, and attachers never unlink it
    -- F9), the forward includes its row, the response is written, and nobody
    is waiting for it. Publishing ``response_seq`` also *clears* it from the
    pending scan, so a worker that died between its doorbell and its answer
    costs exactly one wasted row, once, rather than a hot loop. The server must
    never crash on this; a self-play run that dies because a worker did is the
    failure mode this whole protocol exists to avoid.

    Server death is *not* handled here: the run fails loudly, and per-shard
    respawn with generation counters is pre-registered as P1 (design D2/Q4).

    Args:
        model_factory: Called **once, inside the serving process** to build the
            model. A factory, not a model, because a torch module must never be
            pickled across a process boundary (design D2) -- the factory loads a
            checkpoint by path.
        config: Shapes and timing; must match every client's.
        clients: One handle per worker this shard serves. Sticky assignment;
            the orchestrator decides it.
    """

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        config: InferenceConfig,
        clients: Sequence[SlotHandle],
    ) -> None:
        self._model_factory = model_factory
        self._config = config
        self._clients = tuple(clients)
        self._max_batch = _row_ceiling(config, len(self._clients))

    @property
    def max_batch(self) -> int:
        """The effective batch ceiling in ROWS (F16); see :func:`_row_ceiling`.

        ``min(max_batch, max_wave * len(clients))``, which at the default
        ``max_wave == 1`` is the ``min(max_batch, len(clients))`` this has
        always been.
        """
        return self._max_batch

    def serve(self, stop_event: synchronize.Event) -> None:
        """Serve until `stop_event` is set, running the pipelined loop.

        The dispatch is unconditional (design review D4): there is no flag,
        because a flag whose default were serial would mean every behavioural
        suite tested a path production does not run. When `stop_event` is
        observed, :func:`serve_pipelined` drains both unanswered cohorts before
        returning, so no worker is left polling a slot nobody will fill.

        Everything expensive -- building the model, attaching the segments --
        happens here rather than in ``__init__`` so that it happens in the
        process that will use it.

        ``stop_event`` is the one synchronisation primitive left in the seam,
        and it is deliberate: it is *orchestrator-owned*, set once at shutdown,
        and never touched by a worker. The orchestrator is the process whose
        death is allowed to fail the run loudly (design D2/Q4), so an event it
        alone writes carries none of the SIGKILL exposure the per-worker
        signalling had.

        The ``clients`` list is FIXED at serve start: there is no re-attach,
        no generation counter, no dynamic registration. Consequently, in P0a
        **worker respawn implies restarting the whole shard** -- a respawned
        worker's fresh segment can never reach a running server, and its
        doorbell would be rung in a slot this server is not scanning (wave-9a
        review F5). The orchestrator must treat any worker death as a shard
        restart; dynamic registration with generations is the pre-registered P1
        ticket (design D2/Q4).
        """
        evaluator = AsyncEvaluator(self._model_factory(), self._config)
        slots = {
            handle.worker_id: attach_slots(handle.shm_name, self._config)
            for handle in self._clients
        }
        try:
            serve_pipelined(slots, evaluator, self._config, stop_event)
        finally:
            for entry in slots.values():
                entry.close()


def _scan_pending(
    slots_map: Mapping[int, Slots],
    pending: dict[int, int],
    exclude: frozenset[int] = frozenset(),
) -> None:
    """Add every newly-rung doorbell in `slots_map` to `pending`.

    ``pending`` maps worker id to the ``request_seq`` value that was observed --
    the *ticket*, which is what the server must eventually publish back as
    ``response_seq``. Re-reading the counter later could see a different value
    only if the protocol were already broken, but carrying the observed value
    forward means the answer is tied to the request it was computed from rather
    than to whatever the slot says at scatter time.

    ``exclude`` is the pipelined server's **in-flight set** (invariant I1,
    design review D1) and it is load-bearing, not an optimisation. A worker
    whose request is in flight still reads ``request_seq > response_seq`` for
    the entire duration of the forward -- ``response_seq`` is published at
    scatter, not at gather -- so a scan run *during* that forward would re-admit
    every one of those workers into the next batch. The consequences are not
    merely wasteful rows: the next gather would ``read_request`` a slot whose
    owner has by then been answered and may be mid-``pack_request``, and the
    next scatter would rewrite response bytes the worker may be mid-
    ``read_response`` of. Both are the torn reads the single-writer discipline
    of :class:`Slots` exists to make impossible. Serial callers pass nothing;
    nothing is ever in flight for them.
    """
    for worker_id, entry in slots_map.items():
        if worker_id in pending or worker_id in exclude:
            continue
        ticket = int(entry.request_seq[0])
        # The single comparison the whole protocol rests on: the worker has
        # rung a doorbell this server has not answered.
        if ticket > int(entry.response_seq[0]):
            pending[worker_id] = ticket


class _Admission(NamedTuple):
    """One admitted ticket: whose it is, which request, and how wide.

    ``rows`` is what makes the batch arithmetic row-aware rather than
    ticket-aware -- a ticket is no longer worth one row (KOM-040 b2).
    """

    worker_id: int
    ticket: int
    rows: int


def _row_ceiling(config: InferenceConfig, slot_count: int) -> int:
    """The effective batch ceiling, **in rows**.

    Two bounds meet here (KOM-040 b2 makes the second row-aware):

    * ``max_batch`` -- what one forward may carry.
    * ``max_wave * slot_count`` -- what the live workers can physically have in
      flight, because each has at most one ticket pending and a ticket is at
      most a full wave (design D2/F16, restated per row).

    At ``max_wave == 1`` this is exactly the old ``min(max_batch,
    len(clients))``. The result is never below ``max_wave`` when there is at
    least one slot, because :class:`InferenceConfig` refuses ``max_wave >
    max_batch`` -- which is what makes :func:`_admit`'s "rank 1 always fits"
    claim a theorem rather than a hope.
    """
    return min(config.max_batch, config.max_wave * max(slot_count, 1))


def _fairness_order(pending: Mapping[int, int]) -> list[tuple[int, int]]:
    """Pending tickets, least-served first.

    The queue the doorbells replaced was FIFO and therefore fair for free; a
    plain scan is not, and a fixed scan order would let low-numbered workers
    starve their siblings all the way to :class:`InferenceTimeout`.

    Fairness comes back without any server-side state, out of the counters
    themselves: ``request_seq`` is exactly the number of round trips that
    worker has *started*, so the lowest tickets belong to the workers that have
    been served least. With waves, "least-served" therefore means fewest
    ROUND TRIPS, not fewest rows -- at equal rank a wave client receives up to
    ``max_wave`` rows per trip against a scalar client's one; harmless under a
    homogeneous production config, visible in mixed-width tests (b2 review
    F4). A worker passed over keeps its ticket while its
    siblings' climb, so it strictly rises in this ordering and must be served.
    Worker id breaks ties, only so the choice is deterministic.
    """
    return sorted(pending.items(), key=lambda item: (item[1], item[0]))


def _admit(
    slots_map: Mapping[int, Slots],
    pending: Mapping[int, int],
    ceiling_rows: int,
    config: InferenceConfig,
) -> tuple[list[_Admission], list[int]]:
    """Assemble one batch: fairness-ordered, **no-split** (KOM-040 b2, W4).

    Walk the pending tickets in :func:`_fairness_order` and take each one whole
    or not at all:

    * The **rank-1 ticket always fits**. The batch starts empty and
      ``ceiling_rows >= max_wave`` (see :func:`_row_ceiling`), so the widest
      possible wave is admissible at rank 1.
    * A lower-ranked wave that does not fit the remaining space is **skipped**,
      never split. A ticket answered half-way cannot be published: there is one
      counter, ``response_seq``, and publishing it means "every row of this
      wave is in the slot". Half an answer under a whole counter is exactly the
      torn read the protocol exists to make impossible.
    * Ranks *after* a skipped one may still fill the remaining space. Leaving
      the tail of the batch empty to preserve a strict order would waste rows
      the GPU is paying for either way, and it buys nothing: fairness is
      already carried by the ordering, not by the packing.

    **Starvation bound.** An unserved wave keeps its ticket while every wave
    that *is* served resubmits with a higher one, so its rank strictly rises on
    every forward it misses. It therefore reaches rank 1 in a bounded number of
    forwards, and at rank 1 it is admitted by the first bullet. No wave can be
    skipped indefinitely, however wide it is.

    Returns:
        ``(admitted, poisoned)``. ``poisoned`` holds the worker ids whose
        ``row_count`` header failed the W5 tripwire: they are dropped from this
        batch and the caller must stop scanning them. Poisoning one client is
        not a reason to lose the shard -- every other worker is still served,
        and the offender's own blocking client dies on its timeout, which is
        the fatal-by-design policy it already lives under.
    """
    admitted: list[_Admission] = []
    poisoned: list[int] = []
    used = 0
    for worker_id, ticket in _fairness_order(pending):
        try:
            rows = _wave_rows(slots_map[worker_id], config)
        except InferenceProtocolError:
            poisoned.append(worker_id)
            continue
        if used + rows > ceiling_rows:
            # Skipped, not split. Keep walking: a narrower wave behind this one
            # may still fit, and this one's rank rises for the next forward.
            continue
        admitted.append(_Admission(worker_id, ticket, rows))
        used += rows
    return admitted, poisoned


def _collected_rows(
    slots_map: Mapping[int, Slots], pending: Mapping[int, int], config: InferenceConfig
) -> int:
    """How many rows the pending tickets would contribute, for the window bound.

    A *bound*, not the tripwire: a ``row_count`` outside ``1..max_wave`` is
    clamped here so the collecting loop still terminates, and the loud refusal
    happens once, in :func:`_admit`, where the ticket is actually read.
    """
    total = 0
    for worker_id in pending:
        rows = int(slots_map[worker_id].row_count[0])
        total += min(max(rows, 1), config.max_wave)
    return total


def serve_once(
    slots_map: Mapping[int, Slots],
    evaluator: BatchEvaluator,
    config: InferenceConfig,
) -> int:
    """Run one scan/gather/forward/scatter step. Returns the rows served.

    Factored out of :meth:`ShardServer.serve` so that the doorbell protocol is
    testable without spawning anything: given a dict of attached slots and any
    :class:`BatchEvaluator`, this is the server's entire per-iteration
    behaviour, and ``serve`` is the loop plus the idle sleep around it.

    Returns ``0`` when no doorbell was ringing -- the caller's cue to sleep
    rather than to scan again immediately.

    Args:
        slots_map: Worker id -> that worker's attached :class:`Slots`.
        evaluator: The batch forward. One call per invocation, never more.
        config: Supplies ``max_batch``, ``batch_timeout_ms`` and the transport
            mode (``sparse_policy``).
    """
    # The effective ceiling, in ROWS: each worker has at most one ticket in
    # flight and a ticket is at most one wave (design D2/F16, KOM-040 b2).
    ceiling_rows = _row_ceiling(config, len(slots_map))
    pending: dict[int, int] = {}
    _scan_pending(slots_map, pending)
    if not pending:
        return 0

    # The batching window opens at the FIRST pending discovery, exactly as the
    # queue-draining version opened it at the first dequeued id -- and it closes
    # on collected ROWS, not on collected tickets.
    deadline = time.monotonic() + config.batch_timeout_ms / 1000.0
    while _collected_rows(slots_map, pending, config) < ceiling_rows:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(remaining, _SERVER_IDLE_POLL_S))
        _scan_pending(slots_map, pending)

    # Anything above the ceiling -- or skipped whole by the no-split rule --
    # stays pending and is picked up next iteration, without an idle sleep in
    # between (serve_once returned nonzero).
    served, _poisoned = _admit(slots_map, pending, ceiling_rows, config)
    if not served:
        # Every pending ticket failed the header tripwire. Nothing to forward,
        # and the callers' "0 means sleep" contract is the right answer.
        return 0
    obs, legal = _gather(slots_map, served)

    evaluation = evaluator.evaluate(obs, legal)

    if config.sparse_policy:
        # The serial reference has no device-side pack to reach for -- it holds
        # a dense answer from a plain BatchEvaluator -- so it packs on the CPU.
        # That is the same wire format the pipelined path writes, which is what
        # keeps this function usable as the serial reference in both modes.
        _scatter_sparse(slots_map, pack_sparse_batch(evaluation, legal), served)
    else:
        _scatter(slots_map, evaluation, served)
    return int(obs.shape[0])


def _gather(
    slots_map: Mapping[int, Slots], served: Sequence[_Admission]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Copy every admitted ticket's rows out of its slot, stacked.

    Copies, per :func:`read_request`, one row at a time (W7): the batch
    outlives the read, and the worker owns its slot again the moment its answer
    lands. Batch row order is ticket order, each ticket contributing its
    ``rows`` rows contiguously -- which is what lets :func:`_scatter` walk the
    same list with one cursor.
    """
    obs_rows: list[npt.NDArray[np.float32]] = []
    legal_rows: list[npt.NDArray[np.bool_]] = []
    for admission in served:
        entry = slots_map[admission.worker_id]
        for row in range(admission.rows):
            obs_rows.append(np.array(entry.obs[row], dtype=np.float32))
            legal_rows.append(np.array(entry.legal[row], dtype=np.bool_))
    return np.stack(obs_rows), np.stack(legal_rows)


def _scatter(
    slots_map: Mapping[int, Slots],
    evaluation: NetEval,
    served: Sequence[_Admission],
) -> None:
    """Write each ticket's rows into its slot and publish that worker's ticket.

    Bytes first, then the counter that announces them -- the same ordering rule
    as the worker's, in the opposite direction. Publishing the *gathered*
    ticket rather than an increment is what makes ``response_seq ==
    request_seq`` mean "the answer to THAT request" -- and now, with waves,
    "every row of that request".

    ``row_count_echo`` goes in with the rest of the payload, before the
    counter: it is the width the client checks its own submission against (W5).
    """
    row = 0
    for admission in served:
        entry = slots_map[admission.worker_id]
        for slot_row in range(admission.rows):
            write_response(entry, evaluation, row + slot_row, slot_row)
        entry.row_count_echo[0] = admission.rows
        entry.response_seq[0] = admission.ticket
        row += admission.rows


def _scatter_sparse(
    slots_map: Mapping[int, Slots],
    packed: PackedPolicy,
    served: Sequence[_Admission],
) -> None:
    """:func:`_scatter` for the sparse transport: ``K`` logits, then the ticket.

    Same ordering rule, same published-ticket semantics, same per-row copies;
    the only difference is how many bytes each row's policy region receives and
    that its ``policy_count`` entry says so.
    """
    row = 0
    for admission in served:
        entry = slots_map[admission.worker_id]
        for slot_row in range(admission.rows):
            write_sparse_response(entry, packed, row + slot_row, slot_row)
        entry.row_count_echo[0] = admission.rows
        entry.response_seq[0] = admission.ticket
        row += admission.rows


def _packed_result(
    handle: ForwardHandle, legal: npt.NDArray[np.bool_]
) -> PackedPolicy:
    """Block for `handle`'s rows and get them packed to their legal support.

    Both production handles pack themselves -- the CUDA one on the device
    before the D2H, the CPU one right after its synchronous forward -- and both
    are recognised through :class:`_PacksPolicy`. Anything else (a test's
    hand-completed handle) is packed here from its dense answer, which is
    correct but keeps the dense transfer it would have paid anyway.
    """
    if isinstance(handle, _PacksPolicy):
        return handle.packed_result()
    return pack_sparse_batch(handle.result(), legal)


def _publish(
    slots_map: Mapping[int, Slots],
    handle: ForwardHandle,
    legal: npt.NDArray[np.bool_],
    served: Sequence[_Admission],
    config: InferenceConfig,
) -> None:
    """Wait for `handle` and write its rows into `served`'s slots.

    The one place the transport mode is chosen on the server side, so that the
    three publish sites in :func:`serve_pipelined` -- the steady-state scatter
    and the drain's two windows -- cannot drift apart. `legal` is the mask batch
    the forward was launched with, needed only by the CPU fallback pack.
    """
    if config.sparse_policy:
        _scatter_sparse(slots_map, _packed_result(handle, legal), served)
    else:
        _scatter(slots_map, handle.result(), served)


def serve_pipelined(
    slots_map: Mapping[int, Slots],
    launcher: ForwardLauncher,
    config: InferenceConfig,
    stop_event: synchronize.Event,
) -> None:
    """Serve until `stop_event`, overlapping the next collect with the forward.

    The production loop (KOM-032). :func:`serve_once` does scan -> window ->
    forward -> scatter with nothing happening during the forward; here the
    collect for batch N+1 runs *while* batch N is on the GPU, so a worker that
    resubmits during a forward is already in the next batch when it lands.

    Two states, and the difference between them is what the batching window
    means:

    **Idle** (nothing in flight). Scan; if nothing is ringing, sleep
    ``_SERVER_IDLE_POLL_S`` -- an idle shard must not spin a core, and that
    promise survives the pipeline unchanged. Otherwise open the
    ``batch_timeout_ms`` window at the first pending discovery and keep
    scanning until the window closes or the batch reaches the effective
    ceiling, byte for byte what :func:`serve_once` does today.

    **In flight.** Launch, then collect the next batch until the handle reports
    ready or the ceiling is reached -- **ignoring** ``batch_timeout_ms``
    entirely (invariant I3, design review D8). With a 0.5 ms window and a
    multi-millisecond forward, honouring the window here would close every
    collect long before the GPU finished and the pipeline's batches would stay
    as small as the serial loop's. This poll is the only busy-ish loop in the
    server and it runs only while the GPU is busy, at
    ``_SERVER_IDLE_POLL_S`` granularity.

    Three invariants the body is written around:

    * **I1** -- the collect for batch N+1 never admits a worker whose ticket is
      in flight. See :func:`_scan_pending`'s ``exclude``: without it the scan
      re-admits every in-flight worker, because ``response_seq`` is published
      at scatter and not before.
    * **I2** -- the doorbell protocol is untouched in both directions. Every
      read and write here goes through the same ``_scan_pending`` /
      :func:`read_request` / :func:`write_response` calls the serial loop uses,
      in the same payload-before-counter order. Workers cannot tell which loop
      is running.
    * **I4** -- ``stop_event`` leaves **two** cohorts unanswered, not one
      (design review D2): the batch that was launched and not yet scattered,
      and the batch that was collected and not yet launched. Both are drained
      below before returning, and one **final scan** (after the in-flight
      scatter, so I1 still holds) admits any doorbell rung before the drain --
      a stop landing between a worker's ring and the next scan must not drop
      the request, or the worker wedges until ``request_timeout_s`` and dies,
      which the serial loop never did to it. The drain is deliberately
      **not** in a ``finally``: an exception exit means the server is dying,
      and a run whose server died must fail loudly rather than tidy up.

    Args:
        slots_map: Worker id -> that worker's attached :class:`Slots`.
        launcher: Starts one forward per batch (:class:`AsyncEvaluator` in
            production).
        config: Supplies ``max_batch``, ``batch_timeout_ms`` and the transport
            mode (``sparse_policy``).
        stop_event: Orchestrator-owned; set once, at shutdown.
    """
    # The effective ceiling, in ROWS (design D2/F16 restated per row).
    ceiling_rows = _row_ceiling(config, len(slots_map))
    pending: dict[int, int] = {}
    in_flight: list[_Admission] = []
    # Workers whose row_count header failed the W5 tripwire. They are never
    # scanned again: their bytes cannot be trusted, and re-reading the same bad
    # header every iteration would be a hot loop rather than a diagnosis. The
    # shard keeps serving everybody else -- one client's poison is not the
    # run's (W5).
    poisoned: set[int] = set()
    # The mask batch that went into the in-flight forward, kept alongside it:
    # the sparse publish path needs it if the launcher's handle does not pack
    # itself, and it is reassigned in the same breath as `in_flight` so the two
    # can never describe different batches.
    in_flight_legal: npt.NDArray[np.bool_] = np.zeros(
        (0, config.num_actions), dtype=np.bool_
    )
    handle: ForwardHandle | None = None

    while not stop_event.is_set():
        if not pending:
            # Idle state: nothing was collected while the last forward ran.
            _scan_pending(slots_map, pending, exclude=frozenset(poisoned))
            if not pending:
                time.sleep(_SERVER_IDLE_POLL_S)
                continue
            # The window opens at the FIRST pending discovery, as it always
            # has, and closes on collected ROWS.
            deadline = time.monotonic() + config.batch_timeout_ms / 1000.0
            while _collected_rows(slots_map, pending, config) < ceiling_rows:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, _SERVER_IDLE_POLL_S))
                _scan_pending(slots_map, pending, exclude=frozenset(poisoned))

        # Anything above the ceiling -- or skipped whole by the no-split rule --
        # stays in `pending` and rides into the next batch. It is already
        # collected, so it never sees an idle sleep, and its fairness rank rises
        # while it waits (see _admit's starvation bound).
        in_flight, newly_poisoned = _admit(slots_map, pending, ceiling_rows, config)
        poisoned.update(newly_poisoned)
        for worker_id in newly_poisoned:
            del pending[worker_id]
        for admission in in_flight:
            del pending[admission.worker_id]
        if not in_flight:
            # Everything collected was poison. Nothing to launch; fall back to
            # the idle sleep rather than spinning on an empty batch.
            time.sleep(_SERVER_IDLE_POLL_S)
            continue
        in_flight_obs, in_flight_legal = _gather(slots_map, in_flight)
        handle = launcher.launch(in_flight_obs, in_flight_legal)

        # I1: these ids are unanswerable until the scatter below, so the scan
        # must not see them. This is the blocker from the design review.
        excluded = frozenset(
            admission.worker_id for admission in in_flight
        ) | frozenset(poisoned)
        _scan_pending(slots_map, pending, exclude=excluded)
        # I3: bounded by the forward, not by batch_timeout_ms -- and bounded in
        # ROWS, so a collect that has gathered its ceiling of rows stops even
        # though it holds fewer tickets than there are workers.
        while (
            not handle.query()
            and _collected_rows(slots_map, pending, config) < ceiling_rows
        ):
            if stop_event.is_set():
                break
            time.sleep(_SERVER_IDLE_POLL_S)
            _scan_pending(slots_map, pending, exclude=excluded)

        if stop_event.is_set():
            # Leave both cohorts to the drain, which owns them in one place.
            break

        _publish(slots_map, handle, in_flight_legal, in_flight, config)
        handle = None

    # I4, window (a): the launched batch. `_publish` blocks -- the forward is
    # already running and its rows are owed to workers that are blocked on them.
    if handle is not None:
        _publish(slots_map, handle, in_flight_legal, in_flight, config)
    # I4, one final scan, after the in-flight scatter (before it, the scan
    # would re-admit in-flight workers -- I1's duplicate serve): a doorbell
    # rung before this scan is answered even if the loop above never saw it.
    # Without this, a stop that lands between a worker's ring and the next
    # scan drops the request and the worker wedges for request_timeout_s --
    # the union suite's stop-mid-submit test caught exactly that under
    # full-suite scheduling skew. One scan, not a loop of them: rings that
    # happen after it lost the race with stop by microseconds, which is the
    # residual the blocking client's timeout owns.
    _scan_pending(slots_map, pending, exclude=frozenset(poisoned))
    # I4, window (b): everything collected but never launched, served by a
    # final serial pass -- launch, wait, scatter, no overlap left to exploit.
    # No further scan runs, so nothing new is admitted. The loop runs until no
    # pending ticket is left, which may take MORE THAN ONE forward now that a
    # batch is bounded in rows and a wave may be skipped whole: each pass
    # admits at least the rank-1 ticket, so `pending` strictly shrinks and this
    # terminates.
    while pending:
        served, drain_poisoned = _admit(slots_map, pending, ceiling_rows, config)
        for worker_id in drain_poisoned:
            del pending[worker_id]
        for admission in served:
            del pending[admission.worker_id]
        if not served:
            # Unreachable while the rank-1 guarantee holds: an empty admission
            # means every pending ticket was poison, and those were just
            # deleted. `break` rather than `continue` so a future regression is
            # a dropped drain, not a spin.
            break
        drain_obs, drain_legal = _gather(slots_map, served)
        _publish(
            slots_map, launcher.launch(drain_obs, drain_legal), drain_legal,
            served, config,
        )


def serve_shard(
    model_factory: Callable[[], torch.nn.Module],
    config: InferenceConfig,
    clients: Sequence[SlotHandle],
    stop_event: synchronize.Event,
) -> None:
    """Module-level entry point for a spawned shard process.

    A spawned process can only target something importable by name (design D2),
    so the server's process boundary is this function rather than a bound
    method.
    """
    ShardServer(model_factory, config, clients).serve(stop_event)
