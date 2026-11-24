"""Shared rig for the KOM-121 inference-server behavioural suite.

Everything in this module exists so the five test modules next to it can talk
to :mod:`komugi.selfplay` (KOM-021a, implemented concurrently by a different
worker) without each of them re-deriving the same process plumbing.

Three rules the whole suite obeys, and the reasons they are rules:

* **spawn only.**  ``multiprocessing.get_context("spawn")``, never the default
  context.  The production process model (design D2) is spawned workers, and
  fork would silently share torch/CUDA state that spawn does not -- a fork-only
  green is a lie about the thing being tested.
* **every wall is explicit.**  ``pytest-timeout`` is not in the dev shell (see
  ``flake.nix``), so the mandate that "a deadlock can never hang the gate" is
  served by :func:`join_all`: every spawned process is joined inside a wall,
  anything still alive at the wall is SIGKILLed and reported as a failure.  A
  hung server fails this suite in bounded time instead of wedging CI.
* **the module under test is imported lazily.**  :func:`selfplay` goes through
  ``importlib`` rather than a top-level ``import komugi.selfplay`` for two
  reasons: the module does not exist yet (this suite is written red-first
  against the spec), and spawned children re-import it in their own
  interpreter anyway.

The torch stub nets deliberately do NOT come from :mod:`komugi.net`: the
inference server's contract is "a callable returning something with
``policy_logits`` / ``soft_policy_logits`` / ``wdl_logits`` / ``moves_left``",
and pinning that contract with a real KomugiNet would couple this suite to
KOM-005a's shapes and cost seconds per forward pass for nothing.
"""

from __future__ import annotations

import contextlib
import importlib
import queue as queue_mod
import time
from collections.abc import Iterator, Sequence
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

#: Small everywhere, per the KOM-121 spec: the suite is about the transport,
#: not about the net.  ``num_actions`` 81 == 9*9 also lets the stub policy head
#: be a 1x1 convolution read straight off the board.
IN_PLANES = 4
NUM_ACTIONS = 81
BOARD = 9

#: Cells in one observation.  The encoder stub's hash is a sum over these, so
#: the constant is load-bearing for the decode side too.
OBS_CELLS = IN_PLANES * BOARD * BOARD

#: The spawn context.  Module-level so every queue, event and process in the
#: suite comes from the same context (mixing contexts is a class of bug this
#: suite would otherwise have to debug rather than test).
CTX = get_context("spawn")

#: Wall clocks.  Generous on purpose -- a spawned child imports torch from
#: scratch (seconds on a cold page cache), and a flaky wall is worse than no
#: wall.  They bound *hangs*, they are not latency assertions.
SPAWN_WALL_S = 300.0
JOIN_GRACE_S = 10.0
BARRIER_WALL_S = 120.0
POLL_S = 0.05


def selfplay() -> ModuleType:
    """The module under test, imported in whichever interpreter asks."""
    return importlib.import_module("komugi.selfplay")


# ---------------------------------------------------------------------------
# stub nets
# ---------------------------------------------------------------------------


class StubOutput(NamedTuple):
    """KomugiNet-shaped forward output, built here so the suite owns it.

    Field names and order mirror ``komugi.net.NetOutput`` exactly -- that is
    the shape the server is specified to consume.
    """

    policy_logits: torch.Tensor
    soft_policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class EncoderStub(nn.Module):
    """A net whose output *is* a hash of its input.

    For a batch row filled with the constant ``code``:

    * ``policy_logits[n, a] == 10 * code * OBS_CELLS + a`` -- the ramp makes the
      row non-constant, so a transpose or a stale buffer is visible too, and
      every value stays an exactly-representable float32 integer (the largest
      code this suite uses puts the product well under 2**24).
    * ``moves_left[n] == code``.
    * ``wdl_logits[n]`` is a one-hot at ``code % 3``, which survives the
      server's softmax as an argmax -- a third, independent identity channel.

    That is the whole point of the permutation test: a scatter/gather bug that
    hands worker A's response to worker B cannot produce a self-consistent
    triple, because every channel decodes back to the *submitter's* code.
    """

    def forward(self, x: torch.Tensor) -> StubOutput:
        rows = x.shape[0]
        total = x.reshape(rows, -1).sum(dim=1)
        ramp = torch.arange(NUM_ACTIONS, dtype=x.dtype, device=x.device)
        policy = total.unsqueeze(1) * 10.0 + ramp.unsqueeze(0)
        code = torch.round(total / float(OBS_CELLS))
        wdl = F.one_hot(code.long() % 3, 3).to(x.dtype) * 20.0
        return StubOutput(policy, policy.clone(), wdl, code)


class RandomNet(nn.Module):
    """A tiny deterministic conv net: real ``nn`` parameters, fixed weights.

    Parameters (not plain tensor attributes) so ``model.to(device)`` and
    ``model.eval()`` behave the way the server expects; the values come from a
    seeded numpy generator rather than torch's global RNG, so a server process
    and the in-process reference build bit-identical weights without either
    one perturbing global state.
    """

    def __init__(self, seed: int = 20260822) -> None:
        super().__init__()
        self.conv = nn.Conv2d(IN_PLANES, 8, kernel_size=3, padding=1)
        self.policy = nn.Conv2d(8, 1, kernel_size=1)
        self.wdl = nn.Linear(8, 3)
        self.mlh = nn.Linear(8, 1)
        rng = np.random.default_rng(seed)
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name in sorted(params):  # name order, so it is spawn-stable
                param = params[name]
                values = (rng.standard_normal(tuple(param.shape)) * 0.3).astype(
                    np.float32
                )
                param.copy_(torch.from_numpy(values))

    def forward(self, x: torch.Tensor) -> StubOutput:
        hidden = torch.relu(self.conv(x))
        policy = self.policy(hidden).reshape(x.shape[0], NUM_ACTIONS)
        pooled = hidden.mean(dim=(2, 3))
        return StubOutput(
            policy, policy.clone(), self.wdl(pooled), self.mlh(pooled).reshape(-1)
        )


def encoder_factory() -> EncoderStub:
    """Model factory for the server process (module-level == picklable)."""
    model = EncoderStub()
    model.eval()
    return model


def random_factory() -> RandomNet:
    """Model factory for the equivalence corpus, weights fixed by seed."""
    model = RandomNet()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# observations, encode/decode
# ---------------------------------------------------------------------------


def encode_obs(code: int) -> npt.NDArray[np.float32]:
    """An observation that :class:`EncoderStub` hashes back to `code`."""
    return np.full((IN_PLANES, BOARD, BOARD), float(code), dtype=np.float32)


def all_legal() -> npt.NDArray[np.bool_]:
    """Every action legal.

    The permutation and SIGKILL tests use this on purpose: if the server (or a
    future sparse-return mode) masks illegal logits, an all-legal mask keeps
    the identity encoding intact, so the test measures routing and nothing
    else.
    """
    return np.ones(NUM_ACTIONS, dtype=np.bool_)


def as_scalar(value: Any) -> float:
    """First element of anything scalar-ish (0-d array, (1,) slice, float)."""
    return float(np.asarray(value).reshape(-1)[0])


def check_response(response: Any, code: int) -> str | None:
    """Return a description of how `response` fails to decode to `code`.

    Returns ``None`` when it decodes cleanly.  A string rather than an
    exception because the caller is usually a spawned worker reporting through
    a queue, where a traceback is the less useful artefact.
    """
    policy = np.asarray(response.policy_logits, dtype=np.float64)
    if policy.shape != (NUM_ACTIONS,):
        return f"policy_logits shape {policy.shape}, want {(NUM_ACTIONS,)}"

    deramped = policy - np.arange(NUM_ACTIONS, dtype=np.float64)
    spread = float(np.max(deramped) - np.min(deramped))
    # The encoded values are exact float32 integers (< 2**24), so a correct
    # response de-ramps to a spread of exactly 0; the tolerance is a few ulps
    # at that magnitude (ulp(1.8e5) == 0.0156), not a real allowance.  The
    # smallest wrong answer is a whole code apart -- 10*OBS_CELLS == 3240.
    if spread > 0.05:
        return (
            f"policy row is not a single code + ramp: de-ramped spread {spread} "
            f"(head {policy[:4].tolist()})"
        )
    decoded = float(np.mean(deramped)) / (10.0 * OBS_CELLS)
    if abs(decoded - code) > 1e-2:
        return f"policy_logits decode to code {decoded}, submitted {code}"

    moves_left = as_scalar(response.moves_left)
    if abs(moves_left - code) > 1e-2:
        return f"moves_left {moves_left}, submitted code {code}"

    wdl = np.asarray(response.wdl, dtype=np.float64)
    if wdl.shape != (3,):
        return f"wdl shape {wdl.shape}, want (3,)"
    if abs(float(wdl.sum()) - 1.0) > 1e-4:
        return f"wdl does not sum to 1 (got {wdl.tolist()}) -- softmax missing?"
    if int(np.argmax(wdl)) != code % 3:
        return f"wdl argmax {int(np.argmax(wdl))}, want {code % 3} for code {code}"
    return None


# ---------------------------------------------------------------------------
# config / slots
# ---------------------------------------------------------------------------


def make_config(**overrides: Any) -> Any:
    """``InferenceConfig`` at this suite's small shapes."""
    fields: dict[str, Any] = {"in_planes": IN_PLANES, "num_actions": NUM_ACTIONS}
    fields.update(overrides)
    return selfplay().InferenceConfig(**fields)


def _create_slots(config: Any, worker_ids: Sequence[int]) -> list[Any]:
    """Create one real segment per worker, wrapped in the impl's ``SlotHandle``.

    The orchestrator owns segment creation (design D2). The real contract
    (wave-9a review F1, amended by KOM-021b): ``create_slots(config)`` builds
    ONE segment and returns its views; the per-worker handle -- (shm_name,
    worker_id) -- is the orchestrator's to assemble. So per worker id the rig
    creates a segment, records the OS-chosen name, and closes the parent's
    views (client and server re-attach by name in their own processes).

    No event is paired with it any more: with the doorbell protocol the
    request/response counters live INSIDE the segment, so the name is the whole
    signalling channel and there is no cross-process lock for a SIGKILLed
    worker to orphan.
    """
    sp = selfplay()
    handles: list[Any] = []
    for worker_id in worker_ids:
        created = sp.create_slots(config)
        name = created.shm.name
        created.close()
        handles.append(sp.SlotHandle(shm_name=name, worker_id=worker_id))
    return handles


def unlink_segment(name: str) -> None:
    """Orchestrator-side removal: the creator unlinks, nobody else (D2)."""
    try:
        segment = SharedMemory(name=name)
    except FileNotFoundError:
        return
    segment.close()
    segment.unlink()


@contextlib.contextmanager
def slots(config: Any, worker_ids: Sequence[int]) -> Iterator[list[Any]]:
    """Create one slot per worker id, unlinking every segment on the way out."""
    handles = _create_slots(config, worker_ids)
    try:
        yield handles
    finally:
        for handle in handles:
            unlink_segment(handle.shm_name)


def make_client(config: Any, handle: Any) -> Any:
    """``SlotClient`` built from a handle -- the worker side of one slot."""
    return selfplay().SlotClient(handle.shm_name, handle.worker_id, config)


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------


def server_main(
    factory: Any,
    config: Any,
    handles: Sequence[Any],
    stop_event: Any,
) -> None:
    """Spawned-server entry point: build the server, serve until stopped."""
    server = selfplay().ShardServer(factory, config, list(handles))
    server.serve(stop_event)


def spawn(target: Any, *args: Any, name: str | None = None) -> BaseProcess:
    """Start a spawn-context process running `target`."""
    proc = CTX.Process(target=target, args=args, name=name)
    proc.start()
    return proc


def drain(source: Any) -> list[Any]:
    """Everything currently on a queue, without blocking."""
    items: list[Any] = []
    while True:
        try:
            items.append(source.get_nowait())
        except queue_mod.Empty:
            return items


def collect(source: Any, count: int, wall_s: float) -> list[Any]:
    """Block for exactly `count` items, or fewer if `wall_s` runs out."""
    items: list[Any] = []
    deadline = time.monotonic() + wall_s
    while len(items) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return items
        try:
            items.append(source.get(timeout=min(remaining, 1.0)))
        except queue_mod.Empty:
            continue
    return items


def join_all(
    procs: Sequence[BaseProcess], wall_s: float, errors: Any = None
) -> list[str]:
    """Join every process inside one wall; return the failures found.

    Drains `errors` while waiting, because a worker that has put a message on a
    queue does not exit until the feeder thread has flushed it -- joining first
    and reading second is the classic way to turn a reported failure into a
    hang.  Anything still alive at the wall is killed and reported: this is the
    mechanism that keeps a deadlocked server from wedging ``ci/check.sh``.
    """
    failures: list[str] = []
    deadline = time.monotonic() + wall_s
    for proc in procs:
        while proc.is_alive() and time.monotonic() < deadline:
            if errors is not None:
                failures.extend(str(item) for item in drain(errors))
            proc.join(POLL_S)
    if errors is not None:
        failures.extend(str(item) for item in drain(errors))

    stuck = [proc for proc in procs if proc.is_alive()]
    for proc in stuck:
        proc.kill()
        proc.join(JOIN_GRACE_S)
    if stuck:
        failures.append(
            f"still alive at the {wall_s}s wall (killed): "
            f"{[proc.name for proc in stuck]}"
        )
    failures.extend(
        f"{proc.name} exited {proc.exitcode}"
        for proc in procs
        if proc.exitcode not in (0, None)
    )
    return failures


def stop_server(server: BaseProcess, stop_event: Any) -> list[str]:
    """Signal the server to stop and join it inside a wall."""
    stop_event.set()
    return join_all([server], JOIN_GRACE_S * 3)
