"""Shared rig for the KOM-122 self-play worker behavioural suite.

The worker (KOM-022: ``selfplay/worker.py``, ``selfplay/adapter.py``,
``selfplay/records.py``) is being implemented concurrently by a different
worker, so everything here is written against the pinned contract and nothing
else.  The five test modules next to this one each carry their own
``pytest.importorskip`` + ``hasattr(sp, "SelfPlayWorker")`` gate; this module
carries no gate at all, on purpose -- it is imported by *spawned child
processes* (design D7's shape gate, the PYTHONHASHSEED subprocesses), and a
module-level ``pytest.skip`` raises ``Skipped`` in an interpreter that has no
pytest session to catch it, which would turn every child's exit code into a
lie.

Rules this rig obeys, and why each is a rule:

* **Never import :mod:`komugi.net`.**  The worker's contract is "something
  that returns a ``NetEval``"; pinning it with the real net would couple this
  suite to KOM-005a's shapes and pay seconds per forward for nothing.  The
  stubs below are module-level (therefore picklable, therefore spawnable) and
  deterministic (therefore replayable, which
  ``test_seed_discipline.py`` depends on).
* **Real engine shapes.**  The worker plays the real ``komugi.engine.game.Game``,
  so ``A`` is the engine's 37341 and the observation is its 128 planes.  Both
  are read off the engine rather than written down, so a shape change is a
  failure over there and not a silent divergence here.
* **Tiny budgets everywhere.**  ``gumbel_n_full=4`` / ``gumbel_n_fast=2`` per
  the KOM-122 spec: this suite is about perspective, seeds, shapes and shard
  plumbing, and a realistic 256-simulation search would only make it slow.
* **Every wall is explicit.**  ``pytest-timeout`` is not in the dev shell, so
  each spawned process is joined inside :func:`join_all` and each subprocess
  runs under :func:`run_python`'s ``timeout=``.  A wedged child fails the gate
  in bounded time instead of hanging ``ci/check.sh``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.game import Game
from komugi.engine.observation import observation_size
from komugi.train.shards import ShardReader, ShardWriter

# ---------------------------------------------------------------------------
# shapes and small constants
# ---------------------------------------------------------------------------

IN_PLANES, BOARD_H, BOARD_W = observation_size()
"""The real encoder's observation shape -- (128, 9, 9) today."""

NUM_ACTIONS: int = ACTION_SIZE
"""The real engine's flat action space, ``A``."""

MAX_TURNS: int = Game.max_turns()
"""The engine's own turn cap -- read from the engine, not restated as a
literal here.  The worker adds no second cap (design D6/F15), and
``MLH_SCALE`` is ``1 / MAX_TURNS``."""

DRAW_SCORE: float = 0.20
"""DESIGN.html section 3.3's contempt, and ``SelfPlayConfig``'s default."""

SAMPLES_PER_SHARD: int = 32
"""Small, so one game seals several shards and the reader's cross-shard index
arithmetic is exercised rather than assumed."""

CTX = get_context("spawn")
"""Spawn only: the production process model is spawned workers (design D2), and
a fork-only green is a lie about the thing being tested."""

SPAWN_WALL_S: float = 420.0
"""Wall for a spawned worker/server pair.  Generous: a child imports torch from
scratch and then plays a real game.  It bounds *hangs*; it is not a latency
assertion."""

SUBPROCESS_WALL_S: float = 300.0
"""Wall for a ``python -c`` child."""

JOIN_GRACE_S: float = 10.0
POLL_S: float = 0.05

WIN: tuple[float, float, float] = (1.0, 0.0, 0.0)
DRAW: tuple[float, float, float] = (0.0, 1.0, 0.0)
LOSS: tuple[float, float, float] = (0.0, 0.0, 1.0)

HERE: Path = Path(__file__).resolve().parent
"""This directory, so a child interpreter can ``import worker_rig``."""


def selfplay() -> ModuleType:
    """The module under test, imported in whichever interpreter asks.

    ``importlib`` rather than a top-level import because the worker half of
    this package does not exist yet (this suite is red-first) and because
    spawned children re-import it in their own interpreter anyway.
    """
    return importlib.import_module("komugi.selfplay")


def has_worker() -> bool:
    """Whether KOM-022 has landed, for a child process that must not skip."""
    try:
        return hasattr(selfplay(), "SelfPlayWorker")
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# stub evaluators -- never komugi.net
# ---------------------------------------------------------------------------

MISSHAPED_WDL: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
"""Design D7's contract violation: four win/draw/loss components.

Chosen because it is invisible to every check *except* the jaxtyping seam --
the search's own ``(A,)`` logits ``ValueError`` cannot mask it, and a
four-component vector still sums to 1.0 like a distribution should.
"""


class StubBatchEvaluator:
    """A deterministic evaluator, in both shapes the worker might ask for.

    Two entry points on purpose.  The seam's :class:`~komugi.selfplay.NetEval`
    producer is a ``BatchEvaluator`` (``evaluate`` over ``(N, ...)``, design
    D1), while the thing ``worker_main`` actually hands to ``make_evaluator``
    is a ``SlotClient`` (``submit`` over one position).  A stub that answers
    both is a stub that cannot fail this suite for choosing the wrong one, and
    the two paths share one implementation so they cannot disagree.

    **Deterministic in ``obs`` alone.**  The prior is a fixed random vector
    plus a bounded, position-dependent tilt; no call counter, no clock, no
    global RNG.  That is what makes ``test_seed_discipline``'s byte-identical
    replay claim a statement about the *worker's* seeding rather than about
    this class.

    ``wdl`` defaults to ``(0.5, 0, 0.5)`` -- a scalar utility of **exactly
    zero** under :func:`~komugi.selfplay.wdl_utility` for any ``draw_score``.
    That is deliberate: with every value identical the completed-Q vector the
    Gumbel target is built from is constant, so ``softmax(logits + sigma(q))``
    collapses to the masked softmax of the prior and the recorded ``policy_kl``
    must be 0.  It is the "construct one such move deliberately" half of the
    KOM-122 spec, and it costs nothing else.
    """

    def __init__(
        self,
        seed: int = 0,
        wdl: Sequence[float] = (0.5, 0.0, 0.5),
        moves_left: float = 8.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._base = rng.standard_normal(NUM_ACTIONS).astype(np.float32)
        self._tilt = (rng.standard_normal(NUM_ACTIONS) * 0.25).astype(np.float32)
        self._wdl = np.asarray(wdl, dtype=np.float32)
        self._moves_left = float(moves_left)
        self.calls = 0

    def _rows(self, obs: npt.NDArray[np.float32]) -> Any:
        """Build one :class:`NetEval` for a ``(N, in_planes, 9, 9)`` batch."""
        rows = int(obs.shape[0])
        self.calls += rows
        # A bounded function of the position: the raw sum is huge and would
        # swamp the base logits, so it is folded into [0, 1).
        keys = np.asarray(obs, dtype=np.float64).reshape(rows, -1).sum(axis=1)
        mix = (np.remainder(keys, 8.0) / 8.0).astype(np.float32)
        policy = (self._base[None, :] + mix[:, None] * self._tilt[None, :]).astype(
            np.float32
        )
        wdl = np.repeat(self._wdl[None, :], rows, axis=0).astype(np.float32)
        moves_left = np.full((rows,), self._moves_left, dtype=np.float32)
        return selfplay().NetEval(
            policy_logits=policy, wdl=wdl, moves_left=moves_left
        )

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> Any:
        """``BatchEvaluator.evaluate``: a batch in, a batched ``NetEval`` out."""
        del legal  # carried across the seam, applied by the search (D1/D8)
        return self._rows(np.asarray(obs, dtype=np.float32))

    def submit(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> Any:
        """``SlotClient.submit``: one position in, one unbatched ``NetEval`` out."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        batched = self._rows(array if array.ndim == 4 else array[None, ...])
        return selfplay().NetEval(
            policy_logits=np.asarray(batched.policy_logits[0], dtype=np.float32),
            wdl=np.asarray(batched.wdl[0], dtype=np.float32),
            moves_left=np.asarray(batched.moves_left[0], dtype=np.float32),
        )


def misshaped_evaluator(seed: int = 3) -> StubBatchEvaluator:
    """The D7 stub: a ``NetEval`` whose ``wdl`` has four components."""
    return StubBatchEvaluator(seed=seed, wdl=MISSHAPED_WDL)


# ---------------------------------------------------------------------------
# a torch net for the real inference server -- KomugiNet-shaped, not KomugiNet
# ---------------------------------------------------------------------------


class StubOutput(NamedTuple):
    """The three fields the **inference seam** reads, owned by this suite.

    Deliberately *not* a mirror of ``komugi.net.NetOutput``, which carries
    eight fields since the staged heads landed. The serving path reads this
    object through ``inference._NetOutputLike``, a structural ``Protocol`` over
    ``policy_logits``, ``wdl_logits`` and ``moves_left``, so a four-field stub
    satisfies it unchanged -- and growing three fakes to eight fields to track
    a training-time contract they never cross would be work for nothing.
    ``soft_policy_logits`` is here because it predates that Protocol, not
    because the seam wants it.
    """

    policy_logits: torch.Tensor
    soft_policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class TinyNet(nn.Module):
    """A net at the engine's real shapes that costs almost nothing to run.

    One parameter, so ``next(model.parameters())`` gives the server a device to
    follow (:class:`~komugi.selfplay.InProcessEvaluator` places tensors on the
    model's own device and names none itself).  The policy is a fixed ramp plus
    a pooled function of the observation: deterministic, non-constant (so a
    stale or transposed buffer would show), and one broadcast rather than a
    37341-wide matmul.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x: torch.Tensor) -> StubOutput:
        """Return a KomugiNet-shaped output for a ``(N, 128, 9, 9)`` batch."""
        rows = x.shape[0]
        pooled = x.reshape(rows, -1).mean(dim=1, keepdim=True)
        ramp = torch.arange(NUM_ACTIONS, dtype=x.dtype, device=x.device) * 1e-4
        policy = ramp.unsqueeze(0) + pooled * self.scale
        wdl = torch.cat([pooled * 0.1, pooled * 0.0, pooled * -0.1], dim=1)
        return StubOutput(policy, policy.clone(), wdl, pooled.reshape(-1))


def tiny_factory() -> TinyNet:
    """Model factory for the server process (module-level == picklable)."""
    model = TinyNet()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def tiny_selfplay_config(**overrides: Any) -> Any:
    """``SelfPlayConfig`` at the KOM-122 budgets: n_full=4, n_fast=2."""
    from komugi.config import SelfPlayConfig

    fields: dict[str, Any] = {
        "workers": 1,
        "process_batch_size": 1,
        "gumbel_m_full": 4,
        "gumbel_n_full": 4,
        "gumbel_m_fast": 2,
        "gumbel_n_fast": 2,
        "draw_score": DRAW_SCORE,
    }
    fields.update(overrides)
    return SelfPlayConfig(**fields)


def inference_config(**overrides: Any) -> Any:
    """``InferenceConfig`` at the engine's real shapes."""
    fields: dict[str, Any] = {
        "in_planes": IN_PLANES,
        "num_actions": NUM_ACTIONS,
        "max_batch": 4,
        "batch_timeout_ms": 1.0,
    }
    fields.update(overrides)
    return selfplay().InferenceConfig(**fields)


# ---------------------------------------------------------------------------
# building and running a worker
# ---------------------------------------------------------------------------


def build_worker(
    stub: Any,
    shard_dir: Path,
    *,
    worker_id: int = 0,
    incarnation: int = 0,
    run_seed: int = 7,
    games: int = 1,
    selfplay_config: Any = None,
    writer: Any = None,
) -> tuple[Any, Any]:
    """Return ``(SelfPlayWorker, ShardWriter)`` wired the way ``worker_main`` is.

    The evaluator goes through ``make_evaluator(client, draw_score)`` rather
    than being handed to the worker raw, because that adapter is where design
    D1's wdl->utility derivation and design D7's ``@jaxtyped`` seam live -- the
    two things this suite is here to pin.
    """
    sp = selfplay()
    config = selfplay_config if selfplay_config is not None else tiny_selfplay_config()
    shard_writer = (
        writer
        if writer is not None
        else ShardWriter(shard_dir, sp.SCHEMA, samples_per_shard=SAMPLES_PER_SHARD)
    )
    worker_config = sp.WorkerConfig(
        worker_id=worker_id,
        incarnation=incarnation,
        run_seed=run_seed,
        games=games,
        shard_dir=shard_dir,
    )
    # The stub goes in RAW: SelfPlayWorker's pinned contract takes the batch
    # client and wraps it through make_evaluator INTERNALLY -- pre-wrapping
    # here double-wrapped the evaluator and meant these fixtures had never
    # actually run against the worker (wave-9b review F2).
    worker = sp.SelfPlayWorker(stub, config, worker_config, shard_writer)
    return worker, shard_writer


def run_worker(
    stub: Any, shard_dir: Path, **kwargs: Any
) -> None:
    """Build a worker, ``run()`` it, and seal the tail whatever happened."""
    worker, writer = build_worker(stub, shard_dir, **kwargs)
    try:
        worker.run()
    finally:
        writer.close()


def misshape_main() -> None:
    """Design D7's runner: one game against a ``wdl`` of the wrong width.

    Module-level so that ``python -c "import worker_rig; ..."`` can reach it
    with an environment of the test's choosing -- ``KOMUGI_TYPECHECK`` is read
    when :mod:`komugi.shapecheck` is first imported, so it has to be in the
    child's environment before anything else happens, which rules out setting
    it from inside a test.

    Nothing is caught here: the *point* is that the exception escapes and the
    process dies nonzero with beartype/jaxtyping in its traceback.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_worker(misshaped_evaluator(), Path(tmp), games=1)


def sane_main() -> None:
    """The same runner with a well-shaped stub -- design F7's other control.

    Run under ``KOMUGI_TYPECHECK=1`` it proves the armed seam passes traffic it
    should pass, so the misshape child's nonzero exit is attributable to the
    ``(N, 4)`` wdl and not to shape checking being broken in general.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_worker(StubBatchEvaluator(seed=1), Path(tmp), games=1)


# ---------------------------------------------------------------------------
# reading rows back
# ---------------------------------------------------------------------------


def read_rows(directory: Path) -> list[dict[str, torch.Tensor]]:
    """Every row in a shard directory, in write order, schema-validated."""
    reader = ShardReader(directory, schema=selfplay().SCHEMA)
    assert not reader.skipped, (
        f"shard directory {directory} has unreadable shards: "
        f"{[(s.path.name, s.reason) for s in reader.skipped]}"
    )
    return list(reader)


def as_int(row: dict[str, torch.Tensor], name: str) -> int:
    """One scalar int64 field of a row."""
    return int(row[name].item())


def as_float(row: dict[str, torch.Tensor], name: str) -> float:
    """One scalar float32 field of a row."""
    return float(row[name].item())


def as_array(row: dict[str, torch.Tensor], name: str) -> npt.NDArray[np.float64]:
    """One vector field of a row, widened for comparison arithmetic."""
    return np.asarray(row[name].numpy(), dtype=np.float64)


def by_game(
    rows: Iterable[dict[str, torch.Tensor]],
) -> dict[int, list[dict[str, torch.Tensor]]]:
    """Group rows by ``game_id``, preserving write order within each game."""
    games: dict[int, list[dict[str, torch.Tensor]]] = {}
    for row in rows:
        games.setdefault(as_int(row, "game_id"), []).append(row)
    return games


def mover_target(
    absolute: Sequence[int], mover: int
) -> tuple[float, float, float]:
    """Flip an ABSOLUTE ``(white, black, draw)`` outcome into `mover`'s frame.

    The one line design D4 makes the adapter responsible for, written here
    independently so the suite has an oracle rather than a mirror.  A draw is
    ``[0, 1, 0]`` in every frame, which is exactly why the drawn game cannot
    catch the sign bug and the decisive one must.
    """
    if int(absolute[2]) == 1:
        return DRAW
    winner = 0 if int(absolute[0]) == 1 else 1
    return WIN if winner == mover else LOSS


def absolute_from_row(row: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Invert :func:`mover_target` for one row: the absolute outcome it implies.

    Used to derive a single candidate outcome from a game's first row, which
    every other row is then held to.  An implementation that forgot the flip
    agrees with itself on row 0 and contradicts itself on the first row played
    by the other side -- which is the failure this suite exists to produce.
    """
    wdl = as_array(row, "wdl_target")
    mover = as_int(row, "to_play")
    slot = int(np.argmax(wdl))
    if slot == 1:
        return (0, 0, 1)
    winner = mover if slot == 0 else 1 - mover
    return (1, 0, 0) if winner == 0 else (0, 1, 0)


# ---------------------------------------------------------------------------
# scripted games through the real engine
# ---------------------------------------------------------------------------


def random_playout(
    seed: int,
    *,
    max_plies: int = MAX_TURNS,
    prefer_terminal: bool = False,
    candidates: int = 8,
) -> tuple[Game, list[int]]:
    """Play a real ``Game`` by seeded legal-move shuffle; return it and its movers.

    ``prefer_terminal`` samples up to `candidates` legal actions per ply on
    clones and plays one that ends the game decisively if it finds one -- a
    one-ply mate finder, which is what makes a *decisive* game something this
    suite can ask for rather than wait for.  Without it the shuffle is plain.

    The returned mover list is the engine's own ``player`` at each ply, in
    order, because the KOM-122 spec is explicit that parity is not the mover:
    a drafting position need not alternate, so the engine is the authority and
    this list is what the assertions compare against.
    """
    game = Game()
    rng = np.random.default_rng(seed)
    movers: list[int] = []
    for _ in range(max_plies):
        # Termination is the engine's call and nobody else's (design D6): a
        # decided game reports it in win_state, and an undecided one always has
        # a legal move. One valid_moves() per ply, because it is the expensive
        # one.
        if int(np.asarray(game.win_state()).sum()) > 0:
            break
        legal = np.flatnonzero(np.asarray(game.valid_moves(), dtype=np.uint8))
        if legal.size == 0:
            break
        order = rng.permutation(legal)
        action = int(order[0])
        if prefer_terminal:
            for candidate in order[:candidates]:
                probe = game.clone()
                probe.play_action(int(candidate))
                outcome = np.asarray(probe.win_state())
                if int(outcome[0]) == 1 or int(outcome[1]) == 1:
                    action = int(candidate)
                    break
        movers.append(int(game.player))
        game.play_action(action)
    return game, movers


def capped_draw_game(seed: int = 5, plies: int = 8) -> tuple[Game, list[int]]:
    """A real game standing at the engine's adjudicated turn-cap draw.

    Plays `plies` real moves (real board, real repetition keys), then sets the
    ply counter to the cap.  Reaching the cap by play would take hundreds of
    thousands of legal-move generations to test the *walk* rather than the
    adjudication, so the counter is set directly -- the same fixture technique
    ``tests/rules/test_repetition_planes.py`` uses, and legitimate for the same
    reason: the turn cap is enforced against ``Game._turns`` and nothing else
    (``game.py``'s ``win_state``), so a game whose counter reads the cap *is* a
    turn-cap draw as far as every rule in the engine is concerned.
    """
    game, movers = random_playout(seed, max_plies=plies)
    game._turns = MAX_TURNS
    return game, movers


# ---------------------------------------------------------------------------
# processes and walls
# ---------------------------------------------------------------------------


def spawn(target: Any, *args: Any, name: str | None = None) -> BaseProcess:
    """Start a spawn-context process running `target`."""
    proc = CTX.Process(target=target, args=args, name=name)
    proc.start()
    return proc


def join_all(procs: Sequence[BaseProcess], wall_s: float) -> list[str]:
    """Join every process inside one wall; return the failures found.

    Anything still alive at the wall is SIGKILLed and reported.  This is the
    mechanism that keeps a deadlocked worker or server from wedging
    ``ci/check.sh``, since ``pytest-timeout`` is not in the dev shell.
    """
    failures: list[str] = []
    deadline = time.monotonic() + wall_s
    for proc in procs:
        while proc.is_alive() and time.monotonic() < deadline:
            proc.join(POLL_S)
    stuck = [proc for proc in procs if proc.is_alive()]
    for proc in stuck:
        proc.kill()
        proc.join(JOIN_GRACE_S)
    if stuck:
        failures.append(
            f"still alive at the {wall_s}s wall (killed): "
            f"{[proc.name for proc in stuck]}"
        )
    return failures


def child_env(**overrides: str) -> dict[str, str]:
    """This process's environment, plus `overrides`, plus the import paths.

    ``src`` is derived from the installed :mod:`komugi` rather than assumed to
    be on ``PYTHONPATH`` already (``ci/check.sh`` exports it, an ad-hoc ``pytest``
    invocation may not), and this directory is added so the child can
    ``import worker_rig``.
    """
    import komugi

    package = komugi.__file__
    assert package is not None, "komugi has no __file__ to derive src/ from"
    src = str(Path(package).resolve().parent.parent)
    env = dict(os.environ)
    paths = [str(HERE), src]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(overrides)
    return env


def run_python(
    code: str, *, env: dict[str, str] | None = None, wall_s: float = SUBPROCESS_WALL_S
) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter under a wall, capturing both streams.

    ``check=False``: several callers are *asking* for a nonzero exit.  The
    ``timeout`` is the wall -- a child that hangs raises
    ``subprocess.TimeoutExpired`` here instead of stalling the gate.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env if env is not None else child_env(),
        capture_output=True,
        text=True,
        timeout=wall_s,
        check=False,
    )


BEARTYPE_MARKERS: tuple[str, ...] = ("beartype", "jaxtyping")
"""Substrings that identify a shapecheck-seam failure in a traceback.

Design F7's requirement is that the *seam* fired, not that something happened
to fail: the control run at ``KOMUGI_TYPECHECK=0`` asserts the absence of these
same markers, so a stub that merely crashes for an unrelated reason cannot pass
for a shape check.
"""


def names_the_seam(text: str) -> bool:
    """Whether `text` names beartype or jaxtyping (case-insensitively)."""
    lowered = text.lower()
    return any(marker in lowered for marker in BEARTYPE_MARKERS)
