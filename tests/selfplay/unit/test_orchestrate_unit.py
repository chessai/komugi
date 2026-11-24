"""Unit tests for the orchestrator: validation, guards, naming, one replay.

Everything here runs in **one process**. The orchestrator's whole reason to
exist is process-level -- spawning, killing, restarting, unlinking -- and that
behaviour belongs to the KOM-123 companion suite, which owns the spawn walls
and the wedged-worker fixtures; repeating it here would be a slow, flaky copy
that fails for reasons unrelated to what it claims to test.

What is left is what a unit test is good at, and it is the set of decisions that
are cheap to get wrong and silent when wrong: the config's own validation, the
field order of the summary a caller unpacks positionally, the module-level
factory guard of design F18 (which exists precisely so a lambda fails *here*
rather than inside a spawned child), the shard-directory naming that design
F6/F10c makes load-bearing, the resource-tracker discipline of wave-9a review
F4 -- exercised against a real segment, since the tracker is a separate process
and only a real registration can prove the sequence is balanced -- and design
D11's bit-reproducible serial replay.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final, NamedTuple, cast

import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import observation_size
from komugi.selfplay.inference import (
    InferenceConfig,
    attach_untracked,
    create_slots,
)
from komugi.selfplay.memgrid import GridPoint, cgroup_memory_peak, measure_grid
from komugi.selfplay.orchestrate import (
    OrchestratorConfig,
    RunSummary,
    count_rows,
    reregister_and_unlink,
    require_module_level_factory,
    run_selfplay,
    serial_debug,
    worker_shard_dir,
)
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.train.shards import ShardReader

_SHORT_GAME_PLIES: Final[int] = 6
"""Turn cap for the in-test engine subclass -- a whole game in a moment.

The worker has no ply cap of its own by design (F15), so a test that wants a
short game shortens the *engine*, which is the component that owns
termination. Same trick, same reason, as the KOM-122 unit suite.
"""

_RUN_SEED: Final[int] = 20260823


# --------------------------------------------------------------------------
# Stubs -- module-level, because that is what the factory guard demands
# --------------------------------------------------------------------------


class _StubOutput(NamedTuple):
    """KomugiNet-shaped forward output: the four heads, in the net's order."""

    policy_logits: torch.Tensor
    soft_policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class _StubNet(torch.nn.Module):
    """A parameterless net with a constant, exactly-reproducible output.

    Uniform policy logits, so the Gumbel draw -- and therefore the seed -- is
    the only thing deciding a move. Deterministic to the bit, which is what
    lets the replay test attribute any difference between two runs to the
    orchestrator's seed handling rather than to the network.
    """

    def forward(self, x: torch.Tensor) -> _StubOutput:
        """Return one constant evaluation per row of `x`."""
        rows = int(x.shape[0])
        policy = torch.zeros((rows, num_actions()), dtype=torch.float32)
        wdl = torch.zeros((rows, 3), dtype=torch.float32)
        return _StubOutput(policy, policy.clone(), wdl, torch.zeros(rows))


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`."""

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def stub_factory() -> torch.nn.Module:
    """Module-level model factory: what design F18 says a factory must be."""
    net = _StubNet()
    net.eval()
    return net


_MODULE_LEVEL_LAMBDA: Callable[[], torch.nn.Module] = lambda: _StubNet()
"""A lambda at module level: has a module, but no importable qualified name."""


def _closure_factory() -> Callable[[], torch.nn.Module]:
    """Return a factory that closes over a live model -- the F18 anti-pattern."""
    model = _StubNet()

    def factory() -> torch.nn.Module:
        return model

    return factory


def _no_peak() -> int | None:
    """Stand-in for :func:`cgroup_memory_peak` on a machine without one."""
    return None


def _orchestrator(**overrides: Any) -> OrchestratorConfig:
    """A valid config, with the field under test overridden."""
    fields: dict[str, Any] = {
        "run_seed": _RUN_SEED,
        "workers": 2,
        "games_per_worker": 1,
        "shard_root": "/tmp/komugi-unit",
    }
    fields.update(overrides)
    return OrchestratorConfig(**fields)


def _selfplay_config() -> SelfPlayConfig:
    """A two-simulation config: the loop's structure without its cost."""
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=0.5,
        draw_score=0.2,
    )


def _inference_config() -> InferenceConfig:
    """The seam at the engine's real shapes, which the stub net produces."""
    return InferenceConfig(
        in_planes=observation_size()[0], num_actions=num_actions()
    )


# --------------------------------------------------------------------------
# OrchestratorConfig
# --------------------------------------------------------------------------


def test_orchestrator_config_accepts_a_sane_run() -> None:
    """The defaults are the documented ones, and they validate."""
    config = _orchestrator()
    assert config.liveness_timeout_s == 120.0
    assert config.drain_timeout_s == 900.0
    assert config.first_shard_grace_s == 300.0  # its own knob (wave-9c review F3)
    assert config.max_shard_restarts == 2


@pytest.mark.parametrize(
    "override",
    [
        {"run_seed": -1},
        {"workers": 0},
        {"workers": -3},
        {"games_per_worker": 0},
        {"liveness_timeout_s": 0.0},
        {"liveness_timeout_s": -1.0},
        {"liveness_timeout_s": float("nan")},
        {"drain_timeout_s": 0.0},
        {"drain_timeout_s": float("nan")},
        {"first_shard_grace_s": 0.0},
        {"first_shard_grace_s": float("nan")},
        {"max_shard_restarts": -1},
        {"shard_root": ""},
    ],
)
def test_orchestrator_config_rejects_impossible_runs(override: dict[str, Any]) -> None:
    """Every field is validated where the operator can still read the message.

    An unvalidated ``workers=0`` is a run that spawns nothing and reports
    success; an unvalidated timeout is a gate that never fires. Both are
    silent, which is why they are ``ValueError`` at construction.
    """
    with pytest.raises(ValueError):
        _orchestrator(**override)


def test_orchestrator_config_is_frozen() -> None:
    """Config objects are values: a run cannot be re-aimed halfway through."""
    config = _orchestrator()
    with pytest.raises(AttributeError):
        setattr(config, "workers", 4)


# --------------------------------------------------------------------------
# RunSummary
# --------------------------------------------------------------------------


def test_run_summary_field_order_is_the_published_one() -> None:
    """Positional unpacking is part of the API, so the order is pinned here."""
    assert RunSummary._fields == (
        "games_completed",
        "rows_written",
        "shard_dirs",
        "restarts",
        "worker_incarnations",
    )


def test_run_summary_unpacks_positionally() -> None:
    """A caller may write ``games, rows, dirs, restarts, incarnations = ...``."""
    games, rows, dirs, restarts, incarnations = RunSummary(
        games_completed=4,
        rows_written=97,
        shard_dirs=("a", "b"),
        restarts=1,
        worker_incarnations=(1, 1),
    )
    assert (games, rows, dirs, restarts, incarnations) == (4, 97, ("a", "b"), 1, (1, 1))


def test_grid_point_field_order_is_the_published_one() -> None:
    """The design F12 grid is committed with the run; its columns are fixed."""
    assert GridPoint._fields == ("workers", "n_full", "peak_bytes")


# --------------------------------------------------------------------------
# The module-level factory guard (design F18)
# --------------------------------------------------------------------------


def test_factory_guard_accepts_a_module_level_function() -> None:
    """The shape design D2 requires: importable by qualified name."""
    require_module_level_factory(stub_factory)


def test_factory_guard_rejects_a_lambda() -> None:
    """A lambda has no importable name, at module level or anywhere else."""
    local_lambda = cast(Callable[[], torch.nn.Module], lambda: _StubNet())
    for factory in (_MODULE_LEVEL_LAMBDA, local_lambda):
        with pytest.raises(TypeError, match="module-level"):
            require_module_level_factory(factory)


def test_factory_guard_rejects_a_closure() -> None:
    """A closure over a live model is exactly what design F18 forbids."""
    with pytest.raises(TypeError, match="module-level") as excinfo:
        require_module_level_factory(_closure_factory())
    assert "<locals>" in str(excinfo.value)
    assert "D2/F18" in str(excinfo.value)


def test_run_selfplay_rejects_a_closure_before_spawning_anything(
    tmp_path: Path,
) -> None:
    """The guard fires at the call site, not in a child's pickling traceback."""
    with pytest.raises(TypeError, match="module-level"):
        run_selfplay(
            _closure_factory(),
            _orchestrator(shard_root=str(tmp_path)),
            _selfplay_config(),
            _inference_config(),
        )
    assert list(tmp_path.iterdir()) == []


def test_run_selfplay_refuses_multiple_shards(tmp_path: Path) -> None:
    """P0a is one shard; ``shards > 1`` needs the P1 dynamic registration."""
    with pytest.raises(NotImplementedError, match="P1"):
        run_selfplay(
            stub_factory,
            _orchestrator(shard_root=str(tmp_path)),
            _selfplay_config(),
            InferenceConfig(
                in_planes=observation_size()[0], num_actions=num_actions(), shards=2
            ),
        )
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# Shard directory naming (design D10/F6/F10c)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("worker_id", "incarnation", "expected"),
    [(0, 0, "worker_0_r0"), (3, 1, "worker_3_r1"), (11, 27, "worker_11_r27")],
)
def test_worker_shard_dir_is_the_golden_name(
    worker_id: int, incarnation: int, expected: str
) -> None:
    """``worker_<k>_r<i>``, pinned: a zombie must not share its successor's dir.

    The incarnation is in the *name* because design F6's collision is between a
    dying worker's writer and its replacement's -- two writers in one directory
    pick the same shard index and one of them raises ``FileExistsError``.
    """
    assert worker_shard_dir("/runs/iter_7", worker_id, incarnation) == str(
        Path("/runs/iter_7") / expected
    )


def test_worker_shard_dirs_are_unique_per_incarnation() -> None:
    """No two (worker, incarnation) pairs share a directory."""
    names = {
        worker_shard_dir("/runs", worker_id, incarnation)
        for worker_id in range(4)
        for incarnation in range(3)
    }
    assert len(names) == 12


def test_count_rows_of_an_empty_directory_is_zero(tmp_path: Path) -> None:
    """A worker that sealed nothing contributes nothing, and raises nothing."""
    assert count_rows(tmp_path) == 0


# --------------------------------------------------------------------------
# The resource tracker (wave-9a review F4)
# --------------------------------------------------------------------------


def test_reregister_before_unlink_is_balanced_against_a_real_segment(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Create, attach untracked, re-register, unlink -- and no tracker noise.

    The sequence this reproduces is the one wave-9a review F4 describes: the
    creator's registration is keyed by NAME in a single tracker process, so the
    first :func:`~komugi.selfplay.inference.attach_untracked` (which every
    worker and server performs) *deletes* it. A bare ``unlink()`` afterwards
    makes ``SharedMemory`` unregister a name the tracker no longer holds, and
    the tracker daemon answers a missing key with a ``KeyError`` traceback on
    stderr -- noise that reads exactly like a real leak in a CI log.

    Asserted three ways, because the failure is asynchronous and partly out of
    this process: nothing raises, nothing warns, and the segment is really
    gone.
    """
    config = InferenceConfig(in_planes=2, num_actions=8)
    slots = create_slots(config)
    name = slots.shm.name

    # The attach a worker would do -- and the deregistration it takes with it.
    attached = attach_untracked(name)
    attached.close()
    slots.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reregister_and_unlink(slots.shm)
    assert [str(entry.message) for entry in caught] == []

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)

    # The tracker writes its complaints from its own process, so give it a
    # moment before concluding there were none.
    time.sleep(0.2)
    assert "KeyError" not in capfd.readouterr().err


def test_reregister_and_unlink_tolerates_an_already_unlinked_segment() -> None:
    """The postcondition is "the segment is gone", not "we removed it"."""
    slots = create_slots(InferenceConfig(in_planes=2, num_actions=8))
    slots.close()
    reregister_and_unlink(slots.shm)
    reregister_and_unlink(slots.shm)


# --------------------------------------------------------------------------
# The memory grid (design F12)
# --------------------------------------------------------------------------


def test_cgroup_memory_peak_answers_without_raising() -> None:
    """Either a byte count or ``None`` -- never an exception, on any machine.

    ``None`` is a legitimate answer here (cgroup v1, or no memory controller),
    which is exactly why the grid harness raises on it instead of reporting a
    zero: a fabricated peak would be worse evidence than none.
    """
    peak = cgroup_memory_peak()
    assert peak is None or (isinstance(peak, int) and peak >= 0)


def test_measure_grid_refuses_to_fabricate_a_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``memory.peak`` means no grid: callers and tests skip, not guess."""
    monkeypatch.setattr("komugi.selfplay.memgrid.cgroup_memory_peak", _no_peak)
    with pytest.raises(RuntimeError, match="memory.peak"):
        measure_grid(
            stub_factory,
            [(1, 2)],
            _orchestrator(shard_root=str(tmp_path)),
            _selfplay_config(),
            _inference_config(),
        )
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# serial_debug: design D11's bit-reproducible path
# --------------------------------------------------------------------------


def _replay(directory: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Play game 2 of worker 1, incarnation 0, into `directory`."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    return serial_debug(
        stub_factory,
        run_seed=_RUN_SEED,
        worker_id=1,
        incarnation=0,
        game_index=2,
        shard_dir=str(directory),
        selfplay=_selfplay_config(),
        inference=_inference_config(),
    )


def test_serial_debug_replays_one_game_bit_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same seeds twice, into two directories, produce identical rows.

    This is the *whole* replay claim of design D11 -- one worker, an
    in-process evaluator, no shared memory and no batching, so nothing depends
    on batch composition. Production games are promised only up to evaluator
    tolerance and this test deliberately does not pretend otherwise.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    rows = _replay(first, monkeypatch)
    rows_again = _replay(second, monkeypatch)

    assert rows == rows_again > 0
    assert rows <= _SHORT_GAME_PLIES

    left = ShardReader(first, schema=SCHEMA)
    right = ShardReader(second, schema=SCHEMA)
    assert left.skipped == () and right.skipped == ()
    assert len(left) == len(right) == rows
    # The rows are on disk, not merely returned: a debug replay that buffered
    # its tail would reproduce nothing a reader could compare.
    assert count_rows(first) == rows

    for row_left, row_right in zip(left, right):
        for name in SCHEMA.names:
            assert torch.equal(row_left[name], row_right[name]), name


def test_serial_debug_rejects_a_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory rule is enforced on the debug path too, and on purpose.

    ``--serial-debug`` exists to reproduce a *production* game, so accepting a
    factory production would reject would be a trap wearing a reproducibility
    claim (design D2/F18).
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    with pytest.raises(TypeError, match="module-level"):
        serial_debug(
            _closure_factory(),
            run_seed=_RUN_SEED,
            worker_id=0,
            incarnation=0,
            game_index=0,
            shard_dir=str(tmp_path / "never"),
            selfplay=_selfplay_config(),
            inference=_inference_config(),
        )
    assert not (tmp_path / "never").exists()


def test_serial_debug_never_waves_even_when_the_config_could(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3, pinned against ``serial_debug`` ITSELF (b2 wave-2 review F2).

    The worker-side never-waves test drives a replica construction; the
    review's mutant (c) — ``serial_debug`` building a wave evaluator when
    ``max_wave > 1`` — survived it and every other suite. The scenario D3
    exists for is exactly this call: replaying a production WAVE run's config
    through the bit-identical reference path. The reference must stay scalar
    whatever ``max_wave`` says.
    """
    from komugi.search.gumbel import GumbelSearch as _RealSearch

    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    seen: list[object] = []

    def spy(evaluator: Any, config: Any, *, wave_evaluator: Any = None) -> Any:
        seen.append(wave_evaluator)
        return _RealSearch(evaluator, config, wave_evaluator=wave_evaluator)

    monkeypatch.setattr("komugi.selfplay.worker.GumbelSearch", spy)
    wave_capable = InferenceConfig(
        in_planes=observation_size()[0],
        num_actions=num_actions(),
        max_batch=8,
        max_wave=8,
    )
    rows = serial_debug(
        stub_factory,
        run_seed=_RUN_SEED,
        worker_id=1,
        incarnation=0,
        game_index=2,
        shard_dir=str(tmp_path),
        selfplay=_selfplay_config(),
        inference=wave_capable,
    )
    assert rows > 0 and len(seen) > 0
    assert seen == [None] * len(seen), (
        "serial_debug handed the search a wave evaluator: the bit-identical "
        "replay reference must never batch"
    )
