"""Two games per worker process: the knob, the split, the seeds, the wiring.

KOM-040 b1 (design brief rev2, review D1'). The worker spends a measured 57% of
its wall blocked in :meth:`~komugi.selfplay.inference.SlotClient.submit`, and
this is the ticket that hides it: with ``SelfPlayConfig.games_in_flight == 2``
one process plays two games in two threads, each over its **own** segment,
client, writer and game-index range.

What a unit test can prove about that, and therefore what is here:

* the knob validates (only 1 or 2), and 1 -- the default -- provably builds no
  thread at all, which is how "the single-game path is not restructured" stops
  being a promise and starts being a check;
* the budget split, the slot ids and the lane directory names, including that
  the campaign's reader glob really does find both lanes' rows (the whole
  choice of a *suffixed* directory rather than a new naming scheme rests on
  that one fact);
* the seed derivation: two lanes of one worker draw disjoint streams, and so
  does a ``games_in_flight=1`` run of the same ``run_seed``;
* the accounting: a run's ``games_completed`` is split across lanes, not
  multiplied by them.

Everything process-level -- a real interleaved run against a real server, a
SIGKILL with two segments live -- belongs to the KOM-140i behaviour suite,
which owns the spawn walls. Nothing here spawns anything: the two tests that
need a segment create one in this process and release it again.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from multiprocessing import synchronize
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytest
import torch

from komugi.campaign.loop import iteration_source, shard_root
from komugi.config import SelfPlayConfig
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.selfplay import orchestrate as orchestrate_module
from komugi.selfplay import worker as worker_module
from komugi.selfplay.inference import InferenceConfig, create_slots
from komugi.selfplay.orchestrate import (
    OrchestratorConfig,
    RunSummary,
    _Incarnation,
    count_rows,
    lane_game_budget,
    reregister_and_unlink,
    run_selfplay,
    worker_shard_dir,
)
from komugi.selfplay.records import game_seed, move_seed
from komugi.selfplay.worker import (
    LaneSpec,
    SelfPlayWorker,
    WorkerConfig,
    game_id,
    lane_game_index_base,
    lane_slot_id,
    worker_main,
)
from komugi.train.shards import FieldSpec, Schema, ShardWriter

_RUN_SEED: Final[int] = 20260824

_TOY_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="x", dtype="int64", shape=()),
        # Toy in shape, but not in plane-map claim: these rows are read back
        # through ``ConcatSource``, which refuses a row whose
        # ``plane_map_version`` is absent or stale (KOM-002 §4).
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
    )
)
"""A one-int row.

The reader tests below care about *which directories are found*, not about
what is in them, and a production-shaped row carries a 37341-float policy
target. :class:`~komugi.train.shards.ShardReader` adopts the schema of the
first shard it finds when it is given none -- which is exactly how
:func:`~komugi.campaign.loop.iteration_source` builds its readers -- so a toy
row exercises the same glob and the same concatenation.
"""


def _selfplay(**overrides: Any) -> SelfPlayConfig:
    """A two-simulation config, with the field under test overridden."""
    fields: dict[str, Any] = {
        "workers": 1,
        "process_batch_size": 1,
        "gumbel_m_full": 2,
        "gumbel_n_full": 2,
        "gumbel_m_fast": 2,
        "gumbel_n_fast": 2,
    }
    fields.update(overrides)
    return SelfPlayConfig(**fields)


def _write_rows(directory: Path, rows: int) -> None:
    """Seal `rows` toy samples into `directory`, as a lane's writer would."""
    writer = ShardWriter(directory, _TOY_SCHEMA)
    try:
        for value in range(rows):
            writer.append(
                {
                    "x": torch.tensor(value, dtype=torch.int64),
                    "plane_map_version": torch.tensor(
                        PLANE_MAP_VERSION, dtype=torch.int64
                    ),
                }
            )
        writer.flush()
    finally:
        writer.close()


# --------------------------------------------------------------------------
# The knob itself (C1)
# --------------------------------------------------------------------------


def test_games_in_flight_defaults_to_one_game_per_worker() -> None:
    """The default is today's behaviour, which is what makes the knob safe."""
    assert _selfplay().games_in_flight == 1


def test_games_in_flight_accepts_the_interleaved_setting() -> None:
    """Two lanes is the measured setting KOM-040 b1 exists to enable."""
    assert _selfplay(games_in_flight=2).games_in_flight == 2


@pytest.mark.parametrize("lanes", [0, -1, 3, 16])
def test_games_in_flight_rejects_anything_it_has_not_measured(lanes: int) -> None:
    """Loudly, at construction: a silently-clamped knob is a lie about the run.

    Three lanes per process is not "more of the same" -- it is a different
    point on a latency/memory curve nobody has measured -- and zero lanes is a
    worker that plays nothing while reporting success.
    """
    with pytest.raises(ValueError, match="games_in_flight must be 1 or 2"):
        _selfplay(games_in_flight=lanes)


# --------------------------------------------------------------------------
# The budget split (C3)
# --------------------------------------------------------------------------


def test_lane_budget_splits_five_games_three_and_two() -> None:
    """The pinned example: earlier lanes take the remainder."""
    assert lane_game_budget(5, 2, 0) == 3
    assert lane_game_budget(5, 2, 1) == 2


@pytest.mark.parametrize("games", list(range(0, 9)))
def test_lane_budgets_sum_to_the_workers_budget(games: int) -> None:
    """Split, never multiplied: a worker still plays what it was asked for.

    This is the whole of the accounting claim (C8) at its arithmetic root --
    ``RunSummary.games_completed`` credits a cleanly-exited worker with
    ``games_per_worker``, and that stays true only while the lanes partition
    the budget.
    """
    assert sum(lane_game_budget(games, 2, lane) for lane in (0, 1)) == games


def test_lane_budget_of_a_single_lane_is_the_whole_budget() -> None:
    """``games_in_flight=1`` computes what the orchestrator always passed."""
    for games in range(0, 5):
        assert lane_game_budget(games, 1, 0) == games


def test_one_game_over_two_lanes_leaves_the_second_lane_idle() -> None:
    """1 = 1 + 0. An idle lane is honest, not a bug -- see the docstring."""
    assert [lane_game_budget(1, 2, lane) for lane in (0, 1)] == [1, 0]


@pytest.mark.parametrize(("lanes", "lane"), [(2, 2), (2, -1), (0, 0)])
def test_lane_budget_rejects_a_lane_outside_the_fan_out(lanes: int, lane: int) -> None:
    """A lane index nobody launched would silently return a plausible number."""
    with pytest.raises(ValueError):
        lane_game_budget(4, lanes, lane)


# --------------------------------------------------------------------------
# Slot ids and lane directories (C2, C3)
# --------------------------------------------------------------------------


def test_slot_ids_are_unique_across_workers_and_lanes() -> None:
    """The server keys its slot map by this id, so a collision is a crossed wire."""
    ids = [
        lane_slot_id(worker_id, 2, lane) for worker_id in range(8) for lane in (0, 1)
    ]
    assert sorted(ids) == list(range(16))


def test_slot_id_of_a_single_lane_run_is_the_worker_id() -> None:
    """``games_in_flight=1`` numbers the slots exactly as it always did."""
    assert [lane_slot_id(worker_id, 1, 0) for worker_id in range(4)] == [0, 1, 2, 3]


def test_lane_directories_extend_the_existing_name() -> None:
    """``worker_<k>_r<i>_l<lane>``, and ``worker_<k>_r<i>`` without a lane.

    The suffix goes on the end so the campaign's ``worker_*`` glob keeps
    finding the directory: that, and not tidiness, is why the lane is not a
    prefix or a nested directory.
    """
    assert worker_shard_dir("/runs", 3, 1) == str(Path("/runs/worker_3_r1"))
    assert worker_shard_dir("/runs", 3, 1, lane=None) == str(Path("/runs/worker_3_r1"))
    assert worker_shard_dir("/runs", 3, 1, lane=0) == str(Path("/runs/worker_3_r1_l0"))
    assert worker_shard_dir("/runs", 3, 1, lane=1) == str(Path("/runs/worker_3_r1_l1"))


def test_lane_directories_are_unique_and_match_the_reader_glob() -> None:
    """Every (worker, incarnation, lane) gets its own directory, all globbable."""
    names = [
        Path(worker_shard_dir("/runs", worker_id, incarnation, lane=lane)).name
        for worker_id in range(3)
        for incarnation in range(2)
        for lane in range(2)
    ]
    assert len(set(names)) == len(names) == 12
    assert all(fnmatch.fnmatch(name, "worker_*") for name in names)


def test_the_campaign_reader_finds_both_lanes_rows(tmp_path: Path) -> None:
    """C3's pinned decision: no reader change is needed for lane directories.

    Two lanes of one worker, sealing rows into two directories under one
    iteration's shard root. :func:`~komugi.campaign.loop.iteration_source` is
    the campaign's own join -- ``worker_*`` glob into
    :class:`~komugi.selfplay.records.ConcatSource` -- and it must see the sum,
    or half of every interleaved worker's data would be invisible to training
    while every counter still looked right.
    """
    root = shard_root(tmp_path, 0)
    _write_rows(Path(worker_shard_dir(str(root), 0, 0, lane=0)), 3)
    _write_rows(Path(worker_shard_dir(str(root), 0, 0, lane=1)), 2)

    source = iteration_source(tmp_path, 0)
    assert len(source) == 5
    assert sorted(int(source[index]["x"]) for index in range(5)) == [0, 0, 1, 1, 2]


# --------------------------------------------------------------------------
# Seeds (C4)
# --------------------------------------------------------------------------


def _seed_words(worker_id: int, game_index: int) -> tuple[int, ...]:
    """Return four words of the stream game `game_index` would actually draw."""
    seeds = game_seed(_RUN_SEED, worker_id, 0, game_index)
    return tuple(int(word) for word in seeds.generate_state(4, dtype=np.uint32))


def _lane_indices(lane: int, games: int) -> list[int]:
    """Return the game indices lane `lane` plays."""
    base = lane_game_index_base(lane, games)
    return [base + offset for offset in range(games)]


def test_lane_bases_are_disjoint_ranges_including_the_single_game_run() -> None:
    """Lane 0 is offset too, so an interleaved run replays nobody's games."""
    assert lane_game_index_base(0, 3) == 100_000
    assert lane_game_index_base(1, 3) == 200_000
    assert _lane_indices(0, 3) == [100_000, 100_001, 100_002]
    assert _lane_indices(1, 2) == [200_000, 200_001]


def test_two_lanes_of_one_worker_draw_disjoint_seed_streams() -> None:
    """C4: same worker id, same incarnation, and not one shared seed.

    Checked on the *generated words*, not on the indices that produced them:
    the claim being made is about the random streams the games play on, and
    two different spawn keys colliding in ``SeedSequence`` would be invisible
    to an index comparison.
    """
    lane_0 = {_seed_words(2, index) for index in _lane_indices(0, 3)}
    lane_1 = {_seed_words(2, index) for index in _lane_indices(1, 2)}
    single = {_seed_words(2, index) for index in range(5)}

    assert len(lane_0) == 3
    assert len(lane_1) == 2
    assert not lane_0 & lane_1
    assert not (lane_0 | lane_1) & single
    assert len(lane_0 | lane_1 | single) == 10


def test_lane_seed_derivation_is_deterministic_across_re_runs() -> None:
    """A pure function of the spawn key: no clock, no hash, no thread order."""
    first = [_seed_words(2, index) for index in _lane_indices(1, 3)]
    again = [_seed_words(2, index) for index in _lane_indices(1, 3)]
    assert first == again
    assert move_seed(game_seed(_RUN_SEED, 2, 0, 100_000), 0) != move_seed(
        game_seed(_RUN_SEED, 2, 0, 200_000), 0
    )


def test_lane_game_ids_stay_unique_and_inside_the_packing() -> None:
    """``game_id`` packs the same index, so disjoint ranges are disjoint ids."""
    ids = [
        game_id(2, 0, index)
        for lane, games in ((0, 3), (1, 2))
        for index in _lane_indices(lane, games)
    ] + [game_id(2, 0, index) for index in range(5)]
    assert len(set(ids)) == len(ids) == 10


def test_lane_base_refuses_a_budget_that_would_overrun_the_next_lane() -> None:
    """The stride is a bound, and bounds are enforced rather than assumed."""
    assert lane_game_index_base(0, 100_000) == 100_000
    with pytest.raises(ValueError, match="lane's game budget"):
        lane_game_index_base(0, 100_001)
    with pytest.raises(ValueError, match="lane must be >= 0"):
        lane_game_index_base(-1, 1)


def test_worker_plays_its_lanes_index_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run()`` plays ``game_index_base`` onwards -- and ``0`` onwards by default."""
    played: list[int] = []
    writer = ShardWriter(tmp_path, _TOY_SCHEMA)
    try:
        worker = SelfPlayWorker(
            cast(Any, None),
            _selfplay(games_in_flight=2),
            WorkerConfig(
                worker_id=1,
                incarnation=0,
                run_seed=_RUN_SEED,
                games=3,
                shard_dir=str(tmp_path),
                game_index_base=lane_game_index_base(1, 3),
            ),
            writer,
        )
        def _record_zero_rows(index: int) -> int:
            played.append(index)
            return 0

        monkeypatch.setattr(worker, "play_game", _record_zero_rows)
        assert worker.run() == 0
    finally:
        writer.close()
    assert played == [200_000, 200_001, 200_002]


def test_a_default_worker_still_plays_from_game_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No base, no offset: the single-game stream is the one it always was."""
    played: list[int] = []
    writer = ShardWriter(tmp_path, _TOY_SCHEMA)
    try:
        worker = SelfPlayWorker(
            cast(Any, None),
            _selfplay(),
            WorkerConfig(
                worker_id=1,
                incarnation=0,
                run_seed=_RUN_SEED,
                games=3,
                shard_dir=str(tmp_path),
            ),
            writer,
        )
        def _record_zero_rows(index: int) -> int:
            played.append(index)
            return 0

        monkeypatch.setattr(worker, "play_game", _record_zero_rows)
        worker.run()
    finally:
        writer.close()
    assert played == [0, 1, 2]


# --------------------------------------------------------------------------
# worker_main's wiring: no threads by default, one per lane otherwise
# --------------------------------------------------------------------------


class _CountingThreads:
    """A stand-in for the ``threading`` module that records what was built.

    Returns *real* threads, so the interleaved test still runs its lanes for
    real; the point is only that the single-game path never reaches this at
    all.
    """

    def __init__(self) -> None:
        self.created: list[str] = []

    def Thread(self, **kwargs: Any) -> threading.Thread:  # noqa: N802
        """Record the thread's name and build the real thing."""
        self.created.append(str(kwargs.get("name")))
        return threading.Thread(**kwargs)


def _tiny_inference() -> InferenceConfig:
    """A seam at toy shapes: nothing is evaluated, only attached to."""
    return InferenceConfig(in_planes=2, num_actions=8, request_timeout_s=1.0)


def test_the_single_game_path_builds_no_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1, checked rather than promised: ``games_in_flight=1`` is not threaded.

    Zero games, so no evaluation is ever submitted and no server is needed --
    what is under test is the *shape* of the path: one client, one writer,
    inline in the process's main thread, exactly as before this ticket.
    """
    counter = _CountingThreads()
    monkeypatch.setattr(worker_module, "threading", counter)
    inference = _tiny_inference()
    slots = create_slots(inference)
    try:
        worker_main(
            0,
            0,
            _RUN_SEED,
            0,
            str(tmp_path / "worker_0_r0"),
            slots.shm.name,
            _selfplay(),
            inference,
        )
    finally:
        slots.close()
        reregister_and_unlink(slots.shm)
    assert counter.created == []


def test_two_lanes_run_in_two_threads_over_two_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One thread per lane, each with its own segment and its own directory.

    Again with empty budgets: a real interleaved game against a real server is
    the KOM-140i behaviour suite's, and what a unit test can pin is that both
    lanes are constructed, both are joined, and neither borrows the other's
    resources.
    """
    counter = _CountingThreads()
    monkeypatch.setattr(worker_module, "threading", counter)
    inference = _tiny_inference()
    segments = [create_slots(inference), create_slots(inference)]
    dirs = [tmp_path / f"worker_1_r0_l{lane}" for lane in (0, 1)]
    try:
        worker_main(
            1,
            0,
            _RUN_SEED,
            0,
            str(dirs[0]),
            segments[0].shm.name,
            _selfplay(games_in_flight=2),
            inference,
            (
                LaneSpec(
                    lane=1,
                    slot_id=lane_slot_id(1, 2, 1),
                    shm_name=segments[1].shm.name,
                    shard_dir=str(dirs[1]),
                    games=0,
                ),
            ),
        )
    finally:
        for slots in segments:
            slots.close()
            reregister_and_unlink(slots.shm)

    assert counter.created == ["komugi-lane-0-w1", "komugi-lane-1-w1"]
    assert all(directory.is_dir() for directory in dirs)
    assert all(count_rows(directory) == 0 for directory in dirs)


@pytest.mark.parametrize("lanes", [1, 2])
def test_worker_main_refuses_a_lane_count_the_config_did_not_ask_for(
    tmp_path: Path, lanes: int
) -> None:
    """A worker that quietly played one lane of two would halve the run.

    Validated before anything is attached, so the message is the first thing
    the child prints rather than the last.
    """
    extra: tuple[LaneSpec, ...] = (
        LaneSpec(lane=1, slot_id=3, shm_name="never-attached", shard_dir="/x", games=1),
    )
    none: tuple[LaneSpec, ...] = ()
    with pytest.raises(ValueError, match="games_in_flight"):
        worker_main(
            1,
            0,
            _RUN_SEED,
            1,
            str(tmp_path),
            "never-attached",
            _selfplay(games_in_flight=lanes),
            _tiny_inference(),
            extra if lanes == 1 else none,
        )


# --------------------------------------------------------------------------
# RunSummary accounting over lanes (C8)
# --------------------------------------------------------------------------


class _FakeProcess:
    """A child that has already exited cleanly, for the monitor to read."""

    def __init__(self, name: str, exitcode: int | None) -> None:
        self.name = name
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        """Always dead: this stands in for a process that has been reaped."""
        return False

    def join(self, timeout: float | None = None) -> None:
        """Reaping a process that is already gone costs nothing."""

    def kill(self) -> None:
        """SIGKILL against a corpse, which is what ``_teardown`` sends."""


class _FakeEvent:
    """The stop event, reduced to the one method teardown calls."""

    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        """Record the stop signal."""
        self.was_set = True


def test_run_summary_sums_the_lanes_of_every_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C8: rows sum over lanes, games do not double, empty lanes are dropped.

    The processes are stubs -- this is the orchestrator's *arithmetic* under
    test, and a real interleaved run belongs to the behaviour suite -- but the
    directories are real, hold real sealed shards, and are counted by the same
    :func:`~komugi.selfplay.orchestrate.count_rows` a production run uses.
    """
    config = OrchestratorConfig(
        run_seed=_RUN_SEED,
        workers=2,
        games_per_worker=5,
        shard_root=str(tmp_path),
    )
    rows_per_lane = {(0, 0): 3, (0, 1): 2, (1, 0): 4, (1, 1): 0}
    dirs = tuple(
        tuple(
            Path(worker_shard_dir(config.shard_root, worker_id, 0, lane=lane))
            for lane in (0, 1)
        )
        for worker_id in range(2)
    )
    for (worker_id, lane), rows in rows_per_lane.items():
        _write_rows(dirs[worker_id][lane], rows)

    def _fake_launch(*_args: Any, **_kwargs: Any) -> _Incarnation:
        return _Incarnation(
            index=0,
            dirs=dirs,
            segments=[],
            server=cast(BaseProcess, _FakeProcess("server", None)),
            workers=[
                cast(BaseProcess, _FakeProcess("worker-0", 0)),
                cast(BaseProcess, _FakeProcess("worker-1", 0)),
            ],
            stop_event=cast(synchronize.Event, _FakeEvent()),
            started_at=time.monotonic(),
        )

    monkeypatch.setattr("komugi.selfplay.orchestrate._launch", _fake_launch)
    summary = run_selfplay(
        _stub_factory,
        config,
        _selfplay(games_in_flight=2),
        _tiny_inference(),
    )

    assert isinstance(summary, RunSummary)
    # Split, not multiplied: two workers x their (3 + 2)-game budget.
    assert summary.games_completed == 10
    assert summary.rows_written == 9
    # The idle lane sealed nothing, so its directory is bookkeeping and is
    # dropped -- the same rule a wedged incarnation's directory meets.
    assert summary.shard_dirs == (
        str(dirs[0][0]),
        str(dirs[0][1]),
        str(dirs[1][0]),
    )
    assert summary.restarts == 0
    assert summary.worker_incarnations == (0, 0)


def _stub_factory() -> torch.nn.Module:
    """A module-level factory, which is all design F18's guard asks for."""
    return torch.nn.Identity()


# ---------------------------------------------------------------------------
# C6: a lane's failure fails the whole worker, loudly (b1 union review F1)
# ---------------------------------------------------------------------------


def _lane_specs(tmp_path: Path, games: tuple[int, int] = (2, 1)) -> list[LaneSpec]:
    """Two picklable lane specs; the stubs below never touch the resources."""
    return [
        LaneSpec(
            lane=lane,
            slot_id=lane,
            shm_name=f"unused-{lane}",
            shard_dir=str(tmp_path / f"lane{lane}"),
            games=count,
        )
        for lane, count in enumerate(games)
    ]


def _stub_run_lane(
    behaviours: dict[int, BaseException | tuple[int, int]],
) -> Any:
    """A `_run_lane` stand-in: raise-or-record per lane, driven by `behaviours`.

    The real `_run_lane` writes only to its outcome; the stub does exactly
    that, so `_run_lanes`' join/propagate/aggregate logic runs unmodified --
    which is the code under test (b1 union review F1: mutant M3 deleted the
    re-raise and every suite stayed green, because nothing drove this path).
    """

    def run(
        spec: LaneSpec,
        worker_id: int,
        incarnation: int,
        run_seed: int,
        selfplay: SelfPlayConfig,
        inference: InferenceConfig,
        outcome: Any,
    ) -> None:
        behaviour = behaviours[spec.lane]
        if isinstance(behaviour, BaseException):
            outcome.error = behaviour
        else:
            outcome.games, outcome.rows = behaviour

    return run


def test_one_lanes_error_fails_the_worker_with_that_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6: the first lane error re-raises with its own type after both join.

    Without the re-raise a worker whose lane died would exit 0 having played
    half its budget -- "halve the run's throughput and look healthy doing it".
    The review's mutant M3 (`raise failed[0]` -> `pass`) survived 190 tests;
    this test is its killer: the error object itself must come back.
    """
    boom = RuntimeError("lane 0 poisoned")
    monkeypatch.setattr(
        worker_module, "_run_lane", _stub_run_lane({0: boom, 1: (1, 7)})
    )
    with pytest.raises(RuntimeError) as caught:
        worker_module._run_lanes(
            _lane_specs(tmp_path), 3, 0, _RUN_SEED, _selfplay(), InferenceConfig(in_planes=4, num_actions=8)
        )
    assert caught.value is boom, "a different error object was raised"


def test_a_second_lane_failure_surfaces_as_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6: both lanes failing raises the FIRST error and warns about the second.

    The second error is diagnostic an operator would otherwise never see; it
    must neither replace the first (the exit story) nor vanish.
    """
    first = ValueError("lane 0 first")
    second = KeyError("lane 1 second")
    monkeypatch.setattr(
        worker_module, "_run_lane", _stub_run_lane({0: first, 1: second})
    )
    with pytest.warns(RuntimeWarning, match="second self-play lane"):
        with pytest.raises(ValueError) as caught:
            worker_module._run_lanes(
                _lane_specs(tmp_path), 3, 0, _RUN_SEED, _selfplay(), InferenceConfig(in_planes=4, num_actions=8)
            )
    assert caught.value is first


def test_healthy_lanes_sum_their_games_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aggregation half of the same seam: no failure, summed totals."""
    monkeypatch.setattr(
        worker_module, "_run_lane", _stub_run_lane({0: (2, 11), 1: (1, 5)})
    )
    assert worker_module._run_lanes(
        _lane_specs(tmp_path), 3, 0, _RUN_SEED, _selfplay(), InferenceConfig(in_planes=4, num_actions=8)
    ) == (3, 16)


# ---------------------------------------------------------------------------
# C5 (positive half): the liveness gate judges ALL lanes (b1 union review F2)
# ---------------------------------------------------------------------------


def test_a_worker_is_alive_while_any_lane_writes(tmp_path: Path) -> None:
    """One fresh lane keeps the worker alive even when its sibling is stale.

    The gate judges the newest file across ALL lane directories. The review's
    mutant M8 judged lane 0 only -- inert under every other test because the
    rig's gates never fire -- so this drives `_silence_reason` directly: lane
    0 stale far beyond `liveness_timeout_s`, lane 1 fresh, and the worker must
    read as alive.
    """
    import os
    import time as _time

    config = OrchestratorConfig(
        run_seed=_RUN_SEED, workers=1, games_per_worker=1, shard_root=str(tmp_path)
    )
    lane0 = tmp_path / "lane0"
    lane1 = tmp_path / "lane1"
    lane0.mkdir()
    lane1.mkdir()
    now = _time.time()
    stale = lane0 / "shard-000.npz"
    stale.write_bytes(b"x")
    os.utime(stale, (now - 10_000.0, now - 10_000.0))
    fresh = lane1 / "shard-000.npz"
    fresh.write_bytes(b"x")

    assert (
        orchestrate_module._silence_reason([lane0, lane1], 0, 10_000.0, config)
        is None
    ), "a fresh sibling lane must keep the worker alive"

    # And the complement: with BOTH lanes stale the gate must fire.
    os.utime(fresh, (now - 10_000.0, now - 10_000.0))
    reason = orchestrate_module._silence_reason([lane0, lane1], 0, 10_000.0, config)
    assert reason is not None and "worker 0" in reason


def test_never_started_needs_every_lane_empty_past_grace(tmp_path: Path) -> None:
    """The never-started verdict also unions the lanes: any file anywhere counts."""
    config = OrchestratorConfig(
        run_seed=_RUN_SEED, workers=1, games_per_worker=1, shard_root=str(tmp_path)
    )
    lane0 = tmp_path / "lane0"
    lane1 = tmp_path / "lane1"
    lane0.mkdir()
    lane1.mkdir()
    reason = orchestrate_module._silence_reason(
        [lane0, lane1], 0, config.first_shard_grace_s + 60.0, config
    )
    assert reason is not None, "two empty lanes past the grace must read as stuck"
    (lane1 / "shard-000.npz").write_bytes(b"x")
    assert (
        orchestrate_module._silence_reason(
            [lane0, lane1], 0, config.first_shard_grace_s + 60.0, config
        )
        is None
    ), "one lane's first sealed file must clear the never-started verdict"
