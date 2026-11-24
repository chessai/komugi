"""The P0a gate: two workers, two games each, real engine, straight to a window.

This is the acceptance surface design section 3 names for KOM-123 -- "tiny
end-to-end loop ... producing shards the window/trainer consume THROUGH
ConcatSource with >= 2 worker dirs (F5)" -- and it is the one test in this suite
that exercises every piece at once: the orchestrator spawns a server and two
workers, hands each worker its own ``worker_<k>_r0`` directory (design D10),
plays real games through the real engine against a toy net, and the rows come
back out through :class:`~komugi.selfplay.ConcatSource` into a
:class:`~komugi.train.window.RecencyWindow` as ONE ``IterationRef``.

Why that last hop is the interesting one (design F5).  ``RecencyWindow`` takes
exactly one ``SampleSource`` per iteration -- there is no multi-directory glob
anywhere in ``train/window.py`` -- while D10 gives every worker its own
directory.  The two facts are individually fine and do not fit; ``ConcatSource``
is the join, and a run whose shards cannot be registered as one iteration has
produced data the trainer cannot read.  So the window here is built over
readers spanning **both** worker directories, and a draw that never leaves the
first source would fail.

Everything runs once, in a module-scoped fixture, because a real self-play run
costs real seconds and the assertions below are all views of the same run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from komugi.train.window import IterationRef, RecencyWindow, WindowConfig

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_WORKERS = 2
_GAMES_PER_WORKER = 2
_RUN_SEED = 123123


@dataclass(frozen=True)
class Run:
    """One end-to-end run and everything the assertions below read off it."""

    summary: Any
    root: Path
    elapsed: float
    shm_leaked: set[str]
    children_leaked: dict[int, str]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Play the loop once: 2 workers x 2 games, real engine, toy net."""
    root = Path(tmp_path_factory.mktemp("e2e")) / "shards"
    root.mkdir()

    shm_before = rig.shm_snapshot()
    children_before = set(rig.child_pids())

    outcome = rig.walled_run_selfplay(
        rig.tiny_factory,
        rig.orchestrator_config(
            root,
            run_seed=_RUN_SEED,
            workers=_WORKERS,
            games_per_worker=_GAMES_PER_WORKER,
        ),
        rig.tiny_selfplay_config(workers=_WORKERS),
        rig.inference_config(max_batch=_WORKERS),
        wall_s=rig.E2E_WALL_S,
        label="e2e",
    )
    summary = outcome.unwrap()

    # Snapshot the leftovers before anything in this fixture can clean them up
    # by accident, then clean up for the tests that follow this module.
    leaked_shm = rig.new_shm_survivors(shm_before)
    leaked_children = {
        pid: name
        for pid, name in rig.child_pids().items()
        if pid not in children_before
    }
    rig.kill_stragglers()
    return Run(
        summary=summary,
        root=root,
        elapsed=outcome.elapsed,
        shm_leaked=leaked_shm,
        children_leaked=leaked_children,
    )


def test_the_run_completed_every_game_it_was_asked_for(run: Run) -> None:
    """``games_completed`` and ``rows_written``: the run happened at all."""
    assert run.summary.games_completed >= _WORKERS * _GAMES_PER_WORKER, (
        f"asked for {_WORKERS} workers x {_GAMES_PER_WORKER} games, summary "
        f"reports {run.summary.games_completed} completed"
    )
    assert run.summary.rows_written > 0, "a completed game writes one row per move"
    assert run.summary.restarts == 0, (
        f"nothing wedged, so nothing should have restarted (restarts="
        f"{run.summary.restarts})"
    )
    assert rig.summary_incarnations(run.summary) == [0] * _WORKERS, (
        "no restart means every worker finished on its first incarnation"
    )


def test_each_worker_owns_a_first_incarnation_directory(run: Run) -> None:
    """Design D10's layout, on disk: ``worker_<k>_r0``, one per worker.

    The naming is not cosmetic.  ``ShardWriter`` enforces one writer per
    directory, so the ``_r<incarnation>`` suffix is what lets a respawn write at
    all without colliding with a predecessor that may still be dying.
    """
    dirs = rig.shard_dirs_on_disk(run.root)
    assert len(dirs) >= _WORKERS, (
        f"want at least {_WORKERS} shard directories under {run.root}, found "
        f"{rig.describe(dirs)}"
    )
    parsed = [rig.parse_shard_dir(path) for path in dirs]
    assert {worker for worker, _ in parsed} == set(range(_WORKERS))
    assert all(incarnation == 0 for _, incarnation in parsed), (
        f"an unrestarted run writes only _r0 directories, found {rig.describe(dirs)}"
    )
    for path in dirs:
        assert rig.sealed_shards(path), f"{path.name} sealed no shard"

    reported = rig.summary_dirs(run.summary)
    assert len(reported) >= _WORKERS
    for path in reported:
        assert path.exists(), f"summary names {path}, which is not on disk"
        assert re.fullmatch(rig.SHARD_DIR_RE, path.name), (
            f"summary names {path.name!r}, not a worker_<k>_r<i> directory"
        )


def test_every_row_validates_against_the_schema(run: Run) -> None:
    """``SCHEMA.validate`` on every row of every directory the run produced.

    The schema is the contract between self-play and the trainer, and it is
    checked here on the real rows rather than on a constructed sample: a field
    the worker forgot, a dtype the writer widened, or a shape the engine changed
    is a data-generation event, and this is where it surfaces.
    """
    total = 0
    for directory in rig.summary_dirs(run.summary):
        for row in rig.read_rows(directory):
            sp.SCHEMA.validate(row)
            total += 1
    assert total == run.summary.rows_written, (
        f"summary claims {run.summary.rows_written} rows, the shards hold {total}"
    )


def test_both_search_modes_were_recorded(run: Run) -> None:
    """``prob_fast=0.5`` over a few hundred moves must record 0 *and* 1.

    Design F1 makes ``search_mode`` the key the P0b down-weight-versus-discard
    A/B is stratified on, so a run that silently played one mode only would make
    that experiment impossible while looking perfectly healthy.  The draw is
    seeded off the game's own stream (design D6/D11), so this is deterministic
    for a fixed ``run_seed`` rather than merely likely.
    """
    modes: set[int] = set()
    for directory in rig.summary_dirs(run.summary):
        for row in rig.read_rows(directory):
            modes.add(rig.as_int(row, "search_mode"))
    assert modes == {0, 1}, (
        f"search_mode values across the run: {sorted(modes)}; want both full (0) "
        f"and fast (1) at prob_fast=0.5"
    )


def test_one_iteration_spans_both_worker_dirs_through_concat_source(
    run: Run,
) -> None:
    """Design F5's regression, end to end.

    One ``ConcatSource`` over readers for **all** worker directories, registered
    as ONE ``IterationRef``, is the only shape ``RecencyWindow`` accepts and the
    only shape design D10's per-worker directories can be consumed in.  The
    draw is then checked to reach both sides of the source boundary: a
    concatenation that quietly returned its first source twice would have the
    right length and the wrong data.
    """
    dirs = rig.summary_dirs(run.summary)
    assert len(dirs) >= 2, (
        f"the F5 regression needs >= 2 worker dirs, got {rig.describe(dirs)}"
    )
    readers = [rig.reader_for(directory) for directory in dirs]
    for directory, reader in zip(dirs, readers):
        assert len(reader) > 0, f"{directory.name} holds no samples"

    source = sp.ConcatSource(readers)
    assert len(source) == sum(len(reader) for reader in readers)

    window = RecencyWindow(
        [IterationRef(iteration=0, source=source)],
        WindowConfig(size=1),
        seed=17,
    )
    assert len(window) == len(source)
    assert window.weights == (1.0,)

    boundary = len(readers[0])
    picks = window.draw(128)
    rows = [row for _, row in picks]
    assert any(row < boundary for row in rows) and any(row >= boundary for row in rows), (
        f"128 draws over {len(source)} samples never crossed the source boundary "
        f"at {boundary}: the iteration is not really spanning both worker dirs"
    )

    batch = window.sample(8)
    specs = {field.name: field for field in sp.SCHEMA.fields}
    assert set(batch) == set(sp.SCHEMA.names)
    for name, tensor in batch.items():
        assert tuple(tensor.shape) == (8, *specs[name].shape)
        assert tensor.dtype is specs[name].torch_dtype


def test_the_rows_really_come_from_two_different_workers(run: Run) -> None:
    """Provenance, not just plumbing: two worker ids appear in ``game_id``.

    ``game_id`` packs ``(worker_id, incarnation, game_index)``, so this reads the
    workers straight off the data.  Two directories that both happened to hold
    worker 0's games -- a sticky-assignment bug -- would pass the length and
    ordering checks above and fail here.
    """
    workers: set[int] = set()
    for directory in rig.summary_dirs(run.summary):
        rows = rig.read_rows(directory)
        assert rows, f"{directory.name} holds no rows"
        workers.update(rig.worker_of(rig.as_int(row, "game_id")) for row in rows[:1])
    assert workers == set(range(_WORKERS)), (
        f"game_id decodes to worker ids {sorted(workers)}, want {list(range(_WORKERS))}"
    )


def test_the_run_left_no_shared_memory_segment_behind(run: Run) -> None:
    """Design D2/F9: the orchestrator unlinks every segment it created.

    Snapshotted before and after the run.  Python 3.12 has no
    ``SharedMemory(track=...)``, so the resource tracker is not a backstop here
    (bpo-38119) -- the ``finally`` in the orchestrator is the whole mechanism,
    and a survivor in ``/dev/shm`` is the leak it exists to prevent.
    """
    assert not run.shm_leaked, (
        f"segments still in /dev/shm after the run: {sorted(run.shm_leaked)}"
    )


def test_the_run_left_no_child_process_behind(run: Run) -> None:
    """Graceful shutdown drained and joined: no server, no worker, no corpse."""
    assert not run.children_leaked, (
        f"child processes surviving the run: {run.children_leaked}"
    )
