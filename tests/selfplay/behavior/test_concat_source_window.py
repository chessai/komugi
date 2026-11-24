"""Design F5/D10: one training iteration, several worker directories.

Two facts collide in the reviewed design.  Each worker owns its OWN shard
directory, because ``ShardWriter`` enforces one writer per directory and a
respawned worker gets a fresh ``worker_<k>_r<incarnation>`` subdir (D10).  And
``RecencyWindow`` takes exactly ONE ``SampleSource`` per ``IterationRef`` --
there is no multi-directory glob anywhere in ``train/window.py``, and adding
one was rejected in favour of a source that composes (F5).

``ConcatSource`` is that source, and this module is the regression: an
iteration produced by two workers must reach the trainer as one thing, with the
right length, the right rows, in the right order.  Everything here runs on
**real shards written by real ``SelfPlayWorker`` runs** rather than on a list of
dicts, because the failure F5 describes is a plumbing failure -- the pieces are
individually fine and do not fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from komugi.train.shards import ShardReader
from komugi.train.window import IterationRef, RecencyWindow, WindowConfig

import worker_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "SelfPlayWorker"):
    pytest.skip(
        "komugi.selfplay exports no SelfPlayWorker yet (KOM-022)",
        allow_module_level=True,
    )

_RUN_SEED = 424242


@pytest.fixture(scope="module")
def worker_dirs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Two real per-worker shard directories, as D10 lays an iteration out."""
    root = Path(tmp_path_factory.mktemp("iteration"))
    dirs: list[Path] = []
    for worker_id in (0, 1):
        directory = root / f"worker_{worker_id}"
        rig.run_worker(
            rig.StubBatchEvaluator(seed=worker_id),
            directory,
            worker_id=worker_id,
            run_seed=_RUN_SEED,
            games=1,
        )
        dirs.append(directory)
    return dirs


@pytest.fixture(scope="module")
def readers(worker_dirs: list[Path]) -> list[ShardReader]:
    """One reader per worker directory, schema-checked on the way in."""
    built = [ShardReader(directory, schema=sp.SCHEMA) for directory in worker_dirs]
    for reader, directory in zip(built, worker_dirs):
        assert len(reader) > 0, f"{directory} holds no samples"
        assert not reader.skipped, f"{directory} has unreadable shards"
    return built


@pytest.fixture(scope="module")
def concatenated(readers: list[ShardReader]) -> Any:
    """The thing under test: both workers' shards as one ``SampleSource``."""
    return sp.ConcatSource(readers)


def test_two_workers_really_did_write_two_directories(
    worker_dirs: list[Path], readers: list[ShardReader]
) -> None:
    """The premise: separate directories, separate shards, both non-empty.

    If both workers had shared a directory the second would have collided on a
    shard name (``ShardWriter``'s single-writer guard), so this also pins that
    the suite is exercising D10's layout and not accidentally one dir.
    """
    assert worker_dirs[0] != worker_dirs[1]
    for directory in worker_dirs:
        assert sorted(directory.glob("shard-*.npz")), f"{directory} sealed nothing"
    assert all(len(reader) > 0 for reader in readers)


def test_length_is_the_sum_of_its_sources(
    concatenated: Any, readers: list[ShardReader]
) -> None:
    """The one arithmetic fact a window's sampler depends on."""
    assert len(concatenated) == sum(len(reader) for reader in readers)


def test_every_row_is_readable_and_matches_the_schema(concatenated: Any) -> None:
    """No index in range may raise, and every row is a full sample."""
    names = set(sp.SCHEMA.names)
    specs = {field.name: field for field in sp.SCHEMA.fields}
    for index in range(len(concatenated)):
        row = concatenated[index]
        assert set(row) == names, f"row {index} has fields {sorted(row)}"
        for name, tensor in row.items():
            assert isinstance(tensor, torch.Tensor)
            assert tuple(tensor.shape) == tuple(specs[name].shape), (
                f"row {index} field {name!r} has shape {tuple(tensor.shape)}"
            )
            assert tensor.dtype is specs[name].torch_dtype


def test_rows_come_back_in_source_order(
    concatenated: Any, readers: list[ShardReader]
) -> None:
    """Offset arithmetic across a source boundary, checked at the seam itself.

    The interesting indices are the last row of the first source and the first
    row of the second: an off-by-one in the prefix sums shows up there and
    nowhere else, and it would show up as *plausible training data* rather than
    as an error.
    """
    first, second = readers
    boundary = len(first)
    for index in (0, boundary - 1, boundary, len(concatenated) - 1):
        got = concatenated[index]
        want = first[index] if index < boundary else second[index - boundary]
        for name in sorted(want):
            assert torch.equal(got[name], want[name]), (
                f"row {index} field {name!r} is not the row its source holds "
                f"(boundary at {boundary})"
            )


def test_the_concatenation_is_not_one_source_twice(concatenated: Any) -> None:
    """Both workers' rows are present, not the first worker's counted double.

    Two workers seeded differently play different games, so the first and last
    rows of a correct concatenation differ; a length that happens to be right
    because one source was read twice would not survive this.
    """
    observations = {
        concatenated[index]["observation"].numpy().tobytes()
        for index in (0, len(concatenated) - 1)
    }
    assert len(observations) == 2, (
        "the first and last rows of the concatenation are identical, which is "
        "what returning one source twice looks like"
    )


def test_a_recency_window_accepts_the_concatenation_as_one_iteration(
    concatenated: Any,
) -> None:
    """Design F5, end to end: two worker dirs, one ``IterationRef``, one window.

    ``RecencyWindow`` resolves a draw through the iteration number, so the
    registration API is exactly ``IterationRef(iteration, source)`` -- the
    concatenation has to satisfy ``SampleSource`` structurally (``__len__`` and
    integer ``__getitem__``) with nothing else registered anywhere.
    """
    window = RecencyWindow(
        [IterationRef(iteration=3, source=concatenated)],
        WindowConfig(size=1),
        seed=0,
    )
    assert len(window) == len(concatenated)
    assert window.weights == (1.0,)
    assert [ref.iteration for ref in window.members] == [3]

    picks = window.draw(32)
    assert len(picks) == 32
    assert {iteration for iteration, _ in picks} == {3}
    assert all(0 <= row < len(concatenated) for _, row in picks)


def test_the_window_stacks_batches_out_of_both_workers_shards(
    concatenated: Any,
) -> None:
    """A trainer's actual call: ``sample(batch_size)`` over the joined source."""
    window = RecencyWindow(
        [IterationRef(iteration=0, source=concatenated)],
        WindowConfig(size=1),
        seed=11,
    )
    batch = window.sample(8)
    assert set(batch) == set(sp.SCHEMA.names)
    specs = {field.name: field for field in sp.SCHEMA.fields}
    for name, tensor in batch.items():
        assert tuple(tensor.shape) == (8, *specs[name].shape)
        assert tensor.dtype is specs[name].torch_dtype
    assert torch.isfinite(batch["observation"]).all()
    assert np.allclose(
        batch["policy_target"].sum(dim=1).numpy(), 1.0, atol=1e-4
    )


def test_two_iterations_of_concatenations_keep_their_own_rows(
    readers: list[ShardReader],
) -> None:
    """A window over two concatenations, one per iteration, stays separable.

    This is the shape the orchestrator actually produces once a run has more
    than one iteration: each iteration is a ``ConcatSource`` over that
    iteration's worker dirs, and a draw must attribute to the right one.
    """
    older = sp.ConcatSource([readers[0]])
    newer = sp.ConcatSource([readers[1]])
    window = RecencyWindow(
        [
            IterationRef(iteration=0, source=older),
            IterationRef(iteration=1, source=newer),
        ],
        WindowConfig(size=2, decay=0.8),
        seed=3,
    )
    assert len(window) == len(older) + len(newer)
    assert [ref.iteration for ref in window.members] == [1, 0]
    picks = window.draw(64)
    assert {iteration for iteration, _ in picks} == {0, 1}
    for iteration, row in picks:
        source = newer if iteration == 1 else older
        assert 0 <= row < len(source)


def test_a_single_source_concatenation_is_that_source(
    readers: list[ShardReader],
) -> None:
    """The degenerate case the orchestrator hits with one worker configured."""
    single = sp.ConcatSource([readers[0]])
    assert len(single) == len(readers[0])
    for index in (0, len(single) - 1):
        for name, tensor in single[index].items():
            assert torch.equal(tensor, readers[0][index][name])
