"""What a batch is *made of*, counted exactly, not merely "something came back".

``tests/train/test_window.py`` pins the schedule's arithmetic and the draw
frequencies of :meth:`~komugi.train.window.RecencyWindow.draw`. This module
asks the question one level down, where a real replay window is wrong in a way
arithmetic cannot see: **the batch handed to the optimiser has to be the rows
the draw named** -- from the iteration it named, at the row it named -- and the
window has to hold exactly the iterations it claims to hold as the store grows
under it.

Every sample here is *individually identifiable*: each row carries its own
``iteration`` and ``row`` number as int64 fields, written through a real
:class:`~komugi.train.shards.ShardWriter` (two rows per shard, so an iteration
is several shards) and read back through a real
:class:`~komugi.train.shards.ShardReader`. A stacked batch therefore decodes to
the exact provenance of every one of its rows, which is what turns "the window
returned a batch" into "the window returned *these* rows".

Four boundaries, because they are where an off-by-one lives:

* a batch decoded row-for-row against the ``(iteration, row)`` pairs the same
  seed's :meth:`~komugi.train.window.RecencyWindow.draw` reports;
* a window **larger than the store** -- five slots over three iterations, which
  must renormalise over the three rather than pad, starve or index past them;
* a window of **exactly one** iteration -- 100.0% from the newest, with no
  tolerance, which is the only composition assertion in the suite that needs
  none;
* the moment a **new iteration seals and the oldest leaves**: the membership
  changes by exactly one at each end, the evicted iteration becomes
  unreachable through the new window while its data is still perfectly
  readable, and the schedule stays attached to *rank* rather than to identity.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

import pytest
import torch

from komugi.train.shards import FieldSpec, Schema, ShardReader, ShardWriter
from komugi.train.window import IterationRef, RecencyWindow, WindowConfig

_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="iteration", dtype="int64", shape=()),
        FieldSpec(name="row", dtype="int64", shape=()),
    )
)
"""A sample that is nothing but its own provenance.

Two int64 scalars, so a stacked batch is a pair of ``(batch,)`` vectors naming
where every row came from. No features: this module tests composition, and a
payload would only make the assertions slower to read.
"""

_ROWS_PER_ITERATION: Final[int] = 6
_SAMPLES_PER_SHARD: Final[int] = 2
"""Three shards per iteration, so "an iteration" is genuinely several files."""

_FIRST_ITERATION: Final[int] = 10
_ITERATIONS: Final[int] = 8
"""Iterations 10..17, i.e. three more than the default window can hold."""

_SEED: Final[int] = 4242
_EXACT_DRAWS: Final[int] = 64
_COVERAGE_DRAWS: Final[int] = 400
_MIX_DRAWS: Final[int] = 20_000
"""Enough that the five-way mix is resolved far inside the gaps between the
schedule's shares (the smallest gap is ~3 points; the binomial standard error
at 20k draws is ~0.3 of a point)."""

_DEFAULT_RAW: Final[tuple[float, ...]] = (1.0, 0.8, 0.64, 0.512, 0.4096)
"""The documented geometric schedule, newest first, before normalisation."""


def _row(iteration: int, row: int) -> dict[str, torch.Tensor]:
    """Return the sample identifying itself as row `row` of `iteration`."""
    return {
        "iteration": torch.tensor(iteration, dtype=torch.int64),
        "row": torch.tensor(row, dtype=torch.int64),
    }


def _seal_iteration(
    root: Path, iteration: int, *, first_row: int = 0, rows: int = _ROWS_PER_ITERATION
) -> Path:
    """Write `rows` identifiable samples for `iteration` and return its directory.

    Appends rather than replaces: a second call on the same iteration resumes
    after the shards already there (the writer's own
    :func:`~komugi.train.shards._next_shard_index` resumption), which is what
    makes "a new shard seals into an iteration that already has some" a thing
    this module can stage without touching the writer's internals.
    """
    directory = root / f"iter-{iteration:02d}"
    with ShardWriter(
        directory, _SCHEMA, samples_per_shard=_SAMPLES_PER_SHARD
    ) as writer:
        for offset in range(rows):
            writer.append(_row(iteration, first_row + offset))
    return directory


def _reader(directory: Path) -> ShardReader:
    """Return a reader over `directory`, insisting every shard was readable."""
    reader = ShardReader(directory, schema=_SCHEMA)
    assert reader.skipped == (), (
        f"{directory.name} has unreadable shards: {reader.skipped}"
    )
    return reader


def _store(root: Path, iterations: range) -> dict[int, ShardReader]:
    """Seal every iteration in `iterations` and return readers keyed by number."""
    return {
        iteration: _reader(_seal_iteration(root, iteration))
        for iteration in iterations
    }


def _refs(store: dict[int, ShardReader]) -> list[IterationRef]:
    """Return the window refs for `store`, in ascending iteration order."""
    return [
        IterationRef(iteration=iteration, source=source)
        for iteration, source in sorted(store.items())
    ]


def _provenance(batch: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    """Return the ``(iteration, row)`` pair each stacked sample declares."""
    pairs = zip(batch["iteration"].tolist(), batch["row"].tolist(), strict=True)
    return [(int(iteration), int(row)) for iteration, row in pairs]


def _normalised(raw: tuple[float, ...]) -> tuple[float, ...]:
    """Return `raw` scaled to sum to 1."""
    total = sum(raw)
    return tuple(weight / total for weight in raw)


def test_a_reader_serves_an_iterations_rows_in_the_order_they_were_sealed(
    tmp_path: Path,
) -> None:
    """The premise every assertion below rests on: index `i` is row `i`.

    If the reader did not lay an iteration's shards out in seal order, the
    ``(iteration, row)`` pairs the window reports would still be internally
    consistent while naming the wrong samples -- so this is checked first and
    separately rather than assumed by the composition tests.
    """
    source = _reader(_seal_iteration(tmp_path, _FIRST_ITERATION))
    assert len(source) == _ROWS_PER_ITERATION
    for index in range(_ROWS_PER_ITERATION):
        sample = source[index]
        assert int(sample["iteration"].item()) == _FIRST_ITERATION
        assert int(sample["row"].item()) == index


def test_a_sampled_batch_is_exactly_the_rows_the_draw_named(tmp_path: Path) -> None:
    """Row for row: ``sample`` returns what ``draw`` says it drew.

    The two are separate entry points over one generator, and the wire between
    them is a number-keyed lookup: ``sample`` resolves each drawn
    ``(iteration, row)`` through a dict built from the members. A window that
    resolved a draw against the wrong member -- input order instead of recency
    order, slot index instead of iteration number -- would still return a
    perfectly shaped batch of perfectly real samples, drawn at exactly the
    documented frequencies, from the wrong iterations. Only the payload says
    so.
    """
    store = _store(tmp_path, range(_FIRST_ITERATION, _FIRST_ITERATION + _ITERATIONS))
    refs = _refs(store)

    reported = RecencyWindow(refs, WindowConfig(), seed=_SEED).draw(_EXACT_DRAWS)
    batch = RecencyWindow(refs, WindowConfig(), seed=_SEED).sample(_EXACT_DRAWS)

    assert set(batch) == {"iteration", "row"}
    assert batch["iteration"].shape == (_EXACT_DRAWS,)
    assert len(reported) == _EXACT_DRAWS

    # Every drawn pair is resolved against the store itself, so the comparison
    # is "the batch holds the sample that lives at that address", not "the
    # batch holds a sample whose fields happen to match".
    expected = [
        (
            int(store[iteration][row]["iteration"].item()),
            int(store[iteration][row]["row"].item()),
        )
        for iteration, row in reported
    ]
    assert expected == [(iteration, row) for iteration, row in reported], (
        "the store's samples do not identify themselves as the addresses they "
        "live at, so nothing below could distinguish a mis-resolved pick"
    )
    assert _provenance(batch) == expected, (
        "the batch's rows are not the ones draw() named: sample() resolved at "
        "least one pick through the wrong source or the wrong offset"
    )


def test_the_batch_mix_is_the_schedule_the_window_publishes(tmp_path: Path) -> None:
    """Which iterations, in what proportion -- measured off the batch's contents.

    ``window.weights`` is a claim about the future; this is the same claim
    counted after the fact, including the part no tolerance can absorb: the
    shares are strictly ordered newest-to-oldest, and no iteration outside the
    window contributes a single row.
    """
    store = _store(tmp_path, range(_FIRST_ITERATION, _FIRST_ITERATION + _ITERATIONS))
    window = RecencyWindow(_refs(store), WindowConfig(), seed=_SEED)

    newest_first = [17, 16, 15, 14, 13]
    assert [ref.iteration for ref in window.members] == newest_first
    assert window.weights == pytest.approx(_normalised(_DEFAULT_RAW))

    counts: Counter[int] = Counter(
        iteration for iteration, _ in window.draw(_MIX_DRAWS)
    )
    assert set(counts) == set(newest_first), (
        f"the window drew from {sorted(counts)}, not from {sorted(newest_first)}"
    )
    shares = [counts[iteration] / _MIX_DRAWS for iteration in newest_first]
    for share, weight in zip(shares, window.weights, strict=True):
        assert share == pytest.approx(weight, rel=0.05)
    # Recency is an *ordering* claim before it is a numeric one, and this half
    # holds with no tolerance at all: every step back in time draws less.
    for older, newer in zip(shares[1:], shares[:-1], strict=True):
        assert older < newer


def test_a_window_larger_than_the_store_renormalises_over_what_exists(
    tmp_path: Path,
) -> None:
    """Five slots, three iterations: three members, three weights, no padding.

    The failure this excludes is the quiet one -- a window that keeps five
    slots and lets the two empty ones absorb 27% of every batch, so an early
    campaign trains on two thirds of the data it thinks it has.
    """
    store = _store(tmp_path, range(_FIRST_ITERATION, _FIRST_ITERATION + 3))
    window = RecencyWindow(_refs(store), WindowConfig(size=5), seed=_SEED)

    assert [ref.iteration for ref in window.members] == [12, 11, 10]
    assert window.weights == pytest.approx(_normalised(_DEFAULT_RAW[:3]))
    assert sum(window.weights) == pytest.approx(1.0)
    assert len(window) == 3 * _ROWS_PER_ITERATION

    # The relative tilt is untouched by the truncation: each step back still
    # costs exactly one factor of the decay.
    for older, newer in zip(window.weights[1:], window.weights[:-1], strict=True):
        assert older / newer == pytest.approx(0.8)

    batch = window.sample(_COVERAGE_DRAWS)
    seen = _provenance(batch)
    assert {iteration for iteration, _ in seen} == {10, 11, 12}
    for iteration, row in seen:
        assert 0 <= row < _ROWS_PER_ITERATION


def test_a_window_of_exactly_one_iteration_draws_only_that_one(
    tmp_path: Path,
) -> None:
    """The degenerate window, pinned without a tolerance: 100.0% newest.

    ``size=1`` is what a first campaign iteration runs under, and it is the one
    composition in the suite that is exact rather than statistical -- so a
    window that leaked a single row from iteration 16 into a window of {17}
    fails here on that row.
    """
    store = _store(tmp_path, range(_FIRST_ITERATION, _FIRST_ITERATION + _ITERATIONS))
    window = RecencyWindow(_refs(store), WindowConfig(size=1), seed=_SEED)

    newest = _FIRST_ITERATION + _ITERATIONS - 1
    assert [ref.iteration for ref in window.members] == [newest]
    assert window.weights == (1.0,)
    assert len(window) == _ROWS_PER_ITERATION

    seen = _provenance(window.sample(_COVERAGE_DRAWS))
    assert {iteration for iteration, _ in seen} == {newest}
    assert {row for _, row in seen} <= set(range(_ROWS_PER_ITERATION))
    # Not merely "nothing foreign appeared": with 400 draws over 6 rows the
    # single member's every row must have been reachable.
    assert {row for _, row in seen} == set(range(_ROWS_PER_ITERATION))


def test_sealing_a_new_iteration_evicts_exactly_the_oldest(tmp_path: Path) -> None:
    """The moment the window slides: one iteration in, one out, nothing else.

    Both halves are asserted, because only one of them is about the window.
    The evicted iteration's data is *fine* -- its reader still serves all six
    rows -- so its disappearance from the batch is a policy decision, and a
    window that dropped it for any other reason (a reader that broke, a ref
    that was never added) would look identical from the batch alone.
    """
    store = _store(tmp_path, range(_FIRST_ITERATION, _FIRST_ITERATION + 5))
    before = RecencyWindow(_refs(store), WindowConfig(size=5), seed=_SEED)
    assert [ref.iteration for ref in before.members] == [14, 13, 12, 11, 10]

    fresh = _FIRST_ITERATION + 5
    store[fresh] = _reader(_seal_iteration(tmp_path, fresh))
    after = RecencyWindow(_refs(store), WindowConfig(size=5), seed=_SEED)

    assert [ref.iteration for ref in after.members] == [15, 14, 13, 12, 11]
    assert len(after) == 5 * _ROWS_PER_ITERATION
    # The schedule is attached to rank, not to identity: the new newest draws
    # exactly the share the old newest did.
    assert after.weights == before.weights

    seen = _provenance(after.sample(_COVERAGE_DRAWS))
    assert {iteration for iteration, _ in seen} == {11, 12, 13, 14, 15}
    assert all(iteration != _FIRST_ITERATION for iteration, _ in seen), (
        f"iteration {_FIRST_ITERATION} was evicted from the window and still "
        "contributed rows to a batch"
    )

    # And the eviction is the window's doing, not the store's: the dropped
    # iteration is still whole and still readable.
    evicted = store[_FIRST_ITERATION]
    assert len(evicted) == _ROWS_PER_ITERATION
    assert int(evicted[0]["iteration"].item()) == _FIRST_ITERATION
    assert [ref.iteration for ref in before.members] == [14, 13, 12, 11, 10], (
        "building the newer window mutated the older one's membership"
    )


def test_a_shard_sealed_after_construction_joins_the_window_only_on_rebuild(
    tmp_path: Path,
) -> None:
    """A shard sealing mid-iteration changes the mix at the next rebuild, exactly.

    A reader is a snapshot, so the live window keeps drawing from the rows it
    was built over -- which is the property that makes a training loop's
    batches reproducible while self-play is still appending underneath it. The
    exactness is the point: after the rebuild the window reaches the six old
    rows *and* the four new ones, and nothing else.
    """
    directory = _seal_iteration(tmp_path, _FIRST_ITERATION)
    snapshot = _reader(directory)
    live = RecencyWindow(
        [IterationRef(iteration=_FIRST_ITERATION, source=snapshot)],
        WindowConfig(size=1),
        seed=_SEED,
    )
    assert len(live) == _ROWS_PER_ITERATION

    _seal_iteration(
        tmp_path, _FIRST_ITERATION, first_row=_ROWS_PER_ITERATION, rows=4
    )

    old_rows = {row for _, row in _provenance(live.sample(_COVERAGE_DRAWS))}
    assert old_rows == set(range(_ROWS_PER_ITERATION)), (
        "the live window drew a row that did not exist when it was built: its "
        "reader is not the snapshot the training loop's reproducibility "
        "assumes"
    )

    rebuilt_source = _reader(directory)
    assert len(rebuilt_source) == _ROWS_PER_ITERATION + 4
    rebuilt = RecencyWindow(
        [IterationRef(iteration=_FIRST_ITERATION, source=rebuilt_source)],
        WindowConfig(size=1),
        seed=_SEED,
    )
    assert len(rebuilt) == _ROWS_PER_ITERATION + 4

    new_rows = {row for _, row in _provenance(rebuilt.sample(_COVERAGE_DRAWS))}
    assert new_rows == set(range(_ROWS_PER_ITERATION + 4)), (
        "the rebuilt window does not reach exactly the sealed rows: "
        f"{sorted(new_rows)}"
    )
