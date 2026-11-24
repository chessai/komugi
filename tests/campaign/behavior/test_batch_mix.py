"""What mix of iterations a training batch is actually made of.

The recency schedule has been asserted twice in this repository and never yet
*observed*. ``tests/train/test_window_unit.py`` pins the arithmetic of
:func:`~komugi.train.window.recency_weights`, and
``test_window_spans_iterations`` pins that a campaign's shards support a window
and that :meth:`~komugi.train.window.RecencyWindow.draw` tilts -- but ``draw``
and :meth:`~komugi.train.window.RecencyWindow.sample` are an API path
:func:`~komugi.campaign.loop.run_campaign` **never calls**. The loop trains
through :class:`~komugi.campaign.loop.WindowSource`, whose whole idea is that
the tilt is carried by an *index map* rather than by a second stream of
randomness: :func:`~komugi.train.step.training_iteration` draws indices
uniformly over a virtual range, and the range's block sizes are what make a
uniform draw come out recency-weighted.

So the schedule reaching a batch depends on the quota arithmetic in
``WindowSource.__init__``, the ``local * len(source) // quota`` map in its
``__getitem__``, and the uniform draw in ``_sample_batch`` -- three pieces that
no existing test observes together. That is KOM-031-union-review F6, and it is
the kind of gap where a plausible bug is silent: a quota computed from the
member lengths instead of the weights, an off-by-one in the offsets, or a
member accidentally excluded would all still train, still descend, and still
report a perfectly ordinary loss, on a mix that is not the design's.

How this module observes rather than reconstructs
-------------------------------------------------
Nothing here recomputes an index, a quota or a draw. The whole apparatus is:

* **a sentinel on disk.** Each iteration's rows carry an ``iteration_id``
  column whose value *is* the iteration that wrote them. It is a real schema
  field in a real :class:`~komugi.train.shards.ShardWriter` shard tree, laid
  out the way the campaign lays one out (``iter_<i>/shards/worker_<k>_r0``), so
  the rows travel to the batch through
  :func:`~komugi.campaign.loop.window_refs`,
  :class:`~komugi.train.shards.ShardReader` and
  :class:`~komugi.selfplay.records.ConcatSource` exactly as self-play rows do.
* **an injected step.** :class:`RecordingStep` is a :class:`TrainStep` whose
  ``run`` reads the sentinel column off the batch it is handed and returns stub
  metrics. The batch is the genuine one
  :func:`~komugi.train.step.training_iteration` assembled -- this module never
  builds a batch, never calls ``_sample_batch``, and never draws an index.

The forward/backward is what is stubbed, and only that. It is the one part of a
step that says nothing about *which rows arrived*, it is pinned by
``tests/train`` over the real schema and the real net, and running a real net
over the >100k rows this module needs would trade seconds of CPU for no
additional evidence.

Why seven iterations of unequal size
------------------------------------
Seven, with a window of five, so that "no ineligible iteration is ever drawn"
is a claim with something to be wrong about: iterations 0 and 1 are on disk,
readable, and present in ``window_refs``'s output, and must never appear in a
batch. The row counts *decrease* with recency (the newest iteration is the
smallest) because that is the design's sharpest claim, stated in
:mod:`komugi.train.window`'s docstring: the mix is a property of the schedule
alone, not of self-play throughput. A ``WindowSource`` that sized its blocks by
member length would put the newest iteration *last* here, and the shares would
come out inverted rather than merely wrong.

Expected shares are read off :attr:`RecencyWindow.weights`, which is
:class:`~komugi.train.window.WindowConfig`'s schedule for the number of members
present; no percentage is spelled in this file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest
import torch

from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.train.losses import LossAssembler
from komugi.train.shards import FieldSpec, Schema, ShardWriter
from komugi.train.step import (
    OBSERVATION_KEY,
    TOTAL_KEY,
    StepConfig,
    TrainStep,
    training_iteration,
)
from komugi.train.window import RecencyWindow, WindowConfig, recency_weights
from komugi.train.wiring import P0A_RECIPE

camp = pytest.importorskip("komugi.campaign.loop")

if not hasattr(camp, "WindowSource"):
    pytest.skip(
        "komugi.campaign.loop exports no WindowSource yet (KOM-031)",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# the sentinel-bearing source set
# ---------------------------------------------------------------------------

SENTINEL_FIELD: Final[str] = "iteration_id"
"""The column whose value is the iteration that wrote the row.

An ``int64`` scalar per row, so a stacked batch carries a ``(batch,)`` vector
of iteration numbers and "which generation is this batch made of" is a read
rather than an inference.
"""

TOY_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name=OBSERVATION_KEY, dtype="float32", shape=(2,)),
        FieldSpec(name=SENTINEL_FIELD, dtype="int64", shape=()),
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
    )
)
"""A three-field sample: an observation the shape of nothing in particular,
the sentinel, and the plane-map version stamp.

Deliberately not :data:`komugi.selfplay.records.SCHEMA`. The production schema
would make every one of the >100k rows drawn below a (128, 9, 9) float32 plane
stack -- gigabytes of tensor construction to observe an integer column. The
readers, the concat, the window, the quota map and the uniform draw are all
schema-agnostic (:func:`~komugi.campaign.loop.iteration_source` builds its
readers *without* an expected schema precisely so a toy-shaped shard tree is
readable), so the small schema exercises the same code with the same indices.
``plane_map_version`` is carried anyway because
:class:`~komugi.selfplay.records.ConcatSource` (which :class:`RecencyWindow`'s
members read through) checks it on every row regardless of schema.
"""

ITERATIONS_ON_DISK: Final[int] = 7
"""Iterations written. Two more than the default window keeps, so that
"ineligible iterations never appear" has candidates."""

WORKERS_PER_ITERATION: Final[int] = 2
"""Worker directories per iteration -- design D10's layout, and the
``ConcatSource`` join (design F5) that turns them back into one source."""

OLDEST_ROWS: Final[int] = 180
ROWS_PER_ITERATION_DROP: Final[int] = 10
"""Row counts run *down* with recency: iteration 0 holds
:data:`OLDEST_ROWS` rows and each later iteration holds
:data:`ROWS_PER_ITERATION_DROP` fewer. See the module docstring -- a
throughput-sized window would invert the tilt here rather than merely blur it.
"""

SAMPLES_PER_SHARD: Final[int] = 64
"""Small enough that every worker directory holds several shards, so the
reader's cross-shard index arithmetic is in the path too."""

# ---------------------------------------------------------------------------
# the observation budget
# ---------------------------------------------------------------------------

STEPS: Final[int] = 200
BATCH_SIZE: Final[int] = 512
OBSERVATIONS: Final[int] = STEPS * BATCH_SIZE
"""102,400 sentinel observations from one :func:`training_iteration` call.

The tolerance below is what this has to beat. At the newest iteration's ~0.297
share the sampling standard deviation over this many draws is ~0.0014, so
:data:`SHARE_TOLERANCE` is ~21 sigma: a bounded-error assertion, not a coin
flip. The residual from the quota map's integer rounding is smaller still
(~0.0005 at these sizes).
"""

SHARE_TOLERANCE: Final[float] = 0.03
"""Percentage points, absolute, per iteration."""

RNG_SEED: Final[int] = 4242
"""``training_iteration``'s batch-sampling seed."""

OTHER_RNG_SEED: Final[int] = 909
"""A second one, for the half of determinism that says the seed matters."""

WINDOW_SEED: Final[int] = 11
"""Seed handed to :class:`RecencyWindow`. It seeds a generator that
:class:`WindowSource` never draws from -- the tilt is an index map, not a
second stream -- and is passed anyway because the loop passes one."""

DETERMINISM_STEPS: Final[int] = 20
DETERMINISM_BATCH: Final[int] = 64
"""The replay is a sequence comparison, not a distribution estimate, so it runs
at 1/80th of the budget above: 1,280 sentinels that must match position for
position."""

LR: Final[float] = 1e-3


def _rows_for(iteration: int) -> int:
    """Return how many rows iteration `iteration` was written with."""
    return OLDEST_ROWS - ROWS_PER_ITERATION_DROP * iteration


def _sample(iteration: int, row: int) -> dict[str, torch.Tensor]:
    """Return one :data:`TOY_SCHEMA` row, stamped with its iteration.

    The observation carries ``(iteration, row)`` as floats so that a row is
    identifiable in a failure dump; nothing asserts on it. The sentinel is the
    payload.
    """
    return {
        OBSERVATION_KEY: torch.tensor(
            [float(iteration), float(row)], dtype=torch.float32
        ),
        SENTINEL_FIELD: torch.tensor(iteration, dtype=torch.int64),
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
    }


def _write_iteration(root: Path, iteration: int) -> None:
    """Write iteration `iteration`'s rows under the campaign's own layout.

    ``camp.shard_root`` rather than a path this module spells: the layout is
    the loop's, and :func:`~komugi.campaign.loop.window_refs` is what reads it
    back. Rows are split across :data:`WORKERS_PER_ITERATION` directories --
    one :class:`ShardWriter` each, as the writer's one-writer-per-directory
    rule requires -- and the split is uneven when the count is odd, which is
    what a real pair of workers produces.
    """
    directory: Path = camp.shard_root(root, iteration)
    rows = _rows_for(iteration)
    written = 0
    for worker in range(WORKERS_PER_ITERATION):
        share = rows // WORKERS_PER_ITERATION + (
            1 if worker < rows % WORKERS_PER_ITERATION else 0
        )
        with ShardWriter(
            directory / f"worker_{worker}_r0",
            TOY_SCHEMA,
            samples_per_shard=SAMPLES_PER_SHARD,
        ) as writer:
            for _ in range(share):
                writer.append(_sample(iteration, written))
                written += 1
    assert written == rows, f"iteration {iteration} wrote {written} of {rows} rows"


# ---------------------------------------------------------------------------
# the injected step
# ---------------------------------------------------------------------------


class RecordingStep(TrainStep):
    """A :class:`TrainStep` that records the sentinel column of every batch.

    A subclass rather than a stand-in because
    :func:`~komugi.train.step.training_iteration` takes a ``TrainStep`` and
    checks ``step.net is net`` -- and because the batch has to arrive through
    the real call, not through a seam this test invented. ``run`` reads the
    column and returns stub metrics rather than descending: see the module
    docstring for why the optimiser step is the part that can be stubbed
    without weakening the claim.
    """

    def __init__(self, net: torch.nn.Module, batch_size: int) -> None:
        """Build a real step over `net`, and an empty ledger."""
        super().__init__(
            net,
            LossAssembler(P0A_RECIPE),
            StepConfig(batch_size=batch_size, lr=LR),
        )
        self.seen: list[int] = []
        self.batches: int = 0

    def run(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Record `batch`'s iteration column and return a stub breakdown."""
        assert set(batch) == set(TOY_SCHEMA.names), (
            f"batch fields {sorted(batch)} are not the source's "
            f"{sorted(TOY_SCHEMA.names)}: this is not a stacked source batch"
        )
        column = batch[SENTINEL_FIELD]
        assert tuple(column.shape) == (self.config.batch_size,), (
            f"sentinel column has shape {tuple(column.shape)}, want "
            f"({self.config.batch_size},)"
        )
        self.seen.extend(int(value) for value in column.tolist())
        self.batches += 1
        return {TOTAL_KEY: 0.0}


def _record(source: Any, *, steps: int, batch_size: int, seed: int) -> list[int]:
    """Run `steps` batches through the real loop and return their sentinels.

    A fresh net and a fresh :class:`RecordingStep` per call, so two calls share
    no state but the source -- which is the thing under test and holds no
    randomness of its own. No checkpoint and no journal: this call trains
    nothing and has nothing to persist.
    """
    net = torch.nn.Linear(1, 1)
    step = RecordingStep(net, batch_size)
    training_iteration(net, source, step, steps=steps, rng_seed=seed)
    assert step.batches == steps, (
        f"training_iteration ran {step.batches} steps, asked for {steps}"
    )
    assert len(step.seen) == steps * batch_size
    return step.seen


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign-shaped shard tree holding :data:`ITERATIONS_ON_DISK` iterations."""
    directory = tmp_path_factory.mktemp("batch-mix")
    for iteration in range(ITERATIONS_ON_DISK):
        _write_iteration(directory, iteration)
    return directory


@pytest.fixture(scope="module")
def window(root: Path) -> RecencyWindow:
    """The window the loop would build over that tree, at the default config.

    ``window_refs`` over *every* iteration on disk -- the unbounded listing.
    :func:`~komugi.campaign.loop.run_campaign` now bounds construction at the
    window's size (``stop_after``, KOM-041), which is equivalent by the
    argument on :func:`~komugi.campaign.loop.window_refs`; this fixture keeps
    the unbounded shape so the truncation observed here is
    :class:`RecencyWindow`'s own decision.
    """
    refs = camp.window_refs(root, ITERATIONS_ON_DISK - 1)
    return RecencyWindow(refs, WindowConfig(), seed=WINDOW_SEED)


@pytest.fixture(scope="module")
def source(window: RecencyWindow) -> Any:
    """The window as the flat, tilt-carrying index range the trainer samples."""
    return camp.WindowSource(window)


@pytest.fixture(scope="module")
def observed(source: Any) -> list[int]:
    """:data:`OBSERVATIONS` sentinels, in the order the trainer ate them."""
    return _record(source, steps=STEPS, batch_size=BATCH_SIZE, seed=RNG_SEED)


def _shares(observed: list[int]) -> dict[int, float]:
    """Return the observed fraction of all rows contributed by each iteration."""
    counts = Counter(observed)
    total = len(observed)
    return {iteration: count / total for iteration, count in counts.items()}


# ---------------------------------------------------------------------------
# the premise
# ---------------------------------------------------------------------------


def test_the_window_is_a_strict_subset_of_what_is_on_disk(
    window: RecencyWindow, root: Path
) -> None:
    """Seven readable iterations, five eligible, newest first.

    The premise of everything below. If ``window_refs`` returned five refs
    instead of seven -- or if the window kept all of them -- the
    no-ineligible-iteration assertion would pass while testing nothing, and the
    shares would be checked against a schedule computed over the wrong count.
    """
    refs = camp.window_refs(root, ITERATIONS_ON_DISK - 1)
    assert [ref.iteration for ref in refs] == list(range(ITERATIONS_ON_DISK)), (
        f"window_refs found iterations {[ref.iteration for ref in refs]}, want "
        f"all {ITERATIONS_ON_DISK} that were written"
    )
    for ref in refs:
        assert len(ref.source) == _rows_for(ref.iteration), (
            f"iteration {ref.iteration} reads back {len(ref.source)} rows, "
            f"{_rows_for(ref.iteration)} were written"
        )
    members = [ref.iteration for ref in window.members]
    assert members == sorted(members, reverse=True), (
        f"window members {members} are not newest-first"
    )
    assert len(members) == WindowConfig().size
    assert members[0] == ITERATIONS_ON_DISK - 1
    lengths = [len(ref.source) for ref in window.members]
    assert lengths == sorted(lengths), (
        f"member row counts {lengths} do not increase with age; this suite's "
        f"point is that the newest iteration is the smallest one"
    )


# ---------------------------------------------------------------------------
# the three claims
# ---------------------------------------------------------------------------


def test_batch_mix_matches_the_recency_schedule(
    window: RecencyWindow, observed: list[int]
) -> None:
    """Each iteration's share of the trainer's rows is its scheduled weight.

    The F6 claim. ``WindowSource`` turns the weight schedule into block sizes
    over a virtual index range and :func:`_sample_batch` draws that range
    uniformly; if either half were wrong the shares would be the member lengths
    normalised (this tree's lengths run the *other* way, so that failure is
    unmissable here), or uniform, or shifted by an offset bug.

    Expected shares are :attr:`RecencyWindow.weights` -- the schedule
    :class:`~komugi.train.window.WindowConfig` computes for the five members
    present -- and are checked against
    :func:`~komugi.train.window.recency_weights` first so that a window
    reporting a schedule it did not compute cannot define its own target.
    """
    expected_weights = recency_weights(len(window.members), window.config)
    assert window.weights == pytest.approx(expected_weights)
    expected = {
        ref.iteration: weight
        for ref, weight in zip(window.members, window.weights, strict=True)
    }
    shares = _shares(observed)
    assert len(observed) == OBSERVATIONS
    mix = {key: round(value, 4) for key, value in sorted(shares.items())}
    for iteration, want in expected.items():
        got = shares.get(iteration, 0.0)
        assert abs(got - want) <= SHARE_TOLERANCE, (
            f"iteration {iteration} supplied {got:.4f} of the trainer's rows; "
            f"its scheduled weight is {want:.4f} (tolerance "
            f"{SHARE_TOLERANCE}). Full mix: {mix}"
        )
    ordered = [shares.get(ref.iteration, 0.0) for ref in window.members]
    assert ordered == sorted(ordered, reverse=True), (
        f"the observed mix {ordered} is not tilted newest-first"
    )


def test_every_eligible_iteration_appears_and_no_other_ever_does(
    window: RecencyWindow, observed: list[int]
) -> None:
    """Exactly the five window members reach the batches; 0 and 1 never do.

    Two failures in one assertion, because they are the two ways the index map
    can be wrong without being noticed. A member whose block is empty (or
    unreachable past an offset) silently drops a generation from training; a
    row resolved outside its member's block trains on data the window
    *excluded*, which is the recency window failing to be a window at all.
    """
    eligible = {ref.iteration for ref in window.members}
    ineligible = set(range(ITERATIONS_ON_DISK)) - eligible
    assert ineligible, "the on-disk set must be larger than the window"
    drawn = set(observed)
    missing = sorted(eligible - drawn)
    assert not missing, (
        f"iterations {missing} are in the window but supplied no row in "
        f"{OBSERVATIONS} draws"
    )
    intruders = sorted(drawn - eligible)
    assert not intruders, (
        f"iterations {intruders} reached a training batch but are outside the "
        f"window (eligible: {sorted(eligible)}, on disk: "
        f"{sorted(range(ITERATIONS_ON_DISK))})"
    )


def test_same_seed_draws_the_same_batches(source: Any) -> None:
    """The documented replay contract, observed through the real path.

    :func:`~komugi.train.step.training_iteration` promises that "two calls with
    the same `rng_seed` over the same source draw the same batches in the same
    order". Over a ``WindowSource`` that promise is only true because the
    source contributes no randomness of its own -- the tilt is an index map,
    and the window's own PCG64 stream is never drawn from. A ``__getitem__``
    that had reached for that generator would pass every distribution assertion
    above and fail here.

    The second half of the test is the one that keeps the first honest: a
    different seed must produce a different sequence, or "identical" would be
    satisfied by a recorder that recorded nothing interesting.
    """
    first = _record(
        source, steps=DETERMINISM_STEPS, batch_size=DETERMINISM_BATCH, seed=RNG_SEED
    )
    second = _record(
        source, steps=DETERMINISM_STEPS, batch_size=DETERMINISM_BATCH, seed=RNG_SEED
    )
    divergence = next(
        (
            position
            for position, (left, right) in enumerate(zip(first, second, strict=True))
            if left != right
        ),
        len(first),
    )
    assert first == second, (
        "two identical training_iteration calls drew different rows: the first "
        f"divergence is at position {divergence} of {len(first)} "
        f"({first[divergence:divergence + 4]} vs "
        f"{second[divergence:divergence + 4]})"
    )
    other = _record(
        source,
        steps=DETERMINISM_STEPS,
        batch_size=DETERMINISM_BATCH,
        seed=OTHER_RNG_SEED,
    )
    assert other != first, (
        f"seed {OTHER_RNG_SEED} drew exactly the sequence seed {RNG_SEED} drew; "
        f"rng_seed is not reaching the sampler"
    )


def test_every_row_within_a_member_is_reachable_and_drawn_evenly(
    source: Any,
) -> None:
    """The within-member row map, observed over the whole virtual range.

    ``WindowSource``'s docstring claims rows within a member appear either
    ``floor(q/n)`` or ``ceil(q/n)`` times across the member's quota block.
    Nothing above can see a violation: the sentinel is per-iteration, so a
    ``__getitem__`` that resolved every draw in a member to that member's
    *first row* would leave every share, eligibility and determinism assertion
    green while training collapsed onto one row per iteration (fanout-1 union
    review F3 -- the ``row = 0`` mutant survived the original suite). The
    observation column carries ``(iteration, row)`` for exactly this purpose;
    here it is asserted on, over every virtual index the trainer could draw.
    """
    rows_seen: dict[int, Counter[int]] = {}
    for index in range(len(source)):
        observation = source[index][OBSERVATION_KEY]
        iteration, row = (int(value) for value in observation.tolist())
        rows_seen.setdefault(iteration, Counter())[row] += 1
    for iteration, counts in sorted(rows_seen.items()):
        rows = _rows_for(iteration)
        quota = sum(counts.values())
        assert set(counts) == set(range(rows)), (
            f"iteration {iteration} wrote {rows} rows but its quota of "
            f"{quota} resolves only to rows {sorted(counts)}: a row the map "
            f"cannot reach never trains"
        )
        floor, ceil = quota // rows, -(-quota // rows)
        uneven = {
            row: count for row, count in counts.items() if count not in (floor, ceil)
        }
        assert not uneven, (
            f"iteration {iteration}'s rows are drawn unevenly: quota {quota} "
            f"over {rows} rows should give each row {floor} or {ceil} "
            f"appearances, but these deviate: {uneven}"
        )
