"""The recency schedule, finally holding more than one iteration of real data.

``komugi.train.window`` has been correct and unemployed since KOM-020. Its own
suite proves the geometric tilt, the renormalisation and the determinism -- all
over lists of synthetic dicts, and all with iteration numbers invented by the
test. Design §4's actual claim is bigger and had nowhere to live until a
campaign existed: *iteration 1 trains on iteration 1 and iteration 0*, with the
newest tilted up, over shards that real self-play wrote.

This module reconstructs that wiring from the campaign's output and asserts it
holds:

    RecencyWindow([
        IterationRef(0, ConcatSource(readers over root/iter_0/worker_*)),
        IterationRef(1, ConcatSource(readers over root/iter_1/worker_*)),
    ])

Two joins have to work for that line to mean anything, and both are design
findings rather than plumbing. ``ConcatSource`` is design F5: an iteration is
several worker directories (``ShardWriter`` allows one writer each) while an
``IterationRef`` takes exactly one source, so without it the only way to put a
two-worker iteration in a window is to give each worker its own slot in the
recency schedule -- which silently doubles that iteration's share of every
batch. And the ``iteration`` numbers have to be the campaign's, because they
are what the tilt is computed over: number them wrong and the window is a
uniform sampler wearing a schedule's name.

The window is *reconstructed* here rather than reached into the loop for. The
loop is free to build its own (and does, iteration by iteration); what this
test pins is that the data it wrote supports the construction the design
specifies -- which is the part a later refactor can break without noticing.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from komugi.selfplay.records import SCHEMA, ConcatSource
from komugi.train.window import IterationRef, RecencyWindow, WindowConfig, recency_weights

import campaign_rig as rig

camp = pytest.importorskip("komugi.campaign.loop")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )

_DRAWS: int = 400
"""Draws taken when asking whether both iterations are reachable.

With two iterations the rarer one has probability 0.8/1.8 = 0.444, so missing
it 400 times running has probability ~1e-102. This is a bounded-error
assertion, not a flaky one -- and it is deliberately made against ``draw``
rather than against the weights, because the weights are a tuple the window
computes and the draws are what a trainer actually eats.
"""

_WINDOW_SEED: int = 7
"""Any fixed seed. The window owns a PCG64 stream seeded at construction, so
naming the seed is what makes this test reproducible rather than lucky."""


@pytest.fixture(scope="module")
def campaign() -> rig.SharedCampaign:
    """The shared two-iteration campaign."""
    return rig.shared_campaign()


@pytest.fixture(scope="module")
def sources(campaign: rig.SharedCampaign) -> dict[int, ConcatSource]:
    """One ``ConcatSource`` per iteration, over that iteration's worker dirs."""
    campaign.results()  # a failed campaign fails here, not as an empty window
    built: dict[int, ConcatSource] = {}
    for iteration in range(rig.ITERATIONS):
        directory = rig.iteration_dir(campaign.root, iteration)
        worker_dirs = rig.shard_dirs_under(directory)
        assert worker_dirs, f"{directory} holds no worker shard directory"
        built[iteration] = rig.concat_source(worker_dirs)
    return built


@pytest.fixture(scope="module")
def window(sources: dict[int, ConcatSource]) -> RecencyWindow:
    """The design-§4 window over both iterations, at the default tilt."""
    return RecencyWindow(
        [
            IterationRef(iteration=iteration, source=source)
            for iteration, source in sources.items()
        ],
        WindowConfig(),
        seed=_WINDOW_SEED,
    )


def test_each_iteration_reaches_the_window_as_one_source(
    campaign: rig.SharedCampaign, sources: dict[int, ConcatSource]
) -> None:
    """Design F5: several worker directories, one ``SampleSource``, right length.

    The length is the assertion. A ``ConcatSource`` whose prefix sums were
    wrong, or which was handed one reader instead of two, would still index and
    still train -- on half the iteration, forever.
    """
    for iteration, source in sources.items():
        directory = rig.iteration_dir(campaign.root, iteration)
        readers = rig.readers_for(rig.shard_dirs_under(directory))
        assert len(readers) >= 1, f"{directory} produced no non-empty reader"
        assert len(source) == sum(len(reader) for reader in readers), (
            f"iteration {iteration}'s ConcatSource holds {len(source)} rows, but "
            f"its {len(readers)} reader(s) hold "
            f"{[len(reader) for reader in readers]}"
        )
        assert len(source) > 0, f"iteration {iteration} contributed no rows"


def test_two_workers_really_did_write_two_directories(
    campaign: rig.SharedCampaign,
) -> None:
    """The premise of the test above: the concatenation is not concatenating one.

    ``workers=2`` is in this suite's config precisely so the F5 join is
    exercised; if a future change quietly ran one worker, the ``ConcatSource``
    assertions would pass while testing nothing.
    """
    for iteration in range(rig.ITERATIONS):
        directory = rig.iteration_dir(campaign.root, iteration)
        worker_dirs = rig.shard_dirs_under(directory)
        assert len(worker_dirs) >= rig.WORKERS, (
            f"iteration {iteration} ran with workers={rig.WORKERS} but left "
            f"{len(worker_dirs)} shard directories: {rig.describe(worker_dirs)}"
        )


def test_the_window_holds_both_iterations_newest_first(
    window: RecencyWindow, sources: dict[int, ConcatSource]
) -> None:
    """Both campaign iterations are members, ordered by recency, and complete."""
    assert [ref.iteration for ref in window.members] == [1, 0], (
        f"window members are {[ref.iteration for ref in window.members]}, want "
        f"[1, 0] -- newest first"
    )
    per_iteration = {iteration: len(source) for iteration, source in sources.items()}
    assert len(window) == sum(per_iteration.values()), (
        f"the window reaches {len(window)} rows; the two iterations hold "
        f"{per_iteration}"
    )


def test_the_recency_tilt_is_the_design_schedule(window: RecencyWindow) -> None:
    """Two present iterations get the truncated, renormalised geometric weights.

    ``0.8`` decay over two iterations is ``(1/1.8, 0.8/1.8)``. The point is not
    the arithmetic (its own unit suite owns that) but that the campaign's two
    iterations are what the schedule was computed over: a window that had
    silently dropped one would report a single weight of 1.0 here and still
    sample happily.
    """
    expected = recency_weights(len(window.members), window.config)
    assert window.weights == pytest.approx(expected), (
        f"window weights {window.weights} are not the schedule {expected}"
    )
    assert sum(window.weights) == pytest.approx(1.0)
    assert window.weights[0] > window.weights[1], (
        "the newest iteration must be tilted up, not down"
    )


def test_draws_land_in_both_iterations(window: RecencyWindow) -> None:
    """The window indexes rows from BOTH iterations' shard trees.

    The headline claim of the design-§4 wiring, and the one thing no component
    test can make: until a campaign existed, every window in this repository
    was built over a single iteration of synthetic rows.
    """
    picks = window.draw(_DRAWS)
    assert len(picks) == _DRAWS
    drawn = {iteration for iteration, _ in picks}
    assert drawn == {0, 1}, (
        f"{_DRAWS} draws touched iterations {sorted(drawn)}; both of the "
        f"campaign's iterations must be reachable"
    )
    lengths = {ref.iteration: len(ref.source) for ref in window.members}
    for iteration, row in picks:
        assert 0 <= row < lengths[iteration], (
            f"draw ({iteration}, {row}) is outside iteration {iteration}'s "
            f"{lengths[iteration]} rows"
        )


def test_a_batch_from_the_window_is_a_real_trainable_batch(
    window: RecencyWindow,
) -> None:
    """``sample`` stacks rows from two iterations into one schema-shaped batch.

    Stacking across iterations is where a shape or dtype drift between
    generations would surface -- two iterations written by two different nets
    are still two writes of the same ``SCHEMA``, and if they were not, this is
    the call that a trainer would crash inside.
    """
    batch = window.sample(rig.BATCH_SIZE)
    assert set(batch) == set(SCHEMA.names), (
        f"batch fields {sorted(batch)} are not the self-play schema's "
        f"{sorted(SCHEMA.names)}"
    )
    observation = batch["observation"]
    assert observation.shape == (
        rig.BATCH_SIZE,
        rig.IN_PLANES,
        rig.BOARD_H,
        rig.BOARD_W,
    ), f"observation batch has shape {tuple(observation.shape)}"
    assert observation.dtype == torch.float32
    assert torch.isfinite(observation).all(), "a sampled observation holds NaN/inf"


def test_the_two_iterations_are_different_data(
    sources: dict[int, ConcatSource]
) -> None:
    """Iteration 1's rows are not iteration 0's rows.

    Without this the window could be spanning two directories holding the same
    games -- which is what a campaign that replayed iteration 0's seeds under
    iteration 1's name would produce, and which would make the recency tilt a
    weighted average of a thing with itself. Compared as sets of observation
    digests rather than element-wise, because the two iterations need not hold
    the same number of rows; equality of the sets is the failure.
    """
    digests: dict[int, set[str]] = {}
    for iteration, source in sources.items():
        seen: set[str] = set()
        for index in range(len(source)):
            tensor = source[index]["observation"]
            seen.add(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        digests[iteration] = seen
    assert digests[0] != digests[1], (
        "the two iterations hold identical position sets: iteration 1 replayed "
        "iteration 0's games rather than playing new ones"
    )
