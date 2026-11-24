"""The replay window draws from where it says it draws from.

A recency schedule that is documented but not measured is a comment. These
tests pin three things: the schedule's numbers, the empirical draw frequencies
against those numbers, and determinism under a seed -- without which a
training run cannot be replayed.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from komugi.train.window import (
    IterationRef,
    RecencyWindow,
    WindowConfig,
    recency_weights,
)


class _ListSource:
    """An in-memory iteration: the smallest thing satisfying ``SampleSource``.

    Its existence is the point of the protocol -- the window never learns that
    the production source decompresses shards.
    """

    def __init__(self, iteration: int, size: int) -> None:
        self._samples = [
            {
                "features": torch.tensor(
                    [float(iteration), float(row)], dtype=torch.float32
                )
            }
            for row in range(size)
        ]

    def __len__(self) -> int:
        """Return the number of samples held."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample at `index`."""
        return self._samples[index]


def _refs(*sizes: int, first: int = 10) -> list[IterationRef]:
    """Return iteration refs numbered `first`.. with the given sample counts."""
    return [
        IterationRef(iteration=first + offset, source=_ListSource(first + offset, size))
        for offset, size in enumerate(sizes)
    ]


def test_the_documented_schedule_is_the_implemented_one() -> None:
    """Geometric 0.8 decay, newest first, normalised to sum to 1."""
    weights = recency_weights(5, WindowConfig())
    raw = (1.0, 0.8, 0.64, 0.512, 0.4096)
    total = sum(raw)
    assert weights == pytest.approx(tuple(value / total for value in raw))
    assert sum(weights) == pytest.approx(1.0)
    # Every step back in time costs exactly one factor of `decay`.
    for older, newer in zip(weights[1:], weights[:-1], strict=True):
        assert older / newer == pytest.approx(0.8)


def test_decay_one_reproduces_v1s_flat_window() -> None:
    """decay=1.0 is a uniform window -- the behaviour this generalises."""
    weights = recency_weights(3, WindowConfig(size=3, decay=1.0))
    assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_explicit_weights_override_the_schedule() -> None:
    """An explicit tilt is used verbatim, truncated to the window and normalised."""
    config = WindowConfig(size=2, weights=(3.0, 1.0, 9.0))
    assert recency_weights(2, config) == pytest.approx((0.75, 0.25))


def test_a_short_window_renormalises_rather_than_pads() -> None:
    """Three iterations available means three weights that still sum to 1."""
    weights = recency_weights(3, WindowConfig())
    raw = (1.0, 0.8, 0.64)
    assert weights == pytest.approx(tuple(value / sum(raw) for value in raw))


def test_draw_frequencies_match_the_documented_weights() -> None:
    """Empirical draw frequencies track the schedule within 5%.

    Ten independent seeds of 10k draws each. Pooling them is what makes the
    5% assertion a statement about the sampler rather than about one lucky
    seed: at 10k draws the oldest bin's own sampling noise is already ~2.7%
    of its weight, so a per-seed 5% bound would be roughly a two-sigma coin
    flip. The per-seed bound is therefore kept deliberately loose (15%) and
    the tight one is applied to the pooled 100k.
    """
    window_config = WindowConfig()
    expected = recency_weights(5, window_config)
    newest_first = [14, 13, 12, 11, 10]

    pooled: Counter[int] = Counter()
    per_seed: list[Counter[int]] = []
    draws_per_seed = 10_000
    for seed in range(10):
        window = RecencyWindow(_refs(*(100,) * 5), window_config, seed=seed)
        counts: Counter[int] = Counter()
        for iteration, row in window.draw(draws_per_seed):
            assert 0 <= row < 100
            counts[iteration] += 1
        pooled.update(counts)
        per_seed.append(counts)

    total = draws_per_seed * len(per_seed)
    assert sum(pooled.values()) == total
    for slot, iteration in enumerate(newest_first):
        assert pooled[iteration] / total == pytest.approx(expected[slot], rel=0.05)
    for counts in per_seed:
        for slot, iteration in enumerate(newest_first):
            assert counts[iteration] / draws_per_seed == pytest.approx(
                expected[slot], rel=0.15
            )


def test_the_same_seed_gives_the_same_draws() -> None:
    """Determinism: seed in, identical stream out; a different seed diverges."""
    refs = _refs(*(20,) * 5)
    first = RecencyWindow(refs, seed=1234).draw(200)
    again = RecencyWindow(refs, seed=1234).draw(200)
    other = RecencyWindow(refs, seed=1235).draw(200)
    assert first == again
    assert first != other


def test_batches_are_stacked_and_reproducible() -> None:
    """A batch is a dict of stacked tensors, identical under the same seed."""
    refs = _refs(*(20,) * 5)
    window = RecencyWindow(refs, seed=7)
    batch = window.sample(8)
    assert set(batch) == {"features"}
    assert batch["features"].shape == (8, 2)

    replay = RecencyWindow(refs, seed=7)
    assert torch.equal(replay.sample(8)["features"], batch["features"])

    # The generator advances with consumption: the second batch differs.
    assert not torch.equal(window.sample(8)["features"], batch["features"])

    stream = list(RecencyWindow(refs, seed=7).batches(8, 3))
    assert len(stream) == 3
    assert torch.equal(stream[0]["features"], batch["features"])


def test_only_the_most_recent_iterations_are_eligible() -> None:
    """A window of 5 over iterations 10..17 keeps 13..17 and draws only there."""
    refs = _refs(*(10,) * 8)  # iterations 10..17
    window = RecencyWindow(refs, WindowConfig(size=5), seed=0)
    assert [ref.iteration for ref in window.members] == [17, 16, 15, 14, 13]
    assert len(window) == 50
    drawn = {iteration for iteration, _ in window.draw(500)}
    assert drawn <= {13, 14, 15, 16, 17}


def test_iterations_are_ordered_newest_first_regardless_of_input_order() -> None:
    """The caller's list order is not the window's order; the numbers are."""
    refs = _refs(*(5,) * 3)
    shuffled = [refs[1], refs[2], refs[0]]
    window = RecencyWindow(shuffled, seed=0)
    assert [ref.iteration for ref in window.members] == [12, 11, 10]


def test_empty_iterations_are_dropped_and_the_weights_renormalise() -> None:
    """An iteration that produced nothing cannot absorb a share of the batch."""
    refs = _refs(4, 0, 4)  # iteration 11 produced no samples
    window = RecencyWindow(refs, seed=0)
    assert [ref.iteration for ref in window.members] == [12, 10]
    assert window.weights == pytest.approx((1 / 1.8, 0.8 / 1.8))
    assert sum(window.weights) == pytest.approx(1.0)


def test_duplicate_iteration_numbers_are_rejected() -> None:
    """Two refs claiming one iteration number cannot both be drawn from.

    The number is the identity a draw is resolved through, so a duplicate
    gives the pair two shares of the schedule and one source to spend them on:
    the weights describe a window that is not the one being sampled. Caught at
    construction, where the caller still knows which two it merged.
    """
    refs = _refs(4, 4)
    collided = [
        refs[0],
        IterationRef(iteration=refs[0].iteration, source=refs[1].source),
    ]
    with pytest.raises(ValueError, match="duplicate iteration numbers"):
        RecencyWindow(collided, seed=0)


def test_an_entirely_empty_window_is_an_error() -> None:
    """Silently producing empty batches is worse than refusing to start."""
    with pytest.raises(ValueError, match="no iteration with any samples"):
        RecencyWindow(_refs(0, 0), seed=0)
    with pytest.raises(ValueError, match="no iteration with any samples"):
        RecencyWindow([], seed=0)


def test_config_rejects_impossible_schedules() -> None:
    """Sizes, decays and explicit schedules that cannot form a distribution."""
    with pytest.raises(ValueError, match="window size"):
        WindowConfig(size=0)
    with pytest.raises(ValueError, match="decay"):
        WindowConfig(decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        WindowConfig(decay=1.5)
    with pytest.raises(ValueError, match="fewer than the window size"):
        WindowConfig(size=5, weights=(1.0, 0.5))
    with pytest.raises(ValueError, match="positive"):
        WindowConfig(size=2, weights=(1.0, 0.0))
    with pytest.raises(ValueError, match="at least one iteration"):
        recency_weights(0, WindowConfig())


def test_batch_sizes_and_counts_are_validated() -> None:
    """Zero-sized batches and negative counts are caller errors."""
    window = RecencyWindow(_refs(4), seed=0)
    with pytest.raises(ValueError, match="batch_size"):
        window.sample(0)
    with pytest.raises(ValueError, match="count"):
        window.draw(-1)
    with pytest.raises(ValueError, match="count"):
        list(window.batches(2, -1))
    assert window.draw(0) == ()
