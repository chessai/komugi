"""The recency-weighted replay window over recent self-play iterations.

Design §4: "5 iterations with recency weighting (v1's flat window of 3
worked; recency weighting is the standard refinement)". This module is that
sentence, made explicit and testable.

The weight schedule
-------------------
The window keeps the ``size`` most recent iterations. Let ``k = 0`` be the
newest kept iteration, ``k = 1`` the one before it, and so on. The unnormalised
weight is geometric::

    raw[k] = decay ** k          (decay = 0.8 by default)

normalised to sum to 1 over the iterations actually present. With the default
five-iteration window that is::

    raw   = [1.0, 0.8, 0.64, 0.512, 0.4096]     (sum 3.3616)
    p     = [0.2975, 0.2380, 0.1904, 0.1523, 0.1218]

so the newest iteration supplies just under 30% of every batch and the oldest
just over 12% -- a gentle tilt, not a truncation. ``decay = 1.0`` reproduces
v1's flat window exactly, which is what makes this a strict generalisation of
the thing that worked. An explicit ``weights`` tuple overrides the schedule
entirely (newest first) for experiments that want a non-geometric tilt.

**These probabilities are per-draw over iterations, not per-sample.** A draw
picks an iteration by ``p``, then a sample uniformly within it. The
consequence is worth stating because it is a real design choice: an iteration
that produced half as many games still contributes its full ``p`` share, its
individual samples simply appear more often. The alternative -- weighting each
*sample* by its iteration's weight -- makes a short iteration contribute less
in total, which couples the replay mix to self-play throughput jitter. We
would rather the mix be a property of the schedule alone.

Iterations holding zero samples are dropped before normalisation, so an empty
newest iteration cannot silently swallow 30% of every batch.

Determinism
-----------
Sampling runs off a :class:`numpy.random.Generator` seeded at construction
(PCG64). Two windows built from the same iterations with the same seed and
config yield identical draws, in the same order, forever -- the property that
lets a training run be replayed. The generator is owned, never shared, and
never reseeded, so drawing a batch advances only this window's stream.

This module knows nothing about shards or nets: an iteration is anything with
``__len__`` and integer ``__getitem__`` returning a sample dict (see
:class:`SampleSource`), which :class:`~komugi.train.shards.ShardReader`
satisfies and a list of dicts satisfies just as well.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from komugi.train.shards import stack_samples


class SampleSource(Protocol):
    """One iteration's worth of samples, addressable by index.

    Structural on purpose: :class:`~komugi.train.shards.ShardReader` is the
    production implementation, a ``list[dict[str, Tensor]]`` is the one tests
    use, and neither had to be told about the other.
    """

    def __len__(self) -> int:
        """Return how many samples this source holds."""
        ...

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample at `index`."""
        ...


@dataclass(frozen=True, slots=True)
class IterationRef:
    """A self-play iteration in the window: its number and its samples.

    ``iteration`` orders the window (higher is newer) and is what the sampler
    reports back, so a test or a telemetry hook can attribute a draw to the
    generation it came from.
    """

    iteration: int
    source: SampleSource


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """The replay window's shape and recency tilt.

    ``size``
        How many of the most recent iterations are eligible. Design default 5.
    ``decay``
        Geometric factor per iteration of age. 0.8 by default; 1.0 is a flat
        window.
    ``weights``
        Optional explicit schedule, newest first, overriding ``decay``. May be
        longer than the window (the tail is unused) but not shorter, since a
        short schedule would leave an eligible iteration with no weight at all.
    """

    size: int = 5
    decay: float = 0.8
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Reject sizes, decays and schedules that cannot produce a distribution."""
        if self.size < 1:
            raise ValueError(f"window size must be >= 1, got {self.size}")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if self.weights is not None:
            if len(self.weights) < self.size:
                raise ValueError(
                    f"explicit weights have {len(self.weights)} entries, "
                    f"fewer than the window size {self.size}"
                )
            if any(weight <= 0.0 for weight in self.weights):
                raise ValueError(f"explicit weights must be positive, got {self.weights}")


def recency_weights(count: int, config: WindowConfig) -> tuple[float, ...]:
    """Return the normalised draw probabilities for `count` iterations, newest first.

    `count` is the number of *non-empty* iterations in the window, which may be
    below ``config.size``; the schedule is truncated and renormalised, never
    padded. Raises ``ValueError`` for ``count <= 0``.
    """
    if count <= 0:
        raise ValueError(f"need at least one iteration to weight, got {count}")
    if config.weights is not None:
        raw = tuple(config.weights[:count])
    else:
        raw = tuple(config.decay**age for age in range(count))
    total = sum(raw)
    return tuple(weight / total for weight in raw)


class RecencyWindow:
    """Samples batches from the most recent iterations, tilted toward the newest.

    Construction sorts the supplied iterations by ``iteration`` descending,
    keeps the first ``config.size`` that hold at least one sample, and computes
    the draw distribution over exactly those. Raises ``ValueError`` if nothing
    is left, because an empty window silently producing empty batches is how a
    trainer spends an hour learning nothing, and equally if two refs claim the
    same iteration number -- that number is the identity a draw is resolved
    through, so a duplicate would give the pair two shares of the schedule and
    one source to spend them on.
    """

    def __init__(
        self,
        iterations: Sequence[IterationRef],
        config: WindowConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        """Build the window over `iterations` and seed its generator."""
        self._config = config if config is not None else WindowConfig()
        numbers = [ref.iteration for ref in iterations]
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"recency window has duplicate iteration numbers: {duplicates}. "
                "An iteration number identifies a source, so two refs sharing "
                "one make the window's weights and its sampling disagree: the "
                "schedule counts both, while :meth:`sample` resolves a draw "
                "through a number-keyed dict that keeps only the last."
            )
        ordered = sorted(iterations, key=lambda ref: ref.iteration, reverse=True)
        members: list[IterationRef] = []
        for ref in ordered:
            if len(members) == self._config.size:
                break
            if len(ref.source) > 0:
                members.append(ref)
        if not members:
            raise ValueError("recency window has no iteration with any samples")
        self._members = tuple(members)
        self._weights = recency_weights(len(self._members), self._config)
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def config(self) -> WindowConfig:
        """Return the config in force."""
        return self._config

    @property
    def seed(self) -> int:
        """Return the seed the generator was created with."""
        return self._seed

    @property
    def members(self) -> tuple[IterationRef, ...]:
        """Return the iterations in the window, newest first."""
        return self._members

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the draw probability of each member, newest first, summing to 1."""
        return self._weights

    def __len__(self) -> int:
        """Return the total number of samples reachable through the window."""
        return sum(len(ref.source) for ref in self._members)

    def draw(self, count: int) -> tuple[tuple[int, int], ...]:
        """Draw `count` ``(iteration, sample_index)`` pairs, with replacement.

        The raw sampling primitive, exposed because "which iteration did this
        batch come from" is the only thing that makes the weight schedule
        observable -- both to a test and to iteration-mix telemetry.
        Replacement is intended: a window is resampled many times per
        iteration and drawing without it would make the tilt drift as the
        epoch progressed.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return ()
        slots = self._rng.choice(len(self._members), size=count, p=self._weights)
        picks: list[tuple[int, int]] = []
        for slot in slots:
            member = self._members[int(slot)]
            row = int(self._rng.integers(0, len(member.source)))
            picks.append((member.iteration, row))
        return tuple(picks)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw `batch_size` samples and stack them into one batch of tensors."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        by_iteration = {ref.iteration: ref.source for ref in self._members}
        samples = [
            by_iteration[iteration][row] for iteration, row in self.draw(batch_size)
        ]
        return stack_samples(samples)

    def batches(
        self, batch_size: int, count: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield `count` independent batches of `batch_size` stacked samples.

        A generator, so a training loop can stop early without paying for
        batches it never used; the draws for a batch happen when that batch is
        requested, which keeps the generator stream aligned with consumption.
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        for _ in range(count):
            yield self.sample(batch_size)
