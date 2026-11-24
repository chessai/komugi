"""§D5.2's paired bootstrap: an interval around a fit that carries none.

:class:`~komugi.eval.elo.EloResult` carries no dispersion whatsoever
(``elo.py:109-124``), so every interval this campaign publishes has to be
constructed. The pre-registered construction is a **paired bootstrap over pair
units** at ``B = 2000`` replicates, read as a percentile interval.

Three decisions, each of which is a refusal of an easier alternative:

* **The resampling unit is the ``pair_id``**, and both games of a pair are
  resampled together, never split (``records.py:82-87``, audited by
  ``seats.check_paired_books``). They are not independent: they share a book
  prefix and, under §D3, the identical opening position.
* **Resampling is stratified by matchup.** Each matchup's pair units are drawn
  with replacement independently, which preserves every matchup's game count
  and so keeps every replicate's comparison graph connected. An unstratified
  bootstrap over the pooled record list would occasionally drop a bridging
  matchup entirely and produce an unidentified fit for structural reasons that
  have nothing to do with the data.
* **A replicate whose fit does not converge is discarded and counted.**
  ``converged=False`` means the likelihood is unbounded, *not* that the
  optimiser needs more iterations (``elo.py:289-293``), so a clamped rating is
  not a smaller measurement -- it is not a measurement. The discard **rate** is
  returned because a gate whose interval came from 1,400 of 2,000 replicates
  is reporting something other than what it says.

**The direction of the discard bias is pre-registered, not discovered.** The
discards are not missing at random: a replicate is dropped when a resample
drives some matchup toward a sweep and the fit clamps, which is what happens on
the *extreme* replicates -- the ones populating the tails of the statistic.
Dropping them **narrows** the interval, in the direction that makes a gate fire
more readily. §D5.3's condition 5 tolerating a 10% loss is a tolerance for that
specific bias.

The interval level is **pinned here** rather than left to a reader's
assumption: :data:`LOWER_PERCENTILE` and :data:`UPPER_PERCENTILE` are the 2.5th
and 97.5th, i.e. the nominal 95% percentile interval (§6 node 58). No torch --
this is numpy plus :func:`komugi.eval.elo.fit_elo`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from komugi.eval.elo import fit_elo
from komugi.eval.records import GameRecord, MatchSet

REPLICATES: Final[int] = 2000
"""§D5.2's pre-registered ``B``.

A 95% percentile interval reads the 50th and 1,950th order statistics; at
``B = 200`` the endpoints are single observations and the interval's own noise
is comparable to the effect being measured. If the bootstrap's wall clock ever
passes ten minutes the pre-registered response is to aggregate once per
replicate rather than once per fit -- **never** to reduce ``B``.
"""

CONFIDENCE: Final[float] = 0.95
"""The nominal coverage of the percentile interval this module returns."""

LOWER_PERCENTILE: Final[float] = 2.5
"""2.5: the lower endpoint's percentile, pinned rather than assumed (§6 node 58).

Written as the literal, not as ``100.0 * (1.0 - CONFIDENCE) / 2.0``: that
expression evaluates to ``2.500000000000002`` in binary floating point, so the
derived form would publish an endpoint that is not the pre-registered one.
"""

UPPER_PERCENTILE: Final[float] = 97.5
"""97.5: the upper endpoint's percentile, pinned as a literal for the same reason."""

MAX_DISCARD_RATE: Final[float] = 0.10
"""§D5.3 validity condition 5: above this the window is not evaluable."""

RESAMPLE_UNIT: Final[str] = "pair_id"
"""What one bootstrap draw is: the two seat-swapped games of one opening."""

STRATIFIED_BY: Final[str] = "matchup"
"""What the resampling is stratified by, so game counts are preserved."""


@dataclass(frozen=True, slots=True)
class Statistic:
    """One scalar the bootstrap tracks, plus its own convergence verdict.

    ``compute`` returns ``None`` when the replicate must be **discarded** --
    which is how a caller reports ``converged=False`` from its own fits without
    this module having to know how many fits a statistic is made of. A
    replicate is discarded whole, for every statistic, so that all of them are
    computed over the same surviving resamples and their intervals stay
    comparable.
    """

    name: str
    compute: Callable[[MatchSet], float | None]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """One statistic's point estimate, percentile interval and replicate draws."""

    name: str
    point: float | None
    lo: float
    hi: float
    samples: tuple[float, ...]

    @property
    def excludes_zero(self) -> bool:
        """Return whether the interval excludes 0 -- §D5.3's second predicate."""
        return self.lo > 0.0 or self.hi < 0.0


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """The outcome of one paired bootstrap: intervals, discards, wall clock."""

    intervals: Mapping[str, BootstrapInterval]
    replicates: int
    kept: int
    discarded: int
    wall_clock_s: float
    seed: int
    unit: str = RESAMPLE_UNIT
    stratified_by: str = STRATIFIED_BY

    @property
    def discard_rate(self) -> float:
        """Return the fraction of replicates discarded for non-convergence."""
        if self.replicates <= 0:
            return 0.0
        return self.discarded / self.replicates


def percentile_interval(samples: Sequence[float]) -> tuple[float, float]:
    """Return the nominal 95% percentile interval of `samples`.

    The 2.5th and 97.5th percentiles by linear interpolation between order
    statistics -- **not** the 5th and 95th. The level is a pre-registration:
    read at 90% the effective critical value of §D5.3's gate drops from ~95 to
    ~79 Elo at the wide end of its SE chain and the gate fires half again as
    often as the power table says, in the direction §D5.2 names as the biased
    one.

    Raises:
        ValueError: if `samples` is empty.
    """
    if not samples:
        raise ValueError("cannot take a percentile interval of no samples")
    ordered: npt.NDArray[np.float64] = np.sort(
        np.asarray(samples, dtype=np.float64)
    )
    bounds: npt.NDArray[np.float64] = np.asarray(
        np.percentile(ordered, [LOWER_PERCENTILE, UPPER_PERCENTILE]),
        dtype=np.float64,
    )
    return (float(bounds[0]), float(bounds[1]))


def _matchup_key(record: GameRecord) -> tuple[str, str]:
    """Return the unordered matchup a record belongs to, seat-independently."""
    first, second = sorted((record.white, record.black))
    return (first, second)


def pair_units(
    match_set: MatchSet,
) -> dict[tuple[str, str], list[tuple[str, tuple[GameRecord, ...]]]]:
    """Return the resampling units: pair id -> its games, grouped by matchup.

    Ordering is deterministic (matchups sorted, and pairs in first-appearance
    order within a matchup) so that one seed reproduces one bootstrap.
    """
    grouped: dict[tuple[str, str], dict[str, list[GameRecord]]] = {}
    for record in match_set.records:
        pairs = grouped.setdefault(_matchup_key(record), {})
        pairs.setdefault(record.pair_id, []).append(record)
    return {
        matchup: [(pair_id, tuple(games)) for pair_id, games in pairs.items()]
        for matchup, pairs in sorted(grouped.items())
    }


def resample_pairs(
    match_set: MatchSet,
    rng: np.random.Generator,
    *,
    stratify_by_matchup: bool = True,
) -> MatchSet:
    """Return one bootstrap replicate of `match_set`, drawn over pair units.

    Both games of a drawn pair are emitted together, always, so a replicate can
    never hold one game of a pair alone -- the seat-balance property the whole
    ladder's seat-free likelihood rests on (``elo.py:22-29``).

    With `stratify_by_matchup` (the default, and what §D5.2 pre-registers) each
    matchup contributes exactly as many pair units as it originally had.
    """
    return _draw(pair_units(match_set), rng, stratify_by_matchup)


def _draw(
    units: dict[tuple[str, str], list[tuple[str, tuple[GameRecord, ...]]]],
    rng: np.random.Generator,
    stratify_by_matchup: bool,
) -> MatchSet:
    """Draw one replicate from pre-grouped pair units.

    Split out of :func:`resample_pairs` so :func:`pair_bootstrap` groups the
    record set **once** rather than once per replicate: the grouping is a
    Python pass over every game, and §D5.2 budgets the bootstrap's wall clock
    rather than assuming it cheap.
    """
    drawn: list[GameRecord] = []
    if stratify_by_matchup:
        for pairs in units.values():
            indices = rng.integers(0, len(pairs), size=len(pairs))
            for index in indices:
                drawn.extend(pairs[int(index)][1])
        return MatchSet.of(drawn)
    flat = [pair for pairs in units.values() for pair in pairs]
    indices = rng.integers(0, len(flat), size=len(flat))
    for index in indices:
        drawn.extend(flat[int(index)][1])
    return MatchSet.of(drawn)


def fit_statistic(
    name: str,
    *,
    anchor: str,
    records: Callable[[MatchSet], MatchSet],
    value: Callable[[Mapping[str, float]], float],
) -> Statistic:
    """Return a :class:`Statistic` that refits `fit_elo` and reads a value off it.

    ``records`` selects this statistic's own sub-set of a replicate (§D5.1's
    in-family / cross-family / cross-family-without-the-external-anchor-leg
    partition), and
    ``value`` reads the scalar -- a difference of two ratings, which is
    invariant to the sub-fit's own anchor. A non-converged fit, a sub-set that
    lost a player, or an empty sub-set all report ``None``: the replicate is
    discarded rather than contributing a clamp (``elo.py:289-293``).
    """

    def compute(replicate: MatchSet) -> float | None:
        selected = records(replicate)
        if len(selected) == 0:
            return None
        players = selected.players()
        pinned = anchor if anchor in players else players[0]
        result = fit_elo(selected, anchor=pinned)
        if not result.converged:
            return None
        try:
            return value(result.ratings)
        except KeyError:
            return None

    return Statistic(name=name, compute=compute)


def pair_bootstrap(
    match_set: MatchSet,
    statistics: Sequence[Statistic],
    *,
    replicates: int = REPLICATES,
    seed: int,
    stratify_by_matchup: bool = True,
) -> BootstrapResult:
    """Run §D5.2's paired bootstrap and return the percentile intervals.

    Each replicate resamples every matchup's pair units with replacement, then
    computes every statistic on the resampled records. If **any** statistic
    reports ``None`` the whole replicate is discarded and counted, so all the
    returned intervals are read off the same surviving replicates.

    Args:
        match_set: the games to resample. Pair units are read from it directly.
        statistics: the scalars to interval, each carrying its own fits.
        replicates: ``B``; defaults to §D5.2's pre-registered 2,000.
        seed: seeds a local generator. Required -- there is no global-randomness
            path into this function, and the epoch record stores the seed.
        stratify_by_matchup: keep each matchup's game count (the default and
            the pre-registration); ``False`` is provided only so the stratified
            claim is testable against its alternative.

    Returns:
        A :class:`BootstrapResult` whose ``wall_clock_s`` is measured, because
        §D5.2 budgets the bootstrap rather than assuming it cheap.

    Raises:
        ValueError: if `replicates` is not positive, `statistics` is empty, or
            every replicate was discarded (no interval can be reported).
    """
    if replicates <= 0:
        raise ValueError(f"replicates must be positive, got {replicates!r}")
    if not statistics:
        raise ValueError("pair_bootstrap needs at least one statistic")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    units = pair_units(match_set)
    samples: dict[str, list[float]] = {statistic.name: [] for statistic in statistics}
    discarded = 0
    for _ in range(replicates):
        replicate = _draw(units, rng, stratify_by_matchup)
        drawn: dict[str, float] = {}
        for statistic in statistics:
            computed = statistic.compute(replicate)
            if computed is None:
                break
            drawn[statistic.name] = computed
        if len(drawn) != len(statistics):
            discarded += 1
            continue
        for name, sampled in drawn.items():
            samples[name].append(sampled)
    kept = replicates - discarded
    if kept == 0:
        raise ValueError(
            f"all {replicates} bootstrap replicates were discarded for "
            f"non-convergence; there is no interval to report, and a clamped "
            f"rating is not a smaller measurement (elo.py:289-293)"
        )
    point = _point_estimates(match_set, statistics)
    intervals: dict[str, BootstrapInterval] = {}
    for statistic in statistics:
        drawn_samples = samples[statistic.name]
        lo, hi = percentile_interval(drawn_samples)
        intervals[statistic.name] = BootstrapInterval(
            name=statistic.name,
            point=point[statistic.name],
            lo=lo,
            hi=hi,
            samples=tuple(drawn_samples),
        )
    return BootstrapResult(
        intervals=intervals,
        replicates=replicates,
        kept=kept,
        discarded=discarded,
        wall_clock_s=time.perf_counter() - started,
        seed=seed,
    )


def _point_estimates(
    match_set: MatchSet, statistics: Sequence[Statistic]
) -> dict[str, float | None]:
    """Return each statistic's value on the observed data, for the record."""
    return {
        statistic.name: statistic.compute(match_set) for statistic in statistics
    }
