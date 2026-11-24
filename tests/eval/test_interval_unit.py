"""§D5.2 unit nodes: the paired bootstrap and the level it is read at.

Design: ``docs/design/kom-011d-ladder-pool.md`` (rev 3), §6 nodes 19, 20, 21
and 58. ``EloResult`` carries no dispersion at all (``elo.py:109-124``), so
every interval this campaign publishes comes from here, and the three
properties that make it an interval rather than a number -- the pair unit, the
discard accounting, and the percentile level -- each get a node.

Counts are **pinned in the design**, not chosen here: node 20 is 20 trials x
200 replicates over a 4-entrant, 4-matchup planted graph at 32 pairs per
matchup, with a realised-coverage band of ``[0.85, 1.00]``. It is a smoke over
the machinery and it says so: 20 trials cannot separate a realised 0.90 from
0.95 at any useful confidence, which is exactly why the *level* is pinned
separately by node 58 and why U5's real coverage experiment is a committed
artefact (``gungi-az-runs/scripts/pool_coverage.py``) rather than a test node.
"""

from __future__ import annotations

import numpy as np
import pytest

from komugi.eval.elo import fit_elo
from komugi.eval.interval import (
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    BootstrapResult,
    Statistic,
    fit_statistic,
    pair_bootstrap,
    pair_units,
    percentile_interval,
    resample_pairs,
)
from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet
from komugi.eval.synth import simulate_match_set

_TRIALS: int = 20
"""Pinned by §6 node 20 (rev 2's re-scoping, F11): a smoke, not U5's answer."""

_REPLICATES: int = 200
"""Pinned by §6 node 20. The published bootstrap runs at ``B = 2000``."""

_PAIRS_PER_MATCHUP: int = 32
"""Pinned by §6 node 20: 64 games per matchup on the planted graph."""

_COVERAGE_BAND: tuple[float, float] = (0.85, 1.00)
"""Pinned by §6 node 20 (rev 3 narrowed it around the nominal 0.95)."""

_WIDTH_TRIALS: int = 5
"""Trials for F4's interval-width comparison: two bootstraps each, not twenty."""

_PLANTED: dict[str, float] = {"e0": 0.0, "e1": 120.0, "e2": 260.0, "e3": 420.0}
"""Four entrants -- the node's own count -- spanning a realistic 420 Elo."""

_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("e0", "e1"),
    ("e1", "e2"),
    ("e2", "e3"),
    ("e0", "e3"),
)
"""Four matchups: the chain plus one long link, i.e. a *sparse* graph.

A round robin is what ``simulate_match_set`` used to be able to plant and is
not what the pool plays (§D2.2), and a coverage claim measured on a complete
graph is a claim about a different instrument.
"""

_DRAW_PARAM: float = 0.9
_SEED: int = 20260908


def _trial_set(trial: int) -> MatchSet:
    """Return one trial's planted games over the sparse schedule."""
    return simulate_match_set(
        _PLANTED,
        seed=_SEED + trial,
        draw_param=_DRAW_PARAM,
        # No seat advantage here on purpose: this node measures the
        # bootstrap's coverage of the estimator's own sampling error, not the
        # seat mis-specification test_elo_planted.py already exercises.
        seat_advantage=0.0,
        pairs_per_matchup=_PAIRS_PER_MATCHUP,
        schedule=_SCHEDULE,
    )


def _delta_statistic(first: str, second: str) -> Statistic:
    """Return the statistic ``r(second) - r(first)``, refit per replicate."""
    return fit_statistic(
        f"{second}-{first}",
        anchor=first,
        records=lambda match_set: match_set,
        value=lambda ratings: ratings[second] - ratings[first],
    )


def _sweep_pairs(
    winner: str, loser: str, *, pairs: int, decisive: int
) -> list[GameRecord]:
    """Return `pairs` seat-swapped pairs, `decisive` of them 2-0 to `winner`.

    A pendant link at this shape is how a **known** discard rate is planted:
    a replicate is dropped exactly when all of its drawn pair units are the
    decisive ones, because the winner then scores 100% against the whole rest
    of the pool and the likelihood is unbounded (``elo.py:70-77``).
    """
    records: list[GameRecord] = []
    for index in range(pairs):
        pair_id = f"{winner}:{loser}:{index:05d}"
        result = WHITE_WIN if index < decisive else DRAW
        records.append(
            GameRecord(
                white=winner,
                black=loser,
                result=result,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=loser,
                black=winner,
                result=BLACK_WIN if index < decisive else DRAW,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
    return records


def _discard_prone_set(trial: int) -> MatchSet:
    """Return a planted set whose sweep-prone pendant hangs off the statistic.

    30 of 32 pairs decisive gives ``(30/32)**32 ~ 0.13`` -- a discard rate in
    the neighbourhood of §D5.3's 10% condition, planted rather than hoped for:
    a replicate is dropped exactly when every pair unit it draws is one of the
    decisive ones, because the loser then scores 0% against the whole rest of
    the pool and the likelihood is unbounded (``elo.py:70-77``).

    **The pendant hangs off ``e1``, which is an endpoint of the statistic**
    (``e1 - e0``), rather than off ``e2`` as rev 1 of this fixture did (F4):
    §D5.2's registered claim is that the discards **are** the extreme
    replicates *of the statistic*, and a fixture measuring coverage across
    links the discards cannot reach is measuring the claim's complement.

    What that siting does **not** buy, said here so a later reader does not
    over-read the node: the player that saturates is the **pendant**, and
    ``elo.py``'s likelihood equation for ``e1`` is unchanged by a link ``e1``
    wins outright -- the pendant's own equation absorbs it -- so ``e1 - e0``
    itself is still close to missing-at-random over the discards. That is the
    review's own P2 amendment (*"condition 5 is a weak detector"*), and it is
    why the width half below is read on the statistic the clamps actually land
    in.
    """
    base = simulate_match_set(
        {name: _PLANTED[name] for name in ("e0", "e1", "e2")},
        seed=_SEED + trial,
        draw_param=_DRAW_PARAM,
        seat_advantage=0.0,
        pairs_per_matchup=_PAIRS_PER_MATCHUP,
        schedule=(("e0", "e1"), ("e1", "e2")),
    )
    return MatchSet.of(
        [*base.records, *_sweep_pairs("e1", "e3", pairs=32, decisive=30)]
    )


def _clamped_delta_statistic(first: str, second: str) -> Statistic:
    """Return ``r(second) - r(first)``, **keeping** the replicates a fit clamps.

    The counterfactual half of F4's direction claim: identical to
    :func:`_delta_statistic` except that it never reports ``None``, so
    :func:`pair_bootstrap` discards nothing and the interval is read off all
    ``B`` replicates rather than off the survivors. It is not a proposal --
    ``elo.py:289-293``'s clamp is not a measurement and node 21 is what keeps
    it out of the published interval -- it exists only so the width the
    discards would have produced is a number this file can compare against.
    """

    def compute(replicate: MatchSet) -> float | None:
        result = fit_elo(replicate, anchor=first)
        return result.ratings[second] - result.ratings[first]

    return Statistic(name=f"{second}-{first}", compute=compute)


def _mean_widths(sets: list[MatchSet], *, first: str, second: str) -> tuple[float, float]:
    """Return the mean interval width ``(discards dropped, discards retained)``.

    Both runs use the same seed, and :func:`pair_bootstrap` draws its replicate
    before it computes any statistic, so the two runs see the **same** resampled
    record sets in the same order: the only difference between the two widths is
    which replicates reach the percentile.
    """
    dropped: list[float] = []
    retained: list[float] = []
    for trial, match_set in enumerate(sets):
        seed = _SEED + 2000 + trial
        surviving = pair_bootstrap(
            match_set,
            [_delta_statistic(first, second)],
            replicates=_REPLICATES,
            seed=seed,
        ).intervals[f"{second}-{first}"]
        everything = pair_bootstrap(
            match_set,
            [_clamped_delta_statistic(first, second)],
            replicates=_REPLICATES,
            seed=seed,
        ).intervals[f"{second}-{first}"]
        dropped.append(surviving.hi - surviving.lo)
        retained.append(everything.hi - everything.lo)
    return (float(np.mean(dropped)), float(np.mean(retained)))


def _split_pairs(first: str, second: str, *, pairs: int, wins: int) -> list[GameRecord]:
    """Return `pairs` seat-swapped pairs, `wins` of them 2-0 to `first`.

    **No draws at all**, which is load-bearing: with ``total_draws == 0`` the
    fitted ``nu`` is 0 (``elo.py:177-182``) and the MM iteration is exactly the
    one ``test_divergence.py``'s condition-5 fixture drives. A helper that
    padded with draws would fit a different likelihood.
    """
    records: list[GameRecord] = []
    for index in range(pairs):
        pair_id = f"{first}:{second}:{index:05d}"
        first_seat = WHITE_WIN if index < wins else BLACK_WIN
        records.append(
            GameRecord(
                white=first,
                black=second,
                result=first_seat,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=second,
                black=first,
                result=WHITE_WIN - first_seat,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
    return records


def _sweeping_pendant_set() -> MatchSet:
    """Return the graph ``test_divergence.py``'s condition-5 case fits, id for id.

    Six 128-game links at exactly 32-32 between two hubs (``a``, ``b``) and
    three opponents (``c``, ``d``, ``f``), plus a pendant ``e`` that wins
    **all** 64 games against each of ``c`` and ``d``. That is
    ``divergence._case_discard_rate``'s **cross** record set matchup for
    matchup: the same eight links, the same game counts, the same scores, and
    -- because the player list is sorted and ``a`` is the anchor -- the same
    index layout, so ``_aggregate``'s ``games`` matrix and ``score`` vector are
    the ones ``fit_elo`` sees there.

    Built rather than borrowed on purpose: this file may not import another
    test module's private fixtures, and the property under test is a property
    of :func:`~komugi.eval.elo.fit_elo` on this *shape*, not of the pool's ids.
    """
    records: list[GameRecord] = []
    for hub in ("a", "b"):
        for opponent in ("c", "d", "f"):
            records.extend(_split_pairs(hub, opponent, pairs=64, wins=32))
    for opponent in ("c", "d"):
        records.extend(_split_pairs("e", opponent, pairs=32, wins=32))
    return MatchSet.of(records)


def _coverage(
    sets: list[MatchSet], *, first: str, second: str, planted: float
) -> tuple[float, float]:
    """Return ``(coverage, mean discard rate)`` over the planted trials."""
    covered = 0
    discard_rates: list[float] = []
    for trial, match_set in enumerate(sets):
        result: BootstrapResult = pair_bootstrap(
            match_set,
            [_delta_statistic(first, second)],
            replicates=_REPLICATES,
            seed=_SEED + 1000 + trial,
        )
        interval = result.intervals[f"{second}-{first}"]
        if interval.lo <= planted <= interval.hi:
            covered += 1
        discard_rates.append(result.discard_rate)
    return (covered / len(sets), float(np.mean(discard_rates)))


# --------------------------------------------------------------------------
# §6 node 19
# --------------------------------------------------------------------------
def test_the_bootstrap_resamples_pairs_not_games() -> None:
    """§D5.2: the resampling unit is the ``pair_id``, never a single game."""
    # Red against resampling the pooled record list -- which splits pairs (the
    # two games of a pair share a book prefix and, under §D3, the identical
    # opening, so they are not independent draws) and which would also let a
    # replicate drop a bridging matchup entirely.
    match_set = _trial_set(0)
    rng = np.random.default_rng(11)
    replicate = resample_pairs(match_set, rng)

    counts: dict[str, int] = {}
    seats: dict[str, set[tuple[str, str]]] = {}
    for record in replicate.records:
        counts[record.pair_id] = counts.get(record.pair_id, 0) + 1
        seats.setdefault(record.pair_id, set()).add(record.seats)
    assert counts
    for pair_id, count in counts.items():
        assert count % 2 == 0, (pair_id, count)
        assert len(seats[pair_id]) == 2, pair_id

    # Stratified: every matchup keeps its own game count, so no replicate can
    # lose a bridging link for structural reasons.
    original = pair_units(match_set)
    drawn = pair_units(replicate)
    assert set(drawn) == set(original)
    for matchup, pairs in original.items():
        assert sum(len(games) for _, games in drawn[matchup]) == sum(
            len(games) for _, games in pairs
        )
    assert len(replicate) == len(match_set)


# --------------------------------------------------------------------------
# §6 node 20
# --------------------------------------------------------------------------
def test_planted_elo_interval_coverage_is_a_smoke_at_pinned_counts() -> None:
    """A smoke over the machinery at the design's own pinned counts (F11)."""
    # Red against an interval that is not an interval: a point estimate
    # returned as (x, x), an interval computed from the *original* set rather
    # than from the replicates, or a resampler with no randomness at all --
    # each of which drives realised coverage to ~0 and out of the band.
    assert len(_PLANTED) == 4
    assert len(_SCHEDULE) == 4
    assert _TRIALS == 20
    assert _REPLICATES == 200
    assert _PAIRS_PER_MATCHUP == 32

    planted = _PLANTED["e1"] - _PLANTED["e0"]
    coverage, discards = _coverage(
        [_trial_set(trial) for trial in range(_TRIALS)],
        first="e0",
        second="e1",
        planted=planted,
    )
    low, high = _COVERAGE_BAND
    assert low <= coverage <= high, (coverage, discards)

    # F20's direction, at a planted ~10% discard rate, on a fixture whose
    # sweep-prone pendant now hangs off `e1` -- an endpoint of the statistic --
    # rather than off `e2` (F4). §D5.2 registers the discards as the extreme
    # replicates, so dropping them narrows the interval and biases the gate
    # toward firing; the condition 5 tolerance is a tolerance for exactly that,
    # and what must not happen is coverage falling *below* the band while the
    # rate stays legal.
    prone = [_discard_prone_set(trial) for trial in range(_TRIALS)]
    with_discards, realised = _coverage(
        prone, first="e0", second="e1", planted=planted
    )
    # The rate itself, pinned rather than merely non-zero: 30 of 32 pairs
    # decisive is (30/32)**32 ~ 0.13, so a fixture that had drifted to a
    # near-zero or a near-total discard rate would no longer be measuring the
    # regime §D5.3's 10% condition is written for.
    assert 0.05 < realised < 0.20, realised
    assert low <= with_discards <= high, (with_discards, realised)

    # And the direction claim itself, made measurable rather than asserted:
    # the interval read off the survivors is not WIDER than the one that keeps
    # every replicate. Same sets, same seeds, same draws -- pair_bootstrap
    # resamples before it computes any statistic -- so the surviving samples
    # are a strict *subset* of the retained ones and the only difference
    # between the two widths is which replicates reached the percentile.
    #
    # Read on `e3 - e0`, not on `e1 - e0`, and that is a deliberate deviation
    # from the review's wording. On this estimator a replicate is discarded
    # when the resampled pendant is an exact sweep, and the player that then
    # clamps is the PENDANT: `elo.py`'s likelihood equation for `e1` is
    # unchanged by a link it wins outright (the pendant's own equation absorbs
    # it), so the discarded replicates carry an ordinary `e1 - e0` and are
    # missing-at-random with respect to it -- the review's own P2 amendment
    # says exactly this. A width comparison there would be a coin flip written
    # as an assertion. `e3 - e0` is the statistic the discards *do* touch, and
    # on it the direction is a measurement: the dropped replicates are the
    # ~6949-Elo clamps, i.e. the extreme tail, and keeping them widens the
    # interval by orders of magnitude. Five trials rather than twenty because
    # this half runs the bootstrap twice per trial and §6 node 20 pins the
    # coverage counts, not this comparison's.
    dropped_width, retained_width = _mean_widths(
        prone[:_WIDTH_TRIALS], first="e0", second="e3"
    )
    assert dropped_width < retained_width, (dropped_width, retained_width)


# --------------------------------------------------------------------------
# §6 node 21
# --------------------------------------------------------------------------
def test_a_non_converged_replicate_is_discarded_and_counted() -> None:
    """§D5.2 step 3: a clamp is not a smaller measurement, so it is dropped."""
    # Red against an implementation that keeps clamped ratings: `fit_elo`
    # returns a *number* for an unbounded likelihood (it clamps at
    # _MAX_LOG_GAMMA and sets converged=False, elo.py:289-293), so a caller
    # that reads `.ratings` without reading `.converged` silently mixes ~6949
    # Elo clamps into the tails of D* and reports a wider interval as though it
    # were data.
    match_set = _discard_prone_set(0)
    result = pair_bootstrap(
        match_set,
        [_delta_statistic("e0", "e1")],
        replicates=_REPLICATES,
        seed=_SEED,
    )
    assert result.discarded > 0
    assert result.kept == _REPLICATES - result.discarded
    assert 0.0 < result.discard_rate < 1.0
    interval = result.intervals["e1-e0"]
    assert len(interval.samples) == result.kept
    assert interval.lo < interval.hi
    assert all(abs(sample) < 5000.0 for sample in interval.samples)
    assert result.wall_clock_s >= 0.0
    assert result.unit == "pair_id"
    assert result.stratified_by == "matchup"


def test_a_full_sweep_pendant_does_not_fail_the_observed_fit_but_discards_resamples() -> None:
    """The discard mechanism, stated where §D5.3's condition 5 relies on it (F2).

    ``test_divergence.py``'s condition-5 case plants a pendant that sweeps
    **100%** of two links and then asserts the window reaches condition 5 --
    which it can only do if the *observed* fit converges while a substantial
    fraction of its resamples do not. That is a non-obvious property of
    ``elo.py``'s MM iteration and it is asserted nowhere: ``_initial_theta``
    clips a 100%-scoring player at ``_MAX_LOG_GAMMA`` (``elo.py:194-198``), the
    strength step is stationary there, and whether the post-loop saturation
    check (``elo.py:291-292``) fires turns on whether the re-anchoring's
    ``log(new_gamma_anchor)`` residual carries the iterate back under the cap.
    So the same fixture shape can converge on the observed data and fail on a
    resample, and it is that difference the discard rate is made of.

    Red against the two readings that would make the divergence case pass for
    the wrong reason: an ``elo.py`` that reports a swept pendant as
    ``converged=False`` on the observed data (the window would then name
    condition **1**, never condition 5) and one whose replicates never discard
    at all (the rate would be 0 and condition 5 would be unreachable from any
    fixture).
    """
    swept = _sweeping_pendant_set()

    # The pendant really is a 100% sweep, so what follows is about the
    # saturating case and not about a nearly-saturating one.
    pendant_games = [record for record in swept.records if "e" in record.seats]
    assert len(pendant_games) == 128
    assert all(record.winner == "e" for record in pendant_games)

    observed = fit_elo(swept, anchor="a")
    assert observed.converged, "the observed fit is what condition 1 reads"
    assert observed.ratings["e"] > observed.ratings["a"]

    result = pair_bootstrap(
        swept,
        [_delta_statistic("a", "b")],
        replicates=_REPLICATES,
        seed=_SEED,
    )
    assert 0.0 < result.discard_rate < 1.0, result.discard_rate
    assert result.kept == _REPLICATES - result.discarded


# --------------------------------------------------------------------------
# §6 node 58
# --------------------------------------------------------------------------
def test_the_interval_is_the_nominal_95_percent_percentile_interval() -> None:
    """§D5.2/M3: the endpoints are the 2.5th and 97.5th percentiles, pinned."""
    # Red against the 5th/95th, which no other node can see: node 25's planted
    # 120-vs-20 divergence fires under either level, node 40's smoke sits where
    # the threshold binds rather than the interval, and node 20's 20-trial band
    # cannot separate 0.90 from 0.95. The mutant bites at sigma ~ 48, where the
    # effective critical value drops 94.7 -> 79 and power at a true |D| = 50
    # goes 0.18 -> 0.27: the gate firing half again as often as pre-registered.
    assert (LOWER_PERCENTILE, UPPER_PERCENTILE) == (2.5, 97.5)

    # Known quantiles, computed from the order statistics by hand rather than
    # by calling the same numpy function the implementation calls: with 1000
    # samples 0..999, the p-th percentile by linear interpolation is
    # p/100 * (n - 1).
    samples = [float(value) for value in range(1000)]
    lo, hi = percentile_interval(samples)
    assert abs(lo - 24.975) < 1e-9
    assert abs(hi - 974.025) < 1e-9
    assert abs(lo - 49.95) > 1.0  # the 5th percentile: the mutant
    assert abs(hi - 949.05) > 1.0  # the 95th percentile: the mutant

    # And through the real path: the returned endpoints are those percentiles
    # of the replicate draws the same call returned.
    result = pair_bootstrap(
        _trial_set(0),
        [_delta_statistic("e0", "e1")],
        replicates=64,
        seed=_SEED,
    )
    interval = result.intervals["e1-e0"]
    expected = percentile_interval(list(interval.samples))
    assert (interval.lo, interval.hi) == pytest.approx(expected, abs=1e-12)
    assert len(interval.samples) == result.kept
    assert interval.point is not None
