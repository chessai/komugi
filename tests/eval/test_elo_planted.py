"""The planted-Elo recovery test: does the instrument read what is actually there?

Every strength claim komugi makes will be read off :func:`fit_elo`, so the
question this file answers is not "does the optimiser converge" but "does it
recover ratings that were *planted*, from data whose truth is known". Games
are sampled from the Davidson model with a seat advantage the fit does not
model, which is exactly the mis-specification a real ladder will have; the
tolerance below is therefore a statement about the whole pipeline, not just
the arithmetic.

Sizing is deliberate rather than round. Under this design the standard error
of a fitted rating difference is ~10 Elo: the per-game score variance runs
from 0.16 between neighbours down to 0.04 across the full 650-Elo gap, and
1200 games per matchup over a complete six-player graph turns that into an
effective resistance of ~0.0034 in log-strength units. The +/-40 Elo bar is
therefore about four sigma out. A fixed seed makes the test deterministic;
that margin is what makes it non-flaky by design rather than by luck.

Three levels of check:

* recovery -- planted ratings back within +/-40 Elo, order exactly preserved;
* closed form -- a two-player case where the MLE is solvable by hand, so the
  fit is checked against algebra rather than against itself;
* determinism -- same seed, same numbers, bit for bit.
"""

from __future__ import annotations

import itertools
import math

import pytest

from komugi.eval.elo import EloResult, fit_elo, log_likelihood
from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet
from komugi.eval.seats import check_paired_books
from komugi.eval.synth import simulate_match_set

_ANCHOR: str = "e0"
_ANCHOR_ELO: float = 1500.0

_PLANTED: dict[str, float] = {
    "e0": 1500.0,
    "e1": 1630.0,
    "e2": 1770.0,
    "e3": 1900.0,
    "e4": 2030.0,
    "e5": 2150.0,
}
"""Six players spanning 650 Elo -- the ladder's realistic dynamic range."""

_SEED: int = 20260821
_DRAW_PARAM: float = 0.9
_SEAT_ADVANTAGE: float = 35.0
_PAIRS_PER_MATCHUP: int = 600
_MATCHUPS: int = 15  # 6 players choose 2
_EXPECTED_GAMES: int = _MATCHUPS * _PAIRS_PER_MATCHUP * 2

_TOLERANCE_ELO: float = 40.0

_CACHE: dict[int, MatchSet] = {}


def _generate(seed: int) -> MatchSet:
    """Sample the standard planted ladder under `seed`, without caching."""
    return simulate_match_set(
        _PLANTED,
        seed=seed,
        draw_param=_DRAW_PARAM,
        seat_advantage=_SEAT_ADVANTAGE,
        pairs_per_matchup=_PAIRS_PER_MATCHUP,
    )


def _planted_match_set() -> MatchSet:
    """Return the canonical planted ladder, generated once per test session.

    Cached only to keep the suite quick: the records are immutable, and
    :func:`test_generation_is_deterministic_in_the_seed` goes around the cache
    so the caching cannot be what makes determinism look true.
    """
    cached = _CACHE.get(_SEED)
    if cached is None:
        cached = _generate(_SEED)
        _CACHE[_SEED] = cached
    return cached


def _fit() -> EloResult:
    """Fit the canonical planted ladder with the default hyper-parameters."""
    return fit_elo(_planted_match_set(), anchor=_ANCHOR, anchor_elo=_ANCHOR_ELO)


def _two_player_records(wins: int, draws: int, losses: int) -> MatchSet:
    """Return ``wins``/``draws``/``losses`` games for ``a`` against ``b``.

    Seats alternate, so results are translated into a's perspective as they
    are laid down -- the Elo likelihood is seat-blind, but writing the games
    down seat-correctly keeps the fixture honest.
    """
    records: list[GameRecord] = []
    outcomes = [WHITE_WIN] * wins + [DRAW] * draws + [BLACK_WIN] * losses
    for index, outcome in enumerate(outcomes):
        white, black, result = (
            ("a", "b", outcome) if index % 2 == 0 else ("b", "a", WHITE_WIN - outcome)
        )
        records.append(
            GameRecord(
                white=white,
                black=black,
                result=result,
                book_id=f"book{index // 2:04d}",
                pair_id=f"a:b:{index:05d}",
            )
        )
    return MatchSet.of(records)


def test_planted_ratings_are_recovered_within_tolerance() -> None:
    """The headline test: planted ratings come back, anchored, within +/-40 Elo."""
    match_set = _planted_match_set()
    assert len(match_set) == _EXPECTED_GAMES
    assert len(match_set) >= 2000
    assert len(match_set.players()) >= 6
    assert max(_PLANTED.values()) - min(_PLANTED.values()) >= 600.0

    result = _fit()

    assert result.converged
    assert result.iterations < 10000
    assert set(result.ratings) == set(_PLANTED)
    # The anchor is pinned exactly, not approximately.
    assert result.ratings[_ANCHOR] == _ANCHOR_ELO

    errors = {
        player: result.ratings[player] - planted for player, planted in _PLANTED.items()
    }
    assert all(abs(error) <= _TOLERANCE_ELO for error in errors.values()), errors


def test_planted_order_is_exactly_recovered() -> None:
    """Ranking is the ladder's actual product: no inversions, at any gap."""
    result = _fit()
    by_fitted = sorted(result.ratings, key=lambda player: result.ratings[player])
    by_planted = sorted(_PLANTED, key=lambda player: _PLANTED[player])
    assert by_fitted == by_planted


def test_planted_draw_parameter_is_recovered() -> None:
    """The fitted nu tracks the sampled one, so draw mass is not absorbed into Elo."""
    assert _fit().draw_param == pytest.approx(_DRAW_PARAM, abs=0.1)


def test_synthetic_games_are_properly_paired() -> None:
    """The generator's paired-book promise holds -- the fit's seat-free premise."""
    assert check_paired_books(_planted_match_set()) == ()


def test_fit_beats_the_planted_truth_on_its_own_likelihood() -> None:
    """The fit is a maximum, not just a fixed point of the update rule.

    Checked against an independently written, game-by-game likelihood: the fit
    scores at least as well as the planted ratings do, and moving any single
    player -- or the draw parameter -- makes it strictly worse.
    """
    match_set = _planted_match_set()
    result = _fit()
    best = log_likelihood(match_set, result.ratings, result.draw_param)

    assert best >= log_likelihood(match_set, dict(_PLANTED), _DRAW_PARAM)

    for player in ("e1", "e4"):
        for shift in (-10.0, 10.0):
            nudged = dict(result.ratings)
            nudged[player] += shift
            assert log_likelihood(match_set, nudged, result.draw_param) < best
    for scale in (0.9, 1.1):
        assert (
            log_likelihood(match_set, result.ratings, result.draw_param * scale) < best
        )


def test_two_player_closed_form() -> None:
    """A case the Davidson MLE solves by hand: 180 wins, 120 draws, 60 losses.

    Setting both partial derivatives to zero for two players collapses to
    ``g_a / g_b == wins / losses`` and ``nu == draws / sqrt(wins * losses)``,
    i.e. the fit must reproduce the observed frequencies exactly (the model is
    saturated at two players). That makes this an external check on the
    optimiser rather than a regression on its own output.
    """
    wins, draws, losses = 180, 120, 60
    result = fit_elo(_two_player_records(wins, draws, losses), anchor="b")

    expected_delta = 400.0 * math.log10(wins / losses)
    expected_nu = draws / math.sqrt(wins * losses)

    assert result.converged
    assert result.ratings["b"] == 0.0
    assert result.ratings["a"] == pytest.approx(expected_delta, abs=0.05)
    assert result.draw_param == pytest.approx(expected_nu, abs=1e-3)


def test_two_player_even_score_is_a_dead_heat() -> None:
    """Symmetric data gives exactly equal ratings -- no drift from the iteration."""
    result = fit_elo(_two_player_records(100, 50, 100), anchor="a", anchor_elo=2000.0)
    assert result.converged
    assert result.ratings["a"] == 2000.0
    assert result.ratings["b"] == pytest.approx(2000.0, abs=1e-6)
    assert result.draw_param == pytest.approx(50.0 / 100.0, abs=1e-6)


def test_fit_is_deterministic() -> None:
    """Same input, same hyper-parameters: identical EloResult, bit for bit."""
    match_set = _planted_match_set()
    first = fit_elo(match_set, anchor=_ANCHOR, anchor_elo=_ANCHOR_ELO)
    second = fit_elo(match_set, anchor=_ANCHOR, anchor_elo=_ANCHOR_ELO)
    assert first == second
    assert first.iterations == second.iterations
    assert list(first.ratings) == list(second.ratings)
    for player, rating in first.ratings.items():
        assert rating.hex() == second.ratings[player].hex()


def test_generation_is_deterministic_in_the_seed() -> None:
    """The synthetic ladder is a pure function of its seed, with no global state."""
    first = _generate(_SEED)
    second = _generate(_SEED)
    assert first.records == second.records
    assert _generate(_SEED + 1).records != first.records
    assert fit_elo(first, anchor=_ANCHOR) == fit_elo(second, anchor=_ANCHOR)


def test_unbounded_likelihood_is_reported_as_not_converged() -> None:
    """A player who never drops a point has no finite MLE -- say so, don't fake it."""
    records: list[GameRecord] = []
    for opponent in ("b", "c"):
        for game in range(20):
            records.append(
                GameRecord(
                    white="a",
                    black=opponent,
                    result=WHITE_WIN,
                    book_id=f"book{game:04d}",
                    pair_id=f"a:{opponent}:{game:05d}",
                )
            )
    # b vs c is an ordinary, well-identified 8-4-8 mini-match.
    outcomes = [WHITE_WIN] * 8 + [DRAW] * 4 + [BLACK_WIN] * 8
    for game, outcome in enumerate(outcomes):
        records.append(
            GameRecord(
                white="b",
                black="c",
                result=outcome,
                book_id=f"book{game:04d}",
                pair_id=f"b:c:{game:05d}",
            )
        )

    result = fit_elo(MatchSet.of(records), anchor="b")

    assert not result.converged
    # b and c remain identified against each other: an even mini-match is a
    # dead heat, so only `a` is the unbounded direction.
    assert result.ratings["c"] == pytest.approx(result.ratings["b"], abs=1e-6)
    assert result.ratings["a"] > result.ratings["b"]


_POOL_PLANTED: dict[str, float] = {
    "komugi/m0": 1500.0,
    "komugi/m1": 1620.0,
    "v1/anchor": 1560.0,
    "rawmcts/low": 1380.0,
    "rawmcts/high": 1700.0,
}
"""A §D2.2-shaped field: two milestones, the gauge, and a bracket."""

_POOL_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("komugi/m0", "komugi/m1"),
    ("komugi/m1", "v1/anchor"),
    ("komugi/m1", "rawmcts/low"),
    ("komugi/m1", "rawmcts/high"),
    ("komugi/m0", "v1/anchor"),
)
"""The in-family link, the anchor link and the bracket -- not a round robin.

Five of the six matchups §D2.2(b) schedules (the External leg is on the site's
clock and is not planted here). ``komugi/m0`` and ``rawmcts/low`` never meet,
which is the whole point: the pool's graph is sparse, and an estimator checked
only on a complete graph is checked on a shape it will never see.
"""

_POOL_PAIRS: dict[tuple[str, str], int] = {
    ("komugi/m0", "komugi/m1"): 128,
}
"""§D2.2(b)1: the in-family link is played at 256 games, the rest at 128."""


def test_the_round_robin_default_is_unchanged_by_the_schedule_argument() -> None:
    """The default path is byte-for-byte the round robin it always was.

    Red against a schedule argument that reorders, re-seeds or re-books the
    default path: the planted-recovery numbers above are read off it, and every
    stored ladder record was produced by it.
    """
    explicit = tuple(itertools.combinations(sorted(_PLANTED), 2))
    default = simulate_match_set(
        _PLANTED, seed=_SEED, draw_param=_DRAW_PARAM, pairs_per_matchup=3
    )
    scheduled = simulate_match_set(
        _PLANTED,
        seed=_SEED,
        draw_param=_DRAW_PARAM,
        pairs_per_matchup=3,
        schedule=explicit,
    )
    assert default.records == scheduled.records
    assert len(default) == _MATCHUPS * 3 * 2
    assert default.book_ids() == ("book0000", "book0001", "book0002")

    # Non-vacuity: a *different* schedule must produce different games, or the
    # equality above would say nothing about the argument being honoured.
    other = simulate_match_set(
        _PLANTED,
        seed=_SEED,
        draw_param=_DRAW_PARAM,
        pairs_per_matchup=3,
        schedule=explicit[:5],
    )
    assert other.records != default.records
    assert len(other) == 5 * 3 * 2


def test_a_planted_pool_schedule_plays_only_its_own_matchups() -> None:
    """A sparse §D2.2 schedule plants, pairs and fits -- U5's machinery."""
    # Red against a schedule argument that still plays the round robin (the
    # unscheduled komugi/m0 x rawmcts/low link would appear), against one that
    # ignores the per-matchup counts (the in-family link carries G_in alone and
    # is bought 256 games for that reason), and against one whose pair ids
    # collide across matchups -- which check_paired_books would catch.
    match_set = simulate_match_set(
        _POOL_PLANTED,
        seed=_SEED,
        draw_param=_DRAW_PARAM,
        seat_advantage=_SEAT_ADVANTAGE,
        pairs_per_matchup=64,
        schedule=_POOL_SCHEDULE,
        pairs_by_matchup=_POOL_PAIRS,
    )

    played: dict[frozenset[str], int] = {}
    for record in match_set.records:
        played[frozenset(record.seats)] = played.get(frozenset(record.seats), 0) + 1
    assert set(played) == {frozenset(pair) for pair in _POOL_SCHEDULE}
    assert played[frozenset(("komugi/m0", "komugi/m1"))] == 256
    assert all(
        count == 128
        for pair, count in played.items()
        if pair != frozenset(("komugi/m0", "komugi/m1"))
    )
    assert frozenset(("komugi/m0", "rawmcts/low")) not in played
    assert check_paired_books(match_set) == ()

    result = fit_elo(match_set, anchor="v1/anchor", anchor_elo=_POOL_PLANTED["v1/anchor"])
    assert result.converged
    errors = {
        player: result.ratings[player] - planted
        for player, planted in _POOL_PLANTED.items()
    }
    assert all(abs(error) <= 60.0 for error in errors.values()), errors
    order = sorted(result.ratings, key=lambda player: result.ratings[player])
    assert order == sorted(_POOL_PLANTED, key=lambda player: _POOL_PLANTED[player])


def test_a_schedule_that_repeats_a_matchup_is_refused() -> None:
    """Every incoherent schedule argument is refused where it is made (F10).

    The node keeps its name because the ticket's acceptance greps node ids; its
    subject is now all three ways the schedule arguments can be incoherent.
    """
    # Red against a builder that silently plays the matchup twice: the two runs
    # share every pair_id, and the resulting record set fails
    # check_paired_books with GAME_COUNT violations rather than at the point
    # the mistake was made.
    with pytest.raises(ValueError, match="scheduled twice"):
        simulate_match_set(
            _POOL_PLANTED,
            seed=_SEED,
            schedule=(("komugi/m0", "komugi/m1"), ("komugi/m1", "komugi/m0")),
        )
    with pytest.raises(ValueError, match="no planted rating"):
        simulate_match_set(
            _POOL_PLANTED, seed=_SEED, schedule=(("komugi/m0", "nobody"),)
        )

    # F10: `pairs_by_matchup` without a `schedule`. Red against the round-robin
    # early return, which never consults the argument at all -- a caller
    # planting "the in-family link at 256, everything else at 128" over a
    # complete graph gets a uniform round robin and no error, and U5's
    # committed coverage artefact then reports a number about an instrument
    # §D2.2 does not play while the artefact says otherwise.
    with pytest.raises(ValueError, match="pairs_by_matchup"):
        simulate_match_set(
            _POOL_PLANTED,
            seed=_SEED,
            pairs_per_matchup=64,
            pairs_by_matchup=_POOL_PAIRS,
        )

    # Non-vacuity: the same counts *with* a schedule are honoured, not refused
    # -- so the raise above is about the missing schedule and not about the
    # argument being unsupported.
    with_schedule = simulate_match_set(
        _POOL_PLANTED,
        seed=_SEED,
        pairs_per_matchup=64,
        schedule=_POOL_SCHEDULE,
        pairs_by_matchup=_POOL_PAIRS,
    )
    assert len(with_schedule) == 2 * (128 + 4 * 64)


def test_fit_rejects_bad_arguments() -> None:
    """Empty data, an absent anchor and nonsense hyper-parameters all raise."""
    match_set = _two_player_records(3, 2, 1)
    with pytest.raises(ValueError, match="empty match set"):
        fit_elo(MatchSet.of([]), anchor="a")
    with pytest.raises(ValueError, match="does not appear"):
        fit_elo(match_set, anchor="nobody")
    with pytest.raises(ValueError, match="tol must be positive"):
        fit_elo(match_set, anchor="a", tol=0.0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        fit_elo(match_set, anchor="a", max_iter=0)
