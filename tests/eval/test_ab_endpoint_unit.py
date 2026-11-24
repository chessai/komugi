"""KOM-036 §6 acceptance nodes #32-#34: the pre-registered A/B endpoint.

No games are played here and no network is loaded. The endpoint is arithmetic
over counts and structure over records, so this suite cuts at exactly that
seam: the Wilson interval is checked against an **independently derived**
reference rather than against a second copy of its own closed form, the
decision rule is table-driven over the four outcomes §1.6 pre-registers, and
the S1 pool is checked against the three ways review B3 showed rev 1's version
could not be built.

The one thing that does touch the real ladder is the collision half of #34,
which calls :func:`komugi.eval.arena.ladder_players` on two synthetic
checkpoint directories. Those directories hold **empty files with checkpoint
names**: ``list_checkpoints`` parses the name and ``ladder_players`` hands back
a lazy factory, so nothing is ever opened. That is deliberate -- the collision
being reproduced is a property of the *id minting*
(``f"ckpt-{ref.step}"``, ``arena.py:814``), and paying for two real nets to
observe two strings would be paying torch to count to two.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from komugi.eval import arena
from komugi.eval.ab_endpoint import (
    ADOPT_A,
    ADOPT_B,
    ARM_A_PREFIX,
    ARM_B_PREFIX,
    DECISIONS,
    INCONCLUSIVE,
    REFERENCE_ID,
    VOID,
    Z_95,
    ab_endpoint,
    cross_arm_schedule,
    fit_cross_arm_pool,
    namespaced_entrants,
    pool_components,
    pool_entrants,
    wilson_interval,
)
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    MatchSet,
)

GAMES: int = 400
"""The pre-registered match size, ``ENDPOINT_GAMES``."""

STEPS: tuple[int, ...] = (300, 600)
"""Two rung indices per arm: enough to collide, cheap enough to be a unit test."""

MATCHUP_RESULTS: tuple[float, ...] = (WHITE_WIN, WHITE_WIN, BLACK_WIN, DRAW)
"""Four fabricated games per matchup, from **white's** perspective.

Read against the seating below (game ``2k`` seats ``first`` as white, ``2k+1``
seats ``second``), this gives ``first`` 1.5 points of 4 and ``second`` 2.5: a
mixed, mildly drawish result in which **neither** player scores 0% or 100%.
That is the condition the Davidson MLE needs to be finite at all
(``elo.py``'s Identifiability section), so a happy-path fit that reports
``converged=True`` is reporting the pool's structure and not this pattern's
degeneracy.
"""


def _wilson_roots(successes: float, trials: int, z: float) -> tuple[float, float]:
    """Return the Wilson bounds from the quadratic they are the roots of.

    Independent of :func:`komugi.eval.ab_endpoint.wilson_interval`'s closed
    form: the interval is *defined* as the set of null rates ``p`` that the
    observed ``p_hat`` does not reject under the score test, i.e. the roots of

        ``(p_hat - p)**2 == z**2 * p * (1 - p) / n``

    which rearranges to ``a*p**2 + b*p + c == 0`` with the coefficients below.
    Solving that with the ordinary quadratic formula reaches the same two
    numbers by a different arithmetic path, which is what makes the comparison
    a check rather than a tautology.
    """
    point = successes / trials
    a = 1.0 + z * z / trials
    b = -(2.0 * point + z * z / trials)
    c = point * point
    root = math.sqrt(b * b - 4.0 * a * c)
    return ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))


def _score_statistic(point: float, null: float, trials: int) -> float:
    """Return the score test statistic of `point` against `null`."""
    return (point - null) / math.sqrt(null * (1.0 - null) / trials)


def test_the_wilson_interval_matches_a_hand_computed_case() -> None:
    """#32: the interval reproduces its defining quadratic, at both parities.

    Both ``(k, n)`` cases the design names are checked, and the half-integer
    one is the point: the endpoint's statistic is ``wins + 0.5 * draws``, which
    is half-integer whenever ``draws`` is odd, so an implementation that took
    an ``int`` -- or that rounded on the way in -- would be wrong on roughly
    half of all possible matches while passing an integer-only node.
    """
    assert Z_95 == pytest.approx(1.959963984540054, abs=1e-12), (
        "the 95% two-sided quantile is the decision boundary's scale; a "
        "rounded 1.96 moves both bounds by more than this node's tolerance"
    )

    for successes in (228.0, 228.5):
        interval = wilson_interval(successes, GAMES)
        low, high = _wilson_roots(successes, GAMES, Z_95)

        assert interval.point == pytest.approx(successes / GAMES, abs=1e-15)
        assert interval.low == pytest.approx(low, abs=1e-12)
        assert interval.high == pytest.approx(high, abs=1e-12)

        # The defining property, checked directly: at each bound the score
        # statistic is exactly +-z. This is what "Wilson" means, and it is
        # what a Wald interval (p_hat +- z*SE) fails.
        assert _score_statistic(interval.point, interval.low, GAMES) == (
            pytest.approx(Z_95, abs=1e-9)
        )
        assert _score_statistic(interval.point, interval.high, GAMES) == (
            pytest.approx(-Z_95, abs=1e-9)
        )

        assert 0.0 < interval.low < interval.point < interval.high < 1.0

    # The half-integer case is strictly to the right of the integer one: the
    # extra half-win is carried, not truncated away.
    integral = wilson_interval(228.0, GAMES)
    half = wilson_interval(228.5, GAMES)
    assert half.point > integral.point
    assert half.low > integral.low
    assert half.high > integral.high


def test_the_wilson_interval_stays_inside_the_unit_interval_at_a_sweep() -> None:
    """The reason Wilson rather than Wald: the bounds cannot leave ``[0, 1]``."""
    for successes in (0.0, 0.5, 1.0, float(GAMES) - 1.0, float(GAMES)):
        interval = wilson_interval(successes, GAMES)
        assert 0.0 <= interval.low <= interval.high <= 1.0

    with pytest.raises(ValueError, match="successes"):
        wilson_interval(-0.5, GAMES)
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(float(GAMES) + 0.5, GAMES)
    with pytest.raises(ValueError, match="trials"):
        wilson_interval(1.0, 0)
    with pytest.raises(ValueError, match="z"):
        wilson_interval(1.0, GAMES, z=0.0)


DECISION_CASES: tuple[tuple[str, int, int, str], ...] = (
    # (label, wins_a, draws, expected decision)
    ("a clear arm-A win", 220, 17, ADOPT_A),
    ("a clear arm-B win", 163, 17, ADOPT_B),
    ("a small arm-A edge", 205, 0, INCONCLUSIVE),
    ("an exact tie", 200, 0, INCONCLUSIVE),
    ("half the games drawn is still valid", 200, 200, ADOPT_A),
    ("too few decisive games", 100, 250, VOID),
    ("void outranks a decisive-looking interval", 140, 250, VOID),
)
"""Table for #33, one row per branch of the pre-registered rule.

The last row is the one that pins the *order*: at ``140`` wins and ``250``
draws the score rate is ``0.6625`` and the Wilson lower bound is far above
``0.5``, so an implementation that read the interval before the validity guard
would answer ``"A"``. The row before it pins the guard's boundary as ``<`` and
not ``<=``: ``200`` decisive of ``400`` is exactly half and is **not** void.
"""


@pytest.mark.parametrize(
    ("label", "wins_a", "draws", "expected"),
    DECISION_CASES,
    ids=[case[0] for case in DECISION_CASES],
)
def test_the_decision_rule_is_the_pre_registered_three_way(
    label: str, wins_a: int, draws: int, expected: str
) -> None:
    """#33: LCB > 0.5 -> A, UCB < 0.5 -> B, straddling -> A on the tie-break.

    Plus the validity guard, which is checked *first* and voids the result
    outright: a match that is mostly draws has an interval, and reading a
    decision off it would be reading the conservatism of the statistic rather
    than a difference between the arms.
    """
    result = ab_endpoint(wins_a, draws, GAMES)

    assert result.decision == expected, label
    assert result.decision in DECISIONS
    assert result.losses_a == GAMES - wins_a - draws
    assert result.decisive == GAMES - draws
    assert result.score == pytest.approx((wins_a + 0.5 * draws) / GAMES)
    assert result.score == result.interval.point

    if expected == ADOPT_A:
        assert result.interval.low > 0.5
    elif expected == ADOPT_B:
        assert result.interval.high < 0.5
    elif expected == INCONCLUSIVE:
        assert result.interval.contains(0.5)


def test_the_decision_rule_refuses_incoherent_counts() -> None:
    """Counts that could not have come from one match are refused, not decided."""
    for wins_a, draws, games in ((-1, 0, GAMES), (0, -1, GAMES), (300, 300, GAMES)):
        with pytest.raises(ValueError):
            ab_endpoint(wins_a, draws, games)
    with pytest.raises(ValueError, match="games"):
        ab_endpoint(0, 0, 0)


def _write_checkpoint_names(directory: Path, steps: tuple[int, ...]) -> None:
    """Create empty files with parseable checkpoint names in `directory`.

    ``list_checkpoints`` reads the *name* -- ``checkpoint-<9+ digits>-<16 hex
    chars>.pt`` -- and never opens the file, and ``ladder_players`` wraps each
    ref in a lazy factory. So this is enough to mint the ids whose collision is
    the subject of #34's first part, with no torch and no weights.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (directory / f"checkpoint-{step:09d}-{step:016x}.pt").touch()


def _fabricate_matchup(
    first: str, second: str, index: int
) -> tuple[GameRecord, ...]:
    """Return four seat-swapped fabricated games between two pooled entrants.

    Seated the way :func:`komugi.eval.arena.play_match` seats a pair: game
    ``2k`` puts `first` on white, game ``2k + 1`` puts `second` there. The
    ``pair_id`` carries the matchup index so no two matchups share a pair.
    """
    records: list[GameRecord] = []
    for game, result in enumerate(MATCHUP_RESULTS):
        white, black = (first, second) if game % 2 == 0 else (second, first)
        records.append(
            GameRecord(
                white=white,
                black=black,
                result=result,
                book_id=arena.EMPTY_BOOK,
                pair_id=f"matchup-{index}-pair-{game // 2}",
            )
        )
    return tuple(records)


def _pool_records(
    schedule: tuple[tuple[str, str], ...],
) -> MatchSet:
    """Play the whole schedule on paper and pool it into one match set."""
    records: list[GameRecord] = []
    for index, (first, second) in enumerate(schedule):
        records.extend(_fabricate_matchup(first, second, index))
    return MatchSet.of(records)


def test_a_cross_arm_pool_refuses_colliding_ids_and_disconnected_graphs(
    tmp_path: Path,
) -> None:
    """#34: the three ways rev 1's S1 could not be built, each refused.

    Rev 1 specified S1 as "both arms' checkpoint directories at ``every_n``,
    anchored on the shared ``ckpt-300``". All three of its premises are false:
    the ids collide, there is no shared ``ckpt-300`` (the arms are
    bit-identical only at *initialisation* and diverge at the first optimiser
    step), and two pooled within-arm chains are a disconnected graph whose
    inter-arm offset -- the only quantity S1 exists to show -- is a free
    parameter of the joint likelihood. Each failure is silent, which is why
    each is a guard here rather than a note in a report.
    """
    arm_a_dir = tmp_path / "armA" / "ckpt"
    arm_b_dir = tmp_path / "armB" / "ckpt"
    _write_checkpoint_names(arm_a_dir, STEPS)
    _write_checkpoint_names(arm_b_dir, STEPS)

    raw_a = arena.ladder_players(arm_a_dir)
    raw_b = arena.ladder_players(arm_b_dir)

    # --- (1) collision -------------------------------------------------------
    # Both arms run the same number of steps, so both directories mint the same
    # ids. Pooling them unnamespaced must raise, reproducing run_ladder's own
    # duplicate refusal (ladder.py:423-429), which a hand-rolled driver
    # inherits no more than it inherits the per-matchup seeding.
    assert [player.player.id for player in raw_a] == ["ckpt-300", "ckpt-600"]
    assert [player.player.id for player in raw_b] == ["ckpt-300", "ckpt-600"]
    with pytest.raises(ValueError, match="share"):
        pool_entrants(raw_a, raw_b)

    arm_a = namespaced_entrants(raw_a, ARM_A_PREFIX)
    arm_b = namespaced_entrants(raw_b, ARM_B_PREFIX)
    arm_a_ids = [player.player.id for player in arm_a]
    arm_b_ids = [player.player.id for player in arm_b]
    assert arm_a_ids == ["A/ckpt-300", "A/ckpt-600"]
    assert arm_b_ids == ["B/ckpt-300", "B/ckpt-600"]
    # Prefixing, not rebuilding: the ladder's family survives the namespacing,
    # so the pooled set is not half a different family mid-fit.
    assert {player.player.family for player in arm_a + arm_b} == {
        arena.LADDER_FAMILY
    }
    # The factories are untouched objects, not reconstructed loaders.
    assert [player.evaluator_factory for player in arm_a] == [
        player.evaluator_factory for player in raw_a
    ]
    pooled = pool_entrants(arm_a, arm_b)
    assert len(pooled) == len(arm_a) + len(arm_b)

    # --- (2) disconnection ---------------------------------------------------
    # Two within-arm chains and no cross-arm games: exactly what rev 1
    # scheduled. The component check sees it; the fit would not.
    disconnected = _pool_records(
        (
            (arm_a_ids[0], arm_a_ids[1]),
            (arm_b_ids[0], arm_b_ids[1]),
        )
    )
    components = pool_components(disconnected)
    assert len(components) == 2
    assert {frozenset(arm_a_ids), frozenset(arm_b_ids)} == set(components)
    with pytest.raises(ValueError, match="components"):
        fit_cross_arm_pool(disconnected, anchor=arm_a_ids[0])

    # --- (3) the happy path --------------------------------------------------
    # Namespaced ids, an explicit cross-arm link at every rung index, and the
    # shared reference entrant that pins the gauge on a net neither arm made.
    schedule = cross_arm_schedule(arm_a_ids, arm_b_ids)
    assert schedule == (
        ("A/ckpt-300", "B/ckpt-300"),
        ("A/ckpt-300", REFERENCE_ID),
        ("B/ckpt-300", REFERENCE_ID),
        ("A/ckpt-600", "B/ckpt-600"),
        ("A/ckpt-600", REFERENCE_ID),
        ("B/ckpt-600", REFERENCE_ID),
    )
    connected = _pool_records(schedule)
    roster = [*arm_a_ids, *arm_b_ids, REFERENCE_ID]
    assert pool_components(connected) == (frozenset(roster),)

    fit = fit_cross_arm_pool(connected, expected_ids=roster)
    assert fit.converged, "a connected, non-degenerate pool must identify"
    assert sorted(fit.ratings) == sorted(roster)
    assert fit.ratings[REFERENCE_ID] == 0.0, (
        "the gauge is pinned on the shared reference, not on an arm's own "
        "checkpoint: anchoring on an arm would make the reported inter-arm "
        "offset depend on which arm was named"
    )


def test_the_pool_refuses_an_entrant_that_played_no_games(tmp_path: Path) -> None:
    """An unplayed entrant is absent from ``ratings``, not reported as unrated.

    A caller reading the table alongside its roster would not notice; the
    driver checks the roster it was handed against the games it was handed.
    """
    arm_a_dir = tmp_path / "armA" / "ckpt"
    _write_checkpoint_names(arm_a_dir, STEPS)
    arm_a_ids = [
        player.player.id
        for player in namespaced_entrants(
            arena.ladder_players(arm_a_dir), ARM_A_PREFIX
        )
    ]
    match_set = _pool_records(((arm_a_ids[0], REFERENCE_ID),))
    with pytest.raises(ValueError, match="played no games"):
        fit_cross_arm_pool(
            match_set, expected_ids=[*arm_a_ids, REFERENCE_ID]
        )


def test_the_cross_arm_schedule_refuses_mismatched_rung_counts() -> None:
    """The arms must be sampled at the same rung indices, or the links are not links."""
    with pytest.raises(ValueError, match="rung"):
        cross_arm_schedule(["A/ckpt-300", "A/ckpt-600"], ["B/ckpt-300"])
    with pytest.raises(ValueError, match="at least one rung"):
        cross_arm_schedule([], [])


def test_a_drawn_pool_still_carries_the_draw_parameter() -> None:
    """The fabricated pattern is drawish, and the Davidson fit must see it.

    Guards the fabrication rather than the code under test: a pattern with no
    draws would fit ``nu = 0`` and quietly make the happy-path node above a
    Bradley-Terry check rather than the Davidson one the ladder actually runs.
    """
    assert DRAW in MATCHUP_RESULTS
    match_set = _pool_records((("A/ckpt-300", REFERENCE_ID),))
    fit = fit_cross_arm_pool(match_set, expected_ids=["A/ckpt-300", REFERENCE_ID])
    assert fit.draw_param > 0.0
