"""KOM-230: ``scripts/analysis/repin_fireability.py``.

Loaded **by path**, the way ``tests/analysis/test_termination_census.py`` loads
its script: ``scripts/`` is not a package, is not on ``sys.path``, and is
outside ``[tool.mypy] files``, so a plain import would be a hard mypy failure.
Everything the loader returns is ``Any``; every test binds what it reads to an
explicitly annotated local so this module stays strict-clean.

Nothing here asserts a conclusion about campaign 6. The script supplies exact
numbers and the reading of them belongs in the ticket, not in a test that would
have to be edited every time the campaign's draw rate is re-measured. What is
pinned here is that the numbers are the numbers they claim to be: the pass
probability is checked against a brute force written independently in this
file, and the wait is checked against a closed form derived in a comment.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

#: ``<repo>/scripts/analysis`` -- this file is ``<repo>/tests/analysis/…``.
ANALYSIS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "scripts" / "analysis"

#: The module name the script is cached under in ``sys.modules``.
SCRIPT_NAME: Final[str] = "repin_fireability"


def load(name: str) -> ModuleType:
    """Return ``scripts/analysis/<name>.py``, imported once per process."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ANALYSIS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load an analysis script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fireability: Final[Any] = load(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# Degenerate cases -- the ones a reader can check without arithmetic
# ---------------------------------------------------------------------------


def test_a_perfect_player_passes_every_gauntlet() -> None:
    """Strength 1.0 with no draws wins all 64 games, so it passes with 1.0.

    And two consecutive passes are then certain too -- the case that would
    catch a pass probability that had been silently truncated or clipped.
    """
    single: float = fireability.single_gauntlet_pass_probability(1.0, 0.0)
    assert single == pytest.approx(1.0)

    both: float = fireability.consecutive_pass_probability(1.0, 0.0)
    assert both == pytest.approx(1.0)


def test_a_threshold_of_zero_is_always_cleared() -> None:
    """Every outcome scores at least zero, so the enumeration must sum to one.

    This is the total-probability check on the multinomial: if any outcome were
    missed or double-counted, this is where it shows.
    """
    total: float = fireability.single_gauntlet_pass_probability(
        0.5, 0.065, 16, 0.0
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_an_unreachable_threshold_never_fires() -> None:
    """A threshold above the best possible score has probability zero."""
    impossible: float = fireability.single_gauntlet_pass_probability(
        0.9, 0.0, 8, 1.01
    )
    assert impossible == 0.0


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_pass_probability_is_non_decreasing_in_strength() -> None:
    """A stronger player never passes LESS often, at a fixed draw rate.

    The grid stops at 0.94 because a 0.065 draw rate caps the achievable score
    per game at ``1 - 0.065/2 = 0.9675``; asking beyond that is the refusal
    case below, not a monotonicity failure.
    """
    grid = [0.50 + 0.02 * step for step in range(23)]
    probabilities = [
        float(fireability.single_gauntlet_pass_probability(strength, 0.065))
        for strength in grid
    ]

    assert probabilities[0] < probabilities[-1], "the grid must actually move"
    for lower, higher in zip(probabilities, probabilities[1:]):
        assert higher >= lower - 1e-12, (
            f"pass probability fell from {lower} to {higher} as strength rose"
        )


# ---------------------------------------------------------------------------
# Exactness -- brute force written here, NOT calling the function under test
# ---------------------------------------------------------------------------


def test_pass_probability_matches_an_independent_brute_force() -> None:
    """Enumerate all 3**4 game sequences by hand and compare.

    The brute force below shares no code with the script: it derives its own
    per-game probabilities from the same definition (``strength`` is the
    expected score per game, so ``p_win = strength - draw_rate/2``), walks
    every ordered sequence of four game results, and sums the ones whose score
    reaches the threshold. If the script's multinomial coefficients or its
    ``>=`` boundary were wrong, the two would disagree.
    """
    games = 4
    threshold = 0.75
    strength = 0.6
    draw_rate = 0.2

    p_win = strength - 0.5 * draw_rate  # 0.5
    p_draw = draw_rate  # 0.2
    p_loss = 1.0 - draw_rate - p_win  # 0.3
    points = (1.0, 0.5, 0.0)
    probabilities = (p_win, p_draw, p_loss)

    expected = 0.0
    for sequence in itertools.product(range(3), repeat=games):
        score = sum(points[result] for result in sequence) / games
        if score < threshold:
            continue
        probability = 1.0
        for result in sequence:
            probability *= probabilities[result]
        expected += probability

    actual: float = fireability.single_gauntlet_pass_probability(
        strength, draw_rate, games, threshold
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert 0.0 < expected < 1.0, "a vacuous case would prove nothing"


def test_the_threshold_boundary_is_inclusive() -> None:
    """A score exactly equal to the threshold counts as a pass.

    Two games, no draws, threshold 0.5: the outcomes that reach it are one win
    (probability ``2 * p * (1 - p)``) and two wins (``p**2``), so at ``p = 0.5``
    the answer is ``0.5 + 0.25 = 0.75``. A strictly-greater comparison would
    return 0.25 instead.
    """
    actual: float = fireability.single_gauntlet_pass_probability(0.5, 0.0, 2, 0.5)
    assert actual == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_impossible_strength_and_draw_rate_is_refused() -> None:
    """Strength 0.99 at a 0.5 draw rate implies a negative loss probability.

    The draws alone supply 0.25 of the expected score and leave only 0.5 of the
    probability mass, which cannot supply the remaining 0.74. The script must
    raise rather than clamp: a clamped answer would be a confident number about
    a player that cannot exist.
    """
    with pytest.raises(ValueError, match="unreachable"):
        fireability.single_gauntlet_pass_probability(0.99, 0.5)

    with pytest.raises(ValueError, match="unreachable"):
        fireability.outcome_probabilities(0.99, 0.5)


@pytest.mark.parametrize(
    ("strength", "draw_rate"), [(-0.1, 0.0), (1.1, 0.0), (0.6, -0.2), (0.6, 1.5)]
)
def test_out_of_range_arguments_are_refused(
    strength: float, draw_rate: float
) -> None:
    """A probability outside ``[0, 1]`` is a caller bug, not an input."""
    with pytest.raises(ValueError):
        fireability.outcome_probabilities(strength, draw_rate)


def test_a_gauntlet_of_no_games_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one game"):
        fireability.single_gauntlet_pass_probability(0.9, 0.0, 0)


# ---------------------------------------------------------------------------
# The independence assumption, pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.55, 0.75, 0.85, 0.90])
def test_two_consecutive_is_the_square_of_one(strength: float) -> None:
    """Under the stated independence assumption, ``P(both) == P(one)**2``.

    Pinned deliberately: it is the modelling choice the whole expected-wait
    result rests on, and if it ever stops being true -- a correlation term, a
    drift model -- that is a change of question, which should break this test
    rather than quietly change a published number.
    """
    single: float = fireability.single_gauntlet_pass_probability(strength, 0.065)
    both: float = fireability.consecutive_pass_probability(strength, 0.065)
    assert both == pytest.approx(single**2, rel=1e-12)


def test_three_in_a_row_is_the_cube() -> None:
    """The same assumption, at ``runs=3``: the exponent follows ``runs``."""
    single: float = fireability.single_gauntlet_pass_probability(0.88, 0.065)
    triple: float = fireability.consecutive_pass_probability(
        0.88, 0.065, 64, 0.85, 3
    )
    assert triple == pytest.approx(single**3, rel=1e-12)


# ---------------------------------------------------------------------------
# The expected wait, against its closed form
# ---------------------------------------------------------------------------


def test_expected_wait_at_a_half_matches_the_closed_form() -> None:
    """At ``q = 0.5`` the wait for two consecutive passes is exactly 6.

    Derivation (so the 6 is not a magic constant). Let ``E`` be the expected
    number of trials from a clean slate and ``E1`` the expectation once one
    pass is already banked. Then::

        E  = q * (1 + E1) + (1 - q) * (1 + E)
        E1 = q * 1        + (1 - q) * (1 + E)

    Solving gives ``E = (1 + q) / q**2``, the ``runs=2`` case of the general
    ``(1 - q**r) / (q**r * (1 - q))``. At ``q = 0.5`` that is
    ``1.5 / 0.25 = 6``.

    The ``q = 0.5`` is itself exact rather than approximate: a one-game
    "gauntlet" at strength 0.5 with no draws scores 1.0 or 0.0 with equal
    probability, so a threshold of 0.5 is cleared exactly half the time.
    """
    q = 0.5
    closed_form = (1.0 + q) / q**2
    assert closed_form == pytest.approx(6.0)

    direct: float = fireability.expected_wait_for_consecutive_passes(q)
    assert direct == pytest.approx(6.0)

    single: float = fireability.single_gauntlet_pass_probability(0.5, 0.0, 1, 0.5)
    assert single == pytest.approx(q)

    wait: Any = fireability.expected_gauntlets_until_repin(0.5, 0.0, 1, 0.5)
    gauntlets: float = wait.gauntlets
    iterations: float = wait.iterations
    assert wait.pass_probability == pytest.approx(q)
    assert gauntlets == pytest.approx(6.0)
    # The default cadence is one gauntlet every 4 iterations.
    assert iterations == pytest.approx(gauntlets * 4)


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.8, 0.99])
def test_expected_wait_follows_the_general_run_formula(q: float) -> None:
    """``(1 - q**r) / (q**r * (1 - q))`` for ``r`` in ``{2, 3}``."""
    for runs in (2, 3):
        expected = (1.0 - q**runs) / (q**runs * (1.0 - q))
        actual: float = fireability.expected_wait_for_consecutive_passes(q, runs)
        assert actual == pytest.approx(expected, rel=1e-12)


def test_a_player_who_cannot_pass_waits_forever() -> None:
    """``q = 0`` is infinity, and ``q = 1`` fires on the first ``runs``."""
    assert math.isinf(fireability.expected_wait_for_consecutive_passes(0.0))
    assert fireability.expected_wait_for_consecutive_passes(1.0) == pytest.approx(2.0)
    assert fireability.expected_wait_for_consecutive_passes(
        1.0, 3
    ) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# The horizon chain and its inversion
# ---------------------------------------------------------------------------


def test_repin_probability_within_a_short_horizon_is_exact() -> None:
    """Two gauntlets can only re-pin by passing both: ``q**2``.

    Three gauntlets add the disjoint path "fail, then pass twice", so the
    answer is ``q**2 + (1 - q) * q**2``. Written out rather than derived from
    the chain, so the chain is checked against arithmetic.
    """
    q = 0.3
    assert fireability.repin_probability_within(0, q) == 0.0
    assert fireability.repin_probability_within(1, q) == 0.0
    assert fireability.repin_probability_within(2, q) == pytest.approx(q**2)
    assert fireability.repin_probability_within(3, q) == pytest.approx(
        q**2 + (1.0 - q) * q**2
    )


def test_repin_probability_within_is_non_decreasing_in_the_horizon() -> None:
    """More gauntlets cannot lower the chance of having re-pinned."""
    values = [
        float(fireability.repin_probability_within(horizon, 0.2))
        for horizon in range(12)
    ]
    for earlier, later in zip(values, values[1:]):
        assert later >= earlier - 1e-15


def test_strength_for_repin_within_is_the_inverse_of_the_horizon() -> None:
    """The returned strength sits at even odds: just above, and just below."""
    required: float = fireability.strength_for_repin_within(300)
    assert 0.5 < required < 1.0

    at_strength: float = fireability.repin_probability_within(
        300 // 4, fireability.single_gauntlet_pass_probability(required, 0.065)
    )
    assert at_strength == pytest.approx(0.5, abs=1e-6)

    weaker: float = fireability.repin_probability_within(
        300 // 4,
        fireability.single_gauntlet_pass_probability(required - 0.01, 0.065),
    )
    assert weaker < 0.5


def test_strength_for_repin_within_refuses_an_impossible_horizon() -> None:
    """One gauntlet cannot contain two consecutive passes, at any strength."""
    with pytest.raises(ValueError, match="no achievable strength"):
        fireability.strength_for_repin_within(4)


# ---------------------------------------------------------------------------
# The table and the CLI
# ---------------------------------------------------------------------------


def test_the_default_grid_covers_the_pre_registered_strengths() -> None:
    """0.80, 0.85, 0.87 and 0.90 must all be printed."""
    grid: tuple[float, ...] = fireability.DEFAULT_STRENGTHS
    for strength in (0.80, 0.85, 0.87, 0.90):
        assert any(value == pytest.approx(strength) for value in grid)


def test_the_campaign_defaults_are_the_documented_ones() -> None:
    assert fireability.DEFAULT_GAMES == 64
    assert fireability.DEFAULT_THRESHOLD == pytest.approx(0.85)
    assert fireability.DEFAULT_RUNS == 2
    assert fireability.DEFAULT_CADENCE == 4
    assert fireability.DEFAULT_DRAW_RATE == pytest.approx(0.065)


def test_the_table_reports_one_row_per_strength_in_order() -> None:
    rows: tuple[Any, ...] = fireability.fireability_table((0.80, 0.90))
    assert len(rows) == 2
    assert rows[0].strength == pytest.approx(0.80)
    assert rows[1].strength == pytest.approx(0.90)
    assert rows[1].pass_probability > rows[0].pass_probability
    assert rows[1].iterations < rows[0].iterations


def test_main_prints_the_table_and_the_inversion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The smoke: exact, deterministic, and over in milliseconds."""
    status: int = fireability.main([])
    assert status == 0

    printed: str = capsys.readouterr().out
    assert "re-pin fireability" in printed
    assert "0.8500" in printed
    assert "0.8700" in printed
    assert "within 300 iterations" in printed


def test_main_is_deterministic() -> None:
    """No sampling anywhere: two identical calls give identical text."""
    rows_a: tuple[Any, ...] = fireability.fireability_table()
    rows_b: tuple[Any, ...] = fireability.fireability_table()
    assert fireability.format_table(rows_a) == fireability.format_table(rows_b)
