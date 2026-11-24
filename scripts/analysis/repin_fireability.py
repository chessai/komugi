#!/usr/bin/env python3
"""KOM-230: CAN the gauntlet re-pin rule fire at all?

Campaign 6 re-pins its gauntlet anchor only when TWO CONSECUTIVE gauntlets
score at or above ``0.85`` (with the anchor at least 12 iterations old). A
gauntlet is 64 games played as 32 seat-swapped pairs, so its score
``(wins + 0.5*draws)/64`` moves in half-point steps of ``1/128``. The anchor
has not moved in 301 iterations: across 76 gauntlets the observed scores run
min ``0.500``, median ``0.7421875``, max ``0.8671875``, and only 2 of the 76
reached ``0.85`` -- never on consecutive runs.

This script does NOT ask whether the net is strong. It asks whether the RULE
is fireable: **at a given true per-game strength, how often does a 64-game
gauntlet clear the threshold, and how often does that happen twice running?**
The two questions come apart because a 64-game score is a noisy estimate of
strength. A net whose true strength is comfortably above ``0.85`` still fails
a gauntlet about half the time, and two independent coin flips of that kind
land heads-heads only a quarter of the time; push the threshold up against the
top of the achievable range and the waiting time stops being a training
schedule and starts being a design fault. Which of those it is, is exactly
what the table below reports, and it is a statement about the RULE -- it holds
whatever net is plugged into it.

Everything here is exact. One gauntlet is modelled as `games` independent
games, each a win with probability ``p_w``, a draw with probability ``p_d``
and a loss otherwise, and the pass probability is computed by enumerating
every ``(wins, draws)`` outcome under the multinomial pmf. There is no
sampling and no RNG, so two runs of this script at the same arguments print
the same digits and a disagreement is a bug rather than variance.

Usage::

    python scripts/analysis/repin_fireability.py
    python scripts/analysis/repin_fireability.py --draw-rate 0.12 --horizon 200
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

DEFAULT_DRAW_RATE: Final[float] = 0.065
"""Campaign 6's measured gauntlet draw rate, about 6.5%.

This is a PARAMETER and not a detail, because the draw rate sets the variance
of the score and therefore the whole answer. A draw is a half point with no
spread; at a fixed expected score per game, moving mass from wins-and-losses
into draws shrinks the per-game variance, which narrows the distribution of
the 64-game score around its mean. Narrower is a double-edged thing here: it
makes a net whose true strength is BELOW the threshold pass less often (fewer
lucky gauntlets), and a net whose true strength is ABOVE it pass MORE often.
So a reader who assumes the wrong draw rate gets the wrong verdict on
fireability, in a direction that depends on which side of ``0.85`` the net is
on. Pass ``--draw-rate`` and see.
"""

DEFAULT_GAMES: Final[int] = 64
"""Games in one gauntlet: 32 seat-swapped pairs (campaign 6's own cadence)."""

DEFAULT_THRESHOLD: Final[float] = 0.85
"""The pre-registered per-gauntlet score a pass must reach."""

DEFAULT_RUNS: Final[int] = 2
"""Consecutive passing gauntlets the re-pin rule demands."""

DEFAULT_CADENCE: Final[int] = 4
"""Iterations between gauntlets, which converts a wait in gauntlets to one in
iterations -- the unit the campaign is actually budgeted in."""

DEFAULT_HORIZON: Final[int] = 300
"""The horizon the CLI inverts over, in iterations.

Chosen to match the 301 iterations the anchor has in fact stood still: the
inversion then reads "the true strength a net would have needed for a coin
flip's chance of re-pinning in the time we have already spent".
"""

DEFAULT_STRENGTHS: Final[tuple[float, ...]] = (
    0.75,
    0.80,
    0.85,
    0.87,
    0.90,
    0.95,
)
"""The printed strength grid.

It must contain ``0.85`` (a net exactly AT the threshold -- the case that
shows the rule's coin-flip behaviour most starkly), ``0.87`` (just above the
best gauntlet ever observed, ``0.8671875``), and ``0.80``/``0.90`` to bracket
them.
"""

_SCORE_TOLERANCE: Final[float] = 1e-9
"""Slack when testing ``score >= threshold``.

It guards binary representation only, never a real step of the score grid: the
grid's step is ``1/games`` halved, ``0.0078125`` for a 64-game gauntlet, seven
orders of magnitude coarser than this. So no outcome is reclassified by it.
"""


def outcome_probabilities(
    strength: float, draw_rate: float = DEFAULT_DRAW_RATE
) -> tuple[float, float, float]:
    """Return ``(p_win, p_draw, p_loss)`` for one game.

    Args:
        strength: The **expected score per game**, on the same scale as a
            gauntlet score: ``0.5`` is parity, ``1.0`` is winning every game.
            It is not a win rate -- a draw contributes half a point -- so
            ``p_win = strength - draw_rate/2``.
        draw_rate: The probability a game is drawn. See
            :data:`DEFAULT_DRAW_RATE` for why it changes the answer.

    Raises:
        ValueError: If the pair is impossible -- if it implies a negative win
            or loss probability, or if either argument is outside ``[0, 1]``.
            Refused loudly rather than clamped, because a clamp would silently
            answer a question about a different player: ``strength=0.99`` with
            ``draw_rate=0.5`` is not "nearly perfect with many draws", it is
            arithmetically nothing at all (the draws alone supply 0.25 and the
            wins would have to supply 0.74 from the 0.5 that is left).
    """
    if not 0.0 <= draw_rate <= 1.0:
        raise ValueError(f"a draw rate must lie in [0, 1], got {draw_rate!r}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"a strength must lie in [0, 1], got {strength!r}")

    p_win = strength - 0.5 * draw_rate
    p_loss = 1.0 - draw_rate - p_win
    if p_win < -_SCORE_TOLERANCE or p_loss < -_SCORE_TOLERANCE:
        raise ValueError(
            f"strength {strength!r} is unreachable at a draw rate of "
            f"{draw_rate!r}: it implies p_win={p_win!r}, p_loss={p_loss!r}. "
            f"With this draw rate the score per game is confined to "
            f"[{0.5 * draw_rate!r}, {1.0 - 0.5 * draw_rate!r}]"
        )
    return max(p_win, 0.0), draw_rate, max(p_loss, 0.0)


def _log_power(probability: float, count: int) -> float:
    """Return ``count * log(probability)`` with ``0 * log(0) == 0``.

    The convention matters: an outcome that asks for zero draws when draws are
    impossible has probability one from that factor, not zero.
    """
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -math.inf
    return count * math.log(probability)


def single_gauntlet_pass_probability(
    strength: float,
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Return the exact probability one gauntlet scores at or above `threshold`.

    Computed by ENUMERATION, not by sampling: every ``(wins, draws)`` pair with
    ``wins + draws <= games`` is a distinct outcome of the multinomial, its
    probability is the multinomial pmf, and the ones whose score
    ``(wins + 0.5*draws)/games`` reaches `threshold` are summed. For 64 games
    that is 2145 terms and costs microseconds, so there is no reason to
    tolerate Monte Carlo error in a number the whole finding rests on.

    Args:
        strength: The expected score per game; ``0.5`` is parity. See
            :func:`outcome_probabilities` for how ``p_win`` is derived.
        draw_rate: The per-game draw probability.
        games: Games in a gauntlet.
        threshold: The score a pass must reach, inclusive.

    Returns:
        A probability in ``[0, 1]``.

    Raises:
        ValueError: If `games` is not positive, or via
            :func:`outcome_probabilities` for an impossible strength/draw pair.
    """
    if games <= 0:
        raise ValueError(f"a gauntlet needs at least one game, got {games!r}")
    p_win, p_draw, p_loss = outcome_probabilities(strength, draw_rate)

    total = 0.0
    for wins in range(games + 1):
        for draws in range(games - wins + 1):
            score = (wins + 0.5 * draws) / games
            if score < threshold - _SCORE_TOLERANCE:
                continue
            losses = games - wins - draws
            # log of the exact integer multinomial coefficient: log-space
            # because at 64 games the coefficient reaches ~1e29 while the
            # probability factors reach ~1e-30, and the naive product of the
            # two underflows long before the term itself is negligible.
            log_coefficient = math.log(
                math.comb(games, wins) * math.comb(games - wins, draws)
            )
            log_term = (
                log_coefficient
                + _log_power(p_win, wins)
                + _log_power(p_draw, draws)
                + _log_power(p_loss, losses)
            )
            if log_term != -math.inf:
                total += math.exp(log_term)
    return min(max(total, 0.0), 1.0)


def consecutive_pass_probability(
    strength: float,
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
    runs: int = DEFAULT_RUNS,
) -> float:
    """Return the probability `runs` consecutive gauntlets ALL pass.

    Independence across gauntlets is an ASSUMPTION, and it is the modelling
    choice a reader should push on hardest. Two back-to-back gauntlets are the
    same two nets with only the seed changed, so their scores share every
    systematic term -- an opening the challenger does not understand, a
    plan-length the anchor exploits -- and only the seed-driven part is fresh.
    That is positive correlation, and positive correlation makes this number
    **pessimistic**: conditional on the first gauntlet having passed, the
    second is more likely to pass than its marginal rate suggests, so the true
    two-in-a-row rate is somewhat higher than ``q**runs``. The direction is
    worth stating plainly because it means a finding of "unfireable" computed
    here is conservative -- correlation can only help the rule -- while a
    finding of "fireable" would be the fragile one.

    (In the other direction: the training run advances between gauntlets, so
    the two are not even the same challenger. That drift is a trend, not a
    correlation, and it is why :func:`expected_gauntlets_until_repin` reports a
    wait at a FIXED strength rather than pretending to model a moving one.)
    """
    if runs < 1:
        raise ValueError(f"a re-pin needs at least one gauntlet, got {runs!r}")
    pass_probability = single_gauntlet_pass_probability(
        strength, draw_rate, games, threshold
    )
    return pass_probability**runs


def expected_wait_for_consecutive_passes(
    pass_probability: float, runs: int = DEFAULT_RUNS
) -> float:
    """Return the expected number of gauntlets until `runs` in a row pass.

    The standard geometric-run result for independent trials each passing with
    probability ``q``::

        E[trials until `runs` consecutive passes] = (1 - q**runs)
                                                    / (q**runs * (1 - q))

    which for the campaign's ``runs=2`` simplifies to ``(1 + q) / q**2`` (the
    ``(1 - q)`` cancels against ``1 - q**2 = (1 - q)(1 + q)``). The shape is
    the finding in miniature: the cost is quadratic in ``1/q``, so halving the
    per-gauntlet pass rate quadruples the wait.

    Edge cases: ``q == 0`` never fires and returns infinity; ``q == 1`` fires
    on the first `runs` gauntlets, which is the limit of the formula (both
    numerator and denominator vanish there).
    """
    if not 0.0 <= pass_probability <= 1.0:
        raise ValueError(
            f"a pass probability must lie in [0, 1], got {pass_probability!r}"
        )
    if runs < 1:
        raise ValueError(f"a re-pin needs at least one gauntlet, got {runs!r}")
    if pass_probability <= 0.0:
        return math.inf
    if pass_probability >= 1.0:
        return float(runs)
    run_probability = pass_probability**runs
    return (1.0 - run_probability) / (run_probability * (1.0 - pass_probability))


@dataclass(frozen=True)
class RepinWait:
    """One row of the fireability table: what a fixed true strength implies."""

    strength: float
    """The true expected score per game this row is computed at."""

    pass_probability: float
    """Probability ONE gauntlet reaches the threshold."""

    consecutive_probability: float
    """Probability the next ``runs`` gauntlets all reach it."""

    gauntlets: float
    """Expected gauntlets played before the rule fires; ``inf`` if never."""

    iterations: float
    """The same wait in training iterations, at the gauntlet cadence."""


def expected_gauntlets_until_repin(
    strength: float,
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
    runs: int = DEFAULT_RUNS,
    iterations_per_gauntlet: int = DEFAULT_CADENCE,
) -> RepinWait:
    """Return the expected wait until a re-pin, in gauntlets AND iterations.

    The iteration figure is the one that makes the finding legible: "a net at
    true strength 0.85 re-pins after N iterations on average" is a number that
    can be compared against how long the campaign has actually run, whereas a
    wait in gauntlets has to be converted in the reader's head first. At
    campaign 6's cadence one gauntlet is played every
    `iterations_per_gauntlet` iterations, so the conversion is a multiply.

    The wait assumes a strength that does not move. A real run improves, so
    this is the wait for a net FROZEN at `strength` -- which is the right
    question for auditing the rule, since the rule has to be clearable by some
    fixed strength before "keep training" can be the answer.
    """
    if iterations_per_gauntlet <= 0:
        raise ValueError(
            f"the gauntlet cadence must be positive, got "
            f"{iterations_per_gauntlet!r}"
        )
    pass_probability = single_gauntlet_pass_probability(
        strength, draw_rate, games, threshold
    )
    gauntlets = expected_wait_for_consecutive_passes(pass_probability, runs)
    return RepinWait(
        strength=strength,
        pass_probability=pass_probability,
        consecutive_probability=pass_probability**runs,
        gauntlets=gauntlets,
        iterations=gauntlets * iterations_per_gauntlet,
    )


def repin_probability_within(
    gauntlets: int, pass_probability: float, runs: int = DEFAULT_RUNS
) -> float:
    """Return the probability of `runs` consecutive passes within `gauntlets`.

    An exact forward pass over the run-length Markov chain: the state is how
    many gauntlets have passed CONSECUTIVELY so far (0 .. runs-1, plus an
    absorbing "re-pinned"), a pass advances it, a failure resets it to zero.
    Exact rather than the ``1 - (1 - q**runs)**gauntlets`` approximation a
    reader might reach for, which is wrong here because overlapping windows
    are not independent trials.
    """
    if gauntlets < 0:
        raise ValueError(f"a horizon cannot be negative, got {gauntlets!r}")
    if not 0.0 <= pass_probability <= 1.0:
        raise ValueError(
            f"a pass probability must lie in [0, 1], got {pass_probability!r}"
        )
    if runs < 1:
        raise ValueError(f"a re-pin needs at least one gauntlet, got {runs!r}")

    live: list[float] = [0.0] * runs
    live[0] = 1.0
    absorbed = 0.0
    for _ in range(gauntlets):
        nxt: list[float] = [0.0] * runs
        for streak, mass in enumerate(live):
            if mass == 0.0:
                continue
            if streak + 1 == runs:
                absorbed += mass * pass_probability
            else:
                nxt[streak + 1] += mass * pass_probability
            nxt[0] += mass * (1.0 - pass_probability)
        live = nxt
    return min(max(absorbed, 0.0), 1.0)


def strength_for_repin_within(
    iterations: int,
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
    runs: int = DEFAULT_RUNS,
    iterations_per_gauntlet: int = DEFAULT_CADENCE,
    target: float = 0.5,
) -> float:
    """Return the true strength needed to re-pin within `iterations`, at odds
    `target`.

    The inversion of the table: instead of "how long does this strength wait",
    it answers "how strong would the net have had to be for this wait to be
    even odds". That is the form the finding is quoted in, because it can be
    compared directly against the best gauntlet score ever observed -- if the
    required strength sits above what the score grid can even express, the rule
    is unfireable by construction.

    Bisection over strength, which is sound because the pass probability is
    monotonically non-decreasing in strength (raising the strength at a fixed
    draw rate moves mass from losses to wins, which can only raise the score
    distribution) and so therefore is the horizon probability.

    The upper end of the search is ``1 - draw_rate/2``, not ``1``: with draws
    occurring at a fixed rate, no player can score above that, and pretending
    otherwise is the error :func:`outcome_probabilities` refuses.

    Raises:
        ValueError: If no achievable strength reaches `target` within the
            horizon -- which is itself a finding, not a failure, and says the
            rule cannot be made to fire at these settings by strength alone.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"a target probability must lie in (0, 1), got {target!r}")
    if iterations < 0:
        raise ValueError(f"a horizon cannot be negative, got {iterations!r}")
    if iterations_per_gauntlet <= 0:
        raise ValueError(
            f"the gauntlet cadence must be positive, got "
            f"{iterations_per_gauntlet!r}"
        )

    horizon = iterations // iterations_per_gauntlet
    lower = 0.5 * draw_rate
    upper = 1.0 - 0.5 * draw_rate

    def horizon_probability(strength: float) -> float:
        """Probability of re-pinning within the horizon at `strength`."""
        return repin_probability_within(
            horizon,
            single_gauntlet_pass_probability(strength, draw_rate, games, threshold),
            runs,
        )

    if horizon_probability(upper) < target:
        raise ValueError(
            f"no achievable strength re-pins within {iterations} iterations "
            f"({horizon} gauntlets) with probability {target}: even a perfect "
            f"player (strength {upper}, the most a {draw_rate} draw rate "
            f"allows) reaches only {horizon_probability(upper)}"
        )
    if horizon_probability(lower) >= target:
        return lower

    # 80 halvings take the bracket below 1e-24; the loop is fixed-length so the
    # answer is deterministic rather than tolerance-dependent.
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if horizon_probability(middle) >= target:
            upper = middle
        else:
            lower = middle
    return upper


def fireability_table(
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
    runs: int = DEFAULT_RUNS,
    iterations_per_gauntlet: int = DEFAULT_CADENCE,
) -> tuple[RepinWait, ...]:
    """Return one :class:`RepinWait` per strength, in the order given."""
    return tuple(
        expected_gauntlets_until_repin(
            strength, draw_rate, games, threshold, runs, iterations_per_gauntlet
        )
        for strength in strengths
    )


def _format_probability(value: float) -> str:
    """Render a probability so a very small one stays readable."""
    if value == 0.0:
        return "0"
    if value < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _format_wait(value: float) -> str:
    """Render an expected wait, keeping ``inf`` visible as ``inf``."""
    if math.isinf(value):
        return "inf"
    if value >= 1e6:
        return f"{value:.3e}"
    return f"{value:,.1f}"


def format_table(
    rows: Sequence[RepinWait],
    draw_rate: float = DEFAULT_DRAW_RATE,
    games: int = DEFAULT_GAMES,
    threshold: float = DEFAULT_THRESHOLD,
    runs: int = DEFAULT_RUNS,
    iterations_per_gauntlet: int = DEFAULT_CADENCE,
) -> str:
    """Return the human-readable fireability table.

    The header carries every setting the numbers depend on, because a pasted
    table that does not say its draw rate, game count and threshold cannot be
    told apart from a table of a different question.
    """
    lines = [
        f"re-pin fireability -- {games} games/gauntlet, threshold "
        f"{threshold}, {runs} consecutive, draw rate {draw_rate}, "
        f"one gauntlet every {iterations_per_gauntlet} iterations",
        f"{'strength':>9}  {'P(pass)':>12}  {'P(pass x' + str(runs) + ')':>12}  "
        f"{'E[gauntlets]':>14}  {'E[iterations]':>14}",
    ]
    for row in rows:
        lines.append(
            f"{row.strength:>9.4f}  "
            f"{_format_probability(row.pass_probability):>12}  "
            f"{_format_probability(row.consecutive_probability):>12}  "
            f"{_format_wait(row.gauntlets):>14}  "
            f"{_format_wait(row.iterations):>14}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser, whose defaults are campaign 6's own settings."""
    parser = argparse.ArgumentParser(
        prog="repin_fireability.py",
        description=(
            "Exactly how often can a two-consecutive-gauntlet re-pin rule "
            "fire, as a function of true per-game strength?"
        ),
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_STRENGTHS),
        help=(
            "true expected score per game (0.5 is parity); default "
            f"{' '.join(str(value) for value in DEFAULT_STRENGTHS)}"
        ),
    )
    parser.add_argument(
        "--draw-rate",
        type=float,
        default=DEFAULT_DRAW_RATE,
        help=(
            f"per-game draw probability (default {DEFAULT_DRAW_RATE}, campaign "
            f"6's measured rate); it sets the variance of the score and so "
            f"changes the answer in both directions"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help=f"games in one gauntlet (default {DEFAULT_GAMES}: 32 pairs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"score a passing gauntlet must reach (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"consecutive passes a re-pin needs (default {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--cadence",
        type=int,
        default=DEFAULT_CADENCE,
        help=f"iterations between gauntlets (default {DEFAULT_CADENCE})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=(
            f"iterations to invert over (default {DEFAULT_HORIZON}): the "
            f"strength needed for even odds of re-pinning in that time"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the fireability table and the inverted strength; return a status."""
    args = build_parser().parse_args(argv)
    rows = fireability_table(
        args.strengths,
        args.draw_rate,
        args.games,
        args.threshold,
        args.runs,
        args.cadence,
    )
    print(
        format_table(
            rows,
            args.draw_rate,
            args.games,
            args.threshold,
            args.runs,
            args.cadence,
        )
    )

    horizon_gauntlets = args.horizon // args.cadence
    try:
        required = strength_for_repin_within(
            args.horizon,
            args.draw_rate,
            args.games,
            args.threshold,
            args.runs,
            args.cadence,
        )
    except ValueError as error:
        print(f"within {args.horizon} iterations: unreachable -- {error}")
        return 0
    print(
        f"within {args.horizon} iterations ({horizon_gauntlets} gauntlets): "
        f"a true strength of {required:.4f} gives even odds of a re-pin"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
