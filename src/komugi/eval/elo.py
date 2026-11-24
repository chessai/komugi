"""Maximum-likelihood Elo under the Davidson draw model.

The model
=========

Give player ``i`` a rating ``r_i`` in Elo and write its strength on the
multiplicative scale as::

    g_i = 10 ** (r_i / 400)

Davidson (1970) extends Bradley-Terry with a single non-negative draw
parameter ``nu`` that governs how much probability mass sits on the draw, and
places that mass at the *geometric mean* of the two strengths -- so draws are
most likely between evenly matched players and vanish as the mismatch grows.
For a game between ``i`` and ``j``, with ``D_ij = g_i + g_j + nu*sqrt(g_i*g_j)``::

    P(i beats j) = g_i / D_ij
    P(draw)      = nu * sqrt(g_i * g_j) / D_ij
    P(j beats i) = g_j / D_ij

The three terms sum to 1 by construction. ``nu = 0`` recovers plain
Bradley-Terry (no draws); larger ``nu`` means a drawier game.

Note what is *not* in the model: there is no seat (white-advantage) term. The
ladder is built on paired books -- every pair_id is played twice with the
seats swapped (see :func:`komugi.eval.seats.check_paired_books`) -- so the seat
effect cancels to first order in the aggregate and the fitted ratings stay
seat-free. Fitting a seat term instead of designing it away is a separate
ticket, deliberately not smuggled in here.

The likelihood
==============

The likelihood is invariant to which seat a game was played from, so the data
enter only through per-pair aggregates. Let ``n_ij`` be the number of games
between ``i`` and ``j``, ``s_i`` player ``i``'s total score (1 per win, 1/2 per
draw), and ``d`` the total number of drawn games. The exact log-likelihood
maximised here is::

    L(g, nu) = sum_i s_i * log(g_i)
             + d * log(nu)
             - sum_{i<j} n_ij * log(g_i + g_j + nu * sqrt(g_i * g_j))

which is the log of ``prod_games P(observed outcome)`` after collecting terms
(each win contributes ``log g_winner``, each draw contributes
``log nu + (log g_i + log g_j)/2``, and every game contributes ``-log D_ij``).

The fit
=======

``scipy`` is not in this environment, so the optimiser is a hand-rolled
minorization-maximization (MM) iteration in the style of Hunter (2004), which
needs nothing but numpy and is monotone: every step provably does not decrease
``L``. Two minorants do the work, both tight at the current iterate:

* ``-log(x) >= -log(x0) - (x - x0)/x0`` linearises the ``-log D_ij`` terms;
* ``sqrt(g_i*g_j) <= (g_i*sqrt(g_j0/g_i0) + g_j*sqrt(g_i0/g_j0)) / 2`` (AM-GM)
  separates the surviving cross term.

Maximising the resulting separable surrogate gives closed-form updates::

    g_i  <-  s_i / sum_j [ n_ij / D_ij * (1 + nu/2 * sqrt(g_j / g_i)) ]
    nu   <-  d / sum_{i<j} [ n_ij * sqrt(g_i * g_j) / D_ij ]

Ratings are re-anchored (and clamped) on the log scale after every step, which
both fixes the model's one gauge freedom -- ``L`` is invariant to scaling every
``g_i`` by a common constant -- and keeps the iteration numerically boring.

Identifiability
===============

The MLE is finite only if the comparison graph is connected and no subset of
players scores 100% (or 0%) against the rest; otherwise ratings run off to
infinity. The fit clamps at :data:`_MAX_LOG_GAMMA` and reports
``converged=False`` in that case rather than returning a saturated rating that
looks like a measurement. Callers should treat ``converged=False`` as "this
ladder is not identified from this data", not as "try more iterations".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, MatchSet

_ELO_PER_LOG_UNIT: float = 400.0 / math.log(10.0)
"""Elo points per natural-log unit of strength: ``g = exp(r / _ELO_PER_LOG_UNIT)``."""

_MAX_LOG_GAMMA: float = 40.0
"""Clamp on ``|log g|`` (~6949 Elo from the anchor); saturating means unidentified."""

_MAX_NU: float = 1e6
"""Clamp on the draw parameter, for the all-draws degenerate case."""

_TINY: float = 1e-300
"""Denominator floor, purely to keep a degenerate step from dividing by zero."""

_GAMMA_FLOOR: float = 1e-300
"""Strength floor, so a 0%-scoring player yields ``-inf`` free logs."""

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EloResult:
    """The outcome of a fit: ratings, the fitted draw parameter, convergence.

    ``ratings`` is keyed by player id and anchored so the anchor player sits
    exactly at the requested ``anchor_elo``. ``converged`` is ``False`` both
    when the iteration ran out of budget and when the maximum likelihood is
    unbounded (see the module docstring) -- in either case the ratings are not
    a measurement.
    """

    ratings: dict[str, float]
    draw_param: float
    converged: bool
    iterations: int


@dataclass(frozen=True, slots=True)
class _Aggregates:
    """Sufficient statistics of a match set under the Davidson likelihood."""

    players: tuple[str, ...]
    games: Matrix
    """``games[i, j] == games[j, i]``: games played between i and j."""
    score: Vector
    """``score[i]``: 1 per win, 1/2 per draw, summed over both seats."""
    total_draws: float
    """Number of drawn games in the whole match set."""


def _aggregate(match_set: MatchSet) -> _Aggregates:
    """Collapse a match set to the per-pair counts the likelihood depends on.

    Seat is dropped here on purpose: the Davidson likelihood is a function of
    the unordered pair only, so this is exactly the point where the seat-free
    modelling assumption is made visible.
    """
    players = match_set.players()
    index = {player: i for i, player in enumerate(players)}
    size = len(players)
    games: Matrix = np.zeros((size, size), dtype=np.float64)
    score: Vector = np.zeros(size, dtype=np.float64)
    total_draws = 0.0
    for record in match_set.records:
        i = index[record.white]
        j = index[record.black]
        games[i, j] += 1.0
        games[j, i] += 1.0
        if record.result == DRAW:
            total_draws += 1.0
            score[i] += DRAW
            score[j] += DRAW
        elif record.result == WHITE_WIN:
            score[i] += WHITE_WIN
        else:
            assert record.result == BLACK_WIN  # guaranteed by GameRecord
            score[j] += WHITE_WIN
    return _Aggregates(
        players=players, games=games, score=score, total_draws=total_draws
    )


def _initial_draw_param(total_games: float, total_draws: float) -> float:
    """Return the ``nu`` that reproduces the observed draw rate at equal strength.

    With every ``g_i == 1`` the model's draw probability is ``nu / (2 + nu)``,
    so inverting the observed rate is a cheap, data-driven starting point.
    """
    if total_draws <= 0.0:
        return 0.0
    rate = total_draws / total_games
    if rate >= 1.0:
        return _MAX_NU
    return min(2.0 * rate / (1.0 - rate), _MAX_NU)


def _initial_theta(stats: _Aggregates, anchor_index: int) -> Vector:
    """Return a warm start: the log-odds of each player's observed score rate.

    Starting every player at even (``theta = 0``) makes the MM iteration spend
    its first few hundred steps merely spreading the field out. The log-odds of
    the score rate is free, deterministic, and lands within a factor of two of
    the answer over the whole draw-rate range, which is what keeps the default
    iteration budget comfortable rather than marginal.
    """
    played: Vector = stats.games.sum(axis=1)
    odds: Vector = np.maximum(stats.score, _GAMMA_FLOOR) / np.maximum(
        played - stats.score, _GAMMA_FLOOR
    )
    theta: Vector = np.clip(np.log(odds), -_MAX_LOG_GAMMA, _MAX_LOG_GAMMA)
    anchored: Vector = np.clip(
        theta - theta[anchor_index], -_MAX_LOG_GAMMA, _MAX_LOG_GAMMA
    )
    return anchored


def fit_elo(
    match_set: MatchSet,
    anchor: str,
    anchor_elo: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> EloResult:
    """Fit Davidson-model Elo ratings by maximum likelihood.

    Deterministic: the same match set (in the same record order) and the same
    hyper-parameters produce bit-identical ratings, with no randomness and no
    dependence on dict iteration order beyond the sorted player list.

    Args:
        match_set: the games to fit; must be non-empty.
        anchor: player id pinned to `anchor_elo`, fixing the model's gauge.
        anchor_elo: the rating the anchor is pinned to.
        tol: convergence threshold on the largest per-iteration change in any
            log-strength or in the draw parameter.
        max_iter: iteration budget; exhausting it yields ``converged=False``.

    Returns:
        An :class:`EloResult` whose ``ratings`` cover exactly the players that
        appear in `match_set`.

    Raises:
        ValueError: if the match set is empty, `anchor` never played, or the
            hyper-parameters are non-positive.
    """
    if len(match_set) == 0:
        raise ValueError("cannot fit Elo to an empty match set")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol!r}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter!r}")

    stats = _aggregate(match_set)
    if anchor not in stats.players:
        raise ValueError(
            f"anchor {anchor!r} does not appear in the match set; "
            f"players are {list(stats.players)}"
        )
    anchor_index = stats.players.index(anchor)

    games = stats.games
    theta: Vector = _initial_theta(stats, anchor_index)
    gamma: Vector = np.exp(theta)
    nu = _initial_draw_param(float(len(match_set)), stats.total_draws)

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        # --- strength step: MM update at the current (gamma, nu) ------------
        root = np.sqrt(np.outer(gamma, gamma))
        denom = gamma[:, None] + gamma[None, :] + nu * root
        # games has a zero diagonal, so self-pairs drop out of every sum.
        cross = np.sqrt(gamma[None, :] / gamma[:, None])  # [i, j] = sqrt(g_j/g_i)
        coeff: Vector = (games / denom * (1.0 + 0.5 * nu * cross)).sum(axis=1)
        new_gamma: Vector = stats.score / np.maximum(coeff, _TINY)

        # Re-anchor on the log scale: fixes the gauge and bounds the iterate.
        new_theta: Vector = np.log(np.maximum(new_gamma, _GAMMA_FLOOR))
        new_theta = np.clip(
            new_theta - new_theta[anchor_index], -_MAX_LOG_GAMMA, _MAX_LOG_GAMMA
        )
        new_gamma = np.exp(new_theta)

        # --- draw step: MM update at the updated strengths -------------------
        root = np.sqrt(np.outer(new_gamma, new_gamma))
        denom = new_gamma[:, None] + new_gamma[None, :] + nu * root
        # Halved because the matrix sum counts each unordered pair twice.
        weight = 0.5 * float((games * root / denom).sum())
        new_nu = (
            0.0
            if stats.total_draws <= 0.0
            else min(stats.total_draws / max(weight, _TINY), _MAX_NU)
        )

        delta = max(float(np.abs(new_theta - theta).max()), abs(new_nu - nu))
        theta, gamma, nu = new_theta, new_gamma, new_nu
        if delta < tol:
            converged = True
            break

    # A saturated iterate means the likelihood is unbounded, not that the
    # optimiser succeeded: report it as a failure to converge.
    if bool(np.any(np.abs(theta) >= _MAX_LOG_GAMMA)) or nu >= _MAX_NU:
        converged = False

    ratings = {
        player: anchor_elo + float(theta[i]) * _ELO_PER_LOG_UNIT
        for i, player in enumerate(stats.players)
    }
    return EloResult(
        ratings=ratings, draw_param=nu, converged=converged, iterations=iterations
    )


def log_likelihood(match_set: MatchSet, ratings: dict[str, float], nu: float) -> float:
    """Return the exact Davidson log-likelihood of `match_set` at the given fit.

    Written out game by game from the three probabilities in the module
    docstring -- independent of the aggregated form the optimiser uses, so a
    test can check the fit really is a maximum rather than merely a fixed
    point of the update rule.
    """
    if nu < 0.0:
        raise ValueError(f"draw parameter must be non-negative, got {nu!r}")
    total = 0.0
    for record in match_set.records:
        white = math.exp(ratings[record.white] / _ELO_PER_LOG_UNIT)
        black = math.exp(ratings[record.black] / _ELO_PER_LOG_UNIT)
        root = math.sqrt(white * black)
        denom = white + black + nu * root
        if record.result == WHITE_WIN:
            numerator = white
        elif record.result == BLACK_WIN:
            numerator = black
        else:
            numerator = nu * root
        if numerator <= 0.0:
            return -math.inf
        total += math.log(numerator) - math.log(denom)
    return total
