"""Synthetic match generation: ground truth to point the instrument at.

The point of this module is that the Elo fit can be checked against ratings
that are *known*, not merely plausible. Games are sampled straight from the
Davidson probabilities :mod:`komugi.eval.elo` claims to invert, plus an
explicit seat advantage the fit does **not** model -- so a planted-rating test
also measures how much the seat effect leaks into the answer.

Randomness is entirely local: `seed` is required, the generator is built here,
and nothing touches ``numpy.random``'s global state. The same seed and the
same arguments produce the same games, in the same order, forever.

Structure mirrors what a real ladder run produces: every matchup is played as
paired books -- the same opening twice, seats swapped, one ``pair_id`` -- so
generated data passes :func:`komugi.eval.seats.check_paired_books` by
construction, and a test that plants a violation has to do so deliberately.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np

from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet

_ELO_PER_LOG10_UNIT: float = 400.0
"""Elo points per decade of strength ratio: ``g = 10 ** (r / 400)``."""


def davidson_probabilities(
    rating_delta: float, draw_param: float
) -> tuple[float, float, float]:
    """Return ``(P(white win), P(draw), P(black win))`` for a rating gap.

    `rating_delta` is white's rating minus black's, in Elo, *including* any
    seat advantage. Derived by dividing every term of the Davidson model
    through by black's strength, which leaves the gap as the only free
    quantity::

        u = 10 ** (delta / 400)
        D = u + 1 + nu * sqrt(u)
        (u / D, nu * sqrt(u) / D, 1 / D)

    Raises:
        ValueError: if `draw_param` is negative.
    """
    if draw_param < 0.0:
        raise ValueError(f"draw_param must be non-negative, got {draw_param!r}")
    ratio = math.pow(10.0, rating_delta / _ELO_PER_LOG10_UNIT)
    denom = ratio + 1.0 + draw_param * math.sqrt(ratio)
    return (ratio / denom, draw_param * math.sqrt(ratio) / denom, 1.0 / denom)


def _sample_result(
    probabilities: tuple[float, float, float], rng: np.random.Generator
) -> float:
    """Draw one game result from ``(p_white, p_draw, p_black)``."""
    p_white, p_draw, _ = probabilities
    draw = float(rng.random())
    if draw < p_white:
        return WHITE_WIN
    if draw < p_white + p_draw:
        return DRAW
    return BLACK_WIN


def _planned_matchups(
    players: Sequence[str],
    schedule: Sequence[tuple[str, str]] | None,
    pairs_per_matchup: int,
    pairs_by_matchup: Mapping[tuple[str, str], int] | None,
) -> tuple[tuple[str, str, int], ...]:
    """Return ``(first, second, pairs)`` per matchup, in play order.

    With `schedule` ``None`` this is the full round robin over the sorted
    players at `pairs_per_matchup` -- byte for byte what this module produced
    before a schedule could be named, which is the property
    ``tests/eval/test_elo_planted.py`` pins. With a schedule it is exactly the
    named matchups, in the order given, so a **sparse** graph -- §D2.2's
    six-matchup epoch schedule, whose coverage and power are what U5 and
    §D5.3's table question -- can be planted at all.

    Raises:
        ValueError: on a self-matchup, a repeated matchup (which would collide
            on ``pair_id``), an unrated player, a non-positive game count, or
            `pairs_by_matchup` supplied with no schedule to apply it to.
    """
    if schedule is None:
        if pairs_by_matchup is not None:
            raise ValueError(
                f"pairs_by_matchup was given without a schedule, and the round "
                f"robin has no way to honour it: it plays every matchup at "
                f"pairs_per_matchup={pairs_per_matchup!r} and would silently "
                f"return a uniform complete graph. §D2.2's links are not all "
                f"the same size -- the in-family link is 256 games and the §D8 "
                f"visits legs are 32 and 16 -- so a caller planting "
                f"{sorted(pairs_by_matchup)} meant to plant a *schedule*, and "
                f"ignoring the argument would publish a coverage number for an "
                f"instrument that was never played"
            )
        return tuple(
            (first, second, pairs_per_matchup)
            for first, second in itertools.combinations(players, 2)
        )
    known = set(players)
    planned: list[tuple[str, str, int]] = []
    seen: set[frozenset[str]] = set()
    for first, second in schedule:
        if first == second:
            raise ValueError(f"a matchup cannot be a self-link: {first!r}")
        for name in (first, second):
            if name not in known:
                raise ValueError(
                    f"scheduled player {name!r} has no planted rating; players "
                    f"are {sorted(known)}"
                )
        key = frozenset({first, second})
        if key in seen:
            raise ValueError(
                f"matchup {first!r} x {second!r} is scheduled twice; the two "
                f"runs would share a pair_id and fail check_paired_books with "
                f"GAME_COUNT violations, exactly as a real repeat does "
                f"(arena.py:645-654)"
            )
        seen.add(key)
        pairs = pairs_per_matchup
        if pairs_by_matchup is not None:
            pairs = pairs_by_matchup.get(
                (first, second), pairs_by_matchup.get((second, first), pairs)
            )
        if pairs <= 0:
            raise ValueError(
                f"matchup {first!r} x {second!r} must play at least one paired "
                f"book, got {pairs!r}"
            )
        planned.append((first, second, pairs))
    return tuple(planned)


def simulate_match_set(
    ratings: Mapping[str, float],
    *,
    seed: int,
    draw_param: float = 1.0,
    seat_advantage: float = 0.0,
    pairs_per_matchup: int = 1,
    book_ids: Sequence[str] | None = None,
    schedule: Sequence[tuple[str, str]] | None = None,
    pairs_by_matchup: Mapping[tuple[str, str], int] | None = None,
) -> MatchSet:
    """Sample paired-book games from known ratings, round robin or to a schedule.

    By default every unordered matchup is played `pairs_per_matchup` times, and
    each of those is two games from one book with the seats swapped -- so the
    returned set holds ``len(ratings) * (len(ratings) - 1) * pairs_per_matchup``
    games. Passing `schedule` plays **only** the named matchups instead, which
    is what lets a sparse comparison graph be planted: the pool's own schedule
    is six matchups over ~20 entrants (§D2.2), not a round robin, and a
    coverage or power claim measured on a complete graph is a claim about a
    different instrument.

    Args:
        ratings: the planted true rating per player id; at least two players.
        seed: seeds a local ``numpy`` generator. Required -- there is no
            global-randomness path into this function.
        draw_param: the Davidson ``nu`` to sample with.
        seat_advantage: Elo added to whoever holds white. Sampled into the
            data but *not* modelled by the fit, on purpose.
        pairs_per_matchup: paired books per matchup (two games each), and the
            default for any matchup `pairs_by_matchup` does not name.
        book_ids: opening-book labels, cycled across the paired books of a
            matchup; defaults to synthetic ``book0000``-style ids.
        schedule: the unordered matchups to play, in play order. ``None`` -- the
            default -- plays the full round robin, byte for byte as before.
        pairs_by_matchup: per-matchup paired-book counts, keyed by either
            ordering of the pair. The pool's links are not all the same size:
            the in-family link is 256 games and the visits legs are 32 and 16
            (§D2.2(b)1, §D8.3). Requires `schedule`: the round-robin path has
            no matchup list to apply it to, and silently ignoring it is how a
            coverage artefact gets published for an instrument that was never
            planted.

    Returns:
        A :class:`MatchSet` in a deterministic play order: matchups in sorted
        player order (or in `schedule` order), books in index order, white-first
        within each pair.

    Raises:
        ValueError: if fewer than two players, `pairs_per_matchup` is
            non-positive, `book_ids` is supplied but empty, `pairs_by_matchup`
            is supplied without a `schedule`, or `schedule` names an unrated
            player, a self-link or the same matchup twice.
    """
    players = tuple(sorted(ratings))
    if len(players) < 2:
        raise ValueError(f"need at least two players to play, got {len(players)}")
    if pairs_per_matchup <= 0:
        raise ValueError(
            f"pairs_per_matchup must be positive, got {pairs_per_matchup!r}"
        )
    if book_ids is not None and len(book_ids) == 0:
        raise ValueError("book_ids must be non-empty when supplied")

    planned = _planned_matchups(
        players, schedule, pairs_per_matchup, pairs_by_matchup
    )
    longest = max((pairs for _, _, pairs in planned), default=pairs_per_matchup)
    books: Sequence[str] = (
        book_ids
        if book_ids is not None
        else tuple(f"book{index:04d}" for index in range(longest))
    )
    rng = np.random.default_rng(seed)
    records: list[GameRecord] = []
    for first, second, pairs in planned:
        for index in range(pairs):
            book_id = books[index % len(books)]
            pair_id = f"{first}:{second}:{index:05d}"
            for white, black in ((first, second), (second, first)):
                delta = ratings[white] + seat_advantage - ratings[black]
                result = _sample_result(
                    davidson_probabilities(delta, draw_param), rng
                )
                records.append(
                    GameRecord(
                        white=white,
                        black=black,
                        result=result,
                        book_id=book_id,
                        pair_id=pair_id,
                    )
                )
    return MatchSet.of(records)
