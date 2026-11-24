"""Per-seat accounting, and the paired-book audit the ladder's validity rests on.

:mod:`komugi.eval.elo` deliberately fits no white-advantage term: it assumes
the seat effect has been *designed out* of the data by playing every opening
book twice, once from each seat. This module is where that assumption stops
being an assumption.

Two things live here:

* :func:`seat_matrix` -- exact win/draw/loss counts for every ordered
  ``(white, black)`` seating. Nothing is inferred or symmetrised; the numbers
  are what a caller needs to *see* a seat bias before deciding it is small.
* :func:`check_paired_books` -- the audit: every ``pair_id`` must appear
  exactly twice, with the seats swapped and the same ``book_id``. Anything
  else is returned as a :class:`PairViolation` rather than raised, so a caller
  can report every problem in a run at once instead of one per re-run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet

GAME_COUNT: str = "game_count"
"""Violation reason: a pair_id was not played exactly twice."""

SEATS_NOT_SWAPPED: str = "seats_not_swapped"
"""Violation reason: the two games of a pair were not mirror seatings."""

BOOK_MISMATCH: str = "book_mismatch"
"""Violation reason: the two games of a pair started from different books."""


@dataclass(frozen=True, slots=True)
class SeatTally:
    """Outcome counts for one ordered seating, from the white player's view.

    ``wins + draws + losses == games`` always holds; it is asserted by
    :func:`seat_matrix`'s construction rather than left to the caller.
    """

    white: str
    black: str
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        """Return white's score in this seating: 1 per win, 1/2 per draw."""
        return self.wins + DRAW * self.draws

    @property
    def score_rate(self) -> float:
        """Return white's score per game; ``0.0`` when the seating is unplayed."""
        if self.games == 0:
            return 0.0
        return self.score / self.games


@dataclass(frozen=True, slots=True)
class PairViolation:
    """One way a ``pair_id`` failed the paired-book contract."""

    pair_id: str
    reason: str
    detail: str


def seat_matrix(match_set: MatchSet) -> dict[tuple[str, str], SeatTally]:
    """Return a tally per ordered ``(white, black)`` seating that actually occurred.

    Seatings with no games are absent rather than present-and-zero: the keys
    are a record of what was played. Iteration order is sorted by
    ``(white, black)`` so any report built from it is reproducible.
    """
    counts: dict[tuple[str, str], list[int]] = {}
    for record in match_set.records:
        cell = counts.setdefault(record.seats, [0, 0, 0])
        if record.result == WHITE_WIN:
            cell[0] += 1
        elif record.result == DRAW:
            cell[1] += 1
        else:
            assert record.result == BLACK_WIN  # guaranteed by GameRecord
            cell[2] += 1
    return {
        (white, black): SeatTally(
            white=white,
            black=black,
            games=wins + draws + losses,
            wins=wins,
            draws=draws,
            losses=losses,
        )
        for (white, black), (wins, draws, losses) in sorted(counts.items())
    }


def check_paired_books(match_set: MatchSet) -> tuple[PairViolation, ...]:
    """Audit the paired-book structure, returning every violation found.

    A well-formed ``pair_id`` holds exactly two games: same ``book_id``, same
    two players, opposite seats. A pair with the wrong number of games is
    reported and then skipped -- with the partner missing or a third game
    present there is no unambiguous seat or book comparison left to make, and
    piling on derived complaints would just bury the real one.

    Returns:
        Violations sorted by ``(pair_id, reason)``; empty means the match set
        is properly paired.
    """
    grouped: dict[str, list[GameRecord]] = defaultdict(list)
    for record in match_set.records:
        grouped[record.pair_id].append(record)

    violations: list[PairViolation] = []
    for pair_id, records in grouped.items():
        if len(records) != 2:
            violations.append(
                PairViolation(
                    pair_id=pair_id,
                    reason=GAME_COUNT,
                    detail=f"expected 2 games, found {len(records)}",
                )
            )
            continue
        first, second = records
        if first.seats != (second.black, second.white):
            violations.append(
                PairViolation(
                    pair_id=pair_id,
                    reason=SEATS_NOT_SWAPPED,
                    detail=(
                        f"{first.white} vs {first.black} then "
                        f"{second.white} vs {second.black}"
                    ),
                )
            )
        if first.book_id != second.book_id:
            violations.append(
                PairViolation(
                    pair_id=pair_id,
                    reason=BOOK_MISMATCH,
                    detail=f"{first.book_id!r} then {second.book_id!r}",
                )
            )
    return tuple(sorted(violations, key=lambda v: (v.pair_id, v.reason)))
