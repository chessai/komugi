"""Immutable records for the evaluation ladder: players, games, match sets.

A rating table is only as trustworthy as the bookkeeping underneath it, so
every record validates itself at construction. A result outside
``{1.0, 0.5, 0.0}``, an empty identifier, or a game a player plays against
themselves raises where the record is *built* rather than showing up as a
quiet skew in an Elo table several thousand games later.

Two conventions are load-bearing for the rest of ``komugi.eval``:

* ``GameRecord.result`` is always from **white's** perspective. That single
  rule is what lets :mod:`komugi.eval.seats` count seats without re-deriving
  orientation, and what :mod:`komugi.eval.elo` assumes when it accumulates
  score.
* ``pair_id`` names a *paired-book unit*: the same opening book played twice
  with the seats swapped. Nothing here enforces that (a single record cannot
  see its partner) -- :func:`komugi.eval.seats.check_paired_books` audits it
  across a whole :class:`MatchSet`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

WHITE_WIN: Final[float] = 1.0
"""``GameRecord.result`` for a win by the player holding the white seat."""

DRAW: Final[float] = 0.5
"""``GameRecord.result`` for a draw."""

BLACK_WIN: Final[float] = 0.0
"""``GameRecord.result`` for a win by the player holding the black seat."""

VALID_RESULTS: Final[frozenset[float]] = frozenset({WHITE_WIN, DRAW, BLACK_WIN})
"""The only three results a decisive-or-drawn two-player game can record."""

KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(
    {"komugi", "v1", "external", "ab", "rawmcts"}
)
"""Engine families the ladder is built around.

Advisory, not enforced: a new engine gets a ``Player`` before it gets an entry
here, and rejecting unknown families would make the records module a
bottleneck on every experiment. Use :func:`is_known_family` when a caller
actually wants to branch on it.
"""


def is_known_family(family: str) -> bool:
    """Return whether `family` is one of the :data:`KNOWN_FAMILIES`."""
    return family in KNOWN_FAMILIES


def _require_nonempty(value: str, field: str) -> None:
    """Raise ``ValueError`` if `value` is empty or blank."""
    if not value or not value.strip():
        raise ValueError(f"{field} must be a non-empty identifier, got {value!r}")


@dataclass(frozen=True, slots=True)
class Player:
    """One rateable participant: a stable id plus the engine family it belongs to.

    The id is what appears in every :class:`GameRecord`; the family is the
    coarser grouping (``'komugi'``, ``'v1'``, ...) used to report a ladder by
    engine lineage rather than by individual configuration.
    """

    id: str
    family: str

    def __post_init__(self) -> None:
        _require_nonempty(self.id, "Player.id")
        _require_nonempty(self.family, "Player.family")


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A single finished game, scored from white's perspective.

    ``book_id`` names the opening position the game started from and
    ``pair_id`` the two-game seat-swapped unit it belongs to; both are opaque
    strings here and are only given meaning by the paired-book audit in
    :mod:`komugi.eval.seats`.
    """

    white: str
    black: str
    result: float
    book_id: str
    pair_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.white, "GameRecord.white")
        _require_nonempty(self.black, "GameRecord.black")
        _require_nonempty(self.book_id, "GameRecord.book_id")
        _require_nonempty(self.pair_id, "GameRecord.pair_id")
        if self.white == self.black:
            raise ValueError(f"a player cannot play itself: {self.white!r}")
        if self.result not in VALID_RESULTS:
            raise ValueError(
                f"GameRecord.result must be one of {sorted(VALID_RESULTS)}, "
                f"got {self.result!r}"
            )

    @property
    def is_draw(self) -> bool:
        """Return whether the game was drawn."""
        return self.result == DRAW

    @property
    def winner(self) -> str | None:
        """Return the winning player's id, or ``None`` for a draw."""
        if self.result == WHITE_WIN:
            return self.white
        if self.result == BLACK_WIN:
            return self.black
        return None

    @property
    def seats(self) -> tuple[str, str]:
        """Return the ordered ``(white, black)`` seating of this game."""
        return (self.white, self.black)


@dataclass(frozen=True, slots=True)
class MatchSet:
    """An immutable collection of games, the unit every estimator consumes.

    Construct from any iterable with :meth:`of`; the stored ``records`` tuple
    keeps the games in the order they were played, which is what makes every
    downstream fit reproducible from the same input.
    """

    records: tuple[GameRecord, ...]

    @classmethod
    def of(cls, records: Iterable[GameRecord]) -> MatchSet:
        """Build a match set from any iterable of records."""
        return cls(tuple(records))

    def __len__(self) -> int:
        """Return the number of games."""
        return len(self.records)

    def players(self) -> tuple[str, ...]:
        """Return every player id that appears in any seat, sorted."""
        seen: set[str] = set()
        for record in self.records:
            seen.add(record.white)
            seen.add(record.black)
        return tuple(sorted(seen))

    def pair_ids(self) -> tuple[str, ...]:
        """Return every distinct ``pair_id``, sorted."""
        return tuple(sorted({record.pair_id for record in self.records}))

    def book_ids(self) -> tuple[str, ...]:
        """Return every distinct ``book_id``, sorted."""
        return tuple(sorted({record.book_id for record in self.records}))

    def games_played_by(self, player: str) -> int:
        """Return how many games `player` appears in, in either seat."""
        return sum(
            1 for record in self.records if player in (record.white, record.black)
        )

    def score_of(self, player: str) -> float:
        """Return `player`'s total score: 1 per win, 0.5 per draw.

        Seat-independent -- this is the quantity the Elo likelihood is a
        function of, so it is defined once, here.
        """
        total = 0.0
        for record in self.records:
            if record.white == player:
                total += record.result
            elif record.black == player:
                total += WHITE_WIN - record.result
        return total


def roster_families(players: Iterable[Player]) -> dict[str, str]:
    """Return an ``id -> family`` map, rejecting contradictory duplicate ids.

    Re-registering the same id with the same family is harmless and allowed;
    re-registering it with a *different* family is the bug this catches, since
    it would silently merge two engines into one row of a ladder.
    """
    families: dict[str, str] = {}
    for player in players:
        previous = families.get(player.id)
        if previous is not None and previous != player.family:
            raise ValueError(
                f"player {player.id!r} declared in two families: "
                f"{previous!r} and {player.family!r}"
            )
        families[player.id] = player.family
    return families
