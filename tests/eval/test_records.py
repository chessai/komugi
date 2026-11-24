"""Record-level validation: the bad game gets rejected where it is created.

Everything downstream trusts these invariants instead of re-checking them, so
the invariants have to be load-bearing at construction time rather than
documented and hoped for.
"""

from __future__ import annotations

import dataclasses

import pytest

from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    KNOWN_FAMILIES,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
    is_known_family,
    roster_families,
)


def _game(white: str, black: str, result: float, pair: str) -> GameRecord:
    """Build one record with throwaway book/pair labels."""
    return GameRecord(
        white=white, black=black, result=result, book_id="book0000", pair_id=pair
    )


def test_records_are_frozen_and_reject_unknown_fields() -> None:
    """Same contract as komugi.config: typos raise, mutation raises."""
    record = _game("a", "b", DRAW, "p0")
    # setattr, not `record.result = ...`: the direct form is a static error, and
    # what is under test here is the *runtime* guard behind it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, "result", WHITE_WIN)
    with pytest.raises(TypeError):
        GameRecord(white="a", black="b", reslut=DRAW, book_id="k", pair_id="p")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Player(id="a", famliy="komugi")  # type: ignore[call-arg]


def test_invalid_results_are_rejected() -> None:
    """Only win/draw/loss exist; 0.75 is a scoring bug, not a game."""
    for bad in (0.75, -1.0, 2.0):
        with pytest.raises(ValueError, match="result must be one of"):
            _game("a", "b", bad, "p0")


def test_degenerate_identifiers_are_rejected() -> None:
    """Empty ids and self-play are caught at the record, not in a rating table."""
    with pytest.raises(ValueError, match="cannot play itself"):
        _game("a", "a", DRAW, "p0")
    with pytest.raises(ValueError, match="non-empty"):
        _game("", "b", DRAW, "p0")
    with pytest.raises(ValueError, match="non-empty"):
        _game("a", "b", DRAW, "   ")
    with pytest.raises(ValueError, match="non-empty"):
        Player(id="a", family="")


def test_record_orientation_helpers() -> None:
    """The white's-perspective convention, read back out."""
    white_win = _game("a", "b", WHITE_WIN, "p0")
    black_win = _game("a", "b", BLACK_WIN, "p1")
    drawn = _game("a", "b", DRAW, "p2")
    assert (white_win.winner, white_win.is_draw) == ("a", False)
    assert (black_win.winner, black_win.is_draw) == ("b", False)
    assert (drawn.winner, drawn.is_draw) == (None, True)
    assert white_win.seats == ("a", "b")


def test_match_set_summaries() -> None:
    """Player/pair/book views are sorted, deduplicated and seat-independent."""
    match_set = MatchSet.of(
        [
            _game("b", "a", WHITE_WIN, "p0"),
            _game("a", "b", DRAW, "p0"),
            _game("a", "c", BLACK_WIN, "p1"),
        ]
    )
    assert len(match_set) == 3
    assert match_set.players() == ("a", "b", "c")
    assert match_set.pair_ids() == ("p0", "p1")
    assert match_set.book_ids() == ("book0000",)
    assert match_set.games_played_by("a") == 3
    assert match_set.games_played_by("c") == 1
    # a: loss as black, draw as white, loss as white -> 0.5 over three games.
    assert match_set.score_of("a") == 0.5
    assert match_set.score_of("b") == 1.5
    assert match_set.score_of("c") == 1.0
    assert match_set.score_of("nobody") == 0.0


def test_roster_families() -> None:
    """A roster maps ids to families and refuses to merge two engines into one."""
    roster = [Player(id="komugi-t1", family="komugi"), Player(id="v1-a", family="v1")]
    assert roster_families(roster) == {"komugi-t1": "komugi", "v1-a": "v1"}
    assert roster_families([*roster, Player(id="v1-a", family="v1")])["v1-a"] == "v1"
    with pytest.raises(ValueError, match="two families"):
        roster_families([*roster, Player(id="v1-a", family="ab")])


def test_known_families_are_advisory() -> None:
    """The constant documents the ladder; it does not gate new engines."""
    assert {"komugi", "v1", "external", "ab", "rawmcts"} == set(KNOWN_FAMILIES)
    assert is_known_family("komugi")
    assert not is_known_family("stockfish")
    assert Player(id="sf-1", family="stockfish").family == "stockfish"
