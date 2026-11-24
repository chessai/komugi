"""Seat accounting exactness, and the paired-book audit's negative tests.

The seat matrix is pure bookkeeping, so it is tested against a hand-built set
small enough to count by eye: if this drifts, every seat-bias claim built on
it drifts silently. The paired-book audit gets the harder half of the file --
an audit that never fires is indistinguishable from one that cannot fire, so
each violation mode is planted deliberately and asserted to be *reported*.
"""

from __future__ import annotations

from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet
from komugi.eval.seats import (
    BOOK_MISMATCH,
    GAME_COUNT,
    SEATS_NOT_SWAPPED,
    check_paired_books,
    seat_matrix,
)


def _game(
    white: str,
    black: str,
    result: float,
    *,
    book: str = "book0000",
    pair: str = "pair0000",
) -> GameRecord:
    """Build one record, keeping the tables below readable."""
    return GameRecord(
        white=white, black=black, result=result, book_id=book, pair_id=pair
    )


def _paired(
    first: str, second: str, first_result: float, second_result: float, tag: str
) -> list[GameRecord]:
    """Build a well-formed paired book: same book, seats swapped, one pair_id."""
    return [
        _game(first, second, first_result, book=f"book-{tag}", pair=f"pair-{tag}"),
        _game(second, first, second_result, book=f"book-{tag}", pair=f"pair-{tag}"),
    ]


# a-as-white: 2 wins, 1 draw, 1 loss. b-as-white: 1 win, 1 loss.
# a-vs-c is played from one seat only, so (c, a) must be absent, not zero.
_HAND_BUILT = MatchSet.of(
    [
        _game("a", "b", WHITE_WIN, pair="p0"),
        _game("a", "b", WHITE_WIN, pair="p1"),
        _game("a", "b", DRAW, pair="p2"),
        _game("a", "b", BLACK_WIN, pair="p3"),
        _game("b", "a", WHITE_WIN, pair="p4"),
        _game("b", "a", BLACK_WIN, pair="p5"),
        _game("a", "c", DRAW, pair="p6"),
    ]
)


def test_seat_matrix_counts_are_exact() -> None:
    """Every cell matches a hand count, and the parts sum to the whole."""
    matrix = seat_matrix(_HAND_BUILT)

    a_white = matrix[("a", "b")]
    assert (a_white.games, a_white.wins, a_white.draws, a_white.losses) == (4, 2, 1, 1)
    assert a_white.score == 2.5
    assert a_white.score_rate == 2.5 / 4

    b_white = matrix[("b", "a")]
    assert (b_white.games, b_white.wins, b_white.draws, b_white.losses) == (2, 1, 0, 1)
    assert b_white.score == 1.0

    a_c = matrix[("a", "c")]
    assert (a_c.games, a_c.wins, a_c.draws, a_c.losses) == (1, 0, 1, 0)

    for tally in matrix.values():
        assert tally.wins + tally.draws + tally.losses == tally.games
    assert sum(tally.games for tally in matrix.values()) == len(_HAND_BUILT)


def test_seat_matrix_omits_unplayed_seatings_and_sorts_keys() -> None:
    """Keys record what was played, in a reproducible order."""
    matrix = seat_matrix(_HAND_BUILT)
    assert ("c", "a") not in matrix
    assert list(matrix) == [("a", "b"), ("a", "c"), ("b", "a")]
    assert seat_matrix(MatchSet.of([])) == {}


def test_seat_matrix_is_orientation_faithful() -> None:
    """A win as white is not a win as black: the two seatings stay distinct."""
    match_set = MatchSet.of(_paired("a", "b", WHITE_WIN, WHITE_WIN, "0"))
    matrix = seat_matrix(match_set)
    # Both games were won by whoever held white, so each seating shows 1-0-0.
    assert matrix[("a", "b")].wins == 1
    assert matrix[("b", "a")].wins == 1
    assert matrix[("a", "b")].losses == 0


def test_well_formed_pairs_report_nothing() -> None:
    """The audit is silent on data that satisfies the contract."""
    records = _paired("a", "b", WHITE_WIN, DRAW, "0") + _paired(
        "a", "c", DRAW, BLACK_WIN, "1"
    )
    assert check_paired_books(MatchSet.of(records)) == ()


def test_unpaired_record_is_reported() -> None:
    """The negative test: a lone game must not slip through the audit."""
    records = _paired("a", "b", WHITE_WIN, DRAW, "0")
    records.append(_game("a", "c", WHITE_WIN, book="book-lonely", pair="pair-lonely"))

    violations = check_paired_books(MatchSet.of(records))

    assert len(violations) == 1
    assert violations[0].pair_id == "pair-lonely"
    assert violations[0].reason == GAME_COUNT
    assert "found 1" in violations[0].detail


def test_over_played_pair_is_reported() -> None:
    """Three games under one pair_id is just as broken as one."""
    records = _paired("a", "b", WHITE_WIN, DRAW, "0")
    records.append(_game("a", "b", DRAW, book="book-0", pair="pair-0"))

    violations = check_paired_books(MatchSet.of(records))

    assert [(v.pair_id, v.reason) for v in violations] == [("pair-0", GAME_COUNT)]
    assert "found 3" in violations[0].detail


def test_unswapped_seats_are_reported() -> None:
    """Two games from the same seat defeat the point of pairing."""
    records = [
        _game("a", "b", WHITE_WIN, book="book-0", pair="pair-0"),
        _game("a", "b", BLACK_WIN, book="book-0", pair="pair-0"),
    ]
    violations = check_paired_books(MatchSet.of(records))
    assert [(v.pair_id, v.reason) for v in violations] == [
        ("pair-0", SEATS_NOT_SWAPPED)
    ]


def test_mismatched_books_are_reported() -> None:
    """Swapped seats over *different* openings are not a paired book."""
    records = [
        _game("a", "b", WHITE_WIN, book="book-0", pair="pair-0"),
        _game("b", "a", WHITE_WIN, book="book-1", pair="pair-0"),
    ]
    violations = check_paired_books(MatchSet.of(records))
    assert [(v.pair_id, v.reason) for v in violations] == [("pair-0", BOOK_MISMATCH)]
    assert "'book-0' then 'book-1'" == violations[0].detail


def test_independent_violations_are_all_reported() -> None:
    """One pair can be wrong twice over, and both ways are surfaced, sorted."""
    records = [
        _game("a", "b", WHITE_WIN, book="book-0", pair="pair-0"),
        _game("a", "b", WHITE_WIN, book="book-1", pair="pair-0"),
        _game("x", "y", DRAW, book="book-2", pair="pair-1"),
    ]
    violations = check_paired_books(MatchSet.of(records))
    assert [(v.pair_id, v.reason) for v in violations] == [
        ("pair-0", BOOK_MISMATCH),
        ("pair-0", SEATS_NOT_SWAPPED),
        ("pair-1", GAME_COUNT),
    ]


def test_pair_involving_the_wrong_players_is_reported() -> None:
    """A pair_id reused across different opponents shows up as unswapped seats."""
    records = [
        _game("a", "b", WHITE_WIN, book="book-0", pair="pair-0"),
        _game("c", "a", WHITE_WIN, book="book-0", pair="pair-0"),
    ]
    violations = check_paired_books(MatchSet.of(records))
    assert [(v.pair_id, v.reason) for v in violations] == [
        ("pair-0", SEATS_NOT_SWAPPED)
    ]
