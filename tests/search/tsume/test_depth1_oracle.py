"""Mate-in-1 against the one oracle that needs no solver: enumeration.

At depth 1 the question "is there a forced mate" is decidable by brute force
-- apply every pseudo-legal move, ask the engine whether the result is mate
-- so the solver can be checked against a procedure with no search in it at
all. That is done here in **both directions** on every committed fixture
board: the enumerator finds a mating move exactly when the solver answers
MATE, and when it does, the solver's move is one of the enumerator's.

One direction alone would be nearly worthless. "Every move the solver finds
is a mate" is satisfied by a solver that finds nothing; "every position the
enumerator proves, the solver proves" is satisfied by one that answers MATE
everywhere. Together they pin the verdict on 26 boards.
"""

from __future__ import annotations

import pytest
from tsume_fixtures import (
    ALL_FIXTURE_BOARDS,
    MARSHAL_CAPTURE_ROWS,
    MINED,
    MinedPuzzle,
    enumeration_keys,
    expected_mined_verdict,
    forced_mate_keys,
)

from komugi.search.tsume import solve_tsume

#: The three mined rows excluded below are the KOM-1108/KOM-1109 false-mate
#: rows: the recorded ``keys`` were mined before KOM-1109 fixed
#: ``cyengine.is_checkmate`` to treat a defender reply that captures the
#: attacker's marshal as an escape, so the frozen mined key -- the move the
#: old, buggy enumeration called mate-in-1 -- is no longer what today's
#: (correct) enumeration finds for these three. ``mined.json`` is left
#: untouched, a frozen record of that historical run; see
#: ``MARSHAL_CAPTURE_ROWS`` for the full story.
_MINED_FOR_ENUMERATION_REPRODUCTION = tuple(
    m for m in MINED if m.name not in MARSHAL_CAPTURE_ROWS
)


@pytest.mark.parametrize(
    "fixture", _MINED_FOR_ENUMERATION_REPRODUCTION, ids=lambda m: m.name
)
def test_mined_keys_reproduce_by_enumeration(fixture: MinedPuzzle) -> None:
    """The recorded keys are what this engine's enumeration finds today.

    A pin on the fixture data itself: the keys were mined by a different run
    of a different script, and if the engine's SAN or move generation had
    drifted since, every comparison below would be against stale strings.
    Excludes the KOM-1108/KOM-1109 false-mate rows (see above): those were
    mined before the fix and intentionally do not reproduce today.
    """
    assert sorted(enumeration_keys(fixture.fen)) == sorted(fixture.keys)


@pytest.mark.parametrize("name", sorted(MARSHAL_CAPTURE_ROWS))
def test_false_mate_mined_rows_no_longer_enumerate_the_old_key(name: str) -> None:
    """KOM-1109 regression guard: the pre-fix false-mate key is gone from enumeration.

    These three rows were mined when ``cyengine.is_checkmate`` wrongly called
    a position mate although the defender could reply by capturing the
    attacker's marshal (KOM-1108/KOM-1109). Fixed, the one recorded key is no
    longer among the moves enumeration calls mate-in-1 at this root.
    """
    fixture = next(m for m in MINED if m.name == name)
    assert set(fixture.keys) & set(enumeration_keys(fixture.fen)) == set()


@pytest.mark.parametrize("fixture", MINED, ids=lambda m: m.name)
def test_mined_puzzle_verdict_at_depth_one(fixture: MinedPuzzle) -> None:
    """Each mined row gets its own expected verdict -- never a bare comparison.

    Seven are forced mates in one. Three are not, under the KOM-258 defender
    semantics, because the defender can answer the mating move by capturing
    the attacker's marshal; see ``MARSHAL_CAPTURE_ROWS`` for why the engine's
    ``is_checkmate`` disagrees and why the solver is the conservative one.
    """
    expected = expected_mined_verdict(fixture.name)
    result = solve_tsume(fixture.fen, 1, verify_checkmate=True)
    assert result.verdict == expected, f"{fixture.name}: {result.verdict}"
    if expected == "MATE":
        assert len(result.line) == 1
        assert result.line[0] in fixture.keys
    else:
        assert result.line == ()


BOARD_PARAMS = [pytest.param(name, fen, id=name) for name, fen in ALL_FIXTURE_BOARDS]


@pytest.mark.parametrize("name,fen", BOARD_PARAMS)
def test_depth_one_agrees_with_enumeration_in_both_directions(
    name: str, fen: str
) -> None:
    """Enumeration finds a forced mate iff the solver proves one at max_plies=1."""
    keys = forced_mate_keys(fen)
    result = solve_tsume(fen, 1, verify_checkmate=True)
    assert result.verdict in ("MATE", "NO_MATE"), f"{name}: {result.verdict}"
    assert (result.verdict == "MATE") == bool(keys), (
        f"{name}: solver said {result.verdict}, enumeration found {keys}"
    )
    if result.verdict == "MATE":
        assert result.line[0] in keys, f"{name}: {result.line[0]} not in {keys}"


def test_the_depth_one_oracle_is_not_vacuous() -> None:
    """Both answers actually occur across the fixture boards.

    A both-directions equivalence over a set where one side never fires is an
    equivalence between two constants.
    """
    proved = [name for name, fen in ALL_FIXTURE_BOARDS if forced_mate_keys(fen)]
    refuted = [name for name, fen in ALL_FIXTURE_BOARDS if not forced_mate_keys(fen)]
    assert len(proved) >= 5, proved
    assert len(refuted) >= 5, refuted
