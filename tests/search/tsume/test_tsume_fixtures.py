"""The composed puzzles: solved, minimal, and replayable on a second engine.

Three assertions per puzzle, and each one closes a different way for this
suite to pass while being wrong:

*Solved* -- MATE at the recorded depth. UNKNOWN is a failure here, not a
shrug: on a set this small inside the default budget it would mean the search
gave up, and a test that accepted it would accept a solver that always gave
up.

*Minimal* -- NO_MATE, exactly, two plies shallower. Without this a solver
that reported MATE on anything would pass the first assertion on every row.

*Replayable* -- the mate line is played back move by move through
``api.Gungi``, the pure-Python rules engine, and checked at every ply. The
solver runs on the compiled engine; for a wrong line to survive this, the
bug would have to exist in both implementations and agree with itself.
"""

from __future__ import annotations

from typing import Final

import pytest
from tsume_fixtures import (
    FRIEND,
    MINED,
    PERTURBATIONS,
    FriendFixture,
    MinedPuzzle,
    Perturbation,
    checking_moves,
)

from komugi.engine import cyboard, cyengine
from komugi.engine.api import Gungi
from komugi.search.tsume import solve_tsume

MULTI_MOVE = tuple(f for f in FRIEND if f.tsume_plies >= 3)

#: The composer's own line for the corrected F09, in the order they gave it:
#: the general stacks on 2-6-2, the marshal climbs onto it at 2-6-3 (its only
#: legal reply), and the samurai drops behind on 4-4-3.
F09_COMPOSER_LINE: Final = (
    "大(4-4-3)(2-6-2)付",
    "帥(2-5-2)(2-6-3)付",
    "新侍(4-4-3)付",
)


def test_friend_set_is_complete() -> None:
    """The set is the 14 unquarantined puzzles, not however many happen to load."""
    assert len(FRIEND) >= 14
    assert len({f.name for f in FRIEND}) == len(FRIEND)


def test_perturbation_set_is_complete() -> None:
    assert len(PERTURBATIONS) == 2
    assert len({p.name for p in PERTURBATIONS}) == len(PERTURBATIONS)


def test_mined_set_is_complete() -> None:
    """The set is the 9 unquarantined mined rows, not however many happen to load.

    It was 10 until KOM-261 quarantined ``iter_827-g200002``, whose root has
    the defender already in check -- a genuine mate-in-1, but not a tsume
    problem. Stated as a floor for the same reason the friend set is: a row
    may be added, but one going missing is a fixture file that failed to
    load, and that must fail loudly here rather than shrink the suite.
    """
    assert len(MINED) >= 9
    assert len({m.name for m in MINED}) == len(MINED)


@pytest.mark.parametrize("fixture", FRIEND, ids=lambda f: f.name)
def test_friend_fixture_hygiene(fixture: FriendFixture) -> None:
    """Both marshals on the board, and an odd grade.

    The marshals are the load-bearing part: ``cyengine.is_checkmate`` returns
    False, silently, on a board that does not have exactly two of them, so a
    fixture missing one would turn every mate assertion below into a test of
    nothing.
    """
    placement = fixture.fen.split(" ")[0]
    assert "M" in placement, fixture.fen
    assert "m" in placement, fixture.fen
    assert fixture.tsume_plies % 2 == 1
    assert fixture.tsume_plies >= 1


@pytest.mark.parametrize("fixture", PERTURBATIONS, ids=lambda p: p.name)
def test_perturbation_hygiene(fixture: Perturbation) -> None:
    placement = fixture.fen.split(" ")[0]
    assert "M" in placement, fixture.fen
    assert "m" in placement, fixture.fen
    assert fixture.tsume_plies % 2 == 1


@pytest.mark.parametrize("fixture", MINED, ids=lambda m: m.name)
def test_mined_fixture_hygiene(fixture: MinedPuzzle) -> None:
    placement = fixture.fen.split(" ")[0]
    assert "M" in placement, fixture.fen
    assert "m" in placement, fixture.fen
    assert fixture.mate_in == 1
    assert fixture.keys


@pytest.mark.parametrize("fixture", FRIEND, ids=lambda f: f.name)
def test_friend_puzzle_is_mate_at_its_grade(fixture: FriendFixture) -> None:
    result = solve_tsume(fixture.fen, fixture.tsume_plies, verify_checkmate=True)
    assert result.verdict == "MATE", f"{fixture.name}: {result.verdict}"
    assert result.line, "a MATE verdict without a line proves nothing"
    assert len(result.line) % 2 == 1, result.line
    assert len(result.line) == fixture.tsume_plies, result.line


@pytest.mark.parametrize("fixture", MULTI_MOVE, ids=lambda f: f.name)
def test_friend_puzzle_is_not_mate_two_plies_shallower(fixture: FriendFixture) -> None:
    """Exactly NO_MATE -- an UNKNOWN here would prove nothing about minimality."""
    result = solve_tsume(fixture.fen, fixture.tsume_plies - 2, verify_checkmate=True)
    assert result.verdict == "NO_MATE", f"{fixture.name}: {result.verdict}"
    assert result.line == ()


@pytest.mark.parametrize("fixture", PERTURBATIONS, ids=lambda p: p.name)
def test_perturbation_is_not_mate(fixture: Perturbation) -> None:
    """The same puzzle with the mating resource deleted must lose the mate."""
    result = solve_tsume(fixture.fen, fixture.tsume_plies, verify_checkmate=True)
    assert result.verdict == fixture.expect, f"{fixture.name}: {result.verdict}"


@pytest.mark.parametrize("fixture", FRIEND, ids=lambda f: f.name)
def test_friend_mate_line_replays_through_the_api(fixture: FriendFixture) -> None:
    """Every ply of the line is legal to ``api.Gungi``, and it ends in mate.

    The cross-engine referee. ``Gungi.move`` raises on an illegal move, so
    this fails loudly if the solver's SAN or its move semantics have drifted
    from the rules engine; the per-ply check assertion fails if the "checking
    moves only" restriction has; and ``is_checkmate`` at the end fails if the
    line stops short of mate.
    """
    result = solve_tsume(fixture.fen, fixture.tsume_plies)
    assert result.verdict == "MATE"

    game = Gungi(fixture.fen)
    attacker = str(game.turn())
    defender = "b" if attacker == "w" else "w"
    for ply, san in enumerate(result.line):
        assert str(game.turn()) == (attacker if ply % 2 == 0 else defender)
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        if ply % 2 == 0:
            assert bool(game.in_check(defender)), (
                f"{fixture.name}: attacker ply {ply + 1} ({san}) gives no check"
            )
    assert bool(game.is_checkmate()), f"{fixture.name}: line does not end in mate"


def test_f09_corrected_is_the_composers_three_mover() -> None:
    """F09 is pinned as a CORRECTED diagram, and to the composer's own line.

    The printed F09 is unsound and the composer has confirmed it: their
    intended idea -- general to 2-6-2, the marshal forced up onto it at
    2-6-3, samurai dropped on 4-4-3 -- fails on the printed board only
    because the black *fortress* the marshal was standing on stays behind at
    2-5-1 and interposes at 3-5, on the dropped samurai's diagonal. Swapping
    that fortress for a spy, which cannot reach 3-5, leaves every one of the
    composer's three moves intact and makes their line the solver's unique
    solution. So the row below is deliberately NOT the printed position, and
    this test exists to stop it from silently drifting back into one: it pins
    the corrected board to the *composer's* moves rather than to whatever the
    search happens to return.

    The generic friend-set tests already cover MATE-at-grade and replay for
    every row. What is specific here is the line itself, the uniqueness of
    the key move, and the legal-tsume-start condition -- the defender must
    not already be in check at the root, which is what made the earlier
    rank-3 repair candidates artefacts (KOM-261).
    """
    matches = [f for f in FRIEND if f.name == "F09-3move-corrected"]
    assert len(matches) == 1, "the corrected F09 row is missing from the friend set"
    fixture = matches[0]
    assert fixture.tsume_plies == 3
    assert fixture.source_grade == "3te-corrected"

    board = cyboard.board_from_fen(fixture.fen)
    assert not cyengine.in_check(board, 1 - int(board.turn)), (
        "a tsume root may not start with the defender already in check"
    )
    assert checking_moves(fixture.fen) == (F09_COMPOSER_LINE[0],), (
        "the composer's key move must be the only checking move in the position"
    )

    result = solve_tsume(fixture.fen, fixture.tsume_plies, verify_checkmate=True)
    assert result.verdict == "MATE", result.verdict
    assert len(result.line) == 3, result.line
    assert tuple(result.line) == F09_COMPOSER_LINE, result.line

    shallower = solve_tsume(fixture.fen, 1, verify_checkmate=True)
    assert shallower.verdict == "NO_MATE", shallower.verdict
    assert shallower.line == ()

    game = Gungi(fixture.fen)
    attacker = str(game.turn())
    defender = "b" if attacker == "w" else "w"
    for ply, san in enumerate(F09_COMPOSER_LINE):
        assert str(game.turn()) == (attacker if ply % 2 == 0 else defender)
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        if ply % 2 == 0:
            assert bool(game.in_check(defender)), (
                f"attacker ply {ply + 1} ({san}) gives no check"
            )
    assert bool(game.is_checkmate()), "the composer's line does not end in mate"
