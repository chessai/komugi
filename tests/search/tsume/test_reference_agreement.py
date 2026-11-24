"""Two implementations, one spec: df-pn against the naive api-backed oracle.

This is the answer to "who checks the checker at depth > 1". The solver is
df-pn with a transposition table over the compiled engine; the reference is a
plain recursive AND/OR search with no memoisation at all over the pure-Python
rules engine. They share the semantics and nothing else -- not the search,
not the data structures, not the rules implementation. Every fixture is put
to both, with the verdict each one must return written down in advance, so
that "they agree" cannot be satisfied by two UNKNOWNs or by two solvers that
both answer NO_MATE to everything.

The negative side is the part that needs the oracle most: a NO_MATE at depth
5 asserts something about a whole tree, and nothing enumerable checks it.
Hence the no-mate set -- the two published start positions and three quiet
midgame boards from a fixed replay -- and the two perturbation fixtures,
where a mate provably existed until the mating piece was removed.
"""

from __future__ import annotations

import pytest
from reference_tsume import reference_solve
from tsume_fixtures import (
    FRIEND,
    MINED,
    NO_MATE_PLIES,
    NO_MATE_POSITIONS,
    PERTURBATIONS,
    FriendFixture,
    MinedPuzzle,
    Perturbation,
    checking_moves,
    expected_mined_verdict,
)

from komugi.search.tsume import solve_tsume

MULTI_MOVE = tuple(f for f in FRIEND if f.tsume_plies >= 3)
NO_MATE_PARAMS = [pytest.param(name, fen, id=name) for name, fen in NO_MATE_POSITIONS]


@pytest.mark.parametrize("fixture", FRIEND, ids=lambda f: f.name)
def test_reference_agrees_on_friend_puzzles(fixture: FriendFixture) -> None:
    solver = solve_tsume(fixture.fen, fixture.tsume_plies)
    reference = reference_solve(fixture.fen, fixture.tsume_plies)
    assert solver.verdict == "MATE", f"{fixture.name}: {solver.verdict}"
    assert reference == "MATE", fixture.name
    assert solver.verdict == reference


@pytest.mark.parametrize("fixture", MULTI_MOVE, ids=lambda f: f.name)
def test_reference_agrees_on_minimality(fixture: FriendFixture) -> None:
    """Both implementations must fail to find a mate two plies shallower."""
    shallow = fixture.tsume_plies - 2
    solver = solve_tsume(fixture.fen, shallow)
    reference = reference_solve(fixture.fen, shallow)
    assert solver.verdict == "NO_MATE", f"{fixture.name}: {solver.verdict}"
    assert reference == "NO_MATE", fixture.name


@pytest.mark.parametrize("fixture", PERTURBATIONS, ids=lambda p: p.name)
def test_reference_agrees_on_perturbations(fixture: Perturbation) -> None:
    solver = solve_tsume(fixture.fen, fixture.tsume_plies)
    reference = reference_solve(fixture.fen, fixture.tsume_plies)
    assert solver.verdict == "NO_MATE", f"{fixture.name}: {solver.verdict}"
    assert reference == "NO_MATE", fixture.name


@pytest.mark.parametrize("fixture", MINED, ids=lambda m: m.name)
def test_reference_agrees_on_mined_puzzles(fixture: MinedPuzzle) -> None:
    """Including the three rows where both implementations decline the mate.

    Both reach NO_MATE there for the same reason and by different routes: the
    solver generates the reply that captures the attacker's marshal, the
    reference reads it off the api's attack map.
    """
    expected = expected_mined_verdict(fixture.name)
    solver = solve_tsume(fixture.fen, 1)
    reference = reference_solve(fixture.fen, 1)
    assert solver.verdict == expected, f"{fixture.name}: {solver.verdict}"
    assert reference == expected, fixture.name


@pytest.mark.parametrize("name,fen", NO_MATE_PARAMS)
def test_reference_agrees_that_quiet_positions_have_no_mate(
    name: str, fen: str
) -> None:
    """Depth 5, both implementations, expected NO_MATE -- UNKNOWN is a failure."""
    solver = solve_tsume(fen, NO_MATE_PLIES)
    reference = reference_solve(fen, NO_MATE_PLIES)
    assert solver.verdict == "NO_MATE", f"{name}: {solver.verdict}"
    assert reference == "NO_MATE", name


@pytest.mark.parametrize("name,fen", NO_MATE_PARAMS)
def test_no_mate_positions_have_both_marshals(name: str, fen: str) -> None:
    """A missing marshal would make ``is_checkmate`` False everywhere below."""
    placement = fen.split(" ")[0]
    assert "M" in placement, fen
    assert "m" in placement, fen


def test_the_no_mate_set_is_not_all_dead_ends() -> None:
    """At least one no-mate position has a check available at the root.

    A NO_MATE proved by "the attacker has no checking move" is proved by the
    move generator, not by the search. At least one member of this set has to
    make the search actually look at a tree before it says no.
    """
    with_checks = [
        (name, len(checking_moves(fen)))
        for name, fen in NO_MATE_POSITIONS
        if checking_moves(fen)
    ]
    assert with_checks, "every no-mate position is a dead end at the root"


def test_the_no_mate_set_has_the_required_shape() -> None:
    """Two published start positions plus at least three constructed ones."""
    names = [name for name, _ in NO_MATE_POSITIONS]
    assert "INTRO_POSITION" in names
    assert "BEGINNER_POSITION" in names
    assert len([n for n in names if n.startswith("quiet_midgame")]) >= 3
