"""The rules seam: what the solver means by "mate", "reply" and "game over".

The df-pn search is only as good as the four engine questions it asks, and
each of them has a way of being quietly wrong:

* ``is_checkmate`` is constant-False on a board without exactly two marshals,
  so a mate proven on such a board is not a mate at all;
* movegen is pseudo-legal, so "the defender's replies" is a filter the
  adapter applies, not something the engine hands over;
* that filter has to be the *same* filter ``is_checkmate`` uses, or the
  solver proves mates the engine does not recognise;
* and the one place the two deliberately differ -- a reply that captures the
  attacker's marshal -- has to differ in the conservative direction, on
  positions that are named here rather than discovered later.
"""

from __future__ import annotations

import pytest
from tsume_fixtures import (
    ALL_FIXTURE_BOARDS,
    MARSHAL_CAPTURE_ROWS,
    MINED,
    checking_moves,
    enumeration_keys,
    forced_mate_keys,
    marshal_capturing_replies,
)

from komugi.engine import cyboard, cyengine
from komugi.search.tsume import CyTsumeProblem, solve_tsume

BOARD_PARAMS = [pytest.param(name, fen, id=name) for name, fen in ALL_FIXTURE_BOARDS]


@pytest.mark.parametrize("max_plies", [0, -1, 2, 4, 100])
def test_solve_tsume_rejects_impossible_depths(max_plies: int) -> None:
    """Even and non-positive depths are a ValueError, not a rounded-down search."""
    with pytest.raises(ValueError):
        solve_tsume(
            "4m4/4g4/9/9/9/9/9/9/M3G4 -/- w 3 - 1",
            max_plies,
        )


@pytest.mark.parametrize("name,fen", BOARD_PARAMS)
def test_defender_escape_set_matches_is_checkmate(name: str, fen: str) -> None:
    """At every AND node one ply deep, "no escape" and ``is_checkmate`` agree.

    ``CyTsumeProblem(verify_checkmate=True)`` raises the moment they do not,
    so this walks every checking move of every fixture board and generates
    the replies underneath it.
    """
    root = cyboard.board_from_fen(fen)
    problem = CyTsumeProblem(root, verify_checkmate=True)
    and_nodes = 0
    for _, child in problem.children(root):
        problem.children(child)  # raises AssertionError on disagreement
        and_nodes += 1
    assert problem.attacker == int(root.turn)
    assert and_nodes == len(checking_moves(fen))


def test_the_and_node_checks_cover_a_real_number_of_nodes() -> None:
    """The two per-board checks above are not quietly running on nothing.

    Both walk "every AND node one ply deep", which on a board with no
    checking move is no nodes at all. The fixture set has a hundred-odd; if
    that collapses, the equivalences above stop being evidence.

    The floor was 300, measured over a fixture set that still contained the
    mined row ``iter_827-g200002``. That root had the *defender* already in
    check, so every move "gave check" trivially and the row alone
    contributed 232 of the 340 -- more than the other 25 boards put
    together, and not AND nodes in any meaningful sense, since what reached
    them was not a checking move. KOM-261 quarantined the row, and the floor
    is re-measured here at 100 against an actual total of 108 over the 25
    remaining boards. So the floor moved down because a number that was
    mostly an artefact went away, not because coverage was lost.
    """
    total = sum(len(checking_moves(fen)) for _, fen in ALL_FIXTURE_BOARDS)
    assert total >= 100, total


@pytest.mark.parametrize("name,fen", BOARD_PARAMS)
def test_marshal_capture_reply_matches_attack_map(name: str, fen: str) -> None:
    """Two formulations of "the defender can take the attacker's marshal".

    The solver generates the capture; the reference solver, which cannot ask
    ``api.Gungi`` for a reply list at a position the api calls game-over,
    reads the same fact off the attack map. If those two ever come apart, the
    two implementations are answering different questions and their agreement
    stops meaning anything -- so they are compared here at every AND node of
    every fixture board.
    """
    root = cyboard.board_from_fen(fen)
    problem = CyTsumeProblem(root)
    for _, child in problem.children(root):
        generated = bool(marshal_capturing_replies(str(cyboard.board_to_fen(child))))
        attacked = bool(cyengine.in_check(child, problem.attacker))
        assert generated == attacked, (
            f"{name}: generated={generated} attack_map={attacked} "
            f"for {cyboard.board_to_fen(child)!r}"
        )


def test_marshal_capture_exception_set_is_exactly_the_recorded_rows() -> None:
    """``enumeration_keys`` and ``forced_mate_keys`` now fully agree.

    Before KOM-1109, ``enumeration_keys`` -- the engine's own mate-in-1
    answer via ``cyengine.is_checkmate`` -- disagreed with ``forced_mate_keys``
    (mates the defender can refute by capturing the attacker's marshal
    removed) on exactly the rows named in ``MARSHAL_CAPTURE_ROWS``, because
    ``is_checkmate`` did not yet count a marshal-capturing reply as an escape.
    KOM-1109 fixed ``is_checkmate`` itself to count it, so the two functions
    no longer differ anywhere: this is a regression guard that the divergence
    does not reopen, rather than a description of a live one.
    """
    diverged = {
        name
        for name, fen in ALL_FIXTURE_BOARDS
        if sorted(enumeration_keys(fen)) != sorted(forced_mate_keys(fen))
    }
    assert diverged == set()


@pytest.mark.parametrize("name", sorted(MARSHAL_CAPTURE_ROWS))
def test_each_marshal_capture_row_has_a_real_capturing_reply(name: str) -> None:
    """Name the refutation: the mating move, and the reply that takes the marshal.

    Before KOM-1109, ``cyengine.is_checkmate`` wrongly called the child mate
    even though the defender could reply by capturing the attacker's marshal.
    Fixed, ``is_checkmate`` now agrees: every child with such a capturing
    reply is correctly *not* checkmate.
    """
    fixture = next(m for m in MINED if m.name == name)
    root = cyboard.board_from_fen(fixture.fen)
    refuted: list[tuple[str, tuple[str, ...]]] = []
    for san, child in CyTsumeProblem(root).children(root):
        replies = marshal_capturing_replies(str(cyboard.board_to_fen(child)))
        if not replies:
            continue
        refuted.append((san, replies))
        assert not cyengine.is_checkmate(child), (
            f"{name}: {san} is still (wrongly) checkmate despite the "
            f"marshal-capturing reply {replies} (KOM-1109 regression)"
        )
    assert refuted, f"{name}: no move is refuted by a marshal capture"
    assert not forced_mate_keys(fixture.fen)
