"""The independent tsume oracle: naive AND/OR DFS over ``api.Gungi``.

Who checks the checker? At mate-in-1 an exhaustive enumeration does, but a
5-ply verdict -- and above all a *negative* 5-ply verdict -- has no such
oracle, and a solver asked to confirm its own answer will confirm anything.
So KOM-258 answers the question with **two implementations of one spec**:
:mod:`komugi.search.tsume` (df-pn, transposition table, compiled engine) and
this file (plain recursion, no memoisation at all, pure-Python rules). Every
fixture is solved by both and the verdicts must match. For the two to agree
by accident, a bug in the proof-number algebra would have to be mirrored by
an unrelated bug in a recursive `all()`, on a different rules implementation.

Deliberate properties, each of which is the point:

*No memoisation.* Not an oversight -- a transposition table is exactly the
machinery whose interaction with the path-repetition rule this oracle exists
to check, so it has none. The whole fixture set costs a few hundred nodes.

*Backend is* ``komugi.engine.api.Gungi`` *only.* The pure-Python rules
engine, never the compiled ``cy`` modules the solver under test is bound to.
A seam bug in that binding would have to be reproduced here, in different
code, to escape notice. Positions advance the way the api advances them: by
taking the ``"after"`` FEN out of a verbose move record.

*Same semantics, stated in the same order* as
:mod:`komugi.search.tsume` -- game-over guard, path repetition, depth cap,
then the move sets (attacker: checking moves only; defender: the escape set,
with a capture of the attacker's marshal ending the game in the defender's
favour).

One formulation differs on purpose. The solver finds marshal-capturing
replies by generating them; here the same fact is read off the api's attack
map (``in_check`` for the *attacker*, whose turn it is not), because
``Gungi.moves()`` returns ``[]`` at a position it considers game-over and so
cannot be asked for the reply list at a mating node. Two formulations of one
rule, cross-checked at every AND node of every fixture by
``test_semantics.py::test_marshal_capture_reply_matches_attack_map``.
"""

from __future__ import annotations

from typing import Any, Literal

from komugi.engine.api import Gungi

__all__ = ["ReferenceVerdict", "reference_solve"]

#: The oracle never answers UNKNOWN: it has no budget, it just runs.
ReferenceVerdict = Literal["MATE", "NO_MATE"]


def reference_solve(fen: str, max_plies: int) -> ReferenceVerdict:
    """``MATE`` iff the side to move forces mate within *max_plies* plies."""
    if max_plies < 1 or max_plies % 2 == 0:
        raise ValueError(f"max_plies must be odd and >= 1, got {max_plies}")
    game = Gungi(fen)
    attacker = str(game.turn())
    won = _attacker_wins(game.fen(), 0, max_plies, attacker, ())
    return "MATE" if won else "NO_MATE"


def _attacker_wins(
    fen: str,
    depth: int,
    max_plies: int,
    attacker: str,
    path: tuple[str, ...],
) -> bool:
    """Does the attacker force mate from *fen*, reached at ply *depth*?

    Even *depth* is an OR node (attacker to move), odd is an AND node
    (defender to move, in check by construction).
    """
    game = Gungi(fen)
    normalized = str(game.fen())
    key = _position_key(normalized)
    defender = "b" if attacker == "w" else "w"
    or_node = depth % 2 == 0

    if _marshal_missing(normalized):
        # The game ended with a marshal capture, so no mate was delivered.
        return False
    if key in path:
        # Path repetition: conservatively, the attack fails here.
        return False
    if or_node and depth >= max_plies:
        return False
    if not or_node and bool(game.in_check(attacker)):
        # The defender can answer the check by taking the attacker's marshal,
        # which ends the game in the defender's favour.
        return False

    records: list[Any] = list(game.moves({"verbose": True}))
    sub_path = (*path, key)

    if or_node:
        for record in records:
            after = str(record["after"])
            if not bool(Gungi(after).in_check(defender)):
                continue  # not a checking move
            if _attacker_wins(after, depth + 1, max_plies, attacker, sub_path):
                return True
        return False

    escapes = [
        str(record["after"])
        for record in records
        if not bool(Gungi(str(record["after"])).in_check(defender))
    ]
    if not escapes:
        # No reply leaves the defender out of check: checkmate. (An api that
        # calls the position game-over returns no records at all, which lands
        # here for the same reason.)
        return True
    return all(
        _attacker_wins(after, depth + 1, max_plies, attacker, sub_path)
        for after in escapes
    )


def _position_key(fen: str) -> str:
    """Everything in a FEN that changes the legal moves -- i.e. not the clock.

    The api's counterpart to ``cyengine.repetition_key``: placement, hands,
    side to move, mode and draft rights, with the move number dropped so that
    a genuine repetition is not hidden behind an advancing counter.
    """
    return " ".join(fen.split(" ")[:5])


def _marshal_missing(fen: str) -> bool:
    """True when either marshal is off the board.

    Read from the placement field only: a marshal in hand is not on the
    board, and the hand field spells its pieces with the same letters.
    """
    placement = fen.split(" ")[0]
    return placement.count("M") != 1 or placement.count("m") != 1
