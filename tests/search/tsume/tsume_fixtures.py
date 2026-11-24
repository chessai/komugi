"""Fixture data and shared oracles for the KOM-258 tsume suite.

Three groups of positions, all of them committed rather than generated, so
that a verdict that changes is a change in the solver and never in the
question:

``friend``
    Composed puzzles transcribed from a physical set (KOM-257), with the
    white marshal placed at 9-9 so the board is legal -- without it
    ``cyengine.is_checkmate`` is constant-False and the whole suite would
    pass vacuously. Their recorded ``tsume_plies`` is the *minimal* mate
    distance, which is what ``test_fixtures.py`` asserts from both sides:
    MATE at ``tsume_plies``, NO_MATE at ``tsume_plies - 2``.
    F09 appears only as ``F09-3move-corrected``: the printed diagram is
    unsound -- the design review measured no mate at depth 3, 5 or 7 there,
    checks-only or with quiet moves, and the composer has since confirmed the
    mistake -- so what is pinned is the repaired position, not the printed
    one. The group is no longer counted in its own name, which went stale
    every time a row was added.

``perturbations_2``
    Two of the above with the mating resource deleted (F02's hand emptied,
    F08's archer removed). Negative controls: a solver that answered MATE by
    accident would have to keep doing it with the mate gone.

``mined``
    In-distribution mate-in-1s mined from real self-play games, with the
    exhaustive enumeration keys recorded alongside. In-distribution matters:
    the composed set is hand-made and could share a blind spot.
    ``iter_827-g200002`` was quarantined by KOM-261: its mate-in-1 is
    genuine, but its root has the *defender* already in check (this engine
    permits moving into self-check, so self-play produces such positions),
    which makes it not a tsume problem at all and ``solve_tsume`` rightly
    refuses it. Its FEN survives as a literal regression case in
    ``test_review_findings.py``. Like the friend group, this group is no
    longer counted in its own name, which went stale when the row left.

The fixture JSON is copied verbatim from the KOM-258 ticket appendices and is
not regenerated here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from komugi.engine import cyboard, cyengine, cymovegen
from komugi.engine.fen import ADVANCED_POSITION, BEGINNER_POSITION, INTRO_POSITION
from komugi.search.tsume_core import Verdict

FIXTURE_DIR: Final = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class FriendFixture:
    """A composed puzzle and its grade, in plies."""

    name: str
    fen: str
    tsume_plies: int
    source_grade: str


@dataclass(frozen=True)
class Perturbation:
    """A composed puzzle with its mating resource removed."""

    name: str
    fen: str
    tsume_plies: int
    expect: Verdict


@dataclass(frozen=True)
class MinedPuzzle:
    """A mate-in-1 from self-play, with its exhaustive enumeration keys."""

    name: str
    fen: str
    mate_in: int
    keys: tuple[str, ...]


def _load(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open() as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


_FRIEND_JSON: Final = _load("friend.json")
_MINED_JSON: Final = _load("mined.json")

FRIEND: Final = tuple(
    FriendFixture(
        name=str(row["name"]),
        fen=str(row["fen"]),
        tsume_plies=int(row["tsume_plies"]),
        source_grade=str(row["source_grade"]),
    )
    for row in _FRIEND_JSON["friend"]
)

PERTURBATIONS: Final = tuple(
    Perturbation(
        name=str(row["name"]),
        fen=str(row["fen"]),
        tsume_plies=int(row["tsume_plies"]),
        expect="NO_MATE",
    )
    for row in _FRIEND_JSON["perturbations_2"]
)

MINED: Final = tuple(
    MinedPuzzle(
        name=str(row["name"]),
        fen=str(row["fen"]),
        mate_in=int(row["mate_in"]),
        keys=tuple(str(key) for key in row["keys"]),
    )
    for row in _MINED_JSON["puzzles"]
)

#: Every committed fixture board as ``(name, fen)``, for the tests that must
#: run over all of them rather than over one group.
ALL_FIXTURE_BOARDS: Final = tuple(
    [(f.name, f.fen) for f in FRIEND]
    + [(p.name, p.fen) for p in PERTURBATIONS]
    + [(m.name, m.fen) for m in MINED]
)

#: The three mined rows where the defender can answer the recorded mating
#: move by capturing the attacker's marshal, so the solver calls them
#: NO_MATE under the KOM-258 amendment-A defender semantics.
#:
#: Before KOM-1109, this was also where the *engine* read as mate when it was
#: not: ``is_checkmate`` filtered replies by "does this leave me in check", so
#: a reply that takes the attacker's marshal *and* leaves the defender in
#: check was not counted as an escape and the position wrongly read as mate,
#: even though komugi's move generation is pseudo-legal and that capturing
#: reply was playable, ending the game with the attacker's marshal gone.
#: KOM-1109 fixed ``cyengine._has_escaping_move`` to count a marshal-capturing
#: reply as an escape, so ``is_checkmate`` now agrees with the solver on all
#: three rows (see ``test_semantics.py``). The frozen mined ``keys`` for these
#: rows (``mined.json``) still record the pre-fix enumeration result and are
#: deliberately left unreproduced by ``test_mined_keys_reproduce_by_enumeration``.
#: ``test_semantics.py`` pins each of these three rows to the actual
#: capturing reply, so the exception cannot quietly grow.
MARSHAL_CAPTURE_ROWS: Final = frozenset(
    {
        "iter_827-g200000",
        "iter_827-g200001",
        "iter_827-g3000000100000",
    }
)


def expected_mined_verdict(name: str) -> Verdict:
    """The verdict a correct solver must return for mined row *name* at depth 1."""
    return "NO_MATE" if name in MARSHAL_CAPTURE_ROWS else "MATE"


# ---------------------------------------------------------------------------
# The depth-1 enumeration oracle (KOM-257's), and its refinement
# ---------------------------------------------------------------------------


def enumeration_keys(fen: str) -> tuple[str, ...]:
    """Every move whose result ``cyengine.is_checkmate`` calls mate.

    Exhaustive, trivially correct, and therefore the oracle for mate-in-1:
    apply each pseudo-legal move and ask the engine. It is not the oracle for
    anything deeper, which is what the reference solver is for.
    """
    board = cyboard.board_from_fen(fen)
    keys: list[str] = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        if cyengine.is_checkmate(child):
            keys.append(str(move["san"]))
    return tuple(keys)


def marshal_capturing_replies(fen: str) -> tuple[str, ...]:
    """Replies by the side to move that take the *other* side's marshal.

    Generated rather than read off an attack map, which is exactly how
    :class:`komugi.search.tsume.CyTsumeProblem` finds them.
    """
    board = cyboard.board_from_fen(fen)
    mover = int(board.turn)
    target = cyboard.make_code(0, 1 - mover)
    before = int((board.cells_arr == target).sum())
    replies: list[str] = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        if int((child.cells_arr == target).sum()) < before:
            replies.append(str(move["san"]))
    return tuple(replies)


def forced_mate_keys(fen: str) -> tuple[str, ...]:
    """:func:`enumeration_keys` minus the mates the defender can refute by capture.

    The mate-in-1 oracle restated under the solver's semantics: a move is a
    forced mate only if the defender, besides having no escape, also cannot
    end the game by taking the attacker's marshal.
    """
    board = cyboard.board_from_fen(fen)
    keys: list[str] = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        if not cyengine.is_checkmate(child):
            continue
        if marshal_capturing_replies(cyboard.board_to_fen(child)):
            continue
        keys.append(str(move["san"]))
    return tuple(keys)


def checking_moves(fen: str) -> tuple[str, ...]:
    """Every pseudo-legal move of the side to move that gives check."""
    board = cyboard.board_from_fen(fen)
    defender = 1 - int(board.turn)
    checks: list[str] = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        if cyengine.in_check(child, defender):
            checks.append(str(move["san"]))
    return tuple(checks)


# ---------------------------------------------------------------------------
# The no-mate set
# ---------------------------------------------------------------------------

#: How the quiet midgame positions below are reached. A fixed, deterministic
#: opening replayed from ADVANCED_POSITION -- the same sequence every run,
#: with no RNG and nothing recorded by hand: at ply *n* take move
#: ``(n * 31 + 7) % len(moves)`` of the engine's own generated list. The two
#: draft clauses keep the result a *position* rather than a curiosity: for
#: the first 16 plies a draft-finishing move is never taken (otherwise the
#: replay drafts two pieces and plays out a nearly empty board with 24 pieces
#: still in hand, where mates in 5 are everywhere), and after that it is
#: taken as soon as it is offered.
_DRAFT_PLACEMENT_PLIES: Final = 16
#: Plies at which a position is snapshotted, chosen to be after the draft and
#: to include boards where the side to move has a check available.
_SNAPSHOT_PLIES: Final = (20, 28, 40)


def quiet_midgame_fens() -> tuple[str, ...]:
    """Replay the fixed opening and return the snapshotted midgame FENs."""
    board = cyboard.board_from_fen(ADVANCED_POSITION)
    snapshots: list[str] = []
    for ply in range(1, max(_SNAPSHOT_PLIES) + 1):
        moves = cymovegen.generate_all_moves(board)
        drafting = bool(board.draft_w or board.draft_b)
        candidates = moves
        if drafting and ply <= _DRAFT_PLACEMENT_PLIES:
            candidates = [m for m in moves if not m["draft_finished"]] or moves
        elif drafting:
            candidates = [m for m in moves if m["draft_finished"]] or moves
        cyengine.apply_move(board, candidates[(ply * 31 + 7) % len(candidates)])
        if ply in _SNAPSHOT_PLIES:
            snapshots.append(str(cyboard.board_to_fen(board)))
    return tuple(snapshots)


def no_mate_positions() -> tuple[tuple[str, str], ...]:
    """``(name, fen)`` for the positions where neither solver may find a mate.

    The two published start positions -- where a mate in five would mean the
    rules are broken -- plus the quiet midgame boards from the replay above,
    at least one of which has a checking move at the root so that the
    NO_MATE verdicts cannot all come from an empty root move list.
    """
    positions = [
        ("INTRO_POSITION", INTRO_POSITION),
        ("BEGINNER_POSITION", BEGINNER_POSITION),
    ]
    positions += [
        (f"quiet_midgame_ply{ply}", fen)
        for ply, fen in zip(_SNAPSHOT_PLIES, quiet_midgame_fens(), strict=True)
    ]
    return tuple(positions)


NO_MATE_POSITIONS: Final = no_mate_positions()

#: Depth the no-mate set is searched to by both implementations.
NO_MATE_PLIES: Final = 5
