#!/usr/bin/env python3
"""Regenerate ``kom239_repetition_tail.txt`` -- the KOM-239 repetition fixture.

The fixture is a **SAN history**: the full move list from
:data:`komugi.engine.fen.ADVANCED_POSITION` to a position where the side to
move can complete a fourfold repetition in one ply. That is the shape
the Python match server takes on the wire (``{"fen", "history"}``)
and the shape the reproducing test replays, so the fixture exercises the
protocol and the search with one artefact.

Why generated rather than recorded
----------------------------------
KOM-239 asks for the SAN tail of an external-anchor batch's game g10 (komugi
white, three site-adjudicated repetition draws at ``q`` up to +0.97). The
real recording was not available to the code that produced this fixture, so
the fixture is an engine-generated line with **the same structure**: a real
draft, a real post-draft position, and a two-move shuffle repeated until the
next move would be the fourth occurrence of a position. It reproduces the
defect (a search that cannot see the repetition walks into it); it is not a
recording of g10, and the file says so. Dropping the g10 tail in later is a
file swap: the format is ``# seat: <w|b>`` plus one SAN per line.

Running it::

    nix develop -c bash -c 'export PYTHONPATH=$PWD/src; \
        python tests/search/match/fixtures/make_kom239_fixture.py'

It is deterministic (:data:`SEED`), so a rerun that changes the file is a
change in the engine's move generation or SAN dialect, not noise.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Final

import numpy as np

from komugi.engine.action_encoding import encode_move
from komugi.engine.api import Gungi
from komugi.engine.fen import ADVANCED_POSITION

#: The seeded draft's RNG seed. Any seed that finishes both drafts and leaves a
#: shuffle on the board would do; this is the one the committed file used.
SEED: Final[int] = 11

#: How many pieces a side drops before it is allowed to finish its draft. Big
#: enough that the post-draft position has pieces that can walk back and forth
#: (a board with two marshals and nothing else has no shuffle in it), small
#: enough that the fixture stays short.
MIN_PLACEMENTS: Final[int] = 8

#: The SAN of the draft-finishing move, as the reference api spells it.
DRAFT_FINISH: Final[str] = "終"

#: How many times the shuffle cycle is repeated in full. Two, because the
#: position is already on the board once when the shuffling starts: 1 + 2 = 3
#: occurrences, so the fourth -- the drawing one -- is one ply away.
FULL_CYCLES: Final[int] = 2

#: Where the fixture is written.
OUT: Final[Path] = Path(__file__).resolve().parent / "kom239_repetition_tail.txt"


def position_of(fen: str) -> str:
    """Return `fen` without its move number -- what a repetition compares.

    The engine's repetition key is board + hands + turn + drafting rights; the
    move number is a counter, and two positions four plies apart are the *same
    position* with different counters. Comparing whole FENs would find no
    shuffle anywhere.
    """
    return " ".join(fen.split()[:-1])


def _verbose(gungi: Any) -> list[dict[str, Any]]:
    """Return the legal moves at `gungi`'s position, as verbose dicts."""
    return [dict(move) for move in gungi.moves({"verbose": True})]


def _walkers(gungi: Any) -> list[dict[str, Any]]:
    """Return the legal moves that walk a piece onto an empty square.

    A shuffle has to be undoable, so a candidate must have an origin (not a
    drop), must capture nothing, and must not stack onto another piece -- a
    tsuke changes the tower it lands on and cannot be taken back.
    """
    walkers = []
    for move in _verbose(gungi):
        if not move.get("from") or move.get("captured"):
            continue
        if str(move.get("type", "")) != "route":
            continue
        walkers.append(move)
    return walkers


def _move_between(gungi: Any, origin: str, target: str) -> dict[str, Any] | None:
    """Return the legal move from `origin` to `target`, or ``None``.

    Resolved by squares rather than by SAN on purpose: the SAN of "the same
    move" can be spelled differently in a different position (disambiguation),
    so every repetition of the cycle re-reads the SAN the engine gives it here.
    """
    for move in _verbose(gungi):
        if move.get("from") == origin and move.get("to") == target:
            return move
    return None


def draft(gungi: Any, rng: random.Random) -> list[str]:
    """Play a seeded legal draft to its end; return the SANs played."""
    played: list[str] = []
    while gungi.in_draft():
        legal = [str(san) for san in gungi.moves()]
        drops = [san for san in legal if not san.endswith(DRAFT_FINISH)]
        finish = [san for san in legal if san.endswith(DRAFT_FINISH)]
        if finish and (len(played) >= MIN_PLACEMENTS or not drops):
            chosen = finish[0]
        else:
            chosen = rng.choice(drops or legal)
        gungi.move(chosen)
        played.append(chosen)
    return played


def find_cycle(gungi: Any) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Return ``((out, back), (out, back))`` squares for a four-ply shuffle.

    The mover walks a piece out and the opponent answers; then both walk back,
    and the position on the board is the position that stood there four plies
    ago. Searched by replaying candidate lines through the engine rather than
    by reasoning about the rules: what matters is that the FEN comes back, and
    the engine is the only authority on that.
    """
    start = gungi.fen()
    for mine in _walkers(gungi):
        # `Any` on every `Gungi`: the v1 engine copy is only partially
        # annotated, and `move` is one of the untyped methods (the same bind
        # every anchors test makes).
        first: Any = Gungi(start)
        first.move(str(mine["san"]))
        for theirs in _walkers(first):
            second: Any = Gungi(first.fen())
            second.move(str(theirs["san"]))
            my_back = _move_between(second, str(mine["to"]), str(mine["from"]))
            if my_back is None:
                continue
            third: Any = Gungi(second.fen())
            third.move(str(my_back["san"]))
            their_back = _move_between(
                third, str(theirs["to"]), str(theirs["from"])
            )
            if their_back is None:
                continue
            fourth: Any = Gungi(third.fen())
            fourth.move(str(their_back["san"]))
            if position_of(fourth.fen()) == position_of(start):
                return (
                    (str(mine["from"]), str(mine["to"])),
                    (str(theirs["from"]), str(theirs["to"])),
                )
    return None


def shuffle(
    gungi: Any,
    mine: tuple[str, str],
    theirs: tuple[str, str],
    *,
    plies: int,
) -> list[str]:
    """Play `plies` plies of the shuffle `mine`/`theirs`; return the SANs."""
    legs = [
        (mine[0], mine[1]),
        (theirs[0], theirs[1]),
        (mine[1], mine[0]),
        (theirs[1], theirs[0]),
    ]
    played: list[str] = []
    for index in range(plies):
        origin, target = legs[index % len(legs)]
        move = _move_between(gungi, origin, target)
        if move is None:
            raise AssertionError(
                f"the shuffle broke at ply {index}: no move {origin}->{target}"
            )
        san = str(move["san"])
        gungi.move(san)
        played.append(san)
    return played


def build() -> tuple[list[str], str]:
    """Return ``(sans, seat)`` -- the fixture history and the seat to move.

    The history stops one ply short of the fourth occurrence, so the position
    it reaches is the one the search is asked about and the drawing move is one
    of its legal moves.
    """
    rng = random.Random(SEED)
    gungi = Gungi(ADVANCED_POSITION)
    sans = draft(gungi, rng)
    cycle = find_cycle(gungi)
    if cycle is None:
        raise AssertionError(
            f"no four-ply shuffle at the post-draft position of seed {SEED}"
        )
    mine, theirs = cycle
    # Two full cycles put the shuffle's base position on the board three times;
    # three more plies leave the fourth occurrence one move away.
    sans += shuffle(gungi, mine, theirs, plies=4 * FULL_CYCLES + 3)
    _assert_one_ply_from_a_draw(sans)
    return sans, str(gungi.turn())


def _assert_one_ply_from_a_draw(sans: list[str]) -> None:
    """Fail unless the replayed history has a drawing move and a live one.

    The generator's own acceptance check, asked of the engine that adjudicates
    the game rather than of the shuffle's arithmetic: replay the history into a
    ``Game`` and walk its legal moves, and there must be at least one child the
    engine calls a repetition draw and at least one it calls undecided. A
    fixture missing either half would make the reproducing test vacuous.
    """
    from komugi.engine.cyboard import board_from_fen
    from komugi.engine.game import TERMINATION_FOURFOLD_REPETITION, Game

    game = Game(board_from_fen(ADVANCED_POSITION))
    replay: Any = Gungi(ADVANCED_POSITION)
    for san in sans:
        move = dict(replay.move(san))
        game.play_action(int(encode_move(move)))
    drawing = alive = 0
    for action in np.flatnonzero(np.asarray(game.valid_moves(), dtype=bool)):
        child = game.clone()
        child.play_action(int(action))
        if child.termination_reason() == TERMINATION_FOURFOLD_REPETITION:
            drawing += 1
        elif not bool(np.asarray(child.win_state()).any()):
            alive += 1
    if drawing < 1 or alive < 1:
        raise AssertionError(
            f"the history is not one ply from a repetition draw: "
            f"{drawing} drawing children, {alive} live ones"
        )


def main() -> int:
    """Write the fixture and print what it reached."""
    sans, seat = build()
    lines = [
        "# KOM-239 repetition fixture -- generated, NOT the g10 recording.",
        "# make_kom239_fixture.py says why; format: '# seat: <w|b>' then SANs,",
        "# one per line, replayed from ADVANCED_POSITION.",
        f"# seat: {seat}",
        *sans,
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(sans)} SANs, {seat} to move")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
