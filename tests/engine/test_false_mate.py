"""KOM-1109: capturing the opposing marshal is always an escape.

Before this fix, ``cyengine._has_escaping_move`` accepted a defender's reply
only if it left the defender's *own* marshal out of check. A reply that
instead captures the *attacker's* marshal ends the game outright (the
capturer wins) and was never supposed to be filtered out, but the old check
did not special-case it: if such a reply happened to also leave the
defender nominally "in check" (the attacker's marshal being the piece doing
the checking), the position was wrongly scored checkmate.

The fixture row below is the worked example from the KOM-1109 ticket itself
(a ``self_exposure_checkmate_claim`` case): white plays a move that looks
like checkmate, but black can reply by capturing white's marshal. The ticket
also names a oracle fixture file
(``gungi-az-runs/clientside/ligungi/ligungi_oracle.json``) with further rows
of the same class; that file lives outside this worktree's sandbox and could
not be read while implementing this task (every read attempt was refused by
the harness's containment boundary), so only the one row given literally in
the ticket text is pinned here. A successor task with read access to that
file should add the remaining rows following the same pattern.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import pytest

from komugi.engine import action_encoding, cyboard, cyengine, cymovegen
from komugi.engine import api as gungi_api
from komugi.engine.cyboard import Board, board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import (
    TERMINATION_CHECKMATE,
    TERMINATION_MARSHAL_CAPTURED,
    TERMINATION_NONE,
    Game,
)


class FalseMateRow(NamedTuple):
    """A ``self_exposure_checkmate_claim`` fixture: a false-mate move, and the
    capturing reply that refutes it."""

    name: str
    fen: str
    mating_looking_move: str
    capturing_reply: str


#: The KOM-1108/KOM-1109 worked example: white plays
#: 小(2-4-2)(2-5-2)付, which the pre-fix engine scored checkmate, although
#: black can reply 兵(7-7-2)取(8-7-1) and capture white's marshal outright.
FALSE_MATE_ROWS: Final[tuple[FalseMateRow, ...]] = (
    FalseMateRow(
        name="KOM-1108-worked-example",
        fen=(
            "r1|s:n|m2r|s:g|1/4k|a:J|j2/|n:K:F|1ww1f3/1nt6/2c6/a8/1|D:D:G||S:d|6"
            "/1|N:N||S:M|6/9 I1R1F1C1A1T1/d2 w 3 - 33"
        ),
        mating_looking_move="小(2-4-2)(2-5-2)付",
        capturing_reply="兵(7-7-2)取(8-7-1)#",
    ),
)

#: A genuine checkmate (KOM-142's ``MATE_FEN``, re-pinned here): black to
#: move, mated, and no marshal-capture reply exists -- the control for (c).
GENUINE_CHECKMATE_FEN: Final[str] = "8m/7D1/6I1G/9/9/9/9/9/4M4 -/- b 1 - 1"


def _board_of(game: Game) -> Board:
    board: Board = game._board
    return board


def _find_move(board: Board, san: str) -> dict[str, Any]:
    normalized = san.rstrip("=#")
    for mv in cymovegen.generate_all_moves(board):
        if mv["san"] == san or mv["san"] == normalized:
            return mv
    raise AssertionError(f"move not found: {san!r}")


def _play(game: Game, san: str) -> None:
    """Play *san* on *game*, asserting it is legal first."""
    board = _board_of(game)
    move = _find_move(board, san)
    action = action_encoding.encode_cython_move(move)
    mask = game.valid_moves()
    assert mask[action] == 1, f"{san} is not a legal move: {move}"
    game.play_action(action)


def _marshal_capturing_replies(board: Board) -> tuple[str, ...]:
    """SANs of every pseudo-legal reply of the side to move that captures
    the opponent's marshal.

    Counts marshals of the *opponent's* colour before and after each
    pseudo-legal move, the same technique
    ``tests/search/tsume/tsume_fixtures.marshal_capturing_replies`` uses:
    ``cyengine._marshal_captured`` is a ``cdef`` helper with no Python
    binding, so this is reconstructed from the public board state instead.
    """
    mover = int(board.turn)
    target_code = cyboard.make_code(0, 1 - mover)
    before = int((board.cells_arr == target_code).sum())
    replies = []
    for move in cymovegen.generate_all_moves(board):
        child = board.copy()
        cyengine.apply_move(child, move)
        after = int((child.cells_arr == target_code).sum())
        if after < before:
            replies.append(str(move["san"]))
    return tuple(replies)


# ---------------------------------------------------------------------------
# (a) After the false-mate move: not checkmate, and the capture is legal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", FALSE_MATE_ROWS, ids=lambda r: r.name)
def test_false_mate_move_is_not_checkmate(row: FalseMateRow) -> None:
    """A move that looks like checkmate leaves a real marshal-capture escape.

    cyengine.is_checkmate must be False, Game.termination_reason() must not
    be "checkmate", and the capturing reply must be among the generated
    marshal-capturing replies -- all three independent ways of asking the
    same question (KOM-1109).
    """
    game = Game(board_from_fen(row.fen))
    _play(game, row.mating_looking_move)
    board = _board_of(game)

    assert cyengine.is_checkmate(board) is False
    assert game.termination_reason() != TERMINATION_CHECKMATE
    assert row.capturing_reply in _marshal_capturing_replies(board)


@pytest.mark.parametrize("row", FALSE_MATE_ROWS, ids=lambda r: r.name)
def test_false_mate_capture_is_in_valid_moves(row: FalseMateRow) -> None:
    """The defender's marshal-capturing reply is a legal move in the mask."""
    game = Game(board_from_fen(row.fen))
    _play(game, row.mating_looking_move)
    board = _board_of(game)

    move = _find_move(board, row.capturing_reply)
    action = action_encoding.encode_cython_move(move)
    mask = game.valid_moves()
    assert mask[action] == 1
    assert int(mask.sum()) > 0


# ---------------------------------------------------------------------------
# (b) Playing the capture ends the game, capturer wins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", FALSE_MATE_ROWS, ids=lambda r: r.name)
def test_playing_the_capture_ends_the_game_for_the_capturer(
    row: FalseMateRow,
) -> None:
    game = Game(board_from_fen(row.fen))
    _play(game, row.mating_looking_move)
    capturer = int(_board_of(game).turn)
    _play(game, row.capturing_reply)

    assert game.termination_reason() == TERMINATION_MARSHAL_CAPTURED
    win_state = game.win_state()
    assert int(win_state.sum()) == 1
    assert int(win_state[capturer]) == 1


# ---------------------------------------------------------------------------
# (c) A genuine checkmate, with no marshal-capture escape, stays checkmate.
# ---------------------------------------------------------------------------


def test_genuine_checkmate_with_no_marshal_capture_stays_checkmate() -> None:
    board = board_from_fen(GENUINE_CHECKMATE_FEN)
    assert _marshal_capturing_replies(board) == ()
    assert cyengine.is_checkmate(board) is True

    game = Game(board_from_fen(GENUINE_CHECKMATE_FEN))
    assert game.termination_reason() == TERMINATION_CHECKMATE
    assert int(game.valid_moves().sum()) == 0
    win_state = game.win_state()
    assert int(win_state.sum()) == 1
    # Black (1) is mated, so white (0) wins.
    assert int(win_state[0]) == 1


# ---------------------------------------------------------------------------
# (d) The ADVANCED start position is unaffected.
# ---------------------------------------------------------------------------


def test_advanced_start_position_is_unaffected() -> None:
    game = Game(board_from_fen(ADVANCED_POSITION))
    assert int(game.valid_moves().sum()) == 54
    assert game.win_state().tolist() == [0, 0, 0]
    assert game.termination_reason() == TERMINATION_NONE


# ---------------------------------------------------------------------------
# (e) api.Gungi still reports the documented divergence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", FALSE_MATE_ROWS, ids=lambda r: r.name)
def test_api_gungi_still_claims_checkmate_documented_divergence(
    row: FalseMateRow,
) -> None:
    """api.Gungi (the gungi.js-faithful site referee) is NOT fixed by
    KOM-1109 -- it stays faithful to gungi.js until upstream
    jwyce/gungi.js#53 lands -- so it still (wrongly) reports checkmate on
    this exact example. This divergence from cyengine is deliberate and
    documented (KOM-1108/KOM-1109); it is pinned here so a future change
    that accidentally "fixes" api.py is caught as a scope violation.
    """
    g = gungi_api.Gungi(row.fen)
    g.move(row.mating_looking_move)  # type: ignore[no-untyped-call]
    assert g.is_checkmate() is True
