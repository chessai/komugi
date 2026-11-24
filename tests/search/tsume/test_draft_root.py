"""KOM-282: ``solve_tsume`` refuses a root that is still in the draft.

During the draft the side to move does not strictly alternate, but the solver
takes OR/AND from depth parity, and ``cyengine.is_checkmate`` is
constant-False there -- so a MATE found from a draft root would contradict the
engine by definition. The draft FENs are *derived* from a committed fixture
(both marshals on the board, one draft flag set), not transcribed, and the
same fixture without the flag is checked to solve, so the guard is specific
to the draft rather than to the position.
"""

from __future__ import annotations

import pytest
from tsume_fixtures import MARSHAL_CAPTURE_ROWS, MINED

from komugi.engine import cyboard
from komugi.search.tsume import solve_tsume

#: A mined mate-in-1 the solver calls MATE at depth 1.
FIXTURE = next(m for m in MINED if m.name not in MARSHAL_CAPTURE_ROWS)


def _with_draft_flag(fen: str, flag: str) -> str:
    """*fen* with ``draft_w`` or ``draft_b`` set, both marshals still placed."""
    board = cyboard.board_from_fen(fen)
    setattr(board, flag, 1)
    return cyboard.board_to_fen(board)


@pytest.mark.parametrize("flag", ["draft_w", "draft_b"])
def test_draft_root_is_refused_in_both_modes(flag: str) -> None:
    fen = _with_draft_flag(FIXTURE.fen, flag)
    board = cyboard.board_from_fen(fen)
    assert int(getattr(board, flag))

    with pytest.raises(ValueError, match="draft"):
        solve_tsume(fen, 1)
    with pytest.raises(ValueError, match="draft"):
        solve_tsume(fen, 3, attacker_moves="all")


def test_the_same_fixture_without_the_draft_flag_solves() -> None:
    board = cyboard.board_from_fen(FIXTURE.fen)
    assert not int(board.draft_w) and not int(board.draft_b)

    assert solve_tsume(FIXTURE.fen, 1).verdict == "MATE"
    assert solve_tsume(FIXTURE.fen, 3, attacker_moves="all").verdict == "MATE"
