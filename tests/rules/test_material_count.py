"""KOM-245 -- ``Game.material()``: TOTAL material, board plus hand, per side.

The count KOM-245's soft adjudication of a turn-capped game is computed from,
pinned against the four things that can move it and the one thing that must
agree with it:

* the **advanced start** is ``(25, 25)`` -- the dealt hand, which is what
  ``MAX_COUNT`` sums to;
* a **capture** charges only the victim: the piece leaves the game, and no
  engine path ever returns it to a hand;
* a **drop** (新, arata) moves a piece hand -> board and leaves *both* totals
  untouched, which is why the rule is a total and not a board presence: under a
  board count, drafting would read as a material gain and the label would
  decree what has to be learned from outcomes;
* a **betrayal** debits ``k`` from the mover's own hand while converting ``k``
  of the victim's pieces, so the mover's total is unchanged and the victim's
  falls by ``k`` -- ``d`` moves by ``k``, not ``2k`` (design review F7, and the
  half of §1 that was wrong before it);
* the count and the **observation planes** are one number: the analysis
  scripts decode material out of the planes, and a census that disagreed with
  the label would be worse than no census.

The plane decode below is ``scripts/analysis/material_vs_outcome.decode_material``
re-spelled over a single position (colour is the occupancy plane's parity; a
hand plane is a constant ``count / MAX_COUNT``), deliberately re-derived here
rather than imported: a test that imported the script would pin the two against
each other through one implementation, which is the thing it is trying to
avoid.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

from komugi.engine.fen import ADVANCED_POSITION  # noqa: E402
from komugi.engine.game import Game  # noqa: E402
from komugi.engine.observation import (  # noqa: E402
    MAX_COUNT,
    PIECE_TYPES,
    material_counts,
)
from komugi.engine.utils import TACTICIAN  # noqa: E402

from fixtures import (  # noqa: E402
    SCENARIO_B_LOOP,
    board_of,
    drop_action,
    load_game,
    load_position_q,
)

#: Pieces per side in the advanced deal -- ``MAX_COUNT`` summed, not restated.
PIECES_PER_SIDE: int = sum(MAX_COUNT.values())

#: The betrayal fixture of ``test_betrayal_hand_invariant``: a two-piece tower
#: on 5-5 (black 謀 under a white 兵) and TWO 謀 in White's hand, which is what
#: makes the conversion affordable -- one 謀 pays for the placement and one for
#: the flip.  White 4 / black 2 before, white 4 / black 1 after: the mover's
#: total is unchanged and the victim is one down.
BETRAYAL_FEN: str = "4m4/9/9/9/4|t:D|4/9/9/9/4M4 T2/- w 3 - 1"

#: The tower the converting drop lands on, 1-based ``(rank, file)``.
TOWER: tuple[int, int] = (5, 5)

#: Occupancy planes (0-83), hand planes (84-111) -- the block boundaries the
#: decode below reads, derived from the piece list rather than written out.
_OCCUPANCY_PLANES: int = len(PIECE_TYPES) * 3 * 2
_HAND_STOP: int = _OCCUPANCY_PLANES + len(PIECE_TYPES) * 2


def _material_from_planes(observation: npt.NDArray[np.float32]) -> tuple[int, int]:
    """Decode ``(white, black)`` total material out of one position's planes.

    Board pieces are the occupancy block summed by plane parity (even planes
    are white); hand pieces are the constant hand planes multiplied back by
    their ``MAX_COUNT`` and rounded, which is exactly the inverse of the
    ``count / MAX_COUNT`` the encoder writes.
    """
    planes = np.asarray(observation, dtype=np.float64)
    board = planes[:_OCCUPANCY_PLANES]
    max_by_index = np.array([MAX_COUNT[piece] for piece in PIECE_TYPES])
    hand_fraction = planes[_OCCUPANCY_PLANES:_HAND_STOP, 0, 0]
    hand = np.rint(hand_fraction * np.repeat(max_by_index, 2))
    white = board[0::2].sum() + hand[0::2].sum()
    black = board[1::2].sum() + hand[1::2].sum()
    return int(round(float(white))), int(round(float(black)))


def _assert_planes_agree(game: Game) -> None:
    """The board read and the plane decode must be ONE number, at *game*."""
    assert _material_from_planes(game.observation()) == game.material(), (
        "the material the engine counts off the board and the material the "
        "analysis scripts decode out of the planes describe the same position; "
        "a label the census cannot reproduce is not a measurement"
    )


def test_the_advanced_start_is_twenty_five_a_side() -> None:
    """The dealt hand, counted: 25 pieces each, all of them still in hand."""
    game = load_game(ADVANCED_POSITION)

    assert game.material() == (PIECES_PER_SIDE, PIECES_PER_SIDE) == (25, 25)
    _assert_planes_agree(game)


def test_material_counts_is_the_same_answer_as_game_material() -> None:
    """``Game.material`` is a delegation and owns no arithmetic of its own."""
    game = load_position_q()

    assert game.material() == material_counts(board_of(game))


def test_a_drop_leaves_both_totals_unchanged() -> None:
    """新: a piece moves hand -> board, and the TOTAL is invariant to that.

    The property the whole rule rests on -- and the answer to "a side that
    drafts more has more material", which is true of board presence and false
    of a total.
    """
    game = load_position_q()
    before = game.material()
    assert before == (4, 3), "Q is 帥+侍+兵x2 against 帥+槍+馬"

    game.play_action(SCENARIO_B_LOOP[0])  # W 新兵(5-9), out of the hand

    assert game.material() == before
    _assert_planes_agree(game)


def test_a_capture_charges_only_the_captured_side() -> None:
    """取: the victim is one down, the capturer is unchanged.

    A captured piece leaves the game -- this engine never returns one to a hand
    -- so per-side material is monotone non-increasing and ``d`` moves by one.
    """
    game = load_position_q()
    game.play_action(SCENARIO_B_LOOP[0])  # W 新兵(5-9)
    before = game.material()

    game.play_action(SCENARIO_B_LOOP[1])  # B 馬(3-9)取(5-9)

    white, black = game.material()
    assert (white, black) == (before[0] - 1, before[1]), (
        "the captured 兵 is gone from White's total and Black paid nothing for "
        "taking it"
    )
    _assert_planes_agree(game)


def test_a_betrayal_charges_only_the_victim() -> None:
    """裏: the mover pays ``k`` from its hand and gains ``k`` on the board.

    Design review **F7**: §1 claimed ``d`` moved by ``2k``.  It does not -- an
    arata-betray debits the converted count from the mover's own hand, so the
    mover's total is exactly unchanged and only the victim's falls.
    """
    game = load_game(BETRAYAL_FEN)
    before = game.material()
    assert before == (4, 2), "帥+兵+謀x2 in hand against 帥+謀"

    game.play_action(drop_action(TACTICIAN, TOWER, bsel=1))

    white, black = game.material()
    assert (white, black) == (before[0], before[1] - 1), (
        "the converting drop spends two 謀 from the hand and puts two on the "
        "board (one placed, one flipped), so White's total is unchanged; Black "
        "is one down (KOM-245 review F7)"
    )
    _assert_planes_agree(game)
