"""The final-control map: who owns the top of every tower (KOM-010).

Three properties, in the order they can break:

* the empty board is nobody's -- ``CONTROL_EMPTY`` is a real third value, not
  a default that quietly reads as "white";
* only the **top** of a tower decides, so a betrayal or a capture that changes
  the top flips the square while the pieces underneath do not move;
* over every position in the golden playout corpus, the map agrees with a slow
  reference read straight out of the canonical FEN text -- a second
  implementation, sharing no code with the encoder, that pins the orientation
  convention as well as the values.

The orientation under test is the one pre-registered by
``docs/design/kom-002-input-planes-v2.md`` §2: ``s = r * 9 + (8 - f)``, i.e.
the flat index of a row-major observation plane. The FEN reference gets it for
free -- a rank string's ``j``-th cell is file ``9 - j``, so ``f = 8 - j`` and
the string position *is* the square index within the rank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.cyboard import (
    Board,
    board_from_fen,
    board_to_fen,
    make_code,
)
from komugi.engine.observation import (
    CONTROL_BLACK,
    CONTROL_EMPTY,
    CONTROL_SIZE,
    CONTROL_WHITE,
    final_control,
)

_GOLDENS_DIR: Final[Path] = Path(__file__).parent / "goldens"

_PLAYOUT_GOLDENS: Final[tuple[str, ...]] = (
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
    "playouts_betray.json",
)
"""Every playout corpus, ``playouts_betray`` included: betrayal is the move
that changes a tower's top without changing its height, which is precisely the
case a "read the bottom piece" bug would survive."""

_WHITE_SOLDIER: Final[int] = make_code(9, 0)
_BLACK_SOLDIER: Final[int] = make_code(9, 1)
_BLACK_SPY: Final[int] = make_code(7, 1)

_TOWER_RANK: Final[int] = 2
_TOWER_FILE: Final[int] = 3
_TOWER_SQUARE: Final[int] = _TOWER_RANK * 9 + (8 - _TOWER_FILE)
"""``r * 9 + (8 - f)`` spelled once, by hand, so the test states the
convention rather than borrowing the encoder's arithmetic."""


def _square_of(rank: int, file: int) -> int:
    """Return the flat square index of ``cells_arr`` coordinates `rank`/`file`."""
    return rank * 9 + (8 - file)


def _control_from_fen_text(fen: str) -> npt.NDArray[np.int8]:
    """Return the control map read out of a canonical FEN string.

    The slow reference: it parses the placement field character by character
    and never touches :class:`~komugi.engine.cyboard.Board`, so agreeing with
    :func:`~komugi.engine.observation.final_control` is evidence about the
    encoder rather than about a shared helper.

    The placement grammar it needs is small -- a digit is that many empty
    squares, a bare letter is a lone piece (upper case white, lower case
    black), and ``|a:B:c|`` is a tower written bottom to top, so its **last**
    element is the one that owns the square.
    """
    placement = fen.split(" ")[0]
    ranks = placement.split("/")
    assert len(ranks) == 9, f"a placement field has nine ranks, got {len(ranks)}"

    control: list[int] = []
    for rank in ranks:
        row: list[int] = []
        index = 0
        while index < len(rank):
            char = rank[index]
            if char.isdigit():
                row.extend([CONTROL_EMPTY] * int(char))
                index += 1
            elif char == "|":
                close = rank.index("|", index + 1)
                top = rank[index + 1 : close].split(":")[-1]
                row.append(CONTROL_WHITE if top.isupper() else CONTROL_BLACK)
                index = close + 1
            else:
                row.append(CONTROL_WHITE if char.isupper() else CONTROL_BLACK)
                index += 1
        assert len(row) == 9, f"rank {rank!r} describes {len(row)} squares, not 9"
        control.extend(row)
    return np.asarray(control, dtype=np.int8)


def _playout_fens(name: str) -> list[str]:
    """Return every ``fen_before`` in golden `name`, skipping if it is absent."""
    path = _GOLDENS_DIR / name
    if not path.is_file():
        pytest.skip(f"oracle golden {name} not available (goldens corpus not vendored)")
    with path.open(encoding="utf-8") as handle:
        data: Any = json.load(handle)
    return [str(ply["fen_before"]) for game in data["games"] for ply in game]


def test_empty_board_is_all_empty() -> None:
    """A board with no pieces on it is nobody's, on all 81 squares.

    The non-vacuous half is that ``CONTROL_EMPTY`` is 2 and not 0: an encoder
    that allocated its map with ``np.zeros`` and filled only the occupied
    squares would report the empty board as entirely white-controlled, and
    every downstream label would be wrong on exactly the squares nobody is
    fighting over.
    """
    control = final_control(Board())

    assert control.shape == (CONTROL_SIZE,)
    assert control.dtype == np.int8
    assert np.array_equal(control, np.full(CONTROL_SIZE, CONTROL_EMPTY, dtype=np.int8))
    assert CONTROL_EMPTY != CONTROL_WHITE


def test_top_of_tower_decides_control() -> None:
    """Only the top piece owns the square -- and it owns it the moment it changes.

    Three positions of the same tower at one square:

    1. white on tier 1 with black stacked on tier 2 -- black-controlled,
       despite white being the majority of the tower;
    2. the top betrayed (the tier-2 piece becomes white) -- white-controlled,
       with nothing else on the board touched;
    3. the top captured off (tier 2 emptied) -- white-controlled again, now by
       the piece that was underneath all along.

    Every other square stays empty throughout, which is what makes this a test
    of *that* square rather than of a global recount.
    """
    board = Board()
    board.cells_arr[_TOWER_RANK, _TOWER_FILE, 0] = _WHITE_SOLDIER
    board.cells_arr[_TOWER_RANK, _TOWER_FILE, 1] = _BLACK_SPY

    stacked = final_control(board)
    assert int(stacked[_TOWER_SQUARE]) == CONTROL_BLACK, (
        "a tower of [white, black] bottom-to-top is controlled by the black "
        "piece standing on top of it"
    )
    elsewhere = np.delete(stacked, _TOWER_SQUARE)
    assert np.array_equal(elsewhere, np.full(80, CONTROL_EMPTY, dtype=np.int8))

    # Betrayal: the top piece changes colour in place, height unchanged.
    board.cells_arr[_TOWER_RANK, _TOWER_FILE, 1] = _WHITE_SOLDIER
    betrayed = final_control(board)
    assert int(betrayed[_TOWER_SQUARE]) == CONTROL_WHITE
    assert board.height(_TOWER_RANK, _TOWER_FILE) == 2, (
        "a betrayal flips the top piece; it does not shorten the tower"
    )

    # Capture of the top: the tower shortens and the piece below inherits it.
    board.cells_arr[_TOWER_RANK, _TOWER_FILE, 1] = 0
    captured = final_control(board)
    assert int(captured[_TOWER_SQUARE]) == CONTROL_WHITE
    assert board.height(_TOWER_RANK, _TOWER_FILE) == 1

    # And the mirror case, so "white" is not simply what this square returns:
    board.cells_arr[_TOWER_RANK, _TOWER_FILE, 0] = _BLACK_SOLDIER
    assert int(final_control(board)[_TOWER_SQUARE]) == CONTROL_BLACK


def test_the_map_is_indexed_like_a_flattened_observation_plane() -> None:
    """``s = r * 9 + (8 - f)`` -- design KOM-002 §2's orientation, asserted.

    A map that transposed the board, or mirrored the files, would satisfy every
    value-only assertion in this module; this is the one that would not. Two
    squares, chosen off the diagonal and off the centre file so that neither a
    transpose nor a mirror maps them to themselves.
    """
    board = Board()
    board.cells_arr[0, 0, 0] = _WHITE_SOLDIER   # rank 0, file 0 -> square 8
    board.cells_arr[1, 6, 0] = _BLACK_SOLDIER   # rank 1, file 6 -> square 11

    control = final_control(board)
    assert _square_of(0, 0) == 8
    assert _square_of(1, 6) == 11
    assert int(control[8]) == CONTROL_WHITE
    assert int(control[11]) == CONTROL_BLACK
    assert int(np.count_nonzero(control != CONTROL_EMPTY)) == 2


@pytest.mark.parametrize("golden", _PLAYOUT_GOLDENS)
def test_control_map_matches_fen_roundtrip_on_goldens(golden: str) -> None:
    """Every golden position: the encoder's map is the FEN text's map.

    Thousands of real positions -- towers of every height, both colours on top,
    the betray corpus included -- checked against a reference that reads the
    canonical FEN and shares no line of code with the encoder. Non-vacuity is
    asserted rather than assumed: a corpus that somehow produced no towers at
    all would make the agreement uninteresting, so the run tallies how many
    squares of each kind it actually saw.
    """
    fens = _playout_fens(golden)
    assert fens, f"{golden} carries no positions"

    seen = {CONTROL_WHITE: 0, CONTROL_BLACK: 0, CONTROL_EMPTY: 0}
    for fen in fens:
        board = board_from_fen(fen)
        expected = _control_from_fen_text(board_to_fen(board))
        got = final_control(board)
        assert np.array_equal(got, expected), (
            f"control map mismatch in {golden}\n"
            f"  fen:      {fen!r}\n"
            f"  expected: {expected.tolist()}\n"
            f"  got:      {got.tolist()}"
        )
        for value in seen:
            seen[value] += int(np.count_nonzero(got == value))

    assert all(count > 0 for count in seen.values()), (
        f"{golden} exercised only {seen}; a corpus with no white top, no black "
        "top or no empty square cannot distinguish a correct map from a "
        "constant one"
    )
