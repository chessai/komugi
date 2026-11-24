"""Stub for the compiled komugi.engine.cyboard extension (see cyboard.pyx).

Hand-written and kept honest by ci/check.sh's stubtest run: it declares the
python-visible surface only (the ``cdef``/``cdef inline`` helpers and the
C-level fields reachable only through cyboard.pxd are deliberately absent).

It lives next to the .pyx rather than under stubs/ because mypy's `files`
covers both src/komugi and stubs/, and a second `komugi` package tree there
would be a duplicate module name -- the same reason komugi/_probe.pyi sits
next to _probe.pyx.
"""

from typing import Any

# Kanji symbols in canonical order (14 piece types).
PIECE_TYPES: tuple[str, ...]
# kanji -> type_index (0..13)
PIECE_INDEX: dict[str, int]

def make_code(type_index: int, color: int) -> int: ...
def split_code(code: int) -> tuple[int, int]: ...

class Board:
    """Typed Gungi board -- the representation used by the fast engine."""

    # 9x9x3 piece-code grid and its typed memoryview.
    cells_arr: Any
    cells: Any
    # hand[type_index, color] counts, and its typed memoryview.
    hand_arr: Any
    hand: Any

    turn: int
    mode: int
    draft_w: int
    draft_b: int
    move_number: int

    # Hand piece order: list of (type_index, color) in FEN input order.
    _hand_order: list[Any]

    def __init__(self, *args: object, **kwargs: object) -> None: ...  # runtime is zero-arg; lax form is stubtest-prescribed for Cython binding
    def top(self, r: int, f: int) -> int: ...
    def height(self, r: int, f: int) -> int: ...
    def copy(self) -> Board: ...

def board_from_fen(fen: str) -> Board: ...
def board_to_fen(b: Board) -> str: ...
