# cython: binding=True
# cython: embedsignature=True
# cython: embedsignature.format=python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
cyboard — Typed board representation for Gungi (Cython).

Piece-code encoding
-------------------
0 = empty.
Nonzero = piece_type_index * 2 + color, where:
  piece_type_index  0..13  (canonical order, same as PIECE_TYPES below)
  color             0=white, 1=black
So valid codes are 1..28 (inclusive).  code=1 → marshal/white, code=2 →
marshal/black, code=3 → general/white, …, code=28 → tactician/black.

Canonical piece ordering (index 0..13) — matches
``komugi.engine.observation.PIECE_TYPES`` and
``komugi.engine.action_encoding.PIECE_INDEX``:
  0  帥  marshal
  1  大  general
  2  中  lieutenant_general
  3  小  major_general
  4  侍  warrior
  5  槍  lancer
  6  馬  rider
  7  忍  spy
  8  砦  fortress
  9  兵  soldier
  10 砲  cannon
  11 弓  archer
  12 筒  musketeer
  13 謀  tactician

Board cells: numpy int8 array of shape (9, 9, 3).
  cells[r, f, t]  where r=rank-1, f=file-1, t=tier-1.
  A tower of height h fills t=0..h-1 (bottom→top); remaining tiers are 0.

Hand: numpy int32 array of shape (14, 2).
  hand[type_index, color] = count.
"""

import numpy as np
cimport numpy as np

# ---------------------------------------------------------------------------
# Canonical piece ordering (must match observation.PIECE_TYPES)
# ---------------------------------------------------------------------------

# Kanji symbols in canonical order.
PIECE_TYPES = (
    "\u5e25",  # 帥 marshal
    "\u5927",  # 大 general
    "\u4e2d",  # 中 lieutenant_general
    "\u5c0f",  # 小 major_general
    "\u4f8d",  # 侍 warrior
    "\u69cd",  # 槍 lancer
    "\u99ac",  # 馬 rider
    "\u5fcd",  # 忍 spy
    "\u7826",  # 砦 fortress
    "\u5175",  # 兵 soldier
    "\u7832",  # 砲 cannon
    "\u5f13",  # 弓 archer
    "\u7b52",  # 筒 musketeer
    "\u8b00",  # 謀 tactician
)

# kanji -> type_index (0..13)
PIECE_INDEX = {p: i for i, p in enumerate(PIECE_TYPES)}

# FEN letter (lowercase) -> type_index
_FEN_LETTER_TO_IDX = {
    "m": 0, "g": 1, "i": 2, "j": 3, "w": 4, "n": 5, "r": 6,
    "s": 7, "f": 8, "d": 9, "c": 10, "a": 11, "k": 12, "t": 13,
}

# type_index -> FEN letter (lowercase)
_IDX_TO_FEN_LETTER = ["m", "g", "i", "j", "w", "n", "r", "s",
                       "f", "d", "c", "a", "k", "t"]

# ---------------------------------------------------------------------------
# Helpers: piece code <-> (type_index, color)
# ---------------------------------------------------------------------------

def make_code(int type_index, int color):
    """Build a piece code from *type_index* (0..13) and *color* (0=w,1=b)."""
    return type_index * 2 + color + 1

def split_code(int code):
    """Split a nonzero piece code into (type_index, color)."""
    return ((code - 1) >> 1, (code - 1) & 1)

# C-level helpers (for internal use).
cdef inline int _make_code(int ti, int col):
    return ti * 2 + col + 1

cdef inline int _type_index(int code):
    return (code - 1) >> 1

cdef inline int _color(int code):
    return (code - 1) & 1


# ---------------------------------------------------------------------------
# Board cdef class
# ---------------------------------------------------------------------------

cdef class Board:
    """Typed Gungi board — the representation used by the fast engine."""

    def __init__(self):
        self.cells_arr = np.zeros((9, 9, 3), dtype=np.int8)
        self.cells = self.cells_arr
        self.hand_arr = np.zeros((14, 2), dtype=np.int32)
        self.hand = self.hand_arr
        self.turn = 0
        self.mode = 0
        self.draft_w = 0
        self.draft_b = 0
        self.move_number = 1
        self._hand_order = []

    cpdef int top(self, int r, int f):
        """Return the piece code at the top of the tower at (r, f), or 0."""
        cdef int t
        for t in range(2, -1, -1):
            if self.cells[r, f, t] != 0:
                return self.cells[r, f, t]
        return 0

    cpdef int height(self, int r, int f):
        """Return the tower height at (r, f) — 0 if empty."""
        cdef int t
        for t in range(2, -1, -1):
            if self.cells[r, f, t] != 0:
                return t + 1
        return 0

    cpdef Board copy(self):
        """Return an independent deep copy."""
        cdef Board b = Board.__new__(Board)
        b.cells_arr = self.cells_arr.copy()
        b.cells = b.cells_arr
        b.hand_arr = self.hand_arr.copy()
        b.hand = b.hand_arr
        b.turn = self.turn
        b.mode = self.mode
        b.draft_w = self.draft_w
        b.draft_b = self.draft_b
        b.move_number = self.move_number
        b._hand_order = list(self._hand_order)
        return b

    def __reduce__(self):
        """Serialize via FEN for pickle support (e.g. multiprocessing queues).

        FEN round-trip is complete: board_from_fen rebuilds cells_arr (and the
        cells memoryview), hand_arr (and the hand memoryview), turn, mode,
        draft_w, draft_b, move_number, and _hand_order.  There is no Board
        state beyond what FEN encodes — the typed memoryviews (cells, hand) are
        derived views of cells_arr/hand_arr, not independent state.
        """
        return (board_from_fen, (board_to_fen(self),))

    def __repr__(self):
        return (
            f"Board(turn={'wb'[self.turn]}, mode={self.mode}, "
            f"move={self.move_number})"
        )


# ---------------------------------------------------------------------------
# Boundary conversion: FEN -> Board
# ---------------------------------------------------------------------------

def board_from_fen(str fen):
    """Parse a FEN string into a typed :class:`Board`.

    Uses the pure-Python ``gungi.fen.parse_fen`` for correctness, then copies
    the structured data into the typed arrays.  This is a boundary operation —
    not on the hot path.
    """
    from komugi.engine.fen import parse_fen
    from komugi.engine.utils import setup_mode_to_code

    parsed = parse_fen(fen)

    cdef Board b = Board()

    # --- board cells ---
    # parsed["board"] is board[rank-1][9-file], each cell is a tower list.
    # Our cells[r, f, t]: r=rank-1, f=file-1.
    # Pure-Python file index j corresponds to file = 9-j, so f = 8-j.
    cdef int r, j, f, t
    cdef int ti, col, code

    py_board = parsed["board"]
    for r in range(9):
        row = py_board[r]
        for j in range(9):
            f = 8 - j  # convert from 9-file indexing to file-1 indexing
            tower = row[j]
            if tower[0] is None:
                continue
            for t in range(len(tower)):
                p = tower[t]
                if p is None:
                    continue
                kanji = p["type"]
                ti = PIECE_INDEX[kanji]
                col = 0 if p["color"] == "w" else 1
                b.cells[r, f, t] = _make_code(ti, col)

    # --- hand ---
    for hp in parsed["hand"]:
        kanji = hp["type"]
        ti = PIECE_INDEX[kanji]
        col = 0 if hp["color"] == "w" else 1
        b.hand[ti, col] = hp["count"]
        b._hand_order.append((ti, col))

    # --- scalars ---
    b.turn = 0 if parsed["turn"] == "w" else 1
    b.mode = setup_mode_to_code[parsed["mode"]]
    b.draft_w = 1 if parsed["drafting"]["w"] else 0
    b.draft_b = 1 if parsed["drafting"]["b"] else 0
    b.move_number = parsed["moveNumber"]

    return b


# ---------------------------------------------------------------------------
# Boundary conversion: Board -> FEN
# ---------------------------------------------------------------------------

def board_to_fen(Board b):
    """Serialize a typed :class:`Board` back to a canonical FEN string.

    Produces the same output as ``gungi.fen.encode_fen`` on the equivalent
    parsed dict.
    """
    cdef int r, j, f, t, code, ti, col
    cdef int empty, h, cw, cb

    parts = []  # one string per rank

    for r in range(9):
        rank_str = ""
        empty = 0
        # Iterate files in pure-Python board order: j=0..8, file = 9-j,
        # which means f = 8-j.
        for j in range(9):
            f = 8 - j

            # Determine tower height at (r, f).
            h = 0
            for t in range(2, -1, -1):
                if b.cells[r, f, t] != 0:
                    h = t + 1
                    break

            if h == 0:
                empty += 1
                continue

            if empty > 0:
                rank_str += str(empty)
                empty = 0

            if h == 1:
                # Single piece — no tower delimiters.
                code = b.cells[r, f, 0]
                ti = _type_index(code)
                col = _color(code)
                letter = _IDX_TO_FEN_LETTER[ti]
                rank_str += letter.upper() if col == 0 else letter
            else:
                # Multi-piece tower: |p1:p2:...|
                tower_parts = []
                for t in range(h):
                    code = b.cells[r, f, t]
                    ti = _type_index(code)
                    col = _color(code)
                    letter = _IDX_TO_FEN_LETTER[ti]
                    tower_parts.append(letter.upper() if col == 0 else letter)
                rank_str += "|" + ":".join(tower_parts) + "|"

        if empty > 0:
            rank_str += str(empty)
        parts.append(rank_str)

    placement = "/".join(parts)

    # --- hand ---
    # White hand first, then black.  Preserve original FEN order via
    # _hand_order recorded during board_from_fen; fall back to type-index
    # order (0..13) for boards built manually.
    white_hand = ""
    black_hand = ""
    if b._hand_order:
        for ti_col in b._hand_order:
            ti = ti_col[0]
            col = ti_col[1]
            cnt = b.hand[ti, col]
            if cnt <= 0:
                continue
            letter = _IDX_TO_FEN_LETTER[ti]
            if col == 0:
                white_hand += letter.upper() + str(cnt)
            else:
                black_hand += letter + str(cnt)
    else:
        for ti in range(14):
            cw = b.hand[ti, 0]
            cb = b.hand[ti, 1]
            if cw > 0:
                white_hand += _IDX_TO_FEN_LETTER[ti].upper() + str(cw)
            if cb > 0:
                black_hand += _IDX_TO_FEN_LETTER[ti] + str(cb)

    hand_str = (white_hand if white_hand else "-") + "/" + (black_hand if black_hand else "-")

    # --- turn ---
    turn_str = "w" if b.turn == 0 else "b"

    # --- drafting ---
    draft_str = ""
    if b.draft_w:
        draft_str += "w"
    if b.draft_b:
        draft_str += "b"
    if not draft_str:
        draft_str = "-"

    return f"{placement} {hand_str} {turn_str} {b.mode} {draft_str} {b.move_number}"
