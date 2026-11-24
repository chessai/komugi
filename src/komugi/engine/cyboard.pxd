# cython: language_level=3
"""
Declaration file for cyboard — typed board representation for Gungi.

Allows other .pyx modules to ``cimport`` Board and use its C-level fields
directly (no Python overhead on attribute access).
"""

cimport numpy as np

cdef class Board:
    # 9x9x3 grid: cells[r, f, t] = piece code (0=empty, 1..28=piece).
    # r = rank-1, f = file-1, t = tier-1 (bottom-up).
    cdef public np.ndarray cells_arr
    cdef public signed char[:, :, :] cells

    # hand[type_index, color] = count of that piece in that player's hand.
    cdef public np.ndarray hand_arr
    cdef public int[:, :] hand

    # Scalars.
    cdef public int turn        # 0=white, 1=black
    cdef public int mode        # 0=intro, 1=beginner, 2=intermediate, 3=advanced
    cdef public int draft_w     # 1 if white is drafting
    cdef public int draft_b     # 1 if black is drafting
    cdef public int move_number

    # Hand piece order: list of (type_index, color) preserving FEN input order.
    cdef public list _hand_order

    # Fast queries.
    cpdef int top(self, int r, int f)
    cpdef int height(self, int r, int f)
    cpdef Board copy(self)
