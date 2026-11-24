# cython: binding=True
# cython: embedsignature=True
# cython: embedsignature.format=python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
cyengine — Move application and terminal detection for Gungi (Cython).

Faithful port of gungi/move_gen.py (_make_move) and gungi/api.py
(Gungi class check/terminal methods) operating on the typed cyboard.Board.
No FEN strings inside operations — FEN only at boundaries.
"""

import numpy as np
cimport numpy as np

from komugi.engine.cyboard cimport Board
from komugi.engine.cyboard import PIECE_TYPES, PIECE_INDEX

# ---------------------------------------------------------------------------
# Piece-type index constants (canonical order, matches cyboard)
# ---------------------------------------------------------------------------

cdef int MARSHAL     = 0
cdef int GENERAL     = 1
cdef int LT_GENERAL  = 2
cdef int MJ_GENERAL  = 3
cdef int WARRIOR     = 4
cdef int LANCER      = 5
cdef int RIDER       = 6
cdef int SPY         = 7
cdef int FORTRESS    = 8
cdef int SOLDIER     = 9
cdef int CANNON      = 10
cdef int ARCHER      = 11
cdef int MUSKETEER   = 12
cdef int TACTICIAN   = 13

# Move-type codes (must match cymovegen).
cdef int MT_ROUTE   = 0
cdef int MT_TSUKE   = 1
cdef int MT_CAPTURE = 2
cdef int MT_BETRAY  = 3
cdef int MT_ARATA   = 4

# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------

cdef inline int _type_index(int code):
    return (code - 1) >> 1

cdef inline int _color(int code):
    return (code - 1) & 1

cdef inline int _make_code(int ti, int col):
    return ti * 2 + col + 1

cdef inline int _board_top_code(Board b, int rank, int file):
    """Return piece code at top of tower at 1-based (rank, file), or 0."""
    cdef int t
    cdef int r = rank - 1
    cdef int f = file - 1
    for t in range(2, -1, -1):
        if b.cells[r, f, t] != 0:
            return b.cells[r, f, t]
    return 0

cdef inline int _board_height(Board b, int rank, int file):
    """Return tower height at 1-based (rank, file)."""
    cdef int t
    cdef int r = rank - 1
    cdef int f = file - 1
    for t in range(2, -1, -1):
        if b.cells[r, f, t] != 0:
            return t + 1
    return 0


# ---------------------------------------------------------------------------
# Board mutation helpers (1-based rank/file)
# ---------------------------------------------------------------------------

cdef inline void _remove_top(Board b, int rank, int file):
    """Pop the top piece from the tower at 1-based (rank, file)."""
    cdef int h = _board_height(b, rank, file)
    if h > 0:
        b.cells[rank - 1, file - 1, h - 1] = 0


cdef inline void _put(Board b, int rank, int file, int code):
    """Place piece *code* on top of the tower at 1-based (rank, file)."""
    cdef int h = _board_height(b, rank, file)
    b.cells[rank - 1, file - 1, h] = code


cdef void _remove_captured(Board b, int rank, int file, list captured, int mover_color):
    """Remove ALL captured enemy pieces from the tower at (rank, file).

    *captured* is a list of (type_index, tier) tuples — these are the enemies.
    After removal, rebuild the tower from remaining pieces (preserving order).

    Mirrors gungi/utils.py ``remove(square, pieces, board)``:
    the pure-Python version removes by JSON equality; here we match by
    (type_index, color != mover_color, tier).
    """
    cdef int r = rank - 1
    cdef int f = file - 1
    cdef int h = _board_height(b, rank, file)
    if h == 0:
        return

    # Build a set of (type_index, tier_1based) to remove.
    cdef set to_remove = set()
    for cap in captured:
        to_remove.add((cap[0], cap[1]))

    # Collect surviving pieces in order (bottom to top).
    cdef list survivors = []
    cdef int t, code, ti, col
    for t in range(h):
        code = b.cells[r, f, t]
        if code == 0:
            continue
        ti = _type_index(code)
        col = _color(code)
        key = (ti, t + 1)
        if col != mover_color and key in to_remove:
            to_remove.discard(key)  # consume match
            continue
        survivors.append(code)

    # Clear the tower and rewrite.
    for t in range(3):
        b.cells[r, f, t] = 0
    for t in range(len(survivors)):
        b.cells[r, f, t] = survivors[t]


cdef void _convert_captured(Board b, int rank, int file, list captured, int mover_color):
    """Flip the color of matching enemy pieces in the tower at (rank, file).

    Mirrors gungi/utils.py ``convert(square, pieces, board)``.
    *captured* is list of (type_index, tier) tuples — enemies to flip.
    """
    cdef int r = rank - 1
    cdef int f = file - 1
    cdef int h = _board_height(b, rank, file)
    if h == 0:
        return

    # Build a set of (type_index, tier_1based) to convert.
    cdef set to_convert = set()
    for cap in captured:
        to_convert.add((cap[0], cap[1]))

    cdef int t, code, ti, col, new_col
    for t in range(h):
        code = b.cells[r, f, t]
        if code == 0:
            continue
        ti = _type_index(code)
        col = _color(code)
        key = (ti, t + 1)
        if col != mover_color and key in to_convert:
            new_col = 0 if col == 1 else 1
            b.cells[r, f, t] = _make_code(ti, new_col)


# ---------------------------------------------------------------------------
# apply_move — faithful port of move_gen._make_move
#
# Mutates Board *b* in place.  The move dict comes from cymovegen.
# ---------------------------------------------------------------------------

def apply_move(Board b, dict move):
    """Apply *move* (a cymovegen move-record dict) to Board *b* in place.

    This is the Cython equivalent of move_gen._make_move, but operates
    directly on the typed board without FEN round-trips.
    """
    cdef int move_type = move["move_type"]
    cdef int piece_ti = move["piece_type_index"]
    cdef int piece_color = move["color"]
    cdef int from_r = move["from_r"]   # 0-based, -1 for arata
    cdef int from_f = move["from_f"]   # 0-based, -1 for arata
    cdef int to_r = move["to_r"]       # 0-based
    cdef int to_f = move["to_f"]       # 0-based
    cdef int to_tier = move["to_tier"] # 1-based
    cdef list captured = move["captured"]
    cdef bint draft_finished = move["draft_finished"]

    # Convert to 1-based rank/file for internal helpers.
    cdef int to_rank = to_r + 1
    cdef int to_file = to_f + 1
    cdef int from_rank = from_r + 1  # will be 0 for arata
    cdef int from_file = from_f + 1  # will be 0 for arata

    cdef int piece_code = _make_code(piece_ti, piece_color)
    cdef int cap_ti, opp_color

    if move_type == MT_ROUTE or move_type == MT_TSUKE:
        # Remove piece from origin, place at destination.
        _remove_top(b, from_rank, from_file)

    elif move_type == MT_CAPTURE:
        # Remove piece from origin, remove all captured enemies, place.
        _remove_top(b, from_rank, from_file)
        _remove_captured(b, to_rank, to_file, captured, piece_color)

    elif move_type == MT_BETRAY:
        # Remove piece from origin, convert captured enemies, decrement
        # the converted owner's hand, place.
        _remove_top(b, from_rank, from_file)
        _convert_captured(b, to_rank, to_file, captured, piece_color)
        # update_hand(captured, hand, opposite_color=True):
        # For each captured piece, decrement the OPPOSITE color's hand.
        # The captured pieces are enemies (color != piece_color).
        # opposite_color=True means: the color to decrement is the
        # opposite of the captured piece's color.  Since the captured
        # pieces are enemies (opp of mover), opposite of enemy = mover.
        # So we decrement mover's hand for each captured type.
        for cap in captured:
            cap_ti = cap[0]
            b.hand[cap_ti, piece_color] -= 1

    elif move_type == MT_ARATA:
        # Spend the placed piece from hand.
        b.hand[piece_ti, piece_color] -= 1
        # Remove from _hand_order if count drops to 0.
        # (We keep _hand_order intact; zero-count entries are harmless
        # for generate_all_moves since _generate_arata checks count > 0.)

        # If captured (arata-betray): convert + hand-adjust.
        if captured and len(captured) > 0:
            _convert_captured(b, to_rank, to_file, captured, piece_color)
            for cap in captured:
                cap_ti = cap[0]
                b.hand[cap_ti, piece_color] -= 1

        # Draft-finish flag.
        if draft_finished:
            if piece_color == 0:
                b.draft_w = 0
            else:
                b.draft_b = 0
            # Black finishing also ends white's draft.
            if piece_color == 1:
                b.draft_w = 0

    # Place the piece at destination.
    _put(b, to_rank, to_file, piece_code)

    # --- moveNumber increment ---
    # Original: if turn == "b" and (drafting["w"] or not drafting["b"]): move_number += 1
    # Original: if turn == "w" and not drafting["b"] and not drafting["w"] and draftFinished: move_number += 1
    # Note: b.turn and b.draft_* have already been updated for arata draft_finished above.
    # But in the original code, the turn/drafting used are the PRE-move values except
    # for the draftFinished flag update. Let me re-read...
    #
    # In _make_move, the drafting update for arata happens BEFORE moveNumber logic.
    # So the drafting values used for moveNumber are POST-draft-finish-update.
    # But turn hasn't been flipped yet at this point.

    cdef int turn = b.turn  # still pre-flip

    if turn == 1 and (b.draft_w != 0 or b.draft_b == 0):
        b.move_number += 1
    if turn == 0 and b.draft_b == 0 and b.draft_w == 0 and draft_finished:
        b.move_number += 1

    # --- Turn flip ---
    # Original logic:
    # if drafting.w == drafting.b:
    #     if not draftFinished or turn != "w":
    #         turn = flip(turn)
    # else:
    #     if not drafting.b and turn == "b": turn = "w"
    #     elif not drafting.w and turn == "w": turn = "b"

    if b.draft_w == b.draft_b:
        if not draft_finished or turn != 0:
            b.turn = 1 - turn
    else:
        if b.draft_b == 0 and turn == 1:
            b.turn = 0
        elif b.draft_w == 0 and turn == 0:
            b.turn = 1


# ---------------------------------------------------------------------------
# Direction + probe tables (duplicated from cymovegen — cdef functions there
# are not accessible via cimport since cymovegen.pxd doesn't export them)
# ---------------------------------------------------------------------------

cdef int _DIRS[8][2]
_DIRS[0][0] = -1; _DIRS[0][1] =  1   # NW
_DIRS[1][0] = -1; _DIRS[1][1] =  0   # N
_DIRS[2][0] = -1; _DIRS[2][1] = -1   # NE
_DIRS[3][0] =  0; _DIRS[3][1] =  1   # E
_DIRS[4][0] =  0; _DIRS[4][1] = -1   # W
_DIRS[5][0] =  1; _DIRS[5][1] =  1   # SE
_DIRS[6][0] =  1; _DIRS[6][1] =  0   # S
_DIRS[7][0] =  1; _DIRS[7][1] = -1   # SW

cdef int _PROBE_PVAL[14][8]
cdef int _PROBE_PCARRY[14][8]

cdef void _fill_probe(int ti, int v0, int v1, int v2, int v3,
                       int v4, int v5, int v6, int v7,
                       int c0, int c1, int c2, int c3,
                       int c4, int c5, int c6, int c7):
    _PROBE_PVAL[ti][0] = v0; _PROBE_PVAL[ti][1] = v1
    _PROBE_PVAL[ti][2] = v2; _PROBE_PVAL[ti][3] = v3
    _PROBE_PVAL[ti][4] = v4; _PROBE_PVAL[ti][5] = v5
    _PROBE_PVAL[ti][6] = v6; _PROBE_PVAL[ti][7] = v7
    _PROBE_PCARRY[ti][0] = c0; _PROBE_PCARRY[ti][1] = c1
    _PROBE_PCARRY[ti][2] = c2; _PROBE_PCARRY[ti][3] = c3
    _PROBE_PCARRY[ti][4] = c4; _PROBE_PCARRY[ti][5] = c5
    _PROBE_PCARRY[ti][6] = c6; _PROBE_PCARRY[ti][7] = c7

_fill_probe( 0,  1, 1, 1, 1, 1, 1, 1, 1,   1,1,1,1,1,1,1,1)
_fill_probe( 1,  1,99, 1,99,99, 1,99, 1,   1,1,1,1,1,1,1,1)
_fill_probe( 2, 99, 1,99, 1, 1,99, 1,99,   1,1,1,1,1,1,1,1)
_fill_probe( 3,  1, 1, 1, 1, 1, 0, 1, 0,   1,1,1,1,1,1,1,1)
_fill_probe( 4,  1, 1, 1, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
_fill_probe( 5,  1, 1, 1, 0, 0, 0, 1, 0,   1,2,1,1,1,1,1,1)
_fill_probe( 6,  0, 1, 0, 1, 1, 0, 1, 0,   1,2,1,1,1,1,2,1)
_fill_probe( 7,  1, 0, 1, 0, 0, 1, 0, 1,   2,1,2,1,1,2,1,2)
_fill_probe( 8,  0, 1, 0, 1, 1, 1, 0, 1,   1,1,1,1,1,1,1,1)
_fill_probe( 9,  0, 1, 0, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
_fill_probe(10,  0, 3, 0, 1, 1, 0, 1, 0,   1,1,1,1,1,1,1,1)
_fill_probe(11,  2, 2, 2, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
_fill_probe(12,  0, 2, 0, 0, 0, 1, 0, 1,   1,1,1,1,1,1,1,1)
_fill_probe(13,  1, 0, 1, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)

# SQUARES iteration order — rank 1->9, file 9->1.
cdef int _SQUARES[81][2]
cdef int _sq_idx = 0
cdef int _sq_rank, _sq_y
for _sq_rank in range(1, 10):
    for _sq_y in range(9):
        _SQUARES[_sq_idx][0] = _sq_rank
        _SQUARES[_sq_idx][1] = 9 - _sq_y
        _sq_idx += 1


# ---------------------------------------------------------------------------
# _get_available_squares — duplicated from cymovegen for attack detection
# ---------------------------------------------------------------------------

cdef list _get_available_squares(
    Board b,
    int dy, int dx,
    int sy, int sx,
    int py, int px,
    int origin_ti,
    int origin_color,
    int origin_tier,
    int length,
):
    cdef list available = []
    cdef int ry, rx, code, ti, col, tier
    cdef int fy, fx, step
    cdef int side, wing_rank, wing_file, wing_code, wing_tier
    cdef int moving_forward

    # Archer diagonal wing check.
    if origin_ti == ARCHER and abs(dy) == 1 and abs(dx) == 1:
        wing_rank = py + dy
        wing_file = px + dx
        if 1 <= wing_rank <= 9 and 1 <= wing_file <= 9:
            wing_code = _board_top_code(b, wing_rank, wing_file)
            if wing_code != 0:
                wing_tier = _board_height(b, wing_rank, wing_file)
                if wing_tier > origin_tier:
                    return []

    # Reverse probe.
    rx = sx
    ry = sy
    while rx != px or ry != py:
        rx -= dx
        ry -= dy
        if not (1 <= ry <= 9 and 1 <= rx <= 9):
            break
        code = _board_top_code(b, ry, rx)
        if code != 0:
            tier = _board_height(b, ry, rx)
            if tier > origin_tier:
                return []
        side = -1 if origin_color == 1 else 1
        if 1 <= ry + side <= 9 and 1 <= rx <= 9:
            if ry + side == py and rx == px:
                break

    # Forward probe.
    fx = sx
    fy = sy
    step = 0
    while step < length:
        if fx < 1 or fx > 9 or fy < 1 or fy > 9:
            break
        code = _board_top_code(b, fy, fx)
        if code != 0:
            tier = _board_height(b, fy, fx)
            if tier > origin_tier:
                break
        available.append((fy, fx))
        if code != 0:
            if origin_ti != CANNON and origin_ti != MUSKETEER and origin_ti != ARCHER:
                break
            if origin_color == 0:
                moving_forward = 1 if dy < 0 else 0
            else:
                moving_forward = 1 if dy > 0 else 0
            if not moving_forward:
                break
        fx += dx
        fy += dy
        step += 1

    return available


# ---------------------------------------------------------------------------
# Attack / check detection — port of move_gen._get_attacked_squares etc.
# ---------------------------------------------------------------------------

cdef list _get_attacked_squares(Board b, int py, int px):
    """Return all squares attacked by the piece on top of (py, px) [1-based]."""
    cdef int top_code = _board_top_code(b, py, px)
    if top_code == 0:
        return []

    cdef int pc_ti = _type_index(top_code)
    cdef int pc_color = _color(top_code)
    cdef int pc_tier = _board_height(b, py, px)

    cdef int pval, pcarry, dy, dx, sy, sx, length
    cdef int i
    cdef list attacked = []

    for i in range(8):
        pval = _PROBE_PVAL[pc_ti][i]
        pcarry = _PROBE_PCARRY[pc_ti][i]
        if pval < 1:
            continue

        dy = _DIRS[i][0]
        dx = _DIRS[i][1]
        if pc_color == 1:
            dy = -dy
            dx = -dx

        if pval == 99:
            sy = py + dy
            sx = px + dx
            length = 99
        else:
            sy = py + pval * dy
            sx = px + dx
            length = pc_tier + pcarry - 1

        attacked.extend(
            _get_available_squares(b, dy, dx, sy, sx, py, px,
                                   pc_ti, pc_color, pc_tier, length)
        )
    return attacked


cdef bint _is_square_attacked(Board b, int target_rank, int target_file, int by_color):
    """Return True if (target_rank, target_file) is attacked by any piece of by_color."""
    cdef int i, rank, file_
    cdef int top_code
    cdef list attacked

    for i in range(81):
        rank = _SQUARES[i][0]
        file_ = _SQUARES[i][1]
        top_code = _board_top_code(b, rank, file_)
        if top_code == 0:
            continue
        if _color(top_code) != by_color:
            continue
        attacked = _get_attacked_squares(b, rank, file_)
        for sq in attacked:
            if sq[0] == target_rank and sq[1] == target_file:
                return True
    return False


cdef bint _in_check(Board b, int color):
    """Return True if *color*'s marshal is in check."""
    cdef int opposite = 1 - color
    cdef int i, rank, file_, top_code

    for i in range(81):
        rank = _SQUARES[i][0]
        file_ = _SQUARES[i][1]
        top_code = _board_top_code(b, rank, file_)
        if top_code != 0 and _type_index(top_code) == MARSHAL and _color(top_code) == color:
            return _is_square_attacked(b, rank, file_, opposite)

    return False


def in_check(Board b, int color):
    """Public: Return True if *color*'s (0=w, 1=b) marshal is in check."""
    return _in_check(b, color)


# ---------------------------------------------------------------------------
# Marshal-captured check — port of gungi/utils.py is_game_over
# ---------------------------------------------------------------------------

cdef bint _marshal_captured(Board b):
    """Return True when fewer or more than 2 marshals remain on the board."""
    cdef int count = 0
    cdef int r, f, t, code
    for r in range(9):
        for f in range(9):
            for t in range(3):
                code = b.cells[r, f, t]
                if code != 0 and _type_index(code) == MARSHAL:
                    count += 1
    return count != 2


# ---------------------------------------------------------------------------
# Checkmate / stalemate — port of gungi/api.py
#
# Uses cymovegen.generate_all_moves (Python-level import) for move gen,
# and apply_move + in_check on copies for escape testing.
# ---------------------------------------------------------------------------

cdef bint _has_escaping_move(Board b):
    """Return True if the side to move has any pseudo-legal move that does
    not leave its own marshal in check. A reply that captures the opposing
    marshal always counts as an escape: it ends the game (mover wins)
    rather than leaving the mover in check (KOM-1109)."""
    from komugi.engine.cymovegen import generate_all_moves

    cdef int mover = b.turn
    cdef list all_moves = generate_all_moves(b)

    cdef Board copy
    for mv in all_moves:
        copy = b.copy()
        apply_move(copy, mv)
        if _marshal_captured(copy) or not _in_check(copy, mover):
            return True
    return False


def is_checkmate(Board b):
    """Return True if the side to move is in checkmate."""
    if b.draft_w != 0 or b.draft_b != 0:
        return False
    if _marshal_captured(b):
        return False
    return _in_check(b, b.turn) and not _has_escaping_move(b)


def is_stalemate(Board b):
    """Return True if the side to move is in stalemate (not in check, no legal move)."""
    if b.draft_w != 0 or b.draft_b != 0:
        return False
    if _marshal_captured(b):
        return False
    return not _in_check(b, b.turn) and not _has_escaping_move(b)


# ---------------------------------------------------------------------------
# Insufficient material — port of gungi/api.py Gungi.is_insufficient_material
# ---------------------------------------------------------------------------

def is_insufficient_material(Board b):
    """Return True if only two non-adjacent marshals remain (+ no non-marshal hand)."""
    if b.draft_w != 0 or b.draft_b != 0:
        return False

    # Hand must contain only marshals (or be empty).
    cdef int ti
    for ti in range(14):
        if ti == MARSHAL:
            continue
        if b.hand[ti, 0] > 0 or b.hand[ti, 1] > 0:
            return False

    # Board must have exactly 2 marshals and nothing else.
    cdef int r, f, t, code
    cdef int marshal_count = 0
    cdef int non_marshal_count = 0
    cdef int m1_rank = 0, m1_file = 0, m2_rank = 0, m2_file = 0

    for r in range(9):
        for f in range(9):
            for t in range(3):
                code = b.cells[r, f, t]
                if code == 0:
                    continue
                if _type_index(code) == MARSHAL:
                    marshal_count += 1
                    if marshal_count == 1:
                        m1_rank = r + 1
                        m1_file = f + 1
                    elif marshal_count == 2:
                        m2_rank = r + 1
                        m2_file = f + 1
                else:
                    non_marshal_count += 1

    if non_marshal_count > 0:
        return False
    if marshal_count != 2:
        return False

    cdef int rank_diff = abs(m1_rank - m2_rank)
    cdef int file_diff = abs(m1_file - m2_file)
    cdef bint is_adjacent = (rank_diff <= 1 and file_diff <= 1)
    return not is_adjacent


# ---------------------------------------------------------------------------
# Fourfold repetition — repaired per docs/design/rules-audit.md (KOM-027)
#
# The key is the complete engine-visible game state minus anything that
# advances monotonically with time: cells, side to move, draft flags and hands.
# move_number and mode are deliberately excluded — including move_number would
# make every key unique and delete the repetition rule outright.
#
# Occurrence counting itself lives in Game, as an O(1) dict counter keyed on
# these bytes; this module only builds keys and applies the threshold, so no
# helper here ever rescans a history list.
# ---------------------------------------------------------------------------

#: Occurrences of the CURRENT position that draw (fourfold, as gungi.js).
cdef int _REPETITION_THRESHOLD = 4


def repetition_key(Board b):
    """Return the 274-byte repetition key for the current position.

    Layout (rules-audit "Proposed key"), in this order:

    * ``cells_arr.tobytes()`` — 243 bytes, ``int8[9][9][3]``
    * ``bytes([turn, draft_w, draft_b])`` — 3 bytes
    * the hand narrowed to ``uint8[14][2]``, ``.tobytes()`` — 28 bytes

    ``hand_arr`` is ``int32`` but every count fits in a byte, so narrowing is a
    canonicalization rather than an optimization: two keys must never differ
    because of padding.  ``turn`` fixes the side-to-move alias (Scenario A) and
    the hands fix the spent-material alias (Scenario B); both change the legal
    move set, which is the test every included component passes.
    """
    cdef bytes cells = b.cells_arr.tobytes()
    cdef bytes scalars = bytes([b.turn, b.draft_w, b.draft_b])
    cdef object hand_arr = b.hand_arr
    # Defence in depth: astype(np.uint8) WRAPS silently, so a hand count that
    # ever left [0, 255] could merge two genuinely distinct positions into one
    # key -- the one silent failure mode left in the key (review F6).  Two
    # scalar reductions over 28 elements are free next to the tobytes() calls
    # above.  Unreachable through play (counts start at <= 4 and only
    # decrease), but hand_arr is a writable public ndarray, so it is reachable.
    cdef int hand_min = int(hand_arr.min())
    cdef int hand_max = int(hand_arr.max())
    if hand_min < 0 or hand_max > 255:
        raise ValueError(
            "hand counts must fit in uint8 to build a repetition key, got "
            "min=%d max=%d; narrowing this would wrap around and alias two "
            "distinct positions to the same key" % (hand_min, hand_max)
        )
    cdef bytes hand = hand_arr.astype(np.uint8).tobytes()
    return cells + scalars + hand


def placement_key(Board b):
    """Return the 243-byte CELLS-ONLY key — gungi.js@3.4.11's rule (KOM-141).

    ``cells_arr.tobytes()``, which is exactly the LEADING BLOCK of
    ``repetition_key``: ``repetition_key(b)[:243] == placement_key(b)`` for
    every board, and that identity is pinned by a test rather than left to the
    two functions to keep by hand.

    This is the key gungi.js's ``isFourfoldRepetition`` counts — FEN field 1
    and nothing else — and it ALIASES genuinely distinct positions: two states
    that differ only in side to move (rules-audit Scenario A) or only in hand
    contents (Scenario B) share it, which is findings F1-F4.  It is built here
    so that a REFEREE can apply the site's rule and the Elo cost of the
    mismatch can be measured; it is never Komugi's training key.

    This function used to be a deprecated alias that raised, so that rules-audit
    **Q13** — "no call site silently keeps the aliased key" — was enforced by
    the impossibility of calling it.  That guard has MOVED, not gone: it is now
    ``tests/engine/test_repetition_rule_default.py``, which walks every ``Game(``
    call site in ``src/komugi`` and requires each one to either pass an explicit
    ``repetition_rule=`` or sit in a documented default-FULL allowlist.  Reach
    this key through ``Game(repetition_rule=RepetitionRule.PLACEMENT)``, never
    by swapping it in behind a caller's back.
    """
    return b.cells_arr.tobytes()


cpdef bint is_repetition_draw(int count):
    """Return True if an occurrence *count* of the CURRENT position draws.

    ``>= 4``, never ``== 4``: the predicate is monotone, so it cannot be
    switched off by an unadjudicated ply, a bridge-constructed mid-game state
    or a truncated history (rules-audit F2).  *count* is the multiplicity of
    the position now on the board, this occurrence included.
    """
    return count >= _REPETITION_THRESHOLD


# ---------------------------------------------------------------------------
# Cheap non-mate terminal — draft/marshal/fourfold/insufficient (NO movegen)
# ---------------------------------------------------------------------------

cpdef bint is_terminal_nonmate(Board b, int repetition_count):
    """Return True iff the position is terminal for a NON-checkmate/stalemate
    reason: still in draft (False), marshal captured, fourfold repetition,
    or insufficient material.  Runs in O(board-scan) — no move generation.

    *repetition_count* is the current position's occurrence count (see
    ``Game.repetition_count``); pass 0 when no history is being tracked."""
    if b.draft_w != 0 or b.draft_b != 0:
        return False
    if _marshal_captured(b):
        return True
    if is_repetition_draw(repetition_count):
        return True
    if is_insufficient_material(b):
        return True
    return False


# ---------------------------------------------------------------------------
# Draw / game-over — port of gungi/api.py
# ---------------------------------------------------------------------------

def is_draw(Board b, int repetition_count=0):
    """Return True if the position is a draw (stalemate, fourfold, or insufficient)."""
    if is_stalemate(b):
        return True
    if is_repetition_draw(repetition_count):
        return True
    if is_insufficient_material(b):
        return True
    return False


def is_game_over(Board b, int repetition_count=0):
    """Return True if the game is over (marshal captured, checkmate, or draw).

    Returns False during drafting phase, matching api.py precedence.
    """
    if b.draft_w != 0 or b.draft_b != 0:
        return False
    if _marshal_captured(b):
        return True
    if is_repetition_draw(repetition_count):
        return True
    if is_insufficient_material(b):
        return True
    return is_checkmate(b) or is_stalemate(b)
