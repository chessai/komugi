# cython: binding=True
# cython: embedsignature=True
# cython: embedsignature.format=python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
cymovegen — Cython move generation for Gungi.

Faithful port of gungi/move_gen.py operating on the typed cyboard.Board.
No FEN strings are used inside generation — all work is done on integer
arrays.  FEN is only used at the boundary (board_from_fen).
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

# Kanji lookup (for SAN construction).
cdef tuple _KANJI = PIECE_TYPES  # indexed by type_index

# Move-type codes.
cdef int MT_ROUTE   = 0
cdef int MT_TSUKE   = 1
cdef int MT_CAPTURE = 2
cdef int MT_BETRAY  = 3
cdef int MT_ARATA   = 4

# Move-type name strings (for the dict record).
_MT_NAMES = ("route", "tsuke", "capture", "betray", "arata")

# ---------------------------------------------------------------------------
# Action-encoding constants (mirror action_encoding.py exactly)
#
# The mask filler runs with no python objects per move, so it cannot read the
# constants out of action_encoding at call time — they are duplicated here by
# design.  The underscored C values are what the filler uses; the public
# aliases below exist so a test can assert the duplication has not drifted
# (KOM-128 T4).
# ---------------------------------------------------------------------------

cdef int _A0_START = 0       # board, no-stack (route / capture)
cdef int _A1_START = 6561    # board, stack (tsuke / betray), selector-widened
cdef int _B_START  = 32805   # arata, betray-bit widened
cdef int _C_START  = 35073   # arata + draft-finish, betray-bit widened
cdef int _ACTION_SIZE = 37341  # 461 planes * 81 squares

A0_START = _A0_START
A1_START = _A1_START
B_START = _B_START
C_START = _C_START
ACTION_SIZE = _ACTION_SIZE

cdef inline int _sq_index(int rank_1based, int file_1based):
    """Map 1-based (rank, file) to 0..80 matching action_encoding.sq_index."""
    return (rank_1based - 1) * 9 + (9 - file_1based)

# ---------------------------------------------------------------------------
# Direction table: 8 directions — matches gungi.js / move_gen.py exactly.
#   [NW, N, NE, E, W, SE, S, SW]
# Each entry is (dy, dx).
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

# ---------------------------------------------------------------------------
# Probe tables: probe_pval[type_index][dir] and probe_pcarry[type_index][dir].
#
# pval=0 means direction is inactive.
# pval=99 is the sentinel for infinity (slider).
# ---------------------------------------------------------------------------

cdef int _PROBE_PVAL[14][8]
cdef int _PROBE_PCARRY[14][8]

# Helper to fill a row.
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

# 0  帥 marshal:       1  1  1  1  1  1  1  1   (all pcarry=1)
_fill_probe( 0,  1, 1, 1, 1, 1, 1, 1, 1,   1,1,1,1,1,1,1,1)
# 1  大 general:       1 inf 1 inf inf 1 inf 1
_fill_probe( 1,  1,99, 1,99,99, 1,99, 1,   1,1,1,1,1,1,1,1)
# 2  中 lt_general:   inf 1 inf 1  1 inf 1 inf
_fill_probe( 2, 99, 1,99, 1, 1,99, 1,99,   1,1,1,1,1,1,1,1)
# 3  小 mj_general:    1  1  1  1  1  0  1  0
_fill_probe( 3,  1, 1, 1, 1, 1, 0, 1, 0,   1,1,1,1,1,1,1,1)
# 4  侍 warrior:       1  1  1  0  0  0  1  0
_fill_probe( 4,  1, 1, 1, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
# 5  槍 lancer:        1 [1,2] 1  0  0  0  1  0
_fill_probe( 5,  1, 1, 1, 0, 0, 0, 1, 0,   1,2,1,1,1,1,1,1)
# 6  馬 rider:         0 [1,2] 0  1  1  0 [1,2] 0
_fill_probe( 6,  0, 1, 0, 1, 1, 0, 1, 0,   1,2,1,1,1,1,2,1)
# 7  忍 spy:          [1,2] 0 [1,2] 0  0 [1,2] 0 [1,2]
_fill_probe( 7,  1, 0, 1, 0, 0, 1, 0, 1,   2,1,2,1,1,2,1,2)
# 8  砦 fortress:      0  1  0  1  1  1  0  1
_fill_probe( 8,  0, 1, 0, 1, 1, 1, 0, 1,   1,1,1,1,1,1,1,1)
# 9  兵 soldier:       0  1  0  0  0  0  1  0
_fill_probe( 9,  0, 1, 0, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
# 10 砲 cannon:        0  3  0  1  1  0  1  0
_fill_probe(10,  0, 3, 0, 1, 1, 0, 1, 0,   1,1,1,1,1,1,1,1)
# 11 弓 archer:        2  2  2  0  0  0  1  0
_fill_probe(11,  2, 2, 2, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)
# 12 筒 musketeer:     0  2  0  0  0  1  0  1
_fill_probe(12,  0, 2, 0, 0, 0, 1, 0, 1,   1,1,1,1,1,1,1,1)
# 13 謀 tactician:     1  0  1  0  0  0  1  0
_fill_probe(13,  1, 0, 1, 0, 0, 0, 1, 0,   1,1,1,1,1,1,1,1)

# ---------------------------------------------------------------------------
# SQUARES iteration order — rank 1->9, file 9->1 (matches gungi.js / utils.py)
# Stored as (rank, file) pairs, both 1-based.
# ---------------------------------------------------------------------------

cdef int _SQUARES[81][2]

cdef int _sq_idx = 0
cdef int _sq_rank, _sq_y
for _sq_rank in range(1, 10):
    for _sq_y in range(9):
        _SQUARES[_sq_idx][0] = _sq_rank
        _SQUARES[_sq_idx][1] = 9 - _sq_y
        _sq_idx += 1

# ---------------------------------------------------------------------------
# Inline helpers for board access (1-based rank/file)
# ---------------------------------------------------------------------------

cdef inline int _board_top_code(Board b, int rank, int file):
    """Return the piece code at the top of tower at 1-based (rank, file), or 0."""
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

cdef inline int _type_index(int code):
    return (code - 1) >> 1

cdef inline int _color(int code):
    return (code - 1) & 1


# ---------------------------------------------------------------------------
# _get_available_squares — faithful port of move_gen._get_available_squares
#
# All coordinates are 1-based (rank 1..9, file 1..9).
# Returns a list of (rank, file) tuples for reachable squares.
# ---------------------------------------------------------------------------

cdef list _get_available_squares(
    Board b,
    int dy, int dx,        # direction
    int sy, int sx,        # start position (first square to check)
    int py, int px,        # origin piece position
    int origin_ti,         # origin piece type_index
    int origin_color,      # origin piece color (0=w, 1=b)
    int origin_tier,       # origin piece tier (1-based)
    int length,            # max steps (use 99 for inf)
):
    """Return reachable squares along direction from start, given origin piece."""
    cdef list available = []
    cdef int ry, rx, code, ti, col, tier
    cdef int fy, fx, step
    cdef int side, wing_rank, wing_file, wing_code, wing_tier
    cdef int moving_forward

    # -- Archer diagonal wing check --
    if origin_ti == ARCHER and abs(dy) == 1 and abs(dx) == 1:
        wing_rank = py + dy
        wing_file = px + dx
        if 1 <= wing_rank <= 9 and 1 <= wing_file <= 9:
            wing_code = _board_top_code(b, wing_rank, wing_file)
            if wing_code != 0:
                wing_tier = _board_height(b, wing_rank, wing_file)
                if wing_tier > origin_tier:
                    return []

    # -- Reverse probe: walk backward from start toward origin --
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

        # Color-relative "side" step and origin-square break
        side = -1 if origin_color == 1 else 1  # b=-1, w=1
        if 1 <= ry + side <= 9 and 1 <= rx <= 9:
            if ry + side == py and rx == px:
                break

    # -- Forward probe --
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
            # Piece present — check leap eligibility
            if origin_ti != CANNON and origin_ti != MUSKETEER and origin_ti != ARCHER:
                break
            # Leap pieces can only leap when moving forward
            if origin_color == 0:  # white
                moving_forward = 1 if dy < 0 else 0
            else:  # black
                moving_forward = 1 if dy > 0 else 0
            if not moving_forward:
                break

        fx += dx
        fy += dy
        step += 1

    return available


# ---------------------------------------------------------------------------
# Betrayal combinations — faithful port of move_gen._generate_combinations
# and move_gen._get_betrayal_combos.
# ---------------------------------------------------------------------------

cdef list _generate_combinations(list items):
    """Return all non-empty subsets of *items* in recursive pre-order."""
    cdef list result = []

    def helper(start, current):
        if len(current) > 0:
            result.append(list(current))
        for i in range(start, len(items)):
            current.append(items[i])
            helper(i + 1, current)
            current.pop()

    helper(0, [])
    return result


cdef list _get_betrayal_combos(Board b, int rank, int file, int color,
                               int drop_ti):
    """Return all betrayal combo subsets for a tactician move to (rank, file).

    Each combo is a list of (type_index, tier) tuples representing enemy
    pieces in the tower that can be betrayed.
    *color* is the moving player's color (0=w, 1=b).
    Coordinates are 1-based.

    *drop_ti* is the type index of the piece the move SPENDS FROM HAND before
    the conversion is paid for — the dropped piece of an arata — or -1 for a
    board move, whose tactician comes off the board and costs the hand nothing.
    See ``_qualifying_tier_mask`` for why the drop has to be counted (KOM-170).
    """
    cdef int r = rank - 1
    cdef int f = file - 1
    cdef int h = _board_height(b, rank, file)
    if h == 0:
        return []

    # Collect enemies in the tower (bottom to top, preserving order).
    cdef list enemies = []
    cdef int t, code, ti, col
    for t in range(h):
        code = b.cells[r, f, t]
        if code == 0:
            continue
        ti = _type_index(code)
        col = _color(code)
        if col != color:
            enemies.append((ti, t + 1))  # (type_index, tier 1-based)

    if not enemies:
        return []

    # Count enemies by type.
    cdef dict enemy_count = {}
    for e in enemies:
        eti = e[0]
        if eti in enemy_count:
            enemy_count[eti] = enemy_count[eti] + 1
        else:
            enemy_count[eti] = 1

    # A type qualifies iff hand has count >= number of that type in tower,
    # PLUS the copy an arata of that same type has already spent (KOM-170).
    cdef list betrayal_options = []
    for eti, ecount in enemy_count.items():
        if b.hand[eti, color] >= ecount + (1 if eti == drop_ti else 0):
            for e in enemies:
                if e[0] == eti:
                    betrayal_options.append(e)

    if not betrayal_options:
        return []

    return _generate_combinations(betrayal_options)


cdef inline int _qualifying_tier_mask(Board b, int rank, int file, int color,
                                      int drop_ti):
    """Tier-mask Q of the enemies at (rank, file) that *color* may convert.

    The allocation-free twin of ``_get_betrayal_combos``' ``betrayal_options``:
    the same per-type hand test (hand count >= that type's count in the tower,
    all-or-nothing, over-strictness included), reported as a bit per tier
    instead of a list of combinations.  The legal selectors at this square are
    exactly the submasks of the returned mask, which is what lets the mask
    filler emit them without building a single python object.

    *drop_ti* is the hand type the move spends BEFORE paying for the
    conversion — the arata's dropped piece — or -1 for a board move.

    THE DROPPED PIECE IS SPENT FIRST (KOM-170).  ``cyengine.apply_move``
    applies an arata-betray as ``hand[piece_ti] -= 1`` for the placement and
    then ``hand[cap_ti] -= 1`` per converted enemy — gungi.js ``updateHand``
    semantics, docs/design/betrayal-encoding.md:153-167.  When the dropped type
    IS a converted type those two debits hit the same slot, so a hand holding
    exactly ``count`` copies pays ``count + 1`` and the slot underflows; with
    one 謀 per side that is every drop of the last 謀 onto an enemy 謀, and it
    crashed a self-play worker inside ``repetition_key`` (``hand counts must
    fit in uint8 ... min=-1``).  Requiring the extra copy is what makes the
    generated move affordable by construction.

    DIVERGENCE FROM gungi.js, DELIBERATE.  The reference filters on the hand as
    it stands BEFORE the drop (``getBetrayalCombos``, ``move_gen.ts:402-423``,
    quoted at betrayal-encoding.md:178-192) and so offers the same move; it
    does not underflow only because its hand is a LIST whose entry is deleted
    when the count reaches 0, making the second debit a silent no-op
    (``utils.update_hand``).  The reference therefore lets the mover convert an
    enemy 謀 for free, and ``tests/engine/goldens/playouts_betray.json`` records
    it playing exactly that (``新謀(9-9-3)返謀``).  Komugi's hand is a dense
    array where the same move is a negative count, which is not a position, so
    the move is refused here.  The pure-Python oracle (``move_gen.py``) still
    mirrors the reference; ``tests/rules/test_betrayal_hand_invariant.py``
    states the divergence for the rules-audit U-list.

    Mirrors ``action_encoding.qualifying_mask`` (which is TEST/COMPAT-ONLY and
    still spells the pre-KOM-170 test); coordinates are 1-based and *color* is
    the MOVER's color.
    """
    cdef int r = rank - 1
    cdef int f = file - 1
    cdef int t, u, code, code2, ti, count
    cdef int mask = 0

    for t in range(3):
        code = b.cells[r, f, t]
        if code == 0 or _color(code) == color:
            continue
        ti = _type_index(code)
        # Count this type's enemies across the whole tower (>= 1: itself).
        count = 0
        for u in range(3):
            code2 = b.cells[r, f, u]
            if (code2 != 0 and _color(code2) != color
                    and _type_index(code2) == ti):
                count += 1
        if ti == drop_ti:
            count += 1  # the drop already spent one of this type
        if b.hand[ti, color] >= count:
            mask |= 1 << t

    return mask


# ---------------------------------------------------------------------------
# SAN construction — faithful port of move_gen._create_move's SAN logic.
#
# Does NOT apply the move or produce an after-FEN (that is 5B.3).
# The '#' suffix is derived from the captured list (marshal in captured).
# ---------------------------------------------------------------------------

cdef str _build_san(
    int move_type,         # MT_ROUTE, MT_TSUKE, etc.
    int piece_ti,          # piece type_index
    int from_rank,         # 1-based, or 0 for arata
    int from_file,         # 1-based, or 0 for arata
    int from_tier,         # 1-based, or 0 for arata
    int to_rank,           # 1-based
    int to_file,           # 1-based
    int to_tier,           # 1-based
    list captured,         # list of (type_index, tier) tuples, or None/[]
    bint draft_finished,   # True if 終
):
    cdef str kanji = <str>_KANJI[piece_ti]
    cdef str from_str
    cdef str arata, capture_marker, tsuke_marker, betray_marker, draft_done
    cdef str san_suffix, san, game_over

    # From string
    if from_tier != 0:
        from_str = f"({from_rank}-{from_file}-{from_tier})"
    else:
        from_str = ""

    # Markers
    arata = "\u65b0" if move_type == MT_ARATA else ""          # 新
    capture_marker = "\u53d6" if move_type == MT_CAPTURE else ""  # 取

    # Tsuke marker
    if move_type == MT_TSUKE or (to_tier != 1 and to_tier - from_tier > 0):
        tsuke_marker = "\u4ed8"  # 付
    else:
        tsuke_marker = ""

    # Betray marker
    betray_marker = ""
    if move_type == MT_BETRAY or (move_type == MT_ARATA and captured and len(captured) > 0):
        betray_marker = "\u8fd4"  # 返
        for cap in captured:
            betray_marker += <str>_KANJI[cap[0]]

    # Draft done
    draft_done = "\u7d42" if draft_finished else ""  # 終

    # Betray suppresses tsuke
    san_suffix = betray_marker if betray_marker else tsuke_marker

    san = f"{arata}{kanji}{from_str}{capture_marker}({to_rank}-{to_file}-{to_tier}){san_suffix}{draft_done}"

    # Game-over '#' suffix: if captured list contains a marshal
    game_over = ""
    if captured:
        for cap in captured:
            if cap[0] == MARSHAL:
                game_over = "#"
                break

    return san + game_over


# ---------------------------------------------------------------------------
# Move record builder — returns a dict matching the expected interface.
# ---------------------------------------------------------------------------

cdef dict _make_move_record(
    int move_type,
    int piece_ti,
    int piece_color,
    int from_rank, int from_file, int from_tier,  # 1-based, 0 for arata
    int to_rank, int to_file, int to_tier,         # 1-based
    list captured,         # list of (type_index, tier) tuples
    bint draft_finished,
):
    cdef str san = _build_san(
        move_type, piece_ti,
        from_rank, from_file, from_tier,
        to_rank, to_file, to_tier,
        captured, draft_finished,
    )

    return {
        "from_r": from_rank - 1 if from_rank > 0 else -1,
        "from_f": from_file - 1 if from_file > 0 else -1,
        "to_r": to_rank - 1,
        "to_f": to_file - 1,
        "to_tier": to_tier,
        "move_type": move_type,
        "piece_type_index": piece_ti,
        "color": piece_color,
        "captured": captured if captured else [],
        "draft_finished": draft_finished,
        "san": san,
    }


# ---------------------------------------------------------------------------
# generate_moves_for_square — faithful port of move_gen.generate_moves_for_square
#
# Operates on the typed Board. All coordinates 1-based internally.
# Returns list of move-record dicts.
# ---------------------------------------------------------------------------

cdef list _generate_moves_for_square(Board b, int py, int px):
    """Generate all pseudo-legal moves for the piece on top of (py, px) [1-based]."""
    cdef int top_code = _board_top_code(b, py, px)
    if top_code == 0:
        return []

    cdef int pc_ti = _type_index(top_code)
    cdef int pc_color = _color(top_code)

    # Must be the side to move.
    if pc_color != b.turn:
        return []

    cdef int pc_tier = _board_height(b, py, px)

    cdef int pval, pcarry, dy, dx, sy, sx, length
    cdef int i
    cdef list reachable = []

    for i in range(8):
        pval = _PROBE_PVAL[pc_ti][i]
        pcarry = _PROBE_PCARRY[pc_ti][i]

        if pval < 1:
            continue

        dy = _DIRS[i][0]
        dx = _DIRS[i][1]

        # Flip direction for black.
        if pc_color == 1:
            dy = -dy
            dx = -dx

        if pval == 99:
            # Slider
            sy = py + dy
            sx = px + dx
            length = 99
        else:
            sy = py + pval * dy
            sx = px + dx
            length = pc_tier + pcarry - 1

        reachable.extend(
            _get_available_squares(b, dy, dx, sy, sx, py, px,
                                   pc_ti, pc_color, pc_tier, length)
        )

    cdef int max_tier = 3 if b.mode == 3 else 2
    cdef bint marshal_can_stack = (b.mode >= 2)  # intermediate or advanced

    cdef list moves = []
    cdef int sr, sf  # target square (1-based)
    cdef int target_code, target_ti, target_color, target_tier, target_h
    cdef int new_tier, t, code2
    cdef list captured
    cdef bint has_enemy
    cdef list combos

    for sq in reachable:
        sr = sq[0]
        sf = sq[1]
        target_code = _board_top_code(b, sr, sf)
        target_h = _board_height(b, sr, sf)

        if target_code == 0:
            # Empty square → route to tier 1.
            moves.append(_make_move_record(
                MT_ROUTE, pc_ti, pc_color,
                py, px, pc_tier,
                sr, sf, 1,
                [], False,
            ))
        else:
            target_ti = _type_index(target_code)
            target_color = _color(target_code)
            target_tier = target_h  # top tier = height

            # -- Tsuke --
            if target_tier < max_tier and target_ti != MARSHAL:
                if pc_ti != MARSHAL or marshal_can_stack:
                    moves.append(_make_move_record(
                        MT_TSUKE, pc_ti, pc_color,
                        py, px, pc_tier,
                        sr, sf, target_tier + 1,
                        [], False,
                    ))

                    # -- Betrayal (tactician only) --
                    if pc_ti == TACTICIAN:
                        # Check if tower has any enemy pieces.
                        has_enemy = False
                        for t in range(target_h):
                            code2 = b.cells[sr - 1, sf - 1, t]
                            if code2 != 0 and _color(code2) != pc_color:
                                has_enemy = True
                                break
                        if has_enemy:
                            # A board betrayal moves the 謀 off the board, so
                            # nothing is spent from hand first: drop_ti = -1.
                            combos = _get_betrayal_combos(
                                b, sr, sf, pc_color, -1
                            )
                            for combo in combos:
                                moves.append(_make_move_record(
                                    MT_BETRAY, pc_ti, pc_color,
                                    py, px, pc_tier,
                                    sr, sf, target_tier + 1,
                                    combo, False,
                                ))

            # -- Capture (enemy on top) --
            if target_color != pc_color:
                # new_tier = count of friendly pieces in tower + 1
                new_tier = 0
                for t in range(target_h):
                    code2 = b.cells[sr - 1, sf - 1, t]
                    if code2 != 0 and _color(code2) == pc_color:
                        new_tier += 1
                new_tier += 1

                # captured = all enemy pieces in tower
                captured = []
                for t in range(target_h):
                    code2 = b.cells[sr - 1, sf - 1, t]
                    if code2 != 0 and _color(code2) != pc_color:
                        captured.append((_type_index(code2), t + 1))

                moves.append(_make_move_record(
                    MT_CAPTURE, pc_ti, pc_color,
                    py, px, pc_tier,
                    sr, sf, new_tier,
                    captured, False,
                ))

    return moves


# ---------------------------------------------------------------------------
# generate_arata — faithful port of move_gen.generate_arata
#
# Generates arata (hand placement) moves for a specific piece type + color.
# All coordinates 1-based internally.
# ---------------------------------------------------------------------------

cdef list _generate_arata(Board b, int piece_ti, int piece_color):
    """Generate arata moves for piece (piece_ti, piece_color) from hand."""

    # Must be the side to move and have pieces in hand.
    if piece_color != b.turn:
        return []
    if b.hand[piece_ti, piece_color] <= 0:
        return []

    # If marshal is still in hand, only marshal may be placed.
    if b.hand[MARSHAL, piece_color] > 0 and piece_ti != MARSHAL:
        return []

    cdef bint is_draft = (b.draft_b != 0 or b.draft_w != 0)
    cdef int max_tier = 3 if b.mode == 3 else 2

    # Determine eligible ranks.
    cdef list ranks = []
    cdef list maybe = []
    cdef int rank, file_, start, end, step
    cdef int top_code
    cdef bint has_friendly

    if is_draft:
        if piece_color == 0:  # white
            ranks = [7, 8, 9]
        else:  # black
            ranks = [1, 2, 3]
    else:
        if piece_color == 1:  # black
            start = 1; end = 9; step = 1
        else:  # white
            start = 9; end = 1; step = -1

        rank = start
        while True:
            if piece_color == 1:
                if rank > end:
                    break
            else:
                if rank < end:
                    break

            # Check if any top piece on this rank belongs to the player.
            has_friendly = False
            for file_ in range(1, 10):
                top_code = _board_top_code(b, rank, file_)
                if top_code != 0 and _color(top_code) == piece_color:
                    has_friendly = True
                    break

            if has_friendly:
                ranks.extend(maybe)
                ranks.append(rank)
                maybe = []
            else:
                maybe.append(rank)

            rank += step

    # Generate squares: for each rank, file 1..9.
    # Iterate in the canonical SQUARES order within each rank:
    # file 9, 8, ..., 1 (matching f"{rank}-{9-y}" for y in 0..8).
    # But actually in the Python code it's:
    #   squares = [f"{rank}-{i + 1}" for rank in ranks for i in range(9)]
    # which gives file 1, 2, ..., 9. Let me use that order.

    # Compute player hand count for is_last_piece check.
    cdef int player_hand_count = 0
    cdef int ti2
    for ti2 in range(14):
        player_hand_count += b.hand[ti2, piece_color]
    cdef bint is_last_piece = (player_hand_count == 1)

    cdef bint is_drafting_this_color
    if piece_color == 0:
        is_drafting_this_color = (b.draft_w != 0)
    else:
        is_drafting_this_color = (b.draft_b != 0)

    cdef list moves = []
    cdef int sr, sf, target_tier, h
    cdef int target_code, target_ti, target_color
    cdef list combos

    for rank in ranks:
        for sf in range(1, 10):
            sr = rank
            top_code = _board_top_code(b, sr, sf)
            h = _board_height(b, sr, sf)

            if top_code == 0:
                # Empty → place at tier 1.
                target_tier = 1
            else:
                target_ti = _type_index(top_code)
                target_color = _color(top_code)
                # Can stack only on own pieces, below max_tier, not on marshal.
                if (target_color == piece_color and
                        h < max_tier and
                        target_ti != MARSHAL):
                    target_tier = h + 1
                else:
                    continue
            # -- end placement check --

            # Get betrayal combos (tactician arata only).
            combos = []
            if piece_ti == TACTICIAN and h > 0:
                # The drop spends hand[piece_ti] before the conversion is paid
                # for, so it is counted against the same slot (KOM-170).
                combos = _get_betrayal_combos(
                    b, sr, sf, piece_color, piece_ti
                )

            if is_drafting_this_color:
                if not is_last_piece:
                    # Continue-draft variant (no 終).
                    moves.append(_make_move_record(
                        MT_ARATA, piece_ti, piece_color,
                        0, 0, 0,
                        sr, sf, target_tier,
                        [], False,
                    ))
                    for combo in combos:
                        moves.append(_make_move_record(
                            MT_ARATA, piece_ti, piece_color,
                            0, 0, 0,
                            sr, sf, target_tier,
                            combo, False,
                        ))
                # Draft-finish variant (終) — always available.
                moves.append(_make_move_record(
                    MT_ARATA, piece_ti, piece_color,
                    0, 0, 0,
                    sr, sf, target_tier,
                    [], True,
                ))
                for combo in combos:
                    moves.append(_make_move_record(
                        MT_ARATA, piece_ti, piece_color,
                        0, 0, 0,
                        sr, sf, target_tier,
                        combo, True,
                    ))
            else:
                # Non-draft arata.
                moves.append(_make_move_record(
                    MT_ARATA, piece_ti, piece_color,
                    0, 0, 0,
                    sr, sf, target_tier,
                    [], False,
                ))
                for combo in combos:
                    moves.append(_make_move_record(
                        MT_ARATA, piece_ti, piece_color,
                        0, 0, 0,
                        sr, sf, target_tier,
                        combo, False,
                    ))

    return moves


# ---------------------------------------------------------------------------
# Mask-filling move enumeration (fast path — no python objects per move)
# ---------------------------------------------------------------------------

cdef void _fill_moves_for_square(Board b, int py, int px,
                                  unsigned char[::1] mask):
    """Set mask bits for all pseudo-legal moves from (py, px) [1-based]."""
    cdef int top_code = _board_top_code(b, py, px)
    if top_code == 0:
        return

    cdef int pc_ti = _type_index(top_code)
    cdef int pc_color = _color(top_code)

    if pc_color != b.turn:
        return

    cdef int pc_tier = _board_height(b, py, px)

    cdef int pval, pcarry, dy, dx, sy, sx, length
    cdef int i
    cdef list reachable = []

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
        reachable.extend(
            _get_available_squares(b, dy, dx, sy, sx, py, px,
                                   pc_ti, pc_color, pc_tier, length)
        )

    cdef int max_tier = 3 if b.mode == 3 else 2
    cdef bint marshal_can_stack = (b.mode >= 2)

    cdef int from_sq = _sq_index(py, px)
    cdef int sr, sf, to_sq
    cdef int target_code, target_ti, target_color, target_tier, target_h
    cdef int t, code2
    cdef int qualifying, sel
    cdef bint has_enemy

    for sq in reachable:
        sr = sq[0]
        sf = sq[1]
        target_code = _board_top_code(b, sr, sf)
        target_h = _board_height(b, sr, sf)
        to_sq = _sq_index(sr, sf)

        if target_code == 0:
            # Empty → route
            mask[_A0_START + from_sq * 81 + to_sq] = 1
        else:
            target_ti = _type_index(target_code)
            target_color = _color(target_code)
            target_tier = target_h

            # Tsuke (+ betray) — A1, one id per conversion selector.
            if target_tier < max_tier and target_ti != MARSHAL:
                if pc_ti != MARSHAL or marshal_can_stack:
                    # The tsuke itself is selector 0: convert nothing.
                    mask[_A1_START + (from_sq * 4) * 81 + to_sq] = 1

                    # Betrayal (tactician only) — every selector naming none
                    # but qualifying enemies is legal, so at most three more
                    # stores and still no python object per move.
                    if pc_ti == TACTICIAN:
                        has_enemy = False
                        for t in range(target_h):
                            code2 = b.cells[sr - 1, sf - 1, t]
                            if code2 != 0 and _color(code2) != pc_color:
                                has_enemy = True
                                break
                        if has_enemy:
                            qualifying = _qualifying_tier_mask(
                                b, sr, sf, pc_color, -1
                            )
                            for sel in range(1, 4):
                                if (sel & ~qualifying) == 0:
                                    mask[
                                        _A1_START
                                        + (from_sq * 4 + sel) * 81
                                        + to_sq
                                    ] = 1

            # Capture (enemy on top)
            if target_color != pc_color:
                mask[_A0_START + from_sq * 81 + to_sq] = 1


cdef void _fill_arata(Board b, int piece_ti, int piece_color,
                       unsigned char[::1] mask):
    """Set mask bits for arata moves for piece (piece_ti, piece_color)."""
    if piece_color != b.turn:
        return
    if b.hand[piece_ti, piece_color] <= 0:
        return
    if b.hand[MARSHAL, piece_color] > 0 and piece_ti != MARSHAL:
        return

    cdef bint is_draft = (b.draft_b != 0 or b.draft_w != 0)
    cdef int max_tier = 3 if b.mode == 3 else 2

    cdef list ranks = []
    cdef list maybe = []
    cdef int rank, file_, start, end, step
    cdef int top_code
    cdef bint has_friendly

    if is_draft:
        if piece_color == 0:
            ranks = [7, 8, 9]
        else:
            ranks = [1, 2, 3]
    else:
        if piece_color == 1:
            start = 1; end = 9; step = 1
        else:
            start = 9; end = 1; step = -1
        rank = start
        while True:
            if piece_color == 1:
                if rank > end:
                    break
            else:
                if rank < end:
                    break
            has_friendly = False
            for file_ in range(1, 10):
                top_code = _board_top_code(b, rank, file_)
                if top_code != 0 and _color(top_code) == piece_color:
                    has_friendly = True
                    break
            if has_friendly:
                ranks.extend(maybe)
                ranks.append(rank)
                maybe = []
            else:
                maybe.append(rank)
            rank += step

    # Player hand count for is_last_piece check
    cdef int player_hand_count = 0
    cdef int ti2
    for ti2 in range(14):
        player_hand_count += b.hand[ti2, piece_color]
    cdef bint is_last_piece = (player_hand_count == 1)

    cdef bint is_drafting_this_color
    if piece_color == 0:
        is_drafting_this_color = (b.draft_w != 0)
    else:
        is_drafting_this_color = (b.draft_b != 0)

    cdef int sr, sf, h, target_ti, target_color, to_sq
    cdef int bsel

    for rank in ranks:
        for sf in range(1, 10):
            sr = rank
            top_code = _board_top_code(b, sr, sf)
            h = _board_height(b, sr, sf)

            if top_code == 0:
                pass  # empty — can place
            else:
                target_ti = _type_index(top_code)
                target_color = _color(top_code)
                if (target_color == piece_color and
                        h < max_tier and
                        target_ti != MARSHAL):
                    pass  # can stack
                else:
                    continue

            to_sq = _sq_index(sr, sf)

            # One-bit conversion selector: a tactician dropping onto a tower
            # whose tier-1 enemy qualifies may flip it.  A legal drop target is
            # empty or friendly on top, so tier 1 is the only enemy the rules
            # can ever offer — hence one bit, not two.
            bsel = 0
            if piece_ti == TACTICIAN and h > 0:
                # Same hand test as the record path, drop included (KOM-170):
                # the filler and _generate_arata must offer the same drops.
                if _qualifying_tier_mask(b, sr, sf, piece_color, piece_ti) & 1:
                    bsel = 1

            if is_drafting_this_color:
                if not is_last_piece:
                    # Continue-draft (no 終)
                    mask[_B_START + (piece_ti * 2) * 81 + to_sq] = 1
                    if bsel:
                        mask[_B_START + (piece_ti * 2 + 1) * 81 + to_sq] = 1
                # Draft-finish (終)
                mask[_C_START + (piece_ti * 2) * 81 + to_sq] = 1
                if bsel:
                    mask[_C_START + (piece_ti * 2 + 1) * 81 + to_sq] = 1
            else:
                mask[_B_START + (piece_ti * 2) * 81 + to_sq] = 1
                if bsel:
                    mask[_B_START + (piece_ti * 2 + 1) * 81 + to_sq] = 1


# ---------------------------------------------------------------------------
# generate_all_moves — public entry point.
#
# Mirrors gungi.move_gen.generate_all_moves exactly:
#   Draft phase → only arata moves (for all hand pieces).
#   Normal play → board moves (SQUARES order) + arata moves.
#
# Hand piece iteration order must match the FEN hand order preserved in
# Board._hand_order (list of (type_index, color) tuples).
# ---------------------------------------------------------------------------

def generate_all_moves(Board b):
    """Generate all pseudo-legal moves for the side to move.

    Returns a list of move-record dicts, each containing at least 'san'.
    """
    cdef bint is_draft = (b.draft_b != 0 or b.draft_w != 0)
    cdef list result = []
    cdef int i, rank, file_
    cdef int ti, col

    if is_draft:
        # Draft phase: only arata moves for all hand pieces (in FEN order).
        for hp in b._hand_order:
            ti = hp[0]
            col = hp[1]
            result.extend(_generate_arata(b, ti, col))
    else:
        # Normal play: board moves in SQUARES order, then arata.
        for i in range(81):
            rank = _SQUARES[i][0]
            file_ = _SQUARES[i][1]
            result.extend(_generate_moves_for_square(b, rank, file_))

        for hp in b._hand_order:
            ti = hp[0]
            col = hp[1]
            result.extend(_generate_arata(b, ti, col))

    return result


# ---------------------------------------------------------------------------
# fill_legal_mask — native fast path (no python objects per move)
# ---------------------------------------------------------------------------

cpdef void fill_legal_mask(Board b, unsigned char[::1] mask):
    """Fill *mask* (length ACTION_SIZE=37341, pre-zeroed) with 1s for legal moves.

    Same move enumeration as generate_all_moves but emits action ids inline
    via integer arithmetic — no dicts, no SAN strings, no python per-move.
    The conversion selectors (KOM-028) are emitted here too: the filler and
    the record path must agree exactly about which of them are legal.
    """
    cdef bint is_draft = (b.draft_b != 0 or b.draft_w != 0)
    cdef int i, rank, file_
    cdef int ti, col

    if is_draft:
        for hp in b._hand_order:
            ti = hp[0]
            col = hp[1]
            _fill_arata(b, ti, col, mask)
    else:
        for i in range(81):
            rank = _SQUARES[i][0]
            file_ = _SQUARES[i][1]
            _fill_moves_for_square(b, rank, file_, mask)

        for hp in b._hand_order:
            ti = hp[0]
            col = hp[1]
            _fill_arata(b, ti, col, mask)
