"""
AlphaZero action encoding for Gungi.

Bijective flat action space, ACTION_SIZE = 37341 = 461 * 81, four contiguous
blocks.  Every block is plane-major over the destination square, so an id is
always ``block_start + plane * 81 + to_sq``:

  Block A0 [0, 6561)       — board moves, NO-STACK (route / capture)
                             plane = from_sq                (81 planes)
  Block A1 [6561, 32805)   — board moves, STACK    (tsuke / betray)
                             plane = from_sq * 4 + sel     (324 planes)
  Block B  [32805, 35073)  — arata (hand drop)
                             plane = piece_idx * 2 + bsel   (28 planes)
  Block C  [35073, 37341)  — arata + draft-finish (終)
                             plane = piece_idx * 2 + bsel   (28 planes)

Board squares are indexed 0..80 over gungi.utils.SQUARES (rank-file only;
tier is NEVER encoded). The A0/A1 split disambiguates a capture (lands at
tier 1) from a tsuke (stacks on top) at the same (from_sq, to_sq).

Piece types for blocks B/C are indexed 0..13 in the order they appear in
gungi.utils.piece (the canonical name→kanji dict).

THE CONVERSION SELECTOR (KOM-028, docs/design/betrayal-encoding.md).  A
stacking board move carries a two-bit selector ``sel``: bit ``tier - 1`` is
set iff the piece at that tier of the target tower is converted by the move
(``sel_from_captured``).  ``sel = 0`` is the plain tsuke — "convert nothing" —
so declining a betrayal is a distinct, reachable action, and every partial
betrayal the rules engine generates has an id of its own.  Two bits are exact
rather than budgeted: the tsuke/betray branch requires the target tower to be
below max_tier, so tiers 1 and 2 are the only positions a convertible enemy
can occupy.

An arata carries the one-bit selector ``bsel`` instead: a legal drop target is
empty or friendly on top, which leaves tier 1 as the only possible enemy, so
``bsel = 1`` means "convert that tier-1 enemy".  Arata conversions were
entirely unreachable under the v1 layout (no id decoded to one); they are
ordinary actions here.

A selector is legal iff ``(sel & ~Q) == 0``, where ``Q`` is the tier-mask of
the enemies whose *type* qualifies for conversion — the mover's hand must hold
at least as many copies of that type as the tower holds (``qualifying_mask``,
faithful to gungi.js ``getBetrayalCombos``, over-strictness included).
"""

from __future__ import annotations

from typing import Final

from komugi.engine.utils import SQUARES, piece as PIECE_NAME_TO_KANJI

# ---------------------------------------------------------------------------
# Square indexing — rank-file only, strip tier
# ---------------------------------------------------------------------------

_SQ_TO_INDEX: dict[str, int] = {sq: i for i, sq in enumerate(SQUARES)}

ACTION_SIZE: int = 37341   # 461 planes * 81 squares

# Versions the 37341-id space itself (distinct from ACTION_SIZE, which can't
# tell "the encoding changed" from "it didn't"). v1 was the superseded
# 15390-id space (compat.py's V1_ACTION_SIZE); this is v2.
ACTION_ENCODING_VERSION: Final[int] = 2

# Block start offsets — the running sums of the per-block plane counts
# (81 + 324 + 28 + 28 = 461).
A0_START: int = 0       # board, no-stack (route / capture)
A1_START: int = 6561    # board, stack (tsuke / betray), selector-widened
B_START: int = 32805    # arata, betray-bit widened
C_START: int = 35073    # arata + draft-finish, betray-bit widened

# Half-open [start, stop) id range of block C — the 終 ("shu", draft-finish)
# block.  Every id in this range encodes "place hand piece *p* on square *s*
# (converting the tier-1 enemy or not)", and *nothing else*: the block does NOT
# carry its own semantics.  The betray bit widened the block; its bounds and
# its context-dependence are otherwise exactly as before.
#
# BLOCK C IS CONTEXT-DEPENDENT.  What a C-block id means depends entirely on
# whether the SIDE TO MOVE is still drafting (board.draft_w for white,
# board.draft_b for black):
#
#   * mover's draft bit SET   → the move places the piece AND ends that
#     player's draft (the 終 variant).  cymovegen emits the C-block bit for
#     every draft placement square, and — because it gates the continue-draft
#     (block B) variant on ``not is_last_piece`` — the C-block bits are the
#     ONLY bits emitted when the drafting side has a single piece left in hand.
#   * mover's draft bit CLEAR → there is no draft to finish, so the very same
#     id is a plain hand drop (arata).  cymovegen never sets C-block bits in
#     that case (it emits block B), but a consumer reasoning about ids in the
#     abstract must not assume "C-block == 終".
#
# Consequence for any curriculum/root mask built on top of this encoding: it
# may only treat C-block ids as 終 variants while the side to move is still
# drafting.  See ``komugi.engine.game.draft_root_mask``.
SHU_BLOCK: tuple[int, int] = (C_START, ACTION_SIZE)

# ---------------------------------------------------------------------------
# Piece-type indexing (0..13)
# ---------------------------------------------------------------------------

# Canonical order: iteration order of gungi.utils.piece dict.
PIECE_TYPES: list[str] = list(PIECE_NAME_TO_KANJI.values())
PIECE_INDEX: dict[str, int] = {kanji: i for i, kanji in enumerate(PIECE_TYPES)}
_INDEX_TO_PIECE: list[str] = PIECE_TYPES  # inverse


def sq_index(square: str) -> int:
    """Map a square string to 0..80.

    Accepts both 'rank-file' (e.g. '6-1') and 'rank-file-tier' (e.g. '6-1-1').
    The tier component is always stripped.
    """
    parts = square.split("-")
    key = f"{parts[0]}-{parts[1]}"
    return _SQ_TO_INDEX[key]


def sq_from_index(idx: int) -> str:
    """Inverse of sq_index — returns 'rank-file' string."""
    return SQUARES[idx]


def is_c_block(action_id: int) -> bool:
    """Return True iff *action_id* lies in block C (``SHU_BLOCK``).

    This is a purely positional test on the id.  It says the id *can* be a 終
    (draft-finish) placement — NOT that it is one: block C is
    context-dependent (see ``SHU_BLOCK``).  The id is a 終 variant only when
    the side to move still has its draft bit set; after that side's draft it
    is an ordinary arata drop.
    """
    return SHU_BLOCK[0] <= action_id < SHU_BLOCK[1]


# ---------------------------------------------------------------------------
# The conversion selector — the two pure helpers the encoder, the mask filler
# and the tests all key on.
# ---------------------------------------------------------------------------

def sel_from_captured(captured) -> int:
    """Tier-mask of a move record's converted pieces.

    *captured* is a list of ``(type_index, tier)`` tuples (the cymovegen
    record shape) or of verbose Piece dicts carrying a ``"tier"`` key (the
    pure-Python oracle shape); ``None`` and ``[]`` both give 0.

    Bit ``tier - 1`` is set per converted piece, so the empty conversion —
    a plain tsuke or a plain drop — is selector 0.  The assertion pins the
    layout's central claim: a tsuke/betray or arata target tower is always
    below max_tier, so nothing above tier 2 is ever convertible (KOM-128 T6.5).
    """
    s = 0
    for item in captured or ():
        tier = item["tier"] if isinstance(item, dict) else item[1]
        assert 1 <= tier <= 2, "conversion above tier 2 is unreachable"
        s |= 1 << (tier - 1)
    return s


def qualifying_mask(board, rank: int, file: int, color: int) -> int:
    """Tier-mask ``Q`` of the enemies at (*rank*, *file*) that *color* may convert.

    *board* is a cyboard.Board, *rank*/*file* are 1-based, *color* is the
    MOVER's color (0=white, 1=black).  A type qualifies iff the mover's hand
    holds at least as many copies of it as the tower holds — the all-or-nothing
    hand test of gungi.js ``getBetrayalCombos``, replicated bit-for-bit
    including its over-strictness (two enemy 兵 against one 兵 in hand
    disqualify BOTH, singletons included).  Q is the tier-mask of every
    instance of every qualifying type, and the legal selectors at this square
    are exactly the submasks of Q.

    This is the same predicate as cymovegen's ``_get_betrayal_combos``
    ``betrayal_options``, reduced to a mask: one definition shared by the
    legality test, the native mask filler and the tests.
    """
    cells = board.cells_arr
    hand = board.hand_arr
    r = rank - 1
    f = file - 1

    enemies: list[tuple[int, int]] = []
    for t in range(cells.shape[2]):
        code = int(cells[r, f, t])
        if code == 0:
            continue
        if ((code - 1) & 1) != color:
            enemies.append(((code - 1) >> 1, t + 1))

    if not enemies:
        return 0

    counts: dict[int, int] = {}
    for type_index, _ in enemies:
        counts[type_index] = counts.get(type_index, 0) + 1

    mask = 0
    for type_index, tier in enemies:
        if int(hand[type_index, color]) >= counts[type_index]:
            mask |= 1 << (tier - 1)
    return mask


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_move(move: dict) -> int:
    """Encode a verbose Move dict into a flat action id in [0, ACTION_SIZE)."""
    mtype = move["type"]

    if mtype == "arata":
        piece_idx = PIECE_INDEX[move["piece"]]
        to = sq_index(move["to"])
        captured = move.get("captured") or []
        # A drop's only convertible enemy sits at tier 1 (the target tower is
        # friendly on top), so the whole conversion is one bit.
        bsel = 1 if captured else 0
        assert not bsel or sel_from_captured(captured) == 0b01, (
            "an arata conversion can only ever flip the tier-1 enemy"
        )
        plane = piece_idx * 2 + bsel
        if move.get("draftFinished"):
            return C_START + plane * 81 + to
        return B_START + plane * 81 + to

    from_sq = sq_index(move["from"])
    to_sq = sq_index(move["to"])

    if mtype in ("route", "capture"):
        return A0_START + from_sq * 81 + to_sq
    else:
        # tsuke or betray → A1, distinguished by the conversion selector.
        # A tsuke converts nothing by definition — its record carries no
        # captured list to inspect.
        sel = 0 if mtype == "tsuke" else sel_from_captured(move.get("captured"))
        return A1_START + (from_sq * 4 + sel) * 81 + to_sq


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _arata_bsel_matches(move: dict, bsel: int) -> bool:
    """True iff *move*'s conversion is the one block B/C selector *bsel* names.

    ``bsel = 0`` selects the plain drop (nothing converted); ``bsel = 1``
    selects the drop that converts the tier-1 enemy, whose tier-mask is always
    ``0b01``.  Works for both record shapes — the verbose dicts of the
    pure-Python oracle and the cymovegen tuples.
    """
    captured = move.get("captured") or ()
    if not captured:
        return bsel == 0
    return bsel == 1 and sel_from_captured(captured) == 0b01


def decode_action(action_id: int, game) -> dict:
    """Decode an action id back to the engine Move from game.moves().

    Returns the matching Move dict (verbose) from the current legal moves.
    Raises ValueError if the action id does not correspond to a legal move.
    """
    legal = game.moves({"verbose": True})

    if action_id < A1_START:
        # Block A0 — route or capture
        idx = action_id - A0_START
        from_idx, to_idx = divmod(idx, 81)
        from_sq = sq_from_index(from_idx)
        to_sq = sq_from_index(to_idx)

        for m in legal:
            if m["type"] in ("route", "capture"):
                if sq_index(m["from"]) == from_idx and sq_index(m["to"]) == to_idx:
                    return m
        raise ValueError(
            f"No route/capture at ({from_sq}, {to_sq}) for action {action_id}"
        )

    elif action_id < B_START:
        # Block A1 — tsuke or betray, keyed on the conversion selector
        idx = action_id - A1_START
        plane, to_idx = divmod(idx, 81)
        from_idx, sel = divmod(plane, 4)
        from_sq = sq_from_index(from_idx)
        to_sq = sq_from_index(to_idx)

        # Exact match on the tier-mask: sel 0 is the tsuke record, every other
        # selector names one specific betray combination.  A selector with no
        # matching record is an illegal action, not a near miss to round up.
        for m in legal:
            mtype = m["type"]
            if mtype not in ("tsuke", "betray"):
                continue
            if sq_index(m["from"]) != from_idx or sq_index(m["to"]) != to_idx:
                continue
            m_sel = 0 if mtype == "tsuke" else sel_from_captured(m.get("captured"))
            if m_sel == sel:
                return m

        raise ValueError(
            f"No tsuke/betray at ({from_sq}, {to_sq}) with selector {sel} "
            f"for action {action_id}"
        )

    elif action_id < C_START:
        # Block B — arata (no draft finish)
        idx = action_id - B_START
        plane, to_idx = divmod(idx, 81)
        piece_idx, bsel = divmod(plane, 2)
        piece_kanji = _INDEX_TO_PIECE[piece_idx]
        to_sq = sq_from_index(to_idx)

        for m in legal:
            if (
                m["type"] == "arata"
                and m["piece"] == piece_kanji
                and sq_index(m["to"]) == to_idx
                and not m.get("draftFinished")
                and _arata_bsel_matches(m, bsel)
            ):
                return m
        raise ValueError(
            f"No arata {piece_kanji} at {to_sq} with selector {bsel} "
            f"for action {action_id}"
        )

    else:
        # Block C — arata + draft-finish
        idx = action_id - C_START
        plane, to_idx = divmod(idx, 81)
        piece_idx, bsel = divmod(plane, 2)
        piece_kanji = _INDEX_TO_PIECE[piece_idx]
        to_sq = sq_from_index(to_idx)

        for m in legal:
            if (
                m["type"] == "arata"
                and m["piece"] == piece_kanji
                and sq_index(m["to"]) == to_idx
                and m.get("draftFinished")
                and _arata_bsel_matches(m, bsel)
            ):
                return m
        raise ValueError(
            f"No arata+終 {piece_kanji} at {to_sq} with selector {bsel} "
            f"for action {action_id}"
        )


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cython move encoding (fast path — no string intermediary)
# ---------------------------------------------------------------------------

# Precomputed (r, f) 0-based -> sq_index table.
# SQUARES order: rank 1→9, file 9→1.  sq_index = r * 9 + (8 - f).
_RF_TO_SQ: list[list[int]] = [
    [r * 9 + (8 - f) for f in range(9)] for r in range(9)
]

# Move-type codes (must match cymovegen MT_* constants).
_MT_ROUTE   = 0
_MT_TSUKE   = 1
_MT_CAPTURE = 2
_MT_BETRAY  = 3
_MT_ARATA   = 4


def encode_cython_move(move: dict) -> int:
    """Encode a cymovegen move-record dict into a flat action id.

    Uses integer coords directly — no string construction.
    """
    move_type = move["move_type"]

    if move_type == _MT_ARATA:
        piece_idx = move["piece_type_index"]
        to_sq = _RF_TO_SQ[move["to_r"]][move["to_f"]]
        captured = move["captured"]
        bsel = 1 if captured else 0
        assert not bsel or sel_from_captured(captured) == 0b01, (
            "an arata conversion can only ever flip the tier-1 enemy"
        )
        plane = piece_idx * 2 + bsel
        if move["draft_finished"]:
            return C_START + plane * 81 + to_sq
        return B_START + plane * 81 + to_sq

    from_sq = _RF_TO_SQ[move["from_r"]][move["from_f"]]
    to_sq = _RF_TO_SQ[move["to_r"]][move["to_f"]]

    if move_type == _MT_ROUTE or move_type == _MT_CAPTURE:
        return A0_START + from_sq * 81 + to_sq
    else:
        # tsuke or betray → A1, distinguished by the conversion selector.
        sel = 0 if move_type == _MT_TSUKE else sel_from_captured(move["captured"])
        return A1_START + (from_sq * 4 + sel) * 81 + to_sq


def decode_cython_action(action_id: int, board) -> dict:
    """Decode an action id to the matching cymovegen move-record dict.

    *board* is a cyboard.Board.  Generates all moves via cymovegen and finds
    the record whose conversion selector matches the id exactly — the same
    rule as the string-based decode_action.
    """
    from komugi.engine.cymovegen import generate_all_moves

    legal = generate_all_moves(board)

    if action_id < A1_START:
        # Block A0 — route or capture
        idx = action_id - A0_START
        from_idx, to_idx = divmod(idx, 81)
        for m in legal:
            mt = m["move_type"]
            if mt == _MT_ROUTE or mt == _MT_CAPTURE:
                if (_RF_TO_SQ[m["from_r"]][m["from_f"]] == from_idx and
                        _RF_TO_SQ[m["to_r"]][m["to_f"]] == to_idx):
                    return m
        raise ValueError(f"No route/capture for action {action_id}")

    elif action_id < B_START:
        # Block A1 — tsuke or betray, keyed on the conversion selector
        idx = action_id - A1_START
        plane, to_idx = divmod(idx, 81)
        from_idx, sel = divmod(plane, 4)
        for m in legal:
            mt = m["move_type"]
            if mt != _MT_TSUKE and mt != _MT_BETRAY:
                continue
            if (_RF_TO_SQ[m["from_r"]][m["from_f"]] != from_idx or
                    _RF_TO_SQ[m["to_r"]][m["to_f"]] != to_idx):
                continue
            m_sel = 0 if mt == _MT_TSUKE else sel_from_captured(m["captured"])
            if m_sel == sel:
                return m
        raise ValueError(
            f"No tsuke/betray with selector {sel} for action {action_id}"
        )

    elif action_id < C_START:
        # Block B — arata (no draft finish)
        idx = action_id - B_START
        plane, to_idx = divmod(idx, 81)
        piece_idx, bsel = divmod(plane, 2)
        for m in legal:
            if (m["move_type"] == _MT_ARATA and
                    m["piece_type_index"] == piece_idx and
                    _RF_TO_SQ[m["to_r"]][m["to_f"]] == to_idx and
                    not m["draft_finished"] and
                    _arata_bsel_matches(m, bsel)):
                return m
        raise ValueError(
            f"No arata with selector {bsel} for action {action_id}"
        )

    else:
        # Block C — arata + draft-finish
        idx = action_id - C_START
        plane, to_idx = divmod(idx, 81)
        piece_idx, bsel = divmod(plane, 2)
        for m in legal:
            if (m["move_type"] == _MT_ARATA and
                    m["piece_type_index"] == piece_idx and
                    _RF_TO_SQ[m["to_r"]][m["to_f"]] == to_idx and
                    m["draft_finished"] and
                    _arata_bsel_matches(m, bsel)):
                return m
        raise ValueError(
            f"No arata+終 with selector {bsel} for action {action_id}"
        )


def legal_cython_action_ids(board) -> set[int]:
    """Return the action ids of all legal cython moves.

    *board* is a cyboard.Board.  The encoding is injective over a move list
    now that the selector distinguishes the betray family (KOM-028), so the
    set has exactly one id per record — asserted here because a residual
    collapse would otherwise show up only as a silently missing action.
    """
    from komugi.engine.cymovegen import generate_all_moves
    moves = generate_all_moves(board)
    ids = {encode_cython_move(m) for m in moves}
    assert len(ids) == len(moves), (
        f"action encoding is not injective at this position: {len(moves)} "
        f"moves collapsed onto {len(ids)} ids"
    )
    return ids


# ---------------------------------------------------------------------------
# Legacy string-based legal_action_ids (pure-Python oracle path)
# ---------------------------------------------------------------------------

def legal_action_ids(game) -> set[int]:
    """Return the set of action ids for all legal moves.

    One id per legal move: the conversion selector gives every betray
    combination — and the plain tsuke that coexists with it — its own id.
    """
    return {encode_move(m) for m in game.moves({"verbose": True})}
