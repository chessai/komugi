"""v1 <-> v2 action-id shims — TEST-ONLY, delete after the policy head freezes.

KOM-028 widened the action space from 15390 ids to 37341 by giving every
betrayal subset (and the tsuke that declines it) an id of its own.  The frozen
v1 golden corpora are indexed by the OLD ids, and re-recording them would throw
away the very oracle the migration is checked against.  The two functions here
are what let them survive unchanged, and they are the only place v1 ids are
still spoken (docs/design/betrayal-encoding.md, "How old goldens survive"):

* ``pi`` — the projection v2 -> v1.  Pure arithmetic, no board: it simply
  forgets the selector.  For mask goldens the survival property is an
  *equality*, not an approximation:
  ``{pi(x) for x in mask_v2(P)} == mask_v1_golden(P)`` at every frozen
  position P.  ("subset" holds because every v2 bit comes from a branch that
  fired in v1 too; "superset" because wherever v1's A1/B/C bit fired, v2 emits
  at least the sel = 0 / bsel = 0 id — the plain tsuke and the plain drop are
  always legal where the branch fires.)

* ``rho`` — the state-dependent lift v1 -> v2.  It reproduces v1's "maximal
  conversion" decode rule exactly: the full qualifying set Q is the unique
  longest combination, so ``max(betrays, key=len(captured))`` selected it, and
  ``Q = 0`` degraded to the plain tsuke.  Decode goldens are re-asserted as
  ``decode_v2(rho(P, id_v1), P) == golden_move``, and ``pi(rho(P, id)) == id``
  on every v1-reachable id.

  THE CONVERSION IS TACTICIAN-ONLY, IN BOTH ARMS.  v1's decoder could only
  return a conversion where the rules engine had generated one, and only the
  tactician (謀, ``PIECE_INDEX[TACTICIAN] == 13``) ever converts: cymovegen
  emits betray records solely for a 謀 mover (``_fill_moves_for_square``) and
  the drop conversion solely for a 謀 drop (``_fill_arata``).  So rho keys on
  the ACTING PIECE, never on the target tower alone —

    * A1: ``sel = Q(to_sq)`` only when the mover's own 謀 is on top of
      *from_sq*; every other piece's stacking move lifts to the plain tsuke,
      ``sel = 0``, which is what v1 decoded for it.
    * B/C: ``bsel = 1`` only when the DROPPED piece is the 謀 and ``Q != 0``;
      every other drop lifts with ``bsel = 0``.

  Keying on the target square alone (the shape this module shipped with)
  lifted e.g. a plain 兵 drop onto a qualifying tower to a ``bsel = 1`` id
  that the legal mask never sets — see the reproducing tests in
  tests/engine/test_action_encoding.py.

  SPEC CONTRADICTION, RECORDED NOT RESOLVED.  The T7(b) decode-golden
  semantics and the Migration rule disagree about what a v1 id over a
  qualifying tower meant; this module implements what the v1 *decoder* did.
  The contradiction is tracked on the KOM-028 ticket and is deliberately left
  open here.

Nothing on the self-play hot path imports this module; it is import-fenced to
the tests on purpose.  There is deliberately no lift for a *distribution*: a
v1 probability mass on an A1 id would have to be split across four v2 ids with
no evidence for the split, so replay buffers are rewritten, not lifted.
"""

from __future__ import annotations

from komugi.engine.action_encoding import (
    A1_START,
    B_START,
    C_START,
    ACTION_SIZE,
    PIECE_INDEX,
    qualifying_mask,
)
from komugi.engine.utils import TACTICIAN

#: The one piece that converts.  Betray records exist for a 謀 mover only, and
#: arata conversions for a 謀 drop only, so this index gates both rho arms.
_TACTICIAN_INDEX: int = PIECE_INDEX[TACTICIAN]

# ---------------------------------------------------------------------------
# The frozen v1 layout (15390 = 190 planes * 81 squares).  These numbers are
# dead in the engine now and live on only to interpret the v1 goldens.
# ---------------------------------------------------------------------------

V1_ACTION_SIZE: int = 15390

V1_A0_START: int = 0
V1_A1_START: int = 6561
V1_B_START: int = 13122
V1_C_START: int = 14256


def _rank_file(sq: int) -> tuple[int, int]:
    """Inverse of ``sq_index``: 0..80 -> 1-based (rank, file)."""
    return sq // 9 + 1, 9 - (sq % 9)


def _as_board(position):
    """Accept either a cyboard.Board or the FEN string of one."""
    if isinstance(position, str):
        from komugi.engine.cyboard import board_from_fen

        return board_from_fen(position)
    return position


def _top_piece(board, sq: int) -> tuple[int, int] | None:
    """``(type_index, color)`` of the piece on top of the tower at *sq*.

    ``None`` for an empty square.  Same code layout as ``qualifying_mask``:
    a cell holds ``2 * type_index + color + 1``, zero meaning empty.
    """
    rank, file = _rank_file(sq)
    cells = board.cells_arr
    for t in range(cells.shape[2] - 1, -1, -1):
        code = int(cells[rank - 1, file - 1, t])
        if code:
            return (code - 1) >> 1, (code - 1) & 1
    return None


def pi(v2_id: int) -> int:
    """Project a v2 action id onto the v1 id it collapses to.

    Total and pure — the selector is simply dropped:

        A0: from_sq*81 + to_sq                   -> 0     + from_sq*81 + to_sq
        A1: 6561  + (from_sq*4 + sel)*81 + to_sq -> 6561  + from_sq*81 + to_sq
        B:  32805 + (pi*2 + bsel)*81     + to_sq -> 13122 + pi*81      + to_sq
        C:  35073 + (pi*2 + bsel)*81     + to_sq -> 14256 + pi*81      + to_sq
    """
    if not 0 <= v2_id < ACTION_SIZE:
        raise ValueError(f"v2 action id out of range: {v2_id}")

    if v2_id < A1_START:
        return v2_id  # block A0 is unchanged between the layouts

    if v2_id < B_START:
        plane, to_sq = divmod(v2_id - A1_START, 81)
        from_sq = plane // 4
        return V1_A1_START + from_sq * 81 + to_sq

    if v2_id < C_START:
        plane, to_sq = divmod(v2_id - B_START, 81)
        return V1_B_START + (plane // 2) * 81 + to_sq

    plane, to_sq = divmod(v2_id - C_START, 81)
    return V1_C_START + (plane // 2) * 81 + to_sq


def rho(position, v1_id: int) -> int:
    """Lift a v1 action id to the v2 id that means the same move at *position*.

    *position* is a cyboard.Board (or the FEN of one) — the state matters,
    because v1's decoder resolved its ambiguity against the board.  Identity on
    A0.  On A1 the selector becomes the full qualifying mask Q of the target
    tower, but only when the mover's own 謀 is the piece doing the stacking;
    on B/C the bit is set only for a 謀 drop onto a tower with Q non-empty.
    Every other piece converts nothing in v1 or v2, so it lifts with selector
    0 (see the module docstring).
    """
    if not 0 <= v1_id < V1_ACTION_SIZE:
        raise ValueError(f"v1 action id out of range: {v1_id}")

    board = _as_board(position)
    color = board.turn

    if v1_id < V1_A1_START:
        return v1_id

    if v1_id < V1_B_START:
        from_sq, to_sq = divmod(v1_id - V1_A1_START, 81)
        # Only a 謀 on top of from_sq could have betrayed anything, so only
        # its lift carries the target's qualifying mask; for anything else v1
        # decoded the plain tsuke (review finding 2).
        sel = 0
        if _top_piece(board, from_sq) == (_TACTICIAN_INDEX, color):
            rank, file = _rank_file(to_sq)
            # Clamp to the two selector bits (review finding 4).  Q is a mask
            # over ALL tiers, so a qualifying enemy at tier 3 would set bit 2
            # and bleed the plane index into the NEXT from-square's four
            # planes, breaking pi(rho(id)) == id.  Such a tower is never a
            # legal stack target (it is at max_tier), so the bit is noise.
            sel = qualifying_mask(board, rank, file, color) & 0b11
        return A1_START + (from_sq * 4 + sel) * 81 + to_sq

    if v1_id < V1_C_START:
        piece_idx, to_sq = divmod(v1_id - V1_B_START, 81)
        block_start = B_START
    else:
        piece_idx, to_sq = divmod(v1_id - V1_C_START, 81)
        block_start = C_START

    # A drop converts only when the DROPPED piece is the 謀 (review finding 1):
    # cymovegen's _fill_arata gates the bit on ``piece_ti == TACTICIAN``, so a
    # 兵 drop onto a qualifying tower has no bsel = 1 id in the legal mask.  A
    # legal drop target is empty or friendly on top, so a non-empty Q is always
    # the tier-1 bit — the one enemy a drop can flip.
    bsel = 0
    if piece_idx == _TACTICIAN_INDEX:
        rank, file = _rank_file(to_sq)
        bsel = 1 if qualifying_mask(board, rank, file, color) else 0
    return block_start + (piece_idx * 2 + bsel) * 81 + to_sq
