"""
AlphaZero observation encoding for Gungi.

Encodes a Gungi game state as a float32 numpy array of shape (128, 9, 9).

Plane layout (128 planes total — the map pre-registered by
``docs/design/kom-002-input-planes-v2.md`` §2; ``PLANE_MAP_VERSION = 3``, which
is KOM-002's map with planes 115/120 re-based on KOM-165's raised cap):
  0-83     Occupancy: piece_type(14) x tier(3) x color(2)
           plane = piece_idx * 6 + (tier - 1) * 2 + color_idx
           1.0 at (rank, file) where that (type, tier, color) piece sits.
  84-111   Hand: piece_type(14) x color(2)
           plane = 84 + piece_idx * 2 + color_idx
           Constant plane = count / MAX_COUNT for that piece type.
  112      Drafting rights — white: 1.0 if white is drafting, else 0.0.
  113      Drafting rights — black: 1.0 if black is drafting, else 0.0.
  114      Side to move: 0.0 = white, 1.0 = black.
  115      Turn clock: min(plies, MAX_TURNS) / MAX_TURNS — PLIES, the same unit
           the turn-cap draw is adjudicated in, so the plane reaches 1.0
           exactly at the drawing ply (docs/design/rules-audit.md F7).
  116      R2: 1.0 iff the current position has occurred >= 2 times.
  117      R3: 1.0 iff the current position has occurred >= 3 times.
  118      Last move — ORIGIN: from_tier / 3.0 at the origin square, 0
           elsewhere.  All-zero when there is no origin (an arata) or no last
           move at all.
  119      Last move — DESTINATION: to_tier / 3.0 at the destination square, 0
           elsewhere.  All-zero only when there is no last move.
  120      Progress clock: min(progress_plies, MAX_TURNS) / MAX_TURNS, where a
           progress ply is one since the last capture, betrayal or arata.  The
           denominator is plane 115's, so this plane is <= plane 115 always.
  121-127  Reserved — identically 0.0.  Filling one bumps PLANE_MAP_VERSION
           without changing the observation's shape, which is why the version
           is written into shards and checkpoints (design §4).

The two monotone repetition planes replace v1's single scalar plane 116
(min(occurrences, 4) / 4), whose values were computed over the aliased
cells-only key — see docs/design/rules-audit.md, "Encoding-plane
consequences".  Occurrence 4 needs no plane (the game is over) and occurrence
1 needs none (it is the constant-true case).

Plane-map growth is STRICTLY APPEND-ONLY: planes 0-117 keep their index,
their source and their value bit for bit (design §2's invariant P1).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from komugi.engine.api import Gungi
from komugi.engine.cyboard import Board, split_code
from komugi.engine.utils import (
    MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
    WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
    CANNON, ARCHER, MUSKETEER, TACTICIAN,
    WHITE, BLACK,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The engine's turn cap, in PLIES — the unit the turn-cap draw is adjudicated
#: in, not moves.  THE single source of the cap: ``Game.max_turns()``,
#: :data:`komugi.selfplay.records.MLH_SCALE`,
#: :data:`komugi.campaign.game_stats.TURN_CAP_PLIES` and the book's
#: adjudication budget all derive from this line, and planes 115 and 120 are
#: normalised by it.  Raised 256 → 512 by KOM-165.
MAX_TURNS: int = 512

#: Plane-map version.  0 = the alphazero-general reference's 117-plane map,
#: 1 = KOM-027's 118-plane map (R2/R3 replacing v1's scalar repetition plane),
#: 2 = KOM-002's map (last move, progress clock, reserved block),
#: 3 = that same map with planes 115 and 120 re-based on the raised
#: :data:`MAX_TURNS` (KOM-165).  No plane changed index or source and no plane
#: changed MEANING — 115 is still "fraction of the cap elapsed" and 120 still
#: shares its denominator — but the cap moved, so every value of those two
#: planes is a pure rescale of a version-2 value and shards, books and
#: checkpoints from either side must not be mixed.
#:
#: ANY change to the MEANING of any plane bumps this — including filling one of
#: the reserved planes 121-127, which the shard fingerprint provably cannot see
#: because it is a pure function of field names, dtypes and shapes.
PLANE_MAP_VERSION: Final[int] = 3

# Canonical piece ordering — must match the action encoding's arata block.
PIECE_TYPES: tuple[str, ...] = (
    MARSHAL, GENERAL, LIEUTENANT_GENERAL, MAJOR_GENERAL,
    WARRIOR, LANCER, RIDER, SPY, FORTRESS, SOLDIER,
    CANNON, ARCHER, MUSKETEER, TACTICIAN,
)

# kanji -> 0..13 index
PIECE_INDEX: dict[str, int] = {p: i for i, p in enumerate(PIECE_TYPES)}

# Max hand count per piece type (advanced-init counts).
MAX_COUNT: dict[str, int] = {
    MARSHAL: 1,
    GENERAL: 1,
    LIEUTENANT_GENERAL: 1,
    MAJOR_GENERAL: 2,
    WARRIOR: 2,
    LANCER: 3,
    RIDER: 2,
    SPY: 2,
    FORTRESS: 2,
    SOLDIER: 4,
    CANNON: 1,
    ARCHER: 2,
    MUSKETEER: 1,
    TACTICIAN: 1,
}

COLOR_INDEX: dict[str, int] = {WHITE: 0, BLACK: 1}

_NUM_PIECE_TYPES = 14
_NUM_TIERS = 3
_NUM_COLORS = 2
_OCCUPANCY_PLANES = _NUM_PIECE_TYPES * _NUM_TIERS * _NUM_COLORS  # 84
_HAND_PLANES = _NUM_PIECE_TYPES * _NUM_COLORS                    # 28
_DRAFT_PLANES = 2
_SIDE_TO_MOVE_PLANES = 1
_TURN_CLOCK_PLANES = 1
_REPETITION_PLANES = 2
_LAST_MOVE_PLANES = 2
_PROGRESS_PLANES = 1
#: Padding to N = 128.  Zero today; a future plane consumes one of these at the
#: cost of a PLANE_MAP_VERSION bump and no shape change (design §4).
_RESERVED_PLANES = 7
_TOTAL_PLANES = (
    _OCCUPANCY_PLANES        # 84
    + _HAND_PLANES           # 28
    + _DRAFT_PLANES          # 2
    + _SIDE_TO_MOVE_PLANES   # 1
    + _TURN_CLOCK_PLANES     # 1
    + _REPETITION_PLANES     # 2
    + _LAST_MOVE_PLANES      # 2
    + _PROGRESS_PLANES       # 1
    + _RESERVED_PLANES       # 7
)  # 128

# Index of each appended block, named so the encoders never spell a literal.
_PLANE_LAST_MOVE_FROM = 118
_PLANE_LAST_MOVE_TO = 119
_PLANE_PROGRESS = 120
_PLANE_RESERVED_START = 121

#: Tiers run 1..3, so a tier is encoded as ``tier / _MAX_TIER`` ∈ {1/3, 2/3, 1}.
_MAX_TIER = float(_NUM_TIERS)

_BOARD_H = 9
_BOARD_W = 9


def _plies_from_move_number(move_number: int, turn: int) -> int:
    """Plies elapsed at *move_number* with *turn* (0=white, 1=black) to move.

    ``move_number`` counts FULL moves and advances on black's ply, so a
    position reached after n plies reads ``move_number = n // 2 + 1`` with
    ``turn = n % 2`` — inverted here.  Only the standalone encoders need this:
    they are handed a position rather than a game, while ``Game.observation``
    passes its own ply counter (the one the turn cap is adjudicated on).
    """
    return max(0, 2 * (move_number - 1) + turn)


def observation_size() -> tuple[int, int, int]:
    """Return the observation tensor shape: (planes, height, width)."""
    return (_TOTAL_PLANES, _BOARD_H, _BOARD_W)


def observation(game: Gungi) -> np.ndarray:
    """Encode *game* as a float32 array of shape (128, 9, 9).

    Oracle path: the repetition count is taken over ``api.Gungi``'s own
    placement-field key, which keeps the reference engine's aliasing defect on
    purpose (see ``game.py``'s module docstring).  A ``Gungi`` loaded from a
    FEN carries no history, so the count is 1 and both threshold planes are 0
    — which is exactly what the cyboard fast path produces for the same FEN.

    Planes 118-127 are emitted as ZEROS, permanently and by design (KOM-002
    §8 gap 4): an ``api.Gungi`` has FEN-string history but no decoded move
    record and no progress counter, and re-deriving them from FEN diffs would
    be a second implementation of ``apply_move`` that nothing keeps honest.
    The differential test against the fast path stays meaningful because it
    only ever compares POSITIONS, for which the fast path emits zeros too.
    """
    obs = np.zeros((_TOTAL_PLANES, _BOARD_H, _BOARD_W), dtype=np.float32)

    board = game.board()

    # --- Occupancy planes (0..83) -------------------------------------------
    for rank_idx in range(_BOARD_H):          # 0..8
        for file_idx in range(_BOARD_W):      # 0..8
            tower = board[rank_idx][file_idx]
            if tower[0] is None:
                continue
            for piece in tower:
                pi = PIECE_INDEX.get(piece["type"])
                if pi is None:
                    continue
                ci = COLOR_INDEX[piece["color"]]
                tier = piece["tier"]  # 1..3
                plane = pi * 6 + (tier - 1) * 2 + ci
                obs[plane, rank_idx, file_idx] = 1.0

    # --- Hand planes (84..111) ----------------------------------------------
    for hp in game.hand():
        pi = PIECE_INDEX.get(hp["type"])
        if pi is None:
            continue
        ci = COLOR_INDEX[hp["color"]]
        mc = MAX_COUNT.get(hp["type"], 1)
        val = hp["count"] / mc
        plane = _OCCUPANCY_PLANES + pi * 2 + ci
        obs[plane, :, :] = val

    # --- Drafting-rights planes (112, 113) ----------------------------------
    drafting = game.get_drafting_rights()
    if drafting[WHITE]:
        obs[112, :, :] = 1.0
    if drafting[BLACK]:
        obs[113, :, :] = 1.0

    # --- Side-to-move plane (114) -------------------------------------------
    if game.turn() == BLACK:
        obs[114, :, :] = 1.0

    # --- Turn-clock plane (115) ---------------------------------------------
    plies = _plies_from_move_number(game.move_number(), COLOR_INDEX[game.turn()])
    obs[115, :, :] = min(plies, MAX_TURNS) / MAX_TURNS

    # --- Repetition threshold planes (116, 117) -----------------------------
    current_key = game.fen().split(" ")[0]
    count = 1  # current position counts as 1
    for mv in game._history:
        if mv["before"].split(" ")[0] == current_key:
            count += 1
    obs[116, :, :] = 1.0 if count >= 2 else 0.0
    obs[117, :, :] = 1.0 if count >= 3 else 0.0

    # --- Last move (118, 119), progress clock (120), reserved (121-127) -----
    # All zero on this path — see the docstring.

    return obs


# ---------------------------------------------------------------------------
# Fast-path: observation directly from a cyboard.Board
# ---------------------------------------------------------------------------

# Max hand counts indexed by type_index (0..13), matching MAX_COUNT above.
_MAX_COUNT_BY_IDX: tuple[int, ...] = tuple(MAX_COUNT[p] for p in PIECE_TYPES)


def observation_from_cyboard(
    board,
    repetition_count: int | None = None,
    plies: int | None = None,
    last_move: tuple[int, int, int, int] | None = None,
    progress_plies: int = 0,
) -> np.ndarray:
    """Encode a cyboard.Board as a float32 array of shape (128, 9, 9).

    Produces output identical to :func:`observation` but reads typed arrays
    directly — no pure-Python engine needed.

    Every history-dependent quantity is a PARAMETER with a "bare position"
    default; none of them is re-derived here, because a bare ``Board`` cannot
    know any of them:

    * *repetition_count* — the current position's occurrence count, which
      ``Game`` reads off its O(1) counter (``Game.repetition_count``); the
      encoder never scans a history list.  Defaults to a first occurrence.
    * *plies* — the ply counter the turn cap is adjudicated on
      (``Game._turns``).  Defaults to the count derived from the board's move
      number.
    * *last_move* — ``(from_sq, from_tier, to_sq, to_tier)`` as ``Game``
      records it in ``play_action``: 0..80 square indices in the action
      encoding's own ``r * 9 + (8 - f)`` numbering, 1-based tiers, and
      ``from_sq = -1`` / ``from_tier = 0`` for an arata, which has no origin.
      ``None`` means "no last move" — game start, or a position loaded from a
      FEN, which carries no move history.
    * *progress_plies* — plies since the last capture, betrayal or arata.
      Defaults to 0, which is also what a FEN implies: the six FEN fields
      carry no no-progress counter.
    """
    obs = np.zeros((_TOTAL_PLANES, _BOARD_H, _BOARD_W), dtype=np.float32)

    cells = board.cells_arr  # int8 (9, 9, 3)
    hand = board.hand_arr    # int32 (14, 2)

    # --- Occupancy planes (0..83) -------------------------------------------
    for r in range(_BOARD_H):
        for f in range(_BOARD_W):
            j = 8 - f  # file_idx in observation space (matches pure-Python board layout)
            for t in range(3):
                code = int(cells[r, f, t])
                if code == 0:
                    continue
                ti = (code - 1) >> 1
                ci = (code - 1) & 1
                tier = t + 1  # 1-based
                plane = ti * 6 + (tier - 1) * 2 + ci
                obs[plane, r, j] = 1.0

    # --- Hand planes (84..111) ----------------------------------------------
    for ti in range(14):
        for ci in range(2):
            cnt = int(hand[ti, ci])
            if cnt <= 0:
                continue
            mc = _MAX_COUNT_BY_IDX[ti]
            val = cnt / mc
            plane = _OCCUPANCY_PLANES + ti * 2 + ci
            obs[plane, :, :] = val

    # --- Drafting-rights planes (112, 113) ----------------------------------
    if board.draft_w:
        obs[112, :, :] = 1.0
    if board.draft_b:
        obs[113, :, :] = 1.0

    # --- Side-to-move plane (114) -------------------------------------------
    if board.turn == 1:  # black
        obs[114, :, :] = 1.0

    # --- Turn-clock plane (115) ---------------------------------------------
    if plies is None:
        plies = _plies_from_move_number(board.move_number, board.turn)
    obs[115, :, :] = min(plies, MAX_TURNS) / MAX_TURNS

    # --- Repetition threshold planes (116, 117) -----------------------------
    count = 1 if repetition_count is None else repetition_count
    obs[116, :, :] = 1.0 if count >= 2 else 0.0
    obs[117, :, :] = 1.0 if count >= 3 else 0.0

    # --- Last-move planes (118, 119) ----------------------------------------
    # ``s = r * 9 + (8 - f)`` is BOTH the action encoding's square index and
    # the row-major flat index of an observation plane under the ``j = 8 - f``
    # convention above, so the squares need no translation here.
    if last_move is not None:
        from_sq, from_tier, to_sq, to_tier = last_move
        if from_sq >= 0:
            obs[_PLANE_LAST_MOVE_FROM].reshape(81)[from_sq] = from_tier / _MAX_TIER
        obs[_PLANE_LAST_MOVE_TO].reshape(81)[to_sq] = to_tier / _MAX_TIER

    # --- Progress-clock plane (120) -----------------------------------------
    obs[_PLANE_PROGRESS, :, :] = min(progress_plies, MAX_TURNS) / MAX_TURNS

    # --- Reserved planes (121..127) -----------------------------------------
    # Left at their allocation value, 0.0.

    return obs


# ---------------------------------------------------------------------------
# Final-control map (KOM-010)
# ---------------------------------------------------------------------------

#: "White controls this square" -- the top of its tower is a white piece.
CONTROL_WHITE: Final[int] = 0

#: "Black controls this square" -- the top of its tower is a black piece.
CONTROL_BLACK: Final[int] = 1

#: "Nobody controls this square" -- the tower is empty.  Deliberately a third
#: value rather than a second plane: the map is one ``int8`` per square, and
#: "empty" is not "white's" (which ``0`` would make it if empties defaulted).
CONTROL_EMPTY: Final[int] = 2

#: Squares in a control map -- 81, the board flattened row-major.
CONTROL_SIZE: Final[int] = _BOARD_H * _BOARD_W

#: The two colour values, indexed the way the engine indexes colours
#: (``cyboard.split_code``'s second component, ``0`` white / ``1`` black), so
#: the map's colour half needs no translation at any consumer.
_CONTROL_BY_COLOR: Final[tuple[int, int]] = (CONTROL_WHITE, CONTROL_BLACK)


def final_control(board: Board) -> np.ndarray:
    """Return who owns the top of every tower, as ``(81,)`` ``int8``.

    The **absolute** ownership map KOM-010 extracts at game end, and the target
    behind KOM-005's ownership head: ``CONTROL_WHITE`` (0) where the topmost
    piece of the tower is white, ``CONTROL_BLACK`` (1) where it is black, and
    ``CONTROL_EMPTY`` (2) where there is no tower at all.  Only the top matters
    -- a tower is controlled by whoever stands on it, which is why a betrayal
    or a capture that changes the top flips the square without touching the
    pieces underneath.  ``Board.height`` is therefore not consulted:
    ``Board.top`` already returns the highest occupied tier's code (``0`` when
    the square is empty), and reading the height as well would be a second way
    to spell the same question.

    **Orientation.** Squares are indexed exactly as the observation planes are
    when flattened row-major, i.e. ``s = r * 9 + (8 - f)`` with ``r = rank - 1``
    and ``f`` the 0-based file of ``cyboard.Board.cells_arr``'s second axis --
    the convention pre-registered by ``docs/design/kom-002-input-planes-v2.md``
    §2 ("Orientation convention -- stated once, binding for every spatial
    plane") and used by :func:`observation_from_cyboard` above, by the action
    encoding's ``_RF_TO_SQ`` and by the policy head.  There is deliberately no
    second convention here: ``final_control(board).reshape(9, 9)`` lines up
    cell for cell with ``observation_from_cyboard(board)[p]`` for every plane
    ``p``.

    Pure Python over the typed board on purpose: this runs **once per game**,
    at adjudication, not once per move, so the ~81 ``Board.top`` calls it costs
    are invisible next to a single search.

    Args:
        board: A :class:`komugi.engine.cyboard.Board` -- the final position,
            when the caller is the recorder.  Any position is legal input; the
            map is a pure function of the cells.

    Returns:
        ``(81,)`` ``int8``, values in ``{CONTROL_WHITE, CONTROL_BLACK,
        CONTROL_EMPTY}``.
    """
    control = np.full(CONTROL_SIZE, CONTROL_EMPTY, dtype=np.int8)
    for r in range(_BOARD_H):
        for f in range(_BOARD_W):
            code = board.top(r, f)
            if code == 0:
                continue
            _, color_index = split_code(code)
            control[r * _BOARD_W + (8 - f)] = _CONTROL_BY_COLOR[color_index]
    return control


# ---------------------------------------------------------------------------
# Material count (KOM-245)
# ---------------------------------------------------------------------------


def material_counts(board: Board) -> tuple[int, int]:
    """Return ``(white_total, black_total)`` -- pieces each side still holds.

    **TOTAL** material: every piece standing on the board, at ANY of the three
    tiers, plus every piece still in that side's hand.  The count KOM-245's
    soft adjudication of a turn-capped game is computed from, and it is a total
    rather than a board presence on purpose: a drop moves a piece hand -> board
    and the draft does the same, so a board-only count would make drafting look
    like a material gain and the label would decree what must be learned from
    outcomes instead (design §0/§1).

    **What moves it.** A capture removes the victim's piece from the game
    (``cyengine`` never returns it to a hand), and a betrayal converts ``k`` of
    the victim's pieces while debiting ``k`` from the mover's own hand -- so the
    mover's total is *unchanged* by a betrayal and the victim's falls by ``k``
    (review F7).  Per-side material is therefore monotone non-increasing, and
    hand pieces are uncapturable.

    Read exactly the way :func:`observation_from_cyboard` reads the same two
    arrays, so the count and the planes cannot disagree about a position:
    ``cells_arr`` is ``int8 (9, 9, 3)`` holding ``code = 1 + 2*type + colour``
    (``0`` empty), so the colour of a piece is ``(code - 1) & 1``; ``hand_arr``
    is ``int32 (14, 2)`` indexed ``[type, colour]``.  Orientation is irrelevant
    here -- this is a count over cells, not a map, so no square convention is
    involved.

    Pure Python/NumPy over the typed board, like :func:`final_control` beside
    it and for the same reason: the worker asks **once per game**, of the final
    board, not once per move.

    Args:
        board: A :class:`komugi.engine.cyboard.Board`.  Any position is legal
            input; the count is a pure function of the cells and the hands.

    Returns:
        ``(white_total, black_total)`` as plain ``int`` s -- the advanced start
        is ``(25, 25)``.
    """
    cells = np.asarray(board.cells_arr)
    occupied = cells[cells != 0].astype(np.int64)
    # `(code - 1) & 1` is the colour half of `split_code`, spelled the way the
    # occupancy planes above spell it.
    colour = (occupied - 1) & 1
    hand = np.asarray(board.hand_arr, dtype=np.int64)
    white = int(np.count_nonzero(colour == 0)) + int(hand[:, 0].sum())
    black = int(np.count_nonzero(colour == 1)) + int(hand[:, 1].sum())
    return white, black
