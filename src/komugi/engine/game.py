"""
AlphaZero GameState wrapper for Gungi.

Uses the fast Cython engine (cyboard.Board + cymovegen + cyengine) for all
game-play operations.  The pure-Python gungi.api.Gungi is used only as a
differential oracle in tests.

INTENTIONAL DIVERGENCE: this training engine implements the repaired
repetition rule of ``docs/design/rules-audit.md`` (274-byte key including side
to move, draft flags and hands; draw at ``count >= 4`` on the *current* key),
while ``komugi/engine/api.py`` deliberately keeps the reference's cells-only
aliasing defect because it mirrors gungi.js and is the differential oracle.

The divergence is now MEASURABLE as well as documented (KOM-141): a ``Game``
takes a ``repetition_rule=`` keyword, and ``RepetitionRule.PLACEMENT`` builds
gungi.js's cells-only key so a referee can score a game under the site's rule.
``RepetitionRule.FULL`` is the default here and everywhere else in the repo --
self-play, the arena and the bridge all default to it, and
``tests/engine/test_repetition_rule_default.py`` is what keeps that true. The
rule changes only which bytes the occurrence counter is keyed on; no
adjudication predicate, threshold or observation plane depends on it.
"""

from __future__ import annotations

import os
from typing import Final, List, Tuple

import numpy as np

from komugi.engine.game_base import GameState
from komugi.engine import action_encoding, observation
from komugi.engine.cyboard import Board, board_from_fen, board_to_fen
from komugi.engine import cyengine
from komugi.engine import cymovegen
from komugi.engine.fen import (
    INTRO_POSITION,
    BEGINNER_POSITION,
    INTERMEDIATE_POSITION,
    ADVANCED_POSITION,
)
# Re-exported (the redundant alias is the explicit-reexport spelling) so a
# caller that already imports Game can name the rule without a second import.
from komugi.engine.repetition import RepetitionRule as RepetitionRule
from komugi.engine.repetition import (
    FULL_KEY_BYTES,
    PLACEMENT_KEY_BYTES,
    key_bytes_for,
)


# ---------------------------------------------------------------------------
# Square mirroring helper (left-right file flip)
# ---------------------------------------------------------------------------

def _mirror_sq(sq_idx: int) -> int:
    """Mirror a square index by flipping the file: file f -> 10-f."""
    rank_base = (sq_idx // 9) * 9
    file_off = sq_idx % 9
    return rank_base + (8 - file_off)


def _sq_index(r: int, f: int) -> int:
    """0-based ``(rank, file)`` -> the 0..80 square index of the action encoding.

    ``r * 9 + (8 - f)`` is exactly ``action_encoding._RF_TO_SQ[r][f]`` and, under
    the encoder's ``j = 8 - f`` orientation convention, also the ROW-MAJOR FLAT
    INDEX of an observation-plane cell.  The two coincide, which is what lets
    the last-move planes be addressed by action-encoding square ids with no
    second convention to keep honest.
    """
    return r * 9 + (8 - f)


# Move-type codes; these must match cymovegen's MT_* constants, exactly as
# ``action_encoding``'s own copy does — they live in a .pyx and have no
# python-level name to import.
_MT_ROUTE = 0
_MT_TSUKE = 1
_MT_CAPTURE = 2
_MT_BETRAY = 3
_MT_ARATA = 4

#: The move types that RESET the progress clock.  Equivalently (and this is the
#: definition the tests check against): exactly the moves that strictly decrease
#: the potential ``Φ(b) = 2 * hand_arr.sum() + count_nonzero(cells_arr)`` — a
#: capture removes enemy pieces from the board, a betrayal debits the mover's
#: hand per converted piece, and an arata spends a hand piece.  A route and a
#: tsuke move a piece without changing either term, so Φ is unchanged and the
#: position is reversible.  See docs/design/kom-002-input-planes-v2.md §3.3.
_PROGRESS_MOVE_TYPES = frozenset({_MT_CAPTURE, _MT_BETRAY, _MT_ARATA})


def _mirror_action(action_id: int) -> int:
    """Mirror an action id under the left-right file reflection.

    Only the squares move.  A conversion selector names TIERS of the target
    tower (A1's ``sel``, B/C's ``bsel``), and a file reflection permutes
    neither the tiers nor the piece types, so the selector is carried across
    bit-identical — mirroring it would silently corrupt symmetry-augmented
    training targets.
    """
    if action_id < action_encoding.A1_START:
        # Block A0 — plane is from_sq
        idx = action_id - action_encoding.A0_START
        from_sq, to_sq = divmod(idx, 81)
        return action_encoding.A0_START + _mirror_sq(from_sq) * 81 + _mirror_sq(to_sq)
    elif action_id < action_encoding.B_START:
        # Block A1 — plane is from_sq * 4 + sel
        idx = action_id - action_encoding.A1_START
        plane, to_sq = divmod(idx, 81)
        from_sq, sel = divmod(plane, 4)
        return (
            action_encoding.A1_START
            + (_mirror_sq(from_sq) * 4 + sel) * 81
            + _mirror_sq(to_sq)
        )
    elif action_id < action_encoding.C_START:
        # Block B — plane is piece_idx * 2 + bsel
        idx = action_id - action_encoding.B_START
        plane, to_sq = divmod(idx, 81)
        return action_encoding.B_START + plane * 81 + _mirror_sq(to_sq)
    else:
        # Block C — plane is piece_idx * 2 + bsel
        idx = action_id - action_encoding.C_START
        plane, to_sq = divmod(idx, 81)
        return action_encoding.C_START + plane * 81 + _mirror_sq(to_sq)


def _mirror_last_move(
    last_move: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """File-reflect a ``(from_sq, from_tier, to_sq, to_tier)`` record.

    Only the squares move: a left-right reflection permutes files, and the tiers
    of a tower are stacked along an axis it does not touch.  ``from_sq == -1``
    (an arata, which has no origin) is carried through as-is — it is not a
    square and has no mirror.
    """
    if last_move is None:
        return None
    from_sq, from_tier, to_sq, to_tier = last_move
    return (
        _mirror_sq(from_sq) if from_sq >= 0 else -1,
        from_tier,
        _mirror_sq(to_sq),
        to_tier,
    )


def _mirror_pi(pi: np.ndarray) -> np.ndarray:
    """Mirror a full policy vector under left-right file reflection."""
    mirrored = np.zeros_like(pi)
    nonzero_ids = np.nonzero(pi)[0]
    for aid in nonzero_ids:
        mirrored[_mirror_action(int(aid))] = pi[aid]
    return mirrored


# ---------------------------------------------------------------------------
# Draft-curriculum mask helpers (pure functions — no Game/engine semantics)
# ---------------------------------------------------------------------------
#
# These exist to support a self-play-only root mask that discourages an
# extremely early 終 (draft-finish).  They are deliberately side-effect free:
# nothing here mutates a Board, a mask, or any Game/valid_moves/win_state
# behaviour.  A caller that never invokes them observes an unchanged engine.


def draft_placements(board: Board, color: int) -> int:
    """Number of pieces *color* has placed during its draft.

    Counted directly off the board: every piece code of that color across ALL
    THREE TIERS of ``board.cells_arr`` (a code's color is ``(code - 1) & 1``,
    matching cyboard's ``piece_type_index * 2 + color + 1`` encoding).  During
    the draft this equals the number of placements that color has made — each
    draft move is exactly one arata, and nothing is ever removed.

    CONTRACT: only valid while *color* is still drafting.  Raises ValueError
    when that color's draft bit (``board.draft_w`` / ``board.draft_b``) is
    clear, because after the draft the board count drifts from the placement
    count (post-draft drops add pieces, captures remove them).

    The two obvious shortcuts are wrong and must not be used: ``move_number``
    is shared between the players and desyncs as soon as one side plays an
    early 終, and hand deltas overcount because an arata-betray decrements
    extra hand pieces beyond the one placed.
    """
    if color == 0:
        drafting = board.draft_w
    elif color == 1:
        drafting = board.draft_b
    else:
        raise ValueError(f"color must be 0 (white) or 1 (black), got {color!r}")

    if not drafting:
        raise ValueError(
            f"draft_placements({'wb'[color]}) is only defined while that "
            "color's draft bit is set; its draft is already finished"
        )

    cells = np.asarray(board.cells_arr)
    occupied = cells[cells != 0].astype(np.int32)
    return int(np.count_nonzero(((occupied - 1) & 1) == color))


def draft_root_mask(
    board: Board, valid_mask: np.ndarray, min_placements: int
) -> np.ndarray:
    """Root mask that withholds 終 (draft-finish) until *min_placements*.

    Returns a COPY of *valid_mask* with every C-block id zeroed iff all of:
    ``min_placements > 0``, the side to move still has its draft bit set, and
    that side has made fewer than *min_placements* draft placements.  In every
    other case the input mask object is returned unchanged (cheap no-op — in
    particular when both draft bits are clear, as in intro/beginner modes).

    Zeroing C-block ids is only sound while the mover is drafting; that is
    exactly the guard above (see ``action_encoding.SHU_BLOCK``).

    FALLBACK: if zeroing would leave no legal action at all, the original mask
    is returned unchanged.  That happens on the last hand piece, where
    cymovegen gates the continue-draft variant on ``not is_last_piece`` and so
    emits ONLY 終 variants.  The feature no-ops there rather than ever
    producing a zero-legal root.

    Never mutates *valid_mask*, the board, or any Game/valid_moves/win_state
    behaviour.
    """
    if min_placements <= 0:
        return valid_mask

    side = board.turn
    drafting = board.draft_w if side == 0 else board.draft_b
    if not drafting:
        return valid_mask

    if draft_placements(board, side) >= min_placements:
        return valid_mask

    c_start, c_stop = action_encoding.SHU_BLOCK
    if not valid_mask[c_start:c_stop].any():
        return valid_mask  # nothing to withhold
    if not valid_mask[:c_start].any():
        return valid_mask  # 終 is the only legal move — never zero it out

    masked = valid_mask.copy()
    masked[c_start:c_stop] = 0
    return masked


# ---------------------------------------------------------------------------
# Default start FEN (intro mode)
# ---------------------------------------------------------------------------

_DEFAULT_FEN = INTRO_POSITION

_MODE_FENS = {
    "intro": INTRO_POSITION,
    "beginner": BEGINNER_POSITION,
    "intermediate": INTERMEDIATE_POSITION,
    "advanced": ADVANCED_POSITION,
}

_VALID_MODES = sorted(_MODE_FENS)

_env_mode = os.environ.get("GUNGI_MODE", "intro")
if _env_mode not in _MODE_FENS:
    raise ValueError(
        f"GUNGI_MODE={_env_mode!r} is not valid; choose from {_VALID_MODES}"
    )


#: Byte layout of a repetition key (cyengine.repetition_key): 243 cells, then
#: 3 scalars (turn, draft_w, draft_b), then 28 hand slots.
_KEY_CELLS_BYTES = 9 * 9 * 3
_KEY_BYTES = _KEY_CELLS_BYTES + 3 + 14 * 2

# The two lengths are owned by ``komugi.engine.repetition`` (KOM-141); these
# module-level constants are the same numbers and are asserted equal so a
# divergence is a startup failure rather than a wrong draw much later.
assert _KEY_CELLS_BYTES == PLACEMENT_KEY_BYTES
assert _KEY_BYTES == FULL_KEY_BYTES


def _count_keys(history: list) -> dict:
    """Occurrence counter for a repetition-key history list.

    Only used when a Game is handed a history without its counter (a bridge or
    a hand-built fixture); clone/pickle/symmetries pass the counter along, so
    the normal paths never rebuild it.
    """
    counts: dict = {}
    for key in history:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_history(history: list, key: bytes, key_bytes: int = _KEY_BYTES) -> None:
    """Reject a caller-supplied history that cannot belong to *key*'s position.

    Two checks, both cheap and both loud, because this is the one place the
    repaired repetition invariants can be violated from outside (review F1):

    * every entry must be exactly *key_bytes* long — the length of the key THIS
      GAME'S RULE builds (KOM-141: 274 under ``RepetitionRule.FULL``, 243 under
      ``RepetitionRule.PLACEMENT``), which the caller passes in rather than this
      function guessing.  A pre-repair v1 key is 243 bytes (cells only) and can
      never equal a repaired key, so such a history used to load "successfully"
      under the full rule while ``repetition_count()`` returned 0 — every
      pre-load occurrence silently forgotten — and only surfaced much later as
      a ValueError inside ``_mirror_key``.  The check is a length EQUALITY, not
      a floor: handing a 274-byte full-key history to a placement-ruled game is
      refused for the same reason and with the same message.
    * the last entry must be the current position's key.  The history
      convention (v1's, kept by the repair, and relied on by the bridge and
      by ``tests/rules/fixtures.game_with_repeated_history``) is that the
      history CONTAINS the position now on the board; a history ending
      elsewhere would adjudicate the draw one occurrence late.

    An empty history is accepted — it makes no claim to contradict.
    """
    if not history:
        return
    for index, entry in enumerate(history):
        if not isinstance(entry, bytes):
            raise ValueError(
                f"_history[{index}] is {type(entry).__name__}, not bytes; "
                f"every entry must be a {key_bytes}-byte "
                "repetition-key blob"
            )
        if len(entry) != key_bytes:
            raise ValueError(
                f"_history[{index}] is {len(entry)} bytes, expected "
                f"{key_bytes}; a history built under a different repetition "
                "key (a pre-repair 243-byte cells-only history, "
                "docs/design/rules-audit.md F1-F4, or a full-key history "
                "handed to a placement-ruled game, KOM-141) is rejected "
                "rather than loaded with a silently empty repetition count"
            )
    if history[-1] != key:
        raise ValueError(
            "_history must end with the current position's repetition key "
            "(the history contains the position now on the board, this "
            "occurrence included); the given history ends with a different "
            "key, which would adjudicate the repetition draw one occurrence "
            "late"
        )


# ---------------------------------------------------------------------------
# Termination reasons (KOM-142)
# ---------------------------------------------------------------------------
#
# The seven labels ``Game.termination_reason`` can return.  They NAME the
# adjudication ``win_state`` already performs; nothing here is a new rule, and
# no predicate changes because they exist.

#: The game is not over (this includes every position still in the draft).
TERMINATION_NONE: Final[str] = "none"
#: One side's 帥 is gone from the board — ``cyengine`` counts marshals, so this
#: is also what a board carrying *three* marshals would report.
TERMINATION_MARSHAL_CAPTURED: Final[str] = "marshal_captured"
#: The side to move is in check with no escaping move.
TERMINATION_CHECKMATE: Final[str] = "checkmate"
#: The side to move is not in check and has no escaping move.
TERMINATION_STALEMATE: Final[str] = "stalemate"
#: The current position's repetition key has occurred 4+ times (rules-audit
#: KOM-027; the count is ``repetition_count()``, this occurrence included).
TERMINATION_FOURFOLD_REPETITION: Final[str] = "fourfold_repetition"
#: Two non-adjacent 帥 and nothing else, hands empty of non-marshals.
TERMINATION_INSUFFICIENT_MATERIAL: Final[str] = "insufficient_material"
#: The env's own PLY cap (``max_turns``, the engine's cap) — rules-audit F6: an
#: environment invention with no counterpart in the reference rulebook.
TERMINATION_TURN_CAP: Final[str] = "turn_cap"

#: Every label :meth:`Game.termination_reason` may return, "none" included.
#: Listed in the canonical order a census reports them, which is NOT the
#: precedence order — that is documented on the method itself.
TERMINATION_REASONS: Final[Tuple[str, ...]] = (
    TERMINATION_NONE,
    TERMINATION_MARSHAL_CAPTURED,
    TERMINATION_CHECKMATE,
    TERMINATION_STALEMATE,
    TERMINATION_FOURFOLD_REPETITION,
    TERMINATION_INSUFFICIENT_MATERIAL,
    TERMINATION_TURN_CAP,
)


class Game(GameState):

    _START_MODE: str = _env_mode

    @classmethod
    def set_start_mode(cls, mode: str) -> None:
        if mode not in _MODE_FENS:
            raise ValueError(
                f"Unknown start mode {mode!r}; valid modes: {_VALID_MODES}"
            )
        cls._START_MODE = mode
    """Gungi GameState for alphazero-general (Cython engine back-end)."""

    def __init__(
        self,
        board: Board | None = None,
        _history: list | None = None,
        _counts: dict | None = None,
        *,
        repetition_rule: RepetitionRule = RepetitionRule.FULL,
    ):
        if board is None:
            board = board_from_fen(_MODE_FENS[type(self)._START_MODE])
        super().__init__(board)
        # WHICH KEY this game counts occurrences of (KOM-141).  FULL is
        # Komugi's repaired 274-byte key and the default EVERYWHERE; PLACEMENT
        # is gungi.js@3.4.11's cells-only key and is only ever reached by a
        # caller that asked for it by keyword, because it re-introduces the
        # rules-audit F1-F4 aliasing on purpose (referee parity, not training).
        # Keyword-only, so no positional call site can drift onto it.
        self._repetition_rule: RepetitionRule = RepetitionRule(repetition_rule)
        self._player = board.turn  # 0=white, 1=black — integer attribute
        # The current position's repetition key, kept in step with _board so
        # repetition_count() is a single dict lookup (no history rescan).
        self._key: bytes = self._position_key(board)
        # The decoded last move, ``(from_sq, from_tier, to_sq, to_tier)`` with
        # 0..80 square ids and 1-based tiers, or None when no move has been
        # played into this Game — game start, and every Game built from a FEN,
        # which carries no move history.  ``play_action`` is the only writer.
        self._last_move: tuple[int, int, int, int] | None = None
        # Plies since the last progress move (capture / betrayal / arata).  A
        # FEN has no field for it either, so a loaded position starts at 0.
        self._progress_plies: int = 0
        if _history is not None:
            # A caller-supplied history is the only way the counter/key
            # invariants can be broken from outside — check it loudly here
            # rather than mis-adjudicating a draw plies later.
            _validate_history(
                _history, self._key, key_bytes_for(self._repetition_rule)
            )
            self._history: list = _history
            # _counts is the caller's own dict (clone/pickle/symmetries); copy
            # it so no two Games ever share a counter.
            self._key_counts: dict = (
                dict(_counts) if _counts is not None else _count_keys(_history)
            )
        else:
            self._history: list = [self._key]
            self._key_counts: dict = {self._key: 1}

    # ---- the repetition rule (KOM-141) -------------------------------------

    @property
    def repetition_rule(self) -> RepetitionRule:
        """Which key this game counts occurrences of — read-only.

        Read-only on purpose: the counter and the history are built from the
        rule's key, so flipping the rule on a live game would leave a history
        of one length being counted under another.  Build a new ``Game`` (and
        replay into it) instead.
        """
        return self._repetition_rule

    def _position_key(self, board: Board) -> bytes:
        """Return *board*'s key under THIS GAME'S rule.

        The single dispatch point, called at both key sites (``__init__`` and
        ``play_action``), so the seed key and every played key cannot come from
        different builders — which would make every count 1 and delete the
        rule silently.

        Both arms are spelled out and an unknown rule RAISES rather than
        falling through to the full key: a third member added later would
        otherwise be refereed as FULL while ``key_bytes_for`` refused its
        length, and the two disagreements would surface far from here.
        """
        if self._repetition_rule is RepetitionRule.FULL:
            return cyengine.repetition_key(board)
        if self._repetition_rule is RepetitionRule.PLACEMENT:
            placement: bytes = cyengine.placement_key(board)
            return placement
        raise ValueError(
            f"{self._repetition_rule!r} is not a RepetitionRule this engine "
            f"can build a key for"
        )

    # ---- static contract --------------------------------------------------

    @staticmethod
    def num_players() -> int:
        return 2

    @staticmethod
    def action_size() -> int:
        return action_encoding.ACTION_SIZE

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return observation.observation_size()

    @staticmethod
    def max_turns() -> int:
        #: Derived, never restated: the cap is
        #: :data:`komugi.engine.observation.MAX_TURNS`, which is also the
        #: denominator of observation planes 115 and 120.
        return observation.MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return True

    # ---- dynamic contract -------------------------------------------------

    def repetition_count(self) -> int:
        """Occurrences of the CURRENT position's key, this one included.

        O(1): the counter is maintained incrementally by ``play_action``.  The
        rule adjudicates on this number and nothing else (rules-audit Q4), so
        ``valid_moves`` and ``win_state`` both route through it and cannot
        disagree about the threshold.
        """
        return self._key_counts.get(self._key, 0)

    def valid_moves(self) -> np.ndarray:
        mask = np.zeros(self.action_size(), dtype=np.uint8)
        if cyengine.is_terminal_nonmate(self._board, self.repetition_count()):
            return mask
        cymovegen.fill_legal_mask(self._board, mask)
        if mask.sum() == 0:
            return mask
        if cyengine.in_check(self._board, self._board.turn):
            if cyengine.is_checkmate(self._board):
                mask[:] = 0
        elif cyengine.is_stalemate(self._board):
            mask[:] = 0
        return mask

    def play_action(self, action: int) -> None:
        super().play_action(action)
        decoded = action_encoding.decode_cython_action(action, self._board)
        # BEFORE apply_move: the origin tower still exists, and the mover is by
        # construction on top of it, so its tier is the tower's height.
        #
        # ``Board.height``'s COORDINATES ARE 0-BASED (it indexes ``self.cells``
        # directly) while its RETURN is the 1-based top tier, so the decoded
        # record's 0-based ``from_r``/``from_f`` go in unadjusted.  Passing the
        # 1-based coordinates ``apply_move`` uses internally would read a
        # different square and silently record a wrong tier.
        self._last_move = self._decode_last_move(decoded)
        cyengine.apply_move(self._board, decoded)
        if decoded["move_type"] in _PROGRESS_MOVE_TYPES:
            self._progress_plies = 0
        else:
            self._progress_plies += 1
        key = self._position_key(self._board)
        self._key = key
        self._history.append(key)
        self._key_counts[key] = self._key_counts.get(key, 0) + 1
        self._player = self._board.turn
        self._turns += 1

    def _decode_last_move(self, decoded: dict) -> tuple[int, int, int, int]:
        """``(from_sq, from_tier, to_sq, to_tier)`` for a move record.

        MUST be called BEFORE ``cyengine.apply_move``: ``from_tier`` is read off
        the live board, and after the move the origin tower no longer holds the
        mover.

        An arata has no origin — the record carries ``from_r == from_f == -1``
        (``cymovegen._make_move_record``) — and is reported as
        ``from_sq = -1``, ``from_tier = 0``, which is what makes plane 118
        all-zero for a drop.  ``to_tier`` is taken from the record rather than
        recomputed, so every move type keeps the generator's own value: a route
        lands at 1, a tsuke and a betrayal at ``target_tier + 1``, and a
        CAPTURE at *(friendly survivors in the target tower) + 1*, which is 2 or
        3 whenever the mover's own pieces are already in that tower.
        """
        from_r = int(decoded["from_r"])
        from_f = int(decoded["from_f"])
        to_sq = _sq_index(int(decoded["to_r"]), int(decoded["to_f"]))
        to_tier = int(decoded["to_tier"])
        if from_r < 0:
            return (-1, 0, to_sq, to_tier)
        return (
            _sq_index(from_r, from_f),
            self._board.height(from_r, from_f),
            to_sq,
            to_tier,
        )

    def in_draft(self, side: int) -> bool:
        """Return whether *side* (0 white, 1 black) still holds drafting rights.

        The per-side spelling this module already trusts -- ``board.draft_w if
        side == 0 else board.draft_b``, exactly as the placement-floor filter
        writes it for the side to move.

        **Deliberately not the disjunction** :meth:`win_state` gates on.
        Termination is a whole-game property, so ``draft_w or draft_b`` is the
        right question there; a *temperature schedule* is a property of the
        move being searched, so it is the mover's own bit that decides
        (KOM-036 §3.3). The draft is variable length and interleaved with the
        movement phase, so after one side finishes the two predicates disagree
        for every remaining ply of that side -- and under the disjunction the
        faster drafter would play its early middlegame at the draft
        temperature, asymmetrically between seats, in a game whose seat
        asymmetry is the object of study.
        """
        board = self._board
        return bool(board.draft_w if side == 0 else board.draft_b)

    def win_state(self) -> np.ndarray:
        result = np.zeros(3, dtype=np.uint8)
        board = self._board

        # Max-turns draw applies regardless of draft phase.
        if self._turns >= self.max_turns():
            result[2] = 1
            return result

        # Draft phase — no terminal.
        if board.draft_w or board.draft_b:
            return result

        # Terminal precedence: marshal-capture > checkmate > draw conditions
        # (checkmate before repetition is F8/Q10, pinned deliberately).
        count = self.repetition_count()
        if cyengine.is_game_over(board, count):
            if cyengine.is_checkmate(board):
                winner = 1 - board.turn
                result[winner] = 1
            elif cyengine.is_stalemate(board):
                result[2] = 1
            elif cyengine.is_repetition_draw(count):
                result[2] = 1
            elif cyengine.is_insufficient_material(board):
                result[2] = 1
            else:
                # Marshal captured — winner is the side that just moved.
                winner = 1 - board.turn
                result[winner] = 1

        assert result.sum() <= 1, f"win_state has multiple slots set: {result}"
        return result

    def termination_reason(self) -> str:
        """Why this position is terminal — one of :data:`TERMINATION_REASONS`.

        A READ-ONLY label for the adjudication :meth:`win_state` already
        performs.  It re-asks the same predicates in the same order and adds no
        rule of its own: for every position, ``termination_reason() ==
        TERMINATION_NONE`` iff ``win_state()`` is all-zero, and the reason it
        returns otherwise is the branch that set the slot.

        PRECEDENCE (read off ``win_state`` above, branch for branch — a census
        that disagreed with the scoring would be worse than no census):

        1. ``turn_cap`` — ``self._turns >= self.max_turns()``.  FIRST, and
           deliberately: the cap is tested before the draft gate and before any
           board predicate, so it outranks every other reason, and a game that
           somehow hit the engine's cap while still drafting reports it too
           (rules-audit F6: the cap is an env invention, counted in *plies*).
        2. ``none`` while either side still holds drafting rights — the draft
           has no terminal (``cyengine.is_game_over`` returns False there).
        3. ``none`` when ``cyengine.is_game_over(board, repetition_count())``
           is False; from here on the position IS terminal.
        4. ``checkmate``
        5. ``stalemate``
        6. ``fourfold_repetition``
        7. ``insufficient_material``
        8. ``marshal_captured`` — the remaining case, exactly as ``win_state``
           reaches it through its own ``else``.

        Steps 4-8 are ``win_state``'s branch order, and the CHECKMATE-BEFORE-
        FOURFOLD half of it is rules-audit **F8**, pinned deliberately: the
        gate, ``cyengine.is_game_over``, tests fourfold first, so a position
        that is both a checkmate and the 4th occurrence of its key is a WIN in
        this env (``tests/rules/test_repetition_rule.py`` RT-19) and is
        reported here as ``checkmate``.  The reference library assigns no
        result at all for that position, so nothing contradicts the choice yet
        — but changing it must be a deliberate edit that fails RT-19 and this
        module's precedence test, not silent drift.

        Steps 7 and 8 cannot both apply: ``insufficient_material`` requires
        exactly two marshals on the board and ``marshal_captured`` is
        ``count != 2``.
        """
        board = self._board

        if self._turns >= self.max_turns():
            return TERMINATION_TURN_CAP

        if board.draft_w or board.draft_b:
            return TERMINATION_NONE

        count = self.repetition_count()
        if not cyengine.is_game_over(board, count):
            return TERMINATION_NONE

        if cyengine.is_checkmate(board):
            return TERMINATION_CHECKMATE
        if cyengine.is_stalemate(board):
            return TERMINATION_STALEMATE
        if cyengine.is_repetition_draw(count):
            return TERMINATION_FOURFOLD_REPETITION
        if cyengine.is_insufficient_material(board):
            return TERMINATION_INSUFFICIENT_MATERIAL
        return TERMINATION_MARSHAL_CAPTURED

    def observation(self) -> np.ndarray:
        obs = observation.observation_from_cyboard(
            self._board,
            repetition_count=self.repetition_count(),
            plies=self._turns,
            last_move=self._last_move,
            progress_plies=self._progress_plies,
        )
        assert obs.shape == self.observation_size(), (
            f"Observation shape {obs.shape} != {self.observation_size()}"
        )
        return obs

    def final_control(self) -> np.ndarray:
        """Return the (81,) int8 top-of-tower ownership map of THIS position.

        ``observation.final_control`` over this game's board — 0 white, 1
        black, 2 empty, indexed ``s = r * 9 + (8 - f)`` like a flattened
        observation plane (KOM-002 design §2's orientation convention).

        Named "final" for the caller it exists for (KOM-010's recorder asks at
        adjudication, once per game), not because the position must be
        terminal: it is a pure function of the cells and answers for any
        position.
        """
        return observation.final_control(self._board)

    def material(self) -> tuple[int, int]:
        """Return ``(white_total, black_total)`` for THIS position.

        ``observation.material_counts`` over this game's board: every piece on
        the board at any tier, plus every piece still in hand, per colour.  The
        engine owns the read (KOM-245 review F9) for the reason
        :meth:`final_control` states -- the count is a property of the board,
        and a second spelling of it in a recorder would be a second rule.

        Like :meth:`final_control`, it answers for any position rather than
        only a terminal one; KOM-245's caller asks once, of the final board of
        a turn-capped game.
        """
        return observation.material_counts(self._board)

    def __reduce__(self):
        """Pickle support — reconstruct via Game(board, _history, _counts) and
        restore the remaining scalar attrs that clone() also copies.

        The 6th component (the counter) is what makes the round trip exact:
        ``__init__`` validates the history it is handed, so only pickles
        written by this version load at all (see ``_unpickle_game``).

        The 7th and 8th (the last move and the progress clock) are REQUIRED,
        not optional — see ``_unpickle_game`` for why defaulting them would be
        a silent corruption rather than a loud refusal.

        The 9th (the repetition rule, KOM-141) is required on the same terms:
        a placement-ruled game whose rule defaulted back to FULL on load would
        rebuild a 274-byte key against a 243-byte history and refuse, or worse,
        referee the rest of the game under the other rule.
        """
        return (
            _unpickle_game,
            (
                self._board,
                self._history,
                self._player,
                self._turns,
                self.last_action,
                self._key_counts,
                self._last_move,
                self._progress_plies,
                self._repetition_rule,
            ),
        )

    def clone(self) -> 'Game':
        """Fully independent copy — Board.copy() + history and counter copies."""
        new_board = self._board.copy()
        game = Game(
            new_board,
            list(self._history),
            self._key_counts,
            repetition_rule=self._repetition_rule,
        )
        game._player = self._player
        game._turns = self._turns
        game.last_action = self.last_action
        # Tuples are immutable, so sharing the object is already a copy.
        game._last_move = self._last_move
        game._progress_plies = self._progress_plies
        return game

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Game):
            return NotImplemented
        return (
            board_to_fen(self._board) == board_to_fen(other._board)
            and self._player == other._player
            and self._turns == other._turns
        )

    def __hash__(self) -> int:
        return hash(board_to_fen(self._board))

    def symmetries(self, pi: np.ndarray) -> List[Tuple['Game', np.ndarray]]:
        """Identity + left-right file mirror (the only Gungi symmetry)."""
        identity = self.clone()
        mirrored_pi = _mirror_pi(pi)

        # Build mirrored engine state via mirrored FEN
        fen = board_to_fen(self._board)
        fen_parts = fen.split(" ")
        placement = fen_parts[0]
        mirrored_placement = "/".join(
            _mirror_rank(rank_str) for rank_str in placement.split("/")
        )
        fen_parts[0] = mirrored_placement
        mirrored_fen = " ".join(fen_parts)
        new_board = board_from_fen(mirrored_fen)
        mirrored_history = _mirror_history_keys(self._history)
        # The file mirror is a bijection on keys, so the counts carry over
        # unchanged — a mirrored game repeats exactly when the original does.
        mirrored_counts = {
            _mirror_key(key): count for key, count in self._key_counts.items()
        }

        # The mirror is a bijection on keys of EITHER rule (it permutes the
        # cells block and leaves any suffix alone), so the reflected game is
        # refereed under the same rule as the original — carrying it is what
        # makes the mirrored half of an augmented pair the same game.
        mirrored_game = Game(
            new_board,
            mirrored_history,
            mirrored_counts,
            repetition_rule=self._repetition_rule,
        )
        mirrored_game._player = self._player
        mirrored_game._turns = self._turns
        # The last move must be MIRRORED, not copied.  Planes 118/119 mark it on
        # the board, so carrying it across unmirrored would put the original
        # move's squares on the reflected position — a systematically corrupted
        # input in half the symmetry-augmented rows, in range and silent.  A file
        # reflection permutes squares only; tiers are untouched by it.
        mirrored_game._last_move = _mirror_last_move(self._last_move)
        # A file reflection changes no piece count, so the potential Φ — and with
        # it the progress clock — is invariant under it.
        mirrored_game._progress_plies = self._progress_plies
        # Likewise the action id: ``_mirror_action`` is already applied to the
        # policy vector five lines up, and the two must not disagree now that the
        # last move is observable.
        mirrored_game.last_action = (
            None if self.last_action is None else _mirror_action(self.last_action)
        )

        return [(identity, pi), (mirrored_game, mirrored_pi)]


_MISSING = object()


def _unpickle_game(
    board,
    history,
    player,
    turns,
    last_action,
    counts=None,
    last_move=_MISSING,
    progress_plies=_MISSING,
    repetition_rule=_MISSING,
):
    """Reconstruct a Game from pickled components (used by Game.__reduce__).

    *counts* is optional only so that this function can still be *called* with
    the pre-repair 5-tuple shape; such pickles do NOT load.  Their history
    holds 243-byte cells-only keys, which ``Game.__init__`` now rejects with
    ValueError (before that validation they loaded "successfully" and then
    reported ``repetition_count() == 0``, silently forgetting every
    occurrence recorded before the pickle).  Pre-repair data is not
    warm-startable — replay it, do not unpickle it.

    *last_move* and *progress_plies* are REQUIRED (KOM-002), and the sentinel
    default exists only so the missing-argument case raises the named error
    below instead of a bare TypeError.  They are deliberately NOT defaulted to
    "no last move" / 0: unlike *counts*, which triggers a correct rebuild from
    the history, there is nothing to rebuild them from.  A pre-KOM-002 pickle
    of a mid-game position would silently load with a progress clock of 0 —
    in range, and indistinguishable from a game that just captured — and feed
    a wrong plane 120 into training.  Silence must not read as agreement.

    *repetition_rule* (KOM-141) is REQUIRED on exactly the same grounds, and
    its sentinel exists for the same reason.  Defaulting it to FULL would
    silently re-referee a placement-ruled game under Komugi's key, and there is
    nothing to reconstruct it from: a 274-byte history does not distinguish a
    full-key game from a corrupt placement one.  There are NO persisted game
    pickles in this repo (``git ls-files`` has none), so the refusal strands
    nothing — it only forbids an 8-component tuple written by hand.
    """
    if last_move is _MISSING or progress_plies is _MISSING:
        raise ValueError(
            "this pickle predates KOM-002 (input planes v2): it carries no "
            "_last_move / _progress_plies components, and neither can be "
            "reconstructed from the rest of the game state. Such a pickle "
            "would load with a silently wrong progress clock and wrong "
            "last-move planes (118-120), so it is refused. Replay the game, "
            "do not unpickle it."
        )
    if repetition_rule is _MISSING:
        raise ValueError(
            "this pickle predates KOM-141 (per-Game repetition rule): it "
            "carries no _repetition_rule component, and it cannot be "
            "reconstructed from the rest of the game state. Defaulting it to "
            "RepetitionRule.FULL would re-referee a placement-ruled game "
            "under Komugi's 274-byte key -- a different rule, silently -- so "
            "an 8-component pickle is refused. Replay the game, do not "
            "unpickle it."
        )
    game = Game(board, history, counts, repetition_rule=repetition_rule)
    game._player = player
    game._turns = turns
    game.last_action = last_action
    game._last_move = last_move
    game._progress_plies = progress_plies
    return game


# ---------------------------------------------------------------------------
# FEN mirroring helpers
# ---------------------------------------------------------------------------

def _mirror_rank(rank_str: str) -> str:
    """Mirror a single FEN rank string by reversing the token order.

    A FEN rank is a sequence of tokens where each token is either:
    - A digit (empty squares count)
    - A piece letter (upper or lower case)
    - A tower group like |X:Y| or |X:Y:Z|
    """
    tokens = _tokenize_rank(rank_str)
    tokens.reverse()
    return "".join(tokens)


def _tokenize_rank(rank_str: str) -> list[str]:
    """Split a FEN rank into its component tokens."""
    tokens = []
    i = 0
    while i < len(rank_str):
        if rank_str[i] == '|':
            # Tower group — find matching close pipe
            j = rank_str.index('|', i + 1)
            tokens.append(rank_str[i:j + 1])
            i = j + 1
        elif rank_str[i].isdigit():
            tokens.append(rank_str[i])
            i += 1
        else:
            tokens.append(rank_str[i])
            i += 1
    return tokens


def _mirror_key(key_bytes: bytes) -> bytes:
    """File-mirror one repetition key, of EITHER rule's length (KOM-141).

    Only the leading 243-byte cells block is mirrored; whatever follows passes
    through untouched, because a file reflection changes none of it.  That is
    the whole of a ``RepetitionRule.PLACEMENT`` key (243 bytes, nothing after
    it) and the leading block of a ``RepetitionRule.FULL`` key (274 bytes, with
    turn, draft flags and hands in the 31-byte suffix), so ONE mirror serves
    both rules and the two cannot drift apart.

    A key of any OTHER length still raises rather than silently producing a
    wrong key (rules-audit "Data structure"): the two accepted lengths are the
    two rules' lengths and nothing else, so a truncated or foreign blob is
    refused here instead of adjudicating a wrong draw much later.
    """
    if len(key_bytes) not in (PLACEMENT_KEY_BYTES, FULL_KEY_BYTES):
        raise ValueError(
            f"repetition key must be {PLACEMENT_KEY_BYTES} bytes (placement "
            f"rule) or {FULL_KEY_BYTES} bytes (full rule), got "
            f"{len(key_bytes)}"
        )
    cells = np.frombuffer(key_bytes[:_KEY_CELLS_BYTES], dtype=np.int8)
    cells = cells.reshape(9, 9, 3)[:, ::-1, :]  # flip file axis: f -> 8-f
    return cells.tobytes() + key_bytes[_KEY_CELLS_BYTES:]


def _mirror_history_keys(history: list) -> list:
    """Mirror a repetition-key history for the file-mirrored board.

    Each entry is a bytes blob from ``cyengine.repetition_key``.  We mirror by
    reconstructing: deserialize -> flip files -> reserialize.  This is a
    boundary operation, not hot-path.
    """
    return [_mirror_key(key_bytes) for key_bytes in history]
