"""The engine-backed tactical provider: ordinary A0 marshal captures (KOM-222).

:class:`GameMarshalCaptureProvider` implements
:class:`~komugi.search.tactical.TacticalProvider` over
:class:`~komugi.selfplay.adapter.GameState`, and enumerates exactly one class of
move: an **ordinary board capture that removes an enemy 帥**. That is the
decisive-move class the KOM-215 census found the policy prior dropping -- the
self-play loop missed a mate-in-1 in 41-46% of games -- and it is enumerable
from a single move-generation pass, with no clone, no play and no evaluation.

**What it is NOT.** Three deliberate exclusions, each of which is a contract the
tests pin rather than an oversight:

* **Checkmate is out of scope.** ``Game.win_state`` (``game.py:612``) awards the
  win to ``1 - board.turn`` for a marshal capture *or* a checkmate; only the
  first is a capture, so a mating move that captures nothing is not a candidate
  and must never be required to be one.
* **A candidate is not a win.** ``Game.win_state`` checks the **turn cap first**,
  so a capture that ends the game on the cap is adjudicated a **draw**. Candidacy
  therefore means "the search is obliged to look at this", never "this wins".
* **Betrayals and drops are out of scope.** 返 (``MT_BETRAY``) and 新
  (``MT_ARATA``) live in other blocks of the action encoding; this provider
  enumerates block A0 (``id = from_sq * 81 + to_sq``, ``0 <= id < 6561``) only.

A **buried** marshal counts: ``cymovegen`` records ``captured`` as *all* enemy
pieces in the destination tower (``cymovegen.pyx:663-668``), so a capture of a
tower whose top is some other enemy piece but which hides a 帥 underneath is a
candidate, and is found by scanning that list rather than only its top entry.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from komugi.engine.action_encoding import ACTION_SIZE, encode_cython_move
from komugi.engine.cymovegen import generate_all_moves
from komugi.search.state import SearchState
from komugi.selfplay.adapter import GameState

#: ``cymovegen``'s ``MT_CAPTURE``. Restated rather than imported: the constants
#: live in a ``.pyx`` and have no python-level name (``action_encoding.py:359``
#: keeps its own copy for the same reason).
_MT_CAPTURE: int = 2

#: ``cyboard.PIECE_TYPES[0]`` -- 帥, the marshal.
_MARSHAL_TYPE_INDEX: int = 0

#: Width of action block A0 (board move, no stack): ``81 * 81``.
_A0_WIDTH: int = 6561

#: The three slots of ``GameState.absolute_outcome`` -- ``[white, black, draw]``.
_OUTCOME_WIDTH: int = 3


class GameMarshalCaptureProvider:
    """Ordinary A0 captures that remove an enemy marshal, from one movegen pass.

    Satisfies :class:`~komugi.search.tactical.TacticalProvider`; see that
    protocol for the contracts (ascending unique ids drawn only from the
    supplied mask, completeness with conservative false positives permitted, no
    clone / play / evaluate / mutate, categorical terminal outcomes only).
    """

    provider_id: str = "game-marshal-capture-v1"

    def validate_state(self, state: SearchState) -> None:
        """Raise ``TypeError`` unless *state* is a :class:`GameState`."""
        self._checked(state)

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return the legal A0 marshal-removing captures of *state*, ascending.

        One ``generate_all_moves`` pass over the live board, filtered to
        ``MT_CAPTURE`` records whose ``captured`` list holds a 帥, encoded with
        ``encode_cython_move`` and intersected with *legal_mask*. Deliberately
        **not** a per-id ``decode_cython_action`` loop: that call regenerates
        every move of the position each time it is made.

        Neither the board nor *legal_mask* is mutated, and no state is cloned,
        played or evaluated.
        """
        game_state = self._checked(state)
        if legal_mask.shape != (ACTION_SIZE,):
            raise ValueError(
                f"{self.provider_id}: legal_mask must have shape "
                f"({ACTION_SIZE},), got {legal_mask.shape!r}"
            )
        if legal_mask.dtype != np.bool_:
            raise ValueError(
                f"{self.provider_id}: legal_mask must be dtype bool, got "
                f"{legal_mask.dtype!r}"
            )

        # `_board` because `Game` publishes no board accessor; the same private
        # read the engine's own tests make (``tests/rules/fixtures.board_of``).
        board = game_state.game._board  # noqa: SLF001
        candidates: set[int] = set()
        for record in generate_all_moves(board):
            if int(record["move_type"]) != _MT_CAPTURE:
                continue
            if not any(
                int(type_index) == _MARSHAL_TYPE_INDEX
                for type_index, _tier in record["captured"]
            ):
                continue
            action_id = int(encode_cython_move(record))
            # The range guard is an assertion about the encoding (captures land
            # in A0), kept as a filter so a future block cannot leak in here.
            if 0 <= action_id < _A0_WIDTH and bool(legal_mask[action_id]):
                candidates.add(action_id)
        return tuple(sorted(candidates))

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Return ``0``/``1`` for the winner of *terminal_state*, ``None`` for a draw.

        Reads ``GameState.absolute_outcome`` -- the engine's own ``[white,
        black, draw]`` frame -- and never ``terminal_value`` / ``terminal_wdl``,
        both of which fold ``draw_score`` or the mover's perspective into the
        answer. A non-terminal state, or an outcome that is not a categorical
        one-hot, raises: there is no ``argmax`` repair here, because a
        best-effort winner read off a malformed outcome is how a flipped label
        gets into a training row.
        """
        game_state = self._checked(terminal_state)
        if not game_state.is_terminal():
            raise ValueError(
                f"{self.provider_id}: terminal_winner requires a terminal "
                "state, and this position is not terminal"
            )
        outcome = game_state.absolute_outcome()
        if outcome.shape != (_OUTCOME_WIDTH,):
            raise ValueError(
                f"{self.provider_id}: outcome must have shape "
                f"({_OUTCOME_WIDTH},), got {outcome.shape!r}"
            )
        if not np.issubdtype(outcome.dtype, np.integer):
            raise ValueError(
                f"{self.provider_id}: outcome must have an integer dtype, got "
                f"{outcome.dtype!r}"
            )
        slots = [int(outcome[index]) for index in range(_OUTCOME_WIDTH)]
        if any(slot not in (0, 1) for slot in slots) or sum(slots) != 1:
            raise ValueError(
                f"{self.provider_id}: outcome must be a categorical one-hot "
                f"[white, black, draw], got {slots!r}"
            )
        if slots[0] == 1:
            return 0
        if slots[1] == 1:
            return 1
        return None

    def _checked(self, state: SearchState) -> GameState:
        """Return *state* narrowed to :class:`GameState`, or raise ``TypeError``.

        The type check happens **before** anything else touches the argument,
        so a state from another tree fails with a type error rather than an
        attribute error somewhere further in.
        """
        if not isinstance(state, GameState):
            raise TypeError(
                f"{self.provider_id} requires a "
                f"komugi.selfplay.adapter.GameState, got "
                f"{type(state).__name__}"
            )
        return state
