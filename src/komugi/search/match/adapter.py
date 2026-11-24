"""The seam: what ``Mover.pick`` delivers, wearing the match tree's contract.

``Mover.pick`` hands a match player a ``komugi.selfplay.adapter.GameState``.
That object has the five state methods the tree descends with, plus
``observation`` and ``absolute_outcome`` -- but **no** ``terminal_wdl``, so it
is not a :class:`~komugi.search.match.state.MatchState`. It is a
:class:`~komugi.search.match.state.MoverState`, which it satisfies
*structurally*, so this package can type the seam without importing the
self-play adapter at runtime at all.

:class:`MatchGameState` is the one-method difference. ``terminal_wdl`` is
**derived, not stored**::

    terminal_wdl() = float64( outcome[m], outcome[2], outcome[1 - m] )

from the raw ``(3,)`` ``[white, black, draw]`` outcome and the mover ``m``.
There is no draw score anywhere in it, and that is precisely why the scalar
result method those two components also feed -- the one that returns a draw
score on a draw -- is a method this package must not be able to call. A
structural inability to call it is worth more than a comment saying not to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from komugi.search.match.state import MatchState, MoverState
from komugi.selfplay.inference import WDL_SIZE

if TYPE_CHECKING:
    # Design §M2/B5's single named exception to the total runtime ban on
    # `komugi.selfplay.adapter`: type-checking only, in this file only, and
    # for exactly one purpose -- pinning the structural claim the whole seam
    # rests on. If `GameState` ever stops satisfying `MoverState`, mypy fails
    # *here*, at the seam that made the claim, rather than at some caller's
    # `mover: Mover = MatchPlayer(...)` far away from the cause.
    from komugi.selfplay.adapter import GameState

    def _game_state_is_a_mover_state(state: GameState) -> MoverState:
        return state


#: Index of the draw slot in the engine's absolute outcome vector.
_DRAW_SLOT: Final[int] = 2


class MatchGameState:
    """A :class:`MoverState` wearing the match tree's :class:`MatchState`.

    Thin by construction: every method delegates, and the only arithmetic in
    the class is :meth:`terminal_wdl`'s frame change. The wrapped state is
    owned, not shared -- :meth:`clone` clones it -- because :meth:`play`
    mutates in place and the search clones constantly.

    Args:
        state: What the ``Mover`` seam delivered. Typed structurally, so
            wrapping a hand-built test state costs nothing and requires no
            engine.
    """

    __slots__ = ("_state",)

    def __init__(self, state: MoverState) -> None:
        self._state = state

    @property
    def inner(self) -> MoverState:
        """Return the wrapped state (diagnostics and tests)."""
        return self._state

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the wrapped state's legality mask."""
        return self._state.legal_mask()

    def play(self, action: int) -> None:
        """Advance the wrapped state in place by `action`."""
        self._state.play(action)

    def clone(self) -> MatchGameState:
        """Return an independent copy, wrapping a clone of the inner state."""
        return MatchGameState(self._state.clone())

    def is_terminal(self) -> bool:
        """Return whether the engine has adjudicated this position."""
        return self._state.is_terminal()

    def to_play(self) -> int:
        """Return the index of the player to move."""
        return self._state.to_play()

    def observation(self) -> npt.NDArray[np.float32]:
        """Return this position encoded for the network."""
        return self._state.observation()

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` one-hot (win, draw, loss) for the mover here.

        The engine states an outcome absolutely -- ``[white, black, draw]`` --
        and everything above it is mover-relative, so the win slot is the
        mover's, the loss slot is the other player's, and the draw slot is
        shared. **No draw score is applied**: contempt is a property of the
        reader and enters at read time (design §S2), so a result that carried
        one would poison every reuse of this subtree the moment the reader
        changed its mind.

        Returns:
            A one-hot triple on a decided position; the all-zero vector on a
            position with no result yet, which is what the engine reports
            there and is not a claim that the game was drawn.
        """
        outcome = np.asarray(self._state.absolute_outcome(), dtype=np.float64)
        mover = self._state.to_play()
        triple: npt.NDArray[np.float64] = np.zeros(WDL_SIZE, dtype=np.float64)
        triple[0] = outcome[mover]
        triple[1] = outcome[_DRAW_SLOT]
        triple[2] = outcome[1 - mover]
        return triple


def as_match_state(state: MoverState) -> MatchState:
    """Return `state` as a :class:`MatchState`, wrapping it if need be.

    The seam a match player calls on every ``pick``. Kept as a function rather
    than inlined so that the one place a conversion happens is nameable in a
    review.
    """
    return MatchGameState(state)
