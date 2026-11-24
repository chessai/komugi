"""The match search's own contracts: what it is handed, and what it hands back.

Four Protocols and one tuple, all defined here rather than by editing
:mod:`komugi.search.state` (design §M2). The match package **does not consume**
``SearchState``: it copies its two load-bearing conventions -- values are
mover-relative, ``play`` mutates in place while ``clone`` deep-copies -- and
shares none of its Protocols.

The reason the contracts are duplicated rather than imported is a single
missing method and a single unwanted one. ``SearchState`` has no way to state a
result as a win/draw/loss triple, and the match tree stores nothing else; and
its scalar result method folds a draw score into the number it returns, which
is the one thing a tree that derives utility at *read* time must never be able
to reach (design §S2). A structural inability to call it is worth more than a
comment saying not to.

Two state contracts, not one, because the seam that delivers a position and the
seam the tree searches over are different objects:

* :class:`MoverState` is what ``Mover.pick`` actually hands a match player
  (design §M2's B5 resolution). It is the public surface of
  ``komugi.selfplay.adapter.GameState`` that the match adapter needs, declared
  structurally so no module here imports that adapter.
* :class:`MatchState` is what the tree searches over:
  :class:`~komugi.search.match.adapter.MatchGameState` is the wrapper that
  turns one into the other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt


class MoverState(Protocol):
    """What ``Mover.pick`` actually hands a match player. See design §M2.

    Exactly the public surface of ``komugi.selfplay.adapter.GameState`` that
    the match adapter needs, declared structurally so the match package never
    imports it. NOTE the absence of ``terminal_wdl`` and of the scalar
    result method: this is the *input* contract, not the tree's, and the
    scalar one is deliberately unreachable from here (design §M2).
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return one entry per action, True where the action is legal."""
        ...

    def play(self, action: int) -> None:
        """Advance this position in place by `action`."""
        ...

    def clone(self) -> MoverState:
        """Return an independent copy."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the game has been adjudicated here."""
        ...

    def to_play(self) -> int:
        """Return the index of the player to move."""
        ...

    def observation(self) -> npt.NDArray[np.float32]:
        """Return this position encoded for the network."""
        ...

    def absolute_outcome(self) -> npt.NDArray[np.uint8]:
        """Return the raw ``(3,)`` uint8 ``[white, black, draw]`` outcome."""
        ...


class MatchState(Protocol):
    """A search state that can also state its result as a WDL triple.

    The five methods the tree descends with, plus :meth:`observation` for the
    evaluator and :meth:`terminal_wdl` for a decided position.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return one entry per action, True where the action is legal."""
        ...

    def play(self, action: int) -> None:
        """Advance this position in place by `action`."""
        ...

    def clone(self) -> MatchState:
        """Return an independent copy; the search only ever mutates clones."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the game has been adjudicated here."""
        ...

    def to_play(self) -> int:
        """Return the index of the player to move."""
        ...

    def observation(self) -> npt.NDArray[np.float32]:
        """Return this position encoded for the network."""
        ...

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return a finite nonnegative ``(3,)`` WDL distribution for the mover.

        Its mass is one within the match tree's shared absolute simplex
        tolerance. Fractional distributions are valid; ordinary decided-game
        adapters return one-hot outcomes. Never contempt-weighted: a draw is
        ``(0, 1, 0)``, not a draw score. Contempt is a property of the reader
        and enters at read time only (design §S2).
        """
        ...


class WdlEval(NamedTuple):
    """One network evaluation of one position, as the match tree wants it.

    The WDL-carrying counterpart of ``Evaluator``'s ``(logits, value)`` pair:
    the scalar collapse that the self-play adapter performs on the way in is
    exactly what match play moves to read time, so the triple has to survive
    the seam intact (design §M1).

    Attributes:
        policy_logits: ``(A,)`` float32, raw and unmasked, indexed by action
            id. Masking is the tree's job, the same division of labour
            ``Evaluator`` documents.
        wdl: ``(3,)`` float64 probabilities in (win, draw, loss) order,
            summing to 1, in the perspective of the state's ``to_play()``.
            float64 rather than the network's float32 because every
            accumulator downstream of it is float64 (design §A5/§S4).
    """

    policy_logits: npt.NDArray[np.float32]
    wdl: npt.NDArray[np.float64]


WdlEvaluator: TypeAlias = Callable[[MatchState], WdlEval]
"""One non-terminal position in; ``(policy_logits (A,), wdl (3,))`` out.

``wdl`` is a probability vector summing to 1 in the perspective of the state's
``to_play()``, exactly as ``NetEval.wdl``. Raw, unmasked logits, exactly as
``Evaluator``. No contempt anywhere.
"""


class LeafWeighter(Protocol):
    """Turns one leaf evaluation into a backup weight (design §A5).

    The seam the uncertainty head will land behind. It ships now, unwired, so
    that adding the head is a change of one class rather than a change of the
    node schema; with the default
    :class:`~komugi.search.match.config.UnitWeight` every weight is ``1.0``
    and the per-edge weight equals the per-edge visit count exactly.
    """

    def weight(self, evaluation: WdlEval) -> float:
        """Return the backup weight for `evaluation`, strictly positive."""
        ...
