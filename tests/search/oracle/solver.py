"""The exact solver the oracle is built on.

Plain memoised negamax. It is deliberately dumb: no alpha-beta, no move
ordering, no heuristics -- nothing that could be subtly wrong in the same
direction as the search under test. Tic-tac-toe has ~5.5k reachable states, so
brute force with a memo table settles the whole game in well under a second,
and every later test can ask "what is the truth here?" and get an answer that
owes nothing to the search.

Two things are built on top of it:

* :func:`perfect_evaluator` -- the "perfect value, uninformative policy"
  evaluator. Uniform (all-zero) logits, and a value that is exactly right.
  Under it, a search that picks a losing move has no excuse: the information
  needed to avoid it was handed over on the first visit.
* :func:`miscalibrated_evaluator` -- the same perfect value with logits
  deliberately pointed *away* from the good moves. Policy improvement is the
  claim that search fixes a bad prior; this is the bad prior.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

#: Payoffs compared inside this tolerance count as tied, so that "the set of
#: optimal actions" stays a set for floating payoff tables instead of
#: collapsing to whichever action was visited first.
TIE_TOLERANCE: Final = 1e-9


class SolvableState(Protocol):
    """The search-state protocol plus the memo key the solver needs.

    ``key()`` is not part of what ``komugi.search.gumbel`` sees -- the search
    is handed the same objects but only ever calls the six protocol methods
    above it.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]: ...

    def play(self, action: int) -> None: ...

    def clone(self) -> SolvableState: ...

    def is_terminal(self) -> bool: ...

    def terminal_value(self) -> float: ...

    def to_play(self) -> int: ...

    def key(self) -> Hashable: ...


Evaluator = Callable[[SolvableState], tuple[npt.NDArray[np.float32], float]]


@dataclass(frozen=True)
class Solution:
    """The exact verdict for one state.

    `value` is from the perspective of the player to move; `best_actions` is
    every action attaining it, empty at a terminal state.
    """

    value: float
    best_actions: tuple[int, ...]


_MEMO: dict[tuple[str, Hashable], Solution] = {}


def legal_actions(state: SolvableState) -> tuple[int, ...]:
    """The legal actions of `state`, ascending."""
    mask = state.legal_mask()
    return tuple(index for index in range(int(mask.shape[0])) if bool(mask[index]))


def solve(state: SolvableState) -> Solution:
    """Return the exact value of `state` and every action that achieves it."""
    memo_key = (type(state).__name__, state.key())
    cached = _MEMO.get(memo_key)
    if cached is not None:
        return cached

    if state.is_terminal():
        result = Solution(value=state.terminal_value(), best_actions=())
    else:
        best_value = -math.inf
        best: list[int] = []
        for action in legal_actions(state):
            child = state.clone()
            child.play(action)
            # Negamax: the child reports from its own mover's perspective.
            value = -solve(child).value
            if value > best_value + TIE_TOLERANCE:
                best_value, best = value, [action]
            elif value >= best_value - TIE_TOLERANCE:
                best.append(action)
        if not best:
            raise ValueError(f"non-terminal state with no legal actions: {state!r}")
        result = Solution(value=float(best_value), best_actions=tuple(best))

    _MEMO[memo_key] = result
    return result


def action_values(state: SolvableState) -> dict[int, float]:
    """The exact value of each legal action, from the mover's perspective."""
    values: dict[int, float] = {}
    for action in legal_actions(state):
        child = state.clone()
        child.play(action)
        values[action] = -solve(child).value
    return values


def perfect_evaluator(
    state: SolvableState,
) -> tuple[npt.NDArray[np.float32], float]:
    """Uniform logits, exactly correct value: the oracle's evaluator."""
    logits: npt.NDArray[np.float32] = np.zeros(
        int(state.legal_mask().shape[0]), dtype=np.float32
    )
    return logits, solve(state).value


def miscalibrated_evaluator(bias: float = -3.0) -> Evaluator:
    """An evaluator whose prior leans away from the good moves.

    The logit of an action is `bias` times its true value, so with the default
    negative `bias` the prior's favourite action is the *worst* one. The value
    head stays perfect: the only thing wrong is the policy, which is exactly
    the thing search is supposed to be able to repair.
    """

    def evaluate(state: SolvableState) -> tuple[npt.NDArray[np.float32], float]:
        size = int(state.legal_mask().shape[0])
        logits: npt.NDArray[np.float32] = np.zeros(size, dtype=np.float32)
        if not state.is_terminal():
            for action, value in action_values(state).items():
                logits[action] = bias * value
        return logits, solve(state).value

    return evaluate
