"""Structural seams between komugi's major components.

These are ``typing.Protocol`` definitions, not base classes: engines, search
algorithms and inference servers are wired together structurally so a Cython
engine, a pure-python reference engine and a test double are interchangeable
without an inheritance relationship.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

Observation = npt.NDArray[np.float32]
"""Network-ready encoding of a single position."""

LegalMask = npt.NDArray[np.bool_]
"""One entry per action: True where the action is legal in this position."""

Policy = npt.NDArray[np.float32]
"""One entry per action: a (not necessarily normalised) move preference."""

Value = npt.NDArray[np.float32]
"""Per-position scalar value estimate, batched."""


class GameEngine(Protocol):
    """A two-player, perfect-information game with a fixed action space."""

    def reset(self) -> None:
        """Return the engine to the starting position."""
        ...

    def legal_mask(self) -> LegalMask:
        """Return the legality mask for the current position."""
        ...

    def play(self, action: int) -> None:
        """Apply a legal action, advancing to the next position."""
        ...

    def observation(self) -> Observation:
        """Return the network-ready encoding of the current position."""
        ...

    def win_state(self) -> tuple[bool, float]:
        """Return (terminal?, score for the side to move at the root)."""
        ...


class SearchAlgorithm(Protocol):
    """A tree search that turns an observation into a policy and a value."""

    def search(self, obs: Observation) -> tuple[Policy, float]:
        """Search from `obs` and return (improved policy, root value)."""
        ...


class InferenceServer(Protocol):
    """A batching evaluator shared by many self-play workers."""

    def submit(self, batch: Observation) -> tuple[Policy, Value]:
        """Evaluate a batch of observations and return (policies, values)."""
        ...
