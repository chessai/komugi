"""The jaxtyping/beartype shape-checking seam.

Shape and dtype checks are opt-in: they cost real time on every call, so they
are only wired in when ``KOMUGI_TYPECHECK=1`` is set *before this module is
imported*. That bootstrap ordering is the whole point of the pattern -- a
worker started with ``multiprocessing`` "spawn" re-imports this module in a
fresh interpreter and picks the flag up from the inherited environment, which
is what ``tests/test_env.py`` pins down.

Degradation: if jaxtyping/beartype are missing from the environment, the seam
falls back to identity decorators and plain numpy annotations, so importing
komugi never depends on them being installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

TYPECHECK: bool = os.environ.get("KOMUGI_TYPECHECK", "0") == "1"

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    # Static checkers see plain numpy arrays: jaxtyping's ``Float[...]`` form
    # carries the shape contract at runtime only.
    PolicyBatch = npt.NDArray[np.float32]
else:
    try:
        from jaxtyping import Float

        PolicyBatch = Float[np.ndarray, "batch actions"]
    except ImportError:  # degraded: no jaxtyping in this environment
        PolicyBatch = npt.NDArray[np.float32]


def _identity(fn: F) -> F:
    """Return `fn` unchanged -- the decorator used when checking is off."""
    return fn


def checked_seam() -> Callable[[F], F]:
    """Return the seam decorator: jaxtyping+beartype when enabled, else identity."""
    if not TYPECHECK:
        return _identity
    try:
        from beartype import beartype
        from jaxtyping import jaxtyped
    except ImportError:  # degraded: keep the seam callable, drop the checking
        return _identity

    def _decorate(fn: F) -> F:
        return jaxtyped(typechecker=beartype)(fn)

    return _decorate


@checked_seam()
def validate_policy_batch(x: PolicyBatch) -> int:
    """Return the batch size of a ``(batch, actions)`` policy array."""
    return int(x.shape[0])
