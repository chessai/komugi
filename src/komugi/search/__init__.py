"""Pure Gumbel AlphaZero search, generic over any :class:`SearchState`.

This is the self-play search for the komugi engine, and it is *only* the
search: no board, no network, no torch. It takes a state that satisfies
:class:`~komugi.search.state.SearchState` and a callable that satisfies
:data:`~komugi.search.state.Evaluator`, and returns an action plus a training
policy target.

The algorithm is Danihelka et al., "Policy improvement by planning with
Gumbel" (ICLR 2022). Its point is that policy improvement can be *guaranteed*
at small simulation budgets, which the AlphaZero recipe cannot promise: with
16 or 32 simulations at a branching factor near 229, visit counts are far too
sparse to be a policy. Gumbel gets the guarantee by sampling a handful of root
actions without replacement, spending the budget on comparing exactly those
by sequential halving, and taking the training target from *completed
Q-values* rather than from visits.

Four AlphaZero fixtures are deliberately absent -- root Dirichlet noise, root
temperature, forced playouts, and visit-count policy targets. None of them is
an oversight or a TODO. See :mod:`komugi.search.gumbel` for why each one is
gone and what replaces it.
"""

from __future__ import annotations

from komugi.search.gumbel import GumbelConfig, GumbelSearch, SearchResult
from komugi.search.state import Evaluator, SearchState

__all__ = [
    "Evaluator",
    "GumbelConfig",
    "GumbelSearch",
    "SearchResult",
    "SearchState",
]
