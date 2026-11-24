"""Match-play search: PUCT with KataGo's refinements, LCB, and tree reuse.

The *other* configuration. ``komugi.search.gumbel`` is the self-play search --
Gumbel root selection, no root noise, no tree reuse -- and it must stay that
way, because injecting root noise or PUCT root machinery into Gumbel ranking
destroys the policy-improvement guarantee that is the entire reason Gumbel is
there. This package is what plays matches and evaluations instead: classical
PUCT with dynamic variance-scaled cPUCT, subtree value bias correction, an
uncertainty-weighting seam, tree reuse between moves, and LCB move selection,
with root noise off.

**The one property everything else hangs off:** per-node values are stored as
**WDL components with utility derived at read time**. The production self-play
evaluator collapses win/draw/loss to a scalar *inside the adapter*, with
contempt already baked in, so a search fed by that seam cannot store WDL at
all -- the triple is destroyed one frame above it. Match play moves that
collapse point from *entry into the tree* to *read out of the tree*, and that
single relocation is what lets a retained tree survive a mid-game contempt
change instead of being poisoned by it. There is no ``value``, no ``q`` and no
``utility`` field anywhere in the node schema: it is impossible to poison what
is never written.

**This package does not share tree code with Gumbel, deliberately.** The two
algorithms disagree about what a node even stores -- Gumbel holds dense arrays
over the whole 37341-wide action space with a scalar value sum, match play
holds WDL triples, second moments and float weights sparse over the ~229 legal
moves -- so generalising one node class to carry both would edit the self-play
hot path to serve a search self-play never runs. What is shared is the
*conventions*, not the implementation, and the disjointness is proved by tests
rather than asserted: no self-play module imports this package, no module here
imports the Gumbel search, and the self-play closure's diff is empty.

Nothing in production consumes this yet. The arena still constructs
``GumbelSearch``; wiring a match player into it changes what every published
Elo means and belongs to its own ticket with its own gate.
"""

from __future__ import annotations

from komugi.search.match.config import MatchConfig
from komugi.search.match.player import MatchPlayer
from komugi.search.match.search import MatchResult, MatchSearch

__all__ = [
    "MatchConfig",
    "MatchPlayer",
    "MatchResult",
    "MatchSearch",
]
