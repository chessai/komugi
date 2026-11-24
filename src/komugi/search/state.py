"""The two contracts the search is written against: a state, and an evaluator.

The search in this package knows nothing about Gungi. It knows that a position
can enumerate its legal actions as a boolean mask, advance itself by one of
them, copy itself, and say whether it is over and who is to move. That is the
whole of :class:`SearchState`, and it is a :class:`typing.Protocol` rather than
a base class precisely so the Cython engine, a mock, and a three-node toy in a
unit test are all equally valid inputs -- nothing in the tree ever imports a
board.

Two conventions are load-bearing and easy to get subtly wrong, so they are
stated once here and relied on everywhere else:

* **Perspective.** Both :meth:`SearchState.terminal_value` and the value half
  of an :data:`Evaluator` return a number in ``[-1, 1]`` *from the point of
  view of the player to move at that state*. Nothing is expressed in "player
  0's" frame. The search negates a value whenever it crosses an edge between
  states with different :meth:`SearchState.to_play`, which is what makes the
  backup a negamax rather than a minimax with sign bookkeeping at every node.

  The same convention governs the **WDL vector** of KOM-173, where the frame
  change is the permutation ``(w, d, l) -> (l, d, w)`` rather than a negation:
  :meth:`SearchState.terminal_wdl` and the optional third element of an
  :data:`Evaluator`'s answer are both in the mover's frame at that state, and
  the search applies the swap once per change of mover, at exactly the edges
  it negates the scalar at.
* **Mutation.** :meth:`SearchState.play` mutates in place and
  :meth:`SearchState.clone` returns a deep, independent copy. The search only
  ever mutates clones it made itself; the state handed to
  :meth:`~komugi.search.gumbel.GumbelSearch.search` comes back untouched.

An :data:`Evaluator` is called on non-terminal states, with exactly one
carve-out: a **forced-move root that is also terminal** (a terminal state
whose mask still reports one legal action) is evaluated once, solely to supply
:attr:`~komugi.search.gumbel.SearchResult.root_prior_logits` -- its *value* is
still taken from :meth:`SearchState.terminal_value`, never the network
(KOM-040 D9', review F1). Every other terminal leaf is scored by
:meth:`SearchState.terminal_value` and never evaluated, so an implementation
may still assume a mask with at least one legal action -- but not
non-terminality.

A :data:`WaveEvaluator` is the same contract taken a batch at a time: one call,
several positions, answers positionally aligned. It exists because the search
can, within one sequential-halving round, produce several leaves that are
provably independent of one another (KOM-040 b2; see
``docs/reviews/2026-08-24-kom-040-worker-lane-design-review.md``), and sending
them to the network as one wave costs one round trip instead of several.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt


class SearchState(Protocol):
    """A game position the search can enumerate, advance, copy and score.

    Deliberately minimal: every method here is one the tree actually calls.
    Anything richer (move history, undo, hashing, repetition detection) stays
    on the concrete state, because a protocol the search does not need is a
    constraint on implementations for no benefit.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return a shape ``(A,)`` boolean mask, ``True`` where an action is legal.

        ``A`` must equal the ``num_actions`` of the
        :class:`~komugi.search.gumbel.GumbelConfig` driving the search: the
        action space is fixed, and legality varies within it. A terminal state
        may return an all-``False`` mask.
        """
        ...

    def play(self, action: int) -> None:
        """Advance this state in place by `action`, which must be legal here."""
        ...

    def clone(self) -> SearchState:
        """Return a deep, independent copy; mutating it must not touch `self`."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the game has ended in this position."""
        ...

    def terminal_value(self) -> float:
        """Return the game result in ``[-1, 1]``, for the player to move here.

        Only meaningful when :meth:`is_terminal` is ``True``. A loss for the
        side to move is ``-1``; the side to move being checkmated is the usual
        case, so this is usually negative at the leaves the search reaches.
        """
        ...

    def to_play(self) -> int:
        """Return the index (``0`` or ``1``) of the player to move."""
        ...

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` one-hot ``(win, draw, loss)`` for the mover HERE.

        The vector counterpart of :meth:`terminal_value`, in the **leaf
        mover's frame**: ``w`` is "the player to move at this state wins", and
        a draw is ``(0, 1, 0)`` -- never a draw score, because the draw utility
        is the reader's and enters only through the scalar (KOM-173 D5/D6). The
        same rule the match tree's adapter already applies
        (``komugi.search.match.state.MatchState.terminal_wdl``).

        Only meaningful when :meth:`is_terminal` is ``True``. Read by the
        search **only** while the searched-WDL accumulator is live, i.e. when
        the evaluator answers with a three-element tuple (see
        :data:`Evaluator`), so a state used solely with a legacy two-tuple
        evaluator is never asked.
        """
        ...


WdlAnswer = npt.NDArray[np.float32] | npt.NDArray[np.float64]
"""A ``(3,)`` ``(w, d, l)`` probability vector, ``float32`` or ``float64``.

``float32`` is the production shape -- it is what the network head answers and
what :mod:`komugi.selfplay.inference` carries -- and ``float64`` is accepted so
that an evaluator whose arithmetic is exact (a solver, a unit-test stub pinning
an identity at ``1e-12``) does not have to round-trip through ``float32`` to be
heard. The search converts to ``float64`` either way.
"""


EvalAnswer = (
    tuple[npt.NDArray[np.float32], float]
    | tuple[npt.NDArray[np.float32], float, WdlAnswer]
)
"""One evaluation, in either of the two shapes the search accepts.

``(policy_logits, value)`` is the original answer and stays fully supported:
an evaluator that returns it drives the search exactly as it always did, and
the searched-WDL accumulator is never allocated.

``(policy_logits, value, wdl)`` adds the network's ``(3,)`` ``(w, d, l)``
probability vector for that leaf, **in the leaf mover's frame** (KOM-173 §2).
The vector is a :data:`WdlAnswer`, i.e. ``float32`` (the production shape) or
``float64`` (an evaluator whose arithmetic is exact); the search converts to
``float64`` either way. It is carried alongside the scalar and never feeds
selection: whichever shape an evaluator uses, `value` is the only number the
tree ranks on, so a search's ``action``, ``policy_target`` and ``visit_counts``
are bit-identical either way.

The shape is detected per evaluation by its length, which is why the two may
not be mixed within one search, and :func:`~komugi.search.gumbel._backup`
refuses the mix loudly **in both directions** (KOM-210 F3): a root that
answered with a vector and a leaf that did not would leave the accumulator with
a hole, and a leaf that answered with a vector under a root that did not has no
accumulator to be averaged into and could only produce ``NaN`` readouts.
"""


Evaluator = Callable[[SearchState], tuple[npt.NDArray[np.float32], float]]
"""A network wrapper: a state in, ``(policy_logits, value)`` out.

`policy_logits` has shape ``(A,)`` and dtype ``float32``, unnormalised and
*unmasked* -- the search applies the legal mask itself, so an evaluator never
has to know what the mask is doing. `value` is in ``[-1, 1]`` from the
perspective of the state's :meth:`SearchState.to_play`.

Called on non-terminal states, plus the one forced-terminal-root carve-out in
the module docstring. A batching or caching evaluator is free to
be a callable object rather than a function; nothing here requires purity, but
the search's determinism guarantee is inherited from it, so an evaluator whose
output depends on call order makes the search non-reproducible.

This is the **two-tuple** shape and it is unchanged by KOM-173: the arena, the
gauntlet, the ladder, the anchor server and every evaluator stub in the suite
are typed against it and keep working. :data:`WdlEvaluator` is the shape that
also carries the leaf ``(w, d, l)``, and :data:`AnyEvaluator` -- either of the
two -- is what :class:`~komugi.search.gumbel.GumbelSearch` accepts.
"""


WdlEvaluator = Callable[
    [SearchState], tuple[npt.NDArray[np.float32], float, WdlAnswer]
]
"""An :data:`Evaluator` that also answers with the leaf's ``(w, d, l)``.

The third element is that state's win/draw/loss probabilities **in its own
mover's frame**, un-collapsed and with no utility applied (KOM-173 §2). Given
one, :class:`~komugi.search.gumbel.GumbelSearch` accumulates the searched vector
and fills :attr:`~komugi.search.gumbel.SearchResult.root_wdl` and its two
siblings; the scalar it ranks on is still `value` alone, so the search it runs
is the same search.

Produced by ``komugi.selfplay.adapter.make_evaluator(..., with_wdl=True)``, i.e.
by the self-play worker and by nothing else.
"""


AnyEvaluator = Callable[[SearchState], EvalAnswer]
"""Either evaluator shape -- what the search takes, and tells apart by ``len``.

Written as one callable returning the :data:`EvalAnswer` union rather than as
the union ``Evaluator | WdlEvaluator``, and the difference is not cosmetic: a
callable's return type is **covariant**, so both narrow aliases -- and any
callable object whose own annotation is the wide answer, which is what
``komugi.selfplay.adapter``'s seams are -- satisfy this one, while the union
would accept only the two exact shapes and reject anything typed against the
wider answer it is allowed to return.

One shape per search: the root is always evaluated through the single-position
seam, that answer decides whether the accumulator exists, and a leaf that then
disagreed would leave it with a hole (:data:`EvalAnswer`).
"""


WaveEvaluator = Callable[
    [Sequence[SearchState]], Sequence[tuple[npt.NDArray[np.float32], float]]
]
"""The batched form of :data:`Evaluator`: many states in, many answers out.

The returned sequence is **positionally aligned** with the input -- entry ``i``
is the answer for state ``i`` -- and must have exactly the same length; the
search checks that and refuses a wave answered with the wrong number of
results, because a silent misalignment would attach one position's policy to
another position's node and leave no other symptom.

Each entry carries the same semantics as :data:`Evaluator`'s answer, down to the
detail: shape ``(A,)`` ``float32``, raw -- unnormalised and *unmasked* -- logits,
and a value in ``[-1, 1]`` from the perspective of that state's
:meth:`SearchState.to_play`.

Called only on non-terminal states, and never on the forced-terminal-root
carve-out in the module docstring: that root, like every root, is evaluated
through the single-position :data:`Evaluator`, so the carve-out never reaches
this seam. Terminal leaves are scored by
:meth:`SearchState.terminal_value` and are simply absent from the wave -- a
wave in which *every* descent hit a terminal leaf is empty and is not
submitted at all, which is normal endgame search flow rather than an error.

The search never submits an empty sequence, so an implementation may assume at
least one state. Determinism is inherited exactly as for :data:`Evaluator`,
with one addition worth stating: the answers must not depend on how the states
were grouped into waves. An evaluator that is order- or batch-sensitive
(padding-dependent kernels, a cache keyed by position in the batch) makes the
wave-scheduled search disagree with the sequential one, which is the property
KOM-040 b2's equivalence golden exists to pin.
"""


AnyWaveEvaluator = Callable[[Sequence[SearchState]], Sequence[EvalAnswer]]
"""Either wave shape -- what the search takes.

The wave counterpart of :data:`AnyEvaluator`, and written the same way and for
the same reason: :data:`WaveEvaluator` -- the two-tuple batch above -- is a
subtype of it by return covariance (``Sequence`` is covariant too), as is any
batched seam that also carries the leaf vector, and so is
``komugi.selfplay.adapter.WaveSeam``, whose ``__call__`` is annotated with the
wide answer because one object serves both shapes.

**Two modules spell ``WaveEvaluator`` differently, on purpose** (KOM-210 F8b).
This module's :data:`WaveEvaluator` is the narrow **two-tuple** batch, which is
what the arena, the gauntlet and the ladder are typed against;
``komugi.selfplay.adapter.WaveEvaluator`` is the **wide** one -- a callable
answering with ``Sequence[EvalAnswer]``, i.e. structurally this alias -- because
one adapter object serves both the ``with_wdl=False`` and the ``with_wdl=True``
seam and has to be annotated with the answer it may return. They are
deliberately not renamed into one name: the arena's narrow annotation is
load-bearing (it is the shape its batcher promises), and the adapter's name is
the one its own public seams are documented with. Import the one whose module
you are working in, and read a bare ``WaveEvaluator`` in a diff as ambiguous
until you have checked the import.
"""
