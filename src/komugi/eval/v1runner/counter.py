"""§D3.5's v1-side instrumentation: the counted evaluator, and its refusal.

This is the module the whole parity claim rests on, so it is written to be
unit-testable **without v1, without torch and without a net** -- the arithmetic
is here, the network is behind :class:`CountedNet`'s one collaborator, and
the v1runner unit tests (not part of this repository) drive both in the
default shell.

Three rules from §D3, implemented rather than described:

* **Clause 1 -- count asks, not computes.** :meth:`EvalCounter.ask` is called
  *before* the forward, on the path the search takes to reach the network, so a
  memo hit counts exactly as a miss does. A unit that moves when a pure
  optimisation is toggled measures the implementation, not the search. Under
  §D3.2's shape (ii) -- where the memo *is* the tree -- "above the memo" has a
  concrete referent the U-A report must cite: the point where a descent asks
  for ``Ps[s]``, i.e. the expansion test, not the forward beneath it. Which
  shape v1 has is declared on the wire as ``memo_separable`` and is not decided
  here.
* **Clause 3 -- only this move's search.** :meth:`EvalCounter.begin_move`
  resets the tally, and every evaluation that is not part of a ``pick`` (a
  warm-up forward, a checkpoint probe, an observation probe) must simply not
  go through :class:`CountedNet`.
* **§D3.5's report-and-refuse.** The 258th ask inside one move is **refused**,
  not served and not truncated. :class:`BudgetRefused` unwinds v1's search
  inside the v1 process; the runner catches it and answers ``budget_breach``,
  and **the process stays alive**. That aliveness is the entire point of the
  response type (re-review F1): a raise that reached the harness as a process
  death would be booked M1 ``crash`` -- a *pair* void counting toward "three
  crashes abort the match" -- and M8, the *match*-level remedy that exists
  solely for the only side that can trip P2, would be unreachable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, ParamSpec, TypeVar

from komugi.eval.v1runner.protocol import EVAL_CEILING

_P = ParamSpec("_P")
_R = TypeVar("_R")

DEFAULT_OBSERVATION_RANK: Final[int] = 3
"""How many axes ONE v1 observation has: 117 planes x 9 x 9 (§D2.1).

That is *v1's* plane count; Komugi's own map is ``PLANE_MAP_VERSION`` 2, 128
planes. Nothing here compares the two -- this constant is a RANK, not a plane
count, and it is unaffected by either number.

The real value is read off ``Game.observation_size()`` at startup
(``reference.observation_shape``) and handed to :class:`CountedNet`; this is the
fallback for a game class that does not expose it. It is the number that
separates "one position" from "a stack of them", which is the whole of §D3.2's
definition of an evaluation.
"""

_NO_MOVE_IN_PROGRESS: Final[str] = (
    "ask() was called before begin_move(); §D3.2 clause 3 counts only "
    "evaluations made by the search for the current move, so an ask outside a "
    "move has no move to be attributed to"
)


class BudgetRefused(Exception):
    """The counted evaluator refused an ask beyond §D3.3's ceiling (P2 -> M8).

    Carries the numbers rather than a message because the ``budget_breach``
    response must *quantify* the breach, not merely flag it (§D1's list of
    load-bearing wire fields).

    Args:
        attempted: The ordinal of the refused ask -- 258 at a ceiling of 257.
            This, and not the served count, is what goes on the wire: the
            harness's own M8 detail reads "refused evaluation {evals_used}
            within one move" (``v1bridge.py`` ``_accept_move``'s sibling
            branch), and ``BudgetBreach`` says the same on Komugi's side
            ("asked for evaluation {evals_used}", ``v1bridge.py:311-317``).
            Reporting the served 257 instead would report a number that is
            inside the ceiling for an event whose whole content is that it was
            not.
        ceiling: The ceiling that was in force.
    """

    def __init__(self, attempted: int, ceiling: int) -> None:
        super().__init__(
            f"the v1 counted evaluator refused evaluation {attempted} within one "
            f"move; the pre-registered ceiling is {ceiling} (kom-023 §D3.3, "
            f"§D3.5). The search is unwound and the move discarded -- nothing "
            f"truncated is ever scored"
        )
        self.attempted = attempted
        self.ceiling = ceiling


class EvalCounter:
    """Counts every question this move's search puts to the network.

    Args:
        ceiling: §D3.3's per-move ceiling. A non-positive ceiling is rejected
            at construction: it would make even the forced-move root evaluation
            a breach, and P1's ``evals >= 1`` floor unsatisfiable.

    Raises:
        ValueError: If `ceiling` is below 1.
    """

    def __init__(self, ceiling: int = EVAL_CEILING) -> None:
        if ceiling < 1:
            raise ValueError(
                f"a per-move ceiling of {ceiling} forbids the single root "
                f"evaluation §D3.4 pins on a forced move, so P1's evals >= 1 "
                f"floor could never be met"
            )
        self._ceiling = int(ceiling)
        self._attempted = 0
        self._served = 0
        self._refused_at: int | None = None
        self._in_move = False
        self._moves = 0

    @property
    def ceiling(self) -> int:
        """Return the per-move ceiling in force."""
        return self._ceiling

    @property
    def attempted(self) -> int:
        """Return how many asks this move made, refused ones included."""
        return self._attempted

    @property
    def served(self) -> int:
        """Return how many asks this move was **served** -- never above the ceiling."""
        return self._served

    @property
    def refused_at(self) -> int | None:
        """Return the ordinal of this move's first refusal, or ``None``."""
        return self._refused_at

    @property
    def breached(self) -> bool:
        """Return whether this move breached the ceiling."""
        return self._refused_at is not None

    @property
    def moves_counted(self) -> int:
        """Return how many moves have been begun. Diagnostics only."""
        return self._moves

    def begin_move(self) -> None:
        """Start a new move's tally -- §D3.2 clause 3's boundary.

        Resets everything, including a previous move's refusal: a breach voids
        the **match** at the harness, not the move, so the runner keeps serving
        and the next move's count starts clean rather than staying poisoned.
        Carrying the refusal forward would turn one real M8 into a cascade of
        fabricated ones and destroy the evidence for which ply actually
        breached.
        """
        self._attempted = 0
        self._served = 0
        self._refused_at = None
        self._in_move = True
        self._moves += 1

    def end_move(self) -> None:
        """Close the current move. Later asks raise until the next `begin_move`."""
        self._in_move = False

    def ask(self, count: int = 1) -> int:
        """Count `count` questions to the network and return the last ordinal.

        Args:
            count: How many **positions** this one call asks about. §D3.2
                defines an evaluation as "one question put to a network about
                one game position", so a batched forward costs its batch
                length, not one (review v2 F3). A counter that charged one per
                *call* would turn §D3.3's ceiling into a ceiling on calls and
                let a width-``k`` batching net buy ``257 x k`` evaluations with
                every §D3.6 invariant still holding.

        Returns:
            The ordinal of the last ask within the current move: 1 for the root
            evaluation of a single-position call, up to :attr:`ceiling`.

        Raises:
            BudgetRefused: When this move's asks would pass the ceiling, and on
                every ask after that. The refusal is a **construction check**,
                not a truncation mechanism (§D3.5): serving 257 and playing the
                truncated move would trade tree size against budget in a way no
                counter can see, leaving both counters reading 257 and the two
                searches incomparable. A batch that straddles the ceiling is
                refused **whole** -- there is no partial service, because a
                half-served batch is exactly the truncation §D3.5 forbids.
            RuntimeError: If called outside a move (no :meth:`begin_move`).
            ValueError: If `count` is below 1.
        """
        if not self._in_move:
            raise RuntimeError(_NO_MOVE_IN_PROGRESS)
        if count < 1:
            raise ValueError(
                f"an ask covers at least one position, got count={count}; a "
                f"forward about nothing is not an evaluation and must not be "
                f"charged as one"
            )
        self._attempted += count
        if self._attempted > self._ceiling:
            if self._refused_at is None:
                # The ordinal of the first ask this move could not be served.
                # For the one-at-a-time case this is `attempted`; for a batch
                # that straddles the ceiling it is the first position inside
                # the batch, which is the number the breach is *about*.
                self._refused_at = self._ceiling + 1
            raise BudgetRefused(self._attempted, self._ceiling)
        self._served = self._attempted
        return self._served

    def counted(self, forward: Callable[_P, _R]) -> Callable[_P, _R]:
        """Return `forward` wrapped so every call is asked for first.

        The ask happens **before** the call, so a forward that raises still
        costs its evaluation: the search consumed the budget by asking, and a
        counter that only counted successful forwards would let a flaky net
        buy extra questions.
        """

        def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            self.ask()
            return forward(*args, **kwargs)

        return call


def positions_in(observation: Any, rank: int = DEFAULT_OBSERVATION_RANK) -> int:
    """Return how many game positions one ``predict`` argument carries.

    §D3.2 counts *positions*, not calls, so this is the arity the counter is
    charged (review v2 F3). Three shapes are recognised, and the fallback is
    deliberately 1 rather than an exception:

    * an array-like whose ``shape`` has **more** axes than one observation --
      a stacked batch, charged its leading dimension;
    * a ``list``/``tuple`` of observations -- charged its length;
    * anything else -- one position, which is alphazero-general's own
      ``predict(observation) -> (policy, value)`` shape.

    Falling back to 1 rather than refusing keeps an unrecognised-but-single
    observation working; the case that must never silently pass is the *batch*,
    and a batch in either of the two recognised forms is charged in full.
    """
    if isinstance(observation, (list, tuple)):
        return max(1, len(observation))
    shape = getattr(observation, "shape", None)
    if shape is None:
        return 1
    try:
        dimensions = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return 1
    if len(dimensions) <= max(1, rank):
        return 1
    return max(1, dimensions[0])


class CountedNet:
    """v1's network with §D3.5's counter welded to its one entry point.

    The wrapped object is whatever ``--net-factory`` built -- in
    alphazero-general's shape, an ``NNetWrapper`` with ``predict(observation)
    -> (policy, value)``. That method is the **net-call boundary** the task
    pins the counter at, and it is the only attribute this class exposes.

    **There is deliberately no ``__getattr__`` proxy.** A proxy would let v1's
    MCTS reach the underlying net by any other name and put an *uncounted*
    question to the network, which is precisely the failure §D3.5's
    instrumentation exists to make impossible -- and it would fail silently,
    with both counters still reading a plausible number. Without the proxy, a
    search that needs anything else from the net fails loudly with an
    ``AttributeError`` naming the attribute, which is a U-A question the
    ``--net-factory`` author must answer by routing it explicitly.
    """

    def __init__(
        self,
        net: Any,
        counter: EvalCounter,
        *,
        observation_rank: int = DEFAULT_OBSERVATION_RANK,
    ) -> None:
        self._net = net
        self._counter = counter
        self._observation_rank = int(observation_rank) or DEFAULT_OBSERVATION_RANK

    @property
    def counter(self) -> EvalCounter:
        """Return the counter this net reports to."""
        return self._counter

    @property
    def inner(self) -> Any:
        """Return the wrapped net. For the factory's own construction only.

        Named ``inner`` rather than exposed as an attribute proxy so that every
        uncounted use of the network is a visible, greppable ``.inner`` in the
        source of whoever wrote it.
        """
        return self._net

    @property
    def observation_rank(self) -> int:
        """Return how many axes one position's observation has (§D2.1: 3)."""
        return self._observation_rank

    def predict(self, observation: Any) -> Any:
        """Ask the network about `observation`, counting **every position** first.

        The charge is :func:`positions_in`, not 1: §D3.2's unit is a position,
        and a wrapper whose ``predict`` takes a stacked batch -- the shape any
        parallel or wave MCTS wants -- would otherwise make 257 asks buy
        ``257 x k`` evaluations while every §D3.6 invariant still held (review
        v2 F3).

        Raises:
            BudgetRefused: If this move has already used its ceiling, or if
                this call's positions would carry it past.
        """
        self._counter.ask(positions_in(observation, self._observation_rank))
        return self._net.predict(observation)
