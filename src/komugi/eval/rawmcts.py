"""§D6.3's RawMCTS: the pool's net-free ruler, and the only engine with no torch.

``docs/design/kom-011d-ladder-pool.md`` §D6.3 defines the family this module
implements, because ``docs/DESIGN.html:260`` asks for *"RawMCTS at several sim
counts"* and defines it nowhere:

    RawMCTS is a network-free MCTS: uniform prior over the legal set, no value
    network, value-by-random-rollout to a terminal or to the engine's own turn
    cap, PUCT selection with a fixed ``c_puct``, and a per-move budget of ``n``
    simulations. It shares komugi's engine and nothing else.

Every clause is chosen for the property that this opponent **cannot improve**:
a milestone's gain against it is not contaminated by anything in our lineage,
which is the whole anti-mirror argument ``DESIGN.html:260`` makes when it puts
out-of-family opponents in the pool as first-class members.

Four properties are load-bearing, and each is one construction below.

**No torch, at import time or ever** (§5's torch rule, rev 3 M5). This module
imports numpy and the standard library and nothing else -- in particular it does
**not** import :mod:`komugi.selfplay.adapter`, which reaches torch through
``adapter.py:63`` -> ``inference.py:146``, and it does not reach it through a
deferred function-body import either. The seam that seats this mover in a match
(:mod:`komugi.eval.rawmcts_entrant`) *is* torch-tolerant, because a
``BridgeEntrant`` comes from ``v1bridge.py:67`` -> ``arena.py:81``; the split
between the two modules is the whole of that constraint's resolution, and §6
node 29 is what keeps it true in both directions.

**The state is typed structurally.** :class:`MctsState` names exactly the six
operations the search needs, spelled as the real
:class:`komugi.selfplay.adapter.GameState` spells them (``adapter.py:175-265``).
It is deliberately narrower than
:class:`komugi.selfplay.adapter.EvaluableState`: ``observation()``,
``absolute_outcome()``, ``final_control()`` and ``in_draft()`` are the
recorder's and the self-play loop's, and naming them would couple a net-free
ruler to the encoding. §6 node 54 asserts the real ``GameState`` satisfies this
protocol, statically and dynamically.

**Deterministic given the seam's seed.** Rollouts consume randomness, so
:meth:`RawMctsMover.pick` builds a fresh ``Generator(PCG64(seed))`` from the
per-move seed the ``Mover`` seam already hands it (``v1bridge.py:1896``:
``move_seed(seeds, ply)``) -- the codebase's one seed derivation, composed
rather than re-rolled. Nothing else in this module draws randomness, and no
generator survives a ``pick``.

**No tree across a move** (KOM-023 §D3.2 clause 2, which forbids it on any
rated path). The tree is built inside ``pick`` and dropped when it returns, so
:attr:`RawMctsMover.tree_nodes_at_entry` is an honest literal ``0`` on every
move rather than an unmeasured field -- the same discipline
``gumbel_mover.py:83-89`` states for the same reason.

The sign convention, stated once
--------------------------------
``terminal_value()`` is *"for the player to move here"* (``adapter.py:256``),
so every value in the tree is mover-relative and the backup is negamax. This
module never assumes the movers alternate: each node records the mover it was
created under (:attr:`_Node.mover`) and a backed-up value is negated exactly
when the node's mover differs from the leaf's. The engine owns the turn order,
draft phase included, and a hard-coded ply-parity flip would be this module
quietly re-deriving it.
"""

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

DEFAULT_SIMS: Final[int] = 800
"""Simulations per move when a caller does not say.

§D1.1 pre-registers three ``rawmcts`` entrants at ``n in {100, 800, 6400}``;
800 is the middle rung and the gauge fallback of §D4.1.
"""

DEFAULT_C_PUCT: Final[float] = 1.0
"""The PUCT exploration constant, as §D1.3's digest block pins it.

Part of the entrant's identity, not a tuning knob: it is one of the fields
hashed into the ``@sha8`` of ``rawmcts/n<sims>@<sha8>``
(:func:`komugi.eval.rawmcts_entrant.rawmcts_digest`), so changing it mints a new
entrant rather than silently changing an old one.
"""

NO_TREE_REUSE: Final[int] = 0
"""``tree_nodes_at_entry`` for a search that carries nothing across a move.

Spelled here rather than imported from :mod:`komugi.eval.gumbel_mover`, which
reaches torch through the arena: this module's whole claim is that it imports
none, and a shared constant is not worth the import.
"""


class MctsState(Protocol):
    """The six state operations §D6.3's search needs, and nothing else.

    Structural on purpose (§D6.3, rev 3 R1): the caller hands
    :meth:`RawMctsMover.pick` a real
    :class:`komugi.selfplay.adapter.GameState`, and this protocol is what lets
    the module say so to a type checker without importing the module that
    defines it -- and therefore without importing torch.

    The names are the real object's own spellings (``adapter.py:175-265``).
    Rev 2 of the design named ``legal_actions`` and ``result``; **neither
    exists**, and a protocol naming them would either fail ``mypy --strict`` at
    ``rawmcts_entrant``'s ``-> Mover`` annotation or raise ``AttributeError`` at
    the first ``pick``. §6 node 54 pins both spellings.

    :meth:`clone` returns an :class:`MctsState` rather than a bare object
    because it is the load-bearing one: a rollout to a terminal position must
    never mutate the referee's board, so the search descends and rolls out on
    clones only.

    **Why this is not** :class:`komugi.search.state.SearchState`, which names
    the identical six operations and is itself torch-free
    (``state.py:52-95``): §D6.3 pins the ruler's state contract **locally**, and
    §5 puts ``src/komugi/search/**`` outside this work's blast radius entirely
    (*"the pool plays games; it changes no search"*). A frozen entrant whose
    state contract is owned by the search package is frozen only until that
    package's protocol grows a seventh method -- at which point a ruler that
    was supposed to be unable to change would stop type-checking, or worse,
    would quietly start being handed a different object. Twelve lines of
    duplication buys the pool a contract nothing else can move; §6 node 54 is
    what keeps the copy honest against the real state.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return a shape ``(A,)`` boolean mask, ``True`` where legal."""
        ...

    def play(self, action: int) -> None:
        """Advance this state **in place** by `action`, which must be legal."""
        ...

    def clone(self) -> MctsState:
        """Return a deep, independent copy."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the game has ended in this position."""
        ...

    def terminal_value(self) -> float:
        """Return the result in ``[-1, 1]`` for the player to move **here**."""
        ...

    def to_play(self) -> int:
        """Return the index (``0`` white, ``1`` black) of the player to move."""
        ...


def legal_actions(state: MctsState) -> npt.NDArray[np.int64]:
    """Return the ids of every action legal in `state`, ascending.

    The uniform prior of §D6.3 is uniform over exactly this set, so the set is
    derived in one place: a search that expanded over a different set than it
    scored would have a prior that is not uniform over anything.
    """
    return np.flatnonzero(np.asarray(state.legal_mask(), dtype=np.bool_)).astype(
        np.int64
    )


class _Node:
    """One expanded position: its legal edges, and the statistics on them.

    Statistics live on the **parent**, in the parent's own value frame, so a
    selection is one vectorised expression over the edge arrays rather than a
    walk over child objects -- and so an edge that has never been taken needs no
    child object at all. :attr:`mover` is what makes "the parent's frame" a
    fact read from the engine rather than a ply-parity assumption.
    """

    __slots__ = ("actions", "child_value", "child_visits", "children", "mover", "visits")

    def __init__(self, actions: npt.NDArray[np.int64], mover: int) -> None:
        self.actions = actions
        self.mover = mover
        self.children: list[_Node | None] = [None] * int(actions.size)
        self.child_visits: npt.NDArray[np.int64] = np.zeros(
            int(actions.size), dtype=np.int64
        )
        self.child_value: npt.NDArray[np.float64] = np.zeros(
            int(actions.size), dtype=np.float64
        )
        self.visits = 0


def _select(node: _Node, c_puct: float) -> int:
    """Return the edge index PUCT chooses at `node`.

    ``Q + c_puct * P * sqrt(N) / (1 + n)`` with ``P`` uniform over the legal
    set (§D6.3) -- so the prior is a constant and the only thing separating two
    unvisited edges is the deterministic tie-break of :func:`numpy.argmax`,
    which takes the lowest action id. That is a *stated* convention rather than
    an accident: the randomness this mover spends is the rollouts', and a
    tie-break that consumed the generator would make two picks at the same seed
    depend on how many ties they happened to meet.
    """
    visited = node.child_visits > 0
    q = np.zeros(node.child_visits.shape, dtype=np.float64)
    np.divide(
        node.child_value,
        node.child_visits,
        out=q,
        where=visited,
    )
    prior = 1.0 / float(node.actions.size)
    exploration = (
        c_puct
        * prior
        * float(np.sqrt(max(node.visits, 1)))
        / (1.0 + node.child_visits.astype(np.float64))
    )
    return int(np.argmax(q + exploration))


def _rollout(state: MctsState, rng: np.random.Generator) -> float:
    """Play `state` out uniformly at random and return its value, mover-relative.

    Mutates `state`, which is always a clone by the time it gets here.

    The stop condition is the engine's, not a second one of ours: ``is_terminal``
    is *"exactly ``win_state()`` having any slot set"* and *"the engine owns
    every termination rule, the engine's turn cap included"* (``adapter.py:191-198``).
    A second cap here would be a rule this module invented, and the two would
    diverge the first time the engine's moved.

    Returns:
        The result in ``[-1, 1]`` **for the player to move in `state` when it
        was handed over** -- the frame the caller's backup is stated in. A
        position with no legal action that the engine has not adjudicated is
        worth ``0.0``: it is a state this module cannot score and must not
        guess at.
    """
    mover = state.to_play()
    while not state.is_terminal():
        actions = legal_actions(state)
        if actions.size == 0:
            return 0.0
        state.play(int(actions[rng.integers(actions.size)]))
    value = float(state.terminal_value())
    return value if state.to_play() == mover else -value


class RawMctsMover:
    """§D6.3's net-free ruler, as a :class:`~komugi.eval.v1bridge.Mover`.

    Structural, not nominal: nothing here subclasses ``Mover`` (a
    :class:`typing.Protocol` that lives in the torch-importing
    :mod:`komugi.eval.v1bridge`), exactly as :class:`GumbelMover` does not
    (``gumbel_mover.py:181-185``). :func:`komugi.eval.rawmcts_entrant.rawmcts_entrant`'s
    factory is annotated ``-> Mover``, which is where the type checker actually
    verifies the four methods.

    One tree per move, built in :meth:`pick` and dropped when it returns, so
    §D3.2 clause 2 holds by construction and :attr:`tree_nodes_at_entry` is
    honestly ``0``.

    This player consults **no network**, so KOM-023's 257-evaluation ceiling
    does not apply to it and there is nothing to count (§D7.4 records the
    exemption explicitly rather than leaving a reader to infer it was ignored).
    Its budget is ``sims`` simulations per move and nothing else.

    Args:
        sims: Simulations per move. The sim count is half this entrant's
            identity (``rawmcts/n<sims>@<sha8>``), so it is required to be
            positive and is never adjusted at run time.
        player_id: The rateable identity, for diagnostics and error messages.
        c_puct: The PUCT constant; part of the identity digest (§D1.3).

    Raises:
        ValueError: If `sims` is not positive, or `c_puct` is negative.
    """

    def __init__(
        self,
        *,
        sims: int = DEFAULT_SIMS,
        player_id: str = "",
        c_puct: float = DEFAULT_C_PUCT,
    ) -> None:
        if int(sims) <= 0:
            raise ValueError(
                f"a RawMCTS entrant plays a positive simulation budget, got "
                f"{sims!r}: the count is half its identity "
                f"(rawmcts/n<sims>@<sha8>, §D1.2) and a zero-sim ruler measures "
                f"nothing"
            )
        if float(c_puct) < 0.0:
            raise ValueError(f"c_puct must be non-negative, got {c_puct!r}")
        self._sims = int(sims)
        self._player_id = player_id
        self._c_puct = float(c_puct)
        self._pair: int | None = None
        self._seat: int | None = None
        self.sims_used: int = 0
        """Simulations the **last** ``pick`` ran: ``0`` forced, ``sims`` else."""
        self.tree_nodes_at_entry: int = NO_TREE_REUSE
        """Nodes carried into the last ``pick``. Always ``0`` -- no tree reuse."""
        self.sims_per_move: list[int] = []
        """Per-move ``sims_used`` in ply order, across the whole match.

        :class:`~komugi.eval.gumbel_mover.GumbelMover` spells this ``sims``
        (``gumbel_mover.py:266``); here that name is the constructor's budget
        argument, and one word meaning both an ``int`` budget and a ``list``
        of spends is exactly the confusion a reader does not need.
        """

    @property
    def player_id(self) -> str:
        """Return the rateable identity this mover plays as."""
        return self._player_id

    @property
    def n(self) -> int:
        """Return the per-move simulation budget."""
        return self._sims

    @property
    def c_puct(self) -> float:
        """Return the PUCT constant this mover selects with."""
        return self._c_puct

    @property
    def seating(self) -> tuple[int | None, int | None]:
        """Return the ``(pair, seat)`` of the game in progress, if any."""
        return (self._pair, self._seat)

    def pick(self, state: MctsState, *, ply: int, seed: int) -> int:
        """Return the action this search chooses in `state`.

        `state` is not mutated: every simulation descends and rolls out on a
        clone, so the loop's own position is untouched and the referee stays the
        referee.

        `ply` is not read, for the reason ``gumbel_mover.py:311-314`` gives:
        the randomness this mover consumes is `seed`, which the loop already
        derives from the ply (``move_seed(seeds, ply)``), and deriving anything
        from `ply` here would be a second, silently divergent seed schedule.

        A **forced** position -- one legal action -- is played without a
        simulation. There is nothing for a search to choose between, and
        spending the budget would only make the ruler slower.

        Args:
            state: The position to move in, held by the caller's referee.
            ply: The ply index, for the caller's records. Unused here.
            seed: This move's seed. The same state and seed give the same
                action, whatever else has happened in between.

        Returns:
            A legal action id.

        Raises:
            ValueError: If `state` is terminal or has no legal action. Both are
                the caller's bug -- the loop tests ``is_terminal`` before it
                asks anyone to move (``v1bridge.py:1894``) -- and a mover that
                answered anyway would put an illegal action into a rated game.
        """
        del ply
        self.tree_nodes_at_entry = NO_TREE_REUSE
        if state.is_terminal():
            raise ValueError(
                f"{self._player_id or 'a RawMCTS mover'} was asked to move in a "
                f"position the engine has already adjudicated; the game loop "
                f"tests is_terminal() before it asks (v1bridge.py:1894)"
            )
        actions = legal_actions(state)
        if actions.size == 0:
            raise ValueError(
                f"{self._player_id or 'a RawMCTS mover'} was asked to move in a "
                f"position with no legal action; the game loop tests "
                f"is_terminal() before it asks (v1bridge.py:1894)"
            )
        if actions.size == 1:
            self.sims_used = 0
            self.sims_per_move.append(0)
            return int(actions[0])

        rng = np.random.Generator(np.random.PCG64(int(seed)))
        root = _Node(actions, state.to_play())
        for _ in range(self._sims):
            self._simulate(root, state, rng)
        self.sims_used = self._sims
        self.sims_per_move.append(self._sims)
        return int(root.actions[int(np.argmax(root.child_visits))])

    def observe(self, action: int, record: object) -> None:
        """Note a played move. A no-op: this stage carries no tree.

        Deliberately not "unimplemented", for
        ``gumbel_mover.py:362-372``'s reason: §D1 notifies both movers of every
        action so a stateful adapter can keep its own board in step, and this
        mover has no board of its own -- it searches the position it is handed.

        `record` is typed :class:`object` rather than
        ``komugi.eval.v1bridge.MoveRecord`` because naming that type would
        import the arena, and therefore torch, into the one module §5 says
        imports neither. Nothing here reads it.
        """
        del action, record

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game: record the coordinates, drop nothing.

        There is nothing to drop -- no board, no tree, no net. The per-ply
        :attr:`sims_per_move` list survives, so a match's budget record is a
        match's and not the last game's (``gumbel_mover.py:374-382``).
        """
        self._pair = pair
        self._seat = seat

    def close(self) -> None:
        """Release what this holds: nothing. Idempotent, because the seam
        closes a mover on renewal *and* again in the match's ``finally``
        (``v1bridge.py:2084-2090``).
        """

    def _simulate(
        self, root: _Node, state: MctsState, rng: np.random.Generator
    ) -> None:
        """Run one simulation: descend by PUCT, expand once, roll out, back up."""
        position = state.clone()
        node = root
        path: list[tuple[_Node, int]] = []
        leaf_mover = root.mover
        value = 0.0
        while True:
            index = _select(node, self._c_puct)
            path.append((node, index))
            position.play(int(node.actions[index]))
            if position.is_terminal():
                leaf_mover = position.to_play()
                value = float(position.terminal_value())
                break
            child = node.children[index]
            if child is None:
                child_actions = legal_actions(position)
                node.children[index] = _Node(child_actions, position.to_play())
                leaf_mover = position.to_play()
                value = _rollout(position, rng)
                break
            if child.actions.size == 0:
                # An expanded position the engine has not adjudicated and that
                # has no legal move: unreachable through this engine, and
                # scored 0.0 rather than crashed on, because a ruler that
                # raised here would void a whole match over one position it
                # merely cannot score.
                leaf_mover = child.mover
                value = 0.0
                break
            node = child
        for parent, index in reversed(path):
            parent.child_visits[index] += 1
            parent.child_value[index] += (
                value if parent.mover == leaf_mover else -value
            )
            parent.visits += 1
