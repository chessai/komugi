"""Planted fixtures and helpers for the match-play search suite.

A **package**, deliberately, and never a ``planted.py`` beside it: a module and
a package of the same name cannot coexist in one directory, and the design
pre-registered both before catching itself. The sweep scripts that *find*
fixtures live here as siblings (``find_*.py``); they are committed for
provenance and are never run at test time.

What lives here:

* :func:`wdl_from_value` -- the one place a scalar value is turned into a WDL
  triple, so that every fixture agrees on what "a draw" means.
* :func:`wdl_evaluator_from_table`, :func:`solver_wdl_evaluator`,
  :func:`constant_wdl_evaluator` -- the three shapes of
  :data:`~komugi.search.match.state.WdlEvaluator` the suite needs.
  ``logits`` is a real parameter on all three: the oracle's own evaluator is
  uniform-logit, and a fixture that needs a *high-prior* child cannot be
  expressed without it.
* :class:`MatchToyState` -- the oracle's toy games wearing ``MatchState``.
* :class:`PlantedLegalState` -- a state whose legal set at each node is
  *written down*, which is the only way to construct the legal-set **superset**
  the rehome check has to catch: no real position is guaranteed to produce
  one.
* :class:`DeepGame` -- a game deeper than any playout budget, so that a test
  about evaluation counting is not silently answered by terminal descents.
* :class:`CountingEvaluator`, :func:`walk_nodes`, :func:`snapshot_node`,
  :func:`plant_backup` -- instruments.

The toy games themselves are **not modified**: they are self-play test assets,
and the disjointness this ticket is gated on extends to not editing them.

**Importing this package is also what puts ``tests/search/oracle`` on
``sys.path``**, which is why every module in this suite that reads ``solver``
or ``toygames`` imports ``planted`` *first*. pytest inserts a test module's own
directory into ``sys.path`` when it imports it, not its siblings', and
``tests/search/match`` is collected before ``tests/search/oracle`` -- so
without this the oracle's modules are not importable from here yet. A
``conftest.py`` would be the idiomatic home for it, but the test tree has no
packages and one already exists under ``tests/engine``: a second file of that
name is a *duplicate module* to ``mypy``, which is a hard CI failure rather
than a matter of taste.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

_ORACLE_DIR = Path(__file__).resolve().parents[2] / "oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

import numpy as np
import numpy.typing as npt
from solver import SolvableState, solve

from komugi.search.match.node import S1_FIELDS, MatchNode, pack_outer
from komugi.search.match.state import MatchState, WdlEval, WdlEvaluator

#: Anything a fixture may write a WDL triple down as. Both spellings are used:
#: a literal tuple where the numbers are the point, and an array where it comes
#: back out of a helper.
WdlLike: TypeAlias = "Sequence[float] | npt.NDArray[np.float64]"

#: A one-hot draw, spelled once so no fixture has to remember the slot order.
DRAW_WDL: npt.NDArray[np.float64] = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def wdl_from_value(value: float) -> npt.NDArray[np.float64]:
    """Return a WDL triple whose contempt-free utility is exactly `value`.

    The remainder of the probability mass goes to the **draw** slot, so an
    exact ``0.0`` is the one-hot draw ``(0, 1, 0)`` rather than a coin flip
    between a win and a loss. Those two have the same utility at zero
    contempt and completely different utilities at any other, and a fixture
    that meant "drawn" would otherwise silently become "double-edged" the
    moment a test turned contempt on.

    Args:
        value: A utility in ``[-1, 1]``.

    Returns:
        ``(3,)`` float64 probabilities summing to 1 with ``w - l == value``.
    """
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"value must lie in [-1, 1], got {value!r}")
    if value > 0.0:
        return np.array([value, 1.0 - value, 0.0], dtype=np.float64)
    if value < 0.0:
        return np.array([0.0, 1.0 + value, -value], dtype=np.float64)
    return DRAW_WDL.copy()


def _logits_for(
    num_actions: int, logits: npt.NDArray[np.float32] | None
) -> npt.NDArray[np.float32]:
    """Return `logits` verbatim, or an all-zero (uniform) vector."""
    if logits is None:
        return np.zeros(num_actions, dtype=np.float32)
    wide = np.asarray(logits, dtype=np.float32)
    if wide.shape != (num_actions,):
        raise ValueError(f"logits must be ({num_actions},), got {wide.shape!r}")
    return wide


def wdl_evaluator_from_table(
    table: Mapping[Hashable, WdlLike],
    *,
    num_actions: int,
    logits: npt.NDArray[np.float32] | None = None,
    default: WdlLike | None = None,
) -> WdlEvaluator:
    """Return an evaluator reading WDL triples off a written-down table.

    Args:
        table: Keyed by the wrapped toy game's own ``key()``, so a fixture
            names positions the same way the solver's memo does.
        num_actions: Width of the policy vector to return.
        logits: ``(A,)`` float32 returned verbatim for every position, or
            ``None`` for uniform. A *non-uniform* prior is not decoration:
            a fixture that needs a high-prior child cannot be written without
            it, and inventing one at implementation time would be exactly the
            un-pre-registered fixture the design forbids.
        default: The triple to return for a position the table does not name.
            ``None`` makes an unnamed position an error, which is what keeps a
            fixture honest about its own coverage.

    Returns:
        A :data:`WdlEvaluator` over :class:`MatchToyState`.
    """
    prior = _logits_for(num_actions, logits)
    fallback = None if default is None else np.asarray(default, dtype=np.float64)

    def evaluate(state: MatchState) -> WdlEval:
        key = _toy_key(state)
        named = table.get(key)
        if named is None:
            if fallback is None:
                raise KeyError(f"no WDL planted for position {key!r}")
            return WdlEval(prior, fallback.copy())
        return WdlEval(prior, np.asarray(named, dtype=np.float64))

    return evaluate


def solver_wdl_evaluator(
    *,
    num_actions: int,
    logits: npt.NDArray[np.float32] | None = None,
) -> WdlEvaluator:
    """Return the oracle's evaluator, WDL-shaped: exact value, stated prior.

    The perfect-value/uninformative-policy evaluator the Gumbel oracle is
    built on, with its scalar value passed through :func:`wdl_from_value`.
    Under it a search that picks a losing move has no excuse: the information
    needed to avoid it was handed over on the first visit.
    """
    prior = _logits_for(num_actions, logits)

    def evaluate(state: MatchState) -> WdlEval:
        return WdlEval(prior, wdl_from_value(solve(_toy_game(state)).value))

    return evaluate


def constant_wdl_evaluator(
    wdl: WdlLike,
    *,
    num_actions: int,
    logits: npt.NDArray[np.float32] | None = None,
) -> WdlEvaluator:
    """Return an evaluator that answers `wdl` for every position.

    Used where the *backup* is under test rather than the values: with every
    leaf identical, anything that moves in the tree moved because of the
    arithmetic being tested and not because of the position.
    """
    prior = _logits_for(num_actions, logits)
    value = np.asarray(wdl, dtype=np.float64)

    def evaluate(state: MatchState) -> WdlEval:
        return WdlEval(prior, value.copy())

    return evaluate


class CountingEvaluator:
    """A :data:`WdlEvaluator` that records how many times it was called.

    The instrument for "nothing was re-evaluated": a read path that quietly
    asked the network again would leave the stored buffers untouched and pass
    a byte-comparison, so the call count is a second, independent witness.
    """

    def __init__(self, inner: WdlEvaluator) -> None:
        self._inner = inner
        self.calls = 0

    def __call__(self, state: MatchState) -> WdlEval:
        self.calls += 1
        return self._inner(state)


def _toy_game(state: MatchState) -> SolvableState:
    """Return the toy game a :class:`MatchToyState` wraps."""
    if not isinstance(state, MatchToyState):
        raise TypeError(f"expected a MatchToyState, got {type(state).__name__}")
    return state.game


def _toy_key(state: MatchState) -> Hashable:
    """Return the wrapped toy game's memo key."""
    return _toy_game(state).key()


class MatchToyState:
    """A toy game wearing :class:`~komugi.search.match.state.MatchState`.

    Two methods the toy games do not have: :meth:`terminal_wdl`, derived from
    the game's own mover-relative result, and :meth:`observation`, which is
    trivial because no test here has a network to feed.

    Args:
        game: Any of the oracle's toy games, or one of this package's planted
            ones.
    """

    __slots__ = ("_game",)

    def __init__(self, game: SolvableState) -> None:
        self._game = game

    @property
    def game(self) -> SolvableState:
        """Return the wrapped toy game."""
        return self._game

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the wrapped game's legality mask."""
        return self._game.legal_mask()

    def play(self, action: int) -> None:
        """Advance the wrapped game in place."""
        self._game.play(action)

    def clone(self) -> MatchToyState:
        """Return an independent copy."""
        return MatchToyState(self._game.clone())

    def is_terminal(self) -> bool:
        """Return whether the wrapped game is over."""
        return self._game.is_terminal()

    def to_play(self) -> int:
        """Return the index of the player to move."""
        return self._game.to_play()

    def observation(self) -> npt.NDArray[np.float32]:
        """Return a placeholder encoding; no fixture here has a network."""
        return np.zeros(1, dtype=np.float32)

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the one-hot result for the mover here.

        The toy games state their result as ``+1 / 0 / -1`` from the mover's
        own perspective, which :func:`wdl_from_value` turns into a one-hot
        triple -- with a draw landing squarely on the draw slot and **no**
        draw score anywhere in it.
        """
        return wdl_from_value(self._game.terminal_value())

    def __repr__(self) -> str:
        return f"MatchToyState({self._game!r})"


class PlantedLegalState:
    """A state whose legal set at every node is written down, not derived.

    The design asks for a fixture on which the rehome check's **superset**
    direction is guaranteed by construction rather than probable: a position
    whose legal set has grown since the retained child was expanded. No engine
    position is guaranteed to produce one, and a probabilistic fixture dressed
    as a deterministic test is exactly the shape of coverage failure this
    suite is supposed to catch.

    Args:
        legal_sets: Maps a move history to the actions legal after it. A
            history the map does not name is **terminal**.
        num_actions: Width of the action space.
        results: Maps a terminal history to its value for the mover there.
            Unnamed terminal histories are drawn.
        moves: The history this instance stands at.
    """

    __slots__ = ("_legal_sets", "_num_actions", "_results", "_moves")

    def __init__(
        self,
        legal_sets: Mapping[tuple[int, ...], frozenset[int]],
        *,
        num_actions: int,
        results: Mapping[tuple[int, ...], float] | None = None,
        moves: tuple[int, ...] = (),
    ) -> None:
        self._legal_sets = dict(legal_sets)
        self._num_actions = int(num_actions)
        self._results = dict(results or {})
        self._moves = moves

    @property
    def moves(self) -> tuple[int, ...]:
        """Return the history this state stands at."""
        return self._moves

    @property
    def num_actions(self) -> int:
        """Return the width of the action space."""
        return self._num_actions

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the written-down legal set of this history."""
        mask: npt.NDArray[np.bool_] = np.zeros(self._num_actions, dtype=np.bool_)
        for action in self._legal_sets.get(self._moves, frozenset()):
            mask[action] = True
        return mask

    def play(self, action: int) -> None:
        """Extend the history by `action`, which must be legal here."""
        if action not in self._legal_sets.get(self._moves, frozenset()):
            raise ValueError(f"action {action} is not legal at {self._moves!r}")
        self._moves = (*self._moves, action)

    def clone(self) -> PlantedLegalState:
        """Return an independent copy sharing the written-down tables."""
        return PlantedLegalState(
            self._legal_sets,
            num_actions=self._num_actions,
            results=self._results,
            moves=self._moves,
        )

    def is_terminal(self) -> bool:
        """Return whether this history has no written-down legal set."""
        return self._moves not in self._legal_sets

    def to_play(self) -> int:
        """Return the mover, alternating by ply."""
        return len(self._moves) % 2

    def terminal_value(self) -> float:
        """Return the written-down result for the mover here."""
        if not self.is_terminal():
            raise ValueError("terminal result asked of a non-terminal position")
        return self._results.get(self._moves, 0.0)

    def key(self) -> Hashable:
        """Return the memo key: the history identifies the position."""
        return self._moves

    def __repr__(self) -> str:
        return f"PlantedLegalState(moves={self._moves!r})"


class DeepGame:
    """`width` legal actions at every ply, terminal after `depth` of them.

    A game deeper than any playout budget the suite uses, which is what makes
    "no playout reached a terminal position" a property of the fixture instead
    of a hope: design §B1's evaluation identity has an inequality in it
    exactly because terminal descents cost nothing, so a counting test on a
    shallow game measures the game rather than the search.

    Args:
        width: Actions legal at every non-terminal position.
        depth: Plies after which the game is over.
        result: The value of a terminal position for the mover there.
        moves: The history this instance stands at.
    """

    __slots__ = ("_width", "_depth", "_result", "_moves")

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        result: float = 0.0,
        moves: tuple[int, ...] = (),
    ) -> None:
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self._width = int(width)
        self._depth = int(depth)
        self._result = float(result)
        self._moves = moves

    @property
    def num_actions(self) -> int:
        """Return the width of the action space."""
        return self._width

    @property
    def moves(self) -> tuple[int, ...]:
        """Return the history this state stands at."""
        return self._moves

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return every action, or nothing once the depth is reached."""
        mask: npt.NDArray[np.bool_] = np.zeros(self._width, dtype=np.bool_)
        if not self.is_terminal():
            mask[:] = True
        return mask

    def play(self, action: int) -> None:
        """Extend the history by `action`."""
        if self.is_terminal():
            raise ValueError("the game is already over")
        if not 0 <= action < self._width:
            raise ValueError(f"action out of range: {action!r}")
        self._moves = (*self._moves, action)

    def clone(self) -> DeepGame:
        """Return an independent copy."""
        return DeepGame(
            width=self._width,
            depth=self._depth,
            result=self._result,
            moves=self._moves,
        )

    def is_terminal(self) -> bool:
        """Return whether the depth has been reached."""
        return len(self._moves) >= self._depth

    def to_play(self) -> int:
        """Return the mover, alternating by ply."""
        return len(self._moves) % 2

    def terminal_value(self) -> float:
        """Return the written-down result for the mover here."""
        if not self.is_terminal():
            raise ValueError("terminal result asked of a non-terminal position")
        return self._result

    def key(self) -> Hashable:
        """Return the memo key: the history identifies the position."""
        return self._moves

    def __repr__(self) -> str:
        return f"DeepGame(width={self._width}, depth={self._depth})"


def walk_nodes(root: MatchNode | None) -> list[MatchNode]:
    """Return every node under `root` in a fixed pre-order.

    Children are visited in **ascending edge index**, so the order is a
    property of the tree and not of a dict's iteration: a snapshot compared
    against itself across two reads has to be walked the same way both times,
    or the comparison tests the walk.
    """
    if root is None:
        return []
    found: list[MatchNode] = []
    stack: list[MatchNode] = [root]
    while stack:
        node = stack.pop()
        found.append(node)
        for index in sorted(node.children, reverse=True):
            stack.append(node.children[index])
    return found


def snapshot_node(node: MatchNode) -> tuple[bytes, ...]:
    """Return the bytes of every design §S1 field of `node`.

    Read off :data:`S1_FIELDS` rather than a second list written here, so a
    field added to the schema joins this snapshot by being declared once. The
    child dict is rendered as its sorted edge indices: the *shape* of the tree
    is part of what must not move, while the child nodes themselves are
    snapshotted in their own right by the caller's walk.
    """
    captured: list[bytes] = []
    for name in S1_FIELDS:
        value = getattr(node, name)
        if isinstance(value, np.ndarray):
            captured.append(value.tobytes())
        elif isinstance(value, dict):
            captured.append(repr(sorted(value)).encode())
        else:
            captured.append(repr(value).encode())
    return tuple(captured)


def plant_backup(
    node: MatchNode, index: int, values: Sequence[float], *, weight: float = 1.0
) -> None:
    """Accumulate `values` into `node`'s edge `index` as if they were backed up.

    Uses the production accumulation -- the same weighted sum, the same
    :func:`~komugi.search.match.node.pack_outer` -- so a node built this way
    is indistinguishable from one a search produced, while the numbers stay
    written down instead of being a search outcome nobody can predict by
    reading.

    Args:
        node: The parent whose edge statistics are being planted.
        index: Which edge, by position in ``node.actions``.
        values: Utilities to plant, each turned into a WDL triple by
            :func:`wdl_from_value`.
        weight: The backup weight of each value.
    """
    for value in values:
        triple = wdl_from_value(float(value))
        node.visits[index] += 1
        node.weight[index] += weight
        node.wdl_sum[index] += weight * triple
        node.moment[index] += weight * pack_outer(triple)
