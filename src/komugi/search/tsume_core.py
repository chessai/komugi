"""df-pn (Nagai 2002) AND/OR mate search, with no engine in it.

This module is the *algorithm* half of :mod:`komugi.search.tsume`: proof and
disproof numbers, the depth-first threshold recursion, the transposition
table, the path-repetition rule and the line extraction. It knows nothing
about Gungi. Everything it needs about a position arrives through
:class:`TsumeProblem` -- three methods, ``children`` / ``terminal`` /
``digest`` -- so the same search drives the compiled engine (the production
binding in :mod:`komugi.search.tsume`) and the hand-written AND/OR trees the
unit tests build, which is the only way the pn/dn algebra can be tested
against trees whose minimal proofs were computed on paper.

**The tree.** The root is an OR node with the *attacker* to move; its
children are AND nodes with the *defender* to move, and so on: node role is
the parity of the ply depth, never a field on the node. The attacker is
trying to prove ``MATE``; the defender is trying to disprove it. Proof
numbers are the usual ones::

    OR  (attacker):  pn = min over children,  dn = sum over children
    AND (defender):  pn = sum over children,  dn = min over children

with ``pn == 0`` meaning proven (the attacker mates) and ``dn == 0`` meaning
disproven (the attack fails).

**The depth cap** is the tsume depth: ``max_plies`` counts plies of the
attack line, so the attacker may only move from an OR node at depth <
``max_plies``. An OR node at or past the cap is disproven -- the attacker ran
out of moves, not out of ideas. An AND node with no children is checkmate and
is proven; an OR node with no children (no checking move) is disproven.

That reading of "no children at an AND node" is the *problem's* to guarantee,
not this module's: it is only mate if the defender was in check. A binding
whose attacker may play quiet moves reaches AND nodes where the defender is
not in check, and it must report the no-reply ones as terminal ``DISPROVED``
(stalemate) before this module ever sees them -- which is exactly what
:meth:`komugi.search.tsume.CyTsumeProblem.terminal` does.

**Repetition and the TT.** A node whose digest equals an ancestor's on the
current path ends the attack there (disproven): conservative, and it is what
keeps the graph-history-interaction problem out of the search. Such a result
is *path-dependent*, so it is **tainted**, and so is every result computed
from one: a tainted result is never written to the transposition table, and
never reused. Storing one would let a disproof that only holds under this
path launder itself into a proof-of-no-mate everywhere -- the failure
``tests/search/tsume/test_core_synthetic.py`` reproduces with a synthetic
cycle. The TT key carries the remaining depth alongside the digest, because a
result proven or disproven under a 1-ply budget says nothing about the same
position with 5 plies left. Everything *else* that makes two searches ask
different questions of the same position has to live in the digest, which is
what :class:`TsumeTable` spells out for the callers that share one.

**The budget** is counted in node *expansions*: one per call to
``children``. Exceeding it aborts the whole search, and ``solve_problem``
answers ``UNKNOWN`` -- which is never a wrong answer, only an unhelpful one.
Because the counter is checked before every expansion, ``TsumeResult.nodes``
is never above the budget it was given.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Generic, Literal, Protocol, TypeVar

__all__ = [
    "INF",
    "Outcome",
    "TsumeProblem",
    "TsumeResult",
    "TsumeTable",
    "Verdict",
    "solve_problem",
]

#: The "infinite" proof/disproof number. Small enough that sums of a few
#: hundred of them stay inside a machine word, large enough that no real node
#: count reaches it.
INF: Final = 1 << 30

#: A solver verdict. ``UNKNOWN`` means the budget ran out: it is never a
#: claim about the position, only about the search.
Verdict = Literal["MATE", "NO_MATE", "UNKNOWN"]

NodeT = TypeVar("NodeT")


class Outcome(Enum):
    """The two ways a node can be terminal *before* its children are looked at."""

    #: The attacker has won here (used by synthetic trees; the Gungi binding
    #: proves mate through "AND node with no escaping reply" instead).
    PROVED = "PROVED"
    #: The attack has failed here -- e.g. the game is already over because a
    #: marshal was captured.
    DISPROVED = "DISPROVED"


class TsumeProblem(Protocol[NodeT]):
    """The whole of what the search may learn about a position.

    Implementations are free to make ``NodeT`` anything -- a board, an int, a
    string -- as long as the three methods below are pure and deterministic:
    the search calls ``children`` on the same node more than once and relies
    on getting the same sequence in the same order back.
    """

    def children(self, node: NodeT) -> Sequence[tuple[str, NodeT]]:
        """The ``(label, child)`` pairs reachable from *node*, in a fixed order.

        At an OR node these are the attacker's moves (for tsume: the checking
        moves only); at an AND node the defender's replies. The label is what
        ends up in :attr:`TsumeResult.line`.
        """

    def terminal(self, node: NodeT) -> Outcome | None:
        """The node's outcome if it is terminal before expansion, else ``None``."""

    def digest(self, node: NodeT) -> Hashable:
        """An exact, collision-free identity for the position at *node*.

        Exact because it is used both as the transposition-table key and as
        the repetition test: a collision here is a wrong verdict, not a slow
        search.
        """


@dataclass(frozen=True)
class TsumeResult:
    """What a solve returns. ``line`` is non-empty exactly when MATE."""

    verdict: Verdict
    #: The principal variation as labels, attacker first, alternating sides:
    #: the attacker's fastest mate against the defender's longest defence, so
    #: its length is the mate distance in plies (odd).
    line: tuple[str, ...]
    #: Node expansions consumed, never more than the budget.
    nodes: int


class _BudgetExhausted(Exception):
    """Raised inside the recursion the moment the node budget is spent."""


_Key = tuple[Hashable, int]
#: Proof/disproof pair for a child that has never been looked at.
_UNVISITED: Final = (1, 1)


@dataclass
class TsumeTable:
    """Transposition state that may outlive one :func:`solve_problem` call.

    A fresh one is made per solve unless a caller passes its own, which is the
    only way to *share* learned proof/disproof numbers between solves -- and
    the only way a test can demonstrate that sharing is safe.

    **Sharing is safe exactly as far as the keys are.** An entry is keyed by
    ``(problem.digest(node), plies remaining)``, so the remaining depth is
    already accounted for, but nothing else is: two problems that answer
    *different questions* about the same position must not meet in one table.
    That is the whole reason
    :meth:`komugi.search.tsume.CyTsumeProblem.digest` folds its attacker-move
    mode into the digest -- a "no mate within 3 plies by checks alone" entry
    would otherwise satisfy a full-width query and turn a mode selector into
    a wrong answer.
    """

    #: ``key -> (pn, dn)`` for every result that did *not* depend on a
    #: path-repetition cut.
    entries: dict[_Key, tuple[int, int]] = field(default_factory=dict)
    #: Keys whose computed result depended on a path-repetition cut. They are
    #: absent from :attr:`entries` by construction; this set is what lets a
    #: parent inherit the taint from a child it read back rather than recursed
    #: into.
    tainted: set[_Key] = field(default_factory=set)
    #: ``(digest, remaining plies) -> shortest mate line`` memo for the exact
    #: minimax that runs after df-pn has proved a mate exists.
    lines: dict[_Key, tuple[int, tuple[str, ...]] | None] = field(
        default_factory=dict
    )


class _DfPn(Generic[NodeT]):
    """One search. Not reusable: it carries the budget counter and the TT."""

    def __init__(
        self,
        problem: TsumeProblem[NodeT],
        max_plies: int,
        node_budget: int,
        table: TsumeTable | None = None,
    ) -> None:
        self._problem = problem
        self._max_plies = max_plies
        self._budget = node_budget
        self.nodes = 0
        table = TsumeTable() if table is None else table
        self._tt = table.entries
        self._tainted = table.tainted
        # Digest -> depth of the occurrence on the path currently being
        # searched. Membership is the repetition test. Path-local by
        # definition, so it is never shared and never comes from *table*.
        self._path: dict[Hashable, int] = {}
        self._line_memo = table.lines

    # -- small helpers --------------------------------------------------

    def _is_or(self, depth: int) -> bool:
        """OR nodes (attacker to move) sit at even depth; the root is depth 0."""
        return depth % 2 == 0

    def _key(self, digest: Hashable, depth: int) -> _Key:
        return (digest, self._max_plies - depth)

    def _expand(self, node: NodeT) -> Sequence[tuple[str, NodeT]]:
        """Charge one node to the budget and generate *node*'s children."""
        if self.nodes >= self._budget:
            raise _BudgetExhausted
        self.nodes += 1
        return self._problem.children(node)

    def _pre_expansion_value(
        self, node: NodeT, depth: int, digest: Hashable
    ) -> tuple[int, int, bool] | None:
        """``(pn, dn, tainted)`` if *node* resolves without generating moves.

        Three ways that happens, in this order: the problem calls the node
        terminal (for the Gungi binding: the game is already over, so the
        attack did not mate); the node repeats an ancestor on this path (the
        attack fails, and the answer is tainted); or the attacker is out of
        plies.
        """
        outcome = self._problem.terminal(node)
        if outcome is Outcome.PROVED:
            if self._is_or(depth):
                raise ValueError(
                    "terminal() returned PROVED at an OR-parity node: a win "
                    "with the attacker still to move is not a tsume mate"
                )
            return (0, INF, False)
        if outcome is Outcome.DISPROVED:
            return (INF, 0, False)
        if digest in self._path:
            return (INF, 0, True)
        if self._is_or(depth) and depth >= self._max_plies:
            return (INF, 0, False)
        return None

    def _record(self, key: _Key, pn: int, dn: int, tainted: bool) -> None:
        if tainted:
            self._tainted.add(key)
        else:
            self._tt[key] = (pn, dn)

    # -- the search -----------------------------------------------------

    def search_root(self, root: NodeT) -> tuple[int, int]:
        pn, dn, _ = self._mid(root, 0, INF, INF)
        return pn, dn

    def _mid(
        self, node: NodeT, depth: int, th_pn: int, th_dn: int
    ) -> tuple[int, int, bool]:
        """Nagai's MID: search *node* until pn or dn crosses its threshold."""
        digest = self._problem.digest(node)
        key = self._key(digest, depth)

        resolved = self._pre_expansion_value(node, depth, digest)
        if resolved is not None:
            pn, dn, tainted = resolved
            self._record(key, pn, dn, tainted)
            return resolved

        stored = self._tt.get(key)
        if stored is not None:
            pn, dn = stored
            if pn >= th_pn or dn >= th_dn:
                return (pn, dn, False)

        or_node = self._is_or(depth)
        children = self._expand(node)
        if not children:
            # No checking move for the attacker (disproof), or no reply at
            # all for the defender -- which is checkmate, and the whole point
            # of the search.
            pn, dn = (INF, 0) if or_node else (0, INF)
            self._record(key, pn, dn, False)
            return (pn, dn, False)

        child_keys = [
            self._key(self._problem.digest(c), depth + 1) for _, c in children
        ]
        # Values learned during *this* expansion. They must be kept locally as
        # well as in the TT, because a tainted child result is deliberately
        # not stored -- without this the loop below would re-read (1, 1) for
        # it forever and never make progress.
        local: dict[int, tuple[int, int]] = {}
        tainted = False

        self._path[digest] = depth
        try:
            while True:
                values = [
                    local.get(i, self._tt.get(child_keys[i], _UNVISITED))
                    for i in range(len(children))
                ]
                if or_node:
                    pn = min(v[0] for v in values)
                    dn = min(INF, sum(v[1] for v in values))
                else:
                    pn = min(INF, sum(v[0] for v in values))
                    dn = min(v[1] for v in values)
                if pn >= th_pn or dn >= th_dn:
                    break
                index, child_th_pn, child_th_dn = _select_child(
                    values, or_node, pn, dn, th_pn, th_dn
                )
                child_pn, child_dn, child_tainted = self._mid(
                    children[index][1], depth + 1, child_th_pn, child_th_dn
                )
                local[index] = (child_pn, child_dn)
                tainted = tainted or child_tainted
        finally:
            del self._path[digest]

        tainted = tainted or any(k in self._tainted for k in child_keys)
        self._record(key, pn, dn, tainted)
        return (pn, dn, tainted)

    # -- line extraction ------------------------------------------------

    def best_line(self, node: NodeT, depth: int) -> tuple[int, tuple[str, ...]] | None:
        """The optimal line from *node*, or ``None`` if the attack fails there.

        Optimal in the tsume sense, which is what makes the length of a mate
        line a *grade*: the attacker takes the shortest mate, the defender the
        longest defence. df-pn proves a mate *exists*; it says nothing about
        how long the mate is, so this exact minimax over mate distance runs
        afterwards, over the same problem and against the same node budget.

        It is exact but not full-width: an attacker line of length *d* caps
        the search of its siblings at *d - 2* (mate distances from an OR node
        are odd, so a strictly better line is at least two plies shorter),
        which is why a puzzle whose first tried move mates costs two node
        expansions here rather than one per branch.
        """
        result, _ = self._best_line(node, depth, self._max_plies - depth)
        return result

    def _best_line(
        self, node: NodeT, depth: int, limit: int
    ) -> tuple[tuple[int, tuple[str, ...]] | None, bool]:
        """Shortest mate from *node* within *limit* further plies, and its taint.

        *limit* counts plies of both sides, like ``max_plies`` itself. It
        shrinks by one per ply and is part of the memo key, because "no mate
        within 3 plies" is not the same statement as "no mate within 5".
        """
        digest = self._problem.digest(node)
        key = (digest, limit)

        outcome = self._problem.terminal(node)
        if outcome is Outcome.PROVED:
            if self._is_or(depth):
                raise ValueError(
                    "terminal() returned PROVED at an OR-parity node: a win "
                    "with the attacker still to move is not a tsume mate"
                )
            return ((0, ()), False)
        if outcome is Outcome.DISPROVED:
            return (None, False)
        if digest in self._path:
            return (None, True)
        or_node = self._is_or(depth)
        if or_node and limit < 1:
            return (None, False)
        if key in self._line_memo:
            return (self._line_memo[key], False)

        children = self._expand(node)
        best: tuple[int, tuple[str, ...]] | None = None
        tainted = False
        self._path[digest] = depth
        try:
            if not children:
                # No reply at an AND node is checkmate: zero further plies.
                best = None if or_node else (0, ())
            elif or_node:
                budget = limit
                for label, child in children:
                    if budget < 1:
                        break
                    sub, sub_tainted = self._best_line(child, depth + 1, budget - 1)
                    tainted = tainted or sub_tainted
                    if sub is None:
                        continue
                    best = (sub[0] + 1, (label, *sub[1]))
                    budget = best[0] - 2
            else:
                longest: tuple[int, tuple[str, ...]] | None = None
                for label, child in children:
                    sub, sub_tainted = self._best_line(child, depth + 1, limit - 1)
                    tainted = tainted or sub_tainted
                    if sub is None:
                        # One reply survives, so the defender plays it and
                        # the attack fails here, whatever the others do.
                        longest = None
                        break
                    candidate = (sub[0] + 1, (label, *sub[1]))
                    if longest is None or candidate[0] > longest[0]:
                        longest = candidate
                best = longest
        finally:
            del self._path[digest]

        if not tainted:
            self._line_memo[key] = best
        return (best, tainted)


def _select_child(
    values: Sequence[tuple[int, int]],
    or_node: bool,
    pn: int,
    dn: int,
    th_pn: int,
    th_dn: int,
) -> tuple[int, int, int]:
    """Nagai's most-proving child and the thresholds to search it with.

    At an OR node the child with the smallest pn is the most-proving one, and
    it is searched until its pn passes the *second* smallest (at which point
    that sibling becomes more promising) or the node's own threshold. At an
    AND node the same argument runs on dn. Ties go to the earliest child, so
    the search -- and therefore the line it returns -- is deterministic.
    """
    if or_node:
        best_index = min(range(len(values)), key=lambda i: values[i][0])
        rest = [values[i][0] for i in range(len(values)) if i != best_index]
        second = min(rest) if rest else INF
        child_th_pn = min(th_pn, second + 1)
        child_th_dn = min(INF, th_dn - dn + values[best_index][1])
    else:
        best_index = min(range(len(values)), key=lambda i: values[i][1])
        rest = [values[i][1] for i in range(len(values)) if i != best_index]
        second = min(rest) if rest else INF
        child_th_dn = min(th_dn, second + 1)
        child_th_pn = min(INF, th_pn - pn + values[best_index][0])
    return best_index, child_th_pn, child_th_dn


def solve_problem(
    problem: TsumeProblem[NodeT],
    root: NodeT,
    max_plies: int,
    node_budget: int = 100_000,
    *,
    table: TsumeTable | None = None,
) -> TsumeResult:
    """Solve *problem* from *root* within *max_plies* plies and *node_budget* nodes.

    ``max_plies`` is the tsume depth and must be odd and at least 1: a mate is
    delivered by the attacker, so an even cap would spend its last ply on the
    defender and could never prove anything an odd cap did not.

    *table* carries transposition state in and out; the default is a fresh
    one per call, so a solve learns nothing from its predecessors. See
    :class:`TsumeTable` for what makes sharing one sound.
    """
    if max_plies < 1:
        raise ValueError(f"max_plies must be >= 1, got {max_plies}")
    if max_plies % 2 == 0:
        raise ValueError(
            f"max_plies must be odd (a mate lands on an attacker ply), got {max_plies}"
        )

    search: _DfPn[NodeT] = _DfPn(problem, max_plies, node_budget, table)
    try:
        pn, dn = search.search_root(root)
        if pn == 0:
            line = search.best_line(root, 0)
            if line is None:
                raise RuntimeError(
                    "df-pn proved a mate the exact line search could not find; "
                    "the two disagree about the same problem, which is a bug"
                )
            return TsumeResult("MATE", line[1], search.nodes)
    except _BudgetExhausted:
        return TsumeResult("UNKNOWN", (), search.nodes)
    if dn == 0:
        return TsumeResult("NO_MATE", (), search.nodes)
    return TsumeResult("UNKNOWN", (), search.nodes)
