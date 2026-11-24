"""The pn/dn algebra on trees small enough to minimax by eye.

No engine, no board, no FEN: the df-pn core is written against an injected
interface, so its AND/OR arithmetic, its depth cap, its line extraction and
above all its repetition handling can be tested on hand-built trees whose
correct answers were worked out on paper. Bugs in proof-number bookkeeping
are invisible on real puzzles -- a wrong pn just makes the search explore in
a different order and still arrive at the right verdict, until one day it
does not -- so they are cornered here instead.

The centrepiece is ``test_a_stored_repetition_disproof_would_flip_the_verdict``:
it builds the one tree where the transposition-table taint rule is the
difference between MATE and NO_MATE, and shows the flip happening when the
rule is defeated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from komugi.search.tsume_core import (
    INF,
    Outcome,
    TsumeResult,
    _DfPn,
    solve_problem,
)


@dataclass
class TreeProblem:
    """An AND/OR tree given by hand, satisfying the solver's problem protocol.

    ``digests`` is what makes repetition expressible: two distinct nodes with
    the same digest *are* the same position as far as the search is
    concerned, which is exactly the situation a cycle creates on a real
    board.
    """

    edges: Mapping[str, Sequence[tuple[str, str]]]
    terminals: Mapping[str, Outcome] = field(default_factory=dict)
    digests: Mapping[str, str] = field(default_factory=dict)
    #: Every node this problem was asked to expand, in order.
    expanded: list[str] = field(default_factory=list)

    def children(self, node: str) -> Sequence[tuple[str, str]]:
        self.expanded.append(node)
        return self.edges.get(node, ())

    def terminal(self, node: str) -> Outcome | None:
        return self.terminals.get(node)

    def digest(self, node: str) -> str:
        return self.digests.get(node, node)


def full_width_expansions(tree: TreeProblem, root: str, max_plies: int) -> int:
    """How many nodes a searcher with no pruning at all would expand.

    Every node reachable within the depth cap, counted once per visit, which
    is the baseline the df-pn node counts below are asserted to beat.
    """
    total = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth % 2 == 0 and depth >= max_plies:
            continue
        if node in tree.terminals:
            continue
        total += 1
        stack.extend((child, depth + 1) for _, child in tree.edges.get(node, ()))
    return total


def solve(tree: TreeProblem, root: str, max_plies: int) -> TsumeResult:
    return solve_problem(tree, root, max_plies)


# ---------------------------------------------------------------------------
# Verdicts and lines
# ---------------------------------------------------------------------------


def test_an_and_node_with_no_reply_is_mate() -> None:
    """The whole definition of mate in this search, on the smallest tree."""
    tree = TreeProblem(edges={"r": [("a", "A"), ("b", "B")], "B": [("x", "X")]})
    result = solve(tree, "r", 1)
    assert result.verdict == "MATE"
    assert result.line == ("a",)
    assert result.nodes >= 2


def test_an_or_node_with_no_move_is_no_mate() -> None:
    tree = TreeProblem(edges={})
    result = solve(tree, "r", 1)
    assert result.verdict == "NO_MATE"
    assert result.line == ()
    assert result.nodes >= 1


def test_a_surviving_reply_refutes_the_attack() -> None:
    """One escape is enough: the AND node's dn is the minimum, not the sum."""
    tree = TreeProblem(
        edges={
            "r": [("a", "A")],
            "A": [("escape", "E")],
            "E": [],
        }
    )
    assert solve(tree, "r", 1).verdict == "NO_MATE"


def test_the_depth_cap_refuses_a_mate_that_is_one_move_too_deep() -> None:
    tree = TreeProblem(
        edges={
            "r": [("a", "A")],
            "A": [("p", "P")],
            "P": [("m", "M")],
        }
    )
    assert solve(tree, "r", 1).verdict == "NO_MATE"
    deeper = solve(tree, "r", 3)
    assert deeper.verdict == "MATE"
    assert deeper.line == ("a", "p", "m")


def test_the_defender_takes_the_longest_defence() -> None:
    """The line is the grade: a mate in 5 must not be reported as a mate in 3.

    The defender's first reply chooses between being mated at ply 3 and being
    mated at ply 5. The composed puzzles are graded by the second one, so the
    extraction has to maximise at AND nodes -- a solver that reported the
    short branch would make every 5-ply fixture look like a 3-ply fixture.
    """
    tree = TreeProblem(
        edges={
            "r": [("a", "A")],
            "A": [("short", "S"), ("long", "L")],
            "S": [("m1", "M1")],
            "L": [("m2", "L2")],
            "L2": [("defend", "L3")],
            "L3": [("m3", "M3")],
        }
    )
    result = solve(tree, "r", 5)
    assert result.verdict == "MATE"
    assert result.line == ("a", "long", "m2", "defend", "m3")
    assert len(result.line) % 2 == 1


def test_the_attacker_takes_the_shortest_mate() -> None:
    """And the slow move is tried first, so "shortest" is not just "first"."""
    tree = TreeProblem(
        edges={
            "r": [("slow", "SLOW"), ("fast", "FAST")],
            "SLOW": [("d", "D")],
            "D": [("m", "M")],
        }
    )
    result = solve(tree, "r", 3)
    assert result.verdict == "MATE"
    assert result.line == ("fast",)


def test_terminal_outcomes_end_the_search_before_expansion() -> None:
    """A node the problem calls terminal is never expanded.

    This is the seam the Gungi binding uses for "the game is already over
    because a marshal was captured", and it has to fire *before* movegen: on
    such a board the engine's own ``is_checkmate`` answers False.
    """
    tree = TreeProblem(
        edges={"r": [("a", "A"), ("b", "B")], "A": [("x", "X")], "B": [("y", "Y")]},
        terminals={"A": Outcome.DISPROVED, "B": Outcome.PROVED},
    )
    result = solve(tree, "r", 1)
    assert result.verdict == "MATE"
    assert result.line == ("b",)
    assert "A" not in tree.expanded
    assert "B" not in tree.expanded


def test_max_plies_must_be_odd_and_positive() -> None:
    tree = TreeProblem(edges={})
    for bad in (0, -3, 2, 8):
        with pytest.raises(ValueError):
            solve(tree, "r", bad)


# ---------------------------------------------------------------------------
# Repetition, and the transposition-table taint rule
# ---------------------------------------------------------------------------

#: A tree with one position ("K") reachable two ways at the same depth: once
#: below an ancestor that shares its digest -- a repetition, where the rule
#: says the attack fails -- and once on a sibling branch where it is a clean
#: mate. The two are the same transposition-table key, so a disproof stored
#: from the first would be read back as fact on the second.
CYCLE_TREE = TreeProblem(
    edges={
        "r": [("a", "A"), ("b", "B")],
        "A": [("p", "P")],
        "P": [("q", "Q")],
        "Q": [],
        "B": [("s", "S")],
        "S": [("t", "T")],
        "T": [],
    },
    digests={"A": "K", "Q": "K", "T": "K"},
)
#: ("digest", plies remaining) -- the key both Q and T hash to at depth 3.
CYCLE_KEY = ("K", 2)


def test_a_repetition_cut_does_not_poison_a_sibling_branch() -> None:
    """The honest run: the cycle branch fails, the clean branch mates."""
    tree = TreeProblem(
        edges=CYCLE_TREE.edges, terminals={}, digests=CYCLE_TREE.digests
    )
    result = solve(tree, "r", 5)
    assert result.verdict == "MATE"
    assert result.line == ("b", "s", "t")


def test_the_repetition_result_is_tainted_and_never_stored() -> None:
    """The mechanism, checked directly rather than inferred from a verdict."""
    tree = TreeProblem(
        edges=CYCLE_TREE.edges, terminals={}, digests=CYCLE_TREE.digests
    )
    search: _DfPn[str] = _DfPn(tree, 5, 10_000)
    pn, dn = search.search_root("r")
    assert (pn, dn) == (0, INF)
    # The cut was recorded as tainted...
    assert CYCLE_KEY in search._tainted
    # ...and the table never learned the disproof from it. The key is in the
    # table, but holding T's own honest proof: the clean branch stored what
    # it actually searched, which is the whole point of not poisoning it.
    assert search._tt.get(CYCLE_KEY) != (INF, 0)
    assert search._tt.get(CYCLE_KEY) == (0, INF)
    # And the taint travelled up: nothing computed from the cut was stored.
    assert ("A", 4) not in search._tt
    assert ("P", 3) not in search._tt


def test_a_stored_repetition_disproof_would_flip_the_verdict() -> None:
    """Why the taint rule exists, demonstrated by defeating it.

    Seeding the table with exactly what storing the repetition cut would have
    written turns a proven mate into a proven no-mate. Nothing else about the
    search changes. That is the failure the rule prevents, and it is
    reachable on a real board the moment a 7-ply search meets a cycle.
    """
    tree = TreeProblem(
        edges=CYCLE_TREE.edges, terminals={}, digests=CYCLE_TREE.digests
    )
    search: _DfPn[str] = _DfPn(tree, 5, 10_000)
    search._tt[CYCLE_KEY] = (INF, 0)
    pn, dn = search.search_root("r")
    assert (pn, dn) == (INF, 0), "the poisoned table should disprove the root"
    assert "T" not in tree.expanded, "T was answered from the poisoned entry"


# ---------------------------------------------------------------------------
# Node counts
# ---------------------------------------------------------------------------


def _wide_tree() -> TreeProblem:
    """One instant mate at the first branch, and eleven expensive siblings."""
    edges: dict[str, Sequence[tuple[str, str]]] = {
        "r": [(f"m{i}", f"C{i}") for i in range(12)]
    }
    edges["C0"] = []
    for i in range(1, 12):
        edges[f"C{i}"] = [(f"d{j}", f"D{i}{j}") for j in range(5)]
        for j in range(5):
            edges[f"D{i}{j}"] = [(f"e{k}", f"E{i}{j}{k}") for k in range(3)]
    return TreeProblem(edges=edges)


def test_df_pn_expands_far_fewer_nodes_than_a_full_width_search() -> None:
    """The point of proof numbers: stop at the first proof, do not enumerate.

    The bound is asserted as an absolute number, not as a ratio, so that a
    regression into full-width search fails here rather than merely getting
    slower.
    """
    tree = _wide_tree()
    result = solve(tree, "r", 3)
    assert result.verdict == "MATE"
    assert result.line == ("m0",)
    assert result.nodes <= 6, result.nodes
    baseline = full_width_expansions(_wide_tree(), "r", 3)
    assert baseline >= 200
    assert result.nodes < baseline


def test_the_budget_counts_expansions_of_this_problem() -> None:
    """``nodes`` is the number of ``children`` calls, and it is capped."""
    tree = _wide_tree()
    result = solve_problem(tree, "r", 3, node_budget=4)
    assert result.nodes <= 4
    assert len(tree.expanded) == result.nodes
