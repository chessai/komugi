"""A12: tree reuse -- what it saves, what it must refuse, and what it frees.

Three acceptance tests, and each one names the bug it exists to catch. Two
further nodes cover what a rehome does to design §A2's bias table: the key the
re-expanded root carries (review F5) and the per-rehome decay §A4 asks for
(review F4) -- the latter pre-registered as an ``xfail(strict=True)``
obligation while the table had no call sites, and **live** now that
``descend`` performs the decay.

* **T3** -- reuse changes node counts but not the pick. Rehoming to the wrong
  child makes the picks diverge; rehoming that silently clears makes the
  counts equal; reuse that double-counts visits makes the counts equal and the
  picks diverge.
* **T10** -- the rehome consistency check fires in **both** directions. A
  check written as "every stored action is still legal" passes silently on a
  legal set that has *grown*, and the search then runs over a truncated legal
  set and can never select the missing move. That is the failure the check
  exists to prevent, so it is the one the fixture is built to produce.
* **T11** -- the dropped siblings are collectable. A parent back-pointer would
  keep the whole game's tree alive through the retained child, which turns
  tree reuse from an optimisation into an out-of-memory over two hundred
  plies.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

# `planted` before `solver`/`toygames`: importing it is what puts the oracle
# directory on ``sys.path``. See ``planted/__init__.py``.
from planted import (
    CountingEvaluator,
    DeepGame,
    MatchToyState,
    PlantedLegalState,
    constant_wdl_evaluator,
    solver_wdl_evaluator,
    walk_nodes,
)
from solver import solve
from toygames import BOARD_SIZE, TicTacToe

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import NO_PREVIOUS_ACTION, bias_key_of
from komugi.search.match.search import MatchSearch
from komugi.search.match.tree import MatchTree, MatchTreeDesync

#: Three empty cells, so the scripted line below plays the game out. The
#: design's "four-move line" is not available this late, and starting earlier
#: would put the line through positions with several equally optimal moves,
#: where "the two runs agree" is a coin flip rather than a property.
LINE_BOARD = "OO./XXO/.X."

#: The moves actually played, in order. Each is the unique solver-optimal move
#: in its position -- the block that holds the draw, the block that answers the
#: fork threat, and then the only remaining cell.
LINE = (2, 6, 8)

#: Pinned as a literal rather than read from a config default: at nine actions
#: this is comfortably enough for every root child to be expanded, which is
#: what makes "the opponent's reply was already in the tree" a property of the
#: fixture rather than a hope, and a later budget change must not quietly
#: weaken it.
LINE_PLAYOUTS = 64


def run_scripted_line(reuse: bool) -> tuple[list[int], list[int], list[int], int]:
    """Play :data:`LINE` once, returning what each move decided and cost.

    Returns:
        ``(actions returned, nodes at entry, optimal-action check, evaluations)``
        where the third list holds ``1`` where the returned action was one the
        solver calls optimal.
    """
    config = MatchConfig(
        num_actions=BOARD_SIZE, playouts=LINE_PLAYOUTS, tree_reuse=reuse
    )
    evaluator = CountingEvaluator(solver_wdl_evaluator(num_actions=BOARD_SIZE))
    search = MatchSearch(config, evaluator)
    state = MatchToyState(TicTacToe.from_string(LINE_BOARD))

    actions: list[int] = []
    entries: list[int] = []
    optimal: list[int] = []
    for ply, played in enumerate(LINE):
        best = solve(state.game).best_actions
        result = search.search(state, ply=ply)
        actions.append(result.action)
        entries.append(result.nodes_at_entry)
        optimal.append(1 if result.action in best else 0)
        assert result.tree_reuse is reuse
        state.play(played)
        search.observe(played)
    return actions, entries, optimal, evaluator.calls


def test_reuse_changes_node_counts_but_not_the_pick() -> None:
    """T3: same actions, more nodes at entry, strictly fewer evaluations.

    *Honesty note.* The picks are asserted equal **on this planted,
    solver-evaluated fixture**, where both runs converge on the same optimal
    move at every position. Reuse changes the tree, so equality of picks is
    not a general property and this test does not claim it is; asserting it in
    general would be false and would show up as a flaky test rather than as an
    argument.
    """
    fresh_actions, fresh_entries, fresh_optimal, fresh_calls = run_scripted_line(False)
    reuse_actions, reuse_entries, reuse_optimal, reuse_calls = run_scripted_line(True)

    # (i) the action sequences are identical, and both are the solver's.
    assert fresh_actions == reuse_actions
    assert fresh_optimal == [1, 1, 1]
    assert reuse_optimal == [1, 1, 1]

    # (ii) the reuse run enters every move after the first with a tree; the
    # other never does, on any move, including the opponent's reply.
    assert fresh_entries == [0, 0, 0]
    assert reuse_entries[0] == 0
    assert all(entry > 0 for entry in reuse_entries[1:])

    # (iii) and it pays for strictly less of the network.
    assert reuse_calls < fresh_calls


def planted_fork() -> PlantedLegalState:
    """Return the two-ply fixture T10's three cases are built on.

    Two root actions leading to positions whose legal sets are a strict
    superset and subset of each other. Written down rather than searched for,
    because no engine position is *guaranteed* to produce the superset case
    and a probabilistic fixture dressed as a deterministic test proves nothing
    when it passes.
    """
    return PlantedLegalState(
        {
            (): frozenset({0, 1}),
            (0,): frozenset({0, 1, 2}),
            (1,): frozenset({0, 1, 2, 3}),
        },
        num_actions=4,
        results={},
    )


def searched_fork() -> tuple[MatchSearch, PlantedLegalState]:
    """Return a search whose tree covers both children of the fork."""
    config = MatchConfig(num_actions=4, playouts=32, tree_reuse=True)
    search = MatchSearch(
        config, constant_wdl_evaluator([0.3, 0.4, 0.3], num_actions=4)
    )
    state = planted_fork()
    search.search(MatchToyState(state), ply=0)
    root = search.tree.root
    assert root is not None
    assert set(root.children) == {0, 1}
    return search, state


def test_the_rehome_check_fires_on_a_grown_legal_set() -> None:
    """T10(a): the superset case, which a one-directional check cannot see.

    The tree rehomes onto the child whose legal set is ``{0, 1, 2}`` while the
    real position's is ``{0, 1, 2, 3}``. Every stored action is still legal, so
    a check asking only that passes -- and the search then runs over a legal
    set that cannot express action ``3`` at all. Set equality raises instead,
    and the raise is guaranteed by construction rather than probable.
    """
    search, state = searched_fork()
    search.observe(0)
    state.play(1)
    with pytest.raises(MatchTreeDesync) as raised:
        search.search(MatchToyState(state), ply=1)
    assert raised.value.stored_hash != raised.value.actual_hash
    assert raised.value.ply == 1


def test_the_rehome_check_fires_on_a_shrunken_legal_set() -> None:
    """T10(b): the mirror case, so the check is shown to be two-directional."""
    search, state = searched_fork()
    search.observe(1)
    state.play(0)
    with pytest.raises(MatchTreeDesync):
        search.search(MatchToyState(state), ply=1)


def test_the_rehome_check_is_silent_on_a_correct_move() -> None:
    """T10(c): the honest control -- a check that always raised would pass (a).

    And the retained tree is really retained: ``nodes_at_entry > 0`` says the
    move began inside the subtree the previous move paid for.
    """
    search, state = searched_fork()
    search.observe(0)
    state.play(0)
    result = search.search(MatchToyState(state), ply=1)
    assert result.nodes_at_entry > 0
    assert bool(state.legal_mask()[result.action])


#: The two keys the rehome-decay obligation is stated over. Arbitrary ints:
#: the table is keyed by a packed triple and does not care where one came from.
BIAS_KEYS = (11, 22)

#: A WDL error vector, summing to zero like every difference of two
#: probability vectors.
BIAS_DELTA = np.array([0.1, -0.2, 0.1], dtype=np.float64)


def test_the_root_bias_key_names_the_mover_who_played_into_it() -> None:
    """§A2's mover slot is the **parent's**, not the new root's (review F5).

    The key is ``(mover at the parent of the edge, the move, the move before
    it)`` -- which is what ``playout`` passes for an interior edge, using the
    parent's ``to_play``. The root is the one node whose parent has been
    dropped, and taking the mover from the root's *own* ``to_play`` there
    would file one edge under two different keys depending only on whether the
    child happened to be expanded before it was played.

    A two-ply fixture where the two movers genuinely differ: the root moves
    first (``to_play == 0``), action ``0`` is played into a position whose
    mover is ``1``, and the edge belongs to mover ``0``.
    """
    config = MatchConfig(num_actions=4, tree_reuse=True)
    tree = MatchTree(
        config, constant_wdl_evaluator([0.3, 0.4, 0.3], num_actions=4)
    )
    state = planted_fork()
    tree.ensure_root(MatchToyState(state))

    root = tree.root
    assert root is not None
    assert root.to_play == 0
    # A fresh tree has no edge at all, so the key is the "no edge" one.
    assert root.bias_key == bias_key_of(
        0, NO_PREVIOUS_ACTION, NO_PREVIOUS_ACTION, config.num_actions
    )

    # No playouts have run, so action 0 has no expanded child: the tree is
    # dropped and the next search expands its own root -- which is exactly the
    # path on which the root's key used to lose the edge that reached it.
    tree.descend(0)
    assert tree.root is None
    state.play(0)
    played_into = MatchToyState(state)
    assert played_into.to_play() == 1

    tree.ensure_root(played_into)
    new_root = tree.root
    assert new_root is not None
    assert new_root.to_play == 1

    # The mover slot is the parent's `0`, and the action slot is the move
    # actually played. Both differ from what the root's own mover would give.
    assert new_root.bias_key == bias_key_of(
        0, 0, NO_PREVIOUS_ACTION, config.num_actions
    )
    assert new_root.bias_key != bias_key_of(
        1, 0, NO_PREVIOUS_ACTION, config.num_actions
    )
    assert new_root.bias_key != root.bias_key


def test_a_rehome_ages_every_bias_entry() -> None:
    """Design §A4's per-rehome decay, now that ``descend`` performs it.

    Pre-registered as an ``xfail(strict=True)`` node while the table had no
    call sites -- harmless then, because the table was always empty; **not**
    harmless once the backup writes to it, because a retained node's
    ``bias_gen`` would under-count the decays its contribution has actually
    suffered and the next removal would take out mass the entry no longer
    contains: the un-scaled removal bug, re-entering through the reuse path
    (review F4).

    It seeds the table directly, exactly as the backup does, and asks for the
    two things §A4 requires: every entry's generation advances by **exactly
    one**, and every entry's mass is scaled by ``rho``. Seeding by hand rather
    than by searching is deliberate -- the numbers before the rehome are then
    written down, so "scaled by rho" is an arithmetic assertion rather than a
    comparison of two search outcomes.
    """
    search, _ = searched_fork()
    table = search.tree.bias
    assert table.free_prop == search.config.bias_free_prop

    for key in BIAS_KEYS:
        table.update(
            key,
            contribution=np.zeros(3, dtype=np.float64),
            weight=0.0,
            gen=table.generation(key),
            delta=BIAS_DELTA,
            new_weight=4.0,
        )
    before: dict[int, tuple[int, float]] = {}
    for key in BIAS_KEYS:
        entry = table.entry(key)
        assert entry is not None
        before[key] = (entry.gen, entry.weight_sum)

    search.observe(0)

    for key in BIAS_KEYS:
        entry = table.entry(key)
        assert entry is not None
        generation, weight_sum = before[key]
        assert entry.gen == generation + 1
        assert abs(entry.weight_sum - table.free_prop * weight_sum) < 1e-12


def test_discarded_siblings_are_collectable() -> None:
    """T11: after a rehome the siblings die, and the node count says so."""
    config = MatchConfig(num_actions=3, playouts=48, tree_reuse=True)
    search = MatchSearch(
        config, constant_wdl_evaluator([0.25, 0.5, 0.25], num_actions=3)
    )
    state = MatchToyState(DeepGame(width=3, depth=40))
    search.search(state)

    root = search.tree.root
    assert root is not None
    assert set(root.children) == {0, 1, 2}
    retained_size = len(walk_nodes(root.children[0]))
    siblings = [weakref.ref(root.children[index]) for index in (1, 2)]
    played = int(root.actions[0])
    del root

    search.observe(played)
    gc.collect()

    assert all(reference() is None for reference in siblings)
    assert search.tree.node_count == retained_size
    assert retained_size < 49
