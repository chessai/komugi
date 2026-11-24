"""The subtree-value-bias table: its own arithmetic, and its call sites.

Design §A4's two-commit ordering put the table's *call sites* -- the read in
selection and the update in the backup -- in the commit **after** the one that
recorded ``tests/search/match/planted/goldens/t8_lambda0.json``. That ordering
is the whole point: the golden was produced by a build in which this feature
did not exist, so ``lambda = 0`` reproducing it is a statement about the
feature's off-switch rather than a recording of the feature under test. This
file is that later commit, and it holds three groups of tests.

**The table's own arithmetic**, driven directly, playing the part of the nodes
the backup plays. Three properties, all of them the design's B3 correction:

* a contribution is removed **in the frame the entry has decayed to**, so the
  exactness identity ``weight_sum == sum_i rho**(gen - gen_i) * s_i`` holds
  after every update rather than drifting once a second node shares the key;
* an entry below the weight floor reads as **empty**, not as a huge quotient;
* a ``weight_sum`` that goes meaningfully negative **raises** instead of being
  clamped, because a negative weight means the generation bookkeeping is wrong
  and every later ``b(k)`` on that key is arbitrary.

**T8**, the two-commit bisector: at ``subtree_bias_factor = 0.0`` the search
reproduces the bias-free golden exactly -- chosen action and root visit
weights, on **all four** fixtures. Shipped-factor control actions and budgets
remain pinned here. Real-search nonvacuity at P64/P256 and both score consumers
are covered by ``test_kom217_bias_correction.test_real_search_bias_nonvacuity``.
Plus §A2's sum-to-zero invariant over every ``b(k)`` the search actually reads.

The golden's four fixtures are two pairs. ``t1_first`` and ``asym`` are
positions decided by a margin no correction of this size may cross, and they
are the **controls**: the shipped ``lambda`` must return the same *move* on
them. ``div1`` and ``div2`` are the pair an advisor probe (2026-09-04) found
``lambda``-sensitive at this budget -- the same payoff table twice, at two
priors -- under the historical reader. Their corrected trajectories may now
match the golden despite nonzero score adjustments.

**T8b**, the decay bookkeeping driven through a real search rather than by
hand: a single search, no ``descend`` and no clear (rev 3, R1), so every node
that has contributed to a key is still live and the exactness identity is
checkable as an **equality** rather than only as the lower bound tree reuse
leaves.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

# `planted` first, always: importing it is what puts ``tests/search/oracle`` on
# ``sys.path``, and ``toygames`` lives there. See ``planted/__init__.py``.
from planted import (
    DeepGame,
    MatchToyState,
    constant_wdl_evaluator,
    solver_wdl_evaluator,
    walk_nodes,
)
from toygames import TreeGame

from komugi.search.match.bias import BiasTableInvariant, SubtreeValueBias
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.search import MatchSearch

KEY = 12345
RHO = 0.8
FLOOR = 1e-6

#: The golden recorded by ``planted/gen_t8_golden.py`` at the commit **before**
#: the call sites landed, i.e. from a search that had no bias code in it at
#: all.
GOLDEN_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "planted" / "goldens" / "t8_lambda0.json"
)

#: The shipped ``lambda``. Written here rather than read from
#: :class:`MatchConfig` so that a default changed by accident shows up as a
#: failure of the shipped-factor control test instead of silently redefining
#: what it tests.
SHIPPED_LAMBDA = 0.45

#: Every fixture the golden carries, named here so that a golden regenerated
#: with a different set fails the provenance check rather than quietly
#: narrowing what T8 covers.
GOLDEN_FIXTURES = ("t1_first", "asym", "div1", "div2")

#: Historically sensitive under the old reader; trajectory sensitivity does not
#: survive every arithmetic repair. These remain immutable off-switch fixtures.
HISTORICALLY_DIVERGING = ("div1", "div2")

#: ...and the two whose payoffs are lopsided enough (``+-1`` rows, and a table
#: with a 0.4 margin between the best row's minimum and the next) that no
#: correction of this size may reach the **move**. They are the controls:
#: without them "the search changed" could be reported by a wiring that
#: perturbs every position indiscriminately, which is what a sign-reversed
#: correction does.
CONTROLS = ("t1_first", "asym")

#: WDL error vectors. Each sums to zero, because each is a difference of two
#: probability vectors -- which is what makes ``b(k)`` sum to zero too.
DELTA_A = np.array([0.1, -0.2, 0.1], dtype=np.float64)
DELTA_B = np.array([-0.3, 0.2, 0.1], dtype=np.float64)
DELTA_C = np.array([0.05, 0.0, -0.05], dtype=np.float64)


class FakeNode:
    """The three fields design §S1 puts on a node for the table's sake."""

    def __init__(self, gen: int) -> None:
        self.contribution: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.weight = 0.0
        self.gen = gen


def make_table() -> SubtreeValueBias:
    """Return an empty table with the design's constants."""
    return SubtreeValueBias(free_prop=RHO, weight_floor=FLOOR)


def apply(
    table: SubtreeValueBias,
    node: FakeNode,
    delta: npt.NDArray[np.float64],
    new_weight: float,
) -> None:
    """Run one update and write the result back onto `node`, as the tree will."""
    node.contribution, node.weight, node.gen = table.update(
        KEY,
        contribution=node.contribution,
        weight=node.weight,
        gen=node.gen,
        delta=delta,
        new_weight=new_weight,
    )


def expected_weight_sum(table: SubtreeValueBias, nodes: list[FakeNode]) -> float:
    """Return the right-hand side of design §A2's exactness identity."""
    entry = table.entry(KEY)
    assert entry is not None
    return sum(RHO ** (entry.gen - node.gen) * node.weight for node in nodes)


def test_an_absent_or_empty_entry_reads_as_zero() -> None:
    """Below the floor an entry is empty, not a huge quotient."""
    table = make_table()
    np.testing.assert_array_equal(table.read(KEY), np.zeros(3))
    assert table.generation(KEY) == 0
    assert table.entry(KEY) is not None
    np.testing.assert_array_equal(table.read(KEY), np.zeros(3))


def test_the_decay_bookkeeping_is_exact_across_two_nodes_on_one_key() -> None:
    """The B3 correction: removal is scaled to what the mass has decayed to.

    Without the ``rho ** (entry.gen - node.gen)`` scale the removal takes out
    mass the entry does not contain, and this identity is violated on the very
    first update by a *second* node on the key -- which, with a key this coarse,
    is the normal case rather than a corner.
    """
    table = make_table()
    first = FakeNode(table.generation(KEY))
    second = FakeNode(table.generation(KEY))
    nodes = [first, second]

    apply(table, first, DELTA_A, 1.0)
    entry = table.entry(KEY)
    assert entry is not None
    assert entry.gen == 1
    assert abs(entry.weight_sum - 1.0) < 1e-12
    assert abs(entry.weight_sum - expected_weight_sum(table, nodes)) < 1e-12

    apply(table, second, DELTA_B, 2.0)
    assert abs(entry.weight_sum - 2.8) < 1e-12
    assert abs(entry.weight_sum - expected_weight_sum(table, nodes)) < 1e-12

    apply(table, first, DELTA_C, 3.0)
    assert abs(entry.weight_sum - 4.6) < 1e-12
    assert abs(entry.weight_sum - expected_weight_sum(table, nodes)) < 1e-12

    # And the same identity for the error vector, which is what `b(k)` divides.
    predicted: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    for node in nodes:
        predicted = predicted + (RHO ** (entry.gen - node.gen)) * node.contribution
    assert float(np.abs(entry.err_sum - predicted).max()) < 1e-12


def test_the_rehome_decay_ages_every_entry_and_keeps_the_identity() -> None:
    """Design §A4's per-rehome decay must increment the same counter.

    If it did not, a retained node's generation would under-count the decays
    its contribution has actually suffered, and the next removal would take out
    mass the entry no longer contains -- the un-scaled removal bug, re-entering
    through the reuse path.
    """
    table = make_table()
    node = FakeNode(table.generation(KEY))
    apply(table, node, DELTA_A, 4.0)
    entry = table.entry(KEY)
    assert entry is not None
    before = entry.weight_sum

    table.decay_all()
    assert entry.gen == 2
    assert abs(entry.weight_sum - RHO * before) < 1e-12
    assert abs(entry.weight_sum - expected_weight_sum(table, [node])) < 1e-12

    apply(table, node, DELTA_B, 5.0)
    assert abs(entry.weight_sum - expected_weight_sum(table, [node])) < 1e-12
    assert entry.weight_sum > 0.0


def test_the_read_is_an_average_that_sums_to_zero() -> None:
    """``b(k)`` cannot change the total probability mass a node claims."""
    table = make_table()
    node = FakeNode(table.generation(KEY))
    apply(table, node, DELTA_A, 4.0)
    reading = table.read(KEY)
    assert abs(float(reading.sum())) < 1e-12
    assert float(np.abs(reading).max()) <= 1.0
    np.testing.assert_allclose(reading, DELTA_A, atol=1e-12)


def test_a_bogus_removal_raises_instead_of_going_negative() -> None:
    """A negative ``weight_sum`` is a bookkeeping bug, not a small number.

    Clamping it would convert that bug into a silently wrong correction --
    ``err_sum / weight_sum`` either blows up or flips sign, and the clip on the
    utility then turns an unbounded blow-up into a quietly saturated selection
    score, which is worse for diagnosis rather than better.
    """
    table = make_table()
    node = FakeNode(table.generation(KEY))
    apply(table, node, DELTA_A, 1.0)
    with pytest.raises(BiasTableInvariant) as raised:
        table.update(
            KEY,
            contribution=np.zeros(3, dtype=np.float64),
            weight=100.0,
            gen=node.gen,
            delta=DELTA_B,
            new_weight=1.0,
        )
    assert raised.value.key == KEY
    assert raised.value.weight_sum < 0.0


def test_a_tiny_positive_weight_survives_in_storage() -> None:
    """``epsilon_w`` is a rule about **reads**, not a truncation of storage.

    Design §A2's floor says an entry with nearly no weight behind it *reads*
    as empty rather than as a huge quotient. Applying it to storage as well --
    zeroing a legitimately small positive ``weight_sum`` in ``update`` --
    destroys the exactness identity for every node still holding a
    contribution to that key (review F6): the entry then claims less mass than
    the sum of its live contributions, and the next generation-scaled removal
    takes out mass the entry no longer contains.

    Reaching the band is not exotic. With ``rho = 0.8`` a key whose
    contributors stop contributing decays under the floor in ~63 updates,
    which one 200-ply game with a key this coarse supplies easily. Here it is
    reached in two: a contribution of ``1e-6``, then one update by another
    node that puts nothing back, leaving ``0.8e-6`` -- below ``epsilon_w`` and
    unambiguously positive.
    """
    table = make_table()
    first = FakeNode(table.generation(KEY))
    second = FakeNode(table.generation(KEY))

    apply(table, first, DELTA_A, FLOOR)
    entry = table.entry(KEY)
    assert entry is not None
    assert entry.weight_sum == FLOOR

    apply(table, second, DELTA_B, 0.0)
    decayed = RHO * FLOOR
    assert 0.0 < decayed < FLOOR
    # The round trip: what storage holds is what the arithmetic produced.
    assert abs(entry.weight_sum - decayed) < 1e-15
    assert abs(entry.weight_sum - expected_weight_sum(table, [first, second])) < 1e-15
    # ...and the floor still does its job, at the one place it belongs.
    np.testing.assert_array_equal(table.read(KEY), np.zeros(3))

    # The consequence, and the reason this matters at all: a key that comes
    # back to life resumes from where it actually was. A *third* node, so the
    # decayed mass is carried rather than removed -- clamped in storage it is
    # silently dropped instead, and the identity breaks by exactly that much.
    third = FakeNode(table.generation(KEY))
    apply(table, third, DELTA_C, 1.0)
    assert abs(entry.weight_sum - (RHO * decayed + 1.0)) < 1e-15
    assert entry.weight_sum > 1.0
    assert (
        abs(entry.weight_sum - expected_weight_sum(table, [first, second, third]))
        < 1e-15
    )


def golden() -> dict[str, Any]:
    """Return the committed T8 golden, with its provenance checked.

    The provenance is part of the test: a golden with no generating commit
    recorded could have been produced by any build, including this one, and
    "the code equals itself" is precisely the assertion design §A4's two-commit
    ordering exists to prevent.
    """
    payload: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["generated_at_commit"]
    assert "bias-free" in payload["note"]
    fixtures: dict[str, dict[str, Any]] = payload["fixtures"]
    assert set(fixtures) == set(GOLDEN_FIXTURES)
    return payload


def run_golden_fixture(
    fixture: dict[str, Any], factor: float
) -> tuple[int, list[float]]:
    """Search one golden fixture at `factor` and return what the golden records.

    Every argument is taken from the golden file itself rather than restated
    here, so the two runs cannot drift apart in the one way that would make
    the comparison vacuous: a test that searched a *different* fixture than
    the one recorded.
    """
    payoffs = tuple(tuple(float(value) for value in row) for row in fixture["table"])
    width = len(payoffs)
    config = MatchConfig(
        num_actions=width,
        playouts=int(fixture["playouts"]),
        subtree_bias_factor=factor,
    )
    search = MatchSearch(
        config,
        solver_wdl_evaluator(
            num_actions=width,
            logits=np.asarray(fixture["logits"], dtype=np.float32),
        ),
    )
    result = search.search(MatchToyState(TreeGame(payoffs)), ply=0)
    root = search.tree.root
    assert root is not None
    return int(result.action), [float(weight) for weight in root.weight]


def test_lambda_zero_reproduces_the_bias_free_golden() -> None:
    """T8, the off-switch: ``lambda = 0`` is the bias-free search, exactly.

    The golden was generated by ``planted/gen_t8_golden.py`` at the commit
    recorded inside it, where :mod:`komugi.search.match.select` multiplied
    ``lambda`` into a hardcoded zero and the playout loop neither read nor
    wrote the table. Reproducing it here -- same chosen action, same per-edge
    root weights, on **all four** fixtures, controls and historically sensitive
    pair alike -- is what makes the feature **bisectable**: a regression can be
    attributed to the correction only if turning the correction off restores
    the previous behaviour bit for bit.

    The historical ``div1`` and ``div2`` trajectories may also match at the
    shipped factor. Score changes need not cross a discrete selection boundary;
    real F2 nonvacuity is asserted separately at P64 and P256.

    Note what is *not* claimed: the table is still updated at ``lambda = 0``
    (the update path has no ``lambda`` in it), and entries are still created at
    expansion. What must vanish is the correction's effect on every number a
    caller can observe, which is exactly what the comparison covers.
    """
    fixtures: dict[str, dict[str, Any]] = golden()["fixtures"]
    for name in GOLDEN_FIXTURES:
        fixture = fixtures[name]
        action, weights = run_golden_fixture(fixture, 0.0)
        expected = [float(weight) for weight in fixture["root_weights"]]
        assert action == int(fixture["action"]), name
        assert weights == expected, name

        # Teeth: a golden recording one edge with every visit would agree with
        # itself under any correction whatsoever.
        assert sum(expected) == float(fixture["playouts"])
        assert len({weight for weight in expected if weight > 0.0}) > 1


def test_the_shipped_lambda_preserves_control_actions_and_budgets() -> None:
    """T8 controls: preserve robust actions and all four original budgets.

    Historical div1/div2 sensitivity is not a current trajectory obligation.
    The real F2 search test separately requires responsive visits and exact
    PUCT/LCB score deltas at the same retained shipped-factor root.
    """
    fixtures: dict[str, dict[str, Any]] = golden()["fixtures"]
    report: list[str] = []
    for name in GOLDEN_FIXTURES:
        fixture = fixtures[name]
        action, weights = run_golden_fixture(fixture, SHIPPED_LAMBDA)
        golden_action = int(fixture["action"])
        golden_weights = [float(weight) for weight in fixture["root_weights"]]
        report.append(
            f"{name}: golden action {golden_action} weights {golden_weights}; "
            f"lambda={SHIPPED_LAMBDA} action {action} weights {weights}"
        )
        # Whatever the correction moved, it moved *within* the budget: it
        # redistributes visits, it does not create or destroy them.
        assert sum(weights) == float(fixture["playouts"]), name
        if name in CONTROLS:
            assert action == golden_action, (
                f"the correction changed the move on control fixture {name}, "
                f"whose payoffs decide it by a margin no bias term of this "
                f"size may cross:\n" + "\n".join(report)
            )


def test_every_bias_vector_read_by_a_search_sums_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T8's invariant: ``b(k)`` cannot change the mass a node claims.

    ``b(k)`` is an average of differences of two probability vectors, so it
    sums to zero for **any** ``weight_sum`` -- including a near-zero or a
    negative one. That is why the design says plainly that this invariant is
    *not* the guard on B3's defect (T8b is): it is the guard on the
    correction's shape, which is that adding it to a WDL triple leaves
    a triple still summing to one.

    Every reading the search actually takes is recorded, rather than the
    entries being inspected afterwards -- a correction is only as
    mass-preserving as the vectors that reach the arithmetic. The mover-frame
    swap the writer applies is a *permutation* of the three slots, so it moves
    where the mass sits and never how much of it there is: this invariant holds
    on both sides of that fix, and is deliberately not the guard for it.

    It runs at the **default** ``lambda``, over the ``asym`` fixture's table
    read out of the golden -- which is why an earlier revision of this test was
    red for a reason that had nothing to do with the invariant: it reached the
    table through :func:`golden`, whose provenance check still named the
    two-fixture golden after the file had been regenerated with four.
    """
    payoffs = tuple(
        tuple(float(value) for value in row)
        for row in golden()["fixtures"]["asym"]["table"]
    )
    width = len(payoffs)
    search = MatchSearch(
        MatchConfig(num_actions=width, playouts=1000),
        solver_wdl_evaluator(num_actions=width),
    )
    table = search.tree.bias
    original = table.read
    readings: list[npt.NDArray[np.float64]] = []

    def recording_read(key: int) -> npt.NDArray[np.float64]:
        reading = original(key)
        readings.append(reading)
        return reading

    monkeypatch.setattr(table, "read", recording_read)
    search.search(MatchToyState(TreeGame(payoffs)), ply=0)

    assert readings, "the search never consulted the table"
    assert any(float(np.abs(reading).max()) > 0.0 for reading in readings)
    for reading in readings:
        assert bool(np.isfinite(reading).all())
        assert abs(float(reading.sum())) < 1e-12
        assert float(np.abs(reading).max()) <= 1.0


#: T8b's fixture, and every number in it is load-bearing. The depth **exceeds
#: the playout budget**, and one playout deepens a line by at most one node, so
#: no descent can be truncated by a terminal position -- a terminal leaf has no
#: child weight, contributes nothing, and would quietly thin the interleaving
#: this test is about. Two actions with a lopsided prior so the descent drives
#: one line deep, which is what puts **several** nodes on one
#: ``(mover, move, previous move)`` key: every even-depth node on the all-zeros
#: line carries the key ``(mover 1, move 0, previous move 0)``.
T8B_WIDTH = 2
T8B_DEPTH = 128
T8B_PLAYOUTS = 96
T8B_LOGITS = np.array([3.0, 0.0], dtype=np.float32)

#: A deliberately **asymmetric** leaf value. The backup swaps ``(w, d, l)`` at
#: every mover change, so a symmetric triple would make every subtree mean
#: equal to every net value and every ``delta`` exactly zero -- an err_sum
#: identity over nothing but zeros.
T8B_WDL = (0.5, 0.2, 0.3)


def contributors_by_key(root: MatchNode | None) -> dict[int, list[MatchNode]]:
    """Return every live node grouped by its bias key, in a fixed order."""
    grouped: dict[int, list[MatchNode]] = defaultdict(list)
    for node in walk_nodes(root):
        grouped[node.bias_key].append(node)
    return dict(grouped)


def test_the_decay_bookkeeping_is_exact_after_a_real_search() -> None:
    """T8b: the exactness identity, over a search that really interleaves.

    **One search, no ``descend`` and no clear** (rev 3, R1). Nothing frees a
    node during a single search, so "every node that has contributed and not
    been replaced" and "every live node" are the same set by construction, and
    design §A2's identity is checkable as an equality rather than only as the
    lower bound tree reuse leaves behind.

    The three clauses, and each names the mutant it kills:

    1. ``weight_sum >= 0`` always, and **above the floor** for any entry with a
       live contributor that has actually contributed. A removal at the
       undecayed magnitude drives it through zero and negative, which is the
       rev-1 defect;
    2. ``|b(k)|`` bounded by 1 and finite -- a ``delta`` computed as ``0/0`` on
       a node with no child weight shows up here as ``nan``;
    3. the identity itself, for ``weight_sum`` and for ``err_sum``, recomputed
       from every live node on the key. Removal without the
       ``rho**(gen - bias_gen)`` scale violates it on the very first update by
       a second node on the key -- and with this key's coarseness, a second
       node on the key is the normal case rather than a corner.
    """
    config = MatchConfig(
        num_actions=T8B_WIDTH, playouts=T8B_PLAYOUTS, tree_reuse=False
    )
    search = MatchSearch(
        config,
        constant_wdl_evaluator(
            T8B_WDL, num_actions=T8B_WIDTH, logits=T8B_LOGITS
        ),
    )
    search.search(MatchToyState(DeepGame(width=T8B_WIDTH, depth=T8B_DEPTH)), ply=0)

    table = search.tree.bias
    grouped = contributors_by_key(search.tree.root)
    rho = config.bias_free_prop

    # The fixture has to be rich enough for the identity to say anything: one
    # key carrying three distinct nodes, updated at least twenty times between
    # them. Asserted, not assumed -- a shallower search would make every
    # clause below true of a single contributor and prove nothing.
    busiest = max(
        grouped,
        key=lambda key: sum(1 for node in grouped[key] if node.bias_weight > 0.0),
    )
    live = [node for node in grouped[busiest] if node.bias_weight > 0.0]
    busiest_entry = table.entry(busiest)
    assert busiest_entry is not None
    assert len(live) >= 3
    assert busiest_entry.gen >= 20

    for key in table.keys():
        entry = table.entry(key)
        assert entry is not None
        contributors = grouped.get(key, [])
        contributing = [node for node in contributors if node.bias_weight > 0.0]

        # (1) never negative, and never *below the floor* while a live node is
        # holding real mass on it. An entry whose only live contributor is a
        # freshly expanded node still at `bias_weight = 0.0` is legitimately
        # empty, and asserting it above the floor would be wrong.
        assert entry.weight_sum >= 0.0
        if contributing:
            assert entry.weight_sum > table.weight_floor

        # (2) the reading is a bounded, finite average of differences of
        # probability vectors.
        reading = table.read(key)
        assert bool(np.isfinite(reading).all())
        assert float(np.abs(reading).max()) <= 1.0

        # (3) the identity, on both accumulators.
        predicted_weight = sum(
            rho ** (entry.gen - node.bias_gen) * node.bias_weight
            for node in contributors
        )
        predicted_error: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        for node in contributors:
            predicted_error = predicted_error + (
                rho ** (entry.gen - node.bias_gen)
            ) * np.asarray(node.bias_contribution, dtype=np.float64)
        assert abs(entry.weight_sum - predicted_weight) < 1e-9, key
        assert float(np.abs(entry.err_sum - predicted_error).max()) < 1e-9, key


def test_a_contribution_is_stored_in_the_frame_its_key_names() -> None:
    """The mover frame of a contribution is the **parent's**, not the node's.

    This is the unit-level statement of the defect the oracle caught. Design
    §A2 writes ``delta(n) = x_sub(n) - x_net(n)`` with both terms in ``n``'s own
    perspective, and design §S1 stores the statistics the *reader* corrects --
    ``S(a)/W(a)`` -- in the **parent's**, one negamax swap away (§A6). Storing
    the node's own frame therefore hands selection and LCB a correction whose
    win and loss slots are transposed, i.e. one that enters the collapsed
    utility with its sign reversed, and the correction stops discounting an
    edge whose subtree came in below its net value and starts rewarding it.

    Asserted directly against the node's stored ``bias_contribution`` rather
    than through a search outcome, so a failure names the arithmetic instead of
    a visit count. Every edge of :class:`DeepGame` changes the mover, so every
    contribution here is a swapped one, and the ``material`` counter pins that
    the swap actually moves the numbers on a useful fraction of them -- against
    a symmetric ``delta``, on which both frames agree, this test would prove
    nothing.
    """
    config = MatchConfig(
        num_actions=T8B_WIDTH, playouts=T8B_PLAYOUTS, tree_reuse=False
    )
    search = MatchSearch(
        config,
        constant_wdl_evaluator(
            T8B_WDL, num_actions=T8B_WIDTH, logits=T8B_LOGITS
        ),
    )
    search.search(MatchToyState(DeepGame(width=T8B_WIDTH, depth=T8B_DEPTH)), ply=0)

    root = search.tree.root
    assert root is not None
    stack: list[tuple[MatchNode, MatchNode]] = [
        (root, child) for child in root.children.values()
    ]
    checked = 0
    swapped = 0
    material = 0
    while stack:
        parent, node = stack.pop()
        stack.extend((node, below) for below in node.children.values())
        if node.bias_weight <= 0.0:
            continue
        total = node.total_weight()
        # The node's last table update is its final one: a later backup into
        # its subtree has to pass through it and would update it again.
        assert abs(node.bias_weight - total**config.bias_weight_exponent) < 1e-12

        own_frame = node.subtree_wdl_sum() / total - node.wdl_net
        turns = node.to_play != parent.to_play
        key_frame = own_frame[::-1] if turns else own_frame
        assert (
            float(np.abs(node.bias_contribution - node.bias_weight * key_frame).max())
            < 1e-12
        ), node.bias_key
        checked += 1
        swapped += int(turns)
        if turns and float(np.abs(key_frame - own_frame).max()) > 1e-6:
            material += 1

    # Teeth: something was checked, every edge of this fixture turns, and on
    # most of them the two frames are genuinely different vectors -- so the
    # equality above is not one a frame-blind implementation also satisfies.
    assert checked >= 10
    assert swapped == checked
    assert material >= 5


def test_a_non_finite_delta_is_refused() -> None:
    """The first-backup case: ``0/0`` never reaches the table.

    A ``nan`` written into ``err_sum`` is permanent, so the table refuses one
    rather than trusting every future caller to have skipped the update while
    the node still had no child weight.
    """
    table = make_table()
    node = FakeNode(table.generation(KEY))
    with pytest.raises(ValueError, match="non-finite"):
        apply(table, node, np.array([np.nan, 0.0, 0.0]), 1.0)
    entry = table.entry(KEY)
    assert entry is not None
    assert bool(np.isfinite(entry.err_sum).all())
