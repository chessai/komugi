"""Komugi seated at §D1's move seam: the seam, the budget, and a real match.

:mod:`komugi.eval.gumbel_mover` is the near half of the cross-engine harness --
the arena's own Gumbel search wearing
:class:`~komugi.eval.v1bridge.Mover` -- so what has to be proved here is not that
a search runs, but that **this** search runs, spends what §D3 says it may spend,
and is scored by the bridge exactly as the remote half is.

Five claims, one section each:

* the seam is satisfied *structurally* -- ``mover: Mover = GumbelMover(...)``
  type-checks under ``mypy --strict``, which is where the four-method contract is
  actually enforced, and all four run;
* a Komugi entrant plays a real pair against the fake v1 of
  ``test_v1bridge_behavior`` and is scored as an ordinary completed match;
* §D3.4's budget arithmetic holds at both ends of its range: a branching move
  runs ``n`` simulations for at most ``1 + n`` evaluations, and a forced move
  costs exactly one evaluation and runs none;
* the ceiling-plus-first evaluation reaches
  :func:`~komugi.eval.v1bridge.play_bridged_game` as §D6 M8, a **match** void,
  rather than as a crash of the arena;
* the same state and seed choose the same action.

The cheap tests drive a stub evaluator rather than a net: what they measure is
the *search's* consumption of the seam, which is a property of the schedule and
not of the weights, and a b2c16 forward per evaluation would buy nothing but
minutes. The one test that needs a real forward -- the bridged match -- uses
``factory.small_serving_net``, the campaign's own test shape.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.net")
pytest.importorskip("komugi.selfplay")

from komugi.campaign import factory
from komugi.engine.game import Game
from komugi.eval import arena
from komugi.eval.gumbel_mover import (
    NO_TREE_REUSE,
    CountedEvaluator,
    GumbelMover,
    komugi_entrant,
)
from komugi.eval.records import Player
from komugi.eval.v1bridge import (
    COMPLETED,
    M8,
    BudgetBreach,
    Mover,
    VoidBucket,
    VoidScope,
    budget_invariant_breach,
    play_bridged_game,
    play_bridged_match,
    record_of_action,
)
from komugi.eval.v1runner.counter import BudgetRefused, EvalCounter
from komugi.eval.v1runner.protocol import EVAL_CEILING
from komugi.selfplay.adapter import GameState
from komugi.selfplay.inference import (
    BatchEvaluator,
    InferenceConfig,
    InProcessEvaluator,
    NetEval,
)
from test_v1bridge_behavior import DRAW_SCORE, match_config, stub_entrant

KOMUGI_ID: str = "ckpt-test"
"""The Komugi entrant's rateable id, in the ladder's own ``ckpt-<step>`` shape."""

FAST_M: int = 4
FAST_N: int = 8
"""A four-candidate, eight-simulation search: enough to exercise the halving
schedule, the wave seam and the counter, and small enough to run many times."""


class StubEvaluator:
    """A position-only :class:`~komugi.selfplay.inference.BatchEvaluator`.

    Zero policy logits and a constant value, so the search's choice is decided
    entirely by its Gumbel draw -- which is exactly what the determinism test
    wants to pin. It also tallies the rows it was asked about, independently of
    :class:`~komugi.eval.v1runner.counter.EvalCounter`, so a test can check the
    counter against something that is not the counter.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Answer a batch with a flat prior and a drawn value."""
        del legal
        rows = int(np.asarray(obs, dtype=np.float32).shape[0])
        self.rows += rows
        self.calls += 1
        wdl = np.asarray(
            np.tile(np.asarray([0.25, 0.5, 0.25], dtype=np.float32), (rows, 1)),
            dtype=np.float32,
        )
        return NetEval(
            policy_logits=np.zeros((rows, factory.NUM_ACTIONS), dtype=np.float32),
            wdl=wdl,
            moves_left=np.zeros((rows,), dtype=np.float32),
        )


def stub_factory() -> BatchEvaluator:
    """Return a fresh stub evaluator, in the factory shape the mover takes."""
    return StubEvaluator()


def small_net_factory() -> BatchEvaluator:
    """Return the campaign's b2c16 test net behind the real inference seam.

    ``factory.small_serving_net`` is the shape the campaign itself uses for
    tests: the trunk shrinks, the seam does not -- ``in_planes`` and
    ``num_actions`` stay at the production values -- so a bridged game through
    this factory exercises the real class, the real heads and the real wave
    adapter at a cost a CPU can pay for a whole game.
    """
    return InProcessEvaluator(
        factory.small_serving_net(),
        InferenceConfig(in_planes=factory.IN_PLANES, num_actions=factory.NUM_ACTIONS),
    )


def fresh_state() -> GameState:
    """Return the arena's own start position under the pre-registered contempt."""
    return GameState(Game(), DRAW_SCORE)


class OneLegalState(GameState):
    """The start position with its legal set narrowed to a single action.

    §D3.4's forced-move convention has to be tested on a position with exactly
    one legal move, and gungi's opening has 229 of them; reaching a genuinely
    forced position would take a contrived endgame the engine's own tests do not
    provide. Narrowing the mask is the *same* input the search sees in that
    position -- ``GumbelSearch`` decides "forced" from ``legal_mask`` alone
    (``gumbel.py:481-491``) -- so this drives the real forced-move path and not a
    stand-in for it.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return a one-hot mask over this position's first legal action."""
        full = super().legal_mask()
        narrowed = np.zeros_like(full)
        narrowed[int(np.flatnonzero(full)[0])] = True
        return narrowed


# ---------------------------------------------------------------------------
# (a) the seam
# ---------------------------------------------------------------------------


def test_a_gumbel_mover_satisfies_the_mover_seam_and_all_four_methods_run() -> None:
    """``GumbelMover`` is a ``Mover``, statically and then actually.

    The annotation on the first line is the load-bearing part and it is checked
    by ``mypy --strict``, not by this run: ``Mover`` is a
    :class:`typing.Protocol`, nothing here subclasses it, and a mover whose
    ``pick`` took the wrong parameters or returned the wrong type would fail the
    gate at that assignment. What the body then proves is that the four methods
    are real -- ``pick`` returns a **legal** action in the position it was handed
    and leaves that position unplayed, ``observe`` accepts the record the loop
    hands it, ``reset`` records the seating, and ``close`` is idempotent and
    final.
    """
    mover: Mover = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id=KOMUGI_ID,
        m=FAST_M,
        n=FAST_N,
    )
    mover.reset(pair=0, seat=1)
    state = fresh_state()
    before = state.legal_mask()

    action = mover.pick(state, ply=0, seed=17)

    assert bool(before[action]), "pick returned an action that was not legal"
    assert np.array_equal(state.legal_mask(), before), (
        "pick mutated the referee's position: the search must work on clones"
    )
    mover.observe(action, record_of_action(state, action))

    assert isinstance(mover, GumbelMover)
    assert mover.seating == (0, 1)
    assert mover.tree_nodes_at_entry == NO_TREE_REUSE

    mover.close()
    mover.close()
    with pytest.raises(RuntimeError, match="closed"):
        mover.pick(fresh_state(), ply=1, seed=18)


def test_the_entrant_carries_the_ladder_family_and_the_budget_it_plays_at() -> None:
    """``komugi_entrant`` is a ladder entrant, and its budget is the anchor's.

    The family is what the roster map (§D4 amendment 3) reports the row's engine
    from, and ``m``/``n``/``gumbel_scale`` on the entrant are what
    :func:`~komugi.eval.v1bridge.anchor_configuration` publishes as §D3.6 P5's
    configuration block. If the entrant's copies could drift from the mover's,
    the published budget would not be the budget that was played -- so the
    assertion below reads both.
    """
    entrant = komugi_entrant(
        KOMUGI_ID, stub_factory, draw_score=DRAW_SCORE, m=FAST_M, n=FAST_N
    )

    assert entrant.player == Player(id=KOMUGI_ID, family=arena.LADDER_FAMILY)
    assert (entrant.m, entrant.n) == (FAST_M, FAST_N)
    assert entrant.gumbel_scale == arena.DEFAULT_GUMBEL_SCALE

    first = entrant.mover_factory()
    second = entrant.mover_factory()
    assert first is not second, "the factory must build a fresh mover per match"
    assert isinstance(first, GumbelMover)
    assert (first.m, first.n, first.gumbel_scale) == (
        entrant.m,
        entrant.n,
        entrant.gumbel_scale,
    )
    assert first.player_id == KOMUGI_ID
    assert first.draw_score == DRAW_SCORE
    first.close()
    assert isinstance(second, GumbelMover)
    second.close()


def test_the_default_budget_is_the_arena_s_own() -> None:
    """§D3.3's 32/256/1.0, read off the arena rather than restated here.

    The defaults are the point of the ticket: a rung rated against the anchor and
    the same rung rated against another rung have to be the same player. A drift
    in either direction lands here.
    """
    entrant = komugi_entrant(KOMUGI_ID, stub_factory, draw_score=DRAW_SCORE)

    assert (entrant.m, entrant.n) == (arena.DEFAULT_M, arena.DEFAULT_N)
    assert entrant.gumbel_scale == arena.DEFAULT_GUMBEL_SCALE
    assert (arena.DEFAULT_M, arena.DEFAULT_N) == (32, 256)


# ---------------------------------------------------------------------------
# (b) a real bridged match
# ---------------------------------------------------------------------------


# KOM-236: the only node here that spawns the fake v1 as a real subprocess,
# which the gate sandbox cannot run.
@pytest.mark.sandbox_unsafe
def test_a_komugi_entrant_plays_a_scored_pair_against_a_fake_v1() -> None:
    """The whole point of the ticket: a bridged match with Komugi on one side.

    A real subprocess on the far end, a real net on this one, one seat-swapped
    pair played to the engine's adjudication. Four things could only be proved by
    playing it:

    * ``play_bridged_match`` seats a Komugi mover at all -- until this module
      existed, the near side of §D1's seam had no implementation in ``src``;
    * the two engines stay in step for whole games through ``observe``: a Komugi
      action that decoded to a record the fake v1 could not apply desynchronises
      its board, and the very next ply proposes a move the referee calls illegal
      (an M3 void, which the assertion on ``voids`` catches);
    * the pair is scored as an ordinary match -- two records, mirror seats, one
      ``pair_id`` -- so a cross-engine result needs no special handling downstream;
    * no ply breached §D3.3's ceiling. A breach anywhere would have voided the
      match as M8, so ``status == COMPLETED`` with an empty tally *is* the
      whole-game budget assertion.
    """
    match = play_bridged_match(
        komugi_entrant(
            KOMUGI_ID, small_net_factory, draw_score=DRAW_SCORE, m=2, n=2
        ),
        stub_entrant("v1-fake"),
        match_config(games=2),
    )

    assert match.status == COMPLETED, f"{match.status}: {match.reason}"
    assert match.tally.voids == (), f"unexpected voids: {match.tally.by_mode()}"
    assert match.outcome is not None
    assert len(match.outcome.records) == 2
    assert match.pairs_completed == 1

    first, second = match.outcome.records
    assert {first.white, first.black} == {KOMUGI_ID, "v1-fake"}
    assert first.seats == (second.black, second.white), "the seats must swap"
    assert first.pair_id == second.pair_id
    assert match.outcome.summary.shortest_plies > 1


# ---------------------------------------------------------------------------
# (c) §D3.4's budget arithmetic, at both ends
# ---------------------------------------------------------------------------


def test_a_branching_move_runs_n_simulations_for_at_most_one_evaluation_each() -> None:
    """``sims == n`` and ``evals <= 1 + n``, which is why the ceiling is 257.

    This is §D3.3's ceiling justified rather than asserted -- it is ``1 + n``
    because a move costs one root evaluation plus at most one leaf per
    simulation, and a schedule that consulted the network twice for one
    simulation would push ``evals`` past ``1 + sims`` here (RISK C4's first limb,
    which is precisely what :func:`budget_invariant_breach` is stated over). The
    relation is measured at a budget CI can afford and the pinned budget is then
    reached by arithmetic, on the last assertion: a full 256-simulation search
    holds ~257 root-width arrays in memory to prove nothing this does not.

    The counter is checked against the evaluator's *own* tally, which is
    incremented inside the stub and knows nothing about
    :class:`~komugi.eval.v1runner.counter.EvalCounter`. A counting seam that
    missed a path -- the scalar root, say, or the wave -- disagrees with it.
    """
    stub = StubEvaluator()
    mover = GumbelMover(
        lambda: stub, draw_score=DRAW_SCORE, player_id=KOMUGI_ID, m=8, n=16
    )

    mover.pick(fresh_state(), ply=0, seed=5)

    assert mover.sims_used == 16
    assert 1 <= mover.evals_used <= EVAL_CEILING
    assert mover.evals_used <= 1 + mover.sims_used
    assert mover.tree_nodes_at_entry == NO_TREE_REUSE
    assert budget_invariant_breach(mover.evals_used, mover.sims_used) is None
    assert stub.rows == mover.evals_used, (
        "the counter and the evaluator disagree about how many positions the "
        "search asked about: a path is reaching the network uncounted"
    )
    assert mover.evals == [mover.evals_used]
    assert mover.sims == [mover.sims_used]
    assert 1 + arena.DEFAULT_N == EVAL_CEILING, (
        "§D3.3's ceiling is the pinned budget's own 1 + n; if the arena's n "
        "moves, the ceiling this mover enforces is no longer the pre-registered "
        "one and the anchor's configuration block is a lie"
    )


def test_a_forced_move_costs_one_evaluation_and_runs_no_simulation() -> None:
    """§D3.4's forced-move pin: ``evals == 1``, ``sims == 0``, and no exception.

    The convention is not free arithmetic -- it is the one asymmetry P1 checks in
    both directions' worth of detail, and the harness voids a *pair* when a mover
    reports a forced move as zero evaluations
    (``test_v1bridge_behavior.py``'s M10 test). Komugi's own side must therefore
    satisfy what it demands of v1: one root evaluation through the counted
    evaluator, and no simulation to rediscover the only move there is.
    """
    stub = StubEvaluator()
    mover = GumbelMover(
        lambda: stub, draw_score=DRAW_SCORE, player_id=KOMUGI_ID, m=FAST_M, n=FAST_N
    )
    state = OneLegalState(Game(), DRAW_SCORE)
    forced = int(np.flatnonzero(state.legal_mask())[0])

    action = mover.pick(state, ply=0, seed=3)

    assert action == forced
    assert mover.evals_used == 1
    assert mover.sims_used == 0
    assert stub.rows == 1
    assert budget_invariant_breach(mover.evals_used, mover.sims_used) is None


def test_every_move_is_counted_from_zero() -> None:
    """§D3.2 clause 3: the tally is per move, not per game.

    A counter that accumulated across plies would reach the ceiling somewhere in
    the middle of the first game and void a perfectly legal match -- and it would
    do so at a ply that depends on the position, which is the worst possible
    failure to debug.
    """
    mover = GumbelMover(
        stub_factory, draw_score=DRAW_SCORE, player_id=KOMUGI_ID, m=FAST_M, n=FAST_N
    )
    state = fresh_state()

    for ply in range(3):
        action = mover.pick(state, ply=ply, seed=100 + ply)
        state.play(action)
        assert mover.evals_used <= 1 + mover.sims_used

    assert len(mover.evals) == 3
    assert mover.counter.moves_counted == 3
    assert max(mover.evals) <= 1 + FAST_N


def test_the_counted_evaluator_charges_positions_and_not_calls() -> None:
    """A wave of ``k`` leaves costs ``k``, which is what makes the ceiling real.

    Review v2 F3, on the v1 side, and identically here: a per-*call* charge would
    let a batching schedule buy ``257 x k`` evaluations with every §D3.6
    invariant still holding. The refusal is also whole -- a batch that straddles
    the ceiling is not half-served, because a half-served batch is exactly the
    truncation §D3.5 forbids.
    """
    counter = EvalCounter(5)
    counted = CountedEvaluator(StubEvaluator(), counter)
    obs = np.zeros((3, factory.IN_PLANES, 9, 9), dtype=np.float32)
    legal = np.ones((3, factory.NUM_ACTIONS), dtype=np.bool_)

    counter.begin_move()
    counted.evaluate(obs, legal)
    assert counter.served == 3

    with pytest.raises(BudgetRefused):
        counted.evaluate(obs, legal)
    assert counter.served == 3, "a straddling batch must not be part-served"


def test_the_counted_evaluator_counts_calls_and_rows_separately() -> None:
    """kom-011d §6 node 56: ``served`` counts positions, ``calls`` counts forwards.

    Red against a ``calls`` incremented inside ``_rows_in``
    (``gumbel_mover.py:104-123``) or otherwise charged per **row**: under that
    mutant ``calls`` equals ``served`` on every call, so kom-011d's U2 probe
    reads a mean wave size of exactly ``1.0``, §D2.4's cost bracket is read at
    its pessimistic ``w = 8`` end, and the pool's epoch cadence is set several
    times too long -- permanently, because the cadence is pre-registered from
    that measurement. The two halves of the assertion are what separate the two
    numbers: three rows in **one** call, then two rows in **two**.

    The counter's own semantics are untouched here and are pinned next door by
    :func:`test_the_counted_evaluator_charges_positions_and_not_calls`, which
    still passes unchanged -- the new field is additive, and ``EVAL_CEILING``
    is still spent in rows.
    """
    counter = EvalCounter(EVAL_CEILING)
    counted = CountedEvaluator(StubEvaluator(), counter)
    wave = np.zeros((3, factory.IN_PLANES, 9, 9), dtype=np.float32)
    wave_legal = np.ones((3, factory.NUM_ACTIONS), dtype=np.bool_)

    counter.begin_move()
    counted.evaluate(wave, wave_legal)

    assert counter.served == 3, "the budget is still charged in positions"
    assert counted.calls == 1, (
        "a three-leaf wave is ONE forward; a calls counter that moved with the "
        "rows would make the mean wave size read 1.0 whatever the search did"
    )

    scalar_counter = EvalCounter(EVAL_CEILING)
    scalar_counted = CountedEvaluator(StubEvaluator(), scalar_counter)
    single = np.zeros((1, factory.IN_PLANES, 9, 9), dtype=np.float32)
    single_legal = np.ones((1, factory.NUM_ACTIONS), dtype=np.bool_)

    scalar_counter.begin_move()
    scalar_counted.evaluate(single, single_legal)
    scalar_counted.evaluate(single, single_legal)

    assert scalar_counter.served == 2
    assert scalar_counted.calls == 2, (
        "two one-row calls are two forwards: a counter that counted waves "
        "rather than calls would read 1 here"
    )
    assert counted.counter is counter and scalar_counted.counter is scalar_counter

    # F12, the refusal half: a batch the counter REFUSES buys neither a row nor
    # a call. `evaluate` asks first and increments second
    # (``gumbel_mover.py:205-206``), which is the semantics U2 needs -- the mean
    # wave is `served / calls`, and a refused batch was never a forward, so
    # charging it one would put a phantom forward on every move that hits
    # EVAL_CEILING. Red against swapping those two lines (M7), or against
    # wrapping the `ask` in a retry: `calls` would read 1 below. Neither half
    # above can see it, because neither ever refuses a batch.
    refused_counter = EvalCounter(EVAL_CEILING)
    refused_counted = CountedEvaluator(StubEvaluator(), refused_counter)
    refused_counter.begin_move()
    # Spend the ceiling through the counter itself, so the evaluator under test
    # has served nothing at all when the refusal arrives.
    refused_counter.ask(EVAL_CEILING)
    assert refused_counter.served == EVAL_CEILING

    with pytest.raises(BudgetRefused):
        refused_counted.evaluate(single, single_legal)

    assert refused_counted.calls == 0, (
        "a refused batch is not a forward: charging it one biases U2's mean "
        "wave size DOWN, which is §D2.4's pessimistic end and the same "
        "permanent mis-setting of the epoch cadence this node was written "
        "against"
    )
    assert refused_counter.served == EVAL_CEILING, (
        "a straddling batch must not be part-served (§D3.5)"
    )
    assert EVAL_CEILING == 257


# ---------------------------------------------------------------------------
# (d) §D3.5's report-and-refuse, all the way to M8
# ---------------------------------------------------------------------------


def test_the_evaluation_past_the_ceiling_voids_the_match_as_m8() -> None:
    """The 258th ask reaches ``play_bridged_game`` as a void, not as a crash.

    The ceiling is lowered rather than the search inflated: what is under test is
    the *route* an over-budget ask takes -- counter refusal ->
    :class:`~komugi.eval.v1bridge.BudgetBreach` -> §D6 M8, match scope -- and that
    route is identical whatever number the ceiling holds. §D3.5: "there is no
    process boundary to lose it across, and it must not be allowed to propagate
    as a crash of the arena", so the assertion that matters most is that
    ``play_bridged_game`` **returned**.

    M8 is a *match* void and never a pair void: voiding only the affected pair
    would keep exactly the games where the over-spend happened not to show.
    """
    breacher = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id="ckpt-greedy",
        m=FAST_M,
        n=FAST_N,
        ceiling=3,
    )
    opponent = GumbelMover(
        stub_factory, draw_score=DRAW_SCORE, player_id="ckpt-quiet", m=2, n=2
    )

    game = play_bridged_game(
        breacher,
        opponent,
        match_config(games=2),
        np.random.SeedSequence(11),
        pair=0,
        seat=0,
    )

    assert game.is_void
    assert game.outcome is None, "a breached move must never be scored"
    void = game.void
    assert void is not None
    assert void.mode == M8
    assert void.bucket is VoidBucket.BUDGET_BREACH
    assert void.scope is VoidScope.MATCH
    assert void.invariant == "P2"
    assert void.player_id == "ckpt-greedy"
    assert void.ply is not None and void.ply <= 1, (
        "the breach must be booked at the ply it happened on: the breacher "
        "moves first or second, and never later than that at this budget"
    )
    assert void.pair == 0 and void.seat == 0
    assert void.evals_used is not None and void.evals_used > 3, (
        "the void must quantify the breach with the ask that was refused"
    )


def test_a_breach_leaves_the_mover_raising_rather_than_returning_a_move() -> None:
    """The refusal is a construction check, not a truncation mechanism.

    ``pick`` raises; it does not return the best move found so far. A mover that
    served 257 and played a truncated search would trade tree size against budget
    in a way no counter can see, leaving both sides' counters reading 257 and the
    two searches incomparable (§D3.5).
    """
    mover = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id="ckpt-greedy",
        m=FAST_M,
        n=FAST_N,
        ceiling=2,
    )

    with pytest.raises(BudgetBreach) as raised:
        mover.pick(fresh_state(), ply=0, seed=1)

    assert raised.value.player_id == "ckpt-greedy"
    assert raised.value.evals_used > 2


# ---------------------------------------------------------------------------
# (e) determinism
# ---------------------------------------------------------------------------


def test_one_state_and_one_seed_choose_one_action() -> None:
    """The same position and seed give the same move, mover after mover.

    The property the arena's whole determinism claim rests on, restated at the
    move seam: the only randomness in the stack is the per-move seed, so a
    replayed ply is the same ply. Two *different* movers are compared as well as
    two calls on one, because a mover that quietly carried state between picks --
    a tree, a cache, an RNG -- would pass the second comparison and fail this
    one.
    """
    first = GumbelMover(
        stub_factory, draw_score=DRAW_SCORE, player_id=KOMUGI_ID, m=FAST_M, n=FAST_N
    )
    second = GumbelMover(
        stub_factory, draw_score=DRAW_SCORE, player_id=KOMUGI_ID, m=FAST_M, n=FAST_N
    )

    once = first.pick(fresh_state(), ply=0, seed=42)
    twice = first.pick(fresh_state(), ply=0, seed=42)
    elsewhere = second.pick(fresh_state(), ply=7, seed=42)

    assert once == twice
    assert once == elsewhere
    assert first.evals == [first.evals_used, first.evals_used]
    assert second.evals_used == first.evals_used
