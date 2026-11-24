"""Run R5's driver, its mean, its diagnostics, and the gate it decides.

:mod:`komugi.eval.r5` is two things joined by one number, and each half fails in
a different direction, so each is pinned separately here:

* **the driver** -- it must play the pre-registered games (ten pairs, twenty
  games, ceiling 257, ``run_seed = 0xC0FFEE``, the arena's own
  :func:`~komugi.eval.arena.move_seed_for`) and it must report ``Δmean`` over
  **exactly** the plies §D3.6 P4 names: every ply at which that side moved in a
  **scored** pair, forced plies and turn-capped plies included. The two ways to
  get that wrong are silent -- dropping the ``evals == 1`` plies moves the mean
  by more than P4's whole 5.14 budget, and averaging voided pairs in publishes a
  number over games nobody rated;
* **the gate** -- A24: the rated-run entry point refuses to start when the
  committed ``Δmean`` is absent, and dispatches on §D3.6's three-way branch
  rather than on an operator's judgement. A gate that defaulted to "proceed" on a
  missing artefact is not a gate, and it is exactly the failure the pre-flight
  exists to prevent, so the refusal is tested from four different absences.

The end-to-end sections drive the **fake v1** of ``test_v1bridge_behavior`` -- a
real subprocess speaking §D1's protocol -- against a real Komugi net at a budget
CI can pay for. Nothing here needs a v1 checkout, a GPU or a network: R5 itself is
Tier 2 and advisor-executed, and what these tests own is the driver that will run
it.

The fake v1 spends far more per move than a two-simulation Komugi does, which
makes the end-to-end ``Δmean`` strongly negative -- so the branch it lands on
(§D3.6 branch 2, proceed-one-sided) is a real dispatch on a measured number and
not a value planted to make an assertion pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.net")
pytest.importorskip("komugi.selfplay")

# KOM-236: the whole module drives the fake v1 as a real subprocess against a
# real net, which the gate sandbox cannot run.
pytestmark = pytest.mark.sandbox_unsafe

from komugi.engine.action_encoding import (
    A0_START,
    A1_START,
    ACTION_SIZE,
    B_START,
    C_START,
)
from komugi.engine.game import Game
from komugi.eval.arena import ArenaConfig, move_seed_for
from komugi.eval.gumbel_mover import komugi_entrant
from komugi.eval.r5 import (
    BRANCH_FIELD,
    DELTA_MEAN_FIELD,
    KOMUGI_SIDE,
    P4_TOLERANCE,
    PROCEED,
    PROCEED_ONE_SIDED,
    R5_GAMES,
    R5_PAIRS,
    R5_RUN_SEED,
    REFUSE,
    V1_SIDE,
    P4Branch,
    R5Game,
    R5Missing,
    R5Pair,
    R5Result,
    RatedRunRefused,
    _Collector,
    _R5Probe,
    assert_rated_run_permitted,
    d1_orphans,
    ids_without_v1_preimage,
    p4_branch,
    r5_config,
    rated_run_branch,
    run_r5,
)
from komugi.eval.v1bridge import (
    COMPLETED,
    EVAL_CEILING,
    M5,
    BridgeEntrant,
    MoveRecord,
    bridged_book_id,
    resolved_gungi_mode,
)
from komugi.selfplay.adapter import GameState
from test_gumbel_mover import KOMUGI_ID, small_net_factory, stub_factory
from test_v1bridge_behavior import DISAGREE, DRAW_SCORE, stub_entrant

SQUARES: int = 81
"""Every action-id block is plane-major over the destination square."""


def fast_komugi(player_id: str = KOMUGI_ID) -> BridgeEntrant:
    """Return a Komugi entrant at the smallest budget that still searches.

    ``m=2, n=2`` over ``factory.small_serving_net``: the *route* every quantity
    R5 measures travels is identical at two simulations and at 256, and a
    256-simulation cross-engine pair is not a thing CI can pay for. The pinned
    budget is asserted separately, by the rated-mode test below, which is where
    §D3.3's literals actually matter.
    """
    return komugi_entrant(
        player_id, small_net_factory, draw_score=DRAW_SCORE, m=2, n=2
    )


def one_pair() -> ArenaConfig:
    """Return R5's config shrunk to a single pair -- two games, same everything."""
    return r5_config(games=2)


def planted_result(
    games: tuple[R5Game, ...], pairs: tuple[R5Pair, ...] = ()
) -> R5Result:
    """Return an :class:`R5Result` over planted games, with no match behind it.

    The aggregation is a pure function of the per-game records, so planting them
    is how the mean's *definition* gets tested without paying for twenty
    cross-engine games -- and, more to the point, without waiting for a real game
    to happen to contain a forced ply.
    """
    return R5Result(
        games=games,
        pairs=pairs,
        status=COMPLETED,
        reason="",
        pairs_attempted=len({game.pair for game in games}),
        pairs_scored=len({game.pair for game in games if game.scored}),
        ceiling=EVAL_CEILING,
        run_seed=R5_RUN_SEED,
        config={"ceiling": EVAL_CEILING},
    )


def write_artefact(tmp_path: Path, payload: Any, name: str = "r5.json") -> Path:
    """Write `payload` as a committed R5 artefact and return its path."""
    destination = tmp_path / name
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


@pytest.fixture(scope="module")
def clean_config() -> ArenaConfig:
    """Return the one config the shared clean pair below is played under."""
    return one_pair()


@pytest.fixture(scope="module")
def clean_pair(clean_config: ArenaConfig) -> R5Result:
    """Play one clean cross-engine pair once, for every test that reads one.

    Module-scoped because it spawns a subprocess and plays two whole games
    against a real net: three of the assertions below are about different
    properties of *the same run*, and paying for the run three times would buy
    nothing but wall clock. Nothing mutates the result -- :class:`R5Result` is
    frozen and its games are tuples.
    """
    return run_r5(
        fast_komugi(), stub_entrant("v1-fake"), clean_config, rated=False
    )


# ---------------------------------------------------------------------------
# (a) the pre-registered shape
# ---------------------------------------------------------------------------


def test_the_pre_registered_shape_is_ten_pairs_at_the_rated_ceiling() -> None:
    """§D3.5's literals, read off the module rather than restated in a ticket.

    Every one of these is load-bearing and none is a default anyone may tune.
    The ceiling above all: "what shrinks relative to the rated run is the pair
    count, not the budget -- ``Δmean`` is a per-move quantity, so a reduced
    per-move ceiling would measure a different number and would not transfer".
    And ``run_seed`` is pinned so that two people running "the pre-registered
    pre-flight" get the same games (review F8.2).
    """
    config = r5_config()

    assert (R5_PAIRS, R5_GAMES) == (10, 20)
    assert config.games == R5_GAMES
    assert config.pairs == R5_PAIRS
    assert config.seed == R5_RUN_SEED == 0xC0FFEE
    assert config.draw_score == 0.0
    assert config.book_id == bridged_book_id(resolved_gungi_mode())
    assert EVAL_CEILING == 257
    assert P4_TOLERANCE == 5.14
    assert P4_TOLERANCE == pytest.approx(0.02 * EVAL_CEILING), (
        "P4's budget is 2% of the ceiling; if the ceiling moves and the "
        "tolerance does not, the gate is no longer the pre-registered one"
    )


# ---------------------------------------------------------------------------
# (b) diagnostic D-1 (A18): the ids v1 cannot name
# ---------------------------------------------------------------------------


def test_d1_counts_partial_betrayals_and_arata_conversions_and_nothing_else() -> None:
    """The pre-image test, one block at a time, on a planted legal set.

    v1's head is 190 planes to Komugi's 461: 81 route/capture, 81 tsuke/betray
    with **no** selector, and 14 + 14 arata with **no** betray bit. So a route is
    always nameable, an arata conversion never is, and a tsuke/betray is nameable
    only at the *maximal* conversion its square offers -- which is why the
    decline-the-betrayal tsuke below (``sel = 0`` where ``sel = 3`` is also
    legal) is an orphan while the plain tsuke at a square with nothing to convert
    is not.
    """
    route = A0_START + 5 * SQUARES + 9
    plain_tsuke = A1_START + (7 * 4 + 0) * SQUARES + 11
    declined = A1_START + (3 * 4 + 0) * SQUARES + 12
    partial = A1_START + (3 * 4 + 1) * SQUARES + 12
    maximal = A1_START + (3 * 4 + 3) * SQUARES + 12
    drop = B_START + (6 * 2 + 0) * SQUARES + 40
    drop_converting = B_START + (6 * 2 + 1) * SQUARES + 40
    shu_converting = C_START + (2 * 2 + 1) * SQUARES + 3

    orphans = ids_without_v1_preimage(
        [
            route,
            plain_tsuke,
            declined,
            partial,
            maximal,
            drop,
            drop_converting,
            shu_converting,
        ]
    )

    assert orphans == tuple(
        sorted((declined, partial, drop_converting, shu_converting))
    )
    assert route not in orphans, "v1 names every route and capture"
    assert plain_tsuke not in orphans, (
        "a tsuke at a square offering no conversion IS the maximal conversion "
        "there, so v1's canonicalising decoder reaches it"
    )
    assert maximal not in orphans


def test_d1_is_scoped_to_the_square_it_is_measured_at() -> None:
    """A maximal conversion at one square does not excuse a partial at another.

    The grouping key is ``(from_sq, to_sq)``: the qualifying mask is a property
    of the *target tower*, so a run that took one global maximum -- or that
    grouped on the destination alone -- would silently under-count exactly the
    positions B3 is about, the ones with more than one convertible tower in
    reach.
    """
    here_partial = A1_START + (1 * 4 + 1) * SQUARES + 20
    here_maximal = A1_START + (1 * 4 + 3) * SQUARES + 20
    there_only = A1_START + (2 * 4 + 1) * SQUARES + 20
    same_from_other_to = A1_START + (1 * 4 + 2) * SQUARES + 21

    orphans = ids_without_v1_preimage(
        [here_partial, here_maximal, there_only, same_from_other_to]
    )

    assert orphans == (here_partial,)
    assert there_only not in orphans, (
        "selector 1 is the whole of what that square offers, so it is that "
        "square's maximal conversion and v1 reaches it"
    )
    assert same_from_other_to not in orphans


def test_d1_refuses_an_id_from_another_action_space() -> None:
    """An id outside ``[0, ACTION_SIZE)`` has no pre-image *here*, and says so.

    Answering "not an orphan" for a foreign id would let a caller that handed
    over v1's own 15390-wide ids read a reassuring zero out of a measurement that
    never looked at Komugi's space at all.
    """
    with pytest.raises(ValueError, match="outside Komugi's action space"):
        ids_without_v1_preimage([ACTION_SIZE])
    with pytest.raises(ValueError, match="outside Komugi's action space"):
        ids_without_v1_preimage([-1])


def test_d1_is_zero_in_the_opening_position() -> None:
    """A18's second half: the count is ``0`` in the opening position.

    Nothing is stacked at the start, so nothing is convertible, so Komugi's legal
    set and v1's coincide there. A non-zero answer here would mean the predicate
    is counting something other than the conversion asymmetry -- and would make
    every game's D-1 count uninterpretable, because the baseline would not be
    zero.
    """
    assert d1_orphans(GameState(Game(), DRAW_SCORE)) == ()


# ---------------------------------------------------------------------------
# (c) §D3.6 P4's mean: every ply of a scored pair, forced plies included
# ---------------------------------------------------------------------------


def test_the_mean_counts_forced_and_turn_capped_plies() -> None:
    """P4's definition, with the two ``evals == 1`` plies planted deliberately.

    §D3.4: a **forced** position costs exactly one evaluation and runs no
    simulation, and at the turn-capped final ply of every capped game a
    *non-forced* position also costs exactly 1 because every legal move leads to
    a terminal. P4 includes both -- "over every ply at which that side moved in a
    scored pair, forced plies and turn-capped plies included" -- and rev 2's
    exclusion clause was deleted rather than kept (re-review F8).

    The two definitions are 64.4 evaluations apart on this data, twelve times
    P4's entire 5.14 budget: an implementation that dropped the ones would pass a
    smell test and fail the gate for a reason nobody could see.
    """
    forced_first = R5Game(
        pair=0,
        seat=0,
        evals_k=(1, 257, 129, 1),  # ply 0 forced; ply 3 the turn-cap.
        evals_v1=(200, 200),
        scored=True,
    )
    second = R5Game(pair=0, seat=1, evals_k=(100,), evals_v1=(2,), scored=True)
    result = planted_result((forced_first, second))

    assert result.evals_k == (1, 257, 129, 1, 100)
    assert result.mean_k == pytest.approx(488 / 5)
    assert result.mean_k != pytest.approx(486 / 3), (
        "the two evals == 1 plies were dropped: that is rev 2's deleted "
        "exclusion clause, reintroduced"
    )
    assert result.mean_v1 == pytest.approx(402 / 3)
    assert result.delta_mean == pytest.approx(488 / 5 - 402 / 3)


def test_only_scored_pairs_reach_a_mean_and_every_game_reaches_a_diagnostic() -> None:
    """§D3.6, restated once for the whole section, and its exact consequence.

    "Every mean in P4 and P5 is computed over the plies of **scored** pairs
    only." A voided pair's plies are therefore invisible to ``Δmean`` -- but its
    ``no_move`` and D-1 counts are *not*, and must not be: a ``no_move`` **is**
    an M5, an M5 voids the pair, so a diagnostic restricted to scored pairs would
    be identically zero by construction and would report "mechanism 2 never
    fired" no matter how often it fired.
    """
    scored = R5Game(
        pair=0, seat=0, evals_k=(10, 10), evals_v1=(20, 20), scored=True
    )
    scored_mirror = R5Game(
        pair=0, seat=1, evals_k=(10,), evals_v1=(20,), scored=True
    )
    voided = R5Game(
        pair=1,
        seat=0,
        evals_k=(250, 250),
        evals_v1=(1, 1),
        d1_plies=4,
        d1_plies_v1_to_move=2,
        d1_ids=9,
        no_move_plies=1,
        void_mode=M5,
        scored=False,
    )
    result = planted_result((scored, scored_mirror, voided))

    assert result.mean_k == pytest.approx(10.0)
    assert result.mean_v1 == pytest.approx(20.0)
    assert result.delta_mean == pytest.approx(-10.0)

    artefact = result.to_json()
    assert artefact["no_move_plies_total"] == 1, (
        "the voided pair's M5 must still be reported: it is mechanism 2's only "
        "observable shadow"
    )
    assert artefact["d1_plies_total"] == 4
    assert artefact["d1_ids_total"] == 9
    assert len(artefact["per_game"]) == 3
    assert [row["scored"] for row in artefact["per_game"]] == [True, True, False]


def test_a_run_that_scored_nothing_reports_no_delta_mean_at_all() -> None:
    """No scored pair, no mean -- and ``None`` rather than ``0.0``.

    A zero would be a ``Δmean`` inside P4's band, i.e. a run that measured
    nothing dispatching the rated run on branch 1. The mean does not exist, the
    artefact says so, and :func:`p4_branch` refuses it as A24's blocking state.
    """
    result = planted_result(
        (R5Game(pair=0, seat=0, evals_k=(5,), evals_v1=(5,), scored=False),)
    )

    assert result.mean_k is None
    assert result.mean_v1 is None
    assert result.delta_mean is None
    assert result.to_json()[DELTA_MEAN_FIELD] is None
    assert result.to_json()[BRANCH_FIELD] is None
    with pytest.raises(R5Missing, match="no Δmean"):
        _ = result.branch


def test_the_artefact_carries_every_number_d3_5_asks_r5_to_report() -> None:
    """§D3.5's report list, field by field, in one JSON object.

    ``mean_K``, ``mean_v1``, ``Δmean``, the per-pair spread, and the per-game D-1
    and ``no_move`` counts. The per-ply lists are carried too, so the committed
    number can be recomputed by a reader who does not trust this module -- which
    is the whole point of committing an artefact rather than a sentence.
    """
    first = R5Game(pair=0, seat=0, evals_k=(10, 12), evals_v1=(30,), scored=True)
    second = R5Game(
        pair=0, seat=1, evals_k=(14,), evals_v1=(30, 30), d1_plies=2, scored=True
    )
    third = R5Game(pair=1, seat=0, evals_k=(20,), evals_v1=(21,), scored=True)
    fourth = R5Game(pair=1, seat=1, evals_k=(20,), evals_v1=(21,), scored=True)
    pairs = (
        R5Pair(
            pair=0,
            mean_k=12.0,
            mean_v1=30.0,
            delta_mean=-18.0,
            plies_k=3,
            plies_v1=3,
        ),
        R5Pair(
            pair=1,
            mean_k=20.0,
            mean_v1=21.0,
            delta_mean=-1.0,
            plies_k=2,
            plies_v1=2,
        ),
    )
    artefact = planted_result((first, second, third, fourth), pairs).to_json()

    assert artefact["run"] == "R5"
    assert artefact["ceiling"] == EVAL_CEILING
    assert artefact["run_seed"] == R5_RUN_SEED
    assert artefact["mean_K"] == pytest.approx(76 / 5)
    assert artefact["mean_v1"] == pytest.approx(132 / 5)
    assert artefact[DELTA_MEAN_FIELD] == pytest.approx(76 / 5 - 132 / 5)
    assert artefact["mean_K_fraction_of_ceiling"] == pytest.approx(
        (76 / 5) / EVAL_CEILING
    )
    assert artefact["per_pair_delta_mean"] == {
        "min": pytest.approx(-18.0),
        "max": pytest.approx(-1.0),
        "range": pytest.approx(17.0),
    }
    assert [row["pair"] for row in artefact["per_pair"]] == [0, 1]
    assert artefact["d1_plies_total"] == 2
    assert artefact["per_game"][0]["evals_K"] == [10, 12]
    assert artefact["per_game"][0]["mean_K"] == pytest.approx(11.0)
    assert artefact["max_evals_K"] == 20
    assert json.loads(json.dumps(artefact)) == json.loads(json.dumps(artefact)), (
        "the artefact must be JSON-serialisable: it is committed to a ticket"
    )


# ---------------------------------------------------------------------------
# (d) §D3.6's three-way branch, on planted Δmean values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.0, 1.0, -1.0, 5.14, -5.14, 5.139, -5.139])
def test_branch_one_proceeds_inside_p4s_budget(delta: float) -> None:
    """``|Δmean| ≤ 5.14`` → the rated run proceeds, citable symmetrically.

    The boundary is **inclusive** on both sides: P4 is stated as ``|Δmean| ≤
    5.14``, and a strict comparison here would refuse a run the pre-registration
    admits -- moving the threshold, which is the one thing none of the three arms
    may do.
    """
    branch = p4_branch(delta)

    assert branch.branch == PROCEED
    assert branch.rated_run_permitted
    assert not branch.one_sided_only
    assert branch.tolerance == P4_TOLERANCE
    assert branch.delta_mean == pytest.approx(delta)


@pytest.mark.parametrize("delta", [-5.1401, -6.0, -40.0, -257.0])
def test_branch_two_proceeds_one_sided_when_v1_spent_more(delta: float) -> None:
    """``Δmean < −5.14`` → proceed, but the anchor is a lower bound only.

    v1 spent materially more compute than Komugi, which is the *conservative*
    sign: the measured gap understates Komugi if anything. So the run happens and
    the restriction lands on the reporting -- "it may **never** be cited as a
    symmetric 'Komugi vs v1 at equal compute' number, in KOM-011 or anywhere
    else", which is what :attr:`P4Branch.one_sided_only` carries.
    """
    branch = p4_branch(delta)

    assert branch.branch == PROCEED_ONE_SIDED
    assert branch.rated_run_permitted
    assert branch.one_sided_only
    assert "lower bound" in branch.rationale
    assert "one-sided" in branch.rationale


@pytest.mark.parametrize("delta", [5.1401, 6.0, 40.0, 257.0])
def test_branch_three_refuses_the_ceiling_when_komugi_spent_more(
    delta: float,
) -> None:
    """``Δmean > +5.14`` → **no rated anchor run at this ceiling.**

    The dangerous sign: the resulting number would flatter Komugi by an
    unmeasured amount, in exactly the direction §D3.7 identifies. The
    pre-registered outcome is not a retune but an escalation -- KOM-011 may not
    carry a single-number v1 anchor unless §D3 is re-pre-registered as a rev 4,
    before any rated game.
    """
    branch = p4_branch(delta)

    assert branch.branch == REFUSE
    assert not branch.rated_run_permitted
    assert not branch.one_sided_only
    assert "rev 4" in branch.rationale


def test_the_three_arms_partition_the_line_and_the_threshold_never_moves() -> None:
    """Every real ``Δmean`` lands in exactly one arm, at one fixed tolerance.

    Written as a sweep rather than as three cases because the failure this
    catches is a gap or an overlap at the boundary -- the place a hand-written
    three-way ``if`` goes wrong -- and because a tolerance that varied by arm
    would be P4 widened in one direction only.
    """
    arms = {PROCEED: 0, PROCEED_ONE_SIDED: 0, REFUSE: 0}
    for step in range(-2000, 2001):
        branch = p4_branch(step / 100.0)
        assert branch.branch in arms
        assert branch.tolerance == P4_TOLERANCE
        arms[branch.branch] += 1

    assert sum(arms.values()) == 4001, "a Δmean fell into no arm at all"
    assert all(count > 0 for count in arms.values()), (
        f"an arm is unreachable across ±20 evaluations of Δmean: {arms}"
    )
    assert p4_branch(P4_TOLERANCE).branch == PROCEED
    assert p4_branch(-P4_TOLERANCE).branch == PROCEED
    assert p4_branch(P4_TOLERANCE * 1.001).branch == REFUSE
    assert p4_branch(-P4_TOLERANCE * 1.001).branch == PROCEED_ONE_SIDED


# ---------------------------------------------------------------------------
# (e) A24's refusal: no committed Δmean, no rated run
# ---------------------------------------------------------------------------


def test_the_rated_run_refuses_to_start_without_a_committed_artefact(
    tmp_path: Path,
) -> None:
    """A24, the headline: the gate raises, and it names the artefact it wanted.

    "Its absence blocks the rated run outright." A refusal that did not name the
    missing file would be resolved by an operator guessing which of R1, R2, R3
    and R5 was meant -- so the path is asserted to be *in the message*, not just
    the exception type.
    """
    missing = tmp_path / "kom-023-r5.json"

    with pytest.raises(R5Missing) as raised:
        rated_run_branch(missing)

    assert str(missing) in str(raised.value)
    assert "R5" in str(raised.value)

    with pytest.raises(R5Missing):
        assert_rated_run_permitted(missing)


def test_the_gate_refuses_an_artefact_whose_delta_mean_is_absent_or_null(
    tmp_path: Path,
) -> None:
    """Three ways for the number to be missing, one answer to all of them.

    An artefact with no ``delta_mean``, one that reports ``null`` (R5 ran and
    scored no pair), and one that reports something that is not a number. None of
    them is a proceed: P4 is gated on a measurement, and a rated run started
    without one is a run gated on nothing.
    """
    with pytest.raises(R5Missing, match="carries no"):
        rated_run_branch(write_artefact(tmp_path, {"run": "R5"}, "absent.json"))

    with pytest.raises(R5Missing, match="null"):
        rated_run_branch(
            write_artefact(tmp_path, {DELTA_MEAN_FIELD: None}, "null.json")
        )

    with pytest.raises(R5Missing, match="not a number"):
        rated_run_branch(
            write_artefact(tmp_path, {DELTA_MEAN_FIELD: "0.0"}, "text.json")
        )


def test_the_gate_refuses_an_unparseable_or_non_object_artefact(
    tmp_path: Path,
) -> None:
    """A file that is there but is not an R5 result is an absence, not a zero."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(R5Missing, match="not readable JSON"):
        rated_run_branch(broken)

    with pytest.raises(R5Missing, match="not a JSON object"):
        rated_run_branch(write_artefact(tmp_path, [1, 2, 3], "list.json"))


def test_the_gate_re_derives_the_branch_and_refuses_a_tampered_one(
    tmp_path: Path,
) -> None:
    """A24: the dispatch is on the branch function, not on a stored decision.

    The artefact records the branch it derived, which is what a reader of
    ``tickets/KOM-023.md`` reads. If the gate trusted that field, "proceed" would
    be one text edit away from any ``Δmean`` at all -- so the gate recomputes
    from the number and refuses the artefact when the two disagree, rather than
    silently preferring one.
    """
    tampered = write_artefact(
        tmp_path,
        {DELTA_MEAN_FIELD: 42.0, BRANCH_FIELD: {"branch": PROCEED}},
        "tampered.json",
    )

    with pytest.raises(R5Missing, match="unknown provenance"):
        rated_run_branch(tampered)


def test_a_committed_result_round_trips_through_the_gate(tmp_path: Path) -> None:
    """Write the artefact, read it back, get the same branch off the same number.

    The one test that proves writer and reader agree on the field name. They are
    the two ends of a gate that spans a commit and a code review, and a
    disagreement would make the gate either unsatisfiable or vacuous, with
    nothing in either half looking wrong on its own.
    """
    result = planted_result(
        (
            R5Game(pair=0, seat=0, evals_k=(100, 102), evals_v1=(101,), scored=True),
            R5Game(pair=0, seat=1, evals_k=(101,), evals_v1=(100, 102), scored=True),
        )
    )
    written = result.write_json(tmp_path / "kom-023-r5.json")

    branch = assert_rated_run_permitted(written)

    assert result.delta_mean == pytest.approx(0.0)
    assert branch.branch == PROCEED
    assert branch.delta_mean == pytest.approx(0.0)
    assert branch == p4_branch(result.delta_mean)


def test_branch_three_stops_the_rated_run_with_its_own_exception(
    tmp_path: Path,
) -> None:
    """A refusal by branch 3 is not the same failure as a missing artefact.

    "The number is not there" and "the number says no" have different repairs --
    run R5, versus re-pre-register §D3 as a rev 4 around the measured asymmetry
    -- and a caller that caught one type for both would report the second as the
    first, which is how a refusal turns into a to-do.
    """
    refusing = planted_result(
        (
            R5Game(pair=0, seat=0, evals_k=(200,), evals_v1=(100,), scored=True),
        )
    ).write_json(tmp_path / "refuse.json")

    assert rated_run_branch(refusing).branch == REFUSE
    with pytest.raises(RatedRunRefused, match="rev 4"):
        assert_rated_run_permitted(refusing)
    assert not isinstance(RatedRunRefused("x"), R5Missing)


# ---------------------------------------------------------------------------
# (f) the driver, end to end against the fake v1
# ---------------------------------------------------------------------------


def test_the_driver_plays_the_bridged_pair_and_reports_delta_mean(
    clean_pair: R5Result,
) -> None:
    """The whole pre-flight, one pair of it, against a real v1-shaped subprocess.

    Five things could only be proved by playing it:

    * the driver's instrumentation does not break the bridge -- the pair is
      scored as an ordinary completed match, with no void;
    * ``mean_K`` and ``mean_v1`` are taken over exactly the plies that were
      played, which is checked against the per-game rows rather than restated;
    * the per-game evaluation lists are the ones the two counting seams produced,
      and every one is inside the ceiling the artefact names;
    * the artefact serialises;
    * the branch is a real dispatch on a measured number. The fake v1 answers
      ``1 + (seed % 256)`` evaluations per move against a two-simulation Komugi,
      so ``Δmean`` is strongly negative and §D3.6 branch 2 fires -- the
      proceed-one-sided arm, reached by measurement and not by planting.
    """
    result = clean_pair

    assert result.status == COMPLETED, f"{result.status}: {result.reason}"
    assert (result.pairs_attempted, result.pairs_scored) == (1, 1)
    assert len(result.games) == 2
    assert [game.seat for game in result.games] == [0, 1]
    assert all(game.scored for game in result.games)
    assert all(game.void_mode is None for game in result.games)

    assert result.mean_k is not None and result.mean_v1 is not None
    assert result.delta_mean == pytest.approx(result.mean_k - result.mean_v1)
    assert result.evals_k == tuple(
        value for game in result.games for value in game.evals_k
    )
    assert result.mean_k == pytest.approx(
        sum(result.evals_k) / len(result.evals_k)
    )
    assert 1 <= min(result.evals_k) and max(result.evals_k) <= EVAL_CEILING
    assert 1 <= min(result.evals_v1) and max(result.evals_v1) <= EVAL_CEILING

    artefact = result.to_json()
    assert artefact[DELTA_MEAN_FIELD] == pytest.approx(result.delta_mean)
    assert artefact["config"]["book_id"] == bridged_book_id(resolved_gungi_mode())
    assert artefact["config"]["root_noise"] == "off"
    assert artefact["config"]["tree_reuse"] == "off"
    assert json.dumps(artefact)

    assert result.branch.branch == PROCEED_ONE_SIDED, (
        f"the fake v1 outspends a two-simulation Komugi by construction, so "
        f"Δmean must be far below −{P4_TOLERANCE}; got {result.delta_mean}"
    )
    assert len(result.pairs) == 1
    assert result.pairs[0].delta_mean == pytest.approx(result.delta_mean)


def test_the_driver_seeds_every_ply_with_the_arenas_own_derivation(
    clean_pair: R5Result, clean_config: ArenaConfig
) -> None:
    """§D3.5's "same seed derivation ``move_seed_for(config, pair, seat, ply)``".

    Asserted against :func:`~komugi.eval.arena.move_seed_for` itself, ply by ply,
    because "the same games the rated run will play" is the property R5's entire
    transferability rests on and it is a property of the *seeds*. A second
    schedule that happened to agree at ply 0 and diverged at ply 7 would be
    invisible in every other assertion in this file.
    """
    result = clean_pair

    for game in result.games:
        assert game.seeds, "no ply of this game recorded a seed"
        assert game.seeds == tuple(
            move_seed_for(clean_config, game.pair, game.seat, ply)
            for ply in range(len(game.seeds))
        )
    assert result.games[0].seeds != result.games[1].seeds, (
        "the two seats of a pair must draw different Gumbel streams, or the "
        "swap measures the same game twice"
    )


def test_a_no_move_voids_the_pair_and_still_reports_its_m5() -> None:
    """Mechanism 2's observable shadow, counted, on a run that scores nothing.

    The fake v1 is told to answer ``no_move`` at a position Komugi has not
    adjudicated: that is §D6 M5, it voids the pair, and the pair is therefore
    never scored -- so ``Δmean`` does not exist and the rated run is blocked by
    A24 rather than dispatched on a mean taken over one seat's worth of a broken
    pair. The M5 itself is reported per game, which is the only way mechanism 2
    is visible at all.
    """
    result = run_r5(
        fast_komugi(),
        stub_entrant("v1-fake", DISAGREE, 0, 2),
        one_pair(),
        rated=False,
    )

    assert result.status == COMPLETED
    assert (result.pairs_attempted, result.pairs_scored) == (1, 0)
    assert result.pairs == ()
    assert result.delta_mean is None
    assert result.mean_k is None

    voided = result.games[0]
    assert voided.void_mode == M5
    assert voided.no_move_plies == 1
    assert not voided.scored
    assert result.to_json()["no_move_plies_total"] == 1
    assert result.to_json()[DELTA_MEAN_FIELD] is None

    with pytest.raises(R5Missing):
        _ = result.branch


def test_the_diagnostics_are_recorded_for_every_ply_of_every_game(
    clean_pair: R5Result,
) -> None:
    """D-1 and the per-side attribution are accumulated, not left at zero.

    The count itself is position-dependent and this pair is far too short to
    guarantee a conversion opportunity, so what is pinned is the *bookkeeping*:
    the per-side split never exceeds the total, the id total never falls below
    the ply count when a ply fired, and the artefact's totals are the sum of the
    rows. A probe that computed D-1 and forgot to file it -- the plausible
    failure -- fails the last assertion.
    """
    result = clean_pair
    artefact = result.to_json()

    for game in result.games:
        assert 0 <= game.d1_plies_v1_to_move <= game.d1_plies <= game.plies
        assert game.d1_ids >= game.d1_plies
        assert game.no_move_plies == 0
    assert artefact["d1_plies_total"] == sum(
        game.d1_plies for game in result.games
    )
    assert artefact["d1_plies_v1_to_move_total"] == sum(
        game.d1_plies_v1_to_move for game in result.games
    )
    assert artefact["d1_ids_total"] == sum(game.d1_ids for game in result.games)


def test_a_rated_r5_enforces_the_pre_registered_book_id() -> None:
    """``rated=True`` is the default, and it is the bridge's own gate, not a copy.

    R5 measures a *rated* quantity at the rated ceiling, so the pre-registered
    configuration is enforced -- §D3.3's literals (A15) and §D4 amendment 2's
    ``book_id``. The refusal happens at match construction, before either side is
    built, which is why this test costs no subprocess and no net.
    """
    unlabelled = ArenaConfig(
        games=2, seed=R5_RUN_SEED, draw_score=DRAW_SCORE, book_id="empty"
    )

    with pytest.raises(ValueError, match="book_id"):
        run_r5(
            komugi_entrant(KOMUGI_ID, stub_factory, draw_score=DRAW_SCORE, m=2, n=2),
            stub_entrant("v1-fake"),
            unlabelled,
        )


# ---------------------------------------------------------------------------
# (g) the probe's own construction check
# ---------------------------------------------------------------------------


class MuteMover:
    """A :class:`~komugi.eval.v1bridge.Mover` that never says what it spent.

    Exactly the shape a future third engine adapter could land in: four correct
    methods and no per-move accounting. R5's whole output is what the two sides
    spent, so seating this in a pre-flight would produce a ``Δmean`` over an
    empty set of plies -- ``None``, which the gate then reads as "R5 was never
    run". A refusal at construction says the true thing instead.
    """

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Return the first legal action, spending nothing anyone can see."""
        del ply, seed
        return int(state.legal_mask().nonzero()[0][0])

    def observe(self, action: int, record: MoveRecord) -> None:
        """Ignore the notification."""
        del action, record

    def reset(self, *, pair: int, seat: int) -> None:
        """Ignore the seating."""
        del pair, seat

    def close(self) -> None:
        """Nothing to release."""


def test_the_probe_refuses_a_mover_that_cannot_say_what_it_spent() -> None:
    """A mover with no ``evals`` list is refused where it is seated, not later.

    The alternative failure is the quiet one: twenty games played, an artefact
    committed with ``delta_mean: null``, and a gate that correctly refuses the
    rated run for a reason that has nothing to do with the engines.
    """
    with pytest.raises(TypeError, match="no per-move 'evals' list"):
        _R5Probe(MuteMover(), _Collector(), KOMUGI_SIDE)


def test_the_probe_forwards_the_referee_ahead_view_it_wraps() -> None:
    """§D6 M6's diagnostic survives the wrapper, and reads ``None`` without one.

    ``play_bridged_game`` reads ``last_win_state`` off the mover with
    ``getattr``: a wrapper that did not forward it would zero the
    ``referee_ahead`` count for every R5 run and for every rated run that reused
    this driver, silently. Komugi's own mover has no such view, and ``None`` is
    the honest answer there -- ``_referee_ahead`` accepts only a tuple.
    """
    collector = _Collector()
    komugi_probe = _R5Probe(
        komugi_entrant(
            KOMUGI_ID, stub_factory, draw_score=DRAW_SCORE, m=2, n=2
        ).mover_factory(),
        collector,
        KOMUGI_SIDE,
    )

    assert komugi_probe.last_win_state is None
    komugi_probe.close()

    v1_probe = _R5Probe(stub_entrant("v1-fake").mover_factory(), collector, V1_SIDE)
    try:
        assert v1_probe.last_win_state == (0, 0, 0)
    finally:
        v1_probe.close()


def test_the_branch_value_object_carries_both_permissions() -> None:
    """``P4Branch`` is a value, not a string, and the two permissions differ.

    Branch 2 permits the run and forbids the symmetric citation. A caller handed
    the bare text ``"proceed_one_sided"`` would have to re-derive that pairing,
    and the failure of re-deriving it wrongly is a KOM-011 anchor quoted as
    "Komugi vs v1 at equal compute" -- the exact sentence §D3.6 forbids.
    """
    one_sided = P4Branch(
        branch=PROCEED_ONE_SIDED,
        delta_mean=-10.0,
        tolerance=P4_TOLERANCE,
        rationale="planted",
    )

    assert one_sided.rated_run_permitted and one_sided.one_sided_only
    assert one_sided.to_json() == {
        "branch": PROCEED_ONE_SIDED,
        DELTA_MEAN_FIELD: -10.0,
        "tolerance": P4_TOLERANCE,
        "rated_run_permitted": True,
        "one_sided_only": True,
        "rationale": "planted",
    }
