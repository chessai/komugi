"""§D5 nodes: the record sets, the gate, its conditions, and its resolution.

Design: ``docs/design/kom-011d-ladder-pool.md`` (rev 3), §6 nodes 24, 25, 26,
27, 40, 41, 44 and 55.

The fixtures are built two ways, deliberately. The **condition** nodes plant
links by hand at exact win/draw/loss counts, so that a window fails exactly one
validity condition and the node cannot pass by way of another. The **gate**
nodes (25, 40) sample from planted ratings through
:func:`komugi.eval.synth.simulate_match_set` over the schedule's real game
counts, because what they measure -- does the gate fire at a planted
divergence, and does it stay quiet at none -- is a property of the estimator's
sampling distribution and not of any single record set.

Node 40 is an **in-suite smoke** at the counts §D5.3 pins (10 trials at each of
``|D| in {0, 100}``, ``B = 100``). The >=100-trial power measurement behind
§D5.3's table is an artefact -- ``gungi-az-runs/scripts/pool_coverage.py`` --
because a measurement of that size inside ``python -m pytest -q``, which
``ci/check.sh:22`` runs with no deselection, is 10-20 minutes of ``fit_elo``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

import pytest

from komugi.eval import divergence
from komugi.eval.divergence import (
    CONDITION_BOOK_ERA,
    CONDITION_COMPONENTS,
    CONDITION_CONVERGED,
    CONDITION_DISCARD_RATE,
    CONDITION_GAMES_AND_DECISIVE,
    CONDITION_LINKS_SATURATED,
    CONDITION_WINDOW_NOT_FORMED,
    INFORMATIVE_BAND,
    THRESHOLD_SINGLE_ELO,
    THRESHOLD_SUSTAINED_ELO,
    Verdict,
    WindowKind,
    evaluate_window,
    partition_records,
    threshold_for,
    window_gains,
)
from komugi.eval.elo import EloResult, fit_elo
from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord, MatchSet
from komugi.eval.synth import davidson_probabilities, simulate_match_set

_A: str = "komugi/ckpt-3000@aaaa0001"
_B: str = "komugi/ckpt-6000@aaaa0002"
_A400: str = f"{_A}#anchor400"
_B400: str = f"{_B}#anchor400"
_VISITS: str = f"{_B}#n1600"
_ANCHOR: str = "v1/iter-0020@7eef51d0"
_REF_LOW: str = "rawmcts/n100@0f31ab70"
_REF_HIGH: str = "rawmcts/n6400@0f31ab78"
_SPARE: str = "rawmcts/n800@0f31ab77"
_EXTERNAL: str = "external/site-2026-09-08"

_ROSTER_MAP: dict[str, str] = {
    _A: "komugi",
    _B: "komugi",
    _A400: "komugi",
    _B400: "komugi",
    _VISITS: "komugi",
    _ANCHOR: "v1",
    _REF_LOW: "rawmcts",
    _REF_HIGH: "rawmcts",
    _SPARE: "rawmcts",
    _EXTERNAL: "external",
}

_WINDOW: tuple[str, str] = (_A, _B)
_BOOK: str = "E007-3f9c1a20"
_SEED: int = 20260908

_CROSS_PAIRS: int = 64
"""64 seat-swapped pairs is §D2.1's 128 games per matchup."""

_IN_FAMILY_PAIRS: int = 128
"""§D2.2(b)1's 256 games: the link that carries ``G_in`` alone."""

_EXTERNAL_PAIRS: int = 25
"""KOM-017's >=50-game cadence, the declared deviation of §D2.2(b)4."""

_DRIFT_PAIRS: int = 50
"""The site leg length the F1 fixture plants a **drift** at (§D6.1, U4).

Longer than :data:`_EXTERNAL_PAIRS` on purpose, and this is the review's own
"tuning knob for the precondition". The External vertex reaches ``r_cross(B)``
through two series links -- ``external -- B#anchor400`` at ``2 * pairs`` games
and the 128-game ``B#anchor400 -- B`` tie-in -- and the pendant's information
is ``n * p(1-p)``, so at 25 pairs the largest displacement a site swing can
put on ``G_cross`` is ~35 Elo: ``|D|`` lands near 65, only 10 Elo clear of
``T = 75``, which is too close to the threshold to be a *precondition*. At 50
pairs the same swing moves ``G_cross`` by ~55 Elo, which is also the ">=40
Elo" §6 node 45 asks a site-drift stand-in to move ``r_cross(B)`` by.
"""

_DRIFT_WINS: int = 10
_DRIFT_LOSSES: int = 40
"""The site legs' split, planted at the score rate where the pendant pulls hardest.

The pull a pendant puts on its neighbour is the product of the fitted gap and
the link's information, ``|logit(p)| * n * p(1-p)``, which is maximised near
``p ~ 0.19`` and *falls* toward either extreme -- a 0-50 sweep moves less than
a 10-40 loss, not more, because a saturated link carries almost no information.
``10-40`` is ``p = 0.20``.
"""


def _link(
    first: str,
    second: str,
    *,
    pairs: int,
    wins: int,
    losses: int,
    book: str = _BOOK,
) -> list[GameRecord]:
    """Return `pairs` seat-swapped pairs at an exact win/loss/draw split.

    A "win" pair is 2-0 to `first` and a "loss" pair 0-2, so the score rate is
    ``(2 * wins + draws) / (2 * pairs)`` and the decisive count is
    ``2 * (wins + losses)`` -- the two quantities §D5.3's conditions 4 and 6
    read, set independently of each other, which is what lets node 55 hold the
    decisive count fixed while moving the score rate.
    """
    if wins + losses > pairs:
        raise ValueError("wins + losses cannot exceed the pair count")
    records: list[GameRecord] = []
    for index in range(pairs):
        pair_id = f"{first}:{second}:{index:05d}"
        if index < wins:
            first_seat, second_seat = WHITE_WIN, BLACK_WIN
        elif index < wins + losses:
            first_seat, second_seat = BLACK_WIN, WHITE_WIN
        else:
            first_seat, second_seat = DRAW, DRAW
        records.append(
            GameRecord(
                white=first,
                black=second,
                result=first_seat,
                book_id=book,
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=second,
                black=first,
                result=second_seat,
                book_id=book,
                pair_id=pair_id,
            )
        )
    return records


def _even(
    first: str, second: str, *, pairs: int = _CROSS_PAIRS, book: str = _BOOK
) -> list[GameRecord]:
    """Return an evenly-split link: score rate 0.5, every game decisive."""
    return _link(
        first, second, pairs=pairs, wins=pairs // 2, losses=pairs // 2, book=book
    )


def _per_pair_entry_ids(
    records: Sequence[GameRecord], era: str
) -> list[GameRecord]:
    """Return `records` re-booked with **one §D3.2 entry id per pair**, all in `era`.

    Every other fixture in this file carries one ``book_id`` per *link*, which
    is the one shape :func:`~komugi.eval.divergence.book_era` cannot be
    distinguished from the identity on. §D3.2/§D3.4 give each pair its own
    entry -- ``E007-<sha8>``, 64 distinct ids inside a single 128-game matchup
    -- so this is what the real records look like from T2b onward.
    """
    entries: dict[str, str] = {}
    rebooked: list[GameRecord] = []
    for record in records:
        entry = entries.setdefault(record.pair_id, f"{era}-{len(entries):08x}")
        rebooked.append(
            GameRecord(
                white=record.white,
                black=record.black,
                result=record.result,
                book_id=entry,
                pair_id=record.pair_id,
            )
        )
    return rebooked


def _davidson_link(
    first: str,
    second: str,
    *,
    delta: float,
    pairs: int,
    draw_param: float = 0.9,
) -> list[GameRecord]:
    """Return a link whose counts are the Davidson **expectation** at `delta`.

    Planted at the model's own expected counts rather than sampled from them,
    so the fitted gain is the planted one up to integer rounding: the MLE's
    likelihood equations set each player's observed score equal to its expected
    score, and data at expectation satisfy them at the true parameters. That
    makes node 25 a test of the *gate* -- does a 100-Elo divergence fire -- and
    not a coin flip on whether one draw of the sampling distribution happened
    to land above a threshold the design's own power table puts it 0.78 above.
    Node 40, whose whole subject **is** the sampling distribution, samples.
    """
    p_win, p_draw, _ = davidson_probabilities(delta, draw_param)
    wins = round(pairs * p_win)
    draws = round(pairs * p_draw)
    losses = pairs - wins - draws
    return _link(first, second, pairs=pairs, wins=wins, losses=losses)


def _healthy_pool() -> list[GameRecord]:
    """Return a pool that satisfies all six validity conditions.

    Every link is evenly split, so the fit is identified, the graph is one
    component, every endpoint link is at the design's game count with every
    game decisive, and every endpoint link's score rate is 0.5 -- squarely
    inside condition 6's ``[0.15, 0.85]``.
    """
    records: list[GameRecord] = [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _ANCHOR),
        *_even(_A, _REF_LOW),
        *_even(_A, _REF_HIGH),
        *_even(_B, _ANCHOR),
        *_even(_B, _REF_LOW),
        *_even(_B, _REF_HIGH),
    ]
    return records


def _with_external(records: Sequence[GameRecord]) -> list[GameRecord]:
    """Add both tie-ins and both site legs, as §D2.2(b)4-5 schedules them."""
    return [
        *records,
        *_even(_A, _A400),
        *_even(_B, _B400),
        *_link(
            _A400,
            _EXTERNAL,
            pairs=_EXTERNAL_PAIRS,
            wins=13,
            losses=12,
            book="site-start-advanced",
        ),
        *_link(
            _B400,
            _EXTERNAL,
            pairs=_EXTERNAL_PAIRS,
            wins=13,
            losses=12,
            book="site-start-advanced",
        ),
    ]


def _evaluate(
    records: Sequence[GameRecord],
    *,
    window: tuple[str, str] | None = _WINDOW,
    replicates: int = 50,
    kind: WindowKind = WindowKind.SINGLE,
) -> divergence.DivergenceResult:
    """Evaluate the single-epoch gate over `records` at a cheap replicate count."""
    return evaluate_window(
        MatchSet.of(records),
        roster_map=_ROSTER_MAP,
        window=window,
        kind=kind,
        bootstrap_seed=_SEED,
        replicates=replicates,
    )


_HEALTHY: dict[WindowKind, divergence.DivergenceResult] = {}


def _healthy_result(
    kind: WindowKind = WindowKind.SINGLE,
) -> divergence.DivergenceResult:
    """Return the evaluated healthy control, computed once per window kind.

    Cached only to keep the suite quick -- the records are immutable and the
    bootstrap is seeded, so the cache cannot be what makes any assertion true.
    """
    cached = _HEALTHY.get(kind)
    if cached is None:
        cached = _evaluate(_healthy_pool(), kind=kind)
        _HEALTHY[kind] = cached
    return cached


# --------------------------------------------------------------------------
# §6 node 24
# --------------------------------------------------------------------------
def test_the_driver_never_fits_a_merged_record_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§D5.1: three fits over three sets, and the third is a **refit**."""
    # Red against a driver that fits the union of the two game sets and
    # subtracts (the reconciliation §D4.2 rejects: two fits over disjoint games
    # are two estimates of two different things, and the offset between them is
    # not a constant), and against one that computes D_no_external by adjusting
    # D instead of refitting -- which is exactly the propagation R3 says the
    # External edges cause and cannot be undone by arithmetic.
    seen: list[MatchSet] = []

    def spy(match_set: MatchSet, anchor: str, *args: object, **kwargs: object) -> EloResult:
        seen.append(match_set)
        return fit_elo(match_set, anchor=anchor)

    monkeypatch.setattr(divergence, "fit_elo", spy)
    records = _with_external(_healthy_pool())
    gains = window_gains(partition_records(MatchSet.of(records), _ROSTER_MAP), _WINDOW)

    assert len(seen) == 3
    in_family, cross, no_external = seen

    def komugi_rated(record: GameRecord) -> bool:
        return all(
            _ROSTER_MAP[player] == "komugi" and "#" not in player
            for player in record.seats
        )

    assert all(komugi_rated(record) for record in in_family.records)
    assert not any(komugi_rated(record) for record in cross.records)
    assert _EXTERNAL in cross.players()
    assert _EXTERNAL not in no_external.players()
    dropped = [
        record
        for record in cross.records
        if _EXTERNAL in record.seats
    ]
    assert set(no_external.records) == set(cross.records) - set(dropped)
    assert len(dropped) == 4 * _EXTERNAL_PAIRS
    assert gains.d is not None and gains.d_no_external is not None


# --------------------------------------------------------------------------
# §6 node 25
# --------------------------------------------------------------------------
def _planted_pool(*, g_in: float, g_cross: float, seed: int, pairs: int) -> MatchSet:
    """Return a sampled pool where ``D = g_in - g_cross`` is planted.

    The in-family link and the cross links are sampled from **two** planted
    rating tables that agree on ``A`` and differ on ``B``: that is what a
    divergence *is*, and it cannot be planted with a single table.
    """
    in_family = simulate_match_set(
        {_A: 0.0, _B: g_in},
        seed=seed,
        draw_param=0.9,
        pairs_per_matchup=2 * pairs,
        schedule=((_A, _B),),
        book_ids=(_BOOK,),
    )
    cross = simulate_match_set(
        {
            _A: 0.0,
            _B: g_cross,
            _ANCHOR: -60.0,
            _REF_LOW: -200.0,
            _REF_HIGH: 150.0,
        },
        seed=seed + 5000,
        draw_param=0.9,
        pairs_per_matchup=pairs,
        schedule=(
            (_A, _ANCHOR),
            (_A, _REF_LOW),
            (_A, _REF_HIGH),
            (_B, _ANCHOR),
            (_B, _REF_LOW),
            (_B, _REF_HIGH),
        ),
        book_ids=(_BOOK,),
    )
    return MatchSet.of([*in_family.records, *cross.records])


def _expected_count_pool(*, g_in: float, g_cross: float) -> MatchSet:
    """Return a pool planted at the Davidson expectation of two rating tables."""
    cross_ratings = {
        _A: 0.0,
        _B: g_cross,
        _ANCHOR: -60.0,
        _REF_LOW: -200.0,
        _REF_HIGH: 150.0,
    }
    records = list(
        _davidson_link(_A, _B, delta=-g_in, pairs=_IN_FAMILY_PAIRS)
    )
    for first, second in (
        (_A, _ANCHOR),
        (_A, _REF_LOW),
        (_A, _REF_HIGH),
        (_B, _ANCHOR),
        (_B, _REF_LOW),
        (_B, _REF_HIGH),
    ):
        records.extend(
            _davidson_link(
                first,
                second,
                delta=cross_ratings[first] - cross_ratings[second],
                pairs=_CROSS_PAIRS,
            )
        )
    return MatchSet.of(records)


def test_a_planted_in_family_inflation_fires_the_gate_with_the_right_sign() -> None:
    """§D5.3: a 120-vs-20 divergence fires, positive, with an interval off 0."""
    # Red against a gate that never fires (a threshold read in log-strength
    # units, a sign error that cancels D, an interval that always straddles 0),
    # and -- through the equal-gain half -- against one that always fires.
    # D > 0 is in-family inflation: the lineage says it improved more than the
    # outside world agrees with, which is v1's failure verbatim.
    fired = evaluate_window(
        _expected_count_pool(g_in=120.0, g_cross=20.0),
        roster_map=_ROSTER_MAP,
        window=_WINDOW,
        kind=WindowKind.SINGLE,
        bootstrap_seed=_SEED,
        replicates=100,
    )
    assert fired.verdict is Verdict.FIRED, fired
    assert fired.d_no_external is not None and fired.d_no_external > 0.0
    assert fired.sign == 1
    assert fired.lo_no_external is not None and fired.hi_no_external is not None
    assert fired.lo_no_external > 0.0 or fired.hi_no_external < 0.0
    assert fired.threshold == THRESHOLD_SINGLE_ELO
    assert fired.gate_statistic == "d_no_external"

    quiet = evaluate_window(
        _expected_count_pool(g_in=60.0, g_cross=60.0),
        roster_map=_ROSTER_MAP,
        window=_WINDOW,
        kind=WindowKind.SINGLE,
        bootstrap_seed=_SEED,
        replicates=100,
    )
    assert quiet.verdict is Verdict.OK, quiet
    assert quiet.d_no_external is not None
    assert abs(quiet.d_no_external) < THRESHOLD_SINGLE_ELO


# --------------------------------------------------------------------------
# §6 node 26
# --------------------------------------------------------------------------
def _case_converged() -> list[GameRecord]:
    """B sweeps the in-family link: the likelihood is unbounded (elo.py:70-77)."""
    records = [
        record
        for record in _healthy_pool()
        if _A not in record.seats or _B not in record.seats
    ]
    return [
        *records,
        *_link(_B, _A, pairs=_IN_FAMILY_PAIRS, wins=_IN_FAMILY_PAIRS, losses=0),
    ]


def _case_components() -> list[GameRecord]:
    """A and B hang on different reference entrants: two cross components."""
    return [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _REF_LOW),
        *_even(_A, _SPARE),
        *_even(_B, _REF_HIGH),
        *_even(_B, _ANCHOR),
    ]


def _case_book_era() -> list[GameRecord]:
    """A's endpoint links were played on era 1, B's on era 2 (§D3.3)."""
    return [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS, book="E002-aaaaaaa1"),
        *_even(_A, _ANCHOR, book="E001-9ab31f07"),
        *_even(_A, _REF_LOW, book="E001-9ab31f07"),
        *_even(_A, _REF_HIGH, book="E001-9ab31f07"),
        *_even(_B, _ANCHOR, book="E002-aaaaaaa1"),
        *_even(_B, _REF_LOW, book="E002-aaaaaaa1"),
        *_even(_B, _REF_HIGH, book="E002-aaaaaaa1"),
    ]


def _case_games_and_decisive() -> list[GameRecord]:
    """One of B's GPU-path links is a 50-game leg -- below §D2.1's floor."""
    records = [
        record
        for record in _healthy_pool()
        if not (_B in record.seats and _REF_LOW in record.seats)
    ]
    return [*records, *_even(_B, _REF_LOW, pairs=_EXTERNAL_PAIRS)]


def _case_games_and_decisive_draws() -> list[GameRecord]:
    """One of B's GPU-path links is 128 games with only **38** decisive (F6).

    The games clause is satisfied exactly -- 64 pairs is §D2.1's 128 -- so only
    the decisive clause can fire, which is what
    :func:`_case_games_and_decisive` (a 50-game leg) cannot show: that fixture
    trips ``games < MIN_GPU_PATH_GAMES`` first, so ``MIN_DECISIVE = 40 -> 0``
    survives it. 9 wins + 10 losses is ``2 * 19 = 38`` decisive games, two short
    of the floor, with the remaining 45 pairs drawn -- the saturated-draw window
    §D5.4's ``D < 0`` branch is written for, where ``fit_elo`` *"has fit an Elo
    gap out of nothing"* (``arena.py:367-380``).
    """
    records = [
        record
        for record in _healthy_pool()
        if not (_B in record.seats and _REF_LOW in record.seats)
    ]
    return [*records, *_link(_B, _REF_LOW, pairs=_CROSS_PAIRS, wins=9, losses=10)]


def _case_discard_rate() -> list[GameRecord]:
    """Two sweeping pendant links, so 26% of replicates clamp and are dropped.

    The discard mechanism is **not** "a resample drove the link toward a
    sweep": ``elo.py``'s MM iteration does not run a sweeping rating out to
    ``_MAX_LOG_GAMMA`` -- at ``score / games == 1`` the strength step stalls and
    ``delta < tol`` declares convergence. A replicate is discarded exactly when
    the pendant's *resampled* score rate is 1 across all of its links, which
    makes ``_initial_theta`` clip at the cap on entry and the post-loop
    saturation check report ``converged=False``.

    So the fixture plants the sweep itself: ``_SPARE`` is 32-0 against both
    ``_REF_LOW`` and ``_REF_HIGH``, which measures at 0.26 discards over
    ``B = 100`` -- comfortably over condition 5's 10% while the *observed* fit
    still converges, so the node cannot pass by way of condition 1. ``_SPARE``
    is a pendant, not a window endpoint, so condition 6's ``[0.15, 0.85]``
    score-rate floor is not read over these links and cannot fire first.
    """
    return [
        *_healthy_pool(),
        *_link(_SPARE, _REF_LOW, pairs=32, wins=32, losses=0),
        *_link(_SPARE, _REF_HIGH, pairs=32, wins=32, losses=0),
    ]


def _case_links_saturated() -> list[GameRecord]:
    """Every one of B's cross links is a blowout: condition 6 (rev 3, R4)."""
    return [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _ANCHOR),
        *_even(_A, _REF_LOW),
        *_even(_A, _REF_HIGH),
        *_link(_B, _ANCHOR, pairs=_CROSS_PAIRS, wins=62, losses=2),
        *_link(_B, _REF_LOW, pairs=_CROSS_PAIRS, wins=62, losses=2),
        *_link(_B, _REF_HIGH, pairs=_CROSS_PAIRS, wins=62, losses=2),
    ]


_CONDITION_CASES: dict[str, tuple[Callable[[], list[GameRecord]], str]] = {
    "converged": (_case_converged, CONDITION_CONVERGED),
    "components": (_case_components, CONDITION_COMPONENTS),
    "book_era": (_case_book_era, CONDITION_BOOK_ERA),
    "games_and_decisive": (_case_games_and_decisive, CONDITION_GAMES_AND_DECISIVE),
    # Condition 4 is a conjunction, and its two halves need two cases: the row
    # above trips the *games* clause, this one trips only the decisive floor.
    "games_and_decisive_draws": (
        _case_games_and_decisive_draws,
        CONDITION_GAMES_AND_DECISIVE,
    ),
    "discard_rate": (_case_discard_rate, CONDITION_DISCARD_RATE),
    "links_saturated": (_case_links_saturated, CONDITION_LINKS_SATURATED),
    # The seventh case is not one of the six conditions: at E = 1 there is no
    # predecessor, so the window is not formed at all (§D5.1, R7).
    "window_not_formed": (_healthy_pool, CONDITION_WINDOW_NOT_FORMED),
}


@pytest.mark.parametrize("case", sorted(_CONDITION_CASES))
def test_each_validity_condition_yields_not_evaluable_and_names_itself(
    case: str,
) -> None:
    """§D5.3: six conditions, each naming itself, plus the ``E = 1`` state."""
    # Red against an implementation that returns `ok` when it cannot see --
    # the single most dangerous failure this gate can have, because a silent
    # pass is indistinguishable in the report from a measured absence of
    # divergence. Each fixture fails exactly one condition, so a node that
    # passes for the wrong reason names the wrong condition and fails here.
    build, expected = _CONDITION_CASES[case]
    window = None if case == "window_not_formed" else _WINDOW
    result = _evaluate(build(), window=window, replicates=100)
    assert result.verdict is Verdict.NOT_EVALUABLE, result
    assert result.condition == expected
    assert result.message == f"not evaluable: {expected}"

    if case == "discard_rate":
        # F2: the case must reach condition 5 with a rate the condition would
        # NOT have tolerated. Without this the case could pass while carrying a
        # rate under MAX_DISCARD_RATE -- i.e. named by some other reading of the
        # same row -- and condition 5, the tolerance §D5.2's pre-registered
        # discard bias is bought with, would still have no demonstration. The
        # upper bound is the other half: at a rate of 1.0 there is no interval
        # at all and `pair_bootstrap` raises instead (interval.py:324-329).
        assert result.discard_rate is not None
        assert 0.10 < result.discard_rate < 1.0, result.discard_rate

    # The control: the same shape, healthy, is evaluable on the same code path.
    assert _healthy_result().verdict is Verdict.OK, _healthy_result()


def test_the_first_epoch_window_is_not_yet_formed() -> None:
    """§D5.1 at ``E = 1``: there is no predecessor, so there is no window (R7)."""
    # Red against an implementation that raises at E = 1 (the first epoch is
    # still played and still reported), and against one that reports `ok`.
    result = _evaluate(_healthy_pool(), window=None)
    assert result.verdict is Verdict.NOT_EVALUABLE
    assert result.condition == CONDITION_WINDOW_NOT_FORMED
    assert result.message == "not evaluable: window not yet formed"
    assert result.d is None and result.d_no_external is None


# --------------------------------------------------------------------------
# §6 node 27
# --------------------------------------------------------------------------
def test_neither_threshold_is_a_parameter() -> None:
    """§D5.3: the pre-registration has no override, in the p4_branch idiom."""
    # Red against a helpful `threshold=` keyword (or a `**kwargs` that swallows
    # one), which is precisely how a pre-registration gets quietly retuned
    # after its first failure -- the move r5.py:429-436 names as selection on
    # the gate.
    forbidden = ("threshold", "tolerance", "critical", "cutoff", "gate_elo")
    checked = 0
    for name, value in vars(divergence).items():
        if name.startswith("_") or not inspect.isfunction(value):
            continue
        if value.__module__ != divergence.__name__:
            continue
        checked += 1
        signature = inspect.signature(value)
        for parameter in signature.parameters.values():
            lowered = parameter.name.lower()
            assert not any(word in lowered for word in forbidden), (name, lowered)
            assert parameter.kind is not inspect.Parameter.VAR_KEYWORD, name
    assert checked >= 5

    assert THRESHOLD_SINGLE_ELO == 75.0
    assert THRESHOLD_SUSTAINED_ELO == 90.0
    assert threshold_for(WindowKind.SINGLE) == 75.0
    assert threshold_for(WindowKind.SUSTAINED) == 90.0
    assert list(inspect.signature(threshold_for).parameters) == ["kind"]
    assert list(inspect.signature(evaluate_window).parameters) == [
        "match_set",
        "roster_map",
        "window",
        "kind",
        "bootstrap_seed",
        "replicates",
    ]

    # And the value that reaches the record is the pre-registered one.
    for kind, expected in (
        (WindowKind.SINGLE, THRESHOLD_SINGLE_ELO),
        (WindowKind.SUSTAINED, THRESHOLD_SUSTAINED_ELO),
    ):
        assert _healthy_result(kind).threshold == expected


# --------------------------------------------------------------------------
# §6 node 40
# --------------------------------------------------------------------------
_SMOKE_TRIALS: int = 10
"""Pinned by §6 node 40 (rev 3's re-scoping, R2). A smoke, not the power run."""

_SMOKE_REPLICATES: int = 100
"""Pinned by §6 node 40. The published bootstrap runs at ``B = 2000``."""


def _firings(true_d: float) -> int:
    """Return how many of the pinned trials fire at a planted ``|D|``."""
    fired = 0
    for trial in range(_SMOKE_TRIALS):
        result = evaluate_window(
            _planted_pool(
                g_in=60.0 + true_d,
                g_cross=60.0,
                seed=_SEED + 97 * trial,
                pairs=_CROSS_PAIRS,
            ),
            roster_map=_ROSTER_MAP,
            window=_WINDOW,
            kind=WindowKind.SINGLE,
            bootstrap_seed=_SEED + trial,
            replicates=_SMOKE_REPLICATES,
        )
        assert result.verdict is not Verdict.NOT_EVALUABLE, result
        if result.verdict is Verdict.FIRED:
            fired += 1
    return fired


def test_the_gate_cannot_fire_below_its_own_resolution() -> None:
    """§D5.3's power table, as an in-suite smoke over the real game counts."""
    # Red against a gate whose threshold is not in Elo at all (it would fire on
    # every trial, including at a true |D| = 0), and against one whose interval
    # condition is missing or inverted. It is NOT a measurement of the power
    # table: 10 trials cannot resolve 0.55 from 0.78, and the >=100-trial run at
    # B = 2000 over the real §D2.2 schedule lives in
    # gungi-az-runs/scripts/pool_coverage.py with its output committed. This
    # node is what stops the machinery rotting between those runs.
    assert _SMOKE_TRIALS == 10
    assert _SMOKE_REPLICATES == 100

    # The pre-registered false-alarm side: at a true |D| = 0 the gate needs
    # |D-hat| >= 75 at sigma ~ 33, i.e. 2.3 sigma, so 0 or 1 of 10 is the
    # binomial interval this many trials can carry.
    assert _firings(0.0) <= 1

    # The table says 0.55-0.78 at |D| = 100; >= 5 of 10 is that interval's
    # lower side, and the node asserts only what 10 trials can carry.
    assert _firings(100.0) >= 5


# --------------------------------------------------------------------------
# §6 node 41
# --------------------------------------------------------------------------
def test_g_in_is_fit_over_the_declared_record_set() -> None:
    """§D5.1: ``G_in`` is the no-``#`` komugi x komugi links, and nothing else."""
    # Red against an implementation that reads "in-family" as "both players are
    # komugi" -- which sweeps in the #anchor400 tie-in (a *configuration*
    # offset) and §D8's visits legs (a search-budget sweep), contaminating a
    # training gain with two quantities that are not training. Red also against
    # one that accepts run_ladder's records, whose ids are bare `ckpt-<step>`
    # (arena.py:811-818) and name different players entirely.
    records = [
        *_with_external(_healthy_pool()),
        *_even(_VISITS, _B, pairs=16),
        *_even(_VISITS, _ANCHOR, pairs=16),
    ]
    sets = partition_records(MatchSet.of(records), _ROSTER_MAP)

    assert set(sets.in_family.players()) == {_A, _B}
    assert all(
        record.seats in ((_A, _B), (_B, _A)) for record in sets.in_family.records
    )
    assert len(sets.in_family) == 2 * _IN_FAMILY_PAIRS

    # The tie-in is komugi x komugi by family and lands in G_cross; the visits
    # legs land in neither set.
    assert _A400 in sets.cross.players()
    assert _VISITS not in sets.cross.players()
    assert _VISITS not in sets.cross_no_external.players()
    assert _VISITS in sets.visits.players()
    assert len(sets.visits) == 2 * 32

    # A run_ladder record cannot even be classified: family is read from the
    # roster map, never parsed from a prefix, and the map does not carry it.
    ladder_record = GameRecord(
        white="ckpt-3000",
        black="ckpt-6000",
        result=DRAW,
        book_id="empty-advanced",
        pair_id="ckpt-3000:ckpt-6000:00000",
    )
    with pytest.raises(ValueError, match="not in the roster map"):
        partition_records(MatchSet.of([ladder_record]), _ROSTER_MAP)


# --------------------------------------------------------------------------
# §6 node 44
# --------------------------------------------------------------------------
def test_a_external_matchup_does_not_make_every_window_unevaluable() -> None:
    """§D5.3 condition 4's exemption is narrow, and it is the site leg's (F14)."""
    # Red against reading condition 4 over every matchup: the External leg is a
    # >=50-game leg on the site's clock (§D2.2(b)4's declared deviation) and is
    # the only matchup in the pool that cannot be bought more games by spending
    # GPU, so a literal reading marks *every* window not evaluable, permanently
    # and silently. Red also against an exemption written over game count
    # rather than over the path -- a 50-game GPU matchup must still fail.
    with_site_leg = _evaluate(_with_external(_healthy_pool()))
    assert with_site_leg.verdict is not Verdict.NOT_EVALUABLE, with_site_leg
    assert with_site_leg.g_cross is not None
    assert with_site_leg.g_cross_no_external is not None

    same_but_gpu = _evaluate(
        [*_with_external(_healthy_pool()), *_even(_B, _SPARE, pairs=_EXTERNAL_PAIRS)]
    )
    assert same_but_gpu.verdict is Verdict.NOT_EVALUABLE
    assert same_but_gpu.condition == CONDITION_GAMES_AND_DECISIVE


# --------------------------------------------------------------------------
# §6 node 55
# --------------------------------------------------------------------------
def test_a_saturated_bracket_makes_the_window_not_evaluable() -> None:
    """§D5.3 condition 6: "the gate can no longer see" is a named state (R4)."""
    # Red against conditions 1-5 alone, which cannot see a saturated
    # instrument: a blowout satisfies condition 4's decisive floor trivially
    # (every game is decisive), the fit still converges, and the graph is still
    # connected -- so the gate would report `ok` from a window whose sigma has
    # tripled and whose pre-registered power table no longer applies.
    saturated = _evaluate(_case_links_saturated())
    assert saturated.verdict is Verdict.NOT_EVALUABLE
    assert saturated.condition == CONDITION_LINKS_SATURATED
    assert saturated.bracket_saturated == (_B,)

    # (a) The same decisive counts, in band: evaluable. Every link in
    # _healthy_pool() is 128 decisive games, exactly as the blowouts are, so
    # this node cannot pass by way of condition 4.
    control = _healthy_result()
    assert control.verdict is not Verdict.NOT_EVALUABLE

    # (b) The floor sits between one informative link and two.
    two_informative = [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _ANCHOR),
        *_even(_A, _REF_LOW),
        *_even(_A, _REF_HIGH),
        *_even(_B, _ANCHOR),
        *_even(_B, _REF_LOW),
        *_link(_B, _REF_HIGH, pairs=_CROSS_PAIRS, wins=62, losses=2),
    ]
    assert _evaluate(two_informative).verdict is not Verdict.NOT_EVALUABLE

    one_informative = [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _ANCHOR),
        *_even(_A, _REF_LOW),
        *_even(_A, _REF_HIGH),
        *_even(_B, _ANCHOR),
        *_link(_B, _REF_LOW, pairs=_CROSS_PAIRS, wins=62, losses=2),
        *_link(_B, _REF_HIGH, pairs=_CROSS_PAIRS, wins=62, losses=2),
    ]
    thin = _evaluate(one_informative)
    assert thin.verdict is Verdict.NOT_EVALUABLE
    assert thin.condition == CONDITION_LINKS_SATURATED

    # (c) F8: the band's own endpoints, which (a) and (b) cannot see -- every
    # fixture above is at a score rate of exactly 0.5 or >= 0.95, so
    # `(0.15, 0.85) -> (0.45, 0.55)` and any widening short of 0.97 survive
    # them both. The two links below differ by ONE pair swapped from a loss to
    # a win: 108/128 = 0.84375 is inside the band and 110/128 = 0.859375 is
    # outside it, and both carry 120 decisive games, so condition 4 is
    # identical on the two windows and only the band can separate them.
    def _boundary_window(wins: int, losses: int) -> list[GameRecord]:
        return [
            *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
            *_even(_A, _ANCHOR),
            *_even(_A, _REF_LOW),
            *_even(_A, _REF_HIGH),
            *_even(_B, _ANCHOR),
            *_link(_B, _REF_LOW, pairs=_CROSS_PAIRS, wins=wins, losses=losses),
            *_link(_B, _REF_HIGH, pairs=_CROSS_PAIRS, wins=62, losses=2),
        ]

    inside = _evaluate(_boundary_window(52, 8))
    assert inside.verdict is not Verdict.NOT_EVALUABLE, inside

    outside = _evaluate(_boundary_window(53, 7))
    assert outside.verdict is Verdict.NOT_EVALUABLE, outside
    assert outside.condition == CONDITION_LINKS_SATURATED
    assert outside.bracket_saturated == (_B,)

    # And the pre-registration itself, as a literal, in node 58's idiom: the
    # band is derived from `sqrt(0.25/0.1275) = 1.40` (§D5.3 condition 6), so a
    # later widening to [0.05, 0.95] is a change to the instrument's certified
    # sigma and must move a test rather than only a constant.
    assert INFORMATIVE_BAND == (0.15, 0.85)


def test_the_two_windows_carry_their_own_pre_registered_thresholds() -> None:
    """§D5.3: 75 on ``(E-1, E)`` and 90 on ``(E-3, E)``, both every epoch."""
    # Red against a single threshold used for both windows, which would make
    # the sustained window -- the one that recovers the small effect and costs
    # no games -- fire at the single-epoch rate.
    single = _healthy_result(WindowKind.SINGLE)
    sustained = _healthy_result(WindowKind.SUSTAINED)
    assert single.threshold == 75.0
    assert sustained.threshold == 90.0
    assert single.window_kind is WindowKind.SINGLE
    assert sustained.window_kind is WindowKind.SUSTAINED


def test_the_partition_never_lets_an_external_record_into_the_gate_statistic() -> None:
    """§D5.1/R3: the gate reads the without-External fit; ``D`` sits beside it."""
    # Red against a driver that publishes both numbers and tests the
    # with-External one -- the failure R3 raises: the external anchor updates
    # silently between epochs (U4 says nothing here would notice), D moves by
    # the anchor's drift, and §D5.4 declares the primary series broken on the
    # strength of a leg whose rules are unverified.
    records = _with_external(_healthy_pool())
    sets = partition_records(MatchSet.of(records), _ROSTER_MAP)
    assert _EXTERNAL in sets.cross.players()
    assert _EXTERNAL not in sets.cross_no_external.players()
    result = _evaluate(records)
    assert result.gate_statistic == "d_no_external"
    assert result.d is not None and result.lo is not None
    assert result.d_no_external is not None and result.lo_no_external is not None


def test_the_tie_in_is_not_an_informative_link() -> None:
    """§D5.3 condition 6: the ``#anchor400`` tie-in is not one of the two (F5)."""
    # Red against counting every in-band matchup in G_cross_no_external: the
    # §D2.2(b)5 tie-in is komugi x komugi with a `#` suffix, so it lands in
    # `cross` by construction (divergence.py:286-291), it is gpu_path (no
    # External side), and it is a SAME-WEIGHTS link that §D2.2(b)5(i) argues is
    # "as near in strength as any pair in the pool" -- permanently ~0.5 and
    # permanently in band, whatever the reference field has done. Counting it
    # leaves the pre-registered floor of two external links standing at one:
    # mid-campaign the milestone is >= 0.97 against the anchor and both bracket
    # links and 0.5 against its own #anchor400 row, condition 6 counts
    # 1 (tie-in) + 1 (a surviving bracket link) = 2, the window is declared
    # evaluable, and §D5.3's sigma ~ 33-48 and its whole power table are quoted
    # for an instrument whose real sigma is ~89.
    #
    # (a) The discriminating half: B retains exactly ONE external informative
    # link (the anchor) and the tie-in would be its second. Under the unfixed
    # rule this window is evaluable and every assertion below fails.
    one_external_plus_the_tie_in = [
        *_even(_A, _B, pairs=_IN_FAMILY_PAIRS),
        *_even(_A, _ANCHOR),
        *_even(_A, _REF_LOW),
        *_even(_A, _REF_HIGH),
        *_even(_B, _ANCHOR),
        *_link(_B, _REF_LOW, pairs=_CROSS_PAIRS, wins=62, losses=2),
        *_link(_B, _REF_HIGH, pairs=_CROSS_PAIRS, wins=62, losses=2),
        *_even(_B, _B400),
    ]
    result = _evaluate(one_external_plus_the_tie_in)
    assert result.verdict is Verdict.NOT_EVALUABLE, result
    assert result.condition == CONDITION_LINKS_SATURATED
    assert result.bracket_saturated == (_B,)

    # (b) And a tie-in cannot rescue a window that has lost every external link
    # either -- the review's own fixture, kept because it is the state §D5.3
    # condition 6 is written about and it must not become evaluable by growing
    # a configuration link.
    saturated = _evaluate([*_case_links_saturated(), *_even(_B, _B400)])
    assert saturated.verdict is Verdict.NOT_EVALUABLE, saturated
    assert saturated.condition == CONDITION_LINKS_SATURATED
    assert saturated.bracket_saturated == (_B,)


def test_one_era_of_distinct_entry_ids_is_one_era() -> None:
    """§D5.3 condition 3: 64 distinct ``E007-<sha8>`` entries are **one** era (F7)."""
    # Red against `book_era` returning its argument. Every other fixture in
    # this file carries one book_id per link, so the identity map still yields
    # two eras on the straddle case and one on the healthy pool -- the mutant
    # is invisible to node 26. Under §D3.2's real records the book_id is the
    # per-pair ENTRY id (64 distinct ones inside a single 128-game matchup), so
    # the identity map makes every window from the first real epoch report
    # "not evaluable: book era boundary", permanently and for a reason that
    # reads like a finding: T2b lands real books, every gate window goes not
    # evaluable, and the campaign reads it as an era-boundary artefact (§D3.3's
    # bounded loss) for three epochs and never gets a gate at all.
    one_era = _evaluate(_per_pair_entry_ids(_healthy_pool(), "E007"), replicates=100)
    assert one_era.verdict is not Verdict.NOT_EVALUABLE, one_era
    assert one_era.d_no_external is not None

    # The negative half, on the same per-pair shape: one endpoint link really
    # is on the next era, and the era comparison still sees it.
    straddle = [
        *_per_pair_entry_ids(
            [
                record
                for record in _healthy_pool()
                if not (_B in record.seats and _REF_LOW in record.seats)
            ],
            "E007",
        ),
        *_per_pair_entry_ids(_even(_B, _REF_LOW), "E008"),
    ]
    crossed = _evaluate(straddle, replicates=100)
    assert crossed.verdict is Verdict.NOT_EVALUABLE, crossed
    assert crossed.condition == CONDITION_BOOK_ERA


def test_the_verdict_follows_the_without_external_statistic() -> None:
    """§D5.1/R3: the gate's predicate is ``D_no_external``, and the row says so."""
    # Red against a driver that reads `gains.d` and
    # `bootstrap.intervals[PUBLISHED_STATISTIC]` -- the advisor's pre-seeded
    # mutant, which survives every other node in this group: node 24 counts the
    # three fits and checks the record SETS, node 25's fixtures contain no
    # External records at all (so D == D_no_external there), and
    # test_the_partition_never_lets_an_external_record_into_the_gate_statistic
    # asserts only that four fields are non-None and that `gate_statistic` is
    # its dataclass default (divergence.py:443) -- a label, not a claim about
    # what was computed. `external_drift` becomes constantly False under the
    # mutant and no other node reads it.
    #
    # The planted failure is U4's: the external anchor updates silently between
    # E-1 and E, the two external/site-<date> vertices are different players, the
    # anchor pendant moves r_cross(B) relative to r_cross(A), D crosses under
    # T = 75 while D_no_external does not -- and the epoch must still report
    # FIRED, from the statistic whose rules are verified. The in-family gain is
    # 150 and not 120 because the gated interval at B = 200 is ~+-110 wide on
    # this fixture, so the planted D_no_external has to clear T by more than that
    # margin: measured, g_in=120 gives d=36.9 [-88.4, 148.6] and
    # d_no_external=107.5 [-9.5, 236.0] -- both straddle 0, neither arm fires --
    # while g_in=150 gives d=66.5 [-47.2, 176.4] (published, under T, does not
    # fire) and d_no_external=137.0 [26.8, 258.0] (gated, fires).
    records = [
        *_expected_count_pool(g_in=150.0, g_cross=20.0).records,  # D_no_external = 130
        *_even(_A, _A400),
        *_even(_B, _B400),  # §D2.2(b)5 tie-ins
        *_link(
            _A400,
            _EXTERNAL,
            pairs=_DRIFT_PAIRS,
            wins=_DRIFT_WINS,
            losses=_DRIFT_LOSSES,
            book="site-start-advanced",
        ),
        *_link(
            _B400,
            _EXTERNAL,
            pairs=_DRIFT_PAIRS,
            wins=_DRIFT_LOSSES,
            losses=_DRIFT_WINS,
            book="site-start-advanced",
        ),
    ]
    result = _evaluate(records, replicates=200)

    # The fixture's precondition, asserted so it cannot drift into vacuity. If
    # the two statistics ever agree here this fails loudly rather than letting
    # the node pass for free; the site legs' split and length are the tuning
    # knobs (_DRIFT_PAIRS, _DRIFT_WINS).
    assert result.d is not None and result.d_no_external is not None
    gated_fires = abs(result.d_no_external) >= result.threshold and (
        result.lo_no_external is not None
        and result.hi_no_external is not None
        and (result.lo_no_external > 0.0 or result.hi_no_external < 0.0)
    )
    published_fires = abs(result.d) >= result.threshold and (
        result.lo is not None
        and result.hi is not None
        and (result.lo > 0.0 or result.hi < 0.0)
    )
    assert gated_fires != published_fires, (result.d, result.d_no_external)

    # The node itself: the verdict is the WITHOUT-External predicate, and the
    # disagreement is reported rather than silently resolved.
    assert (result.verdict is Verdict.FIRED) == gated_fires
    assert result.external_drift is True
    assert result.gate_statistic == "d_no_external"


