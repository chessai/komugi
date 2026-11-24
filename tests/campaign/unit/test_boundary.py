"""The boundary's escalation: who records first, who pauses first, who is muted.

KOM-022 §3b/§4/§5, acceptances #23, #32, #33 and #36. The escalation is
deliberate and asymmetric, and both halves have a cost that was paid on
purpose:

* the **hash gate**, the **cross-role invariant**, the **EMA-aliasing check**,
  the **fingerprint-series completeness** invariant and a **non-finite loss**
  pause on the *first* occurrence and can never be muted. A frozen self-play
  net is not a false alarm, and ``KL == 0`` on every probe position is
  assumption-free;
* the **band**, the **flat-trainer** verdict and the **gauntlet floor** record
  first and pause on a later consecutive occurrence, because §10's own
  refutation criteria count *recorded violations* and a gate that stopped the
  campaign at the first could only ever produce the second through an operator
  who had already judged the first a false alarm;
* the **loss spike** pauses on the first occurrence -- §5 bound 1 -- because a
  factor of three on a robust tail median is a broken run, not a bad iteration.
  It is the one escalating tripwire that does not wait, and it is the one whose
  mute (two iterations, enough for the baseline to re-establish) was rev 1's
  whole rationale for muting at all.

The escalation counts are read back from the **journal**, not from memory, so
"two consecutive band exits" means the same thing to a campaign that was
restarted between them as to one that was not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from komugi.campaign import boundary
from komugi.campaign.pause import CampaignPaused, Mute, read_pause
from komugi.campaign.pinned import (
    BAND_PATIENCE,
    DENOMINATOR_FLOOR,
    DISPLACEMENT_BAND,
    FLAT_TRAINER,
    FLAT_TRAINER_PATIENCE,
    GAUNTLET_BREACH_PATIENCE,
    GAUNTLET_FLOOR,
    GAUNTLET_FLOOR_TRIPWIRE,
    HASH_GATE,
    LOSS_SPIKE,
    NON_FINITE_LOSS,
    SPIKE_BASELINE_ITERATIONS,
    SPIKE_FACTOR,
    UNMUTABLE,
    WARMUP_ITERATIONS,
    band,
)
from komugi.config import SelfPlayConfig
from komugi.telemetry.journal import write_record
from komugi.telemetry.liveness import self_play_net_advanced
from komugi.telemetry.records import (
    EMA,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)

TAIL_KEY = "total"
"""The trainer's own ``TOTAL_KEY``, restated so this module does not import the
trainer to read one string."""


def _record(iteration: int, **extras: float) -> IterationMetrics:
    """A campaign-written record carrying `extras` and the discriminating marker."""
    payload = {boundary.CAMPAIGN_MARKER: 1.0}
    payload.update(extras)
    return IterationMetrics(
        iteration=iteration,
        games=1,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras=payload,
    )


def _trainer_record(iteration: int, loss_total: float) -> IterationMetrics:
    """The record ``training_iteration`` writes: no marker, a ``loss_`` prefix."""
    return IterationMetrics(
        iteration=iteration,
        games=0,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras={"loss_total": loss_total},
    )


def _baseline(through: int, value: float = 10.0) -> list[IterationMetrics]:
    """`SPIKE_BASELINE_ITERATIONS` campaign records ending just below `through`."""
    return [
        _record(iteration, **{boundary.LOSS_TAIL_MEDIAN: value})
        for iteration in range(through - SPIKE_BASELINE_ITERATIONS, through)
    ]


def _gauntlet_records(
    iteration: int, *, breached: bool, score: float = 0.35
) -> list[IterationMetrics]:
    """The record a gauntlet iteration writes, as the loop writes it.

    A gauntlet that *played* records ``gauntlet_anchor_gen`` and (unless it was
    uninformative) ``gauntlet_score``; a breach adds ``gauntlet_breach``. The
    keys are what tells "this iteration ran a gauntlet and did not breach" from
    "this iteration ran no gauntlet", which is the distinction the floor's
    escalation is counted over.
    """
    extras = {
        boundary.GAUNTLET_ANCHOR_GEN: 0.0,
        boundary.GAUNTLET_SCORE: score if breached else GAUNTLET_FLOOR + 0.1,
    }
    if breached:
        extras[boundary.GAUNTLET_BREACH] = 1.0
    return [_record(iteration, **extras)]


def _run_band(
    exits: int, iteration: int, mutes: Mapping[str, Mute] | None = None
) -> boundary.BoundaryOutcome:
    """Fire the band gate at `iteration` with `exits` prior consecutive exits."""
    records = [
        _record(index, **{boundary.BAND_VIOLATION: 1.0})
        for index in range(max(iteration - exits, 0), iteration)
    ]
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_band(
        [band()[1] * 10.0], iteration, records, outcome, mutes or {}
    )
    return outcome


def _walk_band(ratios: Sequence[float]) -> int | None:
    """Run `ratios` through the band gate one iteration at a time, from warm-up.

    Each iteration's ``extras`` are written into the record list the next one
    reads, which is how the real loop counts its consecutive exits -- from the
    journal, not from memory. Returns the iteration at which the gate paused,
    or ``None`` if it never did.
    """
    records: list[IterationMetrics] = []
    for offset, ratio in enumerate(ratios):
        iteration = WARMUP_ITERATIONS + offset
        outcome = boundary.BoundaryOutcome()
        boundary.evaluate_band([ratio], iteration, records, outcome, {})
        records.append(_record(iteration, **outcome.extras))
        if outcome.tripwire is not None:
            assert outcome.tripwire == DISPLACEMENT_BAND
            return iteration
    return None


# ---------------------------------------------------------------------------
# the escalating gates: record first, pause on the Nth
# ---------------------------------------------------------------------------


def test_the_band_records_on_the_first_exit_and_pauses_on_the_second() -> None:
    """§3b/F7: escalated, not immediate, so Claim 2 stays falsifiable.

    The refutation criterion needs **two** band exits in the first twelve
    post-warm-up iterations, and rev 1's design stopped the campaign at the
    first and refused to restart without an operator having *already* judged it
    a false alarm. A falsification test conditional on a human pre-judging the
    thing it decides is not a test.
    """
    first = _run_band(0, WARMUP_ITERATIONS + 1)
    assert first.extras[boundary.BAND_VIOLATION] == 1.0
    assert first.tripwire is None, "the first exit stopped the campaign"
    assert [v for v in first.verdicts if not v.ok], "the exit was not recorded"

    second = _run_band(1, WARMUP_ITERATIONS + 2)
    assert second.tripwire == DISPLACEMENT_BAND
    assert f"{BAND_PATIENCE} consecutive" in second.detail


def test_the_band_does_not_run_before_the_warm_up() -> None:
    """§3b guard 1: until one EMA horizon, ``r`` is about the initial condition.

    ``EMA_0 = theta_0`` dominates before then, so a band verdict would be a
    statement about where the campaign started rather than about the recursion
    the band is derived from.
    """
    for iteration in range(WARMUP_ITERATIONS):
        outcome = _run_band(5, iteration)
        assert outcome.tripwire is None
        assert boundary.BAND_VIOLATION not in outcome.extras
        assert outcome.verdicts == []


def test_the_band_is_skipped_when_there_is_no_ratio_to_check() -> None:
    """A ratio the flat-trainer guard suppressed is not a band exit."""
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_band([], WARMUP_ITERATIONS + 3, [], outcome, {})

    assert outcome.tripwire is None
    assert outcome.verdicts == []


def test_a_ratio_inside_the_band_records_a_passing_verdict() -> None:
    """The false-alarm side: a healthy ratio must leave no flag behind."""
    low, high = band()
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_band(
        [(low + high) / 2.0], WARMUP_ITERATIONS + 1, [], outcome, {}
    )

    assert outcome.tripwire is None
    assert boundary.BAND_VIOLATION not in outcome.extras
    assert all(verdict.ok for verdict in outcome.verdicts)


def test_a_healthy_measured_ratio_never_pauses_under_the_repinned_band() -> None:
    """KOM-149's whole point: 0.5 is what a healthy campaign actually reports.

    Every root on disk settles to ``r ~ 0.53`` after its warm-up, and under the
    old derived band (floor 2.25) that was a floor exit at *every* one of those
    iterations -- so the campaign paused at 5, 14, 23 and ran on a standing
    operator ack (KOM-147), which is a tripwire carrying no information. Thirty
    consecutive post-warm-up iterations at 0.5 must now pass silently: no
    pause, and no ``band_violation`` flag left behind for a later real exit to
    count as its predecessor.
    """
    assert _walk_band([0.5] * 30) is None


def test_an_ema_that_freezes_mid_campaign_still_pauses_on_the_new_floor() -> None:
    """The floor kept its job at the lowered value: ``r -> 0`` still stops.

    A healthy series that drops to exactly zero at iteration 20 is a target net
    that stopped moving, or an EMA aliased onto the trainer's tensor. It
    records at 20 and pauses at 21, on :data:`BAND_PATIENCE` consecutive exits
    -- the same escalation as before, now reachable only by a series that is
    genuinely broken rather than by every healthy campaign.
    """
    frozen_at = 20
    healthy = [0.5] * (frozen_at - WARMUP_ITERATIONS)

    assert _walk_band(healthy + [0.0]) is None, "the first exit stopped the campaign"
    assert _walk_band(healthy + [0.0] * 5) == frozen_at + BAND_PATIENCE - 1


def test_the_flat_trainer_verdict_pauses_on_the_third_consecutive() -> None:
    """§3b guard 2: one sub-floor iteration is converged; three is stopped.

    Below the floor the ratio is 0/0 noise, so the band gate is **skipped** and
    a distinct verdict fires -- *the trainer is not moving* -- a third
    frozen-loop mode no existing gate covers.
    """
    for prior in range(FLAT_TRAINER_PATIENCE - 1):
        records = [
            _record(index, **{boundary.FLAT_TRAINER_FLAG: 1.0})
            for index in range(10 - prior, 10)
        ]
        outcome = boundary.BoundaryOutcome()
        flat = boundary.evaluate_flat_trainer(
            DENOMINATOR_FLOOR / 100.0, 10, records, outcome, {}
        )
        assert flat, "a sub-floor denominator must skip the band gate"
        assert outcome.extras[boundary.FLAT_TRAINER_FLAG] == 1.0
        assert outcome.tripwire is None

    records = [
        _record(index, **{boundary.FLAT_TRAINER_FLAG: 1.0}) for index in range(8, 10)
    ]
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_flat_trainer(0.0, 10, records, outcome, {})
    assert outcome.tripwire == FLAT_TRAINER
    assert "the trainer is not moving" in outcome.detail


def test_a_denominator_at_the_floor_is_not_below_it() -> None:
    """The boundary is inclusive on the healthy side; the gate is skipped only below."""
    outcome = boundary.BoundaryOutcome()
    flat = boundary.evaluate_flat_trainer(
        DENOMINATOR_FLOOR, 10, [], outcome, {}
    )

    assert not flat
    assert boundary.FLAT_TRAINER_FLAG not in outcome.extras


def test_the_gauntlet_floor_pauses_on_the_second_consecutive_breach() -> None:
    """§4: first breach records and warns; two consecutive pause.

    Auto-rollback is rejected -- the resume path unconditionally takes the
    newest checkpoint, so a rollback would need a second control system
    fighting the one that exists, on a 64-game signal that resolves only gross
    regressions.
    """
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_gauntlet(0.35, 7, [], outcome, {})
    assert outcome.extras[boundary.GAUNTLET_BREACH] == 1.0
    assert outcome.tripwire is None

    # The prior gauntlet, **and the iterations between the two of them**. At
    # cadence 4 the gauntlets are iterations 7 and 11 and the campaign writes a
    # record for 8, 9 and 10 as well; a fixture holding only the flagged one is
    # a journal production never produces, and the escalation passed on it for
    # the wrong reason (review F1).
    prior = _gauntlet_records(7, breached=True) + [
        _record(index) for index in (8, 9, 10)
    ]
    escalated = boundary.BoundaryOutcome()
    boundary.evaluate_gauntlet(0.35, 11, prior, escalated, {})
    assert escalated.tripwire == GAUNTLET_FLOOR_TRIPWIRE
    assert f"{GAUNTLET_BREACH_PATIENCE} consecutive" in escalated.detail


def test_two_breaches_one_cadence_apart_still_escalate() -> None:
    """Review F1: the floor must be reachable on a **production** journal.

    ``consecutive_flag`` resets on every record that does not carry the flag,
    and at ``GAUNTLET_EVERY = 4`` three unflagged campaign records sit between
    one gauntlet and the next -- so the run was ``1`` at every gauntlet, the
    patience of 2 was unreachable, and ``gauntlet_floor`` was a pre-registered
    tripwire that could not fire. The only test guarding it passed on a fixture
    with the intervening iterations left out.

    The fix is what "consecutive" means: consecutive **gauntlets**, counted
    over the records that ran one (``gauntlet_score`` / ``gauntlet_anchor_gen``).

    Red without it: ``evaluate_gauntlet`` at iteration 11 leaves
    ``outcome.tripwire`` at ``None``, so ``assert outcome.tripwire ==
    GAUNTLET_FLOOR_TRIPWIRE`` fires.
    """
    records = _gauntlet_records(7, breached=True) + [
        _record(index) for index in (8, 9, 10)
    ]

    # The defect, stated as the two counts: over every record the run is zero
    # (iteration 10's silence reset it), over the gauntlets it is one.
    assert boundary.consecutive_flag(records, boundary.GAUNTLET_BREACH, 10) == 0
    assert (
        boundary.consecutive_flag(
            records, boundary.GAUNTLET_BREACH, 10, gate_keys=boundary.GAUNTLET_RAN
        )
        == 1
    )

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_gauntlet(0.35, 11, records, outcome, {})

    assert outcome.tripwire == GAUNTLET_FLOOR_TRIPWIRE, (
        "two breaches one cadence apart did not escalate: the iterations "
        "between the gauntlets reset the run, so the floor can never fire"
    )
    assert f"{GAUNTLET_BREACH_PATIENCE} consecutive" in outcome.detail


def test_a_gauntlet_that_scored_above_the_floor_does_reset_the_run() -> None:
    """The other side: a *measured* pass between two breaches is not a run.

    The fix must not turn the run into "two breaches ever". A gauntlet that
    played and scored above the floor is evidence, and it clears the count --
    only the iterations that ran no gauntlet are passed over.
    """
    records = (
        _gauntlet_records(3, breached=True)
        + [_record(index) for index in (4, 5, 6)]
        + _gauntlet_records(7, breached=False)
        + [_record(index) for index in (8, 9, 10)]
    )

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_gauntlet(0.35, 11, records, outcome, {})

    assert outcome.tripwire is None, (
        "a breach at 3 and a breach at 11 with a passing gauntlet at 7 between "
        "them escalated: they are not consecutive"
    )
    assert outcome.extras[boundary.GAUNTLET_BREACH] == 1.0


def test_a_score_at_or_above_the_floor_is_not_a_breach() -> None:
    """A regression detector must not fire on a healthy iteration."""
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_gauntlet(GAUNTLET_FLOOR, 7, [], outcome, {})

    assert outcome.tripwire is None
    assert boundary.GAUNTLET_BREACH not in outcome.extras


def test_degenerate_scalars_are_omitted_not_invented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #33: a measurement that did not happen is a *missing key*.

    The two degenerate gauntlets, both of which used to be executed by no test
    at all (review F4): an **aborted** one (the anchor did not verify, no games
    were played) and an **uninformative** one (fewer than
    ``GAUNTLET_MIN_DECISIVE`` decisive games, so the floor is not evaluated).
    Neither may write ``gauntlet_score``: a score that was never measured must
    not be a number a plotter will average, and a sentinel like ``-1.0`` is
    exactly such a number. Omission is unambiguous; a sentinel is not.

    The floor must also **not** be evaluated on an uninformative gauntlet -- a
    score concentrated at 0.5 by draws makes the tripwire dead, and a floor
    that fires on one is worse than none.
    """
    from komugi.campaign import gauntlet as gauntlet_module
    from komugi.campaign import loop as campaign_loop
    from komugi.eval.arena import ArenaSummary

    campaign = campaign_loop.CampaignConfig(
        iterations=4,
        run_seed=11,
        root=str(tmp_path),
        steps_per_iteration=4,
        batch_size=2,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
    )
    selfplay = SelfPlayConfig(workers=1, process_batch_size=1)
    pin = gauntlet_module.AnchorPin(
        generation=0,
        pinned_at_iteration=0,
        path=str(tmp_path / "anchor.pt"),
        name_digest="0" * 16,
        weight_sha256="a" * 64,
        bn_sha256="b" * 64,
        m=2,
        n=2,
        games=4,
        floor=GAUNTLET_FLOOR,
        draw_score=0.2,
    )

    class _Challenger:
        """The newest EMA, as much of it as ``_run_gauntlet`` reads."""

        path = tmp_path / "ema.pt"
        step = 8

    monkeypatch.setattr(campaign_loop, "_anchor_for", lambda *a, **k: pin)
    monkeypatch.setattr(campaign_loop, "load_checkpoint", lambda _d: _Challenger())

    def abort(*args: object, **kwargs: object) -> object:
        """The anchor did not verify; no games were played."""
        raise gauntlet_module.GauntletAborted("the anchor does not verify")

    monkeypatch.setattr(gauntlet_module, "run_gauntlet", abort)
    aborted = boundary.BoundaryOutcome()
    campaign_loop._run_gauntlet(
        tmp_path, campaign, selfplay, 3, ("w",), [], aborted, {}
    )

    assert aborted.extras == {"gauntlet_aborted": 1.0}, (
        "an aborted gauntlet wrote a score, a wall time or an anchor "
        "generation: nothing was measured"
    )
    assert aborted.tripwire is None
    assert [verdict for verdict in aborted.verdicts if not verdict.ok]

    drawn = gauntlet_module.GauntletResult(
        generation=0,
        # A score below the floor, deliberately: it must still not be read.
        score=0.0,
        summary=ArenaSummary(
            games=4, decisive=0, draws=4, total_plies=40, shortest_plies=10,
            longest_plies=10,
        ),
        wall_s=1.5,
        uninformative=True,
    )
    monkeypatch.setattr(gauntlet_module, "run_gauntlet", lambda *a, **k: drawn)
    uninformative = boundary.BoundaryOutcome()
    campaign_loop._run_gauntlet(
        tmp_path, campaign, selfplay, 3, ("w",), [], uninformative, {}
    )

    extras = uninformative.extras
    assert extras["gauntlet_uninformative"] == 1.0
    assert extras["gauntlet_wall_s"] == 1.5, (
        "the wall time is measured whenever games were played -- §6's cadence "
        "rule is set from it"
    )
    assert extras[boundary.GAUNTLET_ANCHOR_GEN] == 0.0
    assert boundary.GAUNTLET_SCORE not in extras
    assert "gauntlet_decisive" not in extras
    assert boundary.GAUNTLET_BREACH not in extras, (
        "the floor was evaluated on an uninformative gauntlet: a score "
        "concentrated at 0.5 by draws makes the tripwire dead"
    )
    assert uninformative.tripwire is None
    for name, value in extras.items():
        assert math.isfinite(value), f"{name} is {value!r}"
        assert value != -1.0, (
            f"{name} carries a sentinel; a missing measurement is a missing "
            f"key, because -1.0 is a number a plotter will average"
        )


# ---------------------------------------------------------------------------
# the immediate gates
# ---------------------------------------------------------------------------


def test_the_spike_bound_pauses_on_the_first_occurrence() -> None:
    """§5 bound 1: ``L_i > 3 * B_i`` is a broken run, not a bad iteration."""
    records = _baseline(4)
    outcome = boundary.BoundaryOutcome()

    boundary.evaluate_spike(10.0 * SPIKE_FACTOR + 0.1, 4, records, outcome, {})

    assert outcome.tripwire == LOSS_SPIKE
    assert outcome.extras[boundary.SPIKE_FLAG] == 1.0
    assert "broken run, not a bad iteration" in outcome.detail


def test_the_spike_bound_waits_for_a_baseline() -> None:
    """Evaluated only once three previous iterations exist.

    That is what defers the bound to iteration 4 without a separate iteration
    test -- and it is why a resumed campaign reads the baseline back from the
    journal instead of being blind for three iterations after every restart.
    """
    for available in range(SPIKE_BASELINE_ITERATIONS):
        records = [
            _record(index, **{boundary.LOSS_TAIL_MEDIAN: 10.0})
            for index in range(available)
        ]
        outcome = boundary.BoundaryOutcome()
        boundary.evaluate_spike(1e6, available, records, outcome, {})
        assert outcome.tripwire is None


def test_a_loss_exactly_at_the_bound_does_not_pause() -> None:
    """``L_i > 3 * B_i``, strictly: the tolerance is inclusive on the healthy side."""
    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_spike(10.0 * SPIKE_FACTOR, 4, _baseline(4), outcome, {})

    assert outcome.tripwire is None


def test_a_non_finite_step_total_pauses_and_cannot_be_muted() -> None:
    """§5 bound 2, and the ordering hazard it is the mitigation for.

    ``IterationMetrics`` rejects non-finite extras, so a campaign that wrote
    the tail median of a diverged iteration would die *inside* the telemetry
    write and the blanket handler would file a ``class: crash`` marker on a
    diagnostic. Checking first means the marker names the divergence.
    """
    outcome = boundary.BoundaryOutcome()
    muted = {NON_FINITE_LOSS: Mute(NON_FINITE_LOSS, 1, 1, 99)}

    fired = boundary.evaluate_non_finite_loss(
        [1.0, 2.0, float("nan")], outcome, muted
    )

    assert fired
    assert outcome.tripwire == NON_FINITE_LOSS, (
        "a non-finite loss was muted; it has no false-alarm mode"
    )
    assert "poisoned" in outcome.detail


def test_a_finite_history_arms_nothing() -> None:
    """The common case must be free of flags a plotter would read as an event."""
    outcome = boundary.BoundaryOutcome()

    assert not boundary.evaluate_non_finite_loss([1.0, 0.5, 0.25], outcome, {})
    assert outcome.tripwire is None
    assert outcome.extras == {}
    assert outcome.verdicts == []


# ---------------------------------------------------------------------------
# muting, and what a mute does not suppress
# ---------------------------------------------------------------------------


def test_a_mute_suppresses_the_pause_and_never_the_record() -> None:
    """§5: §10's counting rules run over recorded violations, not over pauses."""
    mutes = {DISPLACEMENT_BAND: Mute(DISPLACEMENT_BAND, 5, 5, 13)}
    outcome = _run_band(1, WARMUP_ITERATIONS + 2, mutes)

    assert outcome.tripwire is None
    assert outcome.extras[boundary.BAND_VIOLATION] == 1.0
    assert outcome.extras[boundary.MUTED_FLAG] == 1.0
    assert [verdict for verdict in outcome.verdicts if not verdict.ok]


def test_no_gate_in_the_unmutable_set_honours_a_mute() -> None:
    """§5: the five structural gates ignore `mutes` at the **point of use**.

    :func:`~komugi.campaign.pinned.mute_length` refuses such a mute and
    :func:`~komugi.campaign.pause.read_mutes` refuses it again, so the file
    cannot exist -- but "not mutable at any length" is the kind of claim that
    should hold where the mute is consulted as well as at the two points of
    entry, and the one gate a mute here would silence
    (``ema_alias``: ``KL == 0`` on every probe position, which is
    assumption-free) is the one with no false-alarm mode at all.
    """
    assert UNMUTABLE, "the unmutable set is empty; this assertion would be vacuous"
    for gate in sorted(UNMUTABLE):
        outcome = boundary.BoundaryOutcome()
        outcome.fire(gate, f"planted {gate}", {gate: Mute(gate, 5, 5, 99)})

        assert outcome.tripwire == gate, f"{gate} was muted; it is unmutable"
        assert boundary.MUTED_FLAG not in outcome.extras
        assert [verdict for verdict in outcome.verdicts if not verdict.ok]


def test_a_mutable_gate_named_by_a_mute_for_another_gate_still_fires() -> None:
    """A mute is per-tripwire: suppressing the band must not suppress the spike."""
    outcome = boundary.BoundaryOutcome()
    outcome.fire(
        LOSS_SPIKE, "planted", {DISPLACEMENT_BAND: Mute(DISPLACEMENT_BAND, 5, 5, 13)}
    )

    assert outcome.tripwire == LOSS_SPIKE
    assert boundary.MUTED_FLAG not in outcome.extras


def test_the_first_armed_tripwire_owns_the_pause() -> None:
    """One boundary, one ``PAUSED`` -- the reason names what fired first."""
    outcome = boundary.BoundaryOutcome()
    outcome.fire(LOSS_SPIKE, "the spike", {})
    outcome.fire(DISPLACEMENT_BAND, "the band", {})

    assert outcome.tripwire == LOSS_SPIKE
    assert outcome.detail == "the spike"
    assert len(outcome.verdicts) == 2, "the second violation was not recorded"


# ---------------------------------------------------------------------------
# enforcement: durable first, raise second
# ---------------------------------------------------------------------------


def test_enforce_writes_paused_before_raising(tmp_path: Path) -> None:
    """A watcher must find the marker on disk, not race the exception."""
    outcome = boundary.BoundaryOutcome()
    outcome.fire(LOSS_SPIKE, "planted", {})

    with pytest.raises(CampaignPaused) as raised:
        boundary.enforce(
            tmp_path,
            outcome,
            7,
            ("result-0", "result-1"),
            newest_checkpoint="ckpt/newest.pt",
            last_good_checkpoint="ckpt/previous.pt",
            written_at_step=64,
        )

    pause = read_pause(tmp_path)
    assert pause is not None
    assert pause.tripwire == LOSS_SPIKE
    assert pause.paused_at_iteration == 7
    assert pause.last_good_checkpoint == "ckpt/previous.pt"
    assert pause.written_at_step == 64
    assert raised.value.results == ("result-0", "result-1"), (
        "CampaignPaused must carry the iterations this call did run; a pause "
        "that returned would be indistinguishable from success"
    )


def test_enforce_does_nothing_when_no_tripwire_armed(tmp_path: Path) -> None:
    """A recorded-but-muted violation is not a pause."""
    boundary.enforce(tmp_path, boundary.BoundaryOutcome(), 7, ())

    assert read_pause(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# acceptances #23/#32/#36 -- the discriminator and the baseline
# ---------------------------------------------------------------------------


def test_campaign_records_are_discriminated_by_the_marker(tmp_path: Path) -> None:
    """Acceptances #23 and #32: the marker, not ``games > 0``.

    The journal holds **two** ``IterationMetrics`` per iteration -- the
    trainer's and the campaign's -- and rev 1 told them apart by ``games > 0``.
    That fails exactly when a post-mortem needs it: an iteration whose
    self-play produced no games still trains on the window's older iterations
    and still writes a campaign record, so the discriminator breaks during a
    self-play outage, which is when the journal is being read in anger.
    """
    journal = tmp_path / "journal.jsonl"
    for iteration in range(3):
        write_record(journal, _trainer_record(iteration, 12.0))
        write_record(
            journal,
            IterationMetrics(
                iteration=iteration,
                # Zero games: the case rev 1's discriminator got wrong.
                games=0,
                draw_rate=0.0,
                avg_game_len=0.0,
                policy_entropy=0.0,
                extras={
                    boundary.CAMPAIGN_MARKER: 1.0,
                    boundary.LOSS_TAIL_MEDIAN: 10.0,
                },
            ),
        )

    campaign = boundary.campaign_records(journal)

    assert len(campaign) == 3
    assert all(record.games == 0 for record in campaign)
    assert all(boundary.LOSS_TAIL_MEDIAN in record.extras for record in campaign)
    assert not any("loss_total" in record.extras for record in campaign)


def test_spike_baseline_reads_only_loss_tail_median(tmp_path: Path) -> None:
    """Acceptance #36/F5: ``B_i`` is a median of tail medians, never of last steps.

    The trainer's ``loss_total`` is the **last step's** total. Comparing a
    median of three single-step draws against a median of thirty by a fixed
    factor of 3, on a descending curve, is a bound whose false-positive rate
    nobody can state from the design.
    """
    journal = tmp_path / "journal.jsonl"
    for iteration in range(3):
        # A last-step total an order of magnitude away from the tail median:
        # if the baseline ever read this, the numbers below would not agree.
        write_record(journal, _trainer_record(iteration, 100.0))
        write_record(
            journal, _record(iteration, **{boundary.LOSS_TAIL_MEDIAN: 10.0})
        )

    records = boundary.campaign_records(journal)
    assert boundary.spike_baseline(records, 3) == 10.0

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_spike(31.0, 3, records, outcome, {})
    assert outcome.tripwire == LOSS_SPIKE, (
        "31 is above 3 x 10 and below 3 x 100: the baseline read the trainer's "
        "last-step total instead of the campaign's tail median"
    )


def test_the_baseline_is_none_until_enough_iterations_exist() -> None:
    """A bound with no baseline must be silent rather than assume one."""
    assert boundary.spike_baseline([], 3) is None
    short = [_record(0, **{boundary.LOSS_TAIL_MEDIAN: 10.0})]
    assert boundary.spike_baseline(short, 3) is None


def test_the_baseline_uses_only_iterations_before_this_one() -> None:
    """``B_i`` is the *previous* iterations' median; including ``L_i`` would blunt it."""
    records = _baseline(4) + [_record(4, **{boundary.LOSS_TAIL_MEDIAN: 1000.0})]

    assert boundary.spike_baseline(records, 4) == 10.0


def test_consecutive_flag_counts_a_run_and_resets_on_a_clean_iteration() -> None:
    """Read back from the journal so an escalation survives a resume.

    "Two consecutive band exits" must mean the same thing to a campaign that
    was restarted between them as to one that was not.
    """
    flagged = {boundary.BAND_VIOLATION: 1.0}
    records = [
        _record(0, **flagged),
        _record(1, **flagged),
        _record(2),
        _record(3, **flagged),
    ]

    assert boundary.consecutive_flag(records, boundary.BAND_VIOLATION, 3) == 1
    assert boundary.consecutive_flag(records, boundary.BAND_VIOLATION, 1) == 2
    assert boundary.consecutive_flag(records, boundary.BAND_VIOLATION, 2) == 0


# ---------------------------------------------------------------------------
# the tail median itself
# ---------------------------------------------------------------------------


def test_loss_tail_median_takes_the_median_of_the_last_tenth() -> None:
    """Median so one spike does not define the iteration; the tail because the
    tail is what the next iteration continues from."""
    history = [{TAIL_KEY: float(value)} for value in range(100)]

    # The last tenth is steps 90..99; their median is 94.5.
    assert boundary.loss_tail_median(history, TAIL_KEY) == 94.5


def test_loss_tail_median_always_includes_at_least_one_step() -> None:
    """A short iteration must still produce a number, not an empty median."""
    history = [{TAIL_KEY: 3.0}, {TAIL_KEY: 5.0}]

    assert boundary.loss_tail_median(history, TAIL_KEY) == 5.0


def test_one_spike_does_not_define_the_iteration() -> None:
    """The reason it is a median: a single bad step is not a broken run."""
    # One catastrophic step inside the tail of a hundred-step iteration: nine
    # of the ten tail samples are 1.0, so the median is 1.0 and the iteration
    # is reported as what it was.
    history = [{TAIL_KEY: 1.0} for _ in range(99)] + [{TAIL_KEY: 1e6}]

    assert boundary.loss_tail_median(history, TAIL_KEY) == 1.0


def test_an_empty_history_is_a_programming_error() -> None:
    """``training_iteration`` never returns an empty history; say so loudly."""
    with pytest.raises(ValueError, match="empty step history"):
        boundary.loss_tail_median([], TAIL_KEY)


def test_step_totals_are_plain_floats() -> None:
    """The finiteness check reads these before anything reaches ``extras``."""
    assert boundary.step_totals([{TAIL_KEY: 1}, {TAIL_KEY: 2.5}], TAIL_KEY) == [
        1.0,
        2.5,
    ]


# ---------------------------------------------------------------------------
# a re-run iteration supersedes the attempt that crashed before its checkpoint
# ---------------------------------------------------------------------------


def _fingerprint(iteration: int, digest: str, role: str) -> NetFingerprint:
    """One fingerprint whose whole content is ``digest`` repeated."""
    return NetFingerprint(
        iteration=iteration, role=role, sha256=digest * 64, param_count=7
    )


def test_a_rerun_iterations_fingerprints_supersede_the_crashed_attempts() -> None:
    """A campaign that re-runs an iteration must not pause on its own resume.

    The reachable schedule: a machine dies between sealing iteration ``i``'s
    shards and publishing its checkpoint. ``resume_point`` therefore starts at
    ``i`` again, the loop clears the crashed attempt's shard tree and re-runs
    it -- and the re-run trains a *different* net (nothing is seeded to
    reproduce a net that was never checkpointed) under the same iteration
    number. The journal is append-only, so both attempts' fingerprints are in
    it.

    Read raw, that is two nets claiming one iteration, which
    ``self_play_net_advanced`` calls a contradiction and the boundary would
    turn into a pause on an **unmutable** gate -- a resume failing on the
    evidence that it worked. The newest record is the one that describes what
    is on disk, and the boundary drops the rest.
    """
    crashed = [_fingerprint(0, "a", TRAIN), _fingerprint(0, "a", EMA)]
    rerun = [_fingerprint(0, "b", TRAIN), _fingerprint(0, "b", EMA)]

    # Without the supersede, this is what the gate is handed, and it fails.
    assert not self_play_net_advanced(crashed + rerun, role=TRAIN).ok
    assert "contradictory" in self_play_net_advanced(crashed + rerun, role=TRAIN).reason

    assert boundary.latest_fingerprints(crashed + rerun) == tuple(rerun)

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_liveness(crashed + rerun, 0, outcome, {})
    assert outcome.tripwire is None, (
        "the re-run of a crashed iteration paused the campaign on its own "
        f"resume: {outcome.detail}"
    )
    assert all(verdict.ok for verdict in outcome.verdicts), [
        verdict.reason for verdict in outcome.verdicts if not verdict.ok
    ]


def test_superseding_does_not_defang_a_genuine_freeze() -> None:
    """The other side: dropping duplicates must not drop the freeze signal.

    One flat net across two iterations is the v1 signature, and it stays a
    pause -- the supersede is per ``(iteration, role)``, so it can only ever
    collapse records that claim the *same* iteration, never two iterations that
    agree.
    """
    frozen = [
        _fingerprint(0, "a", TRAIN),
        _fingerprint(0, "a", EMA),
        _fingerprint(1, "a", TRAIN),
        _fingerprint(1, "a", EMA),
        # What iteration 1 served is EMA_0, so the cross-role invariant holds
        # and the tripwire below is the hash gate rather than that one.
        _fingerprint(1, "a", SELF_PLAY),
    ]

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_liveness(frozen, 1, outcome, {})

    assert outcome.tripwire == HASH_GATE
    assert "unchanged across 2 consecutive iterations" in outcome.detail
