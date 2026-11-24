"""The gates, tested against the failures they exist for.

The first test in this file is the v1 regression: six iterations of self-play
against a net whose weights never changed. It ran to completion, it looked
healthy in every metric anyone was plotting, and the check below is the one
that would have stopped it on iteration two.

The rest fence the boundaries -- a healthy run must not trip the gate (a
false alarm gets a gate switched off, which is worse than no gate), a lagged
refresh schedule must be expressible, and the freshness bound has to be
decided at exactly the boundary rather than near it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Final

import pytest
import torch
from torch import nn

from komugi.campaign.factory import NET_ENV, SMALL_NET, new_net
from komugi.campaign.pinned import RATIO_CENTRE, RATIO_TOLERANCE, band
from komugi.telemetry.liveness import (
    LivenessError,
    LivenessVerdict,
    displacement_band,
    displacement_ratio_within_band,
    ema_within_band,
    fingerprint_series_complete,
    log_freshness,
    ratio_band,
    self_play_net_advanced,
    served_net_is_the_previous_ema,
)
from komugi.telemetry.records import EMA, SELF_PLAY, TRAIN, NetFingerprint
from komugi.train.checkpoint import list_checkpoints, save_checkpoint
from komugi.train.digest import bn_digest, canonical_digest, parameter_keys
from komugi.train.losses import LossAssembler
from komugi.train.step import StepConfig, TrainStep
from komugi.train.wiring import P0A_RECIPE

RATIO_FIXTURE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "campaign"
    / "unit"
    / "fixtures"
    / "kom149_ratio_series.json"
)
"""The measured displacement ratios of every root on disk that recorded one.

Nine campaign roots, ``iteration -> r``, 184 points in total (KOM-149 Phase A
plus the six held-out roots the design review asked for). It lives under
``tests/campaign/unit/fixtures`` because it is campaign evidence rather than a
telemetry literal; this module is where it is *judged*.
"""

IN_SAMPLE_CAMPAIGNS: Final[tuple[str, ...]] = ("campaign5", "campaign5b", "campaign6")
"""The roots :data:`~komugi.campaign.pinned.RATIO_CENTRE` was measured on."""

HELD_OUT_CAMPAIGNS: Final[tuple[str, ...]] = (
    "campaign1",
    "campaign2",
    "campaign3",
    "campaign4",
    "campaign4c",
    "campaign4d",
)
"""The roots that took no part in choosing the centre, and pass anyway."""

IN_SAMPLE_POINTS: Final[int] = 77
"""Phase A tabulated 76; campaign 6 has since recorded its iteration 25."""

HELD_OUT_POINTS: Final[int] = 107

OLD_BAND: Final[tuple[float, float]] = (2.25, 36.0)
"""The refuted band, as a **literal**.

Written out rather than recomputed from ``(d/(1-d))**2``: the derivation is
refuted, and a test that re-derives it would keep a refuted model alive as
executable code, which is exactly what deleting ``displacement_band`` was for
(KOM-149 review F3).
"""


def _load_ratio_series() -> dict[str, dict[int, float]]:
    """Return the fixture as ``campaign -> {iteration: r}``, iterations as ints."""
    raw: dict[str, dict[str, float]] = json.loads(
        RATIO_FIXTURE.read_text(encoding="utf-8")
    )
    return {
        campaign: {int(iteration): float(ratio) for iteration, ratio in series.items()}
        for campaign, series in raw.items()
    }


def _points(campaigns: tuple[str, ...]) -> list[tuple[str, int, float]]:
    """Return ``(campaign, iteration, r)`` for every point of `campaigns`."""
    series = _load_ratio_series()
    return [
        (campaign, iteration, ratio)
        for campaign in campaigns
        for iteration, ratio in sorted(series[campaign].items())
    ]


def _fp(iteration: int, digest_seed: str, role: str = SELF_PLAY) -> NetFingerprint:
    """Return a self-play fingerprint whose digest is `digest_seed` repeated."""
    return NetFingerprint(
        iteration=iteration,
        role=role,
        sha256=digest_seed * 64,
        param_count=12_345_678,
    )


def test_frozen_self_play_net_is_not_ok() -> None:
    """The v1 regression: one digest across six iterations must fail the gate."""
    frozen = [_fp(i, "a") for i in range(6)]

    verdict = self_play_net_advanced(frozen)

    assert not verdict.ok
    assert "unchanged across 6 consecutive iterations" in verdict.reason
    assert "0..5" in verdict.reason


def test_two_identical_iterations_are_already_enough_to_fail() -> None:
    """The gate fires at the first repeat, not after a run has been wasted."""
    verdict = self_play_net_advanced([_fp(0, "a"), _fp(1, "a")])

    assert not verdict.ok


def test_a_freeze_late_in_a_healthy_run_is_caught() -> None:
    """A net that stops advancing partway through is still caught."""
    series = [_fp(0, "a"), _fp(1, "b"), _fp(2, "c"), _fp(3, "c"), _fp(4, "c")]

    verdict = self_play_net_advanced(series)

    assert not verdict.ok
    assert "2..4" in verdict.reason


def test_healthy_sequence_is_ok() -> None:
    """Every iteration a new net: the gate passes and says how far it looked."""
    healthy = [_fp(i, seed) for i, seed in enumerate("abcdef")]

    verdict = self_play_net_advanced(healthy)

    assert verdict.ok
    assert "advanced across 6 iterations" in verdict.reason
    assert "0..5" in verdict.reason


def test_a_repeated_digest_that_is_not_consecutive_is_ok() -> None:
    """Two runs of the same weights, separated by a change, is not a freeze."""
    verdict = self_play_net_advanced([_fp(0, "a"), _fp(1, "b"), _fp(2, "a")])

    assert verdict.ok


def test_out_of_order_fingerprints_are_sorted_before_judging() -> None:
    """Journal order does not have to be iteration order."""
    shuffled = [_fp(2, "c"), _fp(0, "a"), _fp(1, "a")]

    verdict = self_play_net_advanced(shuffled)

    assert not verdict.ok
    assert "0..1" in verdict.reason


def test_lag_allows_a_scheduled_refresh_but_not_a_freeze() -> None:
    """A campaign refreshing self-play every third iteration passes at lag 2."""
    every_third = [_fp(0, "a"), _fp(1, "a"), _fp(2, "a"), _fp(3, "b"), _fp(4, "b")]

    assert self_play_net_advanced(every_third, min_iterations_lag=2).ok
    assert not self_play_net_advanced(every_third, min_iterations_lag=1).ok

    stuck = [*every_third, _fp(5, "b"), _fp(6, "b")]
    assert not self_play_net_advanced(stuck, min_iterations_lag=2).ok


def test_negative_lag_is_a_programming_error() -> None:
    """A nonsensical tolerance raises rather than quietly meaning something."""
    with pytest.raises(ValueError):
        self_play_net_advanced([_fp(0, "a")], min_iterations_lag=-1)


def test_other_roles_are_ignored() -> None:
    """A frozen *train* fingerprint is not this gate's business."""
    series = [
        _fp(0, "a", role=TRAIN),
        _fp(1, "a", role=TRAIN),
        _fp(0, "b"),
        _fp(1, "c"),
    ]

    assert self_play_net_advanced(series).ok


def test_no_fingerprints_at_all_is_not_ok() -> None:
    """A gate with no input has verified nothing, and must not report a pass."""
    verdict = self_play_net_advanced([])

    assert not verdict.ok
    assert "nothing to check" in verdict.reason

    only_other_roles = [_fp(0, "a", role=EMA), _fp(1, "b", role=TRAIN)]
    assert not self_play_net_advanced(only_other_roles).ok


def test_a_single_iteration_passes_because_a_freeze_needs_two() -> None:
    """Iteration zero must not be blocked by a check it cannot yet fail."""
    verdict = self_play_net_advanced([_fp(0, "a")])

    assert verdict.ok
    assert "freeze needs" in verdict.reason


def test_duplicate_iteration_with_the_same_net_is_harmless() -> None:
    """A record written twice is a duplicate, not a contradiction."""
    verdict = self_play_net_advanced([_fp(0, "a"), _fp(0, "a"), _fp(1, "b")])

    assert verdict.ok


def test_contradictory_fingerprints_for_one_iteration_fail() -> None:
    """Two different nets claiming one iteration invalidates the comparison."""
    verdict = self_play_net_advanced([_fp(0, "a"), _fp(0, "b"), _fp(1, "c")])

    assert not verdict.ok
    assert "contradictory" in verdict.reason


def test_ema_band_flags_a_target_that_stopped_moving() -> None:
    """A KL of ~0 is the EMA path's frozen-loop signature."""
    verdict = ema_within_band([0.04, 0.03, 1e-9], low=1e-4, high=0.5)

    assert not verdict.ok
    assert "stopped moving" in verdict.reason


def test_ema_band_flags_a_diverging_trainer() -> None:
    """A KL above the ceiling means the target is no longer tracking."""
    verdict = ema_within_band([0.04, 0.03, 3.5], low=1e-4, high=0.5)

    assert not verdict.ok
    assert "not tracking" in verdict.reason


def test_ema_band_passes_a_healthy_series() -> None:
    """In-band values pass and report the mean that was judged."""
    verdict = ema_within_band([0.04, 0.03, 0.05], low=1e-4, high=0.5)

    assert verdict.ok


def test_ema_band_window_averages_the_tail() -> None:
    """A single noisy spike is absorbed by the window; a sustained one is not."""
    spike = [0.02, 0.02, 0.02, 0.9]

    assert ema_within_band(spike, low=1e-4, high=0.5, window=4).ok
    assert not ema_within_band(spike, low=1e-4, high=0.5, window=1).ok


def test_ema_band_window_longer_than_the_series_uses_what_exists() -> None:
    """A short history is judged on its whole self, not padded."""
    assert ema_within_band([0.2], low=0.1, high=0.3, window=64).ok


def test_ema_band_is_inclusive_at_both_edges() -> None:
    """Exactly on the bound is inside it."""
    assert ema_within_band([0.1], low=0.1, high=0.3).ok
    assert ema_within_band([0.3], low=0.1, high=0.3).ok


def test_ema_band_rejects_a_non_finite_sample() -> None:
    """NaN must fail the gate, not silently answer False to every bound."""
    verdict = ema_within_band([0.02, float("nan")], low=1e-4, high=0.5)

    assert not verdict.ok
    assert "non-finite" in verdict.reason


def test_ema_band_with_no_samples_is_not_ok() -> None:
    """No KL series recorded is a failed gate, not a passed one."""
    assert not ema_within_band([], low=1e-4, high=0.5).ok


def test_ema_band_rejects_a_nonsensical_band() -> None:
    """An inverted band or a zero window is a caller bug and raises."""
    with pytest.raises(ValueError):
        ema_within_band([0.1], low=0.5, high=0.1)
    with pytest.raises(ValueError):
        ema_within_band([0.1], low=0.1, high=0.5, window=0)
    with pytest.raises(ValueError):
        ema_within_band([0.1], low=0.1, high=float("inf"))


def test_log_freshness_boundary_is_inclusive() -> None:
    """Exactly at the bound passes; one ulp past it does not."""
    assert log_freshness(60.0, 60.0).ok
    assert log_freshness(59.999, 60.0).ok
    assert not log_freshness(60.001, 60.0).ok


def test_log_freshness_flags_a_stalled_writer() -> None:
    """A log nobody has touched in an hour fails a one-minute bound."""
    verdict = log_freshness(3600.0, 60.0)

    assert not verdict.ok
    assert "stalled or dead" in verdict.reason


def test_log_freshness_rejects_a_future_mtime() -> None:
    """A negative age is clock skew or a copied-in log, never freshness."""
    verdict = log_freshness(-5.0, 60.0)

    assert not verdict.ok
    assert "future" in verdict.reason


def test_log_freshness_rejects_a_non_finite_age() -> None:
    """A NaN age fails rather than comparing False against every bound."""
    assert not log_freshness(float("nan"), 60.0).ok
    assert not log_freshness(float("inf"), 60.0).ok


def test_log_freshness_rejects_a_nonsensical_bound() -> None:
    """A non-positive maximum age could never pass and is a caller bug."""
    with pytest.raises(ValueError):
        log_freshness(1.0, 0.0)
    with pytest.raises(ValueError):
        log_freshness(1.0, -60.0)


def test_zero_age_is_fresh() -> None:
    """A log written this instant is as fresh as it gets."""
    assert log_freshness(0.0, 60.0).ok


def test_verdict_require_raises_only_on_failure() -> None:
    """require() is how a campaign turns a gate into a stop."""
    LivenessVerdict(ok=True, reason="fine").require()

    with pytest.raises(LivenessError, match="frozen"):
        LivenessVerdict(ok=False, reason="frozen net").require()


def test_verdict_is_frozen() -> None:
    """A verdict cannot be talked out of its decision after the fact."""
    verdict = self_play_net_advanced([_fp(0, "a"), _fp(1, "a")])

    # setattr, not `verdict.ok = True`: the direct form is a static error, and
    # what is under test is the runtime guard behind it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(verdict, "ok", True)


# ===========================================================================
# KOM-022 §3a/§3b: the role parameter, the displacement band, the completeness
# invariant, the cross-role invariant, and the digest's field set.
# ===========================================================================


def test_the_role_parameter_gates_the_train_and_ema_series_too() -> None:
    """§3a: the same run-length question, asked one seat upstream.

    A trainer whose *parameters* stopped moving is the v1 freeze one level
    up, and a target that stopped moving is it one level down. The default
    stays ``self_play`` so every pre-KOM-022 caller is unchanged.
    """
    frozen_train = [_fp(i, "a", role=TRAIN) for i in range(4)]
    healthy_selfplay = [_fp(i, chr(ord("b") + i), role=SELF_PLAY) for i in range(4)]
    series = frozen_train + healthy_selfplay

    assert not self_play_net_advanced(series, role=TRAIN).ok
    assert self_play_net_advanced(series).ok, (
        "the default role must still be self_play, or every existing caller "
        "silently changes meaning"
    )


def test_the_reason_names_the_role_that_actually_froze() -> None:
    """The strings go into an operator alert; naming the wrong net is a lie."""
    # `role=EMA` explicitly: the default is `self_play` and stays that way (the
    # test above pins it, and §3a's whole point is that adding the parameter
    # preserves every existing caller), so an EMA series handed to the default
    # filters to nothing and the gate reports a *missing* series rather than a
    # frozen one.
    verdict = self_play_net_advanced(
        [_fp(0, "a", role=EMA), _fp(1, "a", role=EMA)], role=EMA
    )

    assert not verdict.ok
    assert "ema net" in verdict.reason
    assert "self_play" not in verdict.reason


def test_an_unknown_role_is_a_programming_error() -> None:
    """``NetFingerprint`` has three roles; a fourth would silently filter to none."""
    with pytest.raises(ValueError, match="role must be one of"):
        self_play_net_advanced([_fp(0, "a")], role="bn")


def test_the_band_is_the_measured_centre_spread_by_the_declared_tolerance() -> None:
    """Acceptance #10, re-pinned (KOM-149): centre 0.533, band (0.13325, 2.132).

    The band is no longer *derived* -- its derived centre ``(d/(1-d))**2 = 9``
    is refuted below -- so what this test defends is the remaining half of
    acceptance #10: the shipped bounds are exactly the declared centre divided
    and multiplied by the declared tolerance squared, so moving a bound means
    moving a *constant* in a commit, which the root's iteration-0 stamp then
    refuses mid-campaign.
    """
    assert ratio_band(centre=0.533, tolerance=2.0) == pytest.approx(
        (0.13325, 2.132), abs=1e-12
    )

    spread = RATIO_TOLERANCE**2
    assert band() == pytest.approx(
        (RATIO_CENTRE / spread, RATIO_CENTRE * spread), abs=1e-12
    )
    assert band() == pytest.approx((0.13325, 2.132), abs=1e-12), (
        "the pre-registered band moved without RATIO_CENTRE or RATIO_TOLERANCE "
        "moving with it"
    )


def test_the_band_refuses_a_positional_call() -> None:
    """Review F3: ``0.75`` is a valid decay *and* a valid centre.

    ``ratio_band`` replaced ``displacement_band(decay, tolerance)`` with the
    same arity and a different meaning for the first argument, so a call site
    carried over unedited would have type-checked, run, and silently banded
    around the EMA decay. Keyword-only makes that a ``TypeError`` at the call.
    """
    with pytest.raises(TypeError):
        ratio_band(0.533, 2.0)  # type: ignore


@pytest.mark.parametrize(
    ("centre", "tolerance"),
    [
        (0.0, 2.0),
        (-0.533, 2.0),
        (float("inf"), 2.0),
        (float("nan"), 2.0),
        (0.533, 0.5),
        (0.533, float("inf")),
    ],
)
def test_the_band_refuses_constants_it_cannot_spread(
    centre: float, tolerance: float
) -> None:
    """A band from a nonsensical constant is worse than no band.

    A centre of zero collapses the band to a point at zero -- a gate that can
    never fail on the ceiling and always fails on the floor -- and a tolerance
    below 1 inverts the bounds.
    """
    with pytest.raises(ValueError):
        ratio_band(centre=centre, tolerance=tolerance)


def test_the_refuted_derived_band_is_not_callable() -> None:
    """``displacement_band`` is deleted in all but the name it is exported under.

    The name survives only because ``komugi/telemetry/__init__.py`` re-exports
    it and that file was outside this change's allowlist; it cannot return a
    band. The check that matters is that no caller can get the refuted numbers
    back by accident.
    """
    with pytest.raises(NotImplementedError, match="deleted"):
        displacement_band(0.75, 2.0)


def test_every_in_sample_point_is_inside_the_repinned_band() -> None:
    """Campaigns 5, 5b and 6: the roots the centre was measured on.

    In-sample by construction, so this is a consistency check rather than
    evidence -- the evidence is the held-out test below. What it does pin is
    that the rounding of the median (0.5326 -> 0.533) did not push a measured
    point outside the band it was chosen to contain.
    """
    low, high = band()
    points = _points(IN_SAMPLE_CAMPAIGNS)
    assert len(points) == IN_SAMPLE_POINTS

    outside = [
        (campaign, iteration, ratio)
        for campaign, iteration, ratio in points
        if not displacement_ratio_within_band([ratio], low, high).ok
    ]
    assert outside == [], f"in-sample points outside [{low}, {high}]: {outside}"


def test_every_held_out_point_is_inside_the_repinned_band() -> None:
    """Campaigns 1, 2, 3, 4, 4c and 4d: never used to choose the centre.

    Review F2: a threshold read off the campaigns it will judge is tuned to
    pass by construction, which is what KOM-022 §3b rejected. These 107 points
    -- six roots, the largest measured ``r`` among them (1.634 at campaign 3
    iteration 4, inside the warm-up gate) -- took no part in the median, and
    all of them are inside the band. That is the out-of-sample evidence the
    re-pin rests on.
    """
    low, high = band()
    points = _points(HELD_OUT_CAMPAIGNS)
    assert len(points) == HELD_OUT_POINTS

    outside = [
        (campaign, iteration, ratio)
        for campaign, iteration, ratio in points
        if not displacement_ratio_within_band([ratio], low, high).ok
    ]
    assert outside == [], f"held-out points outside [{low}, {high}]: {outside}"

    every = _points(IN_SAMPLE_CAMPAIGNS + HELD_OUT_CAMPAIGNS)
    ratios = [ratio for _, _, ratio in every]
    assert len(ratios) == IN_SAMPLE_POINTS + HELD_OUT_POINTS == 184
    assert max(ratios) == pytest.approx(1.634038), (
        "the ceiling's 1.30x margin is quoted against this maximum"
    )


def test_the_old_band_would_have_failed_at_every_measured_point() -> None:
    """The refutation, pinned: ``[2.25, 36]`` is below every one of the 184 ``r``.

    Not "usually wrong" and not "wrong early" -- wrong at *every* iteration of
    *every* root that ever recorded the ratio, by a factor of at least 1.4 even
    at the largest measured point. A model whose prediction is never once
    inside its own tolerance is not a tripwire that needs re-calibrating; it is
    a tripwire that was measuring nothing, which is why every campaign since
    campaign 1 has run under a standing operator ack (KOM-147).

    ``ok is False`` per point rather than "would have paused": a pause needs
    two consecutive post-warm-up exits (``BAND_PATIENCE``), and the point here
    is the verdict, not the escalation.
    """
    low, high = OLD_BAND
    points = _points(IN_SAMPLE_CAMPAIGNS + HELD_OUT_CAMPAIGNS)
    assert len(points) == 184

    inside = [
        (campaign, iteration, ratio)
        for campaign, iteration, ratio in points
        if displacement_ratio_within_band([ratio], low, high).ok
    ]
    assert inside == [], f"the refuted band contained a measured point: {inside}"


@pytest.mark.parametrize("ratio", [0.0, 0.05])
def test_a_frozen_or_aliased_ema_still_fails_the_new_floor(ratio: float) -> None:
    """The floor kept its job: the re-pin lowered it, it did not remove it.

    ``0.0`` is the alias (the EMA *is* the trainer's tensor, or has stopped
    moving entirely) and ``0.05`` is the sub-alias regime around it. Both are
    below 0.13325. The sharper checks the floor sits in front of -- the
    exact-zero probe check and the flat-trainer denominator floor -- are
    separate, unmutable, and untouched by this ticket.
    """
    low, high = band()
    verdict = displacement_ratio_within_band([ratio], low, high)

    assert not verdict.ok
    assert "alias" in verdict.reason


def test_an_ema_that_stopped_tracking_still_fails_the_new_ceiling() -> None:
    """``r = 5.0`` is 3.8x the largest ratio nine campaigns ever recorded."""
    low, high = band()
    verdict = displacement_ratio_within_band([5.0], low, high)

    assert not verdict.ok
    assert "not tracking" in verdict.reason


def test_the_ratio_gate_names_the_ratio_and_not_the_raw_kl() -> None:
    """§3b: a ratio series through ``ema_within_band`` would misname the alert.

    Two functions rather than two callers of one, because an operator acts on
    the sentence, not on the call site: a verdict saying "EMA/train KL 40 is
    above the band ceiling" about a *dimensionless ratio* names a quantity
    that was never checked.
    """
    low, high = band()

    ceiling = displacement_ratio_within_band([high * 2.0], low, high)
    floor = displacement_ratio_within_band([low / 2.0], low, high)
    inside = displacement_ratio_within_band([(low + high) / 2.0], low, high)

    assert not ceiling.ok and "displacement ratio" in ceiling.reason
    assert "EMA/train KL" not in ceiling.reason
    assert not floor.ok and "alias" in floor.reason
    assert inside.ok


def test_the_ratio_gate_fails_on_a_non_finite_sample() -> None:
    """A ``nan`` compared against a bound answers ``False`` to every bound."""
    low, high = band()
    verdict = displacement_ratio_within_band([float("nan")], low, high)

    assert not verdict.ok
    assert "non-finite" in verdict.reason


def test_the_ratio_gate_with_no_samples_is_not_ok() -> None:
    """A gate whose input never arrived has not verified anything."""
    low, high = band()
    assert not displacement_ratio_within_band([], low, high).ok


# --- the completeness invariant (acceptance #31, at unit scale) -------------


def _complete_series(iterations: int) -> list[NetFingerprint]:
    """Return the ``3N - 1`` fingerprints ``iterations`` healthy iterations owe."""
    series: list[NetFingerprint] = []
    for index in range(iterations):
        digest = chr(ord("a") + index)
        series.append(_fp(index, digest, role=TRAIN))
        series.append(_fp(index, digest, role=EMA))
        if index >= 1:
            series.append(_fp(index, chr(ord("a") + index - 1), role=SELF_PLAY))
    return series


def test_a_complete_series_is_three_n_minus_one() -> None:
    """§3a: 2 records at iteration 0, 3 at every later one, no gaps."""
    for iterations in (1, 2, 5):
        series = _complete_series(iterations)
        assert len(series) == 3 * iterations - 1
        assert fingerprint_series_complete(series, iterations - 1).ok


def test_a_missing_write_fails_the_completeness_gate() -> None:
    """The run-length gate passes *vacuously* on a short series; this does not.

    One missing write -- an early ``continue``, an exception between the EMA
    publish and the record write, a resume that starts mid-sequence -- would
    otherwise silently narrow the freeze gate to the iterations that happened
    to be recorded.
    """
    series = [f for f in _complete_series(3) if not (f.iteration == 1 and f.role == EMA)]

    verdict = fingerprint_series_complete(series, 2)

    assert not verdict.ok
    assert "iteration 1 is missing" in verdict.reason
    assert "'ema'" in verdict.reason


def test_a_gap_in_the_iteration_sequence_fails() -> None:
    """A resume that started mid-sequence is a gap, not a short series."""
    series = [f for f in _complete_series(4) if f.iteration != 2]

    assert not fingerprint_series_complete(series, 3).ok


def test_iteration_zero_owes_two_records_not_three() -> None:
    """There is no served file at iteration 0: ``CHECKPOINT_ENV`` is empty."""
    two = [_fp(0, "a", role=TRAIN), _fp(0, "a", role=EMA)]

    assert fingerprint_series_complete(two, 0).ok


# --- the cross-role invariant (acceptance #9, at unit scale) ---------------


def test_the_cross_role_invariant_holds_from_iteration_one() -> None:
    """§3a/F17: ``self_play_i == ema_{i-1}``, from ``i >= 1`` and not ``i >= 2``.

    Iteration 1 is the *first* iteration at which "did the loop point self-play
    at the EMA at all" is answerable, and it is where the trainer/serving
    binding split shows up first. Rev 1's ``i >= 2`` gave the most dangerous
    edit in the ticket a free iteration.
    """
    series = _complete_series(3)

    assert served_net_is_the_previous_ema(series, 1).ok
    assert served_net_is_the_previous_ema(series, 2).ok
    assert served_net_is_the_previous_ema(series, 0).ok, (
        "iteration 0 serves fresh weights; the question does not arise"
    )


def test_serving_the_train_net_instead_of_the_ema_is_caught() -> None:
    """The F4 defect: ``latest`` repointed instead of split.

    A loop that served the *trainer's* newest checkpoint publishes a
    ``self_play`` digest that is iteration ``i``'s train digest rather than
    iteration ``i-1``'s EMA digest, and nothing else in the system raises.
    """
    series = [
        _fp(0, "a", role=TRAIN),
        _fp(0, "a", role=EMA),
        _fp(1, "b", role=TRAIN),
        _fp(1, "b", role=EMA),
        _fp(1, "b", role=SELF_PLAY),
    ]

    verdict = served_net_is_the_previous_ema(series, 1)

    assert not verdict.ok
    assert "is not playing the net this campaign thinks it published" in verdict.reason


def test_a_missing_served_fingerprint_fails_rather_than_passing() -> None:
    """"Cannot check" is not "checked and fine"."""
    series = [_fp(0, "a", role=TRAIN), _fp(0, "a", role=EMA)]

    assert not served_net_is_the_previous_ema(series, 1).ok


# --- acceptance #42: the digest's key set is the optimiser's parameter set --


def test_digest_key_set_is_the_optimiser_parameter_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance #42, on a real net: parameters in, every buffer out.

    The predicate is "does the optimiser write to this?", not "is this a
    float?" -- which is the whole of rev 3's fix. Asserted against the live
    module rather than against a name list, so a net that grows a module
    cannot drift out of the digest, and a net that grows a *buffer* cannot
    drift into it.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    keys = set(parameter_keys(net))

    assert keys == {name for name, _ in net.named_parameters()}

    buffers = dict(net.named_buffers())
    assert buffers, "the net grew no buffers; this assertion would be vacuous"
    assert keys & set(buffers) == set()
    for forbidden in ("running_mean", "running_var", "num_batches_tracked"):
        assert not any(key.endswith(forbidden) for key in keys), (
            f"{forbidden} is in the liveness digest: it is advanced by the "
            f"forward pass in train() mode, so a digest containing it cannot "
            f"fail on a trainer that stopped learning"
        )

    # ... and it is the set the optimiser is actually constructed over.
    step = TrainStep(
        net, LossAssembler(P0A_RECIPE), StepConfig(batch_size=2, lr=1e-3, seed=1)
    )
    optimised = {
        id(tensor)
        for group in step.optimizer.param_groups
        for tensor in group["params"]
    }
    assert optimised == {id(tensor) for _, tensor in net.named_parameters()}


def test_a_new_module_joins_the_digest_with_no_edit_to_a_name_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance #42's future-proofing half, run the right way round.

    A later module's *trainable* tensors appear in ``named_parameters()``
    automatically and are digested automatically; anything it registers as a
    buffer is excluded automatically, because none of them is evidence that
    learning happened.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    before = set(parameter_keys(net))

    extra = nn.Linear(3, 2)
    extra.register_buffer("tally", torch.zeros(2, dtype=torch.int64))
    net.add_module("later", extra)

    after = set(parameter_keys(net))
    assert after - before == {"later.weight", "later.bias"}
    assert "later.tally" not in after

    state = net.state_dict()
    digest = canonical_digest(state, parameter_keys(net))
    assert digest.scalars == sum(
        int(tensor.numel()) for _, tensor in net.named_parameters()
    )


def test_digesting_a_state_dict_missing_a_parameter_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subset digest is a gate that quietly starts checking less."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    keys = parameter_keys(net)
    state = dict(net.state_dict())
    del state[keys[0]]

    with pytest.raises(KeyError, match="missing"):
        canonical_digest(state, keys)


def test_filename_digest_is_not_the_weight_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #6: the §1 trap, pinned.

    Two checkpoints saved from **identical** weights at different iterations
    have different *filename* digests -- the name covers ``meta_json`` and the
    meta carries ``iteration`` -- and the same ``canonical_digest``.

    **Necessary and not sufficient**, and rev 1 relied on it as if it were
    sufficient: weights built identical by hand carry identical counters *and*
    identical running statistics, so this passes under rev 1's unfailable
    digest and under rev 2's alike. The conditions that actually gate the
    defect are #27 (the digest does not move without learning) and #41 (the
    gate fires when it does not move).
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    keys = parameter_keys(net)
    state = net.state_dict()

    save_checkpoint(tmp_path, 8, state, {}, {"iteration": 0})
    save_checkpoint(tmp_path, 16, state, {}, {"iteration": 1})

    refs = list_checkpoints(tmp_path)
    assert len(refs) == 2
    assert refs[0].digest != refs[1].digest, (
        "the filename digest did not move for a frozen net, so this test "
        "cannot demonstrate the trap it exists for"
    )
    assert canonical_digest(state, keys).sha256 == canonical_digest(
        dict(state), keys
    ).sha256
    assert bn_digest(state, keys).sha256 == bn_digest(dict(state), keys).sha256
