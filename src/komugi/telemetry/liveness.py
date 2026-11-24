"""Liveness gates: the checks that would have stopped the v1 frozen loop.

v1 ran six iterations against a self-play net that never changed. Nothing was
broken in a way that raised -- games were played, losses went down, the log
filled up -- and the only evidence was a hash nobody was recording. The
functions here are the check that failure should have tripped, plus its two
siblings (a target-net drift band and a log-staleness bound).

Every function in this module is **pure**: it takes numbers and records, and
returns a :class:`LivenessVerdict`. No file is opened, no clock is read.
That is what makes each failure mode reproducible in a unit test from a
literal, and it is why freshness takes an *age* rather than a path -- the
caller does the ``stat`` and the clock arithmetic, and the decision boundary
stays testable.

The verdicts distinguish two things a gate can mean by "not ok": the signal
says the run is broken, and the signal is missing entirely. Both are ``ok =
False``, because a gate that cannot see its input is not a gate that passed;
the ``reason`` string says which, and is written to be pasted straight into
an operator's alert.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from komugi.telemetry.records import (
    EMA,
    NET_ROLES,
    SELF_PLAY,
    TRAIN,
    NetFingerprint,
)


class LivenessError(RuntimeError):
    """Raised by :meth:`LivenessVerdict.require` when a gate does not pass."""


@dataclass(frozen=True, slots=True)
class LivenessVerdict:
    """A gate's decision plus the human-readable reason behind it.

    Intentionally has no ``__bool__``: ``if verdict:`` on a frozen dataclass
    would be truthy for a *failing* verdict, which is exactly the silent-pass
    bug this module exists to prevent. Callers read :attr:`ok` or call
    :meth:`require`.
    """

    ok: bool
    reason: str

    def require(self) -> None:
        """Raise :class:`LivenessError` with the reason unless the gate passed."""
        if not self.ok:
            raise LivenessError(self.reason)


def _passed(reason: str) -> LivenessVerdict:
    """Return a passing verdict."""
    return LivenessVerdict(ok=True, reason=reason)


def _failed(reason: str) -> LivenessVerdict:
    """Return a failing verdict."""
    return LivenessVerdict(ok=False, reason=reason)


def self_play_net_advanced(
    fingerprints: Sequence[NetFingerprint],
    min_iterations_lag: int = 0,
    role: str = SELF_PLAY,
) -> LivenessVerdict:
    """Check that a net actually changes from iteration to iteration.

    This is the v1 regression gate. Entries with a role other than `role` are
    ignored, so a caller can hand over a whole journal's fingerprints; the
    remainder is ordered by iteration and inspected for a run of identical
    digests across consecutive iterations.

    `role` defaults to ``'self_play'``, which preserves every existing caller,
    and exists because KOM-022 publishes the same run-length question about
    the ``train`` and ``ema`` series: a trainer whose *parameters* stopped
    moving is the freeze this gate was written for, one seat upstream. The
    role is interpolated into every reason string below, because those strings
    are written to be pasted into an operator's alert and one saying
    "self-play net" about a frozen EMA would be a lie in the alert.

    ``min_iterations_lag`` is the number of iterations the self-play net is
    *allowed* to stay put -- a campaign that refreshes its workers every third
    iteration passes ``2``. A run of ``L`` consecutive iterations sharing one
    digest fails when ``L > min_iterations_lag + 1``, so the default of ``0``
    means "every iteration must be a new net", and the v1 signature (one
    digest repeated twice or more) fails immediately.

    Fewer than two self-play iterations passes with a reason saying so: a
    freeze is a statement about *change*, and there is nothing yet to compare
    at iteration zero. An empty sequence fails -- a gate whose input never
    arrived has not verified anything.
    """
    if min_iterations_lag < 0:
        raise ValueError(
            f"min_iterations_lag must be >= 0, got {min_iterations_lag!r}"
        )
    if role not in NET_ROLES:
        raise ValueError(f"role must be one of {sorted(NET_ROLES)}, got {role!r}")

    series = sorted(
        (f for f in fingerprints if f.role == role),
        key=lambda f: f.iteration,
    )
    if not series:
        return _failed(
            f"no {role} net fingerprints recorded: the freeze gate has "
            f"nothing to check"
        )

    # One entry per iteration. A repeated iteration with a *different* digest
    # means two different nets claim the same iteration, which invalidates the
    # comparison rather than merely failing it.
    unique: list[NetFingerprint] = []
    for fingerprint in series:
        if unique and unique[-1].iteration == fingerprint.iteration:
            if not unique[-1].same_net_as(fingerprint):
                return _failed(
                    f"contradictory {role} fingerprints for iteration "
                    f"{fingerprint.iteration}: {unique[-1].sha256[:12]} and "
                    f"{fingerprint.sha256[:12]}"
                )
            continue
        unique.append(fingerprint)

    if len(unique) < 2:
        return _passed(
            f"only {len(unique)} {role} iteration recorded; a freeze needs "
            f"two to be visible"
        )

    allowed_run = min_iterations_lag + 1
    run_start = 0
    for index in range(1, len(unique) + 1):
        if index < len(unique) and unique[index].same_net_as(unique[run_start]):
            continue
        run = index - run_start
        if run > allowed_run:
            first = unique[run_start]
            last = unique[index - 1]
            return _failed(
                f"{role} net {first.sha256[:12]} unchanged across {run} "
                f"consecutive iterations ({first.iteration}..{last.iteration}); "
                f"at most {allowed_run} allowed at min_iterations_lag="
                f"{min_iterations_lag}"
            )
        run_start = index

    return _passed(
        f"{role} net advanced across {len(unique)} iterations "
        f"({unique[0].iteration}..{unique[-1].iteration})"
    )


def ema_within_band(
    kl_series: Sequence[float],
    low: float,
    high: float,
    window: int = 1,
) -> LivenessVerdict:
    """Check that the EMA net's divergence from the train net stays in a band.

    `kl_series` is the per-iteration KL divergence between the target (EMA)
    net and the net being trained, oldest first. The gate is two-sided on
    purpose: the *floor* is the frozen-loop detector for the EMA path -- a
    target that has stopped moving reports a KL of essentially zero while
    everything else looks healthy -- and the ceiling catches an EMA that has
    been left behind by a diverging trainer.

    The decision is made on the mean of the last `window` samples, so a single
    noisy iteration does not trip a campaign-level gate; ``window=1`` (the
    default) tests the latest sample alone. A window longer than the series
    uses whatever samples exist. Non-finite samples in the window fail the
    gate rather than propagating a ``NaN`` into a comparison that would
    silently answer ``False`` to every bound.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"band bounds must be finite, got [{low!r}, {high!r}]")
    if low > high:
        raise ValueError(f"band must satisfy low <= high, got [{low!r}, {high!r}]")

    if not kl_series:
        return _failed("no EMA/train KL samples recorded: nothing to band-check")

    samples = tuple(kl_series[-window:])
    for sample in samples:
        if not math.isfinite(sample):
            return _failed(
                f"EMA/train KL series contains a non-finite sample: {sample!r}"
            )

    mean = math.fsum(samples) / len(samples)
    span = f"mean over the last {len(samples)} iteration(s)"
    if mean < low:
        return _failed(
            f"EMA/train KL {mean:.6g} ({span}) is below the band floor {low:.6g}: "
            f"the target net has stopped moving"
        )
    if mean > high:
        return _failed(
            f"EMA/train KL {mean:.6g} ({span}) is above the band ceiling "
            f"{high:.6g}: the target net is not tracking the trainer"
        )
    return _passed(
        f"EMA/train KL {mean:.6g} ({span}) is within [{low:.6g}, {high:.6g}]"
    )


def ratio_band(*, centre: float, tolerance: float) -> tuple[float, float]:
    """Return the displacement-ratio band around a **measured** `centre`.

    KOM-022 §3b as amended by KOM-149. The gate is on the dimensionless ratio

    ``r = KL(pi_ema || pi_train_i) / KL(pi_train_{i-1} || pi_train_i)``,

    "how far behind the trainer the EMA is, measured in units of the trainer's
    last step". The band allows a factor `tolerance` **in displacement**, i.e.
    ``tolerance**2`` in KL: ``[centre / t^2, centre * t^2]``.

    **Refuted history, kept here because it is the reason this signature takes
    a centre at all.** Rev 1 *derived* the centre: locally ``KL`` is quadratic
    in the displacement, and with a *steady* per-iteration trainer displacement
    the EMA's lag is ``d / (1 - d)`` of it, giving ``r* = (d / (1 - d))**2`` --
    9 at ``d = 0.75``, band ``[2.25, 36]``. The steady-displacement assumption
    is false: consecutive trainer steps are not collinear, so the EMA (an
    average of the last ~4 nets) lands *closer* to ``theta_i`` than
    ``theta_{i-1}`` does and ``r < 1``. Measured over **184 iterations across
    nine campaign roots**, ``r`` never exceeded **1.634** and its post-warm-up
    median was **0.533** -- the derived 9 is not merely late, it is out of
    range by a factor of ~17 (KOM-147, KOM-149 Phase A;
    ``docs/design/kom-149-band-repin.md``).

    **The rule that replaces it**: the centre is a *measured* post-warm-up
    median, pinned as a constant
    (:data:`~komugi.campaign.pinned.RATIO_CENTRE`) and stamped into the root at
    iteration 0, so moving it mid-campaign is refused by the drift check rather
    than being a silent re-tune. This function derives nothing; it only applies
    the declared tolerance to the declared centre.

    **What the floor still detects** (review F5). The measured/predicted
    scaling ``0.533 / 9 ~ 1/17`` maps the floor ``0.133`` back to an effective
    EMA decay of about **0.6**: a decay regression to ``<= 0.6`` -- an EMA that
    has begun tracking the trainer too closely to be a smoothed net -- exits
    the floor and pauses; a drift inside ``0.6-0.75`` does not. The two sharper
    floor checks are *separate* and untouched by any band re-pin: the
    exact-zero alias check (``KL(pi_ema || pi_train) == 0`` on every probe
    position) and the flat-trainer denominator floor, both unmutable.

    **Pre-registered response to an early ceiling exit**: the early regime
    (iterations 4-8, median ``r ~ 1.0``) sits at about twice the centre, so the
    ceiling's margin is thinnest there -- 1.30x over the largest measured
    ``r``. Two consecutive early ceiling exits in an otherwise healthy campaign
    are answered by **extending the warm-up**
    (:data:`~komugi.campaign.pinned.WARMUP_ITERATIONS`), never by widening the
    band. A band widened to fit the campaign it was supposed to judge is not a
    tripwire.

    Keyword-only on purpose (review F3): the argument this takes is a
    *centre*, and the deleted ``displacement_band(decay, tolerance)`` took a
    *decay*. ``0.75`` is a valid value of both, so a positional call site
    carried over unedited would silently get a band around 0.75 and still pass
    every type check.

    Raises:
        ValueError: On a `centre` that is not finite and positive, or a
            `tolerance` that is not finite and at least 1.
    """
    if not math.isfinite(centre) or centre <= 0.0:
        raise ValueError(f"centre must be finite and > 0, got {centre!r}")
    if not math.isfinite(tolerance) or tolerance < 1.0:
        raise ValueError(f"tolerance must be finite and >= 1, got {tolerance!r}")
    spread = tolerance**2
    return (centre / spread, centre * spread)


def displacement_band(decay: float, tolerance: float) -> tuple[float, float]:
    """Deleted in all but the name: the refuted derived band (KOM-149).

    The derivation this computed -- ``(d/(1-d))**2`` with a ``tolerance**2``
    spread, ``[2.25, 36]`` at ``d = 0.75`` -- is refuted (see
    :func:`ratio_band`), and it is refuted in a way that makes leaving it
    callable dangerous rather than merely untidy: ``decay`` and ``centre`` have
    overlapping ranges, so every old call site type-checks against the new one.
    It therefore raises rather than returning anything.

    It still exists **only** because ``komugi/telemetry/__init__.py`` imports
    and re-exports the name, and that file was outside this change's allowlist;
    deleting the name outright there and here is a one-commit follow-up.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "displacement_band is deleted: its derived centre (d/(1-d))**2 = 9 was "
        "refuted over 184 measured iterations (max r = 1.634, post-warm-up "
        "median 0.533; KOM-149). Use "
        "ratio_band(centre=..., tolerance=...) with the pinned "
        "komugi.campaign.pinned.RATIO_CENTRE, or komugi.campaign.pinned.band()"
    )


def displacement_ratio_within_band(
    ratio_series: Sequence[float],
    low: float,
    high: float,
    window: int = 1,
) -> LivenessVerdict:
    """Check that the EMA's *displacement ratio* stays inside its band.

    The sibling of :func:`ema_within_band` for the quantity KOM-022 §3b
    actually gates. A separate function rather than a second caller of that
    one, because ``ema_within_band``'s reason strings name "EMA/train KL": a
    ratio series passed through it would emit an alert naming a quantity that
    was not checked, and an operator acting on an alert reads the sentence, not
    the call site. The raw KL series is still recorded and still band-checked
    by that function; this one decides.

    Same window and finite-sample discipline: the decision is on the mean of
    the last `window` samples, and a non-finite sample fails the gate rather
    than propagating a ``NaN`` into a comparison that answers ``False`` to
    every bound.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"band bounds must be finite, got [{low!r}, {high!r}]")
    if low > high:
        raise ValueError(f"band must satisfy low <= high, got [{low!r}, {high!r}]")

    if not ratio_series:
        return _failed(
            "no EMA/train displacement ratios recorded: nothing to band-check"
        )

    samples = tuple(ratio_series[-window:])
    for sample in samples:
        if not math.isfinite(sample):
            return _failed(
                f"displacement ratio series contains a non-finite sample: "
                f"{sample!r}"
            )

    mean = math.fsum(samples) / len(samples)
    span = f"mean over the last {len(samples)} iteration(s)"
    if mean < low:
        return _failed(
            f"displacement ratio {mean:.6g} ({span}) is below the band floor "
            f"{low:.6g}: the EMA is tracking the trainer too closely to be a "
            f"smoothed net -- an alias, or a trainer that has stopped"
        )
    if mean > high:
        return _failed(
            f"displacement ratio {mean:.6g} ({span}) is above the band ceiling "
            f"{high:.6g}: the EMA is not tracking the trainer"
        )
    return _passed(
        f"displacement ratio {mean:.6g} ({span}) is within [{low:.6g}, {high:.6g}]"
    )


def fingerprint_series_complete(
    fingerprints: Sequence[NetFingerprint], through_iteration: int
) -> LivenessVerdict:
    """Check that every iteration recorded the fingerprints it owes.

    :func:`self_play_net_advanced` passes *vacuously* on a short series and
    walks the recorded entries without checking that iterations are
    contiguous, so one missing write -- an early ``continue``, an exception
    between the EMA publish and the record write, a resume that starts
    mid-sequence -- silently degrades the gate instead of failing it. This is
    the check that makes "advances per iteration" mean per *iteration* rather
    than per *record*.

    The expected shape (KOM-022 §3a): **2** records at iteration 0 (``train``
    and ``ema``; there is no served file at iteration 0, since
    ``CHECKPOINT_ENV`` is empty), **3** at every iteration from 1 up, and no
    gaps. ``3N - 1`` in total after ``N`` iterations.
    """
    if through_iteration < 0:
        return _failed(
            f"through_iteration must be >= 0, got {through_iteration!r}: no "
            f"iteration has completed, so there is no series to complete"
        )
    seen: dict[int, set[str]] = {}
    for fingerprint in fingerprints:
        seen.setdefault(fingerprint.iteration, set()).add(fingerprint.role)
    for iteration in range(through_iteration + 1):
        expected = {TRAIN, EMA} if iteration == 0 else {TRAIN, EMA, SELF_PLAY}
        roles = seen.get(iteration, set())
        if not expected <= roles:
            return _failed(
                f"iteration {iteration} is missing the "
                f"{sorted(expected - roles)} net fingerprint(s): the liveness "
                f"series has a gap, so the freeze gate is checking fewer "
                f"iterations than the campaign ran"
            )
    expected_total = 3 * (through_iteration + 1) - 1
    return _passed(
        f"fingerprint series is complete through iteration {through_iteration} "
        f"({expected_total} records expected)"
    )


def served_net_is_the_previous_ema(
    fingerprints: Sequence[NetFingerprint], iteration: int
) -> LivenessVerdict:
    """Check the cross-role invariant: iteration `i` served ``EMA_{i-1}``.

    ``fingerprint(self_play, i).sha256 == fingerprint(ema, i-1).sha256`` for
    every ``i >= 1``. No existing gate expresses it, and it is the real audit
    of "did the loop point self-play at the EMA at all" -- the most dangerous
    edit in KOM-022 (the trainer/serving binding split) shows up here first,
    at ``i = 1``, which is the first iteration at which the question is
    answerable: iteration 0 serves fresh weights, and ``EMA_0 := theta_0`` is
    published before iteration 1 begins.

    Iterations below 1 pass with a reason saying the question does not arise.
    """
    if iteration < 1:
        return _passed(
            f"iteration {iteration} serves freshly initialised weights; the "
            f"cross-role invariant starts at iteration 1"
        )
    served = [
        f for f in fingerprints if f.role == SELF_PLAY and f.iteration == iteration
    ]
    target = [
        f for f in fingerprints if f.role == EMA and f.iteration == iteration - 1
    ]
    if not served or not target:
        return _failed(
            f"cannot check what iteration {iteration} served: "
            f"{len(served)} self_play fingerprint(s) for it and {len(target)} "
            f"ema fingerprint(s) for iteration {iteration - 1}"
        )
    if not served[0].same_net_as(target[0]):
        return _failed(
            f"iteration {iteration} served {served[0].sha256[:12]} but "
            f"EMA_{iteration - 1} is {target[0].sha256[:12]}: self-play is not "
            f"playing the net this campaign thinks it published"
        )
    return _passed(
        f"iteration {iteration} served EMA_{iteration - 1} "
        f"({served[0].sha256[:12]})"
    )


def log_freshness(mtime_age_s: float, max_age_s: float) -> LivenessVerdict:
    """Check that a log has been written to recently enough.

    `mtime_age_s` is ``now - mtime`` in seconds, computed by the caller. The
    bound is inclusive: an age of exactly `max_age_s` passes, so a check
    scheduled at the same period as the writer does not fail on rounding.

    A negative age means the file's mtime is in the future -- a clock step, a
    filesystem with a different clock, or a copied-in file -- and fails,
    because a stale log with a doctored mtime is precisely the case a
    freshness gate must not wave through.
    """
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError(f"max_age_s must be finite and positive, got {max_age_s!r}")
    if not math.isfinite(mtime_age_s):
        return _failed(f"log mtime age is not a finite number: {mtime_age_s!r}")
    if mtime_age_s < 0:
        return _failed(
            f"log mtime is {-mtime_age_s:.3g}s in the future: clock skew, or a "
            f"log that was not written by this run"
        )
    if mtime_age_s > max_age_s:
        return _failed(
            f"log last written {mtime_age_s:.3g}s ago, older than the "
            f"{max_age_s:.3g}s bound: the writer is stalled or dead"
        )
    return _passed(
        f"log last written {mtime_age_s:.3g}s ago, within the {max_age_s:.3g}s bound"
    )
