"""The pre-registered constants, in one module, recorded into every root.

KOM-022 §3b/§4/§5. Every threshold this ticket gates on lives here, for one
reason: a threshold that can be edited without the edit being visible is not
pre-registered, it is a number that will eventually be moved to make a red
campaign green.

Two mechanisms make that real, and neither is a promise:

* **The band is measured once and then pinned.** Rev 1 said "derived, not
  chosen: ``(d/(1-d))**2``, so it is pinnable before iteration 0". That
  derivation is **refuted** -- measured ``r`` never came within a factor of ten
  of it over 184 iterations across nine roots (KOM-147, KOM-149;
  ``docs/design/kom-149-band-repin.md``). :func:`band` is now
  :data:`RATIO_CENTRE`, a *measured* post-warm-up median, divided and
  multiplied by ``tolerance**2``
  (:func:`~komugi.telemetry.liveness.ratio_band`). A measured number is exactly
  the thing §3b feared -- "read off the first campaign, tuned to pass" -- so
  the second mechanism is no longer a nicety, it is the whole guard: the centre
  is a declared constant, stamped, and re-checked, so re-tuning it is an edit
  in a commit that a running root will refuse.
* **The constants are stamped into the root at iteration 0** and re-checked at
  every later iteration (:func:`check_against_root`). A mid-campaign change of
  ``RATIO_CENTRE`` or ``RATIO_TOLERANCE`` then makes the gate refuse to run,
  naming both values, which is what makes widening the band mid-campaign
  indistinguishable from tampering. ``git merge-base --is-ancestor`` bans
  back-dating; it does not ban tuning, and this does.

Stated rather than glossed (§11.8): **``RATIO_TOLERANCE = 2`` has no
derivation**, and :data:`RATIO_CENTRE` has no *derivation* either -- it has a
measurement, on roots that are named and whose held-out majority was not used
to fit it. The mitigation here is mechanical, not evidential.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from komugi.telemetry.liveness import ratio_band
from komugi.train.ema import EMA_DECAY

RATIO_TOLERANCE: Final[float] = 2.0
"""Allowed factor **in displacement**, so ``t**2 = 4`` in KL. Undermined only
by the fact that nothing derives it -- see the module docstring."""

RATIO_CENTRE: Final[float] = 0.533
"""The displacement ratio's **measured** centre (KOM-149).

The post-warm-up median of ``r`` is **0.5326** over the 37 iterations at
``iteration >= 14`` of campaigns 5, 5b and 6 -- the three roots that share the
current regime -- rounded to three decimals. That rounding rule is stated
because the rounding is the only judgement in the number: at three decimals the
band is ``(0.13325, 2.132)``, and both bounds are the exact product of this
constant with :data:`RATIO_TOLERANCE` squared.

The centre replaces a *derived* one, ``(d/(1-d))**2 = 9``, which the same
series refutes by a factor of ~17 (:func:`~komugi.telemetry.liveness.ratio_band`
carries the history). Fitting a threshold to measurements is what §3b rejected,
so what makes it admissible is stated rather than assumed: ``r`` is
dimensionless; the six **held-out** roots (campaigns 1, 2, 3, 4, 4c, 4d -- 107
points never used for the centre, max ``r = 1.634``) are all inside the band;
and the constant is stamped into the root, so a later re-tune is a refusal
rather than an edit nobody sees.

**The first PINNED run of this band is campaign 7.** Campaign 6's root was
stamped before this key existed, so :func:`check_against_root` skips it there
(a key a root never recorded is not drift) -- campaign 6 resumes cleanly under
the new band but its tail carries no tamper-evidence for the centre, and it is
in-sample besides. The pre-registered claim runs on campaign 7 and later:
iterations 1-30 with no ``displacement_band`` exit, refuted by two consecutive
recorded exits while every other gate is green.
"""

WARMUP_ITERATIONS: Final[int] = 4
"""The band gate does not run before this iteration: one EMA horizon, until
which ``EMA_0 = theta_0`` dominates and ``r`` is a statement about the initial
condition rather than about the recursion."""

DENOMINATOR_FLOOR: Final[float] = 1e-6
"""Below this the displacement ratio is 0/0 noise and the gate is skipped.

Derived from the **dtype**, not from this net's learning scale: float32 has a
relative resolution of ~1.2e-7, so two nets differing only by float32 rounding
produce per-position KLs at the ~1e-7 nats level, and a mean over 512 positions
in float64 does not reduce a bias of that size. 1e-6 is one decade above that
accumulation's noise floor. It moves only if the dtype does.
"""

FLAT_TRAINER_PATIENCE: Final[int] = 3
"""Consecutive sub-floor denominators before the flat-trainer verdict pauses.

One is a converged or stalled-for-one-iteration trainer; three in a row is a
trainer that has stopped. The first two record and warn."""

BAND_PATIENCE: Final[int] = 2
"""Consecutive band exits before a pause. The first records and warns.

The escalation is what makes §10's Claim 2 falsifiable *inside* one campaign:
its refutation needs two recorded exits in the first twelve post-warm-up
iterations, and a gate that stopped the campaign at the first could only ever
produce the second through an operator who had already judged the first a
false alarm."""

SPIKE_FACTOR: Final[float] = 3.0
"""``L_i > SPIKE_FACTOR * B_i`` pauses. A factor of three on a robust tail
median is a broken run, not a bad iteration; campaign0 descended 95.8 -> 33.7
over ten iterations monotonically, which calibrates the *shape* of a healthy
descent rather than its scale (that run's lr and batch size are unrecorded)."""

SPIKE_BASELINE_ITERATIONS: Final[int] = 3
"""How many previous iterations' ``loss_tail_median`` the baseline is a median
of. The bound is evaluated only once that many exist."""

LOSS_TAIL_FRACTION: Final[float] = 0.10
"""``L_i`` is the median of the last tenth of the iteration's step totals.

Median so one spike does not define the iteration; the tail rather than the
whole iteration because the tail is what the next iteration continues from --
the argument the trainer already makes for recording its last step."""

MIN_ANCHOR_AGE: Final[int] = 12
"""Iterations an anchor must reach before it may be re-pinned (three gauntlets
at cadence 4). Without a floor the steady state is an anchor eight iterations
behind, re-pinned every eight -- a *rolling* anchor at coarser granularity,
which is the shape §4 rejects. With it the anchor is a lagging fixed
reference and "mid-anchor" is an emergent property rather than a claim."""

REPIN_SCORE: Final[float] = 0.85
"""Two consecutive gauntlets at or above this, plus :data:`MIN_ANCHOR_AGE`,
re-pin the anchor."""

GAUNTLET_GAMES: Final[int] = 64
"""Games per gauntlet: 32 seat-swapped pairs."""

GAUNTLET_FLOOR: Final[float] = 0.40
"""One-sided 95% upper bound on the score below 0.50 at n=64:
``s + 1.645 * sqrt(0.25/64) < 0.50`` gives ``s < 0.397``. A regression
detector, not a progress meter -- requiring ``s > 0.5`` at 64 games would fire
on about half of healthy early iterations."""

GAUNTLET_MIN_DECISIVE: Final[int] = 8
"""Below this many decisive games of 64 the gauntlet is **uninformative** and
the floor is not evaluated: a score concentrated at 0.5 by draws makes the
tripwire dead, and a floor that cannot fire is worse than none."""

GAUNTLET_BREACH_PATIENCE: Final[int] = 2
"""Consecutive floor breaches within one anchor generation before a pause."""

GAUNTLET_EVERY: Final[int] = 4
"""Iterations between gauntlets; a gauntlet runs when ``(i + 1) % k == 0``.

**Provisional.** §6's cost model is a derivation from self-play throughput,
not a measurement, and every one of its errors points toward affordability --
so the pre-registered rule is that one 64-game sharded fast-budget gauntlet is
measured *before* this constant is pinned, and ``k`` is then the smallest of
``{1, 2, 4}`` with ``gauntlet_wall / iteration_wall <= 0.25``. Until that
measurement lands (acceptance #17, an ordering constraint on the ticket rather
than a test), 4 is the conservative default and may only be **lowered** by
that rule.
"""

GAUNTLET_SHARD_DEADLINE_S: Final[float] = 3850.0
"""How long a gauntlet may run before it is declared stalled.

**Ten times the worst gauntlet this box has ever recorded**, rounded up:
``gauntlet_wall_s`` over campaign 6's 27 gauntlets is min 129.9 s, median
209.0 s, p90 353.6 s, max 384.2 s (campaign 5: 7 gauntlets, max 304.8 s), so
the pin is ``10 x 385 s``. The sixteen shards run **concurrently** --
:func:`~komugi.campaign.gauntlet.run_gauntlet` sizes its pool at one worker per
request -- so per-shard wall is approximately *whole-gauntlet* wall, and one
number bounds both. That is why the gauntlet spends it as a single absolute
deadline over the whole fan-out rather than restarting a timer per shard: N
shards under a per-shard timer would bound the call at ``N x`` this number,
which at :data:`GAUNTLET_SHARDS` is seventeen hours, and nothing measured
justifies that.

**Why ten, stated rather than rounded to a pretty hour.** The two errors are
not symmetric. A deadline that fires late costs one idle hour on a campaign
that was already wedged; a deadline that fires early costs a real gauntlet:
``GauntletStalled`` -> the boundary's crash ``PAUSED`` -> on the next startup
the unmutable :data:`UNFINISHED_BOUNDARY` -> a manual rollback, and there is no
retry anywhere in that chain. Ten times the worst *observed* run is a decade of
headroom rather than a factor fitted to clear the maximum by a hair, and it is
the smallest multiple under which "a gauntlet that reached this is wedged, not
slow" still holds on a box whose loadavg of 40-60 between iterations is
routine. Raising it further weakens the detector in exact proportion -- the
stall is found one multiple later -- so the number is a bound on pathology, not
a tuning knob for slowness.

Restated from the campaign records by KOM-212 (KOM-180 impl review F2). The
previous 1800 s pin justified itself by "about two minutes of wall for all
sixteen shards together", so "two orders of magnitude inside this bound", and
both halves are false against the journal that was cited: two minutes is below
the observed *minimum*, and the real margin was 4.7x.

It exists because the alternative has no bound at all. ``pool.map`` waits
forever, so an abruptly-dead or wedged shard process hung the campaign until an
operator noticed -- and the operator's kill then landed between the published
checkpoint and the boundary record, i.e. exactly in the window
:data:`UNFINISHED_BOUNDARY` is about (review CP-04 + CP-01).

**Deliberately not stamped into the root.** It is absent from
:data:`CONSTANTS_KEYS` and from :func:`pinned_constants`, so correcting it
cannot make :func:`check_against_root` refuse a campaign that is already
running -- which is what makes a restatement like this one safe to land mid-
campaign. A supervisor's bound is not a measurement the results depend on.
"""

GAUNTLET_SHARDS: Final[int] = 16
"""Processes the 32 pairs are split across.

The gauntlet runs *between* iterations, when all 28 cores of this box are free,
and the shards are exact rather than approximate: every game's randomness is a
pure function of ``(seed, pair, seat, ply)`` and each shard keeps the **global**
pair index, so the union of the shards is the serial match. Sixteen because
that is the number §6's cost model is written against; it is a parallelism
knob, and unlike the cadence it changes no result.
"""

LOSS_SPIKE: Final[str] = "loss_spike"
DISPLACEMENT_BAND: Final[str] = "displacement_band"
FLAT_TRAINER: Final[str] = "flat_trainer"
GAUNTLET_FLOOR_TRIPWIRE: Final[str] = "gauntlet_floor"
HASH_GATE: Final[str] = "hash_gate"
CROSS_ROLE: Final[str] = "cross_role"
EMA_ALIAS: Final[str] = "ema_alias"
NON_FINITE_LOSS: Final[str] = "non_finite_loss"
FINGERPRINT_SERIES: Final[str] = "fingerprint_series"
EMA_PUBLISH_STALLED: Final[str] = "ema_publish_stalled"
"""KOM-022 §2 read off the step numbering: the EMA owed by this boundary was
not published (:func:`~komugi.campaign.loop.ema_publish_verdict`).

Its own name rather than :data:`HASH_GATE`'s, because the two failures are
different events with different diagnoses and ``PAUSED.tripwire`` is what an
operator (and every test of this gate) reads first. ``hash_gate`` is a *digest*
that stopped changing across two boundaries -- a freeze, however reached;
``ema_publish_stalled`` is a *publish* that did not happen at this one, visible
immediately and one whole iteration earlier. Sharing one name made the two
indistinguishable on disk and left the message -- which carries a hex digest --
as the only discriminator (KOM-025 wire-in review, F5).

Unmutable for HASH_GATE's reason verbatim: a served net that stops advancing is
v1's catastrophe, and a publish that did not happen has no false-alarm mode.
"""

UNFINISHED_BOUNDARY: Final[str] = "unfinished_boundary"
"""Startup found a published checkpoint whose iteration never closed.

The newest trainer checkpoint names iteration ``i``, and the journal holds no
campaign record for ``i`` -- so ``i``'s *boundary* (its fingerprints, its
measurements, its gauntlet, its liveness verdict) did not run, however the
process died. :func:`~komugi.campaign.loop.resume_point` reads that checkpoint
as "iteration ``i`` completed" and would enter ``i + 1``, which is the one
inference nothing on disk supports (review CP-01).

Unmutable, and for a sharper reason than the other members: an ack is a
*resume*, and there is nothing here to resume onto. ``FINGERPRINT_SERIES`` is
itself unmutable and :func:`~komugi.campaign.boundary.fingerprint_series_complete`
walks every iteration, so a root that advanced across the gap would pay a full
iteration, pause on the hole, and pause again at every later boundary --
permanently. And a boundary record for ``i`` is never synthesised, because a
fabricated verdict is the one artefact this whole design exists to make
impossible.

**The recovery, in full** (KOM-211; the ``PAUSED``'s own ``detail`` prints it
with the step substituted, and so does the message
:func:`~komugi.campaign.pause.apply_startup_state` raises):

* ``rm <root>/PAUSED``,
* ``rm <root>/ckpt/checkpoint-<(i + 1) * steps_per_iteration>-*.pt`` -- the
  abandoned attempt's trainer checkpoint, the file that makes ``resume_point``
  claim ``i`` completed,
* ``rm <root>/ema/checkpoint-<the same step>-*.pt`` -- its EMA sibling, if the
  kill landed after that publish,
* relaunch; or, equivalently, **fork the root at ``i - 1``** (a fresh root
  branched off ``i - 1``'s checkpoint, which is the ``last_good_checkpoint`` the
  ``PAUSED`` names).

The EMA file is not optional housekeeping. Left behind, it is the newest
published EMA, so the re-run of iteration ``i`` serves the *abandoned attempt's*
smoothed net to self-play instead of ``EMA_{i-1}`` -- the recursion the
displacement band is derived from, changed underneath the band, with a
self-consistent step numbering that no gate can see through. Leaving it was also
how the rollback used to dead-end: with ``EMA_i`` present and ``theta_i``
deleted, :func:`~komugi.train.ema.reconcile_ema` raises ``EMA is ahead of the
trainer``, an error whose text blames two campaigns sharing one root.
"""


MUTE_DEFAULTS: Final[Mapping[str, int]] = {
    LOSS_SPIKE: 2,
    DISPLACEMENT_BAND: 8,
    FLAT_TRAINER: 4,
    GAUNTLET_FLOOR_TRIPWIRE: 8,
}
"""How long an acked tripwire stays muted, **per tripwire**, in iterations.

A single global 2 was rev 1's, and its rationale -- "the minimum that lets a
loss baseline re-establish itself" -- is about the *spike* alone. A band
violation caused by non-constant displacement does not re-establish in two
iterations, so a global mute would expire and the campaign would re-pause on
the tripwire the operator had just acked: the trap the ack mechanism exists to
avoid, reintroduced for a different tripwire. Two EMA horizons for the band,
one for the flat trainer, and for the gauntlet a mute shorter than the
gauntlet cadence would mute nothing at all.

An ``ACK`` may **shorten** these and may never exceed them. A longer mute is a
ticket, not an operator decision.
"""

UNMUTABLE: Final[frozenset[str]] = frozenset(
    {
        HASH_GATE,
        CROSS_ROLE,
        EMA_ALIAS,
        NON_FINITE_LOSS,
        FINGERPRINT_SERIES,
        EMA_PUBLISH_STALLED,
        UNFINISHED_BOUNDARY,
    }
)
"""Tripwires no ``ACK`` may mute at any length.

A frozen self-play net is never a false alarm -- it is the failure this whole
ticket exists to make impossible -- and the aliasing check (``KL == 0`` on
every probe position) is assumption-free, so it has no false-alarm mode to
mute. An EMA the boundary owed and did not publish is the same failure seen a
boundary earlier, on the numbering rather than the weights.

An ``ACK`` naming one of these is **REFUSED**, and refused before anything on
disk is touched: :func:`~komugi.campaign.pause.apply_startup_state` validates
the ack, :func:`~komugi.campaign.pause._validate_ack` raises
:class:`~komugi.campaign.pause.PauseError` ("which cannot be acked") for every
member here, and the ``PAUSED`` is therefore still in place afterwards -- the
pause is not cleared, no mute is written, and the launch does not run. So an ack
is not "a clear that buys no silence" (what a previous revision of this
docstring and of :data:`UNFINISHED_BOUNDARY`'s said); it is not a usable
operator action at all. The recovery is whatever fixes the cause -- for
:data:`UNFINISHED_BOUNDARY`, the rollback spelled out in its docstring -- which
is why the message raised for an unmutable tripwire prints that recipe instead of
the ``ACK`` instruction the mutable ones get.
"""

TRIPWIRES: Final[frozenset[str]] = frozenset(MUTE_DEFAULTS) | UNMUTABLE
"""Every tripwire name a ``PAUSED`` or a mute file may carry."""

CONSTANTS_KEYS: Final[tuple[str, ...]] = (
    "ema_decay",
    "ratio_tolerance",
    "ratio_centre",
    "warmup_iterations",
    "denominator_floor",
    "min_anchor_age",
    "spike_factor",
)
"""Which pinned constants are stamped into the root at iteration 0."""


class ConstantsDrift(RuntimeError):
    """The live constants disagree with the ones this root was started under."""


class MissingConstants(RuntimeError):
    """This root's first campaign record stamped no pre-registered constants."""


def band() -> tuple[float, float]:
    """Return the shipped displacement-ratio band, from the declared constants.

    ``(0.13325, 2.132)``: :data:`RATIO_CENTRE` divided and multiplied by
    :data:`RATIO_TOLERANCE` squared. :data:`EMA_DECAY` no longer enters the
    band -- the derivation that used it is refuted (KOM-149) -- but it is still
    stamped into the root, because the EMA recursion it parameterises is
    unchanged and a campaign that changed it mid-flight would still be a
    different experiment.
    """
    return ratio_band(centre=RATIO_CENTRE, tolerance=RATIO_TOLERANCE)


def pinned_constants() -> dict[str, float]:
    """Return the constants to stamp into a root's iteration-0 journal record."""
    return {
        "ema_decay": float(EMA_DECAY),
        "ratio_tolerance": float(RATIO_TOLERANCE),
        "ratio_centre": float(RATIO_CENTRE),
        "warmup_iterations": float(WARMUP_ITERATIONS),
        "denominator_floor": float(DENOMINATOR_FLOOR),
        "min_anchor_age": float(MIN_ANCHOR_AGE),
        "spike_factor": float(SPIKE_FACTOR),
    }


def check_against_root(recorded: Mapping[str, float]) -> None:
    """Refuse to run if `recorded` disagrees with the live constants.

    `recorded` is the ``extras`` subset stamped at iteration 0 of this root.
    A key that was never recorded is not an error -- a root started before a
    constant existed cannot have stamped it, and refusing there would make
    adding a constant a breaking change to every live root. A key that *is*
    recorded and differs is a hard refusal naming both values.

    Raises:
        ConstantsDrift: On any disagreement.
    """
    live = pinned_constants()
    for key in CONSTANTS_KEYS:
        if key not in recorded:
            continue
        if recorded[key] != live[key]:
            raise ConstantsDrift(
                f"pre-registered constant {key} is {live[key]!r} in this "
                f"process but {recorded[key]!r} in this root's iteration-0 "
                f"record. The band, the anchor age and the spike factor are "
                f"pre-registered per root: changing one mid-campaign is "
                f"indistinguishable from tuning it to pass. Start a new root, "
                f"or restore the constant"
            )


def check_first_record(recorded: Mapping[str, float]) -> None:
    """Refuse a root whose **first campaign record** stamped no constants at all.

    :func:`check_against_root` is deliberately tolerant of a *missing key* -- a
    root started before a constant existed cannot have stamped it -- and that
    tolerance has one degenerate reading its caller must not take: a record
    carrying **none** of the keys is not a root that predates a constant, it is
    a root whose iteration-0 record is not the record being checked. Every
    KOM-022 root stamps the whole set at iteration 0, so a first record with no
    keys means iteration 0's was lost (or the journal was rewritten), and drift
    checking then degrades silently to "nothing recorded, therefore no drift"
    -- the pre-registration gone, with nothing on disk saying so (review F9).

    Raises:
        MissingConstants: If `recorded` carries none of :data:`CONSTANTS_KEYS`.
        ConstantsDrift: On any disagreement among the keys it does carry.
    """
    if not any(key in recorded for key in CONSTANTS_KEYS):
        raise MissingConstants(
            f"this root's first campaign record carries none of the "
            f"pre-registered constants {list(CONSTANTS_KEYS)}: iteration 0 "
            f"stamps all of them, so this record is not iteration 0's and the "
            f"drift check would pass vacuously on every later iteration. "
            f"Refusing to run a campaign whose band, anchor age and spike "
            f"factor are pinned to nothing"
        )
    check_against_root(recorded)


def mute_length(tripwire: str, requested: int | None = None) -> int:
    """Return how long `tripwire` is muted for, honouring a shortening request.

    Raises:
        ValueError: If `tripwire` is unknown or unmutable, or if `requested`
            is negative or exceeds the tripwire's default. An operator may
            shorten a mute and may never lengthen one.
    """
    if tripwire in UNMUTABLE:
        raise ValueError(
            f"{tripwire!r} is not mutable at any length: it has no false-alarm "
            f"mode, so an ack clears its pause and buys no silence"
        )
    if tripwire not in MUTE_DEFAULTS:
        raise ValueError(
            f"unknown tripwire {tripwire!r}; known: {sorted(TRIPWIRES)}"
        )
    default = MUTE_DEFAULTS[tripwire]
    if requested is None:
        return default
    if requested < 0:
        raise ValueError(f"mute_iterations must be >= 0, got {requested!r}")
    if requested > default:
        raise ValueError(
            f"mute_iterations {requested} exceeds {tripwire!r}'s default of "
            f"{default}: an ack may shorten a mute and never lengthen one -- a "
            f"longer mute is a ticket, not an operator decision"
        )
    return requested


def gauntlet_iteration(iteration: int, every: int = GAUNTLET_EVERY) -> bool:
    """Return whether a gauntlet runs at the end of `iteration`.

    ``(i + 1) % every == 0``, so at cadence 4 the gauntlets are iterations 3,
    7, 11 -- the *last* iteration of each block, which is the one whose EMA is
    worth measuring.
    """
    if every < 1:
        raise ValueError(f"gauntlet cadence must be >= 1, got {every!r}")
    return (iteration + 1) % every == 0
