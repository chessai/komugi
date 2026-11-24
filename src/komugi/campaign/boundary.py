"""What the campaign checks at every iteration boundary, and how it stops.

KOM-022 §3a/§3b/§5. Everything here runs **between** iterations, in the
campaign process, over files that are already on disk -- which is what makes it
callable from a test over a planted root without a trainer, an optimiser or a
learning rate anywhere in the picture.

Three fingerprints per iteration, and the *count* is itself an invariant:

=============  ==========================================  =====================
role           tensors hashed                              meaning
=============  ==========================================  =====================
``train``      ``theta_i``'s parameters, as published      the trainer *learned*
``ema``        ``EMA_i``'s parameters, as published        the target moved
``self_play``  the parameters of the file ``CHECKPOINT_ENV``  what self-play served
               pointed at for this iteration
=============  ==========================================  =====================

Iteration 0 writes two of them: there is no served file at iteration 0, because
``CHECKPOINT_ENV`` is empty and the workers initialise freshly. So the journal
holds ``3N - 1`` fingerprints after ``N`` iterations, with no gaps, and
:func:`~komugi.telemetry.liveness.fingerprint_series_complete` gates that
separately -- because the run-length gate passes *vacuously* on a short series,
so one missing write would silently degrade it rather than fail it.

**What this detects, and what it does not.** All three fingerprints are
computed in the parent, from objects the parent controls: ``theta_i`` and
``EMA_i`` are in hand, and the ``self_play`` digest is of the file the loop
itself *named*. So the gate detects **the loop failing to advance the
pointer** -- a real and sufficient bug class, and the one a fingerprint series
in v1 would have caught. It does **not** detect a child that loaded a
*different* net: that child produces identical parent-side fingerprints. The
factory raises on an *unloadable* file, which covers unloadable and not stale;
closing that needs a digest field on the self-play ``RunSummary``, named in the
design as a successor ticket.

Escalation is deliberate and asymmetric. The hash gate, the cross-role
invariant and the EMA-aliasing check pause on the **first** occurrence and can
never be muted: a frozen self-play net is not a false alarm, and ``KL == 0`` on
every probe position is assumption-free. The band, the flat-trainer verdict,
the loss spike and the gauntlet floor **record first and pause on the
second** -- otherwise §10's own refutation criteria (which count *recorded
violations*, not pauses) could only ever be met by an operator who had already
decided the first observation was a false alarm.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import torch

from komugi.campaign import pinned
from komugi.campaign.pause import (
    CampaignPaused,
    Mute,
    PauseRecord,
    TRIPWIRE_CLASS,
    write_pause,
)
from komugi.telemetry.journal import read_records, write_record
from komugi.telemetry.liveness import (
    LivenessVerdict,
    displacement_ratio_within_band,
    fingerprint_series_complete,
    self_play_net_advanced,
    served_net_is_the_previous_ema,
)
from komugi.telemetry.records import (
    EMA,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)
from komugi.train.checkpoint import load_checkpoint, load_checkpoint_file
from komugi.train.digest import bn_digest, canonical_digest, digest_lo

CAMPAIGN_MARKER: Final[str] = "campaign_record"
"""The key that tells the campaign's ``IterationMetrics`` from the trainer's.

Both are written for the same iteration, deliberately (the trainer's docstring
invites the second record), so a consumer assuming one record per iteration
breaks -- and the discriminator has to be **explicit**. It is not ``games >
0``: an iteration whose self-play produced zero games still trains on the
window's older iterations and still writes a campaign record, so that
discriminator fails precisely during a self-play outage, which is when the
journal is being read in anger. The trainer's extras are built entirely from
its own ``loss_`` prefix, so this key is one it can never write.
"""

LOSS_TAIL_MEDIAN: Final[str] = "loss_tail_median"
BN_DIGEST_LO: Final[str] = "bn_digest_lo"
BN_DIGEST_ADVANCED: Final[str] = "bn_digest_advanced"
BAND_VIOLATION: Final[str] = "band_violation"
FLAT_TRAINER_FLAG: Final[str] = "flat_trainer"
SPIKE_FLAG: Final[str] = "loss_spike"
GAUNTLET_BREACH: Final[str] = "gauntlet_breach"
MUTED_FLAG: Final[str] = "tripwire_muted"

GAUNTLET_SCORE: Final[str] = "gauntlet_score"
GAUNTLET_ANCHOR_GEN: Final[str] = "gauntlet_anchor_gen"

GAUNTLET_RAN: Final[tuple[str, ...]] = (GAUNTLET_SCORE, GAUNTLET_ANCHOR_GEN)
"""Which extras mark a record whose iteration actually **ran** a gauntlet.

The floor's escalation counts consecutive breaches, and "consecutive" has to
mean *consecutive gauntlets* rather than *consecutive iterations*: at
:data:`~komugi.campaign.pinned.GAUNTLET_EVERY` = 4 there are three iterations
between gauntlets, each writing a campaign record that carries no
``gauntlet_breach`` -- so counting over every record resets the run at every
one of them and ``GAUNTLET_BREACH_PATIENCE = 2`` is unreachable in production.
A pre-registered tripwire that cannot fire is worse than none (KOM-022 review
F1).

Either key is enough. A gauntlet that **played** records
``gauntlet_anchor_gen`` (and ``gauntlet_score`` unless it was uninformative);
an **aborted** one -- the anchor did not verify, no games were played -- writes
neither, so it neither counts nor resets. That is the conservative reading: an
aborted gauntlet measured nothing, and a run of breaches interrupted by a
measurement that did not happen is still a run.
"""


@dataclass(frozen=True, slots=True)
class Fingerprints:
    """The digests one boundary produced, and the extras that ride along."""

    records: tuple[NetFingerprint, ...]
    extras: Mapping[str, float]
    bn_sha256: str
    bn_lo: float
    """The BatchNorm digest's low 52 bits -- what the journal actually carries.

    Threaded from boundary to boundary as ``previous_bn``, rather than the full
    hex digest, so that the in-process comparison and the one a restarted
    campaign makes against :func:`last_bn_digest_lo` are the *same*
    comparison. A restart that compared a different width of the same hash
    would report an advance it had not observed.
    """


@dataclass(slots=True)
class BoundaryOutcome:
    """What the boundary decided: what to record, and whether to stop.

    A tripwire that fires while muted still fills in ``verdicts`` and its
    ``extras`` flag and leaves ``tripwire`` unset: a mute suppresses the
    **pause**, never the **record**, which is what keeps §10's counting rules
    (over recorded violations, not over pauses) meaningful under an ack.
    """

    extras: dict[str, float] = field(default_factory=dict)
    verdicts: list[LivenessVerdict] = field(default_factory=list)
    tripwire: str | None = None
    detail: str = ""

    def fire(self, tripwire: str, detail: str, mutes: Mapping[str, Mute]) -> None:
        """Record a tripwire, and arm the pause unless this one is muted.

        A gate in :data:`~komugi.campaign.pinned.UNMUTABLE` ignores `mutes`
        entirely. That is defence in depth rather than the mechanism -- such a
        mute is refused when the ``ACK`` is validated and refused again when
        the mute directory is read, so the file cannot exist -- but "not
        mutable at any length" is the kind of claim that should hold at the
        point of use as well as at the two points of entry.
        """
        self.verdicts.append(LivenessVerdict(ok=False, reason=detail))
        if tripwire in mutes and tripwire not in pinned.UNMUTABLE:
            self.extras[MUTED_FLAG] = 1.0
            return
        if self.tripwire is None:
            self.tripwire = tripwire
            self.detail = detail


def campaign_records(
    journal: str | os.PathLike[str],
) -> tuple[IterationMetrics, ...]:
    """Return this root's **campaign-written** iteration records, in order.

    Filtered on :data:`CAMPAIGN_MARKER`, so the trainer's same-iteration record
    -- whose ``loss_total`` is the *last step's* total rather than a tail
    median -- can never reach a baseline that is supposed to be a median of
    tail medians. Two statistics with different variances compared by a fixed
    factor is a bound whose false-positive rate nobody can state.
    """
    read = read_records(journal, IterationMetrics, missing_ok=True)
    return tuple(
        record
        for record in read.records
        if record.extras.get(CAMPAIGN_MARKER, 0.0) == 1.0
    )


def loss_tail_median(
    history: Sequence[Mapping[str, float]],
    key: str,
    fraction: float = pinned.LOSS_TAIL_FRACTION,
) -> float:
    """Return the median of the last `fraction` of an iteration's step totals.

    Median, not mean, so one spike does not define the iteration; the tail, not
    the whole iteration, because the tail is what the next iteration continues
    from -- the same argument the trainer makes for recording its last step
    rather than its mean. At least one step is always included.
    """
    if not history:
        raise ValueError("cannot take a tail median of an empty step history")
    tail = max(1, int(round(len(history) * fraction)))
    return float(statistics.median(float(entry[key]) for entry in history[-tail:]))


def spike_baseline(
    records: Sequence[IterationMetrics], through_iteration: int
) -> float | None:
    """Return the median of the previous iterations' ``loss_tail_median``.

    ``None`` until :data:`~komugi.campaign.pinned.SPIKE_BASELINE_ITERATIONS`
    of them exist, which is what defers the spike bound to iteration 4 without
    a separate iteration test. Read back from the **journal**, so a resumed
    campaign has a baseline instead of starting with none and being blind for
    three iterations after every restart.
    """
    values = [
        record.extras[LOSS_TAIL_MEDIAN]
        for record in records
        if record.iteration < through_iteration and LOSS_TAIL_MEDIAN in record.extras
    ]
    tail = values[-pinned.SPIKE_BASELINE_ITERATIONS :]
    if len(tail) < pinned.SPIKE_BASELINE_ITERATIONS:
        return None
    return float(statistics.median(tail))


def consecutive_flag(
    records: Sequence[IterationMetrics],
    key: str,
    through_iteration: int,
    gate_keys: Sequence[str] = (),
) -> int:
    """Return how many *consecutive* recent iterations raised `key`.

    Counted from the journal rather than from memory, so the escalation
    survives a resume: "two consecutive band exits" must mean the same thing
    to a campaign that was restarted between them as to one that was not.

    `gate_keys` restricts the walk to the records whose iteration actually ran
    the gate in question -- a record carrying none of them is **skipped**,
    neither counting nor resetting. Empty (the default) is right for a gate
    that runs at every boundary, where every record is evidence either way; a
    *cadenced* gate needs it, because the iterations between its runs write
    records that are silent about it, and treating silence as "no violation"
    resets the run before it can ever reach its patience (see
    :data:`GAUNTLET_RAN`).
    """
    run = 0
    for record in sorted(records, key=lambda item: item.iteration):
        if record.iteration > through_iteration:
            break
        if gate_keys and not any(name in record.extras for name in gate_keys):
            continue
        run = run + 1 if record.extras.get(key, 0.0) == 1.0 else 0
    return run


def last_bn_digest_lo(
    journal: str | os.PathLike[str], before_iteration: int
) -> float | None:
    """Return the newest campaign record's ``bn_digest_lo`` below `before_iteration`.

    What a **restart** has instead of memory. ``previous_bn`` is re-initialised
    per ``run_campaign`` call, so without this the first boundary after any
    restart compares against nothing and reports ``bn_digest_advanced = 1.0``
    regardless of the truth -- the one iteration per restart at which a frozen
    BatchNorm series is invisible, which is precisely the iteration a
    post-crash post-mortem reads (KOM-022 review F6).

    Strictly *below* `before_iteration`, so a re-run iteration (the
    crash-before-checkpoint schedule) compares against its predecessor rather
    than against its own crashed attempt.
    """
    newest: IterationMetrics | None = None
    for record in campaign_records(journal):
        if record.iteration >= before_iteration or BN_DIGEST_LO not in record.extras:
            continue
        # `>=` and not `>`: the journal is append-only and a re-run iteration
        # writes a second record under the same number, so the *last* one
        # written is the one describing what is on disk (:func:`latest_fingerprints`
        # makes the same choice for the fingerprint series).
        if newest is None or record.iteration >= newest.iteration:
            newest = record
    if newest is None:
        return None
    return float(newest.extras[BN_DIGEST_LO])


def _fingerprint(
    iteration: int,
    role: str,
    state: Mapping[str, torch.Tensor],
    param_keys: Sequence[str],
) -> NetFingerprint:
    """Return one role's fingerprint over a loaded ``model_state``."""
    digest = canonical_digest(state, param_keys)
    return NetFingerprint(
        iteration=iteration,
        role=role,
        sha256=digest.sha256,
        param_count=digest.scalars,
    )


def record_fingerprints(
    journal: str | os.PathLike[str],
    iteration: int,
    checkpoints: str | os.PathLike[str],
    ema_directory: str | os.PathLike[str],
    served: str,
    param_keys: Sequence[str],
    previous_bn: float | None = None,
) -> Fingerprints:
    """Write this iteration's fingerprints and return them with their extras.

    `served` is the exact ``CHECKPOINT_ENV`` value this iteration ran under --
    empty at iteration 0, a file under ``<root>/ema`` afterwards, and read from
    the *environment* at the call site rather than from whatever local the loop
    meant to put there, so the digest describes the variable the workers were
    actually handed. It is digested as a *file*, which is why the
    parameter/buffer split is by name and works on a loaded state dict as well
    as on a module.

    `previous_bn` is the previous iteration's :data:`BN_DIGEST_LO`, or ``None``
    -- which is what a restart has. ``None`` is then resolved from the journal
    (:func:`last_bn_digest_lo`) rather than treated as "no comparison
    possible", because the alternative reports the BatchNorm series as advancing
    at the first boundary after every restart whatever it did.

    The BatchNorm digest rides along in ``extras`` and is deliberately **not** a
    fourth fingerprint role: a fourth role would have to be added to a shared
    closed enum, would silently break the ``3N - 1`` completeness count, and
    would then be one careless argument away from being handed to the
    run-length gate -- which is the defect this whole split exists to fix.
    """
    train = load_checkpoint(checkpoints)
    if train is None:
        raise RuntimeError(
            f"iteration {iteration} published no train checkpoint under "
            f"{checkpoints}: there is nothing to fingerprint, and a missing "
            f"fingerprint silently weakens the freeze gate"
        )
    ema = load_checkpoint(ema_directory)
    if ema is None:
        raise RuntimeError(
            f"iteration {iteration} has no EMA checkpoint under "
            f"{ema_directory}: the EMA is published after its source, so this "
            f"means the publish did not happen"
        )
    records = [
        _fingerprint(iteration, TRAIN, train.model_state, param_keys),
        _fingerprint(iteration, EMA, ema.model_state, param_keys),
    ]
    if served:
        source = Path(served)
        checkpoint = (
            load_checkpoint(source) if source.is_dir() else load_checkpoint_file(source)
        )
        if checkpoint is None:
            raise RuntimeError(
                f"iteration {iteration} served {served!r}, which does not "
                f"verify: what self-play played cannot be fingerprinted"
            )
        records.append(
            _fingerprint(iteration, SELF_PLAY, checkpoint.model_state, param_keys)
        )
    for record in records:
        write_record(journal, record)

    buffers = bn_digest(train.model_state, param_keys)
    low = digest_lo(buffers.sha256)
    previous = (
        previous_bn
        if previous_bn is not None
        else last_bn_digest_lo(journal, iteration)
    )
    extras = {
        BN_DIGEST_LO: low,
        BN_DIGEST_ADVANCED: (
            0.0 if previous is not None and previous == low else 1.0
        ),
    }
    return Fingerprints(
        records=tuple(records),
        extras=extras,
        bn_sha256=buffers.sha256,
        bn_lo=low,
    )


def latest_fingerprints(
    fingerprints: Sequence[NetFingerprint],
) -> tuple[NetFingerprint, ...]:
    """Return one record per ``(iteration, role)``: the **last** one written.

    An iteration can legitimately be fingerprinted twice, and exactly one
    schedule produces it: a campaign that died between sealing an iteration's
    shards and publishing its checkpoint **re-runs that iteration** on the next
    start (:func:`~komugi.campaign.loop.resume_point` starts after the newest
    checkpoint's iteration, and the loop clears the crashed attempt's shard
    tree first). The re-run trains a different net under the same iteration
    number -- legitimately, because the first attempt's net was never
    checkpointed, so nothing served it and nothing resumed from it -- and the
    journal is append-only, so both attempts' records survive. The newest is
    the one that describes what is on disk.

    Superseding here rather than in the gate is deliberate. "A re-run replaces
    its predecessor" is a fact about *this loop's* resume rule, not about
    fingerprint series in general, and
    :func:`~komugi.telemetry.liveness.self_play_net_advanced` is right to call
    two different nets at one iteration number a contradiction when nobody has
    told it otherwise. Without this, every crash-before-checkpoint resume would
    pause on an unmutable gate at its first boundary -- the resume path failing
    on the evidence that it worked.
    """
    newest: dict[tuple[int, str], NetFingerprint] = {}
    for fingerprint in fingerprints:
        newest[(fingerprint.iteration, fingerprint.role)] = fingerprint
    return tuple(newest.values())


def evaluate_liveness(
    fingerprints: Sequence[NetFingerprint],
    iteration: int,
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> None:
    """Run the three unmutable structural gates over the whole series.

    The hash gate is run for **each** of the three roles: a frozen ``train``
    series is the trainer that stopped learning, a frozen ``ema`` series is the
    target that stopped moving, and a frozen ``self_play`` series is v1's
    catastrophe verbatim. One function, three verdicts, and the role
    interpolated into the reason so an alert names what actually froze.

    A re-run iteration's superseded records are dropped first
    (:func:`latest_fingerprints`), so the gates see the campaign that is on
    disk rather than every attempt that ever wrote a line.
    """
    fingerprints = latest_fingerprints(fingerprints)
    complete = fingerprint_series_complete(fingerprints, iteration)
    outcome.verdicts.append(complete)
    if not complete.ok:
        outcome.fire(pinned.FINGERPRINT_SERIES, complete.reason, mutes)

    for role in (TRAIN, EMA, SELF_PLAY):
        if role == SELF_PLAY and iteration < 1:
            continue
        verdict = self_play_net_advanced(fingerprints, role=role)
        if verdict.ok:
            outcome.verdicts.append(verdict)
            continue
        outcome.fire(pinned.HASH_GATE, verdict.reason, mutes)

    served = served_net_is_the_previous_ema(fingerprints, iteration)
    if served.ok:
        outcome.verdicts.append(served)
    else:
        outcome.fire(pinned.CROSS_ROLE, served.reason, mutes)


def evaluate_band(
    ratio_series: Sequence[float],
    iteration: int,
    records: Sequence[IterationMetrics],
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> None:
    """Band-gate the displacement ratio, with the warm-up and the escalation.

    Skipped before :data:`~komugi.campaign.pinned.WARMUP_ITERATIONS`, where
    ``EMA_0 = theta_0`` still dominates and ``r`` is a statement about the
    initial condition rather than about the recursion.

    The bounds come from :func:`~komugi.campaign.pinned.band` and are never
    written here: KOM-149 re-pinned them on a measured centre without touching
    a line of this function, which is the point of keeping the numbers in one
    module.
    """
    if iteration < pinned.WARMUP_ITERATIONS or not ratio_series:
        return
    low, high = pinned.band()
    verdict = displacement_ratio_within_band(ratio_series, low, high)
    outcome.verdicts.append(verdict)
    if verdict.ok:
        return
    outcome.extras[BAND_VIOLATION] = 1.0
    # KOM-149 F8, on the carry-over across the re-pin: a root that ran under
    # the old band arrives here with two pieces of stale state, both harmless.
    # An ACK's standing mute is keyed by tripwire name, so it simply idles
    # while the band passes; and `consecutive_flag` walks `band_violation`
    # flags written by the *old* band, so a first exit after the re-pin can
    # report a long consecutive run. Neither can cause a pause that a fresh
    # root would not also take: the run only matters once the new band has
    # itself exited, and a single passing iteration breaks the chain.
    run = consecutive_flag(records, BAND_VIOLATION, iteration - 1) + 1
    if run >= pinned.BAND_PATIENCE:
        outcome.fire(
            pinned.DISPLACEMENT_BAND,
            f"{verdict.reason} (on {run} consecutive post-warm-up iterations)",
            mutes,
        )


def evaluate_flat_trainer(
    denominator: float,
    iteration: int,
    records: Sequence[IterationMetrics],
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> bool:
    """Return whether the trainer's own displacement is below the noise floor.

    Below :data:`~komugi.campaign.pinned.DENOMINATOR_FLOOR` the ratio is 0/0
    noise, so the band gate is **skipped** and a distinct verdict fires -- *the
    trainer is not moving* -- a third frozen-loop mode no existing gate covers.
    One sub-floor iteration is a converged or momentarily stalled trainer;
    three in a row is a trainer that has stopped, and that is a pause.
    """
    if denominator >= pinned.DENOMINATOR_FLOOR:
        return False
    outcome.extras[FLAT_TRAINER_FLAG] = 1.0
    run = consecutive_flag(records, FLAT_TRAINER_FLAG, iteration - 1) + 1
    detail = (
        f"the trainer's one-iteration displacement is {denominator:.3g}, below "
        f"the {pinned.DENOMINATOR_FLOOR:.3g} noise floor of a float32 "
        f"accumulation ({run} consecutive): the displacement ratio is "
        f"undefined and the trainer is not moving"
    )
    outcome.verdicts.append(LivenessVerdict(ok=False, reason=detail))
    if run >= pinned.FLAT_TRAINER_PATIENCE:
        outcome.fire(pinned.FLAT_TRAINER, detail, mutes)
    return True


def step_totals(history: Sequence[Mapping[str, float]], key: str) -> list[float]:
    """Return an iteration's per-step totals, as plain floats."""
    return [float(entry[key]) for entry in history]


def evaluate_non_finite_loss(
    totals: Sequence[float],
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> bool:
    """Return whether any of this iteration's step totals was non-finite.

    §5's second bound, and it is **not mutable**: a ``nan`` total is never a
    false alarm. It is evaluated *before* the tail median reaches ``extras``,
    because :class:`~komugi.telemetry.records.IterationMetrics` rejects
    non-finite extras -- so a campaign that computed the median first would
    die inside the telemetry write with a ``ValueError`` naming a key, and the
    blanket crash handler would then file it as ``class: crash`` on a
    checkpoint the trainer had already published. The tripwire says what
    happened; the crash marker would not.

    The trainer usually dies before this is reachable (§5's ordering hazard:
    ``training_iteration`` writes its own journal record, which rejects the
    non-finite extras, and never returns its history). This is the case where
    it does return -- a spike confined to a step the trainer did not record --
    and it is cheap enough to check unconditionally.
    """
    offending = [value for value in totals if not math.isfinite(value)]
    if not offending:
        return False
    outcome.fire(
        pinned.NON_FINITE_LOSS,
        f"{len(offending)} of iteration's {len(totals)} step totals are "
        f"non-finite (first: {offending[0]!r}): the trainer has diverged, and "
        f"the checkpoint it published this iteration is poisoned",
        mutes,
    )
    return True


def evaluate_spike(
    tail_median: float,
    iteration: int,
    records: Sequence[IterationMetrics],
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> None:
    """Fire the loss-spike bound against a baseline of previous tail medians."""
    baseline = spike_baseline(records, iteration)
    if baseline is None:
        return
    if tail_median <= pinned.SPIKE_FACTOR * baseline:
        return
    outcome.extras[SPIKE_FLAG] = 1.0
    outcome.fire(
        pinned.LOSS_SPIKE,
        f"iteration {iteration}'s tail-median loss {tail_median:.6g} is more "
        f"than {pinned.SPIKE_FACTOR:g}x the {baseline:.6g} baseline of the "
        f"previous {pinned.SPIKE_BASELINE_ITERATIONS} iterations: this is a "
        f"broken run, not a bad iteration",
        mutes,
    )


def evaluate_gauntlet(
    score: float,
    iteration: int,
    records: Sequence[IterationMetrics],
    outcome: BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> None:
    """Fire the pre-registered floor, escalating on the second consecutive breach.

    "Consecutive" counts consecutive **gauntlets**, not consecutive iterations:
    the run is walked over the records that actually ran one
    (:data:`GAUNTLET_RAN`). Counting over every record instead makes the
    tripwire unreachable at any cadence above 1, because the iterations in
    between write records that carry no breach flag and reset the run (KOM-022
    review F1).
    """
    if score >= pinned.GAUNTLET_FLOOR:
        return
    outcome.extras[GAUNTLET_BREACH] = 1.0
    run = (
        consecutive_flag(
            records, GAUNTLET_BREACH, iteration - 1, gate_keys=GAUNTLET_RAN
        )
        + 1
    )
    detail = (
        f"gauntlet score {score:.3f} is below the pre-registered floor "
        f"{pinned.GAUNTLET_FLOOR:.2f} ({run} consecutive breach(es) in this "
        f"anchor generation): there is evidence the current net is *worse* "
        f"than a frozen older anchor"
    )
    outcome.verdicts.append(LivenessVerdict(ok=False, reason=detail))
    if run >= pinned.GAUNTLET_BREACH_PATIENCE:
        outcome.fire(pinned.GAUNTLET_FLOOR_TRIPWIRE, detail, mutes)


def enforce(
    root: str | os.PathLike[str],
    outcome: BoundaryOutcome,
    iteration: int,
    results: Sequence[object],
    *,
    newest_checkpoint: str = "",
    newest_ema: str = "",
    last_good_checkpoint: str = "",
    written_at_step: int = 0,
) -> None:
    """Write ``PAUSED`` and raise, if this boundary armed a tripwire.

    Durable first, raise second: a watcher that restarts the campaign must
    find the marker already on disk, not race the exception's propagation.
    """
    if outcome.tripwire is None:
        return
    write_pause(
        root,
        PauseRecord(
            pause_class=TRIPWIRE_CLASS,
            paused_at_iteration=iteration,
            tripwire=outcome.tripwire,
            detail=outcome.detail,
            newest_checkpoint=newest_checkpoint,
            newest_ema=newest_ema,
            last_good_checkpoint=last_good_checkpoint,
            written_at_step=written_at_step,
        ),
    )
    raise CampaignPaused(
        results,
        f"campaign paused at iteration {iteration} on {outcome.tripwire}: "
        f"{outcome.detail}",
    )
