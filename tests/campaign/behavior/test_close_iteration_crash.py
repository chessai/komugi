"""A crash at the iteration BOUNDARY leaves a marker -- and never steals one.

KOM-129, opened from the KOM-025 wire-in review (F12). ``run_campaign`` turned a
crashing *training step* into a ``PAUSED`` marker and re-raised, but the call
that closes the iteration -- ``_close_iteration``: the EMA publish, the
fingerprints, the probe, the gauntlet, the gates and the campaign's own
``IterationMetrics`` write -- sat outside every handler. An ordinary
``Exception`` raised in there fell through to the outer ``except BaseException``,
which marks only non-``Exception``, so the iteration ended with **neither** a
journalled record **nor** a marker: the largest telemetry-loss surface in the
file, and invisible by construction -- the evidence that it happened is exactly
what is missing.

Two claims, and the second is why the first is not free:

* **T1** a plain ``RuntimeError`` from inside the close (here the journal write
  itself, which is the failure mode the ticket names first) propagates *and*
  leaves ``class: crash`` naming the site;
* **T2** a ``CampaignPaused`` from the close's own ``enforce`` step keeps the
  ``class: tripwire`` marker ``enforce`` already wrote. A crash handler that did
  not exclude it would overwrite every auto-pause in the system with
  ``tripwire: crash`` -- the operator would be told "something threw" instead of
  "the loss diverged", and the ack/mute protocol, which keys off the tripwire
  name, would no longer have a name to key off.

The rig is ``test_tripwires``' idiom: self-play and the training step stood in
for, everything between them the production code path. :func:`_schema_row` is
imported from that module rather than copied because it is a fixture of the real
:data:`~komugi.selfplay.records.SCHEMA` -- two copies would drift the day a field
is added, and the copy that drifted would fail here for a reason that has nothing
to do with this file's subject. (Importing a sibling test module is this repo's
existing idiom; see ``tests/selfplay/behavior/test_book_build_cost.py``.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, NoReturn

import campaign_rig
import pytest
import test_tripwires
import torch

from komugi.campaign import gauntlet as gauntlet_module
from komugi.campaign import pinned as pinned_module
from komugi.campaign.gauntlet import GauntletAborted, GauntletLost, GauntletStalled
from komugi.campaign.pause import (
    CRASH_CLASS,
    CRASH_TRIPWIRE,
    TRIPWIRE_CLASS,
    CampaignPaused,
    apply_startup_state,
    read_pause,
)
from komugi.campaign.pinned import NON_FINITE_LOSS
from komugi.selfplay.orchestrate import RunSummary
from komugi.selfplay.records import SCHEMA
from komugi.train.checkpoint import list_checkpoints, save_checkpoint
from komugi.train.shards import ShardWriter

CLOSE_WALL_S: Final[float] = 300.0
"""Wall for a campaign whose two expensive halves are stood in for.

No game is played and no optimiser step is taken, so a correct implementation
finishes in seconds; the wall exists so that one which reached the real
orchestrator fails *this* test rather than hanging ``ci/check.sh``.
"""

SITE_PREFIX: Final[str] = "close_iteration: "
"""What the boundary's crash marker prefixes its ``detail`` with.

The whole point of the prefix: two crash sites, one ``tripwire: crash``, and an
operator reading the file must be able to tell a trainer that died from a
boundary that died -- they ask different first questions. Asserted as a *prefix*
rather than a substring so that a marker which merely mentions the phrase
somewhere in an exception message cannot satisfy it.
"""

FINITE_HISTORY: Final[tuple[dict[str, float], ...]] = (
    {"total": 1.0},
    {"total": 0.9},
)
"""A healthy step history: nothing here arms a tripwire."""

DIVERGED_HISTORY: Final[tuple[dict[str, float], ...]] = (
    {"total": 1.0},
    {"total": float("nan")},
)
"""§5 bound 2, and the real ordering hazard: the checkpoint is published and
*then* the step total turns out to be non-finite."""

STALLED_GAUNTLET: Final[str] = "planted: a gauntlet shard ran past its deadline"
"""What the stalled-gauntlet stand-in raises, to be found in the marker."""


def _every_iteration_plays_a_gauntlet(iteration: int, every: int = 0) -> bool:
    """Stand in for the cadence so iteration 0 reaches the gauntlet.

    The production cadence is 4, so the first real gauntlet is iteration 3 --
    three extra full iterations of boundary work to reach a call this test
    replaces anyway. The *path* under test is everything after that call:
    ``_run_gauntlet``'s handler, ``_close_iteration``, and ``run_campaign``'s
    boundary crash handler. Which iteration it runs at is not part of it, and
    ``tests/campaign/unit/test_pinned.py`` owns the cadence itself.
    """
    return True


def _stalled_gauntlet(*args: object, **kwargs: object) -> NoReturn:
    """``run_gauntlet`` for a shard that is alive and never returning."""
    raise GauntletStalled(STALLED_GAUNTLET)


def _install_fast_iteration(
    monkeypatch: pytest.MonkeyPatch,
    loop_module: Any,
    root: Path,
    history: tuple[dict[str, float], ...],
) -> list[int]:
    """Stand in for self-play and the training step; count the iterations.

    Everything *between* them stays real -- the window build, the EMA
    reconcile, the fingerprints, the probe, the gates, ``enforce`` and the
    journal write are the production paths, which is the only way a test of the
    boundary's crash handling can be about the boundary. Replaced:

    * **self-play**, by four rows written through the real ``ShardWriter`` at
      the real ``SCHEMA`` (four rows of one game is below ``MIN_ENTRIES``, so
      every generation is bookless and no book replay enters the picture);
    * **the training step**, by a net that genuinely advances -- republishing
      identical weights would trip the freeze gate by itself -- and a
      checkpoint published at the loop's own global step, which is what makes
      "the crash marker names the checkpoint this iteration published" a claim
      with a subject.

    Returns the list the self-play stand-in appends to per call.
    """
    calls: list[int] = []

    def selfplay(*args: object, **kwargs: object) -> RunSummary:
        """Write this iteration's shards, as the orchestrator would have."""
        iteration = len(calls)
        calls.append(iteration)
        directory = loop_module.shard_root(root, iteration) / "worker_0_r0"
        with ShardWriter(directory, SCHEMA, samples_per_shard=4) as writer:
            for index in range(4):
                writer.append(test_tripwires._schema_row(index))
        return RunSummary(
            games_completed=1,
            rows_written=4,
            shard_dirs=(str(directory),),
            restarts=0,
            worker_incarnations=(0,),
        )

    def train(
        net: Any, source: Any, step: Any, **kwargs: Any
    ) -> list[dict[str, float]]:
        """Move every parameter, publish at ``base + steps``, report `history`."""
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.mul_(test_tripwires.PARAMETER_GAIN).add_(
                    test_tripwires.PARAMETER_SHIFT
                )
        save_checkpoint(
            kwargs["checkpoint_dir"],
            int(kwargs["global_step_base"]) + int(kwargs["steps"]),
            dict(net.state_dict()),
            step.optimizer.state_dict(),
            {"iteration": int(kwargs["iteration"]), "role": "train"},
        )
        return [dict(entry) for entry in history]

    monkeypatch.setattr(loop_module, "run_selfplay", selfplay)
    monkeypatch.setattr(loop_module, "training_iteration", train)
    return calls


def _walled_campaign(root: Path, iterations: int, label: str) -> Any:
    """Run ``run_campaign`` over `root` behind the rig's wall."""

    def call() -> object:
        with campaign_rig.small_net_env():
            return campaign_rig.campaign_loop().run_campaign(
                campaign_rig.campaign_config(root, iterations),
                campaign_rig.tiny_selfplay_config(),
                orchestrator_overrides=campaign_rig.orchestrator_overrides(),
                inference=campaign_rig.inference_config(),
            )

    return campaign_rig.walled(call, CLOSE_WALL_S, label)


def test_a_crash_inside_close_iteration_leaves_a_crash_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1: the boundary's own crash is marked, at the boundary, and re-raised.

    The failure planted is the ticket's first example and the cheapest real one:
    the journal write at step 6 of ``_close_iteration`` raises. By then the
    iteration has published its checkpoint and its EMA, so the root is in
    exactly the state that makes a blind restart dangerous -- and, before
    KOM-129, exactly the state that carried no evidence at all: the campaign's
    ``IterationMetrics`` record is the thing that failed to be written, and the
    outer handler marks only ``KeyboardInterrupt``/``SystemExit``.

    ``loop.write_record`` is the loop module's own binding, so patching it
    reaches the boundary's record write and nothing else: the fingerprint
    records are written through ``komugi.campaign.boundary``'s own import and
    still land, which is what makes this a crash *mid*-close rather than a
    close that never started.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root, FINITE_HISTORY)

    def failing_write_record(*args: object, **kwargs: object) -> None:
        """The journal write a full disk (or a bug at the seam) fails on."""
        raise RuntimeError("planted: the boundary's journal write failed")

    monkeypatch.setattr(loop_module, "write_record", failing_write_record)

    run = _walled_campaign(root, 1, "close-iteration-crash")

    assert calls == [0], f"self-play ran for iterations {calls}, want exactly [0]"
    assert type(run.error) is RuntimeError, (
        f"the crash was swallowed or replaced: returned {run.results!r}, raised "
        f"{run.error!r}. A marker is an addition to the failure, never a "
        f"substitute for it"
    )
    assert "planted" in str(run.error)

    pause = read_pause(root)
    assert pause is not None, (
        "a crash inside _close_iteration left no PAUSED marker: the iteration "
        "has no journalled record either (its record write is what failed), so "
        "the campaign died leaving no trace of the iteration at all and a "
        "watcher would restart onto the checkpoint it just published"
    )
    assert pause.pause_class == CRASH_CLASS
    assert pause.tripwire == CRASH_TRIPWIRE
    assert pause.detail.startswith(SITE_PREFIX), (
        f"the marker does not name the site it was written at, so a trainer "
        f"crash and a boundary crash are indistinguishable on disk: "
        f"{pause.detail!r}"
    )
    assert "RuntimeError" in pause.detail, (
        f"the marker does not name the exception class: {pause.detail!r}"
    )
    assert "planted" in pause.detail
    assert pause.paused_at_iteration == 0
    assert pause.written_at_step == campaign_rig.STEPS_PER_ITERATION

    # The checkpoint field is the one an operator acts on, and the dangerous
    # answer is a *stale* one: the file the crash may have poisoned is the one
    # this iteration published, and naming the previous generation would send a
    # post-mortem to the wrong bytes. `list_checkpoints` is newest-first.
    refs = list_checkpoints(loop_module.checkpoint_dir(root))
    assert refs, "the stand-in trainer published no checkpoint; the test is vacuous"
    assert pause.newest_checkpoint == str(refs[0].path), (
        f"the marker names {pause.newest_checkpoint!r}, not the checkpoint "
        f"iteration 0 published ({refs[0].path})"
    )
    assert pause.last_good_checkpoint == "", (
        "iteration 0 resumed from nothing, so there is no last good checkpoint "
        f"to name: {pause.last_good_checkpoint!r}"
    )

    # And the marker is a refusal, not a note: the next start stops on it.
    with pytest.raises(CampaignPaused, match=CRASH_TRIPWIRE):
        apply_startup_state(root)


def test_a_tripwire_pause_keeps_its_own_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: the crash handler does not clobber the pause ``enforce`` just wrote.

    ``_close_iteration``'s last step raises ``CampaignPaused`` *after* writing
    ``PAUSED`` -- ``class: tripwire``, named for the gate that fired. That raise
    travels through the new handler, and ``write_pause`` overwrites
    unconditionally, so a handler that marked every ``Exception`` would replace
    the diagnosis of every auto-pause in the system with ``tripwire: crash``:
    the operator loses the gate's name, and ``apply_startup_state`` refuses the
    ack (a crash-class pause is unackable by design), so the un-pause path
    stops matching the stop.

    The tripwire is §5 bound 2, driven the way ``test_tripwires`` drives it: a
    stand-in step that publishes its checkpoint and then reports a ``nan``.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root, DIVERGED_HISTORY)

    run = _walled_campaign(root, 1, "close-iteration-tripwire")

    assert calls == [0], f"self-play ran for iterations {calls}, want exactly [0]"
    assert isinstance(run.error, CampaignPaused), (
        f"a non-finite step total did not stop the campaign: returned "
        f"{run.results!r}, raised {run.error!r}"
    )

    pause = read_pause(root)
    assert pause is not None
    assert pause.pause_class == TRIPWIRE_CLASS, (
        f"the boundary's crash handler overwrote a tripwire pause with "
        f"class {pause.pause_class!r}: {pause.detail!r}"
    )
    assert pause.tripwire == NON_FINITE_LOSS, (
        f"want the gate's own name, got {pause.tripwire!r}: {pause.detail!r}"
    )
    assert not pause.detail.startswith(SITE_PREFIX), (
        f"the marker carries the crash site's prefix, so the tripwire's own "
        f"detail was replaced: {pause.detail!r}"
    )
    assert pause.paused_at_iteration == 0


def test_a_stalled_gauntlet_writes_the_boundary_crash_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3: the gauntlet supervisor's typed errors end as a marked crash stop.

    KOM-180's whole point is that a wedged gauntlet stops being a campaign that
    is still "running" days later; what makes that true is not the deadline but
    what happens to the error it raises. ``GauntletStalled`` and
    ``GauntletLost`` are **siblings** of ``GauntletAborted`` under
    ``RuntimeError``, so ``_run_gauntlet``'s ``except GauntletAborted`` -- which
    folds an unverifiable anchor into a liveness *verdict* and lets the
    iteration continue -- must not swallow them. An infrastructure failure is
    not a measurement: it has to reach ``run_campaign``'s boundary handler and
    leave a crash ``PAUSED`` naming its type, because the operator's first
    question ("which shard, and is the box sick?") is unanswerable from
    ``gauntlet aborted`` in a journal record.

    That was asserted nowhere (KOM-180 impl review F8): the existing gauntlet
    tests stop at ``run_gauntlet`` raising. This drives the real boundary --
    the real ``_anchor_for`` pins iteration 0's checkpoint as anchor gen 0, the
    real challenger is read off ``<root>/ema`` -- and replaces only the call
    that would spawn sixteen shard processes.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root, FINITE_HISTORY)
    monkeypatch.setattr(
        pinned_module, "gauntlet_iteration", _every_iteration_plays_a_gauntlet
    )
    # On the gauntlet module, which `_run_gauntlet` imports and calls through:
    # the seam under test is the *loop's* handling of what this raises.
    monkeypatch.setattr(gauntlet_module, "run_gauntlet", _stalled_gauntlet)

    run = _walled_campaign(root, 1, "close-iteration-gauntlet-stall")

    assert calls == [0], f"self-play ran for iterations {calls}, want exactly [0]"
    assert type(run.error) is GauntletStalled, (
        f"a stalled gauntlet did not stop the campaign as itself: returned "
        f"{run.results!r}, raised {run.error!r}. Folded into a verdict (or "
        f"re-typed) it would be a measurement, and the iteration would go on "
        f"to record a score for a gauntlet that never finished"
    )

    pause = read_pause(root)
    assert pause is not None, (
        "a stalled gauntlet left no PAUSED marker: the campaign died at its "
        "boundary with nothing on disk saying so, which is the state the "
        "supervisor exists to replace"
    )
    assert pause.pause_class == CRASH_CLASS, (
        f"want a crash-class pause for an infrastructure failure, got "
        f"{pause.pause_class!r}: {pause.detail!r}"
    )
    assert pause.tripwire == CRASH_TRIPWIRE
    assert pause.detail.startswith(SITE_PREFIX), (
        f"the marker does not name the boundary as the site: {pause.detail!r}"
    )
    assert GauntletStalled.__name__ in pause.detail, (
        f"the marker does not name the failure's type, so a wedged gauntlet "
        f"and a dead shard read identically on disk: {pause.detail!r}"
    )
    assert "planted" in pause.detail
    assert pause.paused_at_iteration == 0

    # The iteration is NOT committed: the gauntlet runs before the campaign
    # record, so a restart meets `unfinished_boundary` rather than advancing
    # past an iteration whose boundary never finished.
    assert not loop_module.boundary_committed(root, 0), (
        "the iteration recorded itself as committed despite dying in its own "
        "gauntlet"
    )

    # The class relationships the handler above depends on, pinned here rather
    # than left to inspection: make either of these a subclass of
    # `GauntletAborted` and the crash pause silently becomes a liveness verdict.
    assert issubclass(GauntletStalled, RuntimeError)
    assert issubclass(GauntletLost, RuntimeError)
    assert not issubclass(GauntletStalled, GauntletAborted)
    assert not issubclass(GauntletLost, GauntletAborted)
