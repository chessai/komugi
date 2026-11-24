"""Startup refuses to advance past an iteration that checkpointed and never closed.

The review's witness (CP-01) is a root in the state a kill actually leaves: one
published trainer checkpoint whose metadata says ``iteration: 0``, an empty
journal, and no ``PAUSED``. ``resume_point`` reads that checkpoint as "iteration
0 completed", and until this ticket the loop entered iteration 1 -- playing
games with, and training from, a net whose boundary had never been measured. The
damage is permanent rather than momentary: ``fingerprint_series`` is unmutable
and its completeness check walks every iteration, so the resumed root pays a
full iteration, pauses on the hole it created, and pauses again at every later
boundary.

The witness is inverted here into an assertion about the **correct** behaviour,
and the assertion that carries it is the one about a function that was *not*
called: ``run_selfplay`` is replaced by a sentinel, and the campaign must stop
before it. A test that only asserted "``CampaignPaused`` was raised" would pass
on an implementation that played a whole iteration first.

Each test plants its root directly rather than killing a live campaign. That is
not a shortcut -- the state a kill leaves is exactly "checkpoint present,
campaign record absent", and planting it is how the *precise* window gets
tested rather than whichever point in it a race happened to land on (the same
argument ``kom022_rig`` makes for planted fingerprint series).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, NoReturn

import pytest

from komugi.campaign import loop as campaign_loop
from komugi.campaign.boundary import CAMPAIGN_MARKER
from komugi.campaign.factory import DEVICE_ENV, NET_ENV, SMALL_NET, new_net
from komugi.campaign.loop import (
    boundary_committed,
    checkpoint_dir,
    ema_dir,
    journal_path,
)
from komugi.campaign.pause import (
    CRASH_CLASS,
    INTERRUPT_CLASS,
    PAUSE_NAME,
    TRIPWIRE_CLASS,
    CampaignPaused,
    PauseError,
    PauseRecord,
    ack_path,
    apply_startup_state,
    read_pause,
    write_json_durably,
    write_pause,
)
from komugi.campaign.pinned import LOSS_SPIKE, UNFINISHED_BOUNDARY
from komugi.telemetry.journal import write_record
from komugi.telemetry.records import IterationMetrics
from komugi.train.checkpoint import save_checkpoint
from komugi.train.ema import EmaError

import campaign_rig as rig

STEPS_PER_ITERATION: Final[int] = 2
"""The stride the planted checkpoint's step is a multiple of.

Two, matching the review's witness: iteration 0 files at step 2, so the
step-division fallback in ``resume_point`` and its metadata path agree about
which iteration the file publishes.
"""

CHECKPOINT_STEP: Final[int] = 2
"""``(0 + 1) * STEPS_PER_ITERATION`` -- where iteration 0's checkpoint lands."""

ABANDONED_STEP: Final[int] = 4
"""``(1 + 1) * STEPS_PER_ITERATION`` -- iteration 1's, the step the rollback names.

The number the documented recipe substitutes into its two ``rm`` globs, and the
one the ``PAUSED`` carries as ``written_at_step``. Spelled out rather than
computed so that a test asserting on the *message* and a test performing the
*removal* cannot drift apart.
"""


class _SentinelReached(RuntimeError):
    """Self-play was entered. Raised so the positive control stops there."""


class _SelfPlaySentinel:
    """Stands in for ``run_selfplay`` and records whether it was ever called."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        self.calls.append(args)
        raise _SentinelReached("run_selfplay was entered")


def _plant_checkpoint(directory: Path, iteration: int, role: str) -> Path:
    """Publish `iteration`'s checkpoint into `directory` at the step it files at.

    ``(iteration + 1) * STEPS_PER_ITERATION``, the loop's own filing rule, so
    ``resume_point``'s metadata path and its step-division fallback agree about
    every planted file -- and so does
    ``loop._newest_committed_checkpoint``, which has only the step to go on.
    """
    state = {key: value.detach().clone() for key, value in new_net().state_dict().items()}
    return save_checkpoint(
        directory,
        (iteration + 1) * STEPS_PER_ITERATION,
        state,
        {},
        {"iteration": iteration, "role": role},
    )


def _plant_published_checkpoint(root: Path) -> Path:
    """Publish iteration 0's trainer checkpoint into `root`, and nothing else.

    No journal, no EMA, no ``PAUSED``: the whole point is a root whose only
    evidence about iteration 0 is the checkpoint. The EMA is left absent
    deliberately -- and since KOM-211 it must *stay* absent across a refused
    launch, because the refusal is asked before ``reconcile_ema`` runs at all
    (``test_a_refused_root_is_not_written_to``).
    """
    return _plant_checkpoint(checkpoint_dir(root), 0, "train")


def _bookless(*args: Any, **kwargs: Any) -> None:
    """Stand in for ``book_for_iteration``: this generation plays no book."""
    return None


def _commit_boundary(root: Path, iteration: int) -> None:
    """Write `iteration`'s campaign record: the boundary's own commit point."""
    write_record(
        journal_path(root),
        IterationMetrics(
            iteration=iteration,
            games=2,
            draw_rate=0.0,
            avg_game_len=20.0,
            policy_entropy=0.0,
            extras={CAMPAIGN_MARKER: 1.0},
        ),
    )


@pytest.fixture
def sentinel(monkeypatch: pytest.MonkeyPatch) -> _SelfPlaySentinel:
    """A campaign whose self-play is a tripwire, on the tiny CPU net."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(DEVICE_ENV, rig.CPU_DEVICE)
    stub = _SelfPlaySentinel()
    monkeypatch.setattr(campaign_loop, "run_selfplay", stub)
    return stub


def _run(root: Path, *, iterations: int) -> list[Any]:
    """Call ``run_campaign`` over `root` at this module's tiny sizes."""
    result: list[Any] = campaign_loop.run_campaign(
        rig.campaign_config(
            root, iterations, steps_per_iteration=STEPS_PER_ITERATION
        ),
        rig.tiny_selfplay_config(),
        orchestrator_overrides=rig.orchestrator_overrides(),
        inference=rig.inference_config(),
    )
    return result


def test_a_published_checkpoint_without_its_boundary_pauses_instead_of_advancing(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """CP-01, inverted: the resume stops at the hole instead of stepping over it.

    ``iterations=2``, so iteration 1 is genuinely requested and the sentinel is
    genuinely reachable -- the pause has to be what stops it, not an empty
    ``range``.
    """
    root = tmp_path / "root"
    published = _plant_published_checkpoint(root)

    with pytest.raises(CampaignPaused) as raised:
        _run(root, iterations=2)

    assert not sentinel.calls, (
        f"run_selfplay was entered {len(sentinel.calls)} time(s): the campaign "
        f"advanced into iteration 1 although iteration 0's boundary never ran"
    )
    pause = read_pause(root)
    assert pause is not None, "no PAUSED marker was written; a watcher sees nothing"
    assert pause.paused_at_iteration == 0, (
        f"PAUSED names iteration {pause.paused_at_iteration}; the unfinished "
        f"boundary is iteration 0's, not the one that would have followed it"
    )
    assert pause.tripwire == UNFINISHED_BOUNDARY, (
        f"PAUSED carries tripwire {pause.tripwire!r}: the ack/mute protocol and "
        f"every watcher key off this name"
    )
    assert pause.pause_class == CRASH_CLASS
    assert str(published) == pause.newest_checkpoint, (
        "PAUSED does not name the checkpoint whose boundary is missing"
    )
    assert "campaign record" in pause.detail and "iteration 0" in pause.detail
    assert "iteration 0" in str(raised.value)

    assert not boundary_committed(root, 0), (
        "a campaign record for iteration 0 appeared: the recovery path "
        "synthesised the boundary it was supposed to refuse to invent"
    )


def test_an_interrupt_marker_does_not_bypass_the_check(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """A cleared ``INTERRUPT`` says the stop was deliberate, not that it was clean.

    ``apply_startup_state`` archives an interrupt marker and returns "running",
    which is correct about the *marker* and says nothing about what the Ctrl-C
    interrupted. If the boundary check were sited inside that call's pause
    handling it would be skipped on exactly this path -- the most common way a
    campaign is stopped by hand.
    """
    root = tmp_path / "root"
    _plant_published_checkpoint(root)
    write_pause(
        root,
        PauseRecord(
            pause_class=INTERRUPT_CLASS,
            paused_at_iteration=0,
            tripwire="interrupt",
            detail="KeyboardInterrupt: ",
        ),
    )

    with pytest.raises(CampaignPaused):
        _run(root, iterations=2)

    assert not sentinel.calls, (
        "the interrupt marker was cleared and the campaign then advanced into "
        "iteration 1 across an uncommitted boundary"
    )
    assert (root / "PAUSED.0.cleared").exists(), (
        "the interrupt marker was not archived; this test would then be "
        "asserting about the interrupt pause rather than about a new one"
    )
    pause = read_pause(root)
    assert pause is not None and pause.tripwire == UNFINISHED_BOUNDARY, (
        f"after the interrupt was cleared the root holds {pause!r}: the "
        f"unfinished boundary must pause it again"
    )


def test_the_final_requested_iteration_is_also_checked(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """``iterations == i + 1``: the loop body never runs, and the check still fires.

    The tempting implementation puts the check at the top of the loop body,
    where a campaign whose requested iterations are already "done" skips it
    entirely and returns ``[]`` -- reporting a completed campaign whose last
    iteration was never judged. That root is exactly the one an operator reads
    as finished, so it is the worst place to be silent.
    """
    root = tmp_path / "root"
    _plant_published_checkpoint(root)

    with pytest.raises(CampaignPaused):
        _run(root, iterations=1)

    assert not sentinel.calls
    pause = read_pause(root)
    assert pause is not None and pause.tripwire == UNFINISHED_BOUNDARY, (
        "a campaign whose requested iterations are all 'complete' returned "
        "quietly although its last iteration never closed"
    )


def test_a_committed_boundary_resumes_normally(
    tmp_path: Path, sentinel: _SelfPlaySentinel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control: with iteration 0's record present, the resume runs.

    Without this the whole file is satisfied by a check that refuses every
    resume, which would be a far worse defect than the one it fixes. The book
    build is stubbed out (it is a bookless generation here anyway) so that the
    only thing between startup and self-play is the check under test, and the
    sentinel raises rather than returning so no games are played.
    """
    root = tmp_path / "root"
    _plant_published_checkpoint(root)
    _commit_boundary(root, 0)
    monkeypatch.setattr(campaign_loop, "book_for_iteration", _bookless)

    with pytest.raises(_SentinelReached):
        _run(root, iterations=2)

    assert len(sentinel.calls) == 1, (
        f"self-play was entered {len(sentinel.calls)} time(s): a root whose "
        f"iteration 0 is committed must resume into iteration 1"
    )
    assert read_pause(root) is None, (
        f"a committed boundary was paused anyway: {read_pause(root)!r}"
    )


def test_the_tripwire_cannot_be_acked_away(tmp_path: Path) -> None:
    """An ack is a resume, and there is nothing here to resume onto.

    ``UNFINISHED_BOUNDARY`` is in ``UNMUTABLE``, so ``apply_startup_state``
    refuses the ack outright rather than clearing the pause -- the recovery is a
    fork or rollback to ``i - 1``, which no ``ACK`` file can express.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    write_pause(
        root,
        PauseRecord(
            pause_class=CRASH_CLASS,
            paused_at_iteration=3,
            tripwire=UNFINISHED_BOUNDARY,
            detail="planted",
        ),
    )
    write_json_durably(
        ack_path(root), {"acked_iteration": 3, "tripwire": UNFINISHED_BOUNDARY}
    )

    with pytest.raises(PauseError, match="cannot be acked"):
        apply_startup_state(root)

    pause = read_pause(root)
    assert pause is not None and pause.tripwire == UNFINISHED_BOUNDARY, (
        "the ack cleared an unfinished boundary; only a fork or rollback can"
    )


def test_the_planted_root_really_would_have_advanced(tmp_path: Path) -> None:
    """The premise, asserted rather than assumed: this root resumes at iteration 1.

    Every test above rests on ``resume_point`` reading the planted checkpoint as
    "iteration 0 completed". If a future change made it read the root as fresh,
    the pauses above would still fire for some other reason and nothing would
    say so.
    """
    root = tmp_path / "root"
    with rig.small_net_env():
        published = _plant_published_checkpoint(root)
        point = campaign_loop.resume_point(
            rig.campaign_config(root, 2, steps_per_iteration=STEPS_PER_ITERATION)
        )

    assert point.iteration == 1, (
        f"resume_point starts at {point.iteration}; the witness needs a root "
        f"that would have advanced into iteration 1"
    )
    assert point.checkpoint is not None
    assert Path(point.checkpoint.path) == published
    assert point.checkpoint.step == CHECKPOINT_STEP
    assert not boundary_committed(root, 0)


def test_a_fresh_root_is_not_refused(tmp_path: Path) -> None:
    """No checkpoint, no iteration to have left unfinished.

    The check reads "iteration ``point.iteration - 1``", which is ``-1`` on a
    fresh root; a version that dropped the guard would refuse every campaign's
    first launch.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    with rig.small_net_env():
        point = campaign_loop.resume_point(
            rig.campaign_config(root, 2, steps_per_iteration=STEPS_PER_ITERATION)
        )
        campaign_loop._refuse_unfinished_boundary(
            root,
            rig.campaign_config(root, 2, steps_per_iteration=STEPS_PER_ITERATION),
            point,
            "",
        )

    assert point.checkpoint is None
    assert read_pause(root) is None, "a fresh root was paused for having no history"


def test_a_refused_root_is_not_written_to(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """The refusal comes before the repair: a paused root does *no* work at all.

    KOM-180 impl review F1. ``reconcile_ema`` used to run first, so a root inside
    this window had ``EMA_i`` published for the very iteration the next statement
    declared unjudged -- on every refused launch -- and the documented rollback
    (delete ``theta_i``) then died in ``reconcile_ema`` with "EMA is ahead of the
    trainer ... Two campaigns sharing one root is the likely cause": an error
    naming a cause that did not happen.

    The assertion is about the *whole tree*, not just the EMA: the only new file
    a refused launch may leave is ``PAUSED``, and a second refused launch (which
    stops in ``apply_startup_state``, on the marker the first one wrote) may
    leave nothing at all.
    """
    root = tmp_path / "root"
    _plant_published_checkpoint(root)
    before = rig.stamp_tree(root)

    with pytest.raises(CampaignPaused):
        _run(root, iterations=2)

    assert not sentinel.calls
    assert not list(ema_dir(root).glob("*.pt")), (
        f"the refused launch published an EMA: "
        f"{rig.describe(list(ema_dir(root).glob('*.pt')))}. The iteration that "
        f"EMA belongs to is the one this launch declared unjudged"
    )
    assert not journal_path(root).exists(), (
        "the refused launch appended to the journal; a refused root's evidence "
        "must be exactly what the crash left"
    )
    after = rig.stamp_tree(root)
    assert set(after) - set(before) == {PAUSE_NAME}, (
        f"a refused launch created {sorted(set(after) - set(before))}: the only "
        f"file it may write is {PAUSE_NAME}"
    )
    assert {name: stamp for name, stamp in after.items() if name != PAUSE_NAME} == (
        before
    ), "a refused launch rewrote a file that was already on disk"

    pause = read_pause(root)
    assert pause is not None
    assert pause.last_good_checkpoint == "", (
        f"PAUSED names {pause.last_good_checkpoint!r} as the last good "
        f"checkpoint, but no iteration of this root is committed: the honest "
        f"answer is 'start a fresh root'"
    )

    with pytest.raises(CampaignPaused):
        _run(root, iterations=2)

    assert rig.stamp_tree(root) == after, (
        "a second refused launch changed the root; a refusal must be idempotent "
        "down to the bytes"
    )
    assert not sentinel.calls


def test_the_documented_rollback_actually_resumes(
    tmp_path: Path, sentinel: _SelfPlaySentinel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery the pause prescribes, performed exactly, and the campaign runs.

    The review's second finding: the recipe was documented and never executed by
    a test, and it did not work. The root is the two-checkpoint witness -- a
    committed iteration 0 and an uncommitted iteration 1 -- and at the merged
    revision the *first* launch never even reached the refusal: ``reconcile_ema``
    ran first and raised ``EmaError`` on two train checkpoints beside an empty
    ``ema/``, so the operator's first artefact was a migration complaint about a
    root the loop itself had just created.

    Then EXACTLY the documented steps, and nothing else: remove ``PAUSED``,
    remove the abandoned attempt's trainer checkpoint, remove its EMA sibling
    (which the glob finds nothing to remove here -- the kill landed before that
    publish), relaunch. Iteration 1 is re-run from the committed iteration 0 and
    the sentinel is reached exactly once.
    """
    root = tmp_path / "root"
    committed = _plant_checkpoint(checkpoint_dir(root), 0, "train")
    _commit_boundary(root, 0)
    abandoned = _plant_checkpoint(checkpoint_dir(root), 1, "train")
    monkeypatch.setattr(campaign_loop, "book_for_iteration", _bookless)

    with pytest.raises(CampaignPaused):
        _run(root, iterations=2)

    assert not sentinel.calls
    pause = read_pause(root)
    assert pause is not None and pause.tripwire == UNFINISHED_BOUNDARY

    # EXACTLY the documented steps, in the documented order.
    (root / PAUSE_NAME).unlink()
    for stale in checkpoint_dir(root).glob(f"checkpoint-{ABANDONED_STEP:09d}-*.pt"):
        stale.unlink()
    for stale in ema_dir(root).glob(f"checkpoint-{ABANDONED_STEP:09d}-*.pt"):
        stale.unlink()

    assert not abandoned.exists(), (
        f"the documented glob did not match the abandoned checkpoint "
        f"{abandoned.name}: the recipe an operator pastes must name the file the "
        f"loop actually wrote"
    )

    with pytest.raises(_SentinelReached):
        _run(root, iterations=2)

    assert len(sentinel.calls) == 1, (
        f"self-play was entered {len(sentinel.calls)} time(s) after the "
        f"documented rollback: the recovery an operator is handed must actually "
        f"resume the campaign"
    )
    assert committed.exists(), "the rollback removed the checkpoint it rolls back to"
    assert read_pause(root) is None, (
        f"the relaunch paused again: {read_pause(root)!r}. The rollback removed "
        f"the unfinished attempt, so iteration 1 is now the iteration to run"
    )


def test_the_rollback_must_remove_the_ema_sibling(
    tmp_path: Path, sentinel: _SelfPlaySentinel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the recipe's third line exists: without it the relaunch does not start.

    The kill that lands *after* the EMA publish leaves ``EMA_1`` beside
    ``theta_1``. An operator who removes only ``PAUSED`` and the trainer
    checkpoint -- the recovery as KOM-180 documented it -- gets
    ``EmaError: EMA is ahead of the trainer``, an error whose text blames two
    campaigns sharing one root. Removing the EMA sibling as well is what makes
    the recipe terminate, and it is the line whose omission would otherwise be
    *silent* on a root where the EMA happens not to be ahead: the re-run would
    serve the abandoned attempt's smoothed net instead of ``EMA_0``.
    """
    root = tmp_path / "root"
    _plant_checkpoint(checkpoint_dir(root), 0, "train")
    _commit_boundary(root, 0)
    abandoned = _plant_checkpoint(checkpoint_dir(root), 1, "train")
    abandoned_ema = _plant_checkpoint(ema_dir(root), 1, "ema")
    monkeypatch.setattr(campaign_loop, "book_for_iteration", _bookless)

    with pytest.raises(CampaignPaused) as refused:
        _run(root, iterations=2)

    pause = read_pause(root)
    assert pause is not None and pause.tripwire == UNFINISHED_BOUNDARY
    assert pause.newest_ema == str(abandoned_ema), (
        f"PAUSED records {pause.newest_ema!r} as the newest EMA; it must be the "
        f"file that was already on disk ({abandoned_ema}), read rather than "
        f"published by the launch that refused"
    )
    assert str(ema_dir(root)) in str(refused.value), (
        f"the refusal does not name the EMA file the operator must remove:\n"
        f"{refused.value}"
    )

    # The incomplete rollback: PAUSED and theta_1 only.
    (root / PAUSE_NAME).unlink()
    abandoned.unlink()

    with pytest.raises(EmaError, match="ahead of the trainer"):
        _run(root, iterations=2)
    assert not sentinel.calls

    # The documented one: the EMA sibling goes too.
    abandoned_ema.unlink()

    with pytest.raises(_SentinelReached):
        _run(root, iterations=2)
    assert len(sentinel.calls) == 1


def test_the_pause_names_the_checkpoint_to_roll_back_to(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """``last_good_checkpoint`` is the newest checkpoint whose iteration IS closed.

    KOM-180 impl review F4: both crash-pause sites in ``run_campaign`` fill this
    field and the startup refusal left it empty -- although it is the artefact the
    rollback refers to, and derivable from the directory and the journal. The
    refused checkpoint itself is never the answer.
    """
    root = tmp_path / "root"
    committed = _plant_checkpoint(checkpoint_dir(root), 0, "train")
    _commit_boundary(root, 0)
    abandoned = _plant_checkpoint(checkpoint_dir(root), 1, "train")

    with pytest.raises(CampaignPaused):
        _run(root, iterations=2)

    assert not sentinel.calls
    pause = read_pause(root)
    assert pause is not None
    assert pause.last_good_checkpoint == str(committed), (
        f"PAUSED names {pause.last_good_checkpoint!r} as the last good "
        f"checkpoint; iteration 0 is the newest committed one, so the answer is "
        f"{str(committed)!r}"
    )
    assert pause.newest_checkpoint == str(abandoned)
    assert pause.written_at_step == 2 * STEPS_PER_ITERATION, (
        "written_at_step must be the abandoned attempt's own step: it is the "
        "step the rollback's two rm globs are built from"
    )


def test_the_pause_message_does_not_promise_a_refused_ack(
    tmp_path: Path, sentinel: _SelfPlaySentinel
) -> None:
    """What an operator is told to do must not be a thing that returns an error.

    KOM-180 impl review F3: ``apply_startup_state`` printed "To resume, write
    <root>/ACK ... the loop does the rest" into *every* pause, including this
    one -- and ``_validate_ack`` raises ``PauseError`` for every unmutable
    tripwire before anything is cleared. Both messages are checked: the one the
    refusal itself raises, and the one the *next* launch raises off the marker on
    disk, which is the message an operator who walked away actually reads.
    """
    root = tmp_path / "root"
    _plant_published_checkpoint(root)

    with pytest.raises(CampaignPaused) as refused:
        _run(root, iterations=2)
    with pytest.raises(CampaignPaused) as relaunched:
        _run(root, iterations=2)

    messages = (
        ("the refusal", str(refused.value)),
        ("the relaunch", str(relaunched.value)),
    )
    for label, text in messages:
        assert "cannot be acked" in text, (
            f"{label}'s message does not say the tripwire cannot be acked:\n{text}"
        )
        assert "To resume, write" not in text, (
            f"{label}'s message tells the operator to write an ACK, which "
            f"returns a PauseError:\n{text}"
        )

    instruction = str(relaunched.value)
    assert f"rm {root / PAUSE_NAME}" in instruction, (
        f"the message does not name the pause marker to remove:\n{instruction}"
    )
    assert f"checkpoint-{CHECKPOINT_STEP:09d}-*.pt" in instruction, (
        f"the message does not name the abandoned attempt's checkpoints (the "
        f"step is padded to nine digits on disk):\n{instruction}"
    )
    assert str(ema_dir(root)) in instruction, (
        f"the message does not name the EMA sibling, the one removal whose "
        f"omission no gate notices:\n{instruction}"
    )


def test_a_mutable_tripwire_still_gets_the_ack_instruction(tmp_path: Path) -> None:
    """The other half of the branch: an ackable pause still prints the ACK to write.

    The fix above is a *branch* on ``UNMUTABLE``, not a replacement, and the
    mutable path is the one an operator uses routinely -- a loss spike waved
    through. Without this, a change that printed the rollback for everything
    would pass every other test in this file.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    write_pause(
        root,
        PauseRecord(
            pause_class=TRIPWIRE_CLASS,
            paused_at_iteration=4,
            tripwire=LOSS_SPIKE,
            detail="planted",
        ),
    )

    with pytest.raises(CampaignPaused) as raised:
        apply_startup_state(root)

    text = str(raised.value)
    assert "To resume, write" in text and str(ack_path(root)) in text, (
        f"a mutable tripwire's pause no longer tells the operator how to ack "
        f"it:\n{text}"
    )
    assert "cannot be acked" not in text
