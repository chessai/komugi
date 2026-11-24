"""The pause, the ack, and the mute: a stop that can be un-stopped, exactly once.

KOM-022 §5, acceptances #19, #20, #22, #30, #37, #44, #45 and #46. An auto-pause
whose un-pause path does not close is worse than no auto-pause -- the campaign
stops and *stays* stopped -- so the whole file lifecycle is pinned here rather
than left to a call site.

Every transition is driven through :func:`~komugi.campaign.pause.apply_startup_state`
directly, which is the single call ``run_campaign`` makes before it creates a
directory or spawns anything. The rest go through ``run_campaign`` itself:
two where a *correct* implementation raises before any work starts, and --
KOM-025 -- the two liveness gates, whose whole claim is that they are
**automated inside the loop** rather than merely correct over a planted root
(which is where ``test_liveness_gates`` exercises them). All of them run behind
``campaign_rig``'s wall so that a broken implementation fails the test in
bounded time instead of wedging the gate on a real self-play run, and the ones
that need several iterations stand in for self-play and the training step
(:func:`_install_fast_iteration`) so that "several iterations" costs seconds.

The crash windows the design tabulates are entered *on purpose*, through
``apply_startup_state``'s ``hook``: a window that cannot be entered deliberately
is a window nobody tests, and the 6-to-7 window is the one that destroyed rev
2's ack.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import campaign_rig
import pytest
import torch

from komugi.campaign import boundary
from komugi.campaign import pause as pause_module
from komugi.campaign.pause import (
    ACK_NAME,
    CRASH_CLASS,
    CRASH_TRIPWIRE,
    INTERRUPT_CLASS,
    TRIPWIRE_CLASS,
    AckRaceLost,
    CampaignPaused,
    Mute,
    PauseError,
    PauseRecord,
    ack_path,
    apply_startup_state,
    expire_mutes,
    live_mutes,
    mutes_dir,
    pause_path,
    read_mutes,
    read_pause,
    write_json_durably,
    write_pause,
)
from komugi.campaign.pinned import (
    DISPLACEMENT_BAND,
    EMA_PUBLISH_STALLED,
    HASH_GATE,
    LOSS_SPIKE,
    MUTE_DEFAULTS,
    NON_FINITE_LOSS,
)
from komugi.engine.observation import (
    CONTROL_EMPTY,
    CONTROL_SIZE,
    PLANE_MAP_VERSION,
)
from komugi.selfplay.orchestrate import RunSummary
from komugi.selfplay.records import SCHEMA
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics
from komugi.train.checkpoint import (
    list_checkpoints,
    load_checkpoint_file,
    save_checkpoint,
)
from komugi.train.shards import ShardWriter

PAUSED_AT: Final[int] = 12
"""The iteration Schedule B pauses at, kept from the design's own walkthrough."""

REFUSE_WALL_S: Final[float] = 120.0
"""Wall for a ``run_campaign`` call that must raise before it does any work.

A correct implementation raises in milliseconds; the wall exists so that an
implementation which reached ``run_selfplay`` fails *this test* rather than
hanging ``ci/check.sh`` on real games.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pause(
    root: Path,
    *,
    tripwire: str = LOSS_SPIKE,
    iteration: int = PAUSED_AT,
    pause_class: str = TRIPWIRE_CLASS,
) -> Path:
    """Write the ``PAUSED`` a tripwire would have written."""
    root.mkdir(parents=True, exist_ok=True)
    return write_pause(
        root,
        PauseRecord(
            pause_class=pause_class,
            paused_at_iteration=iteration,
            tripwire=tripwire,
            detail=f"planted {tripwire} at iteration {iteration}",
            newest_checkpoint="ckpt/checkpoint-000000104-abcdef0123456789.pt",
            last_good_checkpoint="ckpt/checkpoint-000000096-0123456789abcdef.pt",
            written_at_step=104,
        ),
    )


def _ack(
    root: Path,
    *,
    tripwire: str = LOSS_SPIKE,
    iteration: int = PAUSED_AT,
    mute_iterations: int | None = None,
) -> Path:
    """Write the ``ACK`` an operator would write beside a still-present pause."""
    payload: dict[str, Any] = {
        "acked_iteration": iteration,
        "tripwire": tripwire,
    }
    if mute_iterations is not None:
        payload["mute_iterations"] = mute_iterations
    write_json_durably(ack_path(root), payload)
    return ack_path(root)


def _names(root: Path) -> set[str]:
    """Return the names of every entry directly under `root`."""
    return {entry.name for entry in root.iterdir()}


def _mute_json(root: Path, tripwire: str) -> dict[str, Any]:
    """Return a mute file's parsed contents."""
    text = (mutes_dir(root) / f"{tripwire}.json").read_text(encoding="utf-8")
    parsed: dict[str, Any] = json.loads(text)
    return parsed


def _baseline_records(through: int) -> list[IterationMetrics]:
    """Three campaign records carrying a flat ``loss_tail_median`` baseline."""
    return [
        IterationMetrics(
            iteration=iteration,
            games=1,
            draw_rate=0.0,
            avg_game_len=0.0,
            policy_entropy=0.0,
            extras={
                boundary.CAMPAIGN_MARKER: 1.0,
                boundary.LOSS_TAIL_MEDIAN: 10.0,
            },
        )
        for iteration in range(through - 2, through + 1)
    ]


# ---------------------------------------------------------------------------
# acceptance #22 and #37 -- a paused root does no work at all
# ---------------------------------------------------------------------------


def test_paused_root_refuses_to_start(tmp_path: Path) -> None:
    """Acceptance #22: no directory created, nothing spawned, no game played.

    The check is the first statement of ``run_campaign``, before
    ``root.mkdir``: a supervisor's restart loop on a paused root must re-raise
    without doing work, or it burns a night re-entering a campaign that is
    stopped on purpose.
    """
    root = tmp_path / "root"
    _pause(root)
    before = _names(root)

    def call() -> object:
        with campaign_rig.small_net_env():
            return campaign_rig.campaign_loop().run_campaign(
                campaign_rig.campaign_config(root, 2),
                campaign_rig.tiny_selfplay_config(),
                orchestrator_overrides=campaign_rig.orchestrator_overrides(),
                inference=campaign_rig.inference_config(),
            )

    run = campaign_rig.walled(call, REFUSE_WALL_S, "paused-root")

    assert isinstance(run.error, CampaignPaused), (
        f"run_campaign on a paused root returned {run.results!r} / raised "
        f"{run.error!r}; it must raise CampaignPaused"
    )
    assert LOSS_SPIKE in str(run.error)
    assert ACK_NAME in str(run.error), "the message must tell the operator what to do"
    assert _names(root) == before, (
        "run_campaign touched the root before deciding it was paused"
    )
    assert not (root / "ckpt").exists()
    assert not (root / "ema").exists()


def test_pause_precedes_stale_shard_rmtree(tmp_path: Path) -> None:
    """Acceptance #37: the pause strictly precedes every destructive step.

    ``loop.py`` removes a crashed attempt's partial shard tree *before*
    self-play runs, so a pause evaluated at the orchestrator would destroy that
    tree and only then stop. Siting the check at the top of the loop body
    costs one word and makes the ordering a property rather than a hope.
    """
    root = tmp_path / "root"
    stale = root / "iter_1" / "shards" / "worker_0_r0"
    stale.mkdir(parents=True)
    (stale / "shard-000000.npz").write_bytes(b"partial output of a crashed attempt")
    _pause(root, iteration=0)

    def call() -> object:
        with campaign_rig.small_net_env():
            return campaign_rig.campaign_loop().run_campaign(
                campaign_rig.campaign_config(root, 3),
                campaign_rig.tiny_selfplay_config(),
                orchestrator_overrides=campaign_rig.orchestrator_overrides(),
                inference=campaign_rig.inference_config(),
            )

    run = campaign_rig.walled(call, REFUSE_WALL_S, "pause-before-rmtree")

    assert isinstance(run.error, CampaignPaused)
    assert stale.is_dir(), "the stale shard tree was removed before the pause stopped"
    assert (stale / "shard-000000.npz").read_bytes().startswith(b"partial")


def _schema_row(index: int) -> dict[str, torch.Tensor]:
    """One row of the real self-play :data:`SCHEMA`, at the real shapes.

    The campaign's own writer and the campaign's own schema, so the window the
    loop builds below is the window it builds in production; only the *content*
    is synthetic, and nothing here reads it.
    """
    target = torch.zeros(campaign_rig.NUM_ACTIONS, dtype=torch.float32)
    # FOUR legal actions, never one. :func:`~komugi.campaign.probe.sample_probe`
    # derives the probe's legal mask as ``policy_target > 0``, and a one-action
    # mask makes the masked softmax the point distribution for *any* net -- so
    # ``KL(pi_ema || pi_train)`` would be exactly 0.0 at every probe position
    # however far apart the two nets were, and the unmutable aliasing gate
    # would fire on every campaign this fixture drives past iteration 0. A
    # single-action target is also not a position a real search records.
    for offset in range(4):
        target[(index + offset) % 8] = 0.25
    return {
        "observation": torch.zeros(
            campaign_rig.IN_PLANES,
            campaign_rig.BOARD_H,
            campaign_rig.BOARD_W,
            dtype=torch.float32,
        ),
        "policy_target": target,
        "wdl_target": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "mlh_target": torch.tensor(10.0, dtype=torch.float32),
        "policy_kl": torch.tensor(0.1, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(0, dtype=torch.int64),
        "to_play": torch.tensor(index % 2, dtype=torch.int64),
        "ply": torch.tensor(index, dtype=torch.int64),
        "game_id": torch.tensor(1, dtype=torch.int64),
        "sims_used": torch.tensor(2, dtype=torch.int64),
        # KOM-002 §4. "The campaign's own writer and the campaign's own
        # schema" above is the whole point of this fixture, and the schema has
        # required this field since KOM-002: without it the row is rejected at
        # `writer.append`, before the loop under test runs at all.
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        # KOM-010, and required by ``writer.append`` for the same reason: the
        # campaign's own schema carries the ownership target, so a fixture
        # without it is not a row this pipeline can write.
        # KOM-014: the move the row's mover played. Required by the schema for
        # the same reason as the two fields above -- a fixture without it is not
        # a row this pipeline can write.
        "action": torch.tensor(index % campaign_rig.NUM_ACTIONS, dtype=torch.int64),
        "final_control": torch.full(
            (CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64
        ),
        # KOM-005b, and required by ``writer.append`` for the same reason as
        # every field above it. Distinguishable per row rather than zero-filled,
        # so a consumer that drops one of the three is visible. ``wdl_target``
        # above is a win, so the game was decisive: ``termination`` is 0.
        "termination": torch.tensor(0, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [
                0.5 - (index % 3) / 2.0,
                0.25 - (index % 5) / 4.0,
                1.0 - (index % 7) / 6.0,
            ],
            dtype=torch.float32,
        ),
        "draft_outcome": (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if index % 4 == 0
            else torch.zeros(3, dtype=torch.float32)
        ),
    }


def test_any_exception_writes_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #21: an ``Exception`` in the iteration body is a crash *marker*.

    This is the mitigation for §5's NaN ordering hazard, and it was executed by
    no test (review F4): the trainer publishes its checkpoint **before** it
    writes its journal record, and a non-finite loss dies inside that write --
    so the poisoned checkpoint is already the newest one and a watcher that
    restarted blindly would resume onto it. The handler converts any exception
    into ``PAUSED`` with ``class: crash`` and the last good checkpoint named,
    then re-raises: nothing is swallowed, and the root refuses to start again.

    Self-play and the training step are both replaced here, and only they: the
    loop's own sequencing -- the window, the net, the ``TrainStep``, the crash
    handler, the re-raise -- is the real one, and the shards the window reads
    are written through the real ``ShardWriter`` at the real schema. A test
    that had to play games to reach the handler would cost minutes to assert a
    property of five lines.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    shard_directory = loop_module.shard_root(root, 0) / "worker_0_r0"

    def fake_selfplay(*args: object, **kwargs: object) -> RunSummary:
        """Write iteration 0's shards, as the orchestrator would have."""
        with ShardWriter(shard_directory, SCHEMA, samples_per_shard=4) as writer:
            for index in range(4):
                writer.append(_schema_row(index))
        return RunSummary(
            games_completed=1,
            rows_written=4,
            shard_dirs=(str(shard_directory),),
            restarts=0,
            worker_incarnations=(0,),
        )

    def diverge(*args: object, **kwargs: object) -> object:
        """The exception a diverged step raises out of the telemetry write."""
        raise ValueError("planted: extras value total is not finite")

    monkeypatch.setattr(loop_module, "run_selfplay", fake_selfplay)
    monkeypatch.setattr(loop_module, "training_iteration", diverge)

    def call() -> object:
        with campaign_rig.small_net_env():
            return loop_module.run_campaign(
                campaign_rig.campaign_config(root, 1),
                campaign_rig.tiny_selfplay_config(),
                orchestrator_overrides=campaign_rig.orchestrator_overrides(),
                inference=campaign_rig.inference_config(),
            )

    run = campaign_rig.walled(call, REFUSE_WALL_S, "crash-marker")

    assert isinstance(run.error, ValueError), (
        f"the crash handler swallowed or replaced the exception: {run.error!r}"
    )
    assert "planted" in str(run.error)

    pause = read_pause(root)
    assert pause is not None, (
        "an exception in the iteration body left no PAUSED marker: a watcher "
        "would restart onto a checkpoint the crash may have poisoned"
    )
    assert pause.pause_class == CRASH_CLASS
    assert pause.tripwire == CRASH_TRIPWIRE
    assert "ValueError" in pause.detail, (
        f"the marker does not name the exception class: {pause.detail!r}"
    )
    assert "planted" in pause.detail
    assert pause.paused_at_iteration == 0
    assert pause.written_at_step == campaign_rig.STEPS_PER_ITERATION

    # And the marker is a refusal, not a note: the next start stops on it.
    with pytest.raises(CampaignPaused, match=CRASH_TRIPWIRE):
        apply_startup_state(root)


def test_a_fired_tripwire_stops_the_next_iterations_selfplay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #18: the orchestrator is **never invoked** for iteration i+1.

    Asserted for a *fired* tripwire rather than a planted ``PAUSED`` file: the
    suite had the planted case twice over and the live one nowhere (review F4),
    and the two are different claims -- the planted one is about
    ``apply_startup_state``, this one is about ``enforce`` raising before the
    loop's next pass reaches ``run_selfplay``. What it is worth is the whole
    point of an auto-pause: the iteration after a divergence is not paid for.

    The tripwire is §5 bound 2 (a non-finite step total), fired by a training
    step that publishes its checkpoint and then reports a ``nan`` -- which is
    the real ordering hazard, not a contrivance: the trainer checkpoints
    *before* it writes its journal record.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls: list[int] = []

    def counting_selfplay(*args: object, **kwargs: object) -> RunSummary:
        """Write the iteration's shards and count the invocation."""
        iteration = len(calls)
        calls.append(iteration)
        directory = (
            loop_module.shard_root(root, iteration) / "worker_0_r0"
        )
        with ShardWriter(directory, SCHEMA, samples_per_shard=4) as writer:
            for index in range(4):
                writer.append(_schema_row(index))
        return RunSummary(
            games_completed=1,
            rows_written=4,
            shard_dirs=(str(directory),),
            restarts=0,
            worker_incarnations=(0,),
        )

    def diverged(
        net: Any, source: Any, step: Any, **kwargs: Any
    ) -> list[dict[str, float]]:
        """Publish the checkpoint, then report the step that diverged."""
        save_checkpoint(
            kwargs["checkpoint_dir"],
            int(kwargs["global_step_base"]) + int(kwargs["steps"]),
            dict(net.state_dict()),
            step.optimizer.state_dict(),
            {"iteration": int(kwargs["iteration"]), "role": "train"},
        )
        return [{"total": 1.0}, {"total": float("nan")}]

    monkeypatch.setattr(loop_module, "run_selfplay", counting_selfplay)
    monkeypatch.setattr(loop_module, "training_iteration", diverged)

    def call() -> object:
        with campaign_rig.small_net_env():
            return loop_module.run_campaign(
                campaign_rig.campaign_config(root, 2),
                campaign_rig.tiny_selfplay_config(),
                orchestrator_overrides=campaign_rig.orchestrator_overrides(),
                inference=campaign_rig.inference_config(),
            )

    run = campaign_rig.walled(call, REFUSE_WALL_S, "spike-before-selfplay")

    assert isinstance(run.error, CampaignPaused), (
        f"a non-finite step total did not stop the campaign: {run.error!r}"
    )
    pause = read_pause(root)
    assert pause is not None
    assert pause.tripwire == NON_FINITE_LOSS
    assert pause.pause_class == TRIPWIRE_CLASS
    assert pause.paused_at_iteration == 0

    assert calls == [0], (
        f"the orchestrator was invoked {len(calls)} time(s) for a campaign "
        f"that armed a tripwire at iteration 0: the pause must precede "
        f"iteration 1's self-play, which is the cost it exists to avoid"
    )
    assert not (root / "iter_1").exists(), (
        "iteration 1 left a directory behind; it never ran"
    )


# ---------------------------------------------------------------------------
# KOM-025 -- the liveness gates, automated, inside run_campaign
# ---------------------------------------------------------------------------

PARAMETER_GAIN: Final[float] = 1.25
PARAMETER_SHIFT: Final[float] = 0.01
"""How far the stand-in trainer moves every parameter, per iteration.

A *large*, deterministic step, and both a gain and an offset. The boundary
measures ``KL(pi_ema || pi_train)`` on a probe set and pauses on the unmutable
aliasing gate when it is exactly zero at every position, so a stand-in whose
"training" left the policy distribution unchanged would pause these campaigns
on the wrong tripwire and pass them off as the gate under test. A pure additive
shift is the trap in particular: added uniformly to a layer's output it is a
softmax no-op.
"""

TRIPWIRE_WALL_S: Final[float] = 300.0
"""Wall for a campaign whose iteration body is stood in for.

No game is played and no optimiser step is taken, so a correct implementation
finishes in seconds; the wall exists so that one which reached the real
orchestrator fails these tests rather than hanging ``ci/check.sh``.
"""


def _install_fast_iteration(
    monkeypatch: pytest.MonkeyPatch, loop_module: Any, root: Path
) -> list[int]:
    """Stand in for self-play and the training step, and count the iterations.

    Everything *between* them stays real, and that is the point: the window
    build, the EMA reconcile, the fingerprint writes, the probe, the gates and
    ``enforce`` are the production code paths. What is replaced is the two
    expensive halves --

    * **self-play**, which spawns processes and plays games, by four rows
      written through the real :class:`~komugi.train.shards.ShardWriter` at the
      real :data:`~komugi.selfplay.records.SCHEMA`, so the window the loop
      builds is the window it builds in production and only the *content* is
      synthetic (four rows of one game is deliberately below
      :data:`~komugi.selfplay.book.MIN_ENTRIES`, so every generation is
      bookless and no book replay enters the picture);
    * **the training step**, by a net that genuinely advances and a checkpoint
      published at the loop's own global step. Advancing matters: a stand-in
      that republished identical weights would trip the freeze gate by itself,
      and every test below would pass for the wrong reason.

    Returns the list the self-play stand-in appends an entry to per call, so a
    test can assert which iterations were actually reached.
    """
    calls: list[int] = []

    def selfplay(*args: object, **kwargs: object) -> RunSummary:
        """Write this iteration's shards, as the orchestrator would have."""
        iteration = len(calls)
        calls.append(iteration)
        directory = loop_module.shard_root(root, iteration) / "worker_0_r0"
        with ShardWriter(directory, SCHEMA, samples_per_shard=4) as writer:
            for index in range(4):
                writer.append(_schema_row(index))
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
        """Move every parameter, then publish at ``base + steps``."""
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.mul_(PARAMETER_GAIN).add_(PARAMETER_SHIFT)
        save_checkpoint(
            kwargs["checkpoint_dir"],
            int(kwargs["global_step_base"]) + int(kwargs["steps"]),
            dict(net.state_dict()),
            step.optimizer.state_dict(),
            {"iteration": int(kwargs["iteration"]), "role": "train"},
        )
        return [{"total": 1.0}, {"total": 0.9}]

    monkeypatch.setattr(loop_module, "run_selfplay", selfplay)
    monkeypatch.setattr(loop_module, "training_iteration", train)
    return calls


def _paused_iterations_own_record(root: Path, iteration: int) -> IterationMetrics:
    """Return the campaign's journalled record for `iteration`, or fail loudly.

    The ordering claim this exists for (KOM-025 wire-in review, F2): the
    boundary writes its own :class:`IterationMetrics` **before**
    ``enforce`` raises, so a liveness stop leaves the paused iteration
    journalled and diagnosable. Swapping those two blocks loses that record on
    every stop -- its tail median, its BatchNorm digest, the trainer's extras --
    which is precisely the record a post-mortem of the pause needs, and the one
    a resumed campaign reads its spike baseline and previous BN digest back
    out of. Nothing before this read the journal after a pause.

    Filtered on :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER` and on the
    iteration the ``PAUSED`` file itself names, so a suite that stopped one
    boundary earlier or later cannot satisfy it with a neighbour's record.
    """
    journal = campaign_rig.campaign_loop().journal_path(root)
    read = read_records(journal, IterationMetrics, missing_ok=True)
    written = [
        record
        for record in read.records
        if record.iteration == iteration
        and record.extras.get(boundary.CAMPAIGN_MARKER, 0.0) == 1.0
    ]
    assert written, (
        f"the campaign paused at iteration {iteration} and left no campaign "
        f"record for it in {journal}; the journal holds "
        f"{[(r.iteration, sorted(r.extras)[:3]) for r in read.records]}. The "
        f"boundary must journal the iteration it is about to stop on before "
        f"`enforce` raises"
    )
    return written[-1]


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

    return campaign_rig.walled(call, TRIPWIRE_WALL_S, label)


def test_a_stalled_ema_publish_stops_the_campaign_at_its_own_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-025 / KOM-022 §2: the EMA invariant is enforced, not merely recorded.

    The published ``EMA_i`` is filed at ``theta_i``'s step, after it, so once
    the boundary's reconcile has returned the newest file under ``<root>/ema``
    must be this iteration's. Here the reconcile is made to no-op from the
    second boundary onwards -- the shape of any defect that leaves the EMA
    unpublished -- and the campaign must **stop**, durably, at the boundary
    where it happened.

    Sited at ``run_campaign`` rather than at ``evaluate_liveness``, which is
    where the existing liveness suite works: a gate that is correct over a
    planted root and unreachable from the loop is a gate that has never run.

    **Deleting the ``ema_publish_verdict`` call from ``_close_iteration``**
    still stops this campaign -- the EMA-role run-length gate catches the same
    stall, at the same boundary, with the same pause class -- so the
    discriminator is the tripwire *name*: this gate fires
    :data:`~komugi.campaign.pinned.EMA_PUBLISH_STALLED` and the run-length gate
    fires ``hash_gate``. That is a deterministic distinction; the previous
    ``"16" in pause.detail`` was a substring of a message carrying a hex digest,
    which the same digest could have satisfied by accident (KOM-025 wire-in
    review, F5). ``paused_at_iteration == 1`` and ``calls == [0, 1]`` pin the
    other half of the gate's value: the stop precedes a further whole iteration
    of self-play, training and checkpointing against the stale net.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root)

    real_reconcile = loop_module.reconcile_ema
    reconciles: list[int] = []

    def stalling(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        """Publish for the start-up call and iteration 0, then stop."""
        reconciles.append(len(reconciles))
        if len(reconciles) > 2:
            return ()
        published: tuple[int, ...] = real_reconcile(*args, **kwargs)
        return published

    monkeypatch.setattr(loop_module, "reconcile_ema", stalling)

    run = _walled_campaign(root, 2, "stalled-ema")

    assert isinstance(run.error, CampaignPaused), (
        f"a stalled EMA publish did not stop the campaign: returned "
        f"{run.results!r}, raised {run.error!r}"
    )
    pause = read_pause(root)
    assert pause is not None, (
        "the campaign stopped without leaving PAUSED; a watcher would restart "
        "it straight back onto the stale served net"
    )
    assert pause.pause_class == TRIPWIRE_CLASS
    assert pause.tripwire == EMA_PUBLISH_STALLED, (
        f"want the EMA publish gate's own name, got {pause.tripwire!r}: "
        f"{pause.detail}"
    )
    assert pause.paused_at_iteration == 1, (
        f"the stall happened at iteration 1's boundary and was reported at "
        f"{pause.paused_at_iteration}"
    )
    assert calls == [0, 1], (
        f"self-play ran for iterations {calls}; the pause must precede any "
        f"further iteration"
    )

    # F2: the stopped boundary is journalled, not lost to the raise. The
    # record must be the *paused* iteration's, carrying what the boundary
    # measured -- which is what a post-mortem and a resumed campaign read.
    stopped = _paused_iterations_own_record(root, pause.paused_at_iteration)
    assert boundary.BN_DIGEST_LO in stopped.extras, (
        f"iteration {pause.paused_at_iteration}'s record was written without "
        f"the boundary's own measurements: {sorted(stopped.extras)}"
    )
    # F6: and the gate's failing verdict survives the process, so stalls are
    # countable off the journal rather than only readable in a PAUSED file.
    assert stopped.extras.get(EMA_PUBLISH_STALLED) == 1.0, (
        f"the stalled EMA publish left no flag on the campaign record: "
        f"{sorted(stopped.extras)}"
    )


def test_a_frozen_selfplay_net_stops_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-025: v1's catastrophe, planted into a live loop and caught by it.

    v1 ran six iterations against a self-play net that never changed. Nothing
    raised -- games were played, losses fell, the log filled up -- and the only
    evidence was a hash nobody recorded. The defect
    :mod:`komugi.campaign.boundary` says this gate detects is precisely **the
    loop failing to advance the pointer**, so that is what is planted: the EMA
    is published correctly every iteration, and the loop is made to serve the
    *oldest* one forever.

    The campaign must stop at iteration 2, which is the first boundary at which
    a freeze is visible at all: one digest is not a repetition, so iteration 1
    -- which legitimately serves ``EMA_0`` -- passes.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root)

    real_load = loop_module.load_checkpoint
    ema_directory = loop_module.ema_dir(root)

    def stale(directory: Any, *args: Any, **kwargs: Any) -> Any:
        """Answer "the newest EMA" with the oldest one, and nothing else."""
        if Path(directory) == ema_directory:
            refs = list_checkpoints(directory)
            if refs:
                # `list_checkpoints` is newest-first, so [-1] is generation 0.
                return load_checkpoint_file(refs[-1].path)
        return real_load(directory, *args, **kwargs)

    monkeypatch.setattr(loop_module, "load_checkpoint", stale)

    run = _walled_campaign(root, 3, "frozen-selfplay-net")

    assert isinstance(run.error, CampaignPaused), (
        f"a frozen self-play net did not stop the campaign: returned "
        f"{run.results!r}, raised {run.error!r}"
    )
    pause = read_pause(root)
    assert pause is not None
    assert pause.pause_class == TRIPWIRE_CLASS
    assert pause.tripwire == HASH_GATE, (
        f"want the unmutable freeze gate, got {pause.tripwire!r}: {pause.detail}"
    )
    assert pause.paused_at_iteration == 2, (
        f"a repetition is first visible at iteration 2; the campaign paused at "
        f"{pause.paused_at_iteration}"
    )
    assert "self_play" in pause.detail, (
        f"the alert must name the role that froze, or an operator reads it as "
        f"a frozen trainer: {pause.detail!r}"
    )
    assert calls == [0, 1, 2], (
        f"self-play ran for iterations {calls}; a campaign that froze at "
        f"iteration 2 must not play iteration 3"
    )

    # F2: the frozen boundary is journalled before the raise. Without it the
    # one iteration an operator most needs to read -- the one the campaign
    # stopped on -- is the one iteration with no record.
    stopped = _paused_iterations_own_record(root, pause.paused_at_iteration)
    assert boundary.BN_DIGEST_LO in stopped.extras, (
        f"iteration {pause.paused_at_iteration}'s record carries none of the "
        f"boundary's own measurements: {sorted(stopped.extras)}"
    )
    assert stopped.extras.get(boundary.CAMPAIGN_MARKER) == 1.0

    # And the stop is a refusal, not a note: the next start re-raises on it.
    with pytest.raises(CampaignPaused, match=HASH_GATE):
        apply_startup_state(root)


def test_a_diverged_iteration_pauses_on_its_divergence_not_on_its_consequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-025 wire-in review F10: the gate ORDER at the boundary is load-bearing.

    ``_close_iteration`` runs the loss bounds first and the EMA/hash liveness
    gates after, deliberately: a diverged iteration must pause on its
    divergence rather than on a consequence of it. Nothing pinned that, and
    hoisting the EMA block above the loss bounds survives every other node --
    a campaign that stops is a campaign that stops, and only the tripwire name
    says *why*.

    So both are planted at one boundary: iteration 1's step totals go
    non-finite **and** its EMA publish no-ops. ``BoundaryOutcome.fire`` keeps
    the first tripwire, so the name on disk is the one whose gate ran first,
    and the operator reading it is told about the divergence -- the cause --
    rather than about the stale EMA, which is downstream of a trainer that
    produced NaNs.
    """
    root = tmp_path / "root"
    loop_module = campaign_rig.campaign_loop()
    calls = _install_fast_iteration(monkeypatch, loop_module, root)
    fast_train = loop_module.training_iteration
    real_reconcile = loop_module.reconcile_ema
    reconciles: list[int] = []

    def diverging(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
        """Train and publish as the stand-in does, then diverge from iteration 1."""
        history: list[dict[str, float]] = fast_train(*args, **kwargs)
        if int(kwargs["iteration"]) >= 1:
            return [{"total": 1.0}, {"total": float("nan")}]
        return history

    def stalling(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        """Publish for the start-up call and iteration 0's boundary, then stop."""
        reconciles.append(len(reconciles))
        if len(reconciles) > 2:
            return ()
        published: tuple[int, ...] = real_reconcile(*args, **kwargs)
        return published

    monkeypatch.setattr(loop_module, "training_iteration", diverging)
    monkeypatch.setattr(loop_module, "reconcile_ema", stalling)

    run = _walled_campaign(root, 2, "diverged-and-stalled")

    assert isinstance(run.error, CampaignPaused), (
        f"a diverged iteration did not stop the campaign: returned "
        f"{run.results!r}, raised {run.error!r}"
    )
    pause = read_pause(root)
    assert pause is not None
    assert pause.paused_at_iteration == 1, (
        f"both faults were planted at iteration 1's boundary; the campaign "
        f"paused at {pause.paused_at_iteration}"
    )
    assert pause.tripwire == NON_FINITE_LOSS, (
        f"the campaign stopped on {pause.tripwire!r}, not on the divergence "
        f"that caused it: {pause.detail}"
    )
    assert calls == [0, 1]


# ---------------------------------------------------------------------------
# acceptance #30 -- the whole file lifecycle
# ---------------------------------------------------------------------------


def test_ack_protocol_end_to_end(tmp_path: Path) -> None:
    """Acceptance #30: which files exist before and after, and what a re-run does.

    After the transition: ``PAUSED.<i>.acked`` and ``ACK.<i>.consumed`` are
    present, ``PAUSED`` and ``ACK`` are gone, and ``mutes/<tripwire>.json``
    exists. Archived rather than deleted, so every pause and its ack survive in
    the root for a post-mortem and nothing an operator or a supervisor does
    removes them.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)

    state = apply_startup_state(root)

    assert state.running
    assert state.applied is not None
    assert state.applied.tripwire == LOSS_SPIKE
    assert state.applied.until_iteration == PAUSED_AT + 1 + MUTE_DEFAULTS[LOSS_SPIKE]

    names = _names(root)
    assert f"PAUSED.{PAUSED_AT}.acked" in names
    assert f"ACK.{PAUSED_AT}.consumed" in names
    assert "PAUSED" not in names
    assert "ACK" not in names
    assert read_pause(root) is None
    assert _mute_json(root, LOSS_SPIKE)["until_iteration"] == PAUSED_AT + 3

    # A second run with the consumed ACK still on disk is an ordinary RUNNING
    # start: the consumed name no longer matches, so it cannot be re-applied.
    again = apply_startup_state(root)
    assert again.running
    assert again.applied is None, "the consumed ack was applied twice"
    assert again.mutes[LOSS_SPIKE].until_iteration == PAUSED_AT + 3


def test_an_ack_with_no_pause_is_refused(tmp_path: Path) -> None:
    """Acceptance #30: a stale ack outliving its pause is an error, not a mute.

    Treating it as benign would turn a leftover file into a permanent silence
    on the tripwire it names.
    """
    root = tmp_path / "root"
    root.mkdir()
    _ack(root)
    before = _names(root)

    with pytest.raises(PauseError, match="stale ack"):
        apply_startup_state(root)

    assert _names(root) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("acked_iteration", PAUSED_AT + 1), ("tripwire", DISPLACEMENT_BAND)],
)
def test_a_mismatched_ack_raises_and_changes_nothing(
    tmp_path: Path, field: str, value: object
) -> None:
    """Acceptance #30: an ack must be an ack of *this* pause.

    And the file it must match is still there to be matched against, which is
    the whole point of not deleting ``PAUSED`` first -- the contradiction rev 1
    could not resolve.
    """
    root = tmp_path / "root"
    _pause(root)
    payload: dict[str, Any] = {
        "acked_iteration": PAUSED_AT,
        "tripwire": LOSS_SPIKE,
    }
    payload[field] = value
    write_json_durably(ack_path(root), payload)
    before = campaign_rig.stamp_tree(root)

    with pytest.raises(PauseError):
        apply_startup_state(root)

    assert campaign_rig.stamp_tree(root) == before, (
        "a refused ack changed the root; nothing on disk may move"
    )
    assert read_pause(root) is not None
    assert not mutes_dir(root).exists()


def test_a_malformed_ack_raises_and_changes_nothing(tmp_path: Path) -> None:
    """``ACK`` is written by a human with an editor; a partial write is live.

    A truncated file must be a **refusal**, never a crash part-way through the
    transition.
    """
    root = tmp_path / "root"
    _pause(root)
    ack_path(root).write_text('{"acked_iteration": 12, "trip', encoding="utf-8")
    before = campaign_rig.stamp_tree(root)

    with pytest.raises(PauseError, match="not valid JSON"):
        apply_startup_state(root)

    assert campaign_rig.stamp_tree(root) == before


def test_hash_gate_is_not_mutable(tmp_path: Path) -> None:
    """Acceptance #20: an ``ACK`` naming the hash gate is refused.

    A frozen self-play net is never a false alarm; it is the failure this whole
    ticket exists to make impossible, so there is nothing here for an operator
    to wave through. The refusal happens at step 4, *before* step 5b, so an
    unmutable gate can never acquire a mute file.
    """
    root = tmp_path / "root"
    _pause(root, tripwire=HASH_GATE)
    _ack(root, tripwire=HASH_GATE)

    with pytest.raises(PauseError, match="cannot be acked"):
        apply_startup_state(root)

    assert read_pause(root) is not None
    assert ack_path(root).exists(), "the ack was consumed by a refusal"
    assert not mutes_dir(root).exists()


def test_a_crash_pause_is_cleared_by_an_ack_and_mutes_nothing(tmp_path: Path) -> None:
    """§5: a crash pause is cleared by an ack that buys no silence.

    There is no recurring statistic to mute, so step 5b writes **no** file --
    a distinct deliberate case rather than a failed write.
    """
    root = tmp_path / "root"
    _pause(root, tripwire=CRASH_TRIPWIRE, pause_class=CRASH_CLASS)
    _ack(root, tripwire=CRASH_TRIPWIRE)

    state = apply_startup_state(root)

    assert state.running
    assert state.applied is None
    assert state.mutes == {}
    assert not (mutes_dir(root) / f"{CRASH_TRIPWIRE}.json").exists()
    assert f"PAUSED.{PAUSED_AT}.acked" in _names(root)


def test_an_interrupt_marker_does_not_refuse(tmp_path: Path) -> None:
    """§5's table: a deliberate stop is not a fault.

    Ctrl-C, an ``ssh`` disconnect or a ``SIGTERM``-derived exit leaves a marker
    for the operator and a root that still starts -- the symmetric cost rev 1
    never priced. The marker is archived rather than deleted.
    """
    root = tmp_path / "root"
    _pause(root, tripwire="interrupt", pause_class=INTERRUPT_CLASS, iteration=3)

    state = apply_startup_state(root)

    assert state.running
    assert state.cleared_interrupt
    assert read_pause(root) is None
    assert "PAUSED.3.cleared" in _names(root)


# ---------------------------------------------------------------------------
# acceptance #19 / #44 / #45 -- the mute is a file, and it survives
# ---------------------------------------------------------------------------


def test_ack_unmutes_and_expires(tmp_path: Path) -> None:
    """Acceptance #19: the mute is asserted as a **file**, not as memory.

    ``<root>/mutes/loss_spike.json`` exists after the transition with
    ``from_iteration == i + 1`` and ``until_iteration == i + 3``; it suppresses
    the pause at ``i+1`` and ``i+2``, it does not at ``i+3``, and the loop
    archives it to ``.expired`` at the first boundary past it.

    The window opens at ``i+1`` because the boundary that paused is over: the
    campaign that reads this mute is the one that restarts at ``i+1``, so a
    window anchored at ``i`` would spend its first iteration on an iteration
    nobody runs again and a two-iteration mute would buy one (review F2).
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)
    apply_startup_state(root)

    mute = read_mutes(root)[LOSS_SPIKE]
    assert mute.from_iteration == PAUSED_AT + 1
    assert mute.until_iteration == PAUSED_AT + 3
    assert not mute.covers(PAUSED_AT)
    assert mute.covers(PAUSED_AT + 1) and mute.covers(PAUSED_AT + 2)
    assert not mute.covers(PAUSED_AT + 3)

    assert set(live_mutes(root, PAUSED_AT + 2)) == {LOSS_SPIKE}
    assert live_mutes(root, PAUSED_AT + 3) == {}

    assert expire_mutes(root, PAUSED_AT + 2) == (), "expired early"
    assert expire_mutes(root, PAUSED_AT + 3) == (LOSS_SPIKE,)
    assert not (mutes_dir(root) / f"{LOSS_SPIKE}.json").exists()
    assert f"{LOSS_SPIKE}.{PAUSED_AT + 3}.expired" in _names(mutes_dir(root))


def test_a_two_iteration_mute_covers_two_runnable_iterations(
    tmp_path: Path,
) -> None:
    """Review F2, as the design's own prose states it.

    §5/#19 says an acked spike at iteration ``i`` with ``n = 2`` means
    "iteration ``i+3`` can pause again". That is only true if the two iterations
    the mute buys are ``i+1`` and ``i+2`` -- the two the campaign is going to
    *run*. Shipped as ``from_iteration = i``, the window covered ``i`` and
    ``i+1``, ``i+2`` re-paused on the tripwire the operator had just acked, and
    the design's sentence described a mute one iteration longer than the one on
    disk.

    Red before the fix: the ``still_muted`` assertion at ``PAUSED_AT + 2``
    fires (that iteration was outside the shipped window), and so does the
    ``from_iteration`` equality above.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root, mute_iterations=2)
    applied = apply_startup_state(root).applied
    assert applied is not None

    covered = [
        iteration
        for iteration in range(PAUSED_AT, PAUSED_AT + 5)
        if applied.covers(iteration)
    ]
    assert covered == [PAUSED_AT + 1, PAUSED_AT + 2], (
        f"a two-iteration mute acked at {PAUSED_AT} covers {covered}; the "
        f"design's window is the two iterations the campaign will run next"
    )

    # And through the gate, which is where it matters: the still-true spike is
    # recorded at both muted iterations and pauses at the third.
    for iteration in (PAUSED_AT + 1, PAUSED_AT + 2):
        still_muted = boundary.BoundaryOutcome()
        boundary.evaluate_spike(
            100.0,
            iteration,
            _baseline_records(PAUSED_AT),
            still_muted,
            live_mutes(root, iteration),
        )
        assert still_muted.tripwire is None, (
            f"iteration {iteration} re-paused on the tripwire the operator "
            f"acked at {PAUSED_AT}"
        )
        assert still_muted.extras[boundary.SPIKE_FLAG] == 1.0

    armed = boundary.BoundaryOutcome()
    boundary.evaluate_spike(
        100.0,
        PAUSED_AT + 3,
        _baseline_records(PAUSED_AT),
        armed,
        live_mutes(root, PAUSED_AT + 3),
    )
    assert armed.tripwire == LOSS_SPIKE, "the mute outlived its own window"


def test_a_muted_tripwire_records_and_does_not_pause(tmp_path: Path) -> None:
    """A mute suppresses the **pause**, never the **record**.

    §10's counting rules run over *recorded violations*, not over pauses, so a
    mute that also suppressed the record would make the campaign's own
    refutation criteria unmeetable under an ack.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)
    apply_startup_state(root)

    muted = boundary.BoundaryOutcome()
    boundary.evaluate_spike(
        100.0,
        PAUSED_AT + 1,
        _baseline_records(PAUSED_AT),
        muted,
        live_mutes(root, PAUSED_AT + 1),
    )

    assert muted.extras[boundary.SPIKE_FLAG] == 1.0
    assert muted.extras[boundary.MUTED_FLAG] == 1.0
    assert muted.tripwire is None
    assert [v for v in muted.verdicts if not v.ok], "the violation was not recorded"

    armed = boundary.BoundaryOutcome()
    boundary.evaluate_spike(
        100.0,
        PAUSED_AT + 3,
        _baseline_records(PAUSED_AT),
        armed,
        live_mutes(root, PAUSED_AT + 3),
    )
    assert armed.tripwire == LOSS_SPIKE, "the mute outlived its own expiry"


def test_mute_survives_a_crash_after_the_pause_is_archived(tmp_path: Path) -> None:
    """Acceptance #44: the re-check's Schedule B, planted.

    The process dies **between step 6 and step 7** -- after ``PAUSED`` is
    archived, before the journal line. On restart the root reads ``RUNNING``,
    ``mutes/loss_spike.json`` is there with ``until_iteration == i + 3`` (the
    window is ``i+1 .. i+2``, review F2), the still-true tripwire at ``i + 1``
    is *recorded* and does not pause, and no second ``ACK`` is consumed.

    **Red without step 5b**: under rev 2's protocol the mute was a journal line
    written after the archive, so this crash left ``ACK.<i>.consumed`` and
    ``PAUSED.<i>.acked`` on disk claiming an un-pause that had bought nothing,
    and the campaign re-paused at ``i + 1`` on the tripwire the operator had
    just acked.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root, mute_iterations=2)

    class _Crash(RuntimeError):
        """The SIGKILL, expressed as something a test can catch."""

    def die_after_archive(step: str) -> None:
        """Stop the process in the 6-to-7 window and nowhere else."""
        if step == "archived":
            raise _Crash("kill -9 between step 6 and step 7")

    with pytest.raises(_Crash):
        apply_startup_state(root, hook=die_after_archive)

    names = _names(root)
    assert f"PAUSED.{PAUSED_AT}.acked" in names
    assert f"ACK.{PAUSED_AT}.consumed" in names
    assert "PAUSED" not in names and "ACK" not in names

    # The restart: step 1 reads <root>/mutes/ on the RUNNING path, which is the
    # whole point -- a mute must survive the very restart that has nothing left
    # to remind the loop it exists.
    restarted = apply_startup_state(root)
    assert restarted.running
    assert restarted.applied is None, "a second ack was consumed"
    assert restarted.mutes[LOSS_SPIKE].until_iteration == PAUSED_AT + 3
    assert _names(root) == names, "the restart changed the root"

    outcome = boundary.BoundaryOutcome()
    boundary.evaluate_spike(
        100.0,
        PAUSED_AT + 1,
        _baseline_records(PAUSED_AT),
        outcome,
        live_mutes(root, PAUSED_AT + 1),
    )
    assert outcome.tripwire is None, (
        "the campaign re-paused at i+1 on the tripwire the operator acked"
    )
    assert outcome.extras[boundary.SPIKE_FLAG] == 1.0


def test_the_mute_is_written_before_the_pause_is_archived(tmp_path: Path) -> None:
    """Acceptance #30, extended in rev 3: step 5b strictly precedes step 6.

    A rig that stops the process between the two must find the mute **already**
    on disk. Rev 2 wrote the mute as an epilogue at step 7, after the pause had
    been archived, which is the ordering the crash above destroys.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)

    observed: dict[str, bool] = {}

    def watch(step: str) -> None:
        """Record whether the mute file exists at each named step."""
        observed[step] = (mutes_dir(root) / f"{LOSS_SPIKE}.json").is_file()

    apply_startup_state(root, hook=watch)

    assert observed["claimed"] is False
    assert observed["muted"] is True, "the mute was not written at step 5b"
    assert observed["archived"] is True
    assert pause_path(root).exists() is False


def test_a_crash_between_the_claim_and_the_mute_asks_again(tmp_path: Path) -> None:
    """The 5-to-5b window: ``PAUSED`` present, ack consumed, no mute.

    The next start refuses at step 2, the operator writes a fresh ``ACK``, and
    5b writes exactly the content it would have written -- the mute file is
    keyed by *tripwire*, not by attempt, so the recovery is idempotent. The
    failure mode is "asks again", never "silently resumes".
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)

    class _Crash(RuntimeError):
        """The kill, in the 5-to-5b window."""

    def die_after_claim(step: str) -> None:
        """Stop immediately after the ack is claimed."""
        if step == "claimed":
            raise _Crash("kill -9 between step 5 and step 5b")

    with pytest.raises(_Crash):
        apply_startup_state(root, hook=die_after_claim)

    assert pause_path(root).exists()
    assert not mutes_dir(root).exists()

    with pytest.raises(CampaignPaused):
        apply_startup_state(root)

    _ack(root)
    recovered = apply_startup_state(root)
    assert recovered.applied is not None
    assert recovered.applied.until_iteration == PAUSED_AT + 3


def test_mute_expires_is_cleared_and_is_validated(tmp_path: Path) -> None:
    """Acceptance #45: expiry is content, the archive is tidiness, and invalid mutes refuse.

    A restart *inside* a live eight-iteration band mute keeps muting -- the
    re-check's no-crash variant, where rev 2 resumed with the mute forgotten
    and re-paused on a tripwire the operator had acked three iterations
    earlier.
    """
    root = tmp_path / "root"
    _pause(root, tripwire=DISPLACEMENT_BAND, iteration=20)
    # The ack must be an ack of *this* pause: `_ack`'s default iteration is
    # `PAUSED_AT`, and this pause is at 20, so the iteration is passed too --
    # otherwise `_validate_ack` refuses before a mute is ever written.
    _ack(root, tripwire=DISPLACEMENT_BAND, iteration=20)
    applied = apply_startup_state(root).applied
    assert applied is not None
    assert applied.until_iteration == 21 + MUTE_DEFAULTS[DISPLACEMENT_BAND]

    # A plain restart three iterations in.
    restarted = apply_startup_state(root)
    assert restarted.mutes[DISPLACEMENT_BAND].covers(23)
    assert set(live_mutes(root, 23)) == {DISPLACEMENT_BAND}

    # Expiry is a property of the file's content, not of an action: a mute
    # nobody ever cleared has already stopped muting. The window is 21..28
    # inclusive (review F2), so 29 is the first iteration past it.
    assert live_mutes(root, 29) == {}
    assert (mutes_dir(root) / f"{DISPLACEMENT_BAND}.json").is_file()
    assert expire_mutes(root, 29) == (DISPLACEMENT_BAND,)


@pytest.mark.parametrize(
    ("name", "payload", "match"),
    [
        (
            "no_such_gate",
            {
                "tripwire": "no_such_gate",
                "acked_iteration": 1,
                "from_iteration": 1,
                "until_iteration": 2,
            },
            "unknown tripwire",
        ),
        (
            LOSS_SPIKE,
            {
                "tripwire": LOSS_SPIKE,
                "acked_iteration": 1,
                "from_iteration": 1,
                "until_iteration": 99,
            },
            "may shorten a mute and never lengthen",
        ),
        (
            HASH_GATE,
            {
                "tripwire": HASH_GATE,
                "acked_iteration": 1,
                "from_iteration": 1,
                "until_iteration": 2,
            },
            "not mutable at any length",
        ),
    ],
)
def test_an_invalid_mute_file_refuses_at_startup(
    tmp_path: Path, name: str, payload: Mapping[str, Any], match: str
) -> None:
    """Acceptance #45: an unvalidatable mute must never silently suppress.

    Four cases, all raising and changing nothing: an unknown tripwire, a mute
    longer than its tripwire's default (the ``ACK`` rule enforced on the
    durable copy, which outlives the ``ACK`` that justified it), a gate that is
    unmutable at any length, and a truncated file. The last is covered
    separately below because it cannot be expressed as a payload.
    """
    root = tmp_path / "root"
    root.mkdir()
    write_json_durably(mutes_dir(root) / f"{name}.json", payload)
    before = campaign_rig.stamp_tree(root)

    with pytest.raises(PauseError, match=match):
        apply_startup_state(root)

    assert campaign_rig.stamp_tree(root) == before


def test_a_truncated_mute_file_refuses_at_startup(tmp_path: Path) -> None:
    """The fourth case of acceptance #45's list."""
    root = tmp_path / "root"
    mutes_dir(root).mkdir(parents=True)
    (mutes_dir(root) / f"{LOSS_SPIKE}.json").write_text(
        '{"tripwire": "loss_sp', encoding="utf-8"
    )

    with pytest.raises(PauseError, match="not valid JSON"):
        apply_startup_state(root)


def test_a_mute_keyed_under_the_wrong_name_refuses(tmp_path: Path) -> None:
    """One file per tripwire, or "which mutes are live" is not a listing."""
    root = tmp_path / "root"
    root.mkdir()
    write_json_durably(
        mutes_dir(root) / f"{DISPLACEMENT_BAND}.json",
        Mute(
            tripwire=LOSS_SPIKE,
            acked_iteration=1,
            from_iteration=1,
            until_iteration=2,
        ).as_json(),
    )

    with pytest.raises(PauseError, match="keyed"):
        apply_startup_state(root)


# ---------------------------------------------------------------------------
# acceptance #46 -- the race loser refuses without running
# ---------------------------------------------------------------------------


def test_ack_race_loser_refuses_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #46: exactly one process claims the ack; the loser refuses.

    The interleaving is injected at the documented race point: the loser
    validates, and the winner's ``os.replace`` lands *before* the loser's. The
    loser's call then raises ``FileNotFoundError`` on a source that no longer
    exists, and it refuses **without re-reading the state table** -- re-reading
    after the winner's step 6 would find ``RUNNING`` and run, which is two
    campaigns on one root.

    What this does *not* claim to close, and the design says so: two processes
    started on a root that is already ``RUNNING``. There is no ack to race on
    and §11.2's absent lockfile is unchanged.
    """
    root = tmp_path / "root"
    _pause(root)
    _ack(root)

    real_validate = pause_module._validate_ack

    def steal(ack: Mapping[str, Any], pause: PauseRecord, path: Path) -> int | None:
        """Validate, then let the *other* process's claim land first."""
        result = real_validate(ack, pause, path)
        os.replace(path, path.parent / f"{ACK_NAME}.{PAUSED_AT}.consumed")
        return result

    monkeypatch.setattr(pause_module, "_validate_ack", steal)

    with pytest.raises(AckRaceLost, match="claimed"):
        apply_startup_state(root)

    assert read_pause(root) is not None, "the loser archived a pause it did not claim"
    assert not mutes_dir(root).exists(), "the loser wrote a mute it did not buy"
    assert f"PAUSED.{PAUSED_AT}.acked" not in _names(root)


def test_the_mute_is_the_same_whichever_process_wins(tmp_path: Path) -> None:
    """Acceptance #46's tail: the mute's content is a function of the pause.

    Keyed by tripwire and computed once from the validated ``ACK`` and the
    per-tripwire table, so a re-ack **supersedes** rather than accumulates --
    which is exactly what makes the crash recovery above idempotent.
    """
    contents: list[dict[str, Any]] = []
    for name in ("first", "second"):
        root = tmp_path / name
        _pause(root)
        _ack(root, mute_iterations=1)
        apply_startup_state(root)
        contents.append(_mute_json(root, LOSS_SPIKE))

    assert contents[0] == contents[1]
    assert contents[0]["until_iteration"] == PAUSED_AT + 2, (
        "an ack may shorten a mute; the shortening was ignored"
    )
    assert contents[0]["from_iteration"] == PAUSED_AT + 1
