"""KOM-1114: ``CampaignConfig.window_floor`` excludes pre-fix iterations.

Unit tests over a fake root, in the shape of ``tests/campaign/unit/test_loop_unit.py``'s
``window_refs`` and iteration-boundary fixtures: a ``WindowConfig`` and a shard
tree built by hand (:func:`_write_iteration`), never a real orchestrator.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import torch

from komugi.campaign.factory import CHECKPOINT_ENV, SMALL_CONFIG
from komugi.campaign import boundary
from komugi.campaign import loop as campaign_loop
from komugi.campaign.game_stats import game_stats
from komugi.campaign.loop import (
    CampaignConfig,
    campaign_window_refs,
    checkpoint_dir,
    ema_dir,
    journal_path,
    shard_root,
    window_refs,
)
from komugi.config import SelfPlayConfig
from komugi.net import KomugiNet
from komugi.selfplay.orchestrate import RunSummary
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics
from komugi.terminations import TERMINATION_DECISIVE
from komugi.train.checkpoint import save_checkpoint
from komugi.train.digest import parameter_keys
from komugi.train.step import TOTAL_KEY
from komugi.train.window import IterationRef, WindowConfig


def _campaign(
    root: Path, *, steps_per_iteration: int = 4, window_floor: int | None = None
) -> CampaignConfig:
    """A minimal valid campaign rooted at `root`, varying `window_floor`."""
    return CampaignConfig(
        iterations=1,
        run_seed=11,
        root=str(root),
        steps_per_iteration=steps_per_iteration,
        batch_size=2,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
        window_floor=window_floor,
    )


def _write_iteration(root: Path, iteration: int, rows: int) -> None:
    """Give iteration `iteration` a worker directory holding `rows` samples.

    Mirrors ``test_loop_unit.py``'s helper of the same name: a real shard tree
    built with :class:`~komugi.train.shards.ShardWriter`, not a stand-in.
    """
    from komugi.train.shards import FieldSpec, Schema, ShardWriter

    worker_dir = shard_root(root, iteration) / "worker_0_r0"
    worker_dir.mkdir(parents=True)
    schema = Schema(fields=(FieldSpec(name="x", dtype="float32", shape=(2,)),))
    with ShardWriter(worker_dir, schema, samples_per_shard=4) as writer:
        for row in range(rows):
            writer.append({"x": torch.full((2,), float(iteration * 10 + row))})


# --------------------------------------------------------------------------
# campaign_window_refs
# --------------------------------------------------------------------------


def test_none_floor_matches_todays_unbounded_call(tmp_path: Path) -> None:
    """``floor=None`` is the exact call ``campaign_window_refs`` always made."""
    for iteration in range(10):
        _write_iteration(tmp_path, iteration, rows=2 + iteration)

    config = WindowConfig()
    through = 9
    plain = window_refs(tmp_path, through, retain_last=config.size, stop_after=config.size)

    def _shape(refs: tuple[IterationRef, ...]) -> list[tuple[int, int]]:
        return [(ref.iteration, len(ref.source)) for ref in refs]

    assert _shape(campaign_window_refs(tmp_path, through, config)) == _shape(plain)
    assert _shape(campaign_window_refs(tmp_path, through, config, floor=None)) == _shape(
        plain
    )


def test_a_floor_drops_refs_below_it_and_keeps_the_rest(tmp_path: Path) -> None:
    """A mid-window floor truncates the newest-first walk's tail, nothing else.

    Size-5 default window over iterations 0..20 keeps [16, 17, 18, 19, 20]
    unfiltered; a floor of 18 must drop 16 and 17 and keep 18, 19 and 20 with
    their row counts untouched.
    """
    for iteration in range(21):
        _write_iteration(tmp_path, iteration, rows=2 + iteration)

    config = WindowConfig()
    through = 20
    unfiltered = campaign_window_refs(tmp_path, through, config)
    assert [ref.iteration for ref in unfiltered] == [16, 17, 18, 19, 20]

    floored = campaign_window_refs(tmp_path, through, config, floor=18)
    assert [ref.iteration for ref in floored] == [18, 19, 20]
    assert [len(ref.source) for ref in floored] == [
        len(ref.source) for ref in unfiltered if ref.iteration >= 18
    ]


def test_the_first_post_floor_iteration_has_a_window_of_one(tmp_path: Path) -> None:
    """``through == floor``: nothing earlier than the floor can be a member."""
    for iteration in range(11):
        _write_iteration(tmp_path, iteration, rows=3)

    config = WindowConfig()
    refs = campaign_window_refs(tmp_path, 10, config, floor=10)

    assert [ref.iteration for ref in refs] == [10]


def test_a_future_floor_raises(tmp_path: Path) -> None:
    """A floor beyond the iteration being trained would train on nothing."""
    for iteration in range(5):
        _write_iteration(tmp_path, iteration, rows=3)

    config = WindowConfig()
    with pytest.raises(ValueError, match="window_floor"):
        campaign_window_refs(tmp_path, 4, config, floor=5)


def test_rebuilding_the_window_at_the_same_iteration_resumes_identically(
    tmp_path: Path,
) -> None:
    """Resume rebuilds the window at the iteration it stopped at; same refs.

    ``IterationRef.source`` holds a fresh reader each call, so the comparison
    is iteration numbers and row counts -- what the window's schedule and the
    samples drawn from it actually depend on -- rather than object identity.
    """
    for iteration in range(12):
        _write_iteration(tmp_path, iteration, rows=2 + iteration)

    config = WindowConfig()
    first = campaign_window_refs(tmp_path, 11, config, floor=7)
    second = campaign_window_refs(tmp_path, 11, config, floor=7)

    assert [ref.iteration for ref in first] == [ref.iteration for ref in second]
    assert [len(ref.source) for ref in first] == [len(ref.source) for ref in second]


# --------------------------------------------------------------------------
# the campaign record (extras[WINDOW_FLOOR])
# --------------------------------------------------------------------------

_BOUNDARY_STEPS: Final[int] = 8


def _publish_at(directory: Path, step: int) -> None:
    """Publish a verifying checkpoint of the small net at `step` under `directory`."""
    save_checkpoint(
        directory,
        step,
        KomugiNet(SMALL_CONFIG).state_dict(),
        {},
        {"iteration": step, "role": "ema"},
    )


def _close_a_planted_boundary(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    window_floor: int | None,
    window_oldest: int | None = None,
) -> Mapping[str, float]:
    """Run iteration 0's real boundary over a planted root; return its record.

    The window-floor twin of ``test_loop_unit.py``'s ``_close_a_planted_boundary``:
    same two stand-ins (a planted EMA publish, a skipped probe -- the two
    things a unit test cannot hand a real boundary), with ``window_floor``
    (and, for KOM-1114b, ``window_oldest``) threaded through so this exercises
    the production ``_close_iteration`` extras write rather than a
    reimplementation of it.
    """
    root.mkdir(parents=True, exist_ok=True)
    campaign = _campaign(
        root, steps_per_iteration=_BOUNDARY_STEPS, window_floor=window_floor
    )
    _publish_at(checkpoint_dir(root), _BOUNDARY_STEPS)
    _publish_at(ema_dir(root), _BOUNDARY_STEPS)

    def already_published(*args: object, **kwargs: object) -> tuple[int, ...]:
        return ()

    def no_probe(*args: object, **kwargs: object) -> tuple[None, str]:
        return (None, "")

    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(campaign_loop, "reconcile_ema", already_published)
    monkeypatch.setattr(campaign_loop, "_probe_set", no_probe)

    campaign_loop._close_iteration(
        root,
        campaign,
        SelfPlayConfig(workers=1, process_batch_size=1, games_in_flight=1),
        0,
        parameter_keys(KomugiNet(SMALL_CONFIG)),
        served="",
        resume=None,
        trained=KomugiNet(SMALL_CONFIG),
        history=[{TOTAL_KEY: 1.0}, {TOTAL_KEY: 0.9}],
        games=1,
        rows=4,
        stats=game_stats(shard_root(root, 0)),
        mutes={},
        previous_bn=None,
        pending_mute=None,
        results=[],
        window_oldest=window_oldest,
    )

    read = read_records(journal_path(root), IterationMetrics, missing_ok=True)
    written = [
        record
        for record in read.records
        if record.extras.get(boundary.CAMPAIGN_MARKER, 0.0) == 1.0
    ]
    assert len(written) == 1, (
        f"want exactly one campaign record for the boundary, got "
        f"{[sorted(record.extras) for record in written]}"
    )
    return written[0].extras


def test_the_campaign_record_omits_window_floor_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``window_floor=None`` writes no key: absence must mean 'never excluded'."""
    extras = _close_a_planted_boundary(tmp_path / "root", monkeypatch, window_floor=None)
    assert campaign_loop.WINDOW_FLOOR not in extras


def test_the_campaign_record_carries_window_floor_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``window_floor=3`` stamps the extra at exactly that value."""
    extras = _close_a_planted_boundary(tmp_path / "root", monkeypatch, window_floor=3)
    assert extras.get(campaign_loop.WINDOW_FLOOR) == 3.0


# --------------------------------------------------------------------------
# the campaign record (extras[WINDOW_OLDEST], KOM-1114b)
# --------------------------------------------------------------------------


def test_the_campaign_record_carries_window_oldest_when_floor_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realised window's oldest member, not the floor itself."""
    extras = _close_a_planted_boundary(
        tmp_path / "root", monkeypatch, window_floor=3, window_oldest=5
    )
    assert extras.get(campaign_loop.WINDOW_OLDEST) == 5.0


def test_the_campaign_record_omits_window_oldest_when_floor_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``window_floor=None`` must stay byte-identical: no ``window_oldest`` either.

    `window_oldest` is handed in regardless, so this pins the gate as being on
    `window_floor` and not merely on whether a caller supplied the new
    argument.
    """
    extras = _close_a_planted_boundary(
        tmp_path / "root", monkeypatch, window_floor=None, window_oldest=5
    )
    assert campaign_loop.WINDOW_FLOOR not in extras
    assert campaign_loop.WINDOW_OLDEST not in extras


# --------------------------------------------------------------------------
# CampaignConfig.window_floor validation (KOM-1114b)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("floor", [-1, -5])
def test_a_negative_window_floor_is_rejected(tmp_path: Path, floor: int) -> None:
    """A floor below iteration 0 names an iteration that cannot exist."""
    with pytest.raises(ValueError, match="window_floor"):
        _campaign(tmp_path, window_floor=floor)


def test_a_bool_window_floor_is_rejected(tmp_path: Path) -> None:
    """``bool`` is an ``int`` subclass that would otherwise pass ``>= 0``.

    ``True`` is iteration 1 and ``False`` reads as a floor of iteration 0;
    neither is what a caller who wrote ``window_floor=True`` by mistake (a
    flag-shaped field next to this one) meant.
    """
    with pytest.raises(ValueError, match="window_floor"):
        _campaign(tmp_path, window_floor=True)


def test_a_zero_window_floor_is_accepted(tmp_path: Path) -> None:
    """Zero is a legitimate floor -- iteration 0 -- not a false-y reject."""
    assert _campaign(tmp_path, window_floor=0).window_floor == 0


# --------------------------------------------------------------------------
# run_campaign's own window construction (KOM-1114b, review F2)
# --------------------------------------------------------------------------


def _fake_run_selfplay(
    net_factory: object,
    orchestrator_config: object,
    selfplay_config: object,
    seam: object,
) -> RunSummary:
    """Stand in for the real orchestrator: write one worker's census rows.

    Only the four columns :data:`~komugi.campaign.game_stats.STAT_COLUMNS`
    reads (``tests/campaign/unit/test_game_stats.py``'s ``_CENSUS_SCHEMA``):
    enough for ``game_stats`` and for the window to see non-empty data, and
    nothing a real game would otherwise need, because training itself is
    stubbed below and never looks at a row's content.
    """
    from komugi.train.shards import FieldSpec, Schema, ShardWriter

    worker_dir = Path(orchestrator_config.shard_root) / "worker_0_r0"  # type: ignore[attr-defined]
    worker_dir.mkdir(parents=True, exist_ok=True)
    schema = Schema(
        fields=tuple(
            FieldSpec(name=name, dtype=dtype, shape=())
            for name, dtype in (
                ("game_id", "int64"),
                ("ply", "int64"),
                ("mlh_target", "float32"),
                ("termination", "int64"),
            )
        )
    )
    rows = 3
    with ShardWriter(worker_dir, schema, samples_per_shard=4) as writer:
        for ply in range(rows):
            writer.append(
                {
                    "game_id": torch.tensor(0, dtype=torch.int64),
                    "ply": torch.tensor(ply, dtype=torch.int64),
                    "mlh_target": torch.tensor(0.1, dtype=torch.float32),
                    "termination": torch.tensor(
                        TERMINATION_DECISIVE, dtype=torch.int64
                    ),
                }
            )
    return RunSummary(
        games_completed=1,
        rows_written=rows,
        shard_dirs=(str(worker_dir),),
        restarts=0,
        worker_incarnations=(0,),
    )


def _fake_training_iteration(
    net: torch.nn.Module,
    source: object,
    step: object,
    *,
    steps: int,
    rng_seed: int,
    checkpoint_dir: Path | str | None = None,
    journal_path: Path | str | None = None,
    iteration: int = 0,
    global_step_base: int = 0,
) -> list[dict[str, float]]:
    """Stand in for the real trainer: publish a checkpoint, never read `source`.

    What this test is wiring around: the statement under test
    (``campaign_window_refs(..., floor=campaign.window_floor)``) runs entirely
    *before* this is called, so the window's rows never need to be the shapes
    a real optimiser step requires -- only ``len()``, which the four-column
    rows above already satisfy.
    """
    assert checkpoint_dir is not None, "run_campaign always passes a checkpoint_dir"
    save_checkpoint(
        checkpoint_dir,
        global_step_base + steps,
        net.state_dict(),
        {},
        {"iteration": iteration},
    )
    return [{TOTAL_KEY: 1.0}, {TOTAL_KEY: 0.9}]


def test_run_campaign_wires_window_floor_into_the_window_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: deleting ``floor=campaign.window_floor`` at the call site must fail this.

    ``campaign_window_refs`` is wrapped to record the ``floor`` it is called
    with, and ``run_campaign`` is driven through one real (if faked)
    iteration to reach that call inside the loop body rather than a
    reimplementation of it. Self-play and the trainer are the two things a
    unit test cannot stand up for real -- :func:`_fake_run_selfplay` and
    :func:`_fake_training_iteration` replace exactly those two, and nothing
    between them, which is where the statement under test lives.

    Verified by temporarily deleting ``floor=campaign.window_floor`` from
    ``run_campaign``'s window-construction call and re-running this test: it
    fails (``seen == [None]``, not ``[0]``), because the recording wrapper's
    default then wins.
    """
    root = tmp_path / "root"
    seen: list[int | None] = []
    real_campaign_window_refs = campaign_loop.campaign_window_refs

    def recording(
        root_arg: Path, through: int, config: WindowConfig, *, floor: int | None = None
    ) -> tuple[IterationRef, ...]:
        seen.append(floor)
        return real_campaign_window_refs(root_arg, through, config, floor=floor)

    monkeypatch.setattr(campaign_loop, "campaign_window_refs", recording)
    monkeypatch.setattr(campaign_loop, "run_selfplay", _fake_run_selfplay)
    monkeypatch.setattr(campaign_loop, "training_iteration", _fake_training_iteration)
    monkeypatch.setattr(
        campaign_loop, "_probe_set", lambda root, campaign: (None, "")
    )
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)

    campaign = _campaign(root, window_floor=0)
    selfplay = SelfPlayConfig(workers=1, process_batch_size=1, games_in_flight=1)

    # Only the statement under test -- the recorded `floor` -- is asserted on;
    # everything after the window is built (the EMA gate, the liveness gate,
    # enforcement) is real production code this test did not stand up to
    # satisfy, and a pause or an unrelated crash there must not hide a
    # passing wiring check.
    with contextlib.suppress(Exception):
        campaign_loop.run_campaign(campaign, selfplay)

    assert seen == [0], (
        f"campaign_window_refs saw floor={seen!r}; run_campaign's window "
        f"construction must pass campaign.window_floor through unchanged"
    )
