"""The round trip: self-play -> shards -> train -> checkpoint -> EMA -> a rating.

KOM-015, the house rule (L15/L20) that no pod time is bought until the whole
loop has been run once, end to end, on this machine, with **nothing stood in
for**. Every component here is the production one:

* :class:`~komugi.selfplay.worker.SelfPlayWorker` plays the games, in spawned
  processes, against a real inference server serving a real net;
* the rows land as real ``shard-*.npz`` files and are read back through
  :class:`~komugi.train.shards.ShardReader` under the real
  :data:`komugi.selfplay.records.SCHEMA`;
* :class:`~komugi.train.step.TrainStep` runs real optimiser steps over them and
  :func:`~komugi.train.step.training_iteration` publishes a real checkpoint;
* :func:`~komugi.train.ema.reconcile_ema` publishes the EMA the *next*
  iteration's workers actually serve;
* and :func:`~komugi.eval.ladder.run_ladder` -- the 011a measurement core --
  plays the produced checkpoints against each other through the real engine and
  the real Gumbel search, and fits a real Elo table to the result.

The driver is :func:`~komugi.campaign.loop.run_campaign` itself, not a
reassembly of its parts: a round-trip test that wired the components together
by hand would certify a loop nobody runs.

What is asserted, and what is deliberately not
==============================================
**The artifact chain, never strength.** At this scale a strength claim would be
a claim about search noise. What is claimed instead is that each stage's output
is the next stage's input, on disk, in a form the next stage accepts:

#. shards exist under ``root/iter_<n>/shards/worker_<k>_r<i>`` and parse under
   ``SCHEMA`` with nothing skipped;
#. the trainer's checkpoint exists at a step that advanced by exactly
   ``steps_per_iteration``, stamped with the iteration that wrote it;
#. the *net hash* advanced across the boundary -- iteration 1's ``train``
   fingerprint differs from iteration 0's -- and the net iteration 1's workers
   **served** is bit-identically the EMA iteration 0 published. That pair is the
   loop closing: a checkpoint written, reconciled, and played with;
#. the EMA series is one file per iteration at the trainer's own steps, with
   ``meta`` sourced from the trainer's checkpoint rather than recomputed;
#. the telemetry journal carries a record per iteration from both writers;
#. the eval core returns a **finite** rating for every produced checkpoint, off
   games that pool into one auditable :class:`~komugi.eval.records.MatchSet`;
#. and iteration 0 of the *same* loop at ``loss_recipe="mlh"`` -- campaign 6's
   recipe -- journals the moves-left head's loss, stamps the recipe and the
   search-side lever on its campaign record, writes rows bit-identical to the
   default recipe's, and trains a net that differs from it (KOM-159 T9,
   section 8). One iteration, because the recipe is the trainer's alone.

The two things dialled down, and why
====================================
**The search budget, in both halves.** Self-play runs at ``m=n=2`` and the
ladder's entrants are rebudgeted to the same, by wrapping
:func:`~komugi.eval.arena.ladder_players` with ``dataclasses.replace`` --
the escape hatch that function's own docstring names, and the idiom
``tests/eval/test_ladder_behavior.py`` already uses. It is needed because
:class:`~komugi.eval.ladder.LadderConfig` has **no search-budget knob**:
``ladder_players`` hands out entrants at the production 32/256 and ``run_ladder``
deliberately refuses to let a caller quietly rate a cheaper search than
self-play plays. Rebudgeting the entrants is therefore the only way to run the
real driver at test scale without adding a knob to ``src/``, which this ticket
is not allowed to do. Everything else -- the checkpoints, the loading, the nets,
the engine, the search schedule, the seat pairing, the seeds, the pooling and
the fit -- is real.

**The net shape.** ``KOMUGI_CAMPAIGN_NET=small`` selects
:data:`komugi.campaign.factory.SMALL_CONFIG`, b2c16. Its *trunk* shrinks; the
inference seam does not (``in_planes`` and ``num_actions`` stay at production
values), so the workers, the server and the arena all talk over the shapes the
real run uses.

Bounding the run
================
``pytest-timeout`` is not in the dev shell, and this test owns spawned
processes -- exactly the thing that hangs. So ``run_campaign`` is never called
directly: :func:`walled_run_campaign` runs it on a daemon helper thread and
turns "still running at the wall" into a *failure in bounded time*. A thread
cannot be killed, so the wall additionally kills every child the campaign left
behind, which is what lets the stuck thread unwind. This is the discipline of
``tests/campaign/behavior/campaign_rig.py``, restated rather than imported: that
module sits under a different rootdir-inserted directory, and importing it would
make this suite's importability depend on pytest having collected another one.

The wall is generous and the orchestrator's liveness knobs more so. A wall
bounds a *hang*; it is not a latency assertion, and a tight liveness setting on
a loaded CI box would only manufacture shard restarts that corrupt the counts
asserted below.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import pytest

from komugi.campaign import factory
from komugi.campaign.boundary import CAMPAIGN_MARKER
from komugi.campaign.loop import (
    CampaignConfig,
    IterationResult,
    checkpoint_dir,
    ema_dir,
    iteration_root,
    journal_path,
    run_campaign,
)
from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.observation import observation_size
from komugi.eval import arena, ladder
from komugi.eval.arena import ArenaPlayer
from komugi.eval.records import VALID_RESULTS, MatchSet
from komugi.eval.seats import check_paired_books
from komugi.selfplay.inference import InferenceConfig
from komugi.selfplay.records import SCHEMA
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import (
    EMA,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)
from komugi.train.checkpoint import (
    Checkpoint,
    CheckpointRef,
    list_checkpoints,
    load_checkpoint_file,
)
from komugi.train.shards import SHARD_GLOB, ShardReader

# ---------------------------------------------------------------------------
# the campaign's size
# ---------------------------------------------------------------------------

ITERATIONS: Final[int] = 2
"""Two, and two is not a round number here -- it is the smallest campaign that
*has* a boundary. Every claim about the chain closing (a checkpoint published by
one iteration being the net the next one serves, the net hash advancing, two
coexisting checkpoints for the ladder to rate) is a claim about the seam between
iteration 0 and iteration 1, and a one-iteration run has no seam at all."""

WORKERS: Final[int] = 1
GAMES_PER_WORKER: Final[int] = 1
"""One worker, one game: the smallest self-play run that produces a shard.

Deliberately smaller than ``tests/campaign/behavior``'s two workers. That suite
needs the second worker because its subject is the ``ConcatSource`` fan-in over
several worker directories; this one's subject is the chain, which one worker
exercises end to end. The saving is half the self-play wall of the whole test."""

STEPS_PER_ITERATION: Final[int] = 4
BATCH_SIZE: Final[int] = 8
LR: Final[float] = 1e-3

RUN_SEED: Final[int] = 20260907

DEVICE: Final[str] = "cpu"
"""CPU, not the CUDA box. The chain is device-independent and CPU is what makes
this test the same test on the desktop and in ``ci/check.sh``; a GPU here would
buy nothing at b2c16 and cost the run's portability."""

TINY_M: Final[int] = 2
TINY_N: Final[int] = 2
"""The Gumbel budget both halves run at: 2 root candidates, 2 simulations.

The *smallest real search*, not a stubbed one -- ``GumbelConfig`` accepts
``m >= 1`` and ``n >= 0``, so this still runs the halving schedule, the wave
evaluator and the root sample."""

# ---------------------------------------------------------------------------
# the eval leg's size
# ---------------------------------------------------------------------------

LADDER_GAMES: Final[int] = 2
"""One seat-swapped pair per matchup -- the smallest match the arena will play,
and the smallest a ladder will accept (``MIN_GAMES_PER_MATCHUP``)."""

LADDER_SEED: Final[int] = 11

# ---------------------------------------------------------------------------
# the seam's shapes
# ---------------------------------------------------------------------------

IN_PLANES, BOARD_H, BOARD_W = observation_size()
"""The real encoder's observation shape -- (128, 9, 9) today."""

NUM_ACTIONS: Final[int] = ACTION_SIZE
"""The real engine's flat action space, ``A``."""

MAX_BATCH: Final[int] = 4
"""Server-side row ceiling. Above ``workers * games_in_flight``, which is what
:func:`~komugi.campaign.loop.campaign_batch_fit` refuses a campaign for."""

# ---------------------------------------------------------------------------
# walls and patience
# ---------------------------------------------------------------------------

LIVENESS_S: Final[float] = 600.0
"""Liveness / first-shard / drain patience handed to the orchestrator.

Ten minutes for games that take seconds. These knobs exist to notice a *wedge*;
at this scale the only thing a tight setting could produce is a shard restart
caused by a loaded CI box, which would corrupt the counts asserted below."""

CAMPAIGN_WALL_S: Final[float] = 900.0
"""How long the campaign may run before the test fails instead of hanging."""

KILL_GRACE_S: Final[float] = 10.0
"""How long a straggler child gets to die after ``kill()``."""

# ---------------------------------------------------------------------------
# the layout, as the loop pins it
# ---------------------------------------------------------------------------

SHARD_DIR_RE: Final[re.Pattern[str]] = re.compile(r"worker_(\d+)_r(\d+)")
"""Design D10's per-worker-per-incarnation directory name."""

EMA_ROLE: Final[str] = "ema"
"""What an EMA checkpoint's ``meta['role']`` says it is (KOM-022 section 2)."""


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

SELFPLAY: Final[SelfPlayConfig] = SelfPlayConfig(
    workers=WORKERS,
    process_batch_size=1,
    gumbel_m_full=TINY_M,
    gumbel_n_full=TINY_N,
    gumbel_m_fast=TINY_M,
    gumbel_n_fast=TINY_N,
    prob_fast=0.5,
)
"""How a game is played. ``prob_fast`` is one half rather than the production
0.75 so that a run this short still records both ``search_mode`` values, which
is what makes the trainer's fast/full row split non-degenerate."""

ORCHESTRATOR_OVERRIDES: Final[dict[str, float]] = {
    "liveness_timeout_s": LIVENESS_S,
    "first_shard_grace_s": LIVENESS_S,
    "drain_timeout_s": LIVENESS_S,
}


def campaign_config(root: Path) -> CampaignConfig:
    """The campaign this module runs, over `root`."""
    return CampaignConfig(
        iterations=ITERATIONS,
        run_seed=RUN_SEED,
        root=str(root),
        steps_per_iteration=STEPS_PER_ITERATION,
        batch_size=BATCH_SIZE,
        lr=LR,
        workers=WORKERS,
        games_per_worker=GAMES_PER_WORKER,
        device=DEVICE,
    )


def inference_config() -> InferenceConfig:
    """The seam's shapes, stated rather than defaulted.

    Passed explicitly because a mismatch between the served net and the slot
    geometry surfaces *inside a spawned child*, where it is hardest to read.
    ``request_timeout_s`` is high for the reason :data:`LIVENESS_S` is high.
    """
    return InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=MAX_BATCH,
        batch_timeout_ms=1.0,
        request_timeout_s=240.0,
    )


# ---------------------------------------------------------------------------
# the wall
# ---------------------------------------------------------------------------


class WallExpired(AssertionError):
    """``run_campaign`` was still running when its wall ran out."""


def kill_stragglers() -> list[str]:
    """Kill and join every child process this interpreter still owns.

    ``multiprocessing.active_children`` sees the orchestrator's children because
    ``run_campaign`` runs in *this* process (on the wall's helper thread) and
    the child registry is a module global shared by every context.
    """
    killed: list[str] = []
    for proc in multiprocessing.active_children():
        killed.append(f"{proc.name} (pid {proc.pid})")
        proc.kill()
        proc.join(KILL_GRACE_S)
    return killed


def walled_run_campaign(
    root: Path, config: CampaignConfig | None = None
) -> list[IterationResult]:
    """Run the campaign over `root`, held to :data:`CAMPAIGN_WALL_S`.

    A thread rather than a subprocess, because the campaign must run *in this
    interpreter* for its spawned grandchildren to be reachable by
    :func:`kill_stragglers`. A thread cannot be killed, so the wall does the
    next best thing and kills the campaign's children: that unblocks whatever
    the stuck thread was waiting on, and the thread is a daemon so a truly
    unkillable one cannot stop the interpreter exiting.

    `config` defaults to :func:`campaign_config` over `root`. It is a parameter
    so that section 8's moves-left run gets the *same* wall, the same straggler
    kill and the same failure reporting as this module's main run -- a second
    hand-rolled thread wall would be a second thing to get wrong.
    """
    produced: list[IterationResult] = []
    failure: list[BaseException] = []

    def target() -> None:
        """Run the campaign, capturing whatever it does."""
        try:
            produced.extend(
                run_campaign(
                    campaign_config(root) if config is None else config,
                    SELFPLAY,
                    orchestrator_overrides=ORCHESTRATOR_OVERRIDES,
                    inference=inference_config(),
                )
            )
        except BaseException as exc:  # reported with its traceback, not swallowed
            failure.append(exc)

    thread = threading.Thread(target=target, name="round-trip", daemon=True)
    thread.start()
    thread.join(CAMPAIGN_WALL_S)
    if thread.is_alive():
        killed = kill_stragglers()
        raise WallExpired(
            f"run_campaign was still running after {CAMPAIGN_WALL_S}s under "
            f"{root}. Killed {len(killed)} orphaned child process(es): {killed}"
        )
    if failure:
        raise AssertionError(
            f"run_campaign raised {type(failure[0]).__name__}: {failure[0]}"
        ) from failure[0]
    return produced


# ---------------------------------------------------------------------------
# reading the run back off disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTrip:
    """One finished campaign: where it lives and what it reported."""

    root: Path
    results: list[IterationResult]


def shard_dirs_under(directory: Path) -> list[Path]:
    """Every ``worker_<k>_r<i>`` directory at or below `directory`, sorted.

    Recursive because the contract pins ``root/iter_<n>`` and design D10 pins
    ``worker_<k>_r<i>``, but the ``shards/`` level between them is the loop's
    own choice and not something a test should forbid.
    """
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_dir() and SHARD_DIR_RE.fullmatch(path.name)
        ),
        key=lambda path: str(path),
    )


def readers_for(directory: Path) -> list[ShardReader]:
    """One ``SCHEMA``-checked reader per worker directory under `directory`.

    Constructing the reader *with* ``schema=SCHEMA`` is what makes this a parse
    assertion rather than a file-count one: a shard whose fingerprint does not
    match the production schema lands in ``skipped`` and is invisible to every
    other view, so an empty ``skipped`` is the claim "these bytes are rows the
    real trainer can draw".
    """
    return [
        ShardReader(worker_dir, schema=SCHEMA)
        for worker_dir in shard_dirs_under(directory)
    ]


def checkpoint_refs(directory: Path) -> list[CheckpointRef]:
    """Every checkpoint in `directory`, **oldest** first.

    :func:`~komugi.train.checkpoint.list_checkpoints` answers newest-first; a
    round trip reads in training order.
    """
    return list(reversed(list_checkpoints(directory)))


def loaded(ref: CheckpointRef) -> Checkpoint:
    """Load exactly `ref`, failing the test if it does not verify."""
    checkpoint = load_checkpoint_file(ref.path)
    assert checkpoint is not None, (
        f"{ref.path} does not verify: its embedded digest does not match its "
        f"bytes, or its payload does not decode"
    )
    return checkpoint


def fingerprints(root: Path) -> list[NetFingerprint]:
    """Every :class:`NetFingerprint` the campaign journalled, in write order."""
    return list(read_records(journal_path(root), NetFingerprint, missing_ok=True).records)


def fingerprint_at(
    records: Sequence[NetFingerprint], iteration: int, role: str
) -> NetFingerprint | None:
    """Return the **last** record for `(iteration, role)`, or ``None``.

    Last rather than first: an iteration that crashed before its checkpoint is
    legitimately re-run under the same number, and the journal is append-only,
    so the newest record is the one describing what is on disk.
    """
    matched = [
        record
        for record in records
        if record.iteration == iteration and record.role == role
    ]
    return matched[-1] if matched else None


def iteration_metrics(root: Path) -> list[IterationMetrics]:
    """Every :class:`IterationMetrics` in the campaign's journal, in write order."""
    return list(
        read_records(journal_path(root), IterationMetrics, missing_ok=True).records
    )


def is_campaign_record(record: IterationMetrics) -> bool:
    """Whether `record` is the loop's own, rather than the trainer's.

    Both are written for the same iteration, deliberately, and the
    discriminator is explicit (:data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`)
    rather than inferred from ``games > 0`` -- an iteration whose self-play
    produced nothing still writes a campaign record.
    """
    return record.extras.get(CAMPAIGN_MARKER, 0.0) == 1.0


def describe(paths: Sequence[Path]) -> str:
    """Render a path list for an assertion message."""
    return ", ".join(sorted(str(path) for path in paths)) or "<none>"


# ---------------------------------------------------------------------------
# fixtures: one campaign, one ladder, per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_net_env() -> Iterator[None]:
    """Point the module-level factory at the tiny CPU net for this module.

    The environment, not a closure, because the value has to be visible to
    *spawned children*: they inherit this process's environment, and the server
    that builds the served net is one of them. It stays set for the whole
    module, so the ladder's own ``factory.new_net()`` calls agree with the
    shape the campaign trained.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(factory.NET_ENV, factory.SMALL_NET)
        patch.setenv(factory.DEVICE_ENV, DEVICE)
        yield


@pytest.fixture(scope="module")
def round_trip(
    small_net_env: None, tmp_path_factory: pytest.TempPathFactory
) -> RoundTrip:
    """Run the whole loop once, and hand the same run to every test below.

    Module-scoped because this is by far the most expensive thing here: real
    games, real optimiser steps and two spawned process trees. ``tmp_path_factory``
    rather than ``mkdtemp`` so pytest keeps the tree of a failing run around --
    when an assertion about a shard or a checkpoint fails, the tree it failed
    about is the first thing worth looking at.
    """
    root = tmp_path_factory.mktemp("round-trip")
    return RoundTrip(root=root, results=walled_run_campaign(root))


def tiny_budget_players(patch: pytest.MonkeyPatch) -> None:
    """Make the ladder driver's entrants cheap without faking where they come from.

    The real directory scan and the real checkpoint-loading factories, with only
    the per-entrant search budget replaced -- ``dataclasses.replace``, which
    :func:`~komugi.eval.arena.ladder_players`'s own docstring names as the way
    to rebudget a player. See this module's docstring for why there is no config
    knob to do it with.
    """
    real = arena.ladder_players

    def players(
        directory: str | os.PathLike[str], every_n: int = 1, device: str = "cpu"
    ) -> tuple[ArenaPlayer, ...]:
        """Call the real scan, then rebudget each entrant."""
        return tuple(
            replace(entrant, m=TINY_M, n=TINY_N)
            for entrant in real(directory, every_n, device)
        )

    patch.setattr(ladder, "ladder_players", players)


@pytest.fixture(scope="module")
def rated(round_trip: RoundTrip) -> ladder.LadderResult:
    """Rate the checkpoints the campaign produced, through the real driver.

    The ladder is played over ``<root>/ckpt`` -- the **trainer's** series, which
    is the one whose rungs are a training curve. The contempt is self-play's
    own ``draw_score``, not a literal: design D3's rule is that a ladder fit
    from games played at a different contempt than the self-play that trained
    the checkpoints is a different experiment, and the way to obey it is to
    read the number off the config the workers ran under.
    """
    with pytest.MonkeyPatch.context() as patch:
        tiny_budget_players(patch)
        return ladder.run_ladder(
            checkpoint_dir(round_trip.root),
            ladder.LadderConfig(
                games_per_matchup=LADDER_GAMES,
                seed=LADDER_SEED,
                draw_score=SELFPLAY.draw_score,
            ),
            device=DEVICE,
        )


# ---------------------------------------------------------------------------
# 1. the loop ran
# ---------------------------------------------------------------------------


def test_the_loop_ran_every_iteration_and_measured_each_one(
    round_trip: RoundTrip,
) -> None:
    """Two iterations were asked for; two ran, each having played and trained.

    The premise every other test rests on. The counts are asserted as
    *positive* rather than as exactly ``workers * games_per_worker``: design D10
    lets a restarted shard replay its whole budget under a new incarnation, so
    the orchestrator legitimately reports more games than were asked for when a
    restart happened, and pinning the exact number would make a healthy restart
    a failure.

    The losses are asserted *finite*, not descending. Four steps at batch 8 over
    one game's rows is a smoke run; a suite that asserted descent there would be
    asserting noise. What must hold is that the numbers were measured at all --
    a NaN here is a diverged step the campaign then carried on from.
    """
    results = round_trip.results
    assert len(results) == ITERATIONS, (
        f"run_campaign(iterations={ITERATIONS}) returned {len(results)} "
        f"IterationResult(s): {results}"
    )
    assert [result.iteration for result in results] == list(range(ITERATIONS)), (
        f"iterations must be reported in order from zero, got "
        f"{[result.iteration for result in results]}"
    )
    for result in results:
        assert result.games >= 1, f"iteration {result.iteration} played no games"
        assert result.rows >= 1, f"iteration {result.iteration} wrote no rows"
        for name in ("first_loss", "last_loss"):
            value = float(getattr(result, name))
            assert math.isfinite(value), (
                f"iteration {result.iteration} reports {name}={value}: a "
                f"non-finite loss is a diverged step, and the campaign trained on"
            )


# ---------------------------------------------------------------------------
# 2. self-play -> shards
# ---------------------------------------------------------------------------


def test_selfplay_wrote_shards_that_parse_under_the_production_schema(
    round_trip: RoundTrip,
) -> None:
    """Every iteration left rows on disk that the real schema accepts.

    Three separate claims, and they fail differently. The *layout* claim is that
    ``root/iter_<n>`` holds a ``worker_<k>_r<i>`` directory (design D10) with a
    sealed ``shard-*.npz`` in it. The *parse* claim is that a
    ``ShardReader(..., schema=SCHEMA)`` skips none of them -- a shard whose
    fingerprint disagrees with the production schema is a row the campaign
    counted and the trainer can never draw. The *content* claim is that a row's
    ``observation`` is the encoder's own shape: a reader will happily accept a
    schema-shaped file of zeros, and the shape is what ties these bytes to the
    net the workers were serving.
    """
    for iteration in range(ITERATIONS):
        directory = iteration_root(round_trip.root, iteration)
        assert directory.is_dir(), (
            f"{directory} does not exist; {round_trip.root} holds "
            f"{describe(sorted(round_trip.root.iterdir()))}"
        )
        worker_dirs = shard_dirs_under(directory)
        assert worker_dirs, (
            f"{directory} holds no worker_<k>_r<i> shard directory (design D10)"
        )
        assert sorted(directory.rglob(SHARD_GLOB)), (
            f"{directory} holds no sealed shard file"
        )

        rows = 0
        for reader in readers_for(directory):
            assert not reader.skipped, (
                f"iteration {iteration} wrote shards the production schema "
                f"rejects: "
                f"{[(entry.path.name, entry.reason) for entry in reader.skipped]}"
            )
            rows += len(reader)
        assert rows > 0, (
            f"iteration {iteration} left readable shard files under {directory} "
            f"holding zero rows between them"
        )


def test_the_rows_the_campaign_counted_are_the_rows_on_disk(
    round_trip: RoundTrip,
) -> None:
    """``IterationResult.rows`` is what the trainer's own reader sees.

    Read back through ``ShardReader`` under ``SCHEMA`` rather than counted from
    the ``npz`` headers, because the number that matters downstream is the
    number of samples the *window* can draw. A count taken any other way would
    still agree with itself while disagreeing with the trainer.
    """
    for result in round_trip.results:
        directory = iteration_root(round_trip.root, result.iteration)
        readable = sum(len(reader) for reader in readers_for(directory))
        assert readable == int(result.rows), (
            f"iteration {result.iteration} reported {result.rows} rows, but "
            f"{directory} reads back {readable} under SCHEMA"
        )


def test_a_shard_row_carries_the_encoders_own_shapes(round_trip: RoundTrip) -> None:
    """One row, unpacked: the seam's shapes reached the file.

    ``in_planes`` and ``num_actions`` are the two numbers the small-net config
    deliberately does **not** shrink, precisely so that a behavioural test of
    the loop cannot pass over rows the production trainer could not consume.
    """
    readers = readers_for(iteration_root(round_trip.root, 0))
    populated = [reader for reader in readers if len(reader) > 0]
    assert populated, "iteration 0 left no readable rows to unpack"

    row = populated[0][0]
    assert set(row) == set(SCHEMA.names), (
        f"a shard row carries fields {sorted(row)}, want {sorted(SCHEMA.names)}"
    )
    assert tuple(row["observation"].shape) == (IN_PLANES, BOARD_H, BOARD_W), (
        f"observation is {tuple(row['observation'].shape)}, want "
        f"{(IN_PLANES, BOARD_H, BOARD_W)} -- the encoder's own shape"
    )
    assert tuple(row["policy_target"].shape) == (NUM_ACTIONS,), (
        f"policy_target is {tuple(row['policy_target'].shape)}, want "
        f"{(NUM_ACTIONS,)}"
    )
    assert float(row["policy_target"].sum()) > 0.0, (
        "the policy target is all zeros: the search produced no visit "
        "distribution, so this row records no decision"
    )


# ---------------------------------------------------------------------------
# 3. shards -> trainer -> checkpoint
# ---------------------------------------------------------------------------


def test_the_trainer_published_one_checkpoint_per_iteration_and_advanced_the_step(
    round_trip: RoundTrip,
) -> None:
    """The step counter is global, and iteration 1 did not supersede iteration 0.

    Both halves matter and they are different failures.
    ``training_iteration`` names its checkpoint after ``global_step_base +
    steps``, so without a global base both iterations would file at step 4 and
    ``save_checkpoint``'s same-step supersede would unlink the first -- a
    campaign that keeps exactly one checkpoint however long it runs, and the
    ladder below would have nothing to rate.

    The **gap** is asserted, not the absolute step: "the counter starts at the
    first iteration's steps" and "it starts at zero" are both defensible
    readings of ``global_step_base``, and they differ by a constant. What the
    global counter claims either way is that one iteration of
    ``steps_per_iteration`` steps advances the count by exactly that much.
    """
    steps = [int(result.checkpoint_step) for result in round_trip.results]
    refs = checkpoint_refs(checkpoint_dir(round_trip.root))

    assert len(refs) == ITERATIONS, (
        f"want one trainer checkpoint per iteration under "
        f"{checkpoint_dir(round_trip.root)}, found "
        f"{[ref.path.name for ref in refs]}. One file for two iterations is the "
        f"same-step supersede"
    )
    assert [ref.step for ref in refs] == steps, (
        f"the checkpoints on disk are at steps {[ref.step for ref in refs]}, "
        f"but the campaign reported {steps}"
    )
    gaps = [second - first for first, second in zip(steps, steps[1:])]
    assert gaps == [STEPS_PER_ITERATION] * (ITERATIONS - 1), (
        f"consecutive checkpoint steps {steps} advance by {gaps}, but each "
        f"iteration ran {STEPS_PER_ITERATION} steps"
    )


def test_each_checkpoint_is_stamped_with_the_iteration_that_trained_it(
    round_trip: RoundTrip,
) -> None:
    """A checkpoint names its own provenance, so a rung is attributable.

    ``iteration``, ``steps``, ``global_step_base`` and ``batch_size`` are the
    trainer's stamps; without them a directory of checkpoints is a directory of
    anonymous weights and "which run, which arm, how many steps" is a question
    only a lab notebook could answer. ``in_planes`` is the KOM-002 architecture
    stamp, and it is checked because it is what lets a loader refuse an
    old-plane-count file by name rather than dying inside ``load_state_dict``.
    """
    refs = checkpoint_refs(checkpoint_dir(round_trip.root))
    assert len(refs) == ITERATIONS, f"want {ITERATIONS} checkpoints, got {len(refs)}"

    for iteration, ref in enumerate(refs):
        meta: dict[str, Any] = loaded(ref).meta
        assert meta.get("iteration") == iteration, (
            f"{ref.path.name} is the {iteration}-th checkpoint but its meta "
            f"says iteration={meta.get('iteration')!r}"
        )
        assert meta.get("steps") == STEPS_PER_ITERATION, (
            f"{ref.path.name} claims steps={meta.get('steps')!r}, want "
            f"{STEPS_PER_ITERATION}"
        )
        assert meta.get("global_step_base") == iteration * STEPS_PER_ITERATION, (
            f"{ref.path.name} claims global_step_base="
            f"{meta.get('global_step_base')!r}, want "
            f"{iteration * STEPS_PER_ITERATION}"
        )
        assert meta.get("batch_size") == BATCH_SIZE, (
            f"{ref.path.name} claims batch_size={meta.get('batch_size')!r}, "
            f"want {BATCH_SIZE}"
        )
        assert meta.get("in_planes") == IN_PLANES, (
            f"{ref.path.name} claims in_planes={meta.get('in_planes')!r}, want "
            f"the encoder's {IN_PLANES}"
        )


def test_the_optimiser_state_travels_with_the_weights(round_trip: RoundTrip) -> None:
    """A trainer checkpoint carries AdamW's moments, not just the net.

    The half of a resume that is invisible when it is missing: weights without
    moments is a cold Adam wearing a warm net, and a campaign resumed from one
    trains differently while reporting nothing. It is asserted here rather than
    in the resume test because it is a property of what the trainer *wrote* --
    and because it is what tells the trainer's series apart from the EMA's,
    which carries ``optim_state={}`` by design.
    """
    for ref in checkpoint_refs(checkpoint_dir(round_trip.root)):
        checkpoint = loaded(ref)
        assert checkpoint.model_state, f"{ref.path.name} carries no weights"
        assert checkpoint.optim_state, (
            f"{ref.path.name} carries an empty optim_state: a resume from it "
            f"would restart AdamW's moments from zero"
        )


# ---------------------------------------------------------------------------
# 4. checkpoint -> EMA -> the net the next iteration served
# ---------------------------------------------------------------------------


def test_the_ema_series_is_published_at_the_trainers_own_steps(
    round_trip: RoundTrip,
) -> None:
    """One EMA file per iteration, at the step of the checkpoint it came from.

    Publishing the EMA at its *source's* step is what makes "EMA ahead of the
    trainer" an impossible state rather than an ambiguous one -- and
    ``reconcile_ema`` refuses to start on a root where it is true. Each file is
    identified by what it *says it is* (``meta['role'] == 'ema'``) rather than
    by which directory it sits in, so a loop that moved the directory would fail
    this test rather than pass it vacuously.
    """
    ema_refs = checkpoint_refs(ema_dir(round_trip.root))
    train_steps = [
        ref.step for ref in checkpoint_refs(checkpoint_dir(round_trip.root))
    ]

    assert len(ema_refs) == ITERATIONS, (
        f"want one EMA checkpoint per iteration under {ema_dir(round_trip.root)}, "
        f"found {[ref.path.name for ref in ema_refs]}"
    )
    assert [ref.step for ref in ema_refs] == train_steps, (
        f"the EMA series is at steps {[ref.step for ref in ema_refs]} but the "
        f"trainer published at {train_steps}: an EMA is published at its "
        f"source's step, so these cannot differ"
    )
    for ref in ema_refs:
        checkpoint = loaded(ref)
        assert str(checkpoint.meta.get("role", "")) == EMA_ROLE, (
            f"{ref.path.name} sits in the EMA directory but its meta says "
            f"role={checkpoint.meta.get('role')!r}"
        )
        assert checkpoint.optim_state == {}, (
            f"{ref.path.name} carries an optimiser state; an EMA has no "
            f"optimiser and nothing resumes from it"
        )


def test_the_ema_meta_is_sourced_from_the_trainers_checkpoint(
    round_trip: RoundTrip,
) -> None:
    """The EMA names the iteration, the source step and the arm of *its source*.

    Sourced, never recomputed: ``reconcile_ema`` reads ``iteration`` and
    ``fast_policy_weight`` off the trainer checkpoint it is averaging in, which
    is what makes the trainer's stamp and the EMA's stamp provably the same
    value rather than two independent claims that could disagree. The EMA is the
    artifact most likely to be copied out of a run directory and handed to an
    evaluator, so a disagreement here is a mislabelled experiment.

    The arm is asserted as **agreeing with the source**, in whichever of its two
    states the source is in, rather than as a fixed number: ``ema_meta`` omits
    the key when its source records none, and ``training_iteration`` stamps it
    unconditionally -- including the ``1.0`` default this campaign runs at. So
    at defaults the key is *present* on both, carrying 1.0; a test that demanded
    it be absent would be pinning a behaviour ``src/`` does not have.
    """
    train_refs = checkpoint_refs(checkpoint_dir(round_trip.root))
    ema_refs = checkpoint_refs(ema_dir(round_trip.root))
    assert len(ema_refs) == len(train_refs) == ITERATIONS

    for iteration, (train_ref, ema_ref) in enumerate(zip(train_refs, ema_refs)):
        source: dict[str, Any] = loaded(train_ref).meta
        meta: dict[str, Any] = loaded(ema_ref).meta

        assert meta.get("iteration") == iteration, (
            f"{ema_ref.path.name} claims iteration={meta.get('iteration')!r}, "
            f"want {iteration}"
        )
        assert meta.get("source_step") == train_ref.step, (
            f"{ema_ref.path.name} claims source_step="
            f"{meta.get('source_step')!r} but was published from "
            f"{train_ref.path.name} at step {train_ref.step}"
        )
        for key in ("train_sha256", "ema_sha256", "decay"):
            assert key in meta, f"{ema_ref.path.name} carries no {key!r}"

        assert ("fast_policy_weight" in meta) == (
            "fast_policy_weight" in source
        ), (
            f"{ema_ref.path.name} and its source {train_ref.path.name} disagree "
            f"about whether an arm was recorded at all: "
            f"{meta.get('fast_policy_weight')!r} vs "
            f"{source.get('fast_policy_weight')!r}"
        )
        if "fast_policy_weight" in source:
            assert meta["fast_policy_weight"] == source["fast_policy_weight"], (
                f"{ema_ref.path.name} claims arm "
                f"{meta['fast_policy_weight']!r} but its source claims "
                f"{source['fast_policy_weight']!r}: an EMA may not relabel the "
                f"experiment that produced it"
            )


def test_the_net_hash_advanced_across_the_iteration_boundary(
    round_trip: RoundTrip,
) -> None:
    """The trained net at iteration 1 is not the trained net at iteration 0.

    The v1 catastrophe, stated as a fingerprint comparison: a loop whose
    optimiser silently did nothing -- a zero learning rate, a detached graph, a
    net that was never the one being stepped -- keeps running, keeps
    checkpointing and keeps reporting losses, and the *only* observable is that
    the weights stop changing. Two digests are enough to see it, and the
    campaign writes one per role per iteration for exactly this reason.

    The parameter count is checked alongside it, because it is what catches the
    hash being taken over the wrong buffer rather than the parameters.
    """
    records = fingerprints(round_trip.root)
    assert records, f"{journal_path(round_trip.root)} holds no NetFingerprint"

    first = fingerprint_at(records, 0, TRAIN)
    second = fingerprint_at(records, 1, TRAIN)
    assert first is not None, "iteration 0 fingerprinted no 'train' net"
    assert second is not None, "iteration 1 fingerprinted no 'train' net"
    assert first.param_count == second.param_count > 0, (
        f"the two 'train' fingerprints cover {first.param_count} and "
        f"{second.param_count} parameters: they are not digests of the same net"
    )
    assert not second.same_net_as(first), (
        f"the trained net did not change between iterations 0 and 1 "
        f"(sha256 {first.sha256}): {STEPS_PER_ITERATION} optimiser steps moved "
        f"no weight, which is the frozen-trainer failure the fingerprint series "
        f"exists to make visible"
    )


def test_the_net_iteration_one_served_is_the_ema_iteration_zero_published(
    round_trip: RoundTrip,
) -> None:
    """The loop closing, in one equality.

    This is the round trip's load-bearing claim and the one no component test
    can make. Iteration 0 trains a net and publishes a checkpoint; the EMA is
    reconciled from it; iteration 1's spawned workers are pointed at that EMA
    through ``KOMUGI_CAMPAIGN_CKPT`` and play their games with it. Each of those
    three links is separately tested elsewhere. What is tested here is that they
    are the *same* net all the way along -- and the campaign's own fingerprint
    series can say so, because ``record_fingerprints`` digests the ``self_play``
    role from the file the environment variable actually named, not from
    whatever local the loop meant to put there.

    Iteration 0 has no ``self_play`` fingerprint by design: it serves freshly
    initialised weights, ``CHECKPOINT_ENV`` is empty, and there is no file to
    digest. That absence is asserted too -- a ``self_play`` record at iteration
    0 would mean the loop had served *something* it never wrote down.
    """
    records = fingerprints(round_trip.root)

    assert fingerprint_at(records, 0, SELF_PLAY) is None, (
        "iteration 0 recorded a 'self_play' fingerprint, but it serves freshly "
        "initialised weights from an empty CHECKPOINT_ENV: there is no file to "
        "have digested"
    )

    published = fingerprint_at(records, 0, EMA)
    served = fingerprint_at(records, 1, SELF_PLAY)
    assert published is not None, "iteration 0 published no EMA fingerprint"
    assert served is not None, (
        "iteration 1 recorded no 'self_play' fingerprint: its workers were "
        "handed no checkpoint, so they played with random weights"
    )
    assert served.same_net_as(published), (
        f"iteration 1 served sha256 {served.sha256} but iteration 0 published "
        f"EMA sha256 {published.sha256}: the checkpoint the loop wrote is not "
        f"the net the next iteration's workers played with, which is the v1 "
        f"frozen-self-play loop"
    )


# ---------------------------------------------------------------------------
# 5. the telemetry journal
# ---------------------------------------------------------------------------


def test_the_journal_carries_both_writers_records_for_every_iteration(
    round_trip: RoundTrip,
) -> None:
    """A campaign whose history cannot be read off disk cannot be audited.

    Two records per iteration, from two writers, and both are wanted: the
    trainer's carries the loss extras and honest zeros for the game counts it
    never saw, and the loop's own carries the self-play numbers and the
    boundary gates' verdicts. They are told apart by
    :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER` -- an explicit key, not
    ``games > 0``, because an iteration whose self-play produced nothing still
    writes a campaign record and that is precisely when the journal is read in
    anger.

    ``skipped`` is asserted zero as well. ``read_records`` tolerates a
    crash-truncated line by counting it, and nothing here crashed, so a non-zero
    count means the journal is being written malformed.
    """
    journal = journal_path(round_trip.root)
    assert journal.is_file(), f"{journal} does not exist"

    read = read_records(journal, IterationMetrics, missing_ok=True)
    assert read.skipped == 0, (
        f"{journal} holds {read.skipped} line(s) that do not decode as records"
    )

    records = list(read.records)
    campaign_records = [record for record in records if is_campaign_record(record)]
    trainer_records = [record for record in records if not is_campaign_record(record)]

    for label, subset in (("campaign", campaign_records), ("trainer", trainer_records)):
        covered = {record.iteration for record in subset}
        assert covered >= set(range(ITERATIONS)), (
            f"the {label}'s records cover iterations {sorted(covered)}, want at "
            f"least {list(range(ITERATIONS))}"
        )


def test_the_campaign_record_reports_the_games_that_were_played(
    round_trip: RoundTrip,
) -> None:
    """The loop's own record carries the self-play count the trainer never saw.

    The trainer honestly reports ``games=0`` -- it is handed a window, not a
    ``RunSummary`` -- so without the loop's second record the games an iteration
    played would exist only in the list ``run_campaign`` returns, which is
    exactly what a campaign that dies never returns.
    """
    by_iteration = {
        record.iteration: record
        for record in iteration_metrics(round_trip.root)
        if is_campaign_record(record)
    }
    for result in round_trip.results:
        record = by_iteration.get(result.iteration)
        assert record is not None, (
            f"iteration {result.iteration} left no campaign record in "
            f"{journal_path(round_trip.root)}"
        )
        assert record.games == int(result.games), (
            f"iteration {result.iteration} played {result.games} games but its "
            f"journal record says {record.games}"
        )


# ---------------------------------------------------------------------------
# 6. the checkpoints -> a rating
# ---------------------------------------------------------------------------


def test_the_eval_core_rates_the_checkpoints_the_campaign_produced(
    round_trip: RoundTrip, rated: ladder.LadderResult
) -> None:
    """The last link: the trained artifacts are entrants a ladder can rate.

    The ids are the steps on disk, oldest first, which is what makes a ladder
    readable as a training curve -- and what would silently break if
    ``ladder_players`` ever sampled a newest-first listing.
    """
    steps = [ref.step for ref in checkpoint_refs(checkpoint_dir(round_trip.root))]
    assert rated.players == tuple(f"ckpt-{step}" for step in steps), (
        f"the ladder rated {list(rated.players)}, but {checkpoint_dir(round_trip.root)} "
        f"holds checkpoints at steps {steps}"
    )
    oldest = f"ckpt-{steps[0]}"
    assert rated.anchor == oldest, (
        f"the fit is anchored on {rated.anchor!r}, want the oldest rung "
        f"{oldest!r} -- anchoring on the newest would move the whole table every "
        f"time a checkpoint was written"
    )


def test_the_rating_is_a_finite_number_for_every_produced_player(
    rated: ladder.LadderResult,
) -> None:
    """Every rung gets a rating, and every rating is a real number.

    Finiteness, and nothing about the ordering. Whether iteration 1's net
    outrates iteration 0's after four optimiser steps at batch 8 is a question
    about search noise, and asserting an answer to it would make this test fail
    at random. What the eval core must do is *return a measurement*: a NaN or an
    infinity here means the Davidson fit diverged on real data, which is a
    defect in the rating stack rather than a fact about the nets.

    ``converged`` is deliberately **not** asserted. A two-rung ladder in which
    one side swept its only neighbour is not identified, which is the normal
    early-training case; ``fit_elo`` then returns clamped numbers and
    ``LadderResult.report`` says so. Failing here would be failing on the data.
    """
    assert set(rated.ratings) == set(rated.players), (
        f"the fit rated {sorted(rated.ratings)} but the ladder ran "
        f"{sorted(rated.players)}"
    )
    for player, rating in rated.ratings.items():
        assert math.isfinite(rating), f"{player} is rated {rating}"
    assert math.isfinite(rated.fit.draw_param), (
        f"the Davidson draw parameter came back {rated.fit.draw_param}"
    )
    assert rated.fit.ratings is rated.ratings, (
        "LadderResult.ratings must be the fit's own table, not a copy that "
        "could drift from it"
    )


def test_the_games_the_ladder_played_pool_into_one_auditable_match_set(
    rated: ladder.LadderResult,
) -> None:
    """Real games, correctly paired -- the evidence under the rating.

    A ladder over ``n`` rungs plays ``n - 1`` adjacent matchups, each of
    ``games_per_matchup`` games as seat-swapped pairs. The audit
    (:func:`~komugi.eval.seats.check_paired_books`) is what certifies the
    pairing: a dangling ``pair_id``, an unswapped seat or a doubled game count
    all mean the seat bias the pairing exists to remove is still in the fit.

    ``mean_plies > 1`` is the assertion that these were *games* and not
    immediately-adjudicated positions -- the cheapest way to catch an arena that
    wired up a terminal state and rated the result of nothing.
    """
    expected_matchups = ITERATIONS - 1
    assert len(rated.per_matchup) == expected_matchups, (
        f"a {ITERATIONS}-rung chain is {expected_matchups} matchup(s), got "
        f"{len(rated.per_matchup)}"
    )

    pooled = MatchSet.of(rated.records)
    assert len(pooled) == expected_matchups * LADDER_GAMES
    assert check_paired_books(pooled) == (), (
        f"the pooled match set fails its paired-book audit: "
        f"{check_paired_books(pooled)}"
    )
    assert len(pooled.pair_ids()) == expected_matchups * (LADDER_GAMES // 2), (
        "each seat-swapped pair must contribute exactly one pair id"
    )
    # A set, because ``MatchSet.players`` answers sorted and ``LadderResult``
    # answers in ladder order; the claim is that the same entrants appear, not
    # that two functions with different ordering contracts agree by accident.
    assert set(pooled.players()) == set(rated.players), (
        f"the pooled games name {sorted(pooled.players())}, but the ladder ran "
        f"{sorted(rated.players)}"
    )

    for record in rated.records:
        assert record.result in VALID_RESULTS, (
            f"a game came back with result {record.result!r}"
        )
        assert record.book_id == arena.EMPTY_BOOK
    for summary in rated.per_matchup:
        assert summary.games == LADDER_GAMES
        assert summary.decisive + summary.draws == LADDER_GAMES, (
            f"{summary.ids} reports {summary.decisive} decisive + "
            f"{summary.draws} draws over {summary.games} games"
        )
        assert summary.mean_plies > 1.0, (
            f"{summary.ids} averaged {summary.mean_plies} plies: these were not "
            f"real games"
        )


def test_the_report_names_the_run_it_was_fit_from(
    rated: ladder.LadderResult,
) -> None:
    """The one artifact a human reads carries every rung and the anchor.

    The anchor is carried on the result rather than re-derived from "whichever
    rating is 0.0", because on a balanced ladder *every* rating is exactly 0.0 --
    and the report would then name the oldest rung as the gauge no matter what
    was configured, which is a false statement about the run in the one place
    anybody looks.
    """
    report = rated.report()
    assert f"ladder: {len(rated.players)} players" in report
    assert f"anchor {rated.anchor} at 0.0 Elo" in report
    for player in rated.players:
        assert player in report, f"{player} is missing from the report"


# ---------------------------------------------------------------------------
# 7. the premise
# ---------------------------------------------------------------------------


def test_everything_was_written_under_the_root_it_was_given(
    round_trip: RoundTrip,
) -> None:
    """Nothing escaped ``CampaignConfig.root``.

    A cheap premise check with real teeth: every assertion above looks *under
    the root*, so a loop that quietly wrote its checkpoints or its journal
    somewhere else -- a default directory, the CWD -- would make several of them
    vacuous rather than failing.
    """
    root = round_trip.root
    assert checkpoint_refs(checkpoint_dir(root)), f"no checkpoint under {root}/ckpt"
    assert checkpoint_refs(ema_dir(root)), f"no EMA checkpoint under {root}/ema"
    assert journal_path(root).is_file(), f"no journal under {root}"
    assert sorted(root.rglob(SHARD_GLOB)), f"no shard anywhere under {root}"


# ---------------------------------------------------------------------------
# 8. the same loop at the moves-left recipe (KOM-159 T9)
# ---------------------------------------------------------------------------

MLH_RECIPE_NAME: Final[str] = "mlh"
"""``CampaignConfig.loss_recipe``'s other legal value -- campaign 6's.

A literal, spelled as :data:`komugi.campaign.loop.RECIPE_MLH` spells it, for the
reason :data:`EMA_ROLE` above is one: the name is what a written-down campaign
config carries, so this module asks for campaign 6 the way an operator would
rather than by reaching for the constant the loop resolves it against.
"""

MLH_CAP_ROW_WEIGHT: Final[float] = 1.0
"""The moves-left cap-row arm this run trains at: the identity, not campaign 6's.

Deliberately **not** the default ``0.0``, and the reason is about this test's
scale rather than about the lever. At ``0.0`` every row of a game that reached
the turn cap is censored, so a one-game iteration whose single game happened to
hit the cap composes a zero-mass ``HEAD_MLH`` vector, the assembler omits the
head (``losses.py``), and ``loss_mlh`` would then be absent from the journal for
a reason that is about that one game rather than about the wire this section
tests. At ``1.0`` the multiplier is exactly ones, the head is always scored, and
the arithmetic of the censoring arm is what
``tests/train/behavior/test_mlh_row_weights.py`` is for.
"""

MLH_ITERATIONS: Final[int] = 1
"""One iteration, where the main run takes two.

Two is the smallest campaign that *has a boundary*, and every claim in sections
1-7 is about that boundary. None of this section's are: the trainer's
``loss_mlh``, the campaign record's two levers and the shard comparison below
are all properties of iteration 0. A second iteration would add a self-play leg
whose games are *legitimately* different between the recipes -- iteration 1
serves the EMA of a net the moves-left loss has already moved -- so it would cost
half this module's remaining budget to weaken the comparison below.
"""

MLH_LOSS_KEY: Final[str] = "loss_mlh"
"""What ``training_iteration`` journals the moves-left head's last-step loss as.

``step.py`` writes every scalar of :meth:`TrainStep.run`'s report under a
``loss_`` prefix, and the assembler keys the moves-left head ``mlh``
(:data:`komugi.train.losses.HEAD_MLH`), so the composed key is this. The design
text names ``loss_mlh_last_decile``; that is one of the three summaries
``step.py`` derives per scalar, all four of which are asserted below.
"""

MLH_LOSS_SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "loss_mlh_first_decile",
    "loss_mlh_last_decile",
    "loss_mlh_step_std",
)
"""KOM-005b's per-scalar summaries of the same head, which a stage gate reads."""

LOSS_RECIPE_IS_MLH_KEY: Final[str] = "loss_recipe_is_mlh"
MLH_UTILITY_LAMBDA_KEY: Final[str] = "mlh_utility_lambda"
"""The two keys the boundary stamps on **every** iteration's campaign record.

Design review F10: the one question campaign 6 has to be able to answer off the
journal alone is which iteration the search-side utility came on, so a key
written only when it was interesting would be a key whose absence meant both
"off" and "this campaign predates the field". Literals here for the reason
:data:`MLH_RECIPE_NAME` is one.
"""


def mlh_campaign_config(root: Path) -> CampaignConfig:
    """The main campaign's config, at the moves-left recipe, over `root`.

    One field of :func:`campaign_config` away from the run sections 1-7 assert
    about, plus the cap-row arm :data:`MLH_CAP_ROW_WEIGHT` explains and the
    iteration count :data:`MLH_ITERATIONS` explains. Everything the self-play
    half reads -- ``run_seed``, ``workers``, ``games_per_worker``, and the
    :data:`SELFPLAY` config handed to ``run_campaign`` -- is identical, which is
    what makes the shard comparison below a claim rather than a coincidence.
    """
    return CampaignConfig(
        iterations=MLH_ITERATIONS,
        run_seed=RUN_SEED,
        root=str(root),
        steps_per_iteration=STEPS_PER_ITERATION,
        batch_size=BATCH_SIZE,
        lr=LR,
        workers=WORKERS,
        games_per_worker=GAMES_PER_WORKER,
        device=DEVICE,
        loss_recipe=MLH_RECIPE_NAME,
        mlh_cap_row_weight=MLH_CAP_ROW_WEIGHT,
    )


@pytest.fixture(scope="module")
def mlh_round_trip(
    small_net_env: None, tmp_path_factory: pytest.TempPathFactory
) -> RoundTrip:
    """Run iteration 0 of the same loop at ``loss_recipe="mlh"``.

    Its own root, so nothing it writes can be mistaken for the main run's, and
    module-scoped for the same reason that one is: real games and a real spawned
    process tree.
    """
    root = tmp_path_factory.mktemp("round-trip-mlh")
    return RoundTrip(
        root=root, results=walled_run_campaign(root, mlh_campaign_config(root))
    )


def row_digest(directory: Path) -> str:
    """Digest every row of every shard under `directory`, field by field.

    Read back through :func:`readers_for` -- the production reader under the
    production schema -- rather than over the ``npz`` bytes, because two runs
    may legitimately differ in compression or in file boundaries while writing
    the same rows. What is digested is what the *trainer* would draw: each
    field's name, dtype and shape (so a dtype regression is visible, not merely
    a value one) followed by its bytes, in sorted field order, in reader order.
    """
    digest = hashlib.sha256()
    for reader in readers_for(directory):
        digest.update(f"rows:{len(reader)}\n".encode())
        for index in range(len(reader)):
            row = reader[index]
            for name in sorted(row):
                tensor = row[name].contiguous()
                digest.update(
                    f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode()
                )
                digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trainer_record_at(root: Path, iteration: int) -> IterationMetrics:
    """Return the **last** trainer-written record for `iteration` under `root`."""
    matched = [
        record
        for record in iteration_metrics(root)
        if record.iteration == iteration and not is_campaign_record(record)
    ]
    assert matched, (
        f"{journal_path(root)} holds no trainer record for iteration "
        f"{iteration}; the trainer writes one per training iteration and the "
        f"campaign marker is what tells it from the loop's own"
    )
    return matched[-1]


def campaign_record_at(root: Path, iteration: int) -> IterationMetrics:
    """Return the **last** loop-written record for `iteration` under `root`."""
    matched = [
        record
        for record in iteration_metrics(root)
        if record.iteration == iteration and is_campaign_record(record)
    ]
    assert matched, (
        f"{journal_path(root)} holds no campaign record for iteration "
        f"{iteration}"
    )
    return matched[-1]


def test_the_moves_left_recipe_puts_a_finite_mlh_loss_in_the_journal(
    mlh_round_trip: RoundTrip,
) -> None:
    """T9's load-bearing claim: the head the recipe enables was actually scored.

    ``LOSS_RECIPES[campaign.loss_recipe]`` is a one-line lookup, and a lookup
    that resolved to the wrong constant -- ``P0B_RECIPE`` under the name
    ``"mlh"`` -- is silent everywhere else: the campaign runs, checkpoints,
    publishes an EMA and journals a full set of loss scalars, and the only
    observable is that ``loss_mlh`` is missing from among them. So the
    assertion is on the *key's presence*, off the journal, after a real
    iteration over real shards.

    The three KOM-005b summaries are asserted beside the last-step value
    because they are what a staging gate reads, and they are computed in a
    different place (over the whole step history rather than from
    ``history[-1]``): a head reported by ``run`` but dropped from the
    summaries would leave a bring-up curve ungateable.

    Finiteness, not a value. Four steps at batch 8 over one game's rows is a
    smoke run; what must hold is that the number was measured at all, since a
    NaN here is a diverged step the campaign then carried on from.
    """
    extras = trainer_record_at(mlh_round_trip.root, 0).extras

    assert MLH_LOSS_KEY in extras, (
        f"the trainer's record for iteration 0 carries {sorted(extras)} and not "
        f"{MLH_LOSS_KEY!r}: the campaign ran at loss_recipe="
        f"{MLH_RECIPE_NAME!r}, so the assembler had the moves-left head on and "
        f"every step reported it"
    )
    assert math.isfinite(extras[MLH_LOSS_KEY]), (
        f"{MLH_LOSS_KEY} came back {extras[MLH_LOSS_KEY]}: a non-finite "
        f"moves-left loss is a diverged step, and the campaign trained on"
    )
    for key in MLH_LOSS_SUMMARY_KEYS:
        assert key in extras, (
            f"the trainer reported {MLH_LOSS_KEY} but journalled no {key!r}; a "
            f"staged head's bring-up gate reads the decile means, so a curve "
            f"nothing summarises cannot be gated"
        )
        assert math.isfinite(extras[key]), f"{key} came back {extras[key]}"


def test_the_campaign_record_names_the_recipe_and_the_search_side_lever(
    mlh_round_trip: RoundTrip, round_trip: RoundTrip
) -> None:
    """F10, off a real journal: both keys, both runs, at their configured values.

    The two runs together are the claim. On the moves-left run the flag is 1.0
    and on the P0b run it is 0.0, so neither a hard-coded constant nor a key
    that is only written when it is interesting would pass both -- and the P0b
    row is what makes the *absence* of the key a failure rather than a reading.

    ``mlh_utility_lambda`` is 0.0 on both because this module's
    :data:`SELFPLAY` leaves the search-side utility off: campaign 6 switches the
    recipe on first and the utility later, and the pair of keys is what makes
    which-iteration-which readable off one append-only file.
    """
    mlh = campaign_record_at(mlh_round_trip.root, 0).extras
    p0b = campaign_record_at(round_trip.root, 0).extras

    assert mlh.get(LOSS_RECIPE_IS_MLH_KEY) == 1.0, (
        f"the campaign record of a loss_recipe={MLH_RECIPE_NAME!r} run says "
        f"{LOSS_RECIPE_IS_MLH_KEY}={mlh.get(LOSS_RECIPE_IS_MLH_KEY)!r}: which "
        f"recipe trained an iteration is recorded nowhere else"
    )
    assert p0b.get(LOSS_RECIPE_IS_MLH_KEY) == 0.0, (
        f"the campaign record of a default-recipe run says "
        f"{LOSS_RECIPE_IS_MLH_KEY}={p0b.get(LOSS_RECIPE_IS_MLH_KEY)!r}; the "
        f"flag is written on every iteration's record, carrying zero when the "
        f"recipe is P0b"
    )
    for label, extras in (("moves-left", mlh), ("P0b", p0b)):
        assert extras.get(MLH_UTILITY_LAMBDA_KEY) == SELFPLAY.mlh_utility_lambda, (
            f"the {label} run's campaign record says "
            f"{MLH_UTILITY_LAMBDA_KEY}={extras.get(MLH_UTILITY_LAMBDA_KEY)!r}; "
            f"self-play ran at {SELFPLAY.mlh_utility_lambda!r}"
        )


def test_the_recipe_changed_the_trainer_and_nothing_upstream_of_it(
    mlh_round_trip: RoundTrip, round_trip: RoundTrip
) -> None:
    """The recipe's blast radius, in two halves that fail differently.

    **Upstream is untouched.** Iteration 0's self-play runs before any training
    has happened: the same seed, the same freshly initialised served net, the
    same worker and game counts, so the rows on disk must be *identical* --
    field for field, dtype for dtype. A recipe that reached the search, the
    recorder or the shard schema would be visible here as a different digest,
    and this is the only claim in the tree that could see it.

    **The trainer is not.** The moves-left term backpropagates through the
    trunk, and both runs draw the same batches in the same order from the same
    rows under the same ``run_seed``, so the trained nets cannot agree unless
    the recipe never reached the optimiser. That is the mutation the journal
    assertions above cannot see in full: a lookup resolving ``"mlh"`` to
    ``P0B_RECIPE`` would leave both fingerprints equal.

    The digest comparison assumes neither run restarted a shard: design D10
    lets a restarted worker replay its whole budget under a new incarnation,
    which legitimately changes the rows. At this scale that needs a ten-minute
    stall (:data:`LIVENESS_S`), and the assertion message says so rather than
    leaving a puzzling inequality.
    """
    mlh_shards = iteration_root(mlh_round_trip.root, 0)
    p0b_shards = iteration_root(round_trip.root, 0)

    mlh_rows = sum(len(reader) for reader in readers_for(mlh_shards))
    p0b_rows = sum(len(reader) for reader in readers_for(p0b_shards))
    assert mlh_rows == p0b_rows > 0, (
        f"iteration 0 wrote {mlh_rows} rows at loss_recipe={MLH_RECIPE_NAME!r} "
        f"and {p0b_rows} at the default: the recipe reaches the trainer only, "
        f"and iteration 0's self-play precedes any training at all. A different "
        f"count means one of the runs restarted a shard (design D10) or the "
        f"recipe reached the self-play half"
    )
    assert row_digest(mlh_shards) == row_digest(p0b_shards), (
        f"iteration 0's rows differ between the two recipes ({mlh_shards} vs "
        f"{p0b_shards}) while their counts agree. The recipe is the trainer's "
        f"alone and iteration 0 played before anything was trained, so either "
        f"it reached the search or the recorder, or one run's worker restarted "
        f"and replayed its budget under a new incarnation"
    )

    records = fingerprints(mlh_round_trip.root)
    trained = fingerprint_at(records, 0, TRAIN)
    baseline = fingerprint_at(fingerprints(round_trip.root), 0, TRAIN)
    assert trained is not None, "the moves-left run fingerprinted no 'train' net"
    assert baseline is not None, "the P0b run fingerprinted no 'train' net"
    assert trained.param_count == baseline.param_count > 0, (
        f"the two runs' 'train' fingerprints cover {trained.param_count} and "
        f"{baseline.param_count} parameters: they are not digests of the same "
        f"net, so their inequality below would say nothing about the recipe"
    )
    assert not trained.same_net_as(baseline), (
        f"training the same rows with the same seed under {MLH_RECIPE_NAME!r} "
        f"and under the default recipe produced the same net (sha256 "
        f"{trained.sha256}). The moves-left term is weighted 1.0 and "
        f"backpropagates through the trunk, so the recipe never reached the "
        f"optimiser: LOSS_RECIPES resolved the name to P0b's own constant"
    )
