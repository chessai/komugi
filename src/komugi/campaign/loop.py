"""The campaign: self-play, train, checkpoint, repeat -- and survive a crash.

Every piece below this module already worked in isolation. The orchestrator
plays games into shard directories
(:func:`~komugi.selfplay.orchestrate.run_selfplay`); the recency window mixes
recent iterations (:class:`~komugi.train.window.RecencyWindow`); one training
iteration descends and checkpoints
(:func:`~komugi.train.step.training_iteration`). What did not exist is the
*sequence* -- and the wave-10 review's "L15-shaped gap" is precisely the
observation that each half being green in isolation says nothing about the
chain. This module is the chain, and it makes four claims that the components
individually cannot.

1. A checkpoint per iteration, and they coexist
-----------------------------------------------
:func:`~komugi.train.step.training_iteration` files its checkpoint under
``global_step_base + steps``, and this loop passes
``iteration * steps_per_iteration``. Before that argument existed, every
iteration published under the *same* step number and
:func:`~komugi.train.checkpoint.save_checkpoint`'s same-step supersede unlinked
its predecessor: one file ever existed, and rollback to a previous generation
was impossible (wave-10 review F5). The step counter is global because the
directory is shared.

2. Adam's moments travel across the iteration boundary
------------------------------------------------------
Each iteration rebuilds the training net and its :class:`TrainStep` from the
newest checkpoint, restoring **both** ``load_state_dict`` calls -- the model's
and the optimiser's. Restoring only the weights is a resume that looks correct
and is not: AdamW's per-parameter first and second moments are half of what a
step computes from, so a moment-less resume takes ~ hundreds of steps of
mis-scaled updates at every iteration boundary while reporting a perfectly
ordinary loss. That is untested-but-promised #1 of the wave-10 review, and it
is the reason the rebuild happens through the checkpoint on *every* iteration
rather than by keeping one long-lived :class:`TrainStep` in memory: the
in-memory path would never exercise the resume, so a broken resume would only
be discovered by a crash.

3. Resume costs at most one iteration
-------------------------------------
:func:`run_campaign` inspects the checkpoint directory before it does anything
and continues from the iteration *after* the newest checkpoint's. Iterations
already played are not replayed -- their shards are still on disk and still in
the window -- so a machine that dies mid-campaign loses the iteration that was
running, not the campaign. The unfinished iteration's partial shard tree, if
the crash left one, is cleared before its re-run: a same-seed replay next to
the crashed attempt's rows would double-count the iteration in both the window
and the reported totals (wave-11 union review F3).

4. The recency schedule is finally wired
----------------------------------------
Design §4's window over the five most recent iterations, one
:class:`~komugi.train.window.IterationRef` per iteration whose source is a
:class:`~komugi.selfplay.records.ConcatSource` over that iteration's per-worker
shard directories (design D10 gives each worker its own, because a
:class:`~komugi.train.shards.ShardWriter` owns its directory alone). See
:class:`WindowSource` for how the window's *tilt* reaches
:func:`~komugi.train.step.training_iteration`'s uniform sampler without a
second stream of randomness.

Directory layout
----------------
Everything a campaign owns lives under ``root``::

    <root>/iter_<i>/shards/worker_<k>_r<n>/shard-*.npz   self-play output
    <root>/ckpt/checkpoint-<step>-<digest>.pt            one per iteration
    <root>/books/gen_<i>.json                            the draft book
                                                         iteration i's workers
                                                         read (KOM-014)
    <root>/journal.jsonl                                 append-only telemetry
    <root>/selfplay.jsonl                                one record per
                                                         iteration's self-play
    <root>/telemetry_archive/<log>.<mtime_ns>            each launch's copy of
                                                         the two logs above,
                                                         taken at the next
                                                         launch's start

``books/gen_<i>.json`` names the iteration the book is **for**, not the one it
was mined from: it is built at the end of iteration ``i-1`` (after training, off
self-play's critical path) from the newest qualifying iterations below ``i``,
and rebuilt at the top of ``i`` if it is missing or if ``i`` is the resume
boundary. Iteration 0 has no sources and therefore no book, which is a bookless
generation rather than an error.

The shard root is per-iteration because the orchestrator creates fresh
directories per incarnation *within* a run and would otherwise have to
distinguish this run's restart from last iteration's worker. One directory
level buys "iteration i's data is exactly ``iter_i``", which is what makes the
window a listing rather than bookkeeping that has to survive a crash.

Seeds
-----
Every iteration derives its own orchestrator ``run_seed`` from the campaign's
via :class:`numpy.random.SeedSequence` with the iteration in the spawn key
(:func:`iteration_seed`), the same construction
:func:`komugi.selfplay.records.game_seed` uses one level down. Reusing one seed
across iterations would make every iteration play the *same games* -- the
network changes, but the Gumbel noise and every other draw would not, and a
campaign that re-explored the identical trajectories would look like it was
learning slowly rather than like it was broken.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import os
import shutil
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import numpy as np
import torch

from komugi.campaign.factory import (
    CHECKPOINT_ENV,
    DEVICE_ENV,
    IN_PLANES,
    NUM_ACTIONS,
    new_net,
    serving_net,
)
from komugi.campaign import boundary as boundary_checks
from komugi.campaign import pinned, probe as probe_module
from komugi.campaign.pause import (
    CRASH_CLASS,
    CRASH_TRIPWIRE,
    INTERRUPT_CLASS,
    CampaignPaused,
    Mute,
    PauseRecord,
    apply_startup_state,
    expire_mutes,
    live_mutes,
    pause_path,
    read_pause,
    write_pause,
)
from komugi.campaign.game_stats import GameStats, game_stats
from komugi.campaign.search_profile import (
    ITERATION_PROFILE_FILENAME,
    MODE_OFF,
    SearchProfile,
    append_profile,
    canonical_json,
    latest_profile,
    profile_of,
    static_digest,
    static_fields,
)
from komugi.campaign.telemetry import (
    SelfPlayTelemetry,
    append_selfplay,
    archive_telemetry,
    selfplay_journal_path,
)
from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.net import KomugiNet
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import (
    Book,
    book_path_for,
    books_dir as _book_directory,
    build_book,
    load_book,
)
from komugi.selfplay.inference import InferenceConfig
from komugi.selfplay.orchestrate import OrchestratorConfig, run_selfplay
from komugi.selfplay.records import ConcatSource
from komugi.telemetry.journal import read_records, write_record
from komugi.telemetry.liveness import LivenessVerdict
from komugi.telemetry.records import IterationMetrics, NetFingerprint
from komugi.train.checkpoint import (
    Checkpoint,
    list_checkpoints,
    load_checkpoint,
    require_matching_plane_count,
)
from komugi.train.digest import digest_lo, parameter_keys
from komugi.train.ema import EMA_DECAY, reconcile_ema
from komugi.train.losses import LossAssembler, LossConfig
from komugi.train.shards import SHARD_GLOB, ShardReader
from komugi.train.step import TOTAL_KEY, StepConfig, TrainStep, training_iteration
from komugi.train.window import IterationRef, RecencyWindow, SampleSource, WindowConfig
from komugi.train.wiring import MLH4_RECIPE, MLH_RECIPE, P0B_RECIPE

if TYPE_CHECKING:  # pragma: no cover -- imported lazily below, see the note
    from komugi.campaign import gauntlet as gauntlet_module

# `komugi.campaign.gauntlet` is imported **inside** the two functions that use
# it, not here. It imports `komugi.eval.arena`, which imports
# `komugi.campaign` -- so a module-level import would make
# `import komugi.eval.arena` (a perfectly ordinary thing for the ladder or a
# test to do first) walk back into a half-initialised arena and fail. The
# gauntlet is reached once per cadence, so the deferred import costs nothing
# and the cycle is only ever closed when both modules are already whole.

CPU: Final[str] = "cpu"
CUDA: Final[str] = "cuda"
DEVICES: Final[frozenset[str]] = frozenset({CPU, CUDA})
"""The devices a campaign may run on. Nothing else is spelled anywhere."""

CHECKPOINT_DIRNAME: Final[str] = "ckpt"
EMA_DIRNAME: Final[str] = "ema"
"""Where the served net lives: ``<root>/ema``, a sibling of ``<root>/ckpt``.

Two directories rather than two roles in one, because the checkpoint loader
selects by *directory* and the whole KOM-022 serving change is "self-play reads
one directory, the trainer resumes from the other". A shared directory would
make the newest file the answer to both questions, which is exactly the bug
(the trainer descending from a smoothed net) the split exists to prevent.
"""

JOURNAL_NAME: Final[str] = "journal.jsonl"

EMA_PUBLISHED_STEP: Final[str] = "ema_published_step"
"""The step of the newest published EMA, stamped on every campaign record.

The EMA advance gate is the one boundary check that used to leave *nothing*
behind when it passed: :attr:`~komugi.campaign.boundary.BoundaryOutcome.verdicts`
has no reader anywhere in ``src/``, so a healthy gate was unobservable on disk
and a journal reader could neither count stalls nor tell whether the gate had
run at all (KOM-025 wire-in review, F6). Every other gate stamps a flag
(:data:`~komugi.campaign.boundary.BAND_VIOLATION`,
:data:`~komugi.campaign.boundary.FLAT_TRAINER_FLAG`,
:data:`~komugi.campaign.boundary.SPIKE_FLAG`); this one stamps the number it
decided on.

Read off the published **file**, not from the arithmetic that predicted it, so
that a record carrying a step the campaign's own numbering does not predict is
a readable symptom rather than a tautology. A failing boundary stamps
:data:`~komugi.campaign.pinned.EMA_PUBLISH_STALLED` beside it, so the stalls are
countable across a campaign the same way spikes are.
"""

DRAWS: Final[str] = "draws"
TURN_CAP_GAMES: Final[str] = "turn_cap_games"
CAP_RATE: Final[str] = "cap_rate"
MAX_PLIES: Final[str] = "max_plies"
"""The game-shape extras every campaign record carries (KOM-150).

:class:`~komugi.telemetry.records.IterationMetrics` has fixed slots for the
draw rate and the mean game length, and those two are now measured rather than
reported as zero -- but it has none for *how* the draws happened, and that is
the half that distinguishes a spiral from noise. A campaign whose draw rate
climbs while ``cap_rate`` climbs with it is shuffling to the engine's cap; one
whose draw rate climbs while ``cap_rate`` stays flat is drawing by rule. The
counts ride beside the rates because a rate over an iteration that played four
games is not a measurement, and only the count says so.
"""

CAP_P_MEAN: Final[str] = "cap_p_mean"
CAP_P_GE_03_SHARE: Final[str] = "cap_p_ge_03_share"
CAP_ADJUDICATED_GAMES: Final[str] = "cap_adjudicated_games"
"""The cap-label extras every campaign record carries (KOM-245, review F8).

``cap_rate`` above says how many games the engine had to referee; these three
say what those games were **labelled**. After the switch to material
adjudication a capped game is no longer ``(0, 1, 0)``, and K1 -- "the share of
cap games with ``p >= 0.3`` is 41% +- 10 in the first post-switch iterations"
-- is read off :data:`CAP_P_GE_03_SHARE` and off the stored targets it is
computed from, never off a script that recomputes material from the planes and
therefore cannot fail (review F4).

:data:`CAP_P_MEAN` is over ALL cap games (a plain draw counting 0), which is
the denominator the plan's 0.25 is stated on; :data:`CAP_ADJUDICATED_GAMES` is
the count beside it, because a share over an iteration with two cap games is
not a measurement and only the count says so. All three are ``0`` under
``cap_adjudication = off``, which is what the pre-switch baseline looks like.
"""

EARLY_SHU_RATE: Final[str] = "early_shu_rate"
MEDIAN_PLACEMENTS_W: Final[str] = "median_placements_w"
MEDIAN_PLACEMENTS_B: Final[str] = "median_placements_b"
"""The draft extras every campaign record carries (KOM-156b).

The draw spiral had a sibling nobody was measuring: the **draft** attractor.
Campaign 2 reproduced v1's -- by iteration 14 the median side placed four
pieces and played 終 -- and the diagnosis had to be made by censusing the
shards by hand, because no per-iteration record carried a placement count.
These three are that census, stamped where every other campaign gate is read
from.

:data:`EARLY_SHU_RATE` is the **ramp gate**: a draft-minimum curriculum is
relaxed a rung only while it stays low (v1 held it under 0.3), and the two
medians beside it are what say whether a low rate means the drafts are long or
merely unfinished. The rate rides in ``extras`` rather than as a count because
:attr:`~komugi.campaign.game_stats.GameStats.early_shu_games` travels to
``selfplay.jsonl`` as the count, and ``games`` is on both records.
"""

MLH_UTILITY_LAMBDA: Final[str] = "mlh_utility_lambda"
LOSS_RECIPE_IS_MLH: Final[str] = "loss_recipe_is_mlh"
"""The two MLH extras every campaign record carries (KOM-159, review F10).

Campaign 6 is two levers taken at two pre-registered moments -- the head's loss
from iteration 0, the search-side utility from a later one -- and until these
existed **neither was recorded anywhere a journal reader could find**:
:func:`~komugi.campaign.pinned.pinned_constants` stamps loop constants only, the
shard schema has no column for either, and a `SelfPlayConfig` is not persisted.
So a series of ``loss_mlh`` values carried no way to tell the iterations before
the switch from the ones after, which is the whole content of the 6a/6b split.

:data:`LOSS_RECIPE_IS_MLH` is a float flag rather than the recipe's name because
:class:`~komugi.telemetry.records.IterationMetrics` extras are floats; ``1.0``
is the moves-left *family* -- :data:`~komugi.train.wiring.MLH_RECIPE` and the
weight-4.0 arm :data:`~komugi.train.wiring.MLH4_RECIPE`, which differ only in
the head's weight (see :func:`recipe_is_mlh`) -- and ``0.0`` is every recipe
that leaves the head off. The
trainer's own record carries the lambda too (its
:class:`~komugi.train.step.StepConfig` is handed it): both are written because
the campaign record is the one every gate and every census filters on.
"""

WINDOW_FLOOR: Final[str] = "window_floor"
"""The recency-window floor extra, written only when ``CampaignConfig.window_floor``
is set (KOM-1114).

Unlike :data:`MLH_UTILITY_LAMBDA` above, this key is **not** stamped on every
record: ``None`` is "this campaign has never excluded an iteration from its
window," which is today's behaviour on every campaign that predates KOM-1114,
and a key present with value ``0.0`` would be indistinguishable from a floor
genuinely set to iteration 0. So the key's mere presence is the signal, and its
value is the floor itself -- the iteration a reader can intersect a loss or
disagreement series against to see only what trained after the exclusion.
"""

WINDOW_OLDEST: Final[str] = "window_oldest"
"""The oldest iteration actually sampled this boundary, written alongside
:data:`WINDOW_FLOOR` -- same gate, same KOM-1114b follow-up.

``window_floor`` is the floor a campaign was *asked* to respect; this is what
the realised window actually contained, read off the same
:func:`campaign_window_refs` call :func:`run_campaign` built the training
window from rather than recomputed here. The two agree whenever the floor is
the binding constraint, and differ -- `window_oldest` reading newer -- early
in a floored campaign's life, before enough post-floor iterations exist to
fill the window: exactly the under-full case :func:`campaign_window_refs`'s
own docstring describes as renormalising rather than erroring. Omitted
whenever :data:`WINDOW_FLOOR` is, for the same reason that key is omitted:
absence must stay "this campaign has never excluded an iteration," not a
floor of iteration 0.
"""

SEARCH_PROOF_ON: Final[str] = "search_proof_on"
SEARCH_DISCOVERY_ON: Final[str] = "search_discovery_on"
"""The two search-layer extras every campaign record carries (KOM-225).

Exactly :data:`LOSS_RECIPE_IS_MLH`'s argument, one lever later. Campaign 6
phase b turned KOM-221's proof layer and KOM-220's discovery layer on at
iteration 571 of a run whose earlier iterations played neither, and -- until
these existed -- a journal reader had no way to tell the two halves apart:
``extras`` are floats, the shard schema has no column for a mode, and a
`SelfPlayConfig` is not persisted. ``1.0`` is *on*, so every census can filter
a series down to the iterations that actually played the search it is about.

They are flags rather than mode names because the records are float-only; the
mode strings themselves, with the algorithm identity they name and the rest of
the search's knobs, live in ``<root>/search_profile.jsonl``
(:mod:`komugi.campaign.search_profile`), and these two are the index into it.
Written on EVERY iteration, including the ones that changed nothing: a key
present only when it is interesting is a key whose absence means both "off"
and "this campaign predates the field".
"""

RECIPE_P0B: Final[str] = "p0b"
RECIPE_MLH: Final[str] = "mlh"
RECIPE_MLH4: Final[str] = "mlh4"
LOSS_RECIPES: Final[Mapping[str, LossConfig]] = {
    RECIPE_P0B: P0B_RECIPE,
    RECIPE_MLH: MLH_RECIPE,
    RECIPE_MLH4: MLH4_RECIPE,
}
"""The recipes a campaign may name, and the constants they resolve to.

``"p0b"`` is campaigns 1 through 5, ``"mlh"`` is campaign 6 at moves-left
weight 1.0, and ``"mlh4"`` is KOM-159 §6's pre-registered fallback arm -- the
same recipe at weight 4.0. A closed mapping, so
:attr:`CampaignConfig.loss_recipe` is validated against it before anything
spawns and the trainer's recipe is a *named* production constant either way.
Adding an arm is adding a line here and a new constant in
:mod:`komugi.train.wiring` -- never an edit to an existing one, which is that
module's own rule: the curves campaigns 1-5 produced belong to
:data:`~komugi.train.wiring.P0B_RECIPE` and the name has to keep producing them.
"""


def recipe_is_mlh(loss_recipe: str) -> bool:
    """Whether ``loss_recipe`` names a recipe in the *moves-left family*.

    Both :data:`RECIPE_MLH` and :data:`RECIPE_MLH4` turn the moves-left head on;
    they differ only in its weight. :data:`LOSS_RECIPE_IS_MLH` is that family
    flag, so a journal reader filtering campaign 6 finds the weight-4.0 arm's
    iterations too, and the weight itself is read off the trainer's record.
    """
    return loss_recipe in (RECIPE_MLH, RECIPE_MLH4)

SHARDS_DIRNAME: Final[str] = "shards"
_ITERATION_DIR: Final[str] = "iter_{iteration}"
_WORKER_GLOB: Final[str] = "worker_*"

_FLOAT_KNOBS: Final[frozenset[str]] = frozenset(
    {"liveness_timeout_s", "drain_timeout_s", "first_shard_grace_s"}
)
_INT_KNOBS: Final[frozenset[str]] = frozenset({"max_shard_restarts"})
"""Which :class:`~komugi.selfplay.orchestrate.OrchestratorConfig` fields
``orchestrator_overrides`` may set: the liveness and patience knobs, and
nothing that decides *what* is played. ``run_seed``, ``workers``,
``games_per_worker`` and ``shard_root`` are the campaign's to derive, and an
override that could reach them would let a caller point two iterations at one
shard directory or replay a seed."""


@dataclass(frozen=True)
class CampaignConfig:
    """What a campaign is asked to do, and where it keeps its state.

    Args:
        iterations: How many self-play/train iterations the campaign runs to.
            A *target*, not a count of work to do: a resumed campaign runs
            only the iterations below this number it has not already
            completed (see :func:`resume_point`).
        run_seed: The campaign's entropy. Every iteration's orchestrator seed
            is derived from it (:func:`iteration_seed`) and it is stamped into
            :class:`~komugi.train.step.StepConfig` so a checkpoint identifies
            the run it came from.
        root: Directory the campaign owns entirely -- shards, checkpoints and
            journal (see the module docstring's layout).
        steps_per_iteration: Optimiser steps per training iteration. Also the
            stride of the global checkpoint step counter, which is what makes
            checkpoint step ``(i + 1) * steps_per_iteration`` readable back as
            "iteration ``i`` finished".
        batch_size: Rows per optimiser step.
        lr: AdamW's learning rate.
        workers: Self-play worker processes per iteration.
        games_per_worker: Games each worker plays per iteration.
        device: Where the *training* net lives, and what the serving factory
            is told through :data:`~komugi.campaign.factory.DEVICE_ENV`. One
            of :data:`DEVICES`; ``'cpu'`` by default because CI has no GPU.
        fast_policy_weight: KOM-036's pre-registered A/B arm, passed straight
            through to :class:`~komugi.train.step.StepConfig`. It is the ONE
            field two arms of that experiment differ in besides ``root``:
            ``0.125`` down-weights a fast row's policy and soft-policy loss,
            ``0.0`` discards it PCR-style, and ``1.0`` -- the default -- is
            today's behaviour. Self-play, seeds, budgets and the number of
            optimiser steps are identical across the arms, which is what makes
            "matched compute" structural rather than a discipline anyone has
            to maintain.
        cap_row_value_weight: KOM-160's pre-registered arm (campaign 4D),
            passed straight through to
            :class:`~komugi.train.step.StepConfig` in the same way and for the
            same reason: what the **WDL** loss of a drawn -- in practice
            turn-capped -- game's rows is multiplied by. ``0.25`` is the
            pre-registered value, ``0.6`` its fallback, and ``1.0`` -- the
            default -- is today's behaviour. Self-play, seeds and budgets are
            identical across 4A and 4D, so this float is the one variable.
        loss_recipe: Which :mod:`komugi.train.wiring` recipe the trainer runs,
            by name: ``"p0b"`` -- the default -- is
            :data:`~komugi.train.wiring.P0B_RECIPE`, every campaign through 5,
            and ``"mlh"`` is :data:`~komugi.train.wiring.MLH_RECIPE`, campaign
            6's, which differs from it in the moves-left head being on at
            weight 1.0 (KOM-159 D1). A **name**, not a
            :class:`~komugi.train.losses.LossConfig`: a campaign that could be
            handed an arbitrary recipe would make "which weights produced this
            curve" a property of a call site rather than of a named constant,
            which is the whole reason those constants exist and are never
            edited.
        mlh_cap_row_weight: KOM-159 D4's arm, passed straight through to
            :class:`~komugi.train.step.StepConfig` exactly as the two above
            are: what the **moves-left** loss of a turn-capped game's rows is
            multiplied by. ``0.0`` -- the default -- censors them, which is
            what campaign 6 launches with; ``1.0`` trains on them. It is
            independent of ``loss_recipe`` on purpose (a censoring arm is not a
            recipe, and the recipe does not decide which rows a head sees), and
            it is inert while the head is off.
        allow_profile_switch: Whether this launch may change the search
            profile of a root that already has one (KOM-225). ``False`` -- the
            default -- makes :func:`guard_search_profile` REFUSE a relaunch
            whose search or loss recipe differs from the last line of
            ``<root>/search_profile.jsonl``, naming the fields that differ. A
            relaunch is the one moment a long campaign's configuration can
            change by accident (an edited launcher, a flag dropped from a
            command line), and the damage is invisible: the rows keep arriving,
            in the same directories, produced by a different algorithm. Set it
            to ``True`` exactly when the change is *intended* -- campaign 6's
            570->571 switch is the case this exists for -- and the new profile
            is appended with ``from_iteration`` at the resume point, so the
            file says which iterations played which search. It is not sticky:
            it authorises the one launch it is set on.
        window_floor: KOM-1114's. ``None`` -- the default -- is today's
            behaviour, byte-identical: the recency window's :func:`window_refs`
            walk is unchanged, and nothing is added to the campaign record.
            Set to an iteration number, the training window never refills from
            iterations older than it -- the window renormalises over whatever
            post-floor iterations exist, exactly as an early campaign's short
            window already does over whatever few iterations exist -- and the
            campaign record carries the floor in ``extras``
            (:data:`WINDOW_FLOOR`) so a reader can tell which iterations
            trained under it. For excluding iterations that predate a shape or
            schema change from the window without starting a new root.

    Raises:
        ValueError: On a non-positive count, a negative seed, an empty root, a
            device outside :data:`DEVICES`, a ``fast_policy_weight``,
            ``cap_row_value_weight`` or ``mlh_cap_row_weight`` outside
            ``[0, 1]``, a ``loss_recipe`` that is not a known recipe name, or a
            ``window_floor`` that is negative or a ``bool``.
    """

    iterations: int
    run_seed: int
    root: str
    steps_per_iteration: int
    batch_size: int
    lr: float
    workers: int
    games_per_worker: int
    device: str = CPU
    fast_policy_weight: float = 1.0
    cap_row_value_weight: float = 1.0
    loss_recipe: str = RECIPE_P0B
    mlh_cap_row_weight: float = 0.0
    # KOM-225's, appended for the reason the fields above were: a dataclass's
    # fields are positional parameters. Not validated in `__post_init__` -- a
    # bool has no invalid value, and what it authorises is checked against the
    # root, at `guard_search_profile`, where the comparison lives.
    allow_profile_switch: bool = False
    # KOM-1114's, appended for the same reason `allow_profile_switch` was:
    # every existing positional construction stays valid. Not validated here
    # against the iteration being trained -- that number does not exist yet
    # at construction time -- so the check lives at the window's construction
    # (`campaign_window_refs`), where both numbers are in hand.
    window_floor: int | None = None

    def __post_init__(self) -> None:
        """Reject a campaign that could not run, before it spawns anything."""
        for field, count in (
            ("iterations", self.iterations),
            ("steps_per_iteration", self.steps_per_iteration),
            ("batch_size", self.batch_size),
            ("workers", self.workers),
            ("games_per_worker", self.games_per_worker),
        ):
            if count < 1:
                raise ValueError(f"{field} must be >= 1, got {count!r}")
        if not self.lr > 0.0:
            # `not >` rather than `<=` so a NaN learning rate is rejected here
            # rather than discovered as a net full of NaNs an hour in.
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        if self.run_seed < 0:
            raise ValueError(f"run_seed must be >= 0, got {self.run_seed!r}")
        if not self.root:
            raise ValueError("root must be a non-empty path")
        if self.device not in DEVICES:
            raise ValueError(
                f"device must be one of {sorted(DEVICES)}, got {self.device!r}"
            )
        if not 0.0 <= self.fast_policy_weight <= 1.0:
            # `not <=` rather than `<`/`>`, so a NaN arm is refused here rather
            # than discovered as a campaign whose policy loss is NaN from its
            # first optimiser step. StepConfig repeats the check because it is
            # constructible without a campaign; this one refuses before
            # anything is spawned.
            raise ValueError(
                f"fast_policy_weight must be in [0, 1], got "
                f"{self.fast_policy_weight!r}"
            )
        if not 0.0 <= self.cap_row_value_weight <= 1.0:
            # Same `not <=` form, same reason (KOM-160): a NaN arm refused
            # before anything is spawned rather than discovered as a campaign
            # whose WDL loss is NaN from its first optimiser step.
            raise ValueError(
                f"cap_row_value_weight must be in [0, 1], got "
                f"{self.cap_row_value_weight!r}"
            )
        if not 0.0 <= self.mlh_cap_row_weight <= 1.0:
            # And a third time (KOM-159 D4), for the third head-weight arm.
            raise ValueError(
                f"mlh_cap_row_weight must be in [0, 1], got "
                f"{self.mlh_cap_row_weight!r}"
            )
        if self.loss_recipe not in LOSS_RECIPES:
            # Before anything is spawned, and by name: a typo'd recipe that
            # fell back to P0b would run campaign 6 as campaign 5 with a
            # different root, and the null result would be unattributable.
            raise ValueError(
                f"loss_recipe must be one of {sorted(LOSS_RECIPES)}, got "
                f"{self.loss_recipe!r}"
            )
        if self.window_floor is not None and (
            isinstance(self.window_floor, bool) or self.window_floor < 0
        ):
            # `bool` is rejected explicitly because it is an `int` subclass
            # that would otherwise sail past `< 0`: `True` is iteration 1 and
            # `False` reads as a floor of iteration 0, neither of which is
            # what a caller who wrote `window_floor=True` meant (KOM-1114b).
            raise ValueError(
                f"window_floor must be >= 0, got {self.window_floor!r}"
            )


class IterationResult(NamedTuple):
    """What one campaign iteration produced.

    Measured, never predicted: ``games`` and ``rows`` come from the
    orchestrator's :class:`~komugi.selfplay.orchestrate.RunSummary`, which
    counts the shards on disk rather than the request, and the two losses are
    the trainer's own first and last reports.

    Attributes:
        iteration: Which iteration this was.
        games: Games completed by cleanly exited workers.
        rows: Rows readable in this iteration's shard directories.
        first_loss: Weighted total loss of the iteration's first step.
        last_loss: ... and of its last. The pair is the cheapest honest signal
            that an iteration trained rather than merely ran; a campaign-level
            curve is the journal's job.
        checkpoint_step: Global step the iteration's checkpoint was filed
            under, ``(iteration + 1) * steps_per_iteration``.
    """

    iteration: int
    games: int
    rows: int
    first_loss: float
    last_loss: float
    checkpoint_step: int


class ResumePoint(NamedTuple):
    """Where a campaign would start, and the checkpoint it would start from."""

    iteration: int
    checkpoint: Checkpoint | None


def checkpoint_dir(root: str | os.PathLike[str]) -> Path:
    """Return the one directory every iteration's checkpoint is published to."""
    return Path(root) / CHECKPOINT_DIRNAME


def ema_dir(root: str | os.PathLike[str]) -> Path:
    """Return the directory the EMA -- the net self-play serves -- is published to."""
    return Path(root) / EMA_DIRNAME


def books_dir(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/books``, a sibling of ``ckpt``, ``ema`` and ``iter_<i>``.

    Delegated to :func:`komugi.selfplay.book.books_dir` rather than re-derived:
    the builder (here) and the reader (a worker, through
    :attr:`~komugi.config.SelfPlayConfig.book_path`) must agree on the layout,
    and two functions spelling ``"books"`` separately is exactly how they would
    stop agreeing. It is restated here because this module's docstring is where
    the campaign's layout is documented, beside :func:`checkpoint_dir` and
    :func:`ema_dir` (KOM-014 §7).
    """
    return _book_directory(root)


def journal_path(root: str | os.PathLike[str]) -> Path:
    """Return the campaign's append-only telemetry journal."""
    return Path(root) / JOURNAL_NAME


def campaign_logs(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Return every telemetry log a relaunch must not be able to destroy.

    The root journal and its self-play sibling, in the order an archive lists
    them. One function rather than two literals at the call site, because
    "which files are the campaign's history" is the question
    :func:`~komugi.campaign.telemetry.archive_telemetry` is answering and a
    third log added later must not have to be remembered in two places.
    """
    return (journal_path(root), selfplay_journal_path(root))


def trainer_extras(
    journal: str | os.PathLike[str], iteration: int
) -> dict[str, float]:
    """Return the **trainer's** own record's extras for `iteration`, from disk.

    :func:`~komugi.train.step.training_iteration` journals one
    :class:`~komugi.telemetry.records.IterationMetrics` per iteration carrying
    every loss scalar of the last step, ``steps``, ``batch_size`` and -- since
    KOM-005b -- three whole-iteration summaries per scalar:
    ``loss_<name>_first_decile``, ``loss_<name>_last_decile`` and
    ``loss_<name>_step_std``. The campaign writes a *second* record for the
    same iteration (that function's docstring invites it), and until this
    existed the two halves never met: the staging ladder's condition 1 reads a
    decile mean off ``extras``, and the campaign record -- the one carrying the
    marker every campaign-level reader filters on -- did not have one.

    Read back **from the journal** rather than recomputed from the step
    history the loop is still holding. The decile mean and the population
    standard deviation are statistics over the whole iteration, and a second
    implementation of them here would be a second thing to get wrong: two
    numbers under one name, agreeing until the day a slice or a rounding
    differs, with no test able to say which is the campaign's. The trainer
    computed them at their source and fsynced them; this reads that.

    The trainer's record is the one for `iteration` **without**
    :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER` -- the explicit
    discriminator, not ``games == 0``, for the reason that constant's docstring
    gives. The **last** such record wins: an iteration that genuinely ran twice
    (the crash-before-checkpoint re-run, whose first attempt's checkpoint was
    never published) appends a second one, and the newest describes the
    attempt that survived, exactly as
    :func:`~komugi.campaign.boundary.latest_fingerprints` decides for the
    fingerprint series.

    An empty mapping when there is no such record -- a journal that does not
    exist yet, or a trainer that was not asked to write one. The campaign's
    record is still written, carrying everything the boundary measured; a
    missing trainer record must not cost the iteration its telemetry.
    """
    newest: IterationMetrics | None = None
    for record in read_records(journal, IterationMetrics, missing_ok=True).records:
        if record.iteration != iteration:
            continue
        if record.extras.get(boundary_checks.CAMPAIGN_MARKER, 0.0) == 1.0:
            continue
        newest = record
    if newest is None:
        return {}
    return dict(newest.extras)


def boundary_extras(
    trainer: Mapping[str, float],
    measured: Mapping[str, float],
    *,
    rows: int,
    pending_mute: Mute | None,
) -> dict[str, float]:
    """Merge one boundary's extras: `trainer` first, `measured` **overriding**.

    The precedence, and why it is this way round: the campaign record's
    *subject* is what the boundary measured -- its fingerprints
    (:data:`~komugi.campaign.boundary.BN_DIGEST_LO`), its flags, its
    ``rows`` -- and the trainer's numbers ride along so that the one record
    every campaign-level reader filters on
    (:data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`) also carries the loss
    scalars and KOM-005b's decile means. So `trainer` goes in first and
    `measured` writes over it, never the other way: on a collision the number
    the boundary observed at *this* boundary wins over one read back off
    another record.

    That the rule is unobservable today is the reason it is stated here rather
    than only at the call site. The two namespaces are disjoint -- but
    :data:`~komugi.campaign.boundary.LOSS_TAIL_MEDIAN` and
    :data:`~komugi.campaign.boundary.SPIKE_FLAG` sit *inside* the ``loss_``
    prefix :func:`~komugi.train.step.training_iteration` owns, so a loss head
    named ``tail_median`` or ``spike`` would collide -- and inverting the order
    would then feed :func:`~komugi.campaign.boundary.spike_baseline` the
    trainer's number across a restart, silently (KOM-025 wire-in review F4,
    mutant M11).

    Pure, and separated from :func:`_close_iteration` for exactly that reason:
    a merge whose precedence is only expressible by standing up a whole
    iteration boundary is a merge whose precedence no test pins.

    Args:
        trainer: The trainer's own record's extras for this iteration, as
            :func:`trainer_extras` read them back off the journal. Empty when
            there is no such record, which costs the campaign record nothing.
        measured: What this boundary measured -- today
            :attr:`~komugi.campaign.boundary.Fingerprints.extras`. Overrides
            `trainer` key for key.
        rows: Rows this iteration's self-play wrote, as counted on disk.
        pending_mute: The mute this boundary is applying, or ``None``. Step 7
            of the ack protocol is telemetry *about* the mute; the mute itself
            is the file written at step 5b, which is why losing these two keys
            to a crash costs a row and not the operator's ack.

    Returns:
        A fresh dict -- neither argument is mutated -- carrying the merge,
        :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`, ``rows`` and the two
        mute keys. Everything the gates stamp afterwards is added to it by
        :func:`_close_iteration`, which is where the order of the gates lives.
    """
    extras: dict[str, float] = dict(trainer)
    extras.update(measured)
    extras[boundary_checks.CAMPAIGN_MARKER] = 1.0
    extras["rows"] = float(rows)
    if pending_mute is not None:
        extras["mute_applied"] = 1.0
        extras["mute_until_iteration"] = float(pending_mute.until_iteration)
    return extras


def published_ema_step(ema_directory: str | os.PathLike[str]) -> float | None:
    """Return the newest published EMA's step, or ``None`` if there is none.

    What :data:`EMA_PUBLISHED_STEP` is stamped from. A float because that is
    what an :class:`~komugi.telemetry.records.IterationMetrics` extra is, and
    because the step numbering is far below float64's exact-integer range.

    ``list_checkpoints`` is newest-first (:func:`~komugi.train.ema.reconcile_ema`
    reverses it to replay ascending), so ``[0]`` is the file this boundary was
    supposed to have written -- the same end of the same listing
    :func:`ema_publish_verdict` decides on, deliberately, so the number the
    journal carries and the number the gate judged cannot drift apart.
    """
    published = list_checkpoints(ema_directory)
    if not published:
        return None
    return float(published[0].step)


def owed_ema_step(
    checkpoints: str | os.PathLike[str], predicted_step: int
) -> int:
    """Return the step this boundary's EMA owes: the train checkpoint's own.

    KOM-022 §2 publishes ``EMA_i`` at ``theta_i``'s step, so the step the EMA
    must carry is a property of a **file on disk** -- the newest checkpoint
    under ``<root>/ckpt`` -- and not of the loop's own arithmetic.
    `predicted_step` is that arithmetic (``(i + 1) * steps_per_iteration``), and
    it is used only when there is no train checkpoint at all to read.

    Why read rather than re-derive (KOM-025 wire-in review, F8): the gate is
    **unmutable**, so a wrong expectation does not degrade to a warning -- it
    stops a healthy campaign, or (the dangerous direction) passes a stalled one
    because both sides of the comparison were computed from the same
    expression. ``training_iteration`` files at ``global_step_base + steps``
    and the loop passes both, so the arithmetic *should* reproduce the file's
    step; keeping it as a cross-check in the message rather than as the
    comparand is what turns a disagreement between them into a diagnosable
    line instead of an invisible assumption.
    """
    published = list_checkpoints(checkpoints)
    if not published:
        return predicted_step
    return published[0].step


def ema_publish_verdict(
    ema_directory: str | os.PathLike[str],
    expected_step: int,
    *,
    predicted_step: int | None = None,
) -> LivenessVerdict:
    """Check the KOM-022 EMA invariant at an iteration boundary.

    §2 publishes ``EMA_i = d * EMA_{i-1} + (1 - d) * theta_i`` **after** the
    train checkpoint and at that checkpoint's step, so once
    :func:`~komugi.train.ema.reconcile_ema` has returned at the end of
    iteration ``i`` the newest file under ``<root>/ema`` must be the one at
    ``(i + 1) * steps_per_iteration``. Anything else means the recursion did
    not advance across this boundary: the published EMA is a previous
    iteration's, and the net self-play is about to serve is one generation (or
    more) stale.

    Why this and not only the hash gate. The run-length gate
    (:func:`~komugi.telemetry.liveness.self_play_net_advanced` at role
    ``ema``) does catch a stalled EMA -- but it needs **two** boundaries to see
    it, because a freeze is a statement about change and one digest is not a
    repetition. A publish that silently no-ops at iteration ``i`` therefore
    costs a whole further iteration of self-play against the stale net before
    anything stops, and that iteration is played, trained on and checkpointed.
    This is the same invariant read off the *step numbering* instead of the
    weights, which is available immediately and is one comparison of two
    integers. Both stay: this one is early and cheap, the hash gate is the one
    that does not care how a freeze was reached.

    Pure, like everything in :mod:`komugi.telemetry.liveness`, except for the
    directory listing it cannot avoid -- the caller supplies the step it
    expects, so the decision boundary is testable from a literal.

    Args:
        ema_directory: ``<root>/ema``, whose newest file is judged.
        expected_step: The step the EMA owes, read off the newest **train**
            checkpoint by :func:`owed_ema_step`. This is the comparand.
        predicted_step: The boundary's own arithmetic,
            ``(i + 1) * steps_per_iteration``, carried into the message as a
            cross-check and never compared against. When the two disagree the
            message says so: the file is authoritative, and a disagreement is
            itself worth reading (KOM-025 wire-in review, F8).
    """
    # The arithmetic, quoted rather than trusted: it is only ever a second
    # derivation of `expected_step`, and the interesting case is the one where
    # the two disagree.
    cross_check = (
        ""
        if predicted_step is None or predicted_step == expected_step
        else (
            f" (this boundary's own arithmetic predicts step "
            f"{predicted_step}, which the train checkpoint on disk does not "
            f"agree with; the file is what the EMA must track)"
        )
    )
    published = list_checkpoints(ema_directory)
    if not published:
        return LivenessVerdict(
            ok=False,
            reason=(
                f"no EMA checkpoint under {ema_directory} after this "
                f"iteration's publish: the served net was never written, so "
                f"self-play would replay the previous generation. The train "
                f"checkpoint this boundary owed an EMA for is at step "
                f"{expected_step}{cross_check}"
            ),
        )
    # `list_checkpoints` is newest-first (`reconcile_ema` reverses it to replay
    # ascending); the newest is the one this boundary was supposed to write.
    newest = published[0]
    if newest.step != expected_step:
        return LivenessVerdict(
            ok=False,
            reason=(
                f"the EMA did not advance across this boundary: the newest "
                f"published EMA is step {newest.step} ({newest.path.name}) and "
                f"this iteration's train checkpoint is step "
                f"{expected_step}{cross_check}. "
                f"EMA_i is published after theta_i and at its step (KOM-022 "
                f"§2), so self-play is about to serve a stale generation"
            ),
        )
    return LivenessVerdict(
        ok=True,
        reason=f"the EMA advanced to step {expected_step} at this boundary",
    )


def iteration_root(root: str | os.PathLike[str], iteration: int) -> Path:
    """Return the directory holding iteration `iteration`'s self-play output."""
    return Path(root) / _ITERATION_DIR.format(iteration=iteration)


def shard_root(root: str | os.PathLike[str], iteration: int) -> Path:
    """Return the ``shard_root`` the orchestrator writes iteration `i` under."""
    return iteration_root(root, iteration) / SHARDS_DIRNAME


def search_profile_extras(selfplay: SelfPlayConfig) -> dict[str, float]:
    """Return the two search-layer flags iteration records carry (KOM-225).

    A function rather than two lines inside :func:`_close_iteration` so the
    mapping from modes to floats is testable without standing up an iteration:
    the values are ``1.0``/``0.0``, and a flag inverted here would be a wrong
    answer to every census that filters campaign 6b's iterations out of
    campaign 6a's. See :data:`SEARCH_PROOF_ON`.
    """
    return {
        SEARCH_PROOF_ON: float(selfplay.proof_mode != MODE_OFF),
        SEARCH_DISCOVERY_ON: float(selfplay.discovery_mode != MODE_OFF),
    }


def guard_search_profile(
    root: str | os.PathLike[str],
    profile: SearchProfile,
    allow_switch: bool,
) -> Literal["appended", "unchanged", "switched"]:
    """Bind `root`'s search profile to `profile`, or refuse the launch.

    The resume guard (KOM-225 §2). Three outcomes, and the middle one is the
    common case:

    * ``'appended'`` -- the root has no profile yet, so `profile` becomes its
      first line. Note that this is also what a *legacy* root gets on its next
      launch: the line's ``from_iteration`` says which iterations it covers,
      and the ones below it carry no profile and must never be read as having
      played this one.
    * ``'unchanged'`` -- the stored profile's :func:`static_digest` matches, so
      nothing is written. A relaunch of an unchanged campaign leaves the file
      byte-identical however many times it happens, which is what makes the
      number of lines equal to the number of real changes.
    * ``'switched'`` -- the digests differ and `allow_switch` is set, so the
      new profile is appended and the file records both regimes.

    Called once per launch, right after :func:`resume_point`, before any
    iteration runs: a refusal must cost nothing, and an authorised switch must
    be on the platter before the first game it describes is played.

    Raises:
        ValueError: If the digests differ and `allow_switch` is ``False``,
            naming every field that differs (with both values) and
            ``allow_profile_switch``. This is the whole point of the guard: a
            relaunch that quietly changed the search would produce rows
            indistinguishable from the ones before it, in the same
            directories, and no later reader could separate them.
    """
    latest = latest_profile(root)
    if latest is None:
        append_profile(root, profile)
        return "appended"
    if static_digest(latest) == static_digest(profile):
        return "unchanged"
    if allow_switch:
        append_profile(root, profile)
        return "switched"
    raise ValueError(
        f"{Path(root)} last recorded a different search profile: "
        f"{_profile_diff(latest, profile)}. A resumed campaign may not change "
        f"the search or the loss recipe by accident -- set "
        f"allow_profile_switch=True to record the change deliberately, or "
        f"restore the configuration the recorded profile describes"
    )


def _profile_diff(stored: SearchProfile, wanted: SearchProfile) -> str:
    """Name the static fields `stored` and `wanted` disagree on, and how.

    Over the canonical dicts rather than the dataclass fields, so the fields
    compared are exactly the ones the digest is taken over: a field the digest
    blanks (``schema``, ``written_at``, ``main_sha``, ``from_iteration``)
    differs on every launch -- or, for ``schema``, on every deploy that bumps
    it -- and naming it would bury the one field that matters.

    Fields are compared as their *JSON renderings* -- the same rendering
    :func:`~komugi.campaign.search_profile.canonical_json` feeds the digest --
    so the diff and the digest agree by construction. Comparing the Python
    values instead would call ``-0.0`` different from ``0.0`` (and ``1`` from
    ``1.0``) where the digest calls them the same, and refuse a launch while
    naming no field at all. The values are still *reported* as themselves.
    """
    left = static_fields(stored)
    right = static_fields(wanted)
    differing = [
        key
        for key in sorted(right)
        if _rendered(left.get(key)) != _rendered(right.get(key))
    ]
    return ", ".join(f"{key}: {left.get(key)!r} -> {right[key]!r}" for key in differing)


def _rendered(value: object) -> str:
    """Return `value` as the digest sees it: canonical JSON."""
    return json.dumps(value, sort_keys=True)


def write_iteration_profile(root: str | os.PathLike[str], iteration: int) -> None:
    """Copy the profile in force under `root` into ``iter_<i>/``.

    So an iteration's rows and the provenance of the search that produced them
    travel together: a shard tree moved, archived or censused on its own still
    says which algorithm played it, which the root-level log cannot once it is
    separated from the tree.

    The copy is the canonical rendering of the profile *in force*, which is not
    always the bytes of any line on disk. After a relaunch that appended nothing
    on a build with a newer schema, the profile in force is the migrated older
    line: its ``schema`` stays the writer's and the newer keys carry their
    defaults, so the copy says both which build recorded the search and what
    that search was in today's vocabulary. The next switch appends a
    current-schema line, and the copy follows it from that iteration on.

    Silently does nothing when the root has no profile -- a legacy root, whose
    iterations genuinely have no recorded search -- because an invented
    ``off``/``off`` file would be a measurement nobody made.
    """
    profile = latest_profile(root)
    if profile is None:
        return
    directory = iteration_root(root, iteration)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ITERATION_PROFILE_FILENAME).write_text(
        canonical_json(profile) + "\n", encoding="utf-8"
    )


def iteration_seed(run_seed: int, iteration: int) -> int:
    """Return iteration `iteration`'s orchestrator seed, derived from `run_seed`.

    ``SeedSequence(run_seed, spawn_key=(iteration,))`` reduced to one 32-bit
    word, because :class:`~komugi.selfplay.orchestrate.OrchestratorConfig`
    takes an ``int`` and hands it on as
    :class:`~numpy.random.SeedSequence`'s entropy one level down. Derivation
    by spawn key rather than by ``run_seed + iteration``: adjacent seeds are
    what makes two runs correlated, and the whole point of a per-iteration
    seed is that iteration 4 is not a continuation of iteration 3's stream.

    A pure function of two integers -- no clock, no :func:`hash` -- so a
    resumed campaign derives exactly the seeds the crashed one would have.
    """
    if run_seed < 0:
        raise ValueError(f"run_seed must be >= 0, got {run_seed}")
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    child = np.random.SeedSequence(run_seed, spawn_key=(iteration,))
    return int(child.generate_state(1, dtype=np.uint32)[0])


def iteration_source(
    root: str | os.PathLike[str], iteration: int, *, retain_arrays: bool = True
) -> ConcatSource:
    """Return one iteration's samples: every worker directory read as one source.

    ``retain_arrays`` is passed through to every
    :class:`~komugi.train.shards.ShardReader`: retained readers hold the
    iteration's decompressed arrays (~GBs per production iteration -- see the
    reader's docstring) and serve rows at memory speed; unretained readers
    hold O(one shard) and re-read. :func:`window_refs` retains only the
    iterations the recency window can actually use (KOM-033 review F1), and
    with its ``stop_after`` bound does not call this function at all for
    iterations older than the window's last member (KOM-041): each call globs
    the iteration's worker directories and scans every shard header, so an
    unbounded walk costs the campaign one such scan per iteration it has ever
    run, on every training iteration.

    Design D10 gives each worker its own shard directory and design F5 gives
    :class:`~komugi.train.window.RecencyWindow` exactly one source per
    iteration; :class:`~komugi.selfplay.records.ConcatSource` is the join, and
    this function is where the campaign's layout is turned into it.

    Directories holding no readable shard are dropped rather than concatenated
    as empty: a wedged incarnation's empty directory is bookkeeping, not data
    (the same adjudication
    :func:`~komugi.selfplay.orchestrate.run_selfplay` makes about
    ``RunSummary.shard_dirs``), and a reader over one costs a wasted scan.

    The readers are built **without** an expected schema, so each adopts the
    schema of the first valid shard it finds and holds the rest of its own
    directory to it. Pinning
    :data:`komugi.selfplay.records.SCHEMA` here instead would make this
    function reject any shard set that is not production-shaped -- including
    the toy-shaped ones a test rig writes -- by silently skipping every shard
    and presenting an empty iteration, which is the least debuggable failure
    available. A genuine mix of two generations still fails loudly, one level
    up, where :func:`~komugi.train.shards.stack_samples` refuses to stack rows
    whose key sets disagree.
    """
    directory = shard_root(root, iteration)
    readers: list[SampleSource] = []
    if not directory.is_dir():
        return ConcatSource(readers)
    for worker_dir in sorted(directory.glob(_WORKER_GLOB)):
        if not worker_dir.is_dir() or not any(worker_dir.glob(SHARD_GLOB)):
            continue
        reader = ShardReader(worker_dir, retain_arrays=retain_arrays)
        if len(reader) > 0:
            readers.append(reader)
    return ConcatSource(readers)


def window_refs(
    root: str | os.PathLike[str],
    through: int,
    *,
    retain_last: int | None = None,
    stop_after: int | None = None,
) -> tuple[IterationRef, ...]:
    """Return one :class:`IterationRef` per iteration ``0..through`` with data.

    Every iteration the campaign has ever run, not just the ones this process
    played: after a resume the earlier iterations' shards are still on disk and
    still belong in the window. :class:`~komugi.train.window.RecencyWindow`
    does the truncating to its ``size`` most recent, which is the one place
    the §4 schedule is decided -- unless the caller declares that size here as
    ``stop_after``, below, in which case the listing stops where the window
    would have truncated anyway.

    Only the newest ``retain_last`` iterations with data are built with
    retained arrays (``None`` means :class:`WindowConfig`'s default size, so
    the retained set is exactly what the default window can use). Retaining
    everything would pin the campaign's whole decompressed history -- measured
    ~1.5-2.2 GB per production iteration, growing without bound -- for
    iterations the window provably never samples (KOM-033 review F1).

    ``stop_after`` bounds *construction* the way ``retain_last`` bounds
    retention. ``None`` walks every iteration ``through..0`` and builds a
    reader set for each -- one glob of the iteration's worker directories plus
    a header scan per shard, so a campaign pays O(campaign length) scans per
    training iteration and O(length^2) over its life (KOM-041, KOM-032 review
    F7). An ``N >= 1`` stops the newest-first walk as soon as ``N`` iterations
    *with data* have been found; empty iterations are skipped as ever and do
    not consume the budget, so the bound is on non-empty refs, not on
    directories examined. The result is still ascending by iteration.

    The equivalence is exact when ``N`` is the consuming window's ``size``:
    :meth:`~komugi.train.window.RecencyWindow.__init__` sorts the refs
    descending and keeps the first ``size`` of them holding at least one
    sample, and this function only ever returns non-empty refs -- so the ref
    at newest-first position ``size`` and beyond can never become a member,
    whatever it holds. A ``stop_after`` *smaller* than that ``size`` is a
    different window, not a cheaper one: the members past ``N`` would simply
    not exist, and the schedule would renormalise over what was left. That is
    why the bound is the caller's to declare rather than this function's to
    assume, and why the production call site in :func:`run_campaign` passes
    the same ``WindowConfig().size`` it hands to
    :class:`~komugi.train.window.RecencyWindow`. Callers who want eligibility
    to stay the window's decision over the whole history leave it ``None``.

    Raises:
        ValueError: If ``stop_after`` is given and below 1 -- a bound of zero
            is an empty window, which the window itself would reject one level
            later with a message about samples rather than about this argument.
    """
    if retain_last is None:
        retain_last = WindowConfig().size
    if stop_after is not None and stop_after < 1:
        raise ValueError(f"stop_after must be >= 1, got {stop_after!r}")
    newest_first: list[IterationRef] = []
    kept = 0
    for iteration in range(through, -1, -1):
        if stop_after is not None and kept == stop_after:
            break
        source = iteration_source(root, iteration, retain_arrays=kept < retain_last)
        if len(source) > 0:
            newest_first.append(IterationRef(iteration=iteration, source=source))
            kept += 1
    return tuple(reversed(newest_first))


def campaign_window_refs(
    root: str | os.PathLike[str],
    through: int,
    config: WindowConfig,
    *,
    floor: int | None = None,
) -> tuple[IterationRef, ...]:
    """The one call shape :func:`run_campaign` uses: both bounds from `config`.

    The KOM-041 equivalence -- that bounding construction with ``stop_after``
    yields the very window the unbounded listing would -- holds only when the
    bound is the ``size`` of the :class:`~komugi.train.window.RecencyWindow`
    the refs are handed to. This helper is where that invariant lives: one
    ``config`` supplies ``retain_last`` (what is worth holding decompressed,
    KOM-033 review F1) and ``stop_after`` (how far back to walk at all), the
    same ``config`` the caller passes to the window. Callers wanting the whole
    history call :func:`window_refs` directly and leave ``stop_after`` unset.

    ``floor`` (:attr:`CampaignConfig.window_floor`) drops every ref whose
    iteration is below it, after the newest-first walk above has already
    found them -- the walk itself is unbounded by ``floor``, so the KOM-041
    equivalence above is untouched and a ``None`` floor is the exact
    byte-identical call this function always made. The window built from what
    is left renormalises over it exactly as an early campaign's short window
    already renormalises over however few iterations exist (KOM-1114).

    Raises:
        ValueError: If ``floor`` is given and is greater than ``through`` --
            the iteration about to be trained. A floor beyond it would leave
            nothing in the window to train on, and the ValueError says so
            rather than letting :class:`~komugi.train.window.RecencyWindow`
            raise its own "empty window" error over a filtered-away ``refs``.
    """
    if floor is not None and floor > through:
        raise ValueError(
            f"window_floor {floor!r} is beyond iteration {through!r} about "
            "to be trained: a floor in the future would leave nothing to "
            "train on"
        )
    refs = window_refs(root, through, retain_last=config.size, stop_after=config.size)
    if floor is None:
        return refs
    return tuple(ref for ref in refs if ref.iteration >= floor)


class WindowSource:
    """A :class:`~komugi.train.window.RecencyWindow` seen as a flat source.

    :func:`~komugi.train.step.training_iteration` draws batches by sampling
    **indices uniformly** from a :class:`~komugi.train.window.SampleSource`,
    while the window expresses recency as a *probability* over iterations. The
    two do not compose directly, and the two obvious fixes are both wrong:

    * calling :meth:`~komugi.train.window.RecencyWindow.sample` from inside
      ``__getitem__`` would make the source's answer depend on a second
      generator, so a batch would no longer be a function of the index the
      trainer drew and ``rng_seed`` would stop describing the run (the
      determinism section of :mod:`komugi.train.step`);
    * ignoring the tilt and concatenating the iterations would train on a
      window whose mix is whatever self-play throughput happened to be, which
      is the coupling :mod:`komugi.train.window` exists to refuse.

    So the schedule is realised as an **index map** instead. Member ``k`` is
    given a contiguous block of ``quota[k]`` virtual indices, sized so the
    blocks are in the window's weight proportions, and a virtual index maps
    into that member's rows by ``row = local * len(source) // quota[k]``.
    Uniform sampling over the virtual range then reproduces the recency
    distribution exactly, up to integer rounding, with no randomness of its
    own.

    The scale is chosen so that ``quota[k] >= len(source_k)`` for every
    member -- the total is ``max(len_k / weight_k)`` rather than the sum of
    the lengths -- because a quota below a member's length would make part of
    that iteration's data *unreachable*: the map is monotone, so a quota of
    200 over 1000 rows would sample 200 of them and never the rest. With the
    default five-iteration schedule and equal-sized iterations the virtual
    range is about 8.2x the real one, which costs nothing: it is an integer
    range, and each row is simply reachable through several indices.

    Rows within a member appear either ``floor(q/n)`` or ``ceil(q/n)`` times,
    so the within-iteration draw is uniform to within one part in ``q``.
    """

    def __init__(self, window: RecencyWindow) -> None:
        """Lay out `window`'s members as blocks of a flat index range."""
        self._window = window
        members = window.members
        weights = window.weights
        # Every member is non-empty (the window drops empty ones) and every
        # weight is positive, so this maximum is finite and the quotas below
        # each cover their own member.
        scale = max(
            len(ref.source) / weight
            for ref, weight in zip(members, weights, strict=True)
        )
        quotas = tuple(
            max(len(ref.source), int(round(weight * scale)))
            for ref, weight in zip(members, weights, strict=True)
        )
        offsets = [0]
        for quota in quotas:
            offsets.append(offsets[-1] + quota)
        self._quotas = quotas
        self._offsets = tuple(offsets)

    @property
    def window(self) -> RecencyWindow:
        """Return the window this is a view of."""
        return self._window

    @property
    def quotas(self) -> tuple[int, ...]:
        """Return each member's slice of the virtual index range, newest first."""
        return self._quotas

    def __len__(self) -> int:
        """Return the size of the virtual index range, not the sample count.

        Deliberately not ``sum(len(member))``: this is the range
        :func:`~komugi.train.step.training_iteration` draws from, and it is
        the range that carries the tilt. The real sample count is
        ``len(source.window)``.
        """
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample virtual index `index` resolves to.

        Negative indices count from the end, matching every other source in
        the trainer; anything outside the range raises ``IndexError`` rather
        than wrapping, because a silent wrap would train on the wrong
        iteration.
        """
        total = len(self)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"index {index} out of range for {total} virtual slots")
        slot = bisect.bisect_right(self._offsets, index) - 1
        member = self._window.members[slot]
        local = index - self._offsets[slot]
        row = local * len(member.source) // self._quotas[slot]
        return member.source[row]


def resume_point(campaign: CampaignConfig) -> ResumePoint:
    """Return the iteration `campaign` would start at, and its checkpoint.

    The newest checkpoint that *verifies* decides -- damaged ones are skipped
    by :func:`~komugi.train.checkpoint.load_checkpoint` with a warning, so a
    torn write costs one iteration rather than the campaign. Its recorded
    ``iteration`` names the last iteration that completed, so the campaign
    continues at the one after it; if that metadata is missing or malformed
    the global step is the fallback, since the loop files iteration ``i`` at
    step ``(i + 1) * steps_per_iteration`` and the division inverts it.

    An empty (or absent) checkpoint directory is ``(0, None)`` -- a fresh
    campaign, not an error.
    """
    latest = load_checkpoint(checkpoint_dir(campaign.root))
    if latest is None:
        return ResumePoint(iteration=0, checkpoint=None)
    recorded = latest.meta.get("iteration")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded >= 0:
        return ResumePoint(iteration=recorded + 1, checkpoint=latest)
    return ResumePoint(
        iteration=latest.step // campaign.steps_per_iteration, checkpoint=latest
    )


def boundary_committed(root: str | os.PathLike[str], iteration: int) -> bool:
    """Return whether `iteration`'s **boundary** record is in the journal.

    The campaign record -- the one carrying
    :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`, written by
    :func:`_close_iteration` -- is the commit point of an iteration, and it is
    written *after* everything the boundary measures: the fingerprints, the
    probe, the gauntlet and the liveness verdict. So its presence is the one
    readable statement on disk that iteration ``i`` finished rather than merely
    checkpointed.

    The trainer's same-iteration record is explicitly **not** it (the marker is
    the discriminator, for the reason that constant's docstring gives): the
    trainer publishes its checkpoint and journals, and the iteration can still
    die in the whole boundary that follows.

    **Known residual: the record is written just BEFORE ``enforce``** (KOM-180
    impl review F5, and campaign-and-recovery CP-01 item (d)).
    :func:`_close_iteration` writes this record and then calls
    :func:`~komugi.campaign.boundary.enforce`, which is what turns an armed
    tripwire into a ``PAUSED``; between the two sits a full
    :func:`~komugi.train.checkpoint.load_checkpoint` -- seconds, every
    iteration. A kill inside that window leaves a root this function calls
    committed while the tripwire that armed at ``i`` (a loss spike, a band
    violation) never became a pause, and the next startup resumes past it
    silently: ``enforce`` for ``i`` is never run again. The window is small and
    the failure is *lossy rather than destructive* -- the armed condition is on
    the record, so it is readable after the fact -- which is why KOM-180 chose
    this marker anyway and KOM-212 recorded the residual rather than moving it.

    Moving the commit point is not a local edit: every root alive today
    (campaign 6 carries markers for iterations 0..108+) was written under this
    rule, so a marker written *after* ``enforce`` would make each of those
    iterations read as uncommitted and refuse the resume this design exists to
    protect. Closing it needs a second, additive marker plus a startup rule that
    treats its absence as "unknown, fall back to the campaign record" -- a
    schema change with its own migration, i.e. a ticket, not a fix in passing.
    """
    for record in read_records(
        journal_path(root), IterationMetrics, missing_ok=True
    ).records:
        if record.iteration != iteration:
            continue
        if record.extras.get(boundary_checks.CAMPAIGN_MARKER, 0.0) == 1.0:
            return True
    return False


def _newest_committed_checkpoint(
    root: Path, campaign: CampaignConfig, refused: Path
) -> str:
    """Return the newest checkpoint whose own iteration IS committed, or ``""``.

    The artefact the rollback refers to, and the one field an operator reading a
    ``PAUSED`` needs that the refused checkpoint's path does not give them:
    *which* file is the last judged state of this campaign. Both crash-pause
    sites in :func:`run_campaign` fill ``last_good_checkpoint`` from the
    iteration's own resume checkpoint; startup has no such binding, so it walks
    the directory instead (KOM-180 impl review F4).

    Newest-first, so the first committed hit is the answer. The iteration a
    listed file publishes comes from its step -- ``list_checkpoints`` returns
    :class:`~komugi.train.checkpoint.CheckpointRef` s, which carry no ``meta``
    -- by inverting the loop's own filing rule (iteration ``i`` files at
    ``(i + 1) * steps_per_iteration``), the same inversion
    :func:`resume_point` falls back to. `refused` is skipped by path rather
    than trusted to be uncommitted: it is the checkpoint this refusal is about,
    and if its metadata and its step ever disagreed about which iteration it
    publishes, naming it as the rollback target would send the operator back to
    the state being refused.

    ``""`` when no checkpoint's iteration is committed -- a root whose *first*
    iteration is the unfinished one, where the recovery is a fresh root rather
    than a rollback.
    """
    stride = campaign.steps_per_iteration
    if stride <= 0:
        # No stride, no inversion. This is one field of a diagnostic message on
        # the way to a pause: it does not get to raise ZeroDivisionError and
        # replace the diagnosis with an arithmetic error.
        return ""
    for ref in list_checkpoints(checkpoint_dir(root)):
        if ref.path == refused:
            continue
        iteration = ref.step // stride - 1
        if iteration >= 0 and boundary_committed(root, iteration):
            return str(ref.path)
    return ""


def _refuse_unfinished_boundary(
    root: Path, campaign: CampaignConfig, point: ResumePoint, served: str
) -> None:
    """Pause rather than advance past a checkpoint whose boundary never closed.

    :func:`resume_point` reads the newest checkpoint's ``iteration`` as "the
    last iteration that completed" and continues at the one after it. That
    inference is only sound if the iteration's *boundary* also ran, and the two
    are separate writes with a whole boundary between them -- the trainer
    publishes ``theta_i`` and journals, then the EMA, the fingerprints, the
    probe, the gauntlet and the liveness verdict happen, and only then is the
    campaign record written. A kill anywhere in that window (an operator ending
    a hung gauntlet, the OOM killer, a host reboot) leaves a published
    checkpoint for ``i`` with no boundary for it.

    Resuming there was silently destructive rather than merely lossy. The gap is
    **permanent**: :func:`~komugi.campaign.boundary.fingerprint_series_complete`
    walks every iteration and ``fingerprint_series`` is unmutable, so the
    resumed root pays a full iteration, pauses on the hole, and pauses again at
    every later boundary -- and nothing it recorded in between attributes the
    failure to the resume (review CP-01).

    So startup refuses, with the tripwire's own name
    (:data:`~komugi.campaign.pinned.UNFINISHED_BOUNDARY`) rather than a generic
    crash, and **never** synthesises the missing record or verdict: the
    boundary's measurements are of a state that no longer exists, and inventing
    them would put a fabricated liveness verdict in the one file the whole
    design treats as evidence.

    **The recovery, exactly** (KOM-211; an ``ACK`` is refused by
    :func:`~komugi.campaign.pause.apply_startup_state` and clears nothing):
    remove ``<root>/PAUSED``, remove the abandoned attempt's trainer checkpoint
    ``<root>/ckpt/checkpoint-<(i + 1) * steps_per_iteration>-*.pt`` **and** its
    EMA sibling ``<root>/ema/checkpoint-<same step>-*.pt``, and relaunch -- or
    fork the root at ``i - 1``. The EMA file has to go with the trainer's: left
    in place it is the newest published EMA, so the re-run of iteration ``i``
    plays its self-play against the *abandoned attempt's* smoothed net instead
    of ``EMA_{i-1}``, which silently changes the very recursion the
    displacement band is derived from -- and no gate notices, because the
    numbering is self-consistent. Both paths are named in the ``PAUSED``'s
    ``detail`` and in the message this raises, with the concrete step
    substituted, because the operator reading them is mid-incident.

    This runs at the very top of :func:`run_campaign`, before
    :func:`~komugi.train.ema.reconcile_ema`: a refused root must not be written
    to at all, and reconciling first published ``EMA_i`` for the iteration this
    function then declared unjudged -- which also made the rollback above
    dead-end in ``EmaError: EMA is ahead of the trainer`` (KOM-180 impl review
    F1).

    Raises:
        CampaignPaused: If the newest checkpoint's iteration has no campaign
            record. ``PAUSED`` is written first, so a watcher finds the marker
            rather than racing the exception.
    """
    checkpoint = point.checkpoint
    if checkpoint is None:
        return
    # `resume_point` starts *after* the newest checkpoint's iteration, both on
    # the metadata path and on the step-division fallback, so the iteration
    # under examination is always the one before the start.
    completed = point.iteration - 1
    if completed < 0 or boundary_committed(root, completed):
        return
    abandoned_step = (completed + 1) * campaign.steps_per_iteration
    # The two files the rollback removes, as globs an operator can paste: the
    # digest in a checkpoint's name is content-derived, so neither path can be
    # spelled exactly without reading the directory, and the step is padded to
    # nine digits exactly as `save_checkpoint` writes it.
    abandoned_train = checkpoint_dir(root) / f"checkpoint-{abandoned_step:09d}-*.pt"
    abandoned_ema = ema_dir(root) / f"checkpoint-{abandoned_step:09d}-*.pt"
    last_good = _newest_committed_checkpoint(root, campaign, Path(checkpoint.path))
    detail = (
        f"checkpoint {checkpoint.path} publishes iteration {completed}, but "
        f"{journal_path(root)} holds no campaign record for iteration "
        f"{completed} (an IterationMetrics carrying "
        f"{boundary_checks.CAMPAIGN_MARKER!r}): that iteration's boundary -- its "
        f"fingerprints, its measurements, its gauntlet and its liveness verdict "
        f"-- never ran. Resuming would enter iteration {completed + 1} on an "
        f"unjudged net and leave a permanent hole in the fingerprint series. "
        f"Recovery is a rollback to iteration {completed - 1} (a fresh root, "
        f"where that is -1): rm {pause_path(root)}, rm {abandoned_train}, "
        f"rm {abandoned_ema} -- then relaunch; or fork the root at iteration "
        f"{completed - 1} from "
        f"{last_good if last_good else 'no committed checkpoint (start fresh)'}. "
        f"The EMA file must go with the trainer checkpoint: left behind it is "
        f"the newest published EMA, so the re-run of iteration {completed} "
        f"would play self-play against the abandoned attempt's smoothed net "
        f"instead of EMA_{completed - 1}, silently changing the recursion the "
        f"displacement band is derived from, and no gate notices. This tripwire "
        f"cannot be acked -- an ACK naming it is refused with a PauseError and "
        f"clears nothing -- and no boundary record will be synthesised"
    )
    write_pause(
        root,
        PauseRecord(
            pause_class=CRASH_CLASS,
            paused_at_iteration=completed,
            tripwire=pinned.UNFINISHED_BOUNDARY,
            detail=detail,
            newest_checkpoint=str(checkpoint.path),
            newest_ema=served,
            last_good_checkpoint=last_good,
            written_at_step=abandoned_step,
        ),
    )
    raise CampaignPaused((), f"{root}: {detail}")


def _orchestrator_config(
    campaign: CampaignConfig,
    iteration: int,
    overrides: Mapping[str, float] | None,
) -> OrchestratorConfig:
    """Build iteration `iteration`'s orchestrator config, applying `overrides`.

    Only the liveness and patience knobs may be overridden
    (:data:`_FLOAT_KNOBS`, :data:`_INT_KNOBS`); an unknown key raises rather
    than being ignored, because a silently dropped ``liveness_timeout_s`` is a
    run that hangs for the default two minutes per stall and an operator who
    believes otherwise.
    """
    base = OrchestratorConfig(
        run_seed=iteration_seed(campaign.run_seed, iteration),
        workers=campaign.workers,
        games_per_worker=campaign.games_per_worker,
        shard_root=str(shard_root(campaign.root, iteration)),
    )
    if not overrides:
        return base
    # `dict[str, Any]`, not `dict[str, float | int]`: `dataclasses.replace`'s
    # keywords are typed field by field, so an unpacked mapping is checked
    # against *every* field's type -- `int` for `max_shard_restarts`, `str` for
    # `shard_root` -- and no single union satisfies them all. The `int`/`float`
    # coercions below are the real narrowing, and the allowlist above is what
    # decides which field each value can reach.
    changes: dict[str, Any] = {}
    for name, value in overrides.items():
        if name in _INT_KNOBS:
            changes[name] = int(value)
        elif name in _FLOAT_KNOBS:
            changes[name] = float(value)
        else:
            raise ValueError(
                f"orchestrator_overrides has unknown key {name!r}; only "
                f"{sorted(_FLOAT_KNOBS | _INT_KNOBS)} may be overridden -- the "
                f"seed, fan-out and shard root are the campaign's to derive"
            )
    # `replace` rather than **kwargs at construction so the defaults live in
    # exactly one place (the orchestrator's own dataclass) and __post_init__
    # validates the merged result.
    return dataclasses.replace(base, **changes)


def _restore_env(name: str, previous: str | None) -> None:
    """Put ``os.environ[name]`` back the way it was, including 'unset'."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def campaign_batch_fit(
    campaign: CampaignConfig,
    selfplay: SelfPlayConfig,
    inference: InferenceConfig,
) -> None:
    """Refuse a campaign whose batch ceiling cannot hold its own slots (F8).

    A campaign fans out to ``campaign.workers * selfplay.games_in_flight``
    inference slots -- one per lane, because an interleaved worker gives each
    game in flight its own segment and its own client -- and every one of them
    can hold a ticket at the same instant. An
    :class:`~komugi.selfplay.inference.InferenceConfig` whose ``max_batch``
    sits below that product does not fail: it silently caps *every* forward
    (:func:`~komugi.selfplay.inference._row_ceiling` takes the min), so the
    extra lanes buy queueing rather than batch width and the run is slower than
    the same box managed before interleaving was switched on -- with nothing
    downstream saying so. The default ``max_batch`` of 16 is already below the
    product at eight workers with two games in flight, which is production
    shape, so the quiet miss is the likely case and not the exotic one
    (KOM-040 b1 review F8, design D11' -- cross-config invariants need an
    enforcement point). An experiment driving
    :func:`~komugi.selfplay.orchestrate.run_selfplay` directly stays free to
    cap deliberately; the campaign is the long unattended run nobody is
    watching, so this is where the invariant is enforced, in the shape
    :func:`~komugi.selfplay.orchestrate.validate_wave_fit` set for the seam's
    other cross-config check.

    Raises:
        ValueError: Naming all three fields and the slot count they imply, so
            the reader knows which one to move.
    """
    slots = campaign.workers * selfplay.games_in_flight
    if inference.max_batch < slots:
        raise ValueError(
            f"InferenceConfig.max_batch={inference.max_batch} is below the "
            f"{slots} inference slots this campaign runs "
            f"(CampaignConfig.workers={campaign.workers} x "
            f"SelfPlayConfig.games_in_flight={selfplay.games_in_flight}): every "
            f"forward would be capped at {inference.max_batch} rows while "
            f"{slots} slots wait their turn, silently, for the whole campaign. "
            f"Raise max_batch to at least {slots}, or run fewer workers/lanes "
            f"(KOM-040 b1 review F8, design D11')"
        )


def _loaded_net(
    state: Mapping[str, torch.Tensor], device: str
) -> torch.nn.Module:
    """Return a net of the environment's shape holding `state`, in ``eval()``.

    ``eval()`` because every use of this is a *measurement*: a train-mode
    forward would update the net's BatchNorm running statistics from the probe
    set, which is a diagnostic silently editing the thing it measures.
    """
    net = new_net()
    net.load_state_dict(dict(state))
    net.to(device)
    net.eval()
    return net


def _net_for_resume(resume: Checkpoint | None) -> KomugiNet:
    """Return the training net, carrying `resume`'s weights when there is one.

    Two lines of :func:`run_campaign` live here rather than inline so the
    resume path's plane-count refusal is reachable from a test: the guard is on
    the Checkpoint the loader returned, before the weights go in (KOM-002 §5.3,
    review F4), and a guard nothing exercises is a guard that can be deleted
    without a gate noticing. ``resume_point`` has already read this
    checkpoint's meta for the resume *iteration*, so this is not the first
    thing that happens to a stale checkpoint -- benign, because it still fires
    ahead of any load.

    ``None`` is a fresh campaign, not an error: there is nothing to check and
    nothing to load.
    """
    net = new_net()
    if resume is not None:
        require_matching_plane_count(resume, net.config.trunk.in_planes)
        net.load_state_dict(resume.model_state)
    return net


def _probe_set(
    root: Path, campaign: CampaignConfig
) -> tuple[probe_module.ProbeSet | None, str]:
    """Return the frozen probe set and its digest, sampling it at iteration 0.

    Sampled **once**, from iteration 0's shard tree, and never resampled: the
    band gate's reference distribution must not move under the gate, or a band
    exit could not be attributed to the nets rather than to the data. What that
    costs is stated in :mod:`komugi.campaign.probe` rather than discovered
    later -- iteration 0 plays with fresh weights, so the reference positions
    are the ones random play reaches.
    """
    path = probe_module.probe_path(root)
    if path.is_file():
        return probe_module.load_probe(path), probe_module.probe_digest(path)
    source = iteration_source(root, 0, retain_arrays=False)
    if len(source) == 0:
        return (None, "")
    probe = probe_module.sample_probe(
        source,
        probe_module.PROBE_SIZE,
        probe_module.probe_seed(campaign.run_seed, probe_module.PROBE_DOMAIN),
    )
    digest = probe_module.write_probe(path, probe)
    probe_module.record_probe_digest(root, digest)
    return probe, digest


def _disagreement(
    root: Path,
    campaign: CampaignConfig,
    iteration: int,
    probe: probe_module.ProbeSet,
    trained: torch.nn.Module,
    ema_state: Mapping[str, torch.Tensor],
    previous_state: Mapping[str, torch.Tensor] | None,
) -> tuple[dict[str, float], bool]:
    """Measure the EMA/train disagreement, and say whether the EMA is an alias.

    Three quantities, all ``KL(pi_ema || pi_train)`` in the design's fixed
    direction (the generating net first, because that weights the regions the
    data actually comes from):

    * ``ema_train_kl_ref`` on the frozen probe set -- §3b's numerator;
    * ``train_step_kl`` = ``KL(pi_train_{i-1} || pi_train_i)`` on the same set
      -- the trainer's own one-iteration displacement, which is what makes the
      ratio scale-free;
    * ``ema_train_kl_fresh`` on rows sampled from *this* iteration's shards,
      whose ratio to the reference is §7's OOD statistic.

    Every derived scalar is guarded, because an unguarded one takes the
    campaign down while it is writing telemetry -- and the blanket pause
    handler then wedges it on a diagnostic that feeds no control law. A
    denominator below the noise floor omits the ratio instead of dividing by
    it; ``D_ref`` below the floor omits ``ood_ratio`` and writes
    ``ood_ratio_undefined`` instead. Omission, not a sentinel: a missing key is
    unambiguous, whereas a ``-1.0`` is a number a plotter will happily average.
    """
    extras: dict[str, float] = {}
    ema_net = _loaded_net(ema_state, campaign.device)
    reference, zeros = probe_module.policy_kl(ema_net, trained, probe)
    extras["ema_train_kl_ref"] = reference

    if previous_state is not None:
        previous_net = _loaded_net(previous_state, campaign.device)
        displacement, _ = probe_module.policy_kl(previous_net, trained, probe)
        extras["train_step_kl"] = displacement
        if displacement >= pinned.DENOMINATOR_FLOOR:
            extras["displacement_ratio"] = reference / displacement

    fresh_source = iteration_source(root, iteration, retain_arrays=False)
    if len(fresh_source) > 0:
        seed = probe_module.probe_seed(
            campaign.run_seed, probe_module.OOD_DOMAIN, iteration
        )
        fresh = probe_module.sample_probe(
            fresh_source, probe_module.PROBE_SIZE, seed
        )
        fresh_kl, _ = probe_module.policy_kl(ema_net, trained, fresh)
        extras["ema_train_kl_fresh"] = fresh_kl
        if reference < pinned.DENOMINATOR_FLOOR:
            extras["ood_ratio_undefined"] = 1.0
        else:
            extras["ood_ratio"] = fresh_kl / reference

    # The aliasing bug: an EMA update that bound the trainer's tensor by
    # reference instead of copying it agrees with the trainer *exactly*, on
    # every position. That check needs no model at all, which is why it pauses
    # on the first occurrence and can never be muted.
    return extras, zeros == len(probe)


def _anchor_for(
    root: Path,
    iteration: int,
    param_keys: Sequence[str],
    selfplay: SelfPlayConfig,
    records: Sequence[IterationMetrics],
) -> gauntlet_module.AnchorPin | None:
    """Return the anchor this iteration's gauntlet plays, pinning one if needed.

    Generation 0's anchor is ``theta_0`` -- at iteration 1 there is no middle,
    and a randomly-seeded campaign's first checkpoint is the only fixed
    reference that exists. Re-pinning is pre-registered: two consecutive
    gauntlets at or above :data:`~komugi.campaign.pinned.REPIN_SCORE`
    **and** an anchor at least
    :data:`~komugi.campaign.pinned.MIN_ANCHOR_AGE` iterations old. Without the
    age floor the steady state is an anchor eight iterations behind, re-pinned
    every eight -- a rolling anchor at coarser granularity, under which "score
    below floor" means a different thing every generation and no floor could be
    pre-registered at all.
    """
    from komugi.campaign import gauntlet as gauntlet_module

    pins = gauntlet_module.read_pins(root)
    if not pins:
        oldest = list_checkpoints(checkpoint_dir(root))
        if not oldest:
            return None
        return gauntlet_module.pin_anchor(
            root,
            0,
            iteration,
            oldest[-1].path,
            param_keys,
            m=selfplay.gumbel_m_fast,
            n=selfplay.gumbel_n_fast,
            games=pinned.GAUNTLET_GAMES,
            floor=pinned.GAUNTLET_FLOOR,
            draw_score=selfplay.draw_score,
        )

    current = pins[-1]
    scores = [
        record.extras[boundary_checks.GAUNTLET_SCORE]
        for record in sorted(records, key=lambda item: item.iteration)
        if record.extras.get(boundary_checks.GAUNTLET_ANCHOR_GEN)
        == float(current.generation)
        and boundary_checks.GAUNTLET_SCORE in record.extras
    ]
    old_enough = iteration - current.pinned_at_iteration >= pinned.MIN_ANCHOR_AGE
    saturated = len(scores) >= 2 and all(
        value >= pinned.REPIN_SCORE for value in scores[-2:]
    )
    if not (old_enough and saturated):
        return current
    newest_ema = load_checkpoint(ema_dir(root))
    if newest_ema is None:
        return current
    return gauntlet_module.pin_anchor(
        root,
        current.generation + 1,
        iteration,
        newest_ema.path,
        param_keys,
        m=current.m,
        n=current.n,
        games=current.games,
        floor=current.floor,
        draw_score=current.draw_score,
    )


def _close_iteration(
    root: Path,
    campaign: CampaignConfig,
    selfplay: SelfPlayConfig,
    iteration: int,
    param_keys: Sequence[str],
    *,
    served: str,
    resume: Checkpoint | None,
    trained: torch.nn.Module,
    history: Sequence[Mapping[str, float]],
    games: int,
    rows: int,
    stats: GameStats,
    mutes: Mapping[str, Mute],
    previous_bn: float | None,
    pending_mute: Mute | None,
    results: Sequence[object],
    window_oldest: int | None = None,
) -> float:
    """Publish the EMA, record everything, run the gates, and stop if one fired.

    The iteration boundary, in order, and the order is the design's:

    1. **publish the EMA** -- after the train checkpoint, never before, which is
       what makes "EMA ahead of the trainer" an impossible state rather than an
       ambiguous one;
    2. **fingerprint** all three roles into the journal;
    3. **measure** the EMA/train disagreement on the frozen probe set and on
       this iteration's fresh rows;
    4. **gauntlet**, on the cadence;
    5. **gate liveness**: the EMA's own step advance
       (:func:`ema_publish_verdict`, the KOM-022 §2 recursion read off the
       numbering, firing :data:`~komugi.campaign.pinned.EMA_PUBLISH_STALLED` --
       its own name, so a publish that did not happen is distinguishable on
       disk from a digest that stopped changing) and then the fingerprint
       series' run-length, completeness and cross-role gates. Both unmutable,
       both sited after the loss bounds so a diverged iteration pauses on its
       divergence rather than on a consequence;
    6. **write the campaign's own** :class:`~komugi.telemetry.records.IterationMetrics`
       -- a second record for the same iteration, which the trainer's own
       docstring explicitly invites, carrying the marker that tells the two
       apart and the trainer's ``extras`` carried across from its record
       (:func:`trainer_extras`, merged under :func:`boundary_extras`'s
       precedence);
    7. **enforce**: if a tripwire armed, ``PAUSED`` is on disk before the raise.

    Returns the BatchNorm digest of this iteration's train checkpoint (its low
    52 bits, the width the journal carries), so the next boundary can say
    whether data flowed through the net at all -- a series that answers a
    different question from the liveness digest and is wired to no gate.

    `window_oldest` is the oldest iteration actually in the training window
    `run_campaign` built this boundary with -- :data:`WINDOW_OLDEST`, written
    only alongside :data:`WINDOW_FLOOR` (KOM-1114b). ``None`` writes neither
    key, which is also what a caller with no window to report (every unit test
    that stands this function up directly over a planted root, rather than
    through `run_campaign`) gets by default.
    """
    checkpoints = checkpoint_dir(root)
    ema_directory = ema_dir(root)
    journal = journal_path(root)

    reconcile_ema(
        checkpoints,
        ema_directory,
        param_keys,
        decay=EMA_DECAY,
        steps_per_iteration=campaign.steps_per_iteration,
    )
    # The environment, not the local: `served` is what this loop *meant* to
    # publish, and the fingerprint's subject is what the workers were actually
    # told to load. The two agree in a correct loop
    # (test_weights_are_actually_served asserts the equality inside a run); a
    # loop that set the variable from another binding then fails the cross-role
    # invariant here instead of writing a journal that names an EMA no game was
    # ever played with (KOM-022 review F10, mutant M13).
    serving_env = os.environ.get(CHECKPOINT_ENV, "")
    fingerprints = boundary_checks.record_fingerprints(
        journal,
        iteration,
        checkpoints,
        ema_directory,
        serving_env,
        param_keys,
        previous_bn,
    )

    records = boundary_checks.campaign_records(journal)
    # The trainer's half of the iteration, carried across rather than
    # recomputed (:func:`trainer_extras`): the loss scalars, `steps`,
    # `batch_size` and KOM-005b's per-scalar decile means and step dispersion.
    # It goes in FIRST, so every key the boundary measured overrides it -- the
    # campaign's `loss_tail_median`, its fingerprints and its flags are this
    # record's subject, and the trainer's numbers ride along. That precedence
    # is :func:`boundary_extras`'s, stated and pinned there rather than here,
    # because a rule spelled only inside this function can only be tested by
    # standing up a whole boundary (review F4, mutant M11).
    extras: dict[str, float] = boundary_extras(
        trainer_extras(journal, iteration),
        fingerprints.extras,
        rows=rows,
        pending_mute=pending_mute,
    )

    if not records:
        extras.update(pinned.pinned_constants())
    else:
        # `check_first_record`, not `check_against_root`: the first campaign
        # record of a root is the one that stamped the constants, so a first
        # record carrying *none* of them is a root whose iteration-0 record was
        # lost -- and "no keys recorded" silently means "no drift" (KOM-022
        # review F9).
        pinned.check_first_record(records[0].extras)

    outcome = boundary_checks.BoundaryOutcome(extras=extras)

    # §5 bounds 1 and 2, in that order and both before the telemetry write.
    # The finiteness check comes first because `IterationMetrics` rejects a
    # non-finite extra: writing the tail median of a diverged iteration would
    # take the campaign down *inside* the telemetry write, and the blanket
    # handler would then file a `class: crash` marker on a diagnostic rather
    # than a `class: tripwire` one naming the divergence.
    totals = boundary_checks.step_totals(history, TOTAL_KEY)
    if not boundary_checks.evaluate_non_finite_loss(totals, outcome, mutes):
        tail_median = boundary_checks.loss_tail_median(history, TOTAL_KEY)
        extras[boundary_checks.LOSS_TAIL_MEDIAN] = tail_median
        boundary_checks.evaluate_spike(
            tail_median, iteration, records, outcome, mutes
        )

    probe, probe_sha256 = _probe_set(root, campaign)
    if probe is not None:
        extras["probe_sha256_lo"] = digest_lo(probe_sha256)
        measured, aliased = _disagreement(
            root,
            campaign,
            iteration,
            probe,
            trained,
            _newest_ema_state(ema_directory),
            resume.model_state if resume is not None else None,
        )
        extras.update(measured)
        if aliased and iteration >= 1:
            # From iteration 1, and not before. At iteration 0 the equality is
            # this design's own construction, not a bug: §2 publishes
            # `EMA_0 := theta_0` -- bit-identical, deliberately, so that
            # iteration 1 serves exactly what the pre-EMA campaign served and
            # smoothing first bites at iteration 2. An aliasing check that ran
            # at iteration 0 would pause every healthy campaign on its first
            # boundary, which is the false-alarm mode that gets a gate switched
            # off. By iteration 1 the EMA is `0.75*theta_0 + 0.25*theta_1` and
            # an exact zero really does mean the update bound the trainer's
            # tensors by reference.
            outcome.fire(
                pinned.EMA_ALIAS,
                "KL(pi_ema || pi_train) is exactly 0.0 on every probe "
                "position: the EMA is not a smoothed net but an alias of the "
                "trainer's own tensors",
                mutes,
            )
        if "train_step_kl" in extras:
            flat = boundary_checks.evaluate_flat_trainer(
                extras["train_step_kl"], iteration, records, outcome, mutes
            )
            if not flat:
                ratios = [
                    record.extras["displacement_ratio"]
                    for record in records
                    if "displacement_ratio" in record.extras
                ]
                if "displacement_ratio" in extras:
                    ratios.append(extras["displacement_ratio"])
                boundary_checks.evaluate_band(
                    ratios, iteration, records, outcome, mutes
                )

    if pinned.gauntlet_iteration(iteration):
        _run_gauntlet(
            root,
            campaign,
            selfplay,
            iteration,
            param_keys,
            records,
            outcome,
            mutes,
        )

    # The KOM-022 EMA invariant, on the step numbering, and the run-length
    # hash gate, on the weights -- in that order, and *after* the loss bounds
    # above so that a diverged iteration still pauses on its divergence rather
    # than on a consequence of it. Both are unmutable: a served net that stops
    # advancing is v1's catastrophe verbatim, and it has no false-alarm mode.
    #
    # The step the EMA owes is read off the newest **train checkpoint file**
    # (`owed_ema_step`), not re-derived: an unmutable gate whose two sides are
    # the same expression evaluated twice cannot fail, and the arithmetic
    # travels in the message as a cross-check instead (review F8).
    predicted_step = (iteration + 1) * campaign.steps_per_iteration
    advanced = ema_publish_verdict(
        ema_directory,
        owed_ema_step(checkpoints, predicted_step),
        predicted_step=predicted_step,
    )
    # What the gate saw, on the record, pass or fail: `outcome.verdicts` is
    # in-memory only, so without this a *passing* EMA gate left nothing on disk
    # and no reader could count stalls or tell that the gate had run (F6).
    step_published = published_ema_step(ema_directory)
    if step_published is not None:
        extras[EMA_PUBLISHED_STEP] = step_published
    if advanced.ok:
        outcome.verdicts.append(advanced)
    else:
        extras[pinned.EMA_PUBLISH_STALLED] = 1.0
        outcome.fire(pinned.EMA_PUBLISH_STALLED, advanced.reason, mutes)

    boundary_checks.evaluate_liveness(
        read_records(journal, NetFingerprint, missing_ok=True).records,
        iteration,
        outcome,
        mutes,
    )

    # The game shape, measured off the iteration's own sealed shards by
    # :func:`~komugi.campaign.game_stats.game_stats` and handed in (KOM-150).
    # Two of `IterationMetrics`' three fixed floats are now measurements; the
    # counts they are rates OF go in `extras`, because a rate without its
    # denominator cannot be read.
    extras[DRAWS] = float(stats.draws)
    extras[TURN_CAP_GAMES] = float(stats.turn_cap_games)
    extras[CAP_RATE] = stats.cap_rate
    extras[MAX_PLIES] = float(stats.max_plies)
    # How those cap games were LABELLED (KOM-245), off the same census: the
    # count, the mean p over every cap game, and K1's p >= 0.3 share.
    extras[CAP_ADJUDICATED_GAMES] = float(stats.cap_adjudicated_games)
    extras[CAP_P_MEAN] = stats.cap_p_mean
    extras[CAP_P_GE_03_SHARE] = stats.cap_p_ge_03_share
    # The draft, off the same census (KOM-156b). `IterationMetrics` has no
    # fixed slot for any of it, so all three ride in `extras` -- the rate
    # because it is what a curriculum ramp is gated on, the medians because a
    # rate without the shape behind it cannot say whether the drafts are long
    # or simply unfinished.
    extras[EARLY_SHU_RATE] = stats.early_shu_rate
    extras[MEDIAN_PLACEMENTS_W] = stats.median_placements_w
    extras[MEDIAN_PLACEMENTS_B] = stats.median_placements_b

    # KOM-159's two levers, on EVERY iteration's record and not only on the
    # ones that changed (design review F10). A key written only when it is
    # interesting is a key whose absence means two different things -- "off"
    # and "this campaign predates the field" -- and the whole point is that the
    # iteration the utility came on is readable off the journal alone.
    extras[MLH_UTILITY_LAMBDA] = float(selfplay.mlh_utility_lambda)
    extras[LOSS_RECIPE_IS_MLH] = float(recipe_is_mlh(campaign.loss_recipe))

    # KOM-1114's, only when set (see :data:`WINDOW_FLOOR`): unlike the two
    # levers above, absence here means "this campaign has never excluded an
    # iteration from its window," and that must stay distinguishable from a
    # floor of iteration 0.
    if campaign.window_floor is not None:
        extras[WINDOW_FLOOR] = float(campaign.window_floor)
        # The realised window's oldest member (KOM-1114b), not the floor
        # itself: see :data:`WINDOW_OLDEST`. `None` when the caller has no
        # window to report, which keeps this key absent exactly when
        # `window_oldest` was never in hand -- never when `window_floor` is
        # merely unset, since this whole branch is already gated on that.
        if window_oldest is not None:
            extras[WINDOW_OLDEST] = float(window_oldest)

    # KOM-225's two search-layer flags, on every record for exactly the reason
    # above: the iteration the proof and discovery layers came on is readable
    # off the journal alone, and the profile they index into is
    # `<root>/search_profile.jsonl`.
    extras.update(search_profile_extras(selfplay))

    write_record(
        journal,
        IterationMetrics(
            iteration=iteration,
            games=games,
            # The self-play detail an operator reads lives in selfplay.jsonl
            # (games, rows, restarts, wall); this record's subject is `extras`.
            # `draw_rate` and `avg_game_len` used to be reported as zero here
            # because the orchestrator does not measure them -- and the
            # previous architecture then died of a draw spiral nobody could
            # see. They are now the shard census's own numbers.
            # `policy_entropy` stays 0.0: nothing measures it. The search
            # writes `policy_target` per row and no consumer reduces it to an
            # entropy, so a number here would be an invention rather than a
            # measurement -- the same honesty the other two just stopped
            # needing.
            draw_rate=stats.draw_rate,
            avg_game_len=stats.mean_plies,
            policy_entropy=0.0,
            extras=extras,
        ),
    )

    newest = load_checkpoint(checkpoints)
    boundary_checks.enforce(
        root,
        outcome,
        iteration,
        results,
        newest_checkpoint=str(newest.path) if newest is not None else "",
        newest_ema=served,
        last_good_checkpoint=str(resume.path) if resume is not None else "",
        written_at_step=(iteration + 1) * campaign.steps_per_iteration,
    )
    return fingerprints.bn_lo


def _newest_ema_state(ema_directory: Path) -> Mapping[str, torch.Tensor]:
    """Return the newest EMA checkpoint's weights, or raise if there is none.

    The plane-count guard runs HERE rather than in :func:`_loaded_net`, because
    this is the last place the *checkpoint* still exists: ``_loaded_net`` takes
    a bare state dict, and by then the meta that records which plane map the
    weights were trained under has been dropped. Guarding on the returned
    object, before the ``load_state_dict`` in ``_loaded_net``, is the same rule
    every other load site follows (KOM-002 §5.3, review F3) -- without it a
    stale EMA checkpoint takes the KL probe down inside torch, mid-boundary,
    with a shape traceback that names neither the plane map nor the ticket.
    """
    newest = load_checkpoint(ema_directory)
    if newest is None:
        raise RuntimeError(
            f"no EMA checkpoint under {ema_directory} after publishing one: "
            f"the disagreement signal has no serving net to measure"
        )
    require_matching_plane_count(newest, IN_PLANES)
    return newest.model_state


def _run_gauntlet(
    root: Path,
    campaign: CampaignConfig,
    selfplay: SelfPlayConfig,
    iteration: int,
    param_keys: Sequence[str],
    records: Sequence[IterationMetrics],
    outcome: boundary_checks.BoundaryOutcome,
    mutes: Mapping[str, Mute],
) -> None:
    """Play the cadence's gauntlet and fold its verdict into `outcome`.

    Every scalar is guarded the way §7 requires: an **aborted** gauntlet (the
    anchor did not verify) and an **uninformative** one (fewer than
    :data:`~komugi.campaign.pinned.GAUNTLET_MIN_DECISIVE` decisive games) omit
    the score keys entirely and write their own flag instead, because a score
    that was never measured must not be a number a plotter can average.
    ``gauntlet_wall_s`` is written whenever games were played, including an
    uninformative gauntlet -- §6's cadence rule is set from that measurement.

    :class:`~komugi.campaign.gauntlet.GauntletAborted` is the one gauntlet
    failure folded into a verdict, because an unverifiable anchor is a statement
    about *this campaign's* files. The supervisor's two -- ``GauntletStalled``
    (a shard still running at the gauntlet's one absolute deadline) and
    ``GauntletLost`` (a shard that died without
    returning) -- are deliberately **not** caught here: they are infrastructure
    failures, not measurements, so they propagate out of ``_close_iteration``
    into ``run_campaign``'s boundary ``except Exception`` handler, which writes
    the ordinary ``close_iteration: `` crash ``PAUSED`` and re-raises (pinned by
    ``test_close_iteration_crash.py``'s
    ``test_a_stalled_gauntlet_writes_the_boundary_crash_pause``). That is
    the same treatment every other boundary exception gets, and it is what makes
    a hung gauntlet end as a marked stop rather than as a campaign that is still
    "running" days later (review CP-04).
    """
    from komugi.campaign import gauntlet as gauntlet_module

    pin = _anchor_for(root, iteration, param_keys, selfplay, records)
    if pin is None:
        return
    challenger = load_checkpoint(ema_dir(root))
    if challenger is None:
        return
    try:
        played = gauntlet_module.run_gauntlet(
            root,
            challenger.path,
            f"ema-{challenger.step}",
            pin,
            param_keys,
            seed=gauntlet_module.gauntlet_seed(campaign.run_seed, iteration),
            # CPU, explicitly, never inherited: §6's between-iterations
            # argument is about 28 free cores, not about 16 processes sharing
            # one GPU, and pin_determinism runs only in those shard processes.
            device=CPU,
            shards=pinned.GAUNTLET_SHARDS,
            min_decisive=pinned.GAUNTLET_MIN_DECISIVE,
        )
    except gauntlet_module.GauntletAborted as exc:
        outcome.extras["gauntlet_aborted"] = 1.0
        outcome.verdicts.append(
            LivenessVerdict(ok=False, reason=f"gauntlet aborted: {exc}")
        )
        return

    outcome.extras["gauntlet_wall_s"] = played.wall_s
    outcome.extras[boundary_checks.GAUNTLET_ANCHOR_GEN] = float(played.generation)
    if played.uninformative:
        outcome.extras["gauntlet_uninformative"] = 1.0
        return
    outcome.extras[boundary_checks.GAUNTLET_SCORE] = played.score
    outcome.extras["gauntlet_decisive"] = float(played.summary.decisive)
    boundary_checks.evaluate_gauntlet(
        played.score, iteration, records, outcome, mutes
    )


def build_next_book(
    root: Path, campaign: CampaignConfig, iteration: int
) -> Path | None:
    """Build ``gen_<iteration+1>.json`` at the **primary** site (KOM-014 §2.1).

    Called at the end of iteration `iteration`, after ``training_iteration``
    returned and its checkpoint was published: iteration `iteration`'s shards
    are complete and have just been digest-verified by the training window
    build, the next iteration's stale-shard ``rmtree`` cannot reach a source
    (every source is strictly older), and the ~5 GB of decompression §2.4 sizes
    is off self-play's critical path.

    Two policies, both load-bearing:

    * **Skipped on the last iteration** (R10b). The loop is
      ``range(start, campaign.iterations)``, so at
      ``iteration + 1 == campaign.iterations`` this would build a book for an
      iteration that never runs -- tens of seconds of DEFLATE for a file no
      worker opens. A campaign later extended past that boundary rebuilds it at
      the fallback site, byte-identically (§2.3).
    * **Failure is caught and logged here**, and only here: a trained iteration
      must not die because a book did not build. The fallback site does the
      opposite, for the reason :func:`book_for_iteration` gives.

    Returns:
        The path written, or ``None`` when the build was skipped, produced too
        thin a book to write, or failed.
    """
    if iteration + 1 >= campaign.iterations:
        return None
    try:
        return build_book(root, iteration + 1, campaign.run_seed)
    except Exception as exc:  # noqa: BLE001 -- reported, and deliberately survived
        warnings.warn(
            f"the draft book for iteration {iteration + 1} did not build at the "
            f"primary site ({type(exc).__name__}: {exc}); iteration "
            f"{iteration + 1} will rebuild it at its fallback site, and plays "
            f"bookless if that fails too",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def validate_book(path: Path, iteration: int) -> Book:
    """Load and check the book iteration `iteration` is about to play from (§4.1).

    **Before any worker is spawned**, exactly once, in the process that built
    it. Worker-side refusal (§6.3, §6.5) is right and stays, but it is on the
    wrong side of the spawn boundary to be the *only* check: a book that exists
    and cannot be loaded fails inside every worker, and the orchestrator then
    restarts each shard up to ``max_shard_restarts`` under a fresh incarnation
    before the iteration fails -- so an operator sees a restart storm with the
    real error N incarnations deep.

    Five checks: :func:`~komugi.selfplay.book.load_book`'s (action space,
    ``GUNGI_MODE``, plane map version, entry bounds, action ids), that the
    book is *this* iteration's, and **one entry replayed end to end** through
    the real engine
    -- the cheapest check that the book and this engine agree. The replayed
    entry is chosen deterministically, as ``build_seed % len(entries)``, so a
    resumed campaign checks the same one.

    Raises:
        ValueError: Naming the book's path and the offending entry.
    """
    book = load_book(path)
    if book.generation != iteration:
        raise ValueError(
            f"book {path} says it is generation {book.generation}, but "
            f"iteration {iteration} is about to play from it: a prefix mined "
            f"from a window that includes this iteration would train on its own "
            f"output"
        )
    if not book.entries:
        raise ValueError(f"book {path} holds no entries")
    entry = book.entries[book.build_seed % len(book.entries)]
    state = GameState(Game(), 0.0)
    for ply, action in enumerate(entry.actions):
        if state.is_terminal() or not bool(state.legal_mask()[action]):
            raise ValueError(
                f"book {path} entry {entry.id} does not replay on this engine: "
                f"action {action} at prefix ply {ply} is illegal (or the "
                f"position is already terminal). Every worker would fail on it, "
                f"one restart storm deep"
            )
        state.play(action)
    return book


def book_for_iteration(
    root: Path, campaign: CampaignConfig, iteration: int, start: int
) -> Path | None:
    """Return the book iteration `iteration` plays from, building it if needed.

    The **fallback** site (§2.1): it builds iff the file is absent or
    `iteration` is the resume boundary. An existing file at the boundary is
    **overwritten, not trusted** -- trusting it would make a book depend on
    which process wrote it, and §2.3's byte-determinism is precisely the
    property that makes rebuilding free.

    **Failure raises here**, unlike at the primary site: a build that fails
    twice over the same data is a defect, and failing here is failing before
    any worker is spawned. So is a book that will not validate (§4.1).

    Returns:
        The path workers should read, or ``None`` for a bookless generation --
        iteration 0, a window too thin to clear the entry floor, or the ≤ 5
        generations after a schema bump. All three are expected shapes, not
        errors (§2.1).
    """
    path = book_path_for(root, iteration)
    if iteration == start or not path.exists():
        built = build_book(root, iteration, campaign.run_seed)
        if built is None:
            return None
        path = built
    validate_book(path, iteration)
    return path


def _selfplay_for_iteration(
    selfplay: SelfPlayConfig, book_file: Path | None
) -> SelfPlayConfig:
    """Return `selfplay` with the campaign's own book decision written into it.

    ``run_campaign`` OWNS ``book_path``, the way it already owns ``run_seed``,
    the fan-out and the shard root (the principle :data:`_FLOAT_KNOBS` exists to
    enforce): the caller's value is replaced, not merged. A bookless generation
    additionally forces ``book_fraction`` to zero -- ``SelfPlayConfig`` refuses
    a positive fraction with no path, and the alternative readings are both
    worse (raising would kill iteration 0 of every booked campaign; leaving the
    caller's path in place would point iteration ``i``'s workers at some other
    generation's book).
    """
    if book_file is None:
        return dataclasses.replace(selfplay, book_path="", book_fraction=0.0)
    return dataclasses.replace(selfplay, book_path=str(book_file))


def run_campaign(
    campaign: CampaignConfig,
    selfplay: SelfPlayConfig,
    orchestrator_overrides: Mapping[str, float] | None = None,
    inference: InferenceConfig | None = None,
) -> list[IterationResult]:
    """Run `campaign` to completion, resuming if its root already holds one.

    One iteration is, in order:

    (a) **self-play.** :data:`~komugi.campaign.factory.CHECKPOINT_ENV` is
        pointed at the newest checkpoint (empty for a fresh campaign's
        iteration 0) and :data:`~komugi.campaign.factory.DEVICE_ENV` at
        ``campaign.device``, then
        :func:`~komugi.selfplay.orchestrate.run_selfplay` is handed
        :func:`~komugi.campaign.factory.serving_net` -- the module-level
        factory it insists on, which reads both variables inside the spawned
        server (see that module's docstring for why the environment, and not
        a closure, is the channel).
    (a') **the draft book.** Before that spawn, ``<root>/books/gen_<i>.json``
        is built if it is missing or if ``i`` is the resume boundary, validated
        once (:func:`validate_book`), and written into the config the workers
        receive (:func:`_selfplay_for_iteration`) -- so ``book_path`` is the
        campaign's to derive, exactly as ``run_seed`` is. The *usual* build
        happened one iteration earlier, at :func:`build_next_book`, off this
        critical path. **Booking is opt-in**: the campaign derives the path but
        never the fraction, and ``SelfPlayConfig.book_fraction`` defaults to
        ``0.0``, so a run left at the default builds a book every iteration and
        books 0% of its games -- an operator must set
        ``book_fraction=DEFAULT_BOOK_FRACTION``, and §6.1/§9's R1 and R4 are
        claims about a run that did.
    (b) **the window.** One :class:`~komugi.train.window.IterationRef` per
        iteration that has data, newest first and stopping at the
        :class:`~komugi.train.window.WindowConfig` size the window is about to
        truncate to anyway (:func:`window_refs`'s ``stop_after``), newest-tilted
        by :class:`~komugi.train.window.RecencyWindow` and presented to the
        trainer through :class:`WindowSource`.
    (c) **load-or-init.** A net of the environment's shape, the newest
        checkpoint's weights **and** its AdamW moments, on ``campaign.device``.
    (d) **train**, and checkpoint at the globally monotone step
        ``(i + 1) * steps_per_iteration``.

    Between (a) and (b) the iteration's self-play numbers are appended to
    ``<root>/selfplay.jsonl`` and fsynced
    (:func:`~komugi.campaign.telemetry.append_selfplay`). The trainer's own
    journal record reports ``games=0`` -- honestly, since the trainer never
    sees a :class:`~komugi.selfplay.orchestrate.RunSummary` -- so before this
    the games, rows, restarts and self-play duration existed only in the list
    returned below, which is exactly what a campaign that dies never returns
    (KOM-042 / KOM-031 union review F5). Written *before* training rather than
    at iteration end because training is the long stretch in which a campaign
    can die, and only iterations this call actually runs write a line, so a
    resumed campaign appends to the history instead of rewriting it. An
    iteration that is genuinely re-run -- the crash-before-checkpoint case,
    whose stale shard tree is cleared above -- appends a second line for that
    iteration number, describing the attempt that survived.

    A **sibling file** rather than a new ``kind`` in ``journal.jsonl``:
    :func:`komugi.telemetry.journal.read_records` does not tolerate an unknown
    kind, it counts one in the same ``skipped`` tally as a corrupt or
    crash-truncated line, so a shared journal would make every healthy
    campaign look damaged to the existing readers (and teaching those readers a
    third kind is outside this ticket's boundary). The durability discipline is
    the journal's, restated: ``O_APPEND`` plus ``fsync`` per record. See
    :mod:`komugi.campaign.telemetry`.

    Before any of that, at **start**: both logs are archived
    (:func:`~komugi.campaign.telemetry.archive_telemetry`), so a relaunch onto
    a root that already holds a campaign's history cannot cost that history.
    And at every iteration **boundary**, the campaign's own
    :class:`~komugi.telemetry.records.IterationMetrics` carries the trainer's
    journalled ``extras`` -- KOM-005b's per-scalar decile means and step
    dispersion among them -- read back off the journal rather than recomputed
    here (:func:`trainer_extras`), beside the unmutable liveness gates:
    :func:`ema_publish_verdict` on the EMA's step numbering and
    :func:`~komugi.campaign.boundary.evaluate_liveness` on the fingerprint
    series. A gate that does not pass writes ``PAUSED`` and raises
    :class:`~komugi.campaign.pause.CampaignPaused`; none of them warns.

    Args:
        campaign: Iteration count, seed, root, and the training/self-play
            budgets.
        selfplay: How a game is played; passed to every worker unchanged.
        orchestrator_overrides: Liveness and patience knobs for
            :class:`~komugi.selfplay.orchestrate.OrchestratorConfig` (see
            :func:`_orchestrator_config`). ``None`` takes its defaults.
        inference: The seam's shapes and timings. ``None`` builds the P0a
            defaults for the production planes and action space.

    Returns:
        One :class:`IterationResult` per iteration **this call ran**, in
        order. A resumed campaign returns only the iterations it played; a
        campaign whose root already holds ``iterations`` completed iterations
        returns an empty list rather than redoing work.

    Before the first iteration the three configs are checked *against each
    other* once, by :func:`campaign_batch_fit`: a ``max_batch`` below the
    campaign's ``workers * games_in_flight`` slot count would cap every forward
    for the whole campaign without ever failing (review F8, design D11'). The
    check is here rather than in the orchestrator because a bench is entitled
    to cap on purpose and a campaign is not.

    Raises:
        CampaignPaused: If the root holds an unacked pause, or if the newest
            checkpoint's iteration has no boundary record -- the resume refuses
            to advance across an iteration that checkpointed and never closed
            (:func:`_refuse_unfinished_boundary`).
        RuntimeError: If an iteration's self-play produced no readable sample
            at all. Training on an empty window is an hour spent learning
            nothing, and the window's own ``ValueError`` would name the
            symptom rather than the iteration.
        ValueError: If the seam's ``max_batch`` cannot hold the campaign's
            inference slots -- see :func:`campaign_batch_fit`. Raised before
            any iteration runs.
    """
    root = Path(campaign.root)
    # Before anything is created or spawned: a root holding an unacked PAUSED
    # raises here, and a validated ACK is consumed, muted and archived here.
    # Reading the mute directory happens on *every* path out of this call,
    # including RUNNING -- a mute has to survive the very restart that has
    # nothing left to remind the loop it exists.
    startup = apply_startup_state(root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_dir(root)
    ema_directory = ema_dir(root)
    journal = journal_path(root)
    # The unfinished-boundary refusal (review CP-01) is asked HERE, before a
    # single byte under this root is written or repaired: before the archive
    # directory, before `reconcile_ema`, before anything is spawned. It reads
    # `point.checkpoint` and the journal and nothing else, so it can be asked
    # this early -- and it has to be, because `reconcile_ema` publishes EMA_i
    # for the very iteration the next statement would declare unjudged. Sited
    # after it (as it was when KOM-180 landed) the refusal mutated the root on
    # *every* refused launch, and the documented rollback -- delete iteration
    # i's trainer checkpoint -- then died inside `reconcile_ema` with "EMA is
    # ahead of the trainer ... Two campaigns sharing one root is the likely
    # cause", an error naming a cause that did not happen (KOM-211, KOM-180
    # impl review F1). The rule quoted below governs this check too, and most
    # of all: a paused root must do *no* work at all (acceptance #22).
    #
    # `newest_ema` is therefore READ off disk and never published: whatever
    # `<root>/ema` holds is what the PAUSED records, which on the common path
    # (a kill before the EMA for i was published) is EMA_{i-1}, or nothing at
    # all on a root that never got that far.
    point = resume_point(campaign)
    newest_ema = load_checkpoint(ema_directory)
    _refuse_unfinished_boundary(
        root,
        campaign,
        point,
        str(newest_ema.path) if newest_ema is not None else "",
    )
    # The search this launch will play, bound to the root before anything is
    # spawned (KOM-225): a fresh root records its profile, an unchanged
    # relaunch records nothing, and a relaunch that changed the search or the
    # loss recipe is REFUSED unless `allow_profile_switch` authorises it. The
    # iteration it is recorded as taking effect at is `point.iteration`, which
    # is why it is asked here, immediately after the resume point is known.
    #
    # Sited after the two refusals above and before `archive_telemetry` for
    # that function's own reason: appending a line is *work*, and a paused root
    # must do none (acceptance #22). Everything below this point either spawns
    # or writes, so this is the last moment at which the guard still precedes
    # every game the profile describes.
    guard_search_profile(
        root,
        profile_of(selfplay, campaign, point.iteration),
        campaign.allow_profile_switch,
    )
    # Archive-on-relaunch, before this process appends a single byte and before
    # anything else here can fail: v1 destroyed its telemetry TWICE by opening
    # the run log for writing at launch, and the fix is that the previous
    # launch's bytes are already copied aside by the time this call returns.
    # A copy, never a truncation -- the logs stay in place and stay
    # append-only, because a resumed campaign reads its own baselines,
    # constants and fingerprint series back out of them. See
    # :func:`~komugi.campaign.telemetry.archive_telemetry` for why the
    # truncation token it obtains is deliberately never spent.
    #
    # Sited after `apply_startup_state` and after the unfinished-boundary
    # refusal, which are the two things entitled to run earlier: a paused root
    # must do *no* work at all, and creating an archive directory under it is
    # work (acceptance #22).
    archive_telemetry(root, campaign_logs(root))
    # One process, one key set, derived from the campaign's own net factory so
    # the same parameter/buffer split applies to a live module and to a state
    # dict read back from a file.
    param_keys = parameter_keys(new_net())
    # The EMA is recomputable, not atomic with the trainer: whatever a crash
    # cost is rebuilt here, in ascending step order, before a game is played.
    reconcile_ema(
        checkpoints,
        ema_directory,
        param_keys,
        decay=EMA_DECAY,
        steps_per_iteration=campaign.steps_per_iteration,
    )
    seam = (
        inference
        if inference is not None
        else InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)
    )
    # Cross-config, before anything is spawned or read: the alternative is a
    # campaign that runs for days at a silently capped batch width.
    campaign_batch_fit(campaign, selfplay, seam)

    # `point` was read at the top of this call, by the unfinished-boundary
    # refusal: one read, because a second one would be a second chance for the
    # checkpoint the refusal judged and the checkpoint the loop resumes from to
    # disagree.
    start = point.iteration
    results: list[IterationResult] = []
    previous_ckpt_env = os.environ.get(CHECKPOINT_ENV)
    previous_device_env = os.environ.get(DEVICE_ENV)
    pending_mute = startup.applied
    # `None` on every call, deliberately: the journal is what a restart reads
    # the previous BatchNorm digest back from (`record_fingerprints` resolves a
    # `None` through `last_bn_digest_lo`), so the resumed path and the
    # continuing path are one comparison rather than two.
    previous_bn: float | None = None
    try:
        for iteration in range(start, campaign.iterations):
            # §5: the pause check is the FIRST statement of the loop body, not
            # the call to run_selfplay. The difference is destructive -- the
            # stale-shard rmtree below runs before self-play, so a check sited
            # at the orchestrator would destroy a crashed attempt's partial
            # shard tree and only then stop.
            standing = read_pause(root)
            if standing is not None and standing.pause_class != INTERRUPT_CLASS:
                raise CampaignPaused(
                    results,
                    f"{root} is paused at iteration "
                    f"{standing.paused_at_iteration} on {standing.tripwire!r}: "
                    f"{standing.detail}",
                )
            mutes = live_mutes(root, iteration)
            expire_mutes(root, iteration)

            # Re-read every iteration rather than carrying the last one's
            # path: this is the same query a fresh process would make, so the
            # resumed path and the continuing path are one code path.
            #
            # Two bindings, deliberately (§2). `serving` is what self-play
            # plays; `resume` is what the trainer continues from. A single
            # binding pointed at the EMA directory would silently train the
            # net *from* smoothed weights every iteration -- the recursion the
            # band gate is derived from would no longer describe the system,
            # and nothing would raise.
            serving = load_checkpoint(ema_directory)
            resume = load_checkpoint(checkpoints)
            served = str(serving.path) if serving is not None else ""
            os.environ[CHECKPOINT_ENV] = served
            os.environ[DEVICE_ENV] = campaign.device

            # An iteration this loop is about to run is by definition
            # unfinished -- resume_point() places the start *after* the newest
            # checkpoint's iteration -- so anything already under its shard
            # tree is a crashed attempt's partial output. Left in place it
            # would sit alongside the same-seed replay below and double the
            # iteration's rows in both the window and the reported counts
            # (wave-11 union review F3).
            stale = shard_root(root, iteration)
            if stale.exists():
                shutil.rmtree(iteration_root(root, iteration))

            # The iteration directory is created here -- after the rmtree above
            # and before the orchestrator writes a shard into it -- and the
            # profile in force goes in with it, so this iteration's rows and
            # the provenance of the search that produced them travel together
            # (KOM-225). Written on the re-run path too, for free: the rmtree
            # took the previous attempt's copy with the rest of the tree.
            write_iteration_profile(root, iteration)

            # The fallback build site and the §4.1 validation, both BEFORE the
            # first spawn (KOM-014 §2.1, N4): a book that cannot be loaded or
            # replayed fails here as one exception, rather than inside every
            # worker as a `max_shard_restarts` storm with the real error N
            # incarnations deep. `run_campaign` owns the path the way it owns
            # the seed -- the caller's `book_path` never reaches a worker.
            book_file = book_for_iteration(root, campaign, iteration, start)
            started = time.monotonic()
            summary = run_selfplay(
                serving_net,
                _orchestrator_config(campaign, iteration, orchestrator_overrides),
                _selfplay_for_iteration(selfplay, book_file),
                seam,
            )
            # Written here rather than at the end of the iteration, and fsynced
            # before the training step starts (KOM-042): the numbers this
            # persists are the ones a *dead* campaign is otherwise missing, and
            # a training step is the longest stretch of an iteration in which
            # it can die. A write failure here (disk full, root unwritable)
            # aborts the iteration by design: a root that cannot take a
            # hundred-byte fsynced write cannot take the checkpoint either,
            # and failing loudly before training beats discovering it after
            # (fanout-1 union review F6).
            # The shard tree is sealed and will not change again this
            # iteration, so it is censused ONCE, here, and the same
            # measurement is stamped on both records below (KOM-150). Two
            # reads would be two chances for the file and the journal to
            # disagree about the same iteration.
            stats = game_stats(shard_root(root, iteration))
            append_selfplay(
                selfplay_journal_path(root),
                SelfPlayTelemetry(
                    iteration=iteration,
                    games=summary.games_completed,
                    rows=summary.rows_written,
                    restarts=summary.restarts,
                    shard_dirs=len(summary.shard_dirs),
                    wall_s=time.monotonic() - started,
                    draws=stats.draws,
                    turn_cap_games=stats.turn_cap_games,
                    mean_plies=stats.mean_plies,
                    max_plies=stats.max_plies,
                    median_placements_w=stats.median_placements_w,
                    median_placements_b=stats.median_placements_b,
                    early_shu_games=stats.early_shu_games,
                ),
            )

            # One config decides the schedule, which iterations are worth
            # holding decompressed, and how far back the listing walks at all:
            # refs beyond the window's size can never be sampled, so retaining
            # them would pin the campaign's whole history in RAM (KOM-033
            # review F1) and *constructing* them would re-scan every iteration
            # the campaign has ever run, once per training iteration -- cost
            # quadratic in campaign length (KOM-041). The same `size` reaches
            # both window_refs and RecencyWindow below, which is what makes the
            # bounded walk equivalent to the unbounded one rather than a
            # narrower window.
            window_config = WindowConfig()
            refs = campaign_window_refs(
                root, iteration, window_config, floor=campaign.window_floor
            )
            if not refs:
                raise RuntimeError(
                    f"iteration {iteration} has no readable samples under "
                    f"{root}: self-play reported {summary.games_completed} "
                    f"games and {summary.rows_written} rows, and the window "
                    f"would be empty. Refusing to run a training iteration "
                    f"that could only train on nothing"
                )
            source = WindowSource(
                RecencyWindow(
                    refs,
                    window_config,
                    seed=iteration_seed(campaign.run_seed, iteration),
                )
            )

            # Guard and load in one place (:func:`_net_for_resume`), so the
            # refusal on this path is exercised by a test rather than by
            # reading (KOM-002 §5.3, review F4).
            net = _net_for_resume(resume)
            net.to(campaign.device)
            step = TrainStep(
                net,
                # Selected by NAME from a closed mapping (:data:`LOSS_RECIPES`),
                # never built here: the recipe a campaign ran under has to be a
                # constant somebody can open, and `CampaignConfig.__post_init__`
                # has already refused anything else.
                LossAssembler(LOSS_RECIPES[campaign.loss_recipe]),
                StepConfig(
                    batch_size=campaign.batch_size,
                    lr=campaign.lr,
                    seed=campaign.run_seed,
                    # The A/B arm. The campaign owns it because the two arms
                    # are two campaigns, and it must reach the trainer for the
                    # experiment to exist at all.
                    fast_policy_weight=campaign.fast_policy_weight,
                    # KOM-160's arm, on the same wire and for the same reason:
                    # 4D and its control are two campaigns differing in this
                    # one field, and it must reach the trainer or both train
                    # at 1.0 and the comparison is a null nobody can attribute.
                    cap_row_value_weight=campaign.cap_row_value_weight,
                    # KOM-159's, likewise -- and its companion, which is not an
                    # arm but the CAP the censoring predicate is evaluated
                    # against. `Game.max_turns()` is the engine's own, read
                    # here (the loop may import the engine; the trainer may
                    # not) so that the trainer never restates a number the
                    # shards were written under.
                    mlh_cap_row_weight=campaign.mlh_cap_row_weight,
                    cap_plies=Game.max_turns(),
                    # Journalled, never consumed by the step: the switch that
                    # turns the search-side utility on is otherwise recorded
                    # nowhere (design review F10).
                    mlh_utility_lambda=selfplay.mlh_utility_lambda,
                ),
            )
            if resume is not None:
                # The half of a resume that is invisible when it is missing:
                # weights without moments is a cold Adam wearing a warm net
                # (wave-10 review, untested-but-promised #1).
                step.optimizer.load_state_dict(resume.optim_state)

            try:
                history = training_iteration(
                    net,
                    source,
                    step,
                    steps=campaign.steps_per_iteration,
                    rng_seed=iteration_seed(campaign.run_seed, iteration),
                    checkpoint_dir=checkpoints,
                    journal_path=journal,
                    iteration=iteration,
                    global_step_base=iteration * campaign.steps_per_iteration,
                )
            except Exception as exc:
                # A crash converted into a crash *with a marker*. This is the
                # only defence against the NaN ordering hazard: the trainer
                # publishes its checkpoint before it writes its journal record,
                # and a non-finite loss dies inside that write -- so the
                # poisoned checkpoint is already the newest one, and a watcher
                # that restarted blindly would resume onto it. The real fix is
                # a finiteness check before `save_checkpoint`, in a file
                # outside this ticket's allowlist; this is the mitigation.
                newest = load_checkpoint(checkpoints)
                write_pause(
                    root,
                    PauseRecord(
                        pause_class=CRASH_CLASS,
                        paused_at_iteration=iteration,
                        tripwire=CRASH_TRIPWIRE,
                        detail=f"{type(exc).__name__}: {exc}",
                        newest_checkpoint=str(newest.path) if newest else "",
                        newest_ema=served,
                        last_good_checkpoint=(
                            str(resume.path) if resume is not None else ""
                        ),
                        written_at_step=(iteration + 1)
                        * campaign.steps_per_iteration,
                    ),
                )
                raise

            result = IterationResult(
                iteration=iteration,
                games=summary.games_completed,
                rows=summary.rows_written,
                first_loss=history[0][TOTAL_KEY],
                last_loss=history[-1][TOTAL_KEY],
                checkpoint_step=(iteration + 1) * campaign.steps_per_iteration,
            )
            results.append(result)

            # The PRIMARY book build (KOM-014 §2.1): after training returned
            # and its checkpoint was published, before the iteration closes.
            # Off self-play's critical path, over a shard tree that was just
            # digest-verified by the window build above, and guarded so that a
            # failed book cannot cost a trained iteration.
            build_next_book(root, campaign, iteration)

            try:
                previous_bn = _close_iteration(
                    root,
                    campaign,
                    selfplay,
                    iteration,
                    param_keys,
                    served=served,
                    resume=resume,
                    trained=net,
                    history=history,
                    games=summary.games_completed,
                    rows=summary.rows_written,
                    # The census taken above, not a second one: the boundary's
                    # record and the self-play line describe the same games.
                    stats=stats,
                    mutes=mutes,
                    previous_bn=previous_bn,
                    pending_mute=pending_mute,
                    results=results,
                    # `refs` is ascending (campaign_window_refs' own
                    # contract), so the first member is the oldest iteration
                    # this boundary's window actually drew from.
                    window_oldest=refs[0].iteration,
                )
            except CampaignPaused:
                # Step 7 of the boundary's own order already ran: `enforce`
                # wrote a `class: tripwire` PAUSED naming the gate that fired,
                # *then* raised. Marking it again below would overwrite the
                # diagnosis with "something threw" -- the operator would read
                # `crash` instead of `loss_spike`, and the ack/mute protocol
                # keys off the tripwire name, so the un-pause path would stop
                # matching the stop. The marker on disk is already the right
                # one; this handler exists to leave it alone.
                raise
            except Exception as exc:
                # The trainer handler's twin, at the OTHER site an iteration
                # can die at (KOM-129, KOM-025 review F12). Everything
                # between the published checkpoint and the boundary's own
                # `IterationMetrics` write can throw -- a journal write that
                # fails, the probe, the gauntlet, a bug at the seam -- and
                # until this existed such an iteration ended with *neither* a
                # record nor a marker: the campaign died leaving no trace of
                # the iteration at all, and a watcher that restarted blindly
                # resumed onto the checkpoint the failure may have poisoned.
                #
                # `close_iteration: ` prefixes the detail so the two crash
                # sites are distinguishable on disk. They ask different first
                # questions -- a trainer crash is about the step, a boundary
                # crash is about the root's telemetry -- and the marker is the
                # only artefact a post-mortem is guaranteed to have.
                newest = load_checkpoint(checkpoints)
                write_pause(
                    root,
                    PauseRecord(
                        pause_class=CRASH_CLASS,
                        paused_at_iteration=iteration,
                        tripwire=CRASH_TRIPWIRE,
                        detail=f"close_iteration: {type(exc).__name__}: {exc}",
                        newest_checkpoint=str(newest.path) if newest else "",
                        newest_ema=served,
                        last_good_checkpoint=(
                            str(resume.path) if resume is not None else ""
                        ),
                        written_at_step=(iteration + 1)
                        * campaign.steps_per_iteration,
                    ),
                )
                raise
            pending_mute = None
    except BaseException as exc:
        # Every `Exception` raised after this iteration's checkpoint was
        # published is marked where it happens -- the training-step handler and
        # the boundary handler, both above; the one step between them,
        # `build_next_book`, catches its own by design. So what reaches here is
        # either a failure from *before* anything of this iteration's was
        # published (the standing-pause check, the stale-shard sweep,
        # self-play, the self-play journal append, the window, the net load),
        # which a restart re-enters safely by re-running the iteration, or a
        # non-`Exception`.
        #
        # The non-`Exception` half is KeyboardInterrupt and SystemExit: a
        # deliberate stop is not a fault, so it leaves a marker for the
        # operator and a root that still starts. Nothing is swallowed either
        # way.
        if not isinstance(exc, Exception):
            write_pause(
                root,
                PauseRecord(
                    pause_class=INTERRUPT_CLASS,
                    paused_at_iteration=len(results) + start,
                    tripwire="interrupt",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )
        raise
    finally:
        _restore_env(CHECKPOINT_ENV, previous_ckpt_env)
        _restore_env(DEVICE_ENV, previous_device_env)
    return results
