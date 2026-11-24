#!/usr/bin/env python3
"""KOM-158 (KOM-151 M1'): what the value head BELIEVES on a turn-cap game.

A campaign whose games increasingly end at the engine's ply cap has two very
different diseases, and ``docs/reviews/2026-09-11-kom-151-design-review.md`` F3
is the finding that they cannot be told apart from what is already on disk:

* **"draw"** -- the net rates a cap position as a confident draw (``d`` near 1),
  the ``[0, 1, 0]`` label every row of a cap game carries is not lying, and the
  loop to attack is the label/row-share feedback (F5.1, rule (c));
* **"I am winning and cannot convert"** -- the net rates the position as won
  (``|w - l|`` large, ``d`` small) and the search cannot finish it, which is a
  *search budget* question and not a label question.

The shard's ``root_value`` column cannot answer that. It is ``v_mix``, a scalar
collapse of ``(w, d, l)`` (``search/gumbel.py:561``), so ``root_value ~ 0`` is
satisfied by "confident draw" **and** by "coin flip" alike; and under a nonzero
``draw_score`` it carries a contempt floor of order 0.05-0.12 that is
mover-relative *by design* (``selfplay/adapter.py:28-39``), so it alternates
sign every ply in the white frame on a perfectly coherent net. A **forward pass
of the iteration's own checkpoint** over the cap positions does answer it:
:class:`~komugi.selfplay.inference.InProcessEvaluator` returns the raw ``wdl``
before any contempt is applied (``NetEval.wdl``, "no scalar value field" by
design), and ``d`` and ``w - l`` are then two separate, unconfounded reads.

The pre-registration this script exists to settle (review "Recommended
experiment order" item 2), on the last :data:`VALUE_PLIES` plies of every cap
game in the white frame:

* median ``d > 0.6``  => the net already believes "draw";
* median ``d < 0.3`` with ``|w - l| > 0.3`` => conversion failure.

How the positions are obtained
------------------------------
By **replay**, the way ``scripts/analysis/termination_census.py`` gets a
termination reason: every self-play row carries the ``action`` actually played
(:data:`komugi.selfplay.records.SCHEMA`), so a game's rows replay on a fresh
:class:`~komugi.engine.game.Game` under
:attr:`~komugi.engine.repetition.RepetitionRule.FULL` and the progress clock,
the repetition count and the encoder's own planes are then read from the
replayed position rather than re-derived here. Three guards keep that honest:

* :func:`mode_guard` -- ``engine/game.py:267`` reads ``GUNGI_MODE`` with default
  ``"intro"``, and a campaign-3 (advanced) shard replayed in an intro process
  fails at ply 0. The shard states its own mode class through the drafting
  planes of its first row; a disagreement is refused with **both** modes named
  (review F7).
* :func:`plane_map_guard` -- the read-side half of KOM-002 §4, mirroring
  :func:`komugi.selfplay.records._check_plane_map_version`: the schema
  fingerprint provably cannot see a plane-map change, so the column is compared
  and the refusal names both versions.
* :func:`observation_crosscheck` -- for a sample of rows, the replayed
  position's planes 115/116/117/120 must equal the **stored** observation's. A
  mismatch means the replay diverged from the played game, and the pass is void
  rather than quietly wrong.

**Booked games are excluded and counted.** A game played from a draft-book
prefix records no row for that prefix (``selfplay/worker.py:832-844``), so its
first row is at ``ply > 0``. Review F6 is right that such a game is replayable
in principle -- book selection is a pure function of the ids
(``worker.py:810-830``) -- but doing it needs the book file, the worker's own
``run_seed`` and ``book_fraction``, none of which is on the shard. So
:func:`replay_game` **accepts** a prefix when a caller has one, the CLI passes
none, and the games are counted in ``book_excluded`` rather than silently
dropped. The ``root_value`` half needs no replay and covers them.

Cost, and why this is a pause-window script
-------------------------------------------
Two passes over the iteration's shards. The first reads seven small
``int64``/``float32`` columns through :func:`~komugi.train.shards.read_columns`,
the projecting reader -- the whole census of game shape for the price of the
columns, exactly as :mod:`komugi.campaign.game_stats` does it. The second reads
``observation`` for the **bounded** row set the guards and the cross-check need
(:data:`CROSSCHECK_ROWS` rows per selected game), not for the iteration. The
forward pass is ~26 cap games x 32 positions plus an equal decisive sample:
seconds on a GPU, a few minutes on CPU for b10c128.

It plays nothing and writes nothing into the campaign root, so it is safe
against a live campaign -- but it **does** hold a net, so run it in a pause
window (``memory/advisor-probes-starve-gates.md``: a probe that competes with a
running gate sinks the gate).

What KOM-162 added, and why (the 4D pre-registration)
-----------------------------------------------------
``docs/reviews/2026-09-11-kom-160-design-review.md`` found four ways this census
as first written cannot score campaign 4D's claims. Each finding is one
extension here; every key the original census wrote is unchanged, and the
default run is byte-for-byte the run it was before.

* **F1 -- the last 32 plies are the least sensitive place to read a per-row
  draw weight.** A weighted cross-entropy multiplies the draw *odds* by ``w``
  uniformly, which moves ``P(draw | x)`` most where cap and decisive rows
  overlap (mid-game) and least where plane 115 already separates them (the
  tail). So ``--window FIRST:LAST`` reads the same two classes at an arbitrary
  **ply range**: cap positions at those plies against positions of decisive
  games that were *still being played* at ``LAST`` (``total_plies > LAST``),
  which is what makes the two samples comparable at a shared ply. That is C3'.
  Without ``--window`` every ``*_win_*`` key is ``null``.
* **F2 -- a signed median cannot see symmetric over-confidence.**
  ``cap_wl_median`` is the median of white-frame ``w - l``; a head that is
  confidently winning for whoever is to move has a median near zero and a huge
  spread. :data:`REPORT_KEYS` therefore carries the **magnitude** beside the
  sign: ``cap_wl_abs_median`` (median ``|w - l|``) and ``cap_wl_spread``
  (``p90 - p10`` of the signed ``w - l``), and the same two for the decisive
  sample. That is C5.
* **F8 -- the journal's ``loss_wdl`` is a weighted mean, so under 4D it
  improves by construction.** ``cap_ce_unweighted`` / ``dec_ce_unweighted`` are
  the **unweighted** cross-entropy of the forward pass against each sampled
  row's stored ``wdl_target`` -- the same mover-relative frame the net answers
  in, so no flip is applied -- which is the like-for-like loss read (M3/C6). A
  shard tree that does not carry the column reports ``null`` rather than a
  number taken against nothing.
* **F6 -- the "non-cap draws" tripwire had no instrument.** ``noncap_draws`` is
  ``draws - turn_cap_games`` for the iteration, by
  :mod:`komugi.campaign.game_stats`'s own two counts. C10 reads it at every
  iteration of 4D.

Usage::

    # iteration 14's rows were played by the EMA at step 14*300 = 4200; the
    # script refuses the file if the journal disagrees (KOM-181).
    GUNGI_MODE=advanced KOMUGI_CAMPAIGN_NET=b10c128 \\
    python scripts/analysis/cap_game_values.py <root> 14 \\
        --ckpt <root>/ema/checkpoint-000004200-<digest>.pt \\
        --device cpu --threads 8 --out campaign3_cap_values_iter14.json

    # the same census with the mid-game read C3' asks for, plies 96-160:
    ... --window 96:160

    # what the head TRAINED on those rows thinks of them -- a different
    # question, asked by naming the role it is a checkpoint of:
    ... --ckpt <root>/ema/checkpoint-000004500-<digest>.pt \\
        --ckpt-role post_training

    # any OTHER net over these rows (iteration 20's head, say) matches no
    # record of iteration 14 and needs the override, which prefixes every key:
    ... --ckpt <root>/ema/checkpoint-000006000-<digest>.pt --later-head

Which net is "the iteration's own checkpoint" (KOM-181, review EP-01/EV-09)
--------------------------------------------------------------------------
The paragraph above says the forward pass is of "the iteration's own
checkpoint", and until KOM-181 nothing here checked it. Iteration N's rows are
produced by the EMA the loop serves **before** self-play -- the EMA at step
``N * 300`` (``campaign/loop.py:2078-2081``) -- while the EMA published at
``(N + 1) * 300`` is the head *trained on those rows*. The KOM-159 censuses
forwarded the second one (iteration 14 against "EMA ckpt 4500"), so every
head-vs-search apportionment read off them is one training iteration out of
frame, and the result JSONs recorded no checkpoint identity, so nothing on disk
could say.

So the ``--ckpt`` is now **resolved against the journal**, not taken on trust:

* ``--ckpt-role served_producer`` (the default) means "the net that played these
  rows", which the journal records as ``self_play@N``; ``--ckpt-role
  post_training`` means the EMA published at the end of the iteration,
  ``ema@N``.
* The file's parameter fingerprint is recomputed with the campaign's own
  :func:`komugi.campaign.boundary._fingerprint` over
  :func:`komugi.train.digest.parameter_keys` -- the identical hash, called
  rather than re-implemented -- and compared with that record's ``sha256``.
  A mismatch is a **refusal** (exit :data:`EXIT_WRONG_NET`) naming both digests;
  iteration 0 has no ``self_play`` record at all (it served fresh weights) and
  is refused by name.
* ``--later-head`` is the deliberate override: the census runs against the
  mismatched net and **every key of the report is prefixed**
  ``laterhead_`` (:func:`later_head_report`), so no such run can be pooled with,
  or mistaken for, an on-role census.
* The report carries the identity either way -- ``census_ckpt_role``,
  ``census_ckpt_path``, ``census_ckpt_step``, ``census_ckpt_sha256``,
  ``census_ckpt_journal_role`` (:data:`IDENTITY_KEYS`) -- and the human summary
  prints it on one line.

KOM-162 (review of KOM-160, F1/F2/F6/F8): ``--window FIRST:LAST`` adds a
mid-game read for both classes; ``cap_wl_abs_median`` / ``cap_wl_spread`` (and
the ``dec_`` twins) score decisiveness by magnitude; ``cap_ce_unweighted`` /
``dec_ce_unweighted`` are the like-for-like value-loss reads; ``noncap_draws``
is the per-iteration non-cap-draw tripwire.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import zipfile
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import torch

from komugi.campaign.boundary import _fingerprint, latest_fingerprints
from komugi.campaign.factory import IN_PLANES, NET_ENV, NUM_ACTIONS, new_net
from komugi.campaign.game_stats import (
    DRAFT_PLANE_BLACK,
    DRAFT_PLANE_WHITE,
    TURN_CAP_PLIES,
)
from komugi.campaign.loop import journal_path, shard_root
from komugi.engine import game as engine_game
from komugi.engine.cyboard import board_from_fen
from komugi.engine.game import TERMINATION_TURN_CAP, Game, RepetitionRule
from komugi.engine.observation import MAX_TURNS, PLANE_MAP_VERSION
from komugi.selfplay.inference import InferenceConfig, InProcessEvaluator, NetEval
from komugi.selfplay.records import MLH_SCALE
from komugi.selfplay.worker import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import EMA, SELF_PLAY, NetFingerprint
from komugi.train.checkpoint import load_checkpoint_file, require_matching_plane_count
from komugi.train.digest import parameter_keys
from komugi.train.shards import SHARD_GLOB, ShardWarning, read_columns

# ---------------------------------------------------------------------------
# what is read, and over what window
# ---------------------------------------------------------------------------

CENSUS_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "ply",
    "mlh_target",
    "termination",
    "to_play",
    "action",
    "root_value",
)
"""The small columns the first pass reads: four ``int64``, two ``float32``.

:data:`komugi.campaign.game_stats.STAT_COLUMNS` plus the three fields a replay
and its value read need -- ``action`` (what was played), ``to_play`` (whose
frame a row's numbers are in) and ``root_value`` (the contempt-contaminated
secondary read this script exists to replace). Still a projection: ~56 bytes a
row against a ~190 KB row.
"""

OBSERVATION_COLUMNS: Final[tuple[str, ...]] = ("game_id", "ply", "observation")
"""The second pass's columns -- the one expensive read, kept to a bounded row set.

``observation`` is 37 KB a row (``(128, 9, 9)`` float32). It is read for the
guard rows and the cross-check rows of the **selected** games only, which is
tens of rows per game rather than an iteration's millions.
"""

WDL_TARGET_COLUMN: Final[str] = "wdl_target"
"""The label the unweighted cross-entropy is scored against (KOM-162, F8).

Read **optionally**, per shard, exactly as :func:`plane_map_guard` treats
:data:`VERSION_COLUMN`: a projection tree that does not declare it is a shape
and not damage, and its rows simply carry no target. It is deliberately not in
:data:`CENSUS_COLUMNS`, whose members are required of every shard read.

``(3,)`` float32 per row, and **mover-relative** -- the pure game outcome in the
perspective of the player who moved in that row, with no contempt in it
(``selfplay/records.py:120-122``). That is the same frame
:class:`~komugi.selfplay.inference.NetEval` answers in, which is why
:func:`cross_entropy` applies no :func:`white_frame` flip: the white frame is
for *reporting* ``w - l`` to a human, and a loss is a loss in either frame only
if both sides of it are in the same one.
"""

WDL_SIZE: Final[int] = 3
"""Win, draw, loss. :data:`komugi.selfplay.records._WDL_SIZE` is private, and
this script already states the shape in :func:`forward_wdl`'s refusal."""

CE_FLOOR: Final[float] = 1e-12
"""The interval ``log`` is taken over is ``[CE_FLOOR, 1]``.

A softmax cannot return an exact zero, but a float32 one rounds to it, and
``-log(0)`` would report an infinite loss for a head that is merely very
confident. Clipping at a floor twelve orders below the smallest number the
report distinguishes turns that into a large finite number, which is what it
is. The ceiling is 1 because the quantity is a probability.
"""

VERSION_COLUMN: Final[str] = "plane_map_version"
"""The column :func:`plane_map_guard` compares. A shard whose own schema does
not declare it (a projection tree, the unit suite's synthetic shards) is not
refused: absence is a shape, not a disagreement. A shard that declares a
*different* version is refused -- see :func:`plane_map_guard`."""

TURN_CLOCK_PLANE: Final[int] = 115
REPETITION_2_PLANE: Final[int] = 116
REPETITION_3_PLANE: Final[int] = 117
PROGRESS_PLANE: Final[int] = 120
"""The four encoder planes the cross-check compares, by the map KOM-002 §2 pins
(:mod:`komugi.engine.observation`'s docstring): 115
``min(plies, MAX_TURNS)/MAX_TURNS``, 116/117 the two monotone repetition planes,
120 ``min(progress_plies, MAX_TURNS)/MAX_TURNS``.

They are the planes that are a pure function of the *history* rather than of the
cells, which is exactly what a replay can get wrong: a board that matches while
the clock does not is a replay that dropped or duplicated a ply.
"""

CROSSCHECK_PLANES: Final[tuple[int, ...]] = (
    TURN_CLOCK_PLANE,
    REPETITION_2_PLANE,
    REPETITION_3_PLANE,
    PROGRESS_PLANE,
)

VALUE_PLIES: Final[int] = 32
"""How many of a game's last plies the forward pass evaluates (design §3.1).

The question is about the *end* of a cap game -- where a shuffle has settled and
where the draw label is least plausible as a description of the position -- not
about its opening, which is shared with every decisive game.
"""

PROGRESS_WINDOW: Final[int] = 64
"""The window the progress clock and the repetition share are read over (F8).

Wider than :data:`VALUE_PLIES` on purpose: review F8 shows that plane 120's
*non-zero share* cannot discriminate a shuffle (it is ~85-90% in an ordinary
decisive game, and an arata resets the clock), so what is reported is the
clock's **magnitude** over a window long enough for a real progress move to have
shown up in it.
"""

CROSSCHECK_ROWS: Final[int] = 8
"""Rows per selected game whose stored observation is fetched and compared.

A sample, for the reason design §3.1 item 4 gives -- the cross-check exists to
catch a replay that *diverged*, and a divergence is not row-local: once the
replay is one ply out, plane 115 disagrees for every row after it. The sample is
the game's LAST rows, which is where a divergence anywhere earlier surfaces.
"""

REPETITION_SHUFFLE: Final[int] = 2
"""``repetition_count() >= 2`` -- the position has been here before.

Two, not four: four is the *draw*, which a cap game by definition did not reach
(it was adjudicated by the cap), so counting fourfolds here would count zero by
construction. Two is the shuffle measure U2 asks for.
"""

FORWARD_BATCH: Final[int] = 32
"""Positions per forward pass. Bounded because the legality mask crossing the
seam is ``(N, 37341)`` bool -- 1.2 MB at this batch, 37 MB at a thousand."""

WHITE: Final[int] = 0
BLACK: Final[int] = 1
"""The ``to_play`` codes (:mod:`komugi.selfplay.worker`)."""

MLH_PLY_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("b0", 0, 60),
    ("b1", 60, 160),
    ("b2", 160, 300),
    ("b3", 300, 1 << 30),
)
"""The ply bands the moves-left calibration is stratified by (KOM-159 F3).

Half-open ``[lo, hi)``, so every row lands in exactly one. The boundaries are
the design's: the draft and the opening (where every game's answer is "a long
way"), the mid-game window the go/no-go is read in, the long tail, and the
positions past 300 plies that are almost all on their way to the cap. A single
pooled MAE over all of them would be dominated by whichever band the iteration
happened to produce the most rows of, and the bands' errors are not the same
quantity -- the head's job at ply 20 and at ply 320 are different problems.
"""

MLH_ONEPLY_THRESHOLD: Final[float] = 8.0
"""Pre-registered ceiling on ``mlh_oneply_median_abs_delta``, in plies (F3).

The noise floor of ``f(m)``: the utility reads *differences* of the head's
answer between a root and a position one ply deeper, and a perfectly consistent
head answers exactly one ply less. ``median |m^(t) - m^(t+1) - 1| < 8`` is
``|f| < 0.25`` of pure noise at the pre-registered 32-ply slope. Reported, not
enforced -- this script scores no claim, it measures one.
"""

P10: Final[float] = 10.0
P90: Final[float] = 90.0
"""The reported tails. A median alone cannot say whether a distribution centred
at ``d = 0.5`` is every position at 0.5 or half at 0.1 and half at 0.9, and
those are different findings about the same net."""

REPORT_KEYS: Final[tuple[str, ...]] = (
    "iteration",
    "games",
    "cap_games",
    "decisive_sample",
    "book_excluded",
    "replay_failures",
    "crosscheck_ok",
    "cap_d_median",
    "cap_d_p10",
    "cap_d_p90",
    "cap_wl_median",
    "cap_wl_p10",
    "cap_wl_p90",
    "dec_d_median",
    "dec_d_p10",
    "dec_d_p90",
    "dec_wl_median",
    "dec_wl_p10",
    "dec_wl_p90",
    "cap_root_value_abs_mean",
    "dec_root_value_abs_mean",
    "cap_progress_median",
    "dec_progress_median",
    "cap_repetition_share",
    "cap_row_share",
    "cap_seat_split",
    "mode",
    "window",
    "cap_d_win_median",
    "cap_d_win_p10",
    "cap_d_win_p90",
    "cap_wl_win_median",
    "dec_d_win_median",
    "dec_d_win_p10",
    "dec_d_win_p90",
    "dec_wl_win_median",
    "cap_wl_abs_median",
    "cap_wl_spread",
    "dec_wl_abs_median",
    "dec_wl_spread",
    "cap_ce_unweighted",
    "dec_ce_unweighted",
    "noncap_draws",
    "mlh_n_dec_games",
    "mlh_n_dec_rows",
    "mlh_mae_dec",
    "mlh_mae_dec_b0",
    "mlh_mae_dec_b1",
    "mlh_mae_dec_b2",
    "mlh_mae_dec_b3",
    "mlh_bias_dec",
    "mlh_bias_dec_b0",
    "mlh_bias_dec_b1",
    "mlh_bias_dec_b2",
    "mlh_bias_dec_b3",
    "mlh_mae_dec_win",
    "mlh_oneply_median_abs_delta",
    "mlh_oneply_n",
)
"""Exactly the keys :func:`run_census` writes -- the JSON contract, in one place.

``cap_d_*`` / ``cap_wl_*`` are the forward pass's raw ``d`` and ``w - l`` over
the last :data:`VALUE_PLIES` plies of the cap games, in the **white** frame;
``dec_*`` are the same over the matched decisive sample. ``cap_root_value_*`` is
the stored scalar, kept as the secondary read F3 asks for -- it is the number a
contempt artefact flips the sign of between campaign 2 and campaign 3 while a
belief does not.

Everything from ``window`` down to ``noncap_draws`` is KOM-162's, and the module
docstring says what each one is for. Three of them are **nullable**, which is a
reported value and not a gap: ``*_win_*`` is ``null`` for a run given no
``--window`` (and for a class the window selected no position of), and
``*_ce_unweighted`` is ``null`` for a shard tree that carries no
:data:`WDL_TARGET_COLUMN`. A ``0.0`` in either place would be a number a reader
could mistake for a finding.

The ``mlh_*`` block is KOM-159's, on **its own sample** -- see
:func:`mlh_calibration` for why it cannot share the decisive one -- and every
number in it is in **plies**. All of them except the two counts are nullable,
for the same reason: an iteration with no decisive game measures no calibration,
and reporting a 0-ply mean absolute error for it would pass the go/no-go.

:data:`IDENTITY_KEYS` is **not** in here: those five keys say which net was
forwarded, which is a fact about the run rather than a measurement
:func:`run_census` takes, and this function is pure over an already-read
population that holds no checkpoint. The JSON a run writes is the two sets
together, and :func:`main` refuses to write anything else.
"""

IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "census_ckpt_role",
    "census_ckpt_path",
    "census_ckpt_step",
    "census_ckpt_sha256",
    "census_ckpt_journal_role",
)
"""Which net the report's numbers came out of (KOM-181, review EP-01/EV-09).

The five keys that make the module docstring's "the iteration's own checkpoint"
checkable *after the fact*, by a reader who has only the JSON:

* ``census_ckpt_role`` -- the role that was **asked for**,
  :data:`SERVED_PRODUCER` or :data:`POST_TRAINING`;
* ``census_ckpt_journal_role`` -- the journal role that maps to
  (:data:`CKPT_ROLES`), i.e. the record the digest was checked against. Under
  ``--later-head`` it is the role the file *failed* to match, and every key of
  such a report carries the :data:`LATER_HEAD_PREFIX` that says so;
* ``census_ckpt_path`` -- the checkpoint, **resolved**: a relative path in a
  results file is a path nobody can follow six months later;
* ``census_ckpt_step`` -- its training step, which is the number that made the
  KOM-159 confusion visible in hindsight (``N * 300`` against ``(N + 1) * 300``);
* ``census_ckpt_sha256`` -- the parameter digest actually computed here, so the
  claim can be re-checked against the journal without the file.
"""

SERVED_PRODUCER: Final[str] = "served_producer"
"""``--ckpt-role`` for the net that **played** the iteration's rows.

The default, because it is what every claim of the form "the head believed X
about these positions" needs, and because it is the one the KOM-159 censuses
silently did not use. The loop serves the newest EMA before self-play and
publishes the trained one afterwards, so this is the EMA at step ``N * 300`` and
the journal knows it as ``self_play@N``.
"""

POST_TRAINING: Final[str] = "post_training"
"""``--ckpt-role`` for the EMA published at the **end** of the iteration.

A legitimate thing to census -- "what did training on these rows produce" -- and
the journal's ``ema@N``, at step ``(N + 1) * 300``. It is simply not the
producer, and the difference is exactly one training iteration.
"""

CKPT_ROLES: Final[Mapping[str, str]] = {
    SERVED_PRODUCER: SELF_PLAY,
    POST_TRAINING: EMA,
}
"""``--ckpt-role`` -> the journal ``net_fingerprint`` role that records it.

The mapping is the whole of KOM-181's claim, so it is one table rather than a
condition spelled out at each use: ``boundary.record_fingerprints`` writes
``train``, ``ema`` and ``self_play`` per iteration, ``self_play`` being a digest
of the file ``CHECKPOINT_ENV`` actually named for that iteration's workers.
"""

LATER_HEAD_PREFIX: Final[str] = "laterhead_"
"""What every key of an off-role report is prefixed with (:func:`later_head_report`).

A prefix rather than a flag key, because a flag is a thing a reader has to
remember to look at: the failure this ticket exists to fix is a JSON that was
pooled with on-role censuses by a script reading ``cap_d_median``, and a
``laterhead_cap_d_median`` simply is not that key.
"""

EXIT_WRONG_NET: Final[int] = 2
"""Exit status for "this checkpoint is not the net you asked for".

Distinct from 1 so a shell loop can tell a refused *identity* -- the fixable
mistake of naming the wrong file -- from the script failing.
"""

_DAMAGE: Final[tuple[type[Exception], ...]] = (
    zipfile.BadZipFile,
    zlib.error,
    OSError,
    ValueError,
    KeyError,
    EOFError,
)
"""Everything a rotted shard can throw on the way back in -- the same tuple
:mod:`komugi.campaign.game_stats` catches, caught as a tuple rather than as bare
``Exception`` so a genuine bug here still escapes as a traceback."""


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


class ReplayError(ValueError):
    """A game's rows could not be replayed into a :class:`Game`."""


class BookedGameError(ReplayError):
    """The rows start past ply 0 and no prefix was supplied.

    Named rather than folded into :class:`ReplayError` because it is the one
    replay failure that is **expected**, counted (``book_excluded``) and not a
    finding: ~10% of campaign 2/3 games are booked.
    """


class IllegalActionError(ReplayError):
    """A recorded action is not legal at its ply in the replayed position.

    The failure :func:`mode_guard` exists to pre-empt, and the one a shard that
    does not belong to this engine produces. It names the ply, because "which
    ply" is the difference between a wrong start position (ply 0) and a replay
    that drifted (anywhere else).
    """


class ModeMismatch(ValueError):
    """The shard's start mode and this process's ``GUNGI_MODE`` disagree."""


class PlaneMapMismatch(ValueError):
    """The shard was encoded under a different plane map than this process."""


class CheckpointRoleError(ValueError):
    """``--ckpt`` is not the net ``--ckpt-role`` asked for (KOM-181).

    The base of the two refusals :func:`checkpoint_identity` raises, so
    :func:`main` catches one name and returns :data:`EXIT_WRONG_NET` for both:
    from the caller's side "there is no record to check against" and "the record
    says a different net" are the same answer -- *this census cannot say it
    forwarded the producer*.
    """


class MissingFingerprint(CheckpointRoleError):
    """The journal carries no fingerprint for this ``(iteration, role)``.

    An **error and not a skip**, which is the whole point: the census that
    started this ticket was wrong precisely because nothing checked, and a guard
    that quietly passes when the evidence is absent reproduces that exactly. The
    expected cases are a root with no journal at all, and iteration 0 under
    :data:`SERVED_PRODUCER` -- iteration 0 served freshly initialised weights, so
    there is no producer checkpoint to name and no ``self_play`` record to name
    it with.
    """


class WrongProducer(CheckpointRoleError):
    """The checkpoint's parameters are not the ones the journal records.

    Attributes:
        identity: What the file **is** -- role, path, step and its own computed
            digest. Carried on the exception so ``--later-head`` can report the
            truth about the net it went on to forward rather than re-deriving
            it.
        expected: The digest the journal holds for the requested role.
    """

    def __init__(
        self, message: str, *, identity: CheckpointIdentity, expected: str
    ) -> None:
        super().__init__(message)
        self.identity = identity
        self.expected = expected


# ---------------------------------------------------------------------------
# the shard, as games
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One self-play row, reduced to what a replay and a value read need.

    Attributes:
        ply: The row's own zero-based ply.
        to_play: Whose move it is, and therefore whose frame every other number
            on the row is in.
        action: The action id actually played, which is what the replay needs.
        root_value: ``v_mix`` at the root -- the contempt-contaminated scalar
            this script exists to replace, kept as the secondary read.
        wdl_target: The row's stored ``wdl_target``, mover-relative, or ``None``
            for a shard that does not carry the column. The label
            :func:`cross_entropy` scores the forward pass against (KOM-162, F8).
            Defaulted so a caller that is testing the replay -- which does not
            read it -- states only the four fields it is about.
    """

    ply: int
    to_play: int
    action: int
    root_value: float
    wdl_target: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One game's rows, in ply order, plus the two game-level facts.

    Attributes:
        game_id: The packed ``(worker, incarnation, game_index)`` triple.
        termination: ``0`` decisive, ``1`` draw (the turn cap included).
        total_plies: The game's **whole** length, book prefix included,
            reconstructed as ``ply + mlh_target / MLH_SCALE`` at the deepest
            row -- :mod:`komugi.campaign.game_stats`'s arithmetic, reused rather
            than re-derived, so the two censuses cannot disagree about which
            games hit the cap.
        rows: The rows **on disk**, ascending by ply. Shorter than
            ``total_plies`` for a booked game, which is what :attr:`booked`
            reads off.
    """

    game_id: int
    termination: int
    total_plies: int
    rows: tuple[Row, ...]

    @property
    def booked(self) -> bool:
        """Return whether this game's rows begin past ply 0.

        The mechanical detection design §3.1 specifies: a book prefix is
        replayed and never recorded (``worker.py:832-844``), so a booked game's
        first row is the first *searched* ply and nothing on the shard says what
        came before it.
        """
        return bool(self.rows) and self.rows[0].ply > 0

    def tail(self, plies: int) -> tuple[Row, ...]:
        """Return this game's last `plies` rows, or all of them if it is shorter."""
        return self.rows[-plies:] if plies > 0 else ()


def _games_from_columns(
    columns: Mapping[str, npt.NDArray[Any]],
) -> tuple[GameRecord, ...]:
    """Group concatenated shard columns into games, ascending by id.

    One ``lexsort`` on ``(ply, game_id)`` -- lexsort's LAST key is the primary
    one -- so a game's rows are contiguous and its deepest row is the last of
    its run, which is where :data:`~komugi.selfplay.records.MLH_SCALE`'s float32
    remainder carries the most precision (``game_stats`` says why).
    """
    game_id = columns["game_id"]
    ply = columns["ply"]
    order = np.lexsort((ply, game_id))
    game_id = game_id[order]
    ply = ply[order]
    remaining = columns["mlh_target"].astype(np.float64)[order]
    termination = columns["termination"][order]
    to_play = columns["to_play"][order]
    action = columns["action"][order]
    root_value = columns["root_value"].astype(np.float64)[order]
    # NaN is the "this shard declared no wdl_target" marker wdl_targets writes,
    # and it is one a real target cannot collide with: the label is a one-hot of
    # the game's outcome. A caller that passes no such column at all -- a test
    # building games by hand -- reads the same way.
    stored_wdl = columns.get(WDL_TARGET_COLUMN)
    targets = (
        np.full((game_id.size, WDL_SIZE), np.nan, dtype=np.float64)
        if stored_wdl is None
        else np.asarray(stored_wdl, dtype=np.float64)[order]
    )
    labelled = ~np.any(np.isnan(targets), axis=1)

    starts = np.concatenate((np.ones(1, dtype=bool), game_id[1:] != game_id[:-1]))
    first = np.flatnonzero(starts)
    last = np.concatenate(
        (first[1:] - 1, np.array([game_id.size - 1], dtype=first.dtype))
    )
    # Rounded rather than truncated: mlh_target is a float32 and int() of
    # 59.999996 is 59 (game_stats.py:533-535).
    totals = np.rint(
        ply[last].astype(np.float64) + remaining[last] / MLH_SCALE
    ).astype(np.int64)

    games: list[GameRecord] = []
    for index, (start, stop) in enumerate(zip(first, last)):
        rows = tuple(
            Row(
                ply=int(ply[row]),
                to_play=int(to_play[row]),
                action=int(action[row]),
                root_value=float(root_value[row]),
                wdl_target=(
                    tuple(float(value) for value in targets[row])
                    if labelled[row]
                    else None
                ),
            )
            for row in range(int(start), int(stop) + 1)
        )
        games.append(
            GameRecord(
                game_id=int(game_id[start]),
                termination=int(termination[start]),
                total_plies=int(totals[index]),
                rows=rows,
            )
        )
    return tuple(games)


def read_games(shard_dir: str | os.PathLike[str]) -> tuple[GameRecord, ...]:
    """Return every game under `shard_dir`, ascending by ``game_id``.

    Every ``shard-*.npz`` **at or below** `shard_dir` is read, at any depth, for
    :func:`komugi.campaign.game_stats.game_stats`'s reason: the layout between
    an iteration root and its worker directories is the loop's business.

    A shard that cannot be read is **skipped with a**
    :class:`~komugi.train.shards.ShardWarning` rather than raised on -- a
    diagnostic that abandons an iteration because one shard rotted is worse than
    one that reports the other 999 -- and a shard whose plane map is not this
    process's IS raised on (:func:`plane_map_guard`), because every position it
    would contribute is encoded in a language this process does not speak.
    """
    collected: dict[str, list[npt.NDArray[Any]]] = {
        name: [] for name in (*CENSUS_COLUMNS, WDL_TARGET_COLUMN)
    }
    for path in sorted(Path(shard_dir).rglob(SHARD_GLOB)):
        try:
            schema, count, columns = read_columns(path, CENSUS_COLUMNS)
        except _DAMAGE as error:
            warnings.warn(
                f"skipping unreadable shard {path}: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        if count == 0:
            continue
        plane_map_guard(path, schema.names)
        for name in CENSUS_COLUMNS:
            collected[name].append(columns[name])
        collected[WDL_TARGET_COLUMN].append(
            wdl_targets(path, schema.names, rows=count)
        )

    if not collected["game_id"]:
        return ()
    return _games_from_columns(
        {name: np.concatenate(parts) for name, parts in collected.items()}
    )


def wdl_targets(
    path: Path, names: Sequence[str], *, rows: int
) -> npt.NDArray[np.float64]:
    """Return `path`'s ``wdl_target`` column, or a NaN block if it has none.

    A second, one-column read rather than a widened
    :data:`CENSUS_COLUMNS`, for the reason :data:`WDL_TARGET_COLUMN` gives and
    the one :func:`plane_map_guard` already makes the same trade for: the census
    columns are *required* of every shard, so adding an optional field to them
    would turn every projection tree -- the unit suite's synthetic shards
    included -- into an unreadable shard skipped with a warning.

    A shard that declares the column but whose block is the wrong shape is
    treated as declaring none rather than raised on: it contributes no CE rows,
    which the ``null`` in the report then says out loud.
    """
    missing = np.full((rows, WDL_SIZE), np.nan, dtype=np.float64)
    if WDL_TARGET_COLUMN not in set(names):
        return missing
    try:
        _schema, count, columns = read_columns(path, (WDL_TARGET_COLUMN,))
    except _DAMAGE as error:
        warnings.warn(
            f"skipping unreadable {WDL_TARGET_COLUMN} of shard {path}: {error}",
            ShardWarning,
            stacklevel=2,
        )
        return missing
    values = np.asarray(columns[WDL_TARGET_COLUMN], dtype=np.float64)
    if count != rows or values.shape != (rows, WDL_SIZE):
        warnings.warn(
            f"{path} declares {WDL_TARGET_COLUMN} of shape {values.shape!r} for "
            f"{rows} rows, not {(rows, WDL_SIZE)!r}; its rows carry no label and "
            f"the unweighted cross-entropy is taken without them",
            ShardWarning,
            stacklevel=2,
        )
        return missing
    return values


def plane_map_guard(path: Path, names: Sequence[str]) -> int | None:
    """Refuse `path` if it was encoded under a different plane map (KOM-002 §4).

    The read-side half of the reserved-plane mitigation, mirroring
    :func:`komugi.selfplay.records._check_plane_map_version` and its
    distinction, which review F6's citation audit is explicit about: a
    *fingerprint* mismatch is skipped with a warning, only a **plane-map**
    mismatch raises. The fingerprint is a hash of names, dtypes and shapes, so
    filling one of the encoder's reserved planes 121-127 changes what plane 120
    *means* while leaving all three untouched -- and plane 120 is precisely what
    this script reads.

    A shard whose own schema does not declare the column returns ``None``
    without a refusal: the unit suite's synthetic shards and any future
    projection tree are legitimate shapes, exactly as
    :func:`komugi.campaign.game_stats._draft_rows` treats a missing draft column.

    Returns:
        The shard's plane-map version, or ``None`` if it declares none.

    Raises:
        PlaneMapMismatch: With **both** versions named.
    """
    if VERSION_COLUMN not in set(names):
        return None
    _schema, count, columns = read_columns(path, (VERSION_COLUMN,))
    if count == 0:
        return None
    version = int(columns[VERSION_COLUMN][0])
    if version != PLANE_MAP_VERSION:
        raise PlaneMapMismatch(
            f"{path} was encoded under plane map version {version}, but this "
            f"process encodes version {PLANE_MAP_VERSION}; the observation's "
            f"shape is identical across a plane-map change (the reserved planes "
            f"exist for exactly that), so the schema fingerprint cannot see "
            f"this, and planes {CROSSCHECK_PLANES} would be compared against "
            f"whatever the other map put there"
        )
    return version


def cap_games(
    shard_dir: str | os.PathLike[str], *, cap_plies: int = TURN_CAP_PLIES
) -> set[int]:
    """Return the ``game_id`` of every turn-capped game under `shard_dir`.

    **Identical by construction** to what
    :func:`komugi.campaign.game_stats.game_stats` counts as ``turn_cap_games``
    on the same tree: the same reconstructed length (``ply + mlh_target /
    MLH_SCALE`` at the deepest row) compared ``>=`` against the same
    :data:`~komugi.campaign.game_stats.TURN_CAP_PLIES`. The predicate is the
    **length**, not ``termination``, for that reason and no other -- the two
    numbers have to be the same number, or this script and the campaign's own
    telemetry describe different populations and neither can be checked against
    the other.

    A capped game whose ``termination`` is not a draw is warned about rather
    than dropped: the engine adjudicates the cap as a draw
    (``engine/game.py:614-616``), so such a game is a writer bug worth seeing.
    """
    return cap_ids(read_games(shard_dir), cap_plies=cap_plies)


def cap_ids(
    games: Sequence[GameRecord], *, cap_plies: int = TURN_CAP_PLIES
) -> set[int]:
    """Return the capped games' ids out of an already-read population."""
    found: set[int] = set()
    for game in games:
        if game.total_plies < cap_plies:
            continue
        if game.termination == TERMINATION_DECISIVE:
            warnings.warn(
                f"game {game.game_id} reached {game.total_plies} plies but is "
                f"recorded as decisive; the engine adjudicates the cap as a "
                f"draw, so this row's writer disagrees with its engine",
                ShardWarning,
                stacklevel=2,
            )
        found.add(game.game_id)
    return found


def decisive_sample(
    games: Sequence[GameRecord],
    *,
    size: int,
    cap_plies: int = TURN_CAP_PLIES,
) -> tuple[GameRecord, ...]:
    """Return up to `size` decisive games, longest first (design §3.1 item 2).

    "Matched on total plies where possible" means the **longest** decisive games
    the iteration has: they are the ones nearest the cap, and therefore the ones
    whose late positions are comparable to a cap game's late positions. Taking a
    uniform sample instead would compare the end of a 256-ply shuffle against
    the end of a 60-ply miniature, where a decisive value is expected and says
    nothing.

    Deterministic -- sorted by length, ties broken by ``game_id`` -- so two runs
    of this script over one iteration report the same sample.
    """
    decisive = [
        game
        for game in games
        if game.termination == TERMINATION_DECISIVE and game.total_plies < cap_plies
    ]
    decisive.sort(key=lambda game: (-game.total_plies, game.game_id))
    return tuple(decisive[:size])


def window_decisive_sample(
    games: Sequence[GameRecord],
    *,
    size: int,
    last: int,
    cap_plies: int = TURN_CAP_PLIES,
) -> tuple[GameRecord, ...]:
    """Return the decisive games still being played at ply `last` (KOM-162, F1).

    The window read compares the two classes **at a shared ply range**, which
    only means something for a decisive game that had not already ended inside
    it: a game that finished at ply 120 contributes nothing at plies 96-160
    except its first 25 plies, and pooling those against a cap game's would
    compare a middlegame with the moment before a mate.

    So the predicate is ``total_plies > last``, and the sample is the longest
    such games -- which is exactly :func:`decisive_sample`'s output filtered,
    because that list is already sorted by descending length: once one game in
    it is short enough to fail, so is every game after it. Written as a filter
    rather than as a second sort for that reason, so the two samples cannot
    drift apart, and so a window run replays no game the tail run did not.
    """
    return tuple(
        game
        for game in decisive_sample(games, size=size, cap_plies=cap_plies)
        if game.total_plies > last
    )


def all_decisive_games(
    games: Sequence[GameRecord], *, cap_plies: int = TURN_CAP_PLIES
) -> tuple[GameRecord, ...]:
    """Return **every** decisive game of the iteration, ascending by id.

    The moves-left calibration's population, and deliberately not
    :func:`decisive_sample`: that one is sized to the cap count and sorted
    longest-first, which is the right matched sample for "what does the head
    believe at the end of a cap game" and the wrong one for "is the head
    calibrated". At iteration 14 of campaign 5 the cap count was 2, so the
    matched sample was two games and 64 positions -- a standard error of
    15-20 plies against a 40-ply threshold, on the read that decides whether
    the utility is switched on (KOM-159 design review F3).

    Longest-first ordering is dropped with it: this is a census of the class,
    so it is ordered by ``game_id`` and nothing is truncated.
    """
    return tuple(
        sorted(
            (
                game
                for game in games
                if game.termination == TERMINATION_DECISIVE
                and game.total_plies < cap_plies
            ),
            key=lambda game: game.game_id,
        )
    )


def noncap_draws(
    games: Sequence[GameRecord],
    *,
    capped: set[int] | None = None,
    cap_plies: int = TURN_CAP_PLIES,
) -> int:
    """Return ``draws - turn_cap_games`` over `games` (R5's tripwire, F6).

    The two counts are :class:`komugi.campaign.game_stats.GameStats`'s own, by
    construction rather than by resemblance: ``draws`` is
    ``termination == TERMINATION_DRAW`` (``game_stats.py:538,548``) and the cap
    count is :func:`cap_ids`, which is already
    ``total_plies >= cap_plies`` (``game_stats.py:549``) and already asserted
    equal to ``turn_cap_games`` on one tree by the unit suite. Recomputed here
    rather than fetched because :func:`run_census` is pure over an
    already-read population -- it holds no path to hand ``game_stats`` -- and a
    number that travelled in from a second read of the same shards could
    disagree with the ``cap_games`` printed beside it.

    What it measures: a draw the engine reached by **repetition or stalemate**
    rather than by running out of turns. Campaign 4D down-weights rows labelled
    draw on the premise that essentially every draw is the cap; a non-cap draw
    class that grows is that premise failing, and C10 pre-registers ``<= 1`` at
    every iteration.

    A negative result is possible and is a finding, not a floor: it means a game
    reached the cap and was recorded as decisive, which :func:`cap_ids` has
    already warned about by id.

    Args:
        games: The iteration's whole population, not a sample: both counts are
            of the iteration.
        capped: The cap set if the caller already has it, so a census that has
            called :func:`cap_ids` once does not re-emit its warnings here.
        cap_plies: The length a game counts as capped at, passed through.
    """
    drawn = sum(1 for game in games if game.termination == TERMINATION_DRAW)
    ids = cap_ids(games, cap_plies=cap_plies) if capped is None else capped
    return drawn - len(ids)


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def _drafting_modes() -> tuple[str, ...]:
    """Return the start modes whose ply-0 position has BOTH draft bits set.

    Computed from :data:`komugi.engine.game._MODE_FENS` rather than listed, so a
    new mode classifies itself. Today: ``intermediate`` and ``advanced`` start
    from an empty board with both hands full (``fen.py:28-35``, the ``wb`` draft
    field), ``intro`` and ``beginner`` start from a laid-out board past the
    draft.
    """
    drafting: list[str] = []
    for mode, fen in engine_game._MODE_FENS.items():
        board = board_from_fen(fen)
        if board.draft_w and board.draft_b:
            drafting.append(mode)
    return tuple(sorted(drafting))


def engine_mode() -> str:
    """Return the start mode this process resolved ``GUNGI_MODE`` to.

    Read at call time from :mod:`komugi.engine.game`'s own module-level
    ``_env_mode`` (``game.py:267``) rather than captured at import, because that
    is the value :class:`~komugi.engine.game.Game` will actually start from and
    a census that named a different one would be reporting a guess.
    """
    return str(engine_game._env_mode)


def mode_guard(
    first_row_observation: npt.NDArray[Any], *, mode: str | None = None
) -> str:
    """Return the engine mode, refusing if the shard was written in another.

    Review F7: ``engine/game.py:267`` defaults ``GUNGI_MODE`` to ``"intro"``, so
    a campaign-3 (advanced) shard replayed in a default process fails at ply 0
    with an illegal action -- a confusing failure several hundred replays deep,
    when the real fault is one environment variable. The shard states its own
    mode class in its first row: drafting planes
    :data:`~komugi.campaign.game_stats.DRAFT_PLANE_WHITE` **and**
    :data:`~komugi.campaign.game_stats.DRAFT_PLANE_BLACK` both set means both
    sides still hold drafting rights, which at ply 0 is true of exactly the
    drafting modes.

    What this can and cannot say, stated rather than implied: the planes
    separate the drafting modes from the non-drafting ones and **not**
    ``advanced`` from ``intermediate``, whose ply-0 positions differ only in the
    FEN's mode field. That is the whole discrimination available from an
    observation, and it is enough for the failure F7 is about.

    Args:
        first_row_observation: The stored ``observation`` of a game's row at
            **ply 0**. A booked game's first row is past the draft's start and
            makes no claim about the mode, which is why the caller picks an
            unbooked game.
        mode: The engine mode to check against; :func:`engine_mode` when
            omitted. A parameter so a test can state both sides of the
            comparison instead of mutating a process-global.

    Returns:
        The engine mode, which is then reported beside every number taken under
        it.

    Raises:
        ModeMismatch: Naming both the shard's implied mode class and the
            engine's own mode.
    """
    resolved = engine_mode() if mode is None else mode
    planes = np.asarray(first_row_observation)
    flat = planes.reshape(planes.shape[0], -1)
    shard_drafts = bool(
        np.any(flat[DRAFT_PLANE_WHITE] != 0.0) and np.any(flat[DRAFT_PLANE_BLACK] != 0.0)
    )
    drafting = _drafting_modes()
    engine_drafts = resolved in drafting
    if shard_drafts == engine_drafts:
        return resolved

    if shard_drafts:
        raise ModeMismatch(
            f"the shard's first row has both drafting planes "
            f"({DRAFT_PLANE_WHITE}/{DRAFT_PLANE_BLACK}) set, so it was written "
            f"in a drafting mode -- one of {drafting} -- but this process "
            f"resolved GUNGI_MODE to {resolved!r}, which starts past the draft; "
            f"replaying it here fails at ply 0. Export GUNGI_MODE=advanced "
            f"(review F7)"
        )
    raise ModeMismatch(
        f"the shard's first row has no drafting planes "
        f"({DRAFT_PLANE_WHITE}/{DRAFT_PLANE_BLACK}) set, so it was written in a "
        f"non-drafting mode -- one of "
        f"{tuple(sorted(set(engine_game._MODE_FENS) - set(drafting)))} -- but "
        f"this process resolved GUNGI_MODE to {resolved!r}, which starts in the "
        f"draft; replaying it here fails at ply 0 (review F7)"
    )


def observation_crosscheck(
    game: Game,
    stored_obs: npt.NDArray[Any],
    planes: Sequence[int] = CROSSCHECK_PLANES,
) -> bool:
    """Return whether the replayed position's `planes` equal the stored row's.

    Design §3.1 item 4, and the thing that makes every other number here
    trustworthy: a replay that dropped or duplicated a ply still produces a
    perfectly plausible board, but plane 115 (the turn clock) is off by exactly
    that many plies and planes 116/117/120 follow the history too. So the four
    history planes are compared and the cells are not -- the cells are what the
    board and the action list agree on by construction.

    Exact equality is not asserted: both sides are float32 quotients of the same
    integers (``min(plies, MAX_TURNS) / MAX_TURNS``), so they agree bit for bit
    in practice,
    but the comparison is a tolerance because a float equality that is *usually*
    exact is a test that fails once a year for a reason nobody can reproduce.
    """
    replayed = np.asarray(game.observation(), dtype=np.float64)
    stored = np.asarray(stored_obs, dtype=np.float64)
    if replayed.shape != stored.shape:
        return False
    return all(
        bool(np.allclose(replayed[plane], stored[plane], atol=1e-6))
        for plane in planes
    )


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------


def new_game() -> Game:
    """Return a fresh game under the rule self-play and the ladder play under.

    :attr:`~komugi.engine.repetition.RepetitionRule.FULL` **explicitly**, though
    it is also the default: KOM-141 makes the key a choice, and a game refereed
    under one key and replayed under the other ends at a different ply -- which
    is the failure ``termination_census.py`` documents at length and refuses
    rather than classifies.
    """
    return Game(repetition_rule=RepetitionRule.FULL)


Observer = Callable[[Game, Row], None]
"""Called at the position each row was RECORDED at -- before its action is
played -- which is the position the row's ``root_value`` and ``wdl_target``
describe, and the one a forward pass must be given."""


def replay_game(
    rows: Sequence[Row],
    book: Sequence[int] | None = None,
    *,
    game_factory: Callable[[], Game] = new_game,
    observer: Observer | None = None,
) -> Game:
    """Replay `rows` onto a fresh :class:`Game` and return the final position.

    The rows are sorted by ply here rather than trusted to arrive sorted: they
    are concatenated out of a shard tree whose file order is the filesystem's,
    and a replay that played ply 7 before ply 6 would fail as an illegal action
    and be reported as a corrupt shard.

    Args:
        rows: One game's rows. Sorted by ply internally.
        book: The prefix the game started from, as action ids, or ``None``. A
            booked game records no row for its prefix, so without one its rows
            cannot be replayed at all -- review F6 is right that the prefix is
            *recoverable* (book selection is a pure function of the game id,
            ``worker.py:810-830``), but it needs the book file and the worker's
            ``run_seed``/``book_fraction``, which are not on the shard. This
            parameter is where a caller that has them passes them in; the CLI
            does not, and counts the game instead.
        game_factory: Builds the start position. The seam a test uses to replay
            an advanced-mode game without mutating a process-global
            (``tests/selfplay/unit/test_min_placements.py``'s idiom).
        observer: Called as ``observer(game, row)`` at each row's own position,
            before its action is played.

    Returns:
        The game after the last recorded action.

    Raises:
        BookedGameError: If the first row is past ply 0 and `book` is ``None``.
        ReplayError: If `book`'s length does not reach the first row's ply --
            a prefix that stops short (or overshoots) is the wrong prefix, and
            replaying it would silently census some other game.
        IllegalActionError: If an action is not legal at its ply, naming the
            ply. Under a wrong ``GUNGI_MODE`` this fires at ply 0, which is what
            :func:`mode_guard` exists to say more usefully.
    """
    ordered = tuple(sorted(rows, key=lambda row: row.ply))
    if not ordered:
        raise ReplayError("a game with no rows cannot be replayed")

    start = ordered[0].ply
    prefix: tuple[int, ...] = tuple(book or ())
    if start > 0 and book is None:
        raise BookedGameError(
            f"game's first row is at ply {start}, so it was played from a "
            f"{start}-ply book prefix that was replayed and never recorded "
            f"(worker.py:832-844); it is excluded from the replay half and "
            f"counted in book_excluded"
        )
    if len(prefix) != start:
        raise ReplayError(
            f"the supplied book prefix is {len(prefix)} plies but the first "
            f"recorded row is at ply {start}; a prefix that does not end where "
            f"the rows begin is not this game's prefix"
        )

    game = game_factory()
    for ply, action in enumerate(prefix):
        _play(game, ply, action)
    for row in ordered:
        if observer is not None:
            observer(game, row)
        _play(game, row.ply, row.action)
    return game


def _play(game: Game, ply: int, action: int) -> None:
    """Play one recorded action, refusing it if the engine calls it illegal."""
    mask = np.asarray(game.valid_moves(), dtype=np.uint8)
    if not 0 <= action < mask.size or mask[action] != 1:
        raise IllegalActionError(
            f"ply {ply}: action {action!r} is not legal in the replayed "
            f"position; this row was not produced by this engine in mode "
            f"{engine_mode()!r} (see mode_guard)"
        )
    game.play_action(int(action))


@dataclass(frozen=True, slots=True)
class Position:
    """One replayed position, as the census reads it.

    Attributes:
        ply: The row's own zero-based ply.
        to_play: Whose move it is -- the frame the network's ``wdl`` comes back
            in, and therefore what :func:`white_frame` needs.
        progress_plies: Plies since the last capture, betrayal or arata, read
            back out of the encoder's plane 120 (``progress / MAX_TURNS``)
            rather than off ``Game._progress_plies``, so the read goes through a
            public path (design §3.1 item 3).
        repetition: ``Game.repetition_count()`` -- occurrences of this position,
            this one included.
        observation: The encoder's planes for this position, which is what the
            forward pass is given.
        wdl_target: The row's stored ``wdl_target``, carried through the replay
            unchanged so :func:`cross_entropy` scores the forward pass against
            the label that row was actually trained on. ``None`` for a shard
            that declares no such column; defaulted for a caller that is
            testing the windowing rather than the loss.
    """

    ply: int
    to_play: int
    progress_plies: int
    repetition: int
    observation: npt.NDArray[np.float32]
    wdl_target: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class Replay:
    """What one game's replay produced.

    Attributes:
        game_id: The game.
        positions: Its last :data:`PROGRESS_WINDOW` positions, in ply order.
            Bounded because the whole point is the *end* of the game and a
            256-ply game's observations are 9.5 MB.
        windowed: The positions inside the ``--window`` ply range, if one was
            asked for, and empty otherwise. A **second** buffer rather than a
            widened :attr:`positions`, because the two reads want different
            things: the tail read wants the last N plies of every game whatever
            their numbers, and the window read wants one fixed ply range
            wherever it falls. A mid-game window of a 256-ply game is nowhere
            near its last :data:`PROGRESS_WINDOW` positions, and widening the
            tail to reach it would hold the whole game's observations -- 9.5 MB
            a game -- to use sixty of them.
        reason: :meth:`~komugi.engine.game.Game.termination_reason` at the final
            position.
        crosschecked: Rows whose stored observation was compared.
        mismatches: How many of those disagreed. Any is fatal to the pass.
    """

    game_id: int
    positions: tuple[Position, ...]
    reason: str
    crosschecked: int
    mismatches: int
    windowed: tuple[Position, ...] = ()


def replay_with_positions(
    game: GameRecord,
    stored: Mapping[tuple[int, int], npt.NDArray[Any]],
    *,
    book: Sequence[int] | None = None,
    game_factory: Callable[[], Game] = new_game,
    window: int = PROGRESS_WINDOW,
    ply_window: tuple[int, int] | None = None,
) -> Replay:
    """Replay `game`, keeping its last `window` positions and cross-checking.

    `stored` maps ``(game_id, ply)`` to that row's **stored** observation, for
    the bounded sample :data:`CROSSCHECK_ROWS` selects. A row that is not in it
    is simply not cross-checked; a row that is must agree on every plane of
    :data:`CROSSCHECK_PLANES`.

    `ply_window` is KOM-162's ``(first, last)``, **inclusive at both ends**, and
    it fills :attr:`Replay.windowed` in addition to -- never instead of -- the
    tail. A game whose rows do not reach it contributes nothing to it, which is
    a count the caller can see rather than a silent short sample.
    """
    kept: list[Position] = []
    windowed: list[Position] = []
    checked = 0
    mismatched = 0

    def observe(replayed: Game, row: Row) -> None:
        """Record the position this row was taken at, and cross-check it."""
        nonlocal checked, mismatched
        obs = np.asarray(replayed.observation(), dtype=np.float32)
        position = Position(
            ply=row.ply,
            to_play=row.to_play,
            # plane 120 is min(progress_plies, MAX_TURNS) / MAX_TURNS,
            # constant across the board, so any square reads it.
            progress_plies=int(round(float(obs[PROGRESS_PLANE, 0, 0]) * MAX_TURNS)),
            repetition=int(replayed.repetition_count()),
            observation=obs,
            wdl_target=row.wdl_target,
        )
        kept.append(position)
        if len(kept) > window:
            del kept[0]
        if ply_window is not None and ply_window[0] <= row.ply <= ply_window[1]:
            windowed.append(position)
        reference = stored.get((game.game_id, row.ply))
        if reference is not None:
            checked += 1
            if not observation_crosscheck(replayed, reference):
                mismatched += 1

    final = replay_game(
        game.rows, book, game_factory=game_factory, observer=observe
    )
    return Replay(
        game_id=game.game_id,
        positions=tuple(kept),
        reason=str(final.termination_reason()),
        crosschecked=checked,
        mismatches=mismatched,
        windowed=tuple(windowed),
    )


# ---------------------------------------------------------------------------
# the statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgressStats:
    """The two shuffle numbers over one game's tail (review F8)."""

    median_progress: float
    repetition_share: float


def progress_stats(
    positions: Sequence[Position], *, window: int = PROGRESS_WINDOW
) -> ProgressStats:
    """Return the progress clock's median and the repeated-position share.

    Over the last `window` plies of ONE game, which is the unit review F8 puts
    the threshold on: "cap median >= 24 and >= 2x the decisive median". The
    *magnitude* rather than the non-zero share, because plane 120 is non-zero on
    ~85-90% of an ordinary decisive game's plies too -- every ply that is not
    itself a capture, betrayal or arata -- so the share cannot discriminate and
    the magnitude can.

    The repetition share is ``repetition_count() >= REPETITION_SHUFFLE``: a
    position the game has been in before. Not fourfold, which a capped game by
    definition never reached.

    An empty window reads as zeros rather than raising -- a game with no
    replayed positions is a finding the counts beside this report already carry.
    """
    tail = tuple(positions[-window:]) if window > 0 else ()
    if not tail:
        return ProgressStats(median_progress=0.0, repetition_share=0.0)
    clocks = np.asarray([position.progress_plies for position in tail], dtype=np.float64)
    repeated = sum(
        1 for position in tail if position.repetition >= REPETITION_SHUFFLE
    )
    return ProgressStats(
        median_progress=float(np.median(clocks)),
        repetition_share=repeated / len(tail),
    )


class Evaluator(Protocol):
    """The one method the forward pass uses of a batch evaluator.

    A :class:`~komugi.selfplay.inference.InProcessEvaluator` satisfies it; so
    does the unit suite's stub, which is what keeps every number below testable
    without a checkpoint.
    """

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a batch of positions."""


def forward_wdl(
    evaluator: Evaluator,
    observations: Sequence[npt.NDArray[Any]],
    *,
    batch: int = FORWARD_BATCH,
) -> npt.NDArray[np.float32]:
    """Return the raw ``(N, 3)`` ``wdl`` for `observations`, mover-relative.

    **Raw**, which is the whole reason this half exists (review F3): the seam
    carries win/draw/loss probabilities and deliberately has no scalar value
    field, so nothing here has been collapsed through a ``draw_score`` the way
    the shard's ``root_value`` has. Contempt is a property of the reader
    (:func:`~komugi.selfplay.inference.wdl_utility`) and this reader applies
    none.

    The legality mask is all-ones: it is validated and otherwise unused at this
    seam in P0a (:class:`~komugi.selfplay.inference.InProcessEvaluator`), and
    the value head does not read it. Re-deriving each position's true mask would
    cost a full move generation per position to produce a number no output
    depends on.
    """
    if not observations:
        return np.zeros((0, 3), dtype=np.float32)
    stacked = np.stack(
        [np.asarray(obs, dtype=np.float32) for obs in observations]
    )
    out: list[npt.NDArray[np.float32]] = []
    for start in range(0, stacked.shape[0], max(batch, 1)):
        chunk = stacked[start : start + max(batch, 1)]
        legal = np.ones((chunk.shape[0], NUM_ACTIONS), dtype=np.bool_)
        out.append(np.asarray(evaluator.evaluate(chunk, legal).wdl, dtype=np.float32))
    return np.concatenate(out)


def forward_moves_left(
    evaluator: Evaluator,
    observations: Sequence[npt.NDArray[Any]],
    *,
    batch: int = FORWARD_BATCH,
) -> npt.NDArray[np.float64]:
    """Return the moves-left head's answer for `observations`, **in plies**.

    ``MAX_TURNS * clip(moves_left, 0, 1)``, which is the inverse of the target
    the recorder writes (``mlh_target = (total_plies - ply) * MLH_SCALE``,
    ``selfplay/records.py``) and exactly the quantity
    :func:`komugi.selfplay.adapter.mlh_utility_term` reads. The clip is the
    head's own: it is a linear regression with no output squashing
    (``net/heads.py``'s ``MLHHead``), so an untrained one answers outside
    ``[0, 1]`` and an unclipped read would report a negative number of plies.

    A separate function from :func:`forward_wdl` rather than a widened one
    because the two are taken over **different position sets**: the ``wdl``
    reads are the last :data:`VALUE_PLIES` plies of a matched sample, and this
    is every row of every decisive game (:func:`mlh_calibration`). Fusing them
    would mean forward-passing one of the two sets for a number nobody reads.

    Batched for the same reason and with the same bound: the legality mask is
    ``(N, 37341)`` bool and is all-ones because no head under measurement reads
    it.

    Raises:
        ValueError: If the seam returns a different number of answers than the
            chunk it was given -- a plumbing error that would otherwise be
            reported as a calibration.
    """
    if not observations:
        return np.zeros((0,), dtype=np.float64)
    stacked = np.stack([np.asarray(obs, dtype=np.float32) for obs in observations])
    width = max(batch, 1)
    out: list[npt.NDArray[np.float32]] = []
    for start in range(0, stacked.shape[0], width):
        chunk = stacked[start : start + width]
        legal = np.ones((chunk.shape[0], NUM_ACTIONS), dtype=np.bool_)
        answered = np.asarray(
            evaluator.evaluate(chunk, legal).moves_left, dtype=np.float32
        ).reshape(-1)
        if answered.shape[0] != chunk.shape[0]:
            raise ValueError(
                f"the evaluator returned {answered.shape[0]} moves_left values "
                f"for a batch of {chunk.shape[0]} positions; the head answers "
                f"once per position"
            )
        out.append(answered)
    raw = np.concatenate(out).astype(np.float64)
    return np.asarray(MAX_TURNS * np.clip(raw, 0.0, 1.0), dtype=np.float64)


def white_frame(
    wdl: npt.NDArray[Any], to_play: Sequence[int]
) -> npt.NDArray[np.float32]:
    """Return `wdl` re-signed so column 0 is always WHITE's win probability.

    The network answers in the **side to move's** perspective
    (:class:`~komugi.selfplay.inference.NetEval`), so a black-to-move row's
    ``w`` is black's. Pooling the two frames without flipping would report the
    mean of a quantity and its negation -- which is how review F3 shows C11
    scores below 90% on a perfectly coherent net.

    The draw column is untouched: ``d`` has no frame, which is exactly why the
    pre-registration is stated on it.
    """
    values = np.asarray(wdl, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"wdl must be (N, 3), got {values.shape!r}")
    sides = np.asarray(to_play, dtype=np.int64)
    if sides.shape != (values.shape[0],):
        raise ValueError(
            f"to_play must be ({values.shape[0]},), got {sides.shape!r}: one "
            f"side per evaluated position"
        )
    flipped = values.copy()
    black = sides == BLACK
    flipped[black, 0] = values[black, 2]
    flipped[black, 2] = values[black, 0]
    return flipped


def row_share(games: Sequence[GameRecord], ids: Iterable[int]) -> float:
    """Return the share of **rows** the games named by `ids` contribute.

    Review F5.2's number, and it is not the share of *games*: a cap game
    contributes 256 rows while campaign 3's average decisive game contributes
    ~155, so 15.5% of games is ~23% of the rows the value head is trained on.
    The count is of rows **on disk** -- a booked game's prefix is not a training
    row and is not counted here, exactly as it is not counted by the trainer.
    """
    named = set(ids)
    total = sum(len(game.rows) for game in games)
    if total == 0:
        return 0.0
    return sum(len(game.rows) for game in games if game.game_id in named) / total


def seat_split(games: Sequence[GameRecord]) -> dict[str, int]:
    """Return how many of `games` had white to move on their first recorded row.

    U5's read. Not always ply 0: a booked game's first recorded row is the ply
    the book left off at, so an odd-length prefix puts black on it. That is the
    honest statement of the quantity -- "who moved first in the part of this
    game that was searched" -- and a split far from 50/50 is a finding about the
    book as much as about the seats.
    """
    white = sum(1 for game in games if game.rows and game.rows[0].to_play == WHITE)
    return {"white": white, "black": len(games) - white}


def quantiles(values: npt.NDArray[Any]) -> tuple[float, float, float]:
    """Return ``(median, p10, p90)`` of `values`; zeros for an empty sample."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return (0.0, 0.0, 0.0)
    return (
        float(np.median(data)),
        float(np.percentile(data, P10)),
        float(np.percentile(data, P90)),
    )


def cross_entropy(
    wdl: npt.NDArray[Any], targets: Sequence[tuple[float, ...] | None]
) -> float | None:
    """Return the mean unweighted WDL cross-entropy, or ``None`` for no labels.

    Review F8's read (M3), and the reason it exists: the journal's ``loss_wdl``
    is a **weighted** mean (``losses.py:742-746``), so under campaign 4D -- which
    down-weights exactly the rows labelled draw -- it falls relative to the
    control by construction, whatever the head actually learnt. This is the same
    loss with every row counted once, which is the only version of it that
    compares two arms trained under different weights.

    Both sides are **mover-relative**: the network answers in the side to move's
    perspective and ``wdl_target`` is stored in it (see
    :data:`WDL_TARGET_COLUMN`), so no :func:`white_frame` flip is applied. A flip
    of one side only would report the loss of the *opponent's* label.

    ``-sum(target * log(p))`` per row, meaned in float64 -- the ordinary
    cross-entropy against a soft target, which for self-play's one-hot outcome
    label is ``-log`` of the probability the net gave the result that happened.
    Rows whose `targets` entry is ``None`` are dropped, not zero-filled; a pass
    with no labelled row at all returns ``None`` rather than ``0.0``, which is
    the score of a perfect head.

    Raises:
        ValueError: If `wdl` and `targets` are different lengths, or if a target
            is not :data:`WDL_SIZE` long -- either is a plumbing error that
            would otherwise be reported as a loss.
    """
    probabilities = np.asarray(wdl, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != WDL_SIZE:
        raise ValueError(f"wdl must be (N, {WDL_SIZE}), got {probabilities.shape!r}")
    if len(targets) != probabilities.shape[0]:
        raise ValueError(
            f"targets is {len(targets)} long but wdl has "
            f"{probabilities.shape[0]} rows: one stored label per evaluated "
            f"position"
        )
    rows = [index for index, target in enumerate(targets) if target is not None]
    if not rows:
        return None
    labels = np.asarray(
        [targets[index] for index in rows], dtype=np.float64
    )
    if labels.shape != (len(rows), WDL_SIZE):
        raise ValueError(
            f"every stored wdl_target must be {WDL_SIZE} long, got "
            f"{labels.shape!r}"
        )
    logs = np.log(np.clip(probabilities[rows], CE_FLOOR, 1.0))
    return float(np.mean(-np.sum(labels * logs, axis=1)))


def root_value_abs_mean(games: Sequence[GameRecord], *, plies: int = VALUE_PLIES) -> float:
    """Return the mean ``|root_value|`` over each game's last `plies` rows.

    The secondary read F3 keeps and demotes: it needs no replay, so it covers
    the booked games too, and its *sign behaviour across campaigns* is what
    separates a contempt artefact from a belief -- campaign 2 (``+0.20``) and
    campaign 3 (``-0.20``) carry opposite contamination, so an artefact flips
    between them and a belief does not.
    """
    rows = [abs(row.root_value) for game in games for row in game.tail(plies)]
    return float(np.mean(rows)) if rows else 0.0


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------


def crosscheck_rows(
    games: Sequence[GameRecord], *, rows: int = CROSSCHECK_ROWS
) -> set[tuple[int, int]]:
    """Return the ``(game_id, ply)`` keys whose stored observation is wanted.

    Each selected game's **last** `rows` rows, plus its first row -- the first
    because :func:`mode_guard` reads it, the last because a replay that diverged
    at any ply is still diverged at the end (see :data:`CROSSCHECK_ROWS`).
    """
    wanted: set[tuple[int, int]] = set()
    for game in games:
        if not game.rows:
            continue
        wanted.add((game.game_id, game.rows[0].ply))
        for row in game.tail(rows):
            wanted.add((game.game_id, row.ply))
    return wanted


def guard_row(games: Sequence[GameRecord]) -> set[tuple[int, int]]:
    """Return the one ``(game_id, ply)`` key :func:`mode_guard` needs, if any.

    The first **unbooked** game's first row, which is at ply 0 and is therefore
    the only kind of row whose drafting planes say what mode the shard was
    written in. Separate from :func:`crosscheck_rows` because the guard must run
    even when nothing was selected for replay -- an iteration with no cap game
    at all still has a mode, and finding that out is not conditional on there
    being something wrong with it.
    """
    for game in games:
        if not game.booked and game.rows:
            return {(game.game_id, game.rows[0].ply)}
    return set()


def read_observations(
    shard_dir: str | os.PathLike[str], wanted: set[tuple[int, int]]
) -> dict[tuple[int, int], npt.NDArray[np.float32]]:
    """Return the stored observations of the `wanted` rows, and no others.

    The expensive read, bounded: one shard's ``observation`` column is
    materialised at a time and everything outside `wanted` is dropped before the
    next shard is opened, so the high-water mark is one shard rather than an
    iteration (:mod:`komugi.campaign.game_stats` makes the same trade for the
    same reason).

    A shard whose schema does not declare ``observation`` contributes nothing,
    without a warning: that is a projection tree, not damage.
    """
    found: dict[tuple[int, int], npt.NDArray[np.float32]] = {}
    if not wanted:
        return found
    for path in sorted(Path(shard_dir).rglob(SHARD_GLOB)):
        try:
            schema, count, columns = read_columns(path, ("game_id", "ply"))
        except _DAMAGE as error:
            warnings.warn(
                f"skipping unreadable shard {path}: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        if count == 0 or "observation" not in set(schema.names):
            continue
        keys = list(zip(columns["game_id"].tolist(), columns["ply"].tolist()))
        hits = [index for index, key in enumerate(keys) if key in wanted]
        if not hits:
            continue
        try:
            _schema, _count, payload = read_columns(path, OBSERVATION_COLUMNS)
        except _DAMAGE as error:
            warnings.warn(
                f"skipping unreadable observations of shard {path}: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        observation = payload["observation"]
        for index in hits:
            found[(int(keys[index][0]), int(keys[index][1]))] = np.asarray(
                observation[index], dtype=np.float32
            )
    return found


@dataclass(frozen=True, slots=True)
class Sampled:
    """One class of games (capped or decisive) after replay and forward pass."""

    games: tuple[GameRecord, ...]
    replays: tuple[Replay, ...]
    excluded: int
    failures: int
    mismatches: int
    crosschecked: int


def _replay_all(
    games: Sequence[GameRecord],
    stored: Mapping[tuple[int, int], npt.NDArray[Any]],
    *,
    game_factory: Callable[[], Game] = new_game,
    expect: str | None = None,
    ply_window: tuple[int, int] | None = None,
) -> Sampled:
    """Replay every game, counting the two failure modes separately.

    A booked game is ``excluded`` (expected, ~10%); anything else that refuses
    is a ``failure`` and is warned about with its id, because a replay that
    fails on an *unbooked* game means the shard, the engine or ``GUNGI_MODE``
    disagree and the report's other numbers are taken over a smaller population
    than its ``games`` count suggests.

    `expect` is the termination reason the class must end on -- ``turn_cap`` for
    the capped games (``engine/game.py:643-705``). A replayed game that ends
    elsewhere is counted as a failure: the length arithmetic and the engine's
    own adjudication then disagree about what this game was.

    `ply_window` is passed through to every replay, so a windowed run costs one
    replay per game and not two.
    """
    replays: list[Replay] = []
    excluded = 0
    failures = 0
    for game in games:
        if game.booked:
            excluded += 1
            continue
        try:
            replay = replay_with_positions(
                game, stored, game_factory=game_factory, ply_window=ply_window
            )
        except ReplayError as error:
            failures += 1
            warnings.warn(
                f"game {game.game_id} did not replay: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        if expect is not None and replay.reason != expect:
            failures += 1
            warnings.warn(
                f"game {game.game_id} replayed to {replay.reason!r}, not "
                f"{expect!r}; its reconstructed length and the engine's own "
                f"adjudication disagree",
                ShardWarning,
                stacklevel=2,
            )
            continue
        replays.append(replay)
    return Sampled(
        games=tuple(games),
        replays=tuple(replays),
        excluded=excluded,
        failures=failures,
        mismatches=sum(replay.mismatches for replay in replays),
        crosschecked=sum(replay.crosschecked for replay in replays),
    )


def _value_positions(sampled: Sampled, *, plies: int = VALUE_PLIES) -> list[Position]:
    """Return the last `plies` positions of every replayed game, pooled."""
    return [
        position
        for replay in sampled.replays
        for position in replay.positions[-plies:]
    ]


def _window_positions(
    sampled: Sampled, *, keep: Iterable[int] | None = None
) -> list[Position]:
    """Return the in-window positions of the replayed games, pooled.

    `keep` restricts the pool to a subset of the class's games by id, which is
    how the decisive side is narrowed to :func:`window_decisive_sample` without
    replaying it again: the tail run has already replayed a superset of it.
    """
    wanted = None if keep is None else set(keep)
    return [
        position
        for replay in sampled.replays
        if wanted is None or replay.game_id in wanted
        for position in replay.windowed
    ]


@dataclass(frozen=True, slots=True)
class Beliefs:
    """What one forward pass over one class of positions says.

    Attributes:
        positions: How many positions were evaluated. Zero is why every other
            field can be a floor value without lying about it.
        d: ``(median, p10, p90)`` of the draw probability, which has no frame.
        wl: ``(median, p10, p90)`` of ``w - l`` in the **white** frame -- a
            seat-bias read, as review F2 insists it be called.
        wl_abs_median: The median of ``|w - l|``. F2's magnitude read: a head
            that is confidently winning for whichever side moves has a signed
            median near zero and this number near one.
        wl_spread: ``p90 - p10`` of the signed ``w - l``. The second half of
            F2's C5, and the width its table shows the median hiding.
        ce_unweighted: The mean unweighted cross-entropy against the stored
            ``wdl_target`` (F8/M3), or ``None`` when no position carried one.
    """

    positions: int
    d: tuple[float, float, float]
    wl: tuple[float, float, float]
    wl_abs_median: float
    wl_spread: float
    ce_unweighted: float | None


NO_BELIEFS: Final[Beliefs] = Beliefs(
    positions=0,
    d=(0.0, 0.0, 0.0),
    wl=(0.0, 0.0, 0.0),
    wl_abs_median=0.0,
    wl_spread=0.0,
    ce_unweighted=None,
)
"""What an empty class reads as -- the zeros :func:`quantiles` already returns
for an empty sample, so a class with no replayed position reports what it always
did, beside a ``positions`` of 0 that says why."""


def _beliefs(evaluator: Evaluator, positions: Sequence[Position]) -> Beliefs:
    """Return the white-framed quantiles, F2's magnitudes and F8's loss.

    One forward pass for all of them: the CE is scored on the raw mover-relative
    ``wdl`` the evaluator returned and the quantiles on the white-framed copy of
    it, so the two reads cannot be taken over different position sets or
    different nets.
    """
    if not positions:
        return NO_BELIEFS
    wdl = forward_wdl(evaluator, [position.observation for position in positions])
    framed = white_frame(wdl, [position.to_play for position in positions])
    signed = np.asarray(framed[:, 0] - framed[:, 2], dtype=np.float64)
    wl = quantiles(signed)
    return Beliefs(
        positions=len(positions),
        d=quantiles(framed[:, 1]),
        wl=wl,
        wl_abs_median=float(np.median(np.abs(signed))),
        wl_spread=wl[2] - wl[1],
        ce_unweighted=cross_entropy(
            wdl, [position.wdl_target for position in positions]
        ),
    )


def _pooled_progress(sampled: Sampled) -> tuple[float, float]:
    """Return the median per-game progress clock and the mean repetition share.

    The unit is the **game**, not the ply: every cap game contributes a full
    :data:`PROGRESS_WINDOW` of plies and a short decisive game contributes
    fewer, so pooling plies would weight the long games up in a comparison whose
    whole subject is length.
    """
    stats = [progress_stats(replay.positions) for replay in sampled.replays]
    if not stats:
        return (0.0, 0.0)
    return (
        float(np.median([stat.median_progress for stat in stats])),
        float(np.mean([stat.repetition_share for stat in stats])),
    )


def _windowed(beliefs: Beliefs, value: float) -> float | None:
    """Return `value`, or ``None`` if the window selected no position.

    The distinction :data:`REPORT_KEYS` insists on: ``0.0`` is a draw
    probability a reader can act on, and "the window landed past every game in
    this class" is not. It covers the no-``--window`` run too, which reaches
    here holding :data:`NO_BELIEFS`.
    """
    return None if beliefs.positions == 0 else value


@dataclass(frozen=True, slots=True)
class MlhCalibration:
    """The moves-left head's answers over one iteration's decisive games.

    Everything is in **plies** and every array is one entry per evaluated row,
    in ``(game_id, ply)`` order.

    Attributes:
        games: How many decisive games were replayed and evaluated. Booked
            games and replay failures are not in it, for the same reason they
            are not in the belief reads: nothing on the shard says what a
            booked game's prefix was.
        ply: Each row's own zero-based ply -- what the band stratification
            keys on.
        predicted: ``MAX_TURNS * clip(moves_left, 0, 1)`` (see
            :func:`forward_moves_left`).
        actual: ``total_plies - ply``, the target the recorder wrote, in the
            same unit. Exact on a decisive game: the last row's target is 1.
        deltas: ``m^(t) - m^(t+1) - 1`` over **consecutive** rows of one game,
            never across a game boundary. A perfectly self-consistent head
            answers zero; the median of the magnitudes is the noise floor of
            the utility's ``f(m)`` (KOM-159 F3).
    """

    games: int
    ply: npt.NDArray[np.int64]
    predicted: npt.NDArray[np.float64]
    actual: npt.NDArray[np.float64]
    deltas: npt.NDArray[np.float64]


NO_MLH: Final[MlhCalibration] = MlhCalibration(
    games=0,
    ply=np.zeros(0, dtype=np.int64),
    predicted=np.zeros(0, dtype=np.float64),
    actual=np.zeros(0, dtype=np.float64),
    deltas=np.zeros(0, dtype=np.float64),
)
"""What an iteration with no replayable decisive game reads as: no rows, and
therefore ``null`` for every derived key rather than a zero-ply error."""


def _game_moves_left(
    evaluator: Evaluator,
    game: GameRecord,
    *,
    game_factory: Callable[[], Game],
    batch: int,
) -> tuple[list[int], list[float]]:
    """Return `game`'s row plies and the head's answer at each, in ply order.

    Streamed rather than collected: one game's observations are 37 KB a row, so
    a 300-ply game held whole is 11 MB and an iteration held whole is gigabytes.
    The buffer is flushed through :func:`forward_moves_left` every `batch` rows
    and cleared, which bounds the high-water mark at one forward batch whatever
    the iteration's size -- the same trade :func:`read_observations` makes for
    the stored column.
    """
    plies: list[int] = []
    predicted: list[float] = []
    pending: list[npt.NDArray[np.float32]] = []
    width = max(batch, 1)

    def flush() -> None:
        """Forward-pass whatever is buffered and drop the observations."""
        if not pending:
            return
        predicted.extend(
            float(value)
            for value in forward_moves_left(evaluator, pending, batch=width)
        )
        pending.clear()

    def observe(replayed: Game, row: Row) -> None:
        """Buffer the position this row was recorded at."""
        plies.append(row.ply)
        pending.append(np.asarray(replayed.observation(), dtype=np.float32))
        if len(pending) >= width:
            flush()

    replay_game(game.rows, None, game_factory=game_factory, observer=observe)
    flush()
    return plies, predicted


def mlh_calibration(
    evaluator: Evaluator,
    games: Sequence[GameRecord],
    *,
    game_factory: Callable[[], Game] = new_game,
    cap_plies: int = TURN_CAP_PLIES,
    batch: int = FORWARD_BATCH,
) -> MlhCalibration:
    """Forward-pass every row of every decisive game (KOM-159 design review F3).

    **Its own sample, and its own replay.** The belief reads above are taken
    over the last :data:`VALUE_PLIES` plies of a decisive sample *matched to the
    cap count*; this is taken over every row of every decisive game
    (:func:`all_decisive_games`). The two cannot be shared: A2's threshold is
    40 plies of mean absolute error, and at campaign 5's iteration 14 the
    matched sample was two games.

    Cap games are excluded from the population and not merely down-weighted: a
    cap row's ``mlh_target`` is the plies until the *engine's* draw, which is a
    right-censored observation of "how far is the result" and is the very
    quantity the trainer censors (design D4). Scoring the head against it here
    would report a calibration on a label the head was told not to learn.

    A booked game is skipped (its prefix is not on the shard) and a game that
    refuses to replay is skipped with a warning, exactly as :func:`_replay_all`
    treats them -- the ``mlh_n_dec_games`` count says how many were actually
    read.
    """
    plies: list[int] = []
    predicted: list[float] = []
    actual: list[float] = []
    deltas: list[float] = []
    read = 0

    for game in all_decisive_games(games, cap_plies=cap_plies):
        if game.booked:
            continue
        try:
            game_plies, game_predicted = _game_moves_left(
                evaluator, game, game_factory=game_factory, batch=batch
            )
        except ReplayError as error:
            warnings.warn(
                f"game {game.game_id} did not replay for the moves-left "
                f"calibration: {error}",
                ShardWarning,
                stacklevel=2,
            )
            continue
        read += 1
        plies.extend(game_plies)
        predicted.extend(game_predicted)
        actual.extend(float(game.total_plies - ply) for ply in game_plies)
        deltas.extend(
            game_predicted[index] - game_predicted[index + 1] - 1.0
            for index in range(len(game_plies) - 1)
            # Consecutive on the SHARD, not merely adjacent in the list: a
            # gap would make the "-1" wrong by the size of the gap. Self-play
            # writes every ply, so this holds today and says so.
            if game_plies[index + 1] == game_plies[index] + 1
        )

    if not plies:
        return NO_MLH
    return MlhCalibration(
        games=read,
        ply=np.asarray(plies, dtype=np.int64),
        predicted=np.asarray(predicted, dtype=np.float64),
        actual=np.asarray(actual, dtype=np.float64),
        deltas=np.asarray(deltas, dtype=np.float64),
    )


def _mean_or_none(values: npt.NDArray[np.float64]) -> float | None:
    """Return the mean of `values`, or ``None`` for an empty selection."""
    return None if values.size == 0 else float(np.mean(values))


def mlh_report(
    calibration: MlhCalibration, *, window: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Return the ``mlh_*`` keys, all in plies, from one calibration pass.

    ``mlh_mae_dec`` is ``mean|predicted - actual|`` over every row;
    ``mlh_bias_dec`` is the **signed** ``mean(predicted - actual)``, so a head
    that under-predicts long games (design R7's censoring risk) reads negative
    rather than merely large. Both are repeated per :data:`MLH_PLY_BANDS` band,
    because pooling them hides exactly the shape the utility depends on.

    ``mlh_mae_dec_win`` is the same over the ``--window`` band and is the go/no-go
    for ``lambda``; a run given no window writes ``null`` for it rather than
    silently falling back to the pooled number.

    A band, a window or a whole pass with no row reads ``null``: this script
    reports what it measured, and it did not measure a zero-ply error.
    """
    error = calibration.predicted - calibration.actual
    report: dict[str, Any] = {
        "mlh_n_dec_games": calibration.games,
        "mlh_n_dec_rows": int(calibration.ply.size),
        "mlh_mae_dec": _mean_or_none(np.abs(error)),
        "mlh_bias_dec": _mean_or_none(error),
    }
    for name, low, high in MLH_PLY_BANDS:
        band = (calibration.ply >= low) & (calibration.ply < high)
        report[f"mlh_mae_dec_{name}"] = _mean_or_none(np.abs(error[band]))
        report[f"mlh_bias_dec_{name}"] = _mean_or_none(error[band])
    if window is None:
        report["mlh_mae_dec_win"] = None
    else:
        # Inclusive at both ends, as --window is everywhere else in this script.
        inside = (calibration.ply >= window[0]) & (calibration.ply <= window[1])
        report["mlh_mae_dec_win"] = _mean_or_none(np.abs(error[inside]))
    report["mlh_oneply_n"] = int(calibration.deltas.size)
    report["mlh_oneply_median_abs_delta"] = (
        None
        if calibration.deltas.size == 0
        else float(np.median(np.abs(calibration.deltas)))
    )
    return report


def run_census(
    games: Sequence[GameRecord],
    *,
    iteration: int,
    evaluator: Evaluator,
    stored: Mapping[tuple[int, int], npt.NDArray[Any]],
    mode: str,
    cap_plies: int = TURN_CAP_PLIES,
    cap_reason: str | None = TERMINATION_TURN_CAP,
    game_factory: Callable[[], Game] = new_game,
    window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return the report for one iteration. Keys are exactly :data:`REPORT_KEYS`.

    Pure: every I/O decision (which shards, which checkpoint, which device) is
    the caller's, so the whole census runs in a unit test against synthetic
    games and a stub evaluator.

    ``crosscheck_ok`` is the gate on everything taken from the replay. It is
    ``True`` only if at least one row was actually compared and none disagreed:
    a pass where no stored observation was available cross-checked *nothing*,
    and reporting that as ``True`` would be a green light nobody earned.

    `cap_reason` is the termination the capped games must replay to, and it
    travels with `cap_plies`: at the engine's own cap the answer is
    ``turn_cap`` and a game that replays to anything else is a failure, but a
    synthetic tree censused at a *lowered* cap was never adjudicated by the
    engine's cap at all, so such a caller passes ``None`` rather than being told
    every one of its games failed.

    `window` is KOM-162's inclusive ``(first, last)`` ply range, or ``None`` for
    the run this script has always done. It **adds** the ``*_win_*`` keys and
    changes none of the others: the tail read is still the last
    :data:`VALUE_PLIES` plies of the same games, because C3 and C3' are two
    pre-registered claims and neither replaces the other. The decisive side of
    the window is :func:`window_decisive_sample` -- the games still being played
    at ``last`` -- which is a subset of the tail run's own decisive sample, so
    the window costs a forward pass and no extra replay.
    """
    capped = cap_ids(games, cap_plies=cap_plies)
    cap_records = tuple(game for game in games if game.game_id in capped)
    dec_records = decisive_sample(games, size=len(cap_records), cap_plies=cap_plies)

    cap_sampled = _replay_all(
        cap_records,
        stored,
        game_factory=game_factory,
        expect=cap_reason,
        ply_window=window,
    )
    dec_sampled = _replay_all(
        dec_records, stored, game_factory=game_factory, ply_window=window
    )

    cap = _beliefs(evaluator, _value_positions(cap_sampled))
    dec = _beliefs(evaluator, _value_positions(dec_sampled))
    cap_d, cap_wl = cap.d, cap.wl
    dec_d, dec_wl = dec.d, dec.wl
    cap_progress, cap_repetition = _pooled_progress(cap_sampled)
    dec_progress, _dec_repetition = _pooled_progress(dec_sampled)

    if window is None:
        cap_window = dec_window = NO_BELIEFS
    else:
        in_window = {
            game.game_id
            for game in window_decisive_sample(
                games, size=len(cap_records), last=window[1], cap_plies=cap_plies
            )
        }
        cap_window = _beliefs(evaluator, _window_positions(cap_sampled))
        dec_window = _beliefs(
            evaluator, _window_positions(dec_sampled, keep=in_window)
        )

    checked = cap_sampled.crosschecked + dec_sampled.crosschecked
    mismatched = cap_sampled.mismatches + dec_sampled.mismatches

    # KOM-159's block, on its own population and its own replay: see
    # `mlh_calibration` for why it cannot reuse `dec_sampled`. It reads only
    # the moves-left head, so it changes no key above it.
    mlh = mlh_report(
        mlh_calibration(
            evaluator, games, game_factory=game_factory, cap_plies=cap_plies
        ),
        window=window,
    )

    return {
        "iteration": iteration,
        "games": len(games),
        "cap_games": len(cap_records),
        "decisive_sample": len(dec_records),
        "book_excluded": cap_sampled.excluded + dec_sampled.excluded,
        "replay_failures": cap_sampled.failures + dec_sampled.failures,
        "crosscheck_ok": checked > 0 and mismatched == 0,
        "cap_d_median": cap_d[0],
        "cap_d_p10": cap_d[1],
        "cap_d_p90": cap_d[2],
        "cap_wl_median": cap_wl[0],
        "cap_wl_p10": cap_wl[1],
        "cap_wl_p90": cap_wl[2],
        "dec_d_median": dec_d[0],
        "dec_d_p10": dec_d[1],
        "dec_d_p90": dec_d[2],
        "dec_wl_median": dec_wl[0],
        "dec_wl_p10": dec_wl[1],
        "dec_wl_p90": dec_wl[2],
        "cap_root_value_abs_mean": root_value_abs_mean(cap_records),
        "dec_root_value_abs_mean": root_value_abs_mean(dec_records),
        "cap_progress_median": cap_progress,
        "dec_progress_median": dec_progress,
        "cap_repetition_share": cap_repetition,
        "cap_row_share": row_share(games, capped),
        "cap_seat_split": seat_split(cap_records),
        "mode": mode,
        "window": None if window is None else f"{window[0]}:{window[1]}",
        "cap_d_win_median": _windowed(cap_window, cap_window.d[0]),
        "cap_d_win_p10": _windowed(cap_window, cap_window.d[1]),
        "cap_d_win_p90": _windowed(cap_window, cap_window.d[2]),
        "cap_wl_win_median": _windowed(cap_window, cap_window.wl[0]),
        "dec_d_win_median": _windowed(dec_window, dec_window.d[0]),
        "dec_d_win_p10": _windowed(dec_window, dec_window.d[1]),
        "dec_d_win_p90": _windowed(dec_window, dec_window.d[2]),
        "dec_wl_win_median": _windowed(dec_window, dec_window.wl[0]),
        "cap_wl_abs_median": cap.wl_abs_median,
        "cap_wl_spread": cap.wl_spread,
        "dec_wl_abs_median": dec.wl_abs_median,
        "dec_wl_spread": dec.wl_spread,
        "cap_ce_unweighted": cap.ce_unweighted,
        "dec_ce_unweighted": dec.ce_unweighted,
        "noncap_draws": noncap_draws(games, capped=capped, cap_plies=cap_plies),
        **mlh,
    }


# ---------------------------------------------------------------------------
# the net, the CLI
# ---------------------------------------------------------------------------


def load_evaluator(path: str | os.PathLike[str], device: str = "cpu") -> Evaluator:
    """Load `path` and return an evaluator serving exactly those weights.

    :func:`komugi.campaign.gauntlet.checkpoint_player`'s loading idiom, minus
    the arena seat: verify, build the net this campaign's ``KOMUGI_CAMPAIGN_NET``
    names, check the plane count **on the built net before**
    ``load_state_dict`` (KOM-002 §5.3, so a checkpoint from another plane map
    dies by name rather than as a raw ``size mismatch for stem.weight``), place
    it, and ``eval()`` it.

    ``eval()`` is not optional even here, where nothing is being rated: a net
    left in training mode updates its BatchNorm running statistics from whatever
    batch is pushed through it, so the second half of this census would be taken
    from a *different* net than the first.
    """
    checkpoint = load_checkpoint_file(path)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{path} names no checkpoint that verifies; refusing to report what "
            f"freshly initialised weights believe under a checkpoint's name"
        )
    net = new_net()
    require_matching_plane_count(checkpoint, net.config.trunk.in_planes)
    net.load_state_dict(checkpoint.model_state)
    net.to(device)
    net.eval()
    return InProcessEvaluator(
        net, InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)
    )


# ---------------------------------------------------------------------------
# which net produced these rows (KOM-181)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """The net a census forwarded, as the report names it.

    Attributes:
        role: The ``--ckpt-role`` asked for.
        journal_role: The journal role that maps to (:data:`CKPT_ROLES`).
        path: The checkpoint, resolved to an absolute path.
        step: Its training step, off the checkpoint itself rather than off its
            filename -- the filename's number and the payload's are written
            together by :func:`~komugi.train.checkpoint.save_checkpoint`, and
            the payload is the one the loader verified.
        sha256: The parameter digest **computed here**, by the campaign's own
            hash. Equal to the journal's record for `journal_role` unless the
            run was forced through with ``--later-head``.
    """

    role: str
    journal_role: str
    path: Path
    step: int
    sha256: str

    def report(self) -> dict[str, Any]:
        """Return the :data:`IDENTITY_KEYS` block for the JSON report."""
        return {
            "census_ckpt_role": self.role,
            "census_ckpt_path": str(self.path),
            "census_ckpt_step": self.step,
            "census_ckpt_sha256": self.sha256,
            "census_ckpt_journal_role": self.journal_role,
        }


def journal_fingerprint(
    journal: str | os.PathLike[str], *, iteration: int, journal_role: str
) -> NetFingerprint:
    """Return the journal's ``net_fingerprint`` for one ``(iteration, role)``.

    Superseded records are dropped first with
    :func:`~komugi.campaign.boundary.latest_fingerprints` -- the campaign's own
    rule, not a second one invented here: a crash-before-checkpoint resume
    re-runs an iteration and the append-only journal keeps both attempts, of
    which the last describes what is on disk.

    Raises:
        MissingFingerprint: If there is no journal, or no record for this pair.
            Iteration 0 under ``self_play`` is named as the specific case it is:
            the first iteration serves freshly initialised weights
            (``CHECKPOINT_ENV`` empty), so ``record_fingerprints`` writes no
            ``self_play`` record for it and no file can be its producer.
    """
    path = Path(journal)
    if not path.is_file():
        raise MissingFingerprint(
            f"{path} does not exist, so the {journal_role!r} fingerprint of "
            f"iteration {iteration} cannot be checked; a census that cannot say "
            f"WHICH net it forwarded is the defect KOM-181 exists to close, so "
            f"this is a refusal and not a skip"
        )
    for record in latest_fingerprints(read_records(path, NetFingerprint).records):
        if record.iteration == iteration and record.role == journal_role:
            return record
    if journal_role == SELF_PLAY and iteration == 0:
        raise MissingFingerprint(
            f"iteration 0 has no {SELF_PLAY!r} fingerprint in {path}: it played "
            f"on freshly initialised weights, so no checkpoint produced its "
            f"rows and no --ckpt can be their {SERVED_PRODUCER}. Census a later "
            f"iteration, or state --ckpt-role {POST_TRAINING} for the EMA "
            f"iteration 0's training published"
        )
    raise MissingFingerprint(
        f"{path} carries no {journal_role!r} net_fingerprint for iteration "
        f"{iteration}; the census cannot check what it is about to forward "
        f"against anything, and reporting an unchecked digest would be the "
        f"KOM-159 mistake with a hash beside it"
    )


def checkpoint_identity(
    ckpt: str | os.PathLike[str],
    *,
    iteration: int,
    role: str,
    journal: str | os.PathLike[str],
) -> CheckpointIdentity:
    """Return what `ckpt` is, refusing it if it is not `role` at `iteration`.

    The digest is the campaign's, called and not copied:
    :func:`komugi.campaign.boundary._fingerprint` over
    :func:`komugi.train.digest.parameter_keys` of this process's
    :func:`~komugi.campaign.factory.new_net`, which is the same key set in the
    same order the loop hashed the file with when it wrote the journal record.
    Re-implementing the hash here would give a guard that agrees with the
    journal until the day one of them changes.

    The checkpoint is loaded a second time (:func:`load_evaluator` loads it to
    serve it): the alternative is reaching into an evaluator for its module,
    which would make the census's one test seam a hole in the guard -- a stub
    evaluator would fingerprint nothing. The cost is one file read.

    Raises:
        MissingFingerprint: Via :func:`journal_fingerprint`.
        WrongProducer: If the digests differ, naming the role and both digests.
        FileNotFoundError: If `ckpt` names nothing that verifies -- the same
            refusal :func:`load_evaluator` makes, made earlier.
    """
    journal_role = CKPT_ROLES[role]
    expected = journal_fingerprint(
        journal, iteration=iteration, journal_role=journal_role
    )
    path = Path(ckpt).resolve()
    checkpoint = load_checkpoint_file(path)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{path} names no checkpoint that verifies; refusing to report what "
            f"freshly initialised weights believe under a checkpoint's name"
        )
    computed = _fingerprint(
        iteration, journal_role, checkpoint.model_state, parameter_keys(new_net())
    )
    identity = CheckpointIdentity(
        role=role,
        journal_role=journal_role,
        path=path,
        step=int(checkpoint.step),
        sha256=computed.sha256,
    )
    if computed.sha256 != expected.sha256:
        raise WrongProducer(
            f"--ckpt {path} is not the {role} of iteration {iteration}: the "
            f"journal's {journal_role!r} record is sha256 {expected.sha256} "
            f"({expected.param_count} parameters) and this file's parameters "
            f"hash to {computed.sha256} ({computed.param_count} parameters) at "
            f"step {checkpoint.step}. Iteration {iteration}'s rows were played "
            f"by the EMA served BEFORE its training ran, so the EMA published "
            f"after it is a different net by one iteration (review EP-01). "
            f"Pass the {journal_role!r} checkpoint, or --later-head to census "
            f"this one anyway with every report key prefixed "
            f"{LATER_HEAD_PREFIX!r}",
            identity=identity,
            expected=expected.sha256,
        )
    return identity


def later_head_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return `report` with :data:`LATER_HEAD_PREFIX` on **every** key.

    Applied once, to the finished dict, and recursively into the nested
    mappings (``cap_seat_split``): the property that has to hold is that *no*
    key of an off-role report collides with a key of an on-role one, and a
    prefix applied per-key at each construction site is a prefix somebody
    forgets on the key they add next.

    Values are untouched -- they are what the later head believes, which is a
    real measurement of a real net; what the prefix says is that it is not the
    net that played the rows.
    """
    return {
        f"{LATER_HEAD_PREFIX}{key}": (
            later_head_report(value) if isinstance(value, Mapping) else value
        )
        for key, value in report.items()
    }


def _guard_mode(
    games: Sequence[GameRecord],
    stored: Mapping[tuple[int, int], npt.NDArray[Any]],
) -> str:
    """Return the mode, checked against the first unbooked game that has a row.

    Checked against a game with a row at **ply 0** only: a booked game's first
    recorded row is past the draft's start and its drafting planes say nothing
    about the start mode. If no such observation was read -- a shard tree with
    no ``observation`` column -- the engine's mode is reported unchecked, and
    the report's ``mode`` is then a statement about this process rather than
    about the shard.
    """
    for game in games:
        if game.booked or not game.rows:
            continue
        obs = stored.get((game.game_id, game.rows[0].ply))
        if obs is not None:
            return mode_guard(obs)
    warnings.warn(
        "no ply-0 observation was available, so the shard's start mode could "
        "not be checked against this process's GUNGI_MODE; the reported mode is "
        f"this process's ({engine_mode()!r})",
        ShardWarning,
        stacklevel=2,
    )
    return engine_mode()


def parse_window(text: str) -> tuple[int, int]:
    """Return ``FIRST:LAST`` as an inclusive ply range, refusing anything else.

    An :class:`argparse.ArgumentTypeError` rather than a bare
    :class:`ValueError` so a typo prints the usage line and the reason, instead
    of a traceback several hundred shard reads later.
    """
    first, separator, last = text.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"--window takes FIRST:LAST, two ply numbers separated by a colon; "
            f"got {text!r}"
        )
    try:
        bounds = (int(first), int(last))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--window takes FIRST:LAST as whole plies; got {text!r}"
        ) from None
    if bounds[0] < 0:
        raise argparse.ArgumentTypeError(
            f"--window starts at ply {bounds[0]}; a game's first ply is 0"
        )
    if bounds[1] < bounds[0]:
        raise argparse.ArgumentTypeError(
            f"--window {text!r} ends before it begins; the range is inclusive, "
            f"so LAST must be at least FIRST"
        )
    return bounds


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="cap_game_values.py",
        description=(
            "What the value head believes on one iteration's turn-cap games: "
            "replay them, forward-pass the iteration's checkpoint over their "
            "last positions, and report raw d and w-l against a matched "
            "decisive sample (KOM-151 M1'). With --window, the same two classes "
            "are read at a fixed mid-game ply range as well (KOM-162, review "
            "F1); magnitude, unweighted-CE and non-cap-draw keys are reported "
            "either way. The mlh_* keys (KOM-159) are the moves-left head's "
            "calibration in plies, over EVERY decisive game of the iteration."
        ),
    )
    parser.add_argument("root", type=Path, help="the campaign root to read")
    parser.add_argument("iteration", type=int, help="which iteration to census")
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help=(
            "the checkpoint to forward-pass; this iteration's own, so the "
            f"report is what the net BELIEVED while it played. Needs {NET_ENV} "
            "to name the architecture the campaign trained. Its parameter "
            "digest is checked against the journal's record for --ckpt-role "
            "and a mismatch is refused (KOM-181)"
        ),
    )
    parser.add_argument(
        "--ckpt-role",
        choices=sorted(CKPT_ROLES),
        default=SERVED_PRODUCER,
        help=(
            f"which net --ckpt is claimed to be. {SERVED_PRODUCER} (the "
            "default) is the EMA that PLAYED this iteration's rows, served "
            "before self-play at step N*300 and recorded in the journal as "
            f"{SELF_PLAY!r}; {POST_TRAINING} is the EMA published after this "
            f"iteration's training, at (N+1)*300, recorded as {EMA!r}. The two "
            "differ by one training iteration, which is the confound review "
            "EP-01 found in the KOM-159 censuses"
        ),
    )
    parser.add_argument(
        "--later-head",
        action="store_true",
        help=(
            "census a --ckpt whose digest does NOT match --ckpt-role anyway, "
            f"prefixing every key of the report {LATER_HEAD_PREFIX!r} so it "
            "cannot be pooled with an on-role census. For deliberately asking "
            "what a later head thinks of earlier rows; without it a mismatch "
            f"exits {EXIT_WRONG_NET}"
        ),
    )
    parser.add_argument(
        "--device", default="cpu", help="where the net is placed (cpu, cuda)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "torch intra-op threads; kept low so a census run beside a live "
            "campaign does not oversubscribe the box (KOM-146)"
        ),
    )
    parser.add_argument(
        "--window",
        type=parse_window,
        default=None,
        metavar="FIRST:LAST",
        help=(
            "also read both classes over this INCLUSIVE ply range (e.g. "
            "96:160), filling the *_win_* keys: cap-game positions at those "
            "plies against positions of decisive games that were still being "
            "played at LAST. Also fills mlh_mae_dec_win, the KOM-159 go/no-go. "
            f"Omitted, the census reads only the last {VALUE_PLIES} plies as "
            "before, and every *_win_* key and mlh_mae_dec_win is null"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the JSON report; stdout when omitted",
    )
    return parser


def _number(value: Any) -> str:
    """Return `value` to three places, or ``n/a`` for the nullable keys' null.

    The summary is read by a human deciding whether a pre-registered claim
    fired, and ``0.000`` where the census has nothing to say is how a claim gets
    scored against a number nobody measured.
    """
    return "n/a" if value is None else f"{float(value):.3f}"


def format_report(report: Mapping[str, Any]) -> str:
    """Return the human-readable summary, with the pre-registration on it.

    The verdict line is printed beside the number it is read off, because
    "median d = 0.71" means nothing to a reader who does not have review item 2
    open, and a census whose finding has to be looked up is a census that gets
    mis-quoted. The ``net:`` line names the :data:`IDENTITY_KEYS` for the same
    reason: the digest that makes the run checkable is worth nothing if it is
    only ever in a file nobody opens.

    Called on the report **before** :func:`later_head_report` prefixes it, so
    one formatter serves both runs; :func:`main` prints the off-role warning
    beside it.
    """
    cap_d = float(report["cap_d_median"])
    cap_wl = float(report["cap_wl_median"])
    lines: list[str] = []
    if cap_d > 0.6:
        verdict = "the net BELIEVES draw (median d > 0.6) -> labels/row share"
    elif cap_d < 0.3 and abs(cap_wl) > 0.3:
        verdict = "conversion failure (d < 0.3, |w-l| > 0.3) -> search budget"
    else:
        verdict = "neither pre-registered branch fired"

    if report["window"] is not None:
        lines.append(
            f"  window {report['window']}   "
            f"cap d {_number(report['cap_d_win_median'])} "
            f"[{_number(report['cap_d_win_p10'])}, "
            f"{_number(report['cap_d_win_p90'])}] "
            f"w-l {_number(report['cap_wl_win_median'])}   "
            f"dec d {_number(report['dec_d_win_median'])} "
            f"[{_number(report['dec_d_win_p10'])}, "
            f"{_number(report['dec_d_win_p90'])}] "
            f"w-l {_number(report['dec_wl_win_median'])}"
        )
    lines.append(
        f"  |w-l| median cap {float(report['cap_wl_abs_median']):.3f} "
        f"(spread {float(report['cap_wl_spread']):.3f}) vs dec "
        f"{float(report['dec_wl_abs_median']):.3f} "
        f"(spread {float(report['dec_wl_spread']):.3f})"
    )
    lines.append(
        f"  unweighted WDL CE cap {_number(report['cap_ce_unweighted'])} "
        f"vs dec {_number(report['dec_ce_unweighted'])}; "
        f"non-cap draws {report['noncap_draws']}"
    )
    lines.append(
        f"  MLH ({report['mlh_n_dec_games']} decisive games, "
        f"{report['mlh_n_dec_rows']} rows) MAE {_number(report['mlh_mae_dec'])} "
        f"plies [b0 {_number(report['mlh_mae_dec_b0'])}, "
        f"b1 {_number(report['mlh_mae_dec_b1'])}, "
        f"b2 {_number(report['mlh_mae_dec_b2'])}, "
        f"b3 {_number(report['mlh_mae_dec_b3'])}]; "
        f"bias {_number(report['mlh_bias_dec'])}; "
        f"window MAE {_number(report['mlh_mae_dec_win'])}"
    )
    lines.append(
        f"  net: {report['census_ckpt_role']} "
        f"(journal {report['census_ckpt_journal_role']}@{report['iteration']}) "
        f"step {report['census_ckpt_step']} "
        f"sha256 {report['census_ckpt_sha256']} "
        f"{report['census_ckpt_path']}"
    )
    lines.append(
        f"  MLH one-ply |delta| median "
        f"{_number(report['mlh_oneply_median_abs_delta'])} plies over "
        f"{report['mlh_oneply_n']} pairs "
        f"(pre-registered < {MLH_ONEPLY_THRESHOLD:.0f})"
    )
    return "\n".join(
        [
            f"iteration {report['iteration']} ({report['mode']}): "
            f"{report['games']} games, {report['cap_games']} capped "
            f"({float(report['cap_row_share']):.1%} of rows), "
            f"{report['decisive_sample']} decisive sampled",
            f"  excluded (booked) {report['book_excluded']}, "
            f"replay failures {report['replay_failures']}, "
            f"crosscheck_ok {report['crosscheck_ok']}",
            f"  cap   d {cap_d:.3f} "
            f"[{float(report['cap_d_p10']):.3f}, {float(report['cap_d_p90']):.3f}]"
            f"   w-l {cap_wl:+.3f} "
            f"[{float(report['cap_wl_p10']):+.3f}, "
            f"{float(report['cap_wl_p90']):+.3f}]",
            f"  dec   d {float(report['dec_d_median']):.3f} "
            f"[{float(report['dec_d_p10']):.3f}, {float(report['dec_d_p90']):.3f}]"
            f"   w-l {float(report['dec_wl_median']):+.3f} "
            f"[{float(report['dec_wl_p10']):+.3f}, "
            f"{float(report['dec_wl_p90']):+.3f}]",
            f"  progress plies (median) cap {float(report['cap_progress_median']):.1f} "
            f"vs dec {float(report['dec_progress_median']):.1f}; "
            f"repeated-position share {float(report['cap_repetition_share']):.3f}",
            f"  |root_value| cap {float(report['cap_root_value_abs_mean']):.3f} "
            f"vs dec {float(report['dec_root_value_abs_mean']):.3f}; "
            f"seat split {report['cap_seat_split']}",
            *lines,
            f"  verdict: {verdict}",
        ]
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_factory: Callable[[Path, str], Evaluator] = load_evaluator,
) -> int:
    """Census one iteration, write the report, and return an exit status.

    `evaluator_factory` is the test seam, and it is the only one: everything
    else on this path is the real reader over a real shard tree, so a test that
    substitutes a stub net still exercises the shard read, the replay, the
    guards and the report. The checkpoint identity is deliberately outside it
    (:func:`checkpoint_identity` opens the file itself): a guard a stub can
    satisfy is not a guard.

    The identity is resolved **first**, before a single shard is read. A census
    of a real iteration is minutes of replay and forward passes, and a refusal
    that arrives at the end of it is a refusal people learn to pass
    ``--later-head`` to silence.

    Returns:
        ``0``, or :data:`EXIT_WRONG_NET` when ``--ckpt`` is not the net
        ``--ckpt-role`` asked for and ``--later-head`` was not given.
    """
    args = build_parser().parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory; there is nothing to census")
    torch.set_num_threads(max(int(args.threads), 1))

    off_role = False
    try:
        identity = checkpoint_identity(
            args.ckpt,
            iteration=args.iteration,
            role=args.ckpt_role,
            journal=journal_path(root),
        )
    except WrongProducer as mismatch:
        if not args.later_head:
            print(mismatch, file=sys.stderr)
            return EXIT_WRONG_NET
        off_role = True
        identity = mismatch.identity
        print(
            f"--later-head: proceeding against a net that is NOT the "
            f"{args.ckpt_role} of iteration {args.iteration}; every report key "
            f"is prefixed {LATER_HEAD_PREFIX!r}.\n{mismatch}",
            file=sys.stderr,
        )
    except MissingFingerprint as missing:
        print(missing, file=sys.stderr)
        return EXIT_WRONG_NET

    shards = shard_root(root, args.iteration)
    games = read_games(shards)
    if not games:
        raise ValueError(
            f"{shards} holds no readable game rows; iteration "
            f"{args.iteration} sealed nothing this census can read"
        )

    capped = cap_ids(games)
    selected = [game for game in games if game.game_id in capped]
    selected.extend(decisive_sample(games, size=len(selected)))
    stored = read_observations(shards, crosscheck_rows(selected) | guard_row(games))
    mode = _guard_mode(games, stored)

    report = run_census(
        games,
        iteration=args.iteration,
        evaluator=evaluator_factory(args.ckpt, args.device),
        stored=stored,
        mode=mode,
        window=args.window,
    )
    report.update(identity.report())
    expected_keys = set(REPORT_KEYS) | set(IDENTITY_KEYS)
    if set(report) != expected_keys:
        raise ValueError(
            f"the report's keys are {sorted(report)}, not "
            f"{sorted(expected_keys)}; REPORT_KEYS + IDENTITY_KEYS is the JSON "
            f"contract and it is wrong or the report is"
        )

    print(format_report(report))
    if off_role:
        report = later_head_report(report)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(serialized)
    else:
        out: Path = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
