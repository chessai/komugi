"""Draft diversification books: self-generated opening prefixes (KOM-014).

``docs/design/kom-014-draft-books.md`` is the authority; this docstring names
the four decisions a reader needs before the code makes sense.

**What a book is (B1).** One JSON file per campaign generation at
``<root>/books/gen_<i>.json``, holding up to :data:`CAP` *action prefixes* of
:data:`MIN_PLIES`--:data:`MAX_PLIES` plies each, mined from the newest
qualifying iterations **below** ``i``. Never mutated; byte-deterministic for a
given ``(run_seed, iteration, shard tree)``, which is what lets a resumed
campaign rebuild it rather than trust the file it finds.

**Where the actions come from (B2).** :data:`komugi.selfplay.records.SCHEMA`
carries ``action`` -- the move that was actually played on a row. Without it a
shard cannot be replayed at all (§1.1), and every alternative (inverting the
observation encoding, a worker-written sidecar, re-searching with the serving
net) is rejected in §3.1 with a reason.

**What "at high temperature" means (B3).** A prefix is the *recorded* actions
of plies ``0 … d−1`` of one unbooked, ply-contiguous source game, followed by
**one** action drawn from that row's ``policy_target`` **truncated and then
tempered**: ``S`` is the top :data:`SUPPORT_TOP_K` of
``{a : p[a] ≥ SUPPORT_EPSILON}``, and the draw is ``p^(1/τ)`` renormalised over
``S``. Truncating *before* tempering is the whole point -- ``policy_target`` is
nonzero on every legal action, so tempering over its full support would put
~0.30 of the drawn mass on actions the search scored below ``1e-4`` (§3.2). One
tempered ply and no more: after the first deviation there is no recorded target
for the resulting position, so a second tempered ply would be an invention.

**What it costs, and the bound (B5).** :class:`~komugi.train.shards.ShardReader`
cannot be this module's API: it decompresses and digest-verifies a whole
directory at construction and returns every field of a row. The builder instead
projects columns with :func:`~komugi.train.shards.read_columns` -- three
``int64`` columns over the whole window (24 bytes a row) in the index pass, and
``policy_target`` over at most :data:`TARGET_SHARD_BUDGET` shards, each visited
exactly once and released, in the target pass. Nothing here constructs a
``ShardReader`` or materialises a whole row, and acceptance #38 pins that by
monkeypatching both to fail.

**Where this module lives, and why it cannot move.**
:mod:`komugi.campaign.loop` already imports :mod:`komugi.selfplay.orchestrate`,
so a builder under ``campaign/`` that the worker imported would close a cycle.
The consequence is visible below: the campaign's directory layout
(``iter_<i>/shards/worker_*``) is **re-derived here** rather than imported, and
:data:`_ITERATION_DIR` / :data:`_SHARDS_DIRNAME` / :data:`_WORKER_GLOB` are the
duplicated constants. They are duplicated deliberately and are cheap to check
against ``loop.py``'s layout docstring; a cycle is not cheap to unwind.

**Who draws from a book (B4).** :func:`select_book_entry` is the whole of
§4.2: a pure function of ``(run_seed, worker_id, incarnation, game_index)`` --
the same four integers that name the game -- run on a **dedicated** stream
composed from the game's own :class:`numpy.random.SeedSequence` with
:data:`~komugi.selfplay.records.BOOK_SELECT_DOMAIN` in the spawn key. Dedicated
because the game's ``rng`` is documented as the full/fast draw's and nothing
else: a book draw taken from it would shift every subsequent full/fast draw, so
a booked run and a bookless run at one seed would no longer differ in the book
alone. Pure because an offline audit (:mod:`komugi.campaign.book_audit`) has to
recompute which entry a game used from the seed and the book file, with no
record field and no logging.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import zipfile
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from komugi.engine.game import Game
from komugi.engine.observation import MAX_TURNS, PLANE_MAP_VERSION
from komugi.selfplay.adapter import GameState
from komugi.selfplay.records import (
    BOOK_BUILD_DOMAIN,
    BOOK_SELECT_DOMAIN,
    SCHEMA,
    book_seed,
    game_seed,
    num_actions,
)
from komugi.train.shards import SHARD_GLOB, read_columns
from komugi.train.window import WindowConfig

BOOK_VERSION: Final[int] = 1
"""The manifest's format version (§2.2). Bumped when a *reader* must change."""

BOOKS_DIRNAME: Final[str] = "books"
_BOOK_NAME: Final[str] = "gen_{generation}.json"

_ITERATION_DIR: Final[str] = "iter_{iteration}"
_SHARDS_DIRNAME: Final[str] = "shards"
_WORKER_GLOB: Final[str] = "worker_*"
"""The campaign's shard layout, ``<root>/iter_<i>/shards/worker_<k>_r<n>``.

Re-derived rather than imported from :mod:`komugi.campaign.loop`: see the module
docstring on the import cycle that forbids the edge.
"""

_SHARD_RE: Final[re.Pattern[str]] = re.compile(r"shard-(\d{6,})\.npz")
"""Shard index parser, matching :class:`~komugi.train.shards.ShardReader`'s.

Shards are visited in *parsed index* order, not name order, so a directory that
has run past six digits still enumerates chronologically -- which is one of the
five facts §2.3's byte-determinism rests on.
"""

CAP: Final[int] = 1024
"""Maximum entries in a book (§2.3). ~300 KB, sized upward from the consumer:
at ~8 booked games per production iteration a 1024-entry book essentially never
repeats an entry within a generation."""

MIN_ENTRIES: Final[int] = 32
"""Below this, **no file is written** (§3.3). A 3-entry book would send every
booked game of a generation down one of three lines."""

MIN_PLIES: Final[int] = 4
MAX_PLIES: Final[int] = 12
"""The ticket's prefix length range, ``L = d + 1 ∈ [4, 12]``."""

ENTRIES_PER_SOURCE_GAME: Final[int] = 8
"""Per-``(source_iteration, source_game_id)`` acceptance quota (§3.3, N1).

The deviation is always the *last* ply, so ``k`` candidates from one game at one
``d`` differ in one action and score as ``k`` distinct entries. Without this
quota a book could sit on a few dozen real stems and still report
``distinct_entries / cap == 1.00`` -- and a sampler mining the tail would score
it *best*."""

TEMPERATURE: Final[float] = 2.0
"""τ, pre-registered (§0 B3). A **build-time** re-normalisation exponent on a
recorded distribution -- disjoint from KOM-036's search-time Gumbel scale, which
is a noise scale on a live draw (§5.1). Sharing one constant would be a category
error."""

SUPPORT_EPSILON: Final[float] = 1e-3
SUPPORT_TOP_K: Final[int] = 16
"""The normal truncation rule: ``S`` = top ``k`` of ``{a : p[a] ≥ ε}`` (§3.2)."""

SUPPORT_FLOOR: Final[float] = 1e-4
SUPPORT_MIN_K: Final[int] = 2
"""The fallback rule, for a target so diffuse that **no** action clears ``ε``
(§3.2 step 5, R3). ``SUPPORT_FLOOR`` is not a free constant: it is exactly R3's
tail threshold, which is what keeps the tail-mass identity (``tail_mass == 0``)
exact on this branch too. ``k_min = 2`` rather than 1 because a one-action
fallback would silently turn every diffuse draft position into an undeviated
entry -- anti-diversification in the shape of a fix."""

DRAW_BUDGET_MULTIPLIER: Final[int] = 4
"""A build never draws more than ``4 × cap`` candidates (§3.3), so a thin source
window terminates instead of spinning."""

WALK_BOUND_MULTIPLIER: Final[int] = 2
"""The source walk examines at most ``2 × WindowConfig().size`` iterations (§1.8).

Its own bound, which ``window_refs`` does not have: after a fingerprint bump
*every* pre-bump iteration fails the schema guard, and a naive newest-first walk
looking for five qualifying iterations would descend to iteration 0."""

_ADJUDICATION_PLY_CAP: Final[int] = MAX_TURNS
"""The engine's turn cap -- the worst plies-per-game the budget must survive.

Derived from :data:`komugi.engine.observation.MAX_TURNS`, the one place the cap
lives, so KOM-165's 256 → 512 re-derives :data:`TARGET_SHARD_BUDGET` (32 → 64)
instead of leaving a stale literal here."""

_SAMPLES_PER_SHARD: Final[int] = 1024
"""``ShardWriter``'s default ``samples_per_shard``, i.e. game starts per shard
divided by plies per game."""

TARGET_SHARD_BUDGET: Final[int] = math.ceil(
    (CAP / ENTRIES_PER_SOURCE_GAME) * _ADJUDICATION_PLY_CAP / _SAMPLES_PER_SHARD
)
"""How many shards the target pass may open, **derived** and not chosen (§2.4, R5).

``⌈(cap / entries_per_source_game) × adjudication_ply_cap / samples_per_shard⌉``
= 64 at KOM-165's 512-ply cap (it was 32 at 256). The derivation, in one line
each: the quota forces ``⌈cap / 8⌉ = 128`` distinct source games before the cap
is reachable; a game's eligible deviation plies all lie within its first 12 rows,
hence in the shard holding its start; a shard holds ``samples_per_shard``
consecutive rows, i.e. ``samples_per_shard / plies_per_game`` game starts; so
``B`` shards reach ``B × samples_per_shard / plies_per_game`` games, and at the
*worst* plies-per-game the engine permits (the engine's turn cap) that needs
``B ≥ 64``.

Written as the formula rather than as a literal so that a later change to
:data:`CAP`, :data:`ENTRIES_PER_SOURCE_GAME` or the engine's cap re-derives it
instead of silently invalidating it. The budget binds *cost*, not the book's
size: :class:`~komugi.campaign.loop.CampaignConfig`'s ``games_per_worker``
sets how many games a window actually mines, independently of this
ceiling."""

BIT_GENERATOR: Final[str] = "PCG64"
"""Pinned **by name**, not inherited from ``default_rng`` (§2.3, N10).

``default_rng``'s algorithm is documented as an implementation choice, and this
artefact is compared byte-wise across processes and machines (#9).
:mod:`komugi.train.window` pins PCG64 by name for exactly this reason."""

_REPLAY_DRAW_SCORE: Final[float] = 0.0
"""``draw_score`` for the build-time replay. Contempt cannot change legality or
termination, and a book is not a search."""

_FLOAT_PLACES: Final[int] = 6
"""Floats are rounded before serialisation so a platform's ``repr`` cannot enter
the bytes (§2.3 fact 5)."""

_MIN_FLATTENING_HEADROOM: Final[float] = 0.05
_MIN_RESTRICTED_ENTROPY: Final[float] = 0.25
"""The **two-sided** exclusion from ``median_flattening_ratio`` (§3.2, R4).

An already-uniform restriction has nothing to flatten (tempering is a fixed
point there); a near-deterministic one collapses the ratio toward 0 while τ
behaves exactly as specified. Rev 2 excluded only the first end and the
pre-registered band failed at the second."""

_UNREADABLE: Final[tuple[type[Exception], ...]] = (
    zipfile.BadZipFile,
    zlib.error,
    OSError,
    ValueError,
    KeyError,
    EOFError,
)
"""Everything a damaged shard can throw on the way back in, matching
:mod:`komugi.train.shards`' own tuple. Caught only where a shard is being
*qualified* -- an unreadable shard is not this generation's shard, which is a
decision, not a crash."""

_SHARD_MODE: Final[int] = 0o644

_GameKey = tuple[int, int]
"""``(source_iteration, game_id)`` -- a source game's identity across the window.

``game_id`` is unique per ``(worker, incarnation, game_index)`` within a run but
*not* across iterations, so the iteration is part of the key."""


def running_gungi_mode() -> str:
    """Return the start mode this process's engine is configured for (§1.11).

    ``GUNGI_MODE`` is read from the environment at :mod:`komugi.engine.game`
    import time and validated there, so the class attribute -- not a second
    ``os.environ`` read -- is the honest answer to "which start position would a
    prefix be replayed from". A prefix is a list of action ids from a bare
    :class:`~komugi.engine.game.Game`; replayed under a different mode it is a
    different line, or illegal.
    """
    return str(Game._START_MODE)


@dataclass(frozen=True, slots=True)
class BookEntry:
    """One opening prefix, plus the provenance and diagnostics §2.2 records.

    ``actions`` replays in order from a bare :class:`~komugi.engine.game.Game`;
    there is no position blob, because a position is not a portable artefact and
    an action list is (the arena makes the same choice for ``GameOutcome.moves``).

    The last action is the **tempered** one, drawn at ``deviation_ply``; every
    earlier action is a move Komugi's own search actually played. ``deviated``
    records whether the draw happened to return the played move -- a degenerate
    draw is kept, not rejected, because it is a real line of Komugi's own play
    and rejecting it would bias the book away from the highest-probability
    openings.

    Args:
        id: ``gen{G:04d}-{index:06d}-{sha8}`` -- non-blank and stable, so it is
            already a legal ``ArenaConfig.book_id`` if §5's pit books ever
            consume it. This ticket does **not** wire it to the arena.
        actions: The prefix, ``plies`` action ids.
        plies: ``len(actions)``, in ``[MIN_PLIES, MAX_PLIES]``.
        source_iteration: The campaign iteration the stem was mined from.
        source_game_id: The ``game_id`` of the source game.
        deviation_ply: ``d`` -- the ply the tempered action was drawn at.
        deviated: Whether the drawn action differs from the recorded one.
        support_size: ``|S|`` after truncation.
        support_fallback: Whether ``S`` came from the ``SUPPORT_FLOOR`` branch.
        truncated_mass: ``1 − Σ_{a∈S} policy_target[a]`` -- a systematically
            large value means the search is diffuse and ``k`` is too small.
        drawn_target_prob: The drawn action's *recorded* ``policy_target``, i.e.
            what R3's tail mass is computed from.
        restricted_entropy_bits: ``H_S``, the untempered restriction's entropy.
        tempered_entropy_bits: ``H_t``.
        flattening_ratio: ``(H_t − H_S) / (log2|S| − H_S)``, or **``None``** when
            that denominator is zero (R6). ``None`` rather than ``0.0`` because
            ``0.0`` is the value the τ = 1 identity claims: scoring a degenerate
            entry ``0.0`` would make acceptance #7a pass for the wrong reason on
            a book built at τ = 2.
    """

    id: str
    actions: tuple[int, ...]
    plies: int
    source_iteration: int
    source_game_id: int
    deviation_ply: int
    deviated: bool
    support_size: int
    support_fallback: bool
    truncated_mass: float
    drawn_target_prob: float
    restricted_entropy_bits: float
    tempered_entropy_bits: float
    flattening_ratio: float | None

    @property
    def stem(self) -> tuple[int, ...]:
        """Return the prefix **without** its deviation (§3.3).

        The gate moves to stems because exact-tuple dedup is a metric the
        failure mode maximises: mining the tail changes only the last action, so
        it leaves the stem count exactly where it was.
        """
        return self.actions[:-1]

    def as_json(self) -> dict[str, Any]:
        """Return this entry as the manifest's object (§2.2)."""
        return {
            "id": self.id,
            "actions": list(self.actions),
            "plies": self.plies,
            "source_iteration": self.source_iteration,
            "source_game_id": self.source_game_id,
            "deviation_ply": self.deviation_ply,
            "deviated": self.deviated,
            "support_size": self.support_size,
            "support_fallback": self.support_fallback,
            "truncated_mass": self.truncated_mass,
            "drawn_target_prob": self.drawn_target_prob,
            "restricted_entropy_bits": self.restricted_entropy_bits,
            "tempered_entropy_bits": self.tempered_entropy_bits,
            "flattening_ratio": self.flattening_ratio,
        }


@dataclass(frozen=True, slots=True)
class Book:
    """A whole book file, parsed (§2.2).

    Every manifest field is carried, including
    ``source_schema_fingerprint`` (traceability *after* the schema guard, not
    instead of it) and ``plane_map_version`` (a prefix does not depend on the
    plane map -- this field is provenance of the *rows* it was mined from).
    :func:`load_book` refuses a book whose ``plane_map_version`` differs from
    the running encoder's: the shard fingerprint cannot see a plane-map
    change, so this is the only place a stale book is caught by name. The
    field that protects a prefix directly is ``action_size``: an action id
    means nothing across an encoding change, and :func:`load_book` refuses on
    it too.
    """

    book_version: int
    generation: int
    source_iterations: tuple[int, ...]
    gungi_mode: str
    action_size: int
    plane_map_version: int
    source_schema_fingerprint: str
    temperature: float
    support_epsilon: float
    support_top_k: int
    support_floor: float
    support_min_k: int
    target_shard_budget: int
    cap: int
    min_plies: int
    max_plies: int
    entries_per_source_game: int
    build_seed: int
    bit_generator: str
    entries: tuple[BookEntry, ...]
    stats: Mapping[str, Any]

    def __len__(self) -> int:
        """Return how many entries the book holds."""
        return len(self.entries)


def books_dir(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/books``, a sibling of ``ckpt``, ``ema`` and ``iter_<i>``."""
    return Path(root) / BOOKS_DIRNAME


def book_path_for(root: str | os.PathLike[str], iteration: int) -> Path:
    """Return ``<root>/books/gen_<iteration>.json``, whether or not it exists.

    ``gen_<i>`` names the iteration the book is **for** -- the one whose workers
    read it -- not the iteration it was mined from.
    """
    return books_dir(root) / _BOOK_NAME.format(generation=iteration)


def _iteration_shards(root: Path, iteration: int) -> list[Path]:
    """Return every shard file of one iteration, in deterministic order.

    Worker directories sorted by name (as ``iteration_source`` does) and shards
    within one sorted by **parsed index** (as ``ShardReader`` does), so "the
    candidate pool, in order" is a pure function of the shard tree -- §2.3's
    first byte-determinism fact.
    """
    directory = root / _ITERATION_DIR.format(iteration=iteration) / _SHARDS_DIRNAME
    if not directory.is_dir():
        return []
    paths: list[Path] = []
    for worker in sorted(directory.glob(_WORKER_GLOB)):
        if not worker.is_dir():
            continue
        indexed: list[tuple[int, Path]] = []
        for shard in worker.glob(SHARD_GLOB):
            match = _SHARD_RE.fullmatch(shard.name)
            if match is None:
                continue
            indexed.append((int(match.group(1)), shard))
        paths.extend(shard for _, shard in sorted(indexed))
    return paths


def _shard_fingerprint(path: Path) -> str | None:
    """Return a shard's embedded schema fingerprint, or ``None`` if unreadable.

    Reads the reserved keys **alone** (``read_columns(path, ())``): the guard is
    a question about the generation, and a whole payload is not needed to answer
    it. Deliberately not via :class:`~komugi.train.shards.ShardReader`, which
    only warns-and-skips a foreign fingerprint (§1.10) and would hand the builder
    a silently shrunken window rather than a recorded skip.
    """
    try:
        schema, _count, _columns = read_columns(path, ())
    except _UNREADABLE:
        return None
    return schema.fingerprint


@dataclass(frozen=True, slots=True)
class _SourceIteration:
    """One qualifying source iteration and its qualifying shards."""

    iteration: int
    shards: tuple[Path, ...]


def _walk_sources(
    root: Path, iteration: int, window_size: int, skipped: list[dict[str, Any]]
) -> list[_SourceIteration]:
    """Return the newest ≤ `window_size` qualifying iterations below `iteration`.

    §1.8's rule -- the campaign's own newest-first, *iterations-with-data* walk
    (**not** the contiguous ``max(0, i−5) … i−1`` range) -- plus KOM-014's two
    additions: a ``WALK_BOUND_MULTIPLIER × window_size`` bound on the walk, and
    the schema guard.

    **The guard qualifies shards, not iterations** (R11). A mixed-fingerprint
    directory is unreachable through normal operation, but under a
    per-*iteration* rule the index pass would project an ``action`` column that
    does not exist in half of it. Under the per-shard rule the foreign shards are
    simply absent, which makes the affected games' ply set partial and lands them
    in ``rejected_truncated_prefix`` by the existing rule -- no new mechanism.

    A non-qualifying iteration does **not** consume the window budget (it is not
    a source), and is recorded in `skipped` so a straddle of the fingerprint bump
    is *reported* rather than inferred from a quietly smaller window.
    """
    wanted = SCHEMA.fingerprint
    found: list[_SourceIteration] = []
    walked = 0
    candidate = iteration - 1
    while candidate >= 0 and walked < WALK_BOUND_MULTIPLIER * window_size:
        walked += 1
        paths = _iteration_shards(root, candidate)
        if paths:
            good: list[Path] = []
            foreign: list[str] = []
            for path in paths:
                fingerprint = _shard_fingerprint(path)
                if fingerprint == wanted:
                    good.append(path)
                else:
                    foreign.append("unreadable" if fingerprint is None else fingerprint)
            if good and foreign:
                skipped.append(
                    {
                        "iteration": candidate,
                        "reason": "schema",
                        "shards_skipped": len(foreign),
                    }
                )
            elif not good:
                skipped.append(
                    {
                        "iteration": candidate,
                        "fingerprint": foreign[0] if foreign else "",
                        "reason": "schema",
                    }
                )
            if good:
                found.append(_SourceIteration(candidate, tuple(good)))
                if len(found) >= window_size:
                    break
        candidate -= 1
    return found


@dataclass(slots=True)
class _Index:
    """What the index pass (§2.4 pass 1) recovers from the whole window.

    Three ``int64`` columns per shard -- 24 bytes a row -- which is what makes
    every per-game decision affordable **once**, before any candidate exists.
    """

    shards: list[tuple[int, Path]]
    plies: dict[_GameKey, set[int]]
    action_at: dict[tuple[int, int, int], int]
    row_at: dict[tuple[int, int, int], tuple[int, int]]


def _index_window(sources: Sequence[_SourceIteration]) -> _Index:
    """Read ``(game_id, ply, action)`` for every qualifying shard of the window."""
    shards: list[tuple[int, Path]] = []
    for source in sorted(sources, key=lambda item: item.iteration):
        shards.extend((source.iteration, path) for path in source.shards)

    index = _Index(shards=shards, plies={}, action_at={}, row_at={})
    for slot, (source_iteration, path) in enumerate(shards):
        _schema, count, columns = read_columns(path, ("game_id", "ply", "action"))
        game_ids = columns["game_id"]
        ply_values = columns["ply"]
        actions = columns["action"]
        for row in range(count):
            game_id = int(game_ids[row])
            ply = int(ply_values[row])
            key = (source_iteration, game_id)
            index.plies.setdefault(key, set()).add(ply)
            index.action_at[(source_iteration, game_id, ply)] = int(actions[row])
            index.row_at[(source_iteration, game_id, ply)] = (slot, row)
    return index


@dataclass(slots=True)
class _Admission:
    """The per-game verdicts of §3.2 step 2, made once in the index pass.

    ``skipped_*`` are counted in **games**, never in draws, and are deliberately
    outside ``candidates_drawn``'s partition (§2.2, R7): a data-loss signal and a
    sampler-accounting signal must not be added together.
    """

    eligible: dict[_GameKey, list[int]]
    pool: dict[int, list[tuple[_GameKey, int]]]
    booked: int
    truncated: int
    short: int


def _admit_games(index: _Index) -> _Admission:
    """Classify every source game and build the ``shard -> [(game, d)]`` pool.

    Rows are addressed **by their ``ply`` value, never by position in a list**:
    from generation 2 onward ~10% of the source window is itself booked and its
    rows begin at ``ply == L``, so ``rows[k]`` is not ply ``k``. Positional
    indexing there would replay actions from plies ``L … L+d−1`` out of a fresh
    game -- the "plausible wrong prefix" failure §3.1 rejects encoder inversion
    for.
    """
    admission = _Admission(eligible={}, pool={}, booked=0, truncated=0, short=0)
    for key in sorted(index.plies):
        present = index.plies[key]
        lowest = min(present)
        if lowest != 0:
            # A booked source game and a truncated one are kept apart because
            # they mean different things: the first is normal operation, the
            # second is data loss (a skipped non-final shard is the expected
            # shape of a preemption).
            if MIN_PLIES <= lowest <= MAX_PLIES:
                admission.booked += 1
            else:
                admission.truncated += 1
            continue
        length = max(present) + 1
        # [MIN_PLIES-1, MAX_PLIES-1] intersected with [0, length-2]: `length-2`
        # so at least one row after the prefix existed in the source game -- a
        # game that *ended* at ply d is not evidence that d is a live position.
        # The `d in present` filter is the same rule under data loss: a game
        # none of whose eligible plies survived has no deviation row to mine,
        # which is the shortness bucket's meaning and not a fourth case.
        eligible = [
            d
            for d in range(MIN_PLIES - 1, MAX_PLIES)
            if d <= length - 2 and d in present
        ]
        if not eligible:
            admission.short += 1
            continue
        admission.eligible[key] = eligible
        for deviation in eligible:
            slot, _row = index.row_at[(key[0], key[1], deviation)]
            admission.pool.setdefault(slot, []).append((key, deviation))
    return admission


def _truncated_support(
    target: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], bool]:
    """Return the truncated support ``S`` and whether the fallback branch fired.

    ``S`` = top :data:`SUPPORT_TOP_K` of ``{a : p[a] ≥ SUPPORT_EPSILON}``, or --
    when nothing clears ``ε`` -- top :data:`SUPPORT_MIN_K` of
    ``{a : p[a] ≥ SUPPORT_FLOOR}``. Both branches order by ``policy_target``
    descending with ties broken by **ascending action id**, so the support is a
    pure function of the row and byte-determinism survives (§2.3).

    An empty return means no legal action cleared ``SUPPORT_FLOOR``; the caller
    rejects the candidate rather than drawing from an empty array.
    """
    chosen = np.nonzero(target >= SUPPORT_EPSILON)[0]
    fallback = False
    limit = SUPPORT_TOP_K
    if chosen.size == 0:
        chosen = np.nonzero(target >= SUPPORT_FLOOR)[0]
        fallback = True
        limit = SUPPORT_MIN_K
    if chosen.size == 0:
        return np.zeros(0, dtype=np.int64), fallback
    order = np.lexsort((chosen, -target[chosen]))
    return np.asarray(chosen[order][:limit], dtype=np.int64), fallback


def _entropy_bits(distribution: npt.NDArray[np.float64]) -> float:
    """Return the Shannon entropy of `distribution`, in bits.

    ``+ 0.0`` at the end so a one-point distribution reports ``0.0`` rather than
    ``-0.0``: the sign of a zero is invisible in arithmetic and very visible in a
    byte-compared JSON file.
    """
    positive = distribution[distribution > 0.0]
    return float(-np.sum(positive * np.log2(positive))) + 0.0


def _tempered(
    target: npt.NDArray[np.float64],
    support: npt.NDArray[np.int64],
    temperature: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(p_S, p_τ)`` over `support` -- the restriction and its tempering.

    ``τ = 1`` reproduces ``p_S`` **exactly** (``x ** 1.0`` is ``x``, and the two
    normalisations are the identical expression), which is the identity
    acceptance #7a asserts and which no fixture can rig. ``τ → ∞`` is uniform
    *on ``S``*, not on the legal set.
    """
    mass = target[support]
    restricted = mass / mass.sum()
    weights = mass ** (1.0 / temperature)
    return restricted, weights / weights.sum()


def _entry_id(generation: int, index: int, actions: Sequence[int]) -> str:
    """Return ``gen{G:04d}-{index:06d}-{sha8}`` for one entry (§2.2)."""
    canonical = json.dumps(list(actions), separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"gen{generation:04d}-{index:06d}-{digest}"


def _rounded(value: Any) -> Any:
    """Round every float in a JSON payload to :data:`_FLOAT_PLACES`.

    Recursive because the floats are scattered (per-entry diagnostics, the
    manifest's constants, two ``stats`` rates) and because missing one would
    leave a platform's ``repr`` in a file acceptance #9 compares as bytes.
    ``bool`` is checked before ``float`` only for clarity -- it is an ``int`` --
    and ``None`` passes through, which is what makes R6's ``null``
    ``flattening_ratio`` survive serialisation.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, _FLOAT_PLACES)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _fsync_directory(directory: Path) -> None:
    """``fsync`` a directory, so a rename into it survives a power loss."""
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_durably(path: Path, text: str) -> None:
    """Publish `text` at `path` atomically and durably.

    ``save_checkpoint``'s idiom, copied: dot-prefixed temp file, ``fsync`` the
    bytes, ``os.replace`` to publish, ``fsync`` the directory so the *name*
    survives too. A reader therefore sees a book either not at all or entirely.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{os.getpid()}"
    handle = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SHARD_MODE)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(text)
            stream.flush()
        os.fsync(handle)
    finally:
        os.close(handle)
    os.replace(tmp_path, path)
    _fsync_directory(path.parent)


def build_book(
    root: str | os.PathLike[str],
    iteration: int,
    run_seed: int,
    *,
    temperature: float = TEMPERATURE,
    cap: int = CAP,
    min_entries: int = MIN_ENTRIES,
    entries_per_source_game: int = ENTRIES_PER_SOURCE_GAME,
    target_shard_budget: int = TARGET_SHARD_BUDGET,
) -> Path | None:
    """Build ``<root>/books/gen_<iteration>.json`` and return its path, or ``None``.

    ``None`` -- and **no file** -- when the sources yield fewer than
    `min_entries` accepted entries. Iteration 0, and the ≤ 5 generations after a
    fingerprint bump during which the qualifying window is thin, both reach that
    branch; both are the expected shape of the situation and not errors, and
    ``stats.skipped_source_iterations`` in a *written* book is what distinguishes
    them from a builder that quietly found nothing.

    The two passes are §2.4's, and the reason there are two is cost: the index
    pass projects three ``int64`` columns over the whole window, and only the
    *deviation* row of an accepted candidate needs ``policy_target`` -- one
    shard's worth of which is ~153 MB decompressed. The target pass therefore
    schedules at most `target_shard_budget` shards, visits each exactly once in
    ascending ``(source_iteration, shard index)`` order, draws a whole batch from
    that shard's pool while its column is open, and releases it.

    Every draw -- the shard permutation, the game, the deviation ply, the
    tempered action -- comes from **one** ``Generator(PCG64(SeedSequence(...)))``
    consumed in exactly that order, and entries are emitted in draw order rather
    than sorted. That is what makes two builds over one shard tree byte-identical
    (#9); sorting would be equally deterministic but would hide a sampler that
    drew badly.

    Args:
        root: The campaign root. **The only path this function takes** -- the
            mechanical half of "no human lines" (#12): there is no code path by
            which an external line could enter, because there is no argument to
            hand one through.
        iteration: The generation the book is *for*. Sources are strictly below
            it, so no booked game of iteration ``i`` can use a prefix from ``i``.
        run_seed: The campaign's run seed. Combined with
            :data:`~komugi.selfplay.records.BOOK_BUILD_DOMAIN` and `iteration`
            into the build seed.
        temperature: τ. The default is the pre-registered
            :data:`TEMPERATURE`; acceptance #7a builds at 1.0 over the same
            fixture to assert the ``flattening_ratio == 0`` identity.
        cap: Maximum accepted entries.
        min_entries: Below this, no file is written.
        entries_per_source_game: The per-source-game acceptance quota.
        target_shard_budget: How many shards the target pass may open.

    Returns:
        The path written, or ``None`` if the book was too thin to write.
    """
    root_path = Path(root)
    build_seed = book_seed(run_seed, BOOK_BUILD_DOMAIN, iteration)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(build_seed)))

    skipped_source_iterations: list[dict[str, Any]] = []
    sources = _walk_sources(
        root_path, iteration, WindowConfig().size, skipped_source_iterations
    )
    if not sources:
        return None

    index = _index_window(sources)
    admission = _admit_games(index)
    candidate_slots = sorted(admission.pool)
    if not candidate_slots:
        return None

    permutation = [int(slot) for slot in rng.permutation(len(candidate_slots))]
    scheduled = sorted(
        candidate_slots[position] for position in permutation[:target_shard_budget]
    )
    per_visit = max(1, math.ceil(DRAW_BUDGET_MULTIPLIER * cap / len(scheduled)))

    reachable: set[_GameKey] = set()
    for slot in scheduled:
        reachable.update(key for key, _d in admission.pool[slot])

    entries: list[BookEntry] = []
    seen: set[tuple[int, ...]] = set()
    quota: dict[_GameKey, int] = {}
    drawn = 0
    rejected = {
        "rejected_illegal": 0,
        "rejected_terminal": 0,
        "rejected_truncated_prefix": 0,
        "rejected_duplicate": 0,
        "rejected_stem_quota": 0,
        "rejected_empty_support": 0,
    }
    shards_read = 0

    for slot in scheduled:
        if len(entries) >= cap:
            break
        _source_iteration, path = index.shards[slot]
        _schema, _count, columns = read_columns(path, ("policy_target",))
        targets = columns["policy_target"]
        shards_read += 1
        by_game: dict[_GameKey, list[int]] = {}
        for key, deviation in admission.pool[slot]:
            by_game.setdefault(key, []).append(deviation)
        games = sorted(by_game)
        for _draw in range(per_visit):
            if len(entries) >= cap:
                break
            drawn += 1
            game = games[int(rng.integers(0, len(games)))]
            plies = by_game[game]
            deviation = plies[int(rng.integers(0, len(plies)))]
            entry = _draw_candidate(
                rng=rng,
                index=index,
                targets=targets,
                game=game,
                deviation=deviation,
                temperature=temperature,
                generation=iteration,
                position=len(entries),
                seen=seen,
                quota=quota,
                entries_per_source_game=entries_per_source_game,
                rejected=rejected,
            )
            if entry is not None:
                seen.add(entry.actions)
                quota[game] = quota.get(game, 0) + 1
                entries.append(entry)
        # Released before the next visit: peak RSS is ONE column (§2.4).
        del targets, columns

    accounted = len(entries) + sum(rejected.values())
    if accounted != drawn:
        raise RuntimeError(
            "book sampler accounting is not a partition: "
            f"{drawn} candidates drawn but {accounted} accounted for "
            f"({len(entries)} accepted, {rejected}); a draw that reaches no "
            "bucket is a sampler bug, not a rounding difference"
        )

    if len(entries) < min_entries:
        return None

    stats = _build_stats(
        entries=entries,
        drawn=drawn,
        rejected=rejected,
        admission=admission,
        reachable=reachable,
        shards_read=shards_read,
        skipped_source_iterations=skipped_source_iterations,
    )
    payload: dict[str, Any] = {
        "book_version": BOOK_VERSION,
        "generation": iteration,
        "source_iterations": sorted(source.iteration for source in sources),
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": float(temperature),
        "support_epsilon": SUPPORT_EPSILON,
        "support_top_k": SUPPORT_TOP_K,
        "support_floor": SUPPORT_FLOOR,
        "support_min_k": SUPPORT_MIN_K,
        "target_shard_budget": target_shard_budget,
        "cap": cap,
        "min_plies": MIN_PLIES,
        "max_plies": MAX_PLIES,
        "entries_per_source_game": entries_per_source_game,
        "build_seed": build_seed,
        "bit_generator": BIT_GENERATOR,
        "entries": [entry.as_json() for entry in entries],
        "stats": stats,
    }
    # allow_nan=False (R6): the default emits the bare tokens NaN/Infinity,
    # which are not JSON and which a strict reader refuses. A non-finite value
    # is a build-time exception here rather than an unparseable artefact.
    text = json.dumps(
        _rounded(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path = book_path_for(root_path, iteration)
    _write_durably(path, text + "\n")
    return path


def _draw_candidate(
    *,
    rng: np.random.Generator,
    index: _Index,
    targets: npt.NDArray[Any],
    game: _GameKey,
    deviation: int,
    temperature: float,
    generation: int,
    position: int,
    seen: set[tuple[int, ...]],
    quota: Mapping[_GameKey, int],
    entries_per_source_game: int,
    rejected: dict[str, int],
) -> BookEntry | None:
    """Turn one ``(game, d)`` pair into an entry, or count why it was rejected.

    Every candidate lands in **exactly one** bucket -- accepted, or one of the
    six ``rejected_*`` -- which is the partition :func:`build_book` asserts
    before writing. The order of the checks is cost-ascending where that does
    not change the verdict: the quota is a dictionary lookup and the replay is
    hundreds of engine calls, so a game that is already at its quota is rejected
    before it is replayed.

    An illegal action during the replay is *counted*, not raised: the builder
    verified nothing yet, and a shard that disagrees with the engine is exactly
    what this check exists to notice. The loud failure is the worker's, at load
    time (§6.3), where the book has already claimed the prefix is legal.
    """
    source_iteration, game_id = game
    present = index.plies[game]
    if any(ply not in present for ply in range(deviation + 1)):
        # A hole anywhere in 0…d means a shard carrying part of this prefix was
        # skipped, and the replay would silently splice across it.
        rejected["rejected_truncated_prefix"] += 1
        return None
    if quota.get(game, 0) >= entries_per_source_game:
        rejected["rejected_stem_quota"] += 1
        return None

    prefix = [
        index.action_at[(source_iteration, game_id, ply)] for ply in range(deviation)
    ]
    state = GameState(Game(), _REPLAY_DRAW_SCORE)
    for action in prefix:
        if state.is_terminal() or not bool(state.legal_mask()[action]):
            rejected["rejected_illegal"] += 1
            return None
        state.play(action)
    if state.is_terminal():
        rejected["rejected_illegal"] += 1
        return None

    _slot, row = index.row_at[(source_iteration, game_id, deviation)]
    target = np.asarray(targets[row], dtype=np.float64)
    support, fallback = _truncated_support(target)
    if support.size == 0:
        rejected["rejected_empty_support"] += 1
        return None

    restricted, tempered = _tempered(target, support, temperature)
    drawn_action = int(support[int(rng.choice(support.size, p=tempered))])
    # S ⊆ {p > 0} ⊆ the legal set by §1.2 -- verified rather than trusted, since
    # a disagreement means the shard and the engine disagree.
    if not bool(state.legal_mask()[drawn_action]):
        rejected["rejected_illegal"] += 1
        return None
    state.play(drawn_action)
    if state.is_terminal():
        rejected["rejected_terminal"] += 1
        return None

    actions = (*prefix, drawn_action)
    if actions in seen:
        # First occurrence wins, in draw order. Not a prefix-tree dedup: two
        # entries sharing a stem but differing at the deviation are two
        # different opening positions and both belong.
        rejected["rejected_duplicate"] += 1
        return None

    restricted_entropy = _entropy_bits(restricted)
    tempered_entropy = _entropy_bits(tempered)
    headroom = math.log2(support.size) - restricted_entropy
    ratio = (
        (tempered_entropy - restricted_entropy) / headroom if headroom > 0.0 else None
    )
    return BookEntry(
        id=_entry_id(generation, position, actions),
        actions=actions,
        plies=len(actions),
        source_iteration=source_iteration,
        source_game_id=game_id,
        deviation_ply=deviation,
        deviated=drawn_action
        != index.action_at[(source_iteration, game_id, deviation)],
        support_size=int(support.size),
        support_fallback=fallback,
        truncated_mass=1.0 - float(target[support].sum()),
        drawn_target_prob=float(target[drawn_action]),
        restricted_entropy_bits=restricted_entropy,
        tempered_entropy_bits=tempered_entropy,
        flattening_ratio=ratio,
    )


def _build_stats(
    *,
    entries: Sequence[BookEntry],
    drawn: int,
    rejected: Mapping[str, int],
    admission: _Admission,
    reachable: set[_GameKey],
    shards_read: int,
    skipped_source_iterations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the manifest's ``stats`` object (§2.2).

    ``rejected_*`` are per-**draw** and partition ``candidates_drawn``;
    ``skipped_*_source_games`` are per-**game**, counted once each in the index
    pass, and appear in no partition. Mixing the two units was rev 2's bug and is
    why the prefixes differ.

    **``rejected_illegal`` covers "illegal *or already terminal* during the stem
    replay"** -- §3.2 step 4's legality failure and step 6's terminality failure
    share this bucket whenever they happen while replaying plies ``0 … d−1``
    (:func:`_draw_candidate`). ``rejected_terminal`` is narrower than its name
    suggests: it counts only the position reached *after* the tempered action,
    which is the one §3.2 step 6 is actually about. Named here rather than split
    into a seventh bucket because the two are the same verdict about the same
    stem -- "this recorded line is not a live draft position" -- and the R7
    partition is unaffected either way.
    """
    included: list[float] = []
    for entry in entries:
        ratio = entry.flattening_ratio
        if ratio is None:
            continue
        headroom = math.log2(entry.support_size) - entry.restricted_entropy_bits
        if (
            headroom >= _MIN_FLATTENING_HEADROOM
            and entry.restricted_entropy_bits >= _MIN_RESTRICTED_ENTROPY
        ):
            included.append(ratio)
    total = len(entries)
    stats: dict[str, Any] = {
        "candidates_drawn": drawn,
        "distinct_entries": total,
        "distinct_stems": len({entry.stem for entry in entries}),
        "distinct_source_games": len(
            {(entry.source_iteration, entry.source_game_id) for entry in entries}
        ),
        "deviation_rate": (
            sum(1 for entry in entries if entry.deviated) / total if total else 0.0
        ),
        "support_fallback": sum(1 for entry in entries if entry.support_fallback),
        "excluded_from_flattening": total - len(included),
        # Exactly 0 by construction: every action in S clears SUPPORT_FLOOR and
        # R3's tail is `policy_target < SUPPORT_FLOOR`, so the two are disjoint.
        # Computed rather than asserted, so that dropping the truncation shows
        # up here as a number instead of as an unexercised comment.
        "tail_mass": (
            sum(1 for entry in entries if entry.drawn_target_prob < SUPPORT_FLOOR)
            / total
            if total
            else 0.0
        ),
        "median_flattening_ratio": (
            statistics.median(included) if included else None
        ),
        "skipped_booked_source_games": admission.booked,
        "skipped_truncated_source_games": admission.truncated,
        "skipped_short_source_games": admission.short,
        "skipped_budget_source_games": sum(
            1 for key in admission.eligible if key not in reachable
        ),
        "target_shards_read": shards_read,
        "skipped_source_iterations": [dict(item) for item in skipped_source_iterations],
    }
    stats.update(rejected)
    return stats


def _entry_from_json(payload: Mapping[str, Any]) -> BookEntry:
    """Rebuild one :class:`BookEntry` from the manifest, or raise ``ValueError``."""
    ratio = payload.get("flattening_ratio")
    return BookEntry(
        id=str(payload["id"]),
        actions=tuple(int(action) for action in payload["actions"]),
        plies=int(payload["plies"]),
        source_iteration=int(payload["source_iteration"]),
        source_game_id=int(payload["source_game_id"]),
        deviation_ply=int(payload["deviation_ply"]),
        deviated=bool(payload["deviated"]),
        support_size=int(payload["support_size"]),
        support_fallback=bool(payload["support_fallback"]),
        truncated_mass=float(payload["truncated_mass"]),
        drawn_target_prob=float(payload["drawn_target_prob"]),
        restricted_entropy_bits=float(payload["restricted_entropy_bits"]),
        tempered_entropy_bits=float(payload["tempered_entropy_bits"]),
        flattening_ratio=None if ratio is None else float(ratio),
    )


def load_book(path: str | os.PathLike[str]) -> Book:
    """Parse a book file, refusing one this process cannot replay (§6.5).

    Four refusals, all naming *both* values, because "the book is wrong" is not
    an actionable message:

    * ``book_version != BOOK_VERSION`` -- checked **first**, before any other
      manifest key is read, because it is the only field that says whether the
      rest of the manifest means what this reader thinks it means.
      :data:`BOOK_VERSION` is bumped exactly when a reader must change, so a v2
      file parsed by a v1 reader is the one case where every *other* check can
      pass and still be answering the wrong question.
    * ``action_size != num_actions()`` -- **this is the guard that actually
      protects a prefix.** An action id means nothing across an encoding change:
      the same integer decodes to a different move, so a stale book would replay
      a plausible wrong line rather than fail.
    * ``gungi_mode`` differs from the running mode -- a prefix is replayed from a
      bare :class:`~komugi.engine.game.Game`, and ``GUNGI_MODE`` decides which
      position that is (§1.11).
    * ``plane_map_version`` differs from
      :data:`~komugi.engine.observation.PLANE_MAP_VERSION` -- the book was mined
      from shards of another data generation. Harmless in isolation (a book is
      action-id prefixes, not observations), but the campaign's ``_check_book``
      is the only place a stale book is caught *by name* instead of being mined
      into a fresh root unnoticed; the shard fingerprint provably cannot see a
      plane-map change (KOM-165 design review F7).

    Entry-level bounds (``plies`` in range, ``len(actions) == plies``, action ids
    in ``[0, action_size)``) are checked here too, so a hand-edited file fails at
    load rather than at replay.

    Raises:
        ValueError: On unparseable JSON, a missing manifest key, a format
            version this reader does not read, a mismatched action space or
            mode, or an entry that violates its own manifest.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload: Any = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"book {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"book {path} must be a JSON object, got {type(payload)}")

    try:
        version = int(payload["book_version"])
    except KeyError as exc:
        raise ValueError(f"book {path} is missing manifest key {exc}") from exc
    if version != BOOK_VERSION:
        raise ValueError(
            f"book {path} declares book_version {version}, but this process "
            f"reads book_version {BOOK_VERSION}: the constant is bumped exactly "
            "when a reader must change, so parsing this file under the running "
            "version's rules would read one format as if it were another"
        )

    try:
        action_size = int(payload["action_size"])
        mode = str(payload["gungi_mode"])
    except KeyError as exc:
        raise ValueError(f"book {path} is missing manifest key {exc}") from exc

    running_actions = num_actions()
    if action_size != running_actions:
        raise ValueError(
            f"book {path} was built for an action space of {action_size} "
            f"actions, but this process encodes {running_actions}: an action id "
            "means nothing across an encoding change, so the prefixes cannot be "
            "replayed"
        )
    running_mode = running_gungi_mode()
    if mode != running_mode:
        raise ValueError(
            f"book {path} was built under GUNGI_MODE={mode!r}, but this process "
            f"runs {running_mode!r}: a prefix replayed from a different start "
            "position is a different line, or illegal"
        )

    try:
        book_plane_map_version = int(payload["plane_map_version"])
    except KeyError as exc:
        raise ValueError(f"book {path} is missing manifest key {exc}") from exc
    if book_plane_map_version != PLANE_MAP_VERSION:
        raise ValueError(
            f"book {path} was mined from shards encoded under plane map version "
            f"{book_plane_map_version}, but this process encodes plane map "
            f"version {PLANE_MAP_VERSION}: the shard fingerprint cannot see a "
            "plane-map change (it is a pure function of field names, dtypes and "
            "shapes), so this is the only place a cross-generation book is "
            "refused by name rather than replayed silently (KOM-165 F7)"
        )

    entries = tuple(_entry_from_json(item) for item in payload["entries"])
    for entry in entries:
        if not MIN_PLIES <= entry.plies <= MAX_PLIES:
            raise ValueError(
                f"book {path} entry {entry.id} claims {entry.plies} plies, "
                f"outside [{MIN_PLIES}, {MAX_PLIES}]"
            )
        if len(entry.actions) != entry.plies:
            raise ValueError(
                f"book {path} entry {entry.id} carries {len(entry.actions)} "
                f"actions but claims {entry.plies} plies"
            )
        if any(not 0 <= action < action_size for action in entry.actions):
            raise ValueError(
                f"book {path} entry {entry.id} carries an action id outside "
                f"[0, {action_size})"
            )
    return Book(
        book_version=version,
        generation=int(payload["generation"]),
        source_iterations=tuple(int(item) for item in payload["source_iterations"]),
        gungi_mode=mode,
        action_size=action_size,
        plane_map_version=book_plane_map_version,
        source_schema_fingerprint=str(payload["source_schema_fingerprint"]),
        temperature=float(payload["temperature"]),
        support_epsilon=float(payload["support_epsilon"]),
        support_top_k=int(payload["support_top_k"]),
        support_floor=float(payload["support_floor"]),
        support_min_k=int(payload["support_min_k"]),
        target_shard_budget=int(payload["target_shard_budget"]),
        cap=int(payload["cap"]),
        min_plies=int(payload["min_plies"]),
        max_plies=int(payload["max_plies"]),
        entries_per_source_game=int(payload["entries_per_source_game"]),
        build_seed=int(payload["build_seed"]),
        bit_generator=str(payload["bit_generator"]),
        entries=entries,
        stats=dict(payload["stats"]),
    )


DEFAULT_BOOK_FRACTION: Final[float] = 0.10
"""The pre-registered share of games that start from a book (§0 B4, §6.1).

Not :class:`~komugi.config.SelfPlayConfig`'s default -- that one is ``0.0``, and
that class documents why the two numbers differ. This is the value §9's R4
audits production against, named once here so an operator's config and the
audit's expectation cite the same constant."""


@dataclass(frozen=True, slots=True)
class BookChoice:
    """What §4.2's two draws decided for one game.

    Args:
        booked: Whether this game starts from a book prefix.
        entry_index: Index into :attr:`Book.entries`, or ``None`` when the game
            is unbooked. ``None`` rather than ``-1`` because an unbooked game
            has no entry at all, and a sentinel index is the kind of value that
            silently reads back as "the last entry".
    """

    booked: bool
    entry_index: int | None


def book_select_rng(
    run_seed: int, worker_id: int, incarnation: int, game_index: int
) -> np.random.Generator:
    """Return one game's **book** stream -- disjoint from every other stream.

    ``SeedSequence(entropy=game_seed(...).entropy, spawn_key=(*spawn_key,
    BOOK_SELECT_DOMAIN))``, i.e. :func:`~komugi.selfplay.records.move_seed`'s
    spawn-key composition with a domain constant where a ply would be (§4.2).
    Three disjointnesses, each deliberate:

    * from the game's own ``rng`` (``worker.py``'s full/fast draw), which is
      why a booked run and a bookless run at one seed still draw the identical
      full/fast sequence and §9's two arms differ in the book alone;
    * from the per-move Gumbel stream, because
      :func:`~komugi.selfplay.records.move_seed` composes a *ply* and the engine
      adjudicates at the engine's cap (far below ``0x9014_0000`` at any cap this
      engine can be configured with), so ``0x9014_0002`` is unreachable as one;
    * across incarnations and lanes, because both are already in the spawn key
      -- a respawned worker books a different subset, which is correct since
      its predecessor's games are gone.

    ``Generator(PCG64(...))`` with the bit generator named rather than
    ``default_rng``, for the reason :data:`BIT_GENERATOR` gives (R8): this draw
    is recomputed offline, in another process, by the audit.
    """
    seeds = game_seed(run_seed, worker_id, incarnation, game_index)
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                entropy=seeds.entropy,
                spawn_key=(*seeds.spawn_key, BOOK_SELECT_DOMAIN),
            )
        )
    )


def select_book_entry(
    run_seed: int,
    worker_id: int,
    incarnation: int,
    game_index: int,
    *,
    entries: int,
    fraction: float,
) -> BookChoice:
    """Return whether one game is booked, and with which entry (§4.2).

    The two draws, in this order and no other: ``random() < fraction``, then --
    **only if** that came up booked -- ``integers(0, entries)``. The order is
    part of the answer, because the audit replays it; and the second draw is
    conditional so that an unbooked game consumes exactly one number, which is
    what makes the booked *fraction* a property of the first draw alone.

    ``integers(0, n)``, never ``% n``: a modulo over a machine word is biased
    for any ``n`` that is not a power of two, and the cap is 1024 today only by
    choice (§4.2).

    Args:
        run_seed: The seed **the worker was handed** -- in a campaign that is
            ``iteration_seed(campaign.run_seed, iteration)``, not the
            campaign's own seed (§6.7, N3).
        worker_id: The worker's index within the run.
        incarnation: The worker's respawn counter.
        game_index: The game's index within the worker's workload.
        entries: How many entries the book holds.
        fraction: ``SelfPlayConfig.book_fraction``.

    Returns:
        The :class:`BookChoice` this game's stream decided.

    Raises:
        ValueError: If `entries` is below 1. A book with no entry cannot book a
            game, and returning "unbooked" would hide an empty book behind a
            perfectly ordinary-looking 0% booking rate.
    """
    if entries < 1:
        raise ValueError(
            f"cannot select from a book of {entries} entries: an empty book is "
            f"not a bookless run, and reporting it as one would hide it"
        )
    rng = book_select_rng(run_seed, worker_id, incarnation, game_index)
    if not bool(rng.random() < fraction):
        return BookChoice(booked=False, entry_index=None)
    return BookChoice(booked=True, entry_index=int(rng.integers(0, entries)))


@dataclass(frozen=True, slots=True)
class BookedFraction:
    """The §6.1(b) audit's answer over some range of iterations.

    ``truncated`` games are in **neither** the numerator nor the denominator
    (N2): a game one of whose non-final shards was skipped presents
    ``min(ply) > 0`` with no book involved, and a preemption must not read as a
    booking-rate anomaly. The residual R9 states is real and is not fixable from
    ``min(ply)`` alone -- a booked game that also lost rows *within* ``[4, 12]``
    reads as booked with a fabricated ``L`` -- so this class reports the counts it
    can defend and nothing more.
    """

    booked: int
    unbooked: int
    truncated: int
    fraction: float
    interval: tuple[float, float]

    @property
    def classified(self) -> int:
        """Return the denominator: booked plus unbooked, truncated excluded."""
        return self.booked + self.unbooked


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return ``P(X <= successes)`` for ``X ~ Binomial(trials, probability)``.

    Summed in log space through :func:`math.lgamma` so a 1500-trial coefficient
    does not overflow. No SciPy: the dev shell does not carry it, and a 60-step
    bisection over this is milliseconds.
    """
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes >= trials else 0.0
    total = 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    for count in range(successes + 1):
        total += math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(count + 1)
            - math.lgamma(trials - count + 1)
            + count * log_p
            + (trials - count) * log_q
        )
    return min(1.0, total)


def _bisect(target: float, trials: int, successes: int, *, steps: int = 60) -> float:
    """Return the ``p`` solving ``_binomial_cdf(successes, trials, p) == target``.

    The CDF is monotonically decreasing in ``p``, which is what makes plain
    bisection exact enough here and worth more than an approximation nobody can
    check.
    """
    low, high = 0.0, 1.0
    for _step in range(steps):
        middle = (low + high) / 2.0
        if _binomial_cdf(successes, trials, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(
    successes: int, trials: int, *, confidence: float = 0.99
) -> tuple[float, float]:
    """Return the exact-binomial two-sided interval at `confidence`.

    Exact, not normal-approximate, because the audit's whole point is that
    ``[8%, 12%]`` at ``n = 200`` is a ±1σ band that fails one honest run in three
    (§6.1): an interval that is itself an approximation could not have made that
    argument.
    """
    if trials <= 0:
        return (0.0, 1.0)
    alpha = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else _bisect(1.0 - alpha, trials, successes - 1)
    upper = 1.0 if successes == trials else _bisect(alpha, trials, successes)
    return (lower, upper)


def booked_fraction(
    root: str | os.PathLike[str], iterations: Iterable[int]
) -> BookedFraction:
    """Classify every source game of `iterations` as unbooked, booked or truncated.

    §4.3's three-way ``min(ply)`` rule, which every consumer of ``min(ply)`` in
    this design shares and none of them replaces with a two-way one:

    ``m == 0``
        unbooked.
    ``m ∈ [MIN_PLIES, MAX_PLIES]``
        booked, with ``L = m``.
    otherwise
        **truncated** -- excluded from the numerator *and* the denominator,
        reported separately, never silently counted as booked.

    Reads two ``int64`` columns per shard and nothing else.
    """
    root_path = Path(root)
    lowest: dict[_GameKey, int] = {}
    for iteration in iterations:
        for path in _iteration_shards(root_path, iteration):
            if _shard_fingerprint(path) != SCHEMA.fingerprint:
                continue
            _schema, count, columns = read_columns(path, ("game_id", "ply"))
            game_ids = columns["game_id"]
            ply_values = columns["ply"]
            for row in range(count):
                key = (iteration, int(game_ids[row]))
                ply = int(ply_values[row])
                current = lowest.get(key)
                if current is None or ply < current:
                    lowest[key] = ply

    booked = unbooked = truncated = 0
    for value in lowest.values():
        if value == 0:
            unbooked += 1
        elif MIN_PLIES <= value <= MAX_PLIES:
            booked += 1
        else:
            truncated += 1
    classified = booked + unbooked
    return BookedFraction(
        booked=booked,
        unbooked=unbooked,
        truncated=truncated,
        fraction=booked / classified if classified else 0.0,
        interval=clopper_pearson(booked, classified),
    )
