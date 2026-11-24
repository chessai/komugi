"""The draft-book builder, unit-scale (KOM-014 §8 acceptances #1-#13, #19, #33-#36b).

Every fixture here is a **real shard tree**: rows written through the real
:class:`~komugi.train.shards.ShardWriter` at the real
:data:`~komugi.selfplay.records.SCHEMA`, carrying ``action`` values that are
actually legal for the position they claim. That is the expensive choice and it
is the only honest one -- a book is a list of moves that must replay, so a
fixture whose actions are made up would let a builder that splices the wrong
plies pass every test in this file.

The construction is cheap in the one place it can be: a planted game is a fresh
:class:`~komugi.engine.game.Game` walked forward by *legal* actions drawn from a
fixed generator, and its ``policy_target`` is synthesised onto the legal actions
of that same position with a controllable shape. ``observation`` is zeros --
nothing in the builder reads it, and 41 KB a row of noise would buy nothing.

**Two shapes of shape.** :data:`_PEAKED` and :data:`_FLAT` put their mass on
exactly 16 legal actions, which is what §3.2's worked entropy examples are
computed on. That was for a long time the *only* shape in the suite, and the
2026-09-06 implementation review's F1/F2 are what that cost: on a 16-action row
the recorded support, the above-``ε`` set and the top-``k`` are the same set, so
truncation is a no-op, R3's fallback is unreachable, and three separate mutations
of :func:`~komugi.selfplay.book._truncated_support` produce a byte-identical
book. The :class:`_Spread` rows exist to leave that axis: they carry a mass on
*every* legal action of their position, and the three of them
(:data:`_PEAKED_OVER_A_TAIL`, :data:`_UNIFORM_SUB_EPSILON`,
:data:`_ALL_BELOW_FLOOR`) select the three branches §3.2 step 5 can take.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from komugi.engine.game import Game
from komugi.engine.observation import CONTROL_EMPTY, CONTROL_SIZE, PLANE_MAP_VERSION
from komugi.selfplay import book
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import (
    ENTRIES_PER_SOURCE_GAME,
    MIN_PLIES,
    MAX_PLIES,
    TARGET_SHARD_BUDGET,
    Book,
    book_path_for,
    build_book,
    load_book,
)
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.train.shards import Schema, ShardWriter
from komugi.train.window import WindowConfig

_SUPPORT_WIDTH: Final[int] = 16
"""How many legal actions a :data:`_PEAKED`/:data:`_FLAT` row puts mass on.

Exactly :data:`~komugi.selfplay.book.SUPPORT_TOP_K`, so ``|S| == 16`` on those
rows and the flattening ratio's denominator ``log2|S| − H_S`` is the 4-bit
headroom §3.2's worked examples use.

Doubles as the **literal** the spread nodes assert ``support_size`` against: a
node that read ``book.support_top_k`` back out of the manifest it is checking
could not discriminate a change to that constant, which is how mutant M3
survived.
"""

_PEAKED: Final[tuple[float, ...]] = (0.70, 0.10, 0.05) + (0.0115,) * 13
"""§3.2's worked *peaked* restriction: ``H_S = 1.874``, ratio ``0.78`` at τ = 2."""

_FLAT: Final[tuple[float, ...]] = (0.30, 0.20, 0.15, 0.10) + (0.0208,) * 12
"""§3.2's worked *near-uniform* restriction: ``H_S = 3.125``, ratio ``0.76``.

Two shapes rather than one because #7c's claim is **scale-freeness**: the ratio
must answer nearly the same number for two inputs a bit apart in entropy, which
a single-shape fixture could not show and could not fail to show.
"""

_ORDINARY_GAMES: Final[int] = 200
"""Enough source games that #36b's quota does not itself starve the book."""

_ORDINARY_PLIES: Final[int] = 6
"""Six plies gives eligible deviations ``d ∈ {3, 4}`` (``[3,11] ∩ [0, n_s−2]``),
which is the shortest game that exercises the range at all."""

_LONG_PLIES: Final[int] = 15
"""A source game long enough to reach the **upper** prefix bound.

``d ≤ n_s − 2`` caps the deviation at ``length − 2``, so ``d = MAX_PLIES − 1 =
11`` -- the only ``d`` that yields a ``MAX_PLIES``-ply entry -- needs
``n_s ≥ 13``. Every other fixture in this file tops out at 12 plies, which is
why the impl review found ``MAX_PLIES`` unreachable in the whole suite (F3, M4):
``d ≤ 10`` there, so no entry ever exceeded 11 plies and ``range(MIN_PLIES − 1,
MAX_PLIES + 1)`` was a no-op mutation. Fifteen rather than thirteen so the bound
is cleared with margin rather than exactly."""

_LONG_GAMES: Final[int] = 24
"""Enough long games that *some* accepted entry lands on ``d = 11``.

Each game offers nine eligible deviations (``d ∈ {3 … 11}``) and the quota
accepts eight, so one game alone would miss ``d = 11`` about 39% of the time.
Across the ~16 games the cap consumes that is ~4e-7 -- and the draw is seeded,
so it is a fixed outcome, not a coin flip re-tossed per run."""

_BOOKED_GAME_ID: Final[int] = 900
_HOLED_GAME_ID: Final[int] = 901
_TRUNCATED_GAME_ID: Final[int] = 902
_MIXED_GAME_ID: Final[int] = 903
_DIFFUSE_GAME_ID: Final[int] = 910
_PEAKED_TAIL_GAME_ID: Final[int] = 911
_BELOW_FLOOR_GAME_ID: Final[int] = 912

_TRUNCATED_FIRST_PLY: Final[int] = 37
"""Where the preempted source game's surviving rows start (F7, N2).

Outside ``[MIN_PLIES, MAX_PLIES]``, so ``min(ply)`` classifies the game
``truncated`` rather than ``booked`` -- the same construction
``tests/campaign/unit/test_book_audit.py`` plants for the *audit's* copy of the
three-way rule. Until this game existed the builder's copy was unpinned: no
fixture had a source game with ``min(ply) > 0`` outside the range, so collapsing
``_admit_games``' three-way classification to two changed nothing anyone
asserted (mutant M9)."""

_SAMPLES_PER_SHARD: Final[int] = 64
_CAP: Final[int] = 128
_MIN_ENTRIES: Final[int] = 32

_SHORT_TURN_CAP: Final[int] = 5
"""The turn cap :class:`_ShortGame` adjudicates at, chosen so that **both**
verdicts appear in one book: a ``d = 3`` candidate reaches 4 plies and lives, a
``d = 4`` candidate reaches 5 and is terminal the moment the tempered action is
played. See :func:`test_a_prefix_that_ends_the_game_is_rejected`."""


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_TURN_CAP` turns.

    ``Game.win_state`` applies the turn cap *before* the draft-phase early-out
    (``game.py:475-482``), so this is the one termination the builder's replay
    can be made to hit from a four-ply prefix. The idiom is
    ``tests/selfplay/unit/test_worker_unit.py``'s, restated here rather than
    imported for the reason ``test_book_prefixes_replay.py``'s docstring gives
    about cross-directory test imports.
    """

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_TURN_CAP


# ---------------------------------------------------------------------------
# Wide policy_target shapes -- the axis the whole suite used to share
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spread:
    """A ``policy_target`` shape defined over a position's **whole** legal set.

    :data:`_PEAKED` and :data:`_FLAT` put their mass on exactly 16 actions, which
    is the single fixture axis the impl review found makes truncation a no-op:
    ``{a : p[a] > 0}``, ``{a : p[a] ≥ SUPPORT_EPSILON}`` and the top-16 are all
    the *same* set, so a sampler with the truncation deleted, or applied after
    tempering, produces a byte-identical book (F1, mutants M1/M2/M3). A spread
    breaks that identity by giving every legal action a mass.

    Args:
        head: Masses for the first ``len(head)`` legal action ids, in order.
        tail: The mass every *remaining* legal action carries.
    """

    head: tuple[float, ...]
    tail: float


_MIN_SPREAD_WIDTH: Final[int] = 48
"""How many legal actions a spread row needs before it discriminates anything.

Must exceed ``len(_PEAKED_OVER_A_TAIL.head) == 40`` so that row has a real tail,
and exceed ``SUPPORT_TOP_K`` so ``|S|`` is genuinely a *truncation*. Gungi's
branching is ~229 early (max draft branching 577), so this is slack, but it is
asserted rather than assumed: a fixture that silently narrowed to 16 actions
would put the suite straight back on the axis these rows exist to leave."""

_DIFFUSE_MASS: Final[float] = 9.9e-4
"""Uniform mass below ``SUPPORT_EPSILON`` (1e-3) and above ``SUPPORT_FLOOR``
(1e-4) -- the R3 fallback branch's whole precondition, and the review's own
number. Representable in ``float32`` a comfortable ~1e-5 clear of ``ε``."""

_SUB_FLOOR_MASS: Final[float] = 5e-5
"""A mass inside R3's tail: below ``SUPPORT_FLOOR``, so it is invisible to
*both* truncation branches."""

_UNIFORM_SUB_EPSILON: Final[_Spread] = _Spread(head=(), tail=_DIFFUSE_MASS)
"""Every legal action at :data:`_DIFFUSE_MASS`: ``S_ε`` is **empty**, so §3.2
step 5's fallback fires and ``|S| == SUPPORT_MIN_K``."""

_PEAKED_OVER_A_TAIL: Final[_Spread] = _Spread(
    head=(0.50,) + (0.0125,) * 39, tail=_SUB_FLOOR_MASS
)
"""Forty actions above ``ε`` over a sub-``SUPPORT_FLOOR`` tail.

Forty rather than sixteen so ``SUPPORT_TOP_K`` actually *binds*: ``|S| == 16``
strictly below both the above-``ε`` count (40) and the nonzero count (the whole
legal set), and ``truncated_mass == 1 − (0.50 + 15 × 0.0125) == 0.3125 > 0`` --
the first fixture in this file on which truncation discards anything at all."""

_ALL_BELOW_FLOOR: Final[_Spread] = _Spread(head=(), tail=_SUB_FLOOR_MASS)
"""Every legal action inside R3's tail: **both** branches return an empty
support and the candidate is rejected rather than drawn from an empty array.

**These rows are deliberately un-normalised, and that is not an accident of the
fixture.** For ``S_ε`` to be empty every legal action must carry ``< 1e-3``, so a
*normalised* row would need more than 1000 legal actions; ``intro``'s branching
is ~229. The R3 fallback and ``rejected_empty_support`` are therefore only
reachable on an un-normalised row in this start mode -- recorded in the design's
rev-3.2 log. Nothing in the builder reads ``policy_target`` as a distribution
before it truncates (``_truncated_support`` thresholds raw values; ``_tempered``
renormalises over ``S`` afterwards), so the branch under test is the real one."""


# ---------------------------------------------------------------------------
# The synthetic shard tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlantedGame:
    """One source game as it will appear in the shards.

    Args:
        game_id: The ``game_id`` every row carries.
        actions: The legal line, indexed by **absolute** ply.
        supports: Per absolute ply, the action ids the row's ``policy_target``
            puts mass on -- always legal at that position, which is what makes
            the tempered draw replayable.
        written: The absolute plies whose rows are actually written. Anything
            other than ``range(len(actions))`` plants a booked source game (rows
            start at ``L``) or a holed one (a shard was skipped).
        shape: The ``policy_target`` masses, aligned with ``supports``. Ignored
            when `spread` is set.
        spread: A shape over the position's **whole** legal set, for the rows
            that have to leave the 16-action axis (F1, F2). When set,
            ``supports[ply]`` is every legal action id at that ply.
        recorded_offset: Added to every written ply before the row is labelled.
            Non-zero **only** for a game the ``min(ply)`` classification
            *excludes* -- :data:`_TRUNCATED_GAME_ID`, whose ``action`` and
            ``policy_target`` columns the builder therefore never reads
            (``_admit_games`` classifies it and ``continue``s). Relabelling
            rather than walking 43 plies of random play, because a 43-ply random
            line can adjudicate before it gets there and the resulting fixture
            failure would be about the engine, not about the rule under test.
            ``tests/campaign/unit/test_book_audit.py`` makes the same call for
            the same game, with rows it does not even walk an engine to produce.
    """

    game_id: int
    actions: tuple[int, ...]
    supports: tuple[tuple[int, ...], ...]
    written: tuple[int, ...]
    shape: tuple[float, ...]
    spread: _Spread | None = None
    recorded_offset: int = 0


def _synthesise(
    game_id: int,
    plies: int,
    seed: int,
    shape: Sequence[float],
    written: Sequence[int] | None = None,
    *,
    spread: _Spread | None = None,
    recorded_offset: int = 0,
) -> _PlantedGame:
    """Walk a fresh game forward by legal actions, recording what a row needs."""
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    supports: list[tuple[int, ...]] = []
    for _ply in range(plies):
        assert not state.is_terminal(), "a planted game ended before its budget"
        legal = np.nonzero(state.legal_mask())[0]
        if spread is None:
            assert legal.size >= _SUPPORT_WIDTH, (
                "the planted position has fewer than 16 legal actions, so the "
                "synthesised policy_target could not be supported on the legal set"
            )
            supports.append(tuple(int(action) for action in legal[:_SUPPORT_WIDTH]))
        else:
            assert legal.size >= _MIN_SPREAD_WIDTH, (
                f"the planted position has {legal.size} legal actions, below "
                f"{_MIN_SPREAD_WIDTH}: a spread row narrower than that puts the "
                "fixture back on the 16-action axis it exists to leave"
            )
            assert legal.size > len(spread.head), (
                "the spread's head covers every legal action, so the row has no "
                "tail and truncation would again discard nothing"
            )
            supports.append(tuple(int(action) for action in legal))
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return _PlantedGame(
        game_id=game_id,
        actions=tuple(actions),
        supports=tuple(supports),
        written=tuple(range(plies)) if written is None else tuple(written),
        shape=tuple(shape),
        spread=spread,
        recorded_offset=recorded_offset,
    )


def _row(game: _PlantedGame, ply: int, *, schema: Schema) -> dict[str, torch.Tensor]:
    """Return one schema-shaped row of `game` at source ply `ply`.

    The row is *labelled* ``ply + game.recorded_offset``; the content is always
    read at `ply`. The two are the same number for every game but the planted
    preemption -- see :attr:`_PlantedGame.recorded_offset`.
    """
    target = torch.zeros(num_actions(), dtype=torch.float32)
    support = game.supports[ply]
    if game.spread is None:
        for slot, mass in enumerate(game.shape):
            target[support[slot]] = mass
    else:
        head = game.spread.head
        for slot, action in enumerate(support):
            target[action] = head[slot] if slot < len(head) else game.spread.tail
    recorded = ply + game.recorded_offset
    row: dict[str, torch.Tensor] = {
        "observation": torch.zeros(Game.observation_size(), dtype=torch.float32),
        "policy_target": target,
        "wdl_target": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "mlh_target": torch.tensor(0.5, dtype=torch.float32),
        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(0, dtype=torch.int64),
        "to_play": torch.tensor(recorded % 2, dtype=torch.int64),
        "ply": torch.tensor(recorded, dtype=torch.int64),
        "game_id": torch.tensor(game.game_id, dtype=torch.int64),
        "sims_used": torch.tensor(32, dtype=torch.int64),
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        "final_control": torch.full((CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64),
        "action": torch.tensor(game.actions[ply], dtype=torch.int64),
        # KOM-005b's three columns. Distinguishable per row rather than
        # zero-filled: the book builder does not read them, so a zero-fill and
        # a dropped column would be indistinguishable here.
        "termination": torch.tensor(0, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [0.5, 0.25 - 0.5 * (recorded % 2), -0.75], dtype=torch.float32
        ),
        "draft_outcome": (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if recorded < 2
            else torch.zeros(3, dtype=torch.float32)
        ),
    }
    # A pre-bump fixture (#35) is written at a schema with no `action` field;
    # the writer refuses an extra key, so the row is projected onto the schema
    # it is being written under rather than special-cased at every call site.
    return {name: row[name] for name in schema.names}


def plant(
    root: Path,
    iteration: int,
    games: Iterable[_PlantedGame],
    *,
    worker: int = 0,
    schema: Schema = SCHEMA,
    samples_per_shard: int = _SAMPLES_PER_SHARD,
) -> None:
    """Write `games` into ``<root>/iter_<iteration>/shards/worker_<k>_r0``.

    `worker` exists so a fixture can control **which shard** a planted game
    lands in: the target pass visits shards in ascending order and stops at the
    cap, so a game planted at the end of a big tree may never be visited at all.
    A game whose rejection a test asserts on therefore gets its own worker
    directory, which sorts first and is visited first.
    """
    directory = root / f"iter_{iteration}" / "shards" / f"worker_{worker}_r0"
    # flush_every_seconds is pushed out of reach: the writer's *time* trigger
    # would otherwise seal a short shard on a slow machine, and the fixtures
    # here depend on how many rows a shard holds.
    with ShardWriter(
        directory,
        schema,
        samples_per_shard=samples_per_shard,
        flush_every_seconds=86_400.0,
    ) as writer:
        for game in games:
            for ply in game.written:
                writer.append(_row(game, ply, schema=schema))


def _ordinary_games(count: int, *, first_id: int = 0, seed: int = 0) -> list[_PlantedGame]:
    """Return `count` planted games, alternating the two ``policy_target`` shapes."""
    return [
        _synthesise(
            game_id=first_id + index,
            plies=_ORDINARY_PLIES,
            seed=seed + index,
            shape=_PEAKED if index % 2 == 0 else _FLAT,
        )
        for index in range(count)
    ]


def _build(
    root: Path, generation: int, *, temperature: float = book.TEMPERATURE
) -> Book:
    """Build a book over `root` and load it, failing loudly if none was written."""
    path = build_book(
        root,
        generation,
        20260905,
        temperature=temperature,
        cap=_CAP,
        min_entries=_MIN_ENTRIES,
    )
    assert path is not None, "the fixture was expected to yield a writable book"
    return load_book(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def main_plant(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[_PlantedGame]]:
    """A 200-game iteration, plus the three hygiene cases ``min(ply)`` separates.

    The hygiene cases share ``worker_0_r0`` and the 200 ordinary games get
    ``worker_1_r0``, so the hygiene shard is the **first** one the target pass
    visits. Planted at the end of the big directory they would sit past the
    shard the cap stops on, and #34's rejection would be unreachable rather than
    absent -- a test that passes for the wrong reason either way.

    All three values ``min(ply)`` can take are planted, because §4.3's rule is
    three-way and the builder's copy of it was pinned only two ways: ``0``
    (every ordinary game, mined), ``6`` (booked, skipped as normal operation)
    and ``37`` (truncated, skipped as data loss). See
    :data:`_TRUNCATED_FIRST_PLY`.
    """
    root = tmp_path_factory.mktemp("book-main")
    plant(
        root,
        0,
        [
            # Rows begin at ply 6, exactly as §4.3 says a booked game's do; it
            # has no ply-0 stem, so it must never be mined (#33).
            _synthesise(_BOOKED_GAME_ID, 12, 9001, _PEAKED, written=range(6, 12)),
            # Ply 4's row is missing, as a skipped shard would leave it. min(ply)
            # is still 0, so the game IS admitted -- the hole has to be caught
            # per candidate, at the {0…d} ⊆ P check (#34).
            _synthesise(
                _HOLED_GAME_ID,
                12,
                9002,
                _FLAT,
                written=(0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11),
            ),
            # Rows begin at ply 37 -- min(ply) > 0 and OUTSIDE [4, 12], which is
            # the third arm of the rule (F7). A preemption, not a booking.
            _synthesise(
                _TRUNCATED_GAME_ID,
                _ORDINARY_PLIES,
                9003,
                _PEAKED,
                recorded_offset=_TRUNCATED_FIRST_PLY,
            ),
        ],
        worker=0,
    )
    games = _ordinary_games(_ORDINARY_GAMES)
    plant(root, 0, games, worker=1)
    return root, games


@pytest.fixture(scope="module")
def main_tree(main_plant: tuple[Path, list[_PlantedGame]]) -> Path:
    """The campaign root of :func:`main_plant`."""
    return main_plant[0]


@pytest.fixture(scope="module")
def main_games(main_plant: tuple[Path, list[_PlantedGame]]) -> list[_PlantedGame]:
    """The planted ordinary lines, for by-ply action comparisons."""
    return main_plant[1]


@pytest.fixture(scope="module")
def book_tau2(main_tree: Path) -> Book:
    """The pre-registered build: τ = 2.0 over :func:`main_tree`, as ``gen_1``."""
    return _build(main_tree, 1)


@pytest.fixture(scope="module")
def book_tau1(main_tree: Path) -> Book:
    """The same tree at τ = 1.0 -- the identity #7a asserts.

    Built as ``gen_2`` so it cannot overwrite ``gen_1``: iteration 1 holds no
    shards, so the walk skips it and lands on the same iteration-0 sources.
    """
    return _build(main_tree, 2, temperature=1.0)


@pytest.fixture(scope="module")
def small_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 20-source-game iteration -- few enough that the quota must bind (#36a)."""
    root = tmp_path_factory.mktemp("book-small")
    plant(root, 0, _ordinary_games(20, seed=500))
    return root


@pytest.fixture(scope="module")
def small_book(small_tree: Path) -> Book:
    """A book over :func:`small_tree`."""
    return _build(small_tree, 1)


@pytest.fixture(scope="module")
def long_book(tmp_path_factory: pytest.TempPathFactory) -> Book:
    """A book over :data:`_LONG_GAMES` games of :data:`_LONG_PLIES` plies each.

    Its own tree rather than more games in :func:`main_plant`, for two reasons:
    the long games would dilute the hygiene shard #34 depends on being visited
    first, and a 15-ply game offers nine eligible deviations where an ordinary
    one offers two -- which is exactly the draw pressure ``d = 11`` needs and
    exactly the wrong thing to add to the tree #36b's ratio is measured on.
    """
    root = tmp_path_factory.mktemp("book-long")
    plant(
        root,
        0,
        [
            _synthesise(
                game_id=2000 + index,
                plies=_LONG_PLIES,
                seed=2000 + index,
                shape=_PEAKED if index % 2 == 0 else _FLAT,
            )
            for index in range(_LONG_GAMES)
        ],
    )
    return _build(root, 1)


@pytest.fixture(scope="module")
def diffuse_plant(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Book, dict[int, _PlantedGame]]:
    """A book whose first shard carries the three rows §3.2's branches need.

    One row per branch of :func:`~komugi.selfplay.book._truncated_support`, all
    supported on the position's whole legal set rather than on 16 actions:

    * :data:`_PEAKED_OVER_A_TAIL` -- the **normal** branch, with ``k`` binding;
    * :data:`_UNIFORM_SUB_EPSILON` -- the **fallback** branch (R3);
    * :data:`_ALL_BELOW_FLOOR` -- **neither**, i.e. ``rejected_empty_support``.

    They share ``worker_0_r0`` so their shard is visited first, and 20 ordinary
    games in ``worker_1_r0`` carry the book past ``min_entries`` -- a book that
    is not written cannot be read, and every assertion here is a manifest read.

    Returned with the planted games so a node can compare ``support_size``
    against the row's *nonzero* count, which is what "truncation happened" means
    and which the manifest does not carry.
    """
    root = tmp_path_factory.mktemp("book-diffuse")
    special = [
        _synthesise(
            _DIFFUSE_GAME_ID, _ORDINARY_PLIES, 2001, (), spread=_UNIFORM_SUB_EPSILON
        ),
        _synthesise(
            _PEAKED_TAIL_GAME_ID, _ORDINARY_PLIES, 2002, (), spread=_PEAKED_OVER_A_TAIL
        ),
        _synthesise(
            _BELOW_FLOOR_GAME_ID, _ORDINARY_PLIES, 2003, (), spread=_ALL_BELOW_FLOOR
        ),
    ]
    plant(root, 0, special, worker=0)
    plant(root, 0, _ordinary_games(20, seed=2100), worker=1)
    return _build(root, 1), {game.game_id: game for game in special}


@pytest.fixture(scope="module")
def gapped_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Iterations 0, 1, 2, 4, 5 hold data; iteration 3 holds none."""
    root = tmp_path_factory.mktemp("book-gapped")
    for iteration in (0, 1, 2, 4, 5):
        plant(
            root,
            iteration,
            _ordinary_games(6, first_id=iteration * 100, seed=1000 + iteration * 10),
        )
    return root


# ---------------------------------------------------------------------------
# #1-#3, #12 -- the artefact and where it comes from
# ---------------------------------------------------------------------------


def test_a_built_book_has_the_manifest_fields(main_tree: Path, book_tau2: Book) -> None:
    """§2.2's manifest, parsed, with every field a consumer or an audit reads."""
    raw = json.loads(book_path_for(main_tree, 1).read_text(encoding="utf-8"))
    for field in (
        "book_version",
        "generation",
        "source_iterations",
        "gungi_mode",
        "action_size",
        "plane_map_version",
        "source_schema_fingerprint",
        "temperature",
        "cap",
        "min_plies",
        "max_plies",
        "build_seed",
        "entries",
        "stats",
    ):
        assert field in raw, f"the manifest is missing {field!r}"
    assert book_tau2.generation == 1
    assert book_tau2.action_size == num_actions()
    assert book_tau2.plane_map_version == PLANE_MAP_VERSION
    assert book_tau2.source_schema_fingerprint == SCHEMA.fingerprint
    assert book_tau2.bit_generator == "PCG64"
    assert book_tau2.entries, "a book with no entries should not have been written"


def test_the_book_lives_under_the_campaign_root_per_generation(
    tmp_path: Path,
) -> None:
    """``<root>/books/gen_<i>.json``, and nothing else is written."""
    assert book_path_for(tmp_path, 7) == tmp_path / "books" / "gen_7.json"
    plant(tmp_path, 6, _ordinary_games(12, seed=700))
    path = build_book(tmp_path, 7, 4242, cap=_CAP, min_entries=_MIN_ENTRIES)
    assert path == book_path_for(tmp_path, 7)
    assert sorted(item.name for item in (tmp_path / "books").iterdir()) == [
        "gen_7.json"
    ]


def test_iteration_zero_builds_no_book(tmp_path: Path) -> None:
    """No source iteration exists below 0, so there is nothing to mine."""
    assert build_book(tmp_path, 0, 1) is None
    assert not (tmp_path / "books").exists()


def test_the_builder_reads_only_the_campaign_root(tmp_path: Path) -> None:
    """The mechanical half of "no human lines": there is no path to hand one in."""
    parameters = inspect.signature(build_book).parameters
    path_like = [
        name
        for name in parameters
        if any(token in name for token in ("path", "dir", "file", "root"))
    ]
    assert path_like == ["root"], (
        "build_book grew a second path argument; an external opening line could "
        f"then enter the training data through it: {path_like}"
    )

    plant(tmp_path, 0, _ordinary_games(12, seed=800))
    clean = build_book(tmp_path, 1, 31337, cap=_CAP, min_entries=_MIN_ENTRIES)
    assert clean is not None
    expected = clean.read_bytes()

    (tmp_path / "human_theory.sgf").write_text("(;GM[1])\n", encoding="utf-8")
    (tmp_path / "openings.json").write_text('{"lines": []}\n', encoding="utf-8")
    (tmp_path / "iter_0" / "notes.txt").write_text("not a shard\n", encoding="utf-8")
    again = build_book(tmp_path, 1, 31337, cap=_CAP, min_entries=_MIN_ENTRIES)
    assert again is not None
    assert again.read_bytes() == expected, (
        "a stray .sgf/.json/.txt beside the shards changed the book, so the "
        "builder is reading something other than iter_*/shards"
    )


# ---------------------------------------------------------------------------
# #4, #8, #13 -- entry shape, dedup, and the floor
# ---------------------------------------------------------------------------


def test_every_entry_is_four_to_twelve_plies(book_tau2: Book, long_book: Book) -> None:
    """The ticket's ``L = d + 1 ∈ [4, 12]``, and the claim matches the payload.

    Both bounds are **reached**, not merely respected -- which is the half the
    impl review found missing (F3). The lower bound was always exercised: every
    6-ply fixture game offers ``d = 3``. The upper bound was not reachable by any
    fixture in the suite, so widening the eligible range to
    ``range(MIN_PLIES − 1, MAX_PLIES + 1)`` (mutant M4) added only ``d = 12``,
    which needs a 14-ply source game and had none. :func:`long_book` supplies
    them, and the ``MAX_PLIES in lengths`` assertion is what goes red under the
    mutation's *other* direction -- and what makes ``load_book``'s own
    ``plies <= MAX_PLIES`` refusal a check on a value the builder can produce.
    """
    for book_file in (book_tau2, long_book):
        assert len(book_file.entries) >= _MIN_ENTRIES
        for entry in book_file.entries:
            assert MIN_PLIES <= entry.plies <= MAX_PLIES
            assert len(entry.actions) == entry.plies
            assert entry.plies == entry.deviation_ply + 1

    assert MIN_PLIES in {entry.plies for entry in book_tau2.entries}
    lengths = {entry.plies for entry in long_book.entries}
    assert MAX_PLIES in lengths, (
        f"a book over {_LONG_GAMES} games of {_LONG_PLIES} plies produced "
        f"lengths {sorted(lengths)} and never reached {MAX_PLIES}: the upper "
        "prefix bound is still unexercised, so an off-by-one in the eligible "
        "range would go unnoticed"
    )


def test_a_thirteen_ply_entry_is_refused_at_load(
    main_tree: Path, tmp_path: Path
) -> None:
    """A hand-edited entry outside ``[MIN_PLIES, MAX_PLIES]`` fails at *load*.

    The other half of F3: :func:`test_every_entry_is_four_to_twelve_plies` pins
    what the *builder* produces, and this pins that a file which claims
    otherwise -- a hand edit, a book from a future range, a truncated write --
    is refused before any worker replays it, naming the offending number and the
    range it is outside.
    """
    payload = json.loads(book_path_for(main_tree, 1).read_text(encoding="utf-8"))
    entry = dict(payload["entries"][0])
    actions = list(entry["actions"])
    entry["actions"] = actions + [actions[0]] * (MAX_PLIES + 1 - len(actions))
    entry["plies"] = MAX_PLIES + 1
    payload["entries"] = [entry]

    path = tmp_path / "thirteen_plies.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        load_book(path)
    message = str(raised.value)
    assert str(MAX_PLIES + 1) in message
    assert f"[{MIN_PLIES}, {MAX_PLIES}]" in message


def test_a_future_book_version_is_refused(main_tree: Path, tmp_path: Path) -> None:
    """A v2 file is refused by a v1 reader, naming both versions (F10).

    ``BOOK_VERSION``'s docstring says it is bumped when a *reader* must change,
    which is only true if a reader looks: until this node, ``load_book`` parsed
    the field into :class:`Book` and never compared it, so a future file would
    be read under today's rules and every other check -- action space, mode,
    entry bounds -- would pass while answering the wrong question.

    The refusal is asserted to come **first**, on a file that is wrong in two
    ways at once: a v2 book whose ``action_size`` is also foreign must report the
    version, because the version is what says whether ``action_size`` still
    means what this reader thinks it means.
    """
    payload = json.loads(book_path_for(main_tree, 1).read_text(encoding="utf-8"))
    payload["book_version"] = book.BOOK_VERSION + 1
    path = tmp_path / "future_version.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        load_book(path)
    message = str(raised.value)
    assert f"book_version {book.BOOK_VERSION + 1}" in message
    assert f"book_version {book.BOOK_VERSION}" in message

    payload["action_size"] = num_actions() + 1
    both_wrong = tmp_path / "future_version_foreign_actions.json"
    both_wrong.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised_first:
        load_book(both_wrong)
    assert "book_version" in str(raised_first.value), (
        "a v2 book with a foreign action space reported the action space: the "
        "version check has to come first, because it is what says whether the "
        "rest of the manifest means what this reader reads it as"
    )


def test_duplicate_prefixes_are_deduped_and_counted(small_book: Book) -> None:
    """Twenty source games and a 4x draw budget collide; the book must not."""
    tuples = [entry.actions for entry in small_book.entries]
    assert len(set(tuples)) == len(tuples), "the same prefix was written twice"
    assert small_book.stats["rejected_duplicate"] > 0, (
        "a 20-game tree drawn 4x over cannot have produced no collision; the "
        "dedup set is not being consulted"
    )


def test_a_book_below_the_minimum_entry_count_is_not_written(tmp_path: Path) -> None:
    """Two source games cannot clear the quota-bounded floor, so no file appears."""
    plant(tmp_path, 0, _ordinary_games(2, seed=900))
    assert build_book(tmp_path, 1, 5, cap=_CAP, min_entries=_MIN_ENTRIES) is None
    assert not book_path_for(tmp_path, 1).exists()


# ---------------------------------------------------------------------------
# #7a/#7b/#7c -- what tau actually does
# ---------------------------------------------------------------------------


def test_the_flattening_ratio_is_zero_at_tau_one(book_tau1: Book) -> None:
    """``p_τ = p_S`` at τ = 1, for **every** input shape -- the fixture cannot rig it."""
    ratios = [entry.flattening_ratio for entry in book_tau1.entries]
    assert ratios, "no entry carried a ratio"
    for ratio in ratios:
        assert ratio is not None
        assert ratio == pytest.approx(0.0, abs=1e-9)
    assert book_tau1.stats["median_flattening_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_the_tempered_draw_never_leaves_the_truncated_support(book_tau2: Book) -> None:
    """Truncate-then-temper, asserted structurally rather than as an inequality.

    ``tail_mass == 0`` **exactly** goes red the moment an implementer drops the
    truncation: with ``policy_target`` nonzero on every legal action, an
    un-truncated τ = 2 sampler puts ~0.30 of its mass below ``1e-4`` (§3.2's
    arithmetic). The old ``≤ 0.10`` ceiling would not have caught it.
    """
    for entry in book_tau2.entries:
        assert entry.drawn_target_prob >= book_tau2.support_epsilon
        assert entry.support_size <= book_tau2.support_top_k
        assert not entry.support_fallback
    assert book_tau2.stats["tail_mass"] == 0.0


def test_a_diffuse_target_is_truncated_before_it_is_tempered(
    diffuse_plant: tuple[Book, dict[int, _PlantedGame]],
) -> None:
    """``|S| == SUPPORT_TOP_K`` on a row where the top-``k`` is a real choice (F1).

    The node above asserts the *same* three properties, and on its fixture all
    three are vacuous: every planted ``policy_target`` there is nonzero on
    exactly 16 actions, all ≥ 0.0115, so ``{a : p[a] > 0}``,
    ``{a : p[a] ≥ SUPPORT_EPSILON}`` and the top-16 coincide and the book is
    byte-identical with the truncation deleted.

    :data:`_PEAKED_OVER_A_TAIL` separates them: 40 actions clear ``ε`` and the
    whole rest of the legal set carries :data:`_SUB_FLOOR_MASS`. So

    * ``support_size == 16`` is now strictly below **both** the above-``ε`` count
      (40) and the nonzero count (the legal set) -- red against
      ``SUPPORT_TOP_K = 64`` (mutant M3), which returns all 40 and which the old
      ``support_size <= book.support_top_k`` could not catch because it read the
      mutated constant back out of the manifest it was checking;
    * ``truncated_mass > 0`` -- the first fixture on which truncation discards
      anything, so the field stops being a constant zero and starts being a
      measurement.

    Asserted against a literal 16 rather than ``book.support_top_k`` for exactly
    the reason the review gives: the manifest's copy of a constant cannot
    discriminate a change to that constant.
    """
    diffuse_book, planted = diffuse_plant
    game = planted[_PEAKED_TAIL_GAME_ID]
    entries = [
        entry
        for entry in diffuse_book.entries
        if entry.source_game_id == _PEAKED_TAIL_GAME_ID
    ]
    assert entries, "the peaked-over-a-tail row produced no entry to read"
    above_epsilon = len(_PEAKED_OVER_A_TAIL.head)
    for entry in entries:
        nonzero = len(game.supports[entry.deviation_ply])
        assert _SUPPORT_WIDTH < above_epsilon < nonzero, (
            "the fixture no longer separates the three counts, so |S| == 16 "
            "would be satisfiable without any truncation happening"
        )
        assert entry.support_size == _SUPPORT_WIDTH, (
            f"|S| is {entry.support_size} on a row with {above_epsilon} actions "
            f"above epsilon and {nonzero} nonzero: the top-k is not binding"
        )
        assert not entry.support_fallback
        assert entry.truncated_mass > 0.0, (
            "truncation discarded no mass on a row whose tail is 3/4 of the "
            "above-epsilon head; S is not a truncation of anything"
        )
        assert entry.drawn_target_prob >= book.SUPPORT_EPSILON


def test_a_target_below_epsilon_falls_back_to_the_floor_support(
    diffuse_plant: tuple[Book, dict[int, _PlantedGame]],
) -> None:
    """R3's fallback branch, exercised rather than asserted absent (F2).

    Before this row, ``support_fallback`` was only ever asserted **False** and
    ``rejected_empty_support`` only ever appeared as a ``0`` inside the partition
    sum -- so the entire rev-3 R3 branch (:data:`SUPPORT_FLOOR`,
    :data:`SUPPORT_MIN_K`) shipped with zero behavioural coverage.

    :data:`_UNIFORM_SUB_EPSILON` puts every legal action at
    :data:`_DIFFUSE_MASS`, below ``ε`` and above the floor, so ``S_ε`` is empty
    and §3.2 step 5 must fire. This is the node that kills the two mutants #7b
    was written for and cannot see:

    * **M1** (``chosen = np.nonzero(target > 0.0)`` instead of ``>= ε``): the
      ``chosen.size == 0`` test never trips, so ``fallback`` stays ``False`` and
      ``limit`` stays ``SUPPORT_TOP_K`` -- ``support_fallback`` is ``False`` and
      ``support_size`` is 16, and both assertions below go red;
    * **M2** (truncate *after* tempering): ``p^(1/2)`` lifts 9.9e-4 to 0.031,
      far above ``ε``, so ``S_ε`` is no longer empty and the same two assertions
      go red the same way. That is B2 -- the design's most-argued decision --
      finally pinned by its consequence rather than by its restatement.
    """
    diffuse_book, _planted = diffuse_plant
    entries = [
        entry
        for entry in diffuse_book.entries
        if entry.source_game_id == _DIFFUSE_GAME_ID
    ]
    assert entries, "the sub-epsilon row produced no entry to read"
    for entry in entries:
        assert entry.support_fallback is True, (
            "a row on which no action clears epsilon did not take the fallback "
            "branch, so either the epsilon test or the ordering of truncation "
            "and tempering is not what §3.2 says"
        )
        assert entry.support_size == book.SUPPORT_MIN_K
        assert entry.support_size == 2
        assert book.SUPPORT_FLOOR <= entry.drawn_target_prob < book.SUPPORT_EPSILON
    assert diffuse_book.stats["support_fallback"] >= len(entries)
    # R3's identity survives the fallback branch too: SUPPORT_FLOOR *is* the
    # tail threshold, so every drawn action is out of the tail by construction.
    assert diffuse_book.stats["tail_mass"] == 0.0


def test_a_target_below_the_floor_rejects_the_candidate(
    diffuse_plant: tuple[Book, dict[int, _PlantedGame]],
) -> None:
    """Neither branch finds a support, so the candidate is counted, not drawn.

    The other half of F2. ``_truncated_support`` returns an empty array when
    nothing clears :data:`SUPPORT_FLOOR` either, and the caller must reject --
    drawing from an empty array is the failure this return exists to prevent,
    and ``rejected_empty_support`` is where the draw has to land for the R7
    partition to still hold.
    """
    diffuse_book, _planted = diffuse_plant
    assert diffuse_book.stats["rejected_empty_support"] >= 1, (
        "a row whose entire mass is below SUPPORT_FLOOR produced no "
        "rejected_empty_support, so that bucket is still unreachable"
    )
    assert all(
        entry.source_game_id != _BELOW_FLOOR_GAME_ID for entry in diffuse_book.entries
    ), "a row with no action above the floor was mined into an entry anyway"


def test_a_prefix_that_ends_the_game_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.2 step 6, on a fixture that can actually falsify it (F5).

    ``test_no_entry_ends_on_a_terminal_position`` is vacuous as written: every
    prefix the suite builds is ≤ 11 plies from a bare ``Game()`` against the
    engine's adjudication cap, so ``rejected_terminal`` is 0 in every book and
    deleting the check outright (mutant M5) changes nothing.

    Shrinking the cap to :data:`_SHORT_TURN_CAP` makes both verdicts appear in
    one book, deterministically -- no draw decides which:

    * ``d = 3`` -- the replay reaches 3 plies, the tempered action makes 4, and
      ``4 < 5`` is live. Accepted, at exactly ``MIN_PLIES`` plies.
    * ``d = 4`` -- the replay reaches 4 plies, the tempered action makes 5, and
      the engine adjudicates a draw at 5. Rejected, in ``rejected_terminal``.

    So under M5 the ``d = 4`` candidates become 5-ply entries: the counter drops
    to zero *and* a length appears that the correct builder cannot produce.
    Both are asserted, because the second survives a mutant that miscounts.

    ``book.Game`` is what is patched, not the module's own: ``_draw_candidate``
    constructs ``GameState(Game(), _REPLAY_DRAW_SCORE)`` off the builder's
    module global, while the fixture's rows are walked through the real engine
    before the patch is applied.
    """
    plant(tmp_path, 0, _ordinary_games(20, seed=2200))
    monkeypatch.setattr(book, "Game", _ShortGame)
    built = _build(tmp_path, 1)

    assert built.stats["rejected_terminal"] >= 1, (
        "no candidate ended the game on a 5-turn cap, where every d = 4 "
        "candidate must: the non-terminality check after the tempered move is "
        "not being reached"
    )
    assert {entry.plies for entry in built.entries} == {MIN_PLIES}, (
        "an entry longer than 4 plies survived a 5-turn cap, so a prefix that "
        "adjudicates the game was written into the book as an opening"
    )


def test_the_flattening_ratio_is_in_the_registered_band_at_tau_two(
    book_tau2: Book,
) -> None:
    """The pre-registered band, and the scale-freeness the band rests on."""
    median = book_tau2.stats["median_flattening_ratio"]
    assert median is not None
    assert 0.50 <= median <= 0.95

    ratios = {entry.flattening_ratio for entry in book_tau2.entries}
    assert None not in ratios, "a 16-action support cannot have zero headroom"
    values = sorted(float(ratio) for ratio in ratios if ratio is not None)
    assert len(values) >= 2, (
        "the fixture plants a peaked and a near-uniform policy_target, so two "
        "distinct ratios must appear; one means only one shape was mined"
    )
    assert values[-1] - values[0] <= 0.10, (
        "the flattening ratio is supposed to be scale-free in the input's "
        f"sharpness over the retained range; it spread {values}"
    )


# ---------------------------------------------------------------------------
# #10, #11 -- which iterations are sources
# ---------------------------------------------------------------------------


def test_no_entry_comes_from_the_current_or_a_future_iteration(
    gapped_tree: Path,
) -> None:
    """§1.8's newest-first walk over iterations *with data*, not a contiguous range."""
    book_file = _build(gapped_tree, 6)
    assert book_file.source_iterations == (0, 1, 2, 4, 5), (
        "the contiguous rule would answer (1, 2, 3, 4, 5); the campaign's rule "
        "skips the data-less iteration without spending the budget on it"
    )
    for entry in book_file.entries:
        assert entry.source_iteration < book_file.generation
        assert entry.source_iteration in book_file.source_iterations


def test_the_source_window_follows_the_recency_window_size(
    gapped_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derivation, not a literal: shrink the recency window and the sources follow."""
    monkeypatch.setattr(book, "WindowConfig", lambda: WindowConfig(size=2))
    book_file = _build(gapped_tree, 6)
    assert book_file.source_iterations == (4, 5)


# ---------------------------------------------------------------------------
# #19 -- what a book refuses to be loaded as
# ---------------------------------------------------------------------------


def test_a_book_from_another_action_space_or_mode_is_refused(
    main_tree: Path, tmp_path: Path
) -> None:
    """Both refusals name **both** values, because "the book is wrong" is not a fix."""
    payload = json.loads(book_path_for(main_tree, 1).read_text(encoding="utf-8"))

    payload["action_size"] = num_actions() + 1
    foreign_actions = tmp_path / "foreign_actions.json"
    foreign_actions.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        load_book(foreign_actions)
    assert str(num_actions() + 1) in str(raised.value)
    assert str(num_actions()) in str(raised.value)

    payload["action_size"] = num_actions()
    payload["gungi_mode"] = "not-a-mode"
    foreign_mode = tmp_path / "foreign_mode.json"
    foreign_mode.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as raised_mode:
        load_book(foreign_mode)
    assert "not-a-mode" in str(raised_mode.value)
    assert book.running_gungi_mode() in str(raised_mode.value)


def test_a_book_from_another_plane_map_is_refused(
    main_tree: Path, tmp_path: Path
) -> None:
    """KOM-165 review F7: the version was stamped and read, but never refused.

    A book is action-id prefixes, so a cross-generation book is not *unsafe* the
    way a cross-generation shard is -- which is exactly why nothing refused it,
    and why a campaign could quietly mine a fresh root with a book built from
    the previous plane map's shards. The stamp is the only field that says which
    data generation the prefixes were selected under, so this is the only place
    the mistake can be caught by name instead of by an operator noticing.

    Both versions are named for the same reason every other refusal here names
    both: the fix is "rebuild the book from this root", and that is only obvious
    once the message says which root the file came from.

    Deleting the check in ``load_book`` fires the ``pytest.raises`` below.
    """
    payload = json.loads(book_path_for(main_tree, 1).read_text(encoding="utf-8"))
    assert int(payload["plane_map_version"]) == PLANE_MAP_VERSION, (
        "premise: the builder stamps the running plane map"
    )

    stale = PLANE_MAP_VERSION - 1
    payload["plane_map_version"] = stale
    foreign_map = tmp_path / "foreign_plane_map.json"
    foreign_map.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_book(foreign_map)
    message = str(raised.value)
    assert str(stale) in message
    assert str(PLANE_MAP_VERSION) in message
    assert "plane map version" in message


# ---------------------------------------------------------------------------
# #33, #34, #35 -- source hygiene
# ---------------------------------------------------------------------------


def test_a_booked_source_game_is_never_mined(
    book_tau2: Book, main_games: list[_PlantedGame]
) -> None:
    """A source game whose own rows start at ply 6 has no ply-0 stem to mine.

    Extended for F7 with the **third** arm of §4.3's ``min(ply)`` rule. The
    audit's copy of that rule is pinned by its own ply-37 game (#17); the
    builder's copy was not, because no fixture here planted a source game with
    ``min(ply) > 0`` *outside* ``[4, 12]``. Collapsing ``_admit_games``'
    three-way classification into "booked if ``min(ply) != 0``" (mutant M9)
    therefore changed only counters that were asserted ``>= 1`` and ``== 0``,
    and passed.

    The two counts are asserted **exactly**, and against different games, which
    is what makes them a partition rather than two thresholds: under the
    collapse the booked count becomes 2 and the truncated count 0, and both
    lines go red. They are different *meanings* as well as different numbers --
    a booked source game is normal operation from generation 2 onward, a
    truncated one is data loss and an operator should see it as such.
    """
    assert book_tau2.stats["skipped_booked_source_games"] == 1
    assert book_tau2.stats["skipped_truncated_source_games"] == 1, (
        f"the game whose rows start at ply {_TRUNCATED_FIRST_PLY} is a "
        "preemption, not a booking; a builder classifying min(ply) > 0 as "
        "booked would have counted it in the other bucket"
    )
    assert all(entry.source_game_id != _BOOKED_GAME_ID for entry in book_tau2.entries)
    assert all(
        entry.source_game_id != _TRUNCATED_GAME_ID for entry in book_tau2.entries
    )

    by_id = {game.game_id: game for game in main_games}
    checked = 0
    for entry in book_tau2.entries:
        game = by_id.get(entry.source_game_id)
        if game is None:
            continue
        checked += 1
        # Every action but the last is the one RECORDED at that ply -- looked up
        # by ply value, which is the property positional indexing would break.
        assert entry.actions[:-1] == game.actions[: entry.plies - 1]
    assert checked > 0, "no entry came from a game whose line the test knows"


def test_a_game_with_a_hole_in_its_plies_is_rejected(book_tau2: Book) -> None:
    """A missing ply-4 row must not be spliced over; only ``d == 3`` survives it."""
    assert book_tau2.stats["rejected_truncated_prefix"] >= 1
    holed = [
        entry for entry in book_tau2.entries if entry.source_game_id == _HOLED_GAME_ID
    ]
    for entry in holed:
        assert entry.deviation_ply == 3, (
            "the prefix for d >= 4 would have to cross the missing ply-4 row"
        )
    # Three counters, three meanings: a per-draw hole, a per-game booked source,
    # a per-game data loss. They are asserted apart because rev 2 incremented
    # one name in two units (N2, B1 residual 2). The holed game contributes to
    # the FIRST only -- its min(ply) is 0, so it is admitted and mined, and the
    # per-game truncation count belongs to the ply-37 game alone.
    assert book_tau2.stats["skipped_truncated_source_games"] == 1


def test_an_iteration_without_the_action_field_is_not_a_source(
    tmp_path: Path,
) -> None:
    """A pre-bump iteration is skipped by name, and the walk continues past it.

    Two shapes of straddle, because R11 says **the guard qualifies shards, not
    iterations** and only one of them can tell the two rules apart:

    * iteration 1 is *wholly* pre-bump -- skipped with its fingerprint, and the
      walk continues below it rather than spending window budget on it;
    * iteration 2 is **mixed** -- one worker directory at the current schema and
      one at the old, holding rows of the *same* source game. This is the case
      the impl review found untested (F6): with no mixed fixture anywhere,
      ``_walk_sources``' ``shards_skipped`` branch was dead code and a
      per-*iteration* guard (mutant M6) passed everything.

    Under the per-shard rule iteration 2 still qualifies on its good shards, the
    skip is *reported* with a count rather than inferred from a quietly smaller
    window, and the foreign shard's rows are simply absent -- which makes the
    affected game's ply set partial and lands its ``d >= 5`` candidates in
    ``rejected_truncated_prefix`` by the existing rule, no new mechanism. Under
    the per-iteration rule ``source_iterations`` loses 2 outright, which is the
    first assertion below.

    The mixed game gets ``worker_0_r0`` to itself so its shard is a pool of one:
    every draw on that shard is a draw on this game, which is what turns "the
    truncated-prefix rejection is reachable" into "it happens ~150 times".

    Sliced **by name**, not by ``[:-1]``. The guard under test is fingerprint
    equality, so a position slice would keep passing after KOM-005b's append --
    but it would have stopped reconstructing "a schema without ``action``" and
    started reconstructing "a schema without ``draft_outcome``", which is a
    different fixture wearing this test's name. The subject is unchanged; only
    the slice is.
    """
    pre_bump = Schema(fields=SCHEMA.fields[: SCHEMA.names.index("action")])
    assert pre_bump.fingerprint != SCHEMA.fingerprint
    plant(tmp_path, 0, _ordinary_games(8, first_id=0, seed=1200))
    plant(
        tmp_path,
        1,
        _ordinary_games(8, first_id=100, seed=1300),
        schema=pre_bump,
    )
    # One game, one line (same seed), split across two fingerprints: plies
    # 0-3 and 5-11 survive the guard, ply 4 does not.
    plant(
        tmp_path,
        2,
        [
            _synthesise(
                _MIXED_GAME_ID,
                12,
                1350,
                _FLAT,
                written=(0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11),
            )
        ],
        worker=0,
    )
    plant(
        tmp_path,
        2,
        [_synthesise(_MIXED_GAME_ID, 12, 1350, _FLAT, written=(4,))],
        worker=1,
        schema=pre_bump,
    )
    plant(tmp_path, 2, _ordinary_games(8, first_id=200, seed=1400), worker=2)

    built = _build(tmp_path, 3)
    assert built.source_iterations == (0, 2), (
        "a mixed-fingerprint iteration was dropped entirely; the guard "
        "qualifies shards, not iterations (R11), so iteration 2's good shards "
        "are still sources"
    )
    skipped = built.stats["skipped_source_iterations"]
    by_iteration = {int(item["iteration"]): item for item in skipped}
    assert sorted(by_iteration) == [1, 2]
    assert by_iteration[1]["reason"] == "schema"
    assert by_iteration[1]["fingerprint"] == pre_bump.fingerprint
    assert all(entry.source_iteration != 1 for entry in built.entries)

    assert by_iteration[2] == {
        "iteration": 2,
        "reason": "schema",
        "shards_skipped": 1,
    }, (
        "the mixed iteration must be reported with a shard COUNT and no "
        "fingerprint: it did not fail the guard, part of it did"
    )
    assert built.stats["rejected_truncated_prefix"] >= 1, (
        "the foreign shard's ply-4 row is missing from the index, so every "
        "d >= 5 candidate of that game must cross a hole and be rejected"
    )
    for entry in built.entries:
        if entry.source_game_id == _MIXED_GAME_ID:
            assert entry.deviation_ply == 3, (
                "a prefix for d >= 4 would have to splice across the ply the "
                "foreign shard carried"
            )


def test_a_window_that_is_entirely_pre_bump_writes_no_book(tmp_path: Path) -> None:
    """The companion case: a straddle thin enough to leave the generation bookless.

    Sliced by name for the reason the node above gives: ``[:-1]`` would still
    pass after KOM-005b's append while quietly reconstructing a different
    pre-bump generation than the one this test is named for.
    """
    pre_bump = Schema(fields=SCHEMA.fields[: SCHEMA.names.index("action")])
    plant(tmp_path, 0, _ordinary_games(8, seed=1500), schema=pre_bump)
    assert build_book(tmp_path, 1, 77, cap=_CAP, min_entries=_MIN_ENTRIES) is None
    assert not book_path_for(tmp_path, 1).exists()


# ---------------------------------------------------------------------------
# #36a, #36b -- the quota, and the metric the failure mode does not maximise
# ---------------------------------------------------------------------------


def test_entries_are_capped_per_source_game(small_book: Book) -> None:
    """At most 8 entries per ``(source_iteration, source_game_id)`` (§3.3, N1)."""
    counts: dict[tuple[int, int], int] = {}
    for entry in small_book.entries:
        key = (entry.source_iteration, entry.source_game_id)
        counts[key] = counts.get(key, 0) + 1
    assert counts, "the small tree produced no entries"
    assert max(counts.values()) <= ENTRIES_PER_SOURCE_GAME
    assert small_book.stats["rejected_stem_quota"] > 0, (
        "twenty source games drawn 4x over must have hit the quota; if they did "
        "not, the quota is not being enforced"
    )


def test_stems_are_counted_against_a_tail_mining_sampler(book_tau2: Book) -> None:
    """The gate is on **stems**, which mining the deviation ply does not move."""
    stems = book_tau2.stats["distinct_stems"]
    entries = book_tau2.stats["distinct_entries"]
    assert entries >= _MIN_ENTRIES
    assert stems / entries >= 0.25, (
        f"{stems} stems over {entries} entries: the book is a handful of lines "
        "wearing 1024 one-ply variants"
    )
    assert book_tau2.stats["distinct_source_games"] >= 1


# ---------------------------------------------------------------------------
# The stats partition and the derived budget
# ---------------------------------------------------------------------------


def test_the_rejection_buckets_partition_the_candidates(book_tau2: Book) -> None:
    """§2.2's R7 invariant, read off the manifest a build already asserted."""
    stats = book_tau2.stats
    total = (
        stats["distinct_entries"]
        + stats["rejected_illegal"]
        + stats["rejected_terminal"]
        + stats["rejected_truncated_prefix"]
        + stats["rejected_duplicate"]
        + stats["rejected_stem_quota"]
        + stats["rejected_empty_support"]
    )
    assert total == stats["candidates_drawn"]
    # The per-GAME buckets sum to no draw and appear in no partition: a
    # data-loss signal and a sampler-accounting signal must not be added.
    assert stats["skipped_booked_source_games"] >= 1


def test_the_shard_budget_is_derived_from_the_quota_it_must_feed() -> None:
    """§2.4 R5: ``⌈(cap / entries_per_source_game) × ply_cap / samples_per_shard⌉``.

    The ply cap in that formula is the ENGINE's cap, read from
    ``Game.max_turns()`` rather than restated, so KOM-165's 256 → 512 re-derives
    the budget (32 → 64) instead of leaving the book mining a stale number of
    shards. The value is pinned too: a derivation that silently answered
    something other than 64 at the current cap would be a budget change nobody
    asked for.
    """
    assert TARGET_SHARD_BUDGET == 64
    assert book.TARGET_SHARD_BUDGET == -(
        -(book.CAP // book.ENTRIES_PER_SOURCE_GAME * Game.max_turns()) // 1024
    )
