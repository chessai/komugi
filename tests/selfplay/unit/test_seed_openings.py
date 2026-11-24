"""Seeded self-play openings: the file, the draw and the replay (KOM-254 A1).

The mechanism lands **inert**, so the node that makes "inert" falsifiable rather
than asserted is the first one here: at ``seed_openings_prob = 0.0`` with a
valid file configured, a worker writes rows that are **byte-identical** to the
same worker with no seed config at all. Everything else in this module is one of
the two things that claim rests on -- a dedicated seed stream (so the draw
cannot perturb the full/fast sequence or any per-move Gumbel vector) and a
loader that refuses every file this process cannot replay faithfully.

The refusals are not a checklist: each one is a failure the design review
(``docs/reviews/kom-254-design-review.md``) showed could otherwise reach a
shard. A prefix terminal at its cut writes **zero rows and reports success**
(F1). A prefix lifted from a booked source game replays *legally* from ply 0
into a different position, so the curriculum trains on plateaus that never
occurred (F2) -- which is why ``start_ply`` is mandatory and must be ``0``. An
action id means nothing across an encoding change and a prefix means nothing
across a ``GUNGI_MODE`` change (F3). A cut above ``MAX_TURNS -
CONVERSION_HORIZON_PLIES`` is a game the turn cap decides, whose rows then carry
the material-cap label this curriculum exists to contradict (F4). And a
diagnostic that names ``book_path`` in a bookless seeded run points the operator
at an empty string (F5).

The engine walks here are deterministic: every action list is produced by
replaying uniformly-random **legal** engine moves under a fixed numpy seed,
scanning seeds upward until one produces a line of the length the test needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import MAX_TURNS, PLANE_MAP_VERSION
from komugi.selfplay import book
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import (
    BookEntry,
    load_book,
    running_gungi_mode,
    select_book_entry,
)
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import (
    BOOK_BUILD_DOMAIN,
    BOOK_SELECT_DOMAIN,
    DRAFT_THRESHOLD_DOMAIN,
    SCHEMA,
    _MAX_DRAFT_THRESHOLD,
    game_seed,
    num_actions,
)
from komugi.selfplay.seed_openings import (
    CONVERSION_HORIZON_PLIES,
    MAX_SEED_ENTRIES,
    _MAX_DRAFT_THRESHOLD as _SEED_MAX_DRAFT_THRESHOLD,
    MIN_CUT,
    MIN_SEED_ENTRIES,
    SEED_OPENINGS_DOMAIN,
    SEED_OPENINGS_VERSION,
    SeedEntry,
    load_seed_openings,
    recompute_seed_choice,
    select_seed_entry,
)
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig, game_id
from komugi.train.shards import ShardReader, ShardWriter

_PROBE_DOMAIN: Final[int] = 0x9022_0001
_OOD_DOMAIN: Final[int] = 0x9022_0002
"""The campaign probe's two domains, restated rather than imported for the
reason ``write_book`` is duplicated across the test modules that need it: a
cross-package import here would tie this module's importability to
:mod:`komugi.campaign`, which the mechanism does not touch."""

_SHORT_GAME_PLIES: Final[int] = 80
"""Turn cap for the in-test engine: above :data:`MIN_CUT`, so a seeded game
still has plies to search, and low enough to play several games in seconds."""

_RUN_SEED: Final[int] = 20260920
_WORKER_ID: Final[int] = 2
_INCARNATION: Final[int] = 1
_GAMES: Final[int] = 3

_BAND_CUT: Final[int] = MIN_CUT
"""The pinned cut the row-shape nodes use: the lowest one a file may name, so a
test that reads ``min(ply)`` back off a shard is reading the bound as well."""

_MAX_PROB: Final[float] = 0.5
"""The highest share :class:`~komugi.config.SelfPlayConfig` accepts, and
therefore the highest any *worker* node here can run at. A1(ii)'s "prob 1 seeds
every game" is a property of the **draw**, not of a runnable config, so it is
asserted on the pure function
(:func:`~komugi.selfplay.seed_openings.select_seed_entry`) and the worker nodes
pick the games that share seeds out of their own stream."""

_LONG_PREFIX_PLIES: Final[int] = 320
"""R5's ">300-ply prefix", inside the loader's own ``cut_max`` bound (384)."""


class _StubEvaluator:
    """A deterministic evaluator: uniform logits, a fixed WDL, no state."""

    def __init__(self, actions: int) -> None:
        self.actions = actions

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        return NetEval(
            policy_logits=np.zeros((rows, self.actions), dtype=np.float32),
            wdl=np.tile(np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`."""

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def _random_line(seed: int, *, stop_at: int | None = None) -> tuple[int, ...]:
    """Replay uniformly-random legal engine moves from a bare :class:`Game`.

    Walked through the real engine rather than invented, because an action list
    that was made up would be rejected by the worker's own legality check and
    every assertion downstream of it would be about the rejection. Stops at
    `stop_at` plies, or at the engine's own adjudication when that is ``None``.
    """
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    while stop_at is None or len(actions) < stop_at:
        if state.is_terminal():
            break
        legal = np.nonzero(np.asarray(state.game.valid_moves()))[0]
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return tuple(actions)


def legal_actions(plies: int, *, start_seed: int = 0) -> tuple[int, ...]:
    """Return `plies` actions that replay legally and leave the game alive.

    Scans seeds upward until one produces a line that long -- random play ends
    a gungi game anywhere between ~35 and the engine's cap, so a fixed seed is
    not a length. The scan is deterministic, which is what the callers need.
    """
    for seed in range(start_seed, start_seed + 512):
        line = _random_line(seed, stop_at=plies)
        if len(line) == plies:
            return line
    raise AssertionError(f"no seed in 512 produced a {plies}-ply line")


def terminating_line(*, at_least: int, at_most: int) -> tuple[int, ...]:
    """Return a full game's actions -- the last one **ends** it.

    The shape a harvester actually emits (a recorded game's action list), and
    the one KOM-254 AM1 is about: replayed to its end, the last action is the
    terminating move, so a cut there leaves nothing to search.
    """
    for seed in range(512):
        line = _random_line(seed)
        if at_least <= len(line) <= at_most:
            return line
    raise AssertionError("no seed in 512 produced a terminating line in range")


def seed_entry(
    index: int,
    actions: Sequence[int],
    *,
    cut_min: int,
    cut_max: int,
    **overrides: Any,
) -> dict[str, Any]:
    """Return one entry payload, with `overrides` applied last."""
    payload: dict[str, Any] = {
        "id": f"seed-{index:06d}",
        "actions": [int(action) for action in actions],
        "cut_min": cut_min,
        "cut_max": cut_max,
        "start_ply": 0,
        "source_iteration": 771,
        "source_game_id": 1000 + index,
        "class": "eroded-plateau",
    }
    payload.update(overrides)
    return payload


def seed_entries(
    actions: Sequence[int],
    *,
    cut_min: int,
    cut_max: int,
    count: int = MIN_SEED_ENTRIES,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """Return `count` entries over one action list, ids distinct."""
    return [
        seed_entry(index, actions, cut_min=cut_min, cut_max=cut_max, **overrides)
        for index in range(count)
    ]


def write_seed_file(
    path: Path,
    entries: Sequence[Mapping[str, Any]],
    **manifest_overrides: Any,
) -> Path:
    """Write a seed-openings file holding `entries`, and return its path.

    Hand-built rather than produced by a harvester: the harvester is
    advisor-side and out of this ticket's scope, and what is under test is the
    consumer. The manifest carries every key
    :func:`~komugi.selfplay.seed_openings.load_seed_openings` reads, so the file
    is refused for exactly the reasons a real one would be.
    """
    payload: dict[str, Any] = {
        "version": SEED_OPENINGS_VERSION,
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "entries": [dict(entry) for entry in entries],
    }
    payload.update(manifest_overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def write_book(path: Path, prefixes: Sequence[Sequence[int]]) -> Path:
    """Write a minimal but **loadable** draft book holding `prefixes`.

    Duplicated from ``test_book_selection.py`` rather than imported, for the
    reason that module states about its own restatement: a test directory is put
    on ``sys.path`` by pytest only once something in it is collected, so a
    cross-module import would make this file's importability depend on
    collection order.
    """
    entries = [
        BookEntry(
            id=f"gen0001-{index:06d}-0badc0de",
            actions=tuple(int(action) for action in actions),
            plies=len(actions),
            source_iteration=0,
            source_game_id=index,
            deviation_ply=len(actions) - 1,
            deviated=True,
            support_size=16,
            support_fallback=False,
            truncated_mass=0.0,
            drawn_target_prob=0.5,
            restricted_entropy_bits=1.0,
            tempered_entropy_bits=2.0,
            flattening_ratio=0.75,
        )
        for index, actions in enumerate(prefixes)
    ]
    payload: dict[str, Any] = {
        "book_version": book.BOOK_VERSION,
        "generation": 1,
        "source_iterations": [0],
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": book.TEMPERATURE,
        "support_epsilon": book.SUPPORT_EPSILON,
        "support_top_k": book.SUPPORT_TOP_K,
        "support_floor": book.SUPPORT_FLOOR,
        "support_min_k": book.SUPPORT_MIN_K,
        "target_shard_budget": book.TARGET_SHARD_BUDGET,
        "cap": book.CAP,
        "min_plies": book.MIN_PLIES,
        "max_plies": book.MAX_PLIES,
        "entries_per_source_game": book.ENTRIES_PER_SOURCE_GAME,
        "build_seed": 7,
        "bit_generator": book.BIT_GENERATOR,
        "entries": [entry.as_json() for entry in entries],
        "stats": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def build_worker(
    directory: Path,
    *,
    seed_path: Path | None = None,
    prob: float = 0.0,
    book_path: Path | None = None,
    book_fraction: float = 0.0,
    run_seed: int = _RUN_SEED,
    games: int = _GAMES,
) -> tuple[SelfPlayWorker, ShardWriter]:
    """Return a worker over `directory`, and its writer (the caller closes it).

    The writer travels with the worker because a game's rows are only readable
    after the shard is sealed, and ``min(ply)`` off those rows is what the
    row-shape nodes assert on.
    """
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            gumbel_m_full=2,
            gumbel_n_full=2,
            gumbel_m_fast=2,
            gumbel_n_fast=2,
            prob_fast=0.5,
            book_path="" if book_path is None else str(book_path),
            book_fraction=book_fraction,
            seed_openings_path="" if seed_path is None else str(seed_path),
            seed_openings_prob=prob,
        ),
        WorkerConfig(
            worker_id=_WORKER_ID,
            incarnation=_INCARNATION,
            run_seed=run_seed,
            games=games,
            shard_dir=str(directory),
        ),
        writer,
    )
    return worker, writer


def rows_by_game(directory: Path) -> dict[int, list[dict[str, torch.Tensor]]]:
    """Return every written row, grouped by ``game_id`` and ply-ascending."""
    grouped: dict[int, list[dict[str, torch.Tensor]]] = {}
    for row in ShardReader(directory, schema=SCHEMA):
        grouped.setdefault(int(row["game_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda entry: int(entry["ply"]))
    return grouped


@pytest.fixture(scope="module")
def prefix_line() -> tuple[int, ...]:
    """A 70-ply legal line -- long enough for a cut at :data:`MIN_CUT`."""
    return legal_actions(70)


@pytest.fixture(scope="module")
def full_game_line() -> tuple[int, ...]:
    """A whole recorded game whose last action ends it (137 plies today).

    Two plies of headroom below the loader's ``cut_max`` bound, because both
    terminality nodes append an action or two to this line and then name a
    ``cut_max`` inside the longer list -- which is what the tightened bound
    (``cut_max < len(actions)``) makes them do.
    """
    return terminating_line(
        at_least=MIN_CUT + 1,
        at_most=MAX_TURNS - CONVERSION_HORIZON_PLIES - 2,
    )


@pytest.fixture(scope="module")
def long_line() -> tuple[int, ...]:
    """A line longer than :data:`_LONG_PREFIX_PLIES` (R5's >300-ply prefix)."""
    return legal_actions(_LONG_PREFIX_PLIES + 8)


# ---------------------------------------------------------------------------
# The stream: one domain, disjoint from every other (Q1)
# ---------------------------------------------------------------------------


def test_the_seed_domain_is_below_two_to_the_32_and_distinct_from_every_other() -> None:
    """The disjointness argument, as the three facts it rests on.

    ``< 2**32`` is not decoration: numpy splits a larger key element into two
    32-bit words, so a domain at or above it would hash exactly like
    ``(worker_id, incarnation, game_index, DOMAIN)`` and the "disjoint by
    composition" claim would be false. Above the engine's turn cap is what makes
    "unreachable as a ply" a fact about the engine rather than a hope, since
    ``move_seed`` composes ``(*spawn_key, ply)``.
    """
    assert SEED_OPENINGS_DOMAIN < 2**32
    assert SEED_OPENINGS_DOMAIN > Game.max_turns()
    for name, other in (
        ("BOOK_BUILD_DOMAIN", BOOK_BUILD_DOMAIN),
        ("BOOK_SELECT_DOMAIN", BOOK_SELECT_DOMAIN),
        ("DRAFT_THRESHOLD_DOMAIN", DRAFT_THRESHOLD_DOMAIN),
        ("PROBE_DOMAIN", _PROBE_DOMAIN),
        ("OOD_DOMAIN", _OOD_DOMAIN),
    ):
        assert SEED_OPENINGS_DOMAIN != other, (
            f"the seed-openings domain collides with {name}, so the two "
            f"streams are one and whether a game is seeded is correlated with "
            f"whatever that domain decides"
        )


def test_the_seed_draw_does_not_disturb_the_games_own_stream(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """The R1 invariant at its root: nothing here consumes the game's ``rng``."""
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, _digest = load_seed_openings(path)
    seeds = game_seed(_RUN_SEED, _WORKER_ID, _INCARNATION, 3)
    before = np.random.default_rng(seeds).random(8).tolist()
    select_seed_entry(
        _RUN_SEED, _WORKER_ID, _INCARNATION, 3, entries=entries, prob=0.5
    )
    after = np.random.default_rng(seeds).random(8).tolist()
    assert before == after


# ---------------------------------------------------------------------------
# (i) R1 -- prob 0 leaves every row byte-identical
# ---------------------------------------------------------------------------


def test_a_zero_probability_seed_file_leaves_the_rows_bit_identical(
    tmp_path: Path, prefix_line: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(i) The claim that makes "lands inert" falsifiable rather than asserted.

    A worker handed a valid seed file at ``prob = 0.0`` and a worker with no
    seed config at all play the *same* games: same full/fast draws, same
    per-move Gumbel vectors, same rows, field for field. Red against any
    implementation that draws the seed decision from the game's own ``rng``
    (which would shift every subsequent full/fast draw) or from a spawn key
    that collides with ``move_seed``'s.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    seeded_dir = tmp_path / "with-file"
    plain_dir = tmp_path / "without-file"
    with_file, writer_a = build_worker(seeded_dir, seed_path=path, prob=0.0)
    without, writer_b = build_worker(plain_dir)
    for index in range(_GAMES):
        assert with_file.play_game(index) > 0
        assert without.play_game(index) > 0
    writer_a.close()
    writer_b.close()

    left = rows_by_game(seeded_dir)
    right = rows_by_game(plain_dir)
    assert left.keys() == right.keys()
    assert left
    for key, rows in left.items():
        others = right[key]
        assert len(rows) == len(others)
        for row, other in zip(rows, others, strict=True):
            assert row.keys() == other.keys()
            for field in row:
                assert torch.equal(row[field], other[field]), (
                    f"game {key} row {int(row['ply'])} field {field} differs "
                    f"between a prob-0 seed config and no seed config: the "
                    f"mechanism did not land inert"
                )


# ---------------------------------------------------------------------------
# (ii), (viii) -- what a seeded game's rows look like
# ---------------------------------------------------------------------------


def test_probability_one_seeds_every_game(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(ii), first half: ``random()`` returns ``[0, 1)``, so ``< 1.0`` always holds.

    On the draw rather than on a worker, because a config that asked for
    ``prob = 1.0`` is refused: the *ceiling* is 0.5 (seeded games are training
    games and enter every iteration metric). A comparison written ``<=`` would
    pass a rate check and fail at the zero end, which the companion node below
    covers.
    """
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, _digest = load_seed_openings(path)
    for index in range(200):
        assert select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=1.0
        ).seeded
        assert not select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=0.0
        ).seeded


def test_seeded_games_begin_at_the_cut_and_unseeded_ones_at_ply_zero(
    tmp_path: Path, prefix_line: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(ii), and the second half of (viii): a seeded game never writes 0 rows.

    ``cut_min == cut_max`` pins the cut, so ``min(ply)`` read back off the shard
    is the cut itself and the ply column is asserted contiguous from it -- the
    prefix-offset plumbing KOM-014 paid for, at a prefix 4x longer than any book
    entry. The unseeded games of the same run are the control: they still begin
    at ply 0, which is what makes "seeded" a property of the draw and not of the
    file's mere presence.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, _digest = load_seed_openings(path)
    games = 8
    directory = tmp_path / "rows"
    worker, writer = build_worker(
        directory, seed_path=path, prob=_MAX_PROB, games=games
    )
    for index in range(games):
        assert worker.play_game(index) > 0, (
            f"game {index} wrote no row: a seeded game that records nothing is "
            f"KOM-254 AM1's failure, not a short game"
        )
    writer.close()

    grouped = rows_by_game(directory)
    assert len(grouped) == games
    seeded_seen = 0
    for index in range(games):
        rows = grouped[game_id(_WORKER_ID, _INCARNATION, index)]
        plies = [int(row["ply"]) for row in rows]
        choice = select_seed_entry(
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            index,
            entries=entries,
            prob=_MAX_PROB,
        )
        expected = _BAND_CUT if choice.seeded else 0
        seeded_seen += int(choice.seeded)
        assert plies == list(range(expected, expected + len(rows)))
        assert all(0.0 <= float(row["mlh_target"]) <= 1.0 for row in rows)
    assert seeded_seen > 0, f"no game in {games} was seeded at prob {_MAX_PROB}"
    assert seeded_seen < games, (
        f"every game in {games} was seeded at prob {_MAX_PROB}: this node "
        f"cannot see a draw that ignores its probability"
    )


def test_a_prefix_terminal_at_the_cut_raises(
    tmp_path: Path, full_game_line: tuple[int, ...]
) -> None:
    """(viii)/F1: AM1's after-the-last-action check, still reachable.

    KOM-014's replay checks terminality before each action and never after the
    last one, and the builder made a terminal book prefix unreachable. A
    *harvested* full game's last action IS the move that ended it, so without
    the after-check a file like this would write zero rows per game and report
    success.

    The loader now refuses ``cut_max == len(actions)`` (impl review F2), so the
    cut this node needs is reached the way the (v) node below reaches its own:
    the action list is the recorded game plus one more id, and the cut is the
    recorded game's own length. The loader therefore sees ``cut_max <
    len(actions)`` and accepts, the replay plays the whole recorded game, and
    the raise comes from **after** its last action rather than from before one
    -- which is the backstop AM1 exists to be for a malformed harvest.
    """
    actions = (*full_game_line, 0)
    cut = len(full_game_line)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(actions, cut_min=MIN_CUT, cut_max=cut),
    )
    index = _index_drawing_cut(path, cut, prob=_MAX_PROB)
    worker, writer = build_worker(
        tmp_path / "rows", seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    with pytest.raises(ValueError, match="terminal position") as raised:
        worker.play_game(index)
    # After ALL of them: the post-loop check, not the pre-action one.
    assert f"after {cut} of its {cut} plies" in str(raised.value)
    writer.close()


def test_a_prefix_terminal_before_the_cut_raises(
    tmp_path: Path, full_game_line: tuple[int, ...]
) -> None:
    """(v)/R2: a prefix that ends the game short of its cut must raise, not loop.

    The action list is a recorded game plus two more ids and the cut is one past
    the recorded game, so the replay reaches a terminal position with an action
    still to play -- the harvester bug KOM-254 D7's R2 names -- through a file
    the loader accepts (``cut_max < len(actions)``).
    """
    actions = (*full_game_line, 0, 0)
    cut = len(full_game_line) + 1
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(actions, cut_min=MIN_CUT, cut_max=cut),
    )
    index = _index_drawing_cut(path, cut, prob=_MAX_PROB)
    worker, writer = build_worker(
        tmp_path / "rows", seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    with pytest.raises(ValueError, match="terminal position") as raised:
        worker.play_game(index)
    # Strictly before: the replay stopped one action short of the cut.
    assert f"after {cut - 1} of its {cut} plies" in str(raised.value)
    writer.close()


def test_a_book_prefix_terminal_after_its_last_action_raises_the_books_message(
    tmp_path: Path, full_game_line: tuple[int, ...]
) -> None:
    """Impl review F3: AM1's after-check is inherited by the BOOK path too.

    ``_replay_prefix`` and ``_replay_seed_prefix`` are two call sites of one
    replay, so the check KOM-254 added for harvested prefixes now also fires on
    a book entry whose last action ends the game. That is the better behaviour
    -- the old code wrote zero rows and returned 0, silently -- but ``load_book``
    cannot prove the case unreachable: it bounds ``plies`` to ``[4, 12]`` and
    never replays, and an entry's last action is the *tempered* one, so "the
    source game continued past this position" does not carry. Nothing else in
    the suite pins it, and a refactor could drop it back to a 0-row game.

    The entry is hand-built rather than loaded, for the reason the bound above
    gives: a 12-ply book entry that ends a gungi game does not exist to be
    written into a file. The wording asserted is the **book's** -- naming
    ``book_path`` and ``book_fraction``, not the seed file.
    """
    book_path = write_book(
        tmp_path / "books" / "gen_1.json", [legal_actions(4, start_seed=11)]
    )
    worker, writer = build_worker(
        tmp_path / "rows", book_path=book_path, book_fraction=0.5
    )
    entry = BookEntry(
        id="gen0001-000042-0badc0de",
        actions=full_game_line,
        plies=len(full_game_line),
        source_iteration=0,
        source_game_id=42,
        deviation_ply=len(full_game_line) - 1,
        deviated=True,
        support_size=16,
        support_fallback=False,
        truncated_mass=0.0,
        drawn_target_prob=0.5,
        restricted_entropy_bits=1.0,
        tempered_entropy_bits=2.0,
        flattening_ratio=0.75,
    )
    with pytest.raises(ValueError) as raised:
        worker._replay_prefix(GameState(Game(), 0.0), entry)
    writer.close()
    message = str(raised.value)
    plies = len(full_game_line)
    assert f"book {book_path} entry {entry.id} reaches " in message
    assert f"a terminal position after {plies} of its {plies} plies" in message
    assert "the book and this engine disagree" in message


def _first_seeded_index(entries: Sequence[SeedEntry], *, prob: float) -> int:
    """Return the first game index this worker's stream seeds at `prob`.

    The worker nodes run at the config's ceiling rather than at ``prob = 1.0``,
    so which games are seeded is the stream's answer and not the test's.
    """
    for index in range(2000):
        if select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=prob
        ).seeded:
            return index
    raise AssertionError(f"no game index in 2000 was seeded at prob {prob}")


def _index_drawing_cut(path: Path, cut: int, *, prob: float) -> int:
    """Return a game index whose seed draw lands exactly on `cut`.

    Found by replaying the draw offline rather than by construction, which is
    the point: the cut a game gets is a property of its own stream, so a test
    that needs a particular one asks the stream which game has it.
    """
    entries, _digest = load_seed_openings(path)
    for index in range(20_000):
        choice = select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=prob
        )
        if choice.seeded and choice.cut == cut:
            return index
    raise AssertionError(f"no game index in 20000 drew a cut of {cut}")


# ---------------------------------------------------------------------------
# (iii), (xiii) -- the draw is reproducible, offline and with a book present
# ---------------------------------------------------------------------------


def test_the_same_game_reproduces_the_same_entry_and_cut(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(iii): the draw is a pure function of the four integers naming the game.

    And of nothing else -- asserted three ways: the same triple twice, a
    different ``game_index`` giving a different draw somewhere in the span, and
    :func:`recompute_seed_choice` -- which has neither the file nor a worker --
    agreeing entry for entry and cut for cut.
    """
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=MIN_CUT, cut_max=MIN_CUT + 8),
    )
    entries, _digest = load_seed_openings(path)
    seen_cuts: set[int] = set()
    for index in range(64):
        first = select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=0.5
        )
        again = select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=0.5
        )
        assert first == again
        # Two passes, which is what an offline caller does: the first reads the
        # entry back (the band does not touch the first two draws), the second
        # is handed that entry's own band and answers the cut.
        seeded, entry_index, _cut = recompute_seed_choice(
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            index,
            n_entries=len(entries),
            prob=0.5,
            cut_min=MIN_CUT,
            cut_max=MIN_CUT,
        )
        assert seeded == first.seeded
        assert entry_index == first.entry_index
        if not first.seeded:
            continue
        assert first.entry_index is not None
        chosen = entries[first.entry_index]
        assert recompute_seed_choice(
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            index,
            n_entries=len(entries),
            prob=0.5,
            cut_min=chosen.cut_min,
            cut_max=chosen.cut_max,
        ) == (True, first.entry_index, first.cut)
        assert first.cut is not None
        assert chosen.cut_min <= first.cut <= chosen.cut_max
        seen_cuts.add(first.cut)
    assert len(seen_cuts) > 1, (
        "every seeded game in 64 drew the same cut from a 9-wide band: the "
        "third draw is not a draw"
    )


def test_a_degenerate_band_always_yields_its_single_cut(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """Q2: ``cut_min == cut_max == k`` yields ``k``, never ``k`` modulo anything.

    ``integers(lo, hi + 1)`` is inclusive on ``cut_max``; an implementation that
    wrote ``integers(cut_min, cut_max)`` would raise or silently exclude the top
    of every band, and one that used ``%`` would bias it.
    """
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, _digest = load_seed_openings(path)
    seeded_seen = 0
    for index in range(200):
        choice = select_seed_entry(
            _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=0.5
        )
        if choice.seeded:
            seeded_seen += 1
            assert choice.cut == _BAND_CUT
    assert seeded_seen > 0, "no game in 200 was seeded at prob 0.5"


def test_a_booked_and_seeded_game_starts_from_the_seed_and_not_the_book(
    tmp_path: Path, prefix_line: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(xiii)/F6: on a game both draws claim, the seed wins and is recomputable.

    The pair is *found*, not forced: a ``(run_seed, game_index)`` where
    :func:`select_book_entry` says booked **and** the seed draw says seeded. The
    rows must then begin at the cut (52+), not at a 4- or 6-ply book prefix, and
    the offline recompute must agree -- which is what a census needs, since no
    column records any of this and ``book_audit.recompute_choice`` would happily
    answer "booked, entry k" for this very game.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    book_path = write_book(
        tmp_path / "books" / "gen_1.json",
        [legal_actions(4, start_seed=11), legal_actions(6, start_seed=12)],
    )
    seed_path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    loaded_book = load_book(book_path)
    entries, _digest = load_seed_openings(seed_path)
    fraction = 0.5
    prob = 0.5

    both = next(
        index
        for index in range(2000)
        if select_book_entry(
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            index,
            entries=len(loaded_book),
            fraction=fraction,
        ).booked
        and select_seed_entry(
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            index,
            entries=entries,
            prob=prob,
        ).seeded
    )

    directory = tmp_path / "rows"
    worker, writer = build_worker(
        directory,
        seed_path=seed_path,
        prob=prob,
        book_path=book_path,
        book_fraction=fraction,
        games=both + 1,
    )
    assert worker.play_game(both) > 0
    writer.close()

    rows = rows_by_game(directory)[game_id(_WORKER_ID, _INCARNATION, both)]
    book_lengths = {entry.plies for entry in loaded_book.entries}
    assert int(rows[0]["ply"]) == _BAND_CUT
    assert int(rows[0]["ply"]) not in book_lengths, (
        "the game's rows begin where a book prefix would have ended: a seeded "
        "game must skip the book consult entirely"
    )
    assert recompute_seed_choice(
        _RUN_SEED,
        _WORKER_ID,
        _INCARNATION,
        both,
        n_entries=len(entries),
        prob=prob,
        cut_min=_BAND_CUT,
        cut_max=_BAND_CUT,
    )[0] is True


# ---------------------------------------------------------------------------
# (iv), (xii) -- diagnostics name the SEED file, never the book
# ---------------------------------------------------------------------------


def test_an_illegal_seed_prefix_action_names_the_seed_file_entry_and_ply(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(iv)/(xii)/F5: with no book configured, the message must still be useful.

    ``_replay_prefix``'s two messages interpolate ``book_path`` and
    ``book_fraction``; in a bookless seeded run those are ``""`` and ``0.0``, so
    an operator reading the crash would be pointed at an empty string. The seed
    path's own message names the seed file, the entry id and the ply the illegal
    action sits at.

    Entry 0 is left legal on purpose: the worker replays it at load (the
    pre-flight), so a file whose every entry was corrupt would raise in the
    constructor and this node would not be testing the per-game path at all.
    """
    illegal_ply = 55
    illegal = _an_illegal_action_at(prefix_line, illegal_ply)
    corrupt = (*prefix_line[:illegal_ply], illegal, *prefix_line[illegal_ply + 1 :])
    entries = [
        seed_entry(
            index,
            prefix_line if index == 0 else corrupt,
            cut_min=60,
            cut_max=60,
        )
        for index in range(MIN_SEED_ENTRIES)
    ]
    path = write_seed_file(tmp_path / "seeds.json", entries)
    loaded, _digest = load_seed_openings(path)
    index = next(
        game_index
        for game_index in range(2000)
        if (
            choice := select_seed_entry(
                _RUN_SEED,
                _WORKER_ID,
                _INCARNATION,
                game_index,
                entries=loaded,
                prob=_MAX_PROB,
            )
        ).seeded
        and choice.entry_index != 0
    )

    worker, writer = build_worker(
        tmp_path / "rows", seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    with pytest.raises(ValueError) as raised:
        worker.play_game(index)
    writer.close()
    message = str(raised.value)
    assert str(path) in message
    assert f"prefix ply {illegal_ply}" in message
    # The DRAWN entry's id, not the shared ``seed-0000`` prefix: every entry in
    # this file starts with it, so a message that always named entry 0 -- the
    # one entry deliberately left legal -- would pass a prefix assertion.
    assert choice.entry_index is not None
    assert loaded[choice.entry_index].id in message
    assert "book" not in message, (
        f"a bookless seeded run raised a message naming the book: {message}"
    )


def _an_illegal_action_at(actions: Sequence[int], ply: int) -> int:
    """Return an action id that is **illegal** after `ply` of `actions`."""
    state = GameState(Game(), 0.0)
    for action in actions[:ply]:
        state.play(action)
    mask = np.asarray(state.game.valid_moves(), dtype=bool)
    return int(np.nonzero(~mask)[0][0])


# ---------------------------------------------------------------------------
# (vi), (ix), (x), (xi), (xv) -- the config and the loader
# ---------------------------------------------------------------------------


def test_a_seed_probability_without_a_path_is_refused() -> None:
    """(vi): a share is a claim about a file, so one without a file is refused."""
    with pytest.raises(ValueError, match="seed_openings_prob"):
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            seed_openings_path="",
            seed_openings_prob=0.04,
        )


@pytest.mark.parametrize("bad", [-0.1, 0.51, 1.0, float("nan")])
def test_a_seed_probability_outside_the_band_is_refused(bad: float) -> None:
    """(vi): ``[0, 0.5]``, with NaN refused here rather than discovered as 0%."""
    with pytest.raises(ValueError, match="seed_openings_prob"):
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            seed_openings_path="/tmp/does-not-need-to-exist.json",
            seed_openings_prob=bad,
        )


def test_the_default_selfplay_config_seeds_nothing() -> None:
    """Every pre-KOM-254 construction is what it was."""
    config = SelfPlayConfig(workers=1, process_batch_size=1)
    assert config.seed_openings_path == ""
    assert config.seed_openings_prob == 0.0


def test_a_seed_path_without_a_probability_is_allowed(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """The other pairing is legal: a file present and never drawn from -- which
    is how an operator switches the curriculum off without deleting it."""
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        seed_openings_path=str(path),
        seed_openings_prob=0.0,
    )
    assert config.seed_openings_prob == 0.0


def test_a_future_seed_file_version_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(vi): the version is read first, because it says what the rest means."""
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
        version=SEED_OPENINGS_VERSION + 1,
    )
    with pytest.raises(ValueError, match="declares version"):
        load_seed_openings(path)


def test_a_seed_file_from_another_action_space_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(x)/F3: an action id means nothing across an encoding change."""
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
        action_size=num_actions() + 1,
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert str(num_actions() + 1) in message
    assert str(num_actions()) in message


def test_a_seed_file_from_another_gungi_mode_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(x)/F3: a prefix replayed from another start position is another line."""
    other = "intro" if running_gungi_mode() != "intro" else "advanced"
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
        gungi_mode=other,
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert repr(other) in message
    assert repr(running_gungi_mode()) in message


@pytest.mark.parametrize("start_ply", [8, None])
def test_a_prefix_that_does_not_start_at_ply_zero_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...], start_ply: int | None
) -> None:
    """(ix)/F2: the guard a book does not need and a harvest cannot do without.

    ~10% of any source window is itself booked, and a booked game's rows begin
    at ply ``L``; an action list lifted from one replays *legally* from ply 0
    into a different position, so the worker's legality check provably cannot
    catch it. ``start_ply`` is therefore mandatory (``None`` here omits the key
    entirely) and must be ``0``.
    """
    entries = seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT)
    for entry in entries:
        if start_ply is None:
            del entry["start_ply"]
        else:
            entry["start_ply"] = start_ply
    path = write_seed_file(tmp_path / "seeds.json", entries)
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert "start_ply" in message
    assert "seed-000000" in message


def test_a_cut_beyond_the_conversion_horizon_is_refused(tmp_path: Path) -> None:
    """(xi)/F4: ``cut_max`` is bounded by a derivation, not by a literal.

    At the review's old ``480`` only 32 plies remain of a 512-ply engine, so a
    deep-cut game is a cap game *by construction* and its rows carry KOM-245's
    material-cap label -- the very label the curriculum exists to contradict.
    The bound is ``MAX_TURNS - CONVERSION_HORIZON_PLIES``, and both numbers are
    named in the refusal so an operator can see which one moved.
    """
    assert CONVERSION_HORIZON_PLIES == MAX_TURNS // 4
    bound = MAX_TURNS - CONVERSION_HORIZON_PLIES
    # Action ids only have to be inside the action space for the loader; the
    # legality of a 400-ply line is the worker's question, not this one's.
    actions = [0] * (bound + 8)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(actions, cut_min=MIN_CUT, cut_max=bound + 1),
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert str(bound + 1) in message
    assert str(bound) in message
    assert str(MAX_TURNS) in message
    assert str(CONVERSION_HORIZON_PLIES) in message


def test_a_file_below_the_entry_floor_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(xv)/F8: a handful of 52+-ply prefixes is a handful of positions.

    The book's own floor, with more force: a seed prefix is 4-30x longer than a
    book entry and therefore far more specific, and at the permitted ceiling
    ``prob = 0.5`` a one-entry file would send half of self-play into a single
    position.
    """
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(
            prefix_line,
            cut_min=_BAND_CUT,
            cut_max=_BAND_CUT,
            count=MIN_SEED_ENTRIES - 1,
        ),
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert str(MIN_SEED_ENTRIES - 1) in message
    assert str(MIN_SEED_ENTRIES) in message
    assert str(MAX_SEED_ENTRIES) in message


def test_the_restated_draft_threshold_still_equals_the_seed_disciplines() -> None:
    """Impl review F7: the tripwire that replaces a private cross-module import.

    :data:`komugi.selfplay.seed_openings._MAX_DRAFT_THRESHOLD` restates
    :mod:`komugi.selfplay.records`'s own private constant rather than importing
    it, which is the convention that constant's docstring states for itself
    (and the reason ``records`` restates ``komugi.config``'s
    ``_MAX_MIN_PLACEMENTS`` in turn). A restatement needs a test where an import
    needed none: this is it, and it is the one place either private name is
    allowed to be imported.
    """
    assert _SEED_MAX_DRAFT_THRESHOLD == _MAX_DRAFT_THRESHOLD


def test_a_cut_that_could_land_mid_draft_is_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(xv)/F10: ``cut_min >= 52``, so no cut lands inside a draft.

    A full draft is one hand per side plus each side's 終. Below that a cut can
    land mid-draft, where the prefix plies obeyed the *source* run's DRAFTMIN
    thresholds while this game's drawn ``(N_w, N_b)`` bind only the searched
    ones -- a threshold that is silently vacuous rather than applied.
    """
    assert MIN_CUT == 52
    assert MIN_CUT == 2 * _SEED_MAX_DRAFT_THRESHOLD + 2
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=MIN_CUT - 1, cut_max=_BAND_CUT),
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert str(MIN_CUT - 1) in message
    assert str(MIN_CUT) in message


def test_a_cut_at_the_end_of_the_action_list_is_refused_at_load(
    tmp_path: Path, full_game_line: tuple[int, ...]
) -> None:
    """F2: ``cut_max == len(actions)`` is a worker death, so it dies at load.

    A harvested entry's actions are a whole source game, and its last action is
    the move that ended it: the existence of ``actions[cut]`` is exactly what
    proves the position at ``cut`` was non-terminal when the game was played.
    At equality the top of the band is therefore terminal by construction, and
    the replay's AM1 check refuses it -- per game, at draw time, at whichever
    game the stream happens to give that cut, hours into an iteration. The
    pre-flight provably cannot see it (it replays entry 0 at ``cut_min`` only),
    so the loader is the only place the file can be stopped as a *file*.

    This is the natural shape of a real harvest: ``cut_max = min(len(actions),
    384)`` gives equality for every source game shorter than 384 plies.
    """
    cut = len(full_game_line)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(full_game_line, cut_min=MIN_CUT, cut_max=cut),
    )
    with pytest.raises(ValueError) as raised:
        load_seed_openings(path)
    message = str(raised.value)
    assert "seed-000000" in message
    assert f"cut_max {cut}" in message
    assert f"{len(full_game_line)} actions" in message


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"actions": []}, "no actions"),
        ({"actions": [num_actions()] * 70}, "outside"),
        ({"actions": [-1] * 70}, "outside"),
        ({"cut_min": 70, "cut_max": 60}, "above cut_max"),
        ({"cut_max": 90}, "strictly below"),
        ({"cut_max": 70}, "strictly below"),
        ({"id": ""}, "blank id"),
    ],
)
def test_an_entry_that_violates_its_own_bounds_is_refused(
    tmp_path: Path,
    prefix_line: tuple[int, ...],
    overrides: dict[str, Any],
    expected: str,
) -> None:
    """(vi): a hand-edited entry fails at load rather than at replay."""
    merged: dict[str, Any] = {"cut_min": _BAND_CUT, "cut_max": _BAND_CUT, **overrides}
    line = tuple(merged.pop("actions", prefix_line))
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(line, **merged),
    )
    with pytest.raises(ValueError, match=expected):
        load_seed_openings(path)


def test_a_missing_entry_key_and_a_duplicate_id_are_refused(
    tmp_path: Path, prefix_line: tuple[int, ...]
) -> None:
    """(vi): the two file-level shapes a hand edit produces."""
    entries = seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT)
    del entries[3]["cut_max"]
    path = write_seed_file(tmp_path / "missing.json", entries)
    with pytest.raises(ValueError, match="missing key"):
        load_seed_openings(path)

    duplicated = seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT)
    duplicated[7]["id"] = duplicated[0]["id"]
    other = write_seed_file(tmp_path / "duplicate.json", duplicated)
    with pytest.raises(ValueError, match="two entries with id"):
        load_seed_openings(other)


def test_a_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    """(vi): the outermost refusal, naming the file rather than the traceback."""
    path = tmp_path / "seeds.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_seed_openings(path)


# ---------------------------------------------------------------------------
# (xiv) -- the file's hash, and the log line that carries it
# ---------------------------------------------------------------------------


def test_the_loader_returns_the_file_hash_and_the_worker_logs_it(
    tmp_path: Path,
    prefix_line: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(xiv)/F7: the seed file is refreshed per harvest window.

    So "which entry did game 783/12/0/44 use" is answerable only against the
    file the worker actually read, and nothing in the schema records which one
    that was. The loader hashes the exact file bytes and the per-seeded-game log
    line carries the hash's prefix beside the game's identity, the entry id and
    the cut.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, digest = load_seed_openings(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    index = _first_seeded_index(entries, prob=_MAX_PROB)
    worker, writer = build_worker(
        tmp_path / "rows", seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    with caplog.at_level(logging.WARNING, logger="komugi.selfplay.worker"):
        assert worker.play_game(index) > 0
    writer.close()

    choice = select_seed_entry(
        _RUN_SEED, _WORKER_ID, _INCARNATION, index, entries=entries, prob=_MAX_PROB
    )
    assert choice.seeded and choice.entry_index is not None
    lines = [record.getMessage() for record in caplog.records]
    seeded_lines = [line for line in lines if "seeded game" in line]
    assert len(seeded_lines) == 1, lines
    line = seeded_lines[0]
    assert digest[:12] in line
    assert entries[choice.entry_index].id in line
    assert f"cut={_BAND_CUT}" in line
    assert f"worker={_WORKER_ID}" in line
    assert f"incarnation={_INCARNATION}" in line
    assert f"game_index={index}" in line


class _CollectingHandler(logging.Handler):
    """A handler that keeps every record it is given, and formats nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep `record`."""
        self.records.append(record)


def test_the_seeded_line_reaches_a_handler_under_komugis_own_logging_config(
    tmp_path: Path,
    prefix_line: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(xiv)/impl review F1: the level, measured against the config komugi has.

    Which is *none*: ``basicConfig``/``dictConfig`` appear nowhere in ``src/``
    or ``scripts/``, so a worker's records are filtered by the root logger's
    own default -- WARNING -- and ``logging.lastResort`` is WARNING too. An
    INFO line is therefore dropped before any handler, and the flip-on guard
    ("seeded games appear in worker logs at ~4% ± 2%") would read 0% whether
    the curriculum was live or silently seeding nothing.

    So this node attaches a handler to the **root** logger and touches no level
    anywhere: no ``caplog.at_level``, no ``basicConfig``. Raising a level is
    exactly what production does not do, and a fixture that raised one would
    make this pass at any level at all. Red while the line is INFO.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(prefix_line, cut_min=_BAND_CUT, cut_max=_BAND_CUT),
    )
    entries, _digest = load_seed_openings(path)
    index = _first_seeded_index(entries, prob=_MAX_PROB)
    worker, writer = build_worker(
        tmp_path / "rows", seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    handler = _CollectingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        assert worker.play_game(index) > 0
    finally:
        root.removeHandler(handler)
    writer.close()

    seeded_lines = [
        record.getMessage()
        for record in handler.records
        if "seeded game" in record.getMessage()
    ]
    assert len(seeded_lines) == 1, (
        f"a seeded game's announcement did not reach a root handler at the "
        f"default level: {[record.getMessage() for record in handler.records]}"
    )


# ---------------------------------------------------------------------------
# (vii) -- R5, a prefix longer than 300 plies
# ---------------------------------------------------------------------------


def test_a_prefix_longer_than_three_hundred_plies_records_only_searched_plies(
    tmp_path: Path, long_line: tuple[int, ...]
) -> None:
    """(vii)/R5: the prefix-offset plumbing is length-agnostic, measured.

    ``total_plies`` is a ply counter and not ``len(rows)``; ``mlh_target`` is
    ``(total_plies - row.ply) * MLH_SCALE`` off an **absolute** ply, so it stays
    in ``[0, 1]`` at any prefix length; and ``short_term_value`` is bounded by
    ``len(rows)``, never by the ply counter. A 320-ply prefix is 27x a book
    entry's longest, and this node is the only place any of that is exercised
    past 12 plies.

    Run on the real engine (no shortened cap): the point is a prefix that is
    long *relative to the engine's own 512-ply cap*.
    """
    path = write_seed_file(
        tmp_path / "seeds.json",
        seed_entries(
            long_line, cut_min=_LONG_PREFIX_PLIES, cut_max=_LONG_PREFIX_PLIES
        ),
    )
    directory = tmp_path / "rows"
    entries, _ = load_seed_openings(path)
    index = _first_seeded_index(entries, prob=_MAX_PROB)
    worker, writer = build_worker(
        directory, seed_path=path, prob=_MAX_PROB, games=index + 1
    )
    written = worker.play_game(index)
    writer.close()
    assert written > 0

    rows = rows_by_game(directory)[game_id(_WORKER_ID, _INCARNATION, index)]
    assert len(rows) == written
    plies = [int(row["ply"]) for row in rows]
    assert plies == list(range(_LONG_PREFIX_PLIES, _LONG_PREFIX_PLIES + written)), (
        "a seeded game recorded a row for a ply it did not search"
    )
    for row in rows:
        mlh = float(row["mlh_target"])
        assert np.isfinite(mlh) and 0.0 <= mlh <= 1.0
        short_term = row["short_term_value"].numpy()
        assert np.all(np.isfinite(short_term))
        assert np.all(np.abs(short_term) <= 1.0)
