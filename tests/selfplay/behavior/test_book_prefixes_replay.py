"""Every book entry replays legally and stops short of a terminal position (#5, #6).

The two claims that make a book usable at all: a prefix is a list of action ids
handed to a fresh :class:`~komugi.engine.game.Game`, so if any of them is
illegal there the booked game dies in the worker, and if the last one *ends* the
game the "opening" is a finished game.

This module also carries the behaviour directory's **shard-tree rig**
(:func:`plant`, :func:`ordinary_games`), which
``test_book_determinism.py`` and ``test_book_build_cost.py`` import. It lives in
a test module rather than a ``*_rig.py`` because KOM-014a's file allowlist has no
rig module in it; the alternative was three near-identical copies of the same
sixty lines, which is the shape divergent fixtures come in.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from komugi.engine.game import Game
from komugi.engine.observation import CONTROL_EMPTY, CONTROL_SIZE, PLANE_MAP_VERSION
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import Book, build_book, load_book
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.train.shards import ShardWriter

SUPPORT_WIDTH: Final[int] = 16
"""Legal actions a planted ``policy_target`` puts mass on -- exactly ``SUPPORT_TOP_K``."""

SHAPE: Final[tuple[float, ...]] = (0.70, 0.10, 0.05) + (0.0115,) * 13
"""§3.2's worked peaked restriction. Every mass clears ``SUPPORT_EPSILON``, so
the truncated support is the whole planted support and the draw is legal by
construction rather than by luck."""

PLIES: Final[int] = 6
"""Six plies gives eligible deviations ``d ∈ {3, 4}``."""


@dataclass(frozen=True)
class PlantedGame:
    """One source game as the shards will carry it."""

    game_id: int
    actions: tuple[int, ...]
    supports: tuple[tuple[int, ...], ...]


def synthesise(game_id: int, plies: int, seed: int) -> PlantedGame:
    """Walk a fresh game forward by *legal* actions, recording what a row needs.

    The actions are legal because they were taken from ``legal_mask()`` at the
    position that produced them, and the ``policy_target`` support is that same
    mask's first :data:`SUPPORT_WIDTH` entries -- so a tempered draw at any ply
    is a legal move at the position the replay reaches.
    """
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    supports: list[tuple[int, ...]] = []
    for _ply in range(plies):
        legal = np.nonzero(state.legal_mask())[0]
        assert legal.size >= SUPPORT_WIDTH
        supports.append(tuple(int(action) for action in legal[:SUPPORT_WIDTH]))
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return PlantedGame(game_id, tuple(actions), tuple(supports))


def ordinary_games(count: int, *, seed: int = 0) -> list[PlantedGame]:
    """Return `count` planted games of :data:`PLIES` plies each."""
    return [synthesise(index, PLIES, seed + index) for index in range(count)]


def _row(game: PlantedGame, ply: int) -> dict[str, torch.Tensor]:
    """Return one real-:data:`SCHEMA` row of `game` at `ply`."""
    target = torch.zeros(num_actions(), dtype=torch.float32)
    for slot, mass in enumerate(SHAPE):
        target[game.supports[ply][slot]] = mass
    return {
        "observation": torch.zeros(Game.observation_size(), dtype=torch.float32),
        "policy_target": target,
        "wdl_target": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "mlh_target": torch.tensor(0.5, dtype=torch.float32),
        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(0, dtype=torch.int64),
        "to_play": torch.tensor(ply % 2, dtype=torch.int64),
        "ply": torch.tensor(ply, dtype=torch.int64),
        "game_id": torch.tensor(game.game_id, dtype=torch.int64),
        "sims_used": torch.tensor(32, dtype=torch.int64),
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        "final_control": torch.full((CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64),
        "action": torch.tensor(game.actions[ply], dtype=torch.int64),
        # KOM-005b's three columns. This rig is imported by
        # ``test_book_determinism.py`` and ``test_book_build_cost.py``, so both
        # of those go red through this one function on a schema append and
        # green again on this edit alone -- if either is still red afterwards,
        # this fixture is what is wrong, not they.
        "termination": torch.tensor(0, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [0.5, -0.25, 1.0 - 0.5 * (ply % 3)], dtype=torch.float32
        ),
        "draft_outcome": (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if ply < 2
            else torch.zeros(3, dtype=torch.float32)
        ),
    }


def plant(
    root: Path,
    iteration: int,
    games: Iterable[PlantedGame],
    *,
    samples_per_shard: int = 64,
) -> None:
    """Write `games` into ``<root>/iter_<iteration>/shards/worker_0_r0``.

    ``flush_every_seconds`` is pushed out of reach: the writer's *time* trigger
    would otherwise seal a short shard on a slow machine, and fixtures here
    depend on how many rows a shard holds.
    """
    directory = root / f"iter_{iteration}" / "shards" / "worker_0_r0"
    with ShardWriter(
        directory,
        SCHEMA,
        samples_per_shard=samples_per_shard,
        flush_every_seconds=86_400.0,
    ) as writer:
        for game in games:
            for ply in range(len(game.actions)):
                writer.append(_row(game, ply))


def build(root: Path, generation: int, *, cap: int = 96, seed: int = 20260905) -> Book:
    """Build and load a book over `root`, failing loudly if none was written."""
    path = build_book(root, generation, seed, cap=cap, min_entries=32)
    assert path is not None, "the planted tree was expected to yield a book"
    return load_book(path)


@pytest.fixture(scope="module")
def replayable(tmp_path_factory: pytest.TempPathFactory) -> Book:
    """A book over a 24-game iteration."""
    root = tmp_path_factory.mktemp("book-replay")
    plant(root, 0, ordinary_games(24, seed=11))
    return build(root, 1)


def _replay(actions: Sequence[int]) -> GameState:
    """Replay `actions` from a fresh game, asserting legality at every step."""
    state = GameState(Game(), 0.0)
    for position, action in enumerate(actions):
        assert not state.is_terminal(), (
            f"the game ended at ply {position} of a {len(actions)}-ply prefix"
        )
        assert bool(state.legal_mask()[action]), (
            f"action {action} is illegal at ply {position}; the builder verified "
            "legality on this same engine, so the book and the engine disagree"
        )
        state.play(action)
    return state


def test_every_entry_replays_legally_from_a_fresh_game(replayable: Book) -> None:
    """Each entry replays through ``GameState(Game(), 0.0)`` move by move (#5)."""
    assert replayable.entries
    for entry in replayable.entries:
        _replay(entry.actions)


def test_no_entry_ends_on_a_terminal_position(replayable: Book) -> None:
    """A prefix that finishes the game is not an opening (#6, §3.2 step 6)."""
    for entry in replayable.entries:
        state = _replay(entry.actions)
        assert not state.is_terminal(), (
            f"entry {entry.id} ends on an adjudicated position, so a booked game "
            "starting from it would record no rows at all"
        )
