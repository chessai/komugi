"""A corrupt book stops the game it poisons (KOM-014 §8 #18, §6.3).

The invariant is not "an illegal prefix raises" -- ``GameState.play`` delegates
to ``Game.play_action``, which validates, so *something* was always going to
raise. It is that the **worker does not catch it and fall back to an empty
board**. That fallback is the failure this node exists to make impossible: it
would convert a corrupt book into a generation that quietly booked 0% while
every telemetry line reported ``book_fraction``, and the shards would carry
plain unbooked games that no audit could tell apart from honest ones.

So the assertions are two, and the second is the one with teeth: the call
raises, **and** the game wrote no row. A worker that swallowed the exception and
replayed the game from the empty board would satisfy neither.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")

from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.selfplay import book
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import BookEntry, running_gungi_mode
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig
from komugi.train.shards import SHARD_GLOB, ShardWriter

_PREFIX_PLIES: Final[int] = 5
_SHORT_GAME_PLIES: Final[int] = 12


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


def _write_book(path: Path, actions: Sequence[int]) -> Path:
    """Write a loadable one-entry book holding `actions`.

    Restated here rather than imported from ``tests/selfplay/unit`` -- a test
    directory reaches ``sys.path`` only once pytest collects something in it,
    so a cross-directory import would tie this module's importability to
    collection order (the reason ``campaign_rig.py`` restates its own wall
    helpers).
    """
    entry = BookEntry(
        id="gen0001-000000-cafebabe",
        actions=tuple(int(action) for action in actions),
        plies=len(actions),
        source_iteration=0,
        source_game_id=1,
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
        "build_seed": 3,
        "bit_generator": book.BIT_GENERATOR,
        "entries": [entry.as_json()],
        "stats": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _legal_line(plies: int, seed: int) -> tuple[list[int], int]:
    """Return a legal `plies`-ply line, plus an action **illegal** at its end.

    Both come from the same walk, which is what makes the corruption surgical:
    every action but the last is one the engine accepted at the position it was
    played into, so a failure can only be the last one.
    """
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    for _ply in range(plies):
        assert not state.is_terminal()
        legal = np.nonzero(state.legal_mask())[0]
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    illegal = np.nonzero(~state.legal_mask())[0]
    assert illegal.size > 0, "every action is legal here, so nothing can be corrupted"
    return actions, int(illegal[0])


def test_an_illegal_prefix_fails_the_game_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-corrupted entry raises out of ``play_game`` and writes nothing."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    line, illegal = _legal_line(_PREFIX_PLIES, 99)
    corrupted = [*line[:-1], illegal]
    assert corrupted != line, "the corruption did not change the entry"
    book_file = _write_book(tmp_path / "books" / "gen_1.json", corrupted)

    directory = tmp_path / "shards"
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
            prob_fast=0.0,
            book_path=str(book_file),
            book_fraction=1.0,
        ),
        WorkerConfig(
            worker_id=0,
            incarnation=0,
            run_seed=515,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )

    with pytest.raises(ValueError) as raised:
        worker.play_game(0)
    message = str(raised.value)
    assert str(book_file) in message, "the failure must name the book"
    assert "gen0001-000000-cafebabe" in message, "and the entry that failed"
    assert str(illegal) in message

    writer.close()
    assert not list(directory.glob(SHARD_GLOB)), (
        "the worker wrote rows for a game whose book prefix was illegal, so it "
        "fell back to the empty board -- which is exactly the silent 0% booking "
        "rate 6.3 forbids"
    )


def test_a_legal_prefix_of_the_same_walk_still_plays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the *uncorrupted* line plays, so the refusal is the action.

    Without this the node above would pass for a worker that refused every
    book, which is a different bug wearing the same assertion.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    line, _illegal = _legal_line(_PREFIX_PLIES, 99)
    book_file = _write_book(tmp_path / "books" / "gen_1.json", line)

    directory = tmp_path / "shards"
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
            prob_fast=0.0,
            book_path=str(book_file),
            book_fraction=1.0,
        ),
        WorkerConfig(
            worker_id=0,
            incarnation=0,
            run_seed=515,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )
    written = worker.play_game(0)
    writer.close()
    assert written > 0
