"""``final_control`` on a finished game's rows: one map, every mover's frame.

KOM-010 delivers the ownership *target*, and the only thing a target can get
wrong that no shape check would catch is its perspective. ``wdl_target`` has
carried that risk since design F11 and is asserted at both parities; this
module does the same job for the control map, on a real game played through
the real worker into real shards.

The two claims, and why each is falsifiable:

* **mover-relative, per row.** The engine states the map absolutely (0 white,
  1 black, 2 empty). A row whose mover is white must see white's squares as
  ``0``; a row whose mover is black must see those same squares as ``1``.
  A worker that stamped the absolute map into every row would pass every
  "the field is present and int64" check and fail this one on the black rows.
* **one map, differing only by the flip.** Un-flipping each row by its own
  ``to_play`` must recover a single absolute map, identical across every row
  of the game -- because the map describes the *final* position and nothing
  else. A worker that recomputed it per ply (or captured it mid-game) would
  produce rows that disagree, which this test reads directly.

Keyed off ``to_play`` and never off ply parity, deliberately: the draft lets
one side move twice (wave-9b review F7), so parity is not a reliable stand-in
for the mover even where it happens to hold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import INTRO_POSITION
from komugi.engine.game import Game, RepetitionRule
from komugi.engine.observation import (
    CONTROL_BLACK,
    CONTROL_EMPTY,
    CONTROL_SIZE,
    CONTROL_WHITE,
)
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig, mover_relative_control
from komugi.train.shards import ShardReader, ShardWriter

_SHORT_GAME_PLIES: Final[int] = 6
"""Turn cap for the in-test engine subclass -- a whole game in a few seconds,
and long enough that both colours certainly move."""

_RUN_SEED: Final[int] = 20260904


class _StubEvaluator:
    """A deterministic batch evaluator: uniform policy, fixed WDL.

    The game this produces is arbitrary but reproducible; nothing here asserts
    on *which* game was played, only on how its final board was recorded.
    """

    def __init__(self, actions: int) -> None:
        self.actions = actions

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        return NetEval(
            policy_logits=np.zeros((rows, self.actions), dtype=np.float32),
            wdl=np.tile(
                np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1)
            ),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`.

    Only ``max_turns`` changes, so every rule that decides what the final board
    looks like -- move generation, capture, betrayal, stacking -- is the
    production engine's.
    """

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def _play_a_game(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[npt.NDArray[np.int8], list[dict[str, torch.Tensor]]]:
    """Play one short game and return ``(final absolute map, rows)``.

    The worker builds its own :class:`~komugi.engine.game.Game` and never hands
    it back, so the factory patched in here keeps a reference to the instance it
    made. That object *is* the final position: the worker plays into it in
    place and the search only ever mutates clones. It takes the worker's
    ``repetition_rule=`` keyword (KOM-141) and passes it on, so the stub stays
    a faithful stand-in for ``Game`` rather than one that happens to work only
    while the worker calls it with no arguments.

    The start position is pinned to :data:`INTRO_POSITION` rather than left to
    ``Game``'s default, which reads ``GUNGI_MODE`` from the environment: a
    draft-phase start would put six arata on an otherwise empty board, and the
    non-vacuity checks below want a board with both colours and empty squares
    on it.
    """
    created: list[Game] = []

    def factory(
        *, repetition_rule: RepetitionRule = RepetitionRule.FULL
    ) -> Game:
        game = _ShortGame(
            board_from_fen(INTRO_POSITION), repetition_rule=repetition_rule
        )
        created.append(game)
        return game

    monkeypatch.setattr("komugi.selfplay.worker.Game", factory)

    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=0.0,
        draw_score=0.2,
    )
    writer = ShardWriter(directory, SCHEMA)
    try:
        SelfPlayWorker(
            _StubEvaluator(num_actions()),
            config,
            WorkerConfig(
                worker_id=0,
                incarnation=0,
                run_seed=_RUN_SEED,
                games=1,
                shard_dir=str(directory),
            ),
            writer,
        ).play_game(0)
    finally:
        writer.close()

    assert len(created) == 1, "the worker built more than one root game"
    absolute = np.asarray(created[0].final_control(), dtype=np.int8)

    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    rows = list(reader)
    assert rows, "the game produced no rows"
    return absolute, rows


def _relative(row: dict[str, torch.Tensor]) -> npt.NDArray[np.int64]:
    """Return one row's ``final_control`` as a numpy vector."""
    control = row["final_control"].numpy()
    assert control.shape == (CONTROL_SIZE,)
    assert control.dtype == np.int64, "a shard holds int64, never int8 (design F4)"
    return np.asarray(control, dtype=np.int64)


def _absolutise(
    relative: npt.NDArray[np.int64], to_play: int
) -> npt.NDArray[np.int8]:
    """Undo the mover flip: return `relative` back in the engine's colour frame.

    The inverse of :func:`~komugi.selfplay.worker.mover_relative_control`,
    written out here rather than imported, so that "the flip round-trips" is a
    statement about two independently written functions instead of one.
    """
    absolute = np.full(relative.shape, CONTROL_EMPTY, dtype=np.int8)
    absolute[relative == 0] = np.int8(to_play)
    absolute[relative == 1] = np.int8(1 - to_play)
    return absolute


def test_rows_are_mover_relative_at_both_parities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """White's squares read ``0`` on a white row and ``1`` on a black one.

    The empties are the control: they are ``2`` in both frames, so a test that
    only looked at them would pass for a worker that never flipped anything.
    Non-vacuity is therefore asserted on all three -- the final board must
    carry white-topped squares, black-topped squares and empty squares -- and
    on both movers appearing among the rows.
    """
    absolute, rows = _play_a_game(tmp_path / "game", monkeypatch)

    white_squares = np.flatnonzero(absolute == CONTROL_WHITE)
    black_squares = np.flatnonzero(absolute == CONTROL_BLACK)
    empty_squares = np.flatnonzero(absolute == CONTROL_EMPTY)
    assert white_squares.size > 0, "the final board has no white-topped square"
    assert black_squares.size > 0, "the final board has no black-topped square"
    assert empty_squares.size > 0, "the final board has no empty square"

    movers = {int(row["to_play"]) for row in rows}
    assert movers == {CONTROL_WHITE, CONTROL_BLACK}, (
        f"the game's rows were played by {movers}; a one-sided game cannot "
        "distinguish a flipped map from an absolute one"
    )

    for index, row in enumerate(rows):
        mover = int(row["to_play"])
        control = _relative(row)
        white_reads = 0 if mover == CONTROL_WHITE else 1
        black_reads = 0 if mover == CONTROL_BLACK else 1

        assert np.all(control[white_squares] == white_reads), (
            f"row {index} (mover {mover}) reads white's squares as "
            f"{np.unique(control[white_squares]).tolist()}, expected "
            f"{white_reads}"
        )
        assert np.all(control[black_squares] == black_reads), (
            f"row {index} (mover {mover}) reads black's squares as "
            f"{np.unique(control[black_squares]).tolist()}, expected "
            f"{black_reads}"
        )
        assert np.all(control[empty_squares] == CONTROL_EMPTY), (
            f"row {index} labelled an empty square as owned; empty is the one "
            "value the flip must leave alone"
        )


def test_every_row_carries_the_same_absolute_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un-flip each row and you get one map -- the final board's, every time.

    The map is a property of the position the game ENDED in, so a row's ply
    must not enter it. Re-absolutising by each row's own ``to_play`` and
    demanding equality catches both failure modes at once: a per-ply
    recomputation (the maps differ) and a missing flip (the black rows
    absolutise to the colour-swapped map).
    """
    absolute, rows = _play_a_game(tmp_path / "game", monkeypatch)

    for index, row in enumerate(rows):
        recovered = _absolutise(_relative(row), int(row["to_play"]))
        assert np.array_equal(recovered, absolute), (
            f"row {index} (mover {int(row['to_play'])}) un-flips to a "
            f"different board than the game's final one\n"
            f"  expected: {absolute.tolist()}\n"
            f"  got:      {recovered.tolist()}"
        )

    # And the flip is genuinely an involution on this data, which is what makes
    # "un-flip and compare" a fair test rather than a definition.
    for row in rows:
        mover = int(row["to_play"])
        reflipped = mover_relative_control(absolute, mover)
        assert np.array_equal(
            np.asarray(reflipped, dtype=np.int64), _relative(row)
        )
