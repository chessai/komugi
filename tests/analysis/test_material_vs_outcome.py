"""Unit suite for ``scripts/analysis/material_vs_outcome.py`` (KOM-243).

Two halves, and they are deliberately anchored differently.

**The decode** is checked against the REAL encoder: every observation a decode
test reads was produced by :func:`komugi.engine.observation.observation_from_cyboard`
(or by :meth:`komugi.engine.game.Game.observation`) over a real
:class:`~komugi.engine.cyboard.Board`, and the expected piece counts are taken
off that board's own typed arrays -- ``cells_arr`` and ``hand_arr`` -- rather
than off a number written down here.  A test that hand-built the planes would
be asserting that the script agrees with the test's idea of the plane map; this
way it asserts that the script agrees with the encoder, which is the thing that
wrote the shards.

**The tabulation** is checked on shards with a PLANTED relationship (lead > 0 =>
win, lead < 0 => loss), so a report that says ``P(win) = 1`` in the positive
bands and ``0`` in the negative is reporting the plant and not an artefact.
Those observations go through the real encoder too, via synthetic FENs whose
piece counts are chosen: positions no legal game would reach, which is exactly
what a planted distribution has to be.

``scripts/`` is outside ``[tool.mypy] files`` and is not a package, so the
script is loaded by path the way ``test_cap_game_values.py`` loads its own.

The suite never opens a real run directory: every shard it reads it wrote.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION, BEGINNER_POSITION, INTRO_POSITION
from komugi.engine.game import Game, RepetitionRule
from komugi.engine.observation import (
    PLANE_MAP_VERSION,
    observation_from_cyboard,
    observation_size,
)
from komugi.selfplay.inference import NetEval
from komugi.selfplay.worker import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.train.shards import FieldSpec, Schema, ShardWriter

ANALYSIS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "scripts" / "analysis"

SCRIPT_NAME: Final[str] = "material_vs_outcome"


def load(name: str) -> ModuleType:
    """Return ``scripts/analysis/<name>.py``, imported once per process."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ANALYSIS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load an analysis script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script: Final[Any] = load(SCRIPT_NAME)

_OBS_SHAPE: Final[tuple[int, ...]] = observation_size()

_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="game_id", dtype="int64", shape=()),
        FieldSpec(name="ply", dtype="int64", shape=()),
        FieldSpec(name="to_play", dtype="int64", shape=()),
        FieldSpec(name="root_value", dtype="float32", shape=()),
        FieldSpec(name="termination", dtype="int64", shape=()),
        FieldSpec(name="observation", dtype="float32", shape=_OBS_SHAPE),
        FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
    )
)
"""The columns the script requires, restated here rather than imported from the
recorder: a test that imported the writer's schema would be asserting that the
script agrees with itself."""

_ONE_HOT: Final[dict[int, tuple[float, float, float]]] = {
    1: (1.0, 0.0, 0.0),
    0: (0.0, 1.0, 0.0),
    -1: (0.0, 0.0, 1.0),
}
"""``wdl_target`` is ``(win, draw, loss)`` in the row's own mover's frame."""

#: ``(fen letter, max count)`` per piece type, white's case, in the order the
#: advanced hand lists them -- the hands these fixtures build are filled from
#: this list so no type ever exceeds its ``MAX_COUNT`` denominator.
_HAND_LETTERS: Final[tuple[tuple[str, int], ...]] = (
    ("D", 4),
    ("N", 3),
    ("J", 2),
    ("W", 2),
    ("R", 2),
    ("S", 2),
    ("F", 2),
    ("A", 2),
    ("M", 1),
    ("G", 1),
    ("I", 1),
    ("C", 1),
    ("K", 1),
    ("T", 1),
)

_MAX_ON_BOARD: Final[int] = 18
"""Two ranks a side, which is all these fixtures need to plant a lead."""


# ---------------------------------------------------------------------------
# fixtures: real boards, synthetic populations
# ---------------------------------------------------------------------------


def _board_counts(board: Any) -> tuple[int, int, int, int]:
    """Return ``(white board, black board, white hand, black hand)`` off `board`.

    Read straight off the typed arrays the encoder itself reads -- the ground
    truth a decode of the planes has to reproduce.  ``cells_arr`` holds the
    piece code of every tier of every tower (``0`` empty), and a code's colour
    is its low bit after the ``+1`` offset; ``hand_arr`` is ``(14, 2)`` counts.
    """
    cells = np.asarray(board.cells_arr)
    codes = cells[cells > 0].astype(np.int64)
    colours = (codes - 1) & 1
    hand = np.asarray(board.hand_arr).astype(np.int64)
    return (
        int(np.count_nonzero(colours == 0)),
        int(np.count_nonzero(colours == 1)),
        int(hand[:, 0].sum()),
        int(hand[:, 1].sum()),
    )


def _rank(letter: str, count: int) -> str:
    """Return one FEN rank holding `count` copies of `letter`."""
    if count >= 9:
        return letter * 9
    return f"{letter * count}{9 - count}" if count else "9"


def _hand_field(letter_case: str, count: int) -> str:
    """Return a FEN hand field holding `count` pieces, no type over its max."""
    if count == 0:
        return "-"
    parts: list[str] = []
    left = count
    for letter, maximum in _HAND_LETTERS:
        if left <= 0:
            break
        take = min(left, maximum)
        parts.append(f"{letter if letter_case == 'w' else letter.lower()}{take}")
        left -= take
    if left:
        raise ValueError(f"cannot hold {count} pieces in one hand")
    return "".join(parts)


def _planted_fen(
    *,
    white_board: int,
    black_board: int,
    white_hand: int = 0,
    black_hand: int = 0,
    turn: str = "w",
) -> str:
    """Return a FEN with exactly the requested piece counts per side.

    Illegal as a *position* (a side may hold ten soldiers here) and legal as a
    *tensor*: the decode under test counts pieces, and a planted distribution
    needs material the rules would never produce.  Parsed by the engine's own
    ``board_from_fen``, so the planes still come from the real encoder.
    """
    if white_board > _MAX_ON_BOARD or black_board > _MAX_ON_BOARD:
        raise ValueError(f"at most {_MAX_ON_BOARD} board pieces a side here")
    black_ranks = [_rank("d", min(black_board, 9)), _rank("d", max(black_board - 9, 0))]
    white_ranks = [_rank("D", max(white_board - 9, 0)), _rank("D", min(white_board, 9))]
    ranks = [*black_ranks, "9", "9", "9", "9", "9", *white_ranks]
    placement = "/".join(ranks)
    hands = f"{_hand_field('w', white_hand)}/{_hand_field('b', black_hand)}"
    return f"{placement} {hands} {turn} 3 - 1"


def _planted_observation(
    *,
    white_board: int,
    black_board: int,
    white_hand: int = 0,
    black_hand: int = 0,
    turn: str = "w",
) -> npt.NDArray[np.float32]:
    """Encode :func:`_planted_fen`'s position through the real encoder."""
    board = board_from_fen(
        _planted_fen(
            white_board=white_board,
            black_board=black_board,
            white_hand=white_hand,
            black_hand=black_hand,
            turn=turn,
        )
    )
    return np.asarray(observation_from_cyboard(board), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _Row:
    """One synthetic shard row: a position, its outcome and its bookkeeping."""

    game_id: int
    ply: int
    white: int
    black: int
    outcome: int
    termination: int = TERMINATION_DECISIVE
    root_value: float = 0.0
    turn: str = "w"

    def observation(self) -> npt.NDArray[np.float32]:
        """The row's planes, through the real encoder."""
        return _planted_observation(
            white_board=self.white, black_board=self.black, turn=self.turn
        )


def _write_shard(
    directory: Path,
    rows: list[_Row],
    *,
    plane_map_version: int = PLANE_MAP_VERSION,
) -> None:
    """Seal `rows` into `directory` as real shards."""
    directory.mkdir(parents=True, exist_ok=True)
    with ShardWriter(directory, _SCHEMA, samples_per_shard=32) as writer:
        for row in rows:
            writer.append(
                {
                    "game_id": torch.tensor(row.game_id, dtype=torch.int64),
                    "ply": torch.tensor(row.ply, dtype=torch.int64),
                    "to_play": torch.tensor(
                        0 if row.turn == "w" else 1, dtype=torch.int64
                    ),
                    "root_value": torch.tensor(row.root_value, dtype=torch.float32),
                    "termination": torch.tensor(row.termination, dtype=torch.int64),
                    "observation": torch.from_numpy(
                        np.ascontiguousarray(row.observation(), dtype=np.float32)
                    ),
                    "wdl_target": torch.tensor(
                        list(_ONE_HOT[row.outcome]), dtype=torch.float32
                    ),
                    "plane_map_version": torch.tensor(
                        plane_map_version, dtype=torch.int64
                    ),
                }
            )


def _planted_rows(*, leads: list[int], plies: list[int]) -> list[_Row]:
    """Rows where the sign of the lead IS the outcome, at several plies.

    The plant the tabulation tests read back: a positive lead always wins, a
    negative lead always loses, a level position always draws.  White is always
    the mover, so "lead" needs no frame argument.
    """
    base = 10
    rows: list[_Row] = []
    for index, lead in enumerate(leads):
        for ply in plies:
            rows.append(
                _Row(
                    game_id=index,
                    ply=ply,
                    white=base + max(lead, 0),
                    black=base + max(-lead, 0),
                    outcome=int(np.sign(lead)),
                    root_value=float(np.sign(lead)) * 0.5,
                )
            )
    return rows


class _StubEvaluator:
    """A batch evaluator answering one fixed mover-relative ``wdl``.

    Fixed on purpose: what is under test here is that the forward pass is run
    over every kept row and pooled into the right band, not what a net believes
    -- a stub whose answer varied would make the two indistinguishable.
    """

    def __init__(self, wdl: tuple[float, float, float]) -> None:
        self._wdl: npt.NDArray[np.float32] = np.asarray(wdl, dtype=np.float32)
        self.batches: list[int] = []

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return the fixed ``wdl`` for every row of `obs`."""
        count = int(obs.shape[0])
        assert legal.shape[0] == count, "one legality row per position"
        self.batches.append(count)
        return NetEval(
            policy_logits=np.zeros((count, 1), dtype=np.float32),
            wdl=np.asarray(
                np.repeat(self._wdl[None, :], count, axis=0), dtype=np.float32
            ),
            moves_left=np.zeros((count,), dtype=np.float32),
        )


def _band(report: dict[str, Any], label: str) -> dict[str, Any]:
    """Return the ``by_lead`` cell labelled `label`."""
    return next(cell for cell in report["by_lead"] if cell["band"] == label)


# ---------------------------------------------------------------------------
# the decode, against the real encoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen", [INTRO_POSITION, BEGINNER_POSITION, ADVANCED_POSITION]
)
def test_the_start_positions_decode_to_the_boards_own_piece_counts(fen: str) -> None:
    """Board and hand, both sides, equal what the engine's arrays hold."""
    board = board_from_fen(fen)
    observed = np.asarray(observation_from_cyboard(board), dtype=np.float32)

    material = script.decode_material(observed[None, :, :, :])

    white_board, black_board, white_hand, black_hand = _board_counts(board)
    assert material.board[0].tolist() == [white_board, black_board]
    assert material.hand[0].tolist() == [white_hand, black_hand]
    assert material.total[0].tolist() == [
        white_board + white_hand,
        black_board + black_hand,
    ]


def test_the_advanced_start_position_decodes_to_a_full_hand_and_a_level_lead() -> None:
    """A hardcoded read, so the parametrised test above cannot pass vacuously.

    The advanced start is an empty board and a hand of every piece a side owns:
    25 in hand, 0 on the board, lead 0.
    """
    observed = np.asarray(
        observation_from_cyboard(board_from_fen(ADVANCED_POSITION)), dtype=np.float32
    )

    material = script.decode_material(observed[None, :, :, :])

    assert material.board[0].tolist() == [0, 0]
    assert material.hand[0].tolist() == [script.PIECES_PER_SIDE] * 2
    assert int(material.lead[0]) == 0
    assert script.PIECES_PER_SIDE == 25


def test_every_piece_of_a_tower_is_counted() -> None:
    """A three-high tower contributes three pieces, not one.

    The reason the decode sums all 84 occupancy planes rather than reading a
    top-of-tower map: a tier-2 piece lives in a different plane than a tier-1
    piece of the same type and colour, and a captured-material read that saw
    only the top would under-count every stack on the board.
    """
    flat = "4M4/9/9/9/9/9/9/9/4m4 -/- w 3 - 1"
    stacked = "4|M:m:M|4/9/9/9/9/9/9/9/4m4 -/- w 3 - 1"

    flat_material = script.decode_material(
        np.asarray(
            observation_from_cyboard(board_from_fen(flat)), dtype=np.float32
        )[None]
    )
    stacked_material = script.decode_material(
        np.asarray(
            observation_from_cyboard(board_from_fen(stacked)), dtype=np.float32
        )[None]
    )

    assert flat_material.board[0].tolist() == [1, 1]
    assert stacked_material.board[0].tolist() == [2, 2]
    assert int(stacked_material.lead[0]) == 0


def test_a_real_game_decodes_after_its_first_plies() -> None:
    """The decode tracks a real engine through real moves.

    An advanced game opens in the draft, so every ply moves a piece out of a
    hand and onto the board: a decode that read the hand planes wrong, or that
    lost a placed piece, disagrees with the board within a handful of plies.
    """
    board = board_from_fen(ADVANCED_POSITION)
    game = Game(board, repetition_rule=RepetitionRule.FULL)
    rng = np.random.Generator(np.random.PCG64(20260917))

    for _ in range(12):
        legal = np.flatnonzero(np.asarray(game.valid_moves(), dtype=np.uint8))
        if legal.size == 0:
            break
        game.play_action(int(legal[rng.integers(legal.size)]))

    observed = np.asarray(game.observation(), dtype=np.float32)
    material = script.decode_material(observed[None, :, :, :])

    white_board, black_board, white_hand, black_hand = _board_counts(board)
    assert white_board + black_board > 0, "the draft must have placed something"
    assert material.board[0].tolist() == [white_board, black_board]
    assert material.hand[0].tolist() == [white_hand, black_hand]
    assert material.total[0].tolist() == [script.PIECES_PER_SIDE] * 2, (
        "the draft moves pieces between hand and board; it creates none"
    )
    assert int(material.to_play[0]) == (0 if board.turn == 0 else 1)


def test_a_planted_lead_decodes_to_that_lead_in_both_frames() -> None:
    """The fixture builder plants what it claims, mover-relative.

    The anchor for every tabulation test below: they read leads off positions
    this builder produced, so the builder is checked against the decode once,
    here, including the frame flip that makes black-to-move's lead the negative
    of white-to-move's on the same material.
    """
    white_to_move = script.decode_material(
        _planted_observation(white_board=12, black_board=5, white_hand=3)[None]
    )
    black_to_move = script.decode_material(
        _planted_observation(
            white_board=12, black_board=5, white_hand=3, turn="b"
        )[None]
    )

    assert white_to_move.total[0].tolist() == [15, 5]
    assert int(white_to_move.lead[0]) == 10
    assert int(black_to_move.lead[0]) == -10


def test_a_hand_plane_that_is_not_constant_is_refused() -> None:
    """The encoder writes hand planes constant; a varying one is damage."""
    observed = _planted_observation(white_board=3, black_board=3, white_hand=4)[None]
    observed[0, script.HAND_PLANE_START, 4, 4] = 0.5

    with pytest.raises(ValueError, match="not constant"):
        script.decode_material(observed)


def test_material_beyond_one_sides_pieces_is_refused() -> None:
    """A decode above ``PIECES_PER_SIDE`` names the bound rather than tabulating."""
    observed = _planted_observation(white_board=18, black_board=3, white_hand=7)[None]
    observed[0, 0] = 1.0  # 81 white marshals: not a plane map this decode speaks

    with pytest.raises(ValueError, match=r"leaves \[0, 25\]"):
        script.decode_material(observed)


def test_a_foreign_plane_map_is_refused_with_both_versions(tmp_path: Path) -> None:
    """A shard from another plane map is fatal, not skipped."""
    _write_shard(
        tmp_path,
        [_Row(game_id=0, ply=0, white=11, black=10, outcome=1)],
        plane_map_version=PLANE_MAP_VERSION + 1,
    )

    with pytest.raises(script.PlaneMapMismatch) as raised:
        script.read_rows([tmp_path])

    assert str(PLANE_MAP_VERSION + 1) in str(raised.value)
    assert str(PLANE_MAP_VERSION) in str(raised.value)


# ---------------------------------------------------------------------------
# the tabulation, on a planted relationship
# ---------------------------------------------------------------------------


def test_a_planted_relationship_is_reported_as_certainty_in_the_bands(
    tmp_path: Path,
) -> None:
    """lead > 0 => win plants ``P(win) = 1`` above and ``0`` below.

    The test the whole script exists to be trusted by: on shards where material
    decides the game, this report says material decides the game.  A report that
    smeared the bands (a mis-signed lead, a lost frame flip, an off-by-one
    ``digitize``) cannot produce these exact 1.0/0.0 corners.
    """
    _write_shard(
        tmp_path,
        _planted_rows(leads=[-8, -3, 0, 3, 8], plies=[10, 50, 130]),
    )

    rows = script.read_rows([tmp_path])
    report = script.tabulate(
        rows, lead_edges=(-5, -2, 2, 5), ply_edges=(0, 40, 120)
    )

    assert report["rows"] == 15
    assert _band(report, "-5..-2")["p_loss"] == 1.0
    assert _band(report, "-5..-2")["p_win"] == 0.0
    assert _band(report, ">=5")["p_win"] == 1.0
    assert _band(report, ">=5")["p_loss"] == 0.0
    assert _band(report, "-2..2")["p_draw"] == 1.0
    assert _band(report, ">=5")["mean_root_value"] == pytest.approx(0.5)
    assert _band(report, "<-5")["mean_root_value"] == pytest.approx(-0.5)
    # Not 1.0, and it should not be: the outcome has three levels against the
    # lead's five, so a perfectly monotone plant still ties ranks inside the
    # win and loss blocks.  Strong and positive in every populated ply band is
    # the falsifiable statement -- and the mirrored plant below negates it.
    assert report["spearman_overall"] > 0.9
    assert all(
        cell["spearman_lead_outcome"] > 0.9
        for cell in report["spearman_by_ply_band"]
        if cell["n"]
    )
    mirrored = script.tabulate(
        script.read_rows([tmp_path]), lead_edges=(-5, -2, 2, 5), ply_edges=(0,)
    )
    flipped = script.Rows(
        game_id=rows.game_id,
        ply=rows.ply,
        to_play=rows.to_play,
        lead=-rows.lead,
        mover_material=rows.opponent_material,
        opponent_material=rows.mover_material,
        outcome=rows.outcome,
        root_value=rows.root_value,
        termination=rows.termination,
        head_value=None,
        to_play_mismatch=0,
        shards=rows.shards,
    )
    assert script.spearman(flipped.lead, flipped.outcome) == pytest.approx(
        -mirrored["spearman_overall"]
    )
    assert report["by_lead_ply"], "the lead x ply cross-tabulation is emitted"
    assert {cell["ply_band"] for cell in report["by_lead_ply"]} == set(
        report["ply_band_labels"]
    )


def test_a_material_blind_population_reports_no_correlation(tmp_path: Path) -> None:
    """The null the ticket's hypothesis predicts: leads, but no signal.

    Every lead is paired with one win and one loss, so ``P(win)`` is 0.5 in
    every band and Spearman is 0 -- the shape the report has to be able to
    produce, or "material does not predict the outcome in self-play" would be
    unfalsifiable by it.
    """
    rows: list[_Row] = []
    for index, lead in enumerate([-8, -3, 3, 8]):
        for outcome in (1, -1):
            rows.append(
                _Row(
                    game_id=index * 2 + (outcome > 0),
                    ply=60,
                    white=10 + max(lead, 0),
                    black=10 + max(-lead, 0),
                    outcome=outcome,
                )
            )
    _write_shard(tmp_path, rows)

    report = script.tabulate(
        script.read_rows([tmp_path]), lead_edges=(-5, -2, 2, 5), ply_edges=(0, 40, 120)
    )

    assert _band(report, ">=5")["p_win"] == 0.5
    assert _band(report, "<-5")["p_win"] == 0.5
    assert report["spearman_overall"] == pytest.approx(0.0)


def test_the_material_leader_at_ply_100_is_read_over_decisive_games_only(
    tmp_path: Path,
) -> None:
    """Draws and games that never reached the ply are out; level ones are apart.

    Four games reach ply 100: two decisive ones the leader wins, one decisive
    one the leader loses, one drawn one (excluded -- a draw has no winner to
    agree with).  A fifth game ends before the ply and contributes nothing.
    """
    rows = [
        _Row(game_id=1, ply=100, white=14, black=10, outcome=1),
        _Row(game_id=2, ply=100, white=14, black=10, outcome=1),
        _Row(game_id=3, ply=100, white=10, black=14, outcome=1),
        _Row(
            game_id=4,
            ply=100,
            white=14,
            black=10,
            outcome=0,
            termination=TERMINATION_DRAW,
        ),
        _Row(game_id=5, ply=100, white=10, black=10, outcome=1),
        _Row(game_id=6, ply=40, white=18, black=5, outcome=-1),
    ]
    _write_shard(tmp_path, rows)

    at_ply = script.lead_at_ply(script.read_rows([tmp_path]), ply=100)

    assert at_ply["decisive_games_reaching_ply"] == 4
    assert at_ply["level_at_ply"] == 1
    assert at_ply["scored"] == 3
    assert at_ply["leader_won"] == pytest.approx(2.0 / 3.0)


def test_the_bands_are_half_open_and_labelled_from_the_edges() -> None:
    """``--bands -20,-10`` means ``< -20``, ``[-20, -10)`` and ``>= -10``."""
    labels = script.band_labels((-20, -10))
    assert labels == ["<-20", "-20..-10", ">=-10"]

    values = np.asarray([-21, -20, -11, -10, 0], dtype=np.int64)
    assert script.band_of(values, (-20, -10)).tolist() == [0, 1, 1, 2, 2]


def test_the_edges_parser_refuses_an_unordered_list() -> None:
    """Band edges must be strictly increasing, or the bands are not bands."""
    assert script.parse_edges("-5,0,5") == (-5, 0, 5)
    with pytest.raises(Exception, match="increasing"):
        script.parse_edges("5,0,-5")


# ---------------------------------------------------------------------------
# the checkpoint's head, and the CLI
# ---------------------------------------------------------------------------


def test_a_checkpoints_head_values_are_tabulated_beside_the_outcomes(
    tmp_path: Path,
) -> None:
    """``--ckpt``'s forward pass covers every row and pools into the bands."""
    shard_dir = tmp_path / "shards"
    _write_shard(shard_dir, _planted_rows(leads=[-8, 8], plies=[10, 50]))
    evaluator = _StubEvaluator((0.7, 0.2, 0.1))

    rows = script.read_rows([shard_dir], evaluator=evaluator, batch=3)
    report = script.tabulate(rows, lead_edges=(-5, 5), ply_edges=(0, 40))

    assert sum(evaluator.batches) == 4, "every kept row is forward-passed once"
    assert max(evaluator.batches) <= 3, "the pass respects the batch size"
    assert report["head_values"] is True
    assert _band(report, ">=5")["mean_head_value"] == pytest.approx(0.6)
    assert _band(report, "<-5")["mean_head_value"] == pytest.approx(0.6)
    assert _band(report, ">=5")["p_win"] == 1.0, (
        "the head is blind to a lead the outcomes are certain on -- the shape "
        "KOM-243 is looking for"
    )


def test_the_cli_writes_valid_json_end_to_end(tmp_path: Path) -> None:
    """``--shards ... --out ...`` runs over a real shard tree and writes JSON."""
    shard_dir = tmp_path / "shards" / "worker-0"
    _write_shard(shard_dir, _planted_rows(leads=[-6, 0, 6], plies=[20, 100]))
    out = tmp_path / "results" / "material_vs_outcome.json"

    status = script.main(
        [
            "--shards",
            str(tmp_path / "shards"),
            "--out",
            str(out),
            # Attached, because the edges start with a minus sign: argparse
            # reads a detached "-5,..." as an option, and the script's help
            # says so.
            "--bands=-5,-2,2,5",
            "--ply-bands=0,40,120",
        ]
    )

    assert status == 0
    report = json.loads(out.read_text())
    assert report["plane_map_version"] == PLANE_MAP_VERSION
    assert report["rows"] == 6
    assert report["games"] == 3
    assert report["shards"] == 1
    assert report["to_play_mismatch"] == 0
    assert report["ckpt"] is None
    assert report["head_values"] is False
    assert report["lead_bands"] == [-5, -2, 2, 5]
    assert report["board_planes"].startswith("0-83")
    assert report["hand_planes"].startswith("84-111")
    assert _band(report, ">=5")["p_win"] == 1.0
    assert report["lead_at_ply_100"]["scored"] == 2
    assert "Spearman" in script.format_report(report)


def test_the_cli_forwards_the_checkpoint_it_is_given(tmp_path: Path) -> None:
    """``--ckpt`` reaches the report through the factory seam, with its device."""
    shard_dir = tmp_path / "shards"
    _write_shard(shard_dir, _planted_rows(leads=[-6, 6], plies=[30]))
    out = tmp_path / "with_head.json"
    seen: list[tuple[Path, str]] = []

    def factory(path: Path, device: str) -> Any:
        seen.append((path, device))
        return _StubEvaluator((0.1, 0.2, 0.7))

    status = script.main(
        [
            "--shards",
            str(shard_dir),
            "--out",
            str(out),
            "--ckpt",
            str(tmp_path / "checkpoint.pt"),
            "--device",
            "cpu",
            "--bands=-5,5",
        ],
        evaluator_factory=factory,
    )

    assert status == 0
    assert seen == [(tmp_path / "checkpoint.pt", "cpu")]
    report = json.loads(out.read_text())
    assert report["ckpt"] == str(tmp_path / "checkpoint.pt")
    assert report["device"] == "cpu"
    assert _band(report, ">=5")["mean_head_value"] == pytest.approx(-0.6)


def test_an_empty_tree_reports_no_rows_rather_than_failing(tmp_path: Path) -> None:
    """A directory with no shards is an empty report, not a crash."""
    out = tmp_path / "empty.json"

    assert script.main(["--shards", str(tmp_path), "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["rows"] == 0
    assert report["games"] == 0
    assert report["spearman_overall"] is None
    assert all(cell["p_win"] is None for cell in report["by_lead"])
