"""komugi.net.onnx_export (KOM-1186 U3d): the ONNX export module the release
tooling uses, and its self-generated parity positions.
"""

from __future__ import annotations

import ast
import hashlib
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game, draft_root_mask
from komugi.net import onnx_export

#: Seed the byte-identity export (test a) and the legality replay (test b)
#: build their fresh net / playout from.
_EXPORT_SEED = 424242
#: Pinned at the base commit, BEFORE this module existed, from the exporter
#: of that commit: ``fresh_small_net(424242)`` exported with ``batch=1``.
_BASE_EXPORT_SHA256 = (
    "1686f2e4d88e214e6416047109624364d4c08776242becb15a20cd667d071b52"
)
#: Every top-level module :mod:`komugi.net.onnx_export` may import.
_ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "dataclasses", "pathlib", "random", "typing",
     "numpy", "onnx", "torch", "komugi"}
)


def test_export_byte_identical_to_base_commit_exporter() -> None:
    net = onnx_export.fresh_small_net(_EXPORT_SEED)
    with tempfile.TemporaryDirectory() as tmp_str:
        out = Path(tmp_str) / "model.onnx"
        onnx_export.export_net(net, out, batch=1)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest == _BASE_EXPORT_SHA256


# --- parity_positions ---------------------------------------------------


def _replay_oracle(count: int, seed: int) -> list[onnx_export.ParityPosition]:
    """Independent re-implementation of the seeded random playout, built
    from the public engine API rather than calling
    :func:`onnx_export.parity_positions`.

    Asserts the legal-move mask is non-empty every time it is about to draw
    a move -- so this oracle can only ever record a position that either is
    the Advanced start or was reached by playing an action out of
    ``Game.valid_moves()`` -- and returns the same ``(phase, observation)``
    pairs :func:`onnx_export.parity_positions` is expected to.
    """
    needed_draft = count // 2
    needed_play = count - needed_draft
    rng = random.Random(seed)
    game = Game(board=board_from_fen(ADVANCED_POSITION))
    draft_rows: list[np.ndarray] = []
    play_rows: list[np.ndarray] = []
    while len(draft_rows) < needed_draft or len(play_rows) < needed_play:
        obs = np.ascontiguousarray(game.observation(), dtype=np.float32)
        in_draft = game.in_draft(0) or game.in_draft(1)
        bucket, limit = (
            (draft_rows, needed_draft) if in_draft else (play_rows, needed_play)
        )
        if len(bucket) < limit:
            bucket.append(obs)
        mask = draft_root_mask(
            game._board, game.valid_moves(), onnx_export._DRAFT_MIN_PLACEMENTS  # noqa: SLF001
        )
        legal = np.flatnonzero(mask)
        assert legal.size > 0, "oracle ran out of legal moves before collecting enough positions"
        action = int(rng.choice(legal.tolist()))
        game.play_action(action)
    return [onnx_export.ParityPosition("draft", obs) for obs in draft_rows] + [
        onnx_export.ParityPosition("play", obs) for obs in play_rows
    ]


def test_parity_positions_deterministic_for_a_seed() -> None:
    first = onnx_export.parity_positions(8, seed=1)
    second = onnx_export.parity_positions(8, seed=1)
    assert [p.phase for p in first] == [p.phase for p in second]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.observation, b.observation)


def test_parity_positions_differ_across_seeds() -> None:
    first = onnx_export.parity_positions(8, seed=1)
    second = onnx_export.parity_positions(8, seed=2)
    assert any(
        not np.array_equal(a.observation, b.observation)
        for a, b in zip(first, second)
    )


def test_parity_positions_cover_both_phases() -> None:
    positions = onnx_export.parity_positions(8, seed=3)
    phases = {p.phase for p in positions}
    assert phases == {"draft", "play"}
    assert sum(p.phase == "draft" for p in positions) == 4
    assert sum(p.phase == "play" for p in positions) == 4


def test_parity_positions_yields_only_legal_positions_matching_oracle() -> None:
    positions = onnx_export.parity_positions(8, seed=5)
    oracle = _replay_oracle(8, seed=5)
    assert [p.phase for p in positions] == [p.phase for p in oracle]
    for got, want in zip(positions, oracle):
        np.testing.assert_array_equal(got.observation, want.observation)


def test_parity_positions_rejects_count_below_two() -> None:
    with pytest.raises(ValueError):
        onnx_export.parity_positions(1, seed=1)


# --- dependencies ---------------------------------------------------------


def test_onnx_export_module_imports_only_komugi_torch_onnx_numpy_stdlib() -> None:
    source = Path(onnx_export.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            if node.module:
                names = [node.module]
        for name in names:
            assert name.split(".")[0] in _ALLOWED_IMPORT_ROOTS, name
