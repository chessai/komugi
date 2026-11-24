"""DRAFTMIN through the self-play worker: no 終 mass in a recorded target.

The root view of :class:`komugi.selfplay.adapter.GameState` is only worth
having if it reaches the **training data**, and the path it takes to get there
is not obvious: the worker never writes a mask into a row. It records the
search's ``policy_target``, and that target is built over the root node's legal
set -- so "終 is withheld at the root" becomes "終 carries no target mass" only
if the masked view survives every step between the worker's state and the root
node the improved policy is read off.

These nodes assert the end of that path, with the real engine, the real search
and a stub network, because every intermediate step is a place the property
could be lost:

* the search clones the state before building its root node, so a clone that
  dropped the view would leave the target unmasked;
* the worker advances ONE state through a whole game, so a ``play`` that
  dropped the view would switch DRAFTMIN off after ply 0 of every game;
* the boosted stub below wants 終 above everything else, so a target that is
  merely *unlikely* to name it is not the same as one that cannot.

Advanced mode, two placements in: the position campaign 2's draft attractor
lives at, and a position where the engine calls 終 legal -- which every node
here asserts before claiming the mask removed it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import SHU_BLOCK
from komugi.engine.game import Game, draft_placements
from komugi.selfplay.adapter import GameState
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig
from komugi.train.shards import ShardReader, ShardWriter

_C_START: Final[int] = SHU_BLOCK[0]
"""First action id of block C -- "arata + draft-finish", the 終 block."""

_THRESHOLD: Final[int] = 10
"""v1's first curriculum rung, and this module's threshold."""

_OPENING_PLACEMENTS: Final[int] = 2
"""Placements each side has made before the worker's first searched ply."""

_CAP: Final[int] = 40
"""Turn cap for the fixture game: long enough for both drafts to pass the
threshold (eight more placements a side, sixteen plies), short enough that a
two-simulation search finishes the game in seconds."""

_RUN_SEED: Final[int] = 20260910


class _DraftedGame(Game):
    """Advanced mode, two placements a side already made, a short turn cap.

    ``_START_MODE`` is a class attribute rather than a
    ``Game.set_start_mode`` call for the reason
    ``tests/selfplay/unit/test_staged_targets.py`` gives: the setter mutates
    the shared base class for the rest of the process, and this way the
    override reaches this class and nothing else.

    The opening placements are played in ``__init__`` because the worker builds
    its game with no arguments (``Game(repetition_rule=...)``) and there is no
    other seam to hand it a position. Only the no-board construction drafts:
    ``Game.clone`` passes a board, and it also builds a plain ``Game``, so no
    clone the search makes re-runs any of this.

    Every opening placement is the first legal id BELOW block C -- a placement
    that keeps the draft open -- so the position the worker starts from is one
    where both sides are still drafting and each has exactly two pieces down.
    """

    _START_MODE: str = "advanced"

    def __init__(self, board: Any = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(board, *args, **kwargs)
        if board is not None:
            return
        for _ in range(2 * _OPENING_PLACEMENTS):
            mask = np.asarray(self.valid_moves(), dtype=np.bool_)
            legal = np.flatnonzero(mask[:_C_START])
            assert legal.size > 0, "no draft-continuing placement is legal"
            self.play_action(int(legal[0]))

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _CAP


class _ShuBoostEvaluator:
    """A stub network that wants 終 above every other move.

    Uniform logits with one exception: every id in block C is boosted by
    :data:`_BIAS`. The boost is what makes these nodes sharp. An unmasked run
    of this fixture puts most of its root target mass on the draft-finish
    block -- asserted below, as the control -- so a masked run's *exact zero*
    is the mask's doing and not the network's indifference.

    Deterministic to the bit, so the masked and unmasked runs differ in one
    thing only: the threshold.
    """

    _BIAS: Final[float] = 16.0

    def __init__(self, actions: int) -> None:
        self.actions = actions
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one boosted evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        self.calls += 1
        logits = np.zeros((rows, self.actions), dtype=np.float32)
        logits[:, _C_START:] = self._BIAS
        return NetEval(
            policy_logits=logits,
            wdl=np.tile(np.asarray((0.4, 0.2, 0.4), dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


def _config(min_placements: int, min_placements_max: int = 0) -> SelfPlayConfig:
    """A two-simulation self-play config carrying one DRAFTMIN threshold.

    `min_placements_max` defaults to 0 -- the fixed-threshold path -- so every
    node written before KOM-246 asks for exactly the config it always did.
    """
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=0.0,
        draw_score=0.2,
        min_placements=min_placements,
        min_placements_max=min_placements_max,
    )


def _play(
    directory: Path,
    min_placements: int,
    monkeypatch: pytest.MonkeyPatch,
    min_placements_max: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Play one game at `min_placements` and return its recorded rows."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _DraftedGame)
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _ShuBoostEvaluator(num_actions()),
        _config(min_placements, min_placements_max),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=_RUN_SEED,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )
    written = worker.play_game(0)
    writer.close()

    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    rows = list(reader)
    assert written == len(rows) > 0
    return rows


_GOLDEN_DIGEST: Final[str] = (
    "fc901c24a8504b6215e32dad8fde7e06b0d1e9a37651882299d701822cd2b1f3"
)
"""sha256 of the masked fixture's rows, PINNED ON THE PRE-KOM-246 BASE.

The byte-identity oracle review F3 asked for. Every other determinism node in
the suite compares two runs of the *same* code, which passes just as happily
when both runs are wrong; this constant was computed by running
:func:`test_the_masked_fixture_rows_match_the_golden_digest` on the base commit
before a single line of KOM-246 was written, so "``min_placements_max = 0`` is
today's behaviour to the byte" is a claim about the code that shipped rather
than about the code under test.
"""


def _rows_digest(rows: list[dict[str, torch.Tensor]]) -> str:
    """Return a sha256 over `rows`, tensor bytes and all.

    Row order, then key order within a row (sorted, so a schema whose dict
    happens to be built in another order still digests the same), then the
    key's own bytes before its tensor's -- the key is in the hash so that two
    columns of equal dtype and shape cannot swap names unnoticed.
    ``contiguous()`` because ``tobytes`` of a non-contiguous tensor's storage
    would hash padding rather than values.
    """
    digest = hashlib.sha256()
    for row in rows:
        for key in sorted(row):
            digest.update(key.encode("utf-8"))
            digest.update(row[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_the_masked_fixture_rows_match_the_golden_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded rows of the seeded fixture are the pre-KOM-246 bytes.

    Both the default path (``min_placements_max = 0``, no draw made) and the
    degenerate range (``min_placements_max == min_placements``, a draw whose
    only possible answer is ``(10, 10)``) reproduce the bytes the fixed
    threshold produced before the per-colour draw existed -- which is the
    whole determinism claim, asserted against code that no longer exists
    rather than against a second run of the code under test.
    """
    for max_edge in (0, _THRESHOLD):
        digest = _rows_digest(
            _play(
                tmp_path / f"max-{max_edge}",
                _THRESHOLD,
                monkeypatch,
                min_placements_max=max_edge,
            )
        )
        assert digest == _GOLDEN_DIGEST, (
            f"min_placements_max={max_edge}: computed digest {digest}"
        )


def test_the_default_path_makes_no_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the knob off the worker does not reach the draw at all.

    The "no new code on the default path" doctrine, asserted rather than
    asserted-about: a :func:`~komugi.selfplay.records.draft_thresholds` that
    explodes on contact leaves a ``min_placements_max = 0`` game producing the
    golden bytes, so the branch in ``play_game`` is known to be untaken and
    not merely believed to be harmless.
    """

    def _explode(*args: object, **kwargs: object) -> tuple[int, int]:
        raise AssertionError(
            "play_game drew draft thresholds with min_placements_max = 0"
        )

    monkeypatch.setattr("komugi.selfplay.worker.draft_thresholds", _explode)
    digest = _rows_digest(_play(tmp_path, _THRESHOLD, monkeypatch))
    assert digest == _GOLDEN_DIGEST, f"computed digest {digest}"


def _shu_mass(row: dict[str, torch.Tensor]) -> float:
    """Return how much of a row's policy target sits on the 終 block."""
    return float(row["policy_target"][_C_START:].sum())


def test_a_masked_draft_ply_records_no_shu_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first searched ply's target is exactly zero on every 終 id.

    Zero, not small: the improved policy spreads mass over every action in the
    root node's legal set, so an id that is still legal there gets *some* mass
    however unattractive it is. An exact zero is only reachable by the id not
    being in that set at all, which is the claim.
    """
    start = _DraftedGame()
    assert start.in_draft(int(start.player)), "ply 0 must be a draft ply"
    placed = draft_placements(start._board, int(start.player))  # noqa: SLF001
    assert placed == _OPENING_PLACEMENTS
    assert np.asarray(start.valid_moves(), dtype=np.bool_)[_C_START:].any(), (
        "the engine must call 終 legal at the position ply 0 searches"
    )

    rows = _play(tmp_path / "masked", _THRESHOLD, monkeypatch)
    first = rows[0]
    assert _shu_mass(first) == 0.0
    assert int(first["action"]) < _C_START


def test_without_the_threshold_the_same_ply_is_full_of_shu_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: nothing but the threshold changes, and the zero disappears.

    The stub boosts block C by 16 logits, so an unmasked root both considers
    and prefers the draft-finish -- which is campaign 2's attractor in
    miniature, and is what makes the masked run's exact zero meaningful rather
    than a property of a network that never liked 終 anyway.
    """
    rows = _play(tmp_path / "unmasked", 0, monkeypatch)
    assert _shu_mass(rows[0]) > 0.0


def test_the_mask_holds_at_every_root_of_the_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just ply 0: every drafting row below the threshold records no 終.

    The worker advances ONE state through the whole game, so this is the node
    that fails if ``play`` drops the root view: DRAFTMIN would then be alive
    for ply 0 and dead for every ply after it, which no assertion about the
    first row can see.

    The per-row placement count is recovered by REPLAYING the recorded
    ``action`` column into a fresh engine, rather than by trusting a count the
    test kept: the replay reaches each row's position by a path the engine
    itself calls legal, so a column that is off by a ply cannot quietly satisfy
    it.
    """
    rows = _play(tmp_path / "every-root", _THRESHOLD, monkeypatch)

    replay = _DraftedGame()
    checked = 0
    lifted = 0
    for row in rows:
        side = int(replay.player)
        assert int(row["to_play"]) == side, "the replay and the rows disagree"
        if replay.in_draft(side):
            placed = draft_placements(replay._board, side)  # noqa: SLF001
            if placed < _THRESHOLD:
                assert _shu_mass(row) == 0.0, (
                    f"ply {int(row['ply'])}: the mover has placed {placed} of "
                    f"{_THRESHOLD} and its target still names 終"
                )
                checked += 1
            else:
                lifted += 1
        replay.play_action(int(row["action"]))

    assert checked >= 4, "the fixture stopped covering the masked plies"
    assert lifted > 0, (
        "no row reached the threshold: the game never showed that the mask "
        "lifts, so a permanent ban would pass this node"
    )


def test_the_worker_configures_a_root_and_the_engine_sees_none_of_it() -> None:
    """``valid_moves`` and ``win_state`` answer as they would with no threshold.

    The worker's root is a VIEW of the engine's position. Built the way
    ``play_game`` builds it, it hides 終 from the search while the engine it
    wraps goes on calling 終 legal, and the adjudication is untouched -- so no
    rule, no termination and no target outside the policy column can depend on
    this feature.
    """
    game = _DraftedGame()
    config = _config(_THRESHOLD)
    raw = np.asarray(game.valid_moves(), dtype=np.bool_)
    outcome = np.asarray(game.win_state(), dtype=np.uint8)
    assert raw[_C_START:].any(), "the engine must call 終 legal here"

    state = GameState(
        game, config.draw_score, min_placements=config.min_placements, is_root=True
    )
    assert not state.legal_mask()[_C_START:].any()

    assert np.array_equal(np.asarray(game.valid_moves(), dtype=np.bool_), raw)
    assert np.array_equal(np.asarray(game.win_state(), dtype=np.uint8), outcome)
