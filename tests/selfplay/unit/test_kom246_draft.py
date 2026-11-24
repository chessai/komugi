"""KOM-246: the per-colour draft threshold, drawn per game from its own stream.

Campaign 6's draft sits at one length because every game runs at one threshold:
white closes at 12 in 84% of games and black prices its reply against a constant.
The lever is a threshold DRAWN per game and per colour --
:func:`komugi.selfplay.records.draft_thresholds` -- and masked at the root by
the side to move's own entry.

Three separate claims live here, and they fail for different reasons:

* the draw is a **pure function** of six integers (the offline census recomputes
  any recorded game's pair from ``game_id`` alone, so a draw that consulted a
  clock or a counter would make every census answer unverifiable);
* its stream is **disjoint** from the game's replay streams -- ``rng`` and
  :func:`~komugi.selfplay.records.move_seed` -- so turning the lever on does not
  reshuffle the searches of a run at fixed seed;
* the worker actually **builds its root from the drawn pair**, which no
  assertion about the function alone can see.

The engine's asymmetry (design review F1: black's 終 ends both drafts, white's
only its own) is asserted in ``test_min_placements.py``, where the board is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import SHU_BLOCK
from komugi.engine.game import Game
from komugi.selfplay.adapter import GameState
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import (
    BOOK_BUILD_DOMAIN,
    BOOK_SELECT_DOMAIN,
    DRAFT_THRESHOLD_DOMAIN,
    SCHEMA,
    draft_thresholds,
    game_seed,
    move_seed,
    num_actions,
)
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig
from komugi.train.shards import ShardWriter

_C_START: Final[int] = SHU_BLOCK[0]
"""First action id of block C -- "arata + draft-finish", the 終 block."""

_RUN_SEED: Final[int] = 20260910
"""The seed ``test_min_placements_worker`` plays its fixture game under."""

_LO: Final[int] = 4
_HI: Final[int] = 18
"""The pre-registered range (``tickets/KOM-246.md``: 10 is not a measured
optimum)."""

_GAMES: Final[int] = 200
"""Games surveyed for coverage. Large enough that a 15-value range is covered
by both colours with room to spare, small enough to stay a unit test."""

_OPENING_PLACEMENTS: Final[int] = 2
_CAP: Final[int] = 40


class _DraftedGame(Game):
    """Advanced mode, two placements a side already made, a short turn cap.

    A minimal copy of ``test_min_placements_worker``'s fixture rather than an
    import of it: the two modules are not a package, and a cross-module import
    between test files would make this one's collection depend on the other's
    path. ``_START_MODE`` is a class attribute for the reason
    ``tests/selfplay/unit/test_staged_targets.py`` gives -- the setter mutates
    the shared base for the rest of the process.
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
    """A deterministic stub network that wants 終 above every other move."""

    _BIAS: Final[float] = 16.0

    def __init__(self, actions: int) -> None:
        self.actions = actions

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one boosted evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        logits = np.zeros((rows, self.actions), dtype=np.float32)
        logits[:, _C_START:] = self._BIAS
        return NetEval(
            policy_logits=logits,
            wdl=np.tile(np.asarray((0.4, 0.2, 0.4), dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


# --------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------


def test_draft_thresholds_is_pure_and_in_range() -> None:
    """Same six arguments, same pair -- and every draw inside ``[lo, hi]``.

    Purity is the property the offline census rests on (design review F2): the
    pair of a recorded game is recomputed from ``unpack_game_id`` and the
    iteration seed, in another process, months later. A draw that advanced any
    state -- ``SeedSequence.spawn``'s counter, a module-level generator -- would
    answer differently the second time and the census would be recomputing
    something that never happened.
    """
    args = (_RUN_SEED, 1, 0, 0, _LO, _HI)
    assert draft_thresholds(*args) == draft_thresholds(*args)

    for game_index in range(_GAMES):
        pair = draft_thresholds(_RUN_SEED, 1, 0, game_index, _LO, _HI)
        assert len(pair) == 2
        for threshold in pair:
            assert _LO <= threshold <= _HI, (
                f"game {game_index} drew {pair}, outside [{_LO}, {_HI}]"
            )


def test_two_hundred_games_cover_the_range_for_both_colours_independently() -> None:
    """Both colours reach every value, and the joint draw is not the diagonal.

    Coverage alone would pass for a generator that drew one number and used it
    for both sides -- which is the lever NOT being pulled, since an asymmetric
    pair is the entire point (K5 measures the first-closer differential). So
    the disagreement rate is asserted too, and in both directions, because a
    pair that were always ``N_w < N_b`` would be a systematic handicap rather
    than a randomised threshold.
    """
    pairs = [
        draft_thresholds(_RUN_SEED, 1, 0, game_index, _LO, _HI)
        for game_index in range(_GAMES)
    ]
    expected = set(range(_LO, _HI + 1))
    assert {pair[0] for pair in pairs} == expected
    assert {pair[1] for pair in pairs} == expected

    differing = [pair for pair in pairs if pair[0] != pair[1]]
    assert len(differing) >= 150, (
        f"only {len(differing)} of {_GAMES} games drew two different "
        f"thresholds: the colours are not being drawn independently"
    )
    assert sum(1 for pair in pairs if pair[0] < pair[1]) >= 40
    assert sum(1 for pair in pairs if pair[0] > pair[1]) >= 40


def test_the_threshold_stream_is_disjoint_from_the_move_and_book_streams() -> None:
    """F5: the domain constant cannot land on a ply's stream or a book's.

    The composition is ``(*game_seed.spawn_key, DOMAIN)``, so disjointness is a
    claim about the KEY -- and it holds only while every element is below
    ``2**32`` (numpy splits a larger integer into two 32-bit words, which would
    let a fatter key hash onto this one). The domain is checked against that
    bound here, and the streams are compared as the state they actually
    generate rather than as constants that look different.
    """
    seeds = game_seed(_RUN_SEED, 1, 0, 0)
    ours = np.random.SeedSequence(
        entropy=seeds.entropy,
        spawn_key=(*seeds.spawn_key, DRAFT_THRESHOLD_DOMAIN),
    ).generate_state(2)

    for other in (0, 511, BOOK_BUILD_DOMAIN, BOOK_SELECT_DOMAIN):
        theirs = np.random.SeedSequence(
            entropy=seeds.entropy, spawn_key=(*seeds.spawn_key, other)
        ).generate_state(2)
        assert not np.array_equal(ours, theirs), (
            f"the draft-threshold stream collides with the stream keyed "
            f"{other!r}"
        )

    assert DRAFT_THRESHOLD_DOMAIN < 2**32, (
        "a key element at or above 2**32 is split into two words and the "
        "disjointness argument above stops holding"
    )
    assert DRAFT_THRESHOLD_DOMAIN not in (BOOK_BUILD_DOMAIN, BOOK_SELECT_DOMAIN)
    # Against the ply stream in the WIDTH that stream is generated at:
    # ``move_seed`` takes a 64-bit word out of its own sequence, so comparing
    # it to a 32-bit word of ours would be an accident of truncation either
    # way round (review F3).
    ours_64 = np.random.SeedSequence(
        entropy=seeds.entropy,
        spawn_key=(*seeds.spawn_key, DRAFT_THRESHOLD_DOMAIN),
    ).generate_state(1, dtype=np.uint64)[0]
    assert int(ours_64) != move_seed(seeds, 0)


def test_bad_edges_are_refused() -> None:
    """A range that is empty, unreachable, or includes the off switch.

    ``lo = 0`` is refused rather than clamped (design review F7): 0 means the
    mask is OFF, so a range containing it would run a nineteenth of the seats
    with no draft mask at all -- a silently mixed population, which is the one
    thing a lever designed to spread the population must not be.

    Refused at BOTH ends of the chain, which is the point of the second half:
    the config refuses the pair at load, so no run ever reaches a game with
    such a range; the draw refuses it too, so a caller that composes edges from
    somewhere other than a ``SelfPlayConfig`` cannot slip past the one gate.
    """
    with pytest.raises(ValueError, match="lo=0"):
        draft_thresholds(_RUN_SEED, 1, 0, 0, 0, 18)
    with pytest.raises(ValueError, match="lo=5, hi=4"):
        draft_thresholds(_RUN_SEED, 1, 0, 0, 5, 4)
    with pytest.raises(ValueError, match="hi=26"):
        draft_thresholds(_RUN_SEED, 1, 0, 0, 4, 26)

    with pytest.raises(ValueError, match="min_placements"):
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            min_placements=0,
            min_placements_max=18,
        )


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------


def _config(min_placements: int, min_placements_max: int) -> SelfPlayConfig:
    """A two-simulation self-play config carrying the two edges."""
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


def _roots_built(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    min_placements: int,
    min_placements_max: int,
    game_index: int = 0,
) -> list[tuple[int, int]]:
    """Play `game_index` and return the thresholds of every root it built.

    The recording is a SUBCLASS of the real adapter rather than a stub, so the
    game the worker plays is the game it would have played: the assertion is
    about what ``play_game`` passed in, not about a state that never searched.
    """
    built: list[tuple[int, int]] = []

    class _RecordingGameState(GameState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self.min_placements)

    monkeypatch.setattr("komugi.selfplay.worker.Game", _DraftedGame)
    monkeypatch.setattr("komugi.selfplay.worker.GameState", _RecordingGameState)
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
    assert worker.play_game(game_index) > 0
    writer.close()
    assert len(built) == 1, f"play_game built {len(built)} roots, expected one"
    return built


def test_the_worker_builds_its_root_from_the_drawn_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: the root's thresholds ARE the function's answer for that game.

    The end of the chain the census walks backwards: ``game_id`` decodes to
    ``(worker_id, incarnation, game_index)``, those three plus the iteration
    seed and the two edges name a pair, and this node is what makes that pair
    the one the game was actually played under.

    Played at TWO game indices, because one index cannot tell "the root carries
    the drawn pair" from "the root carries a pair drawn once per worker": game
    3 must get game 3's answer, which is a different pair from game 0's -- and
    that the two differ is asserted before it is relied on, so a seed at which
    they collide says to pick another index rather than passing vacuously.
    """
    drawn = _roots_built(tmp_path / "drawn", monkeypatch, _LO, _HI)
    expected = draft_thresholds(_RUN_SEED, 1, 0, 0, _LO, _HI)
    assert drawn[0] == expected
    assert expected[0] != expected[1] or _LO == _HI, (
        "this seed happens to draw a symmetric pair; the node still holds but "
        "no longer distinguishes the pair from a scalar"
    )

    later = draft_thresholds(_RUN_SEED, 1, 0, 3, _LO, _HI)
    assert later != expected, (
        "games 0 and 3 draw the same pair at this seed, so the index cannot be "
        "read off the root; pick another index for this half of the node"
    )
    monkeypatch.undo()
    third = _roots_built(tmp_path / "third", monkeypatch, _LO, _HI, game_index=3)
    assert third[0] == later, (
        "the root of game 3 carries game 3's draw, not a pair drawn once and "
        "reused for every game the worker plays"
    )

    monkeypatch.undo()
    fixed = _roots_built(tmp_path / "fixed", monkeypatch, 10, 0)
    assert fixed[0] == (10, 10), (
        "with the high edge at 0 the root carries the configured scalar, "
        "widened to a pair and nothing else"
    )
