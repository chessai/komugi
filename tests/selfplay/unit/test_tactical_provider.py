"""Differential tests for the KOM-222 tactical provider (unit U2).

Five groups, each pinning one clause of the frozen contract in
``komugi.search.tactical``:

* **T1** the shipped provider really satisfies the protocol, and rejects a
  foreign state by ``TypeError`` before touching it;
* **T2** the real KOM-215 fixture -- the ply-47 position of the anchor game,
  whose black reply 弓(4-6-3)取(8-9-1) is the mate the policy prior dropped --
  is enumerated, and the capture really does end the game for black;
* **T3** a seeded differential against an **independent oracle**: on random
  legal playouts, "removes an enemy marshal" is recomputed from the board by
  clone-and-play, and compared with the provider's one-pass answer for
  completeness, soundness, immutability and shape. The provider is additionally
  handed a state whose ``clone``/``play`` raise, so "never clones, never plays"
  is enforced rather than asserted in prose;
* **T4** ``terminal_winner`` refuses a non-terminal state and every malformed
  outcome, and never repairs one into an answer;
* **T5** the mask is a boundary, not a hint: wrong shape/dtype raises, and an
  action the caller masked off cannot come back.

The fixture helpers and the KOM-215 constants live in test trees pytest does not
put on this directory's import path, so they are loaded by file path under
private module names (see :func:`_load_module`) rather than copied here -- a
restated FEN is a fixture that can rot away from the one it claims to be.
"""

from __future__ import annotations

import os

os.environ.setdefault("GUNGI_MODE", "advanced")

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.cyboard import board_from_fen, board_to_fen, make_code
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game
from komugi.search.tactical import TacticalProvider
from komugi.selfplay.adapter import GameState
from komugi.selfplay.tactical import GameMarshalCaptureProvider

# ---------------------------------------------------------------------------
# Borrowed fixtures
# ---------------------------------------------------------------------------

_TESTS_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative: str) -> ModuleType:
    """Import ``tests/<relative>`` under the private module name *name*.

    The module's own directory goes on ``sys.path`` first, because the modules
    borrowed here import their siblings (``test_kom215_tactics`` imports the
    ``planted`` oracle package that way). The private name keeps the loaded copy
    out of the way of pytest's own collection of the same file.
    """
    path = _TESTS_ROOT / relative
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot import {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURES: Final[ModuleType] = _load_module(
    "kom222_rules_fixtures", "rules/fixtures.py"
)
_KOM215: Final[ModuleType] = _load_module(
    "kom222_kom215_tactics", "search/match/test_kom215_tactics.py"
)

#: ``tests/rules/fixtures.load_game`` -- a ``Game`` on a FEN, history seeded.
_load_game: Any = _FIXTURES.load_game
#: ``tests/rules/fixtures.board_of`` -- a ``Game``'s live ``Board``.
_board_of: Any = _FIXTURES.board_of

#: The anchor game's position before ply 47, white to move.
PLY47_FEN: Final[str] = _KOM215.PLY47_FEN
#: White's recorded ply-47 move, 砦(1-8-2)(1-7-1).
HISTORICAL_WHITE_MOVE: Final[int] = _KOM215.HISTORICAL_WHITE_MOVE
#: Black's ply-48 reply, 弓(4-6-3)取(8-9-1)# -- the marshal capture.
BLACK_MARSHAL_CAPTURE: Final[int] = _KOM215.BLACK_MARSHAL_CAPTURE

# ---------------------------------------------------------------------------
# Constants of this file's own oracle
# ---------------------------------------------------------------------------

#: Width of action block A0, the provider's whole range.
_A0_WIDTH: Final[int] = 6561
#: 帥 in ``cyboard.PIECE_TYPES``.
_MARSHAL: Final[int] = 0

_SEED: Final[int] = 20260917
_GAMES: Final[int] = 8
_MAX_PLIES: Final[int] = 160
#: Sample one position in four. The offset is the game index, because the mover
#: alternates every ply: a fixed ``ply % 4 == 0`` would sample white only.
_SAMPLE_EVERY: Final[int] = 4
#: How many sampled positions must actually offer a candidate. Without a floor,
#: a playout corpus that never reaches a marshal capture passes vacuously.
_MIN_NON_EMPTY: Final[int] = 10


def _state(fen: str) -> GameState:
    """A non-root :class:`GameState` on *fen*, draw score ``0``."""
    return GameState(_load_game(fen), 0.0)


def _marshal_count(state: GameState, color: int) -> int:
    """Number of board cells holding *color*'s 帥.

    Read straight off ``cells_arr`` rather than through anything the provider
    uses, so the oracle below shares no code with what it is checking.
    """
    cells: npt.NDArray[np.int64] = np.asarray(
        _board_of(state.game).cells_arr, dtype=np.int64
    )
    return int(np.count_nonzero(cells == make_code(_MARSHAL, color)))


def _oracle_marshal_removals(
    state: GameState, legal_mask: npt.NDArray[np.bool_]
) -> list[int]:
    """Every legal A0 action of *state* that removes an enemy marshal.

    The independent oracle: play each candidate on a clone and count the enemy
    marshal codes before and after. ``np.nonzero`` is ascending, so the loop can
    stop at the end of block A0.
    """
    enemy = 1 - state.to_play()
    before = _marshal_count(state, enemy)
    removals: list[int] = []
    for raw in np.nonzero(legal_mask)[0]:
        action = int(raw)
        if action >= _A0_WIDTH:
            break
        successor = state.clone()
        successor.play(action)
        if _marshal_count(successor, enemy) < before:
            removals.append(action)
    return removals


class _NoTransitionState(GameState):
    """A :class:`GameState` that fails loudly if it is cloned or played.

    Enumeration is a read of the position (protocol contract "purity"), so the
    provider handed one of these must still answer.
    """

    def clone(self) -> GameState:
        raise AssertionError("the provider cloned the state during enumeration")

    def play(self, action: int, *, keep_root: bool = False) -> None:
        raise AssertionError("the provider played a move during enumeration")


class _StubOutcomeState(GameState):
    """A terminal :class:`GameState` with a hand-written ``absolute_outcome``.

    The engine cannot be made to emit a malformed outcome, and that is exactly
    the input ``terminal_winner`` must refuse rather than repair -- so the
    malformed cases are constructed here.
    """

    def __init__(self, outcome: npt.NDArray[np.uint8]) -> None:
        super().__init__(Game(), 0.0)
        self._outcome = outcome

    def is_terminal(self) -> bool:
        return True

    def absolute_outcome(self) -> npt.NDArray[np.uint8]:
        return self._outcome


def _stub_outcome(values: list[float], dtype: npt.DTypeLike) -> _StubOutcomeState:
    """A :class:`_StubOutcomeState` answering ``np.array(values, dtype)``."""
    return _StubOutcomeState(
        cast("npt.NDArray[np.uint8]", np.array(values, dtype=dtype))
    )


# ---------------------------------------------------------------------------
# T1 -- the protocol
# ---------------------------------------------------------------------------


def test_provider_satisfies_the_frozen_protocol() -> None:
    """T1: identity, structure, and a ``TypeError`` on a foreign state."""
    provider = GameMarshalCaptureProvider()

    assert isinstance(provider, TacticalProvider)
    typed: TacticalProvider = provider
    assert typed.provider_id == "game-marshal-capture-v1"
    for name in ("validate_state", "candidate_actions", "terminal_winner"):
        assert callable(getattr(provider, name)), f"{name} is not callable"

    # A bare ``object()`` has no attribute the provider could reach for, so a
    # ``TypeError`` here is evidence the type check ran FIRST: anything else
    # would surface as an ``AttributeError``.
    with pytest.raises(TypeError):
        provider.validate_state(cast(Any, object()))


# ---------------------------------------------------------------------------
# T2 -- the real KOM-215 fixture
# ---------------------------------------------------------------------------


def test_the_kom215_marshal_capture_is_enumerated_and_wins() -> None:
    """T2: the dropped mate-in-1 of the anchor game is a candidate, and wins."""
    provider = GameMarshalCaptureProvider()
    state = _state(PLY47_FEN)
    state.play(HISTORICAL_WHITE_MOVE)
    assert state.to_play() == 1, "black is to move at ply 48"

    mask = state.legal_mask()
    candidates = provider.candidate_actions(state, mask)

    assert BLACK_MARSHAL_CAPTURE in candidates
    assert list(candidates) == sorted(set(candidates))
    for action in candidates:
        assert isinstance(action, int)
        assert 0 <= action < _A0_WIDTH, f"{action} is outside block A0"
        assert bool(mask[action]), f"{action} is not legal"

    played = state.clone()
    played.play(BLACK_MARSHAL_CAPTURE)
    assert played.is_terminal()
    assert provider.terminal_winner(played) == 1


# ---------------------------------------------------------------------------
# T3 -- the differential
# ---------------------------------------------------------------------------


def test_differential_against_an_independent_playout_oracle() -> None:
    """T3: completeness, soundness, purity and shape over seeded playouts."""
    provider = GameMarshalCaptureProvider()
    rng = np.random.default_rng(_SEED)
    sampled = 0
    non_empty = 0

    for game_index in range(_GAMES):
        state = GameState(Game(board_from_fen(ADVANCED_POSITION)), 0.0)
        for ply in range(_MAX_PLIES):
            if state.is_terminal():
                break
            mask = state.legal_mask()
            legal = np.nonzero(mask)[0]
            if legal.size == 0:
                break

            if (ply + game_index) % _SAMPLE_EVERY == 0:
                sampled += 1
                fen_before = board_to_fen(_board_of(state.game))
                mask_before = mask.tobytes()

                candidates = provider.candidate_actions(state, mask)

                # -- immutability --
                assert mask.tobytes() == mask_before, "the mask was mutated"
                assert board_to_fen(_board_of(state.game)) == fen_before, (
                    "the board was mutated"
                )

                # -- shape --
                assert list(candidates) == sorted(set(candidates)), (
                    f"candidates are not ascending and unique: {candidates!r}"
                )
                assert all(isinstance(action, int) for action in candidates)

                # -- purity: no clone, no play --
                frozen = _NoTransitionState(state.game, 0.0)
                assert provider.candidate_actions(frozen, mask) == candidates

                oracle = _oracle_marshal_removals(state, mask)

                # -- completeness --
                missed = sorted(set(oracle) - set(candidates))
                assert not missed, (
                    f"game {game_index} ply {ply}: marshal-removing actions "
                    f"{missed} are missing from the candidates"
                )

                # -- soundness --
                for action in candidates:
                    assert 0 <= action < _A0_WIDTH
                    assert bool(mask[action])
                    if action in oracle:
                        continue
                    successor = state.clone()
                    successor.play(action)
                    assert successor.is_terminal(), (
                        f"game {game_index} ply {ply}: candidate {action} "
                        "neither removes a marshal nor ends the game"
                    )

                if oracle:
                    non_empty += 1

            state.play(int(rng.choice(legal)))

    assert sampled > 0, "the playouts produced no sampled positions"
    assert non_empty >= _MIN_NON_EMPTY, (
        f"only {non_empty} of {sampled} sampled positions offered a candidate; "
        f"fewer than {_MIN_NON_EMPTY} makes the differential near-vacuous"
    )


# ---------------------------------------------------------------------------
# T4 -- terminal_winner
# ---------------------------------------------------------------------------


def test_terminal_winner_refuses_a_non_terminal_state() -> None:
    """T4: an unfinished game has no winner to read."""
    provider = GameMarshalCaptureProvider()
    with pytest.raises(ValueError):
        provider.terminal_winner(GameState(Game(), 0.0))


def test_terminal_winner_refuses_a_foreign_state() -> None:
    """T4: the type check comes before the terminality check."""
    provider = GameMarshalCaptureProvider()
    with pytest.raises(TypeError):
        provider.terminal_winner(cast(Any, object()))


@pytest.mark.parametrize(
    ("values", "dtype", "why"),
    [
        ([1, 1, 0], np.uint8, "two winners"),
        ([0, 0, 0], np.uint8, "no result at all"),
        ([1, 0], np.uint8, "wrong shape"),
        ([1, 0, 0], np.float64, "float dtype"),
        ([2, 0, 0], np.uint8, "not a one-hot entry"),
    ],
)
def test_terminal_winner_refuses_malformed_outcomes(
    values: list[float], dtype: npt.DTypeLike, why: str
) -> None:
    """T4: every malformed outcome raises -- none is repaired by ``argmax``."""
    provider = GameMarshalCaptureProvider()
    with pytest.raises(ValueError):
        provider.terminal_winner(_stub_outcome(values, dtype))


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 0, 0], 0), ([0, 1, 0], 1), ([0, 0, 1], None)],
)
def test_terminal_winner_reads_categorical_outcomes(
    values: list[float], expected: int | None
) -> None:
    """T4: white, black, and a real draw -- the draw is ``None``, not a score."""
    provider = GameMarshalCaptureProvider()
    assert provider.terminal_winner(_stub_outcome(values, np.uint8)) is expected


# ---------------------------------------------------------------------------
# T5 -- the mask is a boundary
# ---------------------------------------------------------------------------


def test_mask_shape_and_dtype_are_enforced() -> None:
    """T5: a mask that is not ``(ACTION_SIZE,)`` bool is a ``ValueError``."""
    provider = GameMarshalCaptureProvider()
    state = _state(PLY47_FEN)
    state.play(HISTORICAL_WHITE_MOVE)

    with pytest.raises(ValueError):
        provider.candidate_actions(state, np.zeros(10, dtype=np.bool_))
    with pytest.raises(ValueError):
        provider.candidate_actions(
            state, cast(Any, np.zeros(ACTION_SIZE, dtype=np.uint8))
        )


def test_a_filtered_mask_is_never_widened() -> None:
    """T5: an action the caller masked off cannot come back through the seam."""
    provider = GameMarshalCaptureProvider()
    state = _state(PLY47_FEN)
    state.play(HISTORICAL_WHITE_MOVE)

    full = state.legal_mask()
    assert BLACK_MARSHAL_CAPTURE in provider.candidate_actions(state, full)

    filtered = full.copy()
    filtered[BLACK_MARSHAL_CAPTURE] = False
    assert BLACK_MARSHAL_CAPTURE not in provider.candidate_actions(state, filtered)


# -- review finding 1 (2026-09-17): the buried-marshal clause is unreachable from
# legal play (no tsuke, arata or capture landing may stack on a marshal), so it
# needs a FEN fixture; this one kills the top-piece-only mutant.
BURIED_FEN: Final[str] = "9/9/9/9/4|m:s||F:M|3/9/9/9/9 -/- w 3 - 10"


def _buried_marshal_count(state: GameState, color: int) -> int:
    cells = np.asarray(state.game._board.cells_arr, dtype=np.int64)
    return int(np.count_nonzero(cells == make_code(0, color)))


def test_a_buried_enemy_marshal_is_a_candidate() -> None:
    provider = GameMarshalCaptureProvider()
    state = GameState(Game(board_from_fen(BURIED_FEN)), 0.0)
    assert state.to_play() == 0 and not state.is_terminal()
    board = state.game._board
    cells = np.asarray(board.cells_arr)
    # The destination tower really buries the black marshal: bottom is 帥, top is not.
    tower = [int(c) for c in cells[4, 4] if c != 0]
    assert tower[0] == make_code(0, 1) and tower[-1] != make_code(0, 1)

    mask = state.legal_mask()
    enemy = 1 - state.to_play()
    before = _buried_marshal_count(state, enemy)
    oracle = []
    for raw in np.nonzero(mask)[0]:
        action = int(raw)
        if action >= _A0_WIDTH:
            break
        successor = state.clone()
        successor.play(action)
        if _buried_marshal_count(successor, enemy) < before:
            oracle.append(action)
    assert oracle, "fixture rot: nothing removes the buried marshal"

    candidates = provider.candidate_actions(state, mask)
    assert set(oracle) <= set(candidates), f"buried-marshal captures {sorted(set(oracle) - set(candidates))} missing"
    for action in candidates:
        assert bool(mask[action]) and 0 <= action < _A0_WIDTH

