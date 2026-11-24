"""Unit tests for the self-play worker: schema, seeds, perspective, one game.

Everything here runs in one process with a stub evaluator. The seam's
*process-level* behaviour -- a spawned worker, the shape gate firing inside it,
the end-to-end loop through the real net -- belongs to the KOM-122 behavioural
suite; repeating it here would be a slow, flaky copy.

What is left is what a unit test is good at, and it is mostly arithmetic and
frames: the schema's fingerprint (a data-generation identity, so it is pinned
to a literal), the moves-left scale, the index arithmetic of
:class:`~komugi.selfplay.records.ConcatSource`, the seed derivation's three
promises, the absolute->mover frame change at **both** ply parities (design
F11's classic perspective bug), and one real game played end to end into real
shards -- twice, asserting the two runs agree row for row (design F10a's
serial-debug replay claim).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.cyboard import board_from_fen
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule
from komugi.engine.observation import (
    CONTROL_EMPTY,
    CONTROL_SIZE,
    MAX_TURNS,
    PLANE_MAP_VERSION,
)
from komugi.search.gumbel import GumbelConfig, GumbelSearch, SearchResult
from komugi.search.state import AnyEvaluator, AnyWaveEvaluator, SearchState
from komugi.selfplay.adapter import (
    SearchContext,
    EvaluableState,
    GameState,
    make_evaluator,
)
from komugi.selfplay.inference import (
    InferenceConfig,
    NetEval,
    SlotClient,
    wdl_utility,
)
from komugi.selfplay import book
from komugi.selfplay.book import BookEntry, running_gungi_mode
from komugi.selfplay.records import (
    MLH_SCALE,
    SCHEMA,
    SHORT_TERM_HORIZONS,
    ConcatSource,
    game_seed,
    move_seed,
    num_actions,
)
from komugi.selfplay.worker import (
    MODE_FAST,
    MODE_FULL,
    LaneSpec,
    PendingRow,
    SelfPlayWorker,
    WorkerConfig,
    _wave_evaluator_for,
    finalize_rows,
    game_id,
    policy_kl,
    unpack_game_id,
    worker_main,
)
from komugi.train.shards import ShardReader, ShardWriter
from komugi.train.window import SampleSource

PMV1_SCHEMA_FINGERPRINT: Final[str] = (
    "d7532936b77136f3dc4a31f50866e86fd4ed1c324f8142d06e10183f43d1fd78"
)
"""The schema fingerprint at plane-map version 1 -- 118 observation planes and
no ``plane_map_version`` field. Retained as the literal KOM-002 broke: every
shard carrying it is a pre-KOM-002 shard and cannot join a post-KOM-002 window.
"""

SCHEMA_CANONICAL_JSON: Final[str] = (
    '[{"dtype":"float32","name":"observation","shape":[128,9,9]},'
    '{"dtype":"float32","name":"policy_target","shape":[37341]},'
    '{"dtype":"float32","name":"wdl_target","shape":[3]},'
    '{"dtype":"float32","name":"mlh_target","shape":[]},'
    '{"dtype":"float32","name":"policy_kl","shape":[]},'
    '{"dtype":"float32","name":"root_value","shape":[]},'
    '{"dtype":"int64","name":"search_mode","shape":[]},'
    '{"dtype":"int64","name":"to_play","shape":[]},'
    '{"dtype":"int64","name":"ply","shape":[]},'
    '{"dtype":"int64","name":"game_id","shape":[]},'
    '{"dtype":"int64","name":"sims_used","shape":[]},'
    '{"dtype":"int64","name":"plane_map_version","shape":[]},'
    '{"dtype":"int64","name":"final_control","shape":[81]},'
    '{"dtype":"int64","name":"action","shape":[]},'
    '{"dtype":"int64","name":"termination","shape":[]},'
    '{"dtype":"float32","name":"short_term_value","shape":[3]},'
    '{"dtype":"float32","name":"draft_outcome","shape":[3]}]'
)
"""The golden serialisation of :data:`~komugi.selfplay.records.SCHEMA` (design F4).

Pinned to a literal on purpose: the fingerprint is what makes a shard belong to
a data generation, so changing the field set, their order, a dtype or a shape
must fail *here*, visibly, rather than by silently splitting a replay window
into two incompatible halves. Bumping it is a deliberate act with a data
migration attached (design D5).

The pin is the canonical JSON rather than its digest because the fingerprint is
``sha256`` of exactly this text: pinning the text pins the digest, and does it
in a form a reviewer can check against the field list by reading, which a hex
string is not. ``tests/train/test_shards.py`` deliberately pins no second
literal and delegates that job here.

KOM-005b is the migration that last moved it -- the three appended staged-head
columns ``termination``, ``short_term_value`` and ``draft_outcome``, in **one**
bump rather than three so the replay window is invalidated once; KOM-014
moved it before that (the appended ``action`` field), KOM-010 before that (the
appended ``final_control`` field), and KOM-002 before that (118 -> 128 planes
and the appended ``plane_map_version`` field). Each was re-derived from the
field list by hand, never copied from a value the code under test printed.
"""

_SHORT_GAME_PLIES: Final[int] = 6
"""Turn cap for the in-test engine subclass -- a whole game in a few seconds.

A real game runs to the engine's turn cap, and the worker has no ply cap of
its own by design (F15). Rather than teach the worker a test-only limit, the
test shortens the *engine*, which is the component that owns termination.
"""


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class _StubEvaluator:
    """A deterministic :class:`~komugi.selfplay.inference.BatchEvaluator`.

    Uniform policy logits (so the Gumbel draw, and therefore the seed, decides
    the candidate ordering) and a fixed win/draw/loss vector. Deterministic to
    the bit, which is what lets the replay test attribute any difference
    between two runs to the worker rather than to the network.
    """

    def __init__(
        self,
        actions: int,
        wdl: tuple[float, float, float] = (0.4, 0.2, 0.4),
        logit: float = 0.0,
    ) -> None:
        self.actions = actions
        self.wdl = wdl
        self.logit = logit
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        self.calls += 1
        return NetEval(
            policy_logits=np.full((rows, self.actions), self.logit, dtype=np.float32),
            wdl=np.tile(np.asarray(self.wdl, dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ScriptedGame:
    """An engine game whose outcome and side to move are dictated by the test.

    :class:`~komugi.selfplay.adapter.GameState` is a pure delegator: the only
    arithmetic it owns is the absolute->mover frame change, whose inputs are
    exactly ``win_state()`` and ``player``. Scripting those two is therefore the
    tightest possible test of that conversion -- a hand-built checkmate FEN
    would test the engine's adjudication instead, which the engine's own suite
    already does, and the real-engine path is exercised end to end by
    :func:`test_play_game_writes_valid_reproducible_rows` below.
    """

    def __init__(self, outcome: tuple[int, int, int], player: int) -> None:
        self._outcome = np.asarray(outcome, dtype=np.uint8)
        self._player = player

    @property
    def player(self) -> int:
        """Side to move: 0 white, 1 black."""
        return self._player

    def win_state(self) -> npt.NDArray[np.uint8]:
        """Absolute ``[white, black, draw]`` outcome."""
        return self._outcome.copy()

    def valid_moves(self) -> npt.NDArray[np.uint8]:
        """A terminal position has no legal move."""
        return np.zeros(4, dtype=np.uint8)

    def observation(self) -> npt.NDArray[np.float32]:
        """A stand-in encoding; nothing under test reads its contents."""
        return np.zeros((2, 1, 1), dtype=np.float32)

    def in_draft(self, side: int) -> bool:
        """Never drafting: a scripted terminal position is past the draft."""
        return False

    def play_action(self, action: int) -> None:
        """Unreachable: these positions are terminal."""
        raise AssertionError("scripted terminal game was played")

    def clone(self) -> "_ScriptedGame":
        """Return an independent copy."""
        return _ScriptedGame(
            (int(self._outcome[0]), int(self._outcome[1]), int(self._outcome[2])),
            self._player,
        )


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`.

    Only ``max_turns`` changes, and ``win_state`` reads it, so every other rule
    -- move generation, checkmate, repetition -- is the production engine's.
    """

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def _scripted_state(
    outcome: tuple[int, int, int], player: int, draw_score: float
) -> GameState:
    """Return a :class:`GameState` over a scripted terminal position."""
    return GameState(cast(Game, _ScriptedGame(outcome, player)), draw_score)


# --------------------------------------------------------------------------
# Schema and scale
# --------------------------------------------------------------------------


def test_schema_fields_are_the_designed_record() -> None:
    """Design D4's field set, in order, with only the two shard dtypes (F4)."""
    assert SCHEMA.names == (
        "observation",
        "policy_target",
        "wdl_target",
        "mlh_target",
        "policy_kl",
        "root_value",
        "search_mode",
        "to_play",
        "ply",
        "game_id",
        "sims_used",
        "plane_map_version",
        "final_control",
        "action",
        "termination",
        "short_term_value",
        "draft_outcome",
    )
    assert {field.dtype for field in SCHEMA.fields} == {"float32", "int64"}
    by_name = {field.name: field for field in SCHEMA.fields}
    assert by_name["observation"].shape == Game.observation_size()
    assert by_name["policy_target"].shape == (num_actions(),)
    assert by_name["wdl_target"].shape == (3,)
    assert by_name["final_control"].shape == (CONTROL_SIZE,)
    assert by_name["short_term_value"].shape == (len(SHORT_TERM_HORIZONS),)
    assert by_name["draft_outcome"].shape == (3,)
    for scalar in (
        "mlh_target",
        "policy_kl",
        "root_value",
        "search_mode",
        "ply",
        "action",
        "termination",
    ):
        assert by_name[scalar].shape == ()


def test_schema_fingerprint_is_the_golden_generation_id() -> None:
    """The fingerprint is the data generation's identity -- pinned literally.

    Pinned through the canonical JSON the digest is taken over, and paired with
    the plane-map-version-1 digest KOM-002 broke, so the pin says both what the
    generation IS and what it is no longer.
    """
    assert SCHEMA.canonical_json() == SCHEMA_CANONICAL_JSON
    assert SCHEMA.fingerprint != PMV1_SCHEMA_FINGERPRINT
    assert (
        hashlib.sha256(SCHEMA_CANONICAL_JSON.encode("utf-8")).hexdigest()
        == SCHEMA.fingerprint
    )


def test_schema_dry_run_accepts_a_row_and_rejects_a_mis_shaped_one(
    tmp_path: Path,
) -> None:
    """Design F4's dry run: the schema constructs *and* a real row round-trips."""
    writer = ShardWriter(tmp_path / "dry", SCHEMA)
    sample = _blank_sample()
    writer.append(sample)
    assert writer.flush() is not None

    reader = ShardReader(tmp_path / "dry", schema=SCHEMA)
    assert reader.skipped == ()
    assert len(reader) == 1

    broken = dict(sample)
    broken["wdl_target"] = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        writer.append(broken)


def test_mlh_scale_is_pinned_to_the_turn_cap() -> None:
    """Design F15/Q3: one over the engine's turn cap, and nothing else.

    Pinned to the leaf rather than to a literal (KOM-165): the scale, the cap
    and observation plane 115's denominator are one number in one file, so a
    cap change cannot leave a stale scale behind in a shard.
    """
    assert MLH_SCALE == 1.0 / MAX_TURNS
    assert MLH_SCALE * Game.max_turns() == 1.0


def _blank_sample() -> dict[str, torch.Tensor]:
    """Return a schema-shaped sample of zeros."""
    return {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in SCHEMA.fields
    }


# --------------------------------------------------------------------------
# ConcatSource
# --------------------------------------------------------------------------


def _fake_source(start: int) -> list[dict[str, torch.Tensor]]:
    """Two samples tagged `start` and ``start + 1``.

    Each row also claims the running plane map: ``ConcatSource`` checks
    ``plane_map_version`` on every materialised row (KOM-002 §4), and a row
    without the field is refused as a pre-KOM-002 row rather than read.
    """
    return [
        {
            "x": torch.tensor(start + offset, dtype=torch.int64),
            "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        }
        for offset in (0, 1)
    ]


def test_concat_source_spans_its_sources_in_order() -> None:
    """Design D10/F5: three worker dirs read as one iteration's samples."""
    sources: list[SampleSource] = [_fake_source(0), _fake_source(10), _fake_source(20)]
    concat = ConcatSource(sources)

    assert len(concat) == 6
    assert [int(concat[index]["x"]) for index in range(6)] == [0, 1, 10, 11, 20, 21]
    assert int(concat[-1]["x"]) == 21
    assert int(concat[-6]["x"]) == 0


def test_concat_source_raises_past_the_end() -> None:
    """A silent wrap would train on the wrong worker's data."""
    sources: list[SampleSource] = [_fake_source(0), _fake_source(10), _fake_source(20)]
    concat = ConcatSource(sources)
    with pytest.raises(IndexError):
        concat[6]
    with pytest.raises(IndexError):
        concat[-7]


def test_concat_source_skips_empty_sources() -> None:
    """A worker that wrote nothing contributes nothing, and shifts no index."""
    empty: list[dict[str, torch.Tensor]] = []
    sources: list[SampleSource] = [
        empty,
        _fake_source(0),
        empty,
        _fake_source(10),
        empty,
    ]
    concat = ConcatSource(sources)
    assert len(concat) == 4
    assert [int(concat[index]["x"]) for index in range(4)] == [0, 1, 10, 11]


# --------------------------------------------------------------------------
# Seeds (design D11 / F10)
# --------------------------------------------------------------------------


def test_game_seed_is_the_designed_spawn_key() -> None:
    """``SeedSequence(run_seed, spawn_key=(worker, incarnation, game))``, no hash."""
    seeds = game_seed(1234, worker_id=3, incarnation=1, game_index=7)
    assert seeds.entropy == 1234
    assert tuple(seeds.spawn_key) == (3, 1, 7)


def test_move_seed_differs_across_plies() -> None:
    """Design F10d: one seed per game would replay one Gumbel vector all game."""
    seeds = game_seed(99, worker_id=0, incarnation=0, game_index=0)
    drawn = [move_seed(seeds, ply) for ply in range(8)]
    assert len(set(drawn)) == len(drawn)


def test_move_seed_is_a_pure_function_of_its_arguments() -> None:
    """Asked twice, answered identically -- the replay claim rests on this.

    ``SeedSequence.spawn`` would not satisfy it: it advances an internal child
    counter, so the second call would answer differently.
    """
    first = game_seed(5, worker_id=2, incarnation=0, game_index=1)
    second = game_seed(5, worker_id=2, incarnation=0, game_index=1)
    assert [move_seed(first, ply) for ply in range(4)] == [
        move_seed(second, ply) for ply in range(4)
    ]
    assert move_seed(first, 3) == move_seed(first, 3)


def test_move_seed_differs_across_incarnations_and_games() -> None:
    """Design F10b/c: a respawn must not replay its predecessor's games."""
    base = game_seed(5, worker_id=2, incarnation=0, game_index=1)
    respawned = game_seed(5, worker_id=2, incarnation=1, game_index=1)
    other_game = game_seed(5, worker_id=2, incarnation=0, game_index=2)
    other_worker = game_seed(5, worker_id=3, incarnation=0, game_index=1)

    plies = range(4)
    streams = [
        tuple(move_seed(seeds, ply) for ply in plies)
        for seeds in (base, respawned, other_game, other_worker)
    ]
    assert len(set(streams)) == 4


def test_game_id_is_a_pure_function_of_the_same_triple() -> None:
    """Provenance that two runs of one game agree on, so rows compare equal."""
    assert game_id(2, 1, 7) == game_id(2, 1, 7)
    assert len({game_id(2, 1, 7), game_id(2, 0, 7), game_id(3, 1, 7), game_id(2, 1, 8)}) == 4


# --------------------------------------------------------------------------
# Perspective: the absolute -> mover frame change (design D1 / F11)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mover", [0, 1])
def test_terminal_value_is_plus_one_when_the_mover_won(mover: int) -> None:
    """Both parities: a win for the side to move is ``+1`` in either frame."""
    outcome = (1, 0, 0) if mover == 0 else (0, 1, 0)
    state = _scripted_state(outcome, player=mover, draw_score=0.2)
    assert state.terminal_value() == 1.0


@pytest.mark.parametrize("mover", [0, 1])
def test_terminal_value_is_minus_one_when_the_mover_lost(mover: int) -> None:
    """Both parities: the sign flips with the mover, not with the colour."""
    outcome = (0, 1, 0) if mover == 0 else (1, 0, 0)
    state = _scripted_state(outcome, player=mover, draw_score=0.2)
    assert state.terminal_value() == -1.0


@pytest.mark.parametrize("mover", [0, 1])
def test_terminal_draw_is_worth_draw_score_at_either_parity(mover: int) -> None:
    """Design D1: contempt enters where a value enters the tree, mover-relative."""
    state = _scripted_state((0, 0, 1), player=mover, draw_score=0.2)
    assert state.terminal_value() == pytest.approx(0.2)


def test_absolute_outcome_is_untouched_by_draw_score() -> None:
    """The recorder's view: pure ``win_state``, so targets never see contempt."""
    state = _scripted_state((0, 0, 1), player=1, draw_score=0.75)
    assert np.array_equal(state.absolute_outcome(), np.array([0, 0, 1], dtype=np.uint8))
    assert state.absolute_outcome().dtype == np.uint8


def test_terminal_value_is_zero_on_an_undecided_position() -> None:
    """No result yet is not a draw: it is a value nobody may read."""
    state = _scripted_state((0, 0, 0), player=0, draw_score=0.2)
    assert state.terminal_value() == 0.0
    assert state.is_terminal() is False


def test_game_state_satisfies_both_protocols() -> None:
    """The adapter's whole point: one object, two contracts (design §0)."""
    state = _scripted_state((0, 0, 1), player=0, draw_score=0.2)
    evaluable: EvaluableState = state
    searchable: SearchState = state
    assert evaluable.to_play() == 0
    assert searchable.is_terminal() is True
    assert isinstance(state.clone(), GameState)
    assert state.clone().draw_score == state.draw_score


# --------------------------------------------------------------------------
# make_evaluator
# --------------------------------------------------------------------------


def test_make_evaluator_returns_policy_and_the_wdl_utility() -> None:
    """``(policy (A,), float)``, with the value derived client-side (design D1)."""
    actions = 4
    evaluator = make_evaluator(_StubEvaluator(actions, wdl=(0.5, 0.3, 0.2)), 0.2)
    state = _scripted_state((0, 0, 0), player=0, draw_score=0.2)

    policy, value = evaluator(state)

    assert policy.shape == (actions,)
    assert policy.dtype == np.float32
    assert isinstance(value, float)
    # w - l + d * draw_score = 0.5 - 0.2 + 0.3 * 0.2
    assert value == pytest.approx(0.36, abs=1e-6)


def test_make_evaluator_draw_score_moves_only_the_value() -> None:
    """The same network output, two contempts, two utilities."""
    stub = _StubEvaluator(4, wdl=(0.25, 0.5, 0.25))
    neutral = make_evaluator(stub, 0.0)
    contempt = make_evaluator(stub, 0.4)
    state = _scripted_state((0, 0, 0), player=0, draw_score=0.0)

    assert neutral(state)[1] == pytest.approx(0.0, abs=1e-6)
    assert contempt(state)[1] == pytest.approx(0.2, abs=1e-6)


def test_make_evaluator_with_wdl_returns_the_leaf_vector_unchanged() -> None:
    """KOM-173 T6: the self-play seam's third element is the head's own vector.

    ``with_wdl=True`` is what the worker passes and nothing else does. It adds
    the leaf's ``(w, d, l)`` -- in the leaf mover's frame, exactly as the network
    answered it -- and leaves the utility alone: the same two numbers the
    two-tuple seam returns, from the same call, so a reader can compare them
    directly rather than through a tolerance on a re-derivation.
    """
    actions = 4
    wdl = (0.5, 0.3, 0.2)
    plain = make_evaluator(_StubEvaluator(actions, wdl=wdl), 0.2)
    carrying = make_evaluator(_StubEvaluator(actions, wdl=wdl), 0.2, with_wdl=True)
    state = _scripted_state((0, 0, 0), player=0, draw_score=0.2)

    plain_answer = plain(state)
    carrying_answer = carrying(state)

    assert len(plain_answer) == 2
    assert len(carrying_answer) == 3
    assert np.array_equal(carrying_answer[0], plain_answer[0])
    assert carrying_answer[1] == plain_answer[1]
    assert carrying_answer[1] == pytest.approx(0.36, abs=1e-6)
    fields: tuple[object, ...] = carrying_answer
    vector = np.asarray(fields[2], dtype=np.float64)
    np.testing.assert_allclose(vector, np.asarray(wdl), rtol=0.0, atol=1e-7)
    assert abs(float(vector.sum()) - 1.0) <= 1e-6


@pytest.mark.parametrize("draw_score", [0.0, 0.4, -0.4])
def test_the_vector_is_contempt_free_while_the_utility_is_not(
    draw_score: float,
) -> None:
    """T6: ``draw_score`` moves the utility only -- the vector is the pure head.

    Design D1 lets contempt in at the two points a *value* enters the tree; the
    searched vector is not one of them (D5/D6), which is what makes the recorded
    readouts a contempt-free diagnostic even on a run that plays with a draw
    score. Swept over both signs, because a vector that had ``s*d`` folded into
    it would still look right at ``s = 0``.
    """
    wdl = (0.25, 0.5, 0.25)
    seam = make_evaluator(_StubEvaluator(4, wdl=wdl), draw_score, with_wdl=True)
    state = _scripted_state((0, 0, 0), player=0, draw_score=draw_score)

    answer = seam(state)
    fields: tuple[object, ...] = answer

    assert answer[1] == pytest.approx(0.5 * draw_score, abs=1e-6)
    np.testing.assert_allclose(
        np.asarray(fields[2], dtype=np.float64),
        np.asarray(wdl),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("mover", [0, 1])
def test_game_state_terminal_wdl_is_the_mover_frame_one_hot(mover: int) -> None:
    """KOM-173 D5: the terminal vector is the result, in the mover's frame.

    The vector counterpart of the three ``terminal_value`` tests above, and the
    same perspective rule: the win slot is the mover's, the loss slot is the
    other player's, the draw slot is shared. ``draw_score`` is deliberately
    large and deliberately absent from the expected values -- a draw is
    ``(0, 1, 0)`` here even at contempt ``0.75``, which is exactly the difference
    from :meth:`GameState.terminal_value`.
    """
    won = _scripted_state(
        (1, 0, 0) if mover == 0 else (0, 1, 0), player=mover, draw_score=0.75
    )
    lost = _scripted_state(
        (0, 1, 0) if mover == 0 else (1, 0, 0), player=mover, draw_score=0.75
    )
    drawn = _scripted_state((0, 0, 1), player=mover, draw_score=0.75)

    assert np.array_equal(won.terminal_wdl(), np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(lost.terminal_wdl(), np.array([0.0, 0.0, 1.0]))
    assert np.array_equal(drawn.terminal_wdl(), np.array([0.0, 1.0, 0.0]))
    assert drawn.terminal_value() == pytest.approx(0.75), (
        "the scalar still carries contempt; only the vector is free of it"
    )
    for state in (won, lost, drawn):
        vector = state.terminal_wdl()
        assert vector.dtype == np.float64
        assert vector.shape == (3,)
        assert float(vector[0] - vector[2]) == pytest.approx(
            0.0 if bool(state.absolute_outcome()[2]) else state.terminal_value()
        )


def test_game_state_terminal_wdl_is_all_zero_before_the_game_ends() -> None:
    """No result yet is not a draw, exactly as :meth:`absolute_outcome` says."""
    state = _scripted_state((0, 0, 0), player=0, draw_score=0.2)

    assert state.is_terminal() is False
    assert np.array_equal(state.terminal_wdl(), np.zeros(3))


# --------------------------------------------------------------------------
# policy_kl and row finalisation
# --------------------------------------------------------------------------


def test_policy_kl_is_zero_when_the_search_confirms_the_prior() -> None:
    """A target equal to the masked prior is no surprise at all."""
    legal = np.array([True, True, False, True], dtype=np.bool_)
    logits = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)
    target = np.array([1 / 3, 1 / 3, 0.0, 1 / 3], dtype=np.float32)
    assert policy_kl(target, logits, legal) == pytest.approx(0.0, abs=1e-6)


def test_policy_kl_matches_the_hand_computed_value() -> None:
    """Three legal actions, uniform prior, a target the search moved."""
    legal = np.array([True, True, True], dtype=np.bool_)
    logits = np.zeros(3, dtype=np.float32)
    target = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    expected = 0.5 * np.log(0.5 * 3) + 2 * (0.25 * np.log(0.25 * 3))
    assert policy_kl(target, logits, legal) == pytest.approx(float(expected), abs=1e-9)


def _pending(
    ply: int,
    to_play: int,
    *,
    root_value: float = 0.0,
    in_draft: bool = False,
) -> PendingRow:
    """A minimal buffered row at `ply`, moved by `to_play`.

    ``action`` is ``ply`` so that a row's move is distinguishable from every
    other row's in the finalised sample -- a zero everywhere would let a
    ``finalize_rows`` that dropped the field pass (KOM-014 §3.1).

    ``root_value`` and ``in_draft`` default to the values that make the two
    KOM-005b columns uninteresting (a zero short-term lookup, an all-zero
    draft-outcome row), so a test about ``wdl_target`` or ``mlh_target`` is
    undisturbed by them; the staged-target suite passes them explicitly.
    """
    return PendingRow(
        observation=np.zeros(Game.observation_size(), dtype=np.float32),
        policy_target=np.zeros(num_actions(), dtype=np.float32),
        policy_kl=0.0,
        root_value=root_value,
        search_mode=MODE_FULL,
        to_play=to_play,
        ply=ply,
        sims_used=2,
        action=ply,
        in_draft=in_draft,
    )


def _empty_control() -> npt.NDArray[np.int8]:
    """An all-empty absolute control map, for the tests that assert on other fields.

    ``CONTROL_EMPTY`` everywhere is the one map that is invariant under the
    mover flip, so a test about ``wdl_target`` or ``mlh_target`` cannot pass or
    fail because of the map it was handed. The flip itself is asserted by
    ``tests/selfplay/unit/test_final_control_rows.py``.
    """
    return np.full(CONTROL_SIZE, CONTROL_EMPTY, dtype=np.int8)


def test_finalize_rows_flips_wdl_with_the_mover() -> None:
    """Design F11: white wins, so rows alternate ``[1,0,0]`` and ``[0,0,1]``."""
    rows = [_pending(ply, ply % 2) for ply in range(4)]
    samples = finalize_rows(
        rows, np.array([1, 0, 0], dtype=np.uint8), _empty_control(), 4, 77
    )

    wdl = [sample["wdl_target"].numpy().tolist() for sample in samples]
    assert wdl == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] * 2
    assert [int(sample["game_id"]) for sample in samples] == [77] * 4


def test_finalize_rows_labels_a_draw_without_contempt() -> None:
    """Design D1/D4: ``draw_score`` never reaches a target -- draws are [0,1,0]."""
    rows = [_pending(ply, ply % 2) for ply in range(3)]
    samples = finalize_rows(
        rows, np.array([0, 0, 1], dtype=np.uint8), _empty_control(), 3, 0
    )
    for sample in samples:
        assert sample["wdl_target"].numpy().tolist() == [0.0, 1.0, 0.0]


def test_finalize_rows_counts_plies_left_on_the_pinned_scale() -> None:
    """``mlh_target = (total_plies - ply) * MLH_SCALE``, last row = one ply."""
    rows = [_pending(ply, ply % 2) for ply in range(5)]
    samples = finalize_rows(
        rows, np.array([0, 1, 0], dtype=np.uint8), _empty_control(), 5, 0
    )
    assert [float(sample["mlh_target"]) for sample in samples] == pytest.approx(
        [5 * MLH_SCALE, 4 * MLH_SCALE, 3 * MLH_SCALE, 2 * MLH_SCALE, 1 * MLH_SCALE]
    )


def test_finalize_rows_refuses_an_unfinished_game() -> None:
    """Writing zeros for a game still in progress would be an undetectable lie."""
    with pytest.raises(ValueError, match="no outcome"):
        finalize_rows(
            [_pending(0, 0)], np.zeros(3, dtype=np.uint8), _empty_control(), 1, 0
        )


# --------------------------------------------------------------------------
# One whole game, on the real engine
# --------------------------------------------------------------------------


def _selfplay_config(
    prob_fast: float, m: int = 2, n: int = 2, games_in_flight: int = 1
) -> SelfPlayConfig:
    """A two-simulation config: the loop's structure without its cost.

    ``m``/``n`` default to the smallest search that still has a shape. The wave
    tests raise them, because a wave is a halving round's leaves and a round of
    two has nothing to batch.
    """
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=m,
        gumbel_n_full=n,
        gumbel_m_fast=m,
        gumbel_n_fast=n,
        prob_fast=prob_fast,
        draw_score=0.2,
        games_in_flight=games_in_flight,
    )


def _play_one(
    directory: Path, prob_fast: float, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, list[dict[str, torch.Tensor]]]:
    """Play game 0 into `directory` and read the rows back out of the shards."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        _selfplay_config(prob_fast),
        WorkerConfig(
            worker_id=1, incarnation=0, run_seed=20260822, games=1, shard_dir=str(directory)
        ),
        writer,
    )
    written = worker.play_game(0)
    writer.close()

    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    return written, list(reader)


def test_every_row_records_the_action_that_was_played(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-014 §3.1/B2: the ``action`` column is the game, replayable.

    The check is a replay rather than a comparison against a list the test kept,
    because a list the test kept would be the worker's own answer written twice.
    Feeding the recorded column back into a fresh engine reaches the *same*
    adjudicated position by a path that is legal at every step -- which is false
    for a column that is zero-filled, off by a ply, or in the wrong order, and is
    exactly what a draft book will ask of a shard.
    """
    written, rows = _play_one(tmp_path / "actions", 0.0, monkeypatch)
    assert written == len(rows) > 0
    actions = [int(row["action"]) for row in rows]
    assert any(action != 0 for action in actions), (
        "every recorded action is 0, which is what a schema field nobody writes "
        "looks like"
    )

    state = GameState(_ShortGame(), 0.2)
    for index, action in enumerate(actions):
        assert not state.is_terminal(), f"the replay ended early, at ply {index}"
        assert bool(state.legal_mask()[action]), (
            f"row {index} records action {action}, which is illegal at the "
            "position ply {index} reaches"
        )
        state.play(action)
    assert state.is_terminal(), (
        "replaying every recorded action did not reach an adjudicated position, "
        "so the column is not the line the worker actually played"
    )


def test_play_game_writes_valid_reproducible_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real game, twice: valid rows, recorded modes, and identical bytes.

    The reproducibility half is design F10a's serial-debug claim, which is only
    ever promised for a deterministic evaluator in one process -- exactly this
    setup. Production games batch across workers and are promised only up to
    evaluator tolerance.
    """
    written, rows = _play_one(tmp_path / "first", 0.5, monkeypatch)
    again, rows_again = _play_one(tmp_path / "second", 0.5, monkeypatch)

    assert written == len(rows) > 0
    assert again == written

    # The engine adjudicates; the worker imposes no cap of its own (F15).
    assert len(rows) <= _SHORT_GAME_PLIES

    for index, row in enumerate(rows):
        assert set(row) == set(SCHEMA.names)
        SCHEMA.validate(row)
        assert int(row["ply"]) == index
        # The movers alternate; which colour starts is the engine's business.
        assert int(row["to_play"]) == (int(rows[0]["to_play"]) + index) % 2
        assert int(row["game_id"]) == game_id(1, 0, 0)
        assert int(row["search_mode"]) in (MODE_FULL, MODE_FAST)
        assert int(row["sims_used"]) in (0, 2)
        assert float(row["policy_kl"]) >= 0.0
        assert float(row["mlh_target"]) == pytest.approx(
            (len(rows) - index) * MLH_SCALE
        )
        assert float(row["policy_target"].sum()) == pytest.approx(1.0, abs=1e-4)

    # wdl_target is the game's own outcome, expressed for the row's mover.
    draw = all(row["wdl_target"].numpy().tolist() == [0.0, 1.0, 0.0] for row in rows)
    if draw:
        # The shortened cap adjudicates a draw, so every row says so -- and no
        # row says 0.2, which is what contempt leaking into targets would look
        # like (design D1).
        assert all(float(row["wdl_target"].max()) == 1.0 for row in rows)
    else:
        wins = [int(np.argmax(row["wdl_target"].numpy())) for row in rows]
        # Decisive: consecutive movers get opposite labels (win 0 / loss 2).
        assert all(a != b for a, b in zip(wins, wins[1:]))

    for first, second in zip(rows, rows_again):
        for name in SCHEMA.names:
            assert torch.equal(first[name], second[name]), name


@pytest.mark.parametrize(
    ("prob_fast", "expected"), [(0.0, MODE_FULL), (1.0, MODE_FAST)]
)
def test_search_mode_records_the_drawn_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prob_fast: float, expected: int
) -> None:
    """Design F1: the full/fast draw is recorded on every row it produced."""
    _, rows = _play_one(tmp_path / f"mode{expected}", prob_fast, monkeypatch)
    assert rows
    assert {int(row["search_mode"]) for row in rows} == {expected}


def test_a_game_costs_exactly_its_searches_and_not_one_evaluation_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-040 D9': the worker adds **no** evaluation of its own, per move.

    The worker used to evaluate the root once more before every search, purely
    because :class:`~komugi.search.gumbel.SearchResult` did not report the
    prior the search had used -- one wasted network round trip on every move of
    every game. The count is pinned against the ground truth rather than a
    literal: the same game is replayed here as bare searches, with the same
    seeds off the same engine and the same deterministic stub, and the worker
    must cost exactly that. On the old code it costs that plus one per ply.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    directory = tmp_path / "evaluations"
    writer = ShardWriter(directory, SCHEMA)
    played = _StubEvaluator(num_actions())
    worker = SelfPlayWorker(
        played,
        _selfplay_config(0.0),
        WorkerConfig(
            worker_id=1, incarnation=0, run_seed=20260824, games=1, shard_dir=str(directory)
        ),
        writer,
    )
    plies = worker.play_game(0)
    writer.close()

    replayed = _StubEvaluator(num_actions())
    evaluator = make_evaluator(replayed, 0.2)
    seeds = game_seed(20260824, worker_id=1, incarnation=0, game_index=0)
    state = GameState(_ShortGame(), 0.2)
    ply = 0
    while not state.is_terminal():
        config = GumbelConfig(
            num_actions=num_actions(), m=2, n=2, seed=move_seed(seeds, ply)
        )
        state.play(GumbelSearch(evaluator, config).search(state).action)
        ply += 1

    assert ply == plies > 0
    assert replayed.calls >= plies, "every search evaluates its root at least"
    assert played.calls == replayed.calls


def test_run_plays_every_game_and_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run`` plays ``games`` games and leaves nothing buffered (design D10)."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    directory = tmp_path / "run"
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        _selfplay_config(1.0),
        WorkerConfig(
            worker_id=0, incarnation=2, run_seed=7, games=2, shard_dir=str(directory)
        ),
        writer,
    )
    worker.run()

    assert writer.pending == 0
    reader = ShardReader(directory, schema=SCHEMA)
    played = {int(row["game_id"]) for row in reader}
    assert played == {game_id(0, 2, 0), game_id(0, 2, 1)}


# --------------------------------------------------------------------------
# Waves: the worker's end of KOM-040 b2 (design rev2, D1/D2/D3)
# --------------------------------------------------------------------------


class _CountingWaveEvaluator:
    """:func:`~komugi.selfplay.adapter.make_wave_evaluator`'s contract, counted.

    The same three steps the real adapter takes -- gather every state's
    observation and mask, ask the seam **once**, hand back one
    ``(policy, value)`` per row -- over a
    :class:`~komugi.selfplay.inference.BatchEvaluator` stub instead of a
    :class:`~komugi.selfplay.inference.SlotClient`, because what is under test
    here is how many round trips a *move* costs, not how a ticket is packed
    (:mod:`tests.selfplay.unit.test_wave_protocol_unit` owns that).

    One call is one round trip: in production this is one doorbell, one
    forward's latency, one answer.

    **Three-tuples since KOM-173**, and that is not optional here: the worker
    builds its scalar seam with ``with_wdl=True``, the search reads the answer
    shape off the root (which always goes through that seam), and a wave that
    dropped the vector would leave the root's accumulator with a hole -- which
    ``_backup`` refuses by name. One shape per search, on both seams.
    """

    def __init__(self, evaluator: _StubEvaluator, draw_score: float) -> None:
        self._evaluator = evaluator
        self._draw_score = draw_score
        self.widths: list[int] = []

    def __call__(
        self, states: Sequence[SearchState]
    ) -> Sequence[
        tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float32]]
    ]:
        """Evaluate a whole wave in one call, recording how wide it was."""
        evaluables = [cast(EvaluableState, state) for state in states]
        obs = np.stack([evaluable.observation() for evaluable in evaluables])
        legal = np.stack([evaluable.legal_mask() for evaluable in evaluables])
        self.widths.append(len(states))
        evaluation = self._evaluator.evaluate(obs, legal)
        values = wdl_utility(evaluation.wdl, self._draw_score)
        return [
            (
                np.asarray(evaluation.policy_logits[row], dtype=np.float32),
                float(values[row]),
                np.asarray(evaluation.wdl[row], dtype=np.float32),
            )
            for row in range(len(states))
        ]


def _bare_search_calls(
    run_seed: int, selfplay: SelfPlayConfig, m: int, n: int
) -> tuple[int, int]:
    """Replay a game as bare searches; return ``(plies, evaluator calls)``.

    The ground truth the worker is measured against, exactly as the D9'
    accounting test builds it: the same engine, the same per-move seeds, the
    same deterministic stub, and no worker in the loop at all.
    """
    evaluator_stub = _StubEvaluator(num_actions())
    evaluator = make_evaluator(evaluator_stub, selfplay.draw_score)
    seeds = game_seed(run_seed, worker_id=1, incarnation=0, game_index=0)
    state = GameState(_ShortGame(), selfplay.draw_score)
    ply = 0
    while not state.is_terminal():
        config = GumbelConfig(
            num_actions=num_actions(), m=m, n=n, seed=move_seed(seeds, ply)
        )
        state.play(GumbelSearch(evaluator, config).search(state).action)
        ply += 1
    return ply, evaluator_stub.calls


def _play_with(
    directory: Path,
    selfplay: SelfPlayConfig,
    run_seed: int,
    evaluator: _StubEvaluator,
    wave_evaluator: AnyWaveEvaluator | None,
) -> int:
    """Play game 0 into `directory` with (or without) the wave seam."""
    writer = ShardWriter(directory, SCHEMA)
    try:
        return SelfPlayWorker(
            evaluator,
            selfplay,
            WorkerConfig(
                worker_id=1,
                incarnation=0,
                run_seed=run_seed,
                games=1,
                shard_dir=str(directory),
            ),
            writer,
            wave_evaluator,
        ).play_game(0)
    finally:
        writer.close()


def test_a_wave_worker_costs_fewer_round_trips_than_one_leaf_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-040 b2, end to end through the worker: same game, fewer trips.

    Both runs play the *same* game -- same seeds, same engine, same
    deterministic stub -- with a four-candidate search, so a halving round has
    leaves to batch. The counts are the point:

    * The no-wave worker costs exactly what bare searches cost. That is
      today's number, pinned against ground truth rather than a literal, and it
      is what ``max_wave == 1`` must keep costing forever.
    * The wave worker costs strictly fewer, and the trips it does make are
      *wide* -- a round's leaves travel together. Every leaf is still
      evaluated: the wave path deletes round trips, not questions.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    run_seed = 20260828
    selfplay = _selfplay_config(0.0, m=4, n=8)

    sequential_stub = _StubEvaluator(num_actions())
    sequential_plies = _play_with(
        tmp_path / "sequential", selfplay, run_seed, sequential_stub, None
    )

    wave_stub = _StubEvaluator(num_actions())
    wave = _CountingWaveEvaluator(wave_stub, selfplay.draw_score)
    wave_plies = _play_with(tmp_path / "wave", selfplay, run_seed, wave_stub, wave)

    replay_plies, replay_calls = _bare_search_calls(run_seed, selfplay, m=4, n=8)
    assert sequential_plies == replay_plies > 0
    assert sequential_stub.calls == replay_calls, (
        "the no-wave worker no longer costs exactly what its searches cost"
    )

    assert wave_plies == sequential_plies
    assert wave.widths, "the wave seam was never used: the search stayed scalar"
    assert max(wave.widths) >= 2, (
        f"every wave carried one leaf ({wave.widths}); nothing was batched, so "
        f"no round trip was saved"
    )
    assert max(wave.widths) <= selfplay.gumbel_m_full, (
        "a wave was wider than the search's candidate count, which is what "
        "InferenceConfig.max_wave is sized against (validate_wave_fit)"
    )
    assert wave_stub.calls < sequential_stub.calls, (
        f"the wave worker made {wave_stub.calls} round trips against the "
        f"sequential worker's {sequential_stub.calls}: the wave path is wired "
        f"but is not saving anything"
    )
    # Same questions, fewer envelopes: the roots and forced moves that stay on
    # the scalar seam, plus every leaf that travelled in a wave.
    scalar_trips = wave_stub.calls - len(wave.widths)
    assert scalar_trips + sum(wave.widths) == sequential_stub.calls, (
        "the wave search evaluated a different number of positions than the "
        "sequential one; waves reorder round trips, never questions"
    )


def test_a_worker_built_without_a_wave_evaluator_never_waves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design D3: the ``--serial-debug`` construction stays scalar, always.

    :func:`~komugi.selfplay.orchestrate.serial_debug` builds a
    :class:`SelfPlayWorker` positionally, without the wave argument -- it holds
    an in-process evaluator, there is no ticket to widen, and it is the
    bit-identical reference the replay claim rests on. This pins what that
    construction hands the search: ``wave_evaluator=None``, on every move of
    the game, whatever the run's ``max_wave`` says.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    seen: list[AnyWaveEvaluator | None] = []
    real_search = GumbelSearch

    def spy(
        evaluator: AnyEvaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: AnyWaveEvaluator | None = None,
    ) -> GumbelSearch:
        """Record what the worker asked for, then build the real search."""
        seen.append(wave_evaluator)
        return real_search(evaluator, config, wave_evaluator=wave_evaluator)

    monkeypatch.setattr("komugi.selfplay.worker.GumbelSearch", spy)
    plies = _play_with(
        tmp_path / "serial",
        _selfplay_config(0.0, m=4, n=8),
        20260828,
        _StubEvaluator(num_actions()),
        None,
    )
    assert len(seen) == plies > 0
    assert seen == [None] * plies


class _ReadoutRecordingSearch(GumbelSearch):
    """The real search, plus a log of every result it hands the worker.

    A subclass rather than a wrapper function because the worker holds the
    object and calls ``search`` on it: overriding the method is the one way to
    see every move's :class:`~komugi.search.gumbel.SearchResult` without
    changing what the worker does with it. Everything else -- the evaluator,
    the config, the wave seam -- is the real construction, untouched.
    """

    def __init__(
        self,
        evaluator: AnyEvaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: AnyWaveEvaluator | None = None,
        log: list[SearchResult],
    ) -> None:
        super().__init__(evaluator, config, wave_evaluator=wave_evaluator)
        self._log = log

    def search(self, state: SearchState) -> SearchResult:
        """Run the real search, recording its result on the way out."""
        result = super().search(state)
        self._log.append(result)
        return result


@pytest.mark.parametrize("waves", [False, True], ids=["sequential", "wave"])
def test_the_worker_arms_the_vector_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waves: bool
) -> None:
    """KOM-173 §4: the worker's scalar seam is built with ``with_wdl=True``.

    Review F4: that flag was unobservable from the outside. The worker discards
    the three readouts (they are not shard columns yet), the three-tuple wave
    seam is accepted whatever the scalar seam answered, and so ``with_wdl=False``
    at the worker's :func:`~komugi.selfplay.adapter.make_evaluator` call left the
    whole suite green while disarming the accumulator that KOM-173 (ii) is to
    build on.

    ``root_wdl`` is what makes it observable: it is finite **only** if the ROOT's
    answer carried a vector, because that answer is what allocates the
    accumulator -- a disarmed worker reports the all-``NaN`` triple of
    ``test_a_two_tuple_evaluator_leaves_every_readout_nan`` on every move. Both
    schedules are covered, since the root goes through the scalar seam on each.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    results: list[SearchResult] = []
    selfplay = _selfplay_config(0.0, m=4, n=8)

    def spy(
        evaluator: AnyEvaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: AnyWaveEvaluator | None = None,
    ) -> GumbelSearch:
        """Build the real search, told where to record its results."""
        return _ReadoutRecordingSearch(
            evaluator, config, wave_evaluator=wave_evaluator, log=results
        )

    monkeypatch.setattr("komugi.selfplay.worker.GumbelSearch", spy)
    stub = _StubEvaluator(num_actions())
    wave = _CountingWaveEvaluator(stub, selfplay.draw_score) if waves else None
    plies = _play_with(tmp_path / "armed", selfplay, 20260828, stub, wave)

    assert len(results) == plies > 0
    for ply, result in enumerate(results):
        readout = np.asarray(result.root_wdl, dtype=np.float64)
        assert readout.shape == (3,)
        assert bool(np.isfinite(readout).all()), (
            f"move {ply} reported root_wdl={result.root_wdl.tolist()}: the "
            f"worker's scalar seam answered with a two-tuple, so the searched-"
            f"WDL accumulator was never allocated (make_evaluator's with_wdl)"
        )
        assert float(readout.sum()) == pytest.approx(1.0, abs=1e-6)
    if waves:
        assert wave is not None and wave.widths, "the wave seam was never used"


def test_the_wave_seam_is_built_only_above_max_wave_one() -> None:
    """Design D1: ``max_wave > 1`` is the enable, and there is no other knob."""
    client = cast(SlotClient, object())  # never touched at construction time
    selfplay = _selfplay_config(0.0, m=4, n=8)
    assert _wave_evaluator_for(client, selfplay, _inference_config(max_wave=1)) is None
    assert _wave_evaluator_for(client, selfplay, _inference_config(max_wave=4)) is not None


def _mlh_selfplay(lam: float) -> SelfPlayConfig:
    """A four-candidate config at ``mlh_utility_lambda = lam``.

    ``draw_score=0.0`` is not a choice: :class:`SelfPlayConfig` refuses a
    positive lambda beside contempt (KOM-159 review F13), which is why
    :func:`_selfplay_config`'s ``0.2`` cannot be reused here.
    """
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=4,
        gumbel_n_full=8,
        gumbel_m_fast=4,
        gumbel_n_fast=8,
        prob_fast=0.0,
        draw_score=0.0,
        mlh_utility_lambda=lam,
    )


def _unpublishing_wave(
    states: Sequence[SearchState],
) -> Sequence[tuple[npt.NDArray[np.float32], float]]:
    """A bare wave closure: the shape the *search* accepts, naming no context.

    Exactly what :func:`~komugi.selfplay.adapter.make_wave_evaluator` returned
    before KOM-171, and what the worker must refuse while the term is live --
    there is no way to show it shares the scalar seam's root.
    """
    raise AssertionError(
        f"a wave of {len(states)} was evaluated; the refusal under test "
        f"happens at construction, so nothing here should ever be called"
    )


def _worker_over(
    directory: Path,
    selfplay: SelfPlayConfig,
    wave_evaluator: AnyWaveEvaluator | None,
    context: SearchContext,
) -> SelfPlayWorker:
    """Construct a worker over `directory` and hand it back, unplayed.

    Construction is the whole subject: the context-identity guard fires there
    (design R9 -- a search that discovered at its fiftieth leaf that its wave
    answers were centred on the wrong position would already have written a
    tree), so no game is ever played here.
    """
    writer = ShardWriter(directory, SCHEMA)
    try:
        return SelfPlayWorker(
            _StubEvaluator(num_actions()),
            selfplay,
            WorkerConfig(
                worker_id=1,
                incarnation=0,
                run_seed=20260912,
                games=1,
                shard_dir=str(directory),
            ),
            writer,
            wave_evaluator,
            search_context=context,
        )
    finally:
        writer.close()


def test_a_live_wave_seam_on_another_context_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    """KOM-159b review F2: the two seams share ONE context, and it is checked.

    The failure this closes is silent and arithmetic. ``GumbelSearch`` evaluates
    the root through the scalar seam and the halving rounds' leaves through the
    wave seam, so if the wave seam holds a *different*
    :class:`~komugi.selfplay.adapter.SearchContext` then its own first leaf
    sets that context's ``m_root`` -- and every wave leaf of every search is
    scored against the first leaf of the first wave instead of against the
    root. Both seams are individually well-formed (each has a context, so
    ``_require_context`` is satisfied), the worker plays on, and the only
    symptom is a term centred on the wrong ply.

    Three constructions, because the guard has to accept the right one:
    the mismatched pair is refused, a bare closure that names no context at all
    is refused (nothing can be shown about it), and the matched pair -- what
    ``_run_lane`` and ``worker_main`` build -- goes through.
    """
    selfplay = _mlh_selfplay(0.05)
    lane = SearchContext()
    stranger = SearchContext()
    seam = _wave_evaluator_for(
        cast(SlotClient, object()), selfplay, _inference_config(max_wave=4), lane
    )
    assert seam is not None, "max_wave=4 built no wave seam; the pair cannot be made"

    with pytest.raises(ValueError, match="different SearchContext"):
        _worker_over(tmp_path / "mismatched", selfplay, seam, stranger)

    with pytest.raises(ValueError, match="does not publish"):
        _worker_over(tmp_path / "unpublished", selfplay, _unpublishing_wave, lane)

    matched = _worker_over(tmp_path / "matched", selfplay, seam, lane)
    assert matched.config.worker_id == 1, (
        "the production pairing -- one context, both seams -- was refused"
    )


def test_at_lambda_zero_a_mismatched_wave_context_is_accepted(
    tmp_path: Path,
) -> None:
    """The same mismatched pair, at ``lambda = 0``, is not an error (review F9).

    The λ = 0 path must stay the path that predates the term, instruction for
    instruction: the reference is never read, so *which* context each seam
    holds cannot change a single number. A guard that fired here would be a new
    refusal on the arm every campaign through 5 ran, and would break the
    bit-identity claim T6 rests on by making an off lever observable.

    The mismatch is the *same* one the node above refuses, so the two together
    say the check is conditional on the lever rather than absent.
    """
    selfplay = _mlh_selfplay(0.0)
    seam = _wave_evaluator_for(
        cast(SlotClient, object()),
        selfplay,
        _inference_config(max_wave=4),
        SearchContext(),
    )
    assert seam is not None

    worker = _worker_over(tmp_path / "off", selfplay, seam, SearchContext())

    assert worker.config.games == 1, (
        "a lambda = 0 worker was refused for a context nothing reads"
    )
    # And the bare closure -- every pre-KOM-171 caller's shape -- too.
    assert (
        _worker_over(
            tmp_path / "off-bare", selfplay, _unpublishing_wave, SearchContext()
        ).config.games
        == 1
    )


def _inference_config(max_wave: int) -> InferenceConfig:
    """A seam config of this run's shapes, at `max_wave` rows per ticket."""
    return InferenceConfig(
        in_planes=Game.observation_size()[0],
        num_actions=num_actions(),
        max_batch=8,
        max_wave=max_wave,
    )


class _FakeClient:
    """A :class:`~komugi.selfplay.inference.SlotClient` that attaches to nothing.

    The wiring tests below are about *which objects get built for which lane*,
    so they must not need a segment, a server or a game -- what they assert is
    that each lane's wave evaluator was built over that lane's own client, and
    that is visible at construction time.
    """

    def __init__(
        self,
        shm_name: str,
        worker_id: int,
        config: InferenceConfig,
        *,
        lanes: int = 1,
    ) -> None:
        self.shm_name = shm_name
        self.worker_id = worker_id
        self.lanes = lanes

    def close(self) -> None:
        """Nothing is mapped; nothing to unmap."""


class _FakeWriter:
    """A :class:`~komugi.train.shards.ShardWriter` that owns no directory."""

    def __init__(self, shard_dir: str, schema: object) -> None:
        self.shard_dir = shard_dir

    def close(self) -> None:
        """Nothing was buffered; nothing to seal."""


class _RecordingWorker:
    """A :class:`SelfPlayWorker` that records its construction and plays nothing."""

    constructions: ClassVar[list[tuple[object, AnyWaveEvaluator | None]]] = []

    def __init__(
        self,
        evaluator: object,
        selfplay: SelfPlayConfig,
        config: WorkerConfig,
        writer: object,
        wave_evaluator: AnyWaveEvaluator | None = None,
        search_context: SearchContext | None = None,
    ) -> None:
        # ``search_context`` is the per-slot moves-left root reference
        # ``worker_main`` / ``_run_lane`` now hand every worker (KOM-159b);
        # this fake records nothing about it.
        _RecordingWorker.constructions.append((evaluator, wave_evaluator))

    def run(self) -> int:
        """No games, no rows."""
        return 0


def _wire_worker_main(
    monkeypatch: pytest.MonkeyPatch, max_wave: int, games_in_flight: int
) -> list[tuple[object, AnyWaveEvaluator | None]]:
    """Run :func:`worker_main` over fakes; return what it constructed per lane."""
    monkeypatch.setattr("komugi.selfplay.worker.SlotClient", _FakeClient)
    monkeypatch.setattr("komugi.selfplay.worker.ShardWriter", _FakeWriter)
    monkeypatch.setattr("komugi.selfplay.worker.SelfPlayWorker", _RecordingWorker)
    _RecordingWorker.constructions.clear()
    extra_lanes = (
        ()
        if games_in_flight == 1
        else (
            LaneSpec(
                lane=1, slot_id=1, shm_name="lane-1", shard_dir="dir-1", games=1
            ),
        )
    )
    worker_main(
        0,
        0,
        7,
        1,
        "dir-0",
        "lane-0",
        _selfplay_config(0.0, m=4, n=8, games_in_flight=games_in_flight),
        _inference_config(max_wave),
        extra_lanes,
    )
    return list(_RecordingWorker.constructions)


@pytest.mark.parametrize("games_in_flight", [1, 2])
def test_max_wave_one_builds_exactly_the_objects_it_always_did(
    monkeypatch: pytest.MonkeyPatch, games_in_flight: int
) -> None:
    """Design D2: at the default width no wave evaluator exists at all.

    Under one game in flight and under two: a run that did not ask for waves
    must get the worker it had before they existed, and "did not ask" is
    ``max_wave == 1`` -- the width its segments were sized for.
    """
    built = _wire_worker_main(monkeypatch, max_wave=1, games_in_flight=games_in_flight)
    assert len(built) == games_in_flight
    assert [wave for _client, wave in built] == [None] * games_in_flight


@pytest.mark.parametrize("games_in_flight", [1, 2])
def test_max_wave_above_one_wires_each_lane_its_own_wave_seam(
    monkeypatch: pytest.MonkeyPatch, games_in_flight: int
) -> None:
    """Design D2: waves are on, and a lane's wave rides its OWN client.

    The interleaved case is the one worth pinning: a wave is a ticket, a ticket
    belongs to one segment, and a lane submitting on its sibling's client would
    ring a doorbell it does not own. Distinct clients, distinct closures.
    """
    built = _wire_worker_main(monkeypatch, max_wave=4, games_in_flight=games_in_flight)
    assert len(built) == games_in_flight
    assert all(wave is not None for _client, wave in built)
    clients = [client for client, _wave in built]
    assert len({id(client) for client in clients}) == games_in_flight
    assert len({id(wave) for _client, wave in built}) == games_in_flight


def test_game_rng_and_move_seeds_use_distinct_key_lengths() -> None:
    """The mode-draw stream and the move-seed stream must never collide.

    The game rng derives from a length-3 spawn key and every move seed from a
    length-4 key; SeedSequence differentiates key LENGTHS, which is the whole
    independence argument (wave-9b review F8). This pins it directly: the
    game-level stream's first draws differ from every move seed's stream.
    """
    seed_seq = game_seed(7, 3, 0, 5)
    game_stream = np.random.default_rng(seed_seq).integers(0, 2**63, size=4)
    move_streams = [
        np.random.default_rng(move_seed(seed_seq, ply)).integers(0, 2**63, size=4)
        for ply in range(4)
    ]
    for ply, stream in enumerate(move_streams):
        assert not np.array_equal(game_stream, stream), (
            f"move seed at ply {ply} reproduces the game-level stream; the "
            f"mode draw and the Gumbel noise would be coupled"
        )


def test_game_id_components_are_bounds_checked() -> None:
    """A component at the stride collides with a neighbour; it must raise."""
    stride = 1_000_000
    for bad in (
        {"worker_id": stride},
        {"incarnation": stride},
        {"game_index": stride},
        {"worker_id": -1},
    ):
        kwargs = {"worker_id": 1, "incarnation": 0, "game_index": 2, **bad}
        with pytest.raises(ValueError, match="game_id"):
            game_id(**kwargs)


def test_game_id_inverts_to_the_triple_it_packed() -> None:
    """KOM-014 §6.7: an audit recovers the spawn key from a shard's column."""
    for triple in ((0, 0, 0), (1, 0, 2), (27, 3, 999_999), (999_999, 999_999, 0)):
        assert unpack_game_id(game_id(*triple)) == triple
    with pytest.raises(ValueError, match="game_id"):
        unpack_game_id(-1)


# --------------------------------------------------------------------------
# KOM-014 -- a booked game's rows (#14, #15)
# --------------------------------------------------------------------------


_BOOK_PREFIX_PLIES: Final[int] = 4
"""The planted prefix length ``L``. Four is :data:`book.MIN_PLIES`, and it
leaves :data:`_SHORT_GAME_PLIES` - 4 = 2 searched plies -- enough that
``len(rows)`` and the true ply count are both non-zero and different, which is
what #15 needs to be able to fail."""


def _legal_prefix(plies: int, seed: int) -> tuple[int, ...]:
    """Return `plies` actions that replay legally from a fresh :class:`Game`."""
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    for _ply in range(plies):
        assert not state.is_terminal(), "the planted line ended before its budget"
        legal = np.nonzero(state.legal_mask())[0]
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return tuple(actions)


def _write_one_entry_book(path: Path, actions: Sequence[int]) -> Path:
    """Write a loadable one-entry book holding `actions` (KOM-014 §2.2).

    Hand-built rather than mined by :func:`~komugi.selfplay.book.build_book`:
    what is under test here is the *worker's* arithmetic on a booked game, and
    a builder run would make these two nodes depend on a sampler's draw. Every
    manifest key :func:`~komugi.selfplay.book.load_book` reads is present, so
    the file is accepted (and would be refused) for the same reasons a real one
    is. Restated in each test module that needs it rather than imported across
    directories -- see ``tests/campaign/behavior/campaign_rig.py`` on why a
    cross-directory test import depends on pytest's collection order.
    """
    entry = BookEntry(
        id="gen0001-000000-0badc0de",
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


def _play_booked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, list[dict[str, torch.Tensor]], tuple[int, ...]]:
    """Play one **booked** game and read its rows back off the shards."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    prefix = _legal_prefix(_BOOK_PREFIX_PLIES, 2026)
    book_file = _write_one_entry_book(tmp_path / "books" / "gen_1.json", prefix)
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
            # 1.0 rather than 0.1: this node is about what a booked game
            # records, not about how often one happens (#16 owns the rate), and
            # a 10% draw would make it a test that usually tests nothing.
            book_path=str(book_file),
            book_fraction=1.0,
        ),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=20260822,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )
    written = worker.play_game(0)
    writer.close()
    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    return written, list(reader), prefix


def test_a_booked_game_writes_no_rows_for_prefix_plies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#14: the row set is exactly plies ``L … T-1``, and ``min(ply) == L``.

    The prefix is played and not recorded (§4.3): there is no target for a
    recorded stem ply (the only available one is the source generation's, a
    stale label already in the window) and the deviation ply's target would be
    a lie -- a one-hot row teaching the policy head that a deliberately
    flattened sample *is* the improved policy.
    """
    written, rows, prefix = _play_booked(tmp_path, monkeypatch)
    plies = [int(row["ply"]) for row in rows]
    assert plies, "a booked game recorded no row at all"
    total = max(plies) + 1
    assert min(plies) == _BOOK_PREFIX_PLIES == len(prefix)
    assert plies == list(range(_BOOK_PREFIX_PLIES, total))
    assert written == len(rows) == total - _BOOK_PREFIX_PLIES
    assert len(rows) != total, (
        "the prefix cost no rows, so this fixture cannot tell a booked game "
        "from an unbooked one"
    )
    # The line the game actually opened with is the book's, replayed: the first
    # recorded position is the one the prefix reaches.
    state = GameState(_ShortGame(), 0.2)
    for action in prefix:
        assert bool(state.legal_mask()[action])
        state.play(action)
    assert not state.is_terminal()
    assert int(rows[0]["to_play"]) == state.to_play()


def test_a_booked_games_mlh_target_counts_absolute_plies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#15: ``mlh_target == (T - ply) * MLH_SCALE`` with ``T`` the TRUE count.

    **Red against an implementation that passes ``len(rows)`` to**
    ``finalize_rows``. For this fixture ``len(rows) == T - 4``, so that
    implementation would label the first row ``(T - 4 - 4) * MLH_SCALE`` -- a
    *negative* moves-left target, and the labels would be wrong on every booked
    game while every unbooked game stayed correct (§1.5, §6.4).
    """
    _written, rows, _prefix = _play_booked(tmp_path, monkeypatch)
    total = max(int(row["ply"]) for row in rows) + 1
    assert total > len(rows), "the prefix must make the two counts differ"
    for row in rows:
        ply = int(row["ply"])
        assert float(row["mlh_target"]) == pytest.approx((total - ply) * MLH_SCALE)
        assert float(row["mlh_target"]) > 0.0
        assert float(row["mlh_target"]) != pytest.approx((len(rows) - ply) * MLH_SCALE)


# --------------------------------------------------------------------------
# The draft temperature schedule (KOM-036 §3.2, acceptance #14-#17, #21)
# --------------------------------------------------------------------------

_DRAFTING_FEN: Final[str] = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
)
"""Intermediate mode's start: an empty board, full hands, **both draft bits set**.

Used rather than ``Game()`` because ``Game()`` follows ``GUNGI_MODE``, whose
default (``intro``) starts from a laid-out board past the draft -- so no ply of
an intro game is a draft ply and every draft assertion below would be vacuous.
"""

_EXPERIMENT_DRAFT_SCALE: Final[float] = 2.0
"""§3.2's pre-registered draft scale: the root samples from ``softmax(L/2)``."""

_EXPERIMENT_FAST_SCALE: Final[float] = 0.5
"""§2.3's pre-registered fast damping: the root samples from ``softmax(2L)``."""


def _scale_config(fast_scale: float, draft_scale: float) -> SelfPlayConfig:
    """A production-budget config carrying one pair of schedule scales."""
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        prob_fast=0.75,
        draw_score=0.2,
        fast_gumbel_scale=fast_scale,
        draft_gumbel_scale=draft_scale,
    )


def _scale_worker(
    directory: Path, fast_scale: float, draft_scale: float
) -> SelfPlayWorker:
    """A worker whose only interesting property is its two schedule scales."""
    return SelfPlayWorker(
        _StubEvaluator(num_actions()),
        _scale_config(fast_scale, draft_scale),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=20260907,
            games=1,
            shard_dir=str(directory),
        ),
        ShardWriter(directory, SCHEMA),
    )


@pytest.mark.parametrize(
    ("fast_scale", "draft_scale"),
    [
        (_EXPERIMENT_FAST_SCALE, _EXPERIMENT_DRAFT_SCALE),
        # A second, asymmetric pair whose four cells are four *distinct*
        # numbers. The experiment's own pair has 1.0 twice (§3.2's intended
        # collision), so a lookup table keyed on the wrong cell could survive
        # it; 0.25 x 4.0 gives {1.0, 4.0, 0.25, 1.0}... still colliding, so
        # 0.25 x 8.0 is used: {1.0, 8.0, 0.25, 2.0}, all distinct.
        (0.25, 8.0),
    ],
)
def test_the_scale_is_the_product_of_the_draft_and_fast_factors(
    tmp_path: Path, fast_scale: float, draft_scale: float
) -> None:
    """#14: ``_scale_for`` is the §3.2 product over all four cells.

    Multiplicative because the knob is a root temperature (§2.2) and
    temperatures compose on the log-odds scale. The product is asserted against
    an expression built from the two factors rather than against four literals,
    so the node is red against a rule that overrides instead of composing --
    "draft wins" would put the fast draft cell at ``draft_scale`` rather than at
    the product, which is precisely the rejected alternative §3.2 records.
    """
    worker = _scale_worker(tmp_path / "product", fast_scale, draft_scale)
    for fast in (False, True):
        for in_draft in (False, True):
            expected = (draft_scale if in_draft else 1.0) * (
                fast_scale if fast else 1.0
            )
            assert worker._scale_for(fast, in_draft) == expected, (  # noqa: SLF001
                f"cell (fast={fast}, in_draft={in_draft}) is not the product"
            )

    # The four cells of the asymmetric pair are four distinct numbers, so no
    # cell can be passing by coincidence with another.
    if fast_scale != _EXPERIMENT_FAST_SCALE:
        cells = {
            worker._scale_for(fast, in_draft)  # noqa: SLF001
            for fast in (False, True)
            for in_draft in (False, True)
        }
        assert len(cells) == 4


def test_the_scale_is_exactly_one_at_the_shipped_defaults(tmp_path: Path) -> None:
    """§3.4: both knobs default to ``1.0``, so the schedule is OFF as shipped.

    This is what makes landing KOM-036 bit-identical for every existing
    self-play test, golden and replay claim -- multiplying a float64 by exactly
    ``1.0`` is the identity -- and therefore what discharges KOM-014 §5.1's
    ordering constraint by construction rather than by scheduling.
    """
    shipped = SelfPlayConfig(workers=1, process_batch_size=1)
    assert shipped.fast_gumbel_scale == 1.0
    assert shipped.draft_gumbel_scale == 1.0

    worker = _scale_worker(tmp_path / "defaults", 1.0, 1.0)
    for fast in (False, True):
        for in_draft in (False, True):
            assert worker._scale_for(fast, in_draft) == 1.0  # noqa: SLF001


def test_the_gumbel_config_carries_the_scheduled_scale(tmp_path: Path) -> None:
    """#15: ``_gumbel_config`` passes the product and changes nothing else.

    The signature grew an argument and the config grew a field; ``m``, ``n``,
    ``seed`` and ``num_actions`` must be exactly what they were before, per
    cell, or the schedule has quietly become a budget change as well.
    """
    worker = _scale_worker(
        tmp_path / "config", _EXPERIMENT_FAST_SCALE, _EXPERIMENT_DRAFT_SCALE
    )
    config = _scale_config(_EXPERIMENT_FAST_SCALE, _EXPERIMENT_DRAFT_SCALE)
    seed = 4242
    for fast in (False, True):
        budget = (
            (config.gumbel_m_fast, config.gumbel_n_fast)
            if fast
            else (config.gumbel_m_full, config.gumbel_n_full)
        )
        for in_draft in (False, True):
            built = worker._gumbel_config(fast, in_draft, seed)  # noqa: SLF001
            assert (built.m, built.n) == budget
            assert built.seed == seed
            assert built.num_actions == num_actions()
            assert built.gumbel_scale == worker._scale_for(fast, in_draft)  # noqa: SLF001

    # And at the shipped defaults the config is bit-identical to the one this
    # code built before the field was threaded at all.
    off = _scale_worker(tmp_path / "config-off", 1.0, 1.0)
    for fast in (False, True):
        for in_draft in (False, True):
            built = off._gumbel_config(fast, in_draft, seed)  # noqa: SLF001
            assert built == GumbelConfig(
                num_actions=num_actions(),
                m=built.m,
                n=built.n,
                seed=seed,
            )


def test_a_fast_draft_ply_lands_at_the_default_scale(tmp_path: Path) -> None:
    """#16: at ``(draft 2.0, fast 0.5)`` a fast draft ply is exactly ``1.0``.

    §3.2 calls this cell *"intended rather than accidental"*, so it gets its own
    node. At ``prob_fast = 0.75`` three of four draft plies are fast, and the
    rejected "draft overrides fast" rule would put every one of them at ``2.0``;
    the product instead guarantees only that no draft ply is ever damped below
    the paper's default, while the **full** draft searches -- whose policy
    targets the P0b A/B weights most -- are the hot ones.
    """
    worker = _scale_worker(
        tmp_path / "cell", _EXPERIMENT_FAST_SCALE, _EXPERIMENT_DRAFT_SCALE
    )
    # Exactly, not approximately: 0.5 and 2.0 are both binary-exact and their
    # product is 1.0 to the bit, which is what makes this cell the search's own
    # default rather than something near it.
    assert worker._scale_for(True, True) == 1.0  # noqa: SLF001
    assert worker._scale_for(False, True) == _EXPERIMENT_DRAFT_SCALE  # noqa: SLF001
    assert worker._scale_for(True, False) == _EXPERIMENT_FAST_SCALE  # noqa: SLF001
    assert worker._scale_for(False, False) == 1.0  # noqa: SLF001

    # And the config the worker hands the search at that cell is the one the
    # search would have built for itself: DEFAULT gumbel_scale, not a near miss.
    built = worker._gumbel_config(True, True, 7)  # noqa: SLF001
    assert built == GumbelConfig(
        num_actions=num_actions(), m=built.m, n=built.n, seed=7
    )


_REFUSED_SCALE_PAIRS: Final[tuple[tuple[float, float], ...]] = (
    (1e-320, 1e-10),
    (1e200, 1e200),
)
"""§3.2's two counterexamples to finite-and-positive-on-each-factor.

``1e-320`` is a subnormal -- finite and strictly positive -- and
``1e-320 * 1e-10 == 0.0``; ``1e200 * 1e200 == inf``. Either product makes
``GumbelConfig.__post_init__`` raise on the first fast draft ply, hours into a
campaign, which is exactly the failure the config validation exists to prevent.
"""

_ACCEPTED_SCALE_PAIRS: Final[tuple[tuple[float, float], ...]] = (
    (1e-6, 1e-6),
    (1e-6, 1e6),
    (1e6, 1e-6),
    (1e6, 1e6),
    (0.5, 2.0),
    (1.0, 1.0),
    (2.0, 0.5),
)
"""The accepted range, **including its endpoints** and not only the experiment's.

Rev 1's version of #17 was the sweep alone, written from ``{0.5, 1.0, 2.0}``:
green, and certifying a false invariant. The endpoints are what make the sweep
say something -- ``1e6 * 1e6 == 1e12``, comfortably finite.
"""


@pytest.mark.parametrize(("draft_scale", "fast_scale"), _REFUSED_SCALE_PAIRS)
def test_no_configured_scale_pair_can_produce_a_config_gumbel_rejects(
    draft_scale: float, fast_scale: float
) -> None:
    """#17, first half: the bound **fires** on the two composition counterexamples.

    Per this repo's L20 a guard needs a test that proves it fires. Red against
    a finite-and-positive validation, which accepts both of these pairs and then
    produces ``0.0`` and ``inf`` respectively at ``_scale_for``.
    """
    with pytest.raises(ValueError, match="gumbel_scale"):
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            draft_gumbel_scale=draft_scale,
            fast_gumbel_scale=fast_scale,
        )

    # And the product really is the degenerate the guard is protecting against,
    # so this node is not merely asserting that some validation exists.
    product = draft_scale * fast_scale
    assert product == 0.0 or not np.isfinite(product)
    with pytest.raises(ValueError, match="gumbel_scale"):
        GumbelConfig(num_actions=num_actions(), m=2, n=2, seed=1, gumbel_scale=product)


@pytest.mark.parametrize(("fast_scale", "draft_scale"), _ACCEPTED_SCALE_PAIRS)
def test_every_accepted_scale_pair_composes_to_a_config_gumbel_accepts(
    tmp_path: Path, fast_scale: float, draft_scale: float
) -> None:
    """#17, second half: the invariant holds over everything the bound accepts.

    Every composed product is finite and ``> 0``, so
    ``GumbelConfig.__post_init__`` can never fire from this path -- which is the
    invariant, not a coincidence, once the bound is ``[1e-6, 1e6]`` rather than
    finiteness and sign.
    """
    worker = _scale_worker(
        tmp_path / "sweep", fast_scale=fast_scale, draft_scale=draft_scale
    )
    for fast in (False, True):
        for in_draft in (False, True):
            product = worker._scale_for(fast, in_draft)  # noqa: SLF001
            assert np.isfinite(product) and product > 0.0
            # The consumer accepts it -- asserted through the consumer rather
            # than by restating its bound here.
            built = worker._gumbel_config(fast, in_draft, 3)  # noqa: SLF001
            assert built.gumbel_scale == product


# --------------------------------------------------------------------------
# The adapter's mover-relative draft predicate (KOM-036 §3.3, acceptance #21)
# --------------------------------------------------------------------------


def _drafting_state(draft_w: int, draft_b: int, to_play: int) -> GameState:
    """Return a state at a drafting position with the bits and mover as asked.

    ``board.turn`` is set **before** the :class:`Game` is built, because
    ``Game.__init__`` copies it into ``player`` and ``GameState.to_play`` reads
    that. The draft bits are set directly for the reason
    ``tests/engine/test_gamestate.py`` sets them: they are what the predicate
    reads, and the asymmetric combinations are not reachable with an arbitrary
    side to move (the engine's own turn rule never hands the move to the side
    that has finished while the other still drafts).
    """
    board = board_from_fen(_DRAFTING_FEN)
    board.draft_w = draft_w
    board.draft_b = draft_b
    board.turn = to_play
    return GameState(Game(board), 0.2)


def test_gamestate_in_draft_delegates_to_the_game() -> None:
    """#21: ``GameState.in_draft()`` is the game's predicate at **the mover**.

    No argument, and never a fixed side and never the disjunction. The four
    ``(bits, mover)`` combinations below include the two where the mover has
    finished while the other side is still drafting: there ``in_draft()`` must
    be ``False`` and the disjunction would say ``True``, which is the whole
    discrimination review M2 asked for.
    """
    cases = (
        # (draft_w, draft_b, to_play, expected)
        (1, 1, 0, True),
        (1, 1, 1, True),
        (1, 0, 0, True),
        (1, 0, 1, False),  # black has finished, white has not
        (0, 1, 0, False),  # white has finished, black has not
        (0, 1, 1, True),
        (0, 0, 0, False),
        (0, 0, 1, False),
    )
    for draft_w, draft_b, to_play, expected in cases:
        state = _drafting_state(draft_w, draft_b, to_play)
        assert state.to_play() == to_play
        # The delegation, spelled: the adapter supplies the mover and nothing
        # else. This is the equality the node is named for.
        assert state.in_draft() is state.game.in_draft(state.to_play())
        assert state.in_draft() is expected
        if draft_w != draft_b and not expected:
            # The disjunction would have answered True here, and the schedule
            # would have run this mover's search at the draft temperature.
            assert bool(draft_w or draft_b)

    # Nothing is cached: the predicate changes every ply, so a state that
    # answered once must answer the board's *current* bits, not its first ones.
    state = _drafting_state(1, 1, 0)
    assert state.in_draft()
    live_board: Any = state.game._board  # noqa: SLF001
    live_board.draft_w = 0
    assert not state.in_draft(), (
        "GameState cached a per-ply predicate; it must delegate on every call"
    )

    # And it is not on the search's protocol: the search must not learn phases.
    assert not hasattr(EvaluableState, "in_draft")


def test_gamestate_in_draft_tracks_the_mover_across_a_played_game() -> None:
    """The delegation holds on states the **engine** produced, not only built ones.

    Complements the constructed table above: every ply of a real drafting game,
    ``in_draft()`` is the mover's own bit. The engine's turn rule means the
    mover is always the drafting side while any side drafts, so this half cannot
    discriminate the two predicates on its own -- which is exactly why the
    constructed table exists.
    """
    rng = np.random.default_rng(20260907)
    state = GameState(Game(board_from_fen(_DRAFTING_FEN)), 0.2)
    seen: set[bool] = set()
    plies = 0
    while plies < 200 and not state.is_terminal():
        board: Any = state.game._board  # noqa: SLF001
        mover_bit = bool(board.draft_w if state.to_play() == 0 else board.draft_b)
        assert state.in_draft() is mover_bit
        seen.add(mover_bit)
        legal = np.nonzero(state.legal_mask())[0]
        if legal.size == 0:
            break
        state.play(int(rng.choice(legal)))
        plies += 1
    assert plies > 0
    assert True in seen, "the game never started in the draft phase"


# ---------------------------------------------------------------------------
# KOM-141 -- the repetition rule a self-play game is refereed under
# ---------------------------------------------------------------------------


def _rule_worker(directory: Path, rule: RepetitionRule) -> SelfPlayWorker:
    """A worker whose only interesting property is its repetition rule."""
    return SelfPlayWorker(
        _StubEvaluator(num_actions()),
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            draw_score=0.2,
            repetition_rule=rule,
        ),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=20260909,
            games=1,
            shard_dir=str(directory),
        ),
        ShardWriter(directory, SCHEMA),
    )


def test_the_workers_game_is_built_under_the_configs_repetition_rule(
    tmp_path: Path,
) -> None:
    """KOM-141: ``_new_game`` reads the rule rather than assuming one.

    The construction site is a method precisely so this can be asserted without
    a shard rig, an inference server or a played game -- the thing under test is
    one keyword argument, and a test that had to play a game to see it would be
    a test of the loop.

    *Non-vacuity:* both values are built, so the node is red against a
    ``_new_game`` that hardcoded either.
    """
    for rule in (RepetitionRule.FULL, RepetitionRule.PLACEMENT):
        worker = _rule_worker(tmp_path / rule.value, rule)
        assert worker._new_game().repetition_rule is rule  # noqa: SLF001


def test_a_self_play_config_defaults_to_the_full_repetition_rule(
    tmp_path: Path,
) -> None:
    """Nothing flips silently: self-play trains under Komugi's repaired key.

    ``RepetitionRule.PLACEMENT`` re-introduces the rules-audit F1-F4 aliasing on
    purpose and is a REFEREE device; a self-play run must never reach it without
    an operator having written it down. Both halves are asserted -- the config's
    default and the game the worker actually builds from it -- because a default
    that never reached the referee would be a field, not a rule.
    """
    assert SelfPlayConfig(
        workers=1, process_batch_size=1
    ).repetition_rule is RepetitionRule.FULL

    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        SelfPlayConfig(workers=1, process_batch_size=1, draw_score=0.2),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=20260909,
            games=1,
            shard_dir=str(tmp_path / "default"),
        ),
        ShardWriter(tmp_path / "default", SCHEMA),
    )
    assert worker._new_game().repetition_rule is RepetitionRule.FULL  # noqa: SLF001
