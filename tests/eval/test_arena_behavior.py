"""The arena against reality: strength, worker parity, and a real net.

Three claims that a structural test cannot make, each expensive enough to earn
its own module (the unit suite next door is the cheap half):

(a) **A value-head difference reaches the games.** Two stubs, identical in
    every way except their value head, produce a match with decisive games in
    which the material-seeking one converts. What this does NOT certify is a
    win *margin*: measured against the correct code, material greed at this
    budget scores near 50% (a one-ply material edge does not convert to wins
    in this game -- 2026-08-29 impl review, R2), so seat-dispatch and
    result-perspective bugs are pinned by the dedicated orientation and
    budget-dispatch tests in ``test_arena_unit.py``, not by this match.
(b) **An arena game is a self-play game.** Same seeds, same budget, same
    evaluator, no fast searches: the move sequence must be the one
    :class:`~komugi.selfplay.worker.SelfPlayWorker` plays. The arena is a
    *different driver over the same loop*, and this is what says so.
(c) **A real :class:`~komugi.net.KomugiNet` plays a whole game on the CPU** and
    what comes out validates as a record.

Heavy bits are skip-guarded like the rest of the behavioural suites. Budgets are
tiny throughout -- the engine adjudicates at 256 turns and every ply costs
``1 + n`` engine expansions, so a realistic 256-simulation budget would make
this module the slowest thing in CI while testing nothing extra.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.net")
pytest.importorskip("komugi.selfplay")

import torch

from komugi.campaign import factory
from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import MAX_COUNT, PIECE_TYPES, observation_size
from komugi.eval import arena
from komugi.eval.gumbel_mover import CountedEvaluator, komugi_entrant
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
)
from komugi.eval.pool_books import PoolBook
from komugi.eval.seats import check_paired_books
from komugi.eval.v1bridge import (
    M8,
    VOIDED,
    BridgeEntrant,
    VoidBucket,
    VoidScope,
    bridged_book_id,
    play_bridged_match,
    resolved_gungi_mode,
)
from komugi.eval.v1runner.counter import EvalCounter
from komugi.net import KomugiNet
from komugi.search.state import Evaluator, WaveEvaluator
from komugi.selfplay.adapter import GameState, make_evaluator
from komugi.selfplay.inference import InferenceConfig, InProcessEvaluator, NetEval
from komugi.selfplay.records import SCHEMA
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig
from komugi.train.checkpoint import save_checkpoint
from komugi.train.shards import ShardReader, ShardWriter

# The epoch-book fixtures live beside §D3's own nodes, and are shared rather
# than re-typed -- the same reason ``test_gumbel_mover.py`` imports its bridged
# fixtures from ``test_v1bridge_behavior``.
from test_pool_books import epoch_book

DRAW_SCORE: float = 0.20
"""``DESIGN.html`` section 3.3's contempt, and ``SelfPlayConfig``'s default."""

# ---------------------------------------------------------------------------
# the observation planes the "strong" stub reads
# ---------------------------------------------------------------------------

PLANES: int = observation_size()[0]
"""The encoder's plane count (128), read from the encoder."""

PIECES: int = len(PIECE_TYPES)
TIERS: int = 3
COLORS: int = 2

OCCUPANCY_PLANES: int = PIECES * TIERS * COLORS
"""Planes 0..83: ``piece_idx * 6 + (tier - 1) * 2 + color_idx`` -- colour is the
fastest-varying axis, which is what lets the block be reshaped to ``(-1, 2)``."""

HAND_BASE: int = OCCUPANCY_PLANES
"""Planes 84..111: ``84 + piece_idx * 2 + color_idx``, constant at
``count / MAX_COUNT``."""

SIDE_TO_MOVE: int = OCCUPANCY_PLANES + PIECES * COLORS + 2
"""Plane 114 (after the two drafting-rights planes): ``0.0`` white, ``1.0`` black."""

TURN_CLOCK: int = SIDE_TO_MOVE + 1
"""Plane 115."""

REPETITION_BASE: int = TURN_CLOCK + 1
"""Planes 116-117: the R2 / R3 threshold planes."""

LAST_MOVE_BASE: int = REPETITION_BASE + 2
"""Planes 118-119: KOM-002's last-move origin and destination."""

PROGRESS_CLOCK: int = LAST_MOVE_BASE + 2
"""Plane 120: KOM-002's progress clock."""

RESERVED_BASE: int = PROGRESS_CLOCK + 1
"""Plane 121, where KOM-002's reserved block starts."""

RESERVED_PLANES: int = 7
"""Planes 121-127, identically zero. Filling one bumps ``PLANE_MAP_VERSION``
without changing the shape, which is why this module counts them rather than
stopping at the last plane that carries a value."""

HAND_WEIGHT: npt.NDArray[np.float32] = np.asarray(
    [MAX_COUNT[piece] for piece in PIECE_TYPES], dtype=np.float32
)
"""Undoes the hand planes' per-type normalisation, so a hand piece and a board
piece count the same."""


def test_the_plane_indices_match_the_encoder() -> None:
    """The layout this module reads is the layout the encoder writes.

    Derived from the encoder's own constants, then checked to account for every
    plane: 84 occupancy + 28 hand + 2 drafting + side-to-move + clock + 2
    repetition + 2 last-move + progress clock + 7 reserved = 128 (KOM-002 §2).

    The accounting runs to ``PLANES`` on purpose. Before KOM-002 it stopped at
    the last plane that carried a value, which happened to be the plane count
    too -- so when the map grew, the constants this module *reads with*
    (:data:`SIDE_TO_MOVE`, :data:`HAND_BASE`) were still right and the total
    was quietly wrong. A layout change lands here rather than as a silently
    wrong material count inside a strength test.
    """
    assert SIDE_TO_MOVE == 114
    assert LAST_MOVE_BASE == 118, "KOM-002 appends after the repetition planes"
    assert PROGRESS_CLOCK == 120
    assert RESERVED_BASE + RESERVED_PLANES == PLANES


def material_by_colour(
    obs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Return ``(N, 2)`` total material per colour: on the board plus in hand.

    Board **and** hand, because in this engine every move except losing a piece
    conserves the sum: a drop (``arata``) moves a piece from hand to board, a
    betrayal converts enemy pieces and pays for them out of the converter's own
    hand, and a capture removes the captured pieces from the board and gives
    them to nobody. So a colour's total falls when and only when its pieces are
    taken, which is what makes this a material differential. A board-only count
    would instead climb every time either player dropped a piece during the
    draft and read that as material from nowhere.
    """
    rows = int(obs.shape[0])
    board = (
        obs[:, :OCCUPANCY_PLANES]
        .reshape(rows, PIECES * TIERS, COLORS, -1)
        .sum(axis=(1, 3))
    )
    hand = obs[:, HAND_BASE : HAND_BASE + PIECES * COLORS, 0, 0].reshape(
        rows, PIECES, COLORS
    )
    return np.asarray(
        board + (hand * HAND_WEIGHT[None, :, None]).sum(axis=1), dtype=np.float32
    )


def mover_material_edge(obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return ``(N,)`` material advantage of the side to move, from the planes."""
    material = material_by_colour(obs)
    black_to_move = obs[:, SIDE_TO_MOVE, 0, 0] > 0.5
    own = np.where(black_to_move, material[:, 1], material[:, 0])
    opponent = np.where(black_to_move, material[:, 0], material[:, 1])
    return np.asarray(own - opponent, dtype=np.float32)


class FlatPolicyStub:
    """Uniform-zero policy, and a value head chosen by the caller.

    The policy is **all zeros** on purpose: with no prior to speak of, the
    search's root sample is a uniform draw of ``m`` legal actions (the Gumbel
    top-k trick over constant logits), so the only thing separating two
    instances of this class is the value they put on the resulting positions.
    That is the experiment -- a strength difference that is *entirely* the
    evaluation.

    Args:
        material: When true, the value is the mover's material advantage
            squashed into ``(-1, 1)``; when false it is a constant ``0``,
            i.e. a player with no opinion about anything.
        scale: Logistic steepness of the material squash. At the default a
            one-piece advantage is worth ~0.46 on the ``[-1, 1]`` value scale:
            big enough to decide between a capture and a quiet move, small
            enough that the signal does not saturate the moment anything at all
            has been taken.
    """

    def __init__(self, *, material: bool, scale: float = 1.0) -> None:
        self._material = material
        self._scale = float(scale)
        self.rows = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a batch; the legal mask is the search's business, not ours."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        rows = int(array.shape[0])
        self.rows += rows
        if self._material:
            edge = mover_material_edge(array) * self._scale
            win = 1.0 / (1.0 + np.exp(-edge.astype(np.float64)))
        else:
            win = np.full((rows,), 0.5, dtype=np.float64)
        wdl = np.stack([win, np.zeros_like(win), 1.0 - win], axis=1).astype(
            np.float32
        )
        return NetEval(
            policy_logits=np.zeros((rows, factory.NUM_ACTIONS), dtype=np.float32),
            wdl=wdl,
            moves_left=np.zeros((rows,), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# (a) strength
# ---------------------------------------------------------------------------

STRENGTH_GAMES: int = 8
STRENGTH_M: int = 6
STRENGTH_N: int = 6
"""Eight games at a six-candidate, six-simulation search.

Sized by what it can *conclude*, then by cost. The gap between the two stubs is
structural rather than marginal -- one grabs material whenever a capture is
among its six sampled moves, the other plays a uniformly random legal move --
and material compounds, so a match this size separates them. It is still a
handful of games: the assertions below are directional, and they are written to
say so.
"""


def test_the_material_stub_beats_the_indifferent_one() -> None:
    """A value-head difference produces decisive games the material stub joins.

    **What this test certifies -- honestly scoped after the 2026-08-29 impl
    review (finding R2).** Measured against the *correct* code, this exact
    match is close to an even split (4-4 decisive at the pinned seed): a
    one-ply material edge over a branching factor near 229 does not convert
    into a reliable win rate at any budget CI can afford, and pushing scale or
    simulations does not change that (probed at scale 2/4/6 and a 4x budget
    -- still a coin flip). So the floors below assert only that the match
    produces decisive games, that the material stub converts some of them, and
    that it is not strictly out-converted; they are satisfied at parity, and
    they would NOT catch a one-evaluator, seat-dispatch or result-perspective
    bug. Those properties are pinned where they can be pinned exactly:
    ``test_results_are_stated_from_whites_perspective`` (engine-truth replay)
    and ``test_the_mover_searches_with_its_own_budget`` (recording search) in
    ``test_arena_unit.py``, and the one-evaluator shape by
    ``test_the_same_match_replays_exactly``. What remains *this* test's own is
    end-to-end: a value-head asymmetry flows through the real search, the real
    engine and the pairing into well-formed, decisive, auditable records.
    """
    strong = arena.ArenaPlayer(
        player=Player(id="material", family="komugi"),
        evaluator_factory=lambda: FlatPolicyStub(material=True),
        m=STRENGTH_M,
        n=STRENGTH_N,
    )
    weak = arena.ArenaPlayer(
        player=Player(id="indifferent", family="komugi"),
        evaluator_factory=lambda: FlatPolicyStub(material=False),
        m=STRENGTH_M,
        n=STRENGTH_N,
    )
    outcome = arena.play_match(
        strong,
        weak,
        arena.ArenaConfig(games=STRENGTH_GAMES, seed=17, draw_score=DRAW_SCORE),
    )

    decisive = [record for record in outcome.records if not record.is_draw]
    strong_wins = sum(1 for record in decisive if record.winner == "material")
    weak_wins = len(decisive) - strong_wins

    assert outcome.summary.games == STRENGTH_GAMES
    assert outcome.summary.decisive == len(decisive)
    assert decisive, (
        f"the match produced no decisive game at all in {STRENGTH_GAMES} games "
        f"(mean length {outcome.summary.mean_plies:.1f} plies): a material-greedy "
        f"player against an indifferent one should convert something, and a "
        f"match of pure draws would let a bare score threshold pass on nothing"
    )
    assert strong_wins >= 1 and strong_wins >= weak_wins, (
        f"the material stub won {strong_wins} of {len(decisive)} decisive games; "
        f"the indifferent stub won {weak_wins}. Seats: "
        f"{[(r.white, r.black, r.result) for r in outcome.records]}"
    )
    # Whatever the result, the match is still well-formed data.
    assert check_paired_books(outcome.match_set()) == ()
    assert MatchSet.of(outcome.records).games_played_by("material") == STRENGTH_GAMES


# ---------------------------------------------------------------------------
# (b) parity with the self-play worker
# ---------------------------------------------------------------------------

PARITY_M: int = 2
PARITY_N: int = 4
PARITY_SEED: int = 4242
PARITY_PAIR: int = 3
PARITY_SEAT: int = 1


class ParityStub:
    """A deterministic evaluator, identical row by row whatever the batch.

    Row-wise by construction: batching must not change an answer, or the arena
    (which submits a halving round as one wave) and the worker (which submits
    one leaf at a time) would disagree for a reason that has nothing to do with
    either loop.
    """

    def __init__(self, seed: int = 5) -> None:
        rng = np.random.default_rng(seed)
        self._base = rng.standard_normal(factory.NUM_ACTIONS).astype(np.float32)
        self._tilt = (rng.standard_normal(factory.NUM_ACTIONS) * 0.25).astype(
            np.float32
        )

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a batch, one independent row at a time."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        rows = int(array.shape[0])
        keys = array.reshape(rows, -1).astype(np.float64).sum(axis=1)
        mix = (np.remainder(keys, 8.0) / 8.0).astype(np.float32)
        policy = (self._base[None, :] + mix[:, None] * self._tilt[None, :]).astype(
            np.float32
        )
        wdl = np.tile(np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1))
        return NetEval(
            policy_logits=policy,
            wdl=wdl,
            moves_left=np.full((rows,), 8.0, dtype=np.float32),
        )


def _worker_rows(shard_dir: Path) -> list[dict[str, torch.Tensor]]:
    """Play one self-play game into `shard_dir` and return its rows in order.

    Driven exactly as ``tests/selfplay`` drives it: the raw evaluator goes in
    (the worker wraps it through ``make_evaluator`` itself), the writer is
    closed before anything is read, and ``prob_fast=0`` with the fast budget set
    equal to the full one makes every move a full search at the arena's budget
    -- the mode draw still happens, on the game's own stream, and still cannot
    perturb the per-move seeds, which are derived independently (design F10d).
    """
    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=PARITY_M,
        gumbel_n_full=PARITY_N,
        gumbel_m_fast=PARITY_M,
        gumbel_n_fast=PARITY_N,
        prob_fast=0.0,
        draw_score=DRAW_SCORE,
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(shard_dir, SCHEMA)
    try:
        SelfPlayWorker(
            ParityStub(),
            config,
            WorkerConfig(
                worker_id=PARITY_PAIR,
                incarnation=PARITY_SEAT,
                run_seed=PARITY_SEED,
                games=1,
                shard_dir=str(shard_dir),
            ),
            writer,
        ).play_game(0)
    finally:
        writer.close()
    reader = ShardReader(shard_dir, schema=SCHEMA)
    assert not reader.skipped, f"unreadable shards in {shard_dir}"
    return list(reader)


def test_an_arena_game_is_the_self_play_game(tmp_path: Path) -> None:
    """Identical seeds and budgets, both seats one evaluator -> the same game.

    The arena's per-move seed is
    :func:`~komugi.selfplay.records.move_seed` over
    :func:`~komugi.selfplay.records.game_seed`, so an arena game at
    ``(pair, seat)`` and a worker game at ``(worker_id=pair,
    incarnation=seat, game_index=0)`` of the same run seed draw the *same*
    Gumbel vector at every ply. Everything else about the two loops is then
    under test: the seat dispatch, the fresh-config-per-move rule, the "play
    ``result.action``, no temperature, no resign" rule, and the engine-owns-
    termination rule.

    The comparison is by **observation**: the worker records one row per ply and
    a row carries the position it was played from, so replaying the arena's move
    list through a fresh engine and matching every ply's observation (and mover)
    pins the move sequences to each other -- one different move and every
    subsequent position differs.

    It is also, incidentally, an end-to-end check of the search's wave
    independence: the worker here evaluates leaves one at a time, the arena
    submits each halving round as one wave, and the games must still coincide.
    """
    rows = _worker_rows(tmp_path / "shards")

    config = arena.ArenaConfig(games=2, seed=PARITY_SEED, draw_score=DRAW_SCORE)
    outcome = arena.play_arena_game(
        arena.ArenaPlayer(
            player=Player(id="white-seat", family="komugi"),
            evaluator_factory=ParityStub,
            m=PARITY_M,
            n=PARITY_N,
        ),
        arena.ArenaPlayer(
            player=Player(id="black-seat", family="komugi"),
            evaluator_factory=ParityStub,
            m=PARITY_M,
            n=PARITY_N,
        ),
        config,
        pair=PARITY_PAIR,
        seat=PARITY_SEAT,
    )

    assert len(outcome.moves) == len(rows) > 1

    replay = Game()
    for ply, (action, row) in enumerate(zip(outcome.moves, rows, strict=True)):
        assert int(row["ply"].item()) == ply
        assert int(row["to_play"].item()) == int(replay.player), (
            f"mover disagreement at ply {ply}"
        )
        assert np.array_equal(
            np.asarray(replay.observation(), dtype=np.float32),
            row["observation"].numpy(),
        ), f"position disagreement at ply {ply}: the games diverged before here"
        replay.play_action(action)

    assert bool(np.asarray(replay.win_state()).any())
    assert outcome.result in {WHITE_WIN, DRAW, BLACK_WIN}


# ---------------------------------------------------------------------------
# (c) a real net, and the ladder driver that builds one
# ---------------------------------------------------------------------------


def small_net_evaluator() -> InProcessEvaluator:
    """A real :class:`~komugi.net.KomugiNet` at the campaign's *test* shape.

    ``factory.SMALL_CONFIG`` is b2c16: the trunk shrinks, the seam does not --
    ``in_planes`` and ``num_actions`` stay at the production values, so this is
    the real class, the real heads and the real inference seam at a cost a CPU
    can pay for a whole 256-turn game.
    """
    net = KomugiNet(factory.SMALL_CONFIG)
    net.eval()
    return InProcessEvaluator(
        net,
        InferenceConfig(
            in_planes=factory.IN_PLANES, num_actions=factory.NUM_ACTIONS
        ),
    )


def test_a_real_net_plays_a_whole_game_on_the_cpu() -> None:
    """A net-vs-itself pair completes and produces records that validate.

    The self-match is the D14 shape: one set of weights, **two distinct player
    ids**, because a record refuses to name one player in both seats. What it
    smoke-tests is everything the stub tests cannot -- a real forward through
    the wave adapter, a real ``(N, A)`` policy, a real softmaxed wdl -- and its
    only claim about the *result* is that the engine adjudicated one.
    """
    outcome = arena.play_match(
        arena.ArenaPlayer(
            player=Player(id="net-a", family="komugi"),
            evaluator_factory=small_net_evaluator,
            m=2,
            n=2,
        ),
        arena.ArenaPlayer(
            player=Player(id="net-b", family="komugi"),
            evaluator_factory=small_net_evaluator,
            m=2,
            n=2,
        ),
        arena.ArenaConfig(games=2, seed=8, draw_score=DRAW_SCORE),
    )

    assert len(outcome.records) == 2
    assert check_paired_books(outcome.match_set()) == ()
    for record in outcome.records:
        assert record.result in {WHITE_WIN, DRAW, BLACK_WIN}
        assert {record.white, record.black} == {"net-a", "net-b"}
    assert outcome.summary.shortest_plies > 1


def test_the_ladder_is_built_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ladder_players`` reverses ``list_checkpoints`` before sampling.

    ``list_checkpoints`` is newest-first, so sampling every ``N`` th of it
    unreversed would both read backwards as a training curve and -- the silent
    half -- re-anchor the whole sample every time a new checkpoint appeared.
    """
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    state = KomugiNet(factory.SMALL_CONFIG).state_dict()
    for step in (0, 10, 20):
        save_checkpoint(tmp_path, step, state, {}, {"step": step})

    players = arena.ladder_players(tmp_path)
    assert [player.player.id for player in players] == [
        "ckpt-0",
        "ckpt-10",
        "ckpt-20",
    ]
    assert {player.player.family for player in players} == {arena.LADDER_FAMILY}
    assert (players[0].m, players[0].n) == (arena.DEFAULT_M, arena.DEFAULT_N)

    strided = arena.ladder_players(tmp_path, every_n=2)
    assert [player.player.id for player in strided] == ["ckpt-0", "ckpt-20"]

    # The factory is a real one: it loads the checkpoint and answers the seam.
    evaluator = players[-1].evaluator_factory()
    start = Game()
    answer = evaluator.evaluate(
        np.asarray(start.observation(), dtype=np.float32)[None],
        np.asarray(start.valid_moves(), dtype=np.bool_)[None],
    )
    assert answer.policy_logits.shape == (1, factory.NUM_ACTIONS)
    assert answer.wdl.shape == (1, 3)

    with pytest.raises(ValueError, match="every_n"):
        arena.ladder_players(tmp_path, every_n=0)
    assert arena.ladder_players(tmp_path / "nothing-here") == ()


def test_a_stale_checkpoint_is_refused_before_load_state_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-002 §5.3 at the ARENA's load site (review F4).

    A ladder is the one place old checkpoints are *meant* to be opened: it
    exists to rate iteration 3 against iteration 40, and a plane-map change
    lands in the middle of exactly that range. The file verifies, its digest is
    correct, and the weights saved here are the small net's own -- so
    ``load_state_dict`` would succeed if the guard did not run first, and the
    refusal below is about ordering rather than about a shape that happens not
    to fit.

    Asserted through ``ladder_players`` rather than
    ``arena._checkpoint_evaluator`` so the node covers the seam the campaign
    actually calls.

    **Deleting ``require_matching_plane_count`` from
    :func:`komugi.eval.arena._checkpoint_evaluator` fires
    ``pytest.raises(ValueError)`` here** -- the evaluator is built and the
    stale weights are served under a checkpoint's name.
    """
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    state = KomugiNet(factory.SMALL_CONFIG).state_dict()
    save_checkpoint(
        tmp_path, 30, state, {}, {"step": 30, "in_planes": 118, "plane_map_version": 1}
    )

    (player,) = arena.ladder_players(tmp_path)
    with pytest.raises(ValueError) as raised:
        player.evaluator_factory()

    message = str(raised.value)
    assert "118" in message
    assert str(observation_size()[0]) in message
    assert "KOM-002" in message


# ---------------------------------------------------------------------------
# (d) KOM-022 acceptance #13 -- sharding a match is exact, not approximate
# ---------------------------------------------------------------------------

SHARD_M: int = 2
SHARD_N: int = 4
SHARD_GAMES: int = 8
"""Four seat-swapped pairs at a two-candidate, four-simulation search.

The smallest match that can be cut into four non-trivial shards, at the
smallest budget that still runs the real search: what is under test is the
*coordinates*, and a bigger budget would buy nothing but wall clock.
"""


def _pair_key(record: GameRecord) -> tuple[str, str]:
    """Order games by ``(pair_id, white)`` -- the design's sort for #13.

    A total order on the union that is a function of the game's *coordinates*
    rather than of the order it was played in, which is precisely what a
    shard-order comparison must not depend on.
    """
    return (record.pair_id, record.white)


def _shard_entrants() -> tuple[arena.ArenaPlayer, arena.ArenaPlayer]:
    """Two distinct, deterministic entrants, freshly built.

    Built per call rather than shared, because the claim is about a *second*
    run reproducing the first: a player carrying state from the serial run into
    the sharded one would make the comparison meaningless. ``ParityStub`` is a
    pure function of its input rows, so the two runs differ in nothing but how
    the pairs were divided.
    """
    return (
        arena.ArenaPlayer(
            player=Player(id="challenger", family="komugi"),
            evaluator_factory=lambda: ParityStub(seed=11),
            m=SHARD_M,
            n=SHARD_N,
        ),
        arena.ArenaPlayer(
            player=Player(id="anchor-g0", family="komugi"),
            evaluator_factory=lambda: ParityStub(seed=23),
            m=SHARD_M,
            n=SHARD_N,
        ),
    )


def test_sharded_pairs_equal_serial() -> None:
    """KOM-022 acceptance #13: the union of the shards *is* the serial match.

    ``play_pairs`` over ``range(0, pairs)`` in one process and over four
    disjoint sub-ranges produce byte-identical ``GameRecord`` tuples after
    sorting by ``(pair_id, white)``. This is the property §4's sharding rests
    on and the reason the shards keep the **global** pair index: every game's
    randomness is ``game_seeds(config, pair, seat)``, a pure function of the
    match seed and those global coordinates, so a shard that renumbered its
    pairs from zero would play a *different* match and the union would silently
    be some other 64 games -- with the gauntlet's floor evaluated on them.

    It lives here rather than in ``tests/campaign`` because the claim is about
    the arena's per-game determinism; ``play_pairs`` itself is
    :mod:`komugi.campaign.gauntlet`'s, built on the arena's public
    ``play_arena_game`` rather than as new arena API (the design named it as a
    new entry point in ``eval/arena.py``, which is outside KOM-022's file
    allowlist -- the property is identical either way).

    The ply counts are compared too: they are what
    :func:`~komugi.campaign.gauntlet.summarise` reads, and a record tuple can
    match while the lengths beside it do not.
    """
    from komugi.campaign import gauntlet

    config = arena.ArenaConfig(games=SHARD_GAMES, seed=97, draw_score=DRAW_SCORE)
    pairs = config.pairs
    assert pairs >= 4, f"{SHARD_GAMES} games gives {pairs} pair(s) to shard"

    serial_a, serial_b = _shard_entrants()
    serial_records, serial_plies = gauntlet.play_pairs(
        serial_a, serial_b, config, range(0, pairs)
    )
    assert len(serial_records) == SHARD_GAMES

    ranges = gauntlet.shard_ranges(pairs, 4)
    shards: list[tuple[tuple[GameRecord, ...], tuple[int, ...]]] = []
    for piece in ranges:
        shard_a, shard_b = _shard_entrants()
        shards.append(gauntlet.play_pairs(shard_a, shard_b, config, piece))

    sharded_records = [record for records, _plies in shards for record in records]
    sharded_plies = [ply for _records, plies in shards for ply in plies]

    assert sorted(sharded_records, key=_pair_key) == sorted(
        serial_records, key=_pair_key
    ), (
        "the sharded match is not the serial one: a shard played different "
        "games from the pairs it was given, so the union the floor is "
        "evaluated on is not the match anybody specified"
    )
    assert list(sharded_records) == list(serial_records), (
        "contiguous, index-preserving shards concatenate in play order; a "
        "difference here is a shard that renumbered or reordered its pairs"
    )
    assert sharded_plies == list(serial_plies)

    merged = gauntlet.merge_outcomes(shards)
    assert merged.summary == gauntlet.summarise(serial_records, serial_plies)


# ---------------------------------------------------------------------------
# (e) KOM-014 #21a/#21b -- the paired-book audit is live, and untouched
# ---------------------------------------------------------------------------


def test_the_paired_book_audit_reports_a_mismatched_pair() -> None:
    """#21a, the control: :func:`check_paired_books` can fail at all.

    KOM-014's original acceptance asserted only that the audit returns ``()``
    on an arena match set -- which it does today and would do *however wrong*
    the ticket were, because the arena stamps one ``config.book_id`` on both
    seats of a pair and the check compares those two strings and nothing else.
    That is a tautology, not evidence. This node is the half that can fail: a
    hand-built pair carrying two different ``book_id`` s must produce a
    finding, so #21b's empty result below is a *measurement*.
    """
    mismatched = MatchSet.of(
        (
            GameRecord(
                white="a", black="b", result=WHITE_WIN, book_id="gen0001-000000-aa",
                pair_id="p0",
            ),
            GameRecord(
                white="b", black="a", result=BLACK_WIN, book_id="gen0001-000001-bb",
                pair_id="p0",
            ),
        )
    )
    violations = check_paired_books(mismatched)
    assert violations, (
        "two seat-swapped games from different opening books are not a pair; an "
        "audit that reports nothing here reports nothing anywhere"
    )
    assert any("gen0001-000000-aa" in violation.detail for violation in violations)

    matched = MatchSet.of(
        (
            GameRecord(
                white="a", black="b", result=WHITE_WIN, book_id=arena.EMPTY_BOOK,
                pair_id="p0",
            ),
            GameRecord(
                white="b", black="a", result=DRAW, book_id=arena.EMPTY_BOOK,
                pair_id="p0",
            ),
        )
    )
    assert check_paired_books(matched) == ()


def test_a_draft_book_on_disk_does_not_reach_the_arena(tmp_path: Path) -> None:
    """#21b: KOM-014's book is a self-play device and is not wired here (§5.2).

    A *valid* ``<root>/books/gen_1.json`` exists (it loads, through the real
    :func:`~komugi.selfplay.book.load_book`, so this is not a test about a file
    the arena ignored because it was broken) and a ``SelfPlayConfig`` carries
    its path. The match played beside it still stamps :data:`arena.EMPTY_BOOK`
    on every record, and the paired-book audit -- shown live by #21a -- returns
    nothing. That is exactly the claim §5.2 makes: the promise ``EMPTY_BOOK``'s
    docstring holds out is **not** cashed by this ticket, because a match book
    must be frozen per ladder epoch and a draft book is refreshed every
    generation.
    """
    import json

    from komugi.engine.game import Game as EngineGame
    from komugi.engine.observation import PLANE_MAP_VERSION
    from komugi.selfplay import book as book_module
    from komugi.selfplay.adapter import GameState
    from komugi.selfplay.book import BookEntry, load_book, running_gungi_mode
    from komugi.selfplay.records import num_actions

    rng = np.random.default_rng(4)
    state = GameState(EngineGame(), 0.0)
    prefix: list[int] = []
    for _ply in range(4):
        legal = np.nonzero(state.legal_mask())[0]
        chosen = int(legal[int(rng.integers(0, legal.size))])
        prefix.append(chosen)
        state.play(chosen)
    entry = BookEntry(
        id="gen0001-000000-0badc0de",
        actions=tuple(prefix),
        plies=len(prefix),
        source_iteration=0,
        source_game_id=1,
        deviation_ply=len(prefix) - 1,
        deviated=True,
        support_size=16,
        support_fallback=False,
        truncated_mass=0.0,
        drawn_target_prob=0.5,
        restricted_entropy_bits=1.0,
        tempered_entropy_bits=2.0,
        flattening_ratio=0.75,
    )
    payload = {
        "book_version": book_module.BOOK_VERSION,
        "generation": 1,
        "source_iterations": [0],
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": book_module.TEMPERATURE,
        "support_epsilon": book_module.SUPPORT_EPSILON,
        "support_top_k": book_module.SUPPORT_TOP_K,
        "support_floor": book_module.SUPPORT_FLOOR,
        "support_min_k": book_module.SUPPORT_MIN_K,
        "target_shard_budget": book_module.TARGET_SHARD_BUDGET,
        "cap": book_module.CAP,
        "min_plies": book_module.MIN_PLIES,
        "max_plies": book_module.MAX_PLIES,
        "entries_per_source_game": book_module.ENTRIES_PER_SOURCE_GAME,
        "build_seed": 3,
        "bit_generator": book_module.BIT_GENERATOR,
        "entries": [entry.as_json()],
        "stats": {},
    }
    book_file = tmp_path / "books" / "gen_1.json"
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    loaded = load_book(book_file)
    assert loaded.entries, "the fixture's book must be a real, loadable one"

    selfplay = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        book_path=str(book_file),
        book_fraction=1.0,
    )
    assert selfplay.book_path == str(book_file)

    challenger, anchor = _shard_entrants()
    outcome = arena.play_match(
        challenger,
        anchor,
        arena.ArenaConfig(games=2, seed=31, draw_score=DRAW_SCORE),
    )
    assert len(outcome.records) == 2
    assert {record.book_id for record in outcome.records} == {arena.EMPTY_BOOK}
    assert check_paired_books(outcome.match_set()) == ()


# ---------------------------------------------------------------------------
# (f) kom-011d U1/U2 -- the timing probe (§6 node 12b)
# ---------------------------------------------------------------------------

PROBE_GAMES: int = 2
PROBE_SIMS: int = 8
PROBE_M: int = 4
"""The probe's **pinned CI** settings: one seat-swapped pair at a four-candidate,
eight-simulation search over the campaign's b2c16 test net.

Small on purpose. What runs inside ``ci/check.sh`` is a *smoke over the
instrument* -- the three numbers exist, are finite and are positive -- and its
budget is chosen so the whole node stays well under half a minute on a CPU. The
numbers kom-011d's U1 and U2 are actually answered with come from the same node
run at production settings (``m = 32, n = 256``, many more games) through
:data:`PROBE_GAMES_ENV`, :data:`PROBE_SIMS_ENV` and :data:`PROBE_M_ENV`, on the
campaign box, by the advisor -- and that run is the one whose PROBE line is
quoted on the ticket.
"""

PROBE_GAMES_ENV: str = "KOMUGI_PROBE_GAMES"
PROBE_SIMS_ENV: str = "KOMUGI_PROBE_SIMS"
PROBE_M_ENV: str = "KOMUGI_PROBE_M"
"""Environment overrides, so the production run needs no edit to this file.

An edit would be the wrong mechanism twice over: the CI settings would have to
be restored afterwards (and would eventually not be), and the measurement would
then have been taken with a diff that never landed.

``m`` is overridable as well as ``n`` because **U2 is a question about ``m``**:
the wave is a halving round's leaves, so its size is bounded by the root
candidate count, and §D2.4's bracket runs from ``w = 8`` to the ``m = 32``
ceiling. A probe pinned at ``m = 4`` could only ever report a mean wave below
4 and would answer the question about a search nobody plays. The production
run is therefore ``KOMUGI_PROBE_M=32 KOMUGI_PROBE_SIMS=256``, i.e. the pinned
budget (:data:`komugi.eval.arena.DEFAULT_M` / ``DEFAULT_N``), with
``KOMUGI_PROBE_GAMES`` set to whatever the box has time for.
"""

PROBE_CEILING: int = 1_000_000_000
"""The probe's per-move ceiling, deliberately unreachable.

:class:`~komugi.eval.v1runner.counter.EvalCounter` is §D3.3's *per-move* budget
and refuses the ask past its ceiling; this probe is not measuring that budget,
it is measuring calls and rows over a whole match, so it opens one "move" for
the entire run and gives it a ceiling no match can spend. The ceiling that is
under test lives in ``test_gumbel_mover.py``, where the mover opens and closes a
move per ply as it does in a real bridged game.
"""


def _probe_setting(name: str, pinned: int) -> int:
    """Return the probe's `name` setting: `pinned`, unless the environment raises it."""
    raw = os.environ.get(name)
    if raw is None:
        return pinned
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name}={raw!r} must be a positive integer")
    return value


def test_the_arena_timing_probe_reports_games_per_hour_and_calls_per_move() -> None:
    """kom-011d §6 node 12b: U1's games/hour and U2's wave size, measured.

    The two unknowns this answers are the two the pool's whole compute bound is
    a function of (§D2.4): the arena's games/hour on this box, which **is not**
    the self-play rate (``play_match`` is a single-threaded, single-game loop
    with no worker pool and no shard server), and the mean forward batch ``w``,
    which turns ``257 / w`` forward calls per move into milliseconds. Neither is
    derivable by reading, and rev 1's assertion that ``CountedEvaluator`` could
    already report them was false -- it charged rows and held no call counter,
    which is the additive edit §5's T2a row carries and node 56 pins.

    So this node **measures and prints**; it asserts only that the instrument
    produced three finite, positive numbers. It deliberately does not assert
    their magnitude: a threshold on a wall-clock number would be a test of
    whatever machine CI is running on, and the numbers that get quoted are the
    ones from the production-settings run (see :data:`PROBE_GAMES`), not from
    this one.

    The counted evaluator is shared by both entrants -- one net, two distinct
    ids, the D14 self-match shape -- so ``served`` and ``calls`` cover every
    forward the match made, from either seat.
    """
    games = _probe_setting(PROBE_GAMES_ENV, PROBE_GAMES)
    if games % 2 != 0:
        games += 1  # a match is a whole number of seat-swapped pairs
    sims = _probe_setting(PROBE_SIMS_ENV, PROBE_SIMS)
    candidates = _probe_setting(PROBE_M_ENV, PROBE_M)

    counter = EvalCounter(PROBE_CEILING)
    counted = CountedEvaluator(small_net_evaluator(), counter)
    counter.begin_move()

    def seat(name: str) -> arena.ArenaPlayer:
        """Seat the shared counted evaluator under its own rateable id."""
        return arena.ArenaPlayer(
            player=Player(id=name, family="komugi"),
            evaluator_factory=lambda: counted,
            m=candidates,
            n=sims,
        )

    config = arena.ArenaConfig(games=games, seed=20260908, draw_score=DRAW_SCORE)
    started = time.perf_counter()
    outcome = arena.play_match(seat("probe-a"), seat("probe-b"), config)
    elapsed = time.perf_counter() - started

    moves = outcome.summary.total_plies
    assert moves > 0 and elapsed > 0.0
    assert counted.calls > 0, "the match reached the network through the probe"

    games_per_hour = outcome.summary.games / elapsed * 3600.0
    calls_per_move = counted.calls / moves
    mean_wave = counter.served / counted.calls

    print(
        f"PROBE games_per_hour={games_per_hour:.1f} "
        f"calls_per_move={calls_per_move:.3f} "
        f"mean_wave={mean_wave:.3f} "
        f"games={outcome.summary.games} m={candidates} n={sims} "
        f"plies={moves} mean_plies={outcome.summary.mean_plies:.1f} "
        f"evaluations={counter.served} calls={counted.calls} "
        f"elapsed_s={elapsed:.2f}"
    )

    for label, value in (
        ("games_per_hour", games_per_hour),
        ("calls_per_move", calls_per_move),
        ("mean_wave", mean_wave),
    ):
        assert math.isfinite(value) and value > 0.0, f"{label} is {value!r}"
    assert mean_wave >= 1.0, (
        "a forward covers at least the position it was asked about; a mean wave "
        "below 1 means rows and calls are being counted off different things"
    )
    assert check_paired_books(outcome.match_set()) == ()


# ---------------------------------------------------------------------------
# (g) kom-011d §D3 -- the book seam: prefixes, pairs, and today's games
# ---------------------------------------------------------------------------

BOOK_GAMES: int = 8
"""Four seat-swapped pairs: the smallest match that needs four book entries."""

BOOK_SEED: int = 20260908


def _booked_config(book: PoolBook, games: int = BOOK_GAMES) -> arena.ArenaConfig:
    """Return the match config for `book`, declaring it as the epoch book."""
    return arena.ArenaConfig(
        games=games,
        seed=BOOK_SEED,
        draw_score=DRAW_SCORE,
        book_id=book.epoch_book_id,
        book=book,
    )


def _book_entrants() -> tuple[arena.ArenaPlayer, arena.ArenaPlayer]:
    """Two distinct entrants at the no-simulation budget, freshly built.

    ``n = 0`` is a legal search budget -- one root evaluation a move -- and it
    is the cheapest real game this module can play. What is under test here is
    which *position* a game starts from, not how well it is played.
    """
    return (
        arena.ArenaPlayer(
            player=Player(id="booked-a", family="komugi"),
            evaluator_factory=lambda: ParityStub(seed=11),
            m=1,
            n=0,
        ),
        arena.ArenaPlayer(
            player=Player(id="booked-b", family="komugi"),
            evaluator_factory=lambda: ParityStub(seed=23),
            m=1,
            n=0,
        ),
    )


def test_both_games_of_a_pair_start_from_the_same_prefix_and_swap_seats(
    tmp_path: Path,
) -> None:
    """kom-011d §6 node 12a: what the book seam exists for, played.

    Over a four-pair match on a four-entry epoch book: every ``pair_id`` holds
    two games, the seats are mirrored, both carry the **same** ``book_id`` --
    which is now that pair's *entry* id rather than one constant for the whole
    match -- and the two games' first ``plies`` actions are identical. That is
    §D3.4, and it is the reason ``check_paired_books``'s book clause stops being
    vacuous: before books it compared one constant against itself.

    *Non-vacuity, three ways.* (i) The four pairs' ``book_id`` s are asserted
    **distinct**, so an implementation that stamped one id on the whole match
    (today's behaviour) fails. (ii) The prefixes are asserted to be the book's
    own entries in pair order, so an implementation that played *a* book move
    but not entry ``k`` fails. (iii) A three-entry book at four pairs is
    asserted to be refused at construction -- §D3.2 forbids ``K`` below the pair
    count, because reusing an opening inside one matchup is exactly the
    correlated-games failure the ``K`` derivation exists to prevent, so the
    node reads the design's "4-pair match with a 3-entry book" as the refusal
    it now is rather than as a cycling map the design rules out.
    """
    book = epoch_book(tmp_path, epoch=7, k=4, entries=8, seed=61)
    config = _booked_config(book)
    assert config.pairs == len(book) == 4

    outcome = arena.play_match(*_book_entrants(), config)
    assert check_paired_books(outcome.match_set()) == ()

    by_pair: dict[str, list[GameRecord]] = {}
    for record in outcome.records:
        by_pair.setdefault(record.pair_id, []).append(record)
    assert len(by_pair) == 4
    for pair, (pair_id, records) in enumerate(sorted(by_pair.items())):
        first, second = records
        assert first.seats == (second.black, second.white), pair_id
        assert first.book_id == second.book_id == book.entries[pair].id
    assert len({record.book_id for record in outcome.records}) == 4, (
        "every pair of an epoch is played on its own opening (§D3.2): one id "
        "over the whole match is the bookless behaviour wearing a book's name"
    )

    # The moves, not just the ids: every pair opens on entry k, and pair 0's
    # two *seats* open on the same one -- which is the §D3.4 property, and the
    # one node 13 then takes apart ply by ply.
    for pair in range(config.pairs):
        for seat in (0, 1) if pair == 0 else (0,):
            a, b = _book_entrants()
            white, black = (a, b) if seat == 0 else (b, a)
            game = arena.play_arena_game(white, black, config, pair=pair, seat=seat)
            entry = book.entries[pair]
            assert game.moves[: entry.plies] == entry.actions, (
                f"pair {pair} seat {seat} did not open on its own entry"
            )

    # The short book declares *itself* (review F10 refuses a config that names
    # one book and carries another), so the only thing left to refuse is its
    # size -- which is what this half is about.
    short = epoch_book(tmp_path / "small", epoch=7, k=3, entries=8, seed=61)
    with pytest.raises(ValueError, match="pairs"):
        arena.ArenaConfig(
            games=BOOK_GAMES,
            seed=BOOK_SEED,
            draw_score=DRAW_SCORE,
            book_id=short.epoch_book_id,
            book=short,
        )


def test_the_two_games_of_a_pair_diverge_after_the_prefix(tmp_path: Path) -> None:
    """kom-011d §6 node 13: the prefix is shared, the game is not.

    Rev 1's node here -- *"no two pairs produce identical move sequences"* --
    was vacuous once §D3.2 gave every pair its own opening: it is satisfied by
    the openings alone, including by a deterministic mover. The two
    discriminating predicates are *within* a pair and *across seeds*, and both
    are asserted:

    (a) the two games of one pair share their first ``plies`` actions and
        **differ** at some later ply -- red against a deterministic mover seated
        as the komugi entrant, which would replay one game twice from a shared
        opening and is the mistake §D7.1 exists to prevent;
    (b) the same matchup at a different :attr:`ArenaConfig.seed` produces a
        different move sequence for at least one pair -- red against the
        seed-blind re-link F13 corrects, which would make §D5.4's diagnostic
        replay the original games move for move.
    """
    book = epoch_book(tmp_path, epoch=7, k=2, entries=8, seed=71)
    config = _booked_config(book, games=4)

    def sequences(match_config: arena.ArenaConfig) -> list[tuple[int, ...]]:
        """Return every game's move list, in ``(pair, seat)`` order."""
        moves: list[tuple[int, ...]] = []
        for pair in range(match_config.pairs):
            for seat in (0, 1):
                a, b = _book_entrants()
                white, black = (a, b) if seat == 0 else (b, a)
                moves.append(
                    arena.play_arena_game(
                        white, black, match_config, pair=pair, seat=seat
                    ).moves
                )
        return moves

    played = sequences(config)
    for pair in range(config.pairs):
        plies = book.entries[pair].plies
        first, second = played[2 * pair], played[2 * pair + 1]
        assert first[:plies] == second[:plies] == book.entries[pair].actions
        assert first != second, (
            f"pair {pair}'s two games are the identical move sequence: a pair "
            f"is one opening played from both seats, not one game played twice"
        )

    reseeded = sequences(
        arena.ArenaConfig(
            games=config.games,
            seed=config.seed + 1,
            draw_score=DRAW_SCORE,
            book_id=book.epoch_book_id,
            book=book,
        )
    )
    assert played != reseeded, (
        "a different match seed produced the identical games: the book fixes "
        "the opening and the seed must still fix everything after it"
    )
    for pair in range(config.pairs):
        assert reseeded[2 * pair][: book.entries[pair].plies] == (
            book.entries[pair].actions
        ), "the opening is the book's whatever the seed is"


PLY_SEED_FIXTURES: tuple[int, ...] = (131, 132, 133)
"""Selection seeds for the one-entry books the ply-seed node scans.

Three, because the node's second half needs a position where two *different*
seeds pick two *different* actions, and a fixture where they happen to agree
would otherwise turn a genuine discrimination into a silent pass.
"""


def test_a_booked_games_searched_plies_keep_their_true_ply_seed(
    tmp_path: Path,
) -> None:
    """Review F3: a searched move is seeded from its **absolute** ply.

    ``_play_wired_game`` starts ``ply`` at ``len(moves)`` -- the prefix's length
    -- so a move searched at absolute ply ``p`` carries
    :func:`~komugi.eval.arena.move_seed_for`'s ``p``-th seed whether or not a
    book put the position there. §D3.4 does not mandate a convention, which is
    exactly why a test has to hold this one: under ``ply = 0`` (the reviewer's
    mutant M1) every booked game in the pool changes its whole seed schedule
    while node 12a's prefix equality, node 13's two divergences and nodes 53/60's
    bookless pins all stay green, and the drift would surface months later as an
    unexplained non-reproduction.

    The check is a re-derivation, not an observation: the pair's prefix is
    replayed onto a fresh referee, the side to move is wired to its own stub, and
    the search is run again at ``move_seed_for(config, pair, seat, L)``. If that
    reproduces the game's first searched action, the game was seeded from ``L``.

    *Non-vacuity:* the same search at ``move_seed_for(..., 0)`` -- the seed
    ``ply = 0`` would have used -- is asserted to pick a **different** action on
    at least one fixture, so the node cannot pass by way of a position with one
    reasonable move or a stub the seed does not reach.
    """
    discriminated = False
    for fixture in PLY_SEED_FIXTURES:
        book = epoch_book(tmp_path / f"s{fixture}", epoch=7, k=1, entries=8, seed=fixture)
        config = _booked_config(book, games=2)
        entry = book.entries[0]
        offset = entry.plies

        game = arena.play_arena_game(*_book_entrants(), config, pair=0, seat=0)
        assert game.moves[:offset] == entry.actions
        assert len(game.moves) > offset, "the prefix is not the whole game"
        searched = game.moves[offset]

        state = GameState(Game(), DRAW_SCORE)
        arena.replay_prefix(state, entry.actions)
        first, second = _book_entrants()
        mover = first if state.to_play() == 0 else second
        evaluator = mover.evaluator_factory()
        scalar = make_evaluator(evaluator, DRAW_SCORE)
        wave = arena.make_batch_wave_evaluator(evaluator, DRAW_SCORE)

        def action_seeded_at(ply: int) -> int:
            """Return the action this position's own search picks at `ply`'s seed."""
            search = arena.move_search(
                scalar,
                wave,
                m=mover.m,
                n=mover.n,
                gumbel_scale=mover.gumbel_scale,
                seed=arena.move_seed_for(config, 0, 0, ply),
            )
            return int(search.search(state).action)

        assert arena.move_seed_for(config, 0, 0, offset) != arena.move_seed_for(
            config, 0, 0, 0
        ), "the two seeds under comparison must differ, or the node sees nothing"
        assert action_seeded_at(offset) == searched, (
            f"the first searched move of a {offset}-ply booked game is not the "
            f"one ply {offset}'s seed picks: the prefix is not being counted "
            f"into the ply index, so every booked game is played on a different "
            f"Gumbel schedule from the one the module's determinism section "
            f"describes"
        )
        if action_seeded_at(0) != searched:
            discriminated = True
            break

    assert discriminated, (
        f"no scanned fixture {PLY_SEED_FIXTURES} separated the ply-offset seed "
        f"from ply 0's: the assertion above cannot be evidence about the offset "
        f"if both seeds pick the same action -- widen the scan"
    )


VISITS_SIMS: int = 1600
"""§D8.2's ``#n1600`` rung -- a visits variant, four times the rated budget."""


def test_a_visits_variant_breaches_the_rated_ceiling_and_voids() -> None:
    """kom-011d §6 node 18: the mechanical proof of §D8.2's first reason.

    A ``#n1600`` entrant seated on the **rated** path produces an M8 match void
    rather than a score: the 257-evaluation ceiling is per move and per side
    (KOM-023 §D3.3), the search asks past it inside its first move, the counted
    evaluator refuses the batch whole and
    :func:`~komugi.eval.v1bridge.play_bridged_game` resolves the breach as a
    match void with nothing scored. That is why the visits legs are separate
    entrants played off the rated path, rather than a knob on a rated row.

    *Non-vacuity:* the same match with both entrants at the rated budget is
    asserted to complete and score, so the node is red against a harness that
    voids everything, and the discarded-pairs half is asserted too -- a
    match-void keeps nothing (§D6 precedence 4).
    """
    config = arena.ArenaConfig(
        games=2,
        seed=5,
        draw_score=0.0,
        book_id=bridged_book_id(resolved_gungi_mode()),
    )

    def entrant(player_id: str, sims: int) -> BridgeEntrant:
        """Seat the small net at `sims` simulations under the rated ceiling."""
        return komugi_entrant(
            player_id,
            small_net_evaluator,
            draw_score=0.0,
            m=2,
            n=sims,
        )

    breached = play_bridged_match(
        entrant("komugi/ckpt-3000@aaaa0001#n1600", VISITS_SIMS),
        entrant("komugi/ckpt-3000@aaaa0001", 2),
        config,
        rated=True,
    )
    assert breached.status == VOIDED
    assert breached.outcome is None, "a match void keeps nothing (§D6)"
    void = breached.tally.voids[-1]
    assert void.mode == M8
    assert void.bucket is VoidBucket.BUDGET_BREACH
    assert void.scope is VoidScope.MATCH
    assert void.invariant == "P2"

    scored = play_bridged_match(
        entrant("komugi/ckpt-3000@aaaa0001", 2),
        entrant("komugi/ckpt-6000@aaaa0002", 2),
        config,
        rated=True,
    )
    assert scored.status != VOIDED, (
        f"the control match voided too ({scored.reason}): the ceiling breach "
        f"above cannot be evidence about the visits budget if a rated match at "
        f"the rated budget voids as well"
    )
    assert scored.outcome is not None and len(scored.outcome.records) == 2


# KOM-165: re-pinned after the plane-115/120 rescale (cap 512, plane map 3) changed the net's inputs and hence the bookless moves; the move-for-move reference comparison above the pin still passes.
NO_BOOK_DIGEST: str = "287af94abf2b65ab"
"""The bookless match, hashed: :func:`match_digest` over the fixture below.

A stored literal rather than a recomputed value, so that any later change to
this match -- the loop, the seeds, the ids, the engine, numpy -- has to be
noticed and re-pinned by a human rather than absorbed silently.

**What pins it to the *pre*-T2b behaviour is not this constant**, which was
recorded at T2b's own gate; it is :func:`_reference_game` below, which is the
game loop as it stood before the book seam existed, transcribed into this file
and run beside ``play_match``. The digest is the regression pin; the reference
loop is the equivalence proof. (A third, independent one already exists and is
untouched by T2b: ``test_an_arena_game_is_the_self_play_game`` pins this loop
against :class:`~komugi.selfplay.worker.SelfPlayWorker`'s.)
"""


def _reference_game(
    white: arena.ArenaPlayer,
    black: arena.ArenaPlayer,
    config: arena.ArenaConfig,
    *,
    pair: int,
    seat: int,
) -> tuple[float, tuple[int, ...]]:
    """Play one game the way the arena played it **before** the book seam.

    A transcription of ``_play_wired_game`` as it stood at T2b's base commit:
    a bare :class:`~komugi.engine.game.Game`, dispatch on ``state.to_play()``,
    a fresh search per move at the mover's own budget on
    :func:`~komugi.eval.arena.move_seed_for`'s per-ply seed, and the engine
    owning termination. It has no concept of a prefix, so it cannot be broken by
    a prefix bug -- which is what makes it the reference and not a second copy
    of the thing under test.

    The result is read from the **engine**, by replaying the move list through a
    fresh game and reading ``win_state()``, rather than from anything in the
    arena.
    """
    sides: list[
        tuple[arena.ArenaPlayer, Evaluator, WaveEvaluator]
    ] = []
    for entrant in (white, black):
        evaluator = entrant.evaluator_factory()
        sides.append(
            (
                entrant,
                make_evaluator(evaluator, config.draw_score),
                arena.make_batch_wave_evaluator(evaluator, config.draw_score),
            )
        )

    state = GameState(Game(), config.draw_score)
    moves: list[int] = []
    ply = 0
    while not state.is_terminal():
        entrant, scalar, wave = sides[0] if state.to_play() == 0 else sides[1]
        search = arena.move_search(
            scalar,
            wave,
            m=entrant.m,
            n=entrant.n,
            gumbel_scale=entrant.gumbel_scale,
            seed=arena.move_seed_for(config, pair, seat, ply),
        )
        action = int(search.search(state).action)
        state.play(action)
        moves.append(action)
        ply += 1

    replay = Game()
    for action in moves:
        replay.play_action(action)
    win = np.asarray(replay.win_state(), dtype=np.uint8)
    result = DRAW if bool(win[2]) else (WHITE_WIN if bool(win[0]) else BLACK_WIN)
    return result, tuple(moves)


def match_digest(
    records: Sequence[GameRecord], moves: Sequence[Sequence[int]]
) -> str:
    """Return a short digest of a match's records **and** its move sequences.

    Both halves, for ``test_the_same_match_replays_exactly``'s reason: equal
    records alone would be satisfied by two matches that played completely
    different games to the same three-valued results.
    """
    canonical = "|".join(
        f"{record.white}>{record.black}={record.result}"
        f"@{record.book_id}#{record.pair_id}"
        for record in records
    )
    canonical += "||" + ";".join(
        ",".join(str(int(action)) for action in sequence) for sequence in moves
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def test_a_match_with_no_book_is_identical_to_todays() -> None:
    """kom-011d §6 node 53: the bookless path did not move.

    Node 8 pinned the ``pair_id`` s a default ``pair_namespace`` writes; this is
    the other half of T2's byte-for-byte claim -- the **games**. With no
    :class:`~komugi.eval.pool_books.PoolBook` supplied, ``play_match`` produces
    the records and the move sequences the pre-T2b loop produces at the same
    seed: every game is compared, move for move and result for result, against
    :func:`_reference_game`, which is that loop transcribed; and the whole
    fixture is pinned against :data:`NO_BOOK_DIGEST` so a later drift has to be
    re-pinned rather than absorbed.

    Red against every plausible way the seam could leak into a bookless match:
    a prefix loop that starts ``ply`` at 1 rather than 0 (the reference's seeds
    then disagree from the first move), a ``book_id`` that stopped being
    :data:`arena.EMPTY_BOOK`, a namespace applied unconditionally, or a replay
    helper that mutates the referee before the loop.
    """
    config = arena.ArenaConfig(games=2, seed=97, draw_score=DRAW_SCORE)
    assert config.book is None and config.book_id == arena.EMPTY_BOOK
    assert config.pair_namespace == ""

    outcome = arena.play_match(*_shard_entrants(), config)
    moves: list[tuple[int, ...]] = []
    for pair in range(config.pairs):
        for seat in (0, 1):
            challenger, anchor = _shard_entrants()
            white, black = (
                (challenger, anchor) if seat == 0 else (anchor, challenger)
            )
            played = arena.play_arena_game(
                white, black, config, pair=pair, seat=seat
            )

            reference_white, reference_black = _shard_entrants()
            if seat == 1:
                reference_white, reference_black = reference_black, reference_white
            reference_result, reference_moves = _reference_game(
                reference_white, reference_black, config, pair=pair, seat=seat
            )
            assert played.moves == reference_moves, (
                f"pair {pair} seat {seat} no longer plays the pre-T2b game: the "
                f"bookless path has moved"
            )
            assert played.result == reference_result
            assert outcome.records[2 * pair + seat].result == reference_result, (
                f"the match's record for pair {pair} seat {seat} disagrees with "
                f"the pre-T2b loop"
            )
            moves.append(played.moves)

    assert [record.pair_id for record in outcome.records] == [
        "challenger:anchor-g0:00000",
        "challenger:anchor-g0:00000",
    ]
    assert {record.book_id for record in outcome.records} == {arena.EMPTY_BOOK}
    assert [record.seats for record in outcome.records] == [
        ("challenger", "anchor-g0"),
        ("anchor-g0", "challenger"),
    ]
    assert check_paired_books(outcome.match_set()) == ()

    digest = match_digest(outcome.records, moves)
    assert digest == NO_BOOK_DIGEST, (
        f"the bookless match is no longer the one this fixture was captured "
        f"from: digest {digest!r}, expected {NO_BOOK_DIGEST!r}. Either the book "
        f"seam has reached a match that supplied no book, or the engine, the "
        f"search or the stub has changed underneath the pin"
    )
