"""Arena structure: pairing, seeds, determinism, and the records that come out.

Stub evaluators only -- no :mod:`komugi.net`, no checkpoints, no device. What is
under test here is the *harness*: that a match is a whole number of seat-swapped
pairs, that every ply gets its own Gumbel seed, that two runs of one config are
the same match down to the move, and that what comes out is data the rating
stack (:mod:`komugi.eval.seats`, :mod:`komugi.eval.elo`) accepts without
translation.

The games are real -- real :class:`~komugi.engine.game.Game`, real
:class:`~komugi.search.gumbel.GumbelSearch`, real termination -- because the
seat dispatch and the per-move seeding are exactly what a faked game loop would
fake away. Two things keep the bill down, and neither of them fakes anything:

* the reference match is played **once**, by a module-scoped fixture, and
  asserted on by several tests -- a game runs to the engine's 256-turn cap, so
  the suite should pay for each one at most once; and
* tests that only care about the *shape* of a match (pair ids, the audit) use
  :func:`cheap_entrant`, whose ``n = 0`` budget plays the search's own
  no-simulation path: one root evaluation per ply, real everything else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.cyboard import board_to_fen
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule
from komugi.eval import arena
from komugi.eval.elo import fit_elo
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    VALID_RESULTS,
    WHITE_WIN,
    MatchSet,
    Player,
)
from komugi.eval.pool_books import (
    POOL_BOOK_VERSION,
    PoolBook,
    PoolBookEntry,
    entry_id_for,
    epoch_book_id_for,
)
from komugi.eval.seats import check_paired_books
from komugi.search.gumbel import GumbelConfig, GumbelSearch
from komugi.search.state import Evaluator, WaveEvaluator
from komugi.selfplay.adapter import GameState, make_evaluator
from komugi.selfplay.book import running_gungi_mode
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import num_actions

DRAW_SCORE: float = 0.20
"""``DESIGN.html`` section 3.3's contempt, and ``SelfPlayConfig``'s default."""

TINY_M: int = 2
TINY_N: int = 4
"""The smallest budget whose first halving round carries **two** leaves.

``m=2, n=4`` runs rounds of 2 and 1 simulations: the first is one wave of two
positions, which is what makes the wave seam observable at all. At ``n=2`` every
round would carry a single leaf and a batched match would be indistinguishable
from a scalar one.
"""


class StubEvaluator:
    """A deterministic :class:`~komugi.selfplay.inference.BatchEvaluator`.

    Deterministic **in ``obs`` alone**: a fixed prior plus a bounded,
    position-dependent tilt, computed row by row. No call counter in the answer,
    no clock, no global RNG -- which is what makes the determinism test below a
    statement about the arena's seeding rather than about this class. Row-wise
    also means a wave of R positions answers exactly what R scalar calls would,
    so the two search schedules cannot be told apart by this stub.

    ``wdl`` is ``(0.5, 0, 0.5)``: a utility of exactly zero under
    :func:`~komugi.selfplay.inference.wdl_utility` for any ``draw_score``, so
    these games are driven by the prior and the Gumbel draw and nothing else.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._base = rng.standard_normal(ACTION_SIZE).astype(np.float32)
        self._tilt = (rng.standard_normal(ACTION_SIZE) * 0.25).astype(np.float32)
        self.rows = 0
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a ``(N, planes, 9, 9)`` batch; the mask is the search's job."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        rows = int(array.shape[0])
        self.calls += 1
        self.rows += rows
        keys = array.reshape(rows, -1).astype(np.float64).sum(axis=1)
        mix = (np.remainder(keys, 8.0) / 8.0).astype(np.float32)
        policy = (self._base[None, :] + mix[:, None] * self._tilt[None, :]).astype(
            np.float32
        )
        wdl = np.tile(np.asarray([0.5, 0.0, 0.5], dtype=np.float32), (rows, 1))
        return NetEval(
            policy_logits=policy,
            wdl=wdl,
            moves_left=np.full((rows,), 8.0, dtype=np.float32),
        )


def entrant(name: str, seed: int) -> arena.ArenaPlayer:
    """One arena entrant over its own stub, at the suite's tiny budget."""
    return arena.ArenaPlayer(
        player=Player(id=name, family="komugi"),
        evaluator_factory=lambda: StubEvaluator(seed),
        m=TINY_M,
        n=TINY_N,
    )


def cheap_entrant(name: str, seed: int) -> arena.ArenaPlayer:
    """An entrant that spends **no** simulations: one root evaluation a move.

    ``n = 0`` is a legal search budget (it yields a prior-shaped target), and it
    is the cheapest real game this suite can play. Used where the assertion is
    about the *structure* of a match rather than about the search inside it.
    """
    return arena.ArenaPlayer(
        player=Player(id=name, family="komugi"),
        evaluator_factory=lambda: StubEvaluator(seed),
        m=1,
        n=0,
    )


CONFIG: arena.ArenaConfig = arena.ArenaConfig(games=2, seed=11, draw_score=DRAW_SCORE)
"""One pair at the tiny budget: the smallest match that shows a seat swap."""

SHAPE_CONFIG: arena.ArenaConfig = arena.ArenaConfig(
    games=4, seed=5, draw_score=DRAW_SCORE
)
"""Two pairs, played by :func:`cheap_entrant` s -- for the structural tests."""


@pytest.fixture(scope="module")
def baseline() -> arena.MatchOutcome:
    """The module's reference match: ``alpha`` vs ``beta``, one pair, seed 11."""
    return arena.play_match(entrant("alpha", 1), entrant("beta", 2), CONFIG)


@pytest.fixture(scope="module")
def shaped() -> arena.MatchOutcome:
    """A two-pair match at the no-simulation budget, for structural assertions."""
    return arena.play_match(
        cheap_entrant("gamma", 3), cheap_entrant("delta", 4), SHAPE_CONFIG
    )


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("games", [1, 3, 101])
def test_odd_game_counts_are_refused(games: int) -> None:
    """An odd match cannot be seat-balanced, so it is refused at construction."""
    with pytest.raises(ValueError, match="even"):
        arena.ArenaConfig(games=games, seed=0, draw_score=DRAW_SCORE)


def test_negative_and_malformed_configs_are_refused() -> None:
    """The other three ways an ArenaConfig could not describe a match."""
    with pytest.raises(ValueError, match="games must be >= 0"):
        arena.ArenaConfig(games=-2, seed=0, draw_score=DRAW_SCORE)
    with pytest.raises(ValueError, match="draw_score"):
        arena.ArenaConfig(games=2, seed=0, draw_score=1.5)
    with pytest.raises(ValueError, match="book_id"):
        arena.ArenaConfig(games=2, seed=0, draw_score=DRAW_SCORE, book_id="  ")


def test_draw_score_has_no_default() -> None:
    """Contempt is a decision the caller has to make (design review D3)."""
    with pytest.raises(TypeError):
        arena.ArenaConfig(games=2, seed=0)  # type: ignore[call-arg]


def test_a_player_cannot_be_matched_against_itself() -> None:
    """Refused before a single game is played, not by the first record (D14)."""
    with pytest.raises(ValueError, match="distinct ids"):
        arena.play_match(entrant("solo", 1), entrant("solo", 2), CONFIG)


def test_default_budget_is_the_production_full_search() -> None:
    """The default entrant plays what self-play plays, by derivation."""
    player = arena.ArenaPlayer(
        player=Player(id="default", family="komugi"),
        evaluator_factory=lambda: StubEvaluator(0),
    )
    assert (player.m, player.n) == (32, 256)
    assert (
        player.gumbel_scale
        == GumbelConfig(num_actions=1, m=1, n=0, seed=0).gumbel_scale
    )


# ---------------------------------------------------------------------------
# pair structure
# ---------------------------------------------------------------------------


def test_a_pair_is_two_games_with_the_seats_swapped(
    baseline: arena.MatchOutcome,
) -> None:
    """Game 2k is a-as-white, game 2k+1 is b-as-white, under one pair_id."""
    first, second = baseline.records
    assert first.seats == ("alpha", "beta")
    assert second.seats == ("beta", "alpha")
    assert first.pair_id == second.pair_id == "alpha:beta:00000"
    assert first.book_id == second.book_id == arena.EMPTY_BOOK
    assert {record.result for record in baseline.records} <= VALID_RESULTS


def test_pair_ids_are_indexed_and_zero_padded(shaped: arena.MatchOutcome) -> None:
    """Every pair of a longer match gets its own id, in synth.py's format."""
    assert SHAPE_CONFIG.pairs == 2
    assert [record.pair_id for record in shaped.records] == [
        "gamma:delta:00000",
        "gamma:delta:00000",
        "gamma:delta:00001",
        "gamma:delta:00001",
    ]
    # The id names the entrants in the order they were passed, in both seats.
    assert [record.seats for record in shaped.records] == [
        ("gamma", "delta"),
        ("delta", "gamma"),
        ("gamma", "delta"),
        ("delta", "gamma"),
    ]


def test_the_default_pair_namespace_reproduces_todays_pair_ids(
    shaped: arena.MatchOutcome,
) -> None:
    """kom-011d §6 node 8: ``pair_namespace=""`` is today's ids, byte for byte.

    Three halves, and the first is the one the node exists for. The **default**
    match's ids are the exact ``f"{a}:{b}:{k:05d}"`` strings every stored record
    and every other test in this file already carries -- no prefix, no
    separator, nothing. That is what a namespaced arena has to keep true, and it
    is red against the obvious wrong implementation: an unconditional
    ``f"{ns}/{a}:{b}:{k:05d}"`` prefix, which writes ``"/gamma:delta:00000"``
    here and would silently re-key every record the campaign has already stored.

    The second half is the feature: a non-empty namespace *does* prefix, so the
    same two entrants can be played twice -- a pool epoch's re-link (§D2.3) --
    without the two matches colliding on ``pair_id`` and failing
    :func:`~komugi.eval.seats.check_paired_books` with GAME_COUNT violations.
    The third is the refusal: a namespace carrying ``":"`` would make an id that
    cannot be split back into its parts, and it is refused at construction
    rather than discovered by a forensic tool a year later.
    """
    assert SHAPE_CONFIG.pair_namespace == "", "the default is the empty prefix"
    assert [record.pair_id for record in shaped.records] == [
        "gamma:delta:00000",
        "gamma:delta:00000",
        "gamma:delta:00001",
        "gamma:delta:00001",
    ]
    assert arena.pair_id_for(SHAPE_CONFIG, "gamma", "delta", 1) == "gamma:delta:00001"

    namespaced = arena.ArenaConfig(
        games=2, seed=5, draw_score=DRAW_SCORE, pair_namespace="E007"
    )
    outcome = arena.play_match(
        cheap_entrant("gamma", 3), cheap_entrant("delta", 4), namespaced
    )
    assert [record.pair_id for record in outcome.records] == [
        "E007/gamma:delta:00000",
        "E007/gamma:delta:00000",
    ]
    # The namespace changes the id and nothing else: same seed, same games.
    assert [record.seats for record in outcome.records] == [
        ("gamma", "delta"),
        ("delta", "gamma"),
    ]
    assert check_paired_books(outcome.match_set()) == ()

    with pytest.raises(ValueError, match="pair_namespace"):
        arena.ArenaConfig(
            games=2, seed=0, draw_score=DRAW_SCORE, pair_namespace="E007:1"
        )
    with pytest.raises(ValueError, match="/"):
        arena.ArenaConfig(games=2, seed=1, draw_score=0.0, pair_namespace="E007/")


def test_summary_counts_what_was_played(baseline: arena.MatchOutcome) -> None:
    """The summary is an accounting of the same games, not a second opinion."""
    summary = baseline.summary
    assert summary.games == len(baseline.records) == 2
    assert summary.decisive + summary.draws == summary.games
    assert summary.draws == sum(1 for record in baseline.records if record.is_draw)
    assert 0 < summary.shortest_plies <= summary.longest_plies
    assert summary.total_plies >= summary.longest_plies
    assert summary.mean_plies == summary.total_plies / summary.games
    assert summary.draw_rate == summary.draws / summary.games


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def test_every_ply_gets_its_own_gumbel_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """One seed per MOVE, not per game (design F10d).

    A game played on one seed draws the identical Gumbel vector at every ply,
    which is a systematic exploration bias rather than the extra noise it looks
    like. The recording subclass is the only way to see the seed the search was
    actually built with -- the seed never reaches the evaluator, so no stub
    could observe it.
    """
    seen: list[int] = []

    class RecordingSearch(GumbelSearch):
        """The real search, with the config's seed noted on the way in."""

        def __init__(
            self,
            evaluator: Evaluator,
            config: GumbelConfig,
            *,
            wave_evaluator: WaveEvaluator | None = None,
        ) -> None:
            seen.append(config.seed)
            super().__init__(evaluator, config, wave_evaluator=wave_evaluator)

    monkeypatch.setattr(arena, "GumbelSearch", RecordingSearch)
    outcome = arena.play_arena_game(
        cheap_entrant("gamma", 3), cheap_entrant("delta", 4), SHAPE_CONFIG
    )

    assert len(seen) == len(outcome.moves) > 1
    assert seen[0] != seen[1]
    assert len(set(seen)) == len(seen)
    assert seen == [
        arena.move_seed_for(SHAPE_CONFIG, 0, 0, ply) for ply in range(len(seen))
    ]


def test_seed_streams_are_disjoint_across_pairs_and_seats() -> None:
    """``(pair, seat, ply)`` names a stream, and no two of them collide."""
    seeds = [
        arena.move_seed_for(CONFIG, pair, seat, ply)
        for pair in range(3)
        for seat in (0, 1)
        for ply in range(8)
    ]
    assert len(set(seeds)) == len(seeds)
    # Pure: asking twice for one ply's seed answers the same both times.
    assert arena.move_seed_for(CONFIG, 1, 1, 5) == arena.move_seed_for(CONFIG, 1, 1, 5)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_same_match_replays_exactly(baseline: arena.MatchOutcome) -> None:
    """Same entrants, same config -> the same records and the same moves.

    Both halves matter. Equal records alone would be satisfied by two matches
    that played completely different games to the same two results -- with three
    possible results per game that is no coincidence at all -- so the *move
    sequence* is compared too, via the single-game door, which is also what ties
    that door to the match loop: it must replay the game the match played.
    """
    replay = arena.play_match(entrant("alpha", 1), entrant("beta", 2), CONFIG)
    assert replay.records == baseline.records
    assert replay.summary == baseline.summary

    first = arena.play_arena_game(entrant("alpha", 1), entrant("beta", 2), CONFIG)
    again = arena.play_arena_game(entrant("alpha", 1), entrant("beta", 2), CONFIG)
    assert len(first.moves) > 1
    assert first.moves == again.moves
    assert first.result == baseline.records[0].result
    assert len(first.moves) in {
        baseline.summary.shortest_plies,
        baseline.summary.longest_plies,
    }


# ---------------------------------------------------------------------------
# the wave seam
# ---------------------------------------------------------------------------


def test_a_wave_answers_exactly_what_scalar_calls_would() -> None:
    """R positions, one call, R answers -- identical to R scalar evaluations."""
    stub = StubEvaluator(9)
    wave = arena.make_batch_wave_evaluator(stub, DRAW_SCORE)
    scalar = make_evaluator(stub, DRAW_SCORE)

    root = GameState(Game(), DRAW_SCORE)
    other = root.clone()
    other.play(int(np.flatnonzero(other.legal_mask())[0]))

    answers = wave([root, other])
    assert len(answers) == 2
    for state, (policy, value) in zip([root, other], answers, strict=True):
        expected_policy, expected_value = scalar(state)
        assert np.array_equal(policy, expected_policy)
        assert value == expected_value

    with pytest.raises(ValueError, match="at least one state"):
        wave([])


def test_a_halving_round_reaches_the_evaluator_as_one_call() -> None:
    """The point of the adapter (D7): fewer calls than positions evaluated.

    At ``m=2, n=4`` a ply is one root evaluation plus a two-leaf wave plus two
    single-leaf waves -- five positions in four calls. A scalar-only arena would
    show one call per position.
    """
    stub = StubEvaluator(7)
    player = arena.ArenaPlayer(
        player=Player(id="waver", family="komugi"),
        evaluator_factory=lambda: stub,
        m=TINY_M,
        n=TINY_N,
    )
    arena.play_arena_game(player, entrant("beta", 2), CONFIG, pair=4, seat=0)

    assert stub.calls > 0
    assert stub.rows > stub.calls


# ---------------------------------------------------------------------------
# what comes out is what the rating stack eats
# ---------------------------------------------------------------------------


def test_two_matches_pass_the_paired_book_audit(
    baseline: arena.MatchOutcome, shaped: arena.MatchOutcome
) -> None:
    """Matches from different pairings union into an audit-clean match set.

    The pair id carries both entrant ids, so two matches cannot collide on one
    -- which is what lets a whole ladder's games be concatenated and audited in
    a single pass (design review D2).
    """
    union = MatchSet.of(baseline.records + shaped.records)

    assert check_paired_books(union) == ()
    assert len(union.pair_ids()) == 3
    assert union.players() == ("alpha", "beta", "delta", "gamma")
    assert baseline.match_set().records == baseline.records


def test_records_feed_the_elo_fit(baseline: arena.MatchOutcome) -> None:
    """The records go into ``fit_elo`` as they are, with no translation."""
    result = fit_elo(baseline.match_set(), anchor="alpha")
    assert set(result.ratings) == {"alpha", "beta"}
    assert result.ratings["alpha"] == 0.0
    assert all(np.isfinite(rating) for rating in result.ratings.values())


def test_an_empty_match_is_a_legal_no_op() -> None:
    """Zero games is even, plays nothing, and summarises to zeroes."""
    outcome = arena.play_match(
        entrant("alpha", 1),
        entrant("beta", 2),
        arena.ArenaConfig(games=0, seed=1, draw_score=DRAW_SCORE),
    )
    assert outcome.records == ()
    assert outcome.summary == arena.ArenaSummary(0, 0, 0, 0, 0, 0)
    assert outcome.summary.mean_plies == 0.0
    assert outcome.summary.draw_rate == 0.0


# ---------------------------------------------------------------------------
# the three properties the 2026-08-29 impl review found undefended
# (review R1/R3-R5: mutations M7, M2/M3 and M5 survived the original suite)
# ---------------------------------------------------------------------------


def test_results_are_stated_from_whites_perspective() -> None:
    """``GameOutcome.result`` matches the engine's own absolute adjudication.

    The ground truth is computed here, independently of the arena: replay the
    game's move list through a fresh engine and read ``win_state()`` directly
    -- ``[white, black, draw]`` in the engine's absolute frame. A result stated
    from any other perspective (review finding R1: the mover's, which at a
    terminal position is almost always the *loser's*) disagrees with this
    replay on every decisive game, so the corruption that would silently
    invert an entire Elo ladder fails here instead.

    Scans a handful of seeds so that both decisive branches are exercised --
    the scan is deterministic, and the closing assertion makes the coverage
    claim loud: if a stub change ever turns every scanned game into a draw or
    a one-sided sweep, this test fails rather than silently thinning.
    """
    observed: set[float] = set()
    for seed in range(10):
        config = arena.ArenaConfig(games=2, seed=seed, draw_score=DRAW_SCORE)
        outcome = arena.play_arena_game(entrant("east", 1), entrant("west", 2), config)

        replay = Game()
        for action in outcome.moves:
            replay.play_action(action)
        state = np.asarray(replay.win_state(), dtype=np.uint8)
        white_won, black_won, drawn = (bool(x) for x in state)
        assert white_won or black_won or drawn, "replayed game is not terminal"
        truth = DRAW if drawn else (WHITE_WIN if white_won else BLACK_WIN)

        assert outcome.result == truth, (
            f"seed {seed}: arena reported {outcome.result} but the engine "
            f"adjudicated {truth} (win_state {state.tolist()}): the result is "
            f"not stated from white's perspective"
        )
        observed.add(truth)
    assert {WHITE_WIN, BLACK_WIN} <= observed, (
        f"the seed scan produced {sorted(observed)}: without at least one win "
        f"for each colour, the perspective claim above is untested on one of "
        f"its decisive branches -- widen the scan"
    )


def test_the_mover_searches_with_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ply's search carries the budget of the player who holds the move.

    The two entrants get deliberately different ``(m, n, gumbel_scale)`` --
    the mixed-budget match is an advertised feature -- and a recording search
    captures what each constructed ``GumbelConfig`` actually carried. The
    mover per ply is derived independently by replaying the move list through
    a fresh engine and asking it who was to play. Handing the mover its
    opponent's budget (review finding R4, survived mutation M3) misassigns
    every asymmetric ply.

    The closing assertion pins the alternation invariant this suite currently
    relies on: in this engine, with equal armies drafted from the standard
    start, ``to_play == ply % 2`` on every reachable ply (probed over 200
    random legal games; the ``cyengine`` turn-flip has an asymmetric branch,
    but equal draft counts keep it unreachable from ``Game()``). If an engine
    change ever makes that assertion fail, dispatch-by-parity has become
    distinguishable from dispatch-by-``to_play()`` -- strengthen this test to
    cover the asymmetric plies rather than deleting the assertion.
    """
    seen: list[tuple[int, int, float]] = []

    class BudgetRecorder(GumbelSearch):
        """The real search, with each config's budget noted on the way in."""

        def __init__(
            self,
            evaluator: Evaluator,
            config: GumbelConfig,
            *,
            wave_evaluator: WaveEvaluator | None = None,
        ) -> None:
            seen.append((config.m, config.n, config.gumbel_scale))
            super().__init__(evaluator, config, wave_evaluator=wave_evaluator)

    monkeypatch.setattr(arena, "GumbelSearch", BudgetRecorder)
    white = arena.ArenaPlayer(
        player=Player(id="wide", family="komugi"),
        evaluator_factory=lambda: StubEvaluator(1),
        m=2,
        n=4,
        gumbel_scale=1.0,
    )
    black = arena.ArenaPlayer(
        player=Player(id="narrow", family="komugi"),
        evaluator_factory=lambda: StubEvaluator(2),
        m=3,
        n=2,
        gumbel_scale=0.5,
    )
    outcome = arena.play_arena_game(white, black, CONFIG)

    movers: list[int] = []
    replay = Game()
    for action in outcome.moves:
        movers.append(int(replay.player))
        replay.play_action(action)

    assert len(seen) == len(movers) > 1
    budgets = {0: (2, 4, 1.0), 1: (3, 2, 0.5)}
    mismatched = [
        ply
        for ply, (got, mover) in enumerate(zip(seen, movers, strict=True))
        if got != budgets[mover]
    ]
    assert not mismatched, (
        f"plies {mismatched[:5]} searched with the wrong player's budget: "
        f"the mover's own (m, n, gumbel_scale) must be used"
    )
    assert movers == [ply % 2 for ply in range(len(movers))], (
        "to_play stopped alternating with ply parity: the engine now reaches "
        "asymmetric-draft plies, so this test must be strengthened to cover "
        "dispatch on them (see docstring) rather than weakened"
    )


class ContemptStub:
    """An evaluator whose draw mass swings with the position.

    Half the positions it sees (by an arbitrary, deterministic key of the
    observation) are called near-certain draws, the other half sharp; the win
    and loss mass are split evenly, so under
    :func:`~komugi.selfplay.inference.wdl_utility` the *only* thing separating
    the two kinds of position is ``draw_score`` -- at ``+1`` the drawish ones
    are worth ``+0.9``, at ``-1`` they are worth ``-0.9``, and at ``0`` the
    stub is indifferent. A search over this stub therefore steers toward or
    away from the drawish half of the tree as contempt flips, which is what
    lets a test observe ``draw_score`` *reaching the games* rather than just
    sitting validated in the config.
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a batch, one independent row at a time."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        rows = int(array.shape[0])
        keys = array.reshape(rows, -1).astype(np.float64).sum(axis=1)
        drawish = np.remainder(keys, 2.0) < 1.0
        draw_mass = np.where(drawish, 0.9, 0.0)
        win = (1.0 - draw_mass) * 0.5
        wdl = np.stack([win, draw_mass, 1.0 - draw_mass - win], axis=1).astype(
            np.float32
        )
        return NetEval(
            policy_logits=np.zeros((rows, ACTION_SIZE), dtype=np.float32),
            wdl=wdl,
            moves_left=np.zeros((rows,), dtype=np.float32),
        )


def test_draw_score_governs_the_games_played() -> None:
    """The configured contempt changes which moves get played.

    Design review D3 made ``draw_score`` a required field because it is the
    contempt the games are *played under*; the original suite enforced only
    the signature (review finding R5, survived mutation M5: hardwiring ``0.0``
    into the wiring passed every test). Here the same seeded game is played at
    contempt ``-1`` and ``+1`` over a draw-mass-carrying stub: the move
    sequences must differ, which can only happen if the config's value reaches
    :func:`~komugi.selfplay.inference.wdl_utility` through the evaluator
    seams. A bounded seed scan keeps the assertion robust to stub drift; if no
    scanned seed separates the two contempts, either the wiring is broken or
    the stub has stopped carrying draw mass -- both worth failing on.
    """

    def contempt_entrant(name: str) -> arena.ArenaPlayer:
        return arena.ArenaPlayer(
            player=Player(id=name, family="komugi"),
            evaluator_factory=ContemptStub,
            m=TINY_M,
            n=TINY_N,
        )

    for seed in range(8):
        moves: dict[float, tuple[int, ...]] = {}
        for draw_score in (-1.0, 1.0):
            config = arena.ArenaConfig(games=2, seed=seed, draw_score=draw_score)
            moves[draw_score] = arena.play_arena_game(
                contempt_entrant("hopeful"), contempt_entrant("fearful"), config
            ).moves
        if moves[-1.0] != moves[1.0]:
            return
    pytest.fail(
        "contempt -1.0 and +1.0 produced identical games at every scanned "
        "seed: draw_score is not reaching the play (or the stub above has "
        "stopped carrying draw mass)"
    )


# ---------------------------------------------------------------------------
# kom-011d §6 node 14 -- a prefix that has no game left in it
# ---------------------------------------------------------------------------


BOOK_EPOCH: int = 7
"""The epoch every hand-built book in this file is minted for."""

BOOK_SOURCE: str = "tests/eval/test_arena_unit.py"
"""What the hand-built books name as their source -- this file, honestly."""


def _book_of(actions: tuple[int, ...]) -> PoolBook:
    """Return a one-entry epoch book carrying `actions` as its prefix.

    Hand-built rather than selected from a source book, because the point is
    the *type*'s refusal: :class:`~komugi.eval.pool_books.PoolBook` verifies
    every prefix by replay at construction, so there is no path into it -- a
    build, a load, or a fixture like this one -- that produces a book the arena
    cannot play.

    **Every id is derived, not written down** (review F2). The entry's id is
    :func:`~komugi.eval.pool_books.entry_id_for` over its index and its actions
    and the book's is
    :func:`~komugi.eval.pool_books.epoch_book_id_for` over the versioned fields
    below -- the same two derivations ``build_epoch_book`` uses -- because the
    type now refuses an id that is not its own contents' hash. The literal
    ``"E007-0badc0de"`` this fixture used to carry over an arbitrary prefix was
    precisely the artefact F2 named: a book whose id said nothing about the
    openings inside it.
    """
    entries = (
        PoolBookEntry(
            id=entry_id_for(BOOK_EPOCH, 0, actions),
            actions=actions,
            plies=len(actions),
            source_entry_id="gen0001-000000-0badc0de",
        ),
    )
    # Mirrors ``PoolBook.versioned_fields`` because the id has to exist before
    # the instance does; the constructor then checks the mirror by re-deriving.
    fields: dict[str, Any] = {
        "pool_book_version": POOL_BOOK_VERSION,
        "epoch": BOOK_EPOCH,
        "source_book": BOOK_SOURCE,
        "source_book_sha256": "0" * 64,
        "gungi_mode": running_gungi_mode(),
        "action_size": ACTION_SIZE,
        "engine_rules_version": 1,
        "selection_seed": 1,
        "k": 1,
        "entries": [entry.as_json() for entry in entries],
    }
    return PoolBook(
        epoch=BOOK_EPOCH,
        source_book=BOOK_SOURCE,
        source_book_sha256="0" * 64,
        gungi_mode=running_gungi_mode(),
        action_size=ACTION_SIZE,
        engine_rules_version=1,
        selection_seed=1,
        k=1,
        entries=entries,
        epoch_book_id=epoch_book_id_for(fields),
    )


def test_a_prefix_that_ends_in_a_terminal_position_is_refused_before_the_match() -> None:
    """kom-011d §6 node 14: an unplayable opening is a construction error.

    A prefix that leaves a terminal position has no game in it. The refusal is
    at **construction** -- of the book, before any config and before any match
    -- and the counted factory below proves it: zero evaluators were built, so
    zero games were played.

    Red against an implementation that discovers the problem at the loop's first
    ``is_terminal()`` check, which would score an empty game: a
    :class:`~komugi.eval.records.GameRecord` with a result and no moves is
    indistinguishable, in a stored ``MatchSet``, from a game somebody played.

    *Non-vacuity:* the same construction with the *first four plies* of the same
    game -- a legal, non-terminal prefix -- builds a book and plays a match, and
    the factory count is then non-zero. Without that half the node would pass
    against an implementation that refused every book.
    """
    finished = arena.play_arena_game(
        cheap_entrant("gamma", 3), cheap_entrant("delta", 4), SHAPE_CONFIG
    )
    assert len(finished.moves) > 4, "the fixture game must be a real, finished one"

    built: list[str] = []

    def counted(name: str) -> arena.ArenaPlayer:
        """An entrant whose evaluator factory records that it was called."""

        def build() -> StubEvaluator:
            built.append(name)
            return StubEvaluator(1)

        return arena.ArenaPlayer(
            player=Player(id=name, family="komugi"), evaluator_factory=build, m=1, n=0
        )

    with pytest.raises(ValueError, match="terminal"):
        terminal = _book_of(finished.moves)
        arena.play_match(
            counted("alpha"),
            counted("beta"),
            arena.ArenaConfig(
                games=2,
                seed=1,
                draw_score=DRAW_SCORE,
                book_id=terminal.epoch_book_id,
                book=terminal,
            ),
        )
    assert built == [], (
        f"{len(built)} evaluator(s) were built before the prefix was refused: "
        f"the refusal must happen before a game is paid for, not at the first "
        f"is_terminal() check inside one"
    )

    playable = _book_of(finished.moves[:4])
    outcome = arena.play_match(
        counted("alpha"),
        counted("beta"),
        arena.ArenaConfig(
            games=2,
            seed=1,
            draw_score=DRAW_SCORE,
            book_id=playable.epoch_book_id,
            book=playable,
        ),
    )
    assert built == ["alpha", "beta"]
    assert {record.book_id for record in outcome.records} == {playable.entries[0].id}
    assert check_paired_books(outcome.match_set()) == ()


# ---------------------------------------------------------------------------
# the T2b review's findings F6, F8 and F10
# ---------------------------------------------------------------------------


def _legal_prefix(plies: int) -> tuple[int, ...]:
    """Return `plies` legal actions from a fresh game: the lowest id each ply.

    Deterministic and engine-derived, so a prefix built here is legal by the
    same authority the arena replays it under.
    """
    state = GameState(Game(), DRAW_SCORE)
    actions: list[int] = []
    for _ply in range(plies):
        action = int(np.flatnonzero(state.legal_mask())[0])
        actions.append(action)
        state.play(action)
    return tuple(actions)


def _entry_carrying(actions: tuple[int, ...]) -> PoolBookEntry:
    """Return a hand-built entry over `actions`, verified by **nobody**.

    :class:`~komugi.eval.pool_books.PoolBookEntry` checks its id, its plies and
    its provenance and does *not* replay: only :class:`PoolBook` does that. So
    this is the one construction in the package that can hold a prefix
    ``verify_prefix`` would refuse, which is exactly what
    :func:`~komugi.eval.arena.replay_prefix`'s own guards have to be tested
    against -- a book that had been verified could not distinguish a guard that
    is there from one that is not.
    """
    return PoolBookEntry(
        id=entry_id_for(BOOK_EPOCH, 0, actions),
        actions=actions,
        plies=len(actions),
        source_entry_id="gen0001-000000-0badc0de",
    )


def test_a_prefix_action_outside_the_encoding_is_refused() -> None:
    """Review F6: ``replay_prefix`` checks the range, not just the mask.

    The arena's copy of the prefix check used to index ``state.legal_mask()``
    directly. A **negative** action is then wrapped by numpy's negative
    indexing and can read ``True`` off the far end of the mask, at which point
    ``state.play(-1)`` is asked to play an action nobody chose; an out-of-range
    positive raises ``IndexError`` rather than the ``ValueError`` the function
    documents. Both are asked for here, over an entry hand-built past
    ``verify_prefix`` -- because every prefix that reaches the arena through a
    :class:`~komugi.eval.pool_books.PoolBook` has already been verified, which
    is why the guard was unobserved in the first place.

    *Non-vacuity, two halves.* (i) The same helper replays the **legal** prefix
    without complaint and returns it, so the node is red against a
    ``replay_prefix`` that refused everything. (ii) The in-range **illegal**
    action is refused too -- and that half is the one the reviewer's mutant M10
    (delete the legality guard) survives without: no fixture in the suite plants
    an illegal prefix, because ``PoolBook`` cannot hold one.
    """
    prefix = _legal_prefix(2)
    replayed = arena.replay_prefix(GameState(Game(), DRAW_SCORE), prefix)
    assert replayed == prefix, "the control prefix must replay"

    for outside in (-1, num_actions()):
        entry = _entry_carrying((*prefix, outside))
        with pytest.raises(ValueError, match="outside"):
            arena.replay_prefix(GameState(Game(), DRAW_SCORE), entry.actions)

    # An action that is *in range* and illegal in the position it is reached
    # from: the guard mutant M10 deletes, which nothing else in the suite sees.
    reached = GameState(Game(), DRAW_SCORE)
    arena.replay_prefix(reached, prefix)
    mask = np.asarray(reached.legal_mask(), dtype=np.bool_)
    illegal = int(np.flatnonzero(~mask)[0])
    entry = _entry_carrying((*prefix, illegal))
    with pytest.raises(ValueError, match="illegal"):
        arena.replay_prefix(GameState(Game(), DRAW_SCORE), entry.actions)

    # And the same line is refused one layer up, at the type: this entry is
    # only constructible because PoolBookEntry does not replay.
    with pytest.raises(ValueError, match="illegal"):
        _book_of((*prefix, illegal))


def _game_length(
    entrants: tuple[arena.ArenaPlayer, arena.ArenaPlayer],
    config: arena.ArenaConfig,
    seat: int,
) -> int:
    """Return how many moves pair 0's `seat` game runs to, as the match plays it.

    The seats are swapped for ``seat == 1`` exactly as
    :func:`~komugi.eval.arena.play_match` swaps them, so this replays the
    match's own game rather than a differently-seated one at the same seed.
    """
    white, black = entrants if seat == 0 else (entrants[1], entrants[0])
    return len(arena.play_arena_game(white, black, config, pair=0, seat=seat).moves)


def test_the_summary_counts_searched_plies_not_the_book_prefix() -> None:
    """Review F8: a replayed opening is not a game a search was run for.

    ``GameOutcome.moves`` carries the prefix on purpose -- a caller replaying
    the list has to play it -- but :class:`~komugi.eval.arena.ArenaSummary` is
    read as cost: §D2.4 prices a game over its **searched** plies and T2a's
    probe divides measured forward calls by this counter to report calls per
    move. Counting an ``L``-ply opening as played reads both low by ``L`` on
    every booked game and sets the epoch cadence too long.

    *Non-vacuity:* the bookless control asserts the counter still equals the
    full move count when there is no prefix, so the node is red against an
    implementation that subtracted a constant, and the booked half compares
    against the games' **own** replayed lengths rather than against a number
    typed here.
    """
    prefix = _legal_prefix(6)
    book = _book_of(prefix)
    entrants = (cheap_entrant("alpha", 1), cheap_entrant("beta", 2))
    config = arena.ArenaConfig(
        games=2,
        seed=29,
        draw_score=DRAW_SCORE,
        book_id=book.epoch_book_id,
        book=book,
    )
    outcome = arena.play_match(*entrants, config)
    played = [_game_length(entrants, config, seat) for seat in (0, 1)]
    searched = [length - len(prefix) for length in played]

    assert min(played) > len(prefix), "the prefix must not be the whole game"
    assert outcome.summary.total_plies == sum(searched), (
        f"the summary counted {outcome.summary.total_plies} plies over games of "
        f"{played} moves opening on a {len(prefix)}-ply book: the replayed "
        f"prefix is being priced as though a search had been run for it"
    )
    assert outcome.summary.shortest_plies == min(searched)
    assert outcome.summary.longest_plies == max(searched)
    assert outcome.summary.mean_plies == sum(searched) / 2

    bookless = arena.ArenaConfig(games=2, seed=29, draw_score=DRAW_SCORE)
    control = arena.play_match(*entrants, bookless)
    control_played = [_game_length(entrants, bookless, seat) for seat in (0, 1)]
    assert control.summary.total_plies == sum(control_played), (
        "a bookless match has no prefix to exclude, so its counter is the full "
        "move count -- unchanged from before the book seam"
    )


def test_a_config_carrying_a_book_must_declare_it() -> None:
    """Review F10: the in-process twin of the bridged declaration check.

    ``ArenaConfig`` used to accept a :class:`~komugi.eval.pool_books.PoolBook`
    beside the default :data:`~komugi.eval.arena.EMPTY_BOOK` declaration, and
    only ``ladder.arena_config`` and the fixtures set the two consistently. A
    T4 in-family pool match built by hand would then store records whose
    per-game ``book_id`` s are entry ids while the config declared the empty
    board -- the manifest's ``epoch_book_id`` and the records disagreeing, with
    nothing refusing.

    *Non-vacuity:* the declared config constructs and its pair map still reads
    the entry id, and a **bookless** config still defaults to ``EMPTY_BOOK``
    without complaint -- so the node is red against a refusal that fired on
    every config, which would have taken the whole bookless path with it.
    """
    book = _book_of(_legal_prefix(4))

    with pytest.raises(ValueError, match="must declare it"):
        arena.ArenaConfig(games=2, seed=1, draw_score=DRAW_SCORE, book=book)
    with pytest.raises(ValueError, match="must declare it"):
        arena.ArenaConfig(
            games=2,
            seed=1,
            draw_score=DRAW_SCORE,
            book_id=book.entries[0].id,
            book=book,
        )

    declared = arena.ArenaConfig(
        games=2,
        seed=1,
        draw_score=DRAW_SCORE,
        book_id=book.epoch_book_id,
        book=book,
    )
    assert declared.book_id_for_pair(0) == book.entries[0].id
    assert arena.ArenaConfig(games=2, seed=1, draw_score=DRAW_SCORE).book_id == (
        arena.EMPTY_BOOK
    )


# ---------------------------------------------------------------------------
# KOM-141 -- the repetition rule the arena's referee adjudicates under
# ---------------------------------------------------------------------------


def test_new_arena_game_follows_the_configs_repetition_rule() -> None:
    """The seam both arena loops build their referee at.

    A scripted line that exhibits the cells-only aliasing cannot be played
    *through* the arena -- ``new_arena_game`` starts from the engine's own start
    position and the rules-audit scenarios start from position P -- so what is
    pinned here is the threading itself, at the one function
    :func:`komugi.eval.arena._play_wired_game` and
    :func:`komugi.eval.v1bridge.play_bridged_game` both call. The rule's
    *behaviour* is pinned where the scenarios live,
    ``tests/rules/test_repetition_rule_placement.py``.

    *Non-vacuity:* both values are asserted, so the node is red against a
    ``new_arena_game`` that hardcoded either one.
    """
    default = arena.ArenaConfig(games=2, seed=0, draw_score=DRAW_SCORE)
    assert default.repetition_rule is RepetitionRule.FULL, (
        "nothing flips silently: FULL is Komugi's repaired key and the default"
    )
    assert arena.new_arena_game(default).repetition_rule is RepetitionRule.FULL

    parity = arena.ArenaConfig(
        games=2,
        seed=0,
        draw_score=DRAW_SCORE,
        repetition_rule=RepetitionRule.PLACEMENT,
    )
    assert arena.new_arena_game(parity).repetition_rule is (
        RepetitionRule.PLACEMENT
    )


def test_new_arena_game_is_otherwise_the_engines_own_start_position() -> None:
    """The rule is the ONLY thing the config changes about the referee.

    A game that also picked up a cap, a contempt or a different start position
    from the config would make every arena measurement a measurement of this
    function instead of the players.
    """
    config = arena.ArenaConfig(games=2, seed=0, draw_score=DRAW_SCORE)
    wired = arena.new_arena_game(config)
    plain = Game()

    assert board_to_fen(wired._board) == board_to_fen(plain._board)
    assert wired.max_turns() == plain.max_turns()
    assert wired.repetition_count() == plain.repetition_count() == 1
