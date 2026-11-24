"""The perspective oracle: whose frame is each recorded row written in?

Design D4 gives the worker one job that no shape check can police -- turning
the engine's ABSOLUTE ``win_state()`` (``[white, black, draw]``, ``game.py``)
into a ``wdl_target`` in the perspective of *the player who moved on that row*.
Getting that wrong is the classic sign bug: the run trains happily, the loss
falls, and every second position teaches the network the opposite of the truth.
It is invisible to every test that looks at one ply, and invisible to every
test whose game happened to end on the same parity it started on.

So this module attacks it from both ends:

* **Through the worker.**  Real games, real engine, tiny Gumbel budgets.  A
  game's rows must all agree on ONE absolute outcome, each flipped into its own
  row's mover.  An implementation that skips the flip agrees with itself on
  row 0 and contradicts itself on the first row played by the other side, so
  the assertion is made explicitly at both parities (even ``ply`` and odd) and
  keyed off the recorded ``to_play`` rather than off ``ply % 2`` -- the KOM-122
  spec is emphatic that a drafting position need not alternate, so parity is
  not the mover and this suite never pretends it is.
* **Through the adapter directly.**  ``GameState(game, draw_score)`` driven
  down a scripted legal line to a *decisive* terminal (both terminal movers,
  found by seed scan) and to the engine's *adjudicated turn-cap draw* (design
  D6: the engine owns termination, the worker adds no second cap).  There the
  absolute outcome is something this module holds independently -- it is
  ``game.win_state()`` -- so ``absolute_outcome()`` and the side-to-move framing
  of ``terminal_value()`` are checked against an oracle rather than against
  themselves.

The rest of the row contract rides along on the same games, because they are
the expensive part: ``mlh_target``'s 1/cap scale (design D4/Q3), ``ply``,
``search_mode`` (design D6's seeded dual-mode draw), ``policy_kl`` (design
D4/F2, including the deliberately constructed zero), ``root_value`` and
``sims_used``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
import torch

import worker_rig as rig

from komugi.engine.action_encoding import C_START
from komugi.engine.game import Game

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "SelfPlayWorker"):
    pytest.skip(
        "komugi.selfplay exports no SelfPlayWorker yet (KOM-022)",
        allow_module_level=True,
    )

Row = dict[str, torch.Tensor]

_RUN_SEED = 20260822
_WORKER_ID = 1
_MAX_GAMES = 6
"""How many worker games to play looking for a decisive one.  Early-exit: the
loop stops as soon as one appears, so the common cost is one game."""

_SEED_SCAN = 14
"""How many scripted playout seeds to try for the two decisive parities."""


# ---------------------------------------------------------------------------
# fixtures -- the expensive part, built once per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def worker_games(tmp_path_factory: pytest.TempPathFactory) -> dict[int, list[Row]]:
    """Rows of real worker games, grouped by ``game_id``.

    Games are played one at a time and the loop stops at the first decisive
    result, because a decisive game is what the sign bug needs to be visible
    and a drawn one is ``[0, 1, 0]`` in every frame.  Everything played is
    returned, so the drawn games that turn up on the way are asserted on too.
    """
    directory = Path(tmp_path_factory.mktemp("worker_perspective"))
    stub = rig.StubBatchEvaluator(seed=11)
    worker, writer = rig.build_worker(
        stub,
        directory,
        worker_id=_WORKER_ID,
        run_seed=_RUN_SEED,
        games=_MAX_GAMES,
    )
    games: dict[int, list[Row]] = {}
    try:
        for index in range(_MAX_GAMES):
            plies = worker.play_game(index)
            assert int(plies) >= 1, (
                f"play_game({index}) reported {plies!r}; one row per move means "
                "a played game is at least one row"
            )
            writer.flush()
            games = rig.by_game(rig.read_rows(directory))
            if any(_is_decisive(rows) for rows in games.values()):
                break
    finally:
        writer.close()
    assert games, "the worker wrote no rows at all"
    return games


@pytest.fixture(scope="module")
def decisive_game(worker_games: dict[int, list[Row]]) -> list[Row]:
    """The longest decisive worker game -- the one the sign bug shows up in."""
    decisive = [rows for rows in worker_games.values() if _is_decisive(rows)]
    if not decisive:
        pytest.fail(
            f"{_MAX_GAMES} worker games all ended drawn, so the absolute->mover "
            "flip is untested here: a draw is [0, 1, 0] in every frame. Raise "
            "_MAX_GAMES, or bias the stub toward a decisive line; do not relax "
            "the assertion, since this parity check is the whole point of D4."
        )
    return max(decisive, key=len)


@pytest.fixture(scope="module")
def scripted_decisive() -> dict[int, Any]:
    """One decisive scripted game per terminal mover, keyed by that mover.

    Terminal mover, not winner: after a marshal capture or a mate the side *to
    move* is the loser, so this gives ``terminal_value()`` its both-parities
    test -- one game where white is the mated side and one where black is.
    """
    found: dict[int, Any] = {}
    for seed in range(_SEED_SCAN):
        game, _ = rig.random_playout(seed, prefer_terminal=True)
        outcome = np.asarray(game.win_state(), dtype=np.int64)
        if int(outcome[0]) == 0 and int(outcome[1]) == 0:
            continue
        found.setdefault(int(game.player), game)
        if len(found) == 2:
            break
    if len(found) < 2:
        pytest.fail(
            f"{_SEED_SCAN} terminal-seeking playouts produced decisive games for "
            f"movers {sorted(found)} only; both parities are needed to pin the "
            "side-to-move framing of terminal_value()"
        )
    return found


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_decisive(rows: list[Row]) -> bool:
    """Whether a game's rows imply a win rather than a draw."""
    return int(rig.absolute_from_row(rows[0])[2]) == 0


def _ordered(rows: list[Row]) -> list[Row]:
    """Rows of one game, in ply order (they are written in it; this pins it)."""
    plies = [rig.as_int(row, "ply") for row in rows]
    assert plies == sorted(plies), f"rows are not in ply order: {plies}"
    return rows


# ---------------------------------------------------------------------------
# the oracle: one absolute outcome, flipped per row
# ---------------------------------------------------------------------------


def test_every_row_of_every_game_matches_one_absolute_outcome(
    worker_games: dict[int, list[Row]],
) -> None:
    """A game has ONE result, and each row states it in its own mover's frame."""
    for game_id, rows in sorted(worker_games.items()):
        absolute = rig.absolute_from_row(rows[0])
        for row in _ordered(rows):
            mover = rig.as_int(row, "to_play")
            want = rig.mover_target(absolute, mover)
            got = rig.as_array(row, "wdl_target")
            assert tuple(got.tolist()) == pytest.approx(want), (
                f"game {game_id} ply {rig.as_int(row, 'ply')} (mover {mover}): "
                f"wdl_target {got.tolist()} is not {list(want)}, the absolute "
                f"outcome {absolute} flipped into that row's mover (design D4)"
            )


def test_wdl_target_is_a_pure_one_hot_outcome_never_a_contempt_value(
    worker_games: dict[int, list[Row]],
) -> None:
    """Design D1/D4: contempt enters values, never targets.

    ``draw_score`` is 0.20 and the recorded target must not know that.  A row
    carrying ``[0, 0.2, 0]`` or a smoothed outcome is the contamination this
    pins.
    """
    for rows in worker_games.values():
        for row in rows:
            wdl = rig.as_array(row, "wdl_target")
            assert wdl.shape == (3,)
            assert sorted(wdl.tolist()) == pytest.approx([0.0, 0.0, 1.0]), (
                f"wdl_target {wdl.tolist()} is not a one-hot outcome"
            )


def test_the_decisive_game_flips_at_both_parities(decisive_game: list[Row]) -> None:
    """The sign bug, stated as the assertion that catches it.

    Winner-mover rows read ``[1, 0, 0]`` and loser-mover rows read ``[0, 0, 1]``
    *in the same game*, and both an even-``ply`` row and an odd-``ply`` row are
    required to be present so a game that happened to end on its starting
    parity cannot let a missing flip through.
    """
    rows = _ordered(decisive_game)
    absolute = rig.absolute_from_row(rows[0])
    winner = 0 if int(absolute[0]) == 1 else 1

    movers = {rig.as_int(row, "to_play") for row in rows}
    assert movers == {0, 1}, (
        f"the decisive game recorded movers {sorted(movers)}; both sides must "
        "appear for the flip to be observable"
    )
    parities = {rig.as_int(row, "ply") % 2 for row in rows}
    assert parities == {0, 1}, f"only ply parities {sorted(parities)} present"

    for parity in (0, 1):
        at_parity = [row for row in rows if rig.as_int(row, "ply") % 2 == parity]
        assert at_parity, f"no rows at ply parity {parity}"
        for row in at_parity:
            mover = rig.as_int(row, "to_play")
            want = rig.WIN if mover == winner else rig.LOSS
            got = rig.as_array(row, "wdl_target")
            assert tuple(got.tolist()) == pytest.approx(want), (
                f"ply {rig.as_int(row, 'ply')} (parity {parity}, mover {mover}, "
                f"winner {winner}): wdl_target {got.tolist()} != {list(want)}"
            )

    # And the two frames genuinely differ, which is the property a flip-less
    # implementation cannot have: it would write one constant row everywhere.
    distinct = {tuple(rig.as_array(row, "wdl_target").tolist()) for row in rows}
    assert len(distinct) == 2, (
        f"a decisive game produced {len(distinct)} distinct wdl_targets "
        f"({sorted(distinct)}); without the absolute->mover flip every row is "
        "identical, which is exactly the bug this test exists for"
    )


# ---------------------------------------------------------------------------
# moves-left, ply, to_play, search_mode
# ---------------------------------------------------------------------------


def test_mlh_scale_is_one_over_the_engines_turn_cap() -> None:
    """Design D4/Q3 pins the scale HERE, in the records module.

    Against the engine's leaf (KOM-165), not against a literal: the rig reads
    the same ``observation.MAX_TURNS``, so this asserts the derivation rather
    than re-stating a number that would rot at the next cap change.
    """
    assert sp.MLH_SCALE == pytest.approx(1.0 / rig.MAX_TURNS)
    assert sp.MLH_SCALE * rig.MAX_TURNS == pytest.approx(1.0)


def test_mlh_target_descends_one_scale_per_ply_to_one_scale(
    worker_games: dict[int, list[Row]],
) -> None:
    """Plies-until-the-end, scaled by MLH_SCALE; the FINAL row is 1 ply, not 0.

    Adjudicated convention (wave-9b review F3): the KOM-122 spec's "to 0 at
    the final row" wording contradicted the KOM-022 spec's pinned formula
    `mlh_target = (total_plies - ply) * MLH_SCALE`, and the formula wins --
    a label of 0 would claim the RECORDED (pre-move) position is terminal,
    and no recorded position ever is: the engine adjudicates only after the
    row's move is played. So the final row is exactly one scale step from
    the end, and every earlier row is one step above its successor. The
    trainer's MSE is convention-agnostic but must see ONE convention forever.
    """
    scale = float(sp.MLH_SCALE)
    for game_id, rows in sorted(worker_games.items()):
        ordered = _ordered(rows)
        total = len(ordered)
        assert rig.as_float(ordered[-1], "mlh_target") == pytest.approx(
            scale, abs=1e-7
        ), f"game {game_id}: the final row is ONE ply from adjudication, not 0"
        for index, row in enumerate(ordered):
            want = (total - index) * scale
            assert rig.as_float(row, "mlh_target") == pytest.approx(want, abs=1e-6), (
                f"game {game_id} row {index}/{total - 1}: mlh_target "
                f"{rig.as_float(row, 'mlh_target')} != {want} "
                f"({total - index} plies to adjudication x {scale})"
            )
        if len(ordered) > 1:
            steps = {
                round(
                    rig.as_float(ordered[i], "mlh_target")
                    - rig.as_float(ordered[i + 1], "mlh_target"),
                    9,
                )
                for i in range(len(ordered) - 1)
            }
            assert steps == {round(scale, 9)}, (
                f"game {game_id}: mlh_target steps {sorted(steps)} are not a "
                f"constant {scale} per ply"
            )


def test_ply_is_dense_and_starts_at_the_engines_zero(
    worker_games: dict[int, list[Row]],
) -> None:
    """One row per move, numbered by the counter the turn cap is enforced on."""
    for game_id, rows in sorted(worker_games.items()):
        plies = [rig.as_int(row, "ply") for row in _ordered(rows)]
        assert plies[0] == 0, (
            f"game {game_id} starts at ply {plies[0]}; ``Game._turns`` -- the "
            "counter the turn-cap draw is adjudicated against -- starts at 0"
        )
        assert plies == list(range(len(plies))), (
            f"game {game_id} has gaps or repeats in ply: {plies}"
        )
        assert len(plies) <= rig.MAX_TURNS, (
            f"game {game_id} recorded {len(plies)} rows, past the engine's cap "
            f"of {rig.MAX_TURNS} -- the worker must add no second cap (D6/F15)"
        )


def test_to_play_is_the_engines_and_is_never_assumed_from_parity(
    worker_games: dict[int, list[Row]],
) -> None:
    """``to_play`` is a recorded fact about the position, in ``{0, 1}``.

    Deliberately NOT asserted to equal ``ply % 2``: the KOM-122 spec is explicit
    that a drafting position need not alternate, so this test pins the range and
    the presence of both sides, and every perspective assertion above keys off
    this field rather than off parity.
    """
    for game_id, rows in sorted(worker_games.items()):
        movers = {rig.as_int(row, "to_play") for row in rows}
        assert movers <= {0, 1}, f"game {game_id} recorded to_play {sorted(movers)}"
        if len(rows) > 1:
            assert movers == {0, 1}, (
                f"game {game_id} recorded only mover {sorted(movers)} across "
                f"{len(rows)} rows"
            )


def test_search_mode_is_zero_or_one_and_both_modes_get_drawn(
    worker_games: dict[int, list[Row]],
) -> None:
    """Design D6/F1: every move records which mode played it.

    Both values are required only once there are enough rows for the seeded
    ``prob_fast`` draw to be observable -- at the 0.75 default, 40 rows make an
    all-one-mode run a one-in-a-hundred-thousand event, not a flake.
    """
    modes = [rig.as_int(row, "search_mode") for rows in worker_games.values() for row in rows]
    assert set(modes) <= {0, 1}, f"search_mode took values {sorted(set(modes))}"
    if len(modes) >= 40:
        assert set(modes) == {0, 1}, (
            f"{len(modes)} rows recorded only search_mode {sorted(set(modes))}; "
            "self-play is dual-mode (DESIGN section 3.1, design D6) and the mode "
            "is drawn per move from prob_fast"
        )


def test_sims_used_never_exceeds_the_budget_it_was_given(
    worker_games: dict[int, list[Row]],
) -> None:
    """The recorded budget is the one the mode asked for (design D6)."""
    config = rig.tiny_selfplay_config()
    for rows in worker_games.values():
        for row in rows:
            sims = rig.as_int(row, "sims_used")
            mode = rig.as_int(row, "search_mode")
            budget = config.gumbel_n_full if mode == 0 else config.gumbel_n_fast
            assert 0 <= sims <= budget, (
                f"sims_used {sims} is outside [0, {budget}] for search_mode {mode}"
            )


# ---------------------------------------------------------------------------
# policy_kl and root_value
# ---------------------------------------------------------------------------


def test_policy_kl_is_finite_non_negative_and_hits_zero_by_construction(
    worker_games: dict[int, list[Row]],
) -> None:
    """Design D4/F2's surprise weight, and the move built to make it vanish.

    The stub's utility is exactly 0 for every position (``wdl = (0.5, 0, 0.5)``
    collapses to ``w - l + d*draw_score == 0`` for any ``draw_score``), so the
    completed-Q vector the Gumbel target is built from is constant and
    ``softmax(logits + sigma(q))`` is the masked softmax of the prior itself.
    KL of a distribution against itself is 0, so at least one row -- in
    practice nearly all of them -- must record exactly that.  Rows whose search
    reached a terminal leaf legitimately differ, which is why the requirement
    is "some row", not "every row".
    """
    values = [
        rig.as_float(row, "policy_kl")
        for rows in worker_games.values()
        for row in rows
    ]
    assert values, "no rows to check"
    assert all(np.isfinite(value) for value in values), (
        f"policy_kl went non-finite: {[v for v in values if not np.isfinite(v)]}"
    )
    assert min(values) >= -1e-6, (
        f"policy_kl is a KL divergence and cannot be negative, got {min(values)}"
    )
    assert min(values) <= 1e-6, (
        "no row recorded policy_kl ~ 0 even though the stub's constant utility "
        "makes the Gumbel target equal to the masked-softmax prior; either the "
        "KL is not computed against the ROOT PRIOR (design D4/F2) or the two "
        "distributions are not the ones the row holds"
    )


def test_root_value_is_a_finite_diagnostic_in_range(
    worker_games: dict[int, list[Row]],
) -> None:
    """Design D4/F14: diagnostics only, but still a value in ``[-1, 1]``."""
    for rows in worker_games.values():
        for row in rows:
            value = rig.as_float(row, "root_value")
            assert np.isfinite(value), "root_value went non-finite"
            assert -1.0 - 1e-5 <= value <= 1.0 + 1e-5, (
                f"root_value {value} is outside [-1, 1]"
            )


def test_policy_target_is_a_distribution_over_the_action_space(
    worker_games: dict[int, list[Row]],
) -> None:
    """``SearchResult.policy_target`` verbatim (design D4): sums to 1, no mass
    below zero, and never all on one action by accident of shape."""
    for rows in worker_games.values():
        for row in rows:
            target = rig.as_array(row, "policy_target")
            assert target.shape == (rig.NUM_ACTIONS,)
            assert float(target.sum()) == pytest.approx(1.0, abs=1e-4)
            assert float(target.min()) >= 0.0


def test_observation_rows_match_the_schema_and_are_finite(
    worker_games: dict[int, list[Row]],
) -> None:
    """The recorded observation is the encoder's, at the encoder's shape."""
    spec = {field.name: field for field in sp.SCHEMA.fields}["observation"]
    assert tuple(spec.shape) == (rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W)
    for rows in worker_games.values():
        row = rows[0]
        observation = row["observation"].numpy()
        assert observation.shape == (rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W)
        assert np.all(np.isfinite(observation))


# ---------------------------------------------------------------------------
# the adapter, driven down a scripted line: absolute vs mover-framed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_mover", [0, 1])
def test_terminal_value_is_minus_one_for_the_side_that_was_mated(
    scripted_decisive: dict[int, Any], terminal_mover: int
) -> None:
    """Both parities of the side-to-move framing, against an independent oracle.

    The absolute outcome here is not inferred from the adapter: it is
    ``game.win_state()``, read off the engine this module drove itself.  The
    side to move at a decisive terminal is the loser, so its ``terminal_value``
    is ``-1`` -- and the test is parametrised over which colour that is, because
    a frame error that happens to be right for white is the whole hazard.
    """
    game = scripted_decisive[terminal_mover]
    absolute = np.asarray(game.win_state(), dtype=np.int64)
    winner = 0 if int(absolute[0]) == 1 else 1
    state = sp.GameState(game, rig.DRAW_SCORE)

    assert state.is_terminal()
    assert state.to_play() == terminal_mover == int(game.player)
    assert winner != terminal_mover, (
        "the side to move at a decisive terminal is the loser; the engine "
        f"disagrees (winner {winner}, to move {terminal_mover})"
    )
    assert state.terminal_value() == pytest.approx(-1.0), (
        f"terminal_value {state.terminal_value()} for the mated side (mover "
        f"{terminal_mover}, winner {winner}) -- it is side-to-move framed and a "
        "loss is -1 (search/state.py's perspective convention)"
    )


@pytest.mark.parametrize("terminal_mover", [0, 1])
def test_absolute_outcome_is_the_engines_win_state_verbatim(
    scripted_decisive: dict[int, Any], terminal_mover: int
) -> None:
    """``absolute_outcome()`` is the raw ``(3,) uint8``, unflipped and unscaled."""
    game = scripted_decisive[terminal_mover]
    outcome = np.asarray(sp.GameState(game, rig.DRAW_SCORE).absolute_outcome())
    engine = np.asarray(game.win_state())
    assert outcome.shape == (3,)
    assert outcome.dtype == np.uint8, (
        f"absolute_outcome dtype {outcome.dtype}; the engine's frame is "
        "(3,) uint8 [white, black, draw] and the adapter passes it through"
    )
    assert np.array_equal(outcome, engine), (
        f"absolute_outcome {outcome.tolist()} != win_state {engine.tolist()}"
    )
    assert int(outcome.sum()) == 1


@pytest.mark.parametrize("plies", [8, 9])
def test_the_turn_cap_draw_is_worth_draw_score_to_whoever_is_to_move(
    plies: int,
) -> None:
    """Design D1/D6: a draw is ``draw_score`` in the mover's frame, both parities.

    Two lengths so the side to move at the cap differs between the two runs: the
    draw utility is a constant of the *reader's* perspective, so it must be the
    same 0.20 for either colour, and the recorded outcome must be a draw in
    either case.
    """
    game, _ = rig.capped_draw_game(seed=5, plies=plies)
    state = sp.GameState(game, rig.DRAW_SCORE)

    engine = np.asarray(game.win_state())
    assert int(engine[2]) == 1, "the fixture is meant to stand at the turn cap"
    assert state.is_terminal()
    assert np.array_equal(np.asarray(state.absolute_outcome()), engine)
    assert state.terminal_value() == pytest.approx(rig.DRAW_SCORE), (
        f"a draw is worth draw_score to the side to move (mover "
        f"{int(game.player)}), got {state.terminal_value()}"
    )


def test_a_draw_is_the_same_target_in_both_frames() -> None:
    """The reason the drawn game cannot catch the sign bug, stated once.

    ``[0, 1, 0]`` is its own flip, so a worker that never flips passes every
    drawn game.  This test exists so that fact is written down next to the
    decisive one that does the real work.
    """
    assert rig.mover_target((0, 0, 1), 0) == rig.mover_target((0, 0, 1), 1)
    assert rig.mover_target((1, 0, 0), 0) != rig.mover_target((1, 0, 0), 1)


def test_the_adapter_state_satisfies_the_evaluable_search_state_contract() -> None:
    """``EvaluableState`` is ``SearchState`` plus ``observation()`` (design D1)."""
    from komugi.engine.game import Game

    state = sp.GameState(Game(), rig.DRAW_SCORE)
    for name in (
        "legal_mask",
        "play",
        "clone",
        "is_terminal",
        "terminal_value",
        "to_play",
        "observation",
    ):
        assert callable(getattr(state, name)), f"GameState is missing {name}()"

    mask = np.asarray(state.legal_mask())
    assert mask.shape == (rig.NUM_ACTIONS,)
    assert mask.dtype == np.bool_, (
        f"legal_mask dtype {mask.dtype}; SearchState specifies a boolean mask "
        "and the engine's uint8 valid_moves() is the adapter's to convert"
    )
    observation = np.asarray(state.observation())
    assert observation.shape == (rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W)
    assert observation.dtype == np.float32
    assert not state.is_terminal()

    # clone() is deep: the search only ever mutates copies it made itself.
    clone = state.clone()
    action = int(np.flatnonzero(mask)[0])
    clone.play(action)
    assert np.array_equal(
        np.asarray(state.legal_mask()), mask
    ), "playing on a clone changed the original's legal mask"


class _CappedGame(Game):
    """A Game whose turn cap is tiny, so it adjudicates a draw in a few plies.

    Only ``max_turns`` changes and ``win_state`` reads it, so the draw comes
    from the ENGINE's own adjudication -- the worker adds no cap (D6/F15).
    """

    @staticmethod
    def max_turns() -> int:
        return 8


def test_a_drawn_worker_game_writes_pure_draw_rows_at_both_parities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every row of an adjudicated draw is ``[0, 1, 0]`` -- never draw_score.

    The wave-9b review (F4) found the decisive-game fixture early-exits before
    a drawn game is guaranteed, so draw labelling through the WORKER loop was
    unasserted: a finalize_rows bug writing contempt (0.2) into wdl_target, or
    one-hotting the wrong slot, would have passed this suite. The short-cap
    engine makes the drawn game deterministic.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _CappedGame)
    stub = rig.StubBatchEvaluator(seed=3)
    worker, writer = rig.build_worker(stub, tmp_path, games=1)
    try:
        plies = worker.play_game(0)
        writer.flush()
    finally:
        writer.close()

    rows = rig.read_rows(tmp_path)
    assert rows, "the capped game wrote no rows"
    assert len(rows) == int(plies)
    for row in rows:
        assert row["wdl_target"].tolist() == [0.0, 1.0, 0.0], (
            f"drawn-game row at ply {int(row['ply'].item())} has wdl_target "
            f"{row['wdl_target'].tolist()}; a draw is one-hot [0, 1, 0] in "
            f"EVERY frame and contempt must never reach a training target"
        )
    movers = {int(row["to_play"].item()) for row in rows}
    assert movers == {0, 1}, (
        f"drawn-game rows cover movers {sorted(movers)}; both parities must "
        f"be asserted or the per-row flip is untested on one side"
    )


# ---------------------------------------------------------------------------
# The draft temperature schedule, audited from the rows (KOM-036 §3.3, §3.5)
# ---------------------------------------------------------------------------

_DRAFT_PLANE_WHITE: int = 112
_DRAFT_PLANE_BLACK: int = 113
"""``observation.py``'s per-colour drafting-rights planes, in an **absolute** frame.

Absolute, not per-seat: the mover-relative audit rule is
``obs[112 + to_play, 0, 0] > 0``, and that arithmetic is only the mover's flag
because the two planes are ordered white-then-black regardless of who is to
move. Plane 114 carries side-to-move separately.
"""

_DRAFT_SCALE: float = 2.0
"""§3.2's pre-registered draft temperature, the value this audit is run at."""

_DRAFT_RUN_SEED: int = 20260906
"""The seed the sibling draft-boundary fixture already plays this line under.

Copied rather than chosen: ``tests/selfplay/unit/test_staged_targets.py`` runs
the same engine subclass, the same evaluator and the same two-simulation budget
at this seed and reaches the asymmetric plies the audit below needs.
"""


def _scheduled_config() -> Any:
    """The tiny config, plus §3.2's draft scale and nothing else.

    ``prob_fast=0.0`` so every ply is a full search and the composed scale is
    the draft factor alone: these nodes are about *when* the draft factor
    applies, and the product law is the unit suite's ``_scale_for`` node.
    The budgets match the sibling fixture's two-simulation search.
    """
    return rig.tiny_selfplay_config(
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=0.0,
        draft_gumbel_scale=_DRAFT_SCALE,
        fast_gumbel_scale=1.0,
    )


class _DraftBoundaryGame(Game):
    """The real engine in **advanced** mode, with a short turn cap.

    Advanced is a mode that actually has a draft: its start FEN is an empty
    board with both hands full and both draft bits set. Intro -- the process
    default, and what ``GUNGI_MODE`` leaves this class at otherwise -- starts
    from a laid-out board past the draft, so no ply of an intro game is a draft
    ply and this whole audit would be vacuous.

    Set as a **class attribute** rather than through ``Game.set_start_mode``,
    which would mutate the shared base class for the rest of the process. The
    same subclass exists in ``tests/selfplay/unit/test_staged_targets.py``,
    where it pins ``PendingRow.in_draft``; it is copied rather than shared
    because the two suites cut at different seams and a rig shared across the
    unit/behavior boundary would couple them.
    """

    _START_MODE: str = "advanced"

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap.

        Long enough for both sides to finish drafting **and** for the movement
        phase to produce rows -- without post-draft rows the "not all plies are
        hot" half of the audit has nothing to see -- and short enough that a
        four-simulation search finishes it in seconds.
        """
        return 40


class _FinishTheDraftEvaluator:
    """A stub that walks the two sides out of the draft **apart**.

    Uniform logits everywhere except the action encoding's block C -- "arata +
    draft-finish" -- with the bias chosen per row from two planes of the
    observation the worker hands it:

    * plane 114, side to move (``1.0`` for black);
    * plane 85, black's marshal-in-hand count.

    Boost block C for white, and for black once its marshal is down; penalise
    it for black while the marshal is still in hand. The engine's marshal-first
    rule then leaves black exactly one other kind of legal id -- the block-B
    placement of that same marshal, which keeps its draft open. The result is a
    game whose early plies have the two sides at *different* drafting statuses,
    which is the only state in which the mover-relative predicate and the
    disjunction can be told apart at all.
    """

    _SIDE_TO_MOVE_PLANE: int = 114
    _BLACK_MARSHAL_IN_HAND_PLANE: int = 85
    _BIAS: float = 16.0

    def __init__(self, actions: int) -> None:
        self.actions = actions

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> Any:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        black_to_move = obs[:, self._SIDE_TO_MOVE_PLANE, 0, 0] > 0.5
        marshal_in_hand = obs[:, self._BLACK_MARSHAL_IN_HAND_PLANE, 0, 0] > 0.0
        bias: npt.NDArray[np.float32] = np.where(
            black_to_move & marshal_in_hand, -self._BIAS, self._BIAS
        ).astype(np.float32)
        logits = np.zeros((rows, self.actions), dtype=np.float32)
        logits[:, C_START:] = bias[:, None]
        return sp.NetEval(
            policy_logits=logits,
            wdl=np.tile(np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ScheduleSpy(NamedTuple):
    """One ``_gumbel_config`` call, in play order."""

    fast: bool
    in_draft: bool
    scale: float


def _play_a_draft_game(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Row], list[_ScheduleSpy]]:
    """Play one advanced-mode game at ``draft_gumbel_scale=2.0``, spying the scale.

    ``prob_fast=0.0`` so every ply is a full search and the composed scale is
    the draft factor alone: this node is about *when* the draft factor applies,
    and the product law is #14's business.

    The scale is spied at ``_gumbel_config`` because it is deliberately **not**
    recoverable from a row -- §3.5 declines to add a schema field for a value
    that is constant per ``(in_draft, fast)`` cell, and records that the
    schedule's scope is therefore pinned by tests rather than by an audit of
    production shards.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _DraftBoundaryGame)
    calls: list[_ScheduleSpy] = []
    original = sp.SelfPlayWorker._gumbel_config

    def spy(worker: Any, fast: bool, in_draft: bool, seed: int) -> Any:
        built = original(worker, fast, in_draft, seed)
        calls.append(
            _ScheduleSpy(
                fast=bool(fast),
                in_draft=bool(in_draft),
                scale=float(built.gumbel_scale),
            )
        )
        return built

    monkeypatch.setattr(sp.SelfPlayWorker, "_gumbel_config", spy)

    worker, writer = rig.build_worker(
        _FinishTheDraftEvaluator(rig.NUM_ACTIONS),
        directory,
        run_seed=_DRAFT_RUN_SEED,
        games=1,
        selfplay_config=_scheduled_config(),
    )
    try:
        written = worker.play_game(0)
        writer.flush()
    finally:
        writer.close()

    rows = _ordered(sorted(rig.read_rows(directory), key=lambda r: rig.as_int(r, "ply")))
    assert len(rows) == int(written) == len(calls), (
        "one _gumbel_config call per recorded ply is what makes the spy and the "
        "rows line up by index"
    )
    return rows, calls


def _mover_is_drafting(row: Row) -> bool:
    """§3.5's audit rule: ``obs[112 + to_play, 0, 0] > 0``, off the row itself."""
    observation = rig.as_array(row, "observation")
    return bool(observation[_DRAFT_PLANE_WHITE + rig.as_int(row, "to_play"), 0, 0] > 0.0)


def _either_side_is_drafting(row: Row) -> bool:
    """The **disjunction**: the whole-game phase question, kept for contrast."""
    observation = rig.as_array(row, "observation")
    return bool(
        observation[_DRAFT_PLANE_WHITE, 0, 0] > 0.0
        or observation[_DRAFT_PLANE_BLACK, 0, 0] > 0.0
    )


def test_the_schedule_fires_only_when_the_mover_is_drafting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#22: the hot plies are exactly the rows whose **mover** was drafting.

    The audit runs off the shard, using §3.5's rule
    ``obs[112 + to_play, 0, 0] > 0`` -- both the plane index and the mover come
    off the row, so it needs no join and no schema change, and it is the rule a
    successor will re-run against production shards to measure §7's R3/R4.

    This is also the node ``kom-036:1134`` names as pinning that the schedule's
    mover-relative read and KOM-005b's observation-derived ``in_draft`` read
    **agree**: they are different predicates, so what is asserted is the
    relation that has to hold -- a drafting mover implies a drafting *game* --
    together with the game having actually visited a ply where the two differ.
    """
    rows, calls = _play_a_draft_game(tmp_path, monkeypatch)

    mover_drafting = [_mover_is_drafting(row) for row in rows]
    either_drafting = [_either_side_is_drafting(row) for row in rows]

    assert any(mover_drafting), "no recorded ply was a draft ply"
    assert not all(mover_drafting), "the game never left the draft phase"

    # The worker's own read agrees with the shard's, ply for ply.
    assert [call.in_draft for call in calls] == mover_drafting, (
        "the scale's draft read and the recorded observation planes disagree"
    )

    # And the scale it produced is the draft factor exactly on those plies.
    assert [call.scale for call in calls] == [
        _DRAFT_SCALE if hot else 1.0 for hot in mover_drafting
    ]
    assert all(not call.fast for call in calls), "this fixture runs prob_fast=0.0"

    # A drafting mover implies a drafting game; the converse is what the
    # disjunction would wrongly assume, and the next assertion shows the two
    # sets are not the same set.
    for hot, either in zip(mover_drafting, either_drafting, strict=True):
        assert not hot or either


def test_a_one_sided_draft_ply_is_hot_only_for_the_drafter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#22a: at a ply with exactly one draft bit set, only the drafter is hot.

    Two halves, and the split is forced by the engine rather than chosen.

    * **Through play.** The fixture reaches plies where exactly one draft bit
      is set, and at each of them the scale is ``draft_gumbel_scale`` -- because
      at those plies the mover *is* the side still drafting.
    * **Constructed.** The other case -- the mover is the side that has
      **finished** while the other still drafts -- is *unreachable by play*:
      ``cyengine``'s turn rule, in the asymmetric state, only ever hands the
      move back to the side that is still drafting (``draft_w == 0 and
      turn == 0`` sets ``turn = 1``, and black finishing clears white's bit
      too). So the discriminating case is built directly, and the assertion is
      on the scale the worker composes from ``GameState.in_draft()`` at it.
      Under the disjunctive predicate that scale would be ``2.0``; under the
      mover-relative one it is ``1.0``.

    Red under the disjunction, green under the mover-relative predicate -- the
    discrimination #20 and #22 both lacked.
    """
    rows, calls = _play_a_draft_game(tmp_path, monkeypatch)

    # --- through play --------------------------------------------------------
    asymmetric = 0
    for row, call in zip(rows, calls, strict=True):
        observation = rig.as_array(row, "observation")
        white = bool(observation[_DRAFT_PLANE_WHITE, 0, 0] > 0.0)
        black = bool(observation[_DRAFT_PLANE_BLACK, 0, 0] > 0.0)
        if white == black:
            continue
        asymmetric += 1
        mover = rig.as_int(row, "to_play")
        drafter = 0 if white else 1
        assert mover == drafter, (
            "the engine handed the move to the side that had finished; the "
            "constructed half below is then reachable through play and this "
            "node should assert it there instead"
        )
        assert call.in_draft is True
        assert call.scale == _DRAFT_SCALE

    assert asymmetric > 0, (
        "no recorded ply has the two sides at different drafting statuses, so "
        "this node cannot distinguish the mover-relative predicate from the "
        "disjunction -- it is void as written"
    )

    # --- constructed: the mover is the side that has finished ----------------
    monkeypatch.setattr("komugi.selfplay.worker.Game", _DraftBoundaryGame)
    worker, writer = rig.build_worker(
        _FinishTheDraftEvaluator(rig.NUM_ACTIONS),
        tmp_path / "constructed",
        selfplay_config=_scheduled_config(),
    )
    writer.close()

    for draft_w, draft_b in ((1, 0), (0, 1)):
        finished_mover = 1 if draft_w else 0
        drafting_mover = 0 if draft_w else 1

        cold = _state_with(draft_w, draft_b, finished_mover)
        assert cold.in_draft() is False
        assert worker._scale_for(False, cold.in_draft()) == 1.0  # noqa: SLF001

        hot = _state_with(draft_w, draft_b, drafting_mover)
        assert hot.in_draft() is True
        assert worker._scale_for(False, hot.in_draft()) == _DRAFT_SCALE  # noqa: SLF001

        # The disjunction cannot tell these two positions apart: it answers
        # True for both, so it would run the finished side's search hot.
        assert bool(draft_w or draft_b)


def _state_with(draft_w: int, draft_b: int, to_play: int) -> Any:
    """Return a ``GameState`` at an advanced-mode draft position, mover forced.

    ``board.turn`` is written before the :class:`Game` is built, because
    ``Game.__init__`` copies it into ``player`` and ``GameState.to_play`` reads
    that.
    """
    game = _DraftBoundaryGame()
    board: Any = game._board  # noqa: SLF001
    board.draft_w = draft_w
    board.draft_b = draft_b
    board.turn = to_play
    game._player = to_play  # noqa: SLF001
    return sp.GameState(game, rig.DRAW_SCORE)
