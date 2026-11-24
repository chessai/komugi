"""KOM-245 -- soft material adjudication of turn-capped self-play games.

The rule, as ``finalize_rows`` implements it: a game the **engine** stopped at
its turn cap keeps ``termination = TERMINATION_DRAW`` and every other column as
before, but under ``cap_adjudication.mode == "material"`` its ``wdl_target`` is
no longer ``(0, 1, 0)``.  With ``d = white - black`` on the final board (TOTAL
material, board at any tier plus hand) and
``p = p_max * min(1, |d| / d_full)``, every row whose ``to_play`` is the leader
gets ``(p, 1 - p, 0)`` and every other row ``(0, 1 - p, p)``; ``d == 0`` leaves
``(0, 1, 0)`` untouched.

What each test here is pinned against, beyond the arithmetic:

* **The frame is ``row.to_play``, never ply parity** -- the synthetic games
  below all contain the draft's double move (two consecutive rows with the same
  mover), which is exactly the case a parity-keyed flip sign-inverts.
* **Review F2**: ``outcome_value``/``z``, ``short_term_value``'s past-the-end
  clamp and ``draft_outcome`` are *derivations* of ``wdl_target`` and are not
  special-cased anywhere -- so an adjudicated game's draft rows carry a soft
  ``draft_outcome`` and its tail clamps to ``+-p``.  They are asserted here
  because the invariant list used to claim the opposite.
* **Review F3**: the cap is the ENGINE's ``termination_reason()``, passed in by
  the caller.  ``total_plies >= Game.max_turns()`` would read the *class*
  constant, which the unit suite's ``_ShortGame`` (a 6-ply cap) falsifies
  silently -- the real-engine test below is red under that spelling.
* **Review F4**: the stored target on a played game equals the rule applied to
  the engine's own count of the final board, which is also the plane decode the
  analysis scripts use.
* **Nothing else moves**: mode ``off`` on a capped draw, the material mode on a
  draw *below* the cap, and any decisive game are all byte-identical to today's
  rows, field for field.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.engine.cyboard import Board, board_from_fen
from komugi.engine.game import TERMINATION_TURN_CAP, Game
from komugi.engine.observation import (
    CONTROL_EMPTY,
    CONTROL_SIZE,
    MAX_COUNT,
    PIECE_TYPES,
)
from komugi.selfplay.adapter import GameState
from komugi.selfplay.records import SCHEMA, SHORT_TERM_HORIZONS, num_actions
from komugi.selfplay.worker import (
    CAP_ADJUDICATION_OFF,
    MODE_FULL,
    TERMINATION_DECISIVE,
    TERMINATION_DRAW,
    CapAdjudication,
    PendingRow,
    SelfPlayWorker,
    WorkerConfig,
    finalize_rows,
)
from komugi.train.shards import ShardReader, ShardWriter

from test_worker_unit import (  # the fixtures this suite deliberately shares
    _SHORT_GAME_PLIES,
    _StubEvaluator,
    _selfplay_config,
)


class _InstanceCappedGame(Game):
    """The real engine, stopping after :data:`_SHORT_GAME_PLIES` -- per INSTANCE.

    Deliberately **not** ``test_worker_unit._ShortGame``, which shortens the
    class's own ``max_turns()``: under that fixture a worker that re-derived
    the cap as ``total_plies >= Game.max_turns()`` would read the shortened
    constant and look correct, and the second spelling review F3 rejects would
    ship un-caught.

    Here ``max_turns()`` stays the production 512 -- the class constant is a
    cap this game never reaches -- while the two methods that own adjudication
    stop it at six plies. Only the *engine's own answer*,
    ``termination_reason()``, says the game was capped, which is exactly the
    seam under test.

    Everything else -- move generation, checkmate, repetition, and the engine's
    precedence below the cap -- is the production engine's, through ``super()``.
    """

    def win_state(self) -> npt.NDArray[np.uint8]:
        """A draw once six plies have been played; the engine's answer before."""
        if self._turns >= _SHORT_GAME_PLIES:  # noqa: SLF001
            capped = np.zeros(3, dtype=np.uint8)
            capped[2] = 1
            return capped
        return super().win_state()

    def termination_reason(self) -> str:
        """``turn_cap`` at six plies, first and outranking every other reason."""
        if self._turns >= _SHORT_GAME_PLIES:  # noqa: SLF001
            return TERMINATION_TURN_CAP
        return super().termination_reason()


#: ``tests/rules/fixtures.POSITION_Q_FEN`` -- white 4 pieces, black 3, and both
#: draft bits clear, so the game starts in open play and the totals are already
#: unequal at ply 0.  Spelled out rather than imported because ``tests/rules``
#: is not on this suite's import path.
UNEQUAL_FEN: Final[str] = "4m3n/9/r8/9/8W/9/9/9/4M4 D2/- w 3 - 1"


class _UnequalCappedGame(_InstanceCappedGame):
    """:class:`_InstanceCappedGame`, seeded from :data:`UNEQUAL_FEN`.

    The six-ply game the sibling fixture plays starts from the production
    opening, where a drop-and-draft game cannot move either total: it can only
    ever exercise the rule's ``d == 0`` branch.  This one starts a piece up, so
    the *soft* label -- the whole of KOM-245 -- is what the stored targets
    carry, and a rule that silently fell back to ``(0, 1, 0)`` is red.

    Only the default board moves; ``clone()``, ``__reduce__`` and the symmetry
    helpers all pass a board positionally and so are untouched.
    """

    def __init__(self, board: Board | None = None, *args: Any, **kwargs: Any):
        """Default to the unequal position; any explicit board wins."""
        if board is None:
            board = board_from_fen(UNEQUAL_FEN)
        super().__init__(board, *args, **kwargs)

#: The pre-registered constants: a side ``D_FULL`` pieces up is labelled with a
#: ``P_MAX`` win probability, and the ramp is linear below that.  A mate is
#: worth ``1.0``, so pressing to win still beats coasting to the cap.
P_MAX: Final[float] = 0.8
D_FULL: Final[int] = 8

MATERIAL: Final[CapAdjudication] = CapAdjudication(
    mode="material", p_max=P_MAX, d_full=D_FULL
)

#: The movers of the synthetic game: four rows, two colours, and **black moves
#: twice in a row at rows 1-2** -- the draft's double move, which is what makes
#: a ply-parity flip observably wrong.
MOVERS: Final[tuple[int, ...]] = (0, 1, 1, 0)

#: The absolute draw outcome; the engine reports a turn cap as one.
DRAW: Final[npt.NDArray[np.uint8]] = np.array([0, 0, 1], dtype=np.uint8)


def _pending(
    ply: int, to_play: int, *, root_value: float = 0.0, in_draft: bool = False
) -> PendingRow:
    """A minimal buffered row at `ply`, moved by `to_play`."""
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
    """The one control map that is invariant under the mover flip."""
    return np.full(CONTROL_SIZE, CONTROL_EMPTY, dtype=np.int8)


def _rows(*, in_draft: tuple[int, ...] = ()) -> list[PendingRow]:
    """The synthetic four-row game, optionally with some rows in the draft."""
    return [
        _pending(ply, mover, root_value=0.25, in_draft=ply in in_draft)
        for ply, mover in enumerate(MOVERS)
    ]


def _finalize(
    material: tuple[int, int] | None,
    *,
    outcome: npt.NDArray[np.uint8] = DRAW,
    cap_adjudication: CapAdjudication = MATERIAL,
    capped: bool = True,
    in_draft: tuple[int, ...] = (),
) -> list[dict[str, torch.Tensor]]:
    """Finalise the synthetic game under one setting of the rule."""
    return finalize_rows(
        _rows(in_draft=in_draft),
        outcome,
        _empty_control(),
        len(MOVERS),
        7,
        cap_adjudication=cap_adjudication,
        final_material=material,
        capped=capped,
    )


def _wdl(samples: list[dict[str, torch.Tensor]]) -> list[list[float]]:
    """The finalised ``wdl_target`` s, as plain lists."""
    return [sample["wdl_target"].numpy().tolist() for sample in samples]


def _assert_identical(
    left: list[dict[str, torch.Tensor]], right: list[dict[str, torch.Tensor]]
) -> None:
    """Every field of every row must be equal tensor for tensor."""
    assert len(left) == len(right)
    for index, (first, second) in enumerate(zip(left, right)):
        assert set(first) == set(second) == set(SCHEMA.names)
        for name in SCHEMA.names:
            assert torch.equal(first[name], second[name]), f"row {index} {name}"


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_a_capped_draw_is_labelled_by_the_material_lead() -> None:
    """``d = +5`` -> ``p = 0.5``: white's rows win-ish, black's lose-ish."""
    samples = _finalize((20, 15))

    for sample, mover in zip(samples, MOVERS):
        wdl = sample["wdl_target"].numpy().tolist()
        if mover == 0:
            assert wdl == pytest.approx([0.5, 0.5, 0.0])
        else:
            assert wdl == pytest.approx([0.0, 0.5, 0.5])
        # z = w - l, and no second rule computes it (review F2).
        z = float(wdl[0] - wdl[2])
        assert z == pytest.approx(0.5 if mover == 0 else -0.5)


def test_the_label_is_keyed_on_the_rows_mover_and_not_on_ply_parity() -> None:
    """Rows 1 and 2 share a mover -- the draft's double move -- and a label.

    A parity-keyed flip would give this pair opposite labels, which is the
    perspective bug design F11 is about, in the one place the draft makes it
    reachable.
    """
    labels = _wdl(_finalize((20, 15)))

    assert MOVERS[1] == MOVERS[2]
    assert labels[1] == labels[2]
    assert labels[0] == labels[3] != labels[1]


def test_a_black_lead_mirrors_the_white_one() -> None:
    """``d = -3`` is ``d = +3`` with the colours exchanged, and nothing else."""
    white_ahead = _wdl(_finalize((18, 15)))
    black_ahead = _wdl(_finalize((15, 18)))

    expected = P_MAX * 3 / D_FULL
    for row, mover in enumerate(MOVERS):
        leader_label = [expected, 1.0 - expected, 0.0]
        trailer_label = [0.0, 1.0 - expected, expected]
        assert white_ahead[row] == pytest.approx(
            leader_label if mover == 0 else trailer_label
        )
        assert black_ahead[row] == pytest.approx(
            trailer_label if mover == 0 else leader_label
        )


def test_a_level_cap_game_is_still_a_plain_draw() -> None:
    """``d == 0``: nobody is ahead, so the label is exactly today's."""
    assert _wdl(_finalize((17, 17))) == [[0.0, 1.0, 0.0]] * len(MOVERS)
    _assert_identical(
        _finalize((17, 17)), _finalize(None, cap_adjudication=CAP_ADJUDICATION_OFF)
    )


@pytest.mark.parametrize("material", [(23, 15), (25, 5), (15, 23)])
def test_the_ramp_saturates_at_p_max(material: tuple[int, int]) -> None:
    """``|d| >= d_full`` -> ``p = p_max``, and never above it.

    ``0.8`` is inexact in float32, hence the approx (review F9).
    """
    for sample in _finalize(material):
        soft = sample["wdl_target"].numpy()
        assert float(max(soft[0], soft[2])) == pytest.approx(P_MAX)
        assert float(soft[1]) == pytest.approx(1.0 - P_MAX)


def test_every_adjudicated_game_is_still_a_draw() -> None:
    """The termination vocabulary does not move: this is a DRAW, relabelled.

    ``3`` stays reserved for KOM-168's dead-draw adjudication, and the
    trainer's cap-row weight keys on this column.
    """
    for material in ((20, 15), (17, 17), (5, 25)):
        for sample in _finalize(material):
            assert int(sample["termination"]) == TERMINATION_DRAW


# --------------------------------------------------------------------------
# What must NOT move
# --------------------------------------------------------------------------


def test_mode_off_leaves_a_capped_draw_byte_identical() -> None:
    """The switch is a switch: off is today's worker, field for field."""
    _assert_identical(
        _finalize((25, 5), cap_adjudication=CAP_ADJUDICATION_OFF),
        _finalize(None, cap_adjudication=CAP_ADJUDICATION_OFF, capped=False),
    )
    assert _wdl(_finalize((25, 5), cap_adjudication=CAP_ADJUDICATION_OFF)) == [
        [0.0, 1.0, 0.0]
    ] * len(MOVERS)


def test_a_draw_below_the_cap_is_untouched_by_the_material_mode() -> None:
    """A repetition or stalemate draw is a RULE draw: neither side failed.

    Only the cap means "neither side could finish", which is the whole
    justification for a material proxy.
    """
    _assert_identical(
        _finalize((25, 5), capped=False),
        _finalize(None, cap_adjudication=CAP_ADJUDICATION_OFF, capped=False),
    )


@pytest.mark.parametrize("outcome", [(1, 0, 0), (0, 1, 0)])
def test_a_decisive_game_is_byte_identical(outcome: tuple[int, int, int]) -> None:
    """A win is worth 1.0 whatever the material was; nothing here touches it."""
    absolute = np.array(outcome, dtype=np.uint8)
    _assert_identical(
        _finalize((25, 5), outcome=absolute),
        _finalize(None, outcome=absolute, cap_adjudication=CAP_ADJUDICATION_OFF),
    )
    for sample in _finalize((25, 5), outcome=absolute):
        assert int(sample["termination"]) == TERMINATION_DECISIVE
        assert float(sample["wdl_target"].numpy().max(initial=0.0)) == 1.0


# --------------------------------------------------------------------------
# Review F2 -- the derived columns follow, and are not special-cased
# --------------------------------------------------------------------------


def test_cap_adjudication_reaches_draft_outcome_and_the_z_clamp() -> None:
    """``draft_outcome`` is a copy of ``wdl``; the tail clamp is ``z = +-p``.

    Review F2: the design note claimed both were unchanged by the rule. They
    cannot be -- both are derivations of ``wdl_target`` -- and a future
    bring-up of either head would inherit the wrong invariant silently.
    """
    samples = _finalize((20, 15), in_draft=(0, 1))

    assert samples[0]["draft_outcome"].numpy().tolist() == pytest.approx(
        [0.5, 0.5, 0.0]
    )
    assert samples[1]["draft_outcome"].numpy().tolist() == pytest.approx(
        [0.0, 0.5, 0.5]
    )
    # Rows 2-3 were not in the draft, so the column stays masked to zeros.
    for sample in samples[2:]:
        assert sample["draft_outcome"].numpy().tolist() == [0.0, 0.0, 0.0]

    # Every horizon of the last row is past the end of the buffer, so all three
    # slots are the clamp -- the row's own z, by its own mover.
    for index, sample in enumerate(samples):
        if index + min(SHORT_TERM_HORIZONS) < len(samples):
            continue
        expected = 0.5 if MOVERS[index] == 0 else -0.5
        assert sample["short_term_value"].numpy().tolist() == pytest.approx(
            [expected] * len(SHORT_TERM_HORIZONS)
        )


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


def test_an_unknown_mode_is_refused() -> None:
    """A typo'd mode must not be read as "not material", i.e. as off."""
    with pytest.raises(ValueError, match="cap adjudication mode"):
        CapAdjudication(mode="materal", p_max=P_MAX, d_full=D_FULL)


def test_from_selfplay_reads_the_three_config_knobs() -> None:
    """The one place production reads the config (KOM-249 landed the knobs)."""
    config = replace(
        _selfplay_config(0.0),
        cap_adjudication="material",
        cap_adjudication_p_max=0.6,
        cap_adjudication_d_full=4,
    )
    assert CapAdjudication.from_selfplay(config) == CapAdjudication(
        mode="material", p_max=0.6, d_full=4
    )
    assert CapAdjudication.from_selfplay(
        _selfplay_config(0.0)
    ) == CAP_ADJUDICATION_OFF


def test_the_rule_refuses_to_fire_with_no_material() -> None:
    """A labelling rule with no input refuses rather than silently reverting."""
    with pytest.raises(ValueError, match="final_material"):
        _finalize(None)


# --------------------------------------------------------------------------
# Review F3/F4 -- the real engine, end to end
# --------------------------------------------------------------------------


def _material_from_planes(observation: npt.NDArray[Any]) -> tuple[int, int]:
    """Decode ``(white, black)`` totals out of one position's planes.

    ``scripts/analysis/material_vs_outcome.decode_material`` over a single
    position: colour is the occupancy plane's parity and a hand plane is a
    constant ``count / MAX_COUNT``.
    """
    planes = np.asarray(observation, dtype=np.float64)
    occupancy = len(PIECE_TYPES) * 3 * 2
    board = planes[:occupancy]
    max_by_index = np.array([MAX_COUNT[piece] for piece in PIECE_TYPES])
    hand = np.rint(
        planes[occupancy : occupancy + len(PIECE_TYPES) * 2, 0, 0]
        * np.repeat(max_by_index, 2)
    )
    return (
        int(round(float(board[0::2].sum() + hand[0::2].sum()))),
        int(round(float(board[1::2].sum() + hand[1::2].sum()))),
    )


def _play_capped_game(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    game_class: type[Game] = _InstanceCappedGame,
) -> list[dict[str, torch.Tensor]]:
    """Play one game on the shortened real engine, with the rule ON.

    ``game_class`` picks the starting position: the default opening (which no
    six-ply game can unbalance) or :class:`_UnequalCappedGame`'s material lead.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", game_class)
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        replace(_selfplay_config(0.0), cap_adjudication="material"),
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=20260918,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )
    worker.play_game(0)
    writer.close()
    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    return list(reader)


def test_a_short_cap_game_is_adjudicated_by_material_on_the_real_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F3: the caller passes the ENGINE's cap, not a ply comparison.

    RED under ``total_plies >= Game.max_turns()``: :class:`_InstanceCappedGame`
    stops after six plies while its ``max_turns()`` is still the production
    512, so that spelling passes ``capped=False`` here and the rule never fires
    -- which is exactly how it would have shipped un-exercised.

    The six-ply game cannot produce a material lead (drops and the draft leave
    both totals untouched), so what is asserted is that the rule **fires**: the
    material mode, the engine's own ``final_material`` and ``capped=True``
    reach ``finalize_rows``, and the label is then the rule's answer for
    ``d == 0``.
    """
    seen: list[dict[str, Any]] = []
    real = finalize_rows

    def spy(*args: Any, **kwargs: Any) -> list[dict[str, torch.Tensor]]:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("komugi.selfplay.worker.finalize_rows", spy)
    rows = _play_capped_game(tmp_path / "capped", monkeypatch)

    assert len(seen) == 1, "one game, one finalisation"
    passed = seen[0]
    assert passed["capped"] is True, (
        "the worker must read the cap off the engine's termination_reason(); "
        "a class-constant ply comparison reports False for this 6-ply cap"
    )
    assert passed["cap_adjudication"] == CapAdjudication(
        mode="material", p_max=P_MAX, d_full=D_FULL
    )
    white, black = passed["final_material"]
    assert white > 0 and black > 0, "both sides still hold their pieces"

    difference = white - black
    p = P_MAX * min(1.0, abs(difference) / D_FULL)
    for row in rows:
        assert int(row["termination"]) == TERMINATION_DRAW
        wdl = row["wdl_target"].numpy()
        if difference == 0:
            assert wdl.tolist() == [0.0, 1.0, 0.0]
        else:
            leader = 0 if difference > 0 else 1
            assert float(max(wdl[0], wdl[2])) == pytest.approx(p)
            assert (int(np.argmax(wdl)) == 0) == (int(row["to_play"]) == leader)


def test_stored_p_agrees_with_the_plane_decode_on_a_level_cap_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F4, the ``d == 0`` half: the two material reads agree, label ``(0, 1, 0)``.

    This game starts from the production opening, where six plies of drops and
    draft cannot move either total -- so what it pins is the *plumbing*: the
    engine's own count of the final board equals the plane decode the analysis
    scripts use, and the rule's level branch leaves the stored target at
    today's plain draw.  The soft label is pinned by
    :func:`test_stored_p_is_soft_on_a_played_cap_game_with_a_lead`.

    The chain, end to end and with no recomputation shared between its links:
    replay the recorded ``action`` column into a fresh engine, confirm the
    engine calls the final position a turn cap, count its material two ways
    (the engine's own read and the analysis scripts' plane decode -- they must
    be one number), apply the rule by hand, and compare with what is on disk.
    """
    rows = _play_capped_game(tmp_path / "stored", monkeypatch)
    assert 0 < len(rows) <= _SHORT_GAME_PLIES

    state = GameState(_InstanceCappedGame(), 0.2)
    for row in rows:
        state.play(int(row["action"]))

    assert state.game.termination_reason() == TERMINATION_TURN_CAP, (
        "the shortened engine must have stopped this game at its cap; without "
        "that the rule under test never applies"
    )
    material = state.final_material()
    assert material == _material_from_planes(state.observation()), (
        "the engine's count and the plane decode the analysis scripts use must "
        "agree on the final board"
    )

    white, black = material
    difference = white - black
    p = P_MAX * min(1.0, abs(difference) / D_FULL)
    leader = None if difference == 0 else (0 if difference > 0 else 1)
    for row in rows:
        stored = row["wdl_target"].numpy().tolist()
        if leader is None:
            expected = [0.0, 1.0, 0.0]
        elif int(row["to_play"]) == leader:
            expected = [p, 1.0 - p, 0.0]
        else:
            expected = [0.0, 1.0 - p, p]
        assert stored == pytest.approx(expected)


def test_stored_p_is_soft_on_a_played_cap_game_with_a_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F1: the same chain, on a game whose final board is UNEQUAL.

    The level game above can only ever show ``p == 0``, so every stored target
    in it is byte-identical to the pre-KOM-245 worker's: a rule that never
    fired -- or one that fired and produced ``(0, 1, 0)`` regardless -- is
    green there.  Here the engine starts white a piece up, the cap arrives
    before either side can trade it off, and what is asserted is the *soft*
    number itself: ``p > 0``, on every row, with the win mass on the leader's
    rows and the loss mass on the trailer's.
    """
    rows = _play_capped_game(tmp_path / "unequal", monkeypatch, _UnequalCappedGame)
    assert 0 < len(rows) <= _SHORT_GAME_PLIES

    state = GameState(_UnequalCappedGame(), 0.2)
    for row in rows:
        state.play(int(row["action"]))

    assert state.game.termination_reason() == TERMINATION_TURN_CAP, (
        "the shortened engine must have stopped this game at its cap; without "
        "that the rule under test never applies"
    )
    material = state.final_material()
    assert material == _material_from_planes(state.observation()), (
        "the engine's count and the plane decode the analysis scripts use must "
        "agree on the final board"
    )

    white, black = material
    assert white != black, (
        f"the seeded lead was traded away -- the final board is {material}, so "
        f"this test would assert the d == 0 branch and claim nothing about the "
        f"soft label; reseed {UNEQUAL_FEN!r} or the worker's run_seed"
    )

    difference = white - black
    p = P_MAX * min(1.0, abs(difference) / D_FULL)
    assert p > 0.0
    leader = 0 if difference > 0 else 1
    for row in rows:
        assert int(row["termination"]) == TERMINATION_DRAW
        stored = row["wdl_target"].numpy()
        win, loss = float(stored[0]), float(stored[2])
        assert max(win, loss) == pytest.approx(p, abs=1e-6)
        assert (win > 0.0) == (int(row["to_play"]) == leader)
