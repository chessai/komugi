"""KOM-002 §7 D -- the progress clock, plane 120.

``Game._progress_plies`` counts plies since the last **progress move**, and
design §3.3 gives that notion twice: once operationally (the move record's
``move_type`` is a capture, a betrayal or an arata) and once as a potential
function (``Φ(b) = 2 * hand_arr.sum() + count_nonzero(cells_arr)`` strictly
decreases).  The two definitions are supposed to coincide exactly; #14 is what
makes that a falsifiable claim rather than a restatement of the implementation.

The other three pin the plane's relationships: it is bounded above by the turn
clock it shares a denominator with (#15), it is implied by repetition (#16,
invariant P2), and it stays at zero through the draft, where every ply is an
arata by construction (#18).

#17 is the one that looks like it belongs elsewhere and does not: plane 120 is
PERCEPTUAL, and the claim that a perceptual plane cannot touch adjudication is
only worth anything if it is checked against a capture taken BEFORE the plane
existed.  That capture is ``tests/encoding/goldens/pmv1/``'s ``win_state``
array, frozen at ``main@29213f2`` in the same npz as #3's digests (design
rev 4).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine import action_encoding
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game

from fixtures import (
    GOLDEN_CORPUS_PLIES,
    PLANE_PROGRESS,
    PLANE_TURN_CLOCK,
    PMV1_MODES,
    PROGRESS_MOVE_TYPES,
    PURE_SHUFFLE,
    PlySnapshot,
    board_of,
    golden_action_sequences,
    load_game,
    load_pmv1_capture,
    load_position_p,
    observation_planes,
    play_action_checked,
    repetition_count,
    seeded_playout,
    start_mode,
    valid_mask,
    win_state,
)

_PLIES: Final[int] = 120
"""Length of the seeded game #14 and #15 run over."""

_SEED: Final[int] = 20260904


@pytest.fixture(scope="module")
def playout() -> list[PlySnapshot]:
    """A 120-ply seeded game, snapshotted per ply.

    Module-scoped because #14 and #15 are two readings of the same run: they
    should not disagree about which game they are describing, and driving it
    twice would only buy the chance that they do.
    """
    return seeded_playout(_PLIES, _SEED)


def test_move_type_rule_matches_the_potential(playout: list[PlySnapshot]) -> None:
    """#14: ``ΔΦ ≤ 0`` always, and ``ΔΦ < 0`` **iff** the clock reset.

    The cross-check design §3.3 owes: the implementation resets on a move-type
    test, and this asserts that test agrees, move for move, with a quantity
    computed from the BOARD -- so widening the reset condition to a reversible
    move type (a route, a tsuke) fails here immediately.

    The clock reaching 0 is an unambiguous witness of a reset: it starts at 0
    and every non-progress ply increments it, so a 0 after ply 0 can only mean
    the counter was cleared on that ply.
    """
    resets = 0
    for previous, current in zip(playout[:-1], playout[1:], strict=True):
        delta = current.potential - previous.potential
        where = f"ply {current.ply} (move_type={current.move_type})"

        assert delta <= 0, (
            f"{where}: Φ rose by {delta}; no legal move adds material to the "
            "board or to a hand, so this is a rules change, not a clock bug"
        )
        assert (delta < 0) == (current.progress_plies == 0), (
            f"{where}: ΔΦ={delta} but progress_plies={current.progress_plies}; "
            "the potential-function and move-type definitions of a progress "
            "move have come apart"
        )
        assert (current.move_type in PROGRESS_MOVE_TYPES) == (delta < 0), (
            f"{where}: ΔΦ={delta} disagrees with the move-type rule "
            f"{sorted(PROGRESS_MOVE_TYPES)}"
        )

        if delta < 0:
            resets += 1
        else:
            assert current.progress_plies == previous.progress_plies + 1, (
                f"{where}: a reversible move must advance the clock by one"
            )

    assert resets > 0, (
        "the seeded game contained no progress move at all, so the 'iff' was "
        "only ever exercised in one direction"
    )


def test_progress_clock_never_exceeds_the_turn_clock(
    playout: list[PlySnapshot],
) -> None:
    """#15: plane 120 ≤ plane 115, elementwise, at every ply; both 0 at ply 0.

    They share a denominator (``MAX_TURNS``), and ``progress_plies`` is
    bounded by ``_turns`` by construction, so the inequality is a property of
    the encoding rather than of this particular game.  It is also the cheapest
    detector of a *separate* progress cap being introduced: any denominator
    other than plane 115's breaks it somewhere in 120 plies.
    """
    first = playout[0]
    assert first.observation[PLANE_PROGRESS].max() == 0.0
    assert first.observation[PLANE_TURN_CLOCK].max() == 0.0

    for snapshot in playout:
        progress = snapshot.observation[PLANE_PROGRESS]
        turn = snapshot.observation[PLANE_TURN_CLOCK]
        assert np.all(progress <= turn), (
            f"ply {snapshot.ply}: progress clock {progress.max()} exceeds the "
            f"turn clock {turn.max()} (progress_plies="
            f"{snapshot.progress_plies})"
        )
        # Both are constant planes; a spatial value here would mean the block
        # was written through the last-move planes' square indexing by mistake.
        assert np.unique(progress).size == 1, f"ply {snapshot.ply}"


def test_repetition_implies_progress() -> None:
    """#16, invariant P2: ``repetition_count >= 2`` implies ``progress >= 2``.

    Two occurrences of the same 274-byte key have the same cells and the same
    hands, hence the same Φ; Φ is non-increasing, so it was constant across
    the span between them and no progress move happened in it.  The key pins
    the side to move, so that span is a nonzero even number of plies.

    Driven over ``PURE_SHUFFLE``, whose four plies are routes -- the line that
    reaches occurrences 2, 3 and 4 without ever resetting the clock.
    """
    game = load_position_p()
    assert repetition_count(game) == 1, "precondition: a first occurrence"

    checked = 0
    for cycle in range(3):
        for offset, action in enumerate(PURE_SHUFFLE):
            play_action_checked(game, action, cycle * len(PURE_SHUFFLE) + offset + 1)
            if repetition_count(game) >= 2:
                assert game._progress_plies >= 2, (
                    f"cycle {cycle} ply {offset + 1}: the position has occurred "
                    f"{repetition_count(game)} times but the progress clock "
                    f"reads {game._progress_plies}"
                )
                checked += 1

    assert checked > 0, "the shuffle never reached a second occurrence"


def test_adjudication_is_unchanged() -> None:
    """#17: the ``win_state`` sequence of every golden game is unchanged.

    Replays each game of ``tests/engine/goldens/playouts_*.json`` and compares
    the ``win_state()`` after every single ply against the capture taken at
    ``main@29213f2`` -- before ``_progress_plies`` existed at all.  Design §3.3
    argues the clock cannot touch adjudication (the engine has no fifty-move
    rule; its draws are the engine's ply cap, fourfold repetition, stalemate and
    insufficient material); this is what makes that argument falsifiable.

    A reset condition wired into ``win_state`` -- or a progress counter that
    accidentally displaced the turn counter the ply cap reads -- shows up
    here as the first ply whose adjudication moved, in a named game.
    """
    compared = 0
    for mode in PMV1_MODES:
        capture = load_pmv1_capture(mode)
        sequences = golden_action_sequences(mode)
        assert len(sequences) == len(capture.plies_per_game), (
            f"{mode}: corpus has {len(sequences)} games, capture has "
            f"{len(capture.plies_per_game)}"
        )

        with start_mode(mode):
            for game_index, actions in enumerate(sequences):
                game = Game()
                states: list[npt.NDArray[np.uint8]] = []
                for action in actions:
                    game.play_action(action)
                    states.append(win_state(game))
                observed = np.stack(states)
                expected = capture.game_win_state(game_index)
                assert observed.shape == expected.shape, (
                    f"{mode} game {game_index}: replayed {observed.shape[0]} "
                    f"plies, capture holds {expected.shape[0]}"
                )
                if not np.array_equal(observed, expected):
                    ply = int(np.argmax(np.any(observed != expected, axis=1)))
                    raise AssertionError(
                        f"{mode} game {game_index} ply {ply}: win_state is "
                        f"{observed[ply].tolist()} but the pre-change capture "
                        f"records {expected[ply].tolist()} "
                        "([white, black, draw]); plane 120 is perceptual and "
                        "must not reach the rules"
                    )
                compared += int(observed.shape[0])

    assert compared == GOLDEN_CORPUS_PLIES, (
        f"compared {compared} adjudications, expected {GOLDEN_CORPUS_PLIES}: "
        "the golden corpus is not the one the capture was taken over"
    )


def _draft_action(game: Game) -> int:
    """The lowest legal action, preferring one that CONTINUES the draft.

    Deterministic, so the walk below is the same every run.  Block C (終) is
    avoided while any other action exists: finishing the draft early would end
    the window this test measures.  ``cymovegen`` emits only 終 variants on a
    player's LAST hand piece, which is when the fallback fires -- and is how
    the draft eventually ends rather than looping forever.
    """
    mask = valid_mask(game)
    c_start, _ = action_encoding.SHU_BLOCK
    continuing = np.nonzero(mask[:c_start])[0]
    if continuing.size:
        return int(continuing[0])
    legal = np.nonzero(mask)[0]
    assert legal.size, "a drafting position always has at least one legal action"
    return int(legal[0])


def test_the_clock_is_zero_through_the_draft() -> None:
    """#18: in advanced mode, plane 120 is 0.0 at every drafting ply.

    Every draft move is an arata (``action_encoding``'s blocks B and C are the
    only draft moves), and an arata is a progress move, so the clock is reset
    on every single ply and can never leave 0 while both sides are still
    drafting.  A progress rule that stopped counting arata as progress would
    show up here as the clock ticking up during the draft.

    The loop runs while BOTH draft bits are set -- the window in which every
    ply is provably a draft move.  The ply on which the first side finishes is
    covered by the assertion after the loop.
    """
    game = load_game(ADVANCED_POSITION)
    board = board_of(game)
    assert board.draft_w and board.draft_b, "precondition: advanced mode drafts"

    drafted = 0
    while board.draft_w and board.draft_b:
        assert drafted < 200, "the draft never finished; the driver is stuck"
        planes = observation_planes(game)
        assert planes[PLANE_PROGRESS].max() == 0.0, f"draft ply {drafted}"
        assert game._progress_plies == 0, f"draft ply {drafted}"
        play_action_checked(game, _draft_action(game), drafted + 1)
        drafted += 1

    assert drafted >= 10, (
        f"only {drafted} plies of draft were observed, which is too few for "
        "this test to mean anything"
    )
    assert observation_planes(game)[PLANE_PROGRESS].max() == 0.0, (
        "the ply that ended one side's draft is itself an arata"
    )
