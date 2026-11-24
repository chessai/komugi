"""RT-17, RT-18 and the plane count -- the observation after the repair.

Two encoding consequences fall out of the key repair
(``docs/design/rules-audit.md``, "Encoding-plane consequences"):

* F7, plane 115.  v1 divides ``move_number`` (FULL moves) by a cap denominated
  in PLIES, so the plane reads about 0.5 when the game is actually at the cap
  and never reaches 1.0 before the draw fires.  RT-17 pins the repaired unit by
  the only property that matters to the value head: the plane hits 1.0 exactly
  at the ply where the turn-cap draw does.
* The scalar repetition plane 116 is deleted -- its values were computed over
  the aliased key, so they were not low-resolution but wrong -- and replaced by
  two monotone threshold planes R2 and R3.  That took the total 117 -> 118;
  KOM-002 then appended the last-move pair, the progress clock and the
  reserved block, taking it to 128.  R2/R3 keep their indices under that
  change -- the plane map is append-only.

Gated on KOM-027.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

if not hasattr(cyengine, "repetition_key"):
    pytest.skip("KOM-027 repair not yet implemented", allow_module_level=True)

from komugi.engine import observation  # noqa: E402
from komugi.engine.game import Game  # noqa: E402

from fixtures import (  # noqa: E402
    FOURFOLD,
    PLANE_R2,
    PLANE_R3,
    PLANE_TURN_CLOCK,
    PURE_SHUFFLE,
    REPAIRED_PLANE_COUNT,
    load_position_p,
    observation_planes,
    plane_count,
    play_sequence,
    repetition_count,
    win_state,
)

MAX_TURNS: int = Game.max_turns()


def _constant_plane(game: Game, plane: int) -> float:
    """Return the single value on *plane*, asserting it really is constant."""
    planes: npt.NDArray[np.float32] = observation_planes(game)
    board_plane = planes[plane]
    assert board_plane.shape == (9, 9)
    values = np.unique(board_plane)
    assert values.size == 1, (
        f"plane {plane} is specified as a constant 9x9 plane, got {values!r}"
    )
    return float(values[0])


def _at_ply(plies: int) -> Game:
    """A game on P whose ply counter reads *plies*.

    The turn cap is enforced against ``Game._turns`` (``gungi.py:260-262``),
    which is incremented once per ``play_action``.  Reaching the cap by play
    is not possible for a fixture that has to avoid repeating -- and would test
    the walk, not the plane -- so the counter is set directly.  That is also
    what makes this a real test of the contract: plane 115 must agree with the
    same counter the draw is adjudicated on, whatever it derives its value
    from.
    """
    game = load_position_p()
    game._turns = plies
    return game


# ---------------------------------------------------------------------------
# Plane count
# ---------------------------------------------------------------------------


def test_the_observation_declares_the_repaired_plane_count() -> None:
    """117 - 1 scalar repetition plane + 2 threshold planes = 118, then 128.

    The count is read from ``REPAIRED_PLANE_COUNT`` rather than spelled here,
    so this test pins the ENCODER against the fixtures' copy of the map; it is
    KOM-002's appended planes that took the same count from 118 to 128.
    """
    assert plane_count() == REPAIRED_PLANE_COUNT
    assert observation.observation_size() == (REPAIRED_PLANE_COUNT, 9, 9)
    assert Game.observation_size() == (REPAIRED_PLANE_COUNT, 9, 9)


def test_a_real_observation_has_the_declared_shape() -> None:
    """The declared count and the produced tensor do not drift apart."""
    planes = observation_planes(load_position_p())
    assert planes.shape == (REPAIRED_PLANE_COUNT, 9, 9)


# ---------------------------------------------------------------------------
# RT-17 -- turn-clock unit fix (F7)
# ---------------------------------------------------------------------------


def test_rt17_turn_clock_reaches_one_exactly_at_the_cap() -> None:
    """RT-17: plane 115 is 1.0 at the cap and below 1.0 one ply short of it."""
    assert _constant_plane(_at_ply(0), PLANE_TURN_CLOCK) == 0.0
    assert _constant_plane(_at_ply(MAX_TURNS - 1), PLANE_TURN_CLOCK) < 1.0
    assert _constant_plane(_at_ply(MAX_TURNS), PLANE_TURN_CLOCK) == 1.0


def test_rt17_the_plane_saturates_rather_than_overflowing() -> None:
    """``min(plies, MAX_TURNS) / MAX_TURNS`` -- clamped, never above 1.0."""
    assert _constant_plane(_at_ply(MAX_TURNS + 44), PLANE_TURN_CLOCK) == 1.0


def test_rt17_the_plane_is_denominated_in_plies_not_full_moves() -> None:
    """Half the ply budget reads 0.5.

    This is the 2x unit mismatch F7 describes, stated as a value: v1's
    ``move_number`` is a full-move count, so at half the ply budget it reads
    ~0.25 and at the cap ~0.5.  Reading 0.5 at half the budget is only possible
    on the ply scale.
    """
    assert _constant_plane(_at_ply(MAX_TURNS // 2), PLANE_TURN_CLOCK) == 0.5


def test_rt17_the_plane_and_the_draw_flip_at_the_same_ply() -> None:
    """The feature and the rule it encodes agree on where the cap is."""
    first_saturated: int | None = None
    first_drawn: int | None = None

    for plies in range(MAX_TURNS - 4, MAX_TURNS + 5):
        game = _at_ply(plies)
        if first_saturated is None and _constant_plane(game, PLANE_TURN_CLOCK) >= 1.0:
            first_saturated = plies
        if first_drawn is None and int(win_state(game)[2]) == 1:
            first_drawn = plies

    assert first_saturated == MAX_TURNS
    assert first_drawn == MAX_TURNS, "the turn-cap draw ply itself"


# ---------------------------------------------------------------------------
# RT-18 -- repetition-plane semantics
# ---------------------------------------------------------------------------


def test_rt18_no_repetition_plane_is_set_at_occurrence_one() -> None:
    """Occurrence 1 is the constant-true case and needs no plane."""
    game = load_position_p()

    assert repetition_count(game) == 1
    assert _constant_plane(game, PLANE_R2) == 0.0
    assert _constant_plane(game, PLANE_R3) == 0.0


def test_rt18_r2_and_r3_are_monotone_thresholds_over_the_rt01_cycle() -> None:
    """RT-18: R2 iff count >= 2, R3 iff count >= 3, on RT-01's genuine cycle.

    Thresholds rather than a one-hot over {1, 2, 3} (audit Q6): "at least
    twice" is the feature the value head actually needs, and a one-hot would
    force the net to re-learn the ordering.
    """
    game = load_position_p()
    observed: list[tuple[int, float, float]] = [
        (
            repetition_count(game),
            _constant_plane(game, PLANE_R2),
            _constant_plane(game, PLANE_R3),
        )
    ]

    for _ in range(3):
        play_sequence(game, PURE_SHUFFLE)
        observed.append(
            (
                repetition_count(game),
                _constant_plane(game, PLANE_R2),
                _constant_plane(game, PLANE_R3),
            )
        )

    assert [count for count, _, _ in observed] == [1, 2, 3, FOURFOLD], (
        "precondition: the cycle really does advance the count each time"
    )
    for count, r2, r3 in observed:
        assert r2 == float(count >= 2), f"R2 at occurrence {count}"
        assert r3 == float(count >= 3), f"R3 at occurrence {count}"


def test_rt18_the_repetition_planes_are_binary_not_scalar() -> None:
    """Every value either 0.0 or 1.0 -- the deleted plane 116 was a fraction.

    v1's plane was ``min(occurrences, 4) / 4``, so occurrence 3 read 0.75.  If
    the repair had kept a scalar under a new index this test would catch it.
    """
    game = load_position_p()
    for _ in range(3):
        play_sequence(game, PURE_SHUFFLE)
        for plane in (PLANE_R2, PLANE_R3):
            assert _constant_plane(game, plane) in (0.0, 1.0)


def test_rt18_the_planes_track_the_key_not_the_cells() -> None:
    """Scenario A's parity-flipped return sets no repetition plane.

    Same cells as the start position, opposite side to move: under the v1
    aliased key the old scalar plane would already be counting this as a
    repetition.  Under the repaired key it is a first occurrence, and both
    threshold planes stay clear.
    """
    from fixtures import SCENARIO_A_CYCLE

    game = load_position_p()
    play_sequence(game, SCENARIO_A_CYCLE)

    assert repetition_count(game) == 1
    assert _constant_plane(game, PLANE_R2) == 0.0
    assert _constant_plane(game, PLANE_R3) == 0.0
