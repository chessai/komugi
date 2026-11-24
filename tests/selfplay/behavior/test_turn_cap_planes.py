"""KOM-165: the turn clock and the turn cap are the same number, at the cap.

One leaf (``observation.MAX_TURNS``) is claimed to be *both* the ply the engine
adjudicates a draw at *and* the denominator of observation plane 115.  That is
the whole reason the cap change is a plane-map bump rather than a constant edit,
and it is exactly the claim that a test which reads either side alone cannot
falsify.  So this module pins them together, on a REAL game, at the two plies
where a disagreement would show:

* one ply below the cap -- plane 115 reads ``(cap - 1) / cap`` and the game is
  **not** over.  A clip-instead-of-rebase implementation (the alternative
  KOM-165 §2.3 rejects) reads 1.0 here, and an off-by-one in the cap predicate
  adjudicates here;
* at the cap -- plane 115 reads exactly 1.0 and the engine reports a turn-cap
  draw.  "Exactly" is the point: the plane reaches its ceiling on the same ply
  the draw fires, which is what makes plane 115 a usable clock for the net
  rather than a scaled ply counter that happens to be bounded.

The technique is ``worker_rig.capped_draw_game``'s (``worker_rig.py:515-526``):
play a handful of real moves, then set ``Game._turns`` directly.  Walking to the
cap would be hundreds of thousands of legal-move generations spent testing the
walk instead of the adjudication, and the cap is enforced against ``_turns`` and
nothing else (``game.py``'s ``win_state``), so a game whose counter reads the cap
*is* a turn-cap draw to every rule in the engine.

Plane 120 is checked at both plies for its own invariant -- it shares plane
115's denominator, so ``120 <= 115`` always (``observation.py``'s docstring) --
because a separate progress denominator is precisely the kind of second cap the
rebase could have left behind.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

import worker_rig as rig

from komugi.engine import observation
from komugi.engine.game import TERMINATION_NONE, TERMINATION_TURN_CAP, Game

PLANE_TURN_CLOCK: Final[int] = 115
"""Plane 115, spelled out so no assertion below hides behind a name."""

PLANE_PROGRESS: Final[int] = 120
"""Plane 120, likewise."""

_SEED: Final[int] = 20260911
"""Fixed, so a failure replays the same eight opening plies."""

_OPENING_PLIES: Final[int] = 8
"""Real moves played before the counter is set: enough that the board, the
repetition keys and the progress clock are a real position rather than the
start position with a doctored counter."""


def _game_at(turns: int) -> Game:
    """A real game whose ply counter reads *turns*.

    ``random_playout`` plays the opening for real; the assignment is the rig's
    own fixture technique.  ``_progress_plies`` is left where play put it (at
    most :data:`_OPENING_PLIES`), which is what makes the ``120 <= 115``
    assertion below a real comparison rather than ``0 <= x``.
    """
    game, _ = rig.random_playout(_SEED, max_plies=_OPENING_PLIES)
    game._turns = turns
    return game


def _constant(obs: npt.NDArray[np.float32], plane: int) -> float:
    """The value of a constant plane, asserting that it IS constant."""
    values = np.unique(obs[plane])
    assert values.size == 1, f"plane {plane} is not constant: {values.tolist()}"
    return float(values[0])


def test_one_ply_below_the_cap_the_clock_is_short_and_the_game_is_live() -> None:
    """At ``cap - 1``: plane 115 is ``(cap - 1) / cap`` and nothing is decided."""
    cap = observation.MAX_TURNS
    assert Game.max_turns() == cap, "the cap and the plane denominator diverged"

    game = _game_at(cap - 1)
    obs = np.asarray(game.observation(), dtype=np.float32)

    # Exact, not approximate: both sides are dyadic rationals in float32.
    assert _constant(obs, PLANE_TURN_CLOCK) == (cap - 1) / cap

    assert not np.asarray(game.win_state()).any(), (
        "a game one ply below the cap is adjudicated; the cap predicate is "
        "``_turns >= max_turns()`` and must not fire at cap - 1"
    )
    assert game.termination_reason() == TERMINATION_NONE

    assert _constant(obs, PLANE_PROGRESS) <= _constant(obs, PLANE_TURN_CLOCK), (
        "plane 120 exceeds plane 115, so the two clocks no longer share a "
        "denominator -- the progress plane has a cap of its own"
    )


def test_at_the_cap_the_clock_reads_one_and_the_draw_fires() -> None:
    """At the cap: plane 115 is exactly 1.0 and the engine says ``turn_cap``."""
    cap = observation.MAX_TURNS
    game = _game_at(cap)
    obs = np.asarray(game.observation(), dtype=np.float32)

    assert _constant(obs, PLANE_TURN_CLOCK) == 1.0, (
        "plane 115 must reach its ceiling on the same ply the draw is "
        "adjudicated (rules-audit F7), which is what the shared leaf buys"
    )
    assert game.termination_reason() == TERMINATION_TURN_CAP
    assert np.asarray(game.win_state()).tolist() == [0, 0, 1]

    assert _constant(obs, PLANE_PROGRESS) <= _constant(obs, PLANE_TURN_CLOCK)


def test_the_clock_is_not_clipped_below_the_cap() -> None:
    """Plane 115 is strictly increasing across the old cap, not pinned at 1.0.

    The rejected alternative (KOM-165 §2.3) keeps the ``/256`` denominator and
    clips: under it every ply from 256 on reads 1.0 and the net cannot see the
    second half of a long game at all.  Three plies spanning the old cap are
    enough to falsify that, and they are the region the cap change exists to
    expose.
    """
    cap = observation.MAX_TURNS
    readings: list[float] = []
    for turns in (cap // 2 - 1, cap // 2, cap - 1):
        obs = np.asarray(_game_at(turns).observation(), dtype=np.float32)
        readings.append(_constant(obs, PLANE_TURN_CLOCK))
        assert readings[-1] == turns / cap

    assert readings[0] < readings[1] < readings[2] < 1.0, (
        f"the turn clock is clipped or flat across the old cap: {readings}"
    )
