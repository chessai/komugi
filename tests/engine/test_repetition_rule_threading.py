"""KOM-141: the repetition rule survives every way a ``Game`` is copied.

``clone()``, the mirror symmetry and ``pickle`` are the three places a Game is
rebuilt from its parts, and each one goes back through ``Game.__init__``. If any
of them dropped the rule, a placement-refereed game would either refuse to load
(a 243-byte history validated against the 274-byte full key) or -- worse, on the
mirror path, where the history is rebuilt -- carry on being refereed under the
OTHER rule with its counts intact. Both are silent at the copy and only visible
plies later as a draw that did or did not fire.

So each copy is taken MID-LINE and the line is FINISHED ON THE COPY: the
assertion is that the copy still draws at the ply the rule says it should, which
is a claim about the carried counter and not only about the carried flag. The
original is finished separately and must draw at the same ply -- a copy that
consumed its parent's counter would fail that half.

The scenario is the rules-audit's Scenario A, rebuilt here rather than imported:
``tests/rules/fixtures.py`` is only importable from ``tests/rules``, and a second
independent construction of the same 15 plies is worth more than a shared one
anyway.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine import action_encoding
from komugi.engine import game as game_module
from komugi.engine.cyboard import board_from_fen
from komugi.engine.game import Game, _mirror_action
from komugi.engine.repetition import (
    PLACEMENT_KEY_BYTES,
    RepetitionRule,
)

#: ``Game.__reduce__`` and ``_unpickle_game`` carry no annotations (they predate
#: this module and live under the engine's ``ignore_errors`` override), so a
#: strict module cannot call them directly. Reached through ``Any`` handles, the
#: same device ``tests/rules/fixtures.py`` uses for the untyped engine surface.
_UNPICKLE: Any = game_module._unpickle_game


def reduce_components(game: Game) -> Any:
    """Return the component tuple ``Game.__reduce__`` writes."""
    handle: Any = game
    _, components = handle.__reduce__()
    return components


#: rules-audit "Scenario A: side-to-move alias" -- advanced mode, both draft
#: bits clear, empty hands. White 帥(9-5) 侍(5-1); black 帥(1-5) 槍(1-1).
POSITION_P_FEN: str = "4m3n/9/9/9/8W/9/9/9/4M4 -/- w 3 - 1"


def _sq(rank: int, file: int) -> int:
    """1-based ``(rank, file)`` -> the 0..80 square id of the action encoding."""
    return (rank - 1) * 9 + (9 - file)


def _move(src: tuple[int, int], dst: tuple[int, int]) -> int:
    """Action id for a no-stack board move (block A0)."""
    return action_encoding.A0_START + _sq(*src) * 81 + _sq(*dst)


#: Plies 1-5 and 11-15: White's 3-move 帥 triangle against Black's 2-move 槍
#: shuffle. ``w = 3, b = 2``, so the cells return to P with the side to move
#: FLIPPED -- the alias the scenario exists to exhibit.
_CYCLE: tuple[int, ...] = (
    _move((9, 5), (8, 5)),
    _move((1, 1), (2, 1)),
    _move((8, 5), (8, 4)),
    _move((2, 1), (1, 1)),
    _move((8, 4), (9, 5)),
)

#: Plies 6-10: the colour-mirrored cycle, which flips the side to move back.
_MIRROR_CYCLE: tuple[int, ...] = (
    _move((1, 5), (2, 5)),
    _move((5, 1), (4, 1)),
    _move((2, 5), (2, 4)),
    _move((4, 1), (5, 1)),
    _move((2, 4), (1, 5)),
)

#: The full 15-ply line. Cells equal P at plies 0, 5, 10 and 15, so the
#: cells-only PLACEMENT key reaches four occurrences at ply 15 and the full key
#: never does.
SCENARIO_A_15PLY: tuple[int, ...] = _CYCLE + _MIRROR_CYCLE + _CYCLE

#: Where each copy is taken: ten plies in, three cells-only occurrences banked.
SPLIT: int = 10


def placement_game() -> Game:
    """Position P, refereed under gungi.js's cells-only key."""
    return Game(
        board_from_fen(POSITION_P_FEN), repetition_rule=RepetitionRule.PLACEMENT
    )


def play(game: Game, actions: tuple[int, ...]) -> None:
    """Play *actions* in order, asserting each is legal in the position first."""
    for ply, action in enumerate(actions, start=1):
        mask: npt.NDArray[np.uint8] = np.asarray(game.valid_moves(), dtype=np.uint8)
        assert mask[action] == 1, f"ply {ply}: action {action} is not legal"
        game.play_action(action)


def is_drawn(game: Game) -> bool:
    """True iff the engine has scored this position a draw."""
    return int(np.asarray(game.win_state(), dtype=np.uint8)[2]) == 1


def zero_policy() -> npt.NDArray[np.float32]:
    """An all-zero policy vector, the cheapest legal input to ``symmetries``."""
    return np.zeros(Game.action_size(), dtype=np.float32)


# ---------------------------------------------------------------------------
# The line itself, so the copy tests below cannot be vacuously green
# ---------------------------------------------------------------------------


def test_the_uncopied_line_draws_at_ply_15_under_placement_only() -> None:
    """The baseline both copy tests measure against.

    Without this, a bug that made EVERY placement game draw at ply 10 would
    leave the clone and pickle tests passing.
    """
    game = placement_game()
    play(game, SCENARIO_A_15PLY[:SPLIT])
    assert not is_drawn(game), "three occurrences is not four"
    assert game.repetition_count() == 3

    play(game, SCENARIO_A_15PLY[SPLIT:])
    assert is_drawn(game)
    assert game.repetition_count() == 4

    full = Game(board_from_fen(POSITION_P_FEN))
    play(full, SCENARIO_A_15PLY)
    assert not is_drawn(full), "the repaired key sees two states of two each"


# ---------------------------------------------------------------------------
# clone()
# ---------------------------------------------------------------------------


def test_clone_carries_the_rule_and_the_counter() -> None:
    """The clone finishes the line and draws; so does the original, separately."""
    game = placement_game()
    play(game, SCENARIO_A_15PLY[:SPLIT])

    copy = game.clone()
    assert copy.repetition_rule is RepetitionRule.PLACEMENT
    assert copy.repetition_count() == game.repetition_count() == 3
    assert copy._key_counts == game._key_counts
    assert copy._key_counts is not game._key_counts, "counters are never shared"
    assert all(len(entry) == PLACEMENT_KEY_BYTES for entry in copy._history)

    play(copy, SCENARIO_A_15PLY[SPLIT:])
    assert is_drawn(copy), "the clone reached the fourth occurrence"

    # The original was NOT advanced by the clone's five plies.
    assert not is_drawn(game)
    play(game, SCENARIO_A_15PLY[SPLIT:])
    assert is_drawn(game), "and gets there on its own"


def test_clone_of_a_default_game_stays_full() -> None:
    """The control: nothing about the default path moved."""
    game = Game(board_from_fen(POSITION_P_FEN))
    assert game.clone().repetition_rule is RepetitionRule.FULL


# ---------------------------------------------------------------------------
# symmetries() -- the file mirror
# ---------------------------------------------------------------------------


def test_the_mirror_symmetry_carries_the_rule_and_the_counter() -> None:
    """The reflected game is the same game, so it must draw at the same ply.

    The mirror is the one copy path that REBUILDS the history (every key is
    file-reflected), so it is the path where a 243-byte key had to be taught to
    mirror at all. The remaining plies are mirrored with the module's own
    ``_mirror_action`` -- the same permutation the policy vector gets -- and
    the reflected game must draw exactly where the original does.
    """
    game = placement_game()
    play(game, SCENARIO_A_15PLY[:SPLIT])

    symmetries: list[Any] = game.symmetries(zero_policy())
    assert len(symmetries) == 2, "identity and the left-right file mirror"
    identity: Game = symmetries[0][0]
    mirrored: Game = symmetries[1][0]

    assert identity.repetition_rule is RepetitionRule.PLACEMENT
    assert mirrored.repetition_rule is RepetitionRule.PLACEMENT
    assert mirrored.repetition_count() == 3, "a bijection on keys preserves counts"
    assert all(len(entry) == PLACEMENT_KEY_BYTES for entry in mirrored._history)
    assert len(mirrored._key_counts) == len(game._key_counts)

    play(mirrored, tuple(_mirror_action(a) for a in SCENARIO_A_15PLY[SPLIT:]))
    assert is_drawn(mirrored), "the reflected line reaches the same draw"


def test_the_mirror_of_a_default_game_stays_full() -> None:
    """The control, on the mirror path."""
    game = Game(board_from_fen(POSITION_P_FEN))
    mirrored: Game = game.symmetries(zero_policy())[1][0]
    assert mirrored.repetition_rule is RepetitionRule.FULL


# ---------------------------------------------------------------------------
# pickle
# ---------------------------------------------------------------------------


def test_a_pickle_round_trip_carries_the_rule_and_the_counter() -> None:
    """Through ``pickle`` -- the multiprocessing path a worker's games take."""
    game = placement_game()
    play(game, SCENARIO_A_15PLY[:SPLIT])

    revived: Game = pickle.loads(pickle.dumps(game))
    assert revived.repetition_rule is RepetitionRule.PLACEMENT
    assert revived._key_counts == game._key_counts
    assert revived._key == game._key
    assert revived.repetition_count() == 3

    play(revived, SCENARIO_A_15PLY[SPLIT:])
    assert is_drawn(revived)


def test_the_reduce_tuple_carries_the_rule_as_its_ninth_component() -> None:
    """The rule is a component, not an attribute restored after the fact."""
    game = placement_game()
    components = reduce_components(game)

    assert len(components) == 9
    assert components[8] is RepetitionRule.PLACEMENT


def test_an_eight_component_pickle_is_refused_by_name() -> None:
    """No silent default: a pre-KOM-141 tuple raises and names the ticket.

    The same policy ``_unpickle_game`` already applies to the KOM-002
    components, and for the same reason -- there is nothing to reconstruct the
    rule from, so loading such a pickle would re-referee the game under the
    other key. There are no persisted game pickles in this repo, so nothing is
    stranded by the refusal.
    """
    game = placement_game()
    play(game, SCENARIO_A_15PLY[:2])
    components = reduce_components(game)

    with pytest.raises(ValueError, match="KOM-141") as raised:
        _UNPICKLE(*components[:8])
    assert "_repetition_rule" in str(raised.value)

    # The control: at the full arity it loads, under the rule it was written
    # with.
    revived: Game = _UNPICKLE(*components)
    assert revived.repetition_rule is RepetitionRule.PLACEMENT
