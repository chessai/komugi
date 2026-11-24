"""KOM-142: ``Game.termination_reason`` -- one test per reason, plus the pin.

Every reason is reached through the ENGINE, never by writing a label: a
position is loaded (or played into) and the method is asked why it ended.  The
positions are the ones the suite already trusts --

* checkmate: ``tests/rules/fixtures.MATE_AND_REPETITION_FEN``, verified against
  the live engine by ``tests/rules/test_fixtures_sanity.py``;
* stalemate / insufficient material / marshal captured: the crafted terminals
  of ``tests/engine/test_cyengine.py`` (``TestCraftedStalemate``,
  ``TestCraftedInsufficientMaterial``, ``TestMarshalCaptured``);
* fourfold: the 12-ply general shuttle of
  ``test_cyengine.TestCraftedFourfoldRepetition``, replayed here as real
  ``Game.play_action`` moves so the occurrence counter is the engine's own.

They are restated as literals rather than imported.  ``tests/rules/fixtures``
is importable only from a ``tests/rules`` test (the tree has no packages, and
pytest puts a test's own directory on ``sys.path``), and ``tests/engine``
collects first, so a bare ``from fixtures import ...`` here would resolve or
not depending on collection order.

NON-VACUITY.  ``Game`` spells terminality as ``win_state()`` -- there is no
``Game.is_game_over``; the board-level predicate is ``cyengine.is_game_over``,
which by construction cannot see the env's turn cap.  So every terminal case
below asserts BOTH that ``win_state()`` has a slot set and that the reason is
not ``"none"``, and the six board-level cases additionally assert
``cyengine.is_game_over`` agrees.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from komugi.engine import action_encoding, cyengine, observation
from komugi.engine.cyboard import Board, board_from_fen
from komugi.engine.game import (
    TERMINATION_CHECKMATE,
    TERMINATION_FOURFOLD_REPETITION,
    TERMINATION_INSUFFICIENT_MATERIAL,
    TERMINATION_MARSHAL_CAPTURED,
    TERMINATION_NONE,
    TERMINATION_REASONS,
    TERMINATION_STALEMATE,
    TERMINATION_TURN_CAP,
    Game,
)

# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

#: Black to move and mated (``tests/rules/fixtures.MATE_AND_REPETITION_FEN``).
MATE_FEN: Final[str] = "8m/7D1/6I1G/9/9/9/9/9/4M4 -/- b 1 - 1"

#: Black to move, not in check, and every move walks into one.
STALEMATE_FEN: Final[str] = "8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1"

#: Two non-adjacent 帥 and nothing else, both hands empty.
INSUFFICIENT_FEN: Final[str] = "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1"

#: A real position from a v1 playout in which a 帥 has been captured.
MARSHAL_CAPTURED_FEN: Final[str] = (
    "1|g:N|2|W:N|Ad1f/7r1/1nd2Adfr/2|c:G|j2K2/6s1D/1w|W:T|6"
    "/2F4J|F:D|/i8/2|S:w||R:M|3C1 -/- b 3 - 164"
)

#: 帥 + 大 a side, nothing else: the shuttle position whose 4-ply cycle repeats
#: the whole repetition key (cells, side to move and the two empty hands).
SHUTTLE_FEN: Final[str] = "4mg3/9/9/9/9/9/9/9/4MG3 -/- w 3 - 1"

#: An empty board with both draft bits set and both hands full: the draft
#: position of ``test_cyengine.TestDraftPhase``.  A drafting position is never
#: terminal, however the board reads.
DRAFT_FEN: Final[str] = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
)

Square = tuple[int, int]


# ---------------------------------------------------------------------------
# Action construction (``tests/rules/fixtures``' spelling)
# ---------------------------------------------------------------------------


def sq_index(square: Square) -> int:
    """1-based ``(rank, file)`` -> the 0..80 square id of the encoding."""
    rank, file = square
    if not (1 <= rank <= 9 and 1 <= file <= 9):
        raise ValueError(f"square out of range: {square!r}")
    return (rank - 1) * 9 + (9 - file)


def move_action(src: Square, dst: Square) -> int:
    """Action id for a NO-STACK board move ``src`` -> ``dst`` (block A0)."""
    return action_encoding.A0_START + sq_index(src) * 81 + sq_index(dst)


#: The 4-ply shuttle: both 大 step out and back, so ply 4 restores every
#: component of the repetition key.  Occurrences of the start position land on
#: plies 0, 4, 8 and 12 -- the 4th is the draw.
SHUTTLE_CYCLE: Final[tuple[int, ...]] = (
    move_action((9, 4), (8, 4)),   # W  大(9-4)(8-4)
    move_action((1, 4), (2, 4)),   # B  大(1-4)(2-4)
    move_action((8, 4), (9, 4)),   # W  大(8-4)(9-4)
    move_action((2, 4), (1, 4)),   # B  大(2-4)(1-4)
)

#: Plies until the shuttle's 4th occurrence of the start key.
FOURFOLD_PLIES: Final[int] = 12

#: The short cap :class:`_CappedGame` adjudicates on -- two whole shuttle
#: cycles, which is the 3rd occurrence of the start key and so NOT yet a
#: fourfold: the cap must be the only reason available at that ply.
SHORT_CAP: Final[int] = 8


class _CappedGame(Game):
    """A ``Game`` whose turn cap is :data:`SHORT_CAP` plies, not the real one.

    ``termination_reason`` and ``win_state`` both read the cap through
    ``self.max_turns()``, so overriding it exercises the real rule at a ply
    count a test can actually reach: the true cap is hundreds of plies, and no
    hand-scripted shuttle survives that long without repeating a position four
    times first.  ``test_the_real_cap_is_the_observation_leaf`` pins the
    production value so this subclass cannot quietly become the thing under
    test.
    """

    @staticmethod
    def max_turns() -> int:
        return SHORT_CAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load(fen: str) -> Game:
    """A fresh ``Game`` on *fen*, its history seeded with that position."""
    return Game(board_from_fen(fen))


def board_of(game: Game) -> Board:
    """*game*'s live ``Board`` (``Game`` exposes no public accessor)."""
    board: Board = game._board
    return board


def win_state(game: Game) -> npt.NDArray[np.uint8]:
    """``game.win_state()`` as a typed ``[white, black, draw]`` vector."""
    return np.asarray(game.win_state(), dtype=np.uint8)


def play_checked(game: Game, action: int, ply: int) -> None:
    """Play *action*, asserting it is legal first (*ply* names it on failure)."""
    mask = np.asarray(game.valid_moves(), dtype=np.uint8)
    assert mask[action] == 1, (
        f"ply {ply}: action {action} is not legal "
        f"(side to move: {'wb'[board_of(game).turn]}, "
        f"legal action ids: {int(mask.sum())})"
    )
    game.play_action(action)


def assert_terminal(game: Game, reason: str) -> None:
    """Assert *game* really is over and reports *reason*.

    The non-vacuity clause of every test below: a reason is only worth
    checking once the position is terminal at all, and ``"none"`` must never
    be the answer for a terminal one.
    """
    assert game.termination_reason() == reason
    assert game.termination_reason() != TERMINATION_NONE
    assert int(win_state(game).sum()) == 1, (
        f"{reason}: win_state {win_state(game)!r} has no single slot set"
    )


def assert_board_terminal(game: Game, reason: str) -> None:
    """:func:`assert_terminal` plus the board-level gate's own agreement."""
    assert_terminal(game, reason)
    assert cyengine.is_game_over(board_of(game), game.repetition_count()) is True


# ---------------------------------------------------------------------------
# The seven reasons
# ---------------------------------------------------------------------------


def test_the_reason_vocabulary_is_the_seven_labels() -> None:
    """``TERMINATION_REASONS`` is exactly the seven strings, no duplicates."""
    assert TERMINATION_REASONS == (
        "none",
        "marshal_captured",
        "checkmate",
        "stalemate",
        "fourfold_repetition",
        "insufficient_material",
        "turn_cap",
    )
    assert len(set(TERMINATION_REASONS)) == len(TERMINATION_REASONS)


def test_none_while_the_game_is_still_running() -> None:
    """A live position -- and the draft, which has no terminal at all."""
    drafting = load(DRAFT_FEN)
    assert drafting.in_draft(0) and drafting.in_draft(1)
    assert drafting.termination_reason() == TERMINATION_NONE
    assert int(win_state(drafting).sum()) == 0

    running = load(SHUTTLE_FEN)
    assert running.termination_reason() == TERMINATION_NONE
    assert int(win_state(running).sum()) == 0


def test_checkmate() -> None:
    """Black is mated, so White wins and the reason names the mate."""
    game = load(MATE_FEN)
    assert_board_terminal(game, TERMINATION_CHECKMATE)
    assert win_state(game).tolist() == [1, 0, 0]


def test_stalemate() -> None:
    """Black has no move and is not in check -- a draw, reported as such."""
    game = load(STALEMATE_FEN)
    assert_board_terminal(game, TERMINATION_STALEMATE)
    assert win_state(game).tolist() == [0, 0, 1]


def test_insufficient_material() -> None:
    """Two non-adjacent 帥 and nothing else."""
    game = load(INSUFFICIENT_FEN)
    assert_board_terminal(game, TERMINATION_INSUFFICIENT_MATERIAL)
    assert win_state(game).tolist() == [0, 0, 1]


def test_marshal_captured() -> None:
    """A 帥 is off the board: a win for the side that took it, not a draw."""
    game = load(MARSHAL_CAPTURED_FEN)
    assert_board_terminal(game, TERMINATION_MARSHAL_CAPTURED)
    assert int(win_state(game)[2]) == 0


def test_fourfold_repetition_by_playing_the_shuttle() -> None:
    """Twelve real moves to the 4th occurrence, and not one ply sooner.

    The intermediate assertion is the point: ``termination_reason`` must stay
    ``"none"`` for plies 1-11, including plies 4 and 8 where the position has
    already recurred twice and three times.
    """
    game = load(SHUTTLE_FEN)
    for ply in range(1, FOURFOLD_PLIES + 1):
        play_checked(game, SHUTTLE_CYCLE[(ply - 1) % len(SHUTTLE_CYCLE)], ply)
        if ply < FOURFOLD_PLIES:
            assert game.termination_reason() == TERMINATION_NONE, (
                f"ply {ply}: the game ended early "
                f"(repetition count {game.repetition_count()})"
            )

    assert game.repetition_count() == 4
    assert_board_terminal(game, TERMINATION_FOURFOLD_REPETITION)
    assert win_state(game).tolist() == [0, 0, 1]


def test_turn_cap() -> None:
    """The cap fires on ply count alone, with nothing else terminal."""
    game = _CappedGame(board_from_fen(SHUTTLE_FEN))
    for ply in range(1, SHORT_CAP + 1):
        play_checked(game, SHUTTLE_CYCLE[(ply - 1) % len(SHUTTLE_CYCLE)], ply)

    assert game.turns == SHORT_CAP
    # Nothing on the BOARD is terminal here -- the cap is the whole reason.
    assert game.repetition_count() == 3
    assert cyengine.is_game_over(board_of(game), game.repetition_count()) is False
    assert_terminal(game, TERMINATION_TURN_CAP)
    assert win_state(game).tolist() == [0, 0, 1]


def test_the_real_cap_is_the_observation_leaf() -> None:
    """The production cap, which :class:`_CappedGame` stands in for.

    Pinned as a DERIVATION, not as a number: ``observation.MAX_TURNS`` is the
    single source of the cap (KOM-165), and it is also the denominator of
    observation planes 115 and 120, so a cap that disagreed with the leaf would
    put the turn-clock plane out of step with the ply the draw fires at.
    """
    assert Game.max_turns() == observation.MAX_TURNS


# ---------------------------------------------------------------------------
# The precedence pin (rules-audit F8)
# ---------------------------------------------------------------------------


def test_checkmate_outranks_a_fourth_occurrence() -> None:
    """A position that is BOTH a checkmate and a 4th occurrence: ``checkmate``.

    ``win_state`` tests checkmate before repetition while the gate it sits
    behind, ``cyengine.is_game_over``, tests repetition first -- rules-audit
    F8, pinned by ``tests/rules/test_repetition_rule.py`` RT-19 as a WIN.
    ``termination_reason`` re-asks ``win_state``'s order, so it must say
    ``checkmate`` here and the score must be 1-0.

    Reached the only way it can be: per-ply adjudication would have ended the
    game at the FIRST occurrence of a mated position, so the history is handed
    in rather than played (the bridge shape ``Game.__init__`` validates).
    """
    board = board_from_fen(MATE_FEN)
    key = cyengine.repetition_key(board)
    game = Game(board, [key] * 4)

    assert game.repetition_count() == 4
    assert cyengine.is_repetition_draw(game.repetition_count()) is True
    assert cyengine.is_checkmate(board_of(game)) is True

    assert_board_terminal(game, TERMINATION_CHECKMATE)
    assert win_state(game).tolist() == [1, 0, 0]
