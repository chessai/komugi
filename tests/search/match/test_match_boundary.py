"""A1 and A6: the seams the match search implements, and the one it does not.

A1 is satisfiable only because of the design's B5 resolution. ``Mover.pick``
delivers a ``komugi.selfplay.adapter.GameState``, and ``pick``'s parameter type
must be a *supertype* of whatever a caller passes (parameter types are
contravariant), so it cannot be ``MatchState``: that is the TREE's contract, a
narrower thing than the input seam, and a mover handing over anything that is
not one would then be a type error at the caller rather than at the adapter.
(Since KOM-173 ``GameState`` does carry a ``terminal_wdl`` of its own -- the
searched-WDL accumulator's terminal source -- so the two protocols now happen to
overlap on that method. The contravariance argument is unaffected: it is about
which contract ``pick`` may *demand*, not about which one one particular state
satisfies.) Annotating ``pick`` with ``GameState`` would require the import A8
bans. The structural ``MoverState`` satisfies both constraints at once, and the
``mover: Mover = MatchPlayer(...)`` line below is what makes ``mypy --strict``
check it -- the assertion is the annotation, not the call.

A6 pins a claim that is load-bearing for the design's seam argument and would
otherwise be an assertion nobody re-checks: ``protocols.SearchAlgorithm`` is a
**dead** seam. No search in this tree satisfies it, so implementing a new
search against it would produce a match player no existing caller could drive
and would ossify a signature nothing else honours.
"""

from __future__ import annotations

import inspect

import numpy as np
import numpy.typing as npt

# `planted` first, always: importing it is what puts ``tests/search/oracle`` on
# ``sys.path``, and ``toygames`` lives there. See ``planted/__init__.py``.
from planted import wdl_from_value
from toygames import BOARD_SIZE, TicTacToe

from komugi.eval.v1bridge import Mover
from komugi.protocols import SearchAlgorithm
from komugi.search.gumbel import GumbelSearch
from komugi.search.match import MatchConfig, MatchPlayer
from komugi.search.match.adapter import MatchGameState, as_match_state
from komugi.search.match.state import MatchState, MoverState, WdlEval

#: A draw score that would be visible if it leaked: not 0, not 1, and not a
#: number any correct WDL triple contains.
LOUD_DRAW_SCORE = 0.7


class MoverTicTacToe:
    """Tic-tac-toe with exactly the public surface ``Mover.pick`` delivers.

    The same seven methods ``GameState`` exposes to a match player, plus the
    draw score that ``GameState`` also carries -- so that "the wrapper never
    touches it" is a checkable claim rather than a property of a fixture that
    never had one.
    """

    __slots__ = ("_game", "draw_score")

    def __init__(self, game: TicTacToe, draw_score: float = LOUD_DRAW_SCORE) -> None:
        self._game = game
        self.draw_score = draw_score

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the game's legality mask."""
        return self._game.legal_mask()

    def play(self, action: int) -> None:
        """Advance the game in place."""
        self._game.play(action)

    def clone(self) -> MoverTicTacToe:
        """Return an independent copy carrying the same draw score."""
        return MoverTicTacToe(self._game.clone(), self.draw_score)

    def is_terminal(self) -> bool:
        """Return whether the game is over."""
        return self._game.is_terminal()

    def to_play(self) -> int:
        """Return the index of the player to move."""
        return self._game.to_play()

    def observation(self) -> npt.NDArray[np.float32]:
        """Return a placeholder encoding; this suite has no network."""
        return np.zeros(1, dtype=np.float32)

    def absolute_outcome(self) -> npt.NDArray[np.uint8]:
        """Return the raw ``(3,)`` ``[white, black, draw]`` outcome.

        All-zero before the game is decided -- "no result yet", not a draw --
        exactly as the engine's own reports it.
        """
        outcome: npt.NDArray[np.uint8] = np.zeros(3, dtype=np.uint8)
        if not self._game.is_terminal():
            return outcome
        winner = self._game.winner()
        if winner is None:
            outcome[2] = 1
        else:
            outcome[winner] = 1
        return outcome


def flat_evaluator(state: MatchState) -> WdlEval:
    """A blandly non-committal evaluator: uniform prior, every position drawn.

    The seam is what is under test here, not the values, so the values say
    nothing at all.
    """
    return WdlEval(np.zeros(BOARD_SIZE, dtype=np.float32), wdl_from_value(0.0))


def test_match_player_satisfies_the_mover_seam() -> None:
    """A1: the annotation is the assertion; the calls prove it also runs.

    ``mypy --strict`` checks the binding below, which is where the structural
    match actually gets verified. The runtime half exercises all four methods
    so that a player which type-checks but raises on ``observe`` or ``close``
    cannot pass.

    ``pick`` is exercised through the concrete name and not through ``mover``,
    which is the B5 seam showing through: ``Mover.pick`` is declared over
    ``komugi.selfplay.adapter.GameState``, so a call through the ``Mover``-typed
    name is only type-correct with a real ``GameState`` -- an object this
    package must not import and this suite has no engine to build. What the
    seam actually requires is the *contravariant* half, and that is what
    ``MatchPlayer.pick``'s ``MoverState`` annotation and the binding below
    assert together: the fixture is accepted by the wider parameter type, and
    the narrower one the protocol declares is a subtype of it (pinned by
    :func:`test_mover_state_is_a_supertype_of_what_the_seam_delivers`).
    """
    player = MatchPlayer(
        MatchConfig(num_actions=BOARD_SIZE, playouts=8), flat_evaluator
    )
    # THE assertion of A1: `mypy --strict` verifies the structural match here.
    mover: Mover = player

    mover.reset(pair=0, seat=0)
    state = MoverTicTacToe(TicTacToe())
    action = player.pick(state, ply=0, seed=1234)
    assert bool(state.legal_mask()[action])
    # `observe` is called on the concrete player: the seam's `record` is a
    # decoded `MoveRecord`, which a match tree has no use for and this suite
    # has no engine to build.
    player.observe(action, None)
    mover.close()


def test_the_wrapped_state_is_a_match_state() -> None:
    """A1: the converse direction of the seam, checked statically and live."""
    wrapped: MatchState = MatchGameState(MoverTicTacToe(TicTacToe()))
    assert wrapped.to_play() == 0
    assert not wrapped.is_terminal()
    assert int(wrapped.legal_mask().sum()) == BOARD_SIZE
    assert isinstance(wrapped.clone(), MatchGameState)
    assert as_match_state(MoverTicTacToe(TicTacToe())).to_play() == 0


def test_a_drawn_position_yields_a_one_hot_with_no_draw_score_in_it() -> None:
    """A1: ``terminal_wdl`` is derived from the outcome, never from contempt.

    This is the method the whole storage invariant rests on. The state handed
    in carries a loud draw score, and the triple that comes back does not
    contain it anywhere: a draw is ``(0, 1, 0)``, and what a draw is *worth*
    is a question for the reader at read time.
    """
    drawn = TicTacToe.from_string("XXO/OOX/XOX")
    assert drawn.is_terminal() and drawn.winner() is None
    state = MoverTicTacToe(drawn, draw_score=LOUD_DRAW_SCORE)
    triple = MatchGameState(state).terminal_wdl()
    assert triple.dtype == np.float64
    np.testing.assert_array_equal(triple, np.array([0.0, 1.0, 0.0]))
    assert LOUD_DRAW_SCORE not in set(triple.tolist())


def test_a_decided_position_is_one_hot_for_the_mover_here() -> None:
    """A1: the win slot is the mover's and the loss slot is the opponent's."""
    lost = TicTacToe.from_string("XXX/OO./...", to_play=1)
    assert lost.is_terminal() and lost.winner() == 0
    triple = MatchGameState(MoverTicTacToe(lost)).terminal_wdl()
    np.testing.assert_array_equal(triple, np.array([0.0, 0.0, 1.0]))


def test_mover_state_is_a_supertype_of_what_the_seam_delivers() -> None:
    """A1: the contravariance that makes ``pick``'s annotation sound."""
    delivered: MoverState = MoverTicTacToe(TicTacToe())
    assert delivered.clone().to_play() == 0


def test_gumbel_search_does_not_satisfy_the_dead_seam() -> None:
    """A6: ``SearchAlgorithm`` is dead, by signature inspection.

    ``GumbelSearch.search`` takes a ``SearchState`` and returns a
    ``SearchResult``; the protocol declares an ``Observation`` in and a
    ``(Policy, float)`` tuple out. Neither the parameter nor the return type
    matches, and a structural check that only asked "is there a method called
    ``search``" would have said yes to both.
    """
    protocol = inspect.signature(SearchAlgorithm.search)
    concrete = inspect.signature(GumbelSearch.search)
    assert list(protocol.parameters) == ["self", "obs"]
    assert list(concrete.parameters) == ["self", "state"]
    assert list(protocol.parameters) != list(concrete.parameters)
    assert str(protocol.return_annotation) != str(concrete.return_annotation)
    assert "SearchResult" in str(concrete.return_annotation)
