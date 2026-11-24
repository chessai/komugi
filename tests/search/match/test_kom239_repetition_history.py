"""The match search sees a repetition it is given the history for (KOM-239).

The defect: the Python match server rebuilt every position from a
lone FEN, so the ``Game`` it searched had an empty ``_history``, every
position's ``repetition_count()`` was 1, and no move could ever be the fourth
occurrence of anything -- a won game could be handed away because the search
could not perceive the drawing move as a draw.

The fix is upstream of the search -- replay the SAN history and search the
``Game`` that replay produces -- and this file is the evidence that the search
does the rest by itself:

* ``GameState.clone`` clones the ``Game``, and ``Game.clone`` copies
  ``_history`` and ``_key_counts``, so a child carries its parent's keys;
* ``GameState.is_terminal`` is ``Game.win_state().any()``, and ``win_state``
  adjudicates ``cyengine.is_repetition_draw``, so a child that completes the
  repetition *is* terminal;
* ``MatchGameState.terminal_wdl`` turns that into ``(0, 1, 0)`` -- a draw, and
  with the match default contempt a utility of 0.

Nothing in :mod:`komugi.search.match` needed changing, and
:func:`test_the_match_state_carries_the_history_across_a_clone` is what keeps
that true: the seam it pins is the one the whole fix rests on.

The fixture
-----------
``fixtures/kom239_repetition_tail.txt`` is a SAN history from
:data:`komugi.engine.fen.ADVANCED_POSITION` -- the wire shape
the Python match server now accepts -- whose last position is one ply from a fourfold
repetition. It is **generated**, not recorded: ``fixtures/make_kom239_fixture.py``
documents why and regenerates it.

The searches run at :data:`PLAYOUTS` playouts on a stub evaluator rather than
at a much larger budget on a checkpoint, for the reason KOM-233's file gives:
what differs between the two searches here is the *tree*, and a network would
only make the same point slower and less reproducibly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

# Before the engine is imported: ``Game`` reads ``GUNGI_MODE`` for its start
# mode, and the fixture replays from the advanced start.
os.environ.setdefault("GUNGI_MODE", "advanced")

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pytest  # noqa: E402

from komugi.engine.action_encoding import encode_move  # noqa: E402
from komugi.engine.api import Gungi  # noqa: E402
from komugi.engine.cyboard import board_from_fen  # noqa: E402
from komugi.engine.fen import ADVANCED_POSITION  # noqa: E402
from komugi.engine.game import (  # noqa: E402
    TERMINATION_FOURFOLD_REPETITION,
    Game,
)
from komugi.engine.repetition import RepetitionRule  # noqa: E402
from komugi.search.match.adapter import as_match_state  # noqa: E402
from komugi.search.match.config import MatchConfig  # noqa: E402
from komugi.search.match.search import MatchResult, MatchSearch  # noqa: E402
from komugi.search.match.state import MatchState, WdlEval  # noqa: E402
from komugi.selfplay.adapter import GameState  # noqa: E402

#: The engine's action-space width, and :class:`MatchConfig`'s own default.
NUM_ACTIONS: Final[int] = 37341

#: The playout budget both halves of the differential run at. 200, which is
#: where the differential was measured: a production-scale budget of 1600
#: would take minutes per search here and says nothing this does not. The
#: claim is a claim about this budget and the file states it rather than
#: implying it.
PLAYOUTS: Final[int] = 200

#: The root view the Python match server builds a request with, so the position
#: searched here is the position the anchor server would search.
MIN_PLACEMENTS: Final[int] = 10

#: What the stub adds to the repeating move's logit. Modest on purpose: a
#: +10 point mass (KOM-233's ``BOOST``) is a prior no 200-playout search can
#: leave at all -- every playout descends the one edge and the alternatives
#: are never visited, so a search that *can* see the draw still has nothing to
#: choose. ``+2`` is a prior that prefers the shuffle about sevenfold, which is
#: enough to decide a tie and not enough to survive a terminal.
BOOST: Final[float] = 2.0

#: What the stub subtracts from every move outside the two it offers. The
#: reduction the differential needs: a policy spread over ~300 legal moves
#: gives every one of them a couple of visits at this budget and decides
#: nothing, so the stub narrows the position to "the shuffle, or one move that
#: plays on" and lets the tree pick between them.
SUPPRESS: Final[float] = 30.0

#: The stub's WDL for the seat it declares winning, and for the other seat.
WINNING_WDL: Final[tuple[float, float, float]] = (0.9, 0.05, 0.05)
LOSING_WDL: Final[tuple[float, float, float]] = (0.05, 0.05, 0.9)

#: The utility a search must keep on a non-repeating line for the differential
#: to mean anything: the mover really is winning, so taking the draw is a loss
#: of exactly that much.
WINNING_UTILITY: Final[float] = 0.5

#: A drawn terminal, as ``terminal_wdl`` states it.
DRAW_WDL: Final[tuple[float, float, float]] = (0.0, 1.0, 0.0)

#: The rule the site adjudicates under, and the one the Python match server
#: now replays under (KOM-141, KOM-239 F1): gungi.js's repetition key is the cells
#: and nothing else. Komugi's default ``RepetitionRule.FULL`` adds the side to
#: move, the drafting rights and the hands, which makes it a strictly harder
#: test.
RULE: Final[RepetitionRule] = RepetitionRule.PLACEMENT

_FIXTURES: Final[Path] = Path(__file__).resolve().parent / "fixtures"

_FIXTURE: Final[Path] = _FIXTURES / "kom239_repetition_tail.txt"

#: The fixture's header key for the seat to move at the search root.
_SEAT_MARK: Final[str] = "# seat:"

#: Seat letters, as a FEN and the fixture spell them, to engine seat indices.
_SEATS: Final[dict[str, int]] = {"w": 0, "b": 1}


# ---------------------------------------------------------------------------
# The fixture, and the two roots it produces
# ---------------------------------------------------------------------------


def load_fixture(path: Path = _FIXTURE) -> tuple[list[str], int]:
    """Return ``(sans, seat)`` -- `path`'s SAN history and the seat to move."""
    seat = ""
    sans: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(_SEAT_MARK):
            seat = line[len(_SEAT_MARK) :].strip()
        elif line and not line.startswith("#"):
            sans.append(line)
    if seat not in _SEATS or not sans:
        raise AssertionError(f"{path} is not a readable SAN history")
    return sans, _SEATS[seat]


def replay(sans: list[str], rule: RepetitionRule = RULE) -> tuple[Game, str]:
    """Return ``(game, fen)`` for `sans` replayed from the advanced start.

    The same two-engines-in-step replay the Python match server's history
    replay performs -- ``Gungi`` owns the SAN dialect, ``Game`` accumulates the
    repetition keys -- spelled again here so this file tests the *search*
    without importing a script. ``tests/anchors`` tests the script's copy.

    `rule` defaults to the server's :data:`RULE` and is a parameter for one
    reason: the ``FULL`` half of the Scenario A differential below, which is
    what makes "replay under the site's rule" a claim this file can fail.
    """
    # `Any`: the v1 engine copy is only partially annotated and ``move`` is one
    # of the untyped methods.
    gungi: Any = Gungi(ADVANCED_POSITION)
    game = Game(board_from_fen(ADVANCED_POSITION), repetition_rule=rule)
    for san in sans:
        move: Any = gungi.move(san)
        game.play_action(int(encode_move(dict(move))))
    return game, str(gungi.fen())


def state_for(game: Game) -> MatchState:
    """Return `game` as the match state the Python match server searches."""
    return as_match_state(
        GameState(game, 0.0, min_placements=MIN_PLACEMENTS, is_root=True)
    )


def legal_ids(state: MatchState) -> list[int]:
    """Return the ascending legal action ids of `state`."""
    return [int(action) for action in np.flatnonzero(state.legal_mask())]


def repeating_actions(game: Game) -> tuple[int, ...]:
    """Return the legal actions of `game` that the engine calls a repetition.

    The oracle is the engine's own ``termination_reason``, not this file's idea
    of which move recreates which position.
    """
    found: list[int] = []
    for action in np.flatnonzero(np.asarray(game.valid_moves(), dtype=bool)):
        child = game.clone()
        child.play_action(int(action))
        if child.termination_reason() == TERMINATION_FOURFOLD_REPETITION:
            found.append(int(action))
    return tuple(found)


class SeatEvaluator:
    """A flat evaluator that says one seat is winning, over two nominated moves.

    Three properties, and all three are needed for the differential to be a
    differential: the position is *won* for the seat to move at the root (so
    accepting a draw is a real loss), the repeating move carries the larger
    policy prior (so a search that cannot see the draw takes it), and the
    position is narrowed to the shuffle plus one move that plays on (so both
    get enough visits at this budget for the choice to mean something). That is
    the shape of a checkpoint's behaviour under this kind of pressure, at
    1/8th the playouts and none of the weights.

    Args:
        seat: The engine seat index (0 white, 1 black) this evaluator declares
            winning, in every position it is asked about.
        boosted: Action ids that get ``+BOOST`` at every node.
        offered: Action ids left at logit 0. Everything outside `boosted` and
            `offered` gets ``-SUPPRESS``.
    """

    def __init__(
        self,
        *,
        seat: int,
        boosted: tuple[int, ...] = (),
        offered: tuple[int, ...] = (),
    ) -> None:
        self._seat = int(seat)
        self._boosted = np.asarray(boosted, dtype=np.int64)
        self._offered = np.asarray(offered, dtype=np.int64)
        self.calls = 0

    def __call__(self, state: MatchState) -> WdlEval:
        """Return the stub's evaluation of `state`, and count the question."""
        self.calls += 1
        logits: npt.NDArray[np.float32] = np.full(
            NUM_ACTIONS, -np.float32(SUPPRESS), dtype=np.float32
        )
        if self._offered.size:
            logits[self._offered] = np.float32(0.0)
        if self._boosted.size:
            logits[self._boosted] = np.float32(BOOST)
        # Mover-relative, as the seam requires: the same absolute judgement
        # ("`seat` is winning") read from whichever side is on move here.
        winning = int(state.to_play()) == self._seat
        wdl = WINNING_WDL if winning else LOSING_WDL
        return WdlEval(logits, np.asarray(wdl, dtype=np.float64))


def plays_on(game: Game) -> tuple[int, ...]:
    """Return the legal actions of `game` that do not complete the repetition.

    Ascending, so "the alternative the stub offers" is a derived fact rather
    than a hand-picked one: the differential must not depend on which move that
    happens to be.
    """
    drawing = set(repeating_actions(game))
    return tuple(a for a in legal_ids(state_for(game.clone())) if a not in drawing)


def search_state(state: MatchState, evaluator: SeatEvaluator) -> MatchResult:
    """Search `state` once at :data:`PLAYOUTS`, with no tactical provider."""
    search = MatchSearch(MatchConfig(playouts=PLAYOUTS), evaluator)
    return search.search(state)


@pytest.fixture(scope="module")
def fixture_history() -> tuple[list[str], int]:
    """Return the fixture's ``(sans, seat)``."""
    return load_fixture()


@pytest.fixture(scope="module")
def with_history(fixture_history: tuple[list[str], int]) -> Game:
    """Return the replayed game -- the root that carries its repetition keys."""
    return replay(fixture_history[0])[0]


@pytest.fixture(scope="module")
def without_history(fixture_history: tuple[list[str], int]) -> Game:
    """Return the same position rebuilt from its FEN -- yesterday's root."""
    return Game(board_from_fen(replay(fixture_history[0])[1]))


# ---------------------------------------------------------------------------
# 1. The fixture really is one ply from a draw, and only with its history
# ---------------------------------------------------------------------------


def test_the_fixture_is_one_ply_from_a_repetition_draw(with_history: Game) -> None:
    """The replayed root has a drawing move and a move that plays on."""
    drawing = repeating_actions(with_history)
    legal = legal_ids(state_for(with_history.clone()))
    assert drawing, "the fixture's root offers no repetition to complete"
    assert set(drawing) < set(legal), "every legal move draws; nothing to choose"


def test_the_fen_alone_cannot_see_the_repetition(
    with_history: Game, without_history: Game
) -> None:
    """The defect itself: the same position, rebuilt from its FEN, sees none.

    This is why the protocol had to change rather than the search: the
    information is not in a FEN, so no amount of searching one recovers it.
    """
    assert with_history.repetition_count() > without_history.repetition_count()
    assert repeating_actions(without_history) == ()


# ---------------------------------------------------------------------------
# 2. The state seam carries the history the search needs
# ---------------------------------------------------------------------------


def test_the_match_state_carries_the_history_across_a_clone(
    with_history: Game,
) -> None:
    """``clone`` copies the keys, and the copy is independent.

    The search only ever mutates clones, so a ``clone`` that dropped
    ``_history`` would make every interior node history-less and the whole fix
    a no-op -- silently, because nothing else in the tree would change.
    """
    root = GameState(with_history.clone(), 0.0)
    child = root.clone()
    assert isinstance(child, GameState)
    assert child.game._history == root.game._history  # noqa: SLF001
    assert child.game.repetition_count() == root.game.repetition_count()
    child.play(repeating_actions(with_history)[0])
    # The clone advanced; the parent did not, which is what "independent" means.
    assert len(child.game._history) == len(root.game._history) + 1  # noqa: SLF001


def test_the_repeating_child_is_a_terminal_draw(with_history: Game) -> None:
    """A child that completes the repetition is terminal, and worth ``(0,1,0)``.

    ``terminal_wdl`` is what the tree backs up, so this is the number the
    search actually sees on that edge -- a draw, never a draw *score*.
    """
    root = state_for(with_history.clone())
    child = root.clone()
    child.play(repeating_actions(with_history)[0])
    assert child.is_terminal()
    assert tuple(child.terminal_wdl()) == DRAW_WDL


def test_a_non_repeating_child_is_not_terminal(with_history: Game) -> None:
    """...and the moves that play on are not terminal, so the tree searches on."""
    root = state_for(with_history.clone())
    drawing = set(repeating_actions(with_history))
    playing_on = [a for a in legal_ids(root) if a not in drawing]
    assert playing_on, "the fixture leaves no move that plays on"
    for action in playing_on[:5]:
        child = root.clone()
        child.play(action)
        assert not child.is_terminal(), f"action {action} should not be terminal"


# ---------------------------------------------------------------------------
# 3. The reproducing differential: red without the history, green with it
# ---------------------------------------------------------------------------


def test_the_search_with_history_refuses_the_repetition(
    with_history: Game,
) -> None:
    """Green half: given the keys, the search plays on in a won position.

    The stub hands the repeating move the entire policy prior, so this is the
    search *overriding* its own prior on the evidence of a terminal it can now
    see -- and it keeps a winning utility while doing it, which is the whole
    point of refusing the draw rather than merely detecting it.
    """
    drawing = repeating_actions(with_history)
    evaluator = SeatEvaluator(
        seat=int(with_history.player),
        boosted=drawing,
        offered=plays_on(with_history)[:1],
    )
    result = search_state(state_for(with_history.clone()), evaluator)
    assert result.action not in drawing, (
        f"the search took the repetition ({result.action}) with the history in "
        f"hand"
    )
    assert result.root_utility > WINNING_UTILITY, (
        f"root utility {result.root_utility} -- the position must be won for "
        f"the mover, or refusing the draw proves nothing"
    )


def test_the_search_without_history_walks_into_the_repetition(
    with_history: Game, without_history: Game
) -> None:
    """Red half: the same search on the same FEN, history-less, shuffles.

    ``without_history`` is the root a FEN-only rebuild gives for this position.
    The action ids are the position's, so the move the boosted search takes
    here is the *same move* the other test refuses; the only difference is
    whether the tree could see what it completes.
    """
    drawing = repeating_actions(with_history)
    evaluator = SeatEvaluator(
        seat=int(without_history.player),
        boosted=drawing,
        offered=plays_on(with_history)[:1],
    )
    result = search_state(state_for(without_history.clone()), evaluator)
    assert result.action in drawing, (
        f"the history-less search chose {result.action}, not one of the "
        f"repeating moves {drawing}: the differential this file reports does "
        f"not hold at {PLAYOUTS} playouts"
    )

