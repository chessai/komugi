"""KOM-215: does the match search back up a one-ply marshal capture?

The ticket's headline claim is that a 1600-playout match search returned
``+0.86`` from a position where almost every reply loses the marshal in one
ply. This module is the **regression half** of the adversarial investigation:
it pins down what is actually true about the seam, separates the facts that
survive re-derivation from the ones that do not. KOM-218 and KOM-217 promote
the six original strict expected failures to ordinary regression assertions.

Four things are worth knowing before reading a single assertion.

**The FEN in the ticket body is the position one ply LATER, with the mover left
as white.** Its grid is byte-for-byte the board that results from playing
white's recorded ply-47 move :data:`HISTORICAL_WHITE_MOVE` -- the 砦 that
stands on ``1-8-2`` in :data:`PLY47_FEN` has already stepped to ``1-7-1`` in it
-- but its side-to-move field still names white, so the engine loads it as
*white to move again* and generates 379 moves rather than the ply-47
position's 378. Nothing about it is ungrammatical: it parses, it is a legal
position, and it is simply not the position the search was handed. Every claim
about "376 of 378 replies lose" is a claim about :data:`PLY47_FEN`; the two are
contrasted in :func:`test_the_ticket_fen_is_the_post_ply47_board_with_white_to_move`
so the correction is a test rather than a footnote.

**The position is not a proven forced loss.** Two root moves --
:data:`MARSHAL_ESCAPES`, the marshal stepping to 9-9-1 or 9-8-1 -- leave black
no immediate marshal capture at all. ``test_each_recorded_escape_survives_one
_ply`` checks that directly, one ply wide, on the engine. So the correct
reading of the anchor game is "the search played one of the 376 losing moves
and reported ``+0.86`` doing it", not "the search evaluated a lost position at
``+0.86``".

**No test here loads the campaign checkpoint.** The engine-backed tests are
identity and legality facts about the rules layer; every search test runs on a
written-down stub evaluator with a **uniform** policy. The checkpoint-backed
numbers the investigation produced (raw head, root Q, the capture's prior) live
in the review, ``docs/reviews/2026-09-14-kom-215-terminal-coverage.md``, with
their probe scripts and raw output under
``docs/reviews/2026-09-14-kom-215-evidence/``, and they measure the interaction between the
network's priors and the search's coverage.

**The real-engine tactical fixture is filtered, and says so.**
The examined two-marshal constructions encounter no-escape adjudication when
marshal-safe choices disappear. This is not a universal equivalence proof for
all positions and legal winning replies. FilteredEngineState restricts the
offered moves; board transitions and adjudication remain those of the engine.
The full root has eight legal actions, including an immediate white win.
The restricted tree verifies terminal handling, not full-width tactical coverage.

Evidence classification uses the user's original A-D labels:

* A: reaching the terminal expansion arm and bypassing the evaluator.
* B: the correct mover-relative terminal WDL.
* C: the correct mover-change swaps during backup.
* D: whether policy-guided search selects the capturing reply at all.

The checkpoint probes independently measure D. Passing A/B/C tests alone does
not establish D by elimination. Uniform-prior tests below characterize finite
depth coverage without reproducing the checkpoint's tiny capture prior.

The separate defect this file also carries is the subtree-value-bias one: with
a deliberately wrong evaluator and **both** terminals discovered, the shipped
``lambda = 0.45`` still returns the losing move
(:func:`test_the_search_should_choose_the_drawn_move_at_the_shipped_default`,
now promoted by KOM-217). The original defect subtracted the correction from
the whole mean; selection now adds it to the remaining direct neural mass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

# `planted` first: importing it is what puts the oracle directory on
# ``sys.path``. See ``planted/__init__.py``.
from planted import (
    MatchToyState,
    PlantedLegalState,
    constant_wdl_evaluator,
    walk_nodes,
    wdl_evaluator_from_table,
)

from komugi.engine.cyboard import Board, board_from_fen, board_to_fen
from komugi.engine.game import Game
from komugi.search.match.adapter import MatchGameState
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.search import MatchResult, MatchSearch
from komugi.search.match.state import MatchState, WdlEval, WdlEvaluator
from komugi.search.match.tree import MatchTree
from komugi.selfplay.adapter import GameState

# ---------------------------------------------------------------------------
# The recorded position, and the ticket's mis-transcription of it
# ---------------------------------------------------------------------------

#: The anchor game's position **before** ply 47, recovered by replaying the
#: recorded PGN rather than by trusting the ticket body. White to move, 378
#: legal moves.
PLY47_FEN: Final[str] = (
    "1|N:F|1|g:s:s||j:n:n|m3/3n|r:a|c3/"
    "3|r:w:w|j|i:f|3/1t1|f:k:a|5/9/1N7/"
    "|G:D:I||K:S||A:T|S5/M|D:N|6|R:D:D|/9 J2W1R1F1C1/d4 w 3 - 24"
)

#: What ``tickets/KOM-215.md`` records instead. A valid FEN of a different
#: board: the 砦 has already left the ``|N:F|`` stack for the square beside it,
#: which is what white's ply-47 move does -- but the mover field still says
#: ``w``, so the engine loads it as white to move for a second time and reports
#: 379 legal moves. Kept as a literal so the correction is checked, not
#: asserted in prose.
TICKET_FEN: Final[str] = (
    "1NF|g:s:s||j:n:n|m3/3n|r:a|c3/3|r:w:w|j|i:f|3/1t1|f:k:a|5/9/1N7/"
    "|G:D:I||K:S||A:T|S5/M|D:N|6|R:D:D|/9 J2W1R1F1C1/d4 w 3 - 24"
)

#: The move white actually played at ply 47: 砦(1-8-2)(1-7-1).
HISTORICAL_WHITE_MOVE: Final[int] = 83

#: Black's reply at ply 48: 弓(4-6-3)取(8-9-1)#, the marshal capture.
BLACK_MARSHAL_CAPTURE: Final[int] = 2493

#: The two root moves that do **not** allow an immediate marshal capture:
#: 帥(8-9-1)(9-9-1) and 帥(8-9-1)(9-8-1).
MARSHAL_ESCAPES: Final[tuple[int, int]] = (5175, 5176)

#: The engine's action-space width, and :class:`MatchConfig`'s own default.
NUM_ACTIONS: Final[int] = 37341

#: White's index in the engine's ``[white, black, draw]`` frame.
WHITE: Final[int] = 0

#: A mover-relative loss, and a mover-relative win, spelled once each.
MOVER_LOSES: Final[tuple[float, float, float]] = (0.0, 0.0, 1.0)
MOVER_WINS: Final[tuple[float, float, float]] = (1.0, 0.0, 0.0)
MOVER_DRAWS: Final[tuple[float, float, float]] = (0.0, 1.0, 0.0)


def engine_state(fen: str) -> MatchGameState:
    """Return `fen` as a :class:`MatchState`, through the production seam."""
    return MatchGameState(GameState(Game(board_from_fen(fen)), 0.0))


def grid_of(board: Board) -> str:
    """Return just the piece-placement field of `board`, as the engine writes it."""
    return board_to_fen(board).split()[0]


def board_after(fen: str, action: int) -> Board:
    """Return the :class:`Board` `fen` becomes once `action` is played on it."""
    board = board_from_fen(fen)
    state = MatchGameState(GameState(Game(board), 0.0))
    state.play(action)
    return board


def legal_ids(state: MatchState) -> npt.NDArray[np.int32]:
    """Return the ascending legal action ids of `state`."""
    return np.flatnonzero(state.legal_mask()).astype(np.int32)


def played(state: MatchGameState, action: int) -> MatchGameState:
    """Return an independent copy of `state` with `action` played."""
    advanced = state.clone()
    advanced.play(action)
    return advanced


def raw_outcome(state: MatchGameState) -> npt.NDArray[np.float64]:
    """Return the engine's absolute ``[white, black, draw]`` outcome of `state`."""
    return np.asarray(state.inner.absolute_outcome(), dtype=np.float64)


def terminal_nodes(tree: MatchTree) -> list[MatchNode]:
    """Return every terminal node live in `tree`."""
    return [node for node in walk_nodes(tree.root) if node.is_terminal]


def triple(values: Sequence[float]) -> npt.NDArray[np.float64]:
    """Return `values` as a ``(3,)`` float64 WDL vector."""
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Engine-backed identity and capture facts (evidence A)
# ---------------------------------------------------------------------------


def test_the_ticket_fen_is_the_post_ply47_board_with_white_to_move() -> None:
    """The ticket's FEN is one ply later, with the mover field left as white."""
    recorded = engine_state(PLY47_FEN)
    transcribed = engine_state(TICKET_FEN)
    assert int(legal_ids(recorded).shape[0]) == 378
    assert int(legal_ids(transcribed).shape[0]) == 379
    assert not np.array_equal(legal_ids(recorded), legal_ids(transcribed))

    after_white = board_after(PLY47_FEN, HISTORICAL_WHITE_MOVE)
    assert grid_of(after_white) == grid_of(board_from_fen(TICKET_FEN)), (
        "the ticket FEN's board is the post-ply-47 board: if this fails, the "
        "transcription is something other than a stale side-to-move field"
    )
    assert grid_of(board_from_fen(PLY47_FEN)) != grid_of(after_white)
    # The mover is the difference that is left: white in the ticket literal,
    # black in the real position that board belongs to.
    assert transcribed.to_play() == WHITE
    assert played(recorded, HISTORICAL_WHITE_MOVE).to_play() == 1 - WHITE


def test_the_recorded_move_and_the_marshal_capture_are_both_legal() -> None:
    """The rules layer generates both halves of the recorded ply-47/48 pair."""
    root = engine_state(PLY47_FEN)
    assert root.to_play() == WHITE
    assert not root.is_terminal()
    assert HISTORICAL_WHITE_MOVE in legal_ids(root).tolist()

    after_white = played(root, HISTORICAL_WHITE_MOVE)
    assert after_white.to_play() == 1 - WHITE
    assert not after_white.is_terminal()
    replies = legal_ids(after_white).tolist()
    assert BLACK_MARSHAL_CAPTURE in replies


def test_the_capture_is_terminal_and_states_white_lost() -> None:
    """Evidence A, second half, **and** the frame that evidence B needs."""
    captured = played(
        played(engine_state(PLY47_FEN), HISTORICAL_WHITE_MOVE),
        BLACK_MARSHAL_CAPTURE,
    )
    assert captured.is_terminal()
    assert int(legal_ids(captured).shape[0]) == 0
    assert captured.to_play() == WHITE
    assert np.array_equal(raw_outcome(captured), triple((0.0, 1.0, 0.0)))
    assert np.array_equal(captured.terminal_wdl(), triple(MOVER_LOSES))


@pytest.mark.parametrize("escape", MARSHAL_ESCAPES)
def test_each_recorded_escape_survives_one_ply(escape: int) -> None:
    """Neither escape permits a marshal capture on black's next move."""
    root = engine_state(PLY47_FEN)
    assert escape in legal_ids(root).tolist()

    after_escape = played(root, escape)
    assert not after_escape.is_terminal()
    # Capture and quiet moves share route IDs; inspect the resulting outcome,
    # not whether the historical route ID remains legal.

    for reply in legal_ids(after_escape).tolist():
        answered = played(after_escape, int(reply))
        if not answered.is_terminal():
            continue
        assert float(raw_outcome(answered)[1 - WHITE]) == 0.0, (
            f"reply {reply} wins for black after escape {escape}: the escape "
            f"is not an escape"
        )


# ---------------------------------------------------------------------------
# 2. A filtered REAL two-ply tree: the backup, on the engine's own capture
# ---------------------------------------------------------------------------

#: Two bare marshals, white to move. Eight root moves on the real engine, and
#: **not** all of them losing -- one of them captures black's marshal outright.
TWO_MARSHAL_FEN: Final[str] = "9/9/9/4M4/4m4/9/9/9/9 -/- w 3 - 1"

#: The root move that ends the game in **white's** favour immediately: white's
#: marshal takes black's. The fixture's filter excludes it on purpose, and its
#: existence is why the unfiltered position may never be described as
#: all-losing or the filtered pair as "the safe moves".
IMMEDIATE_WHITE_WIN: Final[int] = 2551

#: The filter: two root moves, one reply each, and each reply is a genuine
#: engine marshal capture. Every id here is the engine's.
TWO_MARSHAL_FILTER: Final[Mapping[tuple[int, ...], frozenset[int]]] = {
    (): frozenset({2541, 2543}),
    (2541,): frozenset({3270}),
    (2543,): frozenset({3272}),
}

#: The same fixture masked to a single root move: design's root-width-1 skip.
SINGLE_ROOT_FILTER: Final[Mapping[tuple[int, ...], frozenset[int]]] = {
    (): frozenset({2541}),
    (2541,): frozenset({3270}),
}


class TerminalEvaluated(Exception):
    """The evaluator was handed a position its state reports terminal."""


class FilteredEngineState:
    """A real engine position with a **written-down** set of allowed actions."""

    __slots__ = ("_inner", "_filter", "_moves")

    def __init__(
        self,
        inner: MatchState,
        filter_sets: Mapping[tuple[int, ...], frozenset[int]],
        *,
        moves: tuple[int, ...] = (),
    ) -> None:
        self._inner = inner
        self._filter = filter_sets
        self._moves = moves

    @property
    def inner(self) -> MatchState:
        """Return the wrapped real position."""
        return self._inner

    @property
    def moves(self) -> tuple[int, ...]:
        """Return the history this state stands at."""
        return self._moves

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the engine's mask, narrowed to the fixture's set."""
        mask = np.asarray(self._inner.legal_mask(), dtype=np.bool_)
        permitted = self._filter.get(self._moves)
        if permitted is None:
            return mask
        narrowed: npt.NDArray[np.bool_] = np.zeros_like(mask)
        for action in permitted:
            assert bool(mask[action]), (
                f"the fixture permits {action} at {self._moves!r}, which the "
                f"engine says is illegal"
            )
            narrowed[action] = True
        return narrowed

    def play(self, action: int) -> None:
        """Advance the real position and extend the history."""
        self._inner.play(action)
        self._moves = (*self._moves, action)

    def clone(self) -> FilteredEngineState:
        """Return an independent copy, cloning the real position."""
        return FilteredEngineState(
            self._inner.clone(), self._filter, moves=self._moves
        )

    def is_terminal(self) -> bool:
        """Return the engine's adjudication."""
        return self._inner.is_terminal()

    def to_play(self) -> int:
        """Return the engine's mover."""
        return self._inner.to_play()

    def observation(self) -> npt.NDArray[np.float32]:
        """Return the real encoding; no test here has a network."""
        return self._inner.observation()

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the engine-derived one-hot for the mover here."""
        return self._inner.terminal_wdl()


class MutatedTerminalState(FilteredEngineState):
    """The same fixture with one number, or one bit, deliberately broken."""

    __slots__ = ("_triple", "_blind")

    def __init__(
        self,
        inner: MatchState,
        filter_sets: Mapping[tuple[int, ...], frozenset[int]],
        *,
        moves: tuple[int, ...] = (),
        terminal_triple: Sequence[float] | None = None,
        blind: bool = False,
    ) -> None:
        super().__init__(inner, filter_sets, moves=moves)
        self._triple = terminal_triple
        self._blind = blind

    def clone(self) -> MutatedTerminalState:
        """Return an independent copy carrying the same mutation."""
        return MutatedTerminalState(
            self.inner.clone(),
            self._filter,
            moves=self.moves,
            terminal_triple=self._triple,
            blind=self._blind,
        )

    def is_terminal(self) -> bool:
        """Return the engine's adjudication unless this state is blind to it."""
        return False if self._blind else super().is_terminal()

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the planted triple, or the engine's if none was planted."""
        if self._triple is None:
            return super().terminal_wdl()
        return triple(self._triple)


class TerminalRefusingEvaluator:
    """An evaluator wrapper that **fails** if asked about a decided position."""

    def __init__(self, inner: WdlEvaluator) -> None:
        self._inner = inner
        self.calls = 0
        self.terminal_calls = 0

    def __call__(self, state: MatchState) -> WdlEval:
        """Return the wrapped evaluation, refusing a decided position."""
        self.calls += 1
        if state.is_terminal():
            self.terminal_calls += 1
            raise TerminalEvaluated(
                "the evaluator was asked about a position the state reports "
                "terminal: a decided position must be scored from its result"
            )
        return self._inner(state)


def constant_refusing_evaluator(
    wdl: Sequence[float], *, num_actions: int
) -> TerminalRefusingEvaluator:
    """Return a uniform-logit constant evaluator that refuses terminals."""
    return TerminalRefusingEvaluator(
        constant_wdl_evaluator(wdl, num_actions=num_actions)
    )


def filtered_fixture(
    filter_sets: Mapping[tuple[int, ...], frozenset[int]] = TWO_MARSHAL_FILTER,
) -> FilteredEngineState:
    """Return the two-marshal fixture at its root."""
    return FilteredEngineState(engine_state(TWO_MARSHAL_FEN), filter_sets)


def filtered_search(
    playouts: int,
    *,
    subtree_bias_factor: float = MatchConfig().subtree_bias_factor,
) -> tuple[MatchSearch, TerminalRefusingEvaluator]:
    """Return a default-configured search over the fixture, and its evaluator."""
    evaluator = constant_refusing_evaluator(MOVER_LOSES, num_actions=NUM_ACTIONS)
    config = MatchConfig(
        playouts=playouts, subtree_bias_factor=subtree_bias_factor
    )
    return MatchSearch(config, evaluator), evaluator


def assert_the_forced_loss_reached_the_root(
    result: MatchResult, tree: MatchTree
) -> None:
    """Assert the healthy claim: both captures are in the tree and dominate."""
    assert len(terminal_nodes(tree)) == 2
    assert result.root_utility <= -0.9 + 1e-9


def test_the_filtered_fixture_is_a_narrowing_of_a_real_position() -> None:
    """The fixture's provenance: eight real root moves, narrowed to two."""
    root = engine_state(TWO_MARSHAL_FEN)
    engine_legal = legal_ids(root).tolist()
    assert len(engine_legal) == 8
    for action in TWO_MARSHAL_FILTER[()]:
        assert action in engine_legal
    assert IMMEDIATE_WHITE_WIN not in TWO_MARSHAL_FILTER[()]

    assert IMMEDIATE_WHITE_WIN in engine_legal
    winner = played(root, IMMEDIATE_WHITE_WIN)
    assert winner.is_terminal()
    assert float(raw_outcome(winner)[WHITE]) == 1.0

    fixture = filtered_fixture()
    assert legal_ids(fixture).tolist() == sorted(TWO_MARSHAL_FILTER[()])


def test_the_filtered_replies_are_genuine_engine_marshal_captures() -> None:
    """Each permitted reply ends the game in black's favour, on the engine."""
    for white_move, replies in (
        (2541, TWO_MARSHAL_FILTER[(2541,)]),
        (2543, TWO_MARSHAL_FILTER[(2543,)]),
    ):
        after_white = played(engine_state(TWO_MARSHAL_FEN), white_move)
        for reply in replies:
            captured = played(after_white, reply)
            assert captured.is_terminal()
            assert captured.to_play() == WHITE
            assert np.array_equal(captured.terminal_wdl(), triple(MOVER_LOSES))


def test_a_budget_equal_to_the_root_width_never_reaches_depth_two() -> None:
    """Budget characterization, part 1: two playouts, two edges, no depth."""
    search, evaluator = filtered_search(2)
    result = search.search(filtered_fixture())
    assert result.playouts_used == 2
    assert terminal_nodes(search.tree) == []
    assert result.root_utility == pytest.approx(1.0)
    assert evaluator.terminal_calls == 0
    # Root plus one child per edge: the §B1 identity with no terminal descent.
    assert result.evals_used == 3


def test_the_third_playout_reaches_the_terminal_and_the_mean_lags() -> None:
    """Budget characterization, part 2: the root mean is not a minimax value."""
    search, _ = filtered_search(3)
    result = search.search(filtered_fixture())
    assert len(terminal_nodes(search.tree)) == 1
    assert result.root_utility == pytest.approx(1.0 / 3.0)


def test_a_sufficient_budget_backs_the_forced_loss_up_to_the_root() -> None:
    """The healthy assertion: at 40 playouts both terminals dominate the root."""
    search, evaluator = filtered_search(40)
    result = search.search(filtered_fixture())
    assert_the_forced_loss_reached_the_root(result, search.tree)
    assert result.root_utility == pytest.approx(-0.9)
    assert result.visits.tolist() == [20, 20]
    assert evaluator.terminal_calls == 0


def test_the_evaluator_is_never_asked_about_a_decided_position() -> None:
    """Terminal evaluator guard (A): terminals are scored from their result."""
    search, evaluator = filtered_search(40)
    search.search(filtered_fixture())

    found = terminal_nodes(search.tree)
    assert len(found) == 2, "the terminals were never reached"
    root = search.tree.root
    assert root is not None
    for index in range(root.width):
        child = root.children[index]
        assert child.visits.tolist() == [19], "the terminal was not revisited"

    assert evaluator.terminal_calls == 0
    # Three positions are non-terminal in this tree, and each is evaluated
    # once: the root and the two children.
    assert evaluator.calls == 3


def test_the_real_capture_backs_up_with_the_right_sign_at_both_frames() -> None:
    """Terminal result and backup (B/C), on the engine's capture, through ``_expand``/``_backup``."""
    evaluator = constant_refusing_evaluator(MOVER_LOSES, num_actions=NUM_ACTIONS)
    tree = MatchTree(MatchConfig(playouts=0), evaluator)
    state = filtered_fixture()
    tree.ensure_root(state)
    root = tree.root
    assert root is not None

    tree.playout(state)
    tree.playout(state)
    assert terminal_nodes(tree) == []
    before = root.wdl_sum.copy()

    tree.playout(state)
    terminal = terminal_nodes(tree)
    assert len(terminal) == 1
    assert np.array_equal(terminal[0].wdl_net, triple(MOVER_LOSES))

    increments = root.wdl_sum - before
    touched = [
        index
        for index in range(root.width)
        if float(np.abs(increments[index]).sum()) > 0.0
    ]
    assert len(touched) == 1
    assert np.array_equal(increments[touched[0]], triple(MOVER_LOSES))

    black_parent = root.children[touched[0]]
    assert black_parent.to_play != root.to_play
    assert np.array_equal(black_parent.wdl_sum[0], triple(MOVER_WINS))


def test_a_false_draw_terminal_value_makes_the_healthy_assertion_fail() -> None:
    """Diagnostic control, red half: the healthy assertion is **run** and fails."""
    evaluator = constant_refusing_evaluator(MOVER_LOSES, num_actions=NUM_ACTIONS)
    search = MatchSearch(MatchConfig(playouts=40), evaluator)
    mutant = MutatedTerminalState(
        engine_state(TWO_MARSHAL_FEN),
        TWO_MARSHAL_FILTER,
        terminal_triple=(0.0, 1.0, 0.0),
    )
    result = search.search(mutant)
    assert len(terminal_nodes(search.tree)) == 2, "detection is unchanged"
    assert evaluator.terminal_calls == 0
    with pytest.raises(AssertionError):
        assert_the_forced_loss_reached_the_root(result, search.tree)


def test_a_state_blind_to_its_own_terminal_cannot_be_searched_at_all() -> None:
    """Diagnostic control, red half: bypassing terminal expansion is fatal."""
    evaluator = constant_refusing_evaluator(MOVER_LOSES, num_actions=NUM_ACTIONS)
    search = MatchSearch(MatchConfig(playouts=40), evaluator)
    mutant = MutatedTerminalState(
        engine_state(TWO_MARSHAL_FEN), TWO_MARSHAL_FILTER, blind=True
    )
    with pytest.raises(ValueError, match="no legal action"):
        search.search(mutant)


def test_a_single_legal_root_move_skips_the_playouts_by_contract() -> None:
    """Budget characterization, part 3: the width-1 skip is old and documented."""
    search, evaluator = filtered_search(40)
    result = search.search(filtered_fixture(SINGLE_ROOT_FILTER))
    assert result.playouts_used == 0
    assert result.evals_used == 1
    assert terminal_nodes(search.tree) == []
    assert result.root_utility == pytest.approx(-1.0)
    assert evaluator.calls == 1


# ---------------------------------------------------------------------------
# 3. Coverage, without a checkpoint and without a learned prior
# ---------------------------------------------------------------------------

#: The width of the recorded ply-47 root, reused as the fixture's width so the
#: arithmetic below is the arithmetic of the real position.
COVERAGE_WIDTH: Final[int] = 378

#: A budget deliberately below the root width.
COVERAGE_BUDGET: Final[int] = 200


def coverage_state() -> MatchToyState:
    """Return a 378-wide root where **every** move is refuted one ply later."""
    legal_sets: dict[tuple[int, ...], frozenset[int]] = {
        (): frozenset(range(COVERAGE_WIDTH))
    }
    results: dict[tuple[int, ...], float] = {}
    for action in range(COVERAGE_WIDTH):
        legal_sets[(action,)] = frozenset({0})
        results[(action, 0)] = -1.0
    return MatchToyState(
        PlantedLegalState(
            legal_sets, num_actions=COVERAGE_WIDTH, results=results
        )
    )


def test_a_sub_width_budget_cannot_refute_most_of_a_378_wide_root() -> None:
    """Coverage characterization, uniform policy: budget alone is enough to lose."""
    evaluator = constant_refusing_evaluator(
        MOVER_LOSES, num_actions=COVERAGE_WIDTH
    )
    search = MatchSearch(
        MatchConfig(num_actions=COVERAGE_WIDTH, playouts=COVERAGE_BUDGET),
        evaluator,
    )
    result = search.search(coverage_state())

    assert result.playouts_used == COVERAGE_BUDGET
    visits = result.visits
    assert int(visits.shape[0]) == COVERAGE_WIDTH
    looked_at = int((visits > 0).sum())
    refuted = int((visits >= 2).sum())

    assert looked_at <= COVERAGE_BUDGET
    assert COVERAGE_WIDTH - looked_at >= COVERAGE_WIDTH - COVERAGE_BUDGET
    assert refuted <= COVERAGE_BUDGET // 2
    assert COVERAGE_WIDTH - refuted >= COVERAGE_WIDTH - COVERAGE_BUDGET // 2
    # Every terminal in the tree is the reply behind a twice-visited root edge,
    # and there is exactly one such reply per edge.
    assert len(terminal_nodes(search.tree)) == refuted
    assert evaluator.terminal_calls == 0


# ---------------------------------------------------------------------------
# 4. Discovery-independent bias selection (a second, unrelated defect)
# ---------------------------------------------------------------------------

#: Root has two moves; each leads to a one-move position; both of those end.
#: ``(0, 0)`` loses for the mover there, ``(1, 0)`` is drawn.
BIAS_LEGAL_SETS: Final[Mapping[tuple[int, ...], frozenset[int]]] = {
    (): frozenset({0, 1}),
    (0,): frozenset({0}),
    (1,): frozenset({0}),
}

#: ``(history) -> result for the mover there``. Only terminals are named.
BIAS_RESULTS: Final[Mapping[tuple[int, ...], float]] = {
    (0, 0): -1.0,
    (1, 0): 0.0,
}

#: What each root move's terminal descendant must say, in the mover's own frame
#: at that terminal. Move 0 loses by a full point; move 1 draws.
BIAS_TERMINAL_WDL: Final[Mapping[int, tuple[float, float, float]]] = {
    0: MOVER_LOSES,
    1: MOVER_DRAWS,
}

#: The budgets every bias test runs at. Two of them, because a single budget
#: cannot distinguish "the defect" from "this budget".
BIAS_BUDGETS: Final[tuple[int, int]] = (64, 256)

#: The shipped default, read off the config rather than retyped.
SHIPPED_BIAS_FACTOR: Final[float] = MatchConfig().subtree_bias_factor


def bias_state() -> MatchToyState:
    """Return the planted two-move game at its root."""
    return MatchToyState(
        PlantedLegalState(BIAS_LEGAL_SETS, num_actions=2, results=BIAS_RESULTS)
    )


def bias_search(
    playouts: int, *, subtree_bias_factor: float
) -> tuple[MatchSearch, TerminalRefusingEvaluator]:
    """Return a search over the planted game, and its evaluator."""
    evaluator = TerminalRefusingEvaluator(
        wdl_evaluator_from_table(
            {(0,): MOVER_LOSES},
            num_actions=2,
            default=MOVER_WINS,
        )
    )
    search = MatchSearch(
        MatchConfig(
            num_actions=2,
            playouts=playouts,
            subtree_bias_factor=subtree_bias_factor,
        ),
        evaluator,
    )
    return search, evaluator


def assert_both_bias_terminals_were_reached(
    search: MatchSearch, evaluator: TerminalRefusingEvaluator
) -> None:
    """Assert the discovery preconditions the bias claim rests on."""
    root = search.tree.root
    assert root is not None
    assert root.width == 2
    assert root.actions.tolist() == [0, 1]

    for index, expected in BIAS_TERMINAL_WDL.items():
        assert index in root.children, f"root move {index} was never expanded"
        child = root.children[index]
        assert not child.is_terminal
        assert child.width == 1
        assert 0 in child.children, (
            f"the terminal behind root move {index} was never reached"
        )
        terminal = child.children[0]
        assert terminal.is_terminal
        assert np.array_equal(terminal.wdl_net, triple(expected))
        assert int(child.visits[0]) > 0, (
            f"the terminal behind root move {index} has no incoming visits"
        )

    assert len(terminal_nodes(search.tree)) == 2
    assert evaluator.terminal_calls == 0


@pytest.mark.parametrize("playouts", BIAS_BUDGETS, ids=("P64", "P256"))
def test_both_bias_terminals_are_reached_at_the_shipped_default(
    playouts: int,
) -> None:
    """The precondition of the bias finding: discovery is not the problem."""
    search, evaluator = bias_search(
        playouts, subtree_bias_factor=SHIPPED_BIAS_FACTOR
    )
    search.search(bias_state())
    assert_both_bias_terminals_were_reached(search, evaluator)


@pytest.mark.parametrize("playouts", BIAS_BUDGETS, ids=("P64", "P256"))
def test_the_search_should_choose_the_drawn_move_at_the_shipped_default(
    playouts: int,
) -> None:
    """KOM-217: prefer the draw after independently discovering both terminals."""
    search, evaluator = bias_search(playouts, subtree_bias_factor=SHIPPED_BIAS_FACTOR)
    result = search.search(bias_state())
    assert_both_bias_terminals_were_reached(search, evaluator)
    assert result.action == 1


@pytest.mark.parametrize("playouts", BIAS_BUDGETS, ids=("P64", "P256"))
def test_disabling_the_bias_chooses_the_drawn_move(playouts: int) -> None:
    """The positive control: ``lambda = 0`` gets the same fixture right."""
    search, evaluator = bias_search(playouts, subtree_bias_factor=0.0)
    result = search.search(bias_state())

    assert_both_bias_terminals_were_reached(search, evaluator)
    assert result.action == 1


def test_the_bias_fixture_costs_exactly_three_evaluations() -> None:
    """Three non-terminal positions, three evaluations, at either bias factor."""
    for factor in (0.0, SHIPPED_BIAS_FACTOR):
        search, evaluator = bias_search(64, subtree_bias_factor=factor)
        search.search(bias_state())
        assert evaluator.calls == 3
        assert_both_bias_terminals_were_reached(search, evaluator)


# ---------------------------------------------------------------------------
# 5. Terminal ingress: what a custom state can put into the tree
# ---------------------------------------------------------------------------


class TerminalIngressState:
    """A two-move root whose children are terminal with a planted triple."""

    __slots__ = ("_planted", "_moves")

    def __init__(
        self, planted: Sequence[float], *, moves: tuple[int, ...] = ()
    ) -> None:
        self._planted = tuple(float(value) for value in planted)
        self._moves = moves

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return both actions at the root, and nothing once a move is made."""
        mask: npt.NDArray[np.bool_] = np.zeros(2, dtype=np.bool_)
        if not self._moves:
            mask[:] = True
        return mask

    def play(self, action: int) -> None:
        """Extend the history by `action`."""
        self._moves = (*self._moves, action)

    def clone(self) -> TerminalIngressState:
        """Return an independent copy."""
        return TerminalIngressState(self._planted, moves=self._moves)

    def is_terminal(self) -> bool:
        """Return whether a move has been made."""
        return bool(self._moves)

    def to_play(self) -> int:
        """Return the mover, alternating by ply."""
        return len(self._moves) % 2

    def observation(self) -> npt.NDArray[np.float32]:
        """Return a placeholder encoding."""
        return np.zeros(1, dtype=np.float32)

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the planted triple, whatever it is."""
        return triple(self._planted)


def ingress_search() -> MatchSearch:
    """Return a two-playout search with a valid uniform stub evaluator."""
    return MatchSearch(
        MatchConfig(num_actions=2, playouts=2),
        constant_refusing_evaluator(MOVER_WINS, num_actions=2),
    )


@pytest.mark.parametrize(
    "planted",
    (
        (0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0),
        (-1.0, 1.0, 1.0),
        (0.9, 0.9, 0.9),
    ),
    ids=("zero", "nan", "negative", "off-mass"),
)
def test_a_terminal_that_is_not_a_distribution_should_be_refused(
    planted: tuple[float, float, float],
) -> None:
    """Pre-registered: the terminal ingress should be as strict as the evaluator."""
    with pytest.raises(ValueError):
        ingress_search().search(TerminalIngressState(planted))


def test_a_fractional_terminal_distribution_stays_allowed() -> None:
    """The boundary of the proposal: a *valid* non-one-hot terminal is legal."""
    result = ingress_search().search(TerminalIngressState((0.5, 0.25, 0.25)))
    assert result.playouts_used == 2
    assert result.root_utility == pytest.approx(-0.25)
