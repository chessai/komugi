"""Toy games for the Gumbel-search oracle suite.

Two games, both duck-typing the ``SearchState`` protocol that
``komugi.search.gumbel`` searches over (``legal_mask``/``play``/``clone``/
``is_terminal``/``terminal_value``/``to_play``):

* :class:`TicTacToe` -- nine actions, small enough to solve exactly, and rich
  enough to have real tactics (an immediate win, a forced block, a fork). It
  is the substrate for the oracle: a search that plays a losing move in a
  position where the solver says a draw or a win exists is broken, and no
  amount of "the value net was uncertain" explains it away, because the value
  handed to the search *is* the solver.
* :class:`TreeGame` -- a depth-two game whose payoff table is written down by
  hand. Nothing about it is discovered; every leaf value is chosen by the
  test, so a test can point at exactly which number the search should have
  found and which one it actually reported.

Both carry a :meth:`key` for the solver's memo table, which is not part of the
search protocol -- the search never sees it.

Value convention throughout, matching the contract the search is bound to:
every value is from the perspective of the player *to move at that state*, and
values negamax-propagate (a child's value is negated when it moves to the
parent).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

EMPTY: Final = 0
CROSS: Final = 1  # player 0's mark
NOUGHT: Final = 2  # player 1's mark

BOARD_SIZE: Final = 9

#: The eight ways to win, as cell indices into a row-major 3x3 board.
LINES: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

_SYMBOLS: Final[dict[str, int]] = {".": EMPTY, "X": CROSS, "O": NOUGHT}
_GLYPHS: Final[dict[int, str]] = {EMPTY: ".", CROSS: "X", NOUGHT: "O"}


class TicTacToe:
    """Standard tic-tac-toe on nine actions, player 0 ("X") moving first.

    Mutable in place: :meth:`play` writes the mark and flips the side to move,
    which is what makes :meth:`clone` -- and the search's obligation not to
    mutate the caller's state -- worth testing.
    """

    __slots__ = ("_cells", "_to_play")

    def __init__(self, cells: Sequence[int] | None = None, to_play: int = 0) -> None:
        board = [EMPTY] * BOARD_SIZE if cells is None else list(cells)
        if len(board) != BOARD_SIZE:
            raise ValueError(f"a board has {BOARD_SIZE} cells, got {len(board)}")
        for cell in board:
            if cell not in (EMPTY, CROSS, NOUGHT):
                raise ValueError(f"cell value out of range: {cell!r}")
        if to_play not in (0, 1):
            raise ValueError(f"to_play must be 0 or 1, got {to_play!r}")
        self._cells: list[int] = board
        self._to_play: int = to_play

    @classmethod
    def from_string(cls, text: str, to_play: int | None = None) -> TicTacToe:
        """Build a position from nine ``.``/``X``/``O`` glyphs, row-major.

        Whitespace and ``/`` are separators, so ``"XX./OO./O.X"`` reads as a
        board. With `to_play` left out the side to move is inferred from the
        mark counts, which is the only consistent reading of a reachable
        position: equal counts means X, one extra X means O. A count pair that
        no legal game could have produced raises rather than picking a side
        silently.
        """
        cells: list[int] = []
        for glyph in text:
            if glyph.isspace() or glyph == "/":
                continue
            if glyph not in _SYMBOLS:
                raise ValueError(f"unknown board glyph: {glyph!r}")
            cells.append(_SYMBOLS[glyph])
        if len(cells) != BOARD_SIZE:
            raise ValueError(f"a board has {BOARD_SIZE} cells, got {len(cells)}")
        if to_play is None:
            crosses, noughts = cells.count(CROSS), cells.count(NOUGHT)
            if crosses == noughts:
                to_play = 0
            elif crosses == noughts + 1:
                to_play = 1
            else:
                raise ValueError(
                    f"unreachable mark counts: {crosses} X vs {noughts} O"
                )
        return cls(cells, to_play)

    @property
    def cells(self) -> tuple[int, ...]:
        """The board as a nine-tuple of cell values, safe to hold onto."""
        return tuple(self._cells)

    def to_play(self) -> int:
        return self._to_play

    def winner(self) -> int | None:
        """Return the player index owning a completed line, or None."""
        for a, b, c in LINES:
            mark = self._cells[a]
            if mark != EMPTY and self._cells[b] == mark and self._cells[c] == mark:
                return 0 if mark == CROSS else 1
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or EMPTY not in self._cells

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Empty cells, or nothing at all once the game is over."""
        mask: npt.NDArray[np.bool_] = np.zeros(BOARD_SIZE, dtype=np.bool_)
        if self.is_terminal():
            return mask
        for index, cell in enumerate(self._cells):
            if cell == EMPTY:
                mask[index] = True
        return mask

    def terminal_value(self) -> float:
        """+1/0/-1 from the perspective of the player to move at this state.

        In a reachable terminal position the winner is never the side to move,
        so this is -1 or 0 there; the +1 branch exists for hand-built states
        and keeps the perspective rule honest rather than special-cased.
        """
        if not self.is_terminal():
            raise ValueError("terminal_value() on a non-terminal position")
        won_by = self.winner()
        if won_by is None:
            return 0.0
        return 1.0 if won_by == self._to_play else -1.0

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """The same result as a ``(w, d, l)`` one-hot, same perspective (KOM-173).

        A drawn board is ``(0, 1, 0)`` and a decided one is ``(1, 0, 0)`` or
        ``(0, 0, 1)``, so ``w - l`` is :meth:`terminal_value` exactly -- which is
        what the search's searched-WDL identity is stated against.
        """
        value = self.terminal_value()
        if value == 0.0:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if value > 0.0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def play(self, action: int) -> None:
        if not 0 <= action < BOARD_SIZE:
            raise ValueError(f"action out of range: {action!r}")
        if self._cells[action] != EMPTY:
            raise ValueError(f"cell {action} is already occupied")
        if self.winner() is not None:
            raise ValueError("the game is already decided")
        self._cells[action] = CROSS if self._to_play == 0 else NOUGHT
        self._to_play = 1 - self._to_play

    def clone(self) -> TicTacToe:
        return TicTacToe(self._cells, self._to_play)

    def key(self) -> Hashable:
        return (tuple(self._cells), self._to_play)

    def __repr__(self) -> str:
        rows = "/".join(
            "".join(_GLYPHS[cell] for cell in self._cells[start : start + 3])
            for start in (0, 3, 6)
        )
        return f"TicTacToe({rows!r}, to_play={self._to_play})"


class TreeGame:
    """A depth-two, perfect-information game read straight off a payoff table.

    ``payoffs[a][b]`` is the value for player 0 after player 0 plays `a` and
    player 1 replies `b`. The table is square, so both plies have the same
    action count. Two plies land back on player 0, so at a leaf the player to
    move *is* player 0 and the payoff needs no negation to be reported from
    the to-move perspective.

    The minimax value of the root is therefore ``max_a min_b payoffs[a][b]``,
    which a test can compute in one line without trusting the solver.
    """

    __slots__ = ("_payoffs", "_moves")

    def __init__(
        self, payoffs: Sequence[Sequence[float]], moves: Sequence[int] = ()
    ) -> None:
        table = tuple(tuple(float(value) for value in row) for row in payoffs)
        if not table:
            raise ValueError("payoff table must have at least one row")
        width = len(table)
        for row in table:
            if len(row) != width:
                raise ValueError("payoff table must be square")
            for value in row:
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"payoff out of [-1, 1]: {value!r}")
        played = tuple(int(move) for move in moves)
        if len(played) > 2:
            raise ValueError("the tree game is two plies deep")
        for move in played:
            if not 0 <= move < width:
                raise ValueError(f"action out of range: {move!r}")
        self._payoffs: tuple[tuple[float, ...], ...] = table
        self._moves: tuple[int, ...] = played

    @property
    def num_actions(self) -> int:
        return len(self._payoffs)

    @property
    def moves(self) -> tuple[int, ...]:
        """The actions played so far, root first."""
        return self._moves

    def to_play(self) -> int:
        return len(self._moves) % 2

    def is_terminal(self) -> bool:
        return len(self._moves) == 2

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Every action, at both plies; nothing once the leaf is reached."""
        mask: npt.NDArray[np.bool_] = np.zeros(self.num_actions, dtype=np.bool_)
        if not self.is_terminal():
            mask[:] = True
        return mask

    def terminal_value(self) -> float:
        if not self.is_terminal():
            raise ValueError("terminal_value() on a non-terminal position")
        first, second = self._moves
        return self._payoffs[first][second]

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """The payoff as a ``(w, d, l)`` vector with no draw mass (KOM-173).

        A payoff table is a scalar game, so there is no honest ``d`` to report:
        ``w = (1 + v) / 2`` and ``l = (1 - v) / 2`` is the probability vector
        whose utility is that payoff, which keeps the search's identity
        ``root_wdl[0] - root_wdl[2] == root_value`` checkable on this game too.
        """
        value = self.terminal_value()
        return np.array(
            [(1.0 + value) / 2.0, 0.0, (1.0 - value) / 2.0], dtype=np.float64
        )

    def play(self, action: int) -> None:
        if self.is_terminal():
            raise ValueError("the game is already over")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action out of range: {action!r}")
        self._moves = (*self._moves, action)

    def clone(self) -> TreeGame:
        return TreeGame(self._payoffs, self._moves)

    def key(self) -> Hashable:
        # The payoff table is part of the identity: two tree games one move in
        # are the same *position* and completely different *games*, and a memo
        # keyed on the moves alone would confuse them.
        return (self._payoffs, self._moves)

    def __repr__(self) -> str:
        return f"TreeGame({self._payoffs!r}, moves={self._moves!r})"


@dataclass(frozen=True)
class TacticalCase:
    """A hand-analysed tic-tac-toe position and its exact verdict.

    `optimal` and `value` are written down here by hand and checked against
    the solver in ``test_fixtures.py``; the oracle tests then use them as the
    standard the search has to meet. Two independent derivations of the same
    fact is the point -- if the hand analysis and the negamax disagree, the
    fixture test fails before any search is judged by either.
    """

    name: str
    board: str
    to_play: int
    value: float
    optimal: frozenset[int]

    def state(self) -> TicTacToe:
        """A fresh position; the side to move is asserted, not just inferred."""
        state = TicTacToe.from_string(self.board)
        if state.to_play() != self.to_play:
            raise ValueError(f"{self.name}: board and to_play disagree")
        return state


#: Positions where the right move is forced and the wrong ones are punished
#: immediately, so a search either finds it or visibly does not. Every case
#: has at most four legal actions: with ``m >= 4`` the root candidate set is
#: the whole legal set, so a miss is a search failure and never an unlucky
#: Gumbel draw that left the good move out of the candidates.
TACTICAL_CASES: Final[tuple[TacticalCase, ...]] = (
    TacticalCase(
        # X completes the top row at 2, which also happens to defuse one of
        # O's two threats; anything else loses to 5 or 2 next move.
        name="win_now",
        board="XX./OO./O.X",
        to_play=0,
        value=1.0,
        optimal=frozenset({2}),
    ),
    TacticalCase(
        # O threatens the top row at 2. Blocking it holds the draw; the other
        # two moves hand O the win at once.
        name="must_block_to_draw",
        board="OO./XXO/.X.",
        to_play=0,
        value=0.0,
        optimal=frozenset({2}),
    ),
    TacticalCase(
        # O to move with a win at 5 (2-5-8) and a slower win at 7 that blocks
        # X's 1-4-7 first; the other two moves lose to X playing 7.
        name="o_wins_by_force",
        board="XXO/.X./..O",
        to_play=1,
        value=1.0,
        optimal=frozenset({5, 7}),
    ),
    TacticalCase(
        # O must take 2: it blocks X's top row and forks 2-5-8 with 2-4-6.
        name="o_blocks_into_a_fork",
        board="XX./.O./.XO",
        to_play=1,
        value=1.0,
        optimal=frozenset({2}),
    ),
    TacticalCase(
        # Every move wins for X: 5 and 7 immediately, 8 because the resulting
        # double threat cannot be answered. A search cannot go wrong here,
        # which is exactly what makes it a control.
        name="all_moves_win",
        board="OXO/XX./O..",
        to_play=0,
        value=1.0,
        optimal=frozenset({5, 7, 8}),
    ),
)
