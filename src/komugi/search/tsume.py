"""Budgeted tsume (forced-mate) search over the compiled Gungi engine.

KOM-258. Neural search misses forced mates its policy prior ranks low, and in
a drop game it misses them systematically: the mating move is often a piece
coming off the hand onto a square nothing was attacking, which is exactly the
kind of move a policy trained on ordinary play gives no mass to. Measured on
the KOM-257 mate suite, mate-in-1 is solved and multi-move tsume is not. The
precedent is dlshogi, which pairs its NN-MCTS with a classical **df-pn** mate
searcher for the same reason.

So this module is deliberately classical: an exact AND/OR search with proof
and disproof numbers (:mod:`komugi.search.tsume_core`), bound to the engine
by the adapter below. No network, no evaluation, no heuristics -- the only
numbers in it are proof numbers.

**Semantics** (fixed by the KOM-258 design review; the tests are written
against these words, not against the implementation):

*Attacker* is the side to move at the root. Which of its moves it may play
is the one thing about these semantics that is selectable, by
``attacker_moves`` (KOM-262):

``"checks"`` (the default, and what every existing caller gets)
    **Checking moves only** -- every pseudo-legal move after which the
    *defender* is in check. That is the standard tsume restriction; it is
    what keeps the OR branching factor small enough for the search to be
    exact, and 詰将棋 convention requires it of a composed problem.

``"all"``
    **Every pseudo-legal move**, quiet ones included. This is the general
    forced-mate question rather than the tsume one: a mate whose first move
    takes away the defender's last flight square without giving check is a
    real forced mate, and checks-only cannot see it. The error of the
    default is therefore one-sided -- it never invents a mate, it only
    misses them -- and ``"all"`` buys the recall back at a large cost in
    branching factor.

*Defender*, at an AND node, replies with every pseudo-legal move after which
**the defender is not in check** -- which is to say every *legal* move --
**plus** every reply that captures the attacker's marshal. The second half
matters because komugi's movegen is pseudo-legal: without it, the defender
would be allowed to answer a check with a move that ignores it, and no
multi-move puzzle in the suite is provable. With it, a defender who can take
the attacker's marshal ends the game, and the attack is judged to have failed
there.

Under ``"checks"`` the defender is in check at *every* AND node, so that
reply set is the escape set and nothing more needs saying. Under ``"all"`` it
is not, and the same filter then means something wider: every legal reply,
not just the evasions. The filter is identical either way; what changes is
what an *empty* reply set means, which is the next two paragraphs.

*Mate* is an AND node with no reply at all **where the defender is in
check**. That is asserted equivalent to ``cyengine.is_checkmate`` at every
such node when the problem is built with ``verify_checkmate=True``, which the
test suite does over every fixture.

*Stalemate* -- an AND node with no reply where the defender is **not** in
check -- is unreachable under ``"checks"``, because an AND node is reached
only by a checking move and a side in check is not stalemated. Under
``"all"`` it is reachable, and it is scored as a **failed attack**
(``DISPROVED``): a stalemate is not a checkmate, and the one way this mode
could be worse than useless is by proving a mate that is not there. So
:meth:`CyTsumeProblem.terminal` catches those nodes *before* the search can
read "no children at an AND node" as mate. (Whether gungi's rules call
stalemate a loss, a draw, or nothing at all is a separate question this
solver deliberately does not answer: it reports forced *mates*, and a
stalemate is not one.)

*Depth* counts plies of the attack line: 1手 = 1, 3手 = 3, and a mate must
land on an attacker ply at or below ``max_plies``.

*Repetition* is path-local and conservative: a position repeating an ancestor
of the current path ends the attack there. A mate reachable only by repeating
is therefore not proven -- a documented limitation, and the price of not
having to reason about graph-history interaction. See
:mod:`komugi.search.tsume_core` for the transposition-table taint rule that
keeps such a result from leaking out of the path that produced it.

**Scope.** v1 is offline: the consumers are the KOM-257 mate experiments and
the test suite. Nothing here is wired into selfplay, the engine server or
training.

The trap this adapter is built around: ``is_checkmate`` returns ``False``,
silently, on a board without exactly two marshals. Under the defender
semantics above, boards with a captured marshal genuinely occur as children,
so every node is checked for a missing marshal *before* any move is generated
and treated as game-over-without-mate there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from komugi.engine import cyboard, cyengine, cymovegen
from komugi.engine.cyboard import Board
from komugi.search.tsume_core import (
    Outcome,
    TsumeResult,
    TsumeTable,
    Verdict,
    solve_problem,
)

__all__ = [
    "AttackerMoves",
    "CyTsumeProblem",
    "TsumeResult",
    "TsumeTable",
    "Verdict",
    "solve_tsume",
]

#: Which of the attacker's moves an OR node expands. See this module's
#: docstring; ``"checks"`` is the default everywhere.
AttackerMoves = Literal["checks", "all"]

#: One byte per mode, mixed into :meth:`CyTsumeProblem.digest` so that the
#: transposition table -- which is keyed on the digest -- is partitioned by
#: mode. Without it a "no mate within 3 plies" entry learned under
#: ``"checks"`` would answer a ``"all"`` query, which is the same class of bug
#: as leaving the remaining depth out of the key, and it would silently turn
#: the new mode into a wrong answer rather than a slow one.
_MODE_TAG: Final[dict[str, bytes]] = {"checks": b"\x00", "all": b"\x01"}

#: Piece codes of the two marshals, as they appear in ``Board.cells_arr``.
_MARSHAL_CODES: Final = (cyboard.make_code(0, 0), cyboard.make_code(0, 1))

#: Default node-expansion budget. Generous for the checks-only puzzle sizes
#: this is used on (every row of the 15-puzzle friend set solves far below it
#: under ``"checks"``) and small enough that a pathological position
#: answers UNKNOWN rather than running forever. It is **not** generous for
#: ``attacker_moves="all"``: full width can exceed it on a composed puzzle --
#: F15 at its 7-ply grade needs some 845,000 expansions -- so callers of the
#: full-width mode should expect to raise the budget.
DEFAULT_NODE_BUDGET: Final = 100_000


class CyTsumeProblem:
    """The production binding: :class:`TsumeProblem` over the compiled engine.

    Nodes are :class:`~komugi.engine.cyboard.Board` objects, never mutated --
    every child is a fresh ``copy()``. The digest is
    :func:`cyengine.repetition_key`, the engine's exact 274-byte
    serialization of everything that makes a position a position (cells, side
    to move, draft flags, hands). Exact matters twice over here: it is the
    transposition-table key *and* the repetition test, so a lossy digest would
    not slow the search down, it would answer wrongly.
    """

    def __init__(
        self,
        root: Board,
        *,
        verify_checkmate: bool = False,
        attacker_moves: AttackerMoves = "checks",
    ) -> None:
        #: The side to move at the root, for the whole search.
        self.attacker: Final = int(root.turn)
        self.defender: Final = 1 - self.attacker
        if attacker_moves not in _MODE_TAG:
            raise ValueError(
                f"attacker_moves must be 'checks' or 'all', got {attacker_moves!r}"
            )
        #: Which of the attacker's moves OR nodes expand.
        self.attacker_moves: Final[AttackerMoves] = attacker_moves
        # Hoisted out of the hot path: this is asked once per node.
        self._quiet_attacker: Final = attacker_moves == "all"
        self._mode_tag: Final = _MODE_TAG[attacker_moves]
        # Off by default: the check costs a second full movegen-and-apply
        # pass at every mate leaf. The test suite turns it on.
        self._verify_checkmate = verify_checkmate
        # One-slot memo for _defender_replies, used only under "all", where
        # terminal() has to generate the replies to tell mate from stalemate
        # and children() then asks for the very same set. The search calls
        # terminal(node) immediately before children(node) with nothing in
        # between, so one slot is all it takes; it is a speed-up only, and
        # missing it recomputes rather than answering wrongly. The cached
        # value is a tuple, so no caller can empty the memo through the
        # sequence children() handed it.
        self._reply_memo: tuple[bytes, tuple[tuple[str, Board], ...]] | None = None

    def digest(self, node: Board) -> bytes:
        """The position's exact identity, tagged with the attacker-move mode.

        The tag is not decoration. This value is the transposition-table key,
        and a table may outlive one solve (:class:`TsumeTable`), so two
        problems asking *different questions* about one position would
        otherwise answer each other: "no mate within 3 plies by checks alone"
        is not "no mate within 3 plies", and reading the first as the second
        is how a mode selector turns into a wrong verdict. Tagging the digest
        partitions the table, the line memo and the repetition path all at
        once.
        """
        return self._mode_tag + cyengine.repetition_key(node)

    def terminal(self, node: Board) -> Outcome | None:
        """``DISPROVED`` when the game is over or the defender is stalemated.

        Two ways, and the second exists only under ``attacker_moves="all"``:

        *A captured marshal.* Always the *attacker's* under ``"checks"``: an
        attacker move that took the defender's marshal would leave the
        defender un-checkable, so it is not a checking move and never enters
        the tree. Under ``"all"`` the attacker may well capture the
        defender's marshal, and that is not a mate either -- winning by
        capture is not the question this solver answers -- so the same
        ``DISPROVED`` is the right reading for both.

        *A stalemated defender.* An AND node with no reply at all is checkmate
        only if the defender is in check; if it is not, the position is a
        stalemate, and the search's "no children at an AND node means mate"
        rule would prove a mate that does not exist. Catching it here, before
        expansion, is what keeps that rule sound. Scored as a failed attack,
        which is the conservative reading -- see the module docstring.
        """
        if _marshal_missing(node):
            return Outcome.DISPROVED
        if self._quiet_attacker and int(node.turn) == self.defender:
            # in_check is the cheap half of the test and rules out stalemate
            # outright, so it runs before any move generation.
            if cyengine.in_check(node, self.defender):
                return None
            if not self._defender_replies(node):
                return Outcome.DISPROVED
        return None

    def children(self, node: Board) -> Sequence[tuple[str, Board]]:
        if int(node.turn) == self.attacker:
            return self._attacker_moves(node)
        return self._defender_replies(node)

    def _attacker_moves(self, node: Board) -> list[tuple[str, Board]]:
        """The attacker's moves: the checking ones, or under ``"all"`` every one.

        Attacker moves that leave the attacker's *own* marshal in check are
        not filtered out here, and need not be: the defender's reply set at
        the resulting node contains the capture of that marshal, which
        disproves the node one ply later. That argument does not depend on
        the mode, so ``"all"`` keeps the same pseudo-legal move list rather
        than paying for a legality filter the search already applies.
        """
        quiet_ok = self._quiet_attacker
        out: list[tuple[str, Board]] = []
        for move in cymovegen.generate_all_moves(node):
            child = node.copy()
            cyengine.apply_move(child, move)
            if quiet_ok or cyengine.in_check(child, self.defender):
                out.append((str(move["san"]), child))
        return out

    def _defender_replies(self, node: Board) -> tuple[tuple[str, Board], ...]:
        """Every legal reply, plus any capture of the attacker's marshal.

        "Legal" is the one filter -- pseudo-legal moves after which the
        defender is not in check -- and it is the same filter in both modes.
        Under ``"checks"`` the defender is in check here, so that set is the
        escape set; under ``"all"`` it may not be, and the very same filter
        then admits every legal move, which is exactly the widening the
        general question needs. What the two modes disagree about is not this
        list but what an *empty* one means, and that is
        :meth:`terminal`'s business.
        """
        memo_key = cyengine.repetition_key(node) if self._quiet_attacker else b""
        if self._reply_memo is not None and self._reply_memo[0] == memo_key:
            return self._reply_memo[1]

        escapes: list[tuple[str, Board]] = []
        game_enders: list[tuple[str, Board]] = []
        for move in cymovegen.generate_all_moves(node):
            child = node.copy()
            cyengine.apply_move(child, move)
            if not cyengine.in_check(child, self.defender):
                escapes.append((str(move["san"]), child))
            elif _marshal_missing(child):
                game_enders.append((str(move["san"]), child))
        # The escape-set/is_checkmate equivalence is a statement about a
        # defender *in check*, and under "checks" that is an invariant of the
        # attacker's move filter rather than something to measure. Under
        # "all" it has to be asked, and only when the assertion is armed.
        if self._verify_checkmate and (
            not self._quiet_attacker or cyengine.in_check(node, self.defender)
        ):
            self._assert_checkmate_agreement(node, len(escapes) + len(game_enders))
        replies = tuple(escapes + game_enders)
        if self._quiet_attacker:
            self._reply_memo = (memo_key, replies)
        return replies

    def _assert_checkmate_agreement(self, node: Board, replies: int) -> None:
        """"No reply at all" and ``is_checkmate`` must be the same statement.

        Since KOM-1109, ``cyengine.is_checkmate`` itself treats a reply that
        captures the attacker's marshal as an escape (such a reply ends the
        game rather than leaving the defender's own marshal in check), so it
        is no longer just the escape set that must be empty for ``is_checkmate``
        to hold -- it's the full reply set, escapes plus game-enders. *replies*
        is that total. Any disagreement means this adapter has drifted from
        the engine's own notion of mate -- the one thing the whole search is
        trying to prove. Checked at every AND node, not only the mating ones,
        so a spurious escape is caught too.
        """
        mate = bool(cyengine.is_checkmate(node))
        if (replies == 0) != mate:
            raise AssertionError(
                "defender reply set disagrees with cyengine.is_checkmate: "
                f"{replies} replies but is_checkmate={mate} "
                f"for {cyboard.board_to_fen(node)!r}"
            )


def _marshal_missing(board: Board) -> bool:
    """True when either side's marshal is off the board (the game is over).

    Counted per colour rather than in total: a betrayal that converted a
    marshal would keep the total at two while ending the game just as surely.
    """
    cells = board.cells_arr
    return any(int((cells == code).sum()) != 1 for code in _MARSHAL_CODES)


def solve_tsume(
    fen: str,
    max_plies: int,
    node_budget: int = DEFAULT_NODE_BUDGET,
    *,
    verify_checkmate: bool = False,
    attacker_moves: AttackerMoves = "checks",
    table: TsumeTable | None = None,
) -> TsumeResult:
    """Search *fen* for a forced mate by the side to move within *max_plies*.

    Returns ``MATE`` with the principal variation in SAN (attacker's shortest
    mate against the defender's longest defence, so ``len(line) ==`` the mate
    distance in plies), ``NO_MATE`` when no mate exists under the semantics in
    this module's docstring, or ``UNKNOWN`` when the node budget ran out --
    which is a statement about the search, never about the position.

    *attacker_moves* selects the question. The default ``"checks"`` asks the
    tsume one -- is there an all-checks mate -- and is byte-identical to what
    this function did before the selector existed. ``"all"`` asks the general
    forced-mate one, letting the attacker play quiet moves too: strictly more
    mates found, at a cost one to three orders of magnitude larger (measured
    on the corrected F09: 8 node expansions checks-only against 375 at depth
    3 and 9,447 at depth 5). ``NO_MATE`` under ``"checks"`` means "no
    all-checks mate within *max_plies*" and nothing stronger, which is the
    whole reason the other mode exists.

    *table* lets several solves share transposition state. It is safe across
    modes because :meth:`CyTsumeProblem.digest` tags the key with the mode; it
    is *not* safe across problems that differ in any way the digest does not
    capture. See :class:`~komugi.search.tsume_core.TsumeTable`.

    *max_plies* must be odd and >= 1; anything else is a ValueError rather
    than a quietly rounded-down search. So is a *fen* whose game is already
    over -- either marshal off the board -- which is not a tsume position at
    all and would otherwise come back as a free ``NO_MATE``. So, finally, is
    a *fen* whose **defender is already in check**: a tsume problem is one
    where the attacker's first move *delivers* the check, and when the check
    is already there the checks-only filter of
    :meth:`CyTsumeProblem._attacker_moves` admits nearly every pseudo-legal
    move, so the search reports a confident ``MATE`` whose principal
    variation opens with a move unrelated to the mate. Root only: the
    defender being in check at an *interior* AND node is the normal case and
    the whole search depends on it. That guard applies in both modes --
    ``"all"`` could in principle answer the already-in-check root soundly,
    but relaxing it is a separate question from adding the mode, and refusing
    the position keeps the two modes answering about the same set of roots.

    A *fen* **still in the draft** (``draft_w`` or ``draft_b`` set) is refused
    too, in both modes (KOM-282). During the draft the side to move does not
    strictly alternate, but the search takes OR/AND from depth parity, and
    ``cyengine.is_checkmate`` is constant-False there, so a ``MATE`` found
    from a draft root would contradict the engine by definition.
    """
    root = cyboard.board_from_fen(fen)
    if _marshal_missing(root):
        raise ValueError(
            f"not a tsume position (a marshal is off the board): {fen!r}"
        )
    if int(root.draft_w) or int(root.draft_b):
        raise ValueError(
            f"not a tsume position (the position is still in the draft): {fen!r}"
        )
    problem = CyTsumeProblem(
        root, verify_checkmate=verify_checkmate, attacker_moves=attacker_moves
    )
    # Asked of the problem rather than recomputed from ``root.turn`` here, so
    # that "the defender" means one thing in this module and cannot drift.
    if cyengine.in_check(root, problem.defender):
        raise ValueError(
            f"not a tsume position (the defender is already in check): {fen!r}"
        )
    return solve_problem(problem, root, max_plies, node_budget, table=table)
