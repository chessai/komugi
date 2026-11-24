"""Regression tests for the KOM-258 review findings F1 and F2, and KOM-261.

All three are about a *question the solver should refuse to answer* rather
than about an answer it gets wrong, which is why none of them shows up
anywhere in the 248-test suite: the search returns something plausible in each
case.

**F1 (MEDIUM)** -- ``solve_tsume`` on a FEN whose game is already over (either
marshal off the board) used to return ``NO_MATE`` for free. The root's
``terminal()`` says DISPROVED, no move is ever generated, and the caller gets a
statement about a position that is not a tsume position at all. The fix rejects
such a FEN at the root; *interior* nodes keep the DISPROVED rule, because under
this module's defender semantics a captured marshal genuinely occurs as a child
and means the attack failed there.

**F2 (LOW)** -- ``Outcome.PROVED`` returned at an OR-parity node (attacker to
move) would be honoured as a mate, producing an even-length line and breaking
the odd-mate-distance contract that makes ``len(line)`` a grade. A win with the
attacker still to move is not a tsume mate. The fix raises at both places the
core reads a terminal outcome: ``_pre_expansion_value`` during the df-pn search
and ``_best_line`` during line extraction. PROVED at AND parity -- the
supported shortcut the synthetic trees use -- stays honoured.

**KOM-261** -- found by use rather than by review. A tsume problem starts with
the defender *not* in check: the attacker's first move is what delivers the
check. On a root where the defender is already in check, every attacker move
trivially leaves the defender in check, so the checks-only filter admits
essentially the whole pseudo-legal move list and the solver returns a
confident MATE whose principal variation opens with a move that has nothing to
do with the mate. The fix is the same shape as F1 -- reject at the root, leave
interior nodes alone, because the defender being in check at an interior AND
node is the normal case and every fixture depends on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest
from tsume_fixtures import ALL_FIXTURE_BOARDS, FRIEND

from komugi.engine import cyboard, cyengine
from komugi.engine.api import Gungi
from komugi.search.tsume import solve_tsume
from komugi.search.tsume_core import Outcome, solve_problem


@dataclass
class GraphProblem:
    """A hand-built AND/OR graph satisfying the solver's problem protocol.

    The same shape as ``TreeProblem`` in ``test_core_synthetic.py``, minus the
    digest aliasing these findings do not need: here a node *is* its own
    identity, so ``O0``/``A1``/``O3`` name both the position and the parity the
    search will meet it at.
    """

    edges: Mapping[str, Sequence[tuple[str, str]]]
    terminals: Mapping[str, Outcome] = field(default_factory=dict)
    #: Every node this problem was asked to expand, in order.
    expanded: list[str] = field(default_factory=list)

    def children(self, node: str) -> Sequence[tuple[str, str]]:
        self.expanded.append(node)
        return self.edges.get(node, ())

    def terminal(self, node: str) -> Outcome | None:
        return self.terminals.get(node)

    def digest(self, node: str) -> str:
        return node


# ---------------------------------------------------------------------------
# F1: a marshal-less root is not a tsume position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param("4m4/4g4/9/9/9/9/9/9/4G4 -/- w 3 - 1", id="no-white-M"),
        pytest.param("9/4g4/9/9/9/9/9/9/M3G4 -/- w 3 - 1", id="no-black-m"),
    ],
)
def test_a_marshal_less_fen_is_rejected_not_answered(fen: str) -> None:
    """The game is over on both of these, so neither verdict would be true.

    Answering ``NO_MATE`` here is worse than answering nothing: it is the same
    string the solver returns for a live position it searched and refuted, so a
    caller cannot tell a refutation from a malformed question.
    """
    with pytest.raises(ValueError):
        solve_tsume(fen, 1)


# ---------------------------------------------------------------------------
# F2: PROVED at OR parity is a contradiction, not a mate
# ---------------------------------------------------------------------------


def test_proved_at_or_parity_is_rejected_at_expansion() -> None:
    """The search itself walks into O3, so the ``_pre_expansion_value`` guard fires.

    ``O0 -> A2 -> O3`` with O3 PROVED: the only way to the root's proof runs
    through an OR node the problem calls won, which would make the mate
    distance two plies.
    """
    problem = GraphProblem(
        edges={"O0": (("O0>A2", "A2"),), "A2": (("A2>O3", "O3"),)},
        terminals={"O3": Outcome.PROVED},
    )
    with pytest.raises(ValueError):
        solve_problem(problem, "O0", 3, node_budget=10_000)


def test_proved_at_or_parity_is_rejected_during_line_extraction() -> None:
    """The same contradiction reached with a second, honest proof available.

    The root has a clean AND-parity proof through ``A0``, so df-pn may settle
    the root without ever needing O3's branch; whichever of the two guards the
    traversal reaches first, PROVED at OR parity must stop the solve rather
    than be folded into a line.
    """
    problem = GraphProblem(
        edges={"O0": (("O0>A2", "A2"), ("O0>A0", "A0")), "A2": (("A2>O3", "O3"),)},
        terminals={"O3": Outcome.PROVED, "A0": Outcome.PROVED},
    )
    with pytest.raises(ValueError):
        solve_problem(problem, "O0", 1, node_budget=10_000)


def test_a_proved_root_is_rejected() -> None:
    """An already-won root has no odd mate line: the attacker has not moved yet."""
    problem = GraphProblem(edges={}, terminals={"O0": Outcome.PROVED})
    with pytest.raises(ValueError):
        solve_problem(problem, "O0", 1, node_budget=10_000)


def test_proved_at_and_parity_still_shortcuts_to_mate() -> None:
    """The guard must not cost the supported shortcut the synthetic trees use."""
    problem = GraphProblem(
        edges={"O0": (("O0>A1", "A1"),)},
        terminals={"A1": Outcome.PROVED},
    )
    result = solve_problem(problem, "O0", 1, node_budget=10_000)
    assert result.verdict == "MATE"
    assert result.line == ("O0>A1",)


# ---------------------------------------------------------------------------
# KOM-261: a root where the defender is already in check is not a tsume
# ---------------------------------------------------------------------------

#: The F09 diagram (KOM-257) carrying the two edits of the 2026-09-21 pass-3
#: hand-piece probe: the white |兵:槍:大| tower is lifted off its printed
#: square (rank 4, file 4) and a 侍 samurai is put into white's hand, which is
#: what the probe was varying. Written out rank by rank because the artefact
#: below is a one-square edit to it and the edit has to be legible.
#:
#: FEN geometry: ranks run 1..9 top to bottom, files 9..1 left to right, so the
#: leftmost square of a rank field is file 9. ``|a:b:c|`` is a tower read
#: bottom-to-top, i.e. ``c`` is the top piece. Codes: m=帥 g=大 i=中 w=侍
#: n=槍 f=砦 d=兵, uppercase white.
_F09_PROBE_BASE_RANKS: Final = (
    "9",  # rank 1
    "3f|f:m|4",  # rank 2: black 砦 at file 6; black 帥 on tier 2 over a 砦 at file 5
    "9",  # rank 3: empty in the printed diagram
    "1g7",  # rank 4: black 大 at file 8 (the white tower is lifted off file 4)
    "3I2W2",  # rank 5: white 中 at file 6, white 侍 at file 3
    "9",  # rank 6
    "9",  # rank 7
    "9",  # rank 8
    "M8",  # rank 9: the white 帥 at file 9
)

#: Hand, side to move, mode, drafting, move number: a 侍 in white's hand,
#: nothing in black's, white to move, advanced mode, draft over.
_F09_PROBE_TAIL: Final = " W1/- w 3 - 1"


def _f09_probe_fen(rank3: str) -> str:
    """The probe base with rank 3 replaced by *rank3*."""
    ranks = list(_F09_PROBE_BASE_RANKS)
    ranks[2] = rank3
    return "/".join(ranks) + _F09_PROBE_TAIL


def test_the_f09_pass3_artefact_is_rejected() -> None:
    """The exact position that produced a fake mate-in-3, now refused.

    Pass 3 of the F09 hand-piece probe relocated the white |兵:槍:大| tower to
    rank 3 to unlock drops near the black marshal, and got nine "mate-in-3"
    hits. They were artefacts: putting the tower on rank 3 file 5 puts the
    black 帥 on rank 2 file 5 in check *before white has moved*, so the
    checks-only filter stops filtering. This position returned

        MATE  侍(5-3-1)(6-3-1)  帥(2-5-2)(4-3-1)  新侍(5-3-1)

    -- a line that opens with white's samurai stepping backwards, away from
    the black marshal, which is the tell. It must raise instead.

    The root check is asserted here through ``api.Gungi`` rather than assumed,
    so the test still means what it says if the FEN above is mistyped: a
    typo that lands somewhere without the root check fails loudly on the
    assertion instead of passing vacuously on the wrong position.
    """
    # File 5 is the fifth square from the left, so: four empties, the tower,
    # four empties. |D:N:G| is 兵 (tier 1), 槍 (tier 2), 大 (tier 3).
    fen = _f09_probe_fen("4|D:N:G|4")

    game = Gungi(fen)
    assert str(game.turn()) == "w"
    assert bool(game.in_check("b")), f"probe FEN does not check black: {fen!r}"
    assert not bool(game.in_check("w")), "white must not be the one in check"

    with pytest.raises(ValueError, match="already in check"):
        solve_tsume(fen, 3)


def test_the_f09_probe_base_is_still_answered() -> None:
    """The same position with the tower left off rank 3 is a legal question.

    The pair matters: it shows the rejection is caused by the one-square edit
    that creates the root check, not by anything else about the probe FEN.
    """
    fen = _f09_probe_fen("9")
    assert not bool(Gungi(fen).in_check("b"))
    assert solve_tsume(fen, 3).verdict == "NO_MATE"


def test_a_minimal_already_in_check_root_is_rejected() -> None:
    """The same guard on the smallest position that can trigger it.

    Three pieces: the black 帥 on rank 2 file 5, a white 兵 directly in front
    of it on rank 3 file 5 giving the check, and the white 帥 tucked at rank 9
    file 9 so the board has its two marshals and ``is_checkmate`` is not
    silently constant-False. White to move, and white's check is already
    delivered, so there is no tsume to pose.
    """
    fen = "9/4m4/4D4/9/9/9/9/9/M8 -/- w 3 - 1"
    game = Gungi(fen)
    assert bool(game.in_check("b"))
    assert not bool(game.in_check("w"))

    with pytest.raises(ValueError, match="already in check"):
        solve_tsume(fen, 1)


#: Committed fixtures whose ROOT already has the defender in check, and which
#: the KOM-261 guard therefore rejects. Empty, and required to stay empty.
#:
#: It was not empty when the guard landed. ``iter_827-g200002`` -- a mined
#: mate-in-1, black to move, with a black 小 sitting next to the white 帥 at
#: (8-8-1) -- had exactly that root, which is not a mis-transcription but
#: precisely the case KOM-261's own "blast radius" note predicts: this engine
#: permits moving into self-check, so a side can leave its marshal attacked
#: and the game goes on, and the miner scanning real self-play found one.
#:
#: Its recorded mate-in-1 was genuine (eight enumeration keys, all real
#: checkmates), so the question it raised was a fixture question, not a solver
#: one, and it was settled the same way F09 was: the row is QUARANTINED out of
#: ``fixtures/mined.json`` (KOM-261), because a position whose defender is
#: already in check is not a tsume problem whatever its enumeration says. The
#: evidence did not go with it -- ``test_the_quarantined_mined_row_is_rejected``
#: below keeps the FEN as a literal and re-asserts both halves of the finding.
#:
#: So the set is an over-reach guard again, in both directions: a fixture that
#: starts raising, or a quarantined root sneaking back into the committed set,
#: is a named set difference here rather than a pile of failures elsewhere.
_ALREADY_IN_CHECK_FIXTURE_ROOTS: Final = frozenset[str]()


def test_which_committed_fixtures_have_an_already_in_check_root() -> None:
    """The guard's blast radius over the committed set, stated exactly.

    Runs the guard's own predicate -- ``cyengine.in_check`` against the side
    *not* to move -- over every committed fixture board, so an over-reach (a
    friend or mined row silently starting to raise) shows up as a set
    difference naming the row rather than as a pile of failures elsewhere.
    """
    found = frozenset(
        name
        for name, fen in ALL_FIXTURE_BOARDS
        if cyengine.in_check(
            (board := cyboard.board_from_fen(fen)), 1 - int(board.turn)
        )
    )
    assert found == _ALREADY_IN_CHECK_FIXTURE_ROOTS


#: The FEN of the row the line above used to name: mined from self-play
#: iteration 827, game 200002, position 28, recorded as ``iter_827-g200002``
#: with eight verified enumeration keys, and quarantined out of
#: ``fixtures/mined.json`` by KOM-261. Black to move; the white 帥 at (8-8-1)
#: is already attacked by the black 小 next to it, so the DEFENDER is in check
#: before the attacker has moved. Kept here as a literal precisely because it
#: is no longer in the fixture file: the position is the only surviving copy
#: of the evidence that this case occurs in real self-play data rather than
#: only in hand-edited diagrams.
_QUARANTINED_MINED_ROOT: Final = (
    "m|d:T|7/|w:s||d:d:i|7/|K:F||C:n|1S5/9/1S7/1R1|n:j|5/"
    "|D:D:R||k:a:a|7/Mj7/1W7 I1J2F1A1/g1r1f2 b 3 - 28"
)


def test_the_quarantined_mined_row_is_rejected() -> None:
    """The mined already-in-check root, still refused after leaving the fixtures.

    Two halves, because either alone could pass for the wrong reason. First
    that the position really is what it is claimed to be -- ``api.Gungi``,
    not the guard's own predicate, agrees that black is to move and that
    white is already in check -- so a mistyped FEN fails here instead of
    quietly making the second half trivial. Then that ``solve_tsume``
    raises: the row's mate-in-1 is genuine under enumeration, but the
    question "what mates from here" is not a tsume question at this root and
    must not be answered.
    """
    game = Gungi(_QUARANTINED_MINED_ROOT)
    assert str(game.turn()) == "b"
    assert bool(game.in_check("w")), "the defender must be the one in check"
    assert not bool(game.in_check("b")), "the attacker must not be in check"

    with pytest.raises(ValueError, match="already in check"):
        solve_tsume(_QUARANTINED_MINED_ROOT, 1)


def test_the_quarantined_mined_row_left_the_committed_fixtures() -> None:
    """Quarantine means gone from the fixture set, not merely excused.

    The set above is empty only because this position is no longer loaded;
    stated separately so that re-adding the row fails as "the quarantined
    root is back" rather than as an unexplained ValueError in whichever
    parametrized test reaches it first.
    """
    assert "iter_827-g200002" not in {name for name, _ in ALL_FIXTURE_BOARDS}
    assert _QUARANTINED_MINED_ROOT not in {fen for _, fen in ALL_FIXTURE_BOARDS}


def test_a_fixture_still_solves_with_the_defender_in_check_at_every_and_node() -> None:
    """Interior nodes are untouched: the guard is a root-only input check.

    F11 is a 5手, so its line has two interior AND nodes, and the defender is
    in check at both of them -- the same predicate the new guard evaluates at
    the root, True throughout the tree. The solve is run with
    ``verify_checkmate=True`` so the engine's own ``is_checkmate`` is asserted
    to agree at every AND node along the way.
    """
    fixture = next(f for f in FRIEND if f.name == "F11-5move")

    board = cyboard.board_from_fen(fixture.fen)
    assert not cyengine.in_check(board, 1 - int(board.turn))

    result = solve_tsume(fixture.fen, fixture.tsume_plies, verify_checkmate=True)
    assert result.verdict == "MATE"
    assert len(result.line) == fixture.tsume_plies

    game = Gungi(fixture.fen)
    attacker = str(game.turn())
    defender = "b" if attacker == "w" else "w"
    and_nodes = 0
    for ply, san in enumerate(result.line):
        game.move(san)  # type: ignore[no-untyped-call]  # legacy api.Gungi is untyped
        if ply % 2 == 0 and ply + 1 < len(result.line):
            assert bool(game.in_check(defender)), f"ply {ply + 1} ({san}) is quiet"
            and_nodes += 1
    assert and_nodes == 2
