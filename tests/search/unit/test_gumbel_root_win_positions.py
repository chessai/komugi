"""KOM-235: ``root_win_v1`` on real positions, and the draft it must not scan.

``test_gumbel_root_win.py`` says what the exhaustive root scan *is*, over the
KOM-219/220 toys. This module is the other half the KOM-234 review asked for
(``docs/reviews/2026-09-17-kom-234-root-win.md``, findings F1 and F2): the
production seam -- a real :class:`~komugi.selfplay.adapter.GameState`, real move
generation, the production
:class:`~komugi.selfplay.tactical.GameMarshalCaptureProvider` -- on three
positions whose only wins are of the three shapes the KOM-232 census found were
missed, plus the drafting position where the scan must not run at all.

The claims:

* **The positions are what the census said they are.** Each one's win set is
  recomputed here by an oracle written independently of the survey (clone, play,
  read the engine's own adjudication in the mover's frame) and asserted against
  the recorded ids, and the production provider names **nothing** at any of the
  three roots -- which is exactly why ``root_marshal_v1`` cannot see these wins;
* **``root_marshal_v1`` misses them** at both budgets and every seed, with the
  policy burying every win at ``-30`` logits (the census's failure mode, and the
  same stub shape the match-discovery tests use);
* **``root_win_v1`` plays one** -- a witness, attached for at most one
  clone/play per legal action, with no provider call at the root and one extra
  simulation on top of ``n``;
* **the draft is skipped** (F1): ``Game.win_state`` is all-zero while either
  side still holds drafting rights, so no probe taken there can be terminal and
  the scan is transitions spent on a provably empty result. At the advanced
  start the scan costs nothing; on a post-draft position it runs.

No checkpoint, no GPU, no shard. Trees are dropped as each search returns -- a
256-simulation tree over a 37341-wide action space is hundreds of megabytes --
so every assertion reads off the :class:`~komugi.search.SearchResult` and its
diagnostics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

# Before the engine is imported: ``Game`` reads ``GUNGI_MODE`` for its start
# mode, and every FEN below is an advanced-mode position.
os.environ.setdefault("GUNGI_MODE", "advanced")

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game
from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.selfplay.adapter import GameState
from komugi.selfplay.tactical import GameMarshalCaptureProvider

# ---------------------------------------------------------------------------
# The positions
# ---------------------------------------------------------------------------

#: The KOM-233 fixture: 400 sims, game 2, ply 68, white to move. Three block-C
#: **drops** mate; no capture wins, so the KOM-222 provider names nothing.
MISSED_WIN: Final[str] = (
    "3|g:n:s||r:n:t|1m2/1|f:A|1rc|j:w:s|3/f2|n:k|3|R:J|1/"
    "3j1|d:w|3/5|d:d:d|3/9/1G7/1|D:N:W|7/1|D:M|7 I1F2A1T1/- w 3 - 32"
)

#: The stacking mate: white to move, one win, and it is an **A1 stack** rather
#: than a move to an empty square or a capture.
STACK_MATE: Final[str] = (
    "1|f:s|2d1f|a:d|m/1wr|k:t|1|n:j|1sC/2A3d|n:M|n/J1cgA1a2/1d7/"
    "S1|i:T||I:j:G|1N1D1/2D1N2N1/D1WRW1J1S/F1|K:F:D|1R4 -/- w 3 - 75"
)

#: The quiet mate: black to move, two wins, both **A0 non-captures**. The shape
#: a capture-enumerating provider is blindest to.
QUIET_MATE: Final[str] = (
    "3rk3m/8d/d3|s:g|4/1RT2s1S1/|w:a|5|W:D|2/nA4J2/9/nM1j3n1/"
    "3|D:F|5 -/- b 3 - 172"
)

POSITIONS: Final[dict[str, str]] = {
    "missed_win": MISSED_WIN,
    "stack_mate": STACK_MATE,
    "quiet_mate": QUIET_MATE,
}

RECORDED_WINS: Final[dict[str, tuple[int, ...]]] = {
    "missed_win": (33144, 33145, 33155),
    "stack_mate": (14678,),
    "quiet_mate": (2640, 5411),
}
"""The win sets the KOM-232 census and the KOM-234 review recorded.

Asserted against the engine (:func:`test_the_positions_are_the_recorded_ones`)
rather than trusted: every claim below is "the search played a **winning** move",
and a fixture whose wins had drifted would still pass against its own oracle.
"""

NUM_ACTIONS: Final[int] = 37341
"""The engine's action-space width, as ``MatchConfig`` and the recorder spell it."""

MIN_PLACEMENTS: Final[int] = 10
"""The root view self-play and the anchor server search with (DRAFTMIN)."""

BUDGETS: Final[tuple[tuple[int, int], ...]] = ((8, 32), (32, 256))
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3)

SUPPRESS: Final[float] = 30.0
"""What the stub subtracts from every immediate win -- the census's failure mode.

The checkpoint gave the mate-in-1 it missed a prior of ``4.7e-4``; a ``-30``
logit against uniform zeros is the same statement, and it is what makes the
policy incapable of finding these wins on its own.
"""


# ---------------------------------------------------------------------------
# The oracle, written independently of the survey
# ---------------------------------------------------------------------------


def root_state(fen: str) -> GameState:
    """Return `fen` as the ``GameState`` self-play would search it from."""
    return GameState(
        Game(board_from_fen(fen)), 0.0, min_placements=MIN_PLACEMENTS, is_root=True
    )


def legal_ids(state: GameState) -> list[int]:
    """Return `state`'s legal action ids, ascending."""
    return [int(action) for action in np.flatnonzero(state.legal_mask())]


def _mover_lost(state: GameState) -> bool:
    """Return whether `state` is decided **against** the player to move in it.

    Read off :meth:`GameState.terminal_wdl`, the engine's outcome in the mover's
    frame, which shares no code with the provider's ``terminal_winner`` (that
    reads the absolute vector). So "the search played a win" is checked against
    a different reading of the engine than the one the survey verified with.
    """
    return bool(state.is_terminal() and float(state.terminal_wdl()[2]) == 1.0)


_WINS: dict[str, tuple[int, ...]] = {}


def winning_moves(fen: str) -> tuple[int, ...]:
    """Return every legal move of `fen` that ends the game in the mover's favour.

    Full width over the root's legal mask -- drops, stacks, ordinary moves,
    captures -- so it is the whole truth about the class, memoised by FEN
    because several tests ask for the same answer.
    """
    cached = _WINS.get(fen)
    if cached is not None:
        return cached
    root = root_state(fen)
    wins: list[int] = []
    for action in legal_ids(root):
        probe = root.clone()
        probe.play(action)
        if _mover_lost(probe):
            wins.append(action)
    answer = tuple(wins)
    _WINS[fen] = answer
    return answer


# ---------------------------------------------------------------------------
# The evaluator: uniform, except that every win is buried
# ---------------------------------------------------------------------------


_Answer = tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]


class StubEvaluator:
    """Uniform logits and a flat WDL, with every immediate win at ``-SUPPRESS``.

    No network and no position-dependence beyond the id list: what makes one
    search here differ from another is the tree, which is the thing under test.
    The suppression is applied at every node, so the policy cannot stumble onto
    one of these wins deeper either.

    The ``assert not state.is_terminal()`` is a claim about the survey: a
    witness is attached from the game's own outcome and must cost no network
    row, so an evaluator handed a finished position fails here rather than in a
    row count.
    """

    def __init__(self, suppressed: tuple[int, ...]) -> None:
        self._suppressed = np.asarray(suppressed, dtype=np.int64)
        self.calls = 0

    def __call__(self, state: SearchState) -> _Answer:
        assert not state.is_terminal(), "the evaluator was handed a terminal state"
        self.calls += 1
        logits: npt.NDArray[np.float32] = np.zeros(NUM_ACTIONS, dtype=np.float32)
        logits[self._suppressed] -= np.float32(SUPPRESS)
        wdl = np.full(3, 1.0 / 3.0, dtype=np.float64)
        return logits, 0.0, wdl


# ---------------------------------------------------------------------------
# One search, cached, with its tree already dropped
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    """One searched configuration: the position, the budget, the draw, the mode."""

    position: str
    m: int
    n: int
    seed: int
    mode: str


@dataclass(frozen=True)
class _Run:
    """One finished search: the result and the rows it cost."""

    result: SearchResult
    eval_rows: int


_RUNS: Final[dict[_Case, _Run]] = {}


def _search(case: _Case) -> _Run:
    """Return the search `case` names, computed once per configuration.

    Always ``proof_mode="interval_v1"``: a discovery mode needs a proof layer to
    consume what it attaches (``validate_search_modes``), which is the pairing
    self-play runs.
    """
    cached = _RUNS.get(case)
    if cached is not None:
        return cached
    fen = POSITIONS[case.position]
    evaluator = StubEvaluator(winning_moves(fen))
    search = GumbelSearch(
        evaluator,
        GumbelConfig(
            NUM_ACTIONS,
            case.m,
            case.n,
            case.seed,
            proof_mode="interval_v1",
            discovery_mode=case.mode,
        ),
        tactical_provider=GameMarshalCaptureProvider(),
    )
    run = _Run(result=search.search(root_state(fen)), eval_rows=evaluator.calls)
    _RUNS[case] = run
    return run


def _diagnostics(run: _Run) -> Any:
    """Return `run`'s diagnostics, refusing the ``None`` an active layer forbids."""
    diagnostics = run.result.diagnostics
    assert diagnostics is not None, "an active layer reported no diagnostics"
    return diagnostics


def _cases(mode: str) -> tuple[_Case, ...]:
    """Return every (position, budget, seed) this module searches under `mode`."""
    return tuple(
        _Case(position, m, n, seed, mode)
        for position in POSITIONS
        for m, n in BUDGETS
        for seed in SEEDS
    )


def _id(case: _Case) -> str:
    """Return a stable, readable parametrisation id for `case`."""
    return f"{case.position}-m{case.m}n{case.n}-seed{case.seed}"


# ---------------------------------------------------------------------------
# the positions themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", sorted(POSITIONS))
def test_the_positions_are_the_recorded_ones(position: str) -> None:
    """Each fixture parses, is past its draft, and wins exactly where recorded.

    And the production provider names **nothing** at any of them: that is the
    premise of the whole module -- ``root_marshal_v1`` consults the provider at
    the root and gets an empty list, so no amount of budget makes it verify one
    of these wins.
    """
    fen = POSITIONS[position]
    root = root_state(fen)
    assert not root.is_terminal()
    assert not root.game.in_draft(0) and not root.game.in_draft(1)

    wins = winning_moves(fen)
    assert wins, "a fixture with no immediate win pins nothing"
    assert wins == RECORDED_WINS[position], (
        f"{position}: the engine's win set moved away from the KOM-234 review's"
    )
    assert set(wins) <= set(legal_ids(root))
    assert (
        GameMarshalCaptureProvider().candidate_actions(root, root.legal_mask()) == ()
    ), "the provider named a candidate, so root_marshal_v1 is not blind here"


# ---------------------------------------------------------------------------
# root_marshal_v1 misses; root_win_v1 plays a win
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases("root_marshal_v1"), ids=_id)
def test_root_marshal_v1_misses_the_win(case: _Case) -> None:
    """The provider-driven root list finds none of these wins, at any budget.

    The red half of the module, run rather than asserted in prose: without it a
    mode that did nothing would be indistinguishable from one that worked. The
    provider names nothing at the root, so the survey verifies nothing there,
    and the policy has every win at ``-30``.

    **No ROOT witness**, which is the claim -- ``discovery_status`` is the field
    that says so. Not ``terminal_witnesses == 0``: the KOM-234 review's F2
    sketch wrote it that way, but the *child* survey legitimately attaches reply
    witnesses here (a candidate the opponent mates in reply), and it does --
    ``stack_mate`` at ``m=8, n=32`` seed 2 attaches two of them. Those are
    anti-decisive evidence about a move the search then excludes; they are not
    the decisive win this mode cannot find.
    """
    run = _search(case)
    wins = winning_moves(POSITIONS[case.position])
    diagnostics = _diagnostics(run)

    assert run.result.action not in wins
    assert diagnostics.discovery_status == "candidate_survey_complete", (
        "the provider-driven root list verified a win it cannot name"
    )
    assert diagnostics.root_scan == "provider"
    assert diagnostics.provider_calls >= 1, "the provider was never consulted"


@pytest.mark.parametrize("case", _cases("root_win_v1"), ids=_id)
def test_root_win_v1_plays_a_win(case: _Case) -> None:
    """The exhaustive scan finds the win, attaches it and plays it.

    Every field of the KOM-234 accounting in one place: the scan is what built
    the root list (``root_scan``), it asked the provider nothing at the root
    (``provider_calls == 0`` -- the child survey never runs, because the root
    survey returns at its witness), it spent at most one clone/play per legal
    action, and the witness cost exactly one simulation on top of ``n``.
    """
    fen = POSITIONS[case.position]
    run = _search(case)
    wins = winning_moves(fen)
    diagnostics = _diagnostics(run)
    num_legal = int(root_state(fen).legal_mask().sum())

    assert run.result.action in wins
    assert diagnostics.discovery_status == "root_witness"
    assert diagnostics.root_scan == "exhaustive"
    assert diagnostics.provider_calls == 0
    assert diagnostics.terminal_witnesses == 1
    assert 1 <= diagnostics.candidate_verification_transitions <= num_legal
    assert run.result.sims_used == case.n + 1
    assert run.result.sims_used == int(run.result.visit_counts.sum())
    # A witness is read off the game, so it costs no network row: the whole
    # search is the root plus at most one row per simulation.
    assert run.eval_rows <= 1 + case.n


# ---------------------------------------------------------------------------
# F1: the draft is not scanned
# ---------------------------------------------------------------------------


def test_the_exhaustive_scan_skips_the_draft() -> None:
    """At the advanced start the scan costs nothing, because it can find nothing.

    ``Game.win_state`` is all-zero while ``draft_w or draft_b``, so every one of
    the 27 offered drops would be played, found non-terminal and discarded --
    the KOM-234 review measured 7.5-9.0 s per game of exactly that, 45-50% of
    the game's scan cost, for a result that is empty by construction. The skip
    is per **position**, not per mover: either side still drafting suppresses
    every terminal.

    ``root_scan`` still reports ``"exhaustive"``: the rule the search ran under
    did not change, only what that rule found to do here (F6 -- read the status
    beside it).

    The root is built from :data:`~komugi.engine.fen.ADVANCED_POSITION` rather
    than from a bare ``Game()``: ``engine/game.py`` reads ``GUNGI_MODE`` **once,
    at import**, so in a full-suite run -- where something imported the engine
    before this module's ``setdefault`` -- ``Game()`` is the *intro* start, which
    has no draft and would make this a test of nothing. The FEN is the same
    position and depends on no environment.
    """
    start = root_state(ADVANCED_POSITION)
    assert start.game.in_draft(0) and start.game.in_draft(1)
    assert int(start.legal_mask().sum()) > 1, "the drafting root offers no choice"

    evaluator = StubEvaluator(())
    search = GumbelSearch(
        evaluator,
        GumbelConfig(
            NUM_ACTIONS,
            8,
            32,
            0,
            proof_mode="interval_v1",
            discovery_mode="root_win_v1",
        ),
        tactical_provider=GameMarshalCaptureProvider(),
    )
    result = search.search(start)
    diagnostics = result.diagnostics
    assert diagnostics is not None

    assert diagnostics.candidate_verification_transitions == 0
    assert diagnostics.terminal_witnesses == 0
    assert diagnostics.root_scan == "exhaustive"
    assert diagnostics.discovery_status == "candidate_survey_complete"


def test_the_exhaustive_scan_runs_once_the_draft_is_closed() -> None:
    """The other side of the skip: on a post-draft position the scan probes.

    Without this the F1 filter could be "never scan" and still pass everything
    above it.
    """
    case = _Case("stack_mate", 8, 32, 0, "root_win_v1")
    diagnostics = _diagnostics(_search(case))
    assert diagnostics.candidate_verification_transitions >= 1
    assert diagnostics.discovery_status == "root_witness"
