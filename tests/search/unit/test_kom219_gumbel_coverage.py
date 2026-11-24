"""KOM-219: the audited tactical-coverage limits of the Gumbel search, pinned.

The campaign-6 audit of ``GumbelSearch`` found three *tactical* limits and one
contract that holds. The limits are **missing guarantees in the documented
algorithm**, not violated ones:

* **C1, root candidate exclusion.** Sequential halving can only ever play an
  action the root's Gumbel draw sampled. A winning action the draw missed is
  unreachable no matter how large ``n`` is, because ``n`` buys depth inside the
  candidate set and never widens it.
* **C2, interior selection missing a capture.** An interior node ranks its
  replies on ``prior + sigma(completed Q)``; a reply the network's prior buries
  is never visited within a small budget, so a refuted move keeps a clean score.
* **Late discovery.** A reply found *after* the halving has eliminated the safe
  alternative cannot bring it back: the survivor is played even once its own Q
  has gone negative, and the recorded policy target then favours a move the
  search will not play.

The contract that holds is the terminal bypass: a state the search *reaches*
that is terminal is scored by the game and never handed to the evaluator, and
its outcome is backed up in the correct mover frame -- in scalar and in wave
execution alike.

That audit ran once, by hand, and runs nowhere in CI. This module is its
executable form: every number below is the behaviour of ``main`` today, so the
repairs planned as KOM-220 (root candidate admission), KOM-221 (interior
tactical coverage) and KOM-222 (late-discovery recovery) have a baseline that
fails loudly when they land, rather than a memory of what used to happen.

Each test's docstring says which kind of claim it makes. A **MISSING GUARANTEE**
failing is progress -- it means a repair landed and this file needs rewriting
around the new behaviour. An **UPHELD CONTRACT** failing is a regression.

Nothing here loads a checkpoint, touches the engine or leaves the CPU: the
fixtures are two toy states whose whole tactical content is a single winning
action.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.search import gumbel

_WIDTH: Final[int] = 80
"""The toy action space, wide enough that ``m`` never covers the legal set."""

_ROOT_LEGAL: Final[int] = 64
"""Legal actions at the C1 root -- the winning one is the last of them."""

_ROOT_WIN: Final[int] = 63
"""The root action that wins on the spot, and that the draw rarely samples."""

_INTERIOR_WIN: Final[int] = 76
"""The opponent reply that refutes interior action 0."""

_FAST: Final[tuple[int, int]] = (8, 32)
"""The campaign's fast-search budget, ``(m, n)``."""

_FULL: Final[tuple[int, int]] = (32, 256)
"""The campaign's full-search budget, ``(m, n)``."""

_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 7, 11, 42, 1234)
"""The audited seed sweep, reproduced exactly so the counts below are checkable."""

_Answer = tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]
"""One evaluation: raw logits, the mover's scalar, and the mover's WDL vector."""


class _CoverageState:
    """A toy position whose only tactical content is one winning action.

    Two kinds, each the fixture for one audited limit:

    * ``"root"`` -- 64 legal actions, every one of them immediately terminal.
      Action :data:`_ROOT_WIN` wins for the mover; the other 63 draw. This is
      C1's fixture: whether the win is *found* is decided entirely by the root
      draw, before a single simulation is spent.
    * ``"interior"`` -- two legal actions. Action 1 draws immediately; action 0
      enters an opponent node with 77 replies, of which reply
      :data:`_INTERIOR_WIN` wins for the *opponent* and the other 76 are quiet.
      This is C2's fixture: the refutation is one ply below the root, inside a
      node the search reaches on nearly every simulation.

    ``same`` makes ply 0 a pass, leaving the mover unchanged. It is not used by
    the tests below; it exists so a future repair's test can build the
    same-mover frame without another state class.
    """

    NUM_ACTIONS: Final[int] = _WIDTH

    def __init__(
        self,
        kind: str,
        history: tuple[int, ...] = (),
        same: bool = False,
    ) -> None:
        self.kind = kind
        self.history = history
        self.same = same

    def clone(self) -> _CoverageState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        return _CoverageState(self.kind, self.history, self.same)

    def to_play(self) -> int:
        """Return the mover, alternating every ply unless ply 0 was a pass."""
        plies = max(0, len(self.history) - 1) if self.same else len(self.history)
        return plies % 2

    def is_terminal(self) -> bool:
        """Return whether the game is over: every root child is, and two interior
        lines are -- the immediate draw, and the opponent's refutation."""
        if not self.history:
            return False
        return (
            self.kind == "root"
            or self.history[0] == 1
            or (len(self.history) >= 2 and self.history[1] == _INTERIOR_WIN)
        )

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` outcome in the frame of the mover *here*."""
        win = (self.history[0] == _ROOT_WIN) if self.kind == "root" else False
        draw = (self.kind == "root" and not win) or (
            self.kind == "interior" and self.history[0] == 1
        )
        if draw:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if win == (self.to_play() == 0):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def terminal_value(self) -> float:
        """Return ``w - l`` of :meth:`terminal_wdl`, i.e. the same outcome scalar."""
        wdl = self.terminal_wdl()
        return float(wdl[0] - wdl[2])

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the first ``K`` actions as legal, or nothing at all if terminal."""
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.bool_)
        if self.is_terminal():
            return mask
        if self.kind == "root":
            width = _ROOT_LEGAL
        elif not self.history:
            width = 2
        elif len(self.history) == 1:
            width = _INTERIOR_WIN + 1
        else:
            width = 1
        mask[:width] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        assert bool(self.legal_mask()[action]), (
            f"search played illegal action {action} at history {self.history}"
        )
        self.history = (*self.history, action)


class _CoverageEvaluator:
    """A network stub with two priors, a call counter and a wave log.

    ``"uniform"`` has no opinion at all, so what the search does is a statement
    about the *search*. ``"misleading"`` buries exactly the tactical action --
    :data:`_ROOT_WIN` at a root state, :data:`_INTERIOR_WIN` at the opponent
    node -- under a ``-30`` logit, which is what a network that has not learned
    the pattern looks like. The value is ``+1`` for whoever is to move, so no
    leaf evaluation can supply the tactical signal the prior withheld.

    Handing this stub a terminal state is an error rather than a datum: the
    terminal bypass is one of the contracts under test, and a regression that
    evaluated a reached terminal must fail rather than quietly average its
    invented value into the tree.
    """

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self.calls = 0
        self.wave_sizes: list[int] = []

    def __call__(self, state: SearchState) -> _Answer:
        """Return `(logits, value, wdl)` for `state`, recording that it was asked."""
        assert isinstance(state, _CoverageState), "a foreign state reached the seam"
        assert not state.is_terminal(), "terminal descendant evaluated"
        self.calls += 1
        logits = np.zeros(_WIDTH, dtype=np.float32)
        if self.policy == "misleading":
            if state.kind == "root":
                logits[_ROOT_WIN] = -30.0
            elif len(state.history) == 1:
                logits[_INTERIOR_WIN] = -30.0
        value = 1.0 if state.to_play() == 0 else -1.0
        wdl = np.array(
            [1.0, 0.0, 0.0] if value == 1.0 else [0.0, 0.0, 1.0], dtype=np.float64
        )
        return logits, value, wdl

    def wave(self, states: Sequence[SearchState]) -> Sequence[_Answer]:
        """Answer a whole wave positionally, recording how wide it was."""
        self.wave_sizes.append(len(states))
        return [self(state) for state in states]


class _RootCapture:
    """The root node of the last search, taken from the readouts seam.

    :class:`SearchResult` reports visit counts but not the per-edge Q the
    halving ranked on, and "the played move's own Q went negative" is a claim
    about that accumulator. This installs itself over
    :func:`komugi.search.gumbel._wdl_readouts`, records the root it is handed
    and calls through, so the search under test is the real one -- the idiom
    ``tests/search/unit/test_gumbel_unit.py`` already uses.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root: object = None
        inner = gumbel._wdl_readouts

        def spy(
            root: gumbel._Node, improved: npt.NDArray[np.float64], played: int
        ) -> tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
        ]:
            """Record the root, then produce the readouts for real."""
            self.root = root
            return inner(root, improved, played)

        monkeypatch.setattr(gumbel, "_wdl_readouts", spy)

    @property
    def node(self) -> gumbel._Node:
        """Return the captured root, refusing to pretend when nothing ran."""
        assert isinstance(self.root, gumbel._Node), "no search reached the readouts"
        return self.root


def _edge_q(node: gumbel._Node, action: int) -> float:
    """Return the mean backed-up value of `node`'s edge `action`, in its frame."""
    visits = int(node.child_visits[action])
    assert visits > 0, f"edge {action} was never visited, so it has no Q"
    return float(node.child_value_sums[action]) / visits


def _search(
    kind: str,
    *,
    policy: str,
    m: int,
    n: int,
    seed: int,
    wave: bool,
) -> tuple[SearchResult, _CoverageEvaluator]:
    """Search a fresh `kind` fixture, in scalar or wave execution."""
    evaluator = _CoverageEvaluator(policy)
    search = GumbelSearch(
        evaluator,
        GumbelConfig(num_actions=_WIDTH, m=m, n=n, seed=seed),
        wave_evaluator=evaluator.wave if wave else None,
    )
    return search.search(_CoverageState(kind)), evaluator


@pytest.mark.parametrize("wave", (False, True))
def test_root_candidate_exclusion_cannot_be_fixed_by_more_simulations(
    wave: bool,
) -> None:
    """MISSING GUARANTEE (C1, KOM-220): a win the root draw never sampled stays
    unplayable, and multiplying the simulation budget by eight changes nothing.

    64 legal actions, action 63 wins outright, and the prior puts ``-30`` on it.
    Across the audited seeds the draw samples it on **0 of 8** seeds at the fast
    budget and **0 of 8** at the full budget, so its visit count is zero every
    time. That is the shape of the limit: ``n`` buys depth *inside* the
    candidate set, and the candidate set was fixed before the first simulation.
    The repair has to widen the set (admit an action on a tactical signal), not
    deepen it.
    """
    sampled = {}
    for label, (m, n) in (("fast", _FAST), ("full", _FULL)):
        found = 0
        for seed in _SEEDS:
            result, _ = _search(
                "root", policy="misleading", m=m, n=n, seed=seed, wave=wave
            )
            assert result.sims_used == n
            assert int(result.visit_counts[_ROOT_WIN]) == 0, (
                f"{label} seed {seed} sampled the winning action; the C1 "
                f"baseline was written when no seed did"
            )
            assert result.action != _ROOT_WIN
            found += int(int(result.visit_counts[_ROOT_WIN]) > 0)
        sampled[label] = found

    assert sampled == {"fast": 0, "full": 0}


@pytest.mark.parametrize("wave", (False, True))
def test_a_sampled_immediate_win_is_played_with_exact_certainty(
    wave: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UPHELD CONTRACT: once the win *is* a candidate, the search plays it and
    reports the real outcome -- exactly, not approximately.

    The positive control for this module. Without it the file only establishes
    that the search is bad at finding tactics; with it, the failure C1 describes
    is isolated to *sampling*, because every seed that samples action 63 plays
    it, reads its edge Q as exactly ``+1`` and reports ``played_wdl == (1, 0,
    0)`` in the root mover's frame. The sweep runs both budgets because under a
    uniform prior the win is sampled on only a minority of seeds at either one;
    the final assertion refuses to let the test pass on an empty sweep.
    """
    capture = _RootCapture(monkeypatch)
    qualifying = 0

    for m, n in (_FAST, _FULL):
        for seed in _SEEDS:
            result, _ = _search(
                "root", policy="uniform", m=m, n=n, seed=seed, wave=wave
            )
            if int(result.visit_counts[_ROOT_WIN]) == 0:
                continue
            qualifying += 1
            assert result.action == _ROOT_WIN, (
                f"seed {seed} at (m={m}, n={n}) sampled an immediate win and "
                f"played {result.action} instead"
            )
            assert _edge_q(capture.node, _ROOT_WIN) == 1.0
            assert np.array_equal(
                result.played_wdl, np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )

    assert qualifying > 0, "no seed sampled the win, so this control proved nothing"


@pytest.mark.parametrize("wave", (False, True))
def test_misleading_interior_capture_is_unvisited(
    wave: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MISSING GUARANTEE (C2, KOM-221): an interior refutation the prior buries
    is never visited, so the refuted move keeps a clean score and is played.

    Root action 0 walks into an opponent node where reply 76 wins on the spot
    and the prior puts ``-30`` on it. At the fast budget that reply gets **zero**
    visits: interior selection ranks on ``prior + sigma(completed Q)`` and the
    completion cannot know about a line nobody looked at. The safe action 1 -- an
    immediate draw -- is available the whole time and is not what comes back.
    """
    capture = _RootCapture(monkeypatch)
    m, n = _FAST

    result, _ = _search(
        "interior", policy="misleading", m=m, n=n, seed=0, wave=wave
    )

    child = capture.node.children[0]
    assert int(child.child_visits[_INTERIOR_WIN]) == 0, (
        "the opponent's winning reply was visited; the C2 baseline was written "
        "when it was not"
    )
    assert result.action == 0
    assert result.sims_used == n


@pytest.mark.parametrize("wave", (False, True))
def test_late_discovery_keeps_the_losing_survivor(
    wave: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MISSING GUARANTEE (late discovery, KOM-222): a refutation found after the
    halving has eliminated the safe move cannot bring it back.

    At the full budget the opponent's reply 76 *is* reached -- late, once its 76
    quiet siblings have each been expanded once (78 evaluations: the root, the
    opponent node, and those siblings; the terminal refutation itself is never
    evaluated). From then on it absorbs the rest of action 0's visits and drags
    its edge Q below the drawing action 1's zero. But the halving finished
    before that happened, so action 0 is still the move played, and the policy
    target -- built from the *final* Q -- puts its whole mass on action 1, a move
    this search has already eliminated. The label and the move disagree.
    """
    capture = _RootCapture(monkeypatch)
    m, n = _FULL

    result, _ = _search("interior", policy="uniform", m=m, n=n, seed=0, wave=wave)

    root = capture.node
    assert result.action == 0
    assert [int(root.child_visits[a]) for a in (0, 1)] == [192, 64]
    assert _edge_q(root, 0) < _edge_q(root, 1)
    assert float(result.policy_target[1]) == 1.0
    assert result.sims_used == n


@pytest.mark.parametrize("wave", (False, True))
def test_a_reached_terminal_bypasses_the_evaluator(wave: bool) -> None:
    """UPHELD CONTRACT: a terminal the search reaches is scored by the game, not
    by the network -- in scalar and in wave execution alike.

    Every child of the C1 root is terminal, so the only position that can
    legitimately be evaluated is the root itself: one call, while the search
    still spends all 32 simulations backing up real outcomes. The stub raises on
    a terminal input, so a regression that sent one to the network fails here
    with ``terminal descendant evaluated`` rather than by silently averaging an
    invented value into the tree. This is the guarantee the three missing ones
    above are measured against: the search is not wrong about what it reaches,
    only about what it reaches for.
    """
    m, n = _FAST

    result, evaluator = _search(
        "root", policy="uniform", m=m, n=n, seed=0, wave=wave
    )

    assert evaluator.calls == 1
    assert result.sims_used == n
    assert int(result.visit_counts.sum()) == n
