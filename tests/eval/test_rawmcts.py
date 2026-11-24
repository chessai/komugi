"""§D6.3's net-free ruler: the module boundary, the seed, and the state seam.

Four nodes of ``docs/design/kom-011d-ladder-pool.md`` §6 live here -- 29, 30,
31 and 54. The file itself is deliberately **torch-tolerant**: node 54 builds a
real :class:`komugi.selfplay.adapter.GameState`, which pulls torch in through
``adapter.py:63``. That is exactly why node 29 asks a *spawned interpreter*
about the module rather than asking ``sys.modules`` here -- by the time this
file's own imports have run, the answer would be about the test, not about
:mod:`komugi.eval.rawmcts`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.game import Game
from komugi.eval.rawmcts import MctsState, RawMctsMover
from komugi.selfplay.adapter import GameState

#: ``<repo>`` -- this file is ``<repo>/tests/eval/test_rawmcts.py``.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

PROTOCOL_WITNESS: MctsState = GameState(Game(), 0.0)
"""Node 54's **static** half: the real state, annotated as the protocol.

``mypy --strict`` (``ci/check.sh:19``) checks this assignment structurally, so
a :class:`~komugi.eval.rawmcts.MctsState` that named an operation the real
:class:`~komugi.selfplay.adapter.GameState` does not have -- rev 2's
``legal_actions`` and ``result`` -- fails the gate here, before any test runs.
A module-level binding rather than a local one so that the check is part of
the file's own type-checking surface and cannot be deleted by narrowing a test.
"""


# ---------------------------------------------------------------------------
# node 29 -- the module boundary, asked of a fresh interpreter
# ---------------------------------------------------------------------------

_PURE_PROBE: Final[str] = '''
"""Import the ruler, SEARCH with it, and report whether torch arrived."""
import sys

import numpy as np

from komugi.eval.rawmcts import RawMctsMover


class Toy:
    """The six MctsState names over a two-ply, two-action game."""

    def __init__(self, path=()):
        self.path = path

    def legal_mask(self):
        mask = np.zeros(2, dtype=np.bool_)
        if not self.is_terminal():
            mask[:] = True
        return mask

    def play(self, action):
        self.path = (*self.path, int(action))

    def clone(self):
        return Toy(self.path)

    def is_terminal(self):
        return len(self.path) >= 2

    def terminal_value(self):
        return 1.0 if self.path[-1] == 0 else -1.0

    def to_play(self):
        return len(self.path) % 2


mover = RawMctsMover(sims=4)
mover.pick(Toy(), ply=0, seed=1)
mover.observe(0, None)
mover.reset(pair=0, seat=0)
mover.close()
print("torch" in sys.modules)
'''

_ENTRANT_PROBE: Final[str] = """
import sys

import komugi.eval.rawmcts_entrant  # noqa: F401

print("torch" in sys.modules)
"""


def _torch_after(program: str) -> bool:
    """Return whether `program` leaves torch in a **fresh** interpreter's modules.

    The spawned-child idiom (``kom-036-fast-mode-ab.md:1218-1222``): in-process,
    pytest's collection order decides whether torch is already resident, so an
    ``assert "torch" not in sys.modules`` inside the suite tests the ordering
    rather than the import graph.
    """
    environment = dict(os.environ)
    source = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{source}{os.pathsep}{existing}" if existing else source
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    answer = completed.stdout.strip().splitlines()[-1]
    assert answer in ("True", "False"), completed.stdout
    return answer == "True"


def test_rawmcts_imports_no_torch() -> None:
    """§6 node 29: the ruler's search is torch-free; its entrant seam is not.

    The child does not merely import: it builds a mover and **runs a search**
    before it looks at ``sys.modules``. §5's rule, as rev 3 M5 restates it, is
    the stronger property -- no torch at import time *and* none through a
    deferred, function-body import -- and import-time purity alone is satisfied
    by exactly the mutant that rule names (``def pick(...): from
    komugi.eval.arena import ...``). A probe that only imported would call that
    mutant clean.

    Non-vacuity is the second half. A node that only ever saw the clean side
    would pass against a tree in which *nothing* imports torch, and could not
    tell §D6.3's deliberate module boundary from an accident: the entrant
    returns a ``BridgeEntrant``, and ``v1bridge.py:67`` -> ``arena.py:81`` is a
    torch import that cannot be removed. So the wrong implementation the second
    assertion is red against is the one that moves ``rawmcts_entrant``'s
    factory into ``rawmcts.py`` -- which would make the first assertion fail
    and this one pass, and the pair of them is what says the boundary is where
    the design put it.
    """
    assert not _torch_after(_PURE_PROBE)
    assert _torch_after(_ENTRANT_PROBE)


# ---------------------------------------------------------------------------
# a toy state: three symmetric moves, and a value that is pure rollout noise
# ---------------------------------------------------------------------------


class _ToyState:
    """A depth-4, branching-3 game whose leaf value ignores the opening.

    Built for nodes 30 and 31, and built *symmetric on purpose*: the terminal
    value depends only on the **last** action played, so every root action
    leads to an identically-valued subtree and the visit counts a search ends
    up with are a function of its rollout draws and nothing else. A toy whose
    root actions differed in true value would let a search converge on the best
    one whatever its seed, and node 30's second half could then pass against a
    mover that never looked at the seed at all.

    It implements exactly :class:`~komugi.eval.rawmcts.MctsState` -- the same
    six names, so the protocol is exercised by a second, independent
    implementation rather than only by the real engine.
    """

    BRANCHING: Final[int] = 3
    DEPTH: Final[int] = 4

    def __init__(self, path: tuple[int, ...] = ()) -> None:
        self._path = path

    @property
    def path(self) -> tuple[int, ...]:
        """Return the actions played so far, for a test to assert against."""
        return self._path

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return every action as legal, or none at all once terminal."""
        mask = np.zeros(self.BRANCHING, dtype=np.bool_)
        if not self.is_terminal():
            mask[:] = True
        return mask

    def play(self, action: int) -> None:
        """Append `action` to the path, in place."""
        self._path = (*self._path, int(action))

    def clone(self) -> _ToyState:
        """Return an independent copy; the path tuple is immutable."""
        return _ToyState(self._path)

    def is_terminal(self) -> bool:
        """Return whether :data:`DEPTH` actions have been played."""
        return len(self._path) >= self.DEPTH

    def terminal_value(self) -> float:
        """Return ``+1`` to the mover here when the last action was ``0``."""
        if not self._path:
            return 0.0
        return 1.0 if self._path[-1] == 0 else -1.0

    def to_play(self) -> int:
        """Return the mover: the two seats strictly alternate here."""
        return len(self._path) % 2


_TOY_SIMS: Final[int] = 12
"""Simulations per pick in the toy nodes: enough for the visit counts to
separate, small enough that the whole file is milliseconds."""

_SEEDS: Final[tuple[int, ...]] = tuple(range(1, 25))
"""A fixed seed set for node 30's second half, so the node is deterministic.

Twenty-four rather than two: the assertion is that the seed -> action map is
**not constant**, and on a toy whose root actions are equal in value each seed
lands somewhere by rollout noise. A two-seed version of the same claim would
be a coin flip written as an assertion -- green or red by luck, and pinned
either way once it is committed.
"""


# ---------------------------------------------------------------------------
# node 30 -- the seam's seed, and nothing else
# ---------------------------------------------------------------------------


def test_rawmcts_is_reproducible_from_the_seam_seed_alone() -> None:
    """§6 node 30: same seed, same action; different seeds, not all the same.

    Two wrong implementations this is red against, one per half:

    * the first half is red against a mover that draws from module-level
      randomness (``np.random.default_rng()``, ``random``) or that mixes `ply`
      into its seed -- the two picks below use the **same** seed at different
      plies and must agree, which is the ``gumbel_mover.py:311-314`` rule that
      the ply is the harness's coordinate and the seed is the randomness;
    * the second half is red against a mover that accepts a seed and ignores
      it, i.e. one that is deterministic by construction. On this toy every
      root action has the same true value, so a seed-blind search returns one
      constant action for all twenty-four seeds and the set below is a
      singleton.
      The set form is what makes the node deterministic: asserting that two
      *particular* seeds disagree would be a coin flip written as an assertion.
    """
    state = _ToyState()
    first = RawMctsMover(sims=_TOY_SIMS).pick(state, ply=0, seed=90210)
    again = RawMctsMover(sims=_TOY_SIMS).pick(state, ply=7, seed=90210)
    assert first == again
    assert state.path == (), "pick must not mutate the referee's position"

    chosen = {
        RawMctsMover(sims=_TOY_SIMS).pick(state, ply=0, seed=seed)
        for seed in _SEEDS
    }
    assert len(chosen) > 1


# ---------------------------------------------------------------------------
# the value frame: a toy where one root action wins and the others lose
# ---------------------------------------------------------------------------


class _DecisiveState:
    """A game decided entirely by the action played at :attr:`decision_ply`.

    The player to move at ``decision_ply`` **wins** by playing ``winning`` and
    **loses** by playing anything else; one further ply of noise follows, so
    every simulation reaches its terminal through a :func:`rollout
    <komugi.eval.rawmcts._rollout>` rather than through the leaf branch. That
    puts both halves of §D6.3's negamax discipline under the node --
    ``rawmcts.py:244``'s *"mover-relative"* rollout frame and
    ``rawmcts.py:455-459``'s backup negation -- rather than only the second.

    Deliberately **asymmetric**, which is the property :class:`_ToyState` is
    built to lack: there, every root action has the same true value, so a sign
    inversion in either place is invisible. Here a search whose value frame is
    inverted prefers the losing actions, and says so in its visit counts.
    """

    BRANCHING: Final[int] = 3

    def __init__(
        self,
        *,
        winning: int,
        decision_ply: int = 0,
        path: tuple[int, ...] = (),
    ) -> None:
        self._winning = winning
        self._decision_ply = decision_ply
        self._path = path

    @property
    def path(self) -> tuple[int, ...]:
        """Return the actions played so far, for a test to assert against."""
        return self._path

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return every action as legal, or none at all once terminal."""
        mask = np.zeros(self.BRANCHING, dtype=np.bool_)
        if not self.is_terminal():
            mask[:] = True
        return mask

    def play(self, action: int) -> None:
        """Append `action` to the path, in place."""
        self._path = (*self._path, int(action))

    def clone(self) -> _DecisiveState:
        """Return an independent copy; the path tuple is immutable."""
        return _DecisiveState(
            winning=self._winning,
            decision_ply=self._decision_ply,
            path=self._path,
        )

    def is_terminal(self) -> bool:
        """Return whether the decision and the following noise ply are played."""
        return len(self._path) >= self._decision_ply + 2

    def terminal_value(self) -> float:
        """Return ``+1`` **to the player to move here**, as the engine spells it.

        ``adapter.py:256``'s frame, not the decider's: the value is positive
        exactly when the player on move in this terminal position is the one
        the decision went for. Writing it the other way round would bake the
        very inversion this fixture exists to catch into the fixture.
        """
        decider = self._decision_ply % 2
        decider_won = self._path[self._decision_ply] == self._winning
        return 1.0 if (self.to_play() == decider) == decider_won else -1.0

    def to_play(self) -> int:
        """Return the mover: the two seats strictly alternate here."""
        return len(self._path) % 2


_WIN_SIMS: Final[int] = 16
"""Simulations per pick on the decisive toy, as the review's F3 pins them."""


def test_rawmcts_plays_the_winning_move_at_every_seed() -> None:
    """§D6.3: the ruler prefers winning, at every seed and from either seat.

    The node F3 adds, and the one that makes `rawmcts/n800` a **ruler**: nodes
    30, 31 and 54 assert reproducibility, no tree reuse and legality, and all
    three hold under a sign inversion. Two mutants are what this is red
    against, and both survive the rest of the file:

    * **M4** -- ``rawmcts.py:458``'s backup negation flipped
      (``-value if parent.mover == leaf_mover else value``). The winning action
      then backs up ``-1`` and the losing ones ``+1``, PUCT concentrates its
      visits on a loss, and the search plays to lose. It is §D4.1's
      pre-registered fallback gauge and three of the seven reference entrants,
      so the frozen sub-pool -- 11 matchups, played **once ever** -- would be
      measured against a ruler hundreds of Elo weaker than its id claims, and
      nothing in the pool could detect it, because RawMCTS is by construction
      the thing that cannot move.
    * **M5** -- ``rawmcts.py:388``'s ``argmax`` over the root visit counts
      replaced by ``argmin``. The returned action stays legal, stays a
      deterministic function of the seed and still varies across ``_SEEDS``, so
      nodes 30, 31 and 54 all pass; here it returns one of the two losing
      actions instead of the winner.

    Three halves. The winner is at index ``0``; then at the **last** index, so
    an ``argmax`` tie-break on the lowest action id cannot carry the first
    half; then one ply deep, where the root mover is seat ``1`` rather than
    seat ``0`` -- the case in which a value frame derived from ply parity
    rather than from ``_Node.mover`` gets the sign backwards.
    """
    for seed in _SEEDS:
        first = _DecisiveState(winning=0)
        assert RawMctsMover(sims=_WIN_SIMS).pick(first, ply=0, seed=seed) == 0
        assert first.path == (), "pick must not mutate the referee's position"

        last = _DecisiveState(winning=_DecisiveState.BRANCHING - 1)
        assert (
            RawMctsMover(sims=_WIN_SIMS).pick(last, ply=0, seed=seed)
            == _DecisiveState.BRANCHING - 1
        )

        # One ply deep: the decision belongs to seat 1, and the search reaches
        # it from a root whose mover is seat 1 -- so every backed-up value
        # crosses the negation on its way to the root.
        deep = _DecisiveState(winning=1, decision_ply=1, path=(2,))
        assert deep.to_play() == 1
        assert RawMctsMover(sims=_WIN_SIMS).pick(deep, ply=1, seed=seed) == 1


# ---------------------------------------------------------------------------
# node 31 -- no tree crosses a move
# ---------------------------------------------------------------------------


def test_rawmcts_carries_no_tree_across_a_move_and_reports_zero() -> None:
    """§6 node 31: ``tree_nodes_at_entry == 0``, and ply ``k+1`` is unconditioned.

    KOM-023 §D3.2 clause 2 forbids a retained subtree on any rated path, and
    ``gumbel_mover.py:83-89``'s rule is that the claim must be *reported*
    rather than left unmeasured. The second half is the one that can actually
    fail: a mover that kept its tree between moves would enter ply 1 with the
    ply-0 statistics already on the edges, and the warm mover's action would
    then differ from a cold mover's on the same position at the same seed.
    Red, too, against a mover that reports a hard-coded ``0`` while keeping a
    tree -- the field alone cannot see that, and the cold/warm comparison can.
    """
    warm = RawMctsMover(sims=_TOY_SIMS)
    referee = _ToyState()

    opening = warm.pick(referee, ply=0, seed=4242)
    assert warm.tree_nodes_at_entry == 0
    referee.play(opening)

    warm_reply = warm.pick(referee, ply=1, seed=1337)
    assert warm.tree_nodes_at_entry == 0

    cold = RawMctsMover(sims=_TOY_SIMS)
    cold_reply = cold.pick(_ToyState((opening,)), ply=1, seed=1337)
    assert cold.tree_nodes_at_entry == 0
    assert warm_reply == cold_reply


# ---------------------------------------------------------------------------
# node 54 -- the real GameState against the declared protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _RevTwoState(Protocol):
    """Rev 2's spelling of the protocol, kept as the node's non-vacuity fixture.

    ``legal_actions`` and ``result`` are the two names rev 2 of the design
    declared, and **neither exists** on the real state (``adapter.py:175-265``)
    -- which is why rev 3 R1 re-spelled the protocol. A run-time-checkable copy
    of the wrong spelling is what turns node 54 from "the six names happen to
    be there" into a discriminating check.
    """

    def legal_actions(self) -> object:
        """Rev 2's name for the legal set. Not on the real state."""
        ...

    def result(self) -> float:
        """Rev 2's name for the terminal value. Not on the real state."""
        ...


def test_the_real_game_state_satisfies_the_rawmcts_protocol() -> None:
    """§6 node 54: the real state answers all six names, and the ruler moves.

    Three halves, per the node: the static one is
    :data:`PROTOCOL_WITNESS` above (``mypy --strict`` checks the assignment);
    the dynamic one is below -- each of the six names is present, callable and
    answers in the documented shape, and one ``pick`` runs to a legal action on
    a real position; and the non-vacuity one is :class:`_RevTwoState`, the
    spelling that must **not** match.
    """
    state = GameState(Game(), 0.0)

    mask = state.legal_mask()
    assert mask.dtype == np.bool_
    assert bool(mask.any())
    assert state.is_terminal() is False
    assert state.to_play() in (0, 1)
    assert isinstance(state.terminal_value(), float)

    # clone() is the load-bearing one (§D6.3): a rollout must not move the
    # referee's board, so play() on a clone must leave the original alone.
    clone = state.clone()
    assert clone.game is not state.game
    clone.play(int(np.flatnonzero(mask)[0]))
    assert state.legal_mask().tolist() == mask.tolist()
    assert state.is_terminal() is False

    mover = RawMctsMover(sims=2, player_id="rawmcts/n2@0f31ab77")
    action = mover.pick(state, ply=0, seed=17)
    assert bool(state.legal_mask()[action])
    assert mover.tree_nodes_at_entry == 0

    assert not isinstance(state, _RevTwoState)
    with pytest.raises(AttributeError):
        getattr(state, "legal_actions")()
    with pytest.raises(AttributeError):
        getattr(state, "result")()
