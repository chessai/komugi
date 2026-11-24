"""The engine->search adapter: a :class:`Game` the tree can search and evaluate.

``docs/design/kom-021-selfplay.md`` §0 (review finding F11) records the reason
this module exists: the concrete Cython-backed :class:`komugi.engine.game.Game`
does *not* implement :class:`komugi.search.state.SearchState`. It speaks
``valid_moves`` / ``play_action`` / ``win_state``, and its ``win_state`` is a
``(3,)`` ``uint8`` vector in the **absolute** ``[white, black, draw]`` frame,
while everything in the search is expressed from the point of view of the side
to move. Bridging those two frames in one place, with the sign convention
written down next to the code that applies it, is the whole job here.

Three contracts leave this module:

* :class:`EvaluableState` -- :class:`~komugi.search.state.SearchState` plus
  ``observation()``. The search itself never needs an observation (it hands
  states to an evaluator and asks nothing about them), but the *evaluator* the
  worker builds does, and there is no other typed way to ask a state for one.
  This is the "richer evaluator-side protocol" of design §0.
* :func:`make_evaluator` -- the batch inference seam
  (:class:`~komugi.selfplay.inference.BatchEvaluator`, batch of N) adapted to
  the search's scalar seam (:data:`~komugi.search.state.Evaluator`, one leaf at
  a time). Batching happens in the server, across games (design D1).
* :func:`make_wave_evaluator` -- the same seam in its **wave** shape (KOM-040
  b2): many leaves in, ONE shared-memory round trip, one answer per leaf out,
  in whichever of the two shapes below the factory was asked for. Same
  perspective rules, same masks, same adapter; the only difference is how many
  questions travel per doorbell.

Both factories take ``with_wdl`` (KOM-173 §4): given it, each answer carries the
leaf's ``(w, d, l)`` as a third element -- in the **leaf mover's frame**, the
network's own vector with no utility applied -- which arms the search's
root-only searched-WDL accumulator. Off by default, so every caller that is not
the self-play worker (arena, gauntlet, ladder, anchor server) keeps the
two-tuple seam it had, down to the object it returns. The scalar is the same
number in both shapes: :func:`_value_from_wdl` forms it from ``wdl`` and then
hands ``wdl`` back beside it, so nothing in the utility reads the new element.

Perspective, stated once (design D1/F3b)
----------------------------------------
Both halves of the value pipeline -- :meth:`GameState.terminal_value` and the
scalar the evaluator derives from ``wdl`` -- return a number in ``[-1, 1]``
*for the player to move at that state*, and the draw utility ``draw_score``
enters **only** at those two points, in that node's own perspective. Under
strict negamax a constant mover-relative draw utility oscillates in sign up the
tree (a draw worth ``+0.20`` to the mover backs up as ``-0.20`` to the
opponent), which is not the symmetric draw-avoidance pressure ``DESIGN.html``
§3.3 intends. P0a **accepts and documents** that oscillation rather than
patching it; the pre-registered P0b fix is a WDL-carrying backup, not a
different constant.

Contempt never reaches the training data: :meth:`GameState.absolute_outcome`
hands the recorder the raw ``win_state`` vector, and
:mod:`komugi.selfplay.worker` turns *that* into ``wdl_target``. ``draw_score``
is a search-time preference; ``wdl_target`` is what happened.

Legal masks are computed twice per leaf -- once by the search, once by the
evaluator for the seam's ``legal`` argument. Design §0 accepts the duplication
explicitly; caching it on the state would mean invalidating a cache on every
``play``, which is a correctness risk traded for one cheap engine call.

The moves-left conversion term (KOM-159 D5/D6)
----------------------------------------------
The **only** other thing that may enter a leaf's scalar is
:func:`mlh_utility_term`: ``lambda * g(|v|) * sign(v) * f(m)``, the pressure
that makes the side that is winning prefer the line ending sooner and the side
that is losing prefer the line that lasts. It enters at the **leaf seam only**
(:func:`_value_from_wdl`, both the scalar and the wave path); terminals bypass
the evaluator entirely (``gumbel.py``'s ``_make_node``/``_descend`` read
:meth:`GameState.terminal_value` without asking the network, design review
F11), so :meth:`GameState.terminal_value` is untouched.

Unlike the draw constant, this term is **odd** under the frame swap
``w <-> l``: ``v`` negates, ``|v|`` and therefore ``g`` do not, ``sign(v)``
negates and ``f(m)`` does not, so ``T(swap(wdl), m) == -T(wdl, m)`` exactly.
Negamax's one negation per mover change therefore re-expresses the *same*
preference in the parent's frame rather than oscillating the way ``s*d`` does
(design §3.2; the two-ply arithmetic is worked in the KOM-159 review's F1).

``f`` is centred on **this search's root**, which is what
:class:`SearchContext` carries: the root is the first state a search hands the
evaluator (``GumbelSearch.search`` and ``_forced_move`` both evaluate it before
any leaf, pinned by ``tests/search/unit/test_gumbel_unit.py``), so the first
evaluation after a :meth:`SearchContext.reset` sets the reference and adds
nothing to its own value, and every later one is a difference against it.

At ``lambda == 0`` the term is **skipped entirely** -- not added as ``0.0`` --
and the context is never read (design review F9). That is what keeps the
arena's and the anchor server's two-argument :func:`make_evaluator` call, and
every number below it, bit-identical to what it was.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import numpy as np
import numpy.typing as npt

from komugi.config import SelfPlayConfig
from komugi.engine.game import Game, draft_root_mask
from komugi.engine.observation import MAX_TURNS
from komugi.search.state import (
    EvalAnswer,
    Evaluator,
    SearchState,
    WdlAnswer,
    WdlEvaluator,
)
from komugi.selfplay.inference import (
    BatchEvaluator,
    NetEval,
    SlotClient,
    WDL_SIZE,
    wdl_utility,
)
from komugi.shapecheck import checked_seam

if TYPE_CHECKING:
    # Static checkers see plain numpy arrays; the shape strings below are a
    # runtime contract only, exactly as in :mod:`komugi.shapecheck`.
    PolicyVector = npt.NDArray[np.float32]
    WdlVector = npt.NDArray[np.float32]
    OutcomeVector = npt.NDArray[np.uint8]
    ControlVector = npt.NDArray[np.int8]
else:
    try:
        from jaxtyping import Float, Int8, UInt8

        PolicyVector = Float[np.ndarray, "actions"]
        WdlVector = Float[np.ndarray, "3"]
        OutcomeVector = UInt8[np.ndarray, "3"]
        ControlVector = Int8[np.ndarray, "81"]
    except ImportError:  # degraded: no jaxtyping in this environment
        PolicyVector = npt.NDArray[np.float32]
        WdlVector = npt.NDArray[np.float32]
        OutcomeVector = npt.NDArray[np.uint8]
        ControlVector = npt.NDArray[np.int8]

_DRAW_INDEX = 2
"""Index of the draw slot in the engine's absolute ``win_state`` vector."""


class EvaluableState(Protocol):
    """A :class:`~komugi.search.state.SearchState` that can also be encoded.

    The six :class:`~komugi.search.state.SearchState` methods, plus
    :meth:`observation`. The base protocol deliberately has neither an
    observation nor anything else the tree does not call (see its module
    docstring), so the evaluator seam needs its own, wider contract -- design
    §0's "richer evaluator-side protocol carrying observation()".

    Note that :meth:`clone` returns an :class:`EvaluableState`, not a bare
    ``SearchState``: a clone that lost the ability to be evaluated would be
    useless to the search, which evaluates clones and never the state it was
    handed.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return a shape ``(A,)`` boolean mask, ``True`` where legal."""
        ...

    def play(self, action: int) -> None:
        """Advance this state in place by `action`, which must be legal here."""
        ...

    def clone(self) -> EvaluableState:
        """Return a deep, independent copy -- itself an :class:`EvaluableState`."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the game has ended in this position."""
        ...

    def terminal_value(self) -> float:
        """Return the result in ``[-1, 1]`` for the player to move here."""
        ...

    def to_play(self) -> int:
        """Return the index (``0`` white, ``1`` black) of the player to move."""
        ...

    def observation(self) -> npt.NDArray[np.float32]:
        """Return this position encoded as ``(in_planes, 9, 9)`` float32."""
        ...


class GameState:
    """A :class:`komugi.engine.game.Game` wearing the search's contracts.

    Thin by construction: every method delegates to the engine, and the only
    arithmetic in the class is :meth:`terminal_value`'s frame change. The
    engine object is owned, not shared -- :meth:`clone` clones it -- because
    :meth:`play` mutates in place and the search clones constantly.

    Args:
        game: The engine game. This adapter takes ownership: mutating it from
            outside would desynchronise nothing here (no state is cached) but
            would surprise the search, which assumes only it mutates clones.
        draw_score: What a draw is worth to the side to move, in ``[-1, 1]``.
            ``DESIGN.html`` §3.3's contempt, applied per design D1 at the two
            points a value enters the tree -- here for terminal draws, and in
            :func:`make_evaluator` for network draws. It never touches
            :meth:`absolute_outcome` and therefore never touches the training
            targets.
        min_placements: The DRAFTMIN threshold
            (:attr:`komugi.config.SelfPlayConfig.min_placements`), either one
            ``int`` for both sides or a ``(white, black)`` pair -- KOM-246
            draws a threshold per colour per game
            (:func:`komugi.selfplay.records.draft_thresholds`). Stored as a
            pair either way, and each side is masked by ITS OWN entry. ``0``
            -- the default, and a legal entry of a pair -- is off for that side
            and byte-identical to this class before the knob existed. Only ever
            consulted while `is_root` holds.
        is_root: Whether this state is a **root** the search is asked about,
            rather than a position inside a tree. ``False`` by default, so a
            state is a plain position unless its builder says otherwise, and
            the only builders that say otherwise are the self-play worker's
            per-ply root and the anchor server's per-request root.

    The root view (DRAFTMIN)
    ------------------------
    A positive threshold makes :meth:`legal_mask` a *view*: at a root, and
    only at a root, it hides 終 (draft-finish) until the mover has placed that
    many pieces. The threshold consulted is the SIDE TO MOVE's own entry of the
    pair, so the two colours can be masked to different draft lengths within
    one game (KOM-246) -- and the engine makes that asymmetry one-directional:
    black's 終 ends BOTH drafts while white's ends only its own, so ``N_w`` is a
    floor on white's own close and nothing more, and whenever black closes
    first white's draft ends wherever it stood. A mask whose mover has no draft
    bit left is a no-op (:func:`komugi.engine.game.draft_root_mask`'s second
    guard), which is exactly what white's next root sees there. Nothing else
    changes -- ``Game.valid_moves``, ``win_state``
    and every engine invariant answer exactly as they did, so the mask cannot
    reach an adjudication or a training target except through the actions the
    search declines to consider.

    The root-ness travels a deliberate route, and the route is what makes the
    property hold for both searches that consume this class
    (:class:`komugi.search.gumbel.GumbelSearch` and the match tree):

    * :meth:`clone` **keeps** it. The search does not search the state it was
      handed; it clones it once (``gumbel.py:521``, ``match/tree.py:346``) and
      builds the ROOT NODE from that clone. A clone that dropped the view would
      leave the root node's ``legal`` unmasked while the candidate draw used
      the masked one, and the improved policy -- which spreads mass over every
      action in the node's ``legal`` -- would then put target mass on a 終 the
      search could never play. The recorded policy target is the artefact this
      whole feature exists to shape, so the clone carries the view.
    * :meth:`play` **drops** it. Every interior position in either tree is
      reached by ``clone()`` followed immediately by ``play(edge)``
      (``gumbel.py:653/781``, ``match/tree.py:346-356``), so the first edge
      played turns a root view into an ordinary position and every expansion
      below it sees the engine's true legal set. That is the seam: not "a
      child is constructed with ``is_root=False``", which no call site here
      could enforce, but "a position that has been moved in is no longer a
      root", which is true by construction wherever a search makes a move.

      The self-play loop is the one caller for which it is *not* true -- it
      advances one state through a whole game, so its position after a move is
      the next root -- and it says so, with ``play(action, keep_root=True)``.
      The parameter is keyword-only and defaulted off precisely so that the
      search, which plays through a protocol that has no such parameter, cannot
      reach it.
    """

    def __init__(
        self,
        game: Game,
        draw_score: float,
        *,
        min_placements: int | tuple[int, int] = 0,
        is_root: bool = False,
    ) -> None:
        self._game = game
        self._draw_score = float(draw_score)
        # One threshold widens to the pair it always was: every caller that
        # predates KOM-246 hands an int and means "both sides", and storing
        # `(n, n)` keeps `legal_mask` a single indexed read rather than a
        # branch on the argument's type.
        self._min_placements: tuple[int, int] = (
            (int(min_placements[0]), int(min_placements[1]))
            if isinstance(min_placements, tuple)
            else (int(min_placements), int(min_placements))
        )
        self._is_root = bool(is_root)

    @property
    def game(self) -> Game:
        """Return the wrapped engine game (diagnostics and tests)."""
        return self._game

    @property
    def draw_score(self) -> float:
        """Return the draw utility this state scores terminal draws with."""
        return self._draw_score

    @property
    def min_placements(self) -> tuple[int, int]:
        """Return this state's DRAFTMIN thresholds as ``(white, black)``.

        Always the pair, whatever the constructor was handed: a state built
        with a single ``int`` reports ``(n, n)``. A pair rather than the scalar
        because each side is masked by its own entry (KOM-246); nothing outside
        this class reads the property today, so no caller is being asked to
        unpack something it used to compare.
        """
        return self._min_placements

    @property
    def is_root(self) -> bool:
        """Return whether this state still carries the root view."""
        return self._is_root

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the engine's legality mask as ``(A,)`` bool.

        ``Game.valid_moves`` returns ``uint8``; the search wants ``bool_``, and
        the conversion is a view-free cast of a 37341-element array.

        At a **root** with DRAFTMIN on FOR THE SIDE TO MOVE, the engine's
        answer is passed through :func:`komugi.engine.game.draft_root_mask`
        first, which withholds 終 while the mover is drafting and short of its
        threshold (see this class's "root view" section). The threshold is the
        mover's own entry of the pair -- indexed by ``Game.player``, which IS
        ``board.turn``, the same ``0 white / 1 black`` the helper reads -- so a
        pair with one zero masks one colour and leaves the other on today's
        path. Off -- which is every state of every run that does not set
        ``min_placements``, every interior state of one that does, and the
        zero side of such a pair -- the engine's own array is returned,
        un-copied and unexamined: the default path is the line it was before
        the knob.
        """
        mask = np.asarray(self._game.valid_moves(), dtype=np.bool_)
        threshold = self._min_placements[int(self._game.player)]
        if not (self._is_root and threshold > 0):
            return mask
        # `_board` because `Game` publishes no board accessor; the same private
        # read `komugi.eval.v1bridge` makes of the same attribute. The helper
        # never mutates it, and returns THIS object unchanged whenever there is
        # nothing to withhold (draft over, threshold reached, 終 not legal, or
        # 終 the only legal move).
        masked: npt.NDArray[np.bool_] = draft_root_mask(
            self._game._board, mask, threshold  # noqa: SLF001
        )
        return masked

    def play(self, action: int, *, keep_root: bool = False) -> None:
        """Advance the game in place by `action` (the engine validates it).

        **Drops the root view** unless `keep_root` says otherwise: a position a
        search has moved into is a position inside a tree, and DRAFTMIN is a
        property of the position the search was asked about. This is the seam
        that keeps interior nodes on the engine's true legal set -- see the
        class docstring -- and it works because every interior position in
        either tree is a clone this method has advanced.

        Args:
            action: The action to play, legal at this position.
            keep_root: Keep the root view across the move. For the **self-play
                loop and nothing else**: that loop advances one state through a
                whole game, so its position after a move is the next *root*
                rather than an interior node, and a bare ``play`` would switch
                DRAFTMIN off after the first ply of every game. Keyword-only
                and defaulted off, so the search -- which calls ``play(edge)``
                positionally through the :class:`EvaluableState` protocol, and
                whose protocol has no such parameter -- cannot reach it.
        """
        self._game.play_action(int(action))
        if not keep_root:
            self._is_root = False

    def clone(self) -> GameState:
        """Return an independent copy, carrying the same ``draw_score``.

        Carries the root view too, because a search builds its root node from
        a clone of the state it was handed rather than from that state (class
        docstring). The copy is independent in the way that matters: the game
        is cloned, and the clone's view dies on its own first :meth:`play`.
        """
        return GameState(
            self._game.clone(),
            self._draw_score,
            min_placements=self._min_placements,
            is_root=self._is_root,
        )

    def is_terminal(self) -> bool:
        """Return whether the engine has adjudicated this position.

        The engine owns *every* termination rule, the engine's turn cap included
        (design D6/F15): there is no second cap anywhere in self-play, so this
        is exactly ``win_state()`` having any slot set.
        """
        return bool(np.asarray(self._game.win_state()).any())

    def to_play(self) -> int:
        """Return ``0`` (white) or ``1`` (black) -- the engine's own frame.

        The same indexing as :meth:`absolute_outcome`'s first two slots, which
        is what makes :meth:`terminal_value`'s lookup a lookup rather than a
        translation.
        """
        return int(self._game.player)

    def in_draft(self) -> bool:
        """Return whether **the mover** still holds drafting rights.

        No argument, on purpose: the adapter is where the mover is already
        known, so ``self._game.in_draft(self.to_play())`` makes KOM-036 §3.3's
        mover-relativity structural at the call site rather than a rule the
        worker has to remember every time it builds a search config.

        Not on the :class:`~komugi.search.state.EvaluableState` protocol: the
        search must not learn about phases. This exists for the self-play
        loop's temperature schedule and for diagnostics.
        """
        return self._game.in_draft(self.to_play())

    def observation(self) -> npt.NDArray[np.float32]:
        """Return the ``(in_planes, 9, 9)`` float32 encoding of this position."""
        return np.asarray(self._game.observation(), dtype=np.float32)

    def absolute_outcome(self) -> npt.NDArray[np.uint8]:
        """Return the raw ``(3,)`` ``uint8`` ``[white, black, draw]`` outcome.

        **For the recorder.** ``wdl_target`` is the pure game result (design
        D4), so the worker needs the outcome in the frame the engine states it
        in, and does the per-row mover flip itself. Routing training targets
        through :meth:`terminal_value` instead would fold ``draw_score`` into
        them, which is precisely what design D1 forbids.

        All-zero on a non-terminal position -- "no result yet", not a draw.
        """
        return np.asarray(self._game.win_state(), dtype=np.uint8)

    def final_control(self) -> npt.NDArray[np.int8]:
        """Return the raw ``(81,)`` ``int8`` top-of-tower ownership map.

        **For the recorder**, and absolute for the same reason
        :meth:`absolute_outcome` is: ``0`` white, ``1`` black, ``2`` empty, in
        the engine's own colour frame, with
        :func:`komugi.selfplay.worker.finalize_rows` doing the per-row mover
        flip. One perspective change, in one place, keyed off each row's
        ``to_play`` -- the same discipline ``wdl_target`` is under (design
        F11).

        Answers for any position, terminal or not; the worker asks exactly once,
        of the final board (KOM-010).
        """
        return np.asarray(self._game.final_control(), dtype=np.int8)

    def final_material(self) -> tuple[int, int]:
        """Return ``(white_total, black_total)`` of the current position.

        **For the recorder**, and absolute for the reason
        :meth:`absolute_outcome` and :meth:`final_control` are: KOM-245's cap
        adjudication compares the two totals itself, in the engine's own colour
        frame, and :func:`komugi.selfplay.worker.finalize_rows` does the one
        per-row mover flip.

        A pure delegation to :meth:`komugi.engine.game.Game.material`, which is
        where the count lives: the rule reads the board once, through the
        engine, so a label and the planes beside it cannot disagree about how
        much material was on the final board.
        """
        return self._game.material()

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` one-hot ``(win, draw, loss)`` for the mover here.

        The vector form of :meth:`terminal_value`, and the one the searched-WDL
        accumulator reads at a terminal leaf (KOM-173 D5). ``win_state()`` is
        absolute -- ``[white, black, draw]`` -- so the win slot is the mover's,
        the loss slot is the other player's, and the draw slot is shared:

        * the side to move won   -> ``(1, 0, 0)``
        * the side to move lost  -> ``(0, 0, 1)``
        * draw                   -> ``(0, 1, 0)``

        **No ``draw_score``**, deliberately, and that is the whole difference
        from :meth:`terminal_value`: contempt is a scalar-side preference
        (design D1), while the vector is the pure result. It is also why the
        vector composes with the search's backup as a *permutation* --
        ``(w, d, l) -> (l, d, w)`` leaves ``d`` where it is -- instead of
        oscillating the way a mover-relative draw constant does. The same rule
        the match tree's adapter applies
        (:meth:`komugi.search.match.state.MatchState.terminal_wdl`).

        All-zero on a position with no result yet, exactly as
        :meth:`absolute_outcome` reports it there: "no result" is not a draw,
        and the search only asks this of terminal leaves.
        """
        outcome = np.asarray(self.absolute_outcome(), dtype=np.float64)
        mover = self.to_play()
        triple: npt.NDArray[np.float64] = np.zeros(WDL_SIZE, dtype=np.float64)
        triple[0] = outcome[mover]
        triple[1] = outcome[_DRAW_INDEX]
        triple[2] = outcome[1 - mover]
        return triple

    def terminal_value(self) -> float:
        """Return the result in ``[-1, 1]`` for the player to move *here*.

        The perspective-critical method of this module. ``win_state()`` is
        absolute -- ``[white, black, draw]`` -- and everything above it is
        mover-relative, so:

        * the side to move won   -> ``+1.0``
        * the side to move lost  -> ``-1.0``
        * draw                   -> :attr:`draw_score`

        The draw case is design D1's contempt convention: the draw utility
        enters **only** where a value enters the tree, expressed in the
        perspective of the side to move *at that node*. Under the search's
        negamax backup that constant oscillates in sign on the way up (a draw
        worth ``+draw_score`` to this mover becomes ``-draw_score`` one edge
        up), which is not §3.3's symmetric draw-avoidance and is *accepted and
        documented* as a P0a approximation rather than papered over -- the
        pre-registered P0b fix is a WDL-carrying backup, not a patched scalar.

        Returns ``0.0`` on a non-terminal position, where the value is
        meaningless by the :class:`~komugi.search.state.SearchState` contract
        (the search only calls this on terminal states).
        """
        outcome = self.absolute_outcome()
        if bool(outcome[_DRAW_INDEX]):
            return self._draw_score
        mover = self.to_play()
        if bool(outcome[mover]):
            return 1.0
        if bool(outcome[1 - mover]):
            return -1.0
        return 0.0


@dataclass(frozen=True)
class MlhUtility:
    """The four numbers the moves-left conversion term is built from (D7).

    Frozen and tiny, and deliberately **not** a view onto
    :class:`~komugi.config.SelfPlayConfig`: the term is arithmetic over four
    scalars, and a test that wants to sweep ``lam`` should not have to build a
    whole self-play config with a worker count in it. :meth:`from_selfplay` is
    the one place production reads the config, so a campaign that turns the
    lever on and a unit test that pins its shape cannot disagree about which
    fields it is.

    Attributes:
        lam: The ``lambda`` of ``u = v + s*d + lambda*g(|v|)*sign(v)*f(m)``.
            ``0.0`` is **off**, and off means the term is never computed and
            the reference is never read -- see :func:`_value_from_wdl`.
        tau0: Below ``|v| = tau0`` the gate is shut and the term is exactly
            ``0.0``, for any ``m``.
        tau1: Where the gate is fully open;
            ``g(|v|) = |v| * clip((|v| - tau0)/(tau1 - tau0), 0, 1)``.
        slope_plies: The ply difference from the root's own prediction that
            saturates ``f`` at ``+-1``.
    """

    lam: float
    tau0: float
    tau1: float
    slope_plies: int

    @classmethod
    def from_selfplay(cls, config: SelfPlayConfig) -> MlhUtility:
        """Return the term's parameters as this self-play config states them.

        The config validates the four in ``__post_init__`` (``0 <= tau0 < tau1
        <= 1``, ``slope_plies >= 1``, ``lam`` in ``[0, 1]``, and ``lam > 0``
        only at ``draw_score == 0``), so nothing is re-checked here: a second
        copy of those bounds would be a second thing to keep in step.
        """
        return cls(
            lam=float(config.mlh_utility_lambda),
            tau0=float(config.mlh_gate_tau0),
            tau1=float(config.mlh_gate_tau1),
            slope_plies=int(config.mlh_slope_plies),
        )


class SearchContext:
    """The per-search reference ``f(m)`` is centred on.

    One mutable field, and the reason it is an object rather than a closure
    variable: the scalar seam (:func:`make_evaluator`) and the wave seam
    (:func:`make_wave_evaluator`) are two closures built at two different
    places in the worker, and both must centre on the *same* root. Handing
    them one of these is how they share it -- and, since KOM-171, the sharing
    is checked rather than conventional: :func:`make_wave_evaluator` returns a
    :class:`WaveSeam` that names its context, and
    :meth:`komugi.selfplay.worker.SelfPlayWorker.__init__` refuses a live wave
    seam whose context is not the one its scalar seam holds
    (:func:`require_shared_context`).

    **Reset before every search.** The worker calls :meth:`reset` immediately
    before each ``GumbelSearch.search(state)``; the first evaluation after
    that sets :attr:`m_root` from its own answer and contributes no term (its
    own delta is zero), and every later evaluation of that search is a
    difference against it. A search that never reset would centre on the
    previous ply's root, which is a different position.

    Not thread-safe, and it does not need to be: an interleaved worker builds
    one of these per lane (``worker.py``'s ``_run_lane``), and a lane's search
    is single-threaded.
    """

    __slots__ = ("m_root",)

    def __init__(self) -> None:
        self.m_root: float | None = None
        """The root's predicted plies-to-end, or ``None`` before the root has
        been evaluated. ``None`` is not a defaultable value (design R9): a term
        centred on nothing is not the term, so the code that would read it sets
        it instead."""

    def reset(self) -> None:
        """Forget the previous search's root. Called once per ``search()``."""
        self.m_root = None


def mlh_utility_term(
    wdl: Sequence[float] | WdlVector,
    moves_left: float,
    m_root_plies: float,
    config: MlhUtility,
) -> float:
    """Return ``lambda * g(|v|) * sign(v) * f(m)`` for one leaf (design §3.1).

    Pure, and the whole of the term's arithmetic -- ``_value_from_wdl`` does
    nothing but add this to :func:`~komugi.selfplay.inference.wdl_utility`'s
    answer and clamp. The three properties the design rests on are properties
    of these six lines and are pinned in ``tests/selfplay/unit/test_adapter_mlh.py``:

    * **shut below the gate** -- ``|v| < tau0`` returns exactly ``0.0`` for any
      ``m``, so no draftish or balanced position is moved at all;
    * **odd** -- swapping ``w`` and ``l`` negates the result exactly, which is
      what lets negamax carry it up the tree without the ``s*d`` oscillation
      (module docstring, design §3.2);
    * **bounded** -- ``|term| <= lambda * g(|v|) <= lambda``, saturating once a
      leaf is ``slope_plies`` shorter or longer than the root expected.

    Args:
        wdl: ``(w, d, l)`` in the **mover's** frame at the leaf, as the network
            answered. Only ``w`` and ``l`` are read: the gate is contempt-free
            by construction (``v = w - l``), which is why a non-zero
            ``draw_score`` cannot open or close it.
        moves_left: The head's raw output for this leaf, a fraction of
            :data:`~komugi.engine.observation.MAX_TURNS`. Clipped into
            ``[0, 1]`` before it becomes plies -- the head is a linear
            regression and can answer outside its own scale.
        m_root_plies: The same quantity for **this search's root**, already in
            plies (:class:`SearchContext`).
        config: The four parameters.

    Returns:
        The term, in the leaf mover's frame, ready to be added to that mover's
        utility.
    """
    v = float(wdl[0]) - float(wdl[2])
    magnitude = abs(v)
    ramp = (magnitude - config.tau0) / (config.tau1 - config.tau0)
    gate = magnitude * min(max(ramp, 0.0), 1.0)
    if gate == 0.0:
        # Exactly zero, and returned before anything reads `moves_left`: the
        # "below tau0 the term does not exist" property is an identity, not a
        # product that happens to round to nothing.
        return 0.0
    plies = MAX_TURNS * min(max(float(moves_left), 0.0), 1.0)
    # f(m): +1 a full slope shorter than the root expected, -1 a full slope
    # longer. Relative rather than absolute (design §3.1): an absolute f with
    # this slope would act only in the last 32 plies of a game.
    centred = min(max((float(m_root_plies) - plies) / config.slope_plies, -1.0), 1.0)
    # copysign, not `v / |v|`: it is exact for every finite v. The v == 0 case
    # is unreachable here (the gate would be 0) but would give copysign's +1
    # rather than sign(0) = 0, which is why it is written as a branch.
    sign = 0.0 if v == 0.0 else math.copysign(1.0, v)
    return config.lam * gate * sign * centred


def _require_context(
    mlh: MlhUtility | None, context: SearchContext | None
) -> bool:
    """Return whether the term is live, refusing a live term with no reference.

    The guard design review F9 specifies, in the one place both evaluator
    factories can share it. The refusal is **conditional on ``lam > 0``**: an
    unconditional one would fire on ``arena.py``'s and
    the Python match server's two-argument calls, which pass
    neither argument and must keep playing exactly as they did (D8's
    measurement parity).

    Raised at construction rather than at the first leaf (design R9): a search
    that discovered halfway through that it had no reference would have already
    written half a tree centred on nothing.
    """
    if mlh is None or mlh.lam == 0.0:
        return False
    if context is None:
        raise ValueError(
            "make_evaluator was given an MLH utility with lam="
            f"{mlh.lam!r} but no SearchContext: f(m) is centred on the "
            "search's own root, so there is no reference to centre it on and "
            "no defensible default (KOM-159 R9). Pass the SearchContext the "
            "caller resets before each search, or leave lam at 0.0"
        )
    return True


@checked_seam()
def _value_from_wdl(
    policy_logits: PolicyVector,
    wdl: WdlVector,
    draw_score: float,
    *,
    moves_left: float = 0.0,
    mlh: MlhUtility | None = None,
    context: SearchContext | None = None,
    # PolicyVector, not bare npt.NDArray: numpy >= 2.x's NDArray carries the
    # ScalarT TypeVar, which beartype cannot reduce -- an npt annotation here
    # makes @jaxtyped raise AT DECORATION TIME, so KOMUGI_TYPECHECK=1 kills
    # the worker on import and the D7 gate can never arm (wave-9b review F1).
) -> tuple[PolicyVector, float, WdlVector]:
    """Return `(policy_logits, utility, wdl)` for one position -- the shape seam.

    Design D7 asks for the ``BatchEvaluator -> Evaluator`` adapter to be
    ``@jaxtyped`` when ``KOMUGI_TYPECHECK=1``. The decoration sits *here*,
    on the arrays, rather than on the closure :func:`make_evaluator` returns,
    for one concrete reason: that closure's parameter is annotated
    :class:`~komugi.search.state.SearchState`, a Protocol that is not
    ``@runtime_checkable``, and beartype rejects such an annotation at
    decoration time -- arming the seam would then break the worker's import
    rather than check its shapes. Everything the seam can actually catch (a
    ``wdl`` that is not three floats, a policy that is not a float32 vector)
    crosses this boundary, and the D7 integration test's mis-shaped ``(N, 4)``
    ``wdl`` fails right here.

    With the seam off, a mis-shaped ``wdl`` still fails -- as
    :func:`~komugi.selfplay.inference.wdl_utility`'s ``ValueError``, one frame
    later and with a different type. That difference is what the D7 control run
    asserts on.

    **The moves-left term (KOM-159 D6), when it is on at all.** `mlh` at
    ``None`` or at ``lam == 0.0`` leaves every line below the ``wdl_utility``
    call unexecuted: the returned float is the same object-for-object
    arithmetic it was before this parameter existed, which is what makes the
    bit-identity claim (T6) a claim about code that does not run rather than
    about a ``+ 0.0`` that rounds away. With ``lam != 0.0`` the FIRST
    evaluation of a search sets `context`'s reference from its own
    `moves_left` and adds nothing -- it is the root, and the root's delta
    against itself is zero -- and every later one adds the term and clamps back
    into ``[-1, 1]``.

    **The third return (KOM-173 §4)** is the `wdl` it was handed, unchanged and
    un-copied: the searched-WDL accumulator wants the leaf's own ``(w, d, l)``
    in the leaf mover's frame, which is exactly this argument. Nothing in the
    utility arithmetic reads it -- the scalar is formed from `wdl` above and the
    return merely hands the same array back -- so the utility is bit-identical
    to what it was before this element existed (T6), draw score and conversion
    term included. Whether a caller passes it on to the search is
    :func:`make_evaluator`'s ``with_wdl``.
    """
    utility = wdl_utility(wdl.reshape(1, WDL_SIZE), draw_score)
    value = float(utility[0])
    if mlh is not None and mlh.lam != 0.0:
        if context is None:
            raise ValueError(
                "the MLH utility is on (lam="
                f"{mlh.lam!r}) but no SearchContext reached the leaf seam; "
                "f(m) has nothing to be centred on (KOM-159 R9)"
            )
        if context.m_root is None:
            context.m_root = MAX_TURNS * min(max(float(moves_left), 0.0), 1.0)
        else:
            term = mlh_utility_term(wdl, moves_left, context.m_root, mlh)
            value = min(max(value + term, -1.0), 1.0)
    return policy_logits, value, wdl


def _evaluate_one(
    client: BatchEvaluator | SlotClient,
    obs: npt.NDArray[np.float32],
    legal: npt.NDArray[np.bool_],
) -> NetEval:
    """Evaluate one position through whichever end of the seam `client` is.

    The seam has two shapes and one meaning. :class:`BatchEvaluator` (the
    in-process evaluator, a test stub, a server-side forward) takes a batch and
    returns ``(N, ...)``; :class:`~komugi.selfplay.inference.SlotClient` -- the
    worker's shared-memory end -- deliberately implements no ``evaluate`` at
    all, because a tree search has exactly one position to ask about and *the
    server* is where batching happens (design D1). Both are dispatched here, in
    one place, so that a ``--serial-debug`` run and a production worker take
    identical code paths above this line.

    The returned :class:`~komugi.selfplay.inference.NetEval` is in whichever
    shape its producer used; the caller flattens.
    """
    if isinstance(client, SlotClient):
        return client.submit(obs, legal)
    return client.evaluate(obs[None], legal[None])


def _leaf_moves_left(answer: NetEval) -> float:
    """Return one answer's ``moves_left`` as a scalar.

    Called **only** when the term is live -- the call sites guard it -- so that
    a ``lam == 0`` search does not so much as touch the field: the whole point
    of the F9 skip is that the disabled path is the path that was there before,
    instruction for instruction.

    ``reshape(-1)[0]`` for the same reason the policy and the ``wdl`` use
    ``reshape(-1)``: the seam hands back a batch-of-one ``(1,)`` from a
    :class:`~komugi.selfplay.inference.BatchEvaluator` and an already-stripped
    scalar row from a wave answer.
    """
    return float(
        np.asarray(answer.moves_left, dtype=np.float32).reshape(-1)[0]
    )


@overload
def make_evaluator(
    client: BatchEvaluator | SlotClient,
    draw_score: float,
    mlh: MlhUtility | None = ...,
    context: SearchContext | None = ...,
    *,
    with_wdl: Literal[False] = ...,
) -> Evaluator: ...


@overload
def make_evaluator(
    client: BatchEvaluator | SlotClient,
    draw_score: float,
    mlh: MlhUtility | None = ...,
    context: SearchContext | None = ...,
    *,
    with_wdl: Literal[True],
) -> WdlEvaluator: ...


@overload
def make_evaluator(
    client: BatchEvaluator | SlotClient,
    draw_score: float,
    mlh: MlhUtility | None = ...,
    context: SearchContext | None = ...,
    *,
    with_wdl: bool,
) -> Evaluator | WdlEvaluator: ...


def make_evaluator(
    client: BatchEvaluator | SlotClient,
    draw_score: float,
    mlh: MlhUtility | None = None,
    context: SearchContext | None = None,
    *,
    with_wdl: bool = False,
) -> Evaluator | WdlEvaluator:
    """Adapt a batch inference client to the search's scalar evaluator.

    One leaf in, ``(policy_logits (A,), value)`` out -- or
    ``(policy_logits (A,), value, wdl (3,))`` under ``with_wdl`` (KOM-173 §4) --
    blocking. The value is
    :func:`~komugi.selfplay.inference.wdl_utility` applied to the network's
    win/draw/loss probabilities *at this node*, which is the one place design
    D1 lets ``draw_score`` in on the network side (the other is
    :meth:`GameState.terminal_value`, for terminal draws).

    The state must be an :class:`EvaluableState`: the search's own protocol
    cannot produce an observation, so the closure casts. That cast is checked
    by the engine in practice -- a state without ``observation()`` raises
    ``AttributeError`` on the first leaf -- and by the type checker at every
    call site that builds one of these, which is the worker and nothing else.

    Args:
        client: Anything that evaluates positions:
            :class:`~komugi.selfplay.inference.InProcessEvaluator`, a
            :class:`~komugi.selfplay.inference.SlotClient`, or a test stub.
        draw_score: What a draw is worth to the side to move at the node being
            evaluated.
        mlh: The moves-left conversion term's parameters (KOM-159 D5), or
            ``None`` -- the default, and what the arena, the gauntlet, the
            ladder and the anchor server all get, because they play at
            ``lambda = 0`` for measurement parity (D8). ``None`` and a
            ``MlhUtility`` with ``lam == 0.0`` are the same evaluator: neither
            computes the term, and neither touches `context`.
        context: The per-search reference the term is centred on, reset by the
            caller before every search. Required exactly when ``mlh.lam > 0``,
            and refused **here**, at construction, rather than at the first
            leaf.
        with_wdl: Hand the search each leaf's ``(w, d, l)`` as a third tuple
            element (KOM-173 §4), which is what arms its root-only searched-WDL
            accumulator and fills
            :attr:`~komugi.search.gumbel.SearchResult.root_wdl` and its two
            siblings. ``False`` -- the default -- returns the two-tuple this
            factory always returned, so the arena, the gauntlet, the ladder and
            the anchor server keep answers that are equal *as objects*, not
            merely equal in value; the self-play worker is the one caller that
            passes ``True``. The utility is the same number either way: the
            vector travels beside it and never through it.

    Returns:
        An :data:`~komugi.search.state.Evaluator` closure over `client`, or a
        :data:`~komugi.search.state.WdlEvaluator` one under ``with_wdl``. Two
        separate closures over one shared body, chosen **here** rather than by a
        branch inside the seam: the two-tuple closure drops the third element and
        cannot carry it, so its answer is the same object shape it always was
        whatever a caller later does with the flag.

        Three overloads, so a *literal* flag is typed exactly -- ``False`` (or
        omitted) reveals :data:`~komugi.search.state.Evaluator`, ``True``
        reveals :data:`~komugi.search.state.WdlEvaluator` -- while a flag whose
        static type is a plain ``bool`` (a config field, a test parameter)
        reveals the union rather than being a ``call-overload`` error
        (KOM-210 F2). A caller holding the union narrows it by ``len`` on the
        answer, exactly as the search does.

    Raises:
        ValueError: If `mlh` asks for a positive ``lam`` and `context` is
            ``None`` (design review F9, R9).
    """
    mlh_live = _require_context(mlh, context)

    def answer(state: SearchState) -> tuple[PolicyVector, float, WdlVector]:
        """Evaluate one leaf, blocking on the seam; the shared body.

        Almost always non-terminal; the one exception is the search's
        forced-terminal-root prior evaluation (see ``komugi.search.state``'s
        module docstring), which still carries a one-action legal mask, so
        nothing here needs to special-case it.

        Always forms the ``wdl`` -- it is the argument the scalar is derived
        from, so there is nothing to skip -- and always hands it back. Which of
        the two closures below is returned decides whether it travels any
        further; the *arithmetic* is one body either way, which is what makes
        the two shapes' scalars equal by construction rather than by review (T6).
        """
        evaluable = cast(EvaluableState, state)
        obs = np.ascontiguousarray(evaluable.observation(), dtype=np.float32)
        legal = np.asarray(evaluable.legal_mask(), dtype=np.bool_)
        evaluation = _evaluate_one(client, obs, legal)
        # reshape(-1), not an indexed row: it accepts both the batch-of-one
        # shape (1, A) and the SlotClient's already-stripped (A,), and -- the
        # part that matters for design D7 -- it does not *itself* reject a
        # wrong-sized wdl. A (4,) wdl reaches the shape seam and fails there,
        # as a jaxtyping error, which is the contract the D7 test asserts.
        policy = np.asarray(evaluation.policy_logits, dtype=np.float32).reshape(-1)
        wdl = np.asarray(evaluation.wdl, dtype=np.float32).reshape(-1)
        return _value_from_wdl(
            policy,
            wdl,
            draw_score,
            moves_left=_leaf_moves_left(evaluation) if mlh_live else 0.0,
            mlh=mlh,
            context=context,
        )

    def evaluate(state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Answer in the legacy two-tuple shape (the arena, the anchor server)."""
        logits, value, _ = answer(state)
        return logits, value

    def evaluate_with_wdl(
        state: SearchState,
    ) -> tuple[npt.NDArray[np.float32], float, WdlAnswer]:
        """Answer in the three-tuple shape (the self-play worker, KOM-173 §4)."""
        logits, value, leaf_wdl = answer(state)
        return logits, value, leaf_wdl

    if with_wdl:
        return evaluate_with_wdl
    return evaluate


WaveEvaluator = Callable[[Sequence[SearchState]], Sequence[EvalAnswer]]
"""Many leaves in, one ``(policy_logits (A,), value[, wdl])`` per leaf out.

The **wave** shape of the search seam (KOM-040 b2). Deliberately a plain alias
rather than a Protocol: it is the search-side counterpart of
:data:`~komugi.search.state.Evaluator`, and the search task that will consume it
(the concurrent b2 search-side ticket) types its own parameter against this
signature, not against this module.

**This name is not the same type as**
:data:`komugi.search.state.WaveEvaluator` (KOM-210 F8b), and neither is
renamed. This one answers with ``Sequence[EvalAnswer]`` -- the *wide* answer,
either tuple shape per entry, which is what one object serving both the
``with_wdl=False`` and the ``with_wdl=True`` seam has to be annotated with; the
search module's ``WaveEvaluator`` is the narrow **two-tuple** protocol the
arena, the gauntlet and the ladder are typed against, and its wide sibling
there is :data:`~komugi.search.state.AnyWaveEvaluator`, which this alias is
structurally equal to. A bare ``WaveEvaluator`` in a diff is therefore
ambiguous until the import is checked.
"""


@dataclass(frozen=True)
class WaveSeam:
    """A :data:`WaveEvaluator` that **publishes** the context it centres on.

    A plain closure satisfies :data:`WaveEvaluator`, and that is all the search
    needs -- but it makes the one invariant the two seams share unobservable
    from outside. The scalar seam and the wave seam are built in two different
    places (see :class:`SearchContext`), and a worker whose wave evaluator held
    a *different* :class:`SearchContext` than its scalar one would centre every
    wave leaf on the first leaf of the first wave rather than on the root:
    :func:`_require_context` cannot see that, because each seam on its own is
    perfectly well-formed. The KOM-159b implementation review's F2 asked for the
    identity to be enforced rather than left to the two production call sites'
    convention, so this carries the context out to where the worker can check it
    (:func:`require_shared_context`, called from
    :meth:`komugi.selfplay.worker.SelfPlayWorker.__init__`).

    Callable and nothing else: ``__call__`` forwards to the closure with no
    arithmetic and no branch, so the seam a search sees is byte-for-byte the
    behaviour :func:`make_wave_evaluator` always returned.

    Attributes:
        evaluate_wave: The closure :func:`make_wave_evaluator` built.
        context: The reference that closure centres ``f(m)`` on -- ``None``
            when the term is off, which is every wave built at ``lam == 0``.
    """

    evaluate_wave: WaveEvaluator
    context: SearchContext | None

    def __call__(self, states: Sequence[SearchState]) -> Sequence[EvalAnswer]:
        """Evaluate every state in `states` in one blocking round trip.

        Typed with the **wide** answer, because one class carries both shapes:
        which one a given seam actually produces is fixed by the ``with_wdl``
        its factory was called with, and the search reads the shape off the
        answer's length anyway (:data:`~komugi.search.state.EvalAnswer`).
        """
        return self.evaluate_wave(states)


def require_shared_context(
    wave_evaluator: WaveEvaluator | None,
    mlh: MlhUtility | None,
    context: SearchContext | None,
) -> None:
    """Refuse a **live** wave seam that is not centred on `context` (F2).

    The enforced half of :class:`SearchContext`'s "both seams must share one
    object". Called once, at worker construction, for the same reason
    :func:`_require_context` refuses there (design R9): a search that
    discovered at its fiftieth leaf that its wave answers were centred on a
    different position would already have written a tree.

    **Nothing runs at ``lam == 0``** beyond the one comparison that returns:
    the bit-identity claim (KOM-159 review F9, T6) is about a disabled path
    that executes no new code on the evaluator path, and a λ = 0 run -- every
    campaign through 5, the arena, the gauntlet, the ladder -- is accepted
    whatever its wave evaluator holds, because the reference is never read.

    Args:
        wave_evaluator: The worker's wave seam, or ``None`` (``--serial-debug``
            and every pre-KOM-040 caller).
        mlh: The worker's own reading of the config's four parameters.
        context: The :class:`SearchContext` the worker's *scalar* seam holds --
            the object the wave seam must be the same one of, not an equal one.

    Raises:
        ValueError: If the term is live and `wave_evaluator` either does not
            publish its context (it was not built by
            :func:`make_wave_evaluator`) or publishes one that is not
            `context`.
    """
    if wave_evaluator is None or mlh is None or mlh.lam == 0.0:
        return
    if not isinstance(wave_evaluator, WaveSeam):
        raise ValueError(
            "the MLH utility is on (lam="
            f"{mlh.lam!r}) but this worker's wave evaluator does not publish "
            "the SearchContext it centres f(m) on, so the two seams cannot be "
            "shown to share one root. Build it with make_wave_evaluator "
            "(komugi.selfplay.worker._wave_evaluator_for does), or leave lam "
            "at 0.0 (KOM-159b review F2)"
        )
    if wave_evaluator.context is not context:
        raise ValueError(
            "the MLH utility is on (lam="
            f"{mlh.lam!r}) but this worker's wave evaluator centres f(m) on a "
            "different SearchContext than its scalar seam does: the search "
            "evaluates the root through the scalar seam and its leaves through "
            "the wave one, so the wave leaves would be centred on the first "
            "leaf of the first wave instead of on the root. Pass ONE "
            "SearchContext to both (KOM-159b review F2)"
        )


def make_wave_evaluator(
    client: SlotClient,
    config: SelfPlayConfig,
    mlh: MlhUtility | None = None,
    context: SearchContext | None = None,
    *,
    with_wdl: bool = False,
) -> WaveSeam:
    """Adapt a wave-capable slot client to the search's batched evaluator seam.

    :func:`make_evaluator` blocks the whole search on one leaf at a time, which
    is one shared-memory round trip -- one doorbell, one forward's worth of
    latency -- per leaf. The KOM-040 b1 bench showed why that is the thing to
    attack: the shared batch already **synchronises** the lanes, so widening it
    buys little and cutting the *count* of round trips buys the rest. This
    closure submits a whole wave of leaves as ONE ticket
    (:meth:`~komugi.selfplay.inference.SlotClient.submit_wave`) and unpacks the
    answer row by row.

    **Transport only.** Deciding *which* leaves form a wave is the search's job
    and lives behind the seam this returns; nothing here schedules anything. The
    contract is exactly: give it ``R`` states, get ``R`` ``(policy, value)``
    pairs back, in the same order, having blocked once.

    Observation and legal mask are gathered per state exactly as
    :func:`make_evaluator` does -- same cast to :class:`EvaluableState`, same
    ``ascontiguousarray``, same duplicate mask computation that design §0
    accepts -- so a wave of one is byte-identical to a scalar submit.

    Args:
        client: This worker's :class:`~komugi.selfplay.inference.SlotClient`.
            Its ``max_wave`` bounds ``R``; a longer sequence is refused by
            ``submit_wave`` with a message naming the limit, rather than being
            silently chunked into the extra round trips this seam exists to
            delete. The orchestrator's construction-time check
            (``validate_wave_fit``) is what guarantees the search's ``m`` fits.
        config: The self-play config, for ``draw_score`` -- design D1's one
            network-side entry point for contempt, applied here in the
            perspective of the side to move at each evaluated node, exactly as
            :func:`make_evaluator` applies it.
        mlh: The moves-left term's parameters, or ``None``. Same contract as
            :func:`make_evaluator`'s, and the same reference must be shared
            with it: a worker whose two seams centred on different roots would
            score its wave leaves against a stale ply.
        context: The per-search reference. **The same object** the scalar
            evaluator holds -- ``GumbelSearch`` evaluates the root through the
            scalar seam and its leaves through this one, so a wave leaf whose
            context were a private copy would find no root at all. The returned
            :class:`WaveSeam` publishes it so that identity can be *checked*
            rather than assumed (:func:`require_shared_context`).
        with_wdl: Exactly :func:`make_evaluator`'s flag, and it must be given
            the SAME value (KOM-173 §4): the search reads the answer shape off
            the ROOT, which is always evaluated through the scalar seam, so a
            wave that dropped the vector a root had promised is refused at the
            first backup rather than silently averaged over. ``False`` -- the
            default -- returns the two-tuple wave this factory always returned.

    Returns:
        A :class:`WaveSeam` -- a :data:`WaveEvaluator` over `client` that also
        names the context it is centred on.

    Raises:
        ValueError: If `mlh` asks for a positive ``lam`` and `context` is
            ``None``.
    """
    draw_score = config.draw_score
    mlh_live = _require_context(mlh, context)

    def answers(
        states: Sequence[SearchState],
    ) -> list[tuple[PolicyVector, float, WdlVector]]:
        """Evaluate every state in `states` in one round trip; the shared body.

        The transport, the perspective rules and the utility arithmetic in their
        entirety -- the two closures below differ only in how much of each row
        they hand on, exactly as :func:`make_evaluator`'s two do.
        """
        if not states:
            raise ValueError(
                "a wave must carry at least one state: an empty wave would ring "
                "the seam's doorbell for an answer nobody could write"
            )
        evaluables = [cast(EvaluableState, state) for state in states]
        obs_rows = np.stack(
            [
                np.ascontiguousarray(evaluable.observation(), dtype=np.float32)
                for evaluable in evaluables
            ]
        )
        legal_rows = np.stack(
            [
                np.asarray(evaluable.legal_mask(), dtype=np.bool_)
                for evaluable in evaluables
            ]
        )
        rows = client.submit_wave(obs_rows, legal_rows)
        # reshape(-1) for the same reason make_evaluator uses it: the shape seam
        # below, not this frame, is what rejects a mis-shaped wdl (design D7).
        return [
            _value_from_wdl(
                np.asarray(row.policy_logits, dtype=np.float32).reshape(-1),
                np.asarray(row.wdl, dtype=np.float32).reshape(-1),
                draw_score,
                moves_left=_leaf_moves_left(row) if mlh_live else 0.0,
                mlh=mlh,
                context=context,
            )
            for row in rows
        ]

    def evaluate_wave(
        states: Sequence[SearchState],
    ) -> Sequence[tuple[npt.NDArray[np.float32], float]]:
        """Answer every row in the legacy two-tuple shape."""
        return [(logits, value) for logits, value, _ in answers(states)]

    def evaluate_wave_with_wdl(
        states: Sequence[SearchState],
    ) -> Sequence[tuple[npt.NDArray[np.float32], float, WdlAnswer]]:
        """Answer every row in the three-tuple shape (KOM-173 §4)."""
        return answers(states)

    # The context travels out with the closure rather than only inside it: the
    # worker has to be able to assert that its two seams hold the SAME object
    # (KOM-159b review F2), and a closure cell is not readable from there.
    if with_wdl:
        return WaveSeam(evaluate_wave_with_wdl, context)
    return WaveSeam(evaluate_wave, context)
