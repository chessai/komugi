"""Komugi's side of §D1's move seam: the arena's own search, counted.

The cross-engine harness (:mod:`komugi.eval.v1bridge`) dispatches a game on
:class:`~komugi.eval.v1bridge.Mover` -- "choose one action in this position" --
and KOM-023a/b built only the *remote* half of that seam
(:class:`~komugi.eval.v1bridge.V1RemoteMover`, one v1 subprocess behind a wire).
This module is the near half: a Komugi checkpoint, seated at the same seam, so a
bridged match can actually have a Komugi entrant on one side of it.

Two properties are the whole point, and each is one construction below.

**It is the arena's search, not a search like it.** ``pick`` calls
:func:`komugi.eval.arena.move_search`, the same factory
``arena._play_wired_game`` builds every ladder move with, at
:data:`~komugi.eval.arena.DEFAULT_M` = 32 and :data:`~komugi.eval.arena.DEFAULT_N`
= 256 by default (§D3.3). A rung rated against a v1 anchor and the same rung
rated against another rung must be the *same player*, or the two numbers cannot
be put on one ladder; that is a claim about code paths, and the only honest way
to make it is to share the path rather than to re-type it.

**Every question to the network is counted at the seam** (§D3.5). The evaluator
the search is handed is a :class:`CountedEvaluator`, wrapped *before* either
adapter sees it, so both shapes of the search seam -- the scalar root/forced-move
call (:func:`komugi.selfplay.adapter.make_evaluator`) and the batched wave
(:func:`komugi.eval.arena.make_batch_wave_evaluator`) -- reach the network
through one counter and neither can spend budget the other cannot see. The
charge is *positions*, not calls: a wave of 12 leaves costs 12, because §D3.2's
unit is a question about one position and a per-call charge would let a batching
schedule buy the ceiling many times over.

The 258th evaluation inside one move is **refused**, not served
(:class:`~komugi.eval.v1runner.counter.EvalCounter`'s own rule, reused verbatim),
and the refusal leaves this module as :class:`~komugi.eval.v1bridge.BudgetBreach`
-- which :func:`~komugi.eval.v1bridge.play_bridged_game` already resolves as §D6
M8, a match-level void. Nothing truncated is ever scored.

Import direction, deliberately one-way
--------------------------------------
``komugi.eval.gumbel_mover`` imports :mod:`komugi.eval.arena` and
:mod:`komugi.eval.v1bridge`; neither imports this module, and neither may be made
to. ``v1bridge`` already documents why *it* may import the arena (the seam is
the move, and the arena owns the game loop the bridge mirrors); a back-edge from
either into here would put a Komugi net and its torch import inside the module
the v1 runner's drift test has to keep stdlib-clean by inspection.

What this stage does **not** do
-------------------------------
No tree reuse: ``observe`` is a no-op and :attr:`GumbelMover.tree_nodes_at_entry`
is a literal ``0`` reported every move rather than an unmeasured field. That is
§D3.2 clause 2 satisfied by construction -- :class:`GumbelSearch` keeps no state
between calls at all -- and it is the number a P3 check reads. When tree reuse
lands, this is the field that stops being a constant, and the harness's existing
M9 detector is already watching it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from komugi.eval.arena import (
    DEFAULT_GUMBEL_SCALE,
    DEFAULT_M,
    DEFAULT_N,
    LADDER_FAMILY,
    checkpoint_evaluator,
    make_batch_wave_evaluator,
    move_search,
    pin_determinism,
)
from komugi.eval.records import Player
from komugi.eval.v1bridge import BridgeEntrant, BudgetBreach, MoveRecord, Mover
from komugi.eval.v1runner.counter import BudgetRefused, EvalCounter
from komugi.eval.v1runner.protocol import EVAL_CEILING
from komugi.search.modes import (
    SearchAlgorithmIdentity,
    search_algorithm_identity,
    validate_search_modes,
)
from komugi.search.state import Evaluator, WaveEvaluator
from komugi.selfplay.adapter import GameState, make_evaluator
from komugi.selfplay.inference import BatchEvaluator, NetEval

NO_TREE_REUSE: Final[int] = 0
"""``tree_nodes_at_entry`` for a search that carries nothing across a move.

Reported on every move rather than left unset: §D3.2 clause 2 is a *claim* the
harness checks (M9/P3), and a mover that declined to state its tree size would
make the claim unfalsifiable on the side that currently satisfies it trivially.
"""

EvaluatorSource: TypeAlias = str | os.PathLike[str] | Callable[[], BatchEvaluator]
"""What a :class:`GumbelMover` may be built from: a checkpoint, or a factory.

The path form is the production one -- it goes through
:func:`komugi.eval.arena.checkpoint_evaluator`, i.e. the loader every ladder rung
is built with, plane-count guard included. The factory form is what
``arena.ladder_players`` already hands around (:class:`ArenaPlayer` takes a
factory, not an evaluator, so that the caller decides when a net is built and how
long it lives), and it is how a test seats a tiny net without writing a
checkpoint to disk.
"""


def _refuse_discovery(proof_mode: str, discovery_mode: str, player_id: str) -> None:
    """Validate the mode pair and refuse discovery at the rated seam.

    Called as the **first** statement of every rated construction door
    (:class:`GumbelMover`, :func:`komugi_entrant`), before weights are loaded,
    before an evaluator exists and before any counted row is spent: a
    configuration this seam cannot play must cost nothing and must not be
    mistakable for a mid-match :class:`~komugi.eval.v1bridge.BudgetBreach`.

    Raises:
        ValueError: If the pair is not runnable at all (:mod:`komugi.search.modes`),
            or if `discovery_mode` is anything but ``off``.
    """
    validate_search_modes(proof_mode, discovery_mode)
    if discovery_mode != "off":
        who = player_id or "a rated mover"
        raise ValueError(
            f"{who}: discovery_mode {discovery_mode!r} is not available at the "
            f"rated seam. The KOM-220 discovery layer verifies the root and "
            f"every initial candidate's reply, which can put more rows through "
            f"the counter than §D3.3's ceiling of {EVAL_CEILING} allows, so the "
            f"match would void as M8 on ordinary play. Rating a tactical "
            f"entrant needs a protocol with its own ceiling; use "
            f"discovery_mode='off' here"
        )


def _rows_in(obs: npt.NDArray[np.float32]) -> int:
    """Return how many positions one ``evaluate`` call asks about.

    The batch seam's ``obs`` is ``(N, in_planes, 9, 9)`` in both of its shapes:
    the scalar adapter passes ``obs[None]`` (``adapter.py:310``) and the wave
    adapter stacks the round's leaves. So the count is the leading axis, and a
    scalar-looking array with no leading axis is a caller that is not on this
    seam.

    Raises:
        ValueError: If `obs` has no leading axis to count.
    """
    array = np.asarray(obs)
    if array.ndim < 1:
        raise ValueError(
            f"an evaluation batch must carry a leading row axis, got shape "
            f"{array.shape!r}: §D3.2 counts positions, and a 0-d observation "
            f"names none"
        )
    return int(array.shape[0])


class CountedEvaluator:
    """§D3.5's Komugi-side instrumentation: a counted batch evaluator.

    A :class:`~komugi.selfplay.inference.BatchEvaluator` in every respect, so it
    drops into either adapter of the search seam without either of them knowing.

    The counter sits **above** the network and below nothing: ``evaluate`` is the
    single seam the search reaches a net through (``inference.py:283-289``), so a
    search that wants an uncounted answer has to reach past this object, and the
    only way to do that is :attr:`inner` -- a visible, greppable escape hatch
    rather than an ``__getattr__`` proxy that would leak silently (the same
    argument :class:`~komugi.eval.v1runner.counter.CountedNet` makes on the v1
    side).

    The ask happens **before** the forward, so an evaluation the network fails to
    produce still costs its budget: the search consumed the question by asking
    it, and a counter that only counted successes would let a flaky net buy extra
    ones.

    **Rows and calls are two numbers, and this object holds both** (kom-011d U2).
    The *budget* is rows and only rows -- that is §D3.2's unit and nothing about
    it moves -- but the arena's cost per move is paid per **forward**, which is
    launch-bound on the campaign's box (``tickets/KOM-048.md:10-14``), so a cost
    model needs the mean wave size ``served / calls``. :attr:`calls` is that
    denominator: incremented once per :meth:`evaluate`, **never** inside
    :func:`_rows_in`, and carried here rather than on
    :class:`~komugi.eval.v1runner.counter.EvalCounter` -- the counter is the
    per-move budget, reset at every ``begin_move``, and it is v1's module as much
    as ours, while a wave-size probe is a property of *this* seam over a whole
    match.

    Args:
        inner: The evaluator that actually answers.
        counter: The tally, shared with whoever begins and ends the move.
    """

    def __init__(self, inner: BatchEvaluator, counter: EvalCounter) -> None:
        self._inner = inner
        self._counter = counter
        self._calls = 0

    @property
    def inner(self) -> BatchEvaluator:
        """Return the wrapped evaluator. For construction and diagnostics only."""
        return self._inner

    @property
    def counter(self) -> EvalCounter:
        """Return the counter this evaluator reports to."""
        return self._counter

    @property
    def calls(self) -> int:
        """Return how many ``evaluate`` calls this evaluator has served.

        **Calls, not positions.** One wave of 12 leaves is 12 on the counter's
        ``served`` and exactly ``1`` here; the ratio of the two is the mean wave
        size kom-011d's §D2.4 cost bracket is a function of (U2). Never reset --
        the budget is per move and this is per evaluator, so a match's whole
        forward count is readable at the end of it.
        """
        return self._calls

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Answer one batch, charging it its **row count** first.

        The call is counted **after** the ask and before the forward, which
        keeps :attr:`calls` and the counter's ``served`` consistent as a pair:
        a refused batch is not served, so it buys neither rows nor a call, and
        a batch the network fails on has already cost both. Nothing about the
        row charge or the ceiling depends on it.

        Raises:
            BudgetRefused: If this move has already spent its ceiling, or if this
                call's rows would carry it past. The batch is refused whole:
                serving part of it is the truncation §D3.5 forbids.
        """
        self._counter.ask(_rows_in(obs))
        self._calls += 1
        return self._inner.evaluate(obs, legal)


class GumbelMover:
    """A :class:`~komugi.eval.v1bridge.Mover` that is one Komugi checkpoint.

    Structural, not nominal: nothing here subclasses ``Mover`` (a
    :class:`typing.Protocol`), and the import direction is
    ``gumbel_mover -> v1bridge`` only. :func:`komugi_entrant`'s factory is
    annotated ``-> Mover``, which is where the type checker actually verifies the
    four methods.

    One search per move, built by :func:`komugi.eval.arena.move_search` and
    thrown away: :class:`~komugi.search.gumbel.GumbelSearch` is stateless between
    calls, so §D3.2 clause 2 ("no tree crosses a move") holds by construction and
    :attr:`tree_nodes_at_entry` is honestly ``0``.

    Budget accounting, per move (§D3.4):

    * a **forced** position -- one legal action -- costs exactly one evaluation
      and runs zero simulations, because the search's forced-move path still
      takes the root prior through the counted evaluator
      (``gumbel.py:836-870``);
    * any other position runs exactly ``n`` simulations and costs at most
      ``1 + n`` evaluations: one root, and at most one leaf per simulation (a
      simulation that lands on a terminal node asks nothing). At the pinned
      budget that is 257, which is why §D3.3's ceiling is 257 and not 258.

    Args:
        source: A checkpoint path, or a factory that builds the evaluator; see
            :data:`EvaluatorSource`. Built **now**, once, so a missing or
            unverifiable checkpoint fails at match construction rather than at
            the first ply of the first game.
        player_id: The rateable identity, carried into
            :class:`~komugi.eval.v1bridge.BudgetBreach` so an M8 void names who
            breached.
        draw_score: What a draw is worth to the side to move. Required, with no
            default, for design D3's reason -- it is the contempt the games are
            played under and it reaches both the terminal values and the network
            values. **It must match the** :class:`~komugi.eval.arena.ArenaConfig`
            **the match is played under**; the seam cannot check that, because
            ``pick`` is handed a position and not a config.
        m: Root actions sampled per move.
        n: Simulation budget per move.
        gumbel_scale: Multiplier on the root Gumbel noise.
        ceiling: §D3.3's per-move evaluation ceiling. The 258th ask within one
            move raises :class:`~komugi.eval.v1bridge.BudgetBreach`.
        device: Where a checkpoint's net is placed. Ignored for the factory form,
            which has already decided.
        proof_mode: The proof layer this mover searches with. ``interval_v1`` is
            allowed at this seam: it reads the tree the ordinary search already
            built and asks the network nothing extra, so the counted ceiling it
            plays under is the same 257.
        discovery_mode: **Must be** ``off``. See the refusal below.

    Why discovery is refused here
    -----------------------------
    A rated mover is the seam §D3.3's ceiling is enforced at, and the ceiling is
    257 because the ordinary search asks at most ``1 + n`` questions. The KOM-220
    discovery layer verifies the provider's candidates at the root *and* at every
    initial candidate's reply, which at the pinned budget can put up to 289 rows
    through the same counter -- so a discovery mover would breach on ordinary
    play and void its own match as M8. Rating a tactical entrant needs a ceiling
    negotiated for it, i.e. a protocol change, not a flag; until there is one
    this constructor refuses **before** it loads any weights, builds any
    evaluator or spends any counted row, so the refusal costs nothing and cannot
    be mistaken for a mid-match budget failure.
    """

    def __init__(
        self,
        source: EvaluatorSource,
        *,
        draw_score: float,
        player_id: str = "",
        m: int = DEFAULT_M,
        n: int = DEFAULT_N,
        gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
        ceiling: int = EVAL_CEILING,
        device: str = "cpu",
        proof_mode: str = "off",
        discovery_mode: str = "off",
    ) -> None:
        # First, before `pin_determinism`, before the factory, before a single
        # counted row: an unrunnable configuration must not have side effects.
        _refuse_discovery(proof_mode, discovery_mode, player_id)
        self._proof_mode = proof_mode
        self._discovery_mode = discovery_mode
        pin_determinism()
        self._player_id = player_id
        self._m = int(m)
        self._n = int(n)
        self._gumbel_scale = float(gumbel_scale)
        self._draw_score = float(draw_score)
        self._counter = EvalCounter(ceiling)
        evaluator = _factory_for(source, device)()
        self._evaluator: BatchEvaluator | None = evaluator
        counted = CountedEvaluator(evaluator, self._counter)
        # Both seams over the ONE counted object, exactly as ``arena._wire``
        # puts both seams over one evaluator: a player cannot answer two
        # different ways depending on which schedule reached it, and here it
        # cannot pay two different budgets either.
        self._scalar: Evaluator = make_evaluator(counted, self._draw_score)
        self._wave: WaveEvaluator = make_batch_wave_evaluator(
            counted, self._draw_score
        )
        self._pair: int | None = None
        self._seat: int | None = None
        self.evals_used: int = 0
        """Evaluations the **last** ``pick`` spent, for the harness's record."""
        self.sims_used: int = 0
        """Simulations the last ``pick`` ran: ``0`` forced, ``n`` otherwise."""
        self.tree_nodes_at_entry: int = NO_TREE_REUSE
        """Nodes carried into the last ``pick``. Always ``0`` -- no tree reuse."""
        self.evals: list[int] = []
        """Per-move ``evals_used`` in ply order, across the whole match."""
        self.sims: list[int] = []
        """Per-move ``sims_used`` in ply order, across the whole match."""

    @property
    def player_id(self) -> str:
        """Return the rateable identity this mover plays as."""
        return self._player_id

    @property
    def counter(self) -> EvalCounter:
        """Return the evaluation counter at this player's network seam."""
        return self._counter

    @property
    def m(self) -> int:
        """Return the root candidate count this player searches at."""
        return self._m

    @property
    def n(self) -> int:
        """Return the simulation budget this player searches at."""
        return self._n

    @property
    def gumbel_scale(self) -> float:
        """Return the root noise scale this player searches at."""
        return self._gumbel_scale

    @property
    def draw_score(self) -> float:
        """Return the contempt this player evaluates under."""
        return self._draw_score

    @property
    def proof_mode(self) -> str:
        """Return the proof layer this player searches with."""
        return self._proof_mode

    @property
    def discovery_mode(self) -> str:
        """Return this player's discovery layer -- always ``off`` here."""
        return self._discovery_mode

    @property
    def search_identity(self) -> SearchAlgorithmIdentity:
        """Return the algorithm this mover's modes name, for the record.

        Derived from the modes that actually reach :func:`move_search`, so a
        bridged game is reported under the search it was played with.
        """
        return search_algorithm_identity(self._proof_mode, self._discovery_mode)

    @property
    def seating(self) -> tuple[int | None, int | None]:
        """Return the ``(pair, seat)`` of the game in progress, if any."""
        return (self._pair, self._seat)

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Return the action the arena's search chooses in `state`.

        `state` is not mutated: the search works on clones, so the loop's own
        position is untouched and the referee stays the referee.

        `ply` is not read. It is the harness's coordinate for the move, and the
        randomness this mover consumes is `seed` -- which the loop derives from
        the ply anyway (``move_seed(seeds, ply)``, the same derivation self-play
        uses). Deriving anything from `ply` *here* would be a second, silently
        divergent seed schedule.

        Args:
            state: The position to move in, held by the arena's referee.
            ply: The ply index, for the caller's records. Unused here.
            seed: This move's Gumbel seed. The same state and seed give the same
                action, whatever else has happened in between.

        Returns:
            A legal action id.

        Raises:
            BudgetBreach: On the ceiling-plus-first evaluation inside this move
                (§D3.5). :func:`~komugi.eval.v1bridge.play_bridged_game` catches
                it and voids the **match** as M8; nothing truncated is scored.
            RuntimeError: If the mover has been closed.
        """
        del ply
        if self._evaluator is None:
            raise RuntimeError(
                f"{self._player_id or 'this mover'} was closed and its evaluator "
                f"released; build a fresh one from the entrant's factory rather "
                f"than picking through a dead player"
            )
        self._counter.begin_move()
        try:
            result = move_search(
                self._scalar,
                self._wave,
                m=self._m,
                n=self._n,
                gumbel_scale=self._gumbel_scale,
                seed=seed,
                proof_mode=self._proof_mode,
                discovery_mode=self._discovery_mode,
            ).search(state)
        except BudgetRefused as refused:
            # In-process, so there is no wire to carry a `budget_breach`
            # response: the exception itself is §D3.5's report, and the arena
            # boundary is where it is resolved (v1bridge.py:1907-1924).
            raise BudgetBreach(refused.attempted, self._player_id) from refused
        finally:
            self._counter.end_move()
        self.evals_used = self._counter.served
        self.sims_used = int(result.sims_used)
        self.tree_nodes_at_entry = NO_TREE_REUSE
        self.evals.append(self.evals_used)
        self.sims.append(self.sims_used)
        return int(result.action)

    def observe(self, action: int, record: MoveRecord) -> None:
        """Note a played move. A no-op: this stage carries no tree.

        Deliberately not "unimplemented". §D1 notifies both movers of every
        action so that a stateful adapter can keep its own board in step; this
        mover has no board of its own -- it searches the referee's position,
        handed to it -- and no tree to re-root, so there is nothing for the
        notification to update. When tree reuse lands, this is the method that
        stops being empty.
        """
        del action, record

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game: record the coordinates, drop nothing.

        There is nothing to drop -- no board, no tree, no per-game cache -- and
        the evaluator deliberately survives, exactly as ``arena.play_match``
        builds one net per *match* rather than one per game. The per-ply
        :attr:`evals` and :attr:`sims` lists also survive, so a match's budget
        record is a match's, not the last game's.
        """
        self._pair = pair
        self._seat = seat

    def close(self) -> None:
        """Release the evaluator, and refuse to play afterwards.

        Idempotent, because :class:`~komugi.eval.v1bridge._MoverSlot` closes a
        mover on renewal *and* again in the match's ``finally``. A wrapped
        evaluator that owns something (a shared-memory client, a CUDA context) is
        told; :class:`~komugi.selfplay.inference.InProcessEvaluator` owns only a
        torch module and has nothing to say, which is why the call is
        conditional rather than required by the seam.
        """
        evaluator = self._evaluator
        self._evaluator = None
        if evaluator is None:
            return
        closer = getattr(evaluator, "close", None)
        if callable(closer):
            closer()


def _factory_for(
    source: EvaluatorSource, device: str
) -> Callable[[], BatchEvaluator]:
    """Return the evaluator factory `source` names.

    A path goes through :func:`komugi.eval.arena.checkpoint_evaluator` -- the
    ladder's own loader -- so a checkpoint seated at the move seam and the same
    checkpoint seated as a ladder rung are the same weights, verified the same
    way, on the same device.
    """
    if isinstance(source, (str, os.PathLike)):
        return checkpoint_evaluator(source, device)
    return source


def komugi_entrant(
    player_id: str,
    checkpoint: EvaluatorSource,
    *,
    draw_score: float,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
    ceiling: int = EVAL_CEILING,
    device: str = "cpu",
    proof_mode: str = "off",
    discovery_mode: str = "off",
) -> BridgeEntrant:
    """Return a Komugi :class:`~komugi.eval.v1bridge.BridgeEntrant` for `checkpoint`.

    The Komugi counterpart of :func:`komugi.eval.v1bridge.v1_entrant`, and the
    thing a cross-engine match needs on its near side. Three details are
    load-bearing:

    * the family is :data:`komugi.eval.arena.LADDER_FAMILY`, so the roster map
      that travels with the anchor (§D4 amendment 3) says which engine this row
      is without anyone parsing an id prefix;
    * the factory builds a **fresh** :class:`GumbelMover` per call -- once per
      match, and once more per §D6 M1 respawn -- rather than closing over a live
      one, so a renewed slot gets a freshly loaded net rather than a mover that
      has already been ``close``d;
    * ``m``, ``n`` and ``gumbel_scale`` are carried on the entrant *as well as*
      into the mover. The entrant's copies are what
      :func:`~komugi.eval.v1bridge.anchor_configuration` publishes as §D3.6 P5's
      configuration block, and passing the same three values to both is what
      keeps the published budget the budget that was played.

    Args:
        player_id: The rateable identity, e.g. ``ckpt-120000``.
        checkpoint: A checkpoint path or an evaluator factory
            (:data:`EvaluatorSource`).
        draw_score: The contempt the games are played under. Pass the
            :class:`~komugi.eval.arena.ArenaConfig`'s own ``draw_score``: the
            seam has no way to reconcile the two, and §D3.4 pre-registers ``0.0``
            for every cross-family match.
        m: Root actions sampled per move; the arena's 32 by default.
        n: Simulation budget per move; the arena's 256 by default.
        gumbel_scale: Root noise scale; the search's own default.
        ceiling: §D3.3's per-move evaluation ceiling.
        device: Where a checkpoint's net is placed.
        proof_mode: The proof layer the movers this entrant builds search with,
            carried on the entrant too so the published configuration block names
            the search that was played.
        discovery_mode: Must be ``off``; refused **here**, at entrant
            construction, rather than at the first ``build()`` -- a match that
            cannot be played should not be schedulable (see
            :class:`GumbelMover`).

    Returns:
        The entrant, ready for :func:`~komugi.eval.v1bridge.play_bridged_match`.

    Raises:
        ValueError: On an unrunnable mode pair, or any non-``off``
            `discovery_mode`.
    """
    _refuse_discovery(proof_mode, discovery_mode, player_id)

    def build() -> Mover:
        """Build this side's mover, loading the checkpoint on the call."""
        return GumbelMover(
            checkpoint,
            draw_score=draw_score,
            player_id=player_id,
            m=m,
            n=n,
            gumbel_scale=gumbel_scale,
            ceiling=ceiling,
            device=device,
            proof_mode=proof_mode,
            discovery_mode=discovery_mode,
        )

    return BridgeEntrant(
        player=Player(id=player_id, family=LADDER_FAMILY),
        mover_factory=build,
        m=m,
        n=n,
        gumbel_scale=gumbel_scale,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
    )
