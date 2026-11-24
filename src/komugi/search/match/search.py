"""One move in, one :class:`MatchResult` out.

This is the layer that owns design §B1's evaluation identity::

    evals(move) = [1 if the root was not already expanded else 0]
                + #{ playouts that expanded a new, non-terminal node }

and the two budget regimes §B2 separates. **Parity** -- any rated, anchor or
cross-engine match -- runs with tree reuse **off** and asserts
``nodes_at_entry == 0``, because a retained subtree carries visits paid for out
of an earlier move's budget and a move that searches a tree its own budget did
not buy makes "per-move budget" stop meaning anything. **Free play** runs with
reuse on, has no lower bound of one evaluation, and is for unrated play,
analysis and sparring. The two are not interchangeable and this module keeps
the distinction visible: :attr:`MatchResult.tree_reuse` and
:attr:`MatchResult.nodes_at_entry` are reported on every move, so the regime a
number came from is on the record beside the number.

**KOM-233's decisive/anti-decisive filter lives here, at the root, and nowhere
else.** Teytaud & Teytaud (2010): a *decisive* move ends the game in the
mover's favour and must be played; an *anti-decisive* move hands the same
thing to the opponent and must not be. Both are decided by *playing them on
the rules layer* -- the tree is not touched, no node is created, and no
statistic is written -- so a policy prior of ``1e-6`` on the reply that takes
our 帥 cannot hide it, which is exactly what happened in all three gen-8
losses in the external-anchor games. Two passes, and their order is the design:

* **before** the playouts, every legal root move is played on a clone and the
  first that ends the game in *our* favour is returned outright
  (``playouts_used == 0``) -- a proven win is not a thing to spend a budget
  deciding;
* **after** them, every root action that permits an immediate win in reply is
  struck out of the move choice, and the LCB rule chooses among what is left
  (:func:`~komugi.search.match.select.select_action`'s ``allowed`` mask). If
  *every* root move is refuted the legacy choice stands: a filter with nothing
  to offer must not invent a move.

Both passes are **exhaustive scans of the legal mask**, not surveys of a
generated candidate set. A generator that enumerates one class of move --
ordinary board captures that remove a 帥, say -- is blind to the rest of what
ends a gungi game: a drop that mates (the 付# drops that won the external-anchor games
live in blocks B and C, not A0), a checkmate that captures nothing, a
conversion. The filter is only as exact as its enumeration, so the enumeration
is the rules layer itself: clone, play, ask whether the position is terminal
and who won it.

The cost is ``2 x |legal| + sum(|replies|)`` engine transitions per move -- one
per root move in each pass, plus the replies each root move is answered with --
at roughly 0.5 ms apiece, and it is bounded twice because that product is not
small everywhere:

* **the draft is skipped outright.** While either side still holds drafting
  rights the engine suppresses every terminal (``Game.win_state`` returns
  nothing until ``draft_w`` and ``draft_b`` are both clear), so no move can end
  the game and both scans are guaranteed to find nothing. Measured on one
  drafting position: 676 legal moves, 458,328 transitions, **257 s for a single
  move** -- all of it provably wasted. The same search post-draft, at 836 legal
  moves, costs 42 s. The skip is recorded as
  :attr:`MatchTactical.skipped_reason` ``'draft'`` rather than left to be
  inferred from two zero counters.
* **the reply survey has a transition budget**
  (``MatchSearch(..., tactical_transition_cap=...)``). It checks its own
  running count before every clone, and when the cap is reached it stops
  scanning further root moves, keeps the refutations it has, and says so with
  :attr:`MatchTactical.truncated` -- a partial filter that admits it is partial.
  The decisive pass is never truncated: it is at most ``|legal|`` transitions
  and it is the pass that wins games outright.

Neither pass evaluates anything, so the per-move evaluation ceiling of §B2 is
untouched and :attr:`MatchResult.evals_used` is the same number it would have
been. What the pass did is reported on :attr:`MatchResult.tactical`, and a
search built without a provider is byte-identical to the one that shipped
before, with ``tactical`` ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Any, Protocol

import numpy as np
import numpy.typing as npt

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import MatchNode
from komugi.search.match.select import lcb_scores, node_utility, select_action
from komugi.search.match.state import MatchState, WdlEvaluator
from komugi.search.match.tree import MatchTree


class TacticalProvider(Protocol):
    """The rules-level adjudicator this search may consult.

    A **structural narrowing** of ``komugi.search.tactical.TacticalProvider``,
    declared here for the reason design §M2 gives for every other duplicated
    contract in this package: the match package imports nothing from the wider
    search package or from ``komugi.selfplay``, so the seam has to be declared
    where it is used. ``komugi.selfplay.tactical.GameMarshalCaptureProvider``
    satisfies this protocol without knowing it exists, which is the point.

    Narrowing, because the filter above does **not** enumerate through a
    provider: it scans the legal mask itself. What it needs from a provider is
    the other two thirds of the seam -- an identity to record and a categorical
    reading of who won an already terminal position -- and ``candidate_actions``
    is deliberately absent from this copy so that a provider whose enumeration
    is partial (as every real one is) cannot narrow what the filter looks at.

    The state arguments are ``Any`` because what a provider is handed is the
    **wrapped** state -- ``MatchGameState.inner``, the engine's own
    ``GameState`` -- and :class:`~komugi.search.match.state.MatchState`, the
    only state contract this package has, deliberately cannot describe it.
    :meth:`MatchSearch.search` refuses a state that cannot produce one before
    it touches anything else, and the provider's own ``validate_state`` is what
    types the result.

    The contracts are the original's, restated because they are what the search
    validates: ``terminal_winner`` answers only about an actually terminal
    position, in the engine's absolute frame, with ``0``, ``1`` or ``None`` for
    a real draw -- never a ``bool``, never any other id.
    """

    #: Stable identity of the adjudicator, recorded on every result it shapes.
    provider_id: str

    def validate_state(self, state: Any) -> None:
        """Raise ``TypeError`` if `state` is not a kind this provider supports."""
        ...

    def terminal_winner(self, terminal_state: Any) -> int | None:
        """Return the winner of an already terminal `terminal_state`, or ``None``."""
        ...


@dataclass(frozen=True, slots=True)
class MatchTactical:
    """What the KOM-233 root filter did on one move.

    Diagnostics, not a decision: every field is a record of work already done,
    so a run can answer "did the filter fire, and what did it cost" without
    re-running the search. The two counters are the cost, in the only two
    currencies the pass spends -- provider calls and engine transitions -- and
    neither includes a single network evaluation, because the pass makes none.

    Attributes:
        provider_id: The adjudicator that read the terminals, carried so a
            recorded move names its refuter rather than "whatever was wired in
            at the time".
        root_witness: The action that was played because playing it ended the
            game in the mover's favour, or ``None`` when no legal root move
            did. When it is set it **is** :attr:`MatchResult.action` and the
            move cost zero playouts.
        refuted: The root actions struck out by the reply survey, ascending.
            Empty when the survey found nothing, which is the quiet-position
            case and is not the same as the filter being off.
        overridden: The action the legacy (unfiltered) LCB rule would have
            played, when the filter changed the choice; ``None`` when it did
            not. This is the field that says the filter *mattered*.
        provider_calls: How many times the provider was asked **who won a
            terminal position** -- the only question the filter puts to it, so
            this is one call per terminal position either pass reached, and
            never one per move considered.
        candidate_transitions: How many rules-layer moves were played on
            clones by both passes together. Bounded by
            ``2 x |legal| + sum(|replies|)`` and equal to it when nothing ends
            the game early. Zero when the move was skipped.
        skipped_reason: Why both passes were skipped, or ``None`` when they
            ran. ``'draft'`` is the only reason there is: no move can end the
            game while either side is still drafting, so a scan there is
            hundreds of seconds spent proving nothing.
        truncated: Whether the reply survey stopped early on its transition
            cap. ``True`` means :attr:`refuted` is a **partial** answer -- every
            action in it really is refuted, but the scan did not reach every
            root move -- so a reader must not treat an absent action as safe.
    """

    provider_id: str
    root_witness: int | None = None
    refuted: tuple[int, ...] = ()
    overridden: int | None = None
    provider_calls: int = 0
    candidate_transitions: int = 0
    skipped_reason: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class MatchResult:
    """Everything one match-play move reports about itself.

    Attributes:
        action: The action id to play, chosen by design §A3's LCB rule.
        evals_used: Positions this move asked the network about. On a fresh
            tree this is ``1 + playouts_used`` unless a playout reached a
            terminal position, and never more than ``EVAL_CEILING``.
        playouts_used: Playouts actually run. Zero on a forced move, which
            still pays its one root evaluation -- something downstream needs
            an honest number for the position even when there is no choice to
            make.
        nodes_at_entry: How many nodes the tree held when the move began.
            **Zero on every move of a ``tree_reuse=False`` run**, which is
            what proves the parity regime rather than asserting it.
        tree_reuse: The regime this move ran under, reported rather than
            inferred from a default that a caller may have overridden.
        contempt: The contempt :attr:`root_utility` was derived under. A
            utility without the draw score it was collapsed with is not a
            number anyone can compare across moves.
        root_wdl: ``(3,)`` mean win/draw/loss backed up at the root,
            **contempt-free**: this is stored evidence, not a judgement about
            what a draw is worth.
        root_utility: ``root_wdl`` collapsed under `contempt`.
        actions: ``(L,)`` the root's legal action ids, ascending. Every array
            below is indexed by position in this one, not by action id.
        visits: ``(L,)`` per-edge visit counts.
        weights: ``(L,)`` per-edge backup weights. Equal to `visits` exactly
            while the uncertainty head is absent (design §A5).
        lcb_scores: ``(L,)`` the lower confidence bounds the choice was made
            on, ``-inf`` on an edge with no weight.
        tactical: What the KOM-233 root filter did, or ``None`` when no
            provider was wired in. Last and defaulted, so every existing
            positional construction of this class is unchanged.
    """

    action: int
    evals_used: int
    playouts_used: int
    nodes_at_entry: int
    tree_reuse: bool
    contempt: float
    root_wdl: npt.NDArray[np.float64]
    root_utility: float
    actions: npt.NDArray[np.int32]
    visits: npt.NDArray[np.int32]
    weights: npt.NDArray[np.float64]
    lcb_scores: npt.NDArray[np.float64]
    tactical: MatchTactical | None = None


def _inner_of(state: MatchState) -> Any:
    """Return the engine state `state` wraps, for a provider to read.

    :class:`~komugi.search.match.adapter.MatchGameState` publishes it as
    ``inner``; a hand-built test state need not, and a search asked to consult
    a provider about such a state has no position to hand it. That is a wiring
    error rather than a search failure, so it is refused here, loudly, before
    any work is paid for.

    Raises:
        ValueError: If `state` exposes no ``inner``.
    """
    inner = getattr(state, "inner", None)
    if inner is None:
        raise ValueError(
            "a tactical provider was wired in, but this state exposes no "
            "`inner`: only a MatchGameState carries the engine position a "
            "provider reads, and there is nothing here to hand it"
        )
    return inner


class _TacticalSurvey:
    """One move's worth of rules-layer questions, and what they cost.

    Holds the counters the two passes accumulate, so that
    :meth:`MatchSearch.search` reads as the two passes it is rather than as
    bookkeeping with a search in the middle. Created per ``search`` call and
    discarded with it: nothing here survives a move, and the tree never sees
    it.
    """

    __slots__ = (
        "_provider",
        "_cap",
        "_calls",
        "_transitions",
        "_refuted",
        "_truncated",
    )

    def __init__(self, provider: TacticalProvider, *, transition_cap: int) -> None:
        self._provider = provider
        self._cap = transition_cap
        self._calls = 0
        self._transitions = 0
        self._refuted: tuple[int, ...] = ()
        self._truncated = False

    def record(
        self,
        *,
        root_witness: int | None = None,
        overridden: int | None = None,
        skipped_reason: str | None = None,
    ) -> MatchTactical:
        """Return what this survey did, as the result's diagnostic record."""
        return MatchTactical(
            provider_id=self._provider.provider_id,
            root_witness=root_witness,
            refuted=self._refuted,
            overridden=overridden,
            provider_calls=self._calls,
            candidate_transitions=self._transitions,
            skipped_reason=skipped_reason,
            truncated=self._truncated,
        )

    def root_win(self, state: MatchState) -> int | None:
        """Return a root action that ends the game in the mover's favour.

        **Every** legal move is played, in ascending id order, and the answer
        is the first one the rules layer adjudicates as a win for the side to
        move. Ascending rather than by prior because the pass is exhaustive:
        with no enumeration to bias it there is nothing for a prior order to
        buy, and an id order is the one a replay can reproduce without the
        network. Playing is the only proof accepted -- the turn cap adjudicates
        a capture on the last ply as a draw -- so a move that merely looks
        winning is not one.

        Args:
            state: The root position. Never mutated: every trial is a clone.

        Returns:
            The lowest-id winning action, or ``None`` if none wins here.
        """
        mover = state.to_play()
        for action in _legal_actions(state):
            probe = state.clone()
            probe.play(action)
            self._transitions += 1
            if not probe.is_terminal():
                continue
            if self._winner(probe) == mover:
                return action
        return None

    def refutation_mask(
        self, state: MatchState, root: MatchNode
    ) -> npt.NDArray[np.bool_]:
        """Return ``(L,)`` over `root`'s edges, ``False`` where the reply wins.

        One ply wide and exact in both directions: for each root action, the
        position after it is scanned over its **whole** legal mask and every
        reply is played. A reply that ends the game in the **opponent's**
        favour refutes the root move, and the first such reply is enough, so
        the inner loop stops there.

        A root move that ends the game itself is skipped rather than refuted:
        it has no replies at all, and whether it is a good move is the LCB
        rule's business (a terminal child's result is already in the tree).

        This is the expensive pass -- ``|legal| + sum(|replies|)`` transitions
        -- so it is the pass that carries the budget. The count checked against
        the cap is **this pass's own**, the decisive pass having already been
        paid for in full and being the one nobody would want truncated. When
        the cap is reached the scan stops between root moves, every root move
        it never reached stays permitted (an unscanned move is not a refuted
        one), and :attr:`MatchTactical.truncated` says the answer is partial.
        """
        allowed: npt.NDArray[np.bool_] = np.ones(root.width, dtype=np.bool_)
        refuted: list[int] = []
        spent = 0
        for index in range(root.width):
            if spent >= self._cap:
                self._truncated = True
                break
            action = int(root.actions[index])
            answered = state.clone()
            answered.play(action)
            self._transitions += 1
            spent += 1
            if answered.is_terminal():
                continue
            # The reply's mover, stated from the root rather than read off the
            # answered position: outside the draft the two are the same number,
            # and this one cannot be anything but the opponent. (Inside the
            # draft to_play need not alternate -- which is moot here, because
            # the draft is skipped before this pass is ever called.)
            opponent = 1 - state.to_play()
            for reply in _legal_actions(answered):
                if spent >= self._cap:
                    self._truncated = True
                    break
                refutation = answered.clone()
                refutation.play(reply)
                self._transitions += 1
                spent += 1
                if (
                    refutation.is_terminal()
                    and self._winner(refutation) == opponent
                ):
                    allowed[index] = False
                    refuted.append(action)
                    break
        self._refuted = tuple(refuted)
        return allowed

    def _winner(self, terminal: MatchState) -> int | None:
        """Return who won `terminal` according to the provider, validated.

        The validation is not a formality: this answer is compared against a
        player index and decides whether a move is played or struck out, so an
        adjudicator that returns ``True``, ``2`` or a numpy scalar must be
        refused rather than silently compared. ``bool`` is an ``int`` in Python
        and is not a player.

        Raises:
            ValueError: If the answer is neither ``None`` nor the plain ``int``
                ``0`` or ``1``.
        """
        answer: Any = self._provider.terminal_winner(_inner_of(terminal))
        self._calls += 1
        if answer is None:
            return None
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise ValueError(
                f"{self._provider.provider_id}: terminal_winner answered "
                f"{answer!r}, which is not a plain int player index"
            )
        if answer not in (0, 1):
            raise ValueError(
                f"{self._provider.provider_id}: terminal_winner answered "
                f"{answer}, and the only players are 0 and 1"
            )
        return int(answer)


#: :attr:`MatchTactical.skipped_reason` when the draft suppressed both passes.
_SKIPPED_DRAFT: Final[str] = "draft"


def _draft_is_open(state: MatchState) -> bool:
    """Return whether **either** side still holds drafting rights.

    Either, not the mover's: ``Game.win_state`` suppresses every terminal while
    ``draft_w or draft_b``, so a position where only the opponent is still
    drafting is just as incapable of ending, and a mover-relative reading would
    pay the full survey for it.

    Duck-typed, and quiet about a state that cannot answer. This package may
    not import the engine or the self-play adapter (design §M2), the draft is a
    *phase* and :class:`~komugi.search.match.state.MatchState` deliberately
    knows nothing about phases, and a hand-built test state has no draft at
    all. So: the engine ``Game`` behind the adapter if it is reachable
    (``in_draft(side)``, per side), the adapter's own mover-relative
    ``in_draft()`` if it is not, and ``False`` -- scan, do not skip -- if
    neither exists. Guessing "drafting" for a state that cannot say would turn
    the filter off silently, which is the one failure mode worth refusing.
    """
    inner: Any = getattr(state, "inner", None)
    if inner is None:
        return False
    game: Any = getattr(inner, "game", None)
    per_side: Any = getattr(game, "in_draft", None)
    if per_side is not None:
        return bool(per_side(0)) or bool(per_side(1))
    mover_relative: Any = getattr(inner, "in_draft", None)
    if mover_relative is not None:
        return bool(mover_relative())
    return False


def _legal_actions(state: MatchState) -> list[int]:
    """Return `state`'s legal action ids, ascending.

    The filter's enumeration, and the whole of it: the mask is the rules
    layer's own answer about what may be played, so scanning it is what makes
    both passes exact across drops, board moves and everything else the action
    space encodes.
    """
    return [int(action) for action in np.flatnonzero(state.legal_mask())]


class MatchSearch:
    """A reusable match search: one evaluator, one config, one retained tree.

    Unlike ``GumbelSearch``, instances are **not** stateless between calls:
    the tree is the point. :meth:`observe` is how the caller tells the search
    what was actually played, and :meth:`reset` is how a game boundary drops
    everything.

    Args:
        config: The search's configuration.
        evaluator: The WDL-carrying leaf evaluator, held for the life of the
            search.
        tactical_provider: KOM-233's rules-level terminal adjudicator, or
            ``None`` -- the default, and the shipped behaviour to the byte.
        tactical_transition_cap: The reply survey's transition budget, in
            engine moves played on clones. The default is about ten seconds of
            scanning at the measured ~0.5 ms a transition, which is the right
            order against an anchor that thinks for 10-30 s a move; a survey
            that reaches it keeps what it found and reports
            :attr:`MatchTactical.truncated`. Ignored when no provider is
            wired in.
    """

    #: Reply-survey transitions one move may spend before the scan stops.
    DEFAULT_TRANSITION_CAP: int = 20000

    def __init__(
        self,
        config: MatchConfig,
        evaluator: WdlEvaluator,
        *,
        tactical_provider: TacticalProvider | None = None,
        tactical_transition_cap: int = DEFAULT_TRANSITION_CAP,
    ) -> None:
        self._config = config
        self._tree = MatchTree(config, evaluator)
        self._tactical_provider = tactical_provider
        self._tactical_transition_cap = int(tactical_transition_cap)

    @property
    def tactical_provider(self) -> TacticalProvider | None:
        """Return the tactical provider in force, or ``None``."""
        return self._tactical_provider

    @property
    def tactical_transition_cap(self) -> int:
        """Return the reply survey's per-move transition budget."""
        return self._tactical_transition_cap

    @property
    def config(self) -> MatchConfig:
        """Return the configuration in force."""
        return self._config

    @config.setter
    def config(self, config: MatchConfig) -> None:
        """Install a new configuration, on this search and its tree alike."""
        self._config = config
        self._tree.config = config

    @property
    def tree(self) -> MatchTree:
        """Return the retained tree, for tests and diagnostics."""
        return self._tree

    def observe(self, action: int) -> None:
        """Tell the search that `action` was actually played.

        Rehomes the tree onto the played child under reuse, and drops it
        outright otherwise. Called for **both** sides' moves.
        """
        if self._config.tree_reuse:
            self._tree.descend(action)
        else:
            self._tree.clear()

    def reset(self) -> None:
        """Drop the tree at a game boundary."""
        self._tree.clear()

    def search(self, state: MatchState, *, ply: int = -1) -> MatchResult:
        """Run one move's search from `state` and return what it decided.

        `state` is never mutated: every playout walks a clone.

        Args:
            state: The position to move in.
            ply: The ply number, used only to name a desync in the exception
                it raises.

        Returns:
            The move and the per-move budget record design §B2 requires.

        Raises:
            ValueError: If `state` has no legal action, or is terminal while
                reporting several -- neither is a position a search can say
                anything about.
            MatchTreeDesync: If a retained root describes a different
                position than `state` (design §A4).
        """
        config = self._config
        tree = self._tree
        provider = self._tactical_provider

        if provider is not None:
            # Before the tree, before the root evaluation, before anything: a
            # provider that cannot read this state is a wiring error, and a
            # wiring error found after a move has been paid for is a wiring
            # error found too late.
            provider.validate_state(_inner_of(state))

        if not config.tree_reuse:
            # The parity regime is enforced here rather than trusted to the
            # caller's `observe`: a move that begins with an empty tree cannot
            # spend an earlier move's budget, and `nodes_at_entry == 0` below
            # is then a measurement rather than a promise.
            tree.clear()
        nodes_at_entry = tree.node_count
        evaluations_before = tree.evaluations

        tree.check_consistency(state, ply=ply)

        legal = np.flatnonzero(state.legal_mask())
        if int(legal.shape[0]) == 0:
            raise ValueError(
                "match search on a state with no legal actions: there is no "
                "move to return"
            )
        if state.is_terminal():
            raise ValueError(
                f"match search on a terminal state that reports "
                f"{int(legal.shape[0])} legal actions"
            )

        tree.ensure_root(state)
        root = tree.root
        if root is None:  # pragma: no cover - ensure_root just built it
            raise ValueError("the root vanished between expansion and search")

        survey = (
            _TacticalSurvey(provider, transition_cap=self._tactical_transition_cap)
            if provider is not None
            else None
        )
        # Both passes are skipped wholesale inside the draft: the engine cannot
        # adjudicate a terminal while either side still has pieces to place, so
        # there is nothing for either scan to find and the scan is the most
        # expensive thing the move would do (a measured 257 s for one drafting
        # move). The skip is recorded, not inferred.
        drafting = survey is not None and _draft_is_open(state)
        if survey is not None and not drafting:
            witness = survey.root_win(state)
            if witness is not None:
                # A proven win, played without spending a playout on deciding
                # whether to. The record is the root's statistics as they stand
                # -- one evaluation, no descent -- because that is what this
                # move actually bought.
                return self._result(
                    root,
                    action=witness,
                    playouts_used=0,
                    nodes_at_entry=nodes_at_entry,
                    evaluations_before=evaluations_before,
                    tactical=survey.record(root_witness=witness),
                )

        playouts_used = 0
        if root.width > 1:
            for _ in range(config.playouts):
                tree.playout(state)
                playouts_used += 1

        # The move choice reads the design §A2-corrected child utilities, like
        # the descent does: §A2 names selection *and* LCB as its two readers,
        # and a choice made on uncorrected means would disagree with the
        # search that produced them.
        index = select_action(root, config, bias=tree.bias)
        tactical: MatchTactical | None = None
        if survey is not None and drafting:
            tactical = survey.record(skipped_reason=_SKIPPED_DRAFT)
        elif survey is not None:
            allowed = survey.refutation_mask(state, root)
            # Three cases, and only the middle one is the filter doing
            # anything. Nothing refuted: the legacy choice **verbatim**, not
            # the same choice recomputed under an all-True mask -- the masked
            # path's fallback is max-LCB among the permitted and the legacy
            # fallback is max-weight, so recomputing could move the move on a
            # position where the filter found nothing to say. Everything
            # refuted: the legacy choice again, because a filter with nothing
            # to offer must not invent a preference.
            if bool(allowed.all()):
                filtered = index
            elif bool(allowed.any()):
                filtered = select_action(
                    root, config, bias=tree.bias, allowed=allowed
                )
            else:
                filtered = index
            tactical = survey.record(
                overridden=int(root.actions[index]) if filtered != index else None
            )
            index = filtered
        return self._result(
            root,
            action=int(root.actions[index]),
            playouts_used=playouts_used,
            nodes_at_entry=nodes_at_entry,
            evaluations_before=evaluations_before,
            tactical=tactical,
        )

    def _result(
        self,
        root: MatchNode,
        *,
        action: int,
        playouts_used: int,
        nodes_at_entry: int,
        evaluations_before: int,
        tactical: MatchTactical | None,
    ) -> MatchResult:
        """Return the move record for `action`, read off `root` as it stands.

        The single place a :class:`MatchResult` is built, so that the move a
        proven root win returns is reported on exactly the same statistics --
        and the same §B1 evaluation identity -- as the move a full search
        returns.
        """
        config = self._config
        tree = self._tree
        total = root.total_weight()
        root_wdl = (
            np.asarray(root.subtree_wdl_sum() / total, dtype=np.float64)
            if total > 0.0
            else np.asarray(root.wdl_net, dtype=np.float64).copy()
        )
        return MatchResult(
            action=action,
            evals_used=tree.evaluations - evaluations_before,
            playouts_used=playouts_used,
            nodes_at_entry=nodes_at_entry,
            tree_reuse=config.tree_reuse,
            contempt=config.contempt,
            root_wdl=root_wdl,
            root_utility=node_utility(root, config),
            actions=root.actions.copy(),
            visits=root.visits.copy(),
            weights=root.weight.copy(),
            lcb_scores=lcb_scores(root, config, bias=tree.bias),
            tactical=tactical,
        )
