"""The ``Mover`` implementation: a match player built on :class:`MatchSearch`.

``Mover`` -- ``pick`` / ``observe`` / ``reset`` / ``close`` -- is the only
existing interface in the tree whose shape is "an engine that plays moves in a
game" rather than "a function that scores a position", and two of its four
methods are exactly what tree reuse needs and a stateless search has no use
for: ``observe`` is how the tree learns the opponent's reply so it can rehome,
and ``reset`` is where a retained tree is dropped at a game boundary. That is
not a coincidence, and it is the strongest available evidence that ``Mover`` is
the right altitude for this component.

Two things this module owns beyond the seam.

**The move lifecycle of the evaluation counter (design §B2).** ``pick`` calls
``begin_move`` on entry and ``end_move`` in a ``finally``, once per move, and
nothing else calls either. There is no Komugi-side consumer of ``EvalCounter``
today -- it is v1-side instrumentation -- so the counted path is *constructed*
here rather than reused.

**The refusal translation.** ``EvalCounter`` raises ``BudgetRefused``, and the
bridged game loop catches ``BridgeVoided`` and ``BudgetBreach`` only. An
untranslated refusal would escape uncaught to the arena boundary and land in
the ``crash`` bucket instead of the ``budget_breach`` one -- the same event,
filed as a different kind of failure. So the wrapper translates it at the seam,
carrying the attempted ordinal across, and a unit test asserts the translation
directly.

The ``seed`` argument ``pick`` carries is **accepted, recorded, and ignored**:
match search contains no randomness at all (design §D1), so seed-invariance is
a stronger guarantee than seed-reproducibility, and pinning it is what stops a
later "just add a little tie-break noise" from passing review unnoticed.
"""

from __future__ import annotations

from komugi.eval.v1bridge import BudgetBreach
from komugi.eval.v1runner.counter import BudgetRefused, EvalCounter
from komugi.search.match.adapter import as_match_state
from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchResult, MatchSearch, TacticalProvider
from komugi.search.match.state import MatchState, MoverState, WdlEval, WdlEvaluator


def _no_evaluator(state: MatchState) -> WdlEval:
    """Refuse to search: this player was built without an evaluator.

    A :class:`MatchPlayer` is constructible with no evaluator so that the
    parity defaults can be asserted against a bare constructor (acceptance
    A15). Such a player can report its configuration and nothing else.
    """
    raise ValueError(
        "this MatchPlayer was constructed without an evaluator, so it can "
        "report its configuration but cannot search"
    )


class _CountedEvaluator:
    """A :class:`WdlEvaluator` that charges every position to a counter.

    Charges **per position**, not per call: an evaluation is one question put
    to a network about one game position, and a counter that charged one per
    call would turn the per-move ceiling into a ceiling on calls. This
    revision never batches, so every charge is exactly one.

    Args:
        inner: The evaluator being counted.
        counter: The per-move tally.
        player_id: Named in a :class:`BudgetBreach`, so a voided match says
            which entrant overspent.
    """

    __slots__ = ("_inner", "_counter", "_player_id")

    def __init__(
        self, inner: WdlEvaluator, counter: EvalCounter, player_id: str
    ) -> None:
        self._inner = inner
        self._counter = counter
        self._player_id = player_id

    def __call__(self, state: MatchState) -> WdlEval:
        """Charge one position, then evaluate it.

        The charge happens **before** the call, so an evaluator that raises
        still costs its evaluation: the search consumed the budget by asking.

        Raises:
            BudgetBreach: Translated from ``BudgetRefused`` at this seam, so
                the refusal reaches the game loop as the budget event it is
                rather than as an unrelated crash.
        """
        try:
            self._counter.ask(1)
        except BudgetRefused as refused:
            raise BudgetBreach(refused.attempted, self._player_id) from refused
        return self._inner(state)


class MatchPlayer:
    """A ``Mover`` that picks its move with a match-play PUCT search.

    Args:
        config: The search's configuration. :attr:`MatchConfig.tree_reuse` is
            ``False`` by default and this class does not override it: reuse is
            a mode, off on any measured path (design §B2).
        evaluator: The WDL-carrying leaf evaluator. ``None`` builds a player
            that can report its configuration but refuses to search, which is
            what lets the parity defaults be asserted against a bare
            constructor.
        player_id: Carried into a :class:`BudgetBreach`.
        ceiling: The per-move evaluation ceiling. Defaults to the shared
            ``EVAL_CEILING``, which is the same ceiling the Gumbel side is
            instrumented against -- deliberately, since it is what lets one
            counter instrument both searches.
        tactical_provider: KOM-233's rules-level terminal adjudicator, passed
            straight through to the search. ``None`` -- the default -- is the
            player that shipped before, to the byte. It costs no evaluation, so
            it does not interact with `ceiling` at all.
    """

    def __init__(
        self,
        config: MatchConfig,
        evaluator: WdlEvaluator | None = None,
        *,
        player_id: str = "",
        ceiling: int | None = None,
        tactical_provider: TacticalProvider | None = None,
    ) -> None:
        self._config = config
        self._player_id = player_id
        self._counter = (
            EvalCounter() if ceiling is None else EvalCounter(ceiling=ceiling)
        )
        counted = _CountedEvaluator(
            _no_evaluator if evaluator is None else evaluator,
            self._counter,
            player_id,
        )
        self._search = MatchSearch(
            config, counted, tactical_provider=tactical_provider
        )
        self._results: list[MatchResult] = []
        self._last_seed: int | None = None
        self._pair = -1
        self._seat = -1

    @property
    def config(self) -> MatchConfig:
        """Return the configuration in force."""
        return self._config

    @config.setter
    def config(self, config: MatchConfig) -> None:
        """Install a new configuration -- a mid-game contempt change is this.

        The retained tree stays valid: nothing stored on it depends on a
        config field, so every number the next search reads is re-derived
        under the new one (design §S2).
        """
        self._config = config
        self._search.config = config

    @property
    def tree_reuse(self) -> bool:
        """Return whether this player reuses its tree across moves."""
        return self._config.tree_reuse

    @property
    def counter(self) -> EvalCounter:
        """Return the per-move evaluation counter."""
        return self._counter

    @property
    def tactical_provider(self) -> TacticalProvider | None:
        """Return the tactical provider this player searches with, or ``None``."""
        return self._search.tactical_provider

    @property
    def search_engine(self) -> MatchSearch:
        """Return the underlying search, for tests and diagnostics."""
        return self._search

    @property
    def results(self) -> tuple[MatchResult, ...]:
        """Return every move this player has made, oldest first."""
        return tuple(self._results)

    @property
    def last_result(self) -> MatchResult | None:
        """Return the most recent move's record, or ``None``."""
        return self._results[-1] if self._results else None

    @property
    def last_seed(self) -> int | None:
        """Return the seed of the last ``pick``, recorded and unused.

        Kept so that "the seed was ignored" is checkable rather than merely
        asserted in prose.
        """
        return self._last_seed

    def pick(self, state: MoverState, *, ply: int, seed: int) -> int:
        """Return a legal action id for `state`.

        Annotated with :class:`MoverState` rather than the concrete state the
        seam delivers, which is what lets this class satisfy ``Mover`` under
        ``mypy --strict`` while importing nothing from the self-play adapter:
        parameter types are contravariant, so the annotation must be a
        *supertype* of what the caller passes, and the structural protocol is
        exactly that.

        Args:
            state: The position to move in, as the ``Mover`` seam delivers it.
            ply: The ply number, used only to name a tree desync.
            seed: Accepted, recorded, and ignored -- see the module docstring.

        Returns:
            The action id to play.
        """
        self._last_seed = seed
        self._counter.begin_move()
        try:
            result = self._search.search(as_match_state(state), ply=ply)
        finally:
            self._counter.end_move()
        self._results.append(result)
        return result.action

    def observe(self, action: int, record: object) -> None:
        """Take note of an action actually played, by either side.

        `record` is the decoded move the seam passes along; a match tree needs
        only the action id, so it is accepted and unused rather than
        re-derived.
        """
        self._search.observe(action)

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game: drop the tree and the per-game record."""
        self._pair = pair
        self._seat = seat
        self._search.reset()
        self._results.clear()

    def close(self) -> None:
        """Release the tree. There is no subprocess and no socket to close."""
        self._search.reset()

    @property
    def pair(self) -> int:
        """Return the pair index of the game in progress, or ``-1``."""
        return self._pair

    @property
    def seat(self) -> int:
        """Return the seat this player was assigned, or ``-1``."""
        return self._seat
