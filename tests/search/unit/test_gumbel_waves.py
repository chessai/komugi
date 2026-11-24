"""Wave-scheduled Gumbel search: the same search, asked in batches (KOM-040 b2).

The wave path exists to cut network round trips, and it is worth nothing if it
changes what the search returns. So the load-bearing test here is an
*equivalence golden*: an exactly deterministic stub evaluator, offered to the
search both one position at a time and a wave at a time, must produce
:class:`~komugi.search.SearchResult` values that agree field by field and
bit for bit -- not approximately, not distributionally.

That is a checkable claim rather than a hopeful one because of the independence
argument the KOM-040 design review verified against the search code
(``docs/reviews/2026-08-24-kom-040-worker-lane-design-review.md``): within a
sequential-halving round the surviving root actions' simulations touch disjoint
state, so grouping one visit of each into a wave is a reordering of independent
work. Equivalence alone, though, is also what a scheduler that quietly
degenerated to one position per wave would report -- it would be a correct
search with none of the benefit -- so the second test here observes the wave
*sizes* and pins them to the round's arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.search.gumbel import _halving_sizes
from komugi.search.state import AnyWaveEvaluator, WaveEvaluator, WdlEvaluator


class ToyState:
    """A depth-limited toy position: fixed legal set, alternating movers.

    `history` seeds the position, which is how one toy class supplies both a
    mid-game root (deep enough that no leaf inside the budget is terminal) and
    a near-terminal one (whose leaves are scored by the game and never reach an
    evaluator at all).
    """

    def __init__(
        self,
        *,
        num_actions: int = 6,
        legal: tuple[int, ...] = (0, 1, 2, 3, 4),
        max_depth: int = 4,
        history: tuple[int, ...] = (),
    ) -> None:
        self.num_actions = num_actions
        self.legal = legal
        self.max_depth = max_depth
        self.history = history

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set, or nothing at all once terminal."""
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        if not self.is_terminal():
            for action in self.legal:
                mask[action] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        if not bool(self.legal_mask()[action]):
            raise AssertionError(
                f"search played illegal action {action} at history {self.history}"
            )
        self.history = (*self.history, action)

    def clone(self) -> ToyState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        return ToyState(
            num_actions=self.num_actions,
            legal=self.legal,
            max_depth=self.max_depth,
            history=self.history,
        )

    def is_terminal(self) -> bool:
        """Return whether `max_depth` plies have been played."""
        return len(self.history) >= self.max_depth

    def terminal_value(self) -> float:
        """Return a value in ``[-1, 1]`` for the player to move here."""
        return ((sum(self.history) % 5) - 2) / 2.0

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the ``(3,)`` vector whose ``w - l`` is :meth:`terminal_value`.

        Present so this toy satisfies the whole
        :class:`~komugi.search.state.SearchState` contract (KOM-173 added
        ``terminal_wdl``); the stubs in this file answer with two-tuples, so the
        search here never allocates its searched-WDL accumulator and never calls
        this.
        """
        value = self.terminal_value()
        return np.array(
            [(1.0 + value) / 2.0, 0.0, (1.0 - value) / 2.0], dtype=np.float64
        )

    def to_play(self) -> int:
        """Return the mover, alternating every ply."""
        return len(self.history) % 2


def _answer(state: ToyState) -> tuple[npt.NDArray[np.float32], float]:
    """Return the one true answer for `state`: a pure function of its history.

    Both stubs below are built on this single function, which is what makes the
    equivalence test a statement about the *scheduler* rather than about two
    evaluators that happen to agree. It answers differently for every position
    -- depth enters the logits as well as the move sum -- so a wave whose
    answers were scattered back onto the wrong nodes could not pass by
    coincidence.
    """
    depth = len(state.history)
    base = sum(state.history) + 1
    logits = np.array(
        [
            0.4 * ((action * 7 + base) % 5) + 0.11 * depth + 0.03 * action
            for action in range(state.num_actions)
        ],
        dtype=np.float32,
    )
    value = math.tanh(0.3 * base) * (1.0 if state.to_play() == 0 else -1.0)
    return logits, value


class SingleStub:
    """The :data:`~komugi.search.Evaluator` half of the pair, plus a call count."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[tuple[int, ...]] = []

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `_answer` for `state`, recording that it was asked."""
        assert isinstance(state, ToyState), "the stub was handed a foreign state"
        assert not state.is_terminal(), "an evaluator must not see a terminal state"
        self.calls += 1
        self.seen.append(state.history)
        return _answer(state)


class WaveStub:
    """The :data:`~komugi.search.state.WaveEvaluator` half, on the same `_answer`.

    Records the size of every wave and the positions it held. The two
    assertions inside are contract checks the search must never trip: an empty
    wave is never submitted (an all-terminal round is skipped silently, not
    sent), and no wave holds the same position twice -- which is the observable
    shadow of the independence claim, since each row of a wave belongs to a
    different root action and therefore a different subtree.
    """

    def __init__(self) -> None:
        self.wave_sizes: list[int] = []
        self.seen: list[tuple[tuple[int, ...], ...]] = []

    def __call__(
        self, states: Sequence[SearchState]
    ) -> list[tuple[npt.NDArray[np.float32], float]]:
        """Return one answer per state, positionally aligned with `states`."""
        assert len(states) > 0, "the search submitted an empty wave"
        histories: list[tuple[int, ...]] = []
        answers: list[tuple[npt.NDArray[np.float32], float]] = []
        for state in states:
            assert isinstance(state, ToyState), "the stub was handed a foreign state"
            assert (
                not state.is_terminal()
            ), "a terminal leaf must be scored by the game, not waved"
            histories.append(state.history)
            answers.append(_answer(state))
        assert len(set(histories)) == len(histories), (
            f"a wave held the same position twice: {histories}"
        )
        self.wave_sizes.append(len(states))
        self.seen.append(tuple(histories))
        return answers


def _config(*, m: int, n: int, seed: int) -> GumbelConfig:
    """Return the config the paired searches share."""
    return GumbelConfig(num_actions=6, m=m, n=n, seed=seed)


def _midgame() -> ToyState:
    """A root one ply in, with room below for several plies of non-terminal tree."""
    return ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=7, history=(2,))


def _near_terminal() -> ToyState:
    """A root whose grandchildren are terminal: waves mix evaluated and scored."""
    return ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=3, history=(1,))


def _all_children_terminal() -> ToyState:
    """A root whose every child is terminal: every wave is empty."""
    return ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=1)


def _deep() -> ToyState:
    """A root no simulation can reach the bottom of: no terminal leaf, ever.

    The wave-size arithmetic is exact only when nothing drops out of a wave, so
    the observation test needs a position where terminal leaves cannot occur.
    """
    return ToyState(num_actions=6, legal=(0, 1, 2, 3, 4), max_depth=64)


POSITIONS: list[tuple[str, Callable[[], ToyState]]] = [
    ("midgame", _midgame),
    ("near-terminal", _near_terminal),
    ("all-children-terminal", _all_children_terminal),
]

# Full-search and fast-search shapes (m clips to the toy's five legal actions),
# plus the small and degenerate budgets where the round arithmetic is least
# forgiving.
BUDGETS: list[tuple[int, int]] = [
    (32, 256),
    (8, 32),
    (5, 64),
    (5, 17),
    (4, 16),
    (2, 3),
    (5, 1),
    (3, 0),
]


@pytest.mark.parametrize("seed", (7, 1234))
@pytest.mark.parametrize(("m", "n"), BUDGETS)
@pytest.mark.parametrize(
    "position",
    [factory for _, factory in POSITIONS],
    ids=[name for name, _ in POSITIONS],
)
def test_wave_scheduling_is_state_identical_to_the_sequential_search(
    position: Callable[[], ToyState], m: int, n: int, seed: int
) -> None:
    """The equivalence golden: same seeds, same stub answers, same result exactly.

    Every field is compared, and ``policy_target`` is compared as *bytes*: the
    independence argument says the backups land in disjoint accumulators in an
    unchanged within-accumulator order, so the float sums are bit-identical and
    a tolerance would be admitting less than the design claims. A scheduler
    that reordered visits *within* a root action, or scattered a wave's answers
    onto the wrong leaves, or let a terminal leaf take a network value, fails
    here rather than drifting quietly in a campaign.
    """
    config = _config(m=m, n=n, seed=seed)
    sequential = GumbelSearch(SingleStub(), config).search(position())
    waved = GumbelSearch(
        SingleStub(), config, wave_evaluator=WaveStub()
    ).search(position())

    assert waved.action == sequential.action
    assert waved.sims_used == sequential.sims_used == n
    assert np.array_equal(waved.visit_counts, sequential.visit_counts)
    assert waved.policy_target.tobytes() == sequential.policy_target.tobytes()
    assert waved.root_value == sequential.root_value
    assert np.array_equal(waved.root_prior_logits, sequential.root_prior_logits)
    assert waved == sequential


def _expected_waves(m: int, n: int) -> list[tuple[int, int]]:
    """Return `(wave size, that round's survivor count)` for every wave of a search.

    Re-derived here from the published budget rule -- rounds ``m``,
    ``ceil(m/2)``, ..., ``1``; ``B // rounds_left`` per round and the whole
    remainder to the last; ``divmod`` across the survivors within a round --
    rather than read back off the implementation, so it is an independent
    statement of what the schedule should be. One wave per visit index means
    ``per_action`` waves at the full survivor count and, when the round's
    allocation does not divide evenly, one short wave carrying the remainder.
    """
    sizes = _halving_sizes(m)
    budget_left = n
    waves: list[tuple[int, int]] = []
    for round_index, survivors in enumerate(sizes):
        rounds_left = len(sizes) - round_index
        allocation = budget_left if rounds_left == 1 else budget_left // rounds_left
        per_action, remainder = divmod(allocation, survivors)
        waves.extend((survivors, survivors) for _ in range(per_action))
        if remainder:
            waves.append((remainder, survivors))
        budget_left -= allocation
    return waves


@pytest.mark.parametrize(("m", "n"), [(4, 16), (5, 32), (5, 17), (8, 64), (3, 7)])
def test_the_waves_are_as_wide_as_the_round_allows_and_cost_no_extra_evaluations(
    m: int, n: int
) -> None:
    """Batching actually happens, bounded by the round, and buys nothing on credit.

    Three claims, on a position deep enough that no leaf is terminal so the
    arithmetic is exact:

    (i) no wave is wider than its round's survivor count -- the observed sizes
        equal the independently derived schedule, whose every entry is checked
        against that round's survivors here;
    (ii) the wave path evaluates exactly what the sequential path evaluates:
        the wave rows plus the one root evaluation (which stays on the
        single-position seam) equal the sequential call count on the nose, so
        no leaf is evaluated twice and none is skipped;
    (iii) some wave is wider than one row. Equivalence alone cannot see a
        scheduler that submits one position at a time -- it would be a correct
        search that saved nothing -- so the win is asserted directly.
    """
    config = _config(m=m, n=n, seed=5)
    m_effective = min(m, len(_deep().legal))

    sequential_stub = SingleStub()
    GumbelSearch(sequential_stub, config).search(_deep())

    root_stub = SingleStub()
    wave_stub = WaveStub()
    GumbelSearch(root_stub, config, wave_evaluator=wave_stub).search(_deep())

    expected = _expected_waves(m_effective, n)
    assert all(size <= survivors for size, survivors in expected)
    assert wave_stub.wave_sizes == [size for size, _ in expected]
    assert max(wave_stub.wave_sizes) <= m_effective

    assert root_stub.calls == 1, "only the root goes through the single evaluator"
    assert root_stub.seen == [()]
    assert sum(wave_stub.wave_sizes) + root_stub.calls == sequential_stub.calls
    assert sum(wave_stub.wave_sizes) == n, "every simulation reached a fresh leaf"
    submitted = [history for wave in wave_stub.seen for history in wave]
    assert len(set(submitted)) == len(submitted), "a position was waved twice"

    if m_effective >= 4:
        assert max(wave_stub.wave_sizes) > 1, "the scheduler degenerated to waves of 1"


def test_an_all_terminal_wave_is_skipped_rather_than_submitted() -> None:
    """Every descent hits a terminal child: nothing to evaluate, and that is fine.

    Normal endgame flow, and the case an implementer reading "a K=0 wave is
    refused loudly" would get exactly backwards (design review D8'): the
    refusal belongs to the transport, which this path never reaches. The search
    must back the game's own values up and return a full-budget result.
    """
    config = _config(m=4, n=16, seed=3)
    root_stub = SingleStub()
    wave_stub = WaveStub()

    result = GumbelSearch(
        root_stub, config, wave_evaluator=wave_stub
    ).search(_all_children_terminal())

    assert wave_stub.wave_sizes == []
    assert root_stub.calls == 1
    assert result.sims_used == 16
    assert int(result.visit_counts.sum()) == 16


def test_the_default_wave_evaluator_is_none_and_keeps_the_sequential_path() -> None:
    """Omitting the seam is not merely equivalent to the old search; it *is* it.

    The additive-branch discipline: with no wave evaluator the search runs the
    untouched ``_simulate`` loop once per simulation, and with one it runs that
    loop zero times. Counted through the method itself, so a wave path that
    fell back to per-position simulation -- passing every equivalence assertion
    while batching nothing -- is visible here.
    """
    original = GumbelSearch._simulate
    calls: list[int] = []

    def counting(self: GumbelSearch, *args: Any) -> None:
        """Record the simulation, then run the real one."""
        calls.append(int(args[-1]))
        original(self, *args)

    config = _config(m=4, n=16, seed=9)
    try:
        setattr(GumbelSearch, "_simulate", counting)
        sequential = GumbelSearch(SingleStub(), config).search(_midgame())
        assert len(calls) == sequential.sims_used == 16

        calls.clear()
        waved = GumbelSearch(
            SingleStub(), config, wave_evaluator=WaveStub()
        ).search(_midgame())
    finally:
        setattr(GumbelSearch, "_simulate", original)

    assert calls == [], "the wave path fell back to sequential simulation"
    assert waved == sequential


def test_the_root_and_the_forced_move_stay_on_the_single_evaluator() -> None:
    """Nothing to batch at a root, and the forced-move path never simulates.

    Pinning the seam boundary: a forced move evaluates its root once, through
    the single-position evaluator, and submits no wave at all -- which is what
    keeps the forced-terminal-root carve-out (KOM-040 D9') a statement about
    :data:`~komugi.search.Evaluator` alone.
    """
    forced = ToyState(num_actions=6, legal=(4,), max_depth=3)
    root_stub = SingleStub()
    wave_stub = WaveStub()

    result = GumbelSearch(
        root_stub, _config(m=8, n=64, seed=1), wave_evaluator=wave_stub
    ).search(forced)

    assert result.action == 4
    assert result.sims_used == 0
    assert root_stub.calls == 1
    assert root_stub.seen == [()]
    assert wave_stub.wave_sizes == []


class MiscountingWaveStub:
    """A wave evaluator that answers with the wrong number of results."""

    def __init__(self, *, drop: int) -> None:
        self.drop = drop

    def __call__(
        self, states: Sequence[SearchState]
    ) -> list[tuple[npt.NDArray[np.float32], float]]:
        """Return `_answer` for all but the last `drop` states."""
        kept = states[: len(states) - self.drop] if self.drop else list(states) * 2
        answers: list[tuple[npt.NDArray[np.float32], float]] = []
        for state in kept:
            assert isinstance(state, ToyState), "the stub was handed a foreign state"
            answers.append(_answer(state))
        return answers


@pytest.mark.parametrize("drop", (1, 0))
def test_a_wave_answered_with_the_wrong_number_of_results_is_an_error(
    drop: int,
) -> None:
    """Misalignment is refused where it happens, not attached to the wrong node.

    A wave is positional and nothing else: if the answers and the submitted
    states are not the same length, the search has no way to say which policy
    belongs to which leaf. Attaching them anyway would train on another
    position's prior with no other symptom, so both directions -- too few
    answers and too many -- raise. ``drop = 0`` is the too-many case.
    """
    search = GumbelSearch(
        SingleStub(),
        _config(m=4, n=16, seed=2),
        wave_evaluator=MiscountingWaveStub(drop=drop),
    )

    with pytest.raises(ValueError, match="wave evaluator returned"):
        search.search(_midgame())


def test_the_wave_stub_satisfies_the_published_seam_type() -> None:
    """The alias is the contract the tests are written against, not a comment.

    Annotating the stub as a :data:`~komugi.search.state.WaveEvaluator` makes
    mypy check the seam's shape -- a sequence of states in, a positionally
    aligned sequence of ``(logits, value)`` out -- at the one place a real
    transport will later have to fit.
    """
    checked: WaveEvaluator = WaveStub()
    result = GumbelSearch(
        SingleStub(), _config(m=3, n=12, seed=4), wave_evaluator=checked
    ).search(_midgame())

    assert result.sims_used == 12
    assert isinstance(result, SearchResult)


def _wdl_answer(
    state: ToyState,
) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
    """Return the one true *three-tuple* answer for `state`: history in, vector out.

    The three-tuple counterpart of :func:`_answer` and built on it, so the two
    stub pairs rank on the same logits. The vector is a pure function of the
    same history -- three coprime residues of the move sum, normalised, then
    reversed for the second mover, which is the frame the KOM-173 seam says an
    evaluator answers in (``(w, d, l)`` for the player to move *here*). Because
    the frame flip is part of the answer, a vector attached to a node one ply
    off, or scattered onto another wave row, is a different vector rather than
    a coincidentally equal one.

    The scalar is exactly ``w - l``, so the search's own conditional identity
    (``root_wdl[0] - root_wdl[2] == root_value`` at ``draw_score == 0``) is a
    statement this stub makes true rather than one it merely does not
    contradict.
    """
    logits, _ = _answer(state)
    depth = len(state.history)
    base = sum(state.history) + 1
    weights = np.array(
        [
            1.0 + (base * 3) % 7,
            1.0 + (base * 5) % 4,
            1.0 + (base * 2 + depth) % 5,
        ],
        dtype=np.float64,
    )
    wdl = np.array(weights / float(weights.sum()), dtype=np.float64)
    if state.to_play() == 1:
        wdl = np.array(wdl[::-1], dtype=np.float64)
    return logits, float(wdl[0] - wdl[2]), wdl


class SingleWdlStub:
    """The :data:`~komugi.search.state.WdlEvaluator` half of the three-tuple pair."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[tuple[int, ...]] = []

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return `_wdl_answer` for `state`, recording that it was asked."""
        assert isinstance(state, ToyState), "the stub was handed a foreign state"
        assert not state.is_terminal(), "an evaluator must not see a terminal state"
        self.calls += 1
        self.seen.append(state.history)
        return _wdl_answer(state)


class WaveWdlStub:
    """The :data:`~komugi.search.state.AnyWaveEvaluator` half, on the same answer.

    :class:`WaveStub` with the vector added and the same two contract checks:
    no empty wave is ever submitted, and no wave holds a position twice.
    """

    def __init__(self) -> None:
        self.wave_sizes: list[int] = []
        self.seen: list[tuple[tuple[int, ...], ...]] = []

    def __call__(
        self, states: Sequence[SearchState]
    ) -> list[tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]]:
        """Return one three-tuple per state, positionally aligned with `states`."""
        assert len(states) > 0, "the search submitted an empty wave"
        histories: list[tuple[int, ...]] = []
        answers: list[
            tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]
        ] = []
        for state in states:
            assert isinstance(state, ToyState), "the stub was handed a foreign state"
            assert (
                not state.is_terminal()
            ), "a terminal leaf must be scored by the game, not waved"
            histories.append(state.history)
            answers.append(_wdl_answer(state))
        assert len(set(histories)) == len(histories), (
            f"a wave held the same position twice: {histories}"
        )
        self.wave_sizes.append(len(states))
        self.seen.append(tuple(histories))
        return answers


def _assert_readouts_are_distributions(result: SearchResult) -> None:
    """Assert all three WDL readouts are finite and sum to ``1`` within ``1e-6``.

    The loud half of the claim: a search that carried no vector at all reports
    ``NaN`` readouts (the two-tuple path), and a search whose accumulator was
    fed a mis-framed or mis-scaled row reports something that is not a
    distribution. Both are caught here without reference to the other search,
    so this survives even if the two paths agreed on a wrong answer.
    """
    readouts: list[tuple[str, npt.NDArray[np.float32]]] = [
        ("root_wdl", result.root_wdl),
        ("root_wdl_pi", result.root_wdl_pi),
        ("played_wdl", result.played_wdl),
    ]
    for name, vector in readouts:
        assert vector.shape == (3,), f"{name} has shape {vector.shape}"
        assert bool(np.isfinite(vector).all()), f"{name} is not finite: {vector!r}"
        assert bool((vector >= 0.0).all()), f"{name} has a negative entry: {vector!r}"
        total = float(vector.sum())
        assert abs(total - 1.0) <= 1e-6, f"{name} sums to {total}, not 1: {vector!r}"


@pytest.mark.parametrize(("m", "n"), [(8, 32), (32, 256)])
def test_wave_scheduling_carries_the_same_vector_as_the_sequential_search(
    m: int, n: int
) -> None:
    """The equivalence golden, run again on the three-tuple seam (KOM-209).

    :func:`test_wave_scheduling_is_state_identical_to_the_sequential_search`
    drives the scheduler with two-tuple stubs, so the wave path's *vector*
    plumbing -- ``_descend``'s ``want_wdl`` on a terminal leaf and
    ``_run_one_wave``'s ``wdl`` argument -- is never executed by it and the
    searched-WDL readouts are never compared across the seam. That matters
    because the two failure modes are not equally loud: a **missing** vector
    raises at the first backup, while a **wrong** one (a terminal leaf built in
    the wrong frame, a wave row attached to the wrong node) is silent and ends
    up in training data as a number nobody can trace back.

    So: the same deterministic answers offered one position at a time and a
    wave at a time, on a root whose grandchildren are terminal -- the wave
    stub sees fewer positions than there are simulations, which is the witness
    that terminal leaves really were reached *inside* a wave and took the
    game's own one-hot through ``want_wdl``. The results must be equal field
    for field (:class:`~komugi.search.SearchResult` equality covers all three
    readouts) and every readout must be an honest distribution.

    Red under each of these mutants, green without them:

    * drop ``want_wdl`` in ``_descend`` -- the terminal leaf carries no vector
      and the first backup of such a simulation raises ``ValueError``;
    * drop the ``wdl`` argument in ``_run_one_wave`` -- every waved leaf is
      vectorless and the first backup raises ``ValueError``;
    * build the terminal leaf in the wrong frame (before the move is played) --
      the one-hot and the mover disagree with the sequential path's, the
      accumulated rows differ, and the equality fails.
    """
    config = _config(m=m, n=n, seed=11)

    sequential_stub: WdlEvaluator = SingleWdlStub()
    sequential = GumbelSearch(sequential_stub, config).search(_near_terminal())

    wave_stub = WaveWdlStub()
    checked: AnyWaveEvaluator = wave_stub
    waved = GumbelSearch(
        SingleWdlStub(), config, wave_evaluator=checked
    ).search(_near_terminal())

    assert sum(wave_stub.wave_sizes) < n, (
        "no simulation ended on a terminal leaf, so the wave path's terminal "
        "want_wdl branch was never taken"
    )

    _assert_readouts_are_distributions(sequential)
    _assert_readouts_are_distributions(waved)

    assert np.array_equal(waved.root_wdl, sequential.root_wdl)
    assert np.array_equal(waved.root_wdl_pi, sequential.root_wdl_pi)
    assert np.array_equal(waved.played_wdl, sequential.played_wdl)
    assert waved.action == sequential.action
    assert waved.sims_used == sequential.sims_used == n
    assert np.array_equal(waved.visit_counts, sequential.visit_counts)
    assert waved.policy_target.tobytes() == sequential.policy_target.tobytes()
    assert waved.root_value == sequential.root_value
    assert waved == sequential
