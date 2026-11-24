"""KOM-220: the bounded tactical survey, on the fixtures that motivated it.

``tests/search/unit/test_kom219_gumbel_coverage.py`` pinned two of the audited
limits as *missing guarantees*: **C1**, a root action the Gumbel draw never
sampled is unplayable at any budget, and **C2**, an interior refutation the
prior buries is never visited. This module is the other half of that pair --
the same two toy fixtures, imported from that file so there is exactly one copy
of them, searched with ``discovery_mode = "root_marshal_v1"`` and a tactical
provider that knows where the win is.

What is claimed here, and what is deliberately not:

* the survey **attaches terminal evidence**; it never claims a win by itself. A
  provider is allowed false positives (``TacticalProvider``'s contract), so
  every certification below is the proof layer's -- which is why an active
  discovery mode requires ``proof_mode = "interval_v1"``;
* the survey is **bounded**, and the bounds are checked as equalities wherever
  the fixture makes them exact: at most ``2K`` extra simulations, at most
  ``1 + K`` provider calls, at most ``K`` survey rows and exactly ``1 + K + n``
  network rows on a fixture where every simulation expands a node;
* the survey's **scope** is the root plus each initial candidate's replies, one
  ply deep, and an action outside it is exactly as unknown as it was before
  KOM-220 (``test_an_unsampled_unsafe_action_outside_the_scope_stays_unknown``
  pins that boundary rather than papering over it);
* a wave-scheduled search and a sequential one return the **same
  SearchResult**, diagnostics included, because the survey runs before halving
  and entirely through the scalar node seam;
* ``discovery_mode = "off"`` is untouched, and a malformed provider is refused
  **before** a single clone/play is spent on it.

Nothing here loads a checkpoint, touches the engine or leaves the CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.search import gumbel
from komugi.search.gumbel import _Node as _GumbelNode
from komugi.search.proofs import UNKNOWN, root_edge_bounds
from komugi.search.tactical import TacticalProvider


def _load_coverage_fixtures() -> ModuleType:
    """Import the KOM-219 coverage module by path, under its own module name.

    ``tests/`` is not a package, so there is no import path to it; and loading
    it under a name of its own keeps pytest's collection of the real file
    untouched -- this module borrows its *fixtures*, not its tests.
    """
    path = Path(__file__).with_name("test_kom219_gumbel_coverage.py")
    spec = importlib.util.spec_from_file_location("_kom219_coverage_fixtures", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_COVERAGE: Final[ModuleType] = _load_coverage_fixtures()

_CoverageState: Any = _COVERAGE._CoverageState
"""The C1/C2 toy position (``Any``: it arrives through a by-path import)."""

_CoverageEvaluator: Any = _COVERAGE._CoverageEvaluator
"""The two-prior network stub, with its call counter and wave log."""

_WIDTH: Final[int] = int(_COVERAGE._WIDTH)
_ROOT_LEGAL: Final[int] = int(_COVERAGE._ROOT_LEGAL)
_ROOT_WIN: Final[int] = int(_COVERAGE._ROOT_WIN)
_INTERIOR_WIN: Final[int] = int(_COVERAGE._INTERIOR_WIN)
_FAST: Final[tuple[int, int]] = (8, 32)
_FULL: Final[tuple[int, int]] = (32, 256)
_BUDGETS: Final[tuple[tuple[int, int], ...]] = (_FAST, _FULL)
_SEEDS: Final[tuple[int, ...]] = tuple(int(seed) for seed in _COVERAGE._SEEDS)
_PRIORS: Final[tuple[str, ...]] = ("uniform", "misleading")

_QUIET_WIDTH: Final[int] = 40
"""Legal actions at every node of the no-capture toy below."""

_UNSAFE: Final[int] = 39
"""The scope toy's losing root action: the one the survey is not asked about."""


# --------------------------------------------------------------------------
# The providers
# --------------------------------------------------------------------------


def _coverage_kind(state: SearchState) -> str:
    """Return the coverage fixture's variant tag (``"root"``/``"interior"``)."""
    return str(cast(Any, state).kind)


class ToyProvider:
    """A provider that knows the two toys statically, and clones nothing.

    The real ``GameMarshalCaptureProvider`` (KOM-222) enumerates captures off
    the board; this one just *says* where the win is, because the claim under
    test is what the search does with the answer. It obeys the same contract:
    ascending unique plain ``int``\\ s drawn only from the mask it was handed,
    no clone, no play, no evaluation, and a ``TypeError`` for a state it does
    not support.
    """

    provider_id = "toy-immediate-win-v1"

    def validate_state(self, state: SearchState) -> None:
        """Refuse anything that is not a KOM-219 coverage fixture."""
        if not isinstance(state, _CoverageState):
            raise TypeError(
                f"{self.provider_id} supports the KOM-219 coverage fixture "
                f"only, got {type(state).__name__}"
            )

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return the toy's one must-look-at action, when the mask offers it."""
        self.validate_state(state)
        if legal_mask.shape != (_WIDTH,) or legal_mask.dtype != np.bool_:
            raise ValueError(
                f"legal_mask must be a ({_WIDTH},) bool array, got "
                f"{legal_mask.shape} of {legal_mask.dtype}"
            )
        if _coverage_kind(state) == "root":
            return (_ROOT_WIN,) if bool(legal_mask[_ROOT_WIN]) else ()
        if bool(legal_mask[_INTERIOR_WIN]):
            return (_INTERIOR_WIN,)
        return ()

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Return the winner of an already terminal toy position.

        Derived from the fixture rather than assumed: ``terminal_value`` is in
        the frame of the mover *at* the terminal state, so ``-1`` there means
        the side **not** to move won, and ``0`` is a real draw.
        ``test_the_toy_provider_reads_the_fixtures_terminal_frame`` pins the
        derivation against the fixture itself.
        """
        self.validate_state(terminal_state)
        if not terminal_state.is_terminal():
            raise ValueError("terminal_winner called on a non-terminal state")
        value = float(terminal_state.terminal_value())
        mover = int(terminal_state.to_play())
        if value == 0.0:
            return None
        if value == 1.0:
            return mover
        if value == -1.0:
            return 1 - mover
        raise ValueError(f"toy terminal outcome is not categorical: {value}")


class EmptyProvider:
    """A provider with nothing to say, for the toys that hold no capture."""

    provider_id = "toy-no-candidates-v1"

    def __init__(self) -> None:
        self.calls = 0

    def validate_state(self, state: SearchState) -> None:
        """Accept only the quiet toy, so a wiring slip still fails loudly."""
        if not isinstance(state, _QuietState):
            raise TypeError(f"unsupported state {type(state).__name__}")

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return no candidate at all, anywhere."""
        self.validate_state(state)
        self.calls += 1
        return ()

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Never reached: nothing this provider names is ever verified."""
        raise AssertionError("a provider with no candidates was asked for a winner")


class CountingProvider:
    """A wrapper that counts every question the search asks the provider."""

    def __init__(self, inner: TacticalProvider) -> None:
        self.inner = inner
        self.provider_id = inner.provider_id
        self.validations = 0
        self.candidate_calls = 0
        self.winner_calls = 0

    @property
    def calls(self) -> int:
        """Return every call of any kind, which is what "never consulted" means."""
        return self.validations + self.candidate_calls + self.winner_calls

    def validate_state(self, state: SearchState) -> None:
        """Count, then delegate."""
        self.validations += 1
        self.inner.validate_state(state)

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Count, then delegate."""
        self.candidate_calls += 1
        return self.inner.candidate_actions(state, legal_mask)

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Count, then delegate."""
        self.winner_calls += 1
        return self.inner.terminal_winner(terminal_state)


_POISONS: Final[tuple[str, ...]] = (
    "bool_id",
    "unsorted",
    "duplicate",
    "illegal",
    "out_of_range",
    "negative",
    "not_a_tuple",
)
"""The malformed answers that must be refused before any transition is spent."""


class PoisonProvider:
    """A provider that breaks exactly one clause of the contract.

    Every variant is a *silent* corruption downstream if it is not caught here:
    a ``bool`` indexes the mask as action ``0`` or ``1``, a duplicate turns "one
    witness per candidate" into two backups of the same line, an out-of-mask id
    attaches a child for a move the search may not play.
    """

    provider_id = "toy-poison-v1"

    def __init__(self, poison: str) -> None:
        self.poison = poison

    def validate_state(self, state: SearchState) -> None:
        """Accept the coverage fixture; the poison is in the answers, not here."""
        if not isinstance(state, _CoverageState):
            raise TypeError(f"unsupported state {type(state).__name__}")

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return this variant's malformed answer (or a valid one for the rest)."""
        self.validate_state(state)
        malformed: dict[str, object] = {
            "bool_id": (True,),
            "unsorted": (_ROOT_WIN, 5),
            "duplicate": (5, 5),
            "illegal": (_ROOT_LEGAL + 1,),
            "out_of_range": (_WIDTH + 3,),
            "negative": (-1,),
            "not_a_tuple": [_ROOT_WIN],
        }
        if self.poison in malformed:
            return cast(tuple[int, ...], malformed[self.poison])
        return (_ROOT_WIN,)

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Answer with the poisoned winner, or with the honest one."""
        if self.poison == "winner_bool":
            return cast(int, True)
        if self.poison == "winner_out_of_range":
            return 2
        return ToyProvider().terminal_winner(terminal_state)


# --------------------------------------------------------------------------
# A second toy: no captures anywhere, and an exact row count
# --------------------------------------------------------------------------


class _QuietState:
    """A toy with no tactics at all -- and one variant that has one, unasked.

    Three kinds:

    * ``"quiet"`` -- never terminal, :data:`_QUIET_WIDTH` legal actions at every
      node. Every simulation therefore expands exactly one new node, which is
      what makes ``1 + K + n`` an *equality* rather than a bound;
    * ``"forced"`` -- one legal action at the root, for the bypass;
    * ``"scope"`` -- as ``"quiet"``, except that root action :data:`_UNSAFE`
      walks into a position where the opponent's reply ``0`` wins on the spot.
      The provider is never told about it, so it is the action that must stay
      unknown: outside the initial candidates and outside the survey's scope.
    """

    NUM_ACTIONS: Final[int] = _WIDTH

    def __init__(self, kind: str = "quiet", history: tuple[int, ...] = ()) -> None:
        self.kind = kind
        self.history = history

    def clone(self) -> _QuietState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        return _QuietState(self.kind, self.history)

    def to_play(self) -> int:
        """Return the mover, alternating every ply."""
        return len(self.history) % 2

    def is_terminal(self) -> bool:
        """Return whether this is the scope toy's one adjudicated line."""
        return self.kind == "scope" and self.history[:2] == (_UNSAFE, 0)

    def terminal_value(self) -> float:
        """Return the outcome in the frame of the mover here: a loss for them."""
        assert self.is_terminal(), "terminal_value on a live position"
        return -1.0

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the same outcome as a one-hot, in the same frame."""
        assert self.is_terminal(), "terminal_wdl on a live position"
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set of this kind, or nothing if terminal."""
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.bool_)
        if self.is_terminal():
            return mask
        width = 1 if self.kind == "forced" else _QUIET_WIDTH
        mask[:width] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        assert bool(self.legal_mask()[action]), (
            f"search played illegal action {action} at history {self.history}"
        )
        self.history = (*self.history, action)


class _QuietEvaluator:
    """A two-tuple network stub for :class:`_QuietState`, with a call counter.

    ``misleading`` buries the scope toy's unsafe root action under ``-30``,
    which is what keeps it out of the Gumbel sample the boundary test needs it
    to be outside of. Terminal inputs are an error: the terminal bypass is a
    contract this module relies on for its row arithmetic.
    """

    def __init__(self, policy: str = "uniform") -> None:
        self.policy = policy
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, value)` for `state`, recording that it was asked."""
        assert isinstance(state, _QuietState), "a foreign state reached the seam"
        assert not state.is_terminal(), "terminal descendant evaluated"
        self.calls += 1
        logits = np.zeros(_WIDTH, dtype=np.float32)
        if self.policy == "misleading" and not state.history:
            logits[_UNSAFE] = -30.0
        # A flat value everywhere: this toy's whole content is its shape, so a
        # value that varied would smuggle in an opinion the tests would read as
        # the search's own.
        return logits, 0.0

    def wave(
        self, states: Sequence[SearchState]
    ) -> Sequence[tuple[npt.NDArray[np.float32], float]]:
        """Answer a whole wave positionally."""
        return [self(state) for state in states]


# --------------------------------------------------------------------------
# Spies: the tree as the survey left it, and the survey's own backups
# --------------------------------------------------------------------------


class _BackupLog:
    """Every backup of a search, as ``(root edge, delta on that edge)``.

    Installed over :func:`komugi.search.gumbel._backup` -- the idiom
    ``test_kom219_gumbel_coverage.py`` uses for the readouts seam -- and
    measuring the accumulator *before and after* each call, so "the witness put
    a genuine ``-1`` on that root edge" is a recorded difference rather than an
    inference from a total.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entries: list[tuple[int, float]] = []
        inner = gumbel._backup

        def spy(path: list[tuple[_GumbelNode, int]], leaf: _GumbelNode) -> None:
            """Record the root edge's change, then back up for real."""
            root, edge = path[0]
            before = float(root.child_value_sums[edge])
            inner(path, leaf)
            self.entries.append(
                (edge, float(root.child_value_sums[edge]) - before)
            )

        monkeypatch.setattr(gumbel, "_backup", spy)


class _SurveySpy(GumbelSearch):
    """A search that records the tree sequential halving was handed.

    The override runs at exactly the seam the survey ends at, so what it copies
    is the survey's whole effect and nothing of the ordinary search:
    ``candidate_count`` is the candidate set *after* any augmentation (``K`` or
    ``K + 1``, which is the only way to observe that the draw was widened), and
    ``mark`` splits :class:`_BackupLog` into the survey's backups and halving's.
    """

    log: _BackupLog | None = None
    root: _GumbelNode | None = None
    mark: int = 0
    candidate_count: int = 0
    post_survey_visits: npt.NDArray[np.int64] | None = None
    post_survey_sums: npt.NDArray[np.float64] | None = None

    def _sequential_halving(
        self,
        root: _GumbelNode,
        root_state: SearchState,
        survivors: list[int],
        gumbel: npt.NDArray[np.float64],
    ) -> int:
        """Snapshot the post-survey tree, then run the real halving."""
        self.root = root
        self.candidate_count = len(survivors)
        self.post_survey_visits = root.child_visits.copy()
        self.post_survey_sums = root.child_value_sums.copy()
        self.mark = 0 if self.log is None else len(self.log.entries)
        return super()._sequential_halving(root, root_state, survivors, gumbel)

    @property
    def node(self) -> _GumbelNode:
        """Return the captured root, refusing to pretend when nothing ran."""
        assert self.root is not None, "no search reached sequential halving"
        return self.root

    @property
    def survey_backups(self) -> list[tuple[int, float]]:
        """Return the backups performed before halving started."""
        assert self.log is not None, "no backup log was installed"
        return self.log.entries[: self.mark]


def _config(
    *,
    m: int,
    n: int,
    seed: int,
    proof: str = "interval_v1",
    discovery: str = "root_marshal_v1",
) -> GumbelConfig:
    """Return a toy-sized config with the two optional layers set explicitly."""
    return GumbelConfig(
        num_actions=_WIDTH,
        m=m,
        n=n,
        seed=seed,
        proof_mode=proof,
        discovery_mode=discovery,
    )


def _discovery_search(
    kind: str,
    *,
    policy: str,
    m: int,
    n: int,
    seed: int,
    wave: bool,
    provider: TacticalProvider | None = None,
) -> tuple[SearchResult, Any]:
    """Search a fresh coverage fixture with the discovery layer active."""
    evaluator = _CoverageEvaluator(policy)
    search = GumbelSearch(
        evaluator,
        _config(m=m, n=n, seed=seed),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=ToyProvider() if provider is None else provider,
    )
    return search.search(_CoverageState(kind)), evaluator


def _diagnostics(result: SearchResult) -> gumbel.SearchDiagnostics:
    """Return `result`'s diagnostics, refusing the ``None`` an active layer cannot give."""
    assert result.diagnostics is not None, "an active layer reported no diagnostics"
    return result.diagnostics


def _matrix() -> list[tuple[str, int, int, int, bool]]:
    """Return the audited sweep: priors x seeds x budgets x schedules."""
    return [
        (policy, m, n, seed, wave)
        for policy in _PRIORS
        for m, n in _BUDGETS
        for seed in _SEEDS
        for wave in (False, True)
    ]


# --------------------------------------------------------------------------
# The fixture's own frame, pinned before anything is claimed about it
# --------------------------------------------------------------------------


def test_the_toy_provider_reads_the_fixtures_terminal_frame() -> None:
    """SANITY: the winner derivation is read off the fixture, not assumed.

    Everything below turns on ``winner == the mover the win belongs to``, and
    that comparison is only meaningful if ``terminal_winner`` inverts the
    fixture's frame correctly. So: play the C1 win and read it back. The mover
    at the terminal state is the *opponent*, its ``terminal_value`` is ``-1``,
    and the winner is therefore the player who moved into it -- the root mover.
    A quiet sibling is a real draw, and so is ``None``.
    """
    provider = ToyProvider()
    root = _CoverageState("root")
    assert root.to_play() == 0

    won = root.clone()
    won.play(_ROOT_WIN)
    assert won.is_terminal() and won.to_play() == 1
    assert float(won.terminal_value()) == -1.0
    assert provider.terminal_winner(won) == 0, "the root mover won that line"

    drawn = root.clone()
    drawn.play(0)
    assert float(drawn.terminal_value()) == 0.0
    assert provider.terminal_winner(drawn) is None

    interior = _CoverageState("interior")
    reply = interior.clone()
    reply.play(0)
    assert provider.candidate_actions(reply, reply.legal_mask()) == (_INTERIOR_WIN,)
    reply.play(_INTERIOR_WIN)
    assert reply.is_terminal()
    assert provider.terminal_winner(reply) == 1, "the opponent won that line"


# --------------------------------------------------------------------------
# T_CTOR -- the mode and the provider are validated against each other
# --------------------------------------------------------------------------


def test_a_provider_and_a_discovery_mode_must_agree_at_construction() -> None:
    """T_CTOR: neither half of the pair may be supplied without the other.

    Both directions are refusals rather than shrugs. A provider under
    ``discovery_mode = "off"`` would never be consulted, yet the run's
    provenance would name a generator that had no effect on it; a discovery mode
    with no provider would be recorded as having run a survey that could not
    exist. And an identity that is blank or not a string names nothing at all,
    which is only ever noticed in a manifest, long after the search.
    """
    evaluator = _CoverageEvaluator("uniform")

    with pytest.raises(ValueError, match="discovery_mode is 'off'"):
        GumbelSearch(
            evaluator,
            _config(m=2, n=4, seed=0, proof="interval_v1", discovery="off"),
            tactical_provider=ToyProvider(),
        )
    with pytest.raises(ValueError, match="needs a tactical provider"):
        GumbelSearch(evaluator, _config(m=2, n=4, seed=0))

    blank = ToyProvider()
    blank.provider_id = ""
    with pytest.raises(ValueError, match="non-empty"):
        GumbelSearch(evaluator, _config(m=2, n=4, seed=0), tactical_provider=blank)

    assert evaluator.calls == 0, "a refused construction still asked the network"


# --------------------------------------------------------------------------
# T_ROOT_WIN -- C1 repaired: the draw is widened by a tactical signal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
def test_a_root_win_the_draw_missed_is_verified_and_played(
    policy: str, m: int, n: int, seed: int, wave: bool
) -> None:
    """C1 REPAIRED: the immediate win is played on every seed, prior, budget and
    schedule -- and it costs exactly one extra simulation and no network row.

    ``test_root_candidate_exclusion_cannot_be_fixed_by_more_simulations`` is the
    same fixture with the layer off: action 63 wins outright, the prior gives it
    ``-30``, and the draw samples it on 0 of 8 seeds at either budget, so
    multiplying the budget by eight changes nothing. With the survey the win is
    *verified* rather than sampled: one clone/play at the root proves it
    terminal, the proof layer reads the attached terminal as an exact ``+1`` and
    every other root action is excluded from the target. The accounting is the
    point of the rest of the assertions -- a repair that found the win by
    spending an unbounded amount of search would not be this repair.
    """
    result, evaluator = _discovery_search(
        "root", policy=policy, m=m, n=n, seed=seed, wave=wave
    )
    diagnostics = _diagnostics(result)
    candidates = min(m, _ROOT_LEGAL)

    assert result.action == _ROOT_WIN
    assert int(result.visit_counts[_ROOT_WIN]) >= 1
    assert diagnostics.discovery_status == "root_witness"
    assert diagnostics.discovery_scope == "initial_root_candidates_depth1"
    assert diagnostics.provider_id == "toy-immediate-win-v1"
    assert diagnostics.terminal_witnesses == 1
    assert diagnostics.tactical_sims_used == 1
    assert diagnostics.survey_nn_rows == 0, "a terminal witness costs no network row"
    assert diagnostics.provider_calls == 1, "the root witness ends the survey"
    assert diagnostics.candidate_verification_transitions >= 1
    assert diagnostics.survey_root_transitions == 0

    # The set was widened by at most one action, and the budget is untouched.
    assert int((result.visit_counts > 0).sum()) <= candidates + 1
    assert diagnostics.ordinary_sims_used == n
    assert result.sims_used == n + 1
    assert int(result.visit_counts.sum()) == result.sims_used
    # Every child of this root is terminal, so the root is the only evaluation
    # any of it can legitimately cost.
    assert evaluator.calls == 1

    # The proof layer's target, not the survey's: a candidate is never a proof,
    # but a verified terminal +1 is, and it takes the whole support.
    assert float(result.policy_target[_ROOT_WIN]) > 0.0
    off_win = np.delete(np.asarray(result.policy_target, dtype=np.float64), _ROOT_WIN)
    assert not bool((off_win > 0.0).any()), "a proven win leaves no other support"


@pytest.mark.parametrize(("m", "n"), _BUDGETS)
@pytest.mark.parametrize("wave", (False, True))
def test_without_discovery_the_same_configuration_still_misses_the_root_win(
    m: int, n: int, wave: bool
) -> None:
    """The control for the test above: the proof layer alone does not repair C1.

    Same fixture, same budgets, same seeds, ``proof_mode = "interval_v1"`` and
    discovery **off**. The unsampled win keeps zero visits -- the KOM-219 C1
    expectation, unchanged -- and the proof layer cannot play it, because the
    intervals it reasons over are built from terminals the search *reached* and
    it reached none of that line. Discovery is what reaches it.
    """
    for seed in _SEEDS:
        evaluator = _CoverageEvaluator("misleading")
        search = GumbelSearch(
            evaluator,
            _config(m=m, n=n, seed=seed, discovery="off"),
            wave_evaluator=evaluator.wave if wave else None,
        )
        result = search.search(_CoverageState("root"))

        assert int(result.visit_counts[_ROOT_WIN]) == 0, (
            f"seed {seed} sampled the winning action; the C1 baseline this "
            f"control rests on was written when no seed did"
        )
        assert result.action != _ROOT_WIN
        assert result.sims_used == n
        diagnostics = _diagnostics(result)
        assert diagnostics.discovery_status == "off"
        assert diagnostics.discovery_scope == "off"
        assert diagnostics.provider_id is None
        assert diagnostics.tactical_sims_used == 0


# --------------------------------------------------------------------------
# T_INTERIOR -- C2 repaired: the buried refutation is in the tree first
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
def test_every_initial_candidate_is_surveyed_and_a_refuted_one_is_excluded(
    policy: str, m: int, n: int, seed: int, wave: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2 REPAIRED: the opponent's winning reply is attached before halving, and
    the move it refutes is dropped from both the decision and the target.

    ``test_misleading_interior_capture_is_unvisited`` is this fixture with the
    layer off: reply 76 wins for the opponent, the prior gives it ``-30``, and
    interior selection never looks at it, so root action 0 keeps a clean score
    and is played. Here every initial candidate is played once and its replies
    verified, so the refutation is a real terminal node under action 0 before
    the first ordinary simulation -- and a genuine ``-1`` on that root edge,
    read off the accumulator before and after the backup rather than inferred
    from a total. The proof layer then excludes action 0 exactly (target
    ``0.0``) and plays the immediate draw.
    """
    log = _BackupLog(monkeypatch)
    evaluator = _CoverageEvaluator(policy)
    search = _SurveySpy(
        evaluator,
        _config(m=m, n=n, seed=seed),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=ToyProvider(),
    )
    search.log = log
    result = search.search(_CoverageState("interior"))
    diagnostics = _diagnostics(result)
    candidates = 2  # the interior root offers exactly two actions

    assert diagnostics.discovery_status == "candidate_survey_complete"
    assert diagnostics.survey_root_transitions == candidates
    root = search.node
    assert sorted(root.children) == [0, 1], "an initial candidate was left unattached"

    refuted = root.children[0]
    assert _INTERIOR_WIN in refuted.children, "the refutation was not attached"
    assert refuted.children[_INTERIOR_WIN].is_terminal
    assert (0, -1.0) in search.survey_backups, (
        "the survey never put a real loss on the refuted root edge"
    )

    assert float(result.policy_target[0]) == 0.0
    assert result.action == 1

    # The bounds, on a fixture where each of them is exact.
    assert diagnostics.tactical_sims_used == 3 <= 2 * candidates
    assert diagnostics.provider_calls == 2 <= 1 + candidates
    assert diagnostics.survey_nn_rows == 1 <= candidates
    assert result.sims_used == n + diagnostics.tactical_sims_used
    assert int(result.visit_counts.sum()) == result.sims_used
    rows = (
        diagnostics.root_nn_rows
        + diagnostics.survey_nn_rows
        + diagnostics.ordinary_nn_rows
    )
    assert rows == evaluator.calls
    assert rows <= 1 + candidates + n


# --------------------------------------------------------------------------
# T_SCALAR_WAVE_EQUAL -- one search, two schedules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ("root", "interior"))
@pytest.mark.parametrize(("m", "n"), _BUDGETS)
@pytest.mark.parametrize("policy", _PRIORS)
def test_the_wave_schedule_returns_the_same_result_as_the_sequential_one(
    kind: str, m: int, n: int, policy: str
) -> None:
    """The survey runs before halving and through the scalar node seam, so
    batching the leaves cannot move a number -- diagnostics included.

    KOM-040's wave equivalence rests on the survivors' subtrees being disjoint
    *within a round*; the survey is not in a round at all, it is a prologue that
    both schedules share verbatim. This is the assertion that would fail if a
    future scheduler tried to fold the verification playouts into a wave: the
    tree would still be right, and the row counts would not.
    """
    for seed in _SEEDS:
        sequential, _ = _discovery_search(
            kind, policy=policy, m=m, n=n, seed=seed, wave=False
        )
        waved, _ = _discovery_search(
            kind, policy=policy, m=m, n=n, seed=seed, wave=True
        )
        assert sequential == waved, f"{kind} seed {seed} diverged between schedules"


# --------------------------------------------------------------------------
# T_COUNT_FIXTURE -- the row budget, as an equality
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "n"), _BUDGETS)
@pytest.mark.parametrize("wave", (False, True))
def test_a_no_capture_position_costs_exactly_one_row_per_candidate(
    m: int, n: int, wave: bool
) -> None:
    """The survey's price, paid in full and no more: ``1 + K + n`` rows.

    On a toy where nothing is ever terminal and the provider has nothing to
    offer, every one of the survey's ``K`` candidates costs exactly one
    evaluation and every one of the ``n`` ordinary simulations expands exactly
    one node -- so the bound ``1 + K + n`` is reached exactly, at 41 rows for the
    fast budget and 289 for the full one. That is the honest worst case of
    discovery: ``K`` extra rows and ``K`` extra backups for a position with no
    tactics in it at all.
    """
    provider = EmptyProvider()
    evaluator = _QuietEvaluator()
    search = GumbelSearch(
        evaluator,
        _config(m=m, n=n, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=provider,
    )
    result = search.search(_QuietState("quiet"))
    diagnostics = _diagnostics(result)
    candidates = min(m, _QUIET_WIDTH)

    assert diagnostics.discovery_status == "candidate_survey_complete"
    assert diagnostics.terminal_witnesses == 0
    assert diagnostics.tactical_sims_used == candidates
    assert diagnostics.survey_nn_rows == candidates
    assert diagnostics.ordinary_nn_rows == n
    assert diagnostics.root_nn_rows == 1
    assert evaluator.calls == 1 + candidates + n
    assert evaluator.calls == (41 if m == 8 else 289)
    # One call at the root, one per candidate -- the ceiling, reached.
    assert diagnostics.provider_calls == 1 + candidates
    assert diagnostics.candidate_verification_transitions == 0
    assert diagnostics.survey_root_transitions == candidates
    assert result.sims_used == n + candidates
    assert int(result.visit_counts.sum()) == result.sims_used


# --------------------------------------------------------------------------
# T_BOUNDARIES -- the edges of the budget and of the root
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wave", (False, True))
def test_a_single_candidate_root_still_verifies_and_widens(wave: bool) -> None:
    """``m = 1``: the draw offers one action and the survey may still add the win."""
    result, _ = _discovery_search(
        "root", policy="misleading", m=1, n=8, seed=0, wave=wave
    )
    diagnostics = _diagnostics(result)

    assert result.action == _ROOT_WIN
    assert diagnostics.discovery_status == "root_witness"
    assert diagnostics.tactical_sims_used == 1 <= 2 * 1
    assert diagnostics.provider_calls == 1 <= 1 + 1
    assert result.sims_used == 8 + 1
    assert int(result.visit_counts.sum()) == result.sims_used


@pytest.mark.parametrize("wave", (False, True))
def test_a_zero_budget_search_bypasses_the_survey_after_validating_the_state(
    wave: bool,
) -> None:
    """``n = 0``: nothing is surveyed, and the provider is still validated.

    A survey exists to shape how a budget is spent; with no budget there is
    nothing to shape, and spending verification playouts would make ``n = 0``
    cost more than ``n = 1``. The state check still runs -- it is wiring, not
    work -- so a mismatched provider fails here exactly as it would anywhere
    else, which is why ``provider_calls`` is the number that reads zero.
    """
    counting = CountingProvider(ToyProvider())
    evaluator = _CoverageEvaluator("uniform")
    search = GumbelSearch(
        evaluator,
        _config(m=8, n=0, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=counting,
    )
    result = search.search(_CoverageState("root"))
    diagnostics = _diagnostics(result)

    assert diagnostics.discovery_status == "bypassed_n0"
    assert diagnostics.discovery_scope == "initial_root_candidates_depth1"
    assert diagnostics.tactical_sims_used == 0
    assert diagnostics.provider_calls == 0
    assert counting.candidate_calls == 0 and counting.winner_calls == 0
    assert counting.validations == 1, "the state was not validated"
    assert result.sims_used == 0
    assert int(result.visit_counts.sum()) == 0


@pytest.mark.parametrize("n", (1, 4, 7))
@pytest.mark.parametrize("wave", (False, True))
def test_small_and_odd_budgets_keep_the_accounting_exact(n: int, wave: bool) -> None:
    """``n = 1``, ``n < m`` and an odd ``n``: the totals still add up.

    Sequential halving spends exactly ``n`` however awkwardly ``n`` divides into
    the rounds, and the survey's backups are visits like any other, so
    ``sims_used`` is ``n + tactical`` and equals ``visit_counts.sum()`` -- the
    invariant a reader of the counts relies on.
    """
    result, _ = _discovery_search(
        "interior", policy="uniform", m=8, n=n, seed=3, wave=wave
    )
    diagnostics = _diagnostics(result)

    assert diagnostics.ordinary_sims_used == n
    assert result.sims_used == n + diagnostics.tactical_sims_used
    assert int(result.visit_counts.sum()) == result.sims_used
    assert diagnostics.tactical_sims_used <= 2 * 2
    assert result.action == 1, "the refuted candidate survived a small budget"


@pytest.mark.parametrize("wave", (False, True))
def test_a_winning_candidate_the_draw_already_sampled_is_not_added_twice(
    wave: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The set is widened by at most one action, and by none when it need not be.

    With ``m`` equal to the legal count the draw samples everything, so the
    witness is already a candidate. Observed directly: the candidate list handed
    to sequential halving is ``K``, not ``K + 1`` -- an augmentation that
    appended unconditionally would show up here as a duplicated candidate and a
    halving round one wider than the draw asked for.
    """
    log = _BackupLog(monkeypatch)
    evaluator = _CoverageEvaluator("misleading")
    search = _SurveySpy(
        evaluator,
        _config(m=_ROOT_LEGAL, n=32, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=ToyProvider(),
    )
    search.log = log
    result = search.search(_CoverageState("root"))

    assert search.candidate_count == _ROOT_LEGAL
    assert int((result.visit_counts > 0).sum()) <= _ROOT_LEGAL
    assert result.action == _ROOT_WIN
    assert _diagnostics(result).tactical_sims_used == 1
    assert result.sims_used == 32 + 1


@pytest.mark.parametrize("wave", (False, True))
def test_a_forced_root_bypasses_the_survey(wave: bool) -> None:
    """One legal action: there is no set to widen and no alternative to refute.

    The provider is still named in the record, because "the survey did not run,
    and here is what would have run it" is a value rather than an absence.
    """
    provider = EmptyProvider()
    evaluator = _QuietEvaluator()
    search = GumbelSearch(
        evaluator,
        _config(m=8, n=32, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=provider,
    )
    result = search.search(_QuietState("forced"))
    diagnostics = _diagnostics(result)

    assert result.action == 0
    assert diagnostics.discovery_status == "bypassed_forced_root"
    assert diagnostics.discovery_scope == "initial_root_candidates_depth1"
    assert diagnostics.provider_id == "toy-no-candidates-v1"
    assert diagnostics.tactical_sims_used == 0
    assert diagnostics.ordinary_sims_used == 0
    assert diagnostics.root_nn_rows == 1
    assert diagnostics.survey_nn_rows == 0
    assert diagnostics.ordinary_nn_rows == 0
    assert diagnostics.provider_calls == 0
    assert provider.calls == 0
    assert result.sims_used == 0
    assert evaluator.calls == 1


# --------------------------------------------------------------------------
# T_REFUSALS -- a provider is a seam, not a trusted caller
# --------------------------------------------------------------------------


def _count_plays(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count every ``play`` on the coverage fixture, across all its clones.

    Patched on the class rather than on an instance, because the search never
    plays on the state it was handed -- it plays on clones it makes itself, and
    a per-instance counter would read zero no matter what the search did.
    """
    plays = [0]
    original = _CoverageState.play

    def counting_play(state: Any, action: int) -> None:
        """Count the transition, then make it."""
        plays[0] += 1
        original(state, action)

    monkeypatch.setattr(_CoverageState, "play", counting_play)
    return plays


@pytest.mark.parametrize("poison", _POISONS)
def test_a_malformed_candidate_list_is_refused_before_any_transition(
    poison: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T_REFUSALS: every contract violation raises before a clone is played.

    The ordering is the claim. Validation after the first ``clone``/``play``
    would already have spent the thing the bounds are stated over, and for the
    ``bool`` variant it would have played action ``1`` -- a legal move --
    believing it had been told to. So the counter reads exactly zero: the root
    evaluation does not play, and nothing else has happened yet.
    """
    plays = _count_plays(monkeypatch)
    search = GumbelSearch(
        _CoverageEvaluator("uniform"),
        _config(m=8, n=32, seed=0),
        tactical_provider=PoisonProvider(poison),
    )

    with pytest.raises(ValueError):
        search.search(_CoverageState("root"))
    assert plays[0] == 0, f"{poison} was refused only after a transition"


@pytest.mark.parametrize("poison", ("winner_bool", "winner_out_of_range"))
def test_a_malformed_terminal_winner_is_refused(poison: str) -> None:
    """``True`` is not player ``1``, and there is no player ``2``.

    ``isinstance(True, int)`` is ``True``, so an unchecked ``bool`` would
    compare equal to the mover index and attach a witness for a line that does
    not win -- a backed-up ``+1`` on a move that loses, with no other symptom.
    """
    search = GumbelSearch(
        _CoverageEvaluator("uniform"),
        _config(m=8, n=32, seed=0),
        tactical_provider=PoisonProvider(poison),
    )
    with pytest.raises(ValueError, match="terminal_winner"):
        search.search(_CoverageState("root"))


def test_an_unsupported_state_is_refused_before_the_evaluator_is_called() -> None:
    """``validate_state`` propagates its ``TypeError`` from the top of the search.

    Before the legal mask, before the root evaluation and before the RNG: a
    provider paired with a state it cannot read is a wiring error, and it must
    fail where it is wired rather than at the first candidate it is asked about.
    """
    evaluator = _QuietEvaluator()
    search = GumbelSearch(
        evaluator, _config(m=8, n=32, seed=0), tactical_provider=ToyProvider()
    )

    with pytest.raises(TypeError, match="coverage fixture"):
        search.search(_QuietState("quiet"))
    assert evaluator.calls == 0, "a refused state still reached the network"


# --------------------------------------------------------------------------
# T_OFF_UNCHANGED -- the default path is the search it always was
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ("root", "interior"))
@pytest.mark.parametrize("wave", (False, True))
def test_discovery_off_is_the_legacy_search_and_consults_nothing(
    kind: str, wave: bool
) -> None:
    """T_OFF_UNCHANGED: with discovery off the tree is the legacy one, whether
    the proof layer is on or not, and no provider can be attached to consult.

    The proof layer reads a finished tree without touching it (KOM-221), so a
    proof-on search must agree with a proof-off one on every number that
    describes the *search* -- visit counts, ``sims_used``, the root prior and
    ``root_value`` -- and may differ only in the decision and the target. A
    discovery layer that had leaked into the off path would show up here as a
    visit count that no longer matched.
    """
    provider = CountingProvider(ToyProvider())
    for seed in _SEEDS:
        legacy_evaluator = _CoverageEvaluator("misleading")
        legacy = GumbelSearch(
            legacy_evaluator,
            GumbelConfig(num_actions=_WIDTH, m=8, n=32, seed=seed),
            wave_evaluator=legacy_evaluator.wave if wave else None,
        ).search(_CoverageState(kind))

        proof_evaluator = _CoverageEvaluator("misleading")
        proved = GumbelSearch(
            proof_evaluator,
            _config(m=8, n=32, seed=seed, discovery="off"),
            wave_evaluator=proof_evaluator.wave if wave else None,
        ).search(_CoverageState(kind))

        assert legacy.diagnostics is None
        assert np.array_equal(legacy.visit_counts, proved.visit_counts)
        assert np.array_equal(legacy.root_prior_logits, proved.root_prior_logits)
        assert legacy.root_value == proved.root_value
        assert legacy.sims_used == proved.sims_used == 32
        assert legacy_evaluator.calls == proof_evaluator.calls

        diagnostics = _diagnostics(proved)
        assert diagnostics.discovery_status == "off"
        assert diagnostics.discovery_scope == "off"
        assert diagnostics.provider_id is None
        assert diagnostics.provider_calls == 0
        assert diagnostics.tactical_sims_used == 0
        assert diagnostics.survey_nn_rows == 0
        assert diagnostics.survey_root_transitions == 0
        assert diagnostics.candidate_verification_transitions == 0
        assert diagnostics.terminal_witnesses == 0

    # The only way a provider could be consulted is to be attached, and the
    # off mode refuses to take one at all.
    with pytest.raises(ValueError):
        GumbelSearch(
            _CoverageEvaluator("uniform"),
            _config(m=8, n=32, seed=0, discovery="off"),
            tactical_provider=provider,
        )
    assert provider.calls == 0


# --------------------------------------------------------------------------
# T_UNSAMPLED_UNSAFE_STAYS_UNKNOWN -- the scope boundary, pinned
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wave", (False, True))
def test_an_unsampled_unsafe_action_outside_the_scope_stays_unknown(
    wave: bool,
) -> None:
    """The survey is a **scope**, not a guarantee: what it never looks at stays
    exactly as unknown as it was.

    Root action 39 loses to the opponent's reply ``0``, the prior buries it at
    ``-30`` so the draw never samples it, and the provider is not the kind that
    names it. It therefore ends the search with zero visits and the interval
    ``(-1, 1)`` -- no evidence either way -- which is the honest answer and the
    one this module refuses to overclaim. Widening the scope is a change of
    algorithm (a new ``discovery_mode`` string), not a tightening of this one,
    and this test is what would fail if one were made silently.
    """
    evaluator = _QuietEvaluator("misleading")
    search = _SurveySpy(
        evaluator,
        _config(m=8, n=32, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=EmptyProvider(),
    )
    result = search.search(_QuietState("scope"))

    assert int(result.visit_counts[_UNSAFE]) == 0, (
        "the unsafe action was sampled after all; this boundary test needs it "
        "to lie outside the initial candidates"
    )
    assert result.action != _UNSAFE
    bounds = root_edge_bounds(search.node)
    assert bounds[_UNSAFE] == UNKNOWN
    assert _UNSAFE not in search.node.children, "an unsurveyed action was attached"
    # And the actions the survey did cover are attached, so this is a statement
    # about the scope rather than about a survey that did nothing.
    assert len(search.node.children) >= min(8, _QUIET_WIDTH)
