"""KOM-221: the proof-aware decision layer, pinned against the KOM-219 baseline.

``GumbelConfig.proof_mode = "interval_v1"`` turns on a layer that runs *after*
an ordinary search, reads the finished tree and nothing else, and may overrule
the survivor sequential halving irreversibly settled on. The claims this module
makes about it, in the order they are tested:

* the two mode fields are **validated as a pair**, at construction, before an
  evaluator is touched;
* ``off`` is the legacy search **verbatim** -- same action, same visits, same
  target, and ``diagnostics is None``, which is also what a
  :class:`SearchResult` built from the legacy keyword set carries;
* on the KOM-219 **late-discovery** fixture (C3) the layer does the thing the
  baseline records as missing: the refuted survivor is replaced by the safe
  alternative, the target's mass leaves the refuted edge exactly, and every
  number that describes the *search* rather than the *decision* -- visit counts,
  simulations, root value, root logits -- is unchanged;
* the layer is **noise-independent in its target** and Gumbel-ranked in its
  choice, which the pure-helper tests drive directly on a hand-built tree;
* the two limits it does **not** address (C1 root candidate exclusion, C2
  interior selection) behave exactly as before, so a later repair's effect
  cannot be confused with this one's.

Everything here is CPU-only and runs on the KOM-219 toy fixtures, which are
imported by path: ``tests/`` is not a package, so the coverage module is loaded
through :mod:`importlib` rather than through an import statement.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.search import gumbel
from komugi.search.gumbel import SearchDiagnostics
from komugi.search.modes import search_algorithm_identity


def _load_coverage_module() -> Any:
    """Return the KOM-219 coverage module, loaded from its path.

    ``tests/`` is deliberately not a package, so its modules cannot be imported
    by name. The fixtures there -- the toy states, the stub evaluator and its
    wave seam -- are the exact baseline this file measures the repair against,
    and re-typing them here would let the two drift apart silently.
    """
    path = Path(__file__).with_name("test_kom219_gumbel_coverage.py")
    spec = importlib.util.spec_from_file_location("kom219_coverage", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_coverage: Final[Any] = _load_coverage_module()

_CoverageState: Final[Any] = _coverage._CoverageState
_CoverageEvaluator: Final[Any] = _coverage._CoverageEvaluator
_WIDTH: Final[int] = _coverage._WIDTH
_FAST: Final[tuple[int, int]] = _coverage._FAST
_FULL: Final[tuple[int, int]] = _coverage._FULL
_INTERIOR_WIN: Final[int] = _coverage._INTERIOR_WIN

_REFUTED: Final[int] = 0
"""The interior fixture's losing action: the survivor the halving settles on."""

_SAFE: Final[int] = 1
"""The interior fixture's immediate draw: the move the proof layer rescues."""


def _search(
    kind: str,
    *,
    policy: str,
    m: int,
    n: int,
    seed: int,
    wave: bool = False,
    proof_mode: str = "off",
    history: tuple[int, ...] = (),
) -> tuple[SearchResult, Any]:
    """Search a fresh KOM-219 fixture, returning the result and the stub."""
    evaluator = _CoverageEvaluator(policy)
    search = GumbelSearch(
        evaluator,
        GumbelConfig(
            num_actions=_WIDTH, m=m, n=n, seed=seed, proof_mode=proof_mode
        ),
        wave_evaluator=evaluator.wave if wave else None,
    )
    return search.search(_CoverageState(kind, history)), evaluator


def _legacy_result(action: int = 0, sims_used: int = 4) -> SearchResult:
    """Return a :class:`SearchResult` built from the pre-KOM-221 keyword set."""
    target = np.zeros(4, dtype=np.float32)
    target[action] = 1.0
    wdl = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return SearchResult(
        action=action,
        policy_target=target,
        root_prior_logits=np.zeros(4, dtype=np.float32),
        root_value=0.0,
        visit_counts=np.zeros(4, dtype=np.int64),
        sims_used=sims_used,
        root_wdl=wdl,
        root_wdl_pi=wdl.copy(),
        played_wdl=wdl.copy(),
    )


# --------------------------------------------------------------------------
# T_MODE -- the mode pair is validated before anything runs
# --------------------------------------------------------------------------


def test_the_mode_pair_is_refused_before_an_evaluator_is_touched() -> None:
    """An unrunnable mode pair raises where it is written, not where it bites.

    Three separate refusals, and the third is the one that matters: a discovery
    mode this search cannot honour -- it has no parameter to receive a tactical
    provider through yet -- must not reach the evaluator at all, because a mode
    that silently did nothing would still be recorded in the provenance identity
    as though it had run. The messages are deliberately not matched: the claim
    is that construction fails, not how it words it.
    """
    with pytest.raises(ValueError):
        GumbelConfig(num_actions=8, m=2, n=4, seed=0, proof_mode="bogus")
    with pytest.raises(ValueError):
        GumbelConfig(
            num_actions=8,
            m=2,
            n=4,
            seed=0,
            proof_mode="off",
            discovery_mode="root_marshal_v1",
        )

    evaluator = _CoverageEvaluator("uniform")
    with pytest.raises(ValueError):
        GumbelSearch(
            evaluator,
            GumbelConfig(
                num_actions=_WIDTH,
                m=2,
                n=4,
                seed=0,
                proof_mode="interval_v1",
                discovery_mode="root_marshal_v1",
            ),
        )
    assert evaluator.calls == 0, "a refused config still asked the network"


# --------------------------------------------------------------------------
# T_COMPAT -- the new field is additive
# --------------------------------------------------------------------------


def test_the_diagnostics_field_is_additive_and_is_part_of_equality() -> None:
    """A legacy-built result carries ``None``, and diagnostics do not compare equal.

    Both halves matter. A caller that never heard of KOM-221 must be able to
    build and compare results exactly as before, so the field defaults to
    ``None``; and two results that differ *only* in what the optional layers did
    are not the same result, so provenance cannot be dropped by an equality
    check that silently ignores it.
    """
    legacy = _legacy_result()
    assert legacy.diagnostics is None
    assert legacy == _legacy_result()

    identity = search_algorithm_identity("interval_v1", "off")
    with_diagnostics = dataclasses.replace(
        _legacy_result(),
        diagnostics=SearchDiagnostics(identity=identity, ordinary_sims_used=4),
    )
    other = dataclasses.replace(
        _legacy_result(),
        diagnostics=SearchDiagnostics(identity=identity, ordinary_sims_used=5),
    )
    assert with_diagnostics != legacy
    assert with_diagnostics != other


def test_diagnostics_are_frozen_and_hashable() -> None:
    """Provenance is a record, not a scratch buffer: it cannot be edited."""
    identity = search_algorithm_identity("interval_v1", "off")
    diagnostics = SearchDiagnostics(identity=identity, ordinary_sims_used=7)

    with pytest.raises(dataclasses.FrozenInstanceError):
        diagnostics.ordinary_sims_used = 8  # type: ignore[misc]
    assert hash(diagnostics) == hash(
        SearchDiagnostics(identity=identity, ordinary_sims_used=7)
    )


def test_an_off_mode_search_reports_no_diagnostics() -> None:
    """``off`` is not "the layer ran and changed nothing": the layer did not run."""
    m, n = _FAST
    result, _ = _search("interior", policy="uniform", m=m, n=n, seed=0)

    assert result.diagnostics is None


# --------------------------------------------------------------------------
# T_OFF_IDENTICAL / T_TOY_ACTIVE -- the C3 late-discovery fixture
# --------------------------------------------------------------------------


def test_off_mode_reproduces_the_late_discovery_baseline() -> None:
    """With the layer off, C3 still behaves exactly as KOM-219 recorded it.

    The refuted action 0 is played, the visits split 192/64, and the target puts
    its whole mass on the eliminated action 1 -- the label and the move
    disagreeing, which is the defect. This is the control for the test below:
    it fails if the repair leaked into the default path.
    """
    m, n = _FULL
    result, _ = _search("interior", policy="uniform", m=m, n=n, seed=0)

    assert result.action == _REFUTED
    assert [int(result.visit_counts[a]) for a in (_REFUTED, _SAFE)] == [192, 64]
    assert float(result.policy_target[_SAFE]) == 1.0
    assert result.sims_used == n
    assert result.diagnostics is None


def test_the_proof_layer_rescues_the_safe_move_on_the_late_discovery_fixture() -> None:
    """MISSING GUARANTEE REPAIRED (C3): the refuted survivor is not played.

    The opponent's winning reply *is* reached, late, so action 0's interval is
    exactly ``[-1, -1]`` -- a proof, not an estimate -- and the layer drops it
    from the admissible set. The survivor is therefore inadmissible and an
    eliminated initial candidate is rescued in its place, which is exactly the
    move the label already preferred: the disagreement between the played move
    and the recorded target is closed from the *decision* side.

    Everything that describes the search rather than the decision is asserted
    against the off-mode run and must be bit-identical: the layer spends no
    simulation, asks the network nothing, redraws no noise and mutates no node,
    so a differing visit count here would mean it had reached into the tree.
    """
    m, n = _FULL
    baseline, _ = _search("interior", policy="uniform", m=m, n=n, seed=0)
    result, evaluator = _search(
        "interior", policy="uniform", m=m, n=n, seed=0, proof_mode="interval_v1"
    )

    assert np.array_equal(result.visit_counts, baseline.visit_counts)
    assert result.sims_used == n
    assert result.root_value == baseline.root_value
    assert np.array_equal(result.root_prior_logits, baseline.root_prior_logits)

    assert result.action == _SAFE
    assert float(result.policy_target[_REFUTED]) == 0.0
    assert float(result.policy_target.sum()) == pytest.approx(1.0, abs=1e-6)

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.identity.decision_id == "gumbel_sh_interval_v1"
    assert diagnostics.identity.target_id == "completed_q_proof_support_v1"
    assert diagnostics.legacy_action == _REFUTED
    assert diagnostics.rescue_scope == "original_candidates"
    assert diagnostics.proof_excluded_actions == 1
    assert diagnostics.ordinary_sims_used == n
    assert diagnostics.root_nn_rows == 1
    assert diagnostics.ordinary_nn_rows == evaluator.calls - 1


def test_the_wave_schedule_produces_the_same_proof_decision() -> None:
    """Batching the evaluator changes when the network is asked, nothing else.

    The whole :class:`SearchResult` is compared, diagnostics included, so the
    row counts and the rescue scope are covered too: the wave path reaches the
    same terminals in the same order, therefore proves the same intervals.
    """
    m, n = _FULL
    sequential, _ = _search(
        "interior", policy="uniform", m=m, n=n, seed=0, proof_mode="interval_v1"
    )
    waved, evaluator = _search(
        "interior",
        policy="uniform",
        m=m,
        n=n,
        seed=0,
        wave=True,
        proof_mode="interval_v1",
    )

    assert evaluator.wave_sizes, "the wave seam was never used"
    assert waved == sequential


# --------------------------------------------------------------------------
# T_RESCUE_PURE / T_ALL_LOST -- the decision helper, driven directly
# --------------------------------------------------------------------------


_PURE_WIDTH: Final[int] = 8
"""The hand-built root's action space, wide enough for three offered edges."""


def _pure_config() -> GumbelConfig:
    """Return the config the hand-built decisions below are scored under."""
    return GumbelConfig(
        num_actions=_PURE_WIDTH, m=2, n=4, seed=0, proof_mode="interval_v1"
    )


def _terminal_child(value: float) -> gumbel._Node:
    """Return a terminal opponent-to-move node worth `value` in its own frame."""
    return gumbel._Node(
        to_play=1,
        is_terminal=True,
        value=value,
        legal=np.zeros(_PURE_WIDTH, dtype=np.bool_),
        logits=np.zeros(_PURE_WIDTH, dtype=np.float32),
    )


def _pure_root(refuted_edges: tuple[int, ...]) -> gumbel._Node:
    """Return a root with three offered edges, `refuted_edges` proven lost.

    Every edge carries the same visit count and the same backed-up sum, so the
    completed Q of the unproven edges is identical and the ranking between them
    is decided by the Gumbel draw alone -- which is what makes the draw's effect
    on the decision, and its absence from the target, separable at all.
    """
    legal = np.zeros(_PURE_WIDTH, dtype=np.bool_)
    legal[:3] = True
    root = gumbel._Node(
        to_play=0,
        is_terminal=False,
        value=0.0,
        legal=legal,
        logits=np.zeros(_PURE_WIDTH, dtype=np.float32),
    )
    root.child_visits[:3] = 4
    root.child_value_sums[:3] = 0.0
    for action in refuted_edges:
        # A terminal win for the mover *there* -- the opponent -- so the edge is
        # exactly -1 once mapped back into the root's frame.
        root.children[action] = _terminal_child(1.0)
        root.child_value_sums[action] = -4.0
    return root


def _gumbel_vector(preferred: int) -> npt.NDArray[np.float64]:
    """Return a draw that ranks `preferred` above every other action."""
    vector = np.zeros(_PURE_WIDTH, dtype=np.float64)
    vector[preferred] = 1.0
    return vector


def test_the_rescue_ranks_on_the_draw_while_the_target_ignores_it() -> None:
    """The decision follows the noise; the recorded label does not.

    Two draws that disagree about which admissible alternative is best give two
    different moves, as they must -- the layer rescues by the ordinary root
    score and never by a policy argmax -- and the *same* target, bit for bit,
    because the target is a softmax over the admissible support with no Gumbel
    term in it at all. A target that moved with the draw would make the label a
    sample rather than a statistic.
    """
    config = _pure_config()
    root = _pure_root((0,))

    first, first_target, first_scope, first_excluded = gumbel._proof_decision(
        root, [0, 1, 2], 0, _gumbel_vector(1), config
    )
    second, second_target, second_scope, second_excluded = gumbel._proof_decision(
        root, [0, 1, 2], 0, _gumbel_vector(2), config
    )

    assert (first, second) == (1, 2)
    assert np.array_equal(first_target, second_target)
    assert float(first_target[0]) == 0.0
    assert float(first_target.sum()) == pytest.approx(1.0, abs=1e-12)
    assert first_scope == second_scope == "original_candidates"
    assert first_excluded == second_excluded == 1


def test_a_rescue_with_no_admissible_candidate_ranges_over_the_whole_root() -> None:
    """When nothing the draw offered survives, the choice widens -- and ties low.

    The initial candidate set is the refuted edge and nothing else, so the pool
    the layer would rescue from is empty and it falls back to every admissible
    root edge. With a flat draw the two alternatives score exactly equal, and
    the tie falls to the lowest action id rather than to whatever ``argmax``
    happened to see first.
    """
    config = _pure_config()
    root = _pure_root((0,))

    action, target, scope, excluded = gumbel._proof_decision(
        root, [0], 0, np.zeros(_PURE_WIDTH, dtype=np.float64), config
    )

    assert action == 1
    assert scope == "full_root"
    assert excluded == 1
    assert float(target[0]) == 0.0
    assert float(target[1]) == pytest.approx(float(target[2]), abs=1e-12)


def test_a_position_with_every_edge_lost_still_returns_the_survivor() -> None:
    """Refusing to choose is not a move: an all-lost root keeps the legacy answer.

    Dropping the lost edges would leave an empty admissible set and no target
    that sums to one. So when *every* offered edge is proven lost they are all
    kept, the survivor is admissible by construction, and the layer records that
    it rescued nothing.
    """
    config = _pure_config()
    root = _pure_root((0, 1, 2))

    action, target, scope, excluded = gumbel._proof_decision(
        root, [0, 1], 0, _gumbel_vector(2), config
    )

    assert action == 0
    assert scope == "none"
    assert excluded == 0
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-12)
    assert all(float(target[a]) > 0.0 for a in (0, 1, 2))


# --------------------------------------------------------------------------
# T_NEGATIVE_UNCHANGED -- the limits this repair does not address
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "policy", "budget"),
    (
        ("root", "uniform", _FAST),
        ("interior", "misleading", _FULL),
    ),
    ids=("c1-root-candidate-exclusion", "c2-interior-selection"),
)
def test_the_untouched_limits_are_untouched(
    kind: str, policy: str, budget: tuple[int, int]
) -> None:
    """C1 and C2 are KOM-220's and KOM-222's problems, and still behave as such.

    C1's winning action is never *sampled*, so no terminal under it is ever
    reached and no interval can refute anything the draw did offer; C2's
    refutation is never *reached* within the budget, so there is nothing to
    prove. In both the survivor stays admissible and the decision is the legacy
    one -- which is the point: a later repair's effect on these fixtures must be
    attributable to that repair and not to this one.
    """
    m, n = budget
    baseline, _ = _search(kind, policy=policy, m=m, n=n, seed=0)
    result, _ = _search(
        kind, policy=policy, m=m, n=n, seed=0, proof_mode="interval_v1"
    )

    assert result.action == baseline.action
    assert np.array_equal(result.visit_counts, baseline.visit_counts)
    assert result.diagnostics is not None
    assert result.diagnostics.rescue_scope == "none"


# --------------------------------------------------------------------------
# T_FORCED -- the no-search path still reports provenance
# --------------------------------------------------------------------------


def test_the_forced_move_path_reports_its_row_counts() -> None:
    """One legal move: nothing to decide, but the row counts still have to add up.

    The forced path runs no simulation and evaluates exactly one position -- the
    root -- so a downstream consumer totalling network rows per game must see
    ``1 + 0`` here rather than a missing record it has to guess the shape of.
    """
    state: SearchState = _CoverageState("interior", (0, 5))
    assert int(state.legal_mask().sum()) == 1

    evaluator = _CoverageEvaluator("uniform")
    search = GumbelSearch(
        evaluator,
        GumbelConfig(
            num_actions=_WIDTH, m=2, n=4, seed=0, proof_mode="interval_v1"
        ),
    )
    result = search.search(state)

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.ordinary_sims_used == 0
    assert diagnostics.root_nn_rows == 1
    assert diagnostics.ordinary_nn_rows == 0
    assert result.sims_used == 0
    assert evaluator.calls == 1
