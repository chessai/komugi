"""Value-flat characterisation of the draw-utility parity bias (KOM-157 / M2').

What this pins, and why it is a *characterisation* rather than a bug report
-------------------------------------------------------------------------
Self-play collapses the network's ``(w, d, l)`` head to a scalar with
:func:`~komugi.selfplay.inference.wdl_utility` -- ``u = w - l + s*d``, where
``s`` is ``SelfPlayConfig.draw_score`` -- and hands that one number to the
search, in the perspective of the side to move at that node (design D1). The
search's backup negates once per change of mover
(``komugi/search/gumbel.py:_backup``), so a *constant* mover-relative draw term
``s*d`` sitting at every leaf does not cancel: a leaf at depth ``k`` reaches the
root as ``(-1)^k * s * d``. Which depth-class a root action's evidence comes
from is decided by sequential halving, not by the position, so the sign a root
action carries is an artefact of the visit schedule.

``docs/reviews/2026-09-11-kom-151-design-review.md`` F1/F2 respecified the
KOM-151 M2 measurement around exactly that. The tests below are its concrete
form, on a value-flat toy (``w == l`` everywhere, so ``u`` *is* the draw term
and nothing else):

* ``s = 0.0`` removes the term **identically** -- not approximately. Every root
  Q is 0, ``root_value`` is exactly ``0.0``, and the policy target is the masked
  softmax of the root logits: the search that self-play would have run is the
  prior, undisturbed.
* ``s = +-0.2`` gives every one-visit root candidate a root-frame
  ``Q = -s*d`` exactly, of the *opposite* sign to the contempt that was
  configured, amplified into the recorded ``policy_target`` by sigma's
  ``(c_visit + max_visits)`` scale.

None of this is a defect in the code under test; it is the documented P0a
approximation, measured. This file is the red test for any future
WDL-carrying backup (KOM-151 O3 / design review F9): that change must keep
part (i) passing and must turn part (ii)'s ``-s*d`` into ``0``.

The schedule arithmetic
-----------------------
Fast self-play is ``(m=8, n=32)`` (``komugi/config.py``: ``gumbel_m_fast``,
``gumbel_n_fast``). ``_halving_sizes(8) = [8, 4, 2, 1]`` -- four rounds -- and
the budget split is ``budget_left // rounds_left`` with the last round taking
the remainder outright, so every round here gets exactly ``8`` simulations:

    round 0:  32 // 4 = 8 over 8 survivors -> 1 visit each   (cut to 4)
    round 1:  24 // 3 = 8 over 4 survivors -> 2 more each    (cut to 2)
    round 2:  16 // 2 = 8 over 2 survivors -> 4 more each    (cut to 1)
    round 3:  8 (remainder) on the finalist -> 8 more

giving root visit counts ``1, 1, 1, 1, 3, 3, 7, 15`` -- the four round-0 cuts
stop at **one** visit, which is a depth-1 leaf and nothing else, hence
``Q = -s*d`` exactly.

Full search is ``(m=32, n=256)``: ``_halving_sizes(32)`` is six rounds and
round 0 allocates ``256 // 6 = 42`` over 32 survivors, i.e. ``divmod(42, 32) =
(1, 10)`` -- ten candidates get 2 visits and twenty-two get 1. The round-0 cut
keeps 16, all of which pick up at least two more visits in round 1. So the
"exactly one visit" set is still well defined (it is the round-0 cuts that only
ever got one), it is just no longer all of them: at least ``16 - 10 = 6``
survive the whole search with a single visit. The tests key on
``visit_count == 1`` rather than on a hard-coded action set for that reason.

How a root Q is recovered
-------------------------
:class:`~komugi.search.gumbel.SearchResult` does **not** expose per-action Q --
it carries ``policy_target`` (float32), ``root_prior_logits``, ``root_value``
(``v_mix``) and ``visit_counts``, and nothing else -- so there is no more direct
route than inverting the target. Since ``policy_target`` is proportional to
``exp(logit + sigma(completedQ))`` over the legal set, for any two legal
actions ``a`` and ``b``::

    log pt[a] - log pt[b] = (logit[a] - logit[b]) + scale * (Q[a] - Q[b])
    scale = (c_visit + max visit_counts) * c_scale

Choosing ``b`` **unvisited** pins ``Q[b] = v_mix = root_value`` (``_completed_q``
completes every never-visited action with the node's own mixed value), which
leaves ``Q[a]`` as the only unknown. The float32 target costs about 1e-7 in the
log-ratio, divided by a scale of 65 (fast) or ~135 (full), so the recovered Q is
good to ~1e-9 -- comfortably inside the 1e-5 tolerance used below, and the
reason design review F2 rejected M2's original 1e-12.

The toy
-------
``tests/search/unit/test_gumbel_unit.py``'s ``ToyState`` is deliberately *not*
reused: it terminates at ``max_depth`` plies, and a terminal leaf takes its
value from the game rather than from the stub, which would mix a second value
source into the very arithmetic under measurement. The toy here is never
terminal and offers 40 legal actions at every node (more than ``m = 32``, so a
non-candidate reference action always exists), with movers alternating every
ply.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.observation import MAX_TURNS
from komugi.search import (
    GumbelConfig,
    GumbelSearch,
    SearchResult,
    SearchState,
)
from komugi.search.gumbel import _Node, _select_interior
from komugi.selfplay.adapter import MlhUtility, SearchContext, _value_from_wdl
from komugi.selfplay.inference import wdl_utility

_NUM_ACTIONS: Final = 48
"""Action space of the toy -- larger than the legal set, so masking is exercised."""

_NUM_LEGAL: Final = 40
"""Legal actions at every node: ``> m`` for both production configs."""

_WDL: Final = (0.25, 0.5, 0.25)
"""The stub's fixed head output. ``w == l``, so ``u`` is the draw term alone."""

_DRAW_MASS: Final = _WDL[1]
"""``d`` in ``u = w - l + s*d``."""

_FAST: Final = (8, 32)
"""``(gumbel_m_fast, gumbel_n_fast)`` -- 75% of self-play moves."""

_FULL: Final = (32, 256)
"""``(gumbel_m_full, gumbel_n_full)``."""

_SEED: Final = 20260911
"""Fixed Gumbel seed: the candidate set and the cuts must not wander run to run."""

_Q_TOL: Final = 1e-5
"""Tolerance on a recovered root Q (design review F2: 1e-12 is unreachable)."""


def _logits_for(history: tuple[int, ...]) -> npt.NDArray[np.float32]:
    """Return the stub's logits for the position reached by `history`.

    A pure function of the move history -- the search's determinism guarantee is
    inherited from the evaluator -- seeded by a fold over the history so that
    distinct positions get distinct, unequal logits. The empty history is the
    root, whose logits are therefore a fixed seeded draw with no ties.
    """
    seed = 0x9E3779B9
    for action in history:
        seed = (seed * 1000003 + action + 1) % (2**32)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(scale=1.5, size=_NUM_ACTIONS), dtype=np.float32)


def _legal_mask_array() -> npt.NDArray[np.bool_]:
    """Return the toy's fixed legal mask."""
    mask = np.zeros(_NUM_ACTIONS, dtype=np.bool_)
    mask[:_NUM_LEGAL] = True
    return mask


class FlatToyState:
    """A never-terminal toy position: 40 legal actions, movers alternating.

    Never terminal on purpose. A terminal leaf would be scored by
    :meth:`terminal_value` instead of by the stub evaluator, putting a second
    value source inside the quantity these tests measure; :meth:`terminal_value`
    therefore refuses to answer, and the search never asks.
    """

    def __init__(self) -> None:
        self.history: tuple[int, ...] = ()

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set, identical at every depth."""
        return _legal_mask_array()

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        if not bool(self.legal_mask()[action]):
            raise AssertionError(
                f"search played illegal action {action} at history {self.history}"
            )
        self.history = (*self.history, action)

    def clone(self) -> FlatToyState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        copy = FlatToyState()
        copy.history = self.history
        return copy

    def is_terminal(self) -> bool:
        """Return ``False``: this toy never ends."""
        return False

    def terminal_value(self) -> float:
        """Refuse -- reaching here would mean a value the stub did not supply."""
        raise AssertionError(
            "terminal_value was called on a toy that is never terminal: a leaf "
            "value entered the backup from somewhere other than the stub head"
        )

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Refuse, for the same reason :meth:`terminal_value` does.

        The searched-WDL accumulator reads this at terminal leaves only, and this
        toy has none: a call here would mean a *vector* entered the backup from
        somewhere other than the stub head, which is exactly the second value
        source these measurements exist to exclude.
        """
        raise AssertionError(
            "terminal_wdl was called on a toy that is never terminal: a leaf "
            "vector entered the backup from somewhere other than the stub head"
        )

    def to_play(self) -> int:
        """Return the mover, alternating every ply."""
        return len(self.history) % 2


class FlatWdlEvaluator:
    """The self-play value seam, with the head pinned to a value-flat draw.

    ``value`` is computed by the production
    :func:`~komugi.selfplay.inference.wdl_utility` -- the same call
    ``komugi.selfplay.adapter._value_from_wdl`` makes -- so the search is driven
    exactly as self-play drives it, contempt included, rather than by a
    re-derivation of the formula that could drift from it.
    """

    def __init__(self, draw_score: float) -> None:
        self.draw_score = draw_score
        self.utility = float(
            wdl_utility(np.asarray([_WDL], dtype=np.float32), draw_score)[0]
        )
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, utility)`: position-dependent prior, constant value."""
        assert isinstance(state, FlatToyState), "evaluator was handed a foreign state"
        self.calls += 1
        return _logits_for(state.history), self.utility


def _search(
    *, draw_score: float, m: int, n: int
) -> tuple[SearchResult, GumbelConfig, FlatWdlEvaluator]:
    """Run one search of the toy at `draw_score` and return it with its context."""
    config = GumbelConfig(num_actions=_NUM_ACTIONS, m=m, n=n, seed=_SEED)
    evaluator = FlatWdlEvaluator(draw_score)
    result = GumbelSearch(evaluator, config).search(FlatToyState())
    return result, config, evaluator


def _masked_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Return the softmax of `logits` over the legal set, zero elsewhere."""
    legal = _legal_mask_array()
    scores = logits.astype(np.float64)
    shifted = np.where(legal, scores - float(scores[legal].max()), -np.inf)
    exp = np.where(legal, np.exp(shifted), 0.0)
    return np.asarray(exp / float(exp.sum()), dtype=np.float64)


def _sigma_scale(result: SearchResult, config: GumbelConfig) -> float:
    """Return sigma's multiplier at the root: ``(c_visit + max N) * c_scale``."""
    max_visits = int(result.visit_counts.max())
    return (config.c_visit + max_visits) * config.c_scale


def _reference_action(result: SearchResult) -> int:
    """Return a legal action the search never visited.

    Its completed Q is the root's ``v_mix`` by construction
    (``_completed_q`` completes every unvisited action with it), which is
    reported as :attr:`SearchResult.root_value` -- so it is the one action whose
    Q is known without inverting anything.
    """
    legal = _legal_mask_array()
    unvisited = np.flatnonzero(legal & (result.visit_counts == 0))
    assert unvisited.size > 0, "every legal action was a candidate: no reference left"
    return int(unvisited[0])


def _root_q(result: SearchResult, config: GumbelConfig, action: int) -> float:
    """Return `action`'s root-frame Q, recovered from the policy target.

    See the module docstring: ``SearchResult`` exposes no per-action Q, so this
    inverts ``policy_target ~ exp(logit + sigma(completedQ))`` against an
    unvisited reference action whose completed Q is ``root_value``.
    """
    reference = _reference_action(result)
    logits = result.root_prior_logits.astype(np.float64)
    target = result.policy_target.astype(np.float64)
    log_ratio = math.log(target[action]) - math.log(target[reference])
    logit_gap = float(logits[action] - logits[reference])
    return result.root_value + (log_ratio - logit_gap) / _sigma_scale(result, config)


def _one_visit_actions(result: SearchResult) -> list[int]:
    """Return the root candidates that were visited exactly once."""
    return [int(action) for action in np.flatnonzero(result.visit_counts == 1)]


def test_halving_schedule_is_the_documented_ladder() -> None:
    """Pin the visit ladder the rest of the file reasons from.

    The ``Q = -s*d`` claim is a claim about *one-visit* candidates, so the
    schedule that produces them is pinned first: if a future budget-split change
    moves the ladder, this fails here rather than silently turning the other
    tests into assertions about a different tree.
    """
    root_logits = _logits_for(())[:_NUM_LEGAL]
    assert np.unique(root_logits).size == _NUM_LEGAL, "root logits must be distinct"

    fast, _, _ = _search(draw_score=0.0, m=_FAST[0], n=_FAST[1])
    assert fast.sims_used == _FAST[1]
    visited = fast.visit_counts[fast.visit_counts > 0]
    assert sorted(int(count) for count in visited) == [1, 1, 1, 1, 3, 3, 7, 15]
    assert len(_one_visit_actions(fast)) == 4

    full, _, _ = _search(draw_score=0.0, m=_FULL[0], n=_FULL[1])
    assert full.sims_used == _FULL[1]
    assert int((full.visit_counts > 0).sum()) == _FULL[0]
    # Round 0 hands 2 visits to 10 of the 32 candidates and 1 to the other 22;
    # the cut keeps 16, so at least 6 one-visit candidates always survive.
    assert len(_one_visit_actions(full)) >= 6


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
def test_draw_term_is_identically_absent_at_zero(m: int, n: int) -> None:
    """At ``s = 0`` the search is the prior, exactly -- not approximately.

    With ``w == l`` the whole leaf value is the draw term, so ``s = 0`` makes
    every leaf worth ``0.0``: ``v_mix`` is exactly zero, every completed Q is
    zero, ``sigma`` contributes nothing, and the recorded target collapses to
    the masked softmax of the root logits. That is the property campaign 4A
    buys, stated as an identity rather than as a tolerance.
    """
    result, config, evaluator = _search(draw_score=0.0, m=m, n=n)

    assert evaluator.utility == 0.0
    assert result.root_value == 0.0

    expected = _masked_softmax(result.root_prior_logits)
    legal = _legal_mask_array()
    np.testing.assert_allclose(
        result.policy_target[legal], expected[legal], rtol=1e-6, atol=0.0
    )
    assert np.array_equal(
        result.policy_target[~legal],
        np.zeros(_NUM_ACTIONS - _NUM_LEGAL, dtype=np.float32),
    )

    for action in np.flatnonzero(result.visit_counts > 0):
        assert _root_q(result, config, int(action)) == pytest.approx(0.0, abs=_Q_TOL)


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
@pytest.mark.parametrize("draw_score", [0.2, -0.2], ids=["plus", "minus"])
def test_one_visit_candidates_carry_minus_s_times_d(
    draw_score: float, m: int, n: int
) -> None:
    """A one-visit root candidate's Q is ``-s*d`` exactly, for either sign of s.

    One visit is one depth-1 leaf: the stub's mover-relative ``s*d`` crosses a
    single change of mover on the way up and arrives at the root negated, with
    nothing else averaged into it. Under campaign 3's ``s = -0.20`` that is
    ``+0.1`` at ``d = 0.5`` -- a *pro-draw* bias on every round-0 cut, the
    opposite of the contempt that was configured (design review F1).
    """
    result, config, evaluator = _search(draw_score=draw_score, m=m, n=n)

    assert evaluator.utility == pytest.approx(draw_score * _DRAW_MASS, abs=1e-6)

    one_visit = _one_visit_actions(result)
    assert one_visit, "no one-visit candidate: the halving schedule changed"
    expected = -draw_score * _DRAW_MASS
    for action in one_visit:
        assert _root_q(result, config, action) == pytest.approx(expected, abs=_Q_TOL)


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
def test_one_visit_q_flips_sign_with_the_draw_score(m: int, n: int) -> None:
    """Flipping ``s`` flips the one-visit bias and preserves its magnitude.

    Also pins, at ``(8, 32)``, that the *identity* of the one-visit set does not
    depend on ``s``: after round 0 every candidate carries the same constant Q,
    so ``sigma`` shifts all eight scores equally and the round-0 cut is decided
    by ``g + logit`` alone. At ``(32, 256)`` round 0 is uneven (2 visits for ten
    candidates, 1 for twenty-two), so there the cut can and does move with ``s``
    and only the recovered values are compared.
    """
    plus, plus_config, _ = _search(draw_score=0.2, m=m, n=n)
    minus, minus_config, _ = _search(draw_score=-0.2, m=m, n=n)

    plus_actions = _one_visit_actions(plus)
    minus_actions = _one_visit_actions(minus)
    if (m, n) == _FAST:
        assert plus_actions == minus_actions

    plus_q = [_root_q(plus, plus_config, action) for action in plus_actions]
    minus_q = [_root_q(minus, minus_config, action) for action in minus_actions]

    assert all(value < 0.0 for value in plus_q)
    assert all(value > 0.0 for value in minus_q)
    assert max(plus_q) == pytest.approx(-max(minus_q), abs=_Q_TOL)
    assert min(plus_q) == pytest.approx(-min(minus_q), abs=_Q_TOL)


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
def test_finalist_q_is_finite_and_its_sign_is_reported(
    m: int, n: int, record_property: Callable[[str, object], None]
) -> None:
    """Report -- do not assert -- the finalist's Q under both signs of ``s``.

    The finalist's visits are a *mixture* of depth classes, and which mixture
    depends on the sign of ``s`` itself: for ``s > 0`` a deeper leaf looks worse
    to the interior selection rule, which then keeps opening fresh depth-2
    children, while for ``s < 0`` it tunnels down the line it already has. So
    the finalist's Q is not a constant anyone can pin without pinning that
    dynamic, which is why design review F2 asked for it as a *reported* number.
    Finiteness is the only assertion.
    """
    for label, draw_score in (("plus", 0.2), ("minus", -0.2)):
        result, config, _ = _search(draw_score=draw_score, m=m, n=n)
        finalist = result.action
        value = _root_q(result, config, finalist)
        assert math.isfinite(value)
        assert int(result.visit_counts[finalist]) == int(result.visit_counts.max())
        record_property(f"finalist_action_{label}", finalist)
        record_property(f"finalist_visits_{label}", int(result.visit_counts[finalist]))
        record_property(f"finalist_q_{label}", value)
        record_property(f"finalist_q_sign_{label}", int(np.sign(value)))
        record_property(f"root_value_{label}", result.root_value)
        record_property(f"sigma_scale_{label}", _sigma_scale(result, config))


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
@pytest.mark.parametrize(
    ("draw_score", "floor", "ceiling"),
    [(0.2, 1.0, None), (-0.2, 1.0, None), (0.0, None, 1e-3)],
    ids=["plus", "minus", "zero"],
)
def test_target_departs_from_the_prior_only_when_s_is_nonzero(
    draw_score: float, floor: float | None, ceiling: float | None, m: int, n: int
) -> None:
    """Size the bias where it lands: in the recorded training target.

    ``sigma`` multiplies Q by ``(c_visit + max_visits)`` -- 65 in fast mode,
    ~135 in full -- so a 0.1 bias is several nats of shift in the label the
    network is trained on, the same order as the genuine Q differences the
    search is supposed to be reporting. At ``s = 0`` the target is the prior to
    float32 and the shift is nothing at all.
    """
    result, _, _ = _search(draw_score=draw_score, m=m, n=n)
    prior = _masked_softmax(result.root_prior_logits)
    candidates = np.flatnonzero(result.visit_counts > 0)
    shift = np.abs(
        np.log(result.policy_target[candidates].astype(np.float64))
        - np.log(prior[candidates])
    )

    if floor is not None:
        assert float(shift.max()) >= floor
    if ceiling is not None:
        assert float(shift.max()) <= ceiling


# ===========================================================================
# KOM-159 T3' and T6: the moves-left conversion term on the same toy
# ===========================================================================
#
# The file above measures an **even** leaf term (the draw constant) and finds a
# parity artefact. What follows measures an **odd** one and finds none: the same
# rig, the same recovery, the opposite conclusion. That contrast is the point --
# design §3.2's claim is exactly "T composes with negamax because it is odd",
# and the two halves of this file are the executed version of it.
#
# Design review F4 respecified T3 because the original could not pass: its
# frames were unstated, its winning leaf would not have survived as a one-visit
# candidate, and "the root-frame Q is unchanged under w <-> l" is false (the
# swap negates v as well). What is pinned here is F4's version.

_LAM: Final = 0.05
_TAU0: Final = 0.20
_TAU1: Final = 0.40
_SLOPE: Final = 32
"""The pre-registered parameters (design §3.4)."""

_MLH: Final = MlhUtility(lam=_LAM, tau0=_TAU0, tau1=_TAU1, slope_plies=_SLOPE)
_MLH_OFF: Final = MlhUtility(lam=0.0, tau0=_TAU0, tau1=_TAU1, slope_plies=_SLOPE)

_ROOT_PLIES: Final = 100.0
_SHORT_PLIES: Final = 80.0
_LONG_PLIES: Final = 120.0
"""The root's own predicted length and the two depth-1 leaves' (review F4).

80 and 120 are 20 plies either side of the root, i.e. ``f = +-0.625`` at a
32-ply slope -- inside the saturation shoulder on purpose, so the assertion is
about the ramp rather than about the clip.
"""

_LOSING_WDL: Final = (0.2, 0.1, 0.7)
"""What the two depth-1 leaves answer, **in the leaf mover's frame**.

A depth-1 leaf has the OPPONENT to move, so this is the opponent reading itself
as losing by ``v = -0.5``: the root mover is winning. Stating the frame is F4's
first correction to T3 -- ``(0.8, 0, 0.2)`` at a depth-1 leaf means the *root*
is losing, which is the opposite experiment.
"""

_WINNING_WDL: Final = (0.7, 0.1, 0.2)
"""The same position one ply deeper, where the ROOT mover is to move again."""

_GATE: Final = 0.5
"""``g(0.5) = 0.5 * clip((0.5 - 0.20)/0.20, 0, 1) = 0.5`` -- the ramp is
saturated at ``|v| = 0.5``, so the gate is just ``|v|``."""


def _fraction(plies: float) -> float:
    """Return `plies` on the moves-left head's ``[0, 1]`` scale.

    Every value used here is a multiple of ``1/512``, so the round trip through
    :data:`~komugi.engine.observation.MAX_TURNS` is exact and the assertions can
    be about the term rather than about binary rounding.
    """
    return plies / MAX_TURNS


class MlhToyEvaluator:
    """The self-play leaf seam, driven by a history-keyed head answer.

    Routes through the production
    :func:`komugi.selfplay.adapter._value_from_wdl` -- the same call
    ``make_evaluator``'s closure makes -- so what the search sees is what
    self-play would see, term, clamp and all. The only thing stubbed is the
    network: `answer` maps a move history to the ``(wdl, moves_left)`` the head
    would have returned for that position.

    The context is the seam's own: this class hands the same one to every call,
    exactly as a worker does within one search, so the FIRST evaluation (the
    root, guaranteed by ``GumbelSearch.search``) sets the reference.
    """

    def __init__(
        self,
        answer: Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]],
        *,
        draw_score: float = 0.0,
        mlh: MlhUtility | None = None,
        context: SearchContext | None = None,
    ) -> None:
        self._answer = answer
        self._draw_score = draw_score
        self._mlh = mlh
        self._context = context
        self.calls = 0
        self.seen: list[tuple[int, ...]] = []

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, utility)` for `state`, through the production seam."""
        assert isinstance(state, FlatToyState), "evaluator was handed a foreign state"
        self.calls += 1
        self.seen.append(state.history)
        wdl, moves_left = self._answer(state.history)
        # Two-tuple on purpose: this class is the *reference* the KOM-159
        # identity nodes below compare against, so it hands the search exactly
        # what it handed it before the searched-WDL vector existed. The vector
        # third element that `_value_from_wdl` now also returns is dropped here
        # and carried by `MlhWdlToyEvaluator` instead.
        logits, value, _vector = _value_from_wdl(
            _logits_for(state.history),
            np.asarray(wdl, dtype=np.float32),
            self._draw_score,
            moves_left=moves_left,
            mlh=self._mlh,
            context=self._context,
        )
        return logits, value


def _flat_answer(
    history: tuple[int, ...],
) -> tuple[tuple[float, float, float], float]:
    """Return the value-flat head answer used by the T6 identity runs."""
    del history
    return _WDL, _fraction(_ROOT_PLIES)


def _conversion_answer(
    short_edge: int, long_edge: int
) -> Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]]:
    """Return a head keyed on the ROOT EDGE, not on the depth (review F4.4).

    Two root actions lead into decided sub-trees -- `short_edge` into one the
    head reads as ending 20 plies sooner than the root expected, `long_edge` 20
    plies later -- and everything else is value-flat. Inside a decided sub-tree
    the answer alternates frame with the mover and keeps ``m`` fixed, which is
    the head of a position whose length is *known*: it says "80 plies from the
    root" at every node of that line, whoever is to move.

    That fixed-``m`` reading is what makes the arithmetic exact rather than
    approximate, and it is the cleanest possible statement of design §3.2's
    claim: every leaf of the sub-tree, at every depth class, contributes the
    **same** root-frame value. An even term would alternate instead, which is
    precisely what this file's first half measures for ``s*d``.
    """

    def answer(
        history: tuple[int, ...],
    ) -> tuple[tuple[float, float, float], float]:
        """Return the head's answer for the position reached by `history`."""
        if not history:
            return _WDL, _fraction(_ROOT_PLIES)
        if history[0] == short_edge:
            plies = _SHORT_PLIES
        elif history[0] == long_edge:
            plies = _LONG_PLIES
        else:
            return _WDL, _fraction(_ROOT_PLIES)
        # Odd depth is the opponent to move, and the opponent is the one
        # losing here; even depth is the root mover, winning.
        opponent_to_move = len(history) % 2 == 1
        return (
            (_LOSING_WDL if opponent_to_move else _WINNING_WDL),
            _fraction(plies),
        )

    return answer


def _mlh_search(
    answer: Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]],
    *,
    m: int,
    n: int,
    mlh: MlhUtility | None,
    draw_score: float = 0.0,
) -> tuple[SearchResult, GumbelConfig, MlhToyEvaluator]:
    """Run one search of the toy through the production leaf seam."""
    config = GumbelConfig(num_actions=_NUM_ACTIONS, m=m, n=n, seed=_SEED)
    evaluator = MlhToyEvaluator(
        answer, draw_score=draw_score, mlh=mlh, context=SearchContext()
    )
    result = GumbelSearch(evaluator, config).search(FlatToyState())
    return result, config, evaluator


def _candidate_pair() -> tuple[int, int]:
    """Return two root actions that are certainly round-0 candidates.

    The candidate set is drawn from ``gumbel + root logits`` **before** any leaf
    is evaluated (``gumbel.py``'s ``search``), so it does not depend on the head
    at all: a value-flat probe at the same seed names the same eight actions the
    conversion runs will sample. Reading it off a probe rather than
    re-implementing the draw here keeps the test from having a second copy of
    the sampling rule to drift from.
    """
    probe, _, _ = _search(draw_score=0.0, m=_FAST[0], n=_FAST[1])
    candidates = [int(action) for action in np.flatnonzero(probe.visit_counts > 0)]
    assert len(candidates) >= 2, "the probe found fewer than two root candidates"
    return candidates[0], candidates[1]


def test_the_shorter_win_outranks_the_longer_one_by_exactly_the_term() -> None:
    """T3': ``Q(a) - Q(b) == lambda * g(0.5) * 1.25``, and the shorter win wins.

    Both sub-trees are the *same* decided position -- the root mover winning
    with ``|v| = 0.5`` -- differing only in how long the head thinks it takes:
    80 plies against 120, either side of the root's own 100. ``f`` is therefore
    ``+0.625`` and ``-0.625``, and the gap is ``lambda * g * 1.25 = 0.03125``.

    Recovered through :func:`_root_q`, which inverts the recorded target against
    an unvisited reference action; the difference of two recoveries cancels
    ``root_value``, so the number asserted is the Q gap and nothing else. The
    tolerance is the file's usual 1e-5 (design review F2: float32 targets make
    1e-12 unreachable).

    Both actions end the search with many visits -- a root-frame ``Q`` of ``+0.5``
    is 33 nats over a value-flat sibling and survives every cut, which is F4's
    second correction to the original T3. That does **not** weaken the
    assertion: every leaf of a decided sub-tree contributes the same root-frame
    value here, so ``W(a)/N(a)`` is that value at any visit count. The equality
    holding at 15 visits is the no-oscillation claim, executed.
    """
    short_edge, long_edge = _candidate_pair()
    result, config, _ = _mlh_search(
        _conversion_answer(short_edge, long_edge),
        m=_FAST[0],
        n=_FAST[1],
        mlh=_MLH,
    )

    q_short = _root_q(result, config, short_edge)
    q_long = _root_q(result, config, long_edge)

    assert q_short > q_long
    assert q_short - q_long == pytest.approx(_LAM * _GATE * 1.25, abs=_Q_TOL)
    # The absolute values too, not only the gap: a term of the wrong SIGN at
    # every depth would keep the gap and move both numbers.
    assert q_short == pytest.approx(0.5 + _LAM * _GATE * 0.625, abs=_Q_TOL)
    assert q_long == pytest.approx(0.5 - _LAM * _GATE * 0.625, abs=_Q_TOL)


def test_the_term_reaches_the_recorded_target_amplified_by_sigma() -> None:
    """The gap lands in the training label, at ``sigma`` times its size.

    Where the pressure actually acts (design §3.5): the recorded
    ``policy_target`` is ``softmax(logit + sigma(completedQ))``, so a 0.03125 Q
    gap between two candidates is ``sigma * 0.03125`` nats of log-odds in the
    label -- about 2 nats in fast mode. Unlike the draw constant's, this shift
    is the intended signal, and it is bounded: this asserts the size rather
    than merely the sign.
    """
    short_edge, long_edge = _candidate_pair()
    result, config, _ = _mlh_search(
        _conversion_answer(short_edge, long_edge),
        m=_FAST[0],
        n=_FAST[1],
        mlh=_MLH,
    )

    scale = _sigma_scale(result, config)
    logits = result.root_prior_logits.astype(np.float64)
    target = result.policy_target.astype(np.float64)
    observed = (
        math.log(target[short_edge])
        - math.log(target[long_edge])
        - float(logits[short_edge] - logits[long_edge])
    )

    assert observed == pytest.approx(scale * _LAM * _GATE * 1.25, abs=1e-4)
    assert observed > 1.0, "the term is worth more than a nat in the fast label"


def test_a_depth_two_leaf_reaches_the_root_as_the_same_preference(
    record_property: Callable[[str, object], None]
) -> None:
    """F4's second parametrisation: the depth offset is bounded by ``lambda*g/32``.

    A perfectly calibrated head answers ``m_root - k`` at depth ``k``, so the
    depth-2 leaf of a line the head reads as 80 plies long from the root answers
    79. Its root-frame contribution is then ``lambda * g / 32`` away from the
    depth-1 leaf's -- one ply of slope, 0.00078 of utility, 0.05 nats in a fast
    label. Same sign, same preference, no oscillation: contrast the ``2*s*d``
    gap between depth classes that this file's first half measures for the draw
    constant.

    Reported as well as asserted, per review F1: it is the one residual the
    oddness argument does not remove, and §3.5's "no parity structure enters
    the target" should be read with this number beside it.
    """
    context = SearchContext()
    context.m_root = _ROOT_PLIES

    # Depth 1: the opponent to move, losing. One mover change to the root, so
    # the root-frame contribution is the negation of the leaf's own utility.
    _p1, depth_one, _v1 = _value_from_wdl(
        _logits_for((0,)),
        np.asarray(_LOSING_WDL, dtype=np.float32),
        0.0,
        moves_left=_fraction(_SHORT_PLIES),
        mlh=_MLH,
        context=context,
    )
    # Depth 2: the root mover again, winning, one ply closer to the end. Two
    # mover changes, so the root-frame contribution is the utility itself.
    _p2, depth_two, _v2 = _value_from_wdl(
        _logits_for((0, 1)),
        np.asarray(_WINNING_WDL, dtype=np.float32),
        0.0,
        moves_left=_fraction(_SHORT_PLIES - 1.0),
        mlh=_MLH,
        context=context,
    )

    root_frame_one = -depth_one
    root_frame_two = depth_two
    offset = root_frame_two - root_frame_one
    record_property("depth_offset_utility", offset)
    record_property("depth_offset_nats_fast", offset * 65.0)

    assert root_frame_one > 0.0 and root_frame_two > 0.0, (
        "both depth classes must report the root mover as winning; a sign flip "
        "between them is the oscillation the draw constant has"
    )
    assert abs(offset) <= _LAM * _GATE / _SLOPE * (1.0 + 1e-6)


def test_the_losing_side_prolongs_at_the_interior_node() -> None:
    """The opponent's own choice at the depth-1 node prefers the LONGER loss.

    The third of F4's assertions, pinned on
    :func:`~komugi.search.gumbel._select_interior` -- the function that makes
    the choice -- with the node's two children carrying utilities the production
    leaf seam produced. It is **not** pinned on the finalist's second visit, and
    the reason is mechanical: at a node whose children are all unvisited every
    completed Q is the node's own ``v_mix``, so the first interior choice is the
    argmax of the prior and says nothing about the term. The choice becomes a
    statement about value only once the children have been scored, which is the
    state built here.

    In the opponent's frame the two continuations are worth ``-0.525`` (the
    line that ends 40 plies early) and ``-0.475`` (the one that runs 40 plies
    late). The loser prefers the second: the same utility as "the winner
    prefers the shorter line", re-expressed one edge up, which is what oddness
    means.
    """
    context = SearchContext()
    context.m_root = _ROOT_PLIES

    def continuation(plies: float) -> float:
        """Return the opponent-frame value of a depth-2 leaf `plies` from the end."""
        _policy, utility, _vector = _value_from_wdl(
            _logits_for((0, int(plies))),
            np.asarray(_WINNING_WDL, dtype=np.float32),
            0.0,
            moves_left=_fraction(plies),
            mlh=_MLH,
            context=context,
        )
        # One mover change from the depth-2 leaf up to the depth-1 node.
        return -utility

    quick_loss = continuation(_ROOT_PLIES - 40.0)
    slow_loss = continuation(_ROOT_PLIES + 40.0)
    assert slow_loss > quick_loss, (
        "the losing side must prefer the longer line before the search can"
    )

    legal = np.zeros(_NUM_ACTIONS, dtype=np.bool_)
    legal[0] = legal[1] = True
    node = _Node(
        # The opponent, one ply below a root at to_play 0.
        to_play=1,
        is_terminal=False,
        value=0.0,
        legal=legal,
        # Equal priors, so the choice is the value's alone.
        logits=np.zeros(_NUM_ACTIONS, dtype=np.float32),
    )
    node.child_visits[0] = node.child_visits[1] = 1
    node.child_value_sums[0] = quick_loss
    node.child_value_sums[1] = slow_loss

    config = GumbelConfig(num_actions=_NUM_ACTIONS, m=_FAST[0], n=_FAST[1], seed=_SEED)
    assert _select_interior(node, config) == 1, (
        "the interior rule took the quick loss; the term's sign is inverted "
        "for the side that is behind"
    )


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
@pytest.mark.parametrize(
    "mlh", [None, _MLH_OFF], ids=["no-config", "lambda-zero"]
)
def test_at_lambda_zero_the_search_is_the_pre_change_search(
    mlh: MlhUtility | None, m: int, n: int
) -> None:
    """T6: ``policy_target`` and ``root_value`` are the pre-change numbers, exactly.

    The reference is the search this file already ran before KOM-159 existed --
    :class:`FlatWdlEvaluator`, which never reaches the new parameters -- and the
    comparison is ``array_equal`` on the float32 target and ``==`` on the value,
    not a tolerance. That is only achievable because the disabled term is
    *skipped* rather than added as ``0.0`` (design review F9): an addition, even
    of an exact zero, would still have to be the same addition on both sides,
    and this way there is no addition to compare.

    Both shapes of "off" are covered, because ``None`` is what the arena passes
    and ``lam == 0.0`` is what a campaign-6 worker passes before the switch
    iteration.
    """
    reference, _, _ = _search(draw_score=0.0, m=m, n=n)
    result, _, evaluator = _mlh_search(_flat_answer, m=m, n=n, mlh=mlh)

    assert evaluator.calls > 0
    assert result.root_value == reference.root_value
    assert np.array_equal(result.policy_target, reference.policy_target)
    assert np.array_equal(result.visit_counts, reference.visit_counts)
    assert result.action == reference.action
    assert np.array_equal(result.root_prior_logits, reference.root_prior_logits)


# ===========================================================================
# KOM-173 T3: what the SEARCHED WDL VECTOR does to the same two measurements
# ===========================================================================
#
# The file above measures two leaf utility terms through the scalar the search
# ranks on. The vector of KOM-173 carries neither of them: it is the network's
# own ``(w, d, l)``, permuted once per mover change and averaged, with no draw
# score and no conversion term anywhere in it. So on this same rig:
#
# * at ``s != 0`` the vector's utility ``root_wdl[0] - root_wdl[2]`` and the
#   scalar ``root_value`` DIFFER, and the difference is the ``s``-weighted draw
#   mixture -- the very quantity the first half of this file characterises. That
#   is the honest statement of D4's identity: it holds at ``s = 0``, and the gap
#   at ``s != 0`` is measurable rather than mysterious;
# * at ``lambda > 0`` the vector does not move at all, even though the tree does.
#
# Neither claim touches the scalar: the play-identity of the search under the
# vector is pinned in ``test_gumbel_unit.py`` (T1).


class FlatWdlVectorEvaluator:
    """:class:`FlatWdlEvaluator` in its three-tuple shape (KOM-173).

    The same utility -- computed by the same production
    :func:`~komugi.selfplay.inference.wdl_utility` call, contempt included --
    with the head's ``(w, d, l)`` handed to the search beside it. A separate
    class rather than a flag on the original, so the two-tuple reference the
    identity nodes above compare against stays exactly what it was.
    """

    def __init__(self, draw_score: float) -> None:
        self.draw_score = draw_score
        self.utility = float(
            wdl_utility(np.asarray([_WDL], dtype=np.float32), draw_score)[0]
        )
        self.calls = 0

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return `(logits, utility, wdl)`; the vector is ``float64`` (review F5)."""
        assert isinstance(state, FlatToyState), "evaluator was handed a foreign state"
        self.calls += 1
        return (
            _logits_for(state.history),
            self.utility,
            np.asarray(_WDL, dtype=np.float64),
        )


def _wdl_search(
    *, draw_score: float, m: int, n: int
) -> tuple[SearchResult, GumbelConfig, FlatWdlVectorEvaluator]:
    """Run one vector-carrying search of the toy at `draw_score`."""
    config = GumbelConfig(num_actions=_NUM_ACTIONS, m=m, n=n, seed=_SEED)
    evaluator = FlatWdlVectorEvaluator(draw_score)
    result = GumbelSearch(evaluator, config).search(FlatToyState())
    return result, config, evaluator


def _vector_utility(result: SearchResult) -> float:
    """Return ``root_wdl[0] - root_wdl[2]``, computed in float64."""
    wide = result.root_wdl.astype(np.float64)
    return float(wide[0] - wide[2])


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
def test_the_vector_utility_is_the_root_value_at_zero_draw_score(
    m: int, n: int
) -> None:
    """T3/D4 at ``s = 0``: the two routes agree, and the scalar path is untouched.

    The value-flat head makes every leaf worth exactly ``0``, so ``root_value``
    is exactly ``0.0`` (the first half of this file pins that as an identity) and
    the vector's utility must be ``0`` too -- ``w == l`` at every leaf, and the
    swap cannot break a symmetry it is a permutation of. The scalar fields are
    compared against the two-tuple run with ``array_equal``: at ``s = 0`` the
    vector changes nothing, which is the premise every measurement in this file
    rests on.
    """
    reference, _, _ = _search(draw_score=0.0, m=m, n=n)
    result, _, evaluator = _wdl_search(draw_score=0.0, m=m, n=n)

    assert evaluator.utility == 0.0
    assert result.root_value == reference.root_value == 0.0
    assert np.array_equal(result.policy_target, reference.policy_target)
    assert np.array_equal(result.visit_counts, reference.visit_counts)
    assert result.action == reference.action
    assert _vector_utility(result) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        result.root_wdl.astype(np.float64), np.asarray(_WDL), rtol=0.0, atol=1e-7
    )


@pytest.mark.parametrize(("m", "n"), [_FAST, _FULL], ids=["fast", "full"])
@pytest.mark.parametrize("draw_score", [0.2, -0.2], ids=["plus", "minus"])
def test_the_gap_at_nonzero_draw_score_is_the_s_weighted_draw_mixture(
    draw_score: float, m: int, n: int
) -> None:
    """T3: ``root_value - (root_wdl[0] - root_wdl[2])`` is the draw mixture.

    Three things at once, on the same tree:

    * the vector's utility is **exactly** zero -- every leaf answers ``w == l``
      and the swap keeps it that way, so the whole gap is the scalar's;
    * the gap therefore equals ``root_value``, and ``root_value`` is ``v_mix``
      of the ``+-s*d`` the scalar carries. That number is rebuilt at the bottom
      of this test from the root's prior, the visit counts and the visited
      actions' Q recovered out of ``policy_target`` by :func:`_root_q` -- an
      independent route, which reads nothing the vector produced;
    * the scalar play is **identical** to the two-tuple run at this nonzero
      ``s`` -- review F8's point, and the reason D4's identity may be stated at
      ``s = 0`` without making the *implementation* conditional on ``s``.
    """
    reference, _, _ = _search(draw_score=draw_score, m=m, n=n)
    result, config, evaluator = _wdl_search(draw_score=draw_score, m=m, n=n)

    assert evaluator.utility == pytest.approx(draw_score * _DRAW_MASS, abs=1e-6)
    assert result.action == reference.action
    assert result.root_value == reference.root_value
    assert np.array_equal(result.policy_target, reference.policy_target)
    assert np.array_equal(result.visit_counts, reference.visit_counts)

    assert _vector_utility(result) == pytest.approx(0.0, abs=1e-12)
    gap = result.root_value - _vector_utility(result)
    assert gap == pytest.approx(result.root_value, abs=1e-12)
    assert abs(gap) > 1e-3, "the draw mixture is the thing being measured"

    prior = _masked_softmax(result.root_prior_logits)
    visited = np.flatnonzero(result.visit_counts > 0)
    total = int(result.visit_counts.sum())
    prior_mass = float(prior[visited].sum())
    weighted = sum(
        float(prior[action]) * _root_q(result, config, int(action))
        for action in visited
    )
    expected = (draw_score * _DRAW_MASS + total * weighted / prior_mass) / (
        1.0 + total
    )
    assert gap == pytest.approx(expected, abs=_Q_TOL)


_FRAME_CONSISTENT_ROOT: Final = (0.7, 0.1, 0.2)
"""The root-frame reading every leaf of :func:`_frame_consistent_answer` carries.

``_LOSING_WDL`` at odd depth and ``_WINNING_WDL`` at even depth are each other's
swaps, so a leaf at depth ``k`` contributes ``swap^k`` of its own vector -- which
is this same triple for every ``k``. That is what makes ``root_wdl`` exactly
computable here without knowing the visit schedule, and therefore comparable
across two searches that built *different trees*.
"""


def _frame_consistent_answer(
    short_edge: int, long_edge: int
) -> Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]]:
    """Return :func:`_conversion_answer`'s head, applied on EVERY root edge.

    The conversion head keyed on two edges leaves the rest value-flat, which is
    what T3' needs and what a ``root_wdl`` claim does not: a value-flat leaf's
    vector is symmetric and would dilute the constant. Here every line is a
    decided one -- the root mover winning with ``|v| = 0.5`` -- so every leaf's
    root-frame vector is :data:`_FRAME_CONSISTENT_ROOT`.

    ``moves_left`` still varies, and in the one shape that makes both halves of
    the test bite: ``long_edge``'s line runs 20 plies *past* the root's own
    prediction while every other line ends 20 plies sooner. So the term is live
    and does not average to zero (the scalar moves, measurably), and the two
    lambdas disagree about one candidate's worth (the trees differ) -- while the
    vector, which the term never touches, cannot move either way. `short_edge` is
    accepted for symmetry with :func:`_conversion_answer`'s signature and is the
    default case here.
    """
    del short_edge

    def answer(
        history: tuple[int, ...],
    ) -> tuple[tuple[float, float, float], float]:
        """Return the head's answer for the position reached by `history`."""
        if not history:
            return _WINNING_WDL, _fraction(_ROOT_PLIES)
        plies = _LONG_PLIES if history[0] == long_edge else _SHORT_PLIES
        opponent_to_move = len(history) % 2 == 1
        return (
            (_LOSING_WDL if opponent_to_move else _WINNING_WDL),
            _fraction(plies),
        )

    return answer


class MlhWdlToyEvaluator:
    """:class:`MlhToyEvaluator` in its three-tuple shape.

    Same production seam, same context discipline; the difference is that the
    leaf's ``(w, d, l)`` is handed to the search as well as collapsed into the
    scalar. Which is the whole experiment: the conversion term is *in* the scalar
    and cannot be in the vector, because the vector never passes through
    :func:`~komugi.selfplay.adapter.mlh_utility_term` at all.
    """

    def __init__(
        self,
        answer: Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]],
        *,
        draw_score: float = 0.0,
        mlh: MlhUtility | None = None,
        context: SearchContext | None = None,
    ) -> None:
        self._answer = answer
        self._draw_score = draw_score
        self._mlh = mlh
        self._context = context
        self.calls = 0

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float32]]:
        """Return `(logits, utility, wdl)` through the production leaf seam."""
        assert isinstance(state, FlatToyState), "evaluator was handed a foreign state"
        self.calls += 1
        wdl, moves_left = self._answer(state.history)
        return _value_from_wdl(
            _logits_for(state.history),
            np.asarray(wdl, dtype=np.float32),
            self._draw_score,
            moves_left=moves_left,
            mlh=self._mlh,
            context=self._context,
        )


def _mlh_wdl_search(
    answer: Callable[[tuple[int, ...]], tuple[tuple[float, float, float], float]],
    *,
    m: int,
    n: int,
    mlh: MlhUtility | None,
) -> tuple[SearchResult, MlhWdlToyEvaluator]:
    """Run one vector-carrying search through the production leaf seam."""
    config = GumbelConfig(num_actions=_NUM_ACTIONS, m=m, n=n, seed=_SEED)
    evaluator = MlhWdlToyEvaluator(
        answer, draw_score=0.0, mlh=mlh, context=SearchContext()
    )
    result = GumbelSearch(evaluator, config).search(FlatToyState())
    return result, evaluator


def test_the_vector_carries_no_conversion_term_at_positive_lambda() -> None:
    """T3: ``root_wdl`` is bit-identical at ``lambda = 0`` and ``lambda > 0``.

    Every leaf of this head reads, in the root's frame,
    :data:`_FRAME_CONSISTENT_ROOT` -- whatever its depth, whatever its
    ``moves_left`` -- so ``root_wdl`` is that triple exactly, and it must be the
    *same bytes* at both lambdas even though the ``lambda > 0`` search cut its
    candidates differently and built a different tree. The scalar is where the
    term lives: ``root_value`` moves between the two runs, and the two numbers
    then disagree, which is D6 ("no utility enters the vector") stated as a
    measurement rather than as a promise.
    """
    short_edge, long_edge = _candidate_pair()
    answer = _frame_consistent_answer(short_edge, long_edge)

    off, off_evaluator = _mlh_wdl_search(answer, m=_FAST[0], n=_FAST[1], mlh=_MLH_OFF)
    on, on_evaluator = _mlh_wdl_search(answer, m=_FAST[0], n=_FAST[1], mlh=_MLH)

    assert off_evaluator.calls > 1 and on_evaluator.calls > 1
    expected = np.asarray(_FRAME_CONSISTENT_ROOT, dtype=np.float32)
    np.testing.assert_allclose(off.root_wdl, expected, rtol=0.0, atol=1e-7)
    assert np.array_equal(on.root_wdl, off.root_wdl), (
        "the searched vector moved with lambda; the conversion term is scalar-"
        "only (KOM-173 D6)"
    )
    # The other two readouts are weighted by the improved policy and by the
    # visit counts, which the term legitimately moves; what must not move is the
    # vector each weight is applied to, so they are compared at the float32
    # resolution of the constant rather than bit for bit.
    np.testing.assert_allclose(on.root_wdl_pi, expected, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(on.played_wdl, expected, rtol=0.0, atol=1e-7)

    # The term IS live: the scalar moved, so at lambda > 0 the identity no longer
    # holds -- which is the asymmetry D6 is about.
    assert on.root_value != off.root_value
    assert float(off.root_wdl[0]) - float(off.root_wdl[2]) == pytest.approx(
        off.root_value, abs=1e-6
    )
    assert (
        abs(float(on.root_wdl[0]) - float(on.root_wdl[2]) - on.root_value) > 1e-6
    ), "the conversion term did not reach the scalar: nothing was measured"


def test_the_two_seam_shapes_play_identically_at_positive_lambda() -> None:
    """The bit-identity claim, compared at ``lambda > 0`` (KOM-210 F5).

    ``gumbel.py``'s module docstring claims the scalar play is bit-identical
    between the two answer shapes at **any** ``draw_score`` and any MLH
    ``lambda``. ``s != 0`` is compared directly
    (``test_the_gap_at_nonzero_draw_score_is_the_s_weighted_draw_mixture``);
    ``lambda > 0`` was, before this, only claimed -- the test above compares two
    *vector* runs at two lambdas, never a vector run against the two-tuple
    reference at the same one. Review finding 5.

    Same head, same budget, same seed, same ``lambda``: the only difference is
    whether the leaf's vector is handed to the search beside the scalar
    (:class:`MlhWdlToyEvaluator`) or dropped at the seam
    (:class:`MlhToyEvaluator`, which is the pre-KOM-173 reference and shares the
    production ``_value_from_wdl`` body with it, so the two scalars are equal by
    construction rather than by rounding). ``action``, ``policy_target``,
    ``visit_counts`` and ``root_prior_logits`` are compared with
    ``array_equal`` and ``root_value`` with ``==``: a search that recomputed its
    ranking number from the vector would drop exactly the conversion term --
    which is live here -- and cut a different candidate set.
    """
    short_edge, long_edge = _candidate_pair()
    answer = _frame_consistent_answer(short_edge, long_edge)

    reference, _, reference_evaluator = _mlh_search(
        answer, m=_FAST[0], n=_FAST[1], mlh=_MLH
    )
    carrying, carrying_evaluator = _mlh_wdl_search(
        answer, m=_FAST[0], n=_FAST[1], mlh=_MLH
    )

    assert reference_evaluator.calls == carrying_evaluator.calls > 1
    assert carrying.action == reference.action
    assert carrying.root_value == reference.root_value
    assert np.array_equal(carrying.policy_target, reference.policy_target)
    assert np.array_equal(carrying.visit_counts, reference.visit_counts)
    assert np.array_equal(carrying.root_prior_logits, reference.root_prior_logits)
    assert carrying.sims_used == reference.sims_used

    # The term is live at this lambda -- without this the equality above would
    # hold for the trivial reason that there was no term to disagree about.
    at_zero, _, _ = _mlh_search(answer, m=_FAST[0], n=_FAST[1], mlh=_MLH_OFF)
    assert carrying.root_value != at_zero.root_value, (
        "lambda moved nothing: this comparison would pass at lambda = 0 too, "
        "which is the case the goldens already cover"
    )
    # And the vector really was carried, so the two runs differ in the way the
    # test intends: the reference's readouts are NaN, the carrying run's are not.
    assert bool(np.isnan(reference.root_wdl).all())
    assert bool(np.isfinite(carrying.root_wdl).all())
