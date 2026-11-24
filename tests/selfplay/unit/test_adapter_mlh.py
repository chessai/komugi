"""KOM-159 T4: the moves-left conversion term, at and below the leaf seam.

Four claims, and the design rests on all four (``docs/design/
kom-159-mlh-bringup-and-utility.md`` §3.1-§3.2, review F9):

* **shut below the gate.** ``|w - l| < tau0`` returns exactly ``0.0`` for every
  ``m``. Not "small": the draft and the opening are almost entirely below the
  gate (review F13), and a term that was merely tiny there would still be a
  tie-breaker between two moves the head cannot tell apart.
* **odd under ``w <-> l``.** ``T(swap(wdl), m) == -T(wdl, m)``, which is what
  lets the search's one-negation-per-mover-change backup carry the term up the
  tree as the *same* preference rather than as the alternating artefact the
  draw constant produces (``tests/search/unit/test_draw_utility_parity.py``
  pins that artefact; this pins its absence here).
* **monotone in ``m``, centred on the root.** Shorter than the root expected is
  better for whoever is winning and worse for whoever is losing, zero at the
  root's own estimate, saturating one ``slope_plies`` out.
* **skipped, not zeroed, at ``lambda == 0``.** The disabled path does not read
  the context, does not read the head, and returns the float
  :func:`~komugi.selfplay.inference.wdl_utility` returned -- which is what makes
  the arena's and the anchor server's two-argument
  :func:`~komugi.selfplay.adapter.make_evaluator` call bit-identical to what it
  was (review F9). "Does not read the context" is asserted with a context whose
  every non-dunder read raises, because a test that merely checked the number
  could not tell a skipped term from a term that computed zero.

Everything here is arithmetic and one closure: no engine, no network, no search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.config import SelfPlayConfig
from komugi.engine.observation import MAX_TURNS
from komugi.search.state import Evaluator, SearchState, WdlEvaluator
from komugi.selfplay.adapter import (
    MlhUtility,
    SearchContext,
    _value_from_wdl,
    make_evaluator,
    make_wave_evaluator,
    mlh_utility_term,
)
from komugi.selfplay.inference import NetEval, SlotClient, WDL_SIZE, wdl_utility

_LAM: Final[float] = 0.05
_TAU0: Final[float] = 0.20
_TAU1: Final[float] = 0.40
_SLOPE: Final[int] = 32
"""The pre-registered parameters (design §3.4), restated rather than imported
from a config so a campaign that changes the default does not silently change
what these assertions are about."""

_MLH: Final[MlhUtility] = MlhUtility(
    lam=_LAM, tau0=_TAU0, tau1=_TAU1, slope_plies=_SLOPE
)

_OFF: Final[MlhUtility] = MlhUtility(
    lam=0.0, tau0=_TAU0, tau1=_TAU1, slope_plies=_SLOPE
)
"""The same lever at ``lambda = 0`` -- the shape review F9 insists must behave
exactly like ``None``, so both are asserted wherever "off" is."""

_M_ROOT: Final[float] = 100.0
"""The root's predicted plies-to-end in the tests below. 100 is the design's
own worked example (review F1) and leaves room on both sides of the 32-ply
saturation without running into the clip at 0 or at ``MAX_TURNS``."""

_ACTIONS: Final[int] = 4
"""A tiny action space: the policy vector is carried through untouched, so its
width is irrelevant to everything under test and a narrow one keeps the stubs
readable."""


def _wdl(v: float, draw: float = 0.1) -> tuple[float, float, float]:
    """Return a ``(w, d, l)`` triple with ``w - l == v`` and ``w + d + l == 1``.

    The gate reads ``w - l`` and nothing else, so `draw` is free; it is non-zero
    by default precisely so that a term which accidentally read ``d`` would show
    up rather than cancel. It is also capped at ``1 - |v|``, so ``v = +-1``
    gives a genuine one-hot rather than a triple with a negative entry.
    """
    mass = min(draw, 1.0 - abs(v))
    win = (1.0 - mass + v) / 2.0
    return (win, mass, win - v)


def _swap(wdl: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return `wdl` in the opponent's frame: win and loss exchanged, ``d`` fixed."""
    return (wdl[2], wdl[1], wdl[0])


def _fraction(plies: float) -> float:
    """Return `plies` on the head's own ``[0, 1]`` scale."""
    return plies / MAX_TURNS


def _term(
    v: float, plies: float, *, m_root: float = _M_ROOT, config: MlhUtility = _MLH
) -> float:
    """Return the term for a leaf reading `v` with `plies` left, in plies."""
    return mlh_utility_term(_wdl(v), _fraction(plies), m_root, config)


def _gate(v: float) -> float:
    """Return ``g(|v|)`` -- restated here, so the test does not import the code."""
    magnitude = abs(v)
    ramp = (magnitude - _TAU0) / (_TAU1 - _TAU0)
    return magnitude * min(max(ramp, 0.0), 1.0)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.0, 0.05, 0.1, 0.19, -0.05, -0.19])
@pytest.mark.parametrize("plies", [0.0, 1.0, 50.0, 100.0, 200.0, 512.0])
def test_below_tau0_the_term_is_exactly_zero_for_every_m(
    v: float, plies: float
) -> None:
    """``|v| < tau0`` is off, as an identity rather than as a small number.

    Both signs of `v` and both sides of the root are covered, because "off"
    has to mean off in the direction the term would otherwise push as well as
    in the direction it would pull.
    """
    value = _term(v, plies)
    assert value == 0.0
    assert not math.copysign(1.0, value) < 0.0, "an exact zero, not a negative one"


@pytest.mark.parametrize("v", [0.2, 0.3, 0.4, 0.6, 1.0])
def test_the_gate_ramps_linearly_between_tau0_and_tau1(v: float) -> None:
    """At full saturation the magnitude is ``lambda * g(|v|)``, and ``g`` is the ramp.

    ``tau0`` itself is on the closed side of the ramp and contributes ``0``:
    ``g(tau0) = tau0 * 0``. That is deliberate -- the ramp exists so that two
    siblings straddling the threshold do not differ by a step of
    ``lambda * tau0`` (design §3.4).
    """
    saturated = _term(v, _M_ROOT - _SLOPE)
    assert saturated == pytest.approx(_LAM * _gate(v), abs=1e-12)
    assert _term(_TAU0, _M_ROOT - _SLOPE) == 0.0


# ---------------------------------------------------------------------------
# oddness -- the property negamax needs (design §3.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.25, 0.5, 0.6, 0.9, 1.0])
@pytest.mark.parametrize("plies", [0.0, 40.0, 68.0, 100.0, 132.0, 300.0])
def test_the_term_is_odd_under_the_frame_swap(v: float, plies: float) -> None:
    """``T(swap(wdl), m) == -T(wdl, m)``, to 1e-12 and exactly in the sign.

    The whole of design §3.2 executed on one leaf: ``v`` negates, ``|v|`` and
    therefore ``g`` do not, ``f(m)`` does not, and ``sign(v)`` does -- so the
    product negates. An **even** term (the draw constant ``s*d``) would come
    back equal instead of opposite, which is the failure this asserts against.
    """
    forward = mlh_utility_term(_wdl(v), _fraction(plies), _M_ROOT, _MLH)
    reversed_frame = mlh_utility_term(
        _swap(_wdl(v)), _fraction(plies), _M_ROOT, _MLH
    )

    assert reversed_frame == pytest.approx(-forward, abs=1e-12)
    assert math.copysign(1.0, reversed_frame) == -math.copysign(1.0, forward), (
        "the sign must flip exactly, not merely nearly: a term that agreed in "
        "sign is an even term and oscillates up the tree"
    )


def test_the_winner_prefers_the_short_line_and_the_loser_the_long_one() -> None:
    """The direction the whole design is for, on one pair of leaves.

    A leaf 40 plies shorter than the root expected is worth **more** to a mover
    who is winning there and **less** to one who is losing. Those are the same
    statement in two frames, which is exactly what oddness means.
    """
    short_win = _term(0.6, _M_ROOT - 40.0)
    long_win = _term(0.6, _M_ROOT + 40.0)
    short_loss = _term(-0.6, _M_ROOT - 40.0)
    long_loss = _term(-0.6, _M_ROOT + 40.0)

    assert short_win > 0.0 > long_win
    assert long_loss > 0.0 > short_loss


# ---------------------------------------------------------------------------
# f(m): monotone, centred on the root, saturating
# ---------------------------------------------------------------------------


def test_the_term_is_monotone_in_m_and_zero_at_the_root() -> None:
    """Non-increasing in ``m`` for ``v > 0``, non-decreasing for ``v < 0``, 0 at ``m_root``.

    ``m`` runs from a mate-in-two to a full cap, so the sequence crosses both
    saturation shoulders and the root itself.
    """
    grid = [0.0, 10.0, 68.0, 90.0, 100.0, 110.0, 132.0, 300.0, 512.0]

    winning = [_term(0.6, plies) for plies in grid]
    losing = [_term(-0.6, plies) for plies in grid]

    assert all(
        later <= earlier for earlier, later in zip(winning, winning[1:])
    ), f"a winning leaf's term rose with m: {winning}"
    assert all(
        later >= earlier for earlier, later in zip(losing, losing[1:])
    ), f"a losing leaf's term fell with m: {losing}"
    assert _term(0.6, _M_ROOT) == 0.0
    assert _term(-0.6, _M_ROOT) == 0.0


@pytest.mark.parametrize("beyond", [0.0, 1.0, 8.0, 100.0])
def test_it_saturates_at_lambda_times_the_gate(beyond: float) -> None:
    """One ``slope_plies`` out, ``f`` is clipped and the term stops growing."""
    ceiling = _LAM * _gate(0.6)

    assert _term(0.6, _M_ROOT - _SLOPE - beyond) == pytest.approx(ceiling, abs=1e-12)
    assert _term(0.6, _M_ROOT + _SLOPE + beyond) == pytest.approx(-ceiling, abs=1e-12)


def test_moves_left_is_clipped_before_it_becomes_plies() -> None:
    """A head answering outside ``[0, 1]`` is clipped, not believed.

    The moves-left head is a linear regression with no output squashing, so an
    untrained one answers ``-0.3`` and ``1.4``; those must read as ``0`` and
    ``MAX_TURNS`` plies rather than as negative or super-cap lengths.
    """
    assert mlh_utility_term(_wdl(0.6), -0.3, _M_ROOT, _MLH) == pytest.approx(
        mlh_utility_term(_wdl(0.6), 0.0, _M_ROOT, _MLH), abs=1e-12
    )
    assert mlh_utility_term(_wdl(0.6), 1.4, _M_ROOT, _MLH) == pytest.approx(
        mlh_utility_term(_wdl(0.6), 1.0, _M_ROOT, _MLH), abs=1e-12
    )


# ---------------------------------------------------------------------------
# the composed leaf value
# ---------------------------------------------------------------------------


def _leaf_value(
    wdl: tuple[float, float, float],
    *,
    draw_score: float = 0.0,
    plies: float = 0.0,
    mlh: MlhUtility | None = None,
    context: SearchContext | None = None,
) -> float:
    """Return what the shape seam makes of one answer."""
    policy = np.zeros(_ACTIONS, dtype=np.float32)
    # Three-tuple since KOM-173: the seam hands the leaf's own (w, d, l) back
    # beside the utility. The vector is what this helper's callers are NOT
    # about -- they measure the scalar -- so it is dropped here, and
    # `test_the_seam_hands_back_the_leaf_wdl_unchanged` covers it directly.
    _logits, value, _vector = _value_from_wdl(
        policy,
        np.asarray(wdl, dtype=np.float32),
        draw_score,
        moves_left=_fraction(plies),
        mlh=mlh,
        context=context,
    )
    return value


@pytest.mark.parametrize("v", [0.6, 0.9, 0.98, 1.0, -0.9, -1.0])
@pytest.mark.parametrize("plies", [0.0, 60.0, 100.0, 200.0, 512.0])
def test_the_composed_leaf_utility_stays_inside_the_value_range(
    v: float, plies: float
) -> None:
    """``|u| <= 1`` after the clamp, even for a near-terminal leaf.

    Review F11's case: ``v = 0.99`` with a fully saturated ``f`` composes to
    ``1.04`` before the clamp, which would put a leaf above a real mate. The
    clamp is what keeps the value convention intact; the tie it creates with a
    terminal is broken by the prior, as F11 records.
    """
    context = SearchContext()
    context.m_root = _M_ROOT

    value = _leaf_value(_wdl(v, draw=0.0), plies=plies, mlh=_MLH, context=context)

    assert -1.0 <= value <= 1.0


def test_a_near_terminal_leaf_is_clamped_rather_than_rescaled() -> None:
    """The clamp fires where review F11 says it does, and only there."""
    context = SearchContext()
    context.m_root = _M_ROOT

    clamped = _leaf_value(
        _wdl(0.99, draw=0.0), plies=0.0, mlh=_MLH, context=context
    )
    unclamped = _leaf_value(
        _wdl(0.6, draw=0.0), plies=0.0, mlh=_MLH, context=context
    )

    assert clamped == 1.0
    # 1e-6, not tighter: `wdl_utility` collapses the triple in float32, so the
    # composed number carries float32's ~6e-8 on 0.6 and the term carries it
    # again through the gate.
    assert unclamped == pytest.approx(0.6 + _LAM * _gate(0.6), abs=1e-6)


# ---------------------------------------------------------------------------
# lambda = 0: skipped, not zeroed
# ---------------------------------------------------------------------------


class _HostileContext(SearchContext):
    """A context whose reference cannot be read without failing the test.

    The instrument review F9's claim needs: "the term is skipped" is a claim
    about which lines run, and a value assertion cannot distinguish a skipped
    term from one that computed ``0.0`` and was added. Deliberately a real
    :class:`~komugi.selfplay.adapter.SearchContext` subclass -- the seam is
    ``@jaxtyped`` under ``KOMUGI_TYPECHECK=1`` and a duck-typed stand-in would
    be refused by beartype before the code under test ran.

    **Dunder reads pass through** (KOM-171, KOM-159b review F1). Being a real
    subclass is only half of what the typechecked seam needs: beartype's
    ``isinstance`` machinery reads ``__class__`` off the *instance* before the
    body under test runs, so a trap that fired on every name turned
    ``KOMUGI_TYPECHECK=1`` into ten failures about the instrument rather than
    about the code. The claim this class exists to make is unweakened by the
    carve-out: ``m_root`` and ``reset`` -- the only names the leaf seam could
    reach for, and the only ones the term's arithmetic needs -- are not
    dunders, so a term that ran would still be caught here. Nothing in
    :mod:`komugi.selfplay.adapter` reads a dunder off a context.
    """

    def __getattribute__(self, name: str) -> object:
        """Fail on any **non-dunder** attribute read, naming the attempt.

        ``__getattribute__`` rather than a property on ``m_root``: the base
        declares that name in ``__slots__``, and the point is to catch a read
        of anything the seam might reach for -- ``reset`` included -- not just
        of the one field today's code happens to touch. Construction still
        works, because ``__init__``'s ``self.m_root = None`` is a *write*.
        """
        if name.startswith("__"):
            # The runtime type machinery's own reads, not the code under test:
            # `isinstance` wants `__class__`, and refusing it fails the test
            # before `_value_from_wdl` is entered (see the class docstring).
            return object.__getattribute__(self, name)
        raise AssertionError(
            f"the leaf seam read SearchContext.{name} at lambda = 0; the term "
            f"is supposed to be skipped entirely (KOM-159 design review F9)"
        )


@pytest.mark.parametrize("draw_score", [0.0, 0.2, -0.2])
@pytest.mark.parametrize(
    "wdl", [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (0.34, 0.33, 0.33)]
)
def test_at_lambda_zero_the_leaf_value_is_wdl_utility_bit_for_bit(
    wdl: tuple[float, float, float], draw_score: float
) -> None:
    """``None`` and ``lam == 0.0`` both return exactly what the seam always did.

    ``==`` on the float, not ``approx``: the claim (T2/T6) is bit-identity, and
    the only way to have it is for the disabled path to not execute an addition
    at all. The hostile context proves it did not.
    """
    expected = float(
        wdl_utility(np.asarray([wdl], dtype=np.float32), draw_score)[0]
    )

    assert _leaf_value(wdl, draw_score=draw_score, plies=300.0) == expected
    assert (
        _leaf_value(
            wdl,
            draw_score=draw_score,
            plies=300.0,
            mlh=_OFF,
            context=_HostileContext(),
        )
        == expected
    )
    assert (
        _leaf_value(wdl, draw_score=draw_score, plies=300.0, mlh=None, context=None)
        == expected
    )


# ---------------------------------------------------------------------------
# the reference, and the refusal to run without one
# ---------------------------------------------------------------------------


class _StubState:
    """The smallest thing the scalar seam will accept: an evaluable position.

    Its ``observation`` carries nothing the stub evaluator reads -- the answers
    are scripted in call order -- so it exists only to satisfy the cast the
    adapter makes.
    """

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return an all-legal mask of the toy width."""
        return np.ones(_ACTIONS, dtype=np.bool_)

    def play(self, action: int) -> None:
        """Unreachable: nothing here searches."""
        raise AssertionError(f"the leaf seam played action {action}")

    def clone(self) -> _StubState:
        """Return an independent copy; there is no state to share."""
        return _StubState()

    def is_terminal(self) -> bool:
        """Return ``False``: a terminal never reaches the evaluator."""
        return False

    def terminal_value(self) -> float:
        """Unreachable for the same reason."""
        raise AssertionError("the leaf seam asked a non-terminal for its result")

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Unreachable: KOM-173 reads this only at a terminal leaf."""
        raise AssertionError("the leaf seam asked a non-terminal for its one-hot")

    def to_play(self) -> int:
        """Return the mover; the seam does not read it."""
        return 0

    def observation(self) -> npt.NDArray[np.float32]:
        """Return a one-element encoding -- the stub ignores its content."""
        return np.zeros((1, 1, 1), dtype=np.float32)


class _ScriptedEvaluator:
    """Answers a scripted ``(wdl, moves_left)`` per call, in order."""

    def __init__(
        self, answers: Sequence[tuple[tuple[float, float, float], float]]
    ) -> None:
        self._answers = list(answers)
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return the next scripted answer as a batch of one."""
        assert obs.shape[0] == 1, "the scalar seam submits one position"
        assert legal.shape[0] == 1
        wdl, moves_left = self._answers[self.calls]
        self.calls += 1
        return NetEval(
            policy_logits=np.zeros((1, _ACTIONS), dtype=np.float32),
            wdl=np.asarray([wdl], dtype=np.float32).reshape(1, WDL_SIZE),
            moves_left=np.asarray([moves_left], dtype=np.float32),
        )


def test_the_first_evaluation_of_a_search_sets_the_root_and_adds_nothing() -> None:
    """The root centres the term and is not moved by it (design §3.3, T5's premise).

    Two evaluations through one closure: the first is the root -- it fixes
    ``m_root`` and returns the plain ``wdl`` utility -- and the second is a leaf
    40 plies shorter, which is the root's own winning preference and reads
    ``+lambda*g*40/32`` clipped to ``+lambda*g``.
    """
    root_plies, leaf_plies = 100.0, 60.0
    evaluator = _ScriptedEvaluator(
        [
            (_wdl(0.6, draw=0.0), _fraction(root_plies)),
            (_wdl(0.6, draw=0.0), _fraction(leaf_plies)),
        ]
    )
    context = SearchContext()
    seam = make_evaluator(evaluator, 0.0, _MLH, context)

    _root_policy, root_value = seam(_StubState())
    assert context.m_root == pytest.approx(root_plies, abs=1e-6)
    assert root_value == pytest.approx(0.6, abs=1e-6), (
        "the root's own delta against itself is zero, so it carries no term"
    )

    _leaf_policy, leaf_value = seam(_StubState())
    assert leaf_value == pytest.approx(0.6 + _LAM * _gate(0.6), abs=1e-6)

    # And reset really does forget: the next search's first leaf is its root.
    context.reset()
    assert context.m_root is None


def test_at_lambda_zero_the_seam_never_reads_the_context() -> None:
    """A whole evaluation through the closure, with the reference booby-trapped."""
    evaluator = _ScriptedEvaluator([((0.8, 0.1, 0.1), 0.5)])
    seam = make_evaluator(evaluator, 0.2, _OFF, _HostileContext())

    _policy, value = seam(_StubState())

    assert value == pytest.approx(0.8 - 0.1 + 0.1 * 0.2, abs=1e-6)


def test_a_positive_lambda_with_no_context_is_refused_at_construction() -> None:
    """Both factories refuse, and they refuse **when built** (design R9).

    Not at the first leaf: a search that discovered halfway through that it had
    no reference would already have written half a tree centred on nothing. The
    two-argument call every arena, gauntlet, ladder and anchor search makes is
    checked in the same breath, because an unconditional refusal would break all
    of them (review F9).
    """
    evaluator = _ScriptedEvaluator([((0.8, 0.1, 0.1), 0.5)])

    with pytest.raises(ValueError, match="no SearchContext"):
        make_evaluator(evaluator, 0.0, _MLH, None)
    selfplay = SelfPlayConfig(workers=1, process_batch_size=1)
    with pytest.raises(ValueError, match="no SearchContext"):
        make_wave_evaluator(cast(SlotClient, evaluator), selfplay, _MLH, None)

    # The calls that must keep working, untouched.
    assert make_evaluator(evaluator, 0.0) is not None
    assert make_evaluator(evaluator, 0.0, _OFF, None) is not None
    assert make_evaluator(evaluator, 0.0, None, None) is not None


def test_from_selfplay_reads_the_four_fields_and_nothing_else() -> None:
    """:meth:`MlhUtility.from_selfplay` is the one place the config is read."""
    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        draw_score=0.0,
        mlh_utility_lambda=0.10,
        mlh_gate_tau0=0.15,
        mlh_gate_tau1=0.55,
        mlh_slope_plies=24,
    )

    assert MlhUtility.from_selfplay(config) == MlhUtility(
        lam=0.10, tau0=0.15, tau1=0.55, slope_plies=24
    )
    assert MlhUtility.from_selfplay(
        SelfPlayConfig(workers=1, process_batch_size=1)
    ) == MlhUtility(lam=0.0, tau0=0.20, tau1=0.40, slope_plies=32)


def test_the_search_state_protocol_is_satisfied_by_the_stub() -> None:
    """Guard the stub itself: a drifting protocol must fail here, loudly."""
    state: SearchState = _StubState()
    assert state.to_play() == 0


# ---------------------------------------------------------------------------
# KOM-173 T6: the vector the seam hands back beside the utility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("draw_score", [0.0, 0.2, -0.2])
def test_the_seam_hands_back_the_leaf_wdl_unchanged(draw_score: float) -> None:
    """The third return is the head's own ``(w, d, l)``, untouched by any utility.

    Not the utility's ingredients re-derived, and not a contempt-weighted
    version of them: the array the seam was handed. ``draw_score`` sweeps because
    the scalar it moves and the vector it must not move are produced by the same
    call -- a seam that folded ``s*d`` into the vector would pass at ``s = 0``
    and fail here at ``+-0.2``.
    """
    wdl = (0.5, 0.3, 0.2)
    policy = np.zeros(_ACTIONS, dtype=np.float32)
    vector_in = np.asarray(wdl, dtype=np.float32)

    logits, value, vector_out = _value_from_wdl(policy, vector_in, draw_score)

    assert logits is policy
    assert np.array_equal(vector_out, np.asarray(wdl, dtype=np.float32))
    assert value == pytest.approx(0.5 - 0.2 + 0.3 * draw_score, abs=1e-6)


def test_the_vector_is_the_pre_term_head_answer_at_positive_lambda() -> None:
    """At ``lambda > 0`` the scalar carries the term and the vector does not.

    The conversion term is the one thing that may enter a leaf's *scalar*
    (KOM-159 D6); KOM-173 D6 is its mirror image, that nothing may enter the
    vector. Here the leaf is 40 plies shorter than the root expected, so the
    scalar picks up ``+lambda*g(0.6)`` while the vector is still exactly what
    the head said.
    """
    context = SearchContext()
    context.m_root = _M_ROOT
    wdl = _wdl(0.6, draw=0.0)
    policy = np.zeros(_ACTIONS, dtype=np.float32)

    _logits, value, vector = _value_from_wdl(
        policy,
        np.asarray(wdl, dtype=np.float32),
        0.0,
        moves_left=_fraction(_M_ROOT - 40.0),
        mlh=_MLH,
        context=context,
    )

    assert value == pytest.approx(0.6 + _LAM * _gate(0.6), abs=1e-6)
    assert np.array_equal(vector, np.asarray(wdl, dtype=np.float32))
    assert float(vector[0] - vector[2]) == pytest.approx(0.6, abs=1e-6)


def test_with_wdl_is_the_only_difference_between_the_two_seam_shapes() -> None:
    """``with_wdl`` adds a third element and changes nothing else.

    The two closures are built over the same scripted evaluator and asked about
    the same position, so the policy and the utility are comparable directly:
    ``==`` on the float, because the vector travels beside the scalar rather than
    through it. The default is the two-tuple shape, which is what keeps the
    arena, the gauntlet, the ladder and the anchor server on the seam they had.
    """
    answers = [((0.55, 0.25, 0.2), 0.5), ((0.55, 0.25, 0.2), 0.5)]
    plain = make_evaluator(_ScriptedEvaluator(answers[:1]), 0.2)
    carrying = make_evaluator(
        _ScriptedEvaluator(answers[1:]), 0.2, with_wdl=True
    )

    plain_answer = plain(_StubState())
    carrying_answer = carrying(_StubState())

    assert len(plain_answer) == 2
    assert len(carrying_answer) == 3
    assert np.array_equal(carrying_answer[0], plain_answer[0])
    assert carrying_answer[1] == plain_answer[1]
    # Widened to read the third element: the answer's static type is the union
    # of the two shapes, and the length is what tells them apart -- which is
    # exactly how ``GumbelSearch`` reads it.
    carrying_fields: tuple[object, ...] = carrying_answer
    np.testing.assert_allclose(
        np.asarray(carrying_fields[2], dtype=np.float64),
        np.asarray((0.55, 0.25, 0.2)),
        rtol=0.0,
        atol=1e-7,
    )


def _seam_for(
    evaluator: _ScriptedEvaluator, *, with_wdl: bool
) -> Evaluator | WdlEvaluator:
    """Build a scalar seam from a flag whose static type is a plain ``bool``.

    The static half of KOM-173 (i) review F2, and it is the *annotation* that is
    under test: `with_wdl` here is a ``bool``, not a ``Literal``, so this call
    matches only the third :func:`make_evaluator` overload. Before that overload
    existed the gate's ``mypy --strict`` refused this line with ``No overload
    variant of "make_evaluator" matches argument types ... [call-overload]``,
    which is a failure the runtime assertions below could never have produced.
    """
    return make_evaluator(evaluator, 0.2, with_wdl=with_wdl)


@pytest.mark.parametrize("with_wdl", [False, True], ids=["scalar", "wdl"])
def test_make_evaluator_accepts_a_flag_whose_type_is_bool(with_wdl: bool) -> None:
    """A runtime ``bool`` reveals the union; the two literals still narrow.

    A caller that reads the flag from a config field or a test parameter holds a
    ``bool``, and the union it gets back is narrowed the way the search narrows
    it -- by ``len`` on the answer. The two annotated assignments below are the
    other half of the claim: they fail the gate if the literal overloads ever
    regress to returning the union, which is what would make the third overload
    a weakening rather than an addition.
    """
    answers = [((0.55, 0.25, 0.2), 0.5)]
    seam = _seam_for(_ScriptedEvaluator(answers), with_wdl=with_wdl)

    answer = seam(_StubState())
    assert len(answer) == (3 if with_wdl else 2), answer

    literal_off: Evaluator = make_evaluator(_ScriptedEvaluator(answers), 0.2)
    literal_on: WdlEvaluator = make_evaluator(
        _ScriptedEvaluator(answers), 0.2, with_wdl=True
    )
    assert len(literal_off(_StubState())) == 2
    assert len(literal_on(_StubState())) == 3
