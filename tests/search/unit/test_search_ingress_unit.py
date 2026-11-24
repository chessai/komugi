"""KOM-179: what the Gumbel search refuses at its evaluator seam.

The review's numeric-ingress table (2026-09-12) found the seam checked the
*shape* of what a network answered with and almost nothing about the numbers in
it. Three holes, all with the same signature -- no local symptom, and a
plausible-looking :class:`~komugi.search.gumbel.SearchResult` handed to the
self-play recorder:

* a non-finite scalar ``value`` was ``float()``-ed through, entered
  ``child_value_sums``, and came back out as a ``NaN`` ``root_value`` beside an
  **all-NaN** policy target (witness: ``engine_search_probes.py gumbel``);
* :meth:`~komugi.search.gumbel.GumbelSearch._forced_move` did not run its single
  evaluation through the checks at all, so a ``(1,)`` logit vector was returned
  verbatim as ``root_prior_logits`` -- a field every caller reads as
  ``(num_actions,)``;
* a ``(w, d, l)`` that was not a distribution (negative mass, or unnormalised)
  was accepted, which quietly voids
  ``root_wdl[0] - root_wdl[2] == root_value``.

Every test here asserts the **ingress** error: a :class:`ValueError` naming the
evaluator and the number it returned, raised before any of that can be built.
ES-04, in the same table, is the config half -- a negative ``c_visit`` or
``c_scale`` reverses :func:`~komugi.search.gumbel._sigma`'s monotone Q transform
and is a different algorithm rather than a tuning choice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchState

#: Action-space width every stub in this file answers at.
NUM_ACTIONS = 4


class IngressState:
    """A two-ply toy: a fixed legal set, alternating movers, no rules.

    Deliberately the smallest state that reaches every path under test -- a
    branching root (three legal actions), a forced root (one), and a terminal
    leaf after `max_depth` plies.
    """

    def __init__(
        self,
        *,
        legal: tuple[int, ...] = (0, 1, 2),
        max_depth: int = 2,
        terminal: bool = False,
    ) -> None:
        self.legal = legal
        self.max_depth = max_depth
        self.terminal = terminal
        self.history: tuple[int, ...] = ()

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the fixed legal set, or nothing at all once terminal."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        if self.terminal or not self.is_terminal():
            for action in self.legal:
                mask[action] = True
        return mask

    def play(self, action: int) -> None:
        """Advance in place, refusing anything the mask did not offer."""
        if not bool(self.legal_mask()[action]):
            raise AssertionError(f"search played illegal action {action}")
        self.history = (*self.history, action)

    def clone(self) -> IngressState:
        """Return an independent copy -- `history` is a tuple, so sharing is safe."""
        copy = IngressState(
            legal=self.legal, max_depth=self.max_depth, terminal=self.terminal
        )
        copy.history = self.history
        return copy

    def is_terminal(self) -> bool:
        """Return whether the position is adjudicated."""
        return self.terminal or len(self.history) >= self.max_depth

    def terminal_value(self) -> float:
        """Return a value nothing else in the search could have produced."""
        return -0.5

    def terminal_wdl(self) -> npt.NDArray[np.float64]:
        """Return the vector form of that value: ``(0.25, 0, 0.75)``."""
        return np.array([0.25, 0.0, 0.75], dtype=np.float64)

    def to_play(self) -> int:
        """Return the mover, alternating every ply."""
        return len(self.history) % 2


def _logits(offset: float = 0.0) -> npt.NDArray[np.float32]:
    """Return a full-width, finite, non-uniform prior."""
    return np.array(
        [0.25 * action + offset for action in range(NUM_ACTIONS)], dtype=np.float32
    )


class ScalarStub:
    """A two-tuple evaluator that goes non-finite on a chosen call.

    `nan_on` is one-based and counts calls to *this* object, so a test can put
    the ``NaN`` on the root (call 1) or on a leaf the branching schedule
    expands (call 2) without the two needing different stubs.
    """

    def __init__(self, *, nan_on: int) -> None:
        self.nan_on = nan_on
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, value)`, the value non-finite on call `nan_on`."""
        self.calls += 1
        value = math.nan if self.calls == self.nan_on else 0.25
        return _logits(float(self.calls)), value


class CleanStub:
    """A two-tuple evaluator whose answers are all finite and full width."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return `(logits, value)`, both unremarkable."""
        self.calls += 1
        return _logits(float(self.calls)), 0.25


class NanWaveStub:
    """A wave evaluator that answers the first row of every wave with ``NaN``.

    The wave seam is where a non-finite value is *least* visible: the answer is
    one row of a batch, it is backed up immediately, and by the time the search
    returns there is no row left to point at.
    """

    def __init__(self) -> None:
        self.waves = 0

    def __call__(
        self, states: Sequence[SearchState]
    ) -> list[tuple[npt.NDArray[np.float32], float]]:
        """Return one answer per state, the first of them non-finite."""
        self.waves += 1
        return [
            (_logits(float(row)), math.nan if row == 0 else 0.25)
            for row in range(len(states))
        ]


class WdlStub:
    """A three-tuple evaluator returning a fixed `wdl` at every position.

    The vector is handed over verbatim, including a `wdl` that is not a
    distribution: refusing that is the point of the tests that use it.
    """

    def __init__(self, wdl: Sequence[float]) -> None:
        self.wdl = tuple(float(entry) for entry in wdl)
        self.calls = 0

    def __call__(
        self, state: SearchState
    ) -> tuple[npt.NDArray[np.float32], float, npt.NDArray[np.float64]]:
        """Return `(logits, value, wdl)` for `state`."""
        self.calls += 1
        return (
            _logits(float(self.calls)),
            0.25,
            np.array(self.wdl, dtype=np.float64),
        )


class NarrowLogitStub:
    """A two-tuple evaluator answering with a **width-one** logit vector.

    The plausible way to get one: an evaluator that returns logits over the
    *legal* actions rather than over the whole action space. At a forced
    position that is a ``(1,)`` array, and until KOM-179 it was handed straight
    back as :attr:`~komugi.search.gumbel.SearchResult.root_prior_logits`.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return a one-element prior and a finite value."""
        self.calls += 1
        return np.array([0.5], dtype=np.float32), 0.25


class NanLogitStub:
    """A two-tuple evaluator whose prior is full width but not finite.

    Separate from :class:`NarrowLogitStub` so the two forced-path holes are
    pinned separately: the shape check fires first, so a width-one NaN vector
    would never reach the finiteness check and would not test it.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: SearchState) -> tuple[npt.NDArray[np.float32], float]:
        """Return a full-width prior with a ``NaN`` on the forced action."""
        self.calls += 1
        logits = _logits()
        logits[2] = np.float32(math.nan)
        return logits, 0.25


def _config(*, m: int = 3, n: int = 8, seed: int = 11) -> GumbelConfig:
    """Return a config over :data:`NUM_ACTIONS` with the defaults under test."""
    return GumbelConfig(num_actions=NUM_ACTIONS, m=m, n=n, seed=seed)


def _config_with(field: str, value: float) -> GumbelConfig:
    """Return a config whose `field` is `value`, everything else default.

    Spelled out per field rather than through ``**kwargs`` so the call stays
    typed: a dict of widened values would hide exactly the kind of argument
    mistake the tests below are asserting about.
    """
    if field == "c_visit":
        return GumbelConfig(num_actions=NUM_ACTIONS, m=3, n=8, seed=1, c_visit=value)
    if field == "c_scale":
        return GumbelConfig(num_actions=NUM_ACTIONS, m=3, n=8, seed=1, c_scale=value)
    raise AssertionError(f"unknown field {field!r}")


# --------------------------------------------------------------------------
# The scalar value: one checkpoint, three paths through it
# --------------------------------------------------------------------------


def test_a_nan_scalar_value_is_refused_on_the_branching_path() -> None:
    """A leaf's ``NaN`` value raises at ingress, naming the evaluator.

    The root answers finitely and the *second* evaluation -- a leaf the
    sequential-halving schedule expands -- does not, so this pins the leaf seam
    rather than only the root one. Before KOM-179 the search completed and
    returned ``root_value = nan`` with an all-NaN ``policy_target``.
    """
    evaluator = ScalarStub(nan_on=2)
    search = GumbelSearch(evaluator, _config())

    with pytest.raises(ValueError, match="evaluator returned a non-finite value"):
        search.search(IngressState())

    # The root really was evaluated cleanly first: this is the leaf checkpoint.
    assert evaluator.calls == 2


def test_a_nan_scalar_value_is_refused_on_the_root_of_the_branching_path() -> None:
    """The root's own ``NaN`` value is refused before a simulation runs."""
    evaluator = ScalarStub(nan_on=1)
    search = GumbelSearch(evaluator, _config())

    with pytest.raises(ValueError, match="evaluator returned a non-finite value"):
        search.search(IngressState())

    assert evaluator.calls == 1


def test_a_nan_scalar_value_is_refused_on_the_wave_path() -> None:
    """A wave answer goes through the same checkpoint a single answer does.

    The batched seam is a second entrance to the tree, and the property that
    makes it safe is that it is not a second *checkpoint*: both paths meet in
    ``_evaluated_node``, so this raises the identical error.
    """
    wave = NanWaveStub()
    search = GumbelSearch(CleanStub(), _config(), wave_evaluator=wave)

    with pytest.raises(ValueError, match="evaluator returned a non-finite value"):
        search.search(IngressState())

    assert wave.waves == 1, "the wave seam was never reached"


def test_a_nan_scalar_value_is_refused_on_the_forced_path() -> None:
    """A forced position's single evaluation is checked like any other.

    One legal action, so no simulation runs and the whole result is built from
    that one answer -- which is exactly why an unchecked ``NaN`` here reached
    ``root_value`` with nothing in between to dilute it. No ``SearchResult`` is
    produced at all now.
    """
    evaluator = ScalarStub(nan_on=1)
    search = GumbelSearch(evaluator, _config())

    with pytest.raises(ValueError, match="evaluator returned a non-finite value"):
        search.search(IngressState(legal=(2,)))

    assert evaluator.calls == 1, "the forced path still evaluates exactly once"


def test_a_nan_logit_vector_is_refused_on_the_forced_path() -> None:
    """The forced path's logits are finiteness-checked too, at width one.

    ``_forced_move`` used to ``np.asarray`` the answer and return it, so a
    ``NaN`` prior reached ``root_prior_logits`` and, through it, self-play's
    ``policy_kl``.
    """
    search = GumbelSearch(NanLogitStub(), _config())

    with pytest.raises(ValueError, match="non-finite logits"):
        search.search(IngressState(legal=(2,)))


def test_a_width_one_logit_vector_is_refused_on_the_forced_path() -> None:
    """A ``(1,)`` prior at a forced position is refused, naming the shape.

    Before the fix this search **returned**, with
    ``result.root_prior_logits.shape == (1,)`` -- a field documented and
    consumed as ``(num_actions,)``, and the source of self-play's ``policy_kl``.
    The mis-shaped array simply propagated.
    """
    search = GumbelSearch(NarrowLogitStub(), _config())

    with pytest.raises(ValueError, match=r"logits of shape \(1,\)"):
        search.search(IngressState(legal=(2,)))


def test_a_forced_terminal_root_checks_its_evaluation_too() -> None:
    """The adjudicated forced branch runs the same checkpoint.

    ``root_value`` comes from the game here, so a ``NaN`` from the network
    would never show up in it -- but the same evaluation still supplies
    ``root_prior_logits``, and a ``(1,)`` prior on this branch is as wrong as
    on the other.
    """
    search = GumbelSearch(NarrowLogitStub(), _config())

    with pytest.raises(ValueError, match=r"logits of shape \(1,\)"):
        search.search(IngressState(legal=(2,), terminal=True))


# --------------------------------------------------------------------------
# The (w, d, l) vector on the three-tuple seam
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wdl", "message"),
    [
        ((math.nan, 0.2, 0.3), "must be finite"),
        ((math.inf, 0.2, 0.3), "must be finite"),
        ((-0.1, 0.2, 0.9), "no negative entry"),
        ((0.5, 0.5, 0.5), "must sum to 1"),
        ((0.2, 0.2, 0.2), "must sum to 1"),
    ],
)
def test_a_wdl_that_is_not_a_distribution_is_refused(
    wdl: tuple[float, float, float], message: str
) -> None:
    """The 3-tuple seam admits probability vectors and nothing else.

    ``w - l`` is a number for any triple, so an unnormalised or signed vector
    produces readouts that look like probabilities, fail the KOM-173 identity
    ``root_wdl[0] - root_wdl[2] == root_value``, and name nothing.
    """
    search = GumbelSearch(WdlStub(wdl), _config())

    with pytest.raises(ValueError, match=message) as raised:
        search.search(IngressState())

    assert "wdl" in str(raised.value), "the message must name the vector"


def test_a_wdl_off_unit_mass_within_tolerance_is_accepted() -> None:
    """A float32 head's rounding is not an error: 1e-7 of drift passes.

    The bound is the tolerance's other half. Without it the check would be a
    ban on float32 softmax outputs rather than on unnormalised ones.
    """
    search = GumbelSearch(WdlStub((0.5 + 1e-7, 0.2, 0.3)), _config())

    result = search.search(IngressState())

    assert math.isfinite(float(result.root_value))
    assert bool(np.isfinite(result.root_wdl).all())


def test_a_nonsimplex_wdl_is_refused_on_the_forced_path() -> None:
    """The forced path validates its vector through the same checkpoint."""
    search = GumbelSearch(WdlStub((0.5, 0.5, 0.5)), _config())

    with pytest.raises(ValueError, match="must sum to 1"):
        search.search(IngressState(legal=(2,)))


# --------------------------------------------------------------------------
# ES-04: the Q transform's coefficients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["c_visit", "c_scale"])
@pytest.mark.parametrize("value", [-1.0, -1e-12])
def test_negative_q_transform_coefficients_are_rejected(
    field: str, value: float
) -> None:
    """ES-04: a negative coefficient reverses ``_sigma``, so it is not a config.

    ``sigma(q) = (c_visit + max_visit_count) * c_scale * q`` is monotone
    **increasing** in ``q`` for non-negative coefficients, which is the property
    the improved policy is built on. At ``c_scale = -1`` it is decreasing: the
    search ranks the child it scored worst highest and returns a confident
    target for it, with every shape and every sum still correct.
    """
    with pytest.raises(ValueError, match=field):
        _config_with(field, value)


@pytest.mark.parametrize("field", ["c_visit", "c_scale"])
def test_zero_q_transform_coefficients_stay_legal(field: str) -> None:
    """Zero flattens the Q term rather than inverting it, and stays accepted.

    ``tests/search/unit/test_gumbel_unit.py`` searches with ``c_scale=0.0``
    deliberately -- "ignore the Q term" is a statable configuration -- so the
    ES-04 bound has to be at zero and not just below the default.
    """
    config = _config_with(field, 0.0)

    assert getattr(config, field) == 0.0
