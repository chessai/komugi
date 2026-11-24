"""A9 and A14: the read-time collapse, and the packed second moment.

Three pieces of arithmetic that a shape assertion cannot check, so each is
pinned against an independent reference.

* **A9** -- ``node.utility`` is a float64 re-derivation of what
  ``inference.wdl_utility`` computes. It is re-derived rather than delegated
  because that function is float32-in/float32-out and demands an ``(N, 3)``
  input, so routing every child read of every playout through it would
  downcast every utility and force a reshape allocation in the hottest loop in
  the search. The two therefore *can* drift, and this is the only thing that
  stops them.
* **T-MOMENT** -- the packed ``(6,)`` quadratic form against a dense
  ``q @ M @ q``. A transposed index or a dropped factor of two here produces a
  variance that is merely *wrong* rather than obviously broken.
* **T-MOMENT-b** -- the variance clamp. Cancellation between two nearly equal
  large numbers can leave the difference slightly negative on a converged
  edge, and ``sqrt`` of that is ``nan``, which propagates silently into a move
  choice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import (
    MOMENT_SIZE,
    MatchNode,
    legal_hash_of,
    pack_outer,
    quadratic_form,
    utility,
    utility_reference,
)
from komugi.search.match.select import child_variances, lcb_scores

#: How many random pairs A9 is asserted over.
SAMPLES = 1000

#: A9's tolerance. float32 error on ``w - l + c*d`` for values in ``[-1, 1]``
#: is a few times 1.2e-7, so 1e-6 is loose enough to be stable and tight
#: enough to catch a sign, a slot swap, or a missing draw term.
COLLAPSE_TOLERANCE = 1e-6


def random_wdl(
    rng: np.random.Generator, count: int
) -> npt.NDArray[np.float64]:
    """Return `count` random WDL triples, each summing to exactly 1."""
    raw = rng.random((count, 3)) + 1e-3
    return np.asarray(raw / raw.sum(axis=1, keepdims=True), dtype=np.float64)


def dense_moment(wdl: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the dense ``3x3`` outer product, the reference for the packing."""
    return np.outer(wdl, wdl)


def test_the_collapse_agrees_with_production() -> None:
    """A9: ``utility(x, c)`` matches ``wdl_utility`` within ``1e-6``."""
    rng = np.random.default_rng(20260904)
    triples = random_wdl(rng, SAMPLES)
    contempts = rng.uniform(-1.0, 1.0, size=SAMPLES)
    worst = 0.0
    for index in range(SAMPLES):
        triple = triples[index]
        contempt = float(contempts[index])
        ours = utility(triple, contempt)
        theirs = utility_reference(triple, contempt)
        worst = max(worst, abs(ours - theirs))
    assert worst < COLLAPSE_TOLERANCE


def test_the_collapse_is_the_formula_it_claims_to_be() -> None:
    """A9: and it is ``w - l + c*d``, not merely "close to something"."""
    assert utility(np.array([1.0, 0.0, 0.0]), 0.5) == 1.0
    assert utility(np.array([0.0, 0.0, 1.0]), 0.5) == -1.0
    assert utility(np.array([0.0, 1.0, 0.0]), 0.5) == 0.5
    assert utility(np.array([0.0, 1.0, 0.0]), 0.0) == 0.0
    assert utility(np.array([0.25, 0.5, 0.25]), -0.4) == -0.2


def test_the_packed_quadratic_form_matches_a_dense_reference() -> None:
    """T-MOMENT: packed ``(ww, wd, wl, dd, dl, ll)`` equals ``q' M q``."""
    rng = np.random.default_rng(7)
    triples = random_wdl(rng, SAMPLES)
    weights = rng.uniform(0.25, 4.0, size=SAMPLES)
    contempts = rng.uniform(-1.0, 1.0, size=SAMPLES)
    worst = 0.0
    for index in range(SAMPLES):
        triple = triples[index]
        weight = float(weights[index])
        contempt = float(contempts[index])
        packed = (weight * pack_outer(triple)).reshape(1, MOMENT_SIZE)
        ours = float(quadratic_form(packed, contempt)[0])
        q = np.array([1.0, contempt, -1.0], dtype=np.float64)
        theirs = float(weight * (q @ dense_moment(triple) @ q))
        worst = max(worst, abs(ours - theirs))
    assert worst < 1e-12


def test_the_packing_order_is_the_documented_one() -> None:
    """T-MOMENT: the six slots, named, so a transposition is visible."""
    packed = pack_outer(np.array([2.0, 3.0, 5.0]))
    np.testing.assert_array_equal(
        packed, np.array([4.0, 6.0, 10.0, 9.0, 15.0, 25.0])
    )


def test_identical_backups_give_a_floored_finite_variance() -> None:
    """T-MOMENT-b: 4,000 identical backups, and never a ``nan``.

    The exact variance here is zero, and the floor is what a caller sees
    instead. Without the clamp the difference of two large nearly equal
    numbers can land just below zero, and the square root of that is a ``nan``
    that no shape check catches and that decides a move.
    """
    config = MatchConfig(num_actions=2, var_floor=0.05)
    node = MatchNode(
        to_play=0,
        is_terminal=False,
        actions=np.array([0, 1], dtype=np.int32),
        logits=np.zeros(2, dtype=np.float32),
        wdl_net=np.array([0.4, 0.2, 0.4], dtype=np.float64),
        bias_key=0,
    )
    triple = np.array([0.7, 0.1, 0.2], dtype=np.float64)
    for _ in range(4000):
        node.visits[0] += 1
        node.weight[0] += 1.0
        node.wdl_sum[0] += triple
        node.moment[0] += pack_outer(triple)

    variances = child_variances(node, config)
    assert np.isfinite(variances).all()
    assert float(variances[0]) == config.var_floor**2
    bounds = lcb_scores(node, config)
    assert np.isfinite(bounds[0])
    assert bounds[1] == -np.inf


def test_the_variance_of_a_real_mixture_is_not_the_floor() -> None:
    """T-MOMENT-b's control: the clamp must not flatten a genuine spread.

    A floor that swallowed every variance would pass the test above and make
    the LCB a constant offset, which is the failure the clamp is one line away
    from being.
    """
    config = MatchConfig(num_actions=1, var_floor=0.05)
    node = MatchNode(
        to_play=0,
        is_terminal=False,
        actions=np.array([0], dtype=np.int32),
        logits=np.zeros(1, dtype=np.float32),
        wdl_net=np.array([0.4, 0.2, 0.4], dtype=np.float64),
        bias_key=0,
    )
    for triple in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        for _ in range(50):
            node.visits[0] += 1
            node.weight[0] += 1.0
            node.wdl_sum[0] += triple
            node.moment[0] += pack_outer(triple)
    # A half-and-half mixture of +1 and -1 has mean 0 and variance 1.
    assert abs(float(child_variances(node, config)[0]) - 1.0) < 1e-12


def test_the_legal_hash_is_a_pure_function_of_the_action_ids() -> None:
    """The rehome check's O(1) filter separates the sets it has to separate."""
    a = legal_hash_of(np.array([0, 1, 2], dtype=np.int32))
    b = legal_hash_of(np.array([0, 1, 2, 3], dtype=np.int32))
    c = legal_hash_of(np.array([0, 1, 3], dtype=np.int32))
    assert a != b and a != c and b != c
    assert a == legal_hash_of(np.array([0, 1, 2], dtype=np.int32))
    assert legal_hash_of(np.zeros(0, dtype=np.int32)) == legal_hash_of(
        np.zeros(0, dtype=np.int32)
    )
    assert -(2**63) <= a < 2**63
