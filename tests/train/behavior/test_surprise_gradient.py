"""A surprising sample pulls the update toward itself, in proportion.

``test_recipe_and_weights`` proves the stored ``policy_kl`` reaches *something*:
a step at ``surprise_alpha=1`` lands somewhere a step at ``alpha=0`` does not.
That is the existence claim. This module makes the quantitative one, which is
the claim the design actually rests on -- KataGo's "one of the larger
improvements" is not "the weights are non-trivial", it is **that the samples
search was surprised by get more of the update**.

The experiment
--------------
One batch, eight synthetic rows, split into two groups whose gradients
genuinely disagree (each group's rows want mass on different board squares and
a different WDL class). Two batches are built from it that are **bit-identical
in every field except ``policy_kl``** -- asserted, not assumed, by
:func:`test_the_two_batches_differ_only_in_the_recorded_surprise`. Everything
else is fixed: the net comes from a seeded config, so two constructions are
bit-equal, and the step is deterministic.

Why the update *direction* rather than the weight value
-------------------------------------------------------
The assembler's loss is ``sum(w_i * l_i) / sum(w_i)``, so the batch gradient is
the convex combination ``lambda * g_A + (1 - lambda) * g_B`` with
``lambda = w_A / (w_A + w_B)`` over the two groups. Raising group A's recorded
KL raises ``lambda``, which slides the update along the segment from "what
group B alone would do" to "what group A alone would do". That is a *predicted
direction*, and it is measured where it matters -- on the parameters after
``optimizer.step()``, not on the weight vector.

At ``max_weight = 100`` the extreme rung (``lambda = 0.99``) is group A's own
update to within a percent, which is what makes it a usable anchor: the test
asks whether raising the recorded surprise moves the realised update
monotonically toward that anchor. A trainer that computed the weights, logged
them beautifully and passed the assembler nothing would sit at ``lambda = 0.5``
on every rung and fail the ordering.

The step is AdamW's first, so each coordinate moves by ``-lr * sign(g)`` to
within ``eps``; comparing *directions* by per-coordinate sign is therefore the
comparison the optimiser itself is making, and it is why the assertions below
count sign agreement rather than distances that AdamW has already normalised
away.
"""

from __future__ import annotations

from typing import Final

import pytest
import torch
import torch.nn.functional as torch_functional

import synthetic as syn
from komugi.net import NetOutput
from komugi.train import losses
from komugi.train.shards import stack_samples
from komugi.train.step import StepConfig, TrainStep
from komugi.train.wiring import P0A_RECIPE, outputs_from_net

_ROWS: Final[int] = 8
_GROUP_A: Final[tuple[int, ...]] = (0, 1, 2, 3)
_GROUP_B: Final[tuple[int, ...]] = (4, 5, 6, 7)
"""Two halves of the batch. Every row wants a different square (the synthetic
rule strides by 7 over 81) and the two halves disagree on the WDL class too, so
their gradients genuinely compete rather than merely differing in scale."""

_ALPHA: Final[float] = 1.0
_MAX_WEIGHT: Final[float] = 100.0
"""A high ceiling on purpose: it is what lets the extreme rung stand in for
"group A alone" (``lambda = 100 * 4 / (100 * 4 + 4) = 0.99``) while every rung
still runs the ordinary production path."""

_EXTREME_KL: Final[float] = 99.0
"""``1 + 1 * 99`` is exactly ``_MAX_WEIGHT``: the anchor sits at the clip."""

_LADDER: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 3.0, _EXTREME_KL)
"""Recorded KLs for group A, giving weights 1, 1.5, 2, 4, 100 -- i.e.
``lambda`` of 0.5, 0.6, 0.667, 0.8, 0.99."""

_LR: Final[float] = 1e-3
_STEP_SEED: Final[int] = 0


def _batch(kl_a: float, kl_b: float) -> dict[str, torch.Tensor]:
    """Return the eight-row batch with `kl_a` on group A and `kl_b` on group B.

    Built row by row from :mod:`synthetic` so that ``policy_kl`` is the only
    field that ever varies between two calls: every other tensor is a pure
    function of the row index.
    """
    rows = [
        syn.toy_row(index, policy_kl=kl_a if index in _GROUP_A else kl_b)
        for index in range(_ROWS)
    ]
    return stack_samples(rows)


def _trainer(net: torch.nn.Module, alpha: float) -> TrainStep:
    """Return the production-recipe step over `net` at surprise strength `alpha`."""
    return TrainStep(
        net,
        losses.LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=_ROWS,
            lr=_LR,
            surprise_alpha=alpha,
            surprise_max_weight=_MAX_WEIGHT,
            seed=_STEP_SEED,
        ),
    )


def _update(kl_a: float, kl_b: float = 0.0) -> torch.Tensor:
    """Return one step's parameter delta, flattened, for the given recorded KLs.

    A fresh net each call, from the same seeded config, so the only difference
    between two returned vectors is the ``policy_kl`` column of the batch.
    """
    net = syn.tiny_net()
    before = syn.parameter_snapshot(net)
    report = _trainer(net, _ALPHA).run(_batch(kl_a, kl_b))
    assert report, "the step reported no metrics at all"
    after = syn.parameter_snapshot(net)
    return torch.cat(
        [(after[name] - before[name]).reshape(-1) for name in sorted(before)]
    )


def _sign_agreement(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Return the fraction of moved coordinates on which the two updates agree.

    Restricted to coordinates the reference update actually moved: parameters
    of the staged-off moves-left head receive no gradient at all, and counting
    their shared zero as agreement would dilute every number below toward 1.
    """
    live = reference != 0.0
    assert bool(live.any().item()), "the reference update moved no parameter"
    agree = torch.sign(reference[live]) == torch.sign(other[live])
    return float(agree.to(torch.float64).mean().item())


def test_the_two_batches_differ_only_in_the_recorded_surprise() -> None:
    """The premise of every comparison here, checked field by field."""
    quiet = _batch(0.0, 0.0)
    surprising = _batch(_EXTREME_KL, 0.0)

    assert set(quiet) == set(surprising)
    for name in sorted(quiet):
        if name == "policy_kl":
            continue
        assert torch.equal(quiet[name], surprising[name]), (
            f"{name} differs between the two batches, so a difference in the "
            "resulting update could not be attributed to the weighting"
        )
    assert not torch.equal(quiet["policy_kl"], surprising["policy_kl"])
    assert surprising["policy_kl"].tolist() == [
        _EXTREME_KL if index in _GROUP_A else 0.0 for index in range(_ROWS)
    ]


def test_the_step_is_deterministic_at_a_fixed_recorded_surprise() -> None:
    """Two runs at the same KLs are bit-identical, or nothing else here means anything."""
    first = _update(1.0)
    again = _update(1.0)
    assert torch.equal(first, again), (
        "two steps on identical inputs disagreed: the ordering assertions "
        "below would be measuring noise, not the weighting"
    )


def test_the_recorded_kl_reweights_the_loss_the_step_reports() -> None:
    """The reported policy loss is the weight-normalised mean, to the digit.

    Recomputed independently: the per-sample cross-entropy is read straight off
    a reference forward (the targets are one-hot, so it is the negative log
    probability at the row's own square), and the weights are the documented
    ``clip(1 + alpha * kl, 1, max_weight)``. If the trainer normalised by batch
    size instead of by weight mass, or applied the weights to the wrong axis,
    this number moves.
    """
    batch = _batch(_EXTREME_KL, 0.0)

    reference = syn.tiny_net()
    reference.train()
    prediction: NetOutput = reference(batch["observation"])
    logits = outputs_from_net(prediction)[losses.POLICY_LOGITS]
    log_probs = torch_functional.log_softmax(logits, dim=-1)
    per_sample = -(batch["policy_target"] * log_probs).sum(dim=-1)

    weights = torch.tensor(
        [_MAX_WEIGHT if index in _GROUP_A else 1.0 for index in range(_ROWS)],
        dtype=torch.float32,
    )
    expected = float(
        ((per_sample * weights).sum() / weights.sum()).item()
    )
    unweighted = float(per_sample.mean().item())

    report = _trainer(syn.tiny_net(), _ALPHA).run(batch)
    assert report[losses.HEAD_POLICY] == pytest.approx(expected, rel=1e-5)
    assert abs(expected - unweighted) > 1e-3, (
        "the weighted and unweighted means coincide on this batch, so the "
        "assertion above would pass with the weighting disconnected"
    )
    assert report[losses.HEAD_POLICY] != pytest.approx(unweighted, rel=1e-5)


def test_more_recorded_surprise_pulls_the_update_further_toward_those_rows() -> None:
    """The direction claim: agreement with group A's own update rises with its KL.

    ``lambda`` climbs 0.5 -> 0.6 -> 0.667 -> 0.8 -> 0.99 across the ladder, so
    the realised update must slide monotonically toward the anchor. Consecutive
    rungs are allowed to tie -- a coordinate only flips when the combination
    crosses zero there, and a small step of ``lambda`` may flip none -- but the
    ends may not: a batch that up-weights group A four to one has to agree with
    group A's own update on strictly more coordinates than a balanced batch
    does.
    """
    anchor = _update(_EXTREME_KL)
    agreements = [_sign_agreement(anchor, _update(kl)) for kl in _LADDER]

    assert agreements[-1] == 1.0, "the anchor disagrees with itself"
    for lower, higher in zip(agreements[:-1], agreements[1:], strict=True):
        assert lower <= higher, (
            f"agreement with group A's own update fell as its recorded "
            f"surprise rose: {list(zip(_LADDER, agreements, strict=True))}"
        )
    assert agreements[0] < agreements[3], (
        "up-weighting the surprising rows four to one moved the update no "
        "closer to what those rows alone would do than an unweighted batch "
        f"did: {list(zip(_LADDER, agreements, strict=True))}"
    )


def test_surprise_on_the_other_half_pulls_the_update_the_other_way() -> None:
    """The mirror: the same machinery, the opposite group, the opposite tilt.

    Without this the ordering above could be produced by any monotone
    side-effect of a larger ``policy_kl`` column -- a global scale on the loss,
    say. Here the column's *magnitude* is identical and only which rows carry
    it changes, so the two updates can only differ because the weighting is
    per-sample.
    """
    anchor = _update(_EXTREME_KL)
    balanced = _update(0.0)
    mirrored = _update(0.0, _EXTREME_KL)

    toward_a = _sign_agreement(anchor, balanced)
    toward_b = _sign_agreement(anchor, mirrored)

    assert toward_b < 1.0, (
        "up-weighting group B produced exactly group A's update: the two "
        "halves of this batch do not disagree, so nothing here is measurable"
    )
    assert toward_b < toward_a, (
        "moving the recorded surprise from group A to group B did not move "
        f"the update away from group A ({toward_b:.4f} vs {toward_a:.4f})"
    )


def test_the_ceiling_binds_where_the_step_can_see_it() -> None:
    """Beyond ``max_weight`` the update stops changing -- bit-for-bit.

    The clip is what stops one pathological row owning a batch, and it is only
    a real defence if it binds on the path the optimiser is on. ``kl = 99`` and
    ``kl = 10_000`` both clip to ``max_weight``, so the two updates must be
    identical to the last bit; a ceiling applied to a logged copy of the
    weights and not to the ones passed through would let the second run away.
    """
    at_cap = _update(_EXTREME_KL)
    far_past_cap = _update(10_000.0)
    assert torch.equal(at_cap, far_past_cap), (
        "a recorded KL far beyond the ceiling changed the update, so the "
        "weights the optimiser saw were not the clipped ones"
    )
