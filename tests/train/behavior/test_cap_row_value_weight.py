"""KOM-160's mechanism, at the trainer.

The whole of campaign 4D is one float: ``StepConfig.cap_row_value_weight``, the
multiplier the **WDL** loss of a drawn row is scaled by. The pre-registered arm
is ``0.25``, its fallback ``0.6`` (``155 / 256``, mechanism-2 parity), and
``1.0`` is today.

What this file exists to make checkable, in order of how easy each is to get
wrong:

* **The arm reaches the WDL head and no other head.** ``draft_outcome`` is a
  copy of ``wdl_target`` on draft rows and the uncertainty target is the WDL
  residual; both are staged off today, and both would resolve their own
  ``composed.get(head, weights)``. A multiplier applied to "the value heads"
  rather than to ``HEAD_WDL`` would be invisible until a bring-up, and would
  then silently change what that head was taught (design review F10).
* **The normaliser is the weight mass**, not the batch size, so the arms differ
  in the *composition* of the value gradient and not in its scale. Without
  that, 4D would be a learning-rate ablation on one head in disguise.
* **The multiplier composes with whatever base the assembler resolved** rather
  than replacing it, and composes *independently* of the fast-row multiplier:
  policy takes ``base * fast``, WDL takes ``base * cap``, in one call.
* **Weight 1.0 is the pre-KOM-160 trainer to the bit.** The mapping is passed
  unconditionally, so ``1.0`` runs an all-ones vector through ``weights * 1.0``
  rather than bypassing the seam.
* **An out-of-vocabulary termination is refused.** ``2`` is reserved for resign
  and a resigned row's weight is a decision nobody has made.

Every node here runs either the production ``TrainStep.run`` or the production
assembler over a production forward. A step's update is a deterministic function
of (net state, batch) and the nets are built from a seeded config, so "these two
runs differ" is attributable to the arm and to nothing else.
"""

from __future__ import annotations

import math
from typing import Final

import pytest
import torch

import synthetic as syn
from komugi.modes import MODE_FAST, MODE_FULL
from komugi.terminations import (
    TERMINATION_DECISIVE,
    TERMINATION_DRAW,
    TERMINATION_RESIGN,
)
from komugi.train.losses import (
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    POLICY_LOGITS,
    POLICY_TARGET,
    WDL_LOGITS,
    WDL_TARGET,
    LossAssembler,
    LossConfig,
)
from komugi.train.shards import stack_samples
from komugi.train.step import TOTAL_KEY, StepConfig, TrainStep
from komugi.train.wiring import (
    P0A_RECIPE,
    cap_row_weights,
    fast_row_weights,
    outputs_from_net,
    surprise_weights_from_kl,
    targets_from_batch,
)

_ROWS: Final[int] = 8
_LR: Final[float] = 0.05
"""Large enough that one step separates two different gradients well clear of
float32 noise, which is what the "these parameters differ" assertions need."""

_ARMS: Final[tuple[float, ...]] = (0.0, 0.25, 0.6, 1.0)
"""The discard end, the pre-registered arm, its fallback, and today."""

_DRAW_ROWS: Final[frozenset[int]] = frozenset({1, 4, 6})
"""Which rows of the mixed batch come from a drawn (capped) game.

Three of eight -- close to the 23.2% row share the KOM-158 census measured at
campaign 3 iteration 14, and far enough from half that a normaliser bug cannot
hide behind a symmetric batch."""

_FAST_ROWS: Final[frozenset[int]] = frozenset({1, 3, 5, 7})
"""Which rows are fast searches. Deliberately *overlapping* ``_DRAW_ROWS``
without coinciding with them: row 1 is both, row 4 is only a draw row, row 3 is
only fast. A composition bug that multiplied the two vectors together for every
head would be green on any batch where the strata agreed."""


def _mixed_rows() -> list[dict[str, torch.Tensor]]:
    """Return `_ROWS` rows, three of them from drawn games, half of them fast."""
    return [
        syn.toy_row(
            index,
            policy_kl=0.25 * index,
            search_mode=MODE_FAST if index in _FAST_ROWS else MODE_FULL,
            termination=(
                TERMINATION_DRAW if index in _DRAW_ROWS else TERMINATION_DECISIVE
            ),
        )
        for index in range(_ROWS)
    ]


def _decisive_rows() -> list[dict[str, torch.Tensor]]:
    """Return `_ROWS` rows, every one of them from a decisive game."""
    return [
        syn.toy_row(index, termination=TERMINATION_DECISIVE)
        for index in range(_ROWS)
    ]


def _stepped(
    rows: list[dict[str, torch.Tensor]], cap_row_value_weight: float
) -> dict[str, torch.Tensor]:
    """Run one production step over `rows` at `cap_row_value_weight`.

    Returns the net's parameters afterwards. AdamW's update is a closed form of
    the gradient and its own (fresh) state, so equal parameters after one step
    from equal starting weights means equal gradients, and different parameters
    mean different gradients.
    """
    net = syn.tiny_net()
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=len(rows),
            lr=_LR,
            cap_row_value_weight=cap_row_value_weight,
            seed=0,
        ),
    )
    report = step.run(stack_samples(rows))
    assert math.isfinite(report[TOTAL_KEY]), report
    return syn.parameter_snapshot(net)


def _same(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    """Return whether two parameter dicts are equal tensor for tensor."""
    assert set(left) == set(right)
    return all(torch.equal(left[name], right[name]) for name in left)


def _per_head(
    rows: list[dict[str, torch.Tensor]], *, cap: float | None
) -> dict[str, torch.Tensor]:
    """Score `rows` through the production assembler at cap arm `cap`.

    ``cap=None`` passes **no** ``HEAD_WDL`` entry at all -- the pre-KOM-160
    mapping -- which is what makes "the weighted call" and "the unweighted call"
    two things this file can compare rather than two descriptions of one.

    A fresh net each time, forward in train mode: ``KomugiNet`` is a pure
    function of its seeded config, so two calls start bit-identical, and a
    forward is a pure function of (weights, batch). Re-using one net would fold
    BatchNorm's updated running estimates into the second call.
    """
    batch = stack_samples(rows)
    net = syn.tiny_net()
    net.train()
    outputs = outputs_from_net(net(batch["observation"]))
    targets = targets_from_batch(batch)
    fast = fast_row_weights(batch["search_mode"], 1.0)
    head_sample_weight = {HEAD_POLICY: fast, HEAD_SOFT_POLICY: fast}
    if cap is not None:
        head_sample_weight[HEAD_WDL] = cap_row_weights(batch["termination"], cap)
    _, per_head = LossAssembler(P0A_RECIPE)(
        outputs, targets, head_sample_weight=head_sample_weight
    )
    return per_head


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-sample ``-sum(target * log_softmax(logits))``, computed here.

    Recomputed rather than imported from the module under test: checking the
    assembler's arithmetic with the assembler's arithmetic would make the
    hand-computed claim circular.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    contributions = torch.where(
        target > 0, target * log_probs, torch.zeros_like(log_probs)
    )
    return -contributions.sum(dim=-1)


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> float:
    """Return ``sum(w*v) / sum(w)`` as a float."""
    return float(((values * weights).sum() / weights.sum()).item())


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_the_weighted_wdl_loss_on_a_mixed_batch_is_the_hand_computed_value() -> None:
    """Acceptance 1: the reported WDL loss is the weighted mean, to 1e-6.

    Eight rows, three of them from drawn games, at arm ``0.25``. The expected
    value is accumulated **in float64 from the per-row cross-entropies**, row by
    row, rather than by calling anything in :mod:`komugi.train.losses` -- so
    this is a golden and not a restatement of the implementation.

    The second half is what makes it a statement about the *head*: the policy
    and soft-policy values must be ``torch.equal`` to the call that passed no
    WDL multiplier at all. Not merely close -- equal, because their weight
    vectors are untouched and the arithmetic is the identical sequence of
    float32 operations. An implementation that applied the cap multiplier to
    every head, or that replaced the whole ``head_sample_weight`` mapping rather
    than adding one entry, is red on those two assertions while the first one
    still passes.
    """
    rows = _mixed_rows()
    batch = stack_samples(rows)
    arm = 0.25

    net = syn.tiny_net()
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=_ROWS, lr=_LR, cap_row_value_weight=arm, seed=0
        ),
    )
    metrics = step.run(batch)

    # The same forward, on a net that has not been stepped or had its BatchNorm
    # estimates advanced.
    mirror = syn.tiny_net()
    mirror.train()
    outputs = outputs_from_net(mirror(batch["observation"]))

    wdl_ce = _cross_entropy(
        outputs[WDL_LOGITS].detach().to(torch.float64),
        batch[WDL_TARGET].to(torch.float64),
    )
    numerator = 0.0
    mass = 0.0
    for index in range(_ROWS):
        weight = arm if index in _DRAW_ROWS else 1.0
        numerator += weight * float(wdl_ce[index])
        mass += weight
    assert mass == pytest.approx(5.0 + 3.0 * arm), (
        "the hand-computed weight mass is not five decisive rows plus three "
        "draw rows at the arm; the batch is not the batch this node describes"
    )

    assert metrics[HEAD_WDL] == pytest.approx(numerator / mass, abs=1e-6)

    weighted = _per_head(rows, cap=arm)
    unweighted = _per_head(rows, cap=None)
    assert torch.equal(weighted[HEAD_POLICY], unweighted[HEAD_POLICY]), (
        "the cap-row weight moved the POLICY head; a cap row's policy target "
        "is a real search result and 4D does not touch it"
    )
    assert torch.equal(
        weighted[HEAD_SOFT_POLICY], unweighted[HEAD_SOFT_POLICY]
    ), "the cap-row weight moved the soft-policy head; it keys HEAD_WDL alone"
    assert not torch.equal(weighted[HEAD_WDL], unweighted[HEAD_WDL]), (
        "the WDL head is unchanged at arm 0.25, so the two assertions above "
        "are vacuous and nothing here is being weighted at all"
    )


@pytest.mark.parametrize("arm", _ARMS)
def test_the_normaliser_is_the_weight_mass_on_a_mixed_batch(arm: float) -> None:
    """Acceptance 3: the arm changes the mix, not the loss scale.

    The batch is built so that **every row's WDL cross-entropy is the same
    constant** ``ln 3``: uniform logits over three classes, and any normalised
    target then gives ``-sum(t * log(1/3)) == ln 3``. A weighted mean of a
    constant is that constant *whatever the weights are* -- but only if the
    normaliser is the weight mass. An implementation dividing by the batch size
    or by a fixed row count reports ``ln3 * (mass / n)`` instead, which moves
    with the arm.

    This is the property the design rests on when it says 4D is not a learning
    rate change on the value head: because the gradient magnitude does not move
    between arms, 4A and 4D differ in composition alone.

    ``0.0`` is in the sweep and is *not* a degenerate case here: five of the
    eight rows are decisive, so the mass is 5 rather than 0 and the head is
    reported rather than omitted. (An all-draw batch at ``0.0`` is the omission
    path, which is why ``0.0`` is not a pre-registered arm.)
    """
    termination = torch.tensor(
        [
            TERMINATION_DRAW if index in _DRAW_ROWS else TERMINATION_DECISIVE
            for index in range(_ROWS)
        ]
    )
    uniform = torch.zeros(_ROWS, 3, dtype=torch.float64)
    target = torch.zeros(_ROWS, 3, dtype=torch.float64)
    for index in range(_ROWS):
        target[index, index % 3] = 1.0

    assembler = LossAssembler(LossConfig(policy_enabled=False, wdl_enabled=True))
    cap = cap_row_weights(termination, arm)
    _, per_head = assembler(
        {WDL_LOGITS: uniform},
        {WDL_TARGET: target},
        # Supplied directly, because ``sample_weights`` reads ``policy_target``
        # whatever the config says and this node has one head. Exactly ones, so
        # the composed vector is the cap multiplier itself.
        sample_weight=torch.ones(_ROWS, dtype=torch.float64),
        head_sample_weight={HEAD_WDL: cap},
    )

    assert per_head[HEAD_WDL].item() == pytest.approx(math.log(3.0), abs=1e-12)


# --------------------------------------------------------------------------
# Which rows and which heads the arm reaches
# --------------------------------------------------------------------------


def test_the_cap_weight_never_touches_a_decisive_row() -> None:
    """Acceptance 5: on an all-decisive batch every arm is one step.

    The arm applies to draw rows and to nothing else, so a batch with no draw
    row must be bit-identical across arms. Red against a mapping keyed on the
    *wrong* code, which would down-weight the decisive rows instead -- the
    failure that would look like 4D working while training the opposite
    experiment.

    The vector itself is asserted first, so a reader does not have to infer
    "exactly ones" from "the parameters agreed".
    """
    termination = torch.full((_ROWS,), TERMINATION_DECISIVE, dtype=torch.int64)
    for arm in _ARMS:
        weights = cap_row_weights(termination, arm)
        assert torch.equal(weights, torch.ones(_ROWS)), (
            f"at arm {arm} a decisive-only column produced {weights.tolist()}, "
            f"not ones"
        )

    rows = _decisive_rows()
    reference = _stepped(rows, 1.0)
    for arm in _ARMS:
        assert _same(_stepped(rows, arm), reference), (
            f"arm {arm} moved an all-decisive batch's parameters; the cap-row "
            f"weight is reaching rows whose termination is decisive"
        )


def test_a_draw_rows_policy_target_still_trains_at_the_discard_end() -> None:
    """The node that distinguishes the arm from a row filter.

    A capped game's *search* was a real search: its policy target is as good as
    any other row's, and its moves-left target is the plies-to-cap the MLH head
    exists to learn. A filter that dropped draw rows before the batch was
    assembled would throw both away. The weight keeps them, and this is visible
    even at ``0.0`` -- where the row contributes no WDL mass at all -- because
    the policy heads still read it.

    The complement assertion is what stops it being vacuous: the same
    perturbation on the WDL target of the same row must be inert at ``0.0``.
    """
    rows = _mixed_rows()
    draw_row = min(_DRAW_ROWS)
    baseline = _stepped(rows, 0.0)

    bumped_policy = _mixed_rows()
    target = bumped_policy[draw_row][POLICY_TARGET].clone()
    moved = torch.zeros_like(target)
    moved[(int(target.argmax()) + 1) % target.numel()] = 0.5
    bumped_policy[draw_row][POLICY_TARGET] = 0.5 * target + moved
    assert not _same(_stepped(bumped_policy, 0.0), baseline), (
        "moving a draw row's POLICY target changed nothing at weight 0.0: the "
        "arm has become a row filter and is discarding the search result too"
    )

    bumped_wdl = _mixed_rows()
    bumped_wdl[draw_row][WDL_TARGET] = torch.roll(
        bumped_wdl[draw_row][WDL_TARGET], 1
    )
    assert _same(_stepped(bumped_wdl, 0.0), baseline), (
        "moving a draw row's WDL target changed the update at weight 0.0, "
        "where that row carries no value mass at all"
    )


# --------------------------------------------------------------------------
# Identity at 1.0, and composition
# --------------------------------------------------------------------------


def _pre_change_step(
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Reproduce ``TrainStep.run`` *without* the WDL multiplier, verbatim.

    Everything ``run`` does -- ``net.train()``, the forward, the wiring, the
    assembler, the pre-clear, ``backward``, ``step``, the post-clear -- with one
    difference: ``head_sample_weight`` carries the two policy entries and no
    ``HEAD_WDL``. This is the code path that existed before KOM-160, spelled out
    here so "bit-identical to today" is a comparison against something rather
    than an assertion about a diff.
    """
    net = syn.tiny_net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=_LR, weight_decay=0.0)
    batch = stack_samples(rows)
    net.train()
    outputs = outputs_from_net(net(batch["observation"]))
    targets = targets_from_batch(batch)
    fast = fast_row_weights(batch["search_mode"], 1.0)
    total, _ = LossAssembler(P0A_RECIPE)(
        outputs,
        targets,
        sample_weight=None,
        head_sample_weight={HEAD_POLICY: fast, HEAD_SOFT_POLICY: fast},
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return syn.parameter_snapshot(net)


def test_weight_one_is_bit_identical_to_the_current_trainer() -> None:
    """Acceptance 2: ``1.0`` is the pre-KOM-160 trainer, to the bit.

    The mapping is passed **unconditionally**, so ``w = 1.0`` runs an all-ones
    multiplier through ``weights * 1.0`` rather than bypassing the seam. That is
    what makes this node load-bearing: under a ``!= 1.0`` guard it would pin the
    bypass, and an incorrectly computed ``head_sample_weight`` would still pass.

    The batch is the *mixed* one, so the identity claim is made where it could
    fail -- three of its rows are draw rows, and at any other arm they would be
    weighted.
    """
    rows = _mixed_rows()
    assert _same(_stepped(rows, 1.0), _pre_change_step(rows)), (
        "the step at weight 1.0 differs from the pre-KOM-160 step; "
        "`weights * ones` is not the identity here, so the default arm is not "
        "the status quo it is reported as"
    )
    assert not _same(_stepped(rows, 0.25), _pre_change_step(rows)), (
        "arm 0.25 is ALSO identical to the pre-change step, so the node above "
        "cannot distinguish the seam from a no-op"
    )


def test_the_cap_weight_composes_with_the_fast_weight_and_the_surprise_base() -> None:
    """Acceptance 4: ``base * fast`` on policy, ``base * cap`` on WDL, one call.

    The reachable surprise base is ``StepConfig.surprise_alpha``, not
    ``LossConfig``'s: ``run`` builds targets through ``targets_from_batch``,
    which projects only ``TARGET_KEYS``, and ``prior_logits`` is deliberately
    absent from both that tuple and the schema.

    Four assertions, and each one kills a different implementation. **Red**
    against one that *replaces* the base on the WDL head (it would report
    ``_weighted_mean(ce_wdl, cap)``); against one that applies ``cap`` to the
    policy head as well (the policy assertion fails); against one that leaves
    the WDL head on the base alone (the third); and against one that composes
    the two multipliers together for every head (the fourth). Green only against
    two independent per-head compositions of one shared base.

    The second net is not a convenience: ``run``'s forward runs in train mode
    and updates BatchNorm's running estimates, so recomputing the logits on the
    *same* net afterwards would not reproduce them.
    """
    rows = _mixed_rows()
    batch = stack_samples(rows)
    alpha = 1.0
    max_weight = 5.0
    fast_arm = 0.125
    cap_arm = 0.25

    trained = syn.tiny_net()
    step = TrainStep(
        trained,
        LossAssembler(P0A_RECIPE),
        StepConfig(
            batch_size=_ROWS,
            lr=_LR,
            surprise_alpha=alpha,
            surprise_max_weight=max_weight,
            fast_policy_weight=fast_arm,
            cap_row_value_weight=cap_arm,
            seed=0,
        ),
    )
    metrics = step.run(batch)

    mirror = syn.tiny_net()
    mirror.train()
    outputs = outputs_from_net(mirror(batch["observation"]))

    kl_base = surprise_weights_from_kl(
        batch["policy_kl"], alpha=alpha, max_weight=max_weight
    )
    assert not torch.equal(kl_base, torch.ones_like(kl_base)), (
        "the surprise base is all ones, so this node could not tell "
        "`kl_base * cap` from `cap` alone"
    )
    fast = fast_row_weights(batch["search_mode"], fast_arm)
    cap = cap_row_weights(batch["termination"], cap_arm)
    assert not torch.equal(fast, cap), (
        "the two multipliers coincide on this batch, so the composition claim "
        "below could not tell them apart"
    )

    policy_ce = _cross_entropy(outputs[POLICY_LOGITS], batch[POLICY_TARGET])
    wdl_ce = _cross_entropy(outputs[WDL_LOGITS], batch[WDL_TARGET])

    assert metrics[HEAD_POLICY] == pytest.approx(
        _weighted(policy_ce, kl_base * fast), rel=1e-5
    )
    assert metrics[HEAD_WDL] == pytest.approx(
        _weighted(wdl_ce, kl_base * cap), rel=1e-5
    )
    assert metrics[HEAD_WDL] != pytest.approx(
        _weighted(wdl_ce, kl_base), rel=1e-5
    ), "the WDL head took the surprise base alone; the cap arm never arrived"
    assert metrics[HEAD_WDL] != pytest.approx(
        _weighted(wdl_ce, kl_base * fast * cap), rel=1e-5
    ), "the WDL head took the fast multiplier too; the two heads share a base, not a vector"
    assert HEAD_SOFT_POLICY in metrics


# --------------------------------------------------------------------------
# The vocabulary's edge
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", (TERMINATION_RESIGN, 3, -1))
def test_a_reserved_or_unknown_termination_is_refused(value: int) -> None:
    """Acceptance 6: ``2`` and anything past it raise, naming the column.

    ``2`` is reserved for resign v2 and is refused rather than read as decisive:
    a resigned row's weight is a decision nobody has made, and taking it as
    ``1.0`` would train it at a weight nobody chose. ``3`` stands for the
    vocabulary widening the design contemplates (a cap-specific code), which is
    fingerprint-invisible on the wire -- this refusal is the only thing that
    would make it loud. ``-1`` is the corrupt-row case.

    Parametrised rather than looped so a single surviving value names itself in
    the report.
    """
    termination = torch.full((_ROWS,), TERMINATION_DECISIVE, dtype=torch.int64)
    termination[2] = value
    with pytest.raises(ValueError, match="termination"):
        cap_row_weights(termination, 0.25)


def test_a_wrong_rank_termination_column_is_refused() -> None:
    """A 2-D column would broadcast against the per-row losses, silently.

    The same refusal ``surprise_weights_from_kl`` makes, for the same reason:
    the assembler's own length check sees one entry per sample and a
    ``(batch, 1)`` vector satisfies it after broadcasting while reweighting the
    wrong axis.
    """
    with pytest.raises(ValueError, match="termination"):
        cap_row_weights(torch.zeros(_ROWS, 1, dtype=torch.int64), 0.25)
