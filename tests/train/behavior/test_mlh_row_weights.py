"""KOM-159 D4's mechanism, at the trainer: turn-cap rows out of the MLH loss.

The whole of the censoring is one float and one predicate.
:attr:`~komugi.train.step.StepConfig.mlh_cap_row_weight` is what the
**moves-left** loss of a row from a turn-capped game is multiplied by, and
:func:`~komugi.train.wiring.mlh_row_weights` is the predicate that decides
which rows those are: a draw whose reconstructed length
``round(ply + mlh_target * cap_plies)`` reaches the cap. ``0.0`` is campaign
6's arm; ``1.0`` is the pre-KOM-159 arithmetic.

What this file exists to make checkable, in order of how easy each is to get
wrong:

* **The predicate is the cap, not the draw.** ``termination`` records that a
  game was drawn and not which rule drew it, so a fourfold repetition at ply
  300 is a real ending with a real moves-left label and must keep weight 1.
  A censoring keyed on ``termination`` alone -- the obvious implementation,
  and the one :func:`~komugi.train.wiring.cap_row_weights` legitimately uses
  for the WDL head -- throws those rows away too.
* **The arm reaches the moves-left head and no other.** The WDL head has its
  own cap-row knob (KOM-160) with its own arm; a multiplier applied to "the
  value heads" would silently make the two experiments one.
* **The normaliser is the weight mass**, so the arms differ in *which rows*
  the head learns from and not in the size of its gradient.
* **Weight 1.0 is the pre-KOM-159 trainer.** The mapping is passed
  unconditionally, so ``1.0`` runs an all-ones vector through ``weights * 1.0``
  rather than bypassing the seam.
* **An out-of-vocabulary termination is refused**, exactly as it is for the WDL
  weight: ``2`` is reserved for resign, and a resigned game's moves-left target
  has a definition nobody has written down.

The pattern -- and much of the shape -- is
``tests/train/behavior/test_cap_row_value_weight.py``'s, deliberately: the two
functions are siblings, and a reader comparing them should find the same
claims made the same way.
"""

from __future__ import annotations

import math
from typing import Final

import pytest
import torch

import synthetic as syn
from komugi.terminations import (
    TERMINATION_DECISIVE,
    TERMINATION_DRAW,
    TERMINATION_RESIGN,
)
from komugi.train.losses import (
    HEAD_MLH,
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    MLH_PRED,
    MLH_TARGET,
    LossAssembler,
)
from komugi.train.shards import stack_samples
from komugi.train.step import TOTAL_KEY, StepConfig, TrainStep
from komugi.train.wiring import (
    MLH_RECIPE,
    mlh_row_weights,
    outputs_from_net,
    targets_from_batch,
)

_CAP: Final[int] = 512
"""The turn cap these rows were 'played' under (KOM-165's, today's engine).

Stated here rather than imported from the engine because this file is about
the *arithmetic over a stated cap*: the campaign loop passes
``Game.max_turns()`` and ``tests/train/test_step_unit.py`` pins the trainer's
default against it, so a cap bump is loud there rather than silently
reinterpreting every number below.
"""

_LR: Final[float] = 0.05
_ROWS: Final[int] = 8

_REPETITION_TOTAL: Final[int] = 300
"""Length of the drawn game that did **not** reach the cap.

The design's own example (a fourfold repetition, 0-3 per iteration through
campaign 5). Far enough below 512 that no rounding question arises, and far
enough above zero that a predicate reading ``mlh_target`` alone would still
find it."""


def _row(
    index: int, *, ply: int, total: int, termination: int
) -> dict[str, torch.Tensor]:
    """One synthetic row placed at `ply` of a `total`-ply game.

    The moves-left target is the recorder's own arithmetic
    (``(total - ply) / MAX_TURNS``, :data:`komugi.selfplay.records.MLH_SCALE`),
    which is what makes ``round(ply + mlh_target * cap)`` recover `total`
    rather than approximate it. Every value used here has a power-of-two
    denominator, so the float32 column is exact and the reconstruction is not
    resting on the rounding.
    """
    # A distinct recorded KL per row, so the surprise base this file composes
    # against at the end is not accidentally all ones.
    row = syn.toy_row(index, policy_kl=0.25 * index, termination=termination)
    row["ply"] = torch.tensor(ply, dtype=torch.int64)
    row["mlh_target"] = torch.tensor(
        (total - ply) / _CAP, dtype=torch.float32
    )
    return row


def _mixed_rows() -> list[dict[str, torch.Tensor]]:
    """Eight rows: three from a capped game, one repetition draw, four decisive.

    Row 3 is the repetition draw -- ``termination`` is a draw and the game
    ended at ply 300 -- and it is the row that distinguishes "censor the cap"
    from "censor every draw". Row 2 sits one ply short of the cap in a decisive
    game, so a predicate that forgot ``termination`` would catch it.
    """
    plies = (0, 137, _CAP - 1, 100, 4, 60, 511, 300)
    totals = (_CAP, _CAP, _CAP, _REPETITION_TOTAL, 96, 180, 640, 340)
    terminations = (
        TERMINATION_DRAW,
        TERMINATION_DRAW,
        TERMINATION_DRAW,
        TERMINATION_DRAW,
        TERMINATION_DECISIVE,
        TERMINATION_DECISIVE,
        TERMINATION_DECISIVE,
        TERMINATION_DECISIVE,
    )
    return [
        _row(index, ply=plies[index], total=totals[index], termination=term)
        for index, term in enumerate(terminations)
    ]


_CAP_ROWS: Final[frozenset[int]] = frozenset({0, 1, 2})
"""Which rows of :func:`_mixed_rows` are turn-cap rows -- the censored set.

Three of eight. Row 3 is a draw and is *not* here; row 6 reaches ply 511 in a
decisive 640-ply game and is not here either. Both omissions are the point.
"""


_CENSORED_ROW: Final[int] = 1
"""The cap row the perturbation nodes below move.

Row 1 rather than row 0 because it sits at ply 137 with 375 plies to go, so its
moves-left target can be moved *up* -- keeping the row inside the censored set,
which row 0's target of exactly 1.0 cannot do.
"""


def _expected(rows: list[dict[str, torch.Tensor]], weight: float) -> list[float]:
    """The multiplier this file claims, computed from the row list alone."""
    return [
        weight if index in _CAP_ROWS else 1.0 for index in range(len(rows))
    ]


def _weights_for(
    rows: list[dict[str, torch.Tensor]], weight: float
) -> torch.Tensor:
    """Run the production predicate over `rows` at `weight`."""
    batch = stack_samples(rows)
    return mlh_row_weights(
        batch["ply"],
        batch["mlh_target"],
        batch["termination"],
        _CAP,
        weight,
    )


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight", (0.0, 0.25, 1.0))
def test_the_predicate_censors_the_cap_rows_and_only_those(weight: float) -> None:
    """T7: 0 on a cap row, 1.0 on a decisive row and on a short draw.

    Asserted as the whole vector against a list built from the row
    descriptions, so a predicate that was right about *how many* rows to censor
    and wrong about *which* is red. Swept over three weights because the
    predicate must not depend on the arm: at ``1.0`` the vector is ones and the
    censored set is invisible in the values, which is exactly why the next node
    exists.
    """
    rows = _mixed_rows()
    weights = _weights_for(rows, weight)

    assert weights.tolist() == _expected(rows, weight), (
        f"at weight {weight} the predicate produced {weights.tolist()}, want "
        f"{_expected(rows, weight)}"
    )
    assert weights.shape == (len(rows),)


def test_a_cap_row_is_censored_exactly_and_a_repetition_draw_is_not() -> None:
    """The two rows the design names, on their own, at the launch arm.

    ``0.0`` **exactly** -- ``float(x) == 0.0``, not ``approx`` -- because the
    censoring is a mask and a row weighted 1e-9 is a row still in the loss. And
    ``1.0`` exactly on the repetition draw, which is the assertion that
    separates KOM-159's predicate from ``cap_row_weights``' termination-only
    one: a fourfold repetition at ply 300 ended, its moves-left label is a real
    observation, and censoring it would throw away the only mid-length targets
    the head has.
    """
    rows = _mixed_rows()
    weights = _weights_for(rows, 0.0)

    for index in sorted(_CAP_ROWS):
        assert float(weights[index]) == 0.0, (
            f"row {index} reaches the cap and was weighted "
            f"{float(weights[index])!r}"
        )
    assert float(weights[3]) == 1.0, (
        f"the repetition draw (ply 100 of a {_REPETITION_TOTAL}-ply game) was "
        f"weighted {float(weights[3])!r}: the predicate is reading "
        f"`termination` alone and censoring every draw"
    )
    for index in (4, 5, 6, 7):
        assert float(weights[index]) == 1.0, (
            f"decisive row {index} was weighted {float(weights[index])!r}"
        )


def test_weight_one_is_exactly_ones_on_every_row() -> None:
    """The identity arm, asserted on the vector rather than inferred.

    ``torch.equal`` against ``ones``: this is what makes "the default arm is
    the status quo" checkable at the seam, since the mapping is passed
    unconditionally and ``weights * ones`` is the identity only if the vector
    really is ones.

    **A substitution from the design, recorded here** (KOM-159a impl review,
    finding 5). Design T7 states this claim over *a campaign-5 shard fixture*;
    the rows below are synthetic ones from :func:`_mixed_rows`. The claim is
    about the mapping and not about any distribution of games, so synthetic
    rows carry it -- and they carry it more sharply, because ``_mixed_rows``
    plants cap rows, repetition draws and decisive rows deliberately, where a
    real campaign-5 shard's mix of them is whatever that run happened to play.
    What is *not* covered by the substitution is the shard schema itself: that
    a real shard's ``ply``/``mlh_target``/``termination`` columns arrive in the
    dtypes this predicate reads is the round trip's claim
    (``tests/integration/test_round_trip.py``), not this node's.
    """
    rows = _mixed_rows()
    weights = _weights_for(rows, 1.0)
    assert torch.equal(weights, torch.ones(len(rows))), (
        f"weight 1.0 produced {weights.tolist()}, not ones"
    )


def test_the_boundary_is_at_the_cap_and_is_inclusive() -> None:
    """``>= cap``, one ply either side of it, on a drawn game.

    A game that ends *at* the cap is the censored case and a game that ends one
    ply short of it is not -- the engine's adjudication fires at
    ``turns >= max_turns()``, so "reached the cap" is inclusive. An
    implementation using ``>`` censors nothing at all (no game exceeds the
    cap), which would be a silently inert arm: every claim about weight 0 would
    still pass on any batch with no cap rows in it.
    """
    at_cap = _row(0, ply=8, total=_CAP, termination=TERMINATION_DRAW)
    below = _row(1, ply=8, total=_CAP - 1, termination=TERMINATION_DRAW)
    weights = _weights_for([at_cap, below], 0.0)

    assert float(weights[0]) == 0.0, "a game ending exactly at the cap is censored"
    assert float(weights[1]) == 1.0, (
        "a drawn game ending one ply short of the cap was censored; the "
        "predicate is not reading the row's reconstructed length at all"
    )


def test_the_reconstruction_survives_the_float32_column() -> None:
    """Every ply of a capped game is censored, not merely the early ones.

    ``mlh_target`` is a float32 column and the predicate multiplies it back up
    by 512, so the arithmetic has to land *on* the cap rather than one ULP
    below it. Swept over every eighth ply of a full-length game: an
    implementation that compared without rounding, or that did the
    reconstruction in float32, leaves a scatter of uncensored rows in the
    middle of a stall -- which is the worst available failure, because the arm
    would look like it was working.
    """
    rows = [
        _row(index, ply=ply, total=_CAP, termination=TERMINATION_DRAW)
        for index, ply in enumerate(range(0, _CAP, 8))
    ]
    weights = _weights_for(rows, 0.0)
    survivors = [index for index, value in enumerate(weights.tolist()) if value]
    assert not survivors, (
        f"{len(survivors)} row(s) of a capped game escaped censoring at plies "
        f"{[index * 8 for index in survivors]}"
    )


# --------------------------------------------------------------------------
# The vocabulary's edge, and the malformed batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", (TERMINATION_RESIGN, 3, -1))
def test_a_reserved_or_unknown_termination_is_refused(value: int) -> None:
    """``2`` and anything past it raise, naming the column.

    The same refusal :func:`~komugi.train.wiring.cap_row_weights` makes and for
    the same reason: a resigned game's moves-left target is "plies until
    adjudication" under a definition resign v2 has not written yet, so reading
    it as an ordinary uncensored row would train it at a weight nobody chose.
    """
    rows = _mixed_rows()
    rows[4]["termination"] = torch.tensor(value, dtype=torch.int64)
    with pytest.raises(ValueError, match="termination"):
        _weights_for(rows, 0.0)


def test_a_wrong_rank_or_mismatched_column_is_refused() -> None:
    """A 2-D column would broadcast; three columns of different lengths lie.

    Both are the malformed-batch case rather than the wrong-value case, and
    both are silent without the guard: a ``(batch, 1)`` vector satisfies the
    assembler's own length check after broadcasting, and a shorter ``ply``
    column would pair each row's target with another row's ply.
    """
    ply = torch.zeros(_ROWS, dtype=torch.int64)
    target = torch.zeros(_ROWS, dtype=torch.float32)
    termination = torch.zeros(_ROWS, dtype=torch.int64)

    with pytest.raises(ValueError, match="ply"):
        mlh_row_weights(ply.reshape(_ROWS, 1), target, termination, _CAP, 0.0)
    with pytest.raises(ValueError, match="mlh_target"):
        mlh_row_weights(ply, target.reshape(_ROWS, 1), termination, _CAP, 0.0)
    with pytest.raises(ValueError, match="termination"):
        mlh_row_weights(ply, target, termination.reshape(_ROWS, 1), _CAP, 0.0)
    with pytest.raises(ValueError, match="same batch"):
        mlh_row_weights(ply[:-1], target, termination, _CAP, 0.0)


@pytest.mark.parametrize("cap", (0, -1))
def test_a_cap_below_one_is_refused(cap: int) -> None:
    """A zero cap censors every row in the batch; it is not a turn cap.

    Refused at the call rather than discovered as a moves-left head that
    reports nothing (a whole batch at weight 0 is a zero-mass head, which the
    assembler *legitimately* omits -- so the symptom would be a missing curve,
    not an error).
    """
    with pytest.raises(ValueError, match="cap_plies"):
        mlh_row_weights(
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.int64),
            cap,
            0.0,
        )


# --------------------------------------------------------------------------
# The composed loss
# --------------------------------------------------------------------------


def _mlh_squared_errors(
    batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Per-row ``(pred - target) ** 2`` in float64, computed here.

    Recomputed rather than imported from the module under test: checking the
    assembler's arithmetic with the assembler's arithmetic would make the
    hand-computed claim below circular.
    """
    pred = outputs[MLH_PRED].detach().reshape(-1).to(torch.float64)
    target = batch[MLH_TARGET].reshape(-1).to(torch.float64)
    return (pred - target) ** 2


def test_the_composed_mlh_loss_is_the_hand_computed_weighted_mean() -> None:
    """The reported ``loss_mlh`` is the censored mean, to 1e-6.

    Eight rows, three of them from a capped game, run through the production
    :meth:`~komugi.train.step.TrainStep.run` under
    :data:`~komugi.train.wiring.MLH_RECIPE` at the launch arm. The expected
    value is accumulated in float64 from the per-row squared errors of a
    *mirror* net's forward -- a second net from the same seed, because ``run``
    updates BatchNorm's running estimates and the same net could not reproduce
    its own logits afterwards.

    The mass is asserted first and separately: five rows at weight 1 and three
    at 0. That is what makes this a statement about the **normaliser** -- the
    assembler divides by the weight mass, so censoring changes which rows the
    head learns from and not the scale of its gradient. An implementation
    dividing by the batch size reports ``5/8`` of this number.
    """
    rows = _mixed_rows()
    batch = stack_samples(rows)

    net = syn.tiny_net()
    metrics = TrainStep(
        net,
        LossAssembler(MLH_RECIPE),
        StepConfig(
            batch_size=_ROWS,
            lr=_LR,
            mlh_cap_row_weight=0.0,
            cap_plies=_CAP,
            seed=0,
        ),
    ).run(batch)
    assert math.isfinite(metrics[TOTAL_KEY]), metrics

    mirror = syn.tiny_net()
    mirror.train()
    outputs = outputs_from_net(mirror(batch["observation"]))
    errors = _mlh_squared_errors(batch, outputs)

    numerator = 0.0
    mass = 0.0
    for index in range(_ROWS):
        weight = 0.0 if index in _CAP_ROWS else 1.0
        numerator += weight * float(errors[index])
        mass += weight
    assert mass == pytest.approx(float(_ROWS - len(_CAP_ROWS))), (
        "the hand-computed mass is not five uncensored rows; the batch is not "
        "the batch this node describes"
    )

    assert metrics[HEAD_MLH] == pytest.approx(numerator / mass, abs=1e-6)
    assert metrics[HEAD_MLH] != pytest.approx(
        float(errors.mean()), abs=1e-6
    ), (
        "the censored mean coincides with the plain mean over all eight rows, "
        "so this node cannot tell a censored head from an uncensored one"
    )


def test_the_arm_reaches_the_moves_left_head_and_no_other() -> None:
    """Censoring moves ``loss_mlh`` and leaves policy, soft policy and WDL alone.

    The moves-left knob and the WDL cap-row knob are two experiments on the
    same stratum (KOM-160 down-weights a capped game's *value* target; KOM-159
    censors its *length* target), and they compose only because each reaches
    one head. An implementation applying this multiplier to "the heads that
    care about cap rows" would make campaign 6 an unattributable pair of
    levers.

    Two fresh nets from the same seed, one per arm, for the BatchNorm reason
    above.
    """
    batch = stack_samples(_mixed_rows())

    def stepped(weight: float) -> dict[str, float]:
        """One production step at `weight`, on its own net."""
        return TrainStep(
            syn.tiny_net(),
            LossAssembler(MLH_RECIPE),
            StepConfig(
                batch_size=_ROWS,
                lr=_LR,
                mlh_cap_row_weight=weight,
                cap_plies=_CAP,
                seed=0,
            ),
        ).run(batch)

    censored = stepped(0.0)
    uncensored = stepped(1.0)

    assert censored[HEAD_MLH] != pytest.approx(uncensored[HEAD_MLH], abs=1e-9), (
        "the arm did not change the reported moves-left loss, so "
        "mlh_cap_row_weight is not reaching the assembler at all"
    )
    for head in (HEAD_POLICY, HEAD_SOFT_POLICY, HEAD_WDL):
        assert censored[head] == pytest.approx(uncensored[head], abs=1e-12), (
            f"censoring the moves-left head moved the {head} head; the arm "
            f"keys HEAD_MLH alone"
        )


def test_an_all_cap_batch_omits_the_head_rather_than_dividing_by_zero() -> None:
    """The reachable degenerate case, named rather than discovered in a campaign.

    Every row of a game that reached the cap is a cap row, so a batch drawn
    entirely from one stalled game has zero moves-left mass at the launch arm.
    The assembler's contract is that a zero-mass head is *omitted* -- no key in
    the breakdown, no contribution to the total -- rather than reported as a
    ``0.0`` or divided into a ``NaN``. This is the node that says so for this
    head, and it is why the journal's ``loss_mlh`` series may legitimately have
    a hole in it.
    """
    rows = [
        _row(index, ply=index * 4, total=_CAP, termination=TERMINATION_DRAW)
        for index in range(4)
    ]
    metrics = TrainStep(
        syn.tiny_net(),
        LossAssembler(MLH_RECIPE),
        StepConfig(
            batch_size=len(rows),
            lr=_LR,
            mlh_cap_row_weight=0.0,
            cap_plies=_CAP,
            seed=0,
        ),
    ).run(stack_samples(rows))

    assert HEAD_MLH not in metrics, (
        f"a batch of nothing but censored rows reported {metrics.get(HEAD_MLH)!r} "
        f"for the moves-left head; a zero-mass head is omitted"
    )
    assert math.isfinite(metrics[TOTAL_KEY]), (
        f"the total is {metrics[TOTAL_KEY]!r}: an omitted head became a "
        f"division by zero somewhere in the sum"
    )


def test_the_censored_rows_still_train_every_other_head() -> None:
    """The node that distinguishes the arm from a row filter.

    A capped game's search was a real search and its outcome was a real draw:
    its policy target and its WDL target are as good as any other row's, and
    only its *length* label is censored. A filter that dropped cap rows before
    the batch was assembled would throw all three away. The weight keeps them,
    which is visible even at ``0.0`` -- perturbing a censored row's WDL target
    must still move the update, while perturbing its moves-left target must
    not.
    """
    def stepped(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """One production step over `rows` at the launch arm; parameters after."""
        net = syn.tiny_net()
        TrainStep(
            net,
            LossAssembler(MLH_RECIPE),
            StepConfig(
                batch_size=len(rows),
                lr=_LR,
                mlh_cap_row_weight=0.0,
                cap_plies=_CAP,
                seed=0,
            ),
        ).run(stack_samples(rows))
        return syn.parameter_snapshot(net)

    def same(
        left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
    ) -> bool:
        """Whether two parameter dicts are equal tensor for tensor."""
        assert set(left) == set(right)
        return all(torch.equal(left[name], right[name]) for name in left)

    baseline = stepped(_mixed_rows())

    bumped_wdl = _mixed_rows()
    bumped_wdl[_CENSORED_ROW]["wdl_target"] = torch.roll(
        bumped_wdl[_CENSORED_ROW]["wdl_target"], 1
    )
    assert not same(stepped(bumped_wdl), baseline), (
        "moving a censored row's WDL target changed nothing: the arm has "
        "become a row filter and is discarding the whole row"
    )

    bumped_mlh = _mixed_rows()
    # *Longer*, deliberately: the reconstructed total is an input to the
    # censoring predicate as well as to the loss, so a perturbation towards a
    # shorter game would take the row out of the censored set and this
    # assertion would then be pinning nothing. From 375 plies-to-go to 410 the
    # row stays a cap row, and its squared error moves by a lot.
    bumped_mlh[_CENSORED_ROW]["mlh_target"] = torch.tensor(
        410.0 / _CAP, dtype=torch.float32
    )
    assert _weights_for(bumped_mlh, 0.0).tolist() == _expected(
        bumped_mlh, 0.0
    ), "the perturbation moved the row out of the censored set"
    assert same(stepped(bumped_mlh), baseline), (
        "moving a censored row's moves-left target changed the update, at the "
        "arm where that row carries no moves-left mass at all"
    )


def test_the_multiplier_composes_with_the_assemblers_base_rather_than_replacing_it() -> None:
    """``base * mlh`` on the moves-left head, with a non-trivial base.

    The reachable base is the surprise weighting
    (:attr:`~komugi.train.step.StepConfig.surprise_alpha`), which the assembler
    applies to *every* head. Composed, the moves-left head must see
    ``kl_base * mlh``: an implementation that *replaced* the base on this head
    would report the censored mean unweighted by surprise, which is a different
    number and a different experiment. The two complements are asserted too, so
    the node is red against replacing the base and against dropping the arm.
    """
    rows = _mixed_rows()
    batch = stack_samples(rows)
    alpha = 1.0
    max_weight = 5.0

    metrics = TrainStep(
        syn.tiny_net(),
        LossAssembler(MLH_RECIPE),
        StepConfig(
            batch_size=_ROWS,
            lr=_LR,
            surprise_alpha=alpha,
            surprise_max_weight=max_weight,
            mlh_cap_row_weight=0.0,
            cap_plies=_CAP,
            seed=0,
        ),
    ).run(batch)

    mirror = syn.tiny_net()
    mirror.train()
    outputs = outputs_from_net(mirror(batch["observation"]))
    errors = _mlh_squared_errors(batch, outputs)

    kl_base = (1.0 + alpha * batch["policy_kl"].to(torch.float64)).clamp(
        min=1.0, max=max_weight
    )
    assert not torch.equal(kl_base, torch.ones_like(kl_base)), (
        "the surprise base is all ones, so this node could not tell "
        "`kl_base * mlh` from `mlh` alone"
    )
    censoring = _weights_for(rows, 0.0).to(torch.float64)

    def weighted(weights: torch.Tensor) -> float:
        """``sum(w * e) / sum(w)`` over the batch."""
        return float(((errors * weights).sum() / weights.sum()).item())

    assert metrics[HEAD_MLH] == pytest.approx(
        weighted(kl_base * censoring), rel=1e-5
    )
    assert metrics[HEAD_MLH] != pytest.approx(weighted(kl_base), rel=1e-5), (
        "the moves-left head took the surprise base alone; the censoring "
        "never arrived"
    )
    assert metrics[HEAD_MLH] != pytest.approx(weighted(censoring), rel=1e-5), (
        "the moves-left head took the censoring alone; the multiplier replaced "
        "the assembler's base instead of composing with it"
    )


def test_targets_from_batch_still_passes_the_moves_left_target_itself() -> None:
    """The column is read twice, for two different purposes, and both survive.

    ``mlh_target`` is the head's *target* and -- with ``ply`` -- half of the
    censoring predicate's input. A refactor that moved the column out of
    :data:`~komugi.train.wiring.TARGET_KEYS` while wiring the predicate would
    leave the head scoring against nothing while the weights still looked
    right.
    """
    batch = stack_samples(_mixed_rows())
    targets = targets_from_batch(batch)
    assert targets[MLH_TARGET] is batch[MLH_TARGET]
