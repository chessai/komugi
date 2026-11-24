"""One training step, computed by hand first, so the recipe cannot drift quietly.

Every other test in this suite is a *relative* claim -- the loss fell, the
parameters moved, this batch weighs more than that one. A relative claim
survives almost any change to the loss recipe: swap the soft-policy weight from
8 to 1, sharpen with the wrong exponent, normalise a weighted mean by batch
size instead of weight mass, and the curves still descend and every existing
assertion still passes. This module is the absolute claim. It fixes a batch,
fixes a net, works the arithmetic out longhand, and pins the numbers.

The net is built here rather than taken from :mod:`komugi.net`, and that is the
whole trick: a bias-per-input-row affine map has *no* trunk to hand-simulate,
so the four head losses come out as small multiples of ``ln 2`` and the AdamW
update comes out in closed form. The trainer under test is entirely real --
:class:`~komugi.train.step.TrainStep`, its internally built AdamW,
:class:`~komugi.train.losses.LossAssembler` under
:data:`~komugi.train.wiring.P0A_RECIPE`, and
:func:`~komugi.train.wiring.outputs_from_net` translating a real
:class:`~komugi.net.model.NetOutput`. Only the arithmetic is small enough to
check on paper.

The arithmetic
--------------
Two samples. The observation is a one-hot selector of width 2, so sample ``i``
reads row ``i`` of every parameter matrix and the two samples never interfere.
Every row of the policy and soft-policy parameters is chosen so that its
softmax is exactly ``[1/2, 1/4, 1/8, 1/8]``, and the WDL rows so that theirs is
``[1/2, 1/4, 1/4]``.

Targets are chosen so the sharpening is exact too: ``[16/17, 1/17, 0, 0]``
raised to the 0.25 and renormalised is exactly ``[2/3, 1/3, 0, 0]``, because
``16 ** 0.25 == 2``. With ``L = ln 2``:

===========  ==================================  ===============
head         per-sample                          mean
===========  ==================================  ===============
policy       ``(18/17)L``, ``3L``                ``(69/34)L``
soft policy  ``(4/3)L``, ``3L``                  ``(13/6)L``
wdl          ``2L``, ``L``                       ``(3/2)L``
===========  ==================================  ===============

and the total is ``policy + 8 * soft_policy + wdl`` -- the P0a weights, spelled
out -- which is ``20.862745 * L`` or about ``14.460960``. Move the soft-policy
weight and the total moves by ~1.5 per unit; change the sharpening exponent and
the soft-policy head moves; normalise by batch size instead of weight mass and
every one of them moves.

The update is pinned too. AdamW's *first* step reduces to
``-lr * g / (|g| + eps)`` (the bias corrections cancel exactly at ``t = 1``), so
each parameter moves by ``lr`` in the direction opposite its gradient, and each
gradient here is ``(1/2) * head_weight * (prediction - target)`` on one row --
a sign pattern that can be written down and is written down below.

Seeds
-----
``torch.manual_seed`` is set before each run and the step's own
``StepConfig.seed`` is pinned, but the stronger statement is that this golden
does not *have* an ambient dependence to pin: the net is constructed from
literal tensors, there is no dropout, no sampling and no initialisation, and
:func:`test_the_golden_is_independent_of_the_ambient_rng` proves it by running
the same step under two different global seeds and requiring bit-identical
numbers.
"""

from __future__ import annotations

import math
from typing import Final

import pytest
import torch
from torch import nn

from komugi.net import NetOutput
from komugi.train import losses
from komugi.train.shards import stack_samples
from komugi.train.step import TOTAL_KEY, StepConfig, TrainStep
from komugi.train.wiring import P0A_RECIPE

_LN2: Final[float] = math.log(2.0)
_LR: Final[float] = 1e-3
_SEED: Final[int] = 20260830
_OTHER_SEED: Final[int] = 7

_ACTIONS: Final[int] = 4
_INPUTS: Final[int] = 2
"""One input per sample: the observation is the identity, so sample ``i`` reads
parameter row ``i`` and the two samples' gradients stay separable."""

_POLICY_ROW: Final[tuple[float, ...]] = (0.0, -_LN2, -2.0 * _LN2, -2.0 * _LN2)
"""softmax == ``[1/2, 1/4, 1/8, 1/8]`` exactly."""

_WDL_ROW: Final[tuple[float, ...]] = (0.0, -_LN2, -_LN2)
"""softmax == ``[1/2, 1/4, 1/4]`` exactly."""

_MLH_ROW: Final[tuple[float, ...]] = (0.5, 0.5)
"""The staged-off head's parameters. P0a does not score moves-left, so these
must receive no gradient and must not move -- which is an assertion, not an
aside: a head that trains while the recipe says it is off is a silent change to
what the run is optimising."""

_EXPECTED_POLICY: Final[float] = (69.0 / 34.0) * _LN2  # ~1.406686
_EXPECTED_SOFT: Final[float] = (13.0 / 6.0) * _LN2  # ~1.501819
_EXPECTED_WDL: Final[float] = 1.5 * _LN2  # ~1.039721
_EXPECTED_TOTAL: Final[float] = (
    _EXPECTED_POLICY + 8.0 * _EXPECTED_SOFT + _EXPECTED_WDL
)  # ~14.460960

_TOLERANCE: Final[float] = 1e-5
"""Relative, and generous by float32's standards: the quantities here are
sums of a handful of terms, so the working precision is worth about 1e-6."""


class _GoldenNet(nn.Module):
    """A net whose every output is a hand-chosen logit row.

    Not a toy *model* -- a toy *oracle*. The point is that ``forward`` is
    ``observation @ parameters`` with a one-hot observation, so the loss, the
    gradient and the optimiser's first step are all writable in closed form
    while the trainer above it is the real one.
    """

    def __init__(self) -> None:
        """Fix every parameter to the row the module docstring works from."""
        super().__init__()
        policy = torch.tensor([_POLICY_ROW, _POLICY_ROW], dtype=torch.float32)
        self.policy = nn.Parameter(policy.clone())
        self.soft_policy = nn.Parameter(policy.clone())
        self.wdl = nn.Parameter(
            torch.tensor([_WDL_ROW, _WDL_ROW], dtype=torch.float32)
        )
        self.mlh = nn.Parameter(torch.tensor(_MLH_ROW, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> NetOutput:
        """Return the eight heads for a one-hot `observation` of shape ``(N, 2)``.

        The four staged outputs are fixed **constants**, not parameters, and
        that is the point of this module under KOM-005b: they satisfy
        ``NetOutput``'s eight required fields (D18 gives them no defaults) so
        the golden batch runs, while contributing nothing to any parameter of
        this net. Every asserted number below is therefore unchanged from
        before the staged heads existed -- which is exactly the bit-identity
        claim D2 makes, pinned by a constant rather than by an argument.
        """
        batch = observation.shape[0]
        return NetOutput(
            policy_logits=observation @ self.policy,
            soft_policy_logits=observation @ self.soft_policy,
            wdl_logits=observation @ self.wdl,
            moves_left=observation @ self.mlh,
            uncertainty=torch.zeros(batch, dtype=torch.float32),
            ownership_logits=torch.zeros(batch, 81, 3, dtype=torch.float32),
            short_term_value=torch.zeros(batch, 3, dtype=torch.float32),
            draft_outcome_logits=torch.zeros(batch, 3, dtype=torch.float32),
        )


def _row(
    index: int, policy_target: tuple[float, ...], wdl_target: tuple[float, ...]
) -> dict[str, torch.Tensor]:
    """Return one fixed sample selecting parameter row `index`."""
    observation = torch.zeros(_INPUTS, dtype=torch.float32)
    observation[index] = 1.0
    return {
        "observation": observation,
        "policy_target": torch.tensor(policy_target, dtype=torch.float32),
        "wdl_target": torch.tensor(wdl_target, dtype=torch.float32),
        "mlh_target": torch.tensor(0.25 + 0.5 * index, dtype=torch.float32),
        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
        # A full row. `TrainStep.run` requires the column at every arm
        # (KOM-036 §1.5), and the golden constants below are unchanged by it:
        # at `fast_policy_weight = 1.0` the multiplier is exactly ones and
        # `weights * 1.0` is the identity. That this file's float-goldens
        # still hold is the arm-C bit-identity claim (acceptance #13).
        "search_mode": torch.tensor(0, dtype=torch.int64),
        # A decisive row, on the same terms and with the same consequence
        # (KOM-160): `run` requires the column at every cap_row_value_weight,
        # and at the default 1.0 the multiplier is exactly ones -- so every
        # float-golden below is unchanged, which IS the bit-identity claim.
        "termination": torch.tensor(0, dtype=torch.int64),
        # The ply this row was recorded at, required on those same terms again
        # (KOM-159 D4): the moves-left cap-row weight reconstructs the row's
        # game length from `ply` and `mlh_target` above, and `run` demands both
        # at every `mlh_cap_row_weight`. It cannot move a golden number here
        # under the P0a recipe -- the moves-left head is off, so the composed
        # HEAD_MLH vector never reaches a loss -- and the row is decisive
        # anyway, which the predicate never censors whatever the plies say.
        "ply": torch.tensor(index, dtype=torch.int64),
    }


def _batch() -> dict[str, torch.Tensor]:
    """Return the two-sample golden batch, stacked by the production stacker."""
    return stack_samples(
        [
            # 16/17 and 1/17: the ratio is 16, whose fourth root is 2, so the
            # sharpened target is exactly [2/3, 1/3, 0, 0].
            _row(0, (16.0 / 17.0, 1.0 / 17.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            _row(1, (0.0, 0.0, 1.0 / 17.0, 16.0 / 17.0), (1.0, 0.0, 0.0)),
        ]
    )


def _run(seed: int = _SEED) -> tuple[dict[str, float], _GoldenNet, _GoldenNet]:
    """Run the golden step and return the report, the trained net and a pristine one."""
    torch.manual_seed(seed)
    net = _GoldenNet()
    pristine = _GoldenNet()
    trainer = TrainStep(
        net,
        losses.LossAssembler(P0A_RECIPE),
        StepConfig(batch_size=2, lr=_LR, weight_decay=0.0, seed=_SEED),
    )
    report = trainer.run(_batch())
    return report, net, pristine


def test_the_golden_step_reports_the_hand_computed_losses() -> None:
    """Every head and the total, against constants derived on paper.

    These are the numbers a change to the loss recipe has to argue with. Each
    one is a small multiple of ``ln 2`` because the batch was built to make it
    one -- see the table in the module docstring for the derivation.
    """
    report, _, _ = _run()

    assert set(report) == {
        TOTAL_KEY,
        losses.HEAD_POLICY,
        losses.HEAD_SOFT_POLICY,
        losses.HEAD_WDL,
    }, (
        "the P0a recipe scores policy, soft policy and WDL and stages "
        f"moves-left off; this step reported {sorted(report)}"
    )
    assert report[losses.HEAD_POLICY] == pytest.approx(
        _EXPECTED_POLICY, rel=_TOLERANCE
    )
    assert report[losses.HEAD_SOFT_POLICY] == pytest.approx(
        _EXPECTED_SOFT, rel=_TOLERANCE
    ), (
        "the soft-policy head's loss is not the cross-entropy of the sharpened "
        "target: at exponent 0.25 the target [16/17, 1/17, 0, 0] becomes "
        "exactly [2/3, 1/3, 0, 0], which against logits whose softmax is "
        "[1/2, 1/4, 1/8, 1/8] costs (4/3)ln2 on this sample"
    )
    assert report[losses.HEAD_WDL] == pytest.approx(_EXPECTED_WDL, rel=_TOLERANCE)
    assert report[TOTAL_KEY] == pytest.approx(
        _EXPECTED_TOTAL, rel=_TOLERANCE
    ), (
        "the weighted total is not policy + 8 * soft_policy + wdl: either a "
        "head weight moved or the total is not the weighted sum of the "
        "reported per-head values"
    )


def test_the_total_is_the_recipes_weighted_sum_of_the_reported_heads() -> None:
    """The 8x is visible in the total, and an equal-weight total is far away."""
    report, _, _ = _run()
    weighted = (
        report[losses.HEAD_POLICY]
        + 8.0 * report[losses.HEAD_SOFT_POLICY]
        + report[losses.HEAD_WDL]
    )
    equal = (
        report[losses.HEAD_POLICY]
        + report[losses.HEAD_SOFT_POLICY]
        + report[losses.HEAD_WDL]
    )
    assert report[TOTAL_KEY] == pytest.approx(weighted, rel=_TOLERANCE)
    assert abs(report[TOTAL_KEY] - equal) > 1.0, (
        "the weighted and unweighted totals are within 1.0 of each other, so "
        "the assertion above could not tell a recipe change from a rounding one"
    )


def test_the_golden_update_is_adamws_first_step_in_the_predicted_direction() -> None:
    """Each parameter moves by ``lr`` against its gradient's sign, exactly.

    At ``t = 1`` AdamW's bias corrections cancel and the update is
    ``-lr * g / (|g| + eps)``, so the magnitude is ``lr`` to within a part in a
    million and the *direction* is the whole content. Every sign below comes
    from ``(1/2) * head_weight * (prediction - target)`` on that row:

    * policy row 0: prediction ``[.5, .25, .125, .125]`` against target
      ``[16/17, 1/17, 0, 0]`` -- under-confident on action 0 and over-confident
      on the other three, so the update raises logit 0 and lowers the rest;
    * soft-policy row 0: the same prediction against the *sharpened*
      ``[2/3, 1/3, 0, 0]`` -- which still under-shoots actions 0 **and 1**, so
      logit 1 rises here where it fell for the main head. That difference is
      the whole reason the soft target has its own head, and it is what this
      assertion pins.
    """
    _, net, pristine = _run()

    expected_signs = {
        "policy": torch.tensor(
            [[1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 1.0]]
        ),
        "soft_policy": torch.tensor(
            [[1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]
        ),
        "wdl": torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]),
    }
    trained = dict(net.named_parameters())
    initial = dict(pristine.named_parameters())

    for name, signs in expected_signs.items():
        delta = (trained[name] - initial[name]).detach()
        assert torch.equal(torch.sign(delta), signs), (
            f"{name} moved in {torch.sign(delta).tolist()}, not the predicted "
            f"{signs.tolist()}"
        )
        assert torch.allclose(
            delta.abs(), torch.full_like(delta, _LR), rtol=1e-4
        ), (
            f"{name} did not move by exactly the learning rate: AdamW's first "
            "step is -lr * g / (|g| + eps), so every coordinate with a "
            "gradient moves by lr"
        )

    staged_off = (trained["mlh"] - initial["mlh"]).detach()
    assert torch.equal(staged_off, torch.zeros_like(staged_off)), (
        "the moves-left head is staged OFF in the P0a recipe and its "
        "parameters moved: the step is training a head the recipe does not "
        "score"
    )


def test_the_golden_is_independent_of_the_ambient_rng() -> None:
    """Two global seeds, one set of numbers -- to the last bit.

    A golden that quietly reads ``torch``'s global generator is a flake
    generator: it passes on the machine it was written on and fails whenever a
    test ordering change leaves the RNG somewhere else. There is nothing
    stochastic in this step, and this is what says so.
    """
    first, net_first, _ = _run(_SEED)
    second, net_second, _ = _run(_OTHER_SEED)

    assert first == second, (
        "the same step under two global seeds reported different numbers: "
        "something in the step or the loss is reading the ambient RNG"
    )
    for name, parameter in net_first.named_parameters():
        assert torch.equal(parameter, dict(net_second.named_parameters())[name]), (
            f"{name} landed somewhere else under a different global seed"
        )
