"""The oracle: a training iteration on a solvable task must actually learn it.

Every other test in this directory checks that the trainer is *wired* -- that
keys line up, that files appear, that a weight is honoured. This one checks
the only thing that matters underneath all of that: point the real
:class:`TrainStep` at data whose answer is a deterministic function of the
input, and the net must end up knowing the answer.

The task is :mod:`synthetic`'s: plane 0 is hot at one square, and the policy
target is one-hot at that square's action id; planes 1-3 encode the WDL class
as constant planes. Both rules are learnable by construction, so a failure
here is a statement about the training step, never about the difficulty of
Gungi.

Five claims, one training run (the run is cached, so the tests share it):

(a) the reported total loss falls by at least 30% from the mean of the first
    five steps to the mean of the last five;
(b) afterwards the net's ``argmax`` policy agrees with the rule on at least
    75% of the rows -- from an untrained baseline pinned below 50% (NOT chance:
    the equivariant 1x1 readout answers every row with the same filter);
(b2) and its ``argmax`` WDL class agrees on at least 90%, which pins the
    (win, draw, loss) index order through *training* rather than through the
    loss's reward direction alone;
(c) the ``soft_policy`` head's parameters moved. This is the wave-8 F1
    regression at the behavioural level: while the assembler read its
    sharpened target off ``policy_logits``, the 8x-weighted term trained the
    *main* head and :class:`~komugi.net.heads.SoftPolicyHead` received no
    gradient at all -- silently, at full forward cost. Parameters, not
    ``state_dict``: BatchNorm running statistics move on any forward pass and
    would mask exactly this bug;
(d) every number the step reports is finite, on every step.

The thresholds are deliberately far from the achieved values, and every seed
is fixed: this test must never flake, because a flaky learning-signal test is
one that gets disabled and then the campaign trains on nothing.

``_STEPS`` is 120 rather than the ~60 the task sketched, for the same reason:
120 steps of a 32-row memorisation task on a one-block net costs about a
second, and buying a wide margin on (a) and (b) at that price is the right
trade.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pytest
import torch

import synthetic as syn
from komugi.train import losses

wiring = pytest.importorskip("komugi.train.wiring")
if not hasattr(wiring, "P0A_RECIPE"):
    pytest.skip(
        "komugi.train.wiring is present but exports no P0A_RECIPE: the "
        "KOM-030 training-step contract has not landed yet",
        allow_module_level=True,
    )
step_module = pytest.importorskip("komugi.train.step")

_ROWS: Final[int] = 32
_STEPS: Final[int] = 120
_LR: Final[float] = 3e-3
_SEED: Final[int] = 20_260_823
_EDGE: Final[int] = 5
"""How many steps at each end of the run the loss comparison averages over."""

_REQUIRED_DROP: Final[float] = 0.30
_REQUIRED_ACCURACY: Final[float] = 0.75
_REQUIRED_WDL_ACCURACY: Final[float] = 0.90
"""The WDL rule is a constant plane read through a (mean, max) pool -- the
easiest thing in the dataset -- and the wave-10 reviewer measured 1.0."""
_CHANCE_ACCURACY: Final[float] = 1.0 / syn.ACTIONS

_TOTAL_KEYS: Final[tuple[str, ...]] = ("total", "loss", "total_loss")
"""Names the step's report may give the weighted total, in preference order."""


@dataclass(frozen=True)
class _TrainedRun:
    """Everything the four tests need out of one training iteration."""

    metrics: list[dict[str, float]]
    accuracy: float
    wdl_accuracy: float
    moved: frozenset[str]


def _total_of(record: Mapping[str, float]) -> float:
    """Return the weighted total loss out of one step's report.

    The contract pins ``run(batch) -> dict[str, float]`` but not the key the
    total travels under, so this accepts the plausible spellings and fails
    loudly -- naming what it did find -- rather than silently asserting
    against a per-head number that happens to be present.
    """
    for key in _TOTAL_KEYS:
        if key in record:
            return float(record[key])
    raise AssertionError(
        f"the step reported {sorted(record)}, none of which is a total loss; "
        f"expected one of {list(_TOTAL_KEYS)}"
    )


def _expected_outcomes(count: int) -> torch.Tensor:
    """Return the ``(count,)`` int64 vector of the WDL rule's classes.

    ``syn.rule_outcome`` in (win, draw, loss) index order -- the same order
    :class:`~komugi.net.model.NetOutput` documents for ``wdl_logits`` and the
    schema's ``wdl_target`` carries. That agreement is what makes the accuracy
    assertion below a *behavioural* pin on the index semantics: a net trained
    against a permuted target would still drive its loss down and would fail
    here, on the argmax, at 33%.
    """
    return torch.tensor(
        [syn.rule_outcome(index) for index in range(count)], dtype=torch.int64
    )


@functools.cache
def _trained_run() -> _TrainedRun:
    """Train once and return what happened. Cached: the four tests share it."""
    torch.manual_seed(_SEED)
    rows = syn.toy_rows(_ROWS)
    net = syn.tiny_net()
    trainer: Any = step_module.TrainStep(
        net,
        losses.LossAssembler(wiring.P0A_RECIPE),
        step_module.StepConfig(batch_size=_ROWS, lr=_LR, seed=_SEED),
    )
    before = syn.parameter_snapshot(net)

    # Self-validating oracle (wave-10 review F3): the untrained baseline is
    # NOT chance (1/81) -- the 1x1-conv policy readout is translation-
    # equivariant, so a random net answers the plane-0 bump with the SAME
    # filter at every row and initial accuracy is a per-seed coin flip
    # correlated across rows (measured ~28% at this seed). The oracle only
    # discriminates while the untrained net stays under the 75% bar, so pin
    # the premise: if a future seed/init change pushes the baseline past 0.5,
    # this fails loudly instead of the suite passing a trainer that trains
    # nothing.
    net.eval()
    with torch.no_grad():
        untrained: Any = net(syn.observations(rows))
        untrained_hits = int(
            (torch.argmax(untrained.policy_logits, dim=-1) == syn.expected_actions(_ROWS))
            .sum()
            .item()
        )
    assert untrained_hits / _ROWS < 0.5, (
        f"the UNTRAINED net already matches the rule on {untrained_hits}/{_ROWS} "
        f"rows; the 75% bar no longer discriminates -- change the seed or the "
        f"synthetic rule"
    )
    net.train()

    metrics: list[dict[str, float]] = step_module.training_iteration(
        net, rows, trainer, steps=_STEPS, rng_seed=_SEED
    )
    moved = syn.moved_parameters(net, before)

    # Evaluation mode on purpose: this is how the net is used in self-play, so
    # a policy that is only correct under batch statistics has not learned the
    # rule in the sense that matters.
    net.eval()
    with torch.no_grad():
        outputs: Any = net(syn.observations(rows))
        chosen = torch.argmax(outputs.policy_logits, dim=-1)
        wdl_chosen = torch.argmax(outputs.wdl_logits, dim=-1)
    hits = int((chosen == syn.expected_actions(_ROWS)).sum().item())
    wdl_hits = int((wdl_chosen == _expected_outcomes(_ROWS)).sum().item())
    return _TrainedRun(
        metrics=metrics,
        accuracy=hits / _ROWS,
        wdl_accuracy=wdl_hits / _ROWS,
        moved=moved,
    )


def test_the_iteration_reports_one_record_per_step() -> None:
    """Enough records to average five at each end without them overlapping."""
    metrics = _trained_run().metrics
    assert len(metrics) >= 2 * _EDGE, (
        f"training_iteration(steps={_STEPS}) returned {len(metrics)} records; "
        f"the loss-curve assertion needs at least {2 * _EDGE} to compare "
        "disjoint ends of the run"
    )


def test_the_total_loss_falls_over_the_iteration() -> None:
    """(a) The weighted total drops by >= 30% from the run's start to its end."""
    metrics = _trained_run().metrics
    first = [_total_of(record) for record in metrics[:_EDGE]]
    last = [_total_of(record) for record in metrics[-_EDGE:]]
    start = sum(first) / len(first)
    end = sum(last) / len(last)

    assert start > 0.0, f"the run began at a total loss of {start}"
    drop = (start - end) / start
    assert drop >= _REQUIRED_DROP, (
        f"total loss went {start:.4f} -> {end:.4f} over {len(metrics)} steps, "
        f"a drop of {drop:.1%}; the task is one-hot targets on a fixed 32-row "
        f"dataset, so at least {_REQUIRED_DROP:.0%} is expected"
    )


def test_the_trained_policy_matches_the_synthetic_rule() -> None:
    """(b) The argmax policy agrees with the rule on >= 75% of the rows."""
    accuracy = _trained_run().accuracy
    assert accuracy >= _REQUIRED_ACCURACY, (
        f"the trained policy picked the rule's action on {accuracy:.1%} of "
        f"{_ROWS} rows (chance is {_CHANCE_ACCURACY:.1%}); the target is "
        f"one-hot at the square where input plane 0 is hot"
    )


def test_the_trained_wdl_head_matches_the_synthetic_outcome() -> None:
    """(b2) The WDL argmax agrees with the rule on >= 90% of the rows.

    The wave-10 review's untested-but-promised #2: the synthetic rows carry a
    learnable WDL rule (constant plane ``1 + outcome``, which is exactly what
    survives the head's (mean, max) pooling) and nothing consumed it -- the
    reviewer measured 100% and the suite asserted nothing. 90% is the bar
    because the rule is learnable by construction and chance is 33%.

    What this pins that the wiring tests cannot: (win, draw, loss) index order
    through **training**. The existing pins are static -- the net's field, the
    assembler's key, the schema's column -- and each would survive a
    consistent permutation. This one only passes if the class the *data* calls
    ``outcome`` is the class the *trained net* puts its mass on.
    """
    accuracy = _trained_run().wdl_accuracy
    assert accuracy >= _REQUIRED_WDL_ACCURACY, (
        f"the trained WDL head picked the rule's class on {accuracy:.1%} of "
        f"{_ROWS} rows (chance is {1.0 / syn.WDL_SIZE:.1%}); the target is "
        f"one-hot at the class whose constant plane is raised in the "
        f"observation"
    )


def test_the_soft_policy_head_receives_gradient() -> None:
    """(c) The 8x-weighted auxiliary head really is the head being trained."""
    moved = _trained_run().moved
    assert "soft_policy.conv.weight" in moved and "soft_policy.conv.bias" in moved, (
        "SoftPolicyHead's output convolution did not move during training: "
        "the soft-policy term is still being scored against some other head's "
        "logits (the wave-8 F1 rewire), so the 8x-weighted target trains the "
        f"wrong parameters. Parameters that did move: {sorted(moved)}"
    )
    # ... and the main policy head is still being trained too, so a "fix" that
    # merely swapped which head is starved would not pass here either.
    assert "policy.conv.weight" in moved, (
        "PolicyHead's output convolution did not move: the primary policy "
        "term reached no parameters"
    )


def test_every_reported_loss_is_finite() -> None:
    """(d) No step reports a NaN or an infinity, in any head, at any point."""
    for step_index, record in enumerate(_trained_run().metrics):
        assert record, f"step {step_index} reported nothing"
        for name, value in record.items():
            assert math.isfinite(float(value)), (
                f"step {step_index} reported {name}={value}"
            )
