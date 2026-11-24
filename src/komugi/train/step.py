"""One optimiser step, and the loop that runs a few thousand of them.

Everything below this module is a component with no opinion about training:
the assembler scores tensors, the window draws batches, the checkpoint writer
persists state dicts. This is where they become a *step* -- forward, loss,
backward, update -- and where the two decisions that a training run cannot
avoid making are written down rather than left to a call site.

The optimiser is built inside, and it is AdamW
----------------------------------------------
:class:`TrainStep` constructs ``torch.optim.AdamW(net.parameters(), lr=...,
weight_decay=...)`` itself; no optimiser enters from outside in P0a. That is a
deliberate narrowing. An injected optimiser is the kind of seam that looks free
and is not: the optimiser's state is half of what a checkpoint has to carry, so
a resumed run would be reproducible only if whoever rebuilt the optimiser
matched what the original run happened to be handed. One constructor, one
place, one thing to read out of :class:`StepConfig`.

AdamW rather than Adam because decoupled weight decay is the only form of it
that means what it says on an adaptive optimiser -- L2-in-the-loss decay gets
divided by the per-parameter second-moment estimate, so it decays the
parameters that need it least. ``weight_decay=0.0`` by default: P0a's first
run measures without it.

Determinism
-----------
:meth:`TrainStep.run` is a deterministic function of (net state, batch): the
network has no dropout and no other stochastic layer (see
:mod:`komugi.net.heads`), the loss is arithmetic, and AdamW's update is a
closed form of the gradient and its own state. ``StepConfig.seed`` therefore
does not seed anything here -- there is nothing to seed. It exists so the
config *identifies* a run: it is stamped into checkpoint metadata, and it is
what :func:`training_iteration` derives its batch-sampling stream from when a
caller passes it through as ``rng_seed``. Anything that would make a step
non-deterministic (a stochastic layer, a non-deterministic CUDA kernel) has to
argue with this docstring first.

BatchNorm is the one piece of state a step mutates besides the parameters:
``net.train()`` puts the norm layers in batch-statistics mode and updates their
running estimates. That is training behaviour, not non-determinism -- the same
net and the same batch produce the same update every time -- but it does mean
running the *same* batch twice is not the same as running it once, which the
descent smoke test in ``tests/train/test_step_unit.py`` accounts for.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from komugi.telemetry.journal import write_record
from komugi.telemetry.records import IterationMetrics
from komugi.train.checkpoint import plane_meta, save_checkpoint
from komugi.train.losses import (
    HEAD_MLH,
    HEAD_POLICY,
    HEAD_SOFT_POLICY,
    HEAD_WDL,
    LossAssembler,
)
from komugi.train.shards import stack_samples
from komugi.train.window import SampleSource
from komugi.train.wiring import (
    cap_row_weights,
    fast_row_weights,
    mlh_row_weights,
    outputs_from_net,
    surprise_weights_from_kl,
    targets_from_batch,
)

OBSERVATION_KEY: Final[str] = "observation"
"""The batch field the network is fed, spelled as :data:`komugi.selfplay.records.SCHEMA` does."""

POLICY_KL_KEY: Final[str] = "policy_kl"
"""The batch field surprise weights are computed from (design D4's stored scalar)."""

SEARCH_MODE_KEY: Final[str] = "search_mode"
"""The batch field the fast-row weight is keyed on (KOM-036 §1.2).

A **literal**, spelled as :data:`komugi.selfplay.records.SCHEMA` spells it, in
the same idiom as :data:`OBSERVATION_KEY` and :data:`POLICY_KL_KEY` above --
deliberately not imported from ``records``. ``train/__init__`` loads this module
before it loads :mod:`komugi.train.wiring`, so an entry through ``import
komugi.selfplay`` would reach ``records`` -> ``komugi.train.shards`` ->
``train/__init__`` -> here -> ``from komugi.selfplay.records import ...`` with
``records`` frozen at its own import line and ``SCHEMA`` not yet bound. The mode
*codes* have a leaf to live in (:mod:`komugi.modes`); a column name is a string
this module states.
"""

TERMINATION_KEY: Final[str] = "termination"
"""The batch field the cap-row value weight is keyed on (KOM-160).

A **literal**, for exactly the reason :data:`SEARCH_MODE_KEY` above is one, and
required of every batch for exactly the reason that column is: the multiplier
is applied at every ``cap_row_value_weight`` including the default ``1.0``, so
the arms share one code path. The termination *codes* have their own leaf
(:mod:`komugi.terminations`); the column name is a string this module states.
"""

PLY_KEY: Final[str] = "ply"
MLH_TARGET_KEY: Final[str] = "mlh_target"
"""The two further batch fields the moves-left cap-row weight is keyed on
(KOM-159 D4).

Literals, for the reason :data:`SEARCH_MODE_KEY` is one, and required of every
batch for the reason :data:`TERMINATION_KEY` is: the multiplier is applied at
every ``mlh_cap_row_weight`` including ``1.0``, so the arms share one code
path. ``mlh_target`` is *also* a target the assembler consumes -- it is named
here as a **column** because this module reads it to reconstruct the row's
game length, which is a provenance question and not a loss one.
"""

DEFAULT_CAP_PLIES: Final[int] = 512
"""Fallback for :attr:`StepConfig.cap_plies`: the engine's turn cap today.

A campaign passes :meth:`komugi.engine.game.Game.max_turns` explicitly
(:func:`komugi.campaign.loop.run_campaign`), so this default is only ever the
one a hand-built :class:`StepConfig` gets. It is a literal rather than an
import because importing the engine -- or :mod:`komugi.selfplay.records`, which
derives ``MLH_SCALE`` from the same constant -- from the trainer closes the
cycle :mod:`komugi.train.wiring`'s docstring describes, and because the value
is only used to *reconstruct* a length that a shard already recorded. A test
holds it to :data:`komugi.engine.observation.MAX_TURNS`, so the next cap bump
(KOM-165 raised it from 256) is red rather than silent.
"""

TOTAL_KEY: Final[str] = "total"
"""Key of the weighted total in the metrics :meth:`TrainStep.run` returns.

The per-head keys alongside it are the assembler's own
(``policy``, ``soft_policy``, ``wdl``, ``mlh``, and the staged
``uncertainty``, ``ownership``, ``short_term_value``, ``draft_outcome``), so a
metrics dict and a loss
breakdown are the same vocabulary. ``"total"`` cannot collide with a head name
-- there is no head called total -- but the constant exists so a telemetry
consumer can name it rather than spell it.
"""

_METRIC_PREFIX: Final[str] = "loss_"
"""Prefix under which step metrics travel in :attr:`IterationMetrics.extras`."""


@dataclass(frozen=True, slots=True)
class StepConfig:
    """Everything one optimiser step needs that is not the net or the loss.

    Args:
        batch_size: Samples per step. Must be positive; a zero-length batch
            makes every weighted mean a division by zero.
        lr: AdamW's learning rate. Must be positive -- a zero learning rate is
            a run that trains nothing while looking exactly like one that does,
            which is a thing worth being unable to configure by accident.
        weight_decay: AdamW's decoupled decay. Non-negative; 0.0 is P0a.
        surprise_alpha: Strength of the surprise weighting computed from the
            batch's stored ``policy_kl``. ``0.0`` (the default) means the step
            passes no explicit weights at all -- see :meth:`TrainStep.run`.
        surprise_max_weight: Ceiling of that weighting; at least 1, the clip's
            lower bound.
        fast_policy_weight: The KOM-036 A/B's one knob -- what a fast row's
            policy and soft-policy loss is multiplied by, in ``[0, 1]``. The
            arms are ``0.125`` (down-weight, derived as ``gumbel_n_fast /
            gumbel_n_full``) and ``0.0`` (PCR-style discard); ``1.0`` -- the
            default -- is today's behaviour exactly, and takes the *same* code
            path rather than bypassing the seam, because the multiplier is
            then all ones and ``weights * 1.0`` is the identity. WDL and MLH
            are never touched by it: a fast move still advances the game, so
            its outcome-derived targets are as good as a full move's.
        cap_row_value_weight: KOM-160's one knob -- what the **WDL** loss of a
            row from a drawn (in practice turn-capped) game is multiplied by,
            in ``[0, 1]``. Campaign 4D's pre-registered arm is ``0.25``, its
            fallback ``0.6``; ``1.0`` -- the default -- is today's behaviour
            exactly, and takes the *same* code path rather than bypassing the
            seam, for the reason ``fast_policy_weight`` gives above. It reaches
            ``HEAD_WDL`` and no other head: a cap row's policy target is a
            real search result and its moves-left target is the plies-to-cap
            the MLH head exists to learn, so every other head's treatment of
            cap rows stays a decision that head's own bring-up makes.
        mlh_cap_row_weight: KOM-159 D4's one knob -- what the **moves-left**
            loss of a row from a turn-capped game is multiplied by, in
            ``[0, 1]``. ``0.0`` -- the default -- censors those rows, which is
            what campaign 6 launches with: a stall's ``mlh_target`` is the
            plies until the *engine's* adjudication, a right-censored
            observation of the quantity the head predicts rather than a label
            for it (see :func:`~komugi.train.wiring.mlh_row_weights`). ``1.0``
            is the pre-KOM-159 arithmetic and takes the *same* code path, for
            the reason ``cap_row_value_weight`` gives above. It reaches
            :data:`~komugi.train.losses.HEAD_MLH` and no other head, so the
            WDL head's treatment of cap rows stays ``cap_row_value_weight``'s
            decision and the two knobs compose independently.
        cap_plies: The engine's turn cap the batch's shards were played under
            -- :meth:`komugi.engine.game.Game.max_turns`, which the campaign
            loop passes explicitly. Must be at least 1. It is a plain ``int``
            rather than a constant read from :mod:`komugi.selfplay.records`
            because that import closes the cycle
            :mod:`komugi.train.wiring`'s docstring describes; see
            :data:`DEFAULT_CAP_PLIES` for why the fallback is a literal.
        mlh_utility_lambda: The **search-side** MLH utility strength this
            iteration's self-play played at (KOM-159 D5/D7,
            :attr:`komugi.config.SelfPlayConfig.mlh_utility_lambda`). The
            trainer does not consume it and cannot: it is a property of the
            games, not of the optimiser step.

            It is **not journalled from here** (KOM-171, KOM-159a finisher
            review F5). Campaign 6 turns the utility on at a pre-registered
            iteration and that switch must be readable off the journal
            (design review F10), but the campaign boundary is the single
            writer of the key -- :data:`komugi.campaign.loop.MLH_UTILITY_LAMBDA`
            on every iteration's campaign record -- because one key stamped by
            both the trainer's record and the loop's is one key whose value a
            reader cannot attribute to either half. The field survives as the
            declared shape of what the loop hands the trainer, and is
            validated in ``[0, 1]`` here so a nonsense arm is refused before
            an iteration trains under it.
        seed: Identifies the run. Not consumed by the step itself (see the
            module docstring's determinism section); carried into checkpoint
            metadata.
    """

    batch_size: int
    lr: float
    weight_decay: float = 0.0
    surprise_alpha: float = 0.0
    surprise_max_weight: float = 5.0
    fast_policy_weight: float = 1.0
    cap_row_value_weight: float = 1.0
    mlh_cap_row_weight: float = 0.0
    cap_plies: int = DEFAULT_CAP_PLIES
    mlh_utility_lambda: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a configuration that cannot describe a training step."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not self.lr > 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.surprise_alpha < 0.0:
            raise ValueError(
                f"surprise_alpha must be >= 0, got {self.surprise_alpha}"
            )
        if self.surprise_max_weight < 1.0:
            raise ValueError(
                "surprise_max_weight must be >= 1 (the clip's lower bound), "
                f"got {self.surprise_max_weight}"
            )
        if not 0.0 <= self.fast_policy_weight <= 1.0:
            # `not <=` rather than `<`/`>` so a NaN arm is refused here rather
            # than discovered as a policy head whose loss is NaN from the first
            # batch. Above 1 is refused too: the knob down-weights fast rows,
            # and a value above the full-row weight is a different experiment.
            raise ValueError(
                f"fast_policy_weight must be in [0, 1], got "
                f"{self.fast_policy_weight}"
            )
        if not 0.0 <= self.cap_row_value_weight <= 1.0:
            # `not <=` for the same reason: a NaN arm refused here rather than
            # discovered as a WDL head whose loss is NaN from the first batch.
            # Above 1 is refused too -- the knob down-weights draw rows, and
            # up-weighting them is a different experiment.
            raise ValueError(
                f"cap_row_value_weight must be in [0, 1], got "
                f"{self.cap_row_value_weight}"
            )
        if not 0.0 <= self.mlh_cap_row_weight <= 1.0:
            # `not <=` for the third time and the third reason of the same
            # shape: a NaN arm refused here rather than discovered as a
            # moves-left head whose loss is NaN from the first batch. Above 1
            # is refused too -- the knob censors cap rows, and up-weighting the
            # censored observation is a different experiment.
            raise ValueError(
                f"mlh_cap_row_weight must be in [0, 1], got "
                f"{self.mlh_cap_row_weight}"
            )
        if self.cap_plies < 1:
            # A cap of zero would make `total >= cap_plies` true for every row
            # and censor a whole batch's moves-left head; a negative one is a
            # corrupt config. Neither is a turn cap any engine can play under.
            raise ValueError(f"cap_plies must be >= 1, got {self.cap_plies}")
        if not 0.0 <= self.mlh_utility_lambda <= 1.0:
            # Neither consumed nor journalled from here (see the field's docs:
            # the boundary is the single writer) -- but a value nothing reads is
            # exactly the value nobody notices is nonsense, and a NaN reaching
            # the boundary's own stamp would take the telemetry write down
            # (`IterationMetrics` refuses a non-finite extra) at the end of an
            # iteration that had already trained.
            raise ValueError(
                f"mlh_utility_lambda must be in [0, 1], got "
                f"{self.mlh_utility_lambda}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")


class NonFiniteUpdateError(RuntimeError):
    """A step produced a non-finite loss or gradient and was aborted.

    Raised by :meth:`TrainStep.run` **after** ``backward()`` and **before**
    ``optimizer.step()``, with the gradients cleared, so the parameters and the
    optimiser's moment estimates are exactly what they were when the batch
    arrived. That ordering is the whole guarantee: NaN reaching ``step()`` is
    not a bad update, it is a permanent one -- AdamW's moments become NaN with
    the parameters, every subsequent batch produces NaN, and the run trains to
    nothing while the loss curve simply stops (review TS-01).

    **BatchNorm running statistics are the accepted exception.** ``net.train()``
    updates them during the forward, before there is a loss to inspect, so a
    refused step leaves the norm layers' running mean and variance mutated by
    the batch that poisoned it. Un-mutating them would mean snapshotting every
    buffer on every step to pay for the rare one, and the statistics are an
    exponential average that one batch barely moves. The policy is therefore
    **abort and resume**: a caller that catches this aborts the run and resumes
    from the last committed healthy checkpoint, which carries the buffers as
    they were when it was written. Catching it to *continue* training would
    keep the poisoned statistics.

    A ``RuntimeError`` subclass so it reads as what it is -- a run that cannot
    proceed -- and so an existing ``except RuntimeError`` around a training loop
    still sees it.
    """


def _refuse_nonfinite_update(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer, total: torch.Tensor
) -> None:
    """Raise :class:`NonFiniteUpdateError` if this backward cannot be applied.

    Checks the loss and every gradient that participates -- a parameter whose
    ``.grad`` is ``None`` took no part in this batch and has nothing to check.
    Clears the gradients before raising, so the aborted step leaves nothing
    behind for the next caller of ``step()`` to apply.

    One device sync in the healthy path: the per-parameter ``isfinite().all()``
    reductions stay on device and only the single stacked ``all()`` crosses.
    The per-parameter search that names the offender runs only on the batch
    that is already being refused.

    Args:
        net: The network whose named parameters carry the gradients.
        optimizer: The optimiser to clear before raising.
        total: The scalar loss ``backward()`` was called on.

    Raises:
        NonFiniteUpdateError: Naming the loss value and the first parameter, in
            ``named_parameters`` order, holding a non-finite gradient.
    """
    named_grads: list[tuple[str, torch.Tensor]] = []
    for name, parameter in net.named_parameters():
        gradient = parameter.grad
        if gradient is not None:
            named_grads.append((name, gradient))
    loss_value = float(total.detach().item())
    finite_flags: list[bool] = []
    if named_grads:
        stacked = torch.stack([torch.isfinite(grad).all() for _, grad in named_grads])
        grads_are_finite = bool(stacked.all())
        if not grads_are_finite:
            finite_flags = [bool(flag) for flag in stacked.tolist()]
    else:
        grads_are_finite = True
    if math.isfinite(loss_value) and grads_are_finite:
        return

    offender = "none (the loss itself is non-finite)"
    if not grads_are_finite:
        offender = named_grads[finite_flags.index(False)][0]
    # Clear FIRST: the raise must not leave a non-finite gradient sitting on the
    # parameters for whatever calls step() next.
    optimizer.zero_grad(set_to_none=True)
    raise NonFiniteUpdateError(
        f"refusing to apply this update: loss={loss_value}, first non-finite "
        f"gradient at parameter {offender!r}. Gradients have been cleared and "
        f"optimizer.step() was NOT called, so the parameters and the moment "
        f"estimates are unchanged. Resume from the last committed checkpoint "
        f"(review TS-01); BatchNorm running statistics moved by this batch's "
        f"forward are not rolled back"
    )


class TrainStep:
    """A net, a loss and an optimiser, taking one batch at a time.

    Holds the optimiser -- and therefore the moment-estimate state that makes
    Adam Adam -- so a step is a method call and the state survives between
    them. One instance per run.

    Args:
        net: The network to optimise. Its parameters are handed to AdamW at
            construction, so a net whose parameters are replaced afterwards
            (rather than updated in place) needs a new :class:`TrainStep`.
        assembler: The loss. Its config decides which heads are computed, and
            therefore which keys :meth:`run` returns.
        config: Step hyperparameters.
    """

    def __init__(
        self, net: torch.nn.Module, assembler: LossAssembler, config: StepConfig
    ) -> None:
        """Bind the three, and build the AdamW the module docstring argues for."""
        self._net = net
        self._assembler = assembler
        self._config = config
        self._optimizer = torch.optim.AdamW(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    @property
    def net(self) -> torch.nn.Module:
        """Return the network being optimised."""
        return self._net

    @property
    def assembler(self) -> LossAssembler:
        """Return the loss assembler in force."""
        return self._assembler

    @property
    def config(self) -> StepConfig:
        """Return the frozen step config."""
        return self._config

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Return the internally built optimiser.

        Exposed for exactly two reasons -- checkpointing its ``state_dict``
        and inspecting it in a test. Reaching in to mutate it is how the
        injected-optimiser problem the module docstring rejects gets in
        through the window.
        """
        return self._optimizer

    def run(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Train on one `batch` and return its detached loss breakdown.

        Forward on ``batch['observation']``, translate through
        :mod:`komugi.train.wiring`, assemble, backward, step. Gradients are
        zeroed BOTH before ``backward()`` and after the update: the pre-clear
        is the guarantee -- a caller that ran its own ``backward()`` between
        steps (a gradient probe, a debug pass), or handed this step a net with
        stale grads, would otherwise silently pollute the update DIRECTION
        while the reported loss stayed correct (wave-10 review F2). The
        post-step clear is hygiene, not the defence.

        Every tensor in `batch` is moved to the net's own parameter device
        before the forward. **The net's placement decides**, and nothing here
        names a device -- the same rule
        :class:`~komugi.selfplay.inference.ShardServer` follows on the
        inference side (``no device literals``), for the same reason: a
        trainer that called ``.cuda()`` would make "where does this run" a
        property of two disagreeing modules instead of one. A caller puts the
        net where it wants the run to happen (``net.to('cuda')``) and the
        batches follow. On CPU every ``.to`` is the identity, so the CPU path
        is unchanged -- which is what makes the existing tests the regression
        for this move.

        Surprise weights are computed from ``batch['policy_kl']`` and passed
        explicitly when ``config.surprise_alpha > 0``. At ``alpha == 0`` no
        weights are passed at all, so a batch without a ``policy_kl`` column
        trains fine while the weighting is off, and the assembler's own
        (unweighted) path is the one exercised.

        ``batch['search_mode']`` is required **unconditionally**, at every
        ``config.fast_policy_weight`` including the default ``1.0``. That is
        deliberate: passing the fast-row multiplier only when the arm is
        non-trivial would give arm C its own code path, which is exactly the
        objection KOM-036 §1.2 raises against the row filter it rejected --
        the three arms must differ in one float and in nothing else.

        ``batch['termination']`` is required on the same terms and for the same
        reason (KOM-160): it keys the cap-row multiplier on the WDL head, which
        is passed at every ``config.cap_row_value_weight`` including the default
        ``1.0``, where it is an all-ones vector and the step is bit-identical to
        the pre-KOM-160 trainer.

        ``batch['ply']`` and ``batch['mlh_target']`` are required for the third
        time on the same terms (KOM-159 D4). Together with ``termination`` they
        reconstruct "this row's game reached the turn cap", which is the
        predicate the **moves-left** head's multiplier censors on; it too is
        passed at every ``config.mlh_cap_row_weight``, so ``1.0`` is the
        pre-KOM-159 arithmetic on the same code path rather than a bypass.
        ``mlh_target`` is a target as well as a column, and it is required here
        even under a recipe with the head off -- which is deliberate: which
        columns a *batch* must carry is a property of the trainer, not of the
        recipe in force, and a batch that lost the column would otherwise train
        happily until the iteration the head was switched on.

        Between ``backward()`` and ``step()`` the loss and every participating
        gradient are checked for finiteness (:func:`_refuse_nonfinite_update`).
        A step that cannot be applied raises instead of applying itself, and
        returns no metrics at all -- so a caller building a journal record from
        the returned dict writes nothing for a batch that did not train. The
        BatchNorm caveat is in :class:`NonFiniteUpdateError`'s docstring.

        Returns:
            ``{'total': ..., <head>: ...}`` as plain floats, one entry per
            enabled head plus the weighted total. Detached by construction --
            ``float()`` on a scalar tensor -- so holding the dict cannot keep a
            graph alive.

        Raises:
            ValueError: If the batch's ``policy_target`` is not a distribution
                (:func:`~komugi.train.losses.validate_policy_target`, raised
                from the assembler before any gradient exists).
            NonFiniteUpdateError: If the loss or any gradient is non-finite.
                Gradients are cleared and ``optimizer.step()`` is not called.
        """
        if OBSERVATION_KEY not in batch:
            raise KeyError(
                f"batch is missing {OBSERVATION_KEY!r} (present: {sorted(batch)})"
            )
        # The model's parameter device, read the way the inference server
        # reads it; `torch.empty(0)` is the default for a net with no
        # parameters at all, and its device is the CPU.
        device = next(self._net.parameters(), torch.empty(0)).device
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        self._net.train()
        out = self._net(batch[OBSERVATION_KEY])
        outputs = outputs_from_net(out)
        targets = targets_from_batch(batch)

        sample_weight: torch.Tensor | None = None
        if self._config.surprise_alpha > 0.0:
            if POLICY_KL_KEY not in batch:
                raise KeyError(
                    f"surprise_alpha is {self._config.surprise_alpha} but the "
                    f"batch has no {POLICY_KL_KEY!r} column (present: "
                    f"{sorted(batch)})"
                )
            sample_weight = surprise_weights_from_kl(
                batch[POLICY_KL_KEY],
                alpha=self._config.surprise_alpha,
                max_weight=self._config.surprise_max_weight,
            )

        if SEARCH_MODE_KEY not in batch:
            raise KeyError(
                f"batch has no {SEARCH_MODE_KEY!r} column (present: "
                f"{sorted(batch)}); the fast-row policy weight is keyed on it "
                f"and is applied at every fast_policy_weight, arm C included"
            )
        # UNCONDITIONALLY, at every arm. At fast_policy_weight == 1.0 this is
        # an all-ones vector and `weights * 1.0` is the identity, so arm C is
        # bit-identical to the pre-seam trainer while running the same code
        # path -- which is what makes an incorrectly composed multiplier
        # visible in the golden step rather than hidden behind a `!= 1.0`
        # bypass. `run` builds no base vector of its own: the multiplier
        # composes with whichever base the assembler resolved, so a recipe
        # with surprise weighting on keeps it on the policy head AND gets the
        # fast factor on top (KOM-036 §1.5).
        fast = fast_row_weights(
            batch[SEARCH_MODE_KEY], self._config.fast_policy_weight
        )
        if TERMINATION_KEY not in batch:
            raise KeyError(
                f"batch has no {TERMINATION_KEY!r} column (present: "
                f"{sorted(batch)}); the cap-row value weight is keyed on it "
                f"and is applied at every cap_row_value_weight, 1.0 included"
            )
        # Unconditionally, for the reason above -- and on HEAD_WDL ALONE. The
        # policy heads take `fast`, the WDL head takes `cap`, and the two
        # multipliers never meet: each composes with whichever base the
        # assembler resolved for its own head (KOM-160 §2, design review F10).
        cap = cap_row_weights(
            batch[TERMINATION_KEY], self._config.cap_row_value_weight
        )
        for column in (PLY_KEY, MLH_TARGET_KEY):
            if column not in batch:
                raise KeyError(
                    f"batch has no {column!r} column (present: {sorted(batch)}); "
                    f"the moves-left cap-row weight reconstructs the row's game "
                    f"length from it and is applied at every "
                    f"mlh_cap_row_weight, 1.0 included"
                )
        # The third multiplier, on HEAD_MLH ALONE and on the same terms as the
        # other two: unconditional, so the arms share one code path (KOM-159
        # D4). At `mlh_cap_row_weight = 1.0` it is an all-ones vector, and
        # `weights * 1.0` is the identity, so the pre-KOM-159 arithmetic is
        # reachable through the seam rather than around it. Three multipliers,
        # three head groups, one shared base -- none of them ever meet.
        mlh = mlh_row_weights(
            batch[PLY_KEY],
            batch[MLH_TARGET_KEY],
            batch[TERMINATION_KEY],
            self._config.cap_plies,
            self._config.mlh_cap_row_weight,
        )
        total, per_head = self._assembler(
            outputs,
            targets,
            sample_weight=sample_weight,
            head_sample_weight={
                HEAD_POLICY: fast,
                HEAD_SOFT_POLICY: fast,
                HEAD_WDL: cap,
                HEAD_MLH: mlh,
            },
        )
        # Pre-clear: any grads accumulated OUTSIDE run() must not steer this
        # update (wave-10 review F2).
        self._optimizer.zero_grad(set_to_none=True)
        total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
        # Between the backward and the update, which is the only place this
        # check is worth anything: after `step()` the NaN is in the parameters
        # AND in AdamW's moments, and no later batch can get it out again
        # (review TS-01).
        _refuse_nonfinite_update(self._net, self._optimizer, total)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

        metrics = {TOTAL_KEY: float(total.item())}
        for head, value in per_head.items():
            metrics[head] = float(value.item())
        return metrics


def _sample_batch(
    source: SampleSource, rng: np.random.Generator, batch_size: int
) -> dict[str, torch.Tensor]:
    """Draw `batch_size` rows uniformly **with replacement** and stack them.

    With replacement because a training step is a Monte-Carlo estimate of the
    gradient over the whole source, not a pass over a permutation of it: the
    replay window this eventually reads from is resampled many times per
    iteration, and sampling without replacement would make the batch
    distribution drift as an epoch progressed (the same argument
    :meth:`komugi.train.window.RecencyWindow.draw` makes).
    """
    rows = rng.integers(0, len(source), size=batch_size)
    return stack_samples([source[int(row)] for row in rows])


def training_iteration(
    net: torch.nn.Module,
    source: SampleSource,
    step: TrainStep,
    *,
    steps: int,
    rng_seed: int,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    journal_path: str | os.PathLike[str] | None = None,
    iteration: int = 0,
    global_step_base: int = 0,
) -> list[dict[str, float]]:
    """Run `steps` optimiser steps over `source`, then persist what happened.

    Batches are drawn uniformly with replacement (see :func:`_sample_batch`)
    from a :class:`numpy.random.PCG64` stream seeded through
    :class:`numpy.random.SeedSequence` -- never from :func:`hash`, whose
    randomisation per interpreter would make a resumed run draw a different
    sequence than the one it is resuming (the reasoning
    :mod:`komugi.selfplay.records` spells out for seed derivation generally).
    Two calls with the same `rng_seed` over the same source draw the same
    batches in the same order.

    Args:
        net: The network being trained. `step` must already hold it; it is
            taken separately because the checkpoint records *its* state dict
            and passing it makes that explicit rather than reached-through.
        source: Anything with ``__len__`` and integer ``__getitem__`` -- a
            :class:`~komugi.train.shards.ShardReader`, a
            :class:`~komugi.train.window.RecencyWindow` member, or a list of
            sample dicts.
        step: The bound net/loss/optimiser. Supplies ``config.batch_size``.
        steps: How many optimiser steps to run. Must be positive.
        rng_seed: Seed of the batch-sampling stream.
        checkpoint_dir: Where to write one checkpoint after the last step.
            ``None`` writes none, which is what a test or a smoke run wants.
            The checkpoint's step number is ``global_step_base + steps``.
        journal_path: Journal to append one telemetry record to. ``None``
            writes none. The file's parent directory must exist.
        iteration: The campaign iteration these steps belong to; stamped on
            the telemetry record and the checkpoint metadata.
        global_step_base: How many optimiser steps the run had already taken
            before this call, so the checkpoint is filed under the **globally
            monotone** step ``global_step_base + steps`` rather than under
            this call's local count. Non-negative; ``0`` -- the default --
            reproduces the single-iteration numbering every existing caller
            already has.

            This is the wave-10 review F5 fix, and F5 is worth restating
            because the failure is silent: a campaign loop calling this
            repeatedly with one `checkpoint_dir` and a constant `steps` used
            to publish every iteration under the *same* step number, and
            :func:`~komugi.train.checkpoint.save_checkpoint`'s same-step
            supersede would then **unlink** the previous iteration's file. One
            checkpoint ever existed, rollback to a prior generation was
            impossible, and nothing said so. With a base, iteration ``i``'s
            checkpoint is step ``(i + 1) * steps``: distinct names, files that
            coexist, a directory whose newest is genuinely the newest.

    Returns:
        One metrics dict per step, in order -- exactly what
        :meth:`TrainStep.run` returned each time.

    Raises:
        ValueError: If `steps` is not positive, or `source` is empty (a batch
            cannot be drawn from nothing, and the numpy error for it names
            neither).

    Telemetry mapping -- an honest approximation, not a new schema.
        :class:`~komugi.telemetry.records.IterationMetrics` was shaped for the
        *self-play* half of an iteration, and three of its five scalars
        describe games this function never sees. Rather than widen the record
        (its whole point is that anything experiment-specific goes in
        ``extras``, so adding fields is the failure mode it was built to
        prevent), the trainer records what it honestly knows:

        * ``games``, ``draw_rate``, ``avg_game_len``, ``policy_entropy`` are
          reported as **0**: this call played no games and measured no policy
          entropy. Zero is the schema's floor rather than a measurement, and a
          campaign loop that also runs self-play should write *its own*
          :class:`IterationMetrics` for the same iteration with the real
          numbers -- the journal is append-only and holds both.
        * every loss scalar of the **last** step goes into ``extras`` under a
          ``loss_`` prefix (``loss_total``, ``loss_policy``, ...), plus
          ``steps``, ``batch_size`` and ``cap_row_value_weight`` (KOM-160's
          arm, journalled because it changes what ``loss_wdl`` *means*: below
          1.0 the reported value is the cap-weighted mean, which is not
          like-for-like with another arm's), and KOM-159's
          ``mlh_cap_row_weight`` and ``cap_plies``, because they redefine
          ``loss_mlh`` the same way. The **search-side** ``mlh_utility_lambda``
          is deliberately *not* among them even though
          :class:`StepConfig` carries it: it is a property of the games, the
          campaign boundary journals it on every iteration's own record
          (:data:`komugi.campaign.loop.MLH_UTILITY_LAMBDA`, design review F10),
          and one key written by both halves is one key whose value a reader
          cannot attribute (KOM-159a finisher review F5). The last step rather
          than the mean
          because the mean of a descending curve is a number nothing else in
          the system reports; the final loss is the one comparable with the
          next iteration's.
        * **and, for every one of those loss scalars, three summaries of the
          whole iteration**: ``loss_<name>_first_decile`` and
          ``loss_<name>_last_decile`` (means over the first and last
          ``ceil(steps / 10)`` steps) and ``loss_<name>_step_std``
          (population standard deviation over every step). A staged head's
          gate reads the two decile means to decide whether its bring-up curve
          fell; the dispersion is the secondary measurement short-term value's
          gate records. Neither is recoverable from a last-step scalar -- one
          value per iteration has no dispersion -- which is why they are
          computed here, over the ``history`` this function already holds, and
          not restated downstream. The last-step scalars, ``steps`` and
          ``batch_size`` keep their exact meanings: this is a key addition,
          never a redefinition.

        The ``checkpoint_dir`` ``meta`` block stays last-step-only on purpose:
        a checkpoint records a moment, the journal records the iteration.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if global_step_base < 0:
        raise ValueError(f"global_step_base must be >= 0, got {global_step_base}")
    if len(source) < 1:
        raise ValueError("cannot sample a batch from a source with no samples")
    if step.net is not net:
        # Identity, not equality: the checkpoint records `net`'s weights while
        # every gradient update lands on `step.net`. A mismatch trains one
        # object and persists the other -- silently, with the corruption
        # surfacing only at resume (wave-10 review F1).
        raise ValueError(
            "training_iteration was given a net that is not the one its "
            "TrainStep optimises; the checkpoint would record untrained "
            "weights next to the trained optimizer's moment estimates"
        )

    rng = np.random.default_rng(np.random.SeedSequence(rng_seed))
    history: list[dict[str, float]] = []
    for _ in range(steps):
        batch = _sample_batch(source, rng, step.config.batch_size)
        history.append(step.run(batch))

    last = history[-1]
    extras: dict[str, float] = {
        f"{_METRIC_PREFIX}{name}": value for name, value in last.items()
    }
    # The staging ladder's condition 1 reads a *decile* statistic, and the
    # journal carries one value per loss scalar -- history[-1] -- so it is not
    # computable downstream from what `extras` held before. It is computed
    # here, at its source, over the same `history` this function already holds,
    # for EVERY loss scalar rather than for one head: which head is in bring-up
    # is a property of the campaign, not of this function. `_step_std` is the
    # secondary dispersion measurement, written in the same loop rather than in
    # a second place. This is a key ADDITION -- the last-step scalars,
    # `steps` and `batch_size` are untouched, so every existing reader of
    # `extras` is unaffected.
    decile = math.ceil(len(history) / 10)
    for name in last:
        series = [record[name] for record in history]
        extras[f"{_METRIC_PREFIX}{name}_first_decile"] = statistics.fmean(
            series[:decile]
        )
        extras[f"{_METRIC_PREFIX}{name}_last_decile"] = statistics.fmean(
            series[-decile:]
        )
        extras[f"{_METRIC_PREFIX}{name}_step_std"] = statistics.pstdev(series)
    extras["steps"] = float(steps)
    extras["batch_size"] = float(step.config.batch_size)
    # KOM-160's arm, in the JOURNAL as well as the checkpoint meta below. It is
    # in both because this one changes what `loss_wdl` MEANS: under an arm below
    # 1.0 the reported WDL loss is the cap-weighted mean, so it is not
    # like-for-like with another campaign's, and a reader of the series has to
    # be able to recover the arm from the same append-only file the series is
    # in. `fast_policy_weight` is checkpoint-only because it leaves every
    # reported number's definition alone.
    extras["cap_row_value_weight"] = float(step.config.cap_row_value_weight)
    # KOM-159's two, on the same wire and for the same reason. Below 1.0 the
    # journalled `loss_mlh` is the *censored* mean -- computed over the decisive
    # and repetition rows only -- so it is not like-for-like with an uncensored
    # one, and `cap_plies` rides beside it because the censoring predicate is
    # meaningless without the cap it was evaluated against (a series that
    # straddles a cap bump is two different statistics under one name).
    extras["mlh_cap_row_weight"] = float(step.config.mlh_cap_row_weight)
    extras["cap_plies"] = float(step.config.cap_plies)
    # The search-side lambda is NOT stamped here (KOM-171, KOM-159a finisher
    # review F5): it describes the games, and the campaign boundary -- which is
    # the only place that knows the self-play config was the one those games
    # were played under -- writes it on every iteration's campaign record
    # (`komugi.campaign.loop.MLH_UTILITY_LAMBDA`). Two writers of one key made
    # the trainer/boundary namespaces overlap, which is the invariant that keeps
    # a journal reader able to say which half of an iteration reported a number.

    if checkpoint_dir is not None:
        # The loss extras keep their metric prefix; "steps"/"batch_size" are
        # written ONCE as ints (the **extras spread previously overrode the
        # int literals with the journal's floats -- wave-10 review F4).
        meta: dict[str, Any] = {
            f"{_METRIC_PREFIX}{name}": value for name, value in last.items()
        }
        meta.update(
            iteration=iteration,
            steps=steps,
            global_step_base=global_step_base,
            rng_seed=rng_seed,
            config_seed=step.config.seed,
            lr=step.config.lr,
            weight_decay=step.config.weight_decay,
            surprise_alpha=step.config.surprise_alpha,
            # The KOM-036 arm, stamped beside the other knob that changes what
            # a curve means: a trainer checkpoint names the arm that produced
            # it, and `reconcile_ema` reads this same value back off the source
            # checkpoint so the served EMA cannot claim a different one.
            fast_policy_weight=step.config.fast_policy_weight,
            # KOM-160's arm, beside it and for the same reason -- a checkpoint
            # names the arm that produced it. This one is journalled too (see
            # `extras` above): it redefines `loss_wdl`, which the journal
            # carries and the checkpoint does not have to be read to interpret.
            cap_row_value_weight=step.config.cap_row_value_weight,
            # KOM-159's arm, beside it and for the same reason: a checkpoint
            # names the arm that produced it, and this one decides which rows
            # the moves-left head ever saw.
            mlh_cap_row_weight=step.config.mlh_cap_row_weight,
            batch_size=step.config.batch_size,
            # The architecture facts a loader must agree with before it may
            # call load_state_dict (KOM-002 §5.3). A checkpoint carried no
            # architecture metadata at all before this: an old-plane-count
            # checkpoint died as a raw torch "size mismatch for stem.weight"
            # traceback, which named neither number nor the reason.
            # save_checkpoint stamps these for every writer, so this is the
            # trainer agreeing with the codec rather than the only thing
            # putting them on disk -- one source, spelled once.  The net's own
            # state dict goes in because ``in_planes`` describes THESE weights
            # (review F5): stating the encoder's number here would override the
            # codec's stamp with a claim about the process instead.
            **plane_meta(net.state_dict()),
        )
        save_checkpoint(
            checkpoint_dir,
            global_step_base + steps,
            net.state_dict(),
            step.optimizer.state_dict(),
            meta,
        )

    if journal_path is not None:
        write_record(
            journal_path,
            IterationMetrics(
                iteration=iteration,
                games=0,
                draw_rate=0.0,
                avg_game_len=0.0,
                policy_entropy=0.0,
                extras=extras,
            ),
        )
    return history
