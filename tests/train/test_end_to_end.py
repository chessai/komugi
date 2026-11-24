"""The four modules, wired together, actually train something.

This is the ticket's behavioural gate: samples go through a real
:class:`~komugi.train.shards.ShardWriter`, come back through a real
:class:`~komugi.train.shards.ShardReader`, are drawn by a real
:class:`~komugi.train.window.RecencyWindow`, are scored by a real
:class:`~komugi.train.losses.LossAssembler`, and the result is a net whose
weights moved and whose loss on a *fixed held-out batch* is lower after three
optimiser steps than before them.

The net is deliberately three ``nn.Linear`` layers defined here rather than
anything from ``komugi.net``: the trainer core must not depend on the model
package, and a test that imported one would quietly make it depend on it.

The synthetic task is learnable by construction -- the policy target is
``softmax(W @ features)`` for a fixed ``W`` -- so "the loss went down" is a
statement about the machinery working, not about a lucky initialisation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from komugi.train.checkpoint import load_checkpoint, save_checkpoint
from komugi.train.losses import (
    POLICY_LOGITS,
    WDL_LOGITS,
    LossAssembler,
    LossConfig,
)
from komugi.train.shards import FieldSpec, Schema, ShardReader, ShardWriter
from komugi.train.window import IterationRef, RecencyWindow, WindowConfig

_FEATURES = 4
_ACTIONS = 5

_SCHEMA = Schema(
    fields=(
        FieldSpec(name="features", dtype="float32", shape=(_FEATURES,)),
        FieldSpec(name="policy_target", dtype="float32", shape=(_ACTIONS,)),
        FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),
        FieldSpec(name="prior_logits", dtype="float32", shape=(_ACTIONS,)),
    )
)


class ToyNet(nn.Module):
    """A two-layer trunk with a policy head and a WDL head.

    The smallest thing that is honestly a net: a hidden layer with a
    nonlinearity, and two heads whose gradients therefore interact through
    shared parameters -- which is what makes multi-head loss assembly worth
    testing at all.
    """

    def __init__(self) -> None:
        """Build the trunk and both heads."""
        super().__init__()
        self.trunk = nn.Linear(_FEATURES, 16)
        self.policy = nn.Linear(16, _ACTIONS)
        self.wdl = nn.Linear(16, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the two heads' logits for a batch of features."""
        hidden = torch.relu(self.trunk(features))
        return {POLICY_LOGITS: self.policy(hidden), WDL_LOGITS: self.wdl(hidden)}


def _teacher() -> torch.Tensor:
    """Return the fixed matrix defining the learnable mapping."""
    generator = torch.Generator().manual_seed(11)
    return torch.randn(_ACTIONS, _FEATURES, generator=generator)


def _make_sample(
    generator: torch.Generator, teacher: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return one synthetic self-play sample.

    ``prior_logits`` stands in for what the generating net believed at
    self-play time; here it is uniform, i.e. a net that knew nothing, which
    makes every sample maximally "surprising" and exercises the weighting path
    without changing what is learnable.
    """
    features = torch.randn(_FEATURES, generator=generator)
    logits = teacher @ features
    winner = int(torch.argmax(logits[:3]).item())
    wdl = torch.zeros(3, dtype=torch.float32)
    wdl[winner] = 1.0
    return {
        "features": features,
        "policy_target": torch.softmax(logits, dim=-1),
        "wdl_target": wdl,
        "prior_logits": torch.zeros(_ACTIONS, dtype=torch.float32),
    }


def _write_iteration(directory: Path, seed: int, count: int) -> ShardReader:
    """Write `count` synthetic samples into `directory` and return a reader."""
    teacher = _teacher()
    generator = torch.Generator().manual_seed(seed)
    with ShardWriter(directory, _SCHEMA, samples_per_shard=48) as writer:
        for _ in range(count):
            writer.append(_make_sample(generator, teacher))
    reader = ShardReader(directory, schema=_SCHEMA)
    assert reader.skipped == ()
    assert len(reader) == count
    return reader


def test_three_steps_from_a_real_shard_stream_train_the_net(tmp_path: Path) -> None:
    """Weights move and the held-out loss drops, end to end through all four modules."""
    iterations = [
        IterationRef(
            iteration=index,
            source=_write_iteration(tmp_path / f"iter-{index}", seed=index, count=200),
        )
        for index in range(2)
    ]
    window = RecencyWindow(iterations, WindowConfig(size=5), seed=99)
    assert len(window) == 400
    assert window.weights == pytest.approx((1 / 1.8, 0.8 / 1.8))

    torch.manual_seed(3)
    net = ToyNet()
    optimiser = torch.optim.Adam(net.parameters(), lr=0.05)
    assembler = LossAssembler(
        LossConfig(
            policy_enabled=True,
            policy_weight=1.0,
            wdl_enabled=True,
            wdl_weight=1.0,
            surprise_alpha=0.5,
            surprise_max_weight=4.0,
        )
    )

    held_out = window.sample(64)
    before_weights = {name: tensor.clone() for name, tensor in net.state_dict().items()}

    def held_out_loss() -> float:
        """Return the total loss on the fixed batch, without touching gradients."""
        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = net(held_out["features"])
            total, _ = assembler(outputs, held_out)
        return float(total.item())

    loss_at_step_0 = held_out_loss()

    per_head_seen: set[str] = set()
    for batch in window.batches(64, 3):
        optimiser.zero_grad(set_to_none=True)
        outputs: dict[str, torch.Tensor] = net(batch["features"])
        total, per_head = assembler(outputs, batch)
        assert total.requires_grad
        per_head_seen.update(per_head)
        total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
        optimiser.step()

    loss_at_step_3 = held_out_loss()

    assert per_head_seen == {"policy", "wdl"}
    # (a) the weights actually moved -- every trainable tensor, not just one.
    after = net.state_dict()
    for name, tensor in before_weights.items():
        assert not torch.equal(after[name], tensor), name
    # (b) three steps on the same distribution improved the fixed batch.
    assert loss_at_step_3 < loss_at_step_0


def test_the_trained_net_survives_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """Train, save, reload into a fresh net: identical outputs on a fixed input."""
    reader = _write_iteration(tmp_path / "iter-0", seed=5, count=96)
    window = RecencyWindow([IterationRef(iteration=0, source=reader)], seed=1)

    torch.manual_seed(4)
    net = ToyNet()
    optimiser = torch.optim.Adam(net.parameters(), lr=0.05)
    assembler = LossAssembler(LossConfig(policy_enabled=True, wdl_enabled=True))

    for step, batch in enumerate(window.batches(32, 2)):
        optimiser.zero_grad(set_to_none=True)
        outputs: dict[str, torch.Tensor] = net(batch["features"])
        total, _ = assembler(outputs, batch)
        total.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped
        optimiser.step()
        save_checkpoint(
            tmp_path / "ckpt",
            step,
            net.state_dict(),
            optimiser.state_dict(),
            {"loss": float(total.item())},
        )

    probe = torch.ones(1, _FEATURES)
    with torch.no_grad():
        expected: dict[str, torch.Tensor] = net(probe)

    loaded = load_checkpoint(tmp_path / "ckpt")
    assert loaded is not None
    assert loaded.step == 1
    assert "loss" in loaded.meta

    revived = ToyNet()
    revived.load_state_dict(loaded.model_state)
    with torch.no_grad():
        actual: dict[str, torch.Tensor] = revived(probe)
    assert torch.equal(actual[POLICY_LOGITS], expected[POLICY_LOGITS])
    assert torch.equal(actual[WDL_LOGITS], expected[WDL_LOGITS])


def test_a_crashed_iteration_still_trains_on_what_it_sealed(tmp_path: Path) -> None:
    """The preemption story, end to end: sealed shards feed the window as usual."""
    teacher = _teacher()
    generator = torch.Generator().manual_seed(21)
    writer = ShardWriter(tmp_path / "iter-0", _SCHEMA, samples_per_shard=32)
    for _ in range(100):
        writer.append(_make_sample(generator, teacher))
    assert writer.pending == 4  # 96 sealed, 4 lost with the process
    del writer

    reader = ShardReader(tmp_path / "iter-0", schema=_SCHEMA)
    assert reader.skipped == ()
    assert len(reader) == 96

    window = RecencyWindow([IterationRef(iteration=0, source=reader)], seed=2)
    batch = window.sample(16)
    assert batch["features"].shape == (16, _FEATURES)

    torch.manual_seed(6)
    net = ToyNet()
    outputs: dict[str, torch.Tensor] = net(batch["features"])
    total, per_head = LossAssembler(LossConfig())(outputs, batch)
    assert torch.isfinite(total).item()
    assert set(per_head) == {"policy", "wdl"}
