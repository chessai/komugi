"""What a training iteration leaves on disk, read back with the real readers.

``training_iteration`` takes an optional ``checkpoint_dir`` and an optional
``journal_path``. This module runs one with both set and then puts down the
trainer entirely: the checkpoint comes back through
:func:`komugi.train.checkpoint.load_checkpoint` (digest verification, format
version, ``weights_only`` load -- the real path a resumed run takes), and the
journal through :func:`komugi.telemetry.journal.decode_line` and
:func:`~komugi.telemetry.journal.read_records`.

Two things are deliberately *not* asserted.

**Which step is checkpointed.** The save cadence is the trainer's business, so
the round trip is checked against the checkpoint's own contents -- load it,
put it in a fresh net, and require the net's ``state_dict`` to come back
bit-equal -- plus the one substantive claim that survives any cadence: the
saved weights are *trained* ones, not the initialisation.

**What the journal records mean.** The trainer's per-step reports and
:class:`~komugi.telemetry.records.IterationMetrics` are not the same shape of
thing, and the contract says the mapping is the trainer's own. So this pins
what is checkable from outside: at least one record was written, every line
decodes, and the reader skips nothing. A journal that gained lines the reader
refuses is the failure worth catching -- v1 lost its metrics twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
import torch

import synthetic as syn
from komugi.telemetry import journal
from komugi.telemetry.records import IterationMetrics, NetFingerprint
from komugi.train import losses
from komugi.train.checkpoint import Checkpoint, list_checkpoints, load_checkpoint

wiring = pytest.importorskip("komugi.train.wiring")
if not hasattr(wiring, "P0A_RECIPE"):
    pytest.skip(
        "komugi.train.wiring is present but exports no P0A_RECIPE: the "
        "KOM-030 training-step contract has not landed yet",
        allow_module_level=True,
    )
step_module = pytest.importorskip("komugi.train.step")

_ROWS: Final[int] = 16
_STEPS: Final[int] = 4
_LR: Final[float] = 3e-3
_SEED: Final[int] = 515


def _run_iteration(tmp_path: Path) -> tuple[Path, Path]:
    """Train `_STEPS` steps with both sinks on; return (checkpoint dir, journal)."""
    checkpoint_dir = tmp_path / "checkpoints"
    journal_path = tmp_path / "journal.jsonl"

    torch.manual_seed(_SEED)
    net = syn.tiny_net()
    trainer: Any = step_module.TrainStep(
        net,
        losses.LossAssembler(wiring.P0A_RECIPE),
        step_module.StepConfig(batch_size=_ROWS, lr=_LR, seed=_SEED),
    )
    metrics: list[dict[str, float]] = step_module.training_iteration(
        net,
        syn.toy_rows(_ROWS),
        trainer,
        steps=_STEPS,
        rng_seed=_SEED,
        checkpoint_dir=checkpoint_dir,
        journal_path=journal_path,
    )
    assert metrics, "the iteration reported nothing"
    return checkpoint_dir, journal_path


def test_the_checkpoint_loads_back_into_a_fresh_net(tmp_path: Path) -> None:
    """A checkpoint is published, verifies, and restores bit-equal weights."""
    checkpoint_dir, _ = _run_iteration(tmp_path)

    refs = list_checkpoints(checkpoint_dir)
    assert refs, (
        f"training_iteration was given checkpoint_dir={checkpoint_dir} and "
        "published nothing under it"
    )

    loaded: Checkpoint | None = load_checkpoint(checkpoint_dir)
    assert loaded is not None, (
        "the newest checkpoint failed digest or payload verification, so a "
        "resumed run would silently fall back to an older one"
    )
    assert loaded.step >= 0

    revived = syn.tiny_net()
    assert set(loaded.model_state) == set(revived.state_dict()), (
        "the checkpoint's model state does not describe this net: "
        f"{sorted(set(loaded.model_state) ^ set(revived.state_dict()))}"
    )
    revived.load_state_dict(loaded.model_state)
    for name, tensor in revived.state_dict().items():
        assert torch.equal(tensor, loaded.model_state[name]), (
            f"{name} did not survive the checkpoint round trip bit-equal"
        )

    # The one claim that holds whatever the save cadence is: what was written
    # is a *trained* net. A checkpoint of the initialisation would round-trip
    # just as prettily and be worth nothing on resume.
    pristine = syn.parameter_snapshot(syn.tiny_net())
    assert any(
        not torch.equal(loaded.model_state[name], tensor)
        for name, tensor in pristine.items()
    ), f"the checkpoint at step {loaded.step} holds the untrained weights"


def test_the_journal_gained_readable_records(tmp_path: Path) -> None:
    """Every line the iteration appended decodes through the journal's reader."""
    _, journal_path = _run_iteration(tmp_path)

    assert journal_path.exists(), (
        f"training_iteration was given journal_path={journal_path} and wrote "
        "no journal at all"
    )
    lines = [
        line for line in journal_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert lines, "the journal is empty"

    for number, line in enumerate(lines, start=1):
        try:
            journal.decode_line(line)
        except journal.JournalDecodeError as exc:
            pytest.fail(f"journal line {number} does not decode: {exc}")

    metrics = journal.read_records(journal_path, IterationMetrics)
    fingerprints = journal.read_records(journal_path, NetFingerprint)
    assert metrics.ok and fingerprints.ok, (
        f"the reader skipped {metrics.skipped} unusable line(s)"
    )
    assert len(metrics) + len(fingerprints) == len(lines), (
        "the journal holds lines of neither known kind: "
        f"{len(lines)} written, {len(metrics)} metrics + "
        f"{len(fingerprints)} fingerprints decoded"
    )
    for record in metrics.records:
        assert record.iteration >= 0
        assert record.games >= 0
