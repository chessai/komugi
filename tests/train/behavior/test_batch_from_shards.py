"""Shards to window to trainer, once, at the real schema's shapes.

Everything else in this directory runs on toy shapes -- 4 planes, 81 actions
-- because a learning-signal test wants speed. This module runs the *real*
:data:`komugi.selfplay.records.SCHEMA`: 128 observation planes, 37341 actions,
written by a real :class:`~komugi.train.shards.ShardWriter` into two
directories (one per worker, as design D10 requires), read back through two
:class:`~komugi.train.shards.ShardReader` s joined by a
:class:`~komugi.selfplay.records.ConcatSource`, stacked by the real
:func:`~komugi.train.shards.stack_samples`, and fed to the trainer.

The claim being pinned is a naming one, and it is the seam the whole campaign
converges on: **the schema's field names are the batch's key names**. A worker
writes ``policy_target``; the window stacks it into ``batch["policy_target"]``;
``targets_from_batch`` reads it there. Nothing renames anything in between, so
there is no adapter to get out of step -- which is exactly why the absence of
one has to be asserted rather than assumed.

The net is built *from the schema* (in_planes and num_actions read off the
field specs) rather than from constants, so a schema change that the head
config cannot express fails here with a shape it can name.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

import pytest
import torch

import synthetic as syn
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay.records import SCHEMA, ConcatSource
from komugi.train import losses
from komugi.train.shards import FieldSpec, ShardReader, ShardWriter, stack_samples

wiring = pytest.importorskip("komugi.train.wiring")
if not hasattr(wiring, "P0A_RECIPE"):
    pytest.skip(
        "komugi.train.wiring is present but exports no P0A_RECIPE: the "
        "KOM-030 training-step contract has not landed yet",
        allow_module_level=True,
    )
step_module = pytest.importorskip("komugi.train.step")

_ROWS_PER_DIRECTORY: Final[int] = 4
_DIRECTORIES: Final[int] = 2
_ROWS: Final[int] = _ROWS_PER_DIRECTORY * _DIRECTORIES
_SAMPLES_PER_SHARD: Final[int] = 2
_BOARD_SQUARES: Final[int] = 81
_CHANNELS: Final[int] = 8
_SUPPORT: Final[int] = 3
"""How many actions each synthetic policy target puts mass on."""


def _spec(name: str) -> FieldSpec:
    """Return the schema's spec for `name`, failing loudly if it is gone."""
    for field in SCHEMA.fields:
        if field.name == name:
            return field
    raise AssertionError(f"records.SCHEMA has no field {name!r}: {list(SCHEMA.names)}")


def _observation_shape() -> tuple[int, ...]:
    """Return the schema's ``(planes, 9, 9)`` observation shape."""
    return _spec("observation").shape


def _num_actions() -> int:
    """Return the schema's action-space size."""
    return _spec("policy_target").shape[0]


def _net() -> KomugiNet:
    """Return a thin net shaped by the schema itself."""
    actions = _num_actions()
    planes = _observation_shape()[0]
    assert actions % _BOARD_SQUARES == 0, (
        f"the schema's policy_target is {actions} wide, which is not a whole "
        f"number of {_BOARD_SQUARES}-square planes, so no HeadsConfig can "
        "describe a head that emits it"
    )
    trunk = TrunkConfig(
        in_planes=planes, channels=_CHANNELS, blocks=1, gpool_every=3, seed=3
    )
    heads = HeadsConfig(
        channels=_CHANNELS,
        num_actions=actions,
        policy_planes=actions // _BOARD_SQUARES,
        seed=3,
    )
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _row(index: int) -> dict[str, torch.Tensor]:
    """Return one schema-shaped sample, built field by field off the schema.

    Zeros everywhere the row does not care, so a field added to the schema
    tomorrow arrives here with the right dtype and shape instead of turning
    into a ``ValueError`` from the writer.
    """
    row = {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in SCHEMA.fields
    }
    actions = _num_actions()
    for offset in range(_SUPPORT):
        row["policy_target"][(index * 137 + offset * 5_000) % actions] = (
            1.0 / _SUPPORT
        )
    row["wdl_target"][index % 3] = 1.0
    row["mlh_target"] += (index % 8) / 8.0
    row["observation"][index % _observation_shape()[0], index % 9, index % 9] = 1.0
    row["to_play"] += index % 2
    row["ply"] += index
    row["game_id"] += index // _ROWS_PER_DIRECTORY
    row["sims_used"] += 32
    # Zeros are the right default for every other field, but not for this one:
    # a row that claims plane map 0 is a row ``ConcatSource`` refuses, by
    # design (KOM-002 §4), and these rows are read back through one.
    row["plane_map_version"] += PLANE_MAP_VERSION
    # Nor for this one, and for a sharper reason (KOM-014 §8 #26 / N8): zeros
    # would let the key-set assertions below pass while nothing ever carried an
    # action. `index + 1` is distinct and non-zero per row, so the stacked
    # column can be compared to what was planted rather than merely counted.
    row["action"] += index + 1
    return row


def _write_shards(root: Path) -> ConcatSource:
    """Write the rows into `_DIRECTORIES` worker directories and read them back."""
    readers: list[ShardReader] = []
    for worker in range(_DIRECTORIES):
        directory = root / f"worker-{worker}"
        with ShardWriter(
            directory, SCHEMA, samples_per_shard=_SAMPLES_PER_SHARD
        ) as writer:
            for offset in range(_ROWS_PER_DIRECTORY):
                writer.append(_row(worker * _ROWS_PER_DIRECTORY + offset))
        reader = ShardReader(directory, schema=SCHEMA)
        assert reader.skipped == ()
        assert len(reader.shards) == _ROWS_PER_DIRECTORY // _SAMPLES_PER_SHARD
        assert len(reader) == _ROWS_PER_DIRECTORY
        readers.append(reader)
    return ConcatSource(readers)


def test_the_toy_rows_carry_the_real_schema_field_names() -> None:
    """:mod:`synthetic`'s rows are toy in *shape* only, never in naming."""
    assert set(syn.toy_row(0)) == set(SCHEMA.names), (
        "the rest of this suite trains on synthetic rows; if their key names "
        "drift from the schema's, those tests stop testing the real data path"
    )


def test_a_batch_carries_the_action_that_was_written(tmp_path: Path) -> None:
    """The ``action`` column survives write, read, concat and stack (KOM-014 #26).

    A key-set assertion cannot show this: the toy builder zero-fills field by
    field off the schema, so ``action`` would arrive on every row as ``0`` and a
    pipeline that dropped it entirely would still produce the right *names*. The
    planted values (``index + 1``) are what discriminate a carried column from a
    zero-filled one.
    """
    source = _write_shards(tmp_path)
    batch = stack_samples([source[index] for index in range(len(source))])
    assert batch["action"].dtype == torch.int64
    assert batch["action"].tolist() == [index + 1 for index in range(_ROWS)]


def test_a_batch_stacked_from_shards_feeds_the_p0a_recipe(tmp_path: Path) -> None:
    """Written, read, concatenated, stacked, scored -- with no KeyError in between."""
    source = _write_shards(tmp_path)
    assert len(source) == _ROWS

    batch = stack_samples([source[index] for index in range(len(source))])
    assert set(batch) == set(SCHEMA.names), (
        "the batch's keys are the schema's field names, verbatim: that "
        "identity is what lets targets_from_batch read the batch directly"
    )
    assert batch["observation"].shape == (_ROWS, *_observation_shape())
    assert batch["policy_target"].shape == (_ROWS, _num_actions())
    assert torch.allclose(
        batch["policy_target"].sum(dim=-1), torch.ones(_ROWS)
    ), "every stacked policy target is still a distribution"

    net = _net()
    prediction: Any = net(batch["observation"])
    outputs: dict[str, torch.Tensor] = wiring.outputs_from_net(prediction)
    targets: dict[str, torch.Tensor] = wiring.targets_from_batch(batch)

    # The rewire, at the wiring seam: some key must carry the *soft* head's
    # logits. Which key is the assembler's business, not this test's.
    assert any(
        value is prediction.soft_policy_logits for value in outputs.values()
    ), (
        "outputs_from_net dropped NetOutput.soft_policy_logits, so the "
        "8x-weighted soft-policy term has no head to score"
    )

    total, per_head = losses.LossAssembler(wiring.P0A_RECIPE)(outputs, targets)
    assert set(per_head) == {
        losses.HEAD_POLICY,
        losses.HEAD_SOFT_POLICY,
        losses.HEAD_WDL,
    }
    assert bool(torch.isfinite(total).item())
    for head, value in per_head.items():
        assert bool(torch.isfinite(value).item()), f"{head} loss is {value}"


def test_a_training_step_runs_on_a_batch_straight_off_the_shards(
    tmp_path: Path,
) -> None:
    """The same batch, handed to the real step: finite numbers, no adapter."""
    source = _write_shards(tmp_path)
    batch = stack_samples([source[index] for index in range(len(source))])

    net = _net()
    before = syn.parameter_snapshot(net)
    trainer: Any = step_module.TrainStep(
        net,
        losses.LossAssembler(wiring.P0A_RECIPE),
        step_module.StepConfig(batch_size=_ROWS, lr=1e-3, seed=0),
    )
    report: dict[str, float] = trainer.run(batch)

    assert report, "the step reported no metrics"
    for name, value in report.items():
        assert math.isfinite(float(value)), f"the step reported {name}={value}"
    assert syn.moved_parameters(net, before), (
        "one step on a real batch moved no parameter at all"
    )
