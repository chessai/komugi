"""A sealed shard is verified and decompressed once, then lives in RAM.

``ShardReader``'s construction scan already decompresses and digest-verifies
every shard in order to count its rows, and a sealed shard never changes after
that. Retained mode (the default) keeps the arrays that scan produced; the
properties worth pinning are therefore:

* equivalence -- a retained reader serves byte-identical rows to a
  ``retain_arrays=False`` one, over a full index sweep;
* work-once -- over a retained reader, shuffled row access does *not* grow the
  load or digest counters past the one-per-shard construction cost, while the
  same access pattern over a non-retained reader does. This is the regression
  that keeps the trainer from spending 58% of a step in ``zlib`` and 19%
  re-hashing payloads it has already verified;
* unchanged damage and snapshot semantics -- retention changes when work
  happens, never what is trusted or what is visible.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.train import shards as shards_module
from komugi.train.shards import (
    FieldSpec,
    Schema,
    ShardReader,
    ShardWarning,
    ShardWriter,
)

_SCHEMA = Schema(
    fields=(
        FieldSpec(name="features", dtype="float32", shape=(2, 3)),
        FieldSpec(name="policy_target", dtype="float32", shape=(4,)),
        FieldSpec(name="winner", dtype="int64", shape=()),
    )
)

_ROWS_PER_SHARD = 8


def _sample(index: int) -> dict[str, torch.Tensor]:
    """Return sample `index`, with values that identify it uniquely."""
    return {
        "features": torch.arange(6, dtype=torch.float32).reshape(2, 3) + index,
        "policy_target": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        + index * 0.01,
        "winner": torch.tensor(index % 3, dtype=torch.int64),
    }


def _write_shards(directory: Path, shard_count: int, *, first: int = 0) -> None:
    """Seal `shard_count` full shards of `_ROWS_PER_SHARD` samples each."""
    with ShardWriter(directory, _SCHEMA, samples_per_shard=_ROWS_PER_SHARD) as writer:
        for offset in range(shard_count * _ROWS_PER_SHARD):
            writer.append(_sample(first + offset))


def _install_spies(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count calls to the two functions retention is meant to stop repeating.

    ``_load_shard_arrays`` is the only door to the file and its DEFLATE
    streams, and ``_payload_digest`` is the ``sha256`` over every field's
    bytes. Both wrap the real implementation, so what is measured is work that
    genuinely happened rather than work a stub pretended to do.
    """
    counts: Counter[str] = Counter()
    real_load = shards_module._load_shard_arrays
    real_digest = shards_module._payload_digest

    def counting_load(path: Path) -> dict[str, npt.NDArray[Any]]:
        counts["load"] += 1
        return real_load(path)

    def counting_digest(
        schema: Schema, count: int, arrays: Mapping[str, npt.NDArray[Any]]
    ) -> str:
        counts["digest"] += 1
        return real_digest(schema, count, arrays)

    monkeypatch.setattr(shards_module, "_load_shard_arrays", counting_load)
    monkeypatch.setattr(shards_module, "_payload_digest", counting_digest)
    return counts


def _shuffled_indices(total: int, howmany: int) -> list[int]:
    """Return `howmany` seeded random row indices in ``[0, total)``.

    Seeded so the two readers under comparison see the identical access
    pattern and the non-retained reader's miss count is arithmetic rather than
    a probabilistic hope.
    """
    rng = random.Random(20260823)
    return [rng.randrange(total) for _ in range(howmany)]


def test_retained_and_reread_readers_serve_identical_rows(tmp_path: Path) -> None:
    """Every row of a retained reader equals the re-reading reader's row."""
    _write_shards(tmp_path, shard_count=5)

    retained = ShardReader(tmp_path)
    reread = ShardReader(tmp_path, retain_arrays=False)

    assert retained.retains_arrays is True
    assert reread.retains_arrays is False
    assert retained.skipped == () and reread.skipped == ()
    assert retained.schema == reread.schema == _SCHEMA
    assert [info.count for info in retained.shards] == [info.count for info in reread.shards]
    assert len(retained) == len(reread) == 5 * _ROWS_PER_SHARD

    for index in range(len(retained)):
        left = retained[index]
        right = reread[index]
        assert set(left) == set(right) == set(_SCHEMA.names)
        for name in _SCHEMA.names:
            assert left[name].dtype == right[name].dtype, (index, name)
            assert left[name].shape == right[name].shape, (index, name)
            assert torch.equal(left[name], right[name]), (index, name)

    # Iteration is the same window in the same order, in both modes.
    for left_sample, right_sample in zip(retained, reread, strict=True):
        for name in _SCHEMA.names:
            assert torch.equal(left_sample[name], right_sample[name]), name


def test_a_retained_reader_loads_and_digests_each_shard_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construction pays for K shards; 200 shuffled fetches add nothing."""
    shard_count = 5
    _write_shards(tmp_path, shard_count=shard_count)

    counts = _install_spies(monkeypatch)
    reader = ShardReader(tmp_path)

    # The construction scan: one read and one digest per shard, no more.
    assert counts["load"] == shard_count
    assert counts["digest"] == shard_count
    after_construction = Counter(counts)

    for index in _shuffled_indices(len(reader), 200):
        row = reader[index]
        assert torch.equal(row["winner"], _sample(index)["winner"])
    assert counts == after_construction

    # Full iteration is likewise served from the retained arrays.
    assert len(list(reader)) == shard_count * _ROWS_PER_SHARD
    assert counts == after_construction


def test_a_non_retained_reader_still_re_reads_on_every_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The knob is real in both directions: retention off, the work comes back."""
    shard_count = 5
    _write_shards(tmp_path, shard_count=shard_count)

    counts = _install_spies(monkeypatch)
    reader = ShardReader(tmp_path, retain_arrays=False)
    assert counts["load"] == shard_count
    assert counts["digest"] == shard_count

    indices = _shuffled_indices(len(reader), 200)
    # The one-shard cache hits only when a draw lands in the same shard as the
    # previous one, so the extra work is exactly the number of transitions.
    slots = [index // _ROWS_PER_SHARD for index in indices]
    misses = sum(1 for position, slot in enumerate(slots) if position == 0 or slot != slots[position - 1])
    assert misses > 100, "the sampled pattern must actually cross shards"

    for index in indices:
        reader[index]

    assert counts["load"] == shard_count + misses
    assert counts["digest"] == shard_count + misses


def test_a_retained_reader_needs_no_file_after_construction(tmp_path: Path) -> None:
    """Retained rows survive their files: the reader is holding, not caching."""
    _write_shards(tmp_path, shard_count=3)
    reader = ShardReader(tmp_path)
    expected = [reader[index] for index in range(len(reader))]

    for info in reader.shards:
        info.path.unlink()

    for index, sample in enumerate(expected):
        actual = reader[index]
        for name in _SCHEMA.names:
            assert torch.equal(actual[name], sample[name]), (index, name)


def test_a_corrupt_shard_is_still_skipped_in_retained_mode(tmp_path: Path) -> None:
    """Damage costs its own shard and nothing else, retention or not."""
    _write_shards(tmp_path, shard_count=3)

    # Scramble the middle half of the file: squarely inside the members'
    # DEFLATE streams, clear of the local header and the central directory.
    victim = tmp_path / "shard-000001.npz"
    data = bytearray(victim.read_bytes())
    for offset in range(len(data) // 4, (3 * len(data)) // 4):
        data[offset] ^= 0xFF
    victim.write_bytes(bytes(data))

    with pytest.warns(ShardWarning, match="shard-000001"):
        reader = ShardReader(tmp_path)

    assert reader.retains_arrays is True
    assert [skip.path.name for skip in reader.skipped] == ["shard-000001.npz"]
    assert [info.path.name for info in reader.shards] == [
        "shard-000000.npz",
        "shard-000002.npz",
    ]
    assert len(reader) == 2 * _ROWS_PER_SHARD

    survivors = list(range(_ROWS_PER_SHARD)) + list(
        range(2 * _ROWS_PER_SHARD, 3 * _ROWS_PER_SHARD)
    )
    for position, source in enumerate(survivors):
        expected = _sample(source)
        actual = reader[position]
        for name in _SCHEMA.names:
            assert torch.equal(actual[name], expected[name]), (position, name)


@pytest.mark.parametrize("retain", [True, False])
def test_a_reader_is_a_snapshot_in_either_mode(tmp_path: Path, retain: bool) -> None:
    """A shard sealed after construction is invisible to an existing reader."""
    _write_shards(tmp_path, shard_count=2)
    reader = ShardReader(tmp_path, retain_arrays=retain)
    assert len(reader) == 2 * _ROWS_PER_SHARD

    _write_shards(tmp_path, shard_count=1, first=1000)

    assert len(reader) == 2 * _ROWS_PER_SHARD
    assert len(reader.shards) == 2
    assert len(list(reader)) == 2 * _ROWS_PER_SHARD
    with pytest.raises(IndexError):
        reader[2 * _ROWS_PER_SHARD]

    # A reader built now does see it, so the fixture really did seal a shard.
    assert len(ShardReader(tmp_path, retain_arrays=retain)) == 3 * _ROWS_PER_SHARD


@pytest.mark.parametrize("retain", [True, False])
def test_a_payload_tampered_but_zip_valid_shard_is_skipped(
    tmp_path: Path, retain: bool
) -> None:
    """The digest is the only defence once the zip's own CRC is satisfied.

    The DEFLATE-scramble test above never reaches ``_payload_digest`` -- the
    zip layer rejects the bytes first. The KOM-033 review (F2) built the
    tamper that gets past it: rewrite one field's values inside a structurally
    valid npz, leaving the recorded digest stale, and the verify-then-ignore
    mutant served the poisoned labels through every test in this repo. This
    pins the digest's *consequence* in both modes: the tampered shard is
    skipped for a digest mismatch, and the surviving rows are the untampered
    ones by value.
    """
    _write_shards(tmp_path, shard_count=3)
    victim = tmp_path / "shard-000001.npz"
    with np.load(victim) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    reserved = {
        shards_module.COUNT_KEY,
        shards_module.SCHEMA_KEY,
        shards_module.FINGERPRINT_KEY,
        shards_module.DIGEST_KEY,
    }
    field = next(name for name in sorted(arrays) if name not in reserved)
    arrays[field] = (arrays[field] + 100).astype(arrays[field].dtype)
    with victim.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.warns(ShardWarning, match="digest"):
        reader = ShardReader(tmp_path, retain_arrays=retain)

    assert reader.retains_arrays is retain
    assert [skip.path.name for skip in reader.skipped] == ["shard-000001.npz"]
    assert "digest" in reader.skipped[0].reason
    assert len(reader) == 2 * _ROWS_PER_SHARD

    survivors = list(range(_ROWS_PER_SHARD)) + list(
        range(2 * _ROWS_PER_SHARD, 3 * _ROWS_PER_SHARD)
    )
    for position, source in enumerate(survivors):
        expected = _sample(source)
        actual = reader[position]
        for name in _SCHEMA.names:
            assert torch.equal(actual[name], expected[name]), (position, name)
