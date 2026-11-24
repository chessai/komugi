"""KOM-002 §7 E #23 -- the version check the fingerprint provably cannot make.

``Schema.fingerprint`` is a ``sha256`` over field names, dtypes and shapes. A
future ticket that fills one of the encoder's reserved planes 121-127 changes
what a row MEANS while leaving all three unchanged, so the fingerprint would
wave it straight through and old and new rows would mix silently inside one
training window -- a worse failure than the shape break KOM-002 is fixing.

``plane_map_version`` is the eight bytes that close that gap, and this module
is the whole of its read side: the value is checked when a row is
*materialised*, because a :class:`~komugi.train.window.SampleSource` need not
have a shard behind it and the row itself is the only place the version is
guaranteed to be readable.

**Why this file is not ``tests/train/test_records.py``**, which is the path
design §7 #23 names: ``tests/eval/test_records.py`` already exists, no
directory under ``tests/`` carries an ``__init__.py``, and pytest's default
``prepend`` import mode derives a module name from the BASENAME alone -- so two
``test_records.py`` files are an "import file mismatch" collection error, not a
suite. Every other basename under ``tests/`` is unique, which is the
convention this follows. The design anticipates the substitution: #23 "fixes
the *behaviour*, not the location" (§4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.selfplay.records import SCHEMA, ConcatSource
from komugi.train.shards import ShardReader, ShardWriter
from komugi.train.window import SampleSource

_STALE_VERSION: int = 1
"""KOM-027's 118-plane map. Every shard the in-flight campaign wrote claims it."""


def _row(plane_map_version: int) -> dict[str, torch.Tensor]:
    """A schema-shaped row of zeros claiming `plane_map_version`."""
    row = {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in SCHEMA.fields
    }
    row["plane_map_version"] = torch.tensor(plane_map_version, dtype=torch.int64)
    return row


def _one_shard_source(directory: Path, plane_map_version: int) -> ConcatSource:
    """Write one row claiming `plane_map_version` and read it back."""
    with ShardWriter(directory, SCHEMA, samples_per_shard=1) as writer:
        writer.append(_row(plane_map_version))
    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == (), "the shard itself is well formed"
    assert len(reader) == 1
    return ConcatSource([reader])


def test_a_stale_plane_map_version_is_refused(tmp_path: Path) -> None:
    """A correct-fingerprint, correct-shape row is still refused, by version.

    The shard here is valid in every way the reader can check: its schema is
    the running one, its fingerprint matches, its shapes match. Only the
    encoder that produced its observation differs -- which is precisely the
    case a shape-derived fingerprint is blind to.
    """
    source = _one_shard_source(tmp_path, _STALE_VERSION)

    with pytest.raises(ValueError) as raised:
        source[0]
    message = str(raised.value)
    assert f"version {_STALE_VERSION}" in message, message
    assert f"version {PLANE_MAP_VERSION}" in message, message

    # Both versions named, not just "mismatch": the operator has to know which
    # generation to regenerate, and which one this process speaks.
    assert "Regenerate the shard." in message


def test_a_version_2_row_is_refused_after_the_cap_bump(tmp_path: Path) -> None:
    """KOM-165: a campaign 2-4 shard cannot be trained on under plane map 3.

    The version-1 case above differs from this process in plane COUNT as well,
    so several other guards would also catch it. Version 2 differs in nothing a
    machine can see except this field: 128 planes, the same schema, the same
    fingerprint, and planes 115/120 normalised by a 256-ply cap rather than this
    process's. Mixing those rows into a replay window trains the value head on
    two different turn clocks at once, with no error anywhere.

    Both versions are named for the operator's sake -- "which generation is this
    shard, and which do I speak" is the only actionable question at this point.
    """
    stale = PLANE_MAP_VERSION - 1
    source = _one_shard_source(tmp_path, stale)

    with pytest.raises(ValueError) as raised:
        source[0]
    message = str(raised.value)
    assert f"version {stale}" in message, message
    assert f"version {PLANE_MAP_VERSION}" in message, message
    assert "Regenerate the shard." in message


def test_a_row_without_the_field_is_refused_too() -> None:
    """A row written before the field existed is refused, not defaulted.

    Absence must not read as agreement -- the same rule §5.3 applies to a
    checkpoint whose meta lacks ``in_planes``. Written against a plain list
    source, because that is the shape a pre-KOM-002 row survives in once its
    shard has been read by anything other than this schema.
    """
    row = _row(PLANE_MAP_VERSION)
    del row["plane_map_version"]
    sources: list[SampleSource] = [[row]]

    with pytest.raises(ValueError) as raised:
        ConcatSource(sources)[0]
    message = str(raised.value)
    assert "plane_map_version" in message
    assert f"version {PLANE_MAP_VERSION}" in message


def test_a_current_row_passes(tmp_path: Path) -> None:
    """Non-vacuity: the check refuses stale rows, not every row.

    Without this the two tests above would pass identically if
    ``ConcatSource`` had simply been broken into raising on everything.
    """
    source = _one_shard_source(tmp_path, PLANE_MAP_VERSION)
    sample = source[0]
    assert set(sample) == set(SCHEMA.names)
    assert int(sample["plane_map_version"].item()) == PLANE_MAP_VERSION
    assert sample["observation"].shape == (128, 9, 9)
