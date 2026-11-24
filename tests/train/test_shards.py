"""Shards keep what they were given, and a kill costs only the buffer.

The properties, in the order a campaign would miss them:

* round-trip -- values, dtypes and shapes come back bit-exact, in write order;
* periodic flush -- the time trigger seals a short-but-complete shard, so the
  unsealed backlog is bounded by one ``flush_every_seconds`` interval *of
  appends*; the trigger lives in ``append``, so a writer that goes quiet holds
  its tail until someone calls ``flush``, which is why ``flush`` is public;
* crash tolerance -- a writer that is never closed leaves every sealed shard
  readable and forfeits only what was still buffered;
* atomicity -- no partially written file is ever visible under a shard name;
* damage containment -- a corrupted or foreign shard is skipped with a
  warning and the rest of the directory reads normally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.observation import CONTROL_SIZE, observation_size
from komugi.selfplay.records import SCHEMA
from komugi.train.shards import (
    FINGERPRINT_KEY,
    SHARD_GLOB,
    FieldSpec,
    Schema,
    ShardReader,
    ShardWarning,
    ShardWriter,
    _load_shard_arrays,
    _validated_arrays,
    read_columns,
    stack_samples,
)

_SCHEMA = Schema(
    fields=(
        FieldSpec(name="features", dtype="float32", shape=(2, 3)),
        FieldSpec(name="policy_target", dtype="float32", shape=(4,)),
        FieldSpec(name="winner", dtype="int64", shape=()),
    )
)


def _sample(index: int) -> dict[str, torch.Tensor]:
    """Return sample `index`, with values that identify it uniquely."""
    return {
        "features": torch.arange(6, dtype=torch.float32).reshape(2, 3) + index,
        "policy_target": torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float32
        )
        + index * 0.01,
        "winner": torch.tensor(index % 3, dtype=torch.int64),
    }


def _assert_same_sample(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    """Assert two samples are equal field by field, dtype included."""
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype, name
        assert actual[name].shape == tensor.shape, name
        assert torch.equal(actual[name], tensor), name


class _FakeClock:
    """A monotonic clock the test advances by hand.

    The flush policy is a time policy; testing it with ``time.monotonic``
    would mean sleeping, and sleeping in a test means either a slow suite or a
    flaky one.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


def test_round_trip_is_exact(tmp_path: Path) -> None:
    """Every sample comes back with the same bytes, dtypes and order."""
    samples = [_sample(index) for index in range(7)]
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=3) as writer:
        for sample in samples:
            writer.append(sample)

    reader = ShardReader(tmp_path)
    assert reader.skipped == ()
    assert len(reader) == 7
    # 3 + 3 sealed by the size trigger, 1 by close().
    assert [info.count for info in reader.shards] == [3, 3, 1]
    assert reader.schema == _SCHEMA

    for expected, actual in zip(samples, reader, strict=True):
        _assert_same_sample(actual, expected)
    for index, expected in enumerate(samples):
        _assert_same_sample(reader[index], expected)
    _assert_same_sample(reader[-1], samples[-1])


# ---------------------------------------------------------------------------
# read_columns -- the projecting reader (KOM-014 §2.4)
# ---------------------------------------------------------------------------


def test_read_columns_returns_only_the_named_fields(tmp_path: Path) -> None:
    """A projection is the reserved keys plus exactly what was asked for.

    The whole point of the function: one column out of a shard, not a row out of
    a directory. If it quietly returned everything, the draft-book builder would
    pull ~190 KB a row to read three ``int64`` columns and §2.4's cost argument
    would be false while every other assertion still passed.
    """
    samples = [_sample(index) for index in range(5)]
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=5) as writer:
        for sample in samples:
            writer.append(sample)
    shard = tmp_path / "shard-000000.npz"

    schema, count, columns = read_columns(shard, ("winner",))
    assert schema == _SCHEMA
    assert count == 5
    assert set(columns) == {"winner"}
    assert columns["winner"].dtype == np.dtype("int64")
    assert columns["winner"].tolist() == [index % 3 for index in range(5)]

    # names=() is how a caller asks "what generation is this shard?" without
    # paying for a single row -- the schema guard's whole access pattern.
    bare_schema, bare_count, bare_columns = read_columns(shard, ())
    assert bare_columns == {}
    assert bare_schema.fingerprint == _SCHEMA.fingerprint
    assert bare_count == 5


def test_read_columns_refuses_a_field_the_shard_does_not_have(tmp_path: Path) -> None:
    """Asking for a column that is not in the shard's own schema is an error.

    Not an empty result and not a ``KeyError`` from numpy: the builder projects
    ``action`` across a window that may straddle the fingerprint bump, and
    "this shard predates the field" has to be a message naming the shard.
    """
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=1) as writer:
        writer.append(_sample(0))
    shard = tmp_path / "shard-000000.npz"

    with pytest.raises(ValueError, match="action"):
        read_columns(shard, ("action",))


def test_read_columns_still_refuses_a_shard_whose_schema_lies(tmp_path: Path) -> None:
    """The checks a projection *can* make are still made, digest aside.

    ``__digest__`` covers the whole payload in schema order, so a projection
    cannot verify it -- that gap is stated in the function's docstring and paid
    for elsewhere (§2.4). Everything upstream of it still holds: a shard whose
    embedded fingerprint disagrees with its embedded schema is refused here,
    naming both.
    """
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=1) as writer:
        writer.append(_sample(0))
    shard = tmp_path / "shard-000000.npz"

    arrays = _load_shard_arrays(shard)
    arrays[FINGERPRINT_KEY] = np.array("0" * 64)
    np.savez_compressed(shard, **arrays)  # type: ignore[arg-type]  # numpy stubs mistype savez_compressed's **kwargs

    with pytest.raises(ValueError, match="fingerprint"):
        read_columns(shard, ("winner",))


def test_index_out_of_range_raises(tmp_path: Path) -> None:
    """Random access past the end is an IndexError, not a silent wrap."""
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=2) as writer:
        writer.append(_sample(0))
    reader = ShardReader(tmp_path)
    with pytest.raises(IndexError):
        reader[1]


def test_schema_violations_are_rejected_before_buffering(tmp_path: Path) -> None:
    """A bad sample raises and does not enter the buffer."""
    writer = ShardWriter(tmp_path, _SCHEMA, samples_per_shard=100)
    bad_dtype = _sample(0)
    bad_dtype["winner"] = torch.tensor(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="int64"):
        writer.append(bad_dtype)

    bad_shape = _sample(0)
    bad_shape["policy_target"] = torch.zeros(5, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        writer.append(bad_shape)

    missing = _sample(0)
    del missing["features"]
    with pytest.raises(ValueError, match="missing"):
        writer.append(missing)

    extra = _sample(0)
    extra["surprise"] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="extra"):
        writer.append(extra)

    assert writer.pending == 0


def test_periodic_flush_seals_a_short_shard(tmp_path: Path) -> None:
    """The time trigger publishes a complete shard well short of the size limit.

    This is the "a preemption forfeits minutes, not an iteration" invariant:
    with a 1024-sample shard size, the two samples buffered when the interval
    elapsed are on disk, not waiting for another 1022.
    """
    clock = _FakeClock()
    writer = ShardWriter(
        tmp_path,
        _SCHEMA,
        samples_per_shard=1024,
        flush_every_seconds=10.0,
        clock=clock,
    )
    writer.append(_sample(0))
    writer.append(_sample(1))
    assert writer.pending == 2
    assert writer.sealed_shards == ()

    clock.advance(10.0)
    writer.append(_sample(2))

    assert writer.pending == 0
    assert len(writer.sealed_shards) == 1
    reader = ShardReader(tmp_path)
    assert len(reader) == 3
    assert reader.skipped == ()


def test_writer_copies_so_callers_may_reuse_buffers(tmp_path: Path) -> None:
    """Mutating an appended tensor after the call does not change the shard."""
    scratch = _sample(0)
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as writer:
        writer.append(scratch)
        scratch["features"].fill_(99.0)
        writer.append(_sample(1))

    reader = ShardReader(tmp_path)
    _assert_same_sample(reader[0], _sample(0))


def test_a_killed_writer_loses_only_its_buffer(tmp_path: Path) -> None:
    """Sealed shards survive a writer that is never closed; the buffer does not.

    The kill is simulated the only way a single-process test honestly can: the
    writer object is abandoned without ``close()``, which is exactly the state
    a SIGKILL leaves the *directory* in -- whatever was sealed is on disk,
    whatever was buffered never existed outside the dead process's memory.
    """
    writer = ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4)
    for index in range(10):
        writer.append(_sample(index))
    assert writer.pending == 2  # 8 sealed as two shards, 2 still buffered
    del writer  # the process dies here

    reader = ShardReader(tmp_path)
    assert reader.skipped == ()
    assert len(reader) == 8
    for index in range(8):
        _assert_same_sample(reader[index], _sample(index))


def test_no_half_written_file_is_ever_visible(tmp_path: Path) -> None:
    """After a simulated kill, every visible shard name reads cleanly.

    Atomicity has an observable form: the set of files matching the reader's
    glob contains only whole shards, and any in-progress file is both
    dot-prefixed and outside that glob. Both are asserted, because either one
    alone would let a rename-into-place bug through.
    """
    writer = ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4)
    for index in range(10):
        writer.append(_sample(index))
    del writer

    # A crash *during* a seal leaves a temporary file behind; forge one.
    torn = tmp_path / ".shard-000002.npz.tmp-4242"
    torn.write_bytes(b"PK\x03\x04 half a shard")

    visible = sorted(path.name for path in tmp_path.glob(SHARD_GLOB))
    assert visible == ["shard-000000.npz", "shard-000001.npz"]
    assert torn.name not in visible
    assert torn.name.startswith(".")

    reader = ShardReader(tmp_path)
    assert reader.skipped == ()
    assert len(reader) == 8

    # And a writer restarted on the same directory appends rather than
    # reusing an index, so the torn tmp cannot be mistaken for shard 2.
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as resumed:
        resumed.append(_sample(100))
    assert resumed.sealed_shards[0].name == "shard-000002.npz"
    assert len(ShardReader(tmp_path)) == 9


def test_a_corrupt_shard_is_skipped_and_the_rest_survive(tmp_path: Path) -> None:
    """Flipping bytes inside one shard costs that shard and nothing else."""
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as writer:
        for index in range(12):
            writer.append(_sample(index))

    # Scramble the middle half of the file: that is squarely inside the
    # members' DEFLATE streams and clear of both the leading local header and
    # the trailing central directory, so the failure this provokes is genuine
    # payload damage rather than a malformed container.
    victim = tmp_path / "shard-000001.npz"
    data = bytearray(victim.read_bytes())
    for offset in range(len(data) // 4, (3 * len(data)) // 4):
        data[offset] ^= 0xFF
    victim.write_bytes(bytes(data))

    with pytest.warns(ShardWarning, match="shard-000001"):
        reader = ShardReader(tmp_path)

    assert [skip.path.name for skip in reader.skipped] == ["shard-000001.npz"]
    assert len(reader) == 8
    for position, index in enumerate([0, 1, 2, 3, 8, 9, 10, 11]):
        _assert_same_sample(reader[position], _sample(index))


def test_a_truncated_trailing_shard_is_skipped(tmp_path: Path) -> None:
    """A shard cut off mid-write reads as absent, not as an exception."""
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as writer:
        for index in range(8):
            writer.append(_sample(index))

    victim = tmp_path / "shard-000001.npz"
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])

    with pytest.warns(ShardWarning):
        reader = ShardReader(tmp_path)
    assert len(reader) == 4
    assert len(reader.skipped) == 1


def test_a_foreign_fingerprint_is_skipped(tmp_path: Path) -> None:
    """A shard from another schema generation does not join this window."""
    other = Schema(fields=(FieldSpec(name="features", dtype="float32", shape=(1,)),))
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as writer:
        for index in range(4):
            writer.append(_sample(index))
    with ShardWriter(tmp_path, other, samples_per_shard=4) as writer_b:
        for _ in range(4):
            writer_b.append({"features": torch.zeros(1, dtype=torch.float32)})

    with pytest.warns(ShardWarning, match="fingerprint"):
        reader = ShardReader(tmp_path, schema=_SCHEMA)
    assert len(reader) == 4
    assert [skip.path.name for skip in reader.skipped] == ["shard-000001.npz"]

    # Without an explicit schema the reader adopts the first shard's and holds
    # the rest to it -- one generation plus a warning, never an interleaving.
    with pytest.warns(ShardWarning):
        adopted = ShardReader(tmp_path)
    assert adopted.schema == _SCHEMA
    assert len(adopted) == 4


def test_a_tampered_fingerprint_field_is_caught(tmp_path: Path) -> None:
    """The embedded fingerprint is checked against the embedded schema."""
    with ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4) as writer:
        for index in range(4):
            writer.append(_sample(index))

    victim = tmp_path / "shard-000000.npz"
    loaded = np.load(victim, allow_pickle=False)
    payload = {str(name): loaded[name] for name in loaded.files}
    loaded.close()
    payload[FINGERPRINT_KEY] = np.array("0" * 64)
    with victim.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    with pytest.warns(ShardWarning, match="fingerprint"):
        reader = ShardReader(tmp_path)
    assert len(reader) == 0
    assert reader.schema is None


def test_an_empty_directory_reads_as_empty(tmp_path: Path) -> None:
    """No shards is not an error; it is a window that has not started."""
    reader = ShardReader(tmp_path)
    assert len(reader) == 0
    assert reader.shards == ()
    assert reader.skipped == ()
    assert list(reader) == []


def test_stack_samples_builds_a_batch(tmp_path: Path) -> None:
    """Stacking gives leading-batch-dimension tensors and rejects a mismatch."""
    batch = stack_samples([_sample(0), _sample(1), _sample(2)])
    assert batch["features"].shape == (3, 2, 3)
    assert batch["policy_target"].shape == (3, 4)
    assert batch["winner"].shape == (3,)
    assert batch["winner"].dtype == torch.int64
    assert torch.equal(batch["features"][1], _sample(1)["features"])

    with pytest.raises(ValueError, match="empty"):
        stack_samples([])
    with pytest.raises(ValueError, match="keys"):
        stack_samples([_sample(0), {"features": torch.zeros(1)}])


def test_schema_rejects_reserved_names_and_bad_dtypes() -> None:
    """Container keys and unsupported dtypes are schema errors, not shard errors."""
    with pytest.raises(ValueError, match="reserved"):
        FieldSpec(name="__count__", dtype="float32", shape=(1,))
    with pytest.raises(ValueError, match="dtype"):
        FieldSpec(name="x", dtype="float16", shape=(1,))
    with pytest.raises(ValueError, match="non-positive"):
        FieldSpec(name="x", dtype="float32", shape=(0,))
    with pytest.raises(ValueError, match="savez parameters"):
        FieldSpec(name="file", dtype="float32", shape=(1,))
    with pytest.raises(ValueError, match="identifier"):
        FieldSpec(name="policy target", dtype="float32", shape=(1,))
    with pytest.raises(ValueError, match="duplicate"):
        Schema(
            fields=(
                FieldSpec(name="x", dtype="float32", shape=(1,)),
                FieldSpec(name="x", dtype="int64", shape=(1,)),
            )
        )
    with pytest.raises(ValueError, match="at least one field"):
        Schema(fields=())


def test_schema_json_round_trip() -> None:
    """A schema survives the trip through its own canonical JSON."""
    assert Schema.from_json(_SCHEMA.canonical_json()) == _SCHEMA
    assert Schema.from_json(_SCHEMA.canonical_json()).fingerprint == _SCHEMA.fingerprint
    with pytest.raises(ValueError):
        Schema.from_json("not json")
    with pytest.raises(ValueError):
        Schema.from_json('{"name": "x"}')


def test_a_closed_writer_refuses_appends(tmp_path: Path) -> None:
    """Closing is final; a late append is a bug worth raising on."""
    writer = ShardWriter(tmp_path, _SCHEMA, samples_per_shard=4)
    writer.append(_sample(0))
    writer.close()
    writer.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(_sample(1))
    assert len(ShardReader(tmp_path)) == 1


def test_an_explicit_flush_seals_the_tail_a_quiet_writer_would_hold(
    tmp_path: Path,
) -> None:
    """The heartbeat hook: no appends, no timer, and the tail is still durable.

    The timed trigger is evaluated inside ``append``, so a writer that stops
    appending -- the stalled-worker case -- would sit on its buffer forever.
    ``flush`` is the public escape hatch, and this is the property a caller
    heartbeating it depends on: the samples reach a real, readable shard.
    """
    # An interval long enough that only the explicit flush can seal anything.
    writer = ShardWriter(
        tmp_path, _SCHEMA, samples_per_shard=100, flush_every_seconds=1e6
    )
    for index in range(3):
        writer.append(_sample(index))
    assert writer.pending == 3
    assert list(tmp_path.glob(SHARD_GLOB)) == []

    sealed = writer.flush()
    assert sealed is not None
    assert sealed.exists()
    assert writer.pending == 0
    # Free and idempotent, so a heartbeat may call it at any rate it likes.
    assert writer.flush() is None

    reader = ShardReader(tmp_path, schema=_SCHEMA)
    assert len(reader) == 3
    for index in range(3):
        _assert_same_sample(reader[index], _sample(index))


def test_sealing_refuses_to_publish_over_an_existing_shard(tmp_path: Path) -> None:
    """Two writers on one directory destroyed each other's shards silently.

    Both pick the same next index, and ``os.replace`` would delete a shard
    that has already been counted as safe. Single-writer-per-directory is
    enforced here rather than merely documented.
    """
    writer = ShardWriter(tmp_path, _SCHEMA, samples_per_shard=100)
    writer.append(_sample(0))
    intruder = tmp_path / "shard-000000.npz"
    intruder.write_bytes(b"another writer got here first")

    with pytest.raises(FileExistsError, match="One writer per directory"):
        writer.flush()

    assert intruder.read_bytes() == b"another writer got here first"
    assert writer.pending == 1
    # Nothing half-written was left behind by the refused seal either.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shard-000000.npz"]


def test_writer_rejects_impossible_flush_policies(tmp_path: Path) -> None:
    """A zero shard size or non-positive interval is a config error."""
    with pytest.raises(ValueError, match="samples_per_shard"):
        ShardWriter(tmp_path, _SCHEMA, samples_per_shard=0)
    with pytest.raises(ValueError, match="flush_every_seconds"):
        ShardWriter(tmp_path, _SCHEMA, flush_every_seconds=0.0)


# ---------------------------------------------------------------------------
# KOM-002 §7 E -- the plane-map break, seen from the shard side
# ---------------------------------------------------------------------------

PMV1_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="observation", dtype="float32", shape=(118, 9, 9)),
        FieldSpec(name="policy_target", dtype="float32", shape=(ACTION_SIZE,)),
        FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),
        FieldSpec(name="mlh_target", dtype="float32", shape=()),
        FieldSpec(name="policy_kl", dtype="float32", shape=()),
        FieldSpec(name="root_value", dtype="float32", shape=()),
        FieldSpec(name="search_mode", dtype="int64", shape=()),
        FieldSpec(name="to_play", dtype="int64", shape=()),
        FieldSpec(name="ply", dtype="int64", shape=()),
        FieldSpec(name="game_id", dtype="int64", shape=()),
        FieldSpec(name="sims_used", dtype="int64", shape=()),
    )
)
"""``records.SCHEMA`` as it stood at plane-map version 1 -- 118 observation
planes and no ``plane_map_version`` field.

Rebuilt here rather than imported, because there is nothing left to import it
from: KOM-002 replaced it. Its fingerprint is pinned literally below, which is
what turns "the generation changed" into a checkable statement instead of a
tautology about whatever the schema happens to be today.
"""

PMV1_SCHEMA_FINGERPRINT: Final[str] = (
    "d7532936b77136f3dc4a31f50866e86fd4ed1c324f8142d06e10183f43d1fd78"
)
"""The pre-KOM-002 fingerprint, committed as a literal (design §7 #19)."""


def _pmv1_row() -> dict[str, torch.Tensor]:
    """One zero-filled row shaped for :data:`PMV1_SCHEMA`."""
    return {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in PMV1_SCHEMA.fields
    }


def test_the_schema_fingerprint_changed() -> None:
    """#19: the break is intentional, and the number it broke is on record.

    The shard fingerprint is a ``sha256`` over the canonical JSON of every
    field's name, dtype and shape in order, so widening the observation from
    ``(118, 9, 9)`` to ``(128, 9, 9)`` -- and appending
    ``plane_map_version`` -- necessarily moves it. Pinning the OLD digest is
    what makes this an assertion: "the current fingerprint differs from
    whatever the current fingerprint used to be" would pass forever.
    """
    assert PMV1_SCHEMA.fingerprint == PMV1_SCHEMA_FINGERPRINT, (
        "the reconstruction of the pre-KOM-002 schema no longer hashes to the "
        "digest KOM-002 broke; the reconstruction is what to fix, not the "
        "literal"
    )
    assert SCHEMA.fingerprint != PMV1_SCHEMA_FINGERPRINT
    assert observation_size()[0] == 128
    assert "plane_map_version" in SCHEMA.names


def test_a_118_plane_shard_is_refused(tmp_path: Path) -> None:
    """#20: an old shard cannot enter a new window -- loudly, by fingerprint.

    Two readings of the same refusal:

    * the raw check raises ``ValueError`` naming BOTH fingerprints, which is
      the message an operator has to act on;
    * the reader turns that into a skip plus a ``ShardWarning`` and reads the
      directory as empty, which is what keeps one stale shard from taking a
      campaign down -- while making it impossible for the shard to be silently
      mixed into a post-KOM-002 window.
    """
    with ShardWriter(tmp_path, PMV1_SCHEMA, samples_per_shard=1) as writer:
        writer.append(_pmv1_row())
    shard = tmp_path / "shard-000000.npz"
    assert shard.is_file()

    with pytest.raises(ValueError) as raised:
        _validated_arrays(_load_shard_arrays(shard), SCHEMA)
    message = str(raised.value)
    assert PMV1_SCHEMA_FINGERPRINT in message
    assert SCHEMA.fingerprint in message

    with pytest.warns(ShardWarning, match="fingerprint"):
        reader = ShardReader(tmp_path, schema=SCHEMA)
    assert len(reader) == 0
    assert [skip.path.name for skip in reader.skipped] == ["shard-000000.npz"]
    assert PMV1_SCHEMA_FINGERPRINT in reader.skipped[0].reason


# ---------------------------------------------------------------------------
# KOM-010 -- the ownership target, seen from the shard side
# ---------------------------------------------------------------------------

PRE_KOM010_SCHEMA: Final[Schema] = Schema(
    fields=(
        FieldSpec(name="observation", dtype="float32", shape=(128, 9, 9)),
        FieldSpec(name="policy_target", dtype="float32", shape=(ACTION_SIZE,)),
        FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),
        FieldSpec(name="mlh_target", dtype="float32", shape=()),
        FieldSpec(name="policy_kl", dtype="float32", shape=()),
        FieldSpec(name="root_value", dtype="float32", shape=()),
        FieldSpec(name="search_mode", dtype="int64", shape=()),
        FieldSpec(name="to_play", dtype="int64", shape=()),
        FieldSpec(name="ply", dtype="int64", shape=()),
        FieldSpec(name="game_id", dtype="int64", shape=()),
        FieldSpec(name="sims_used", dtype="int64", shape=()),
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
    )
)
"""``records.SCHEMA`` as it stood immediately before KOM-010 -- twelve fields,
no ``final_control``.

Rebuilt rather than imported, for the reason :data:`PMV1_SCHEMA` is: KOM-010
replaced it, so there is nothing left to import. Note that it is *not* the
plane-map-version-1 schema above -- the observation is already 128 planes and
``plane_map_version`` is already present. Two reconstructions, two literals,
because a shard can be stale in either of two independent ways.
"""

PRE_KOM010_SCHEMA_FINGERPRINT: Final[str] = (
    "cc964546f8546cc0810ea6ebc6f438c0b403c84ed34c65c5f2e610ac849fedf6"
)
"""The pre-KOM-010 fingerprint, committed as a literal (KOM-002 design §7 #19's
pattern: pin the digest the change BROKE, so the assertion has content)."""


def test_the_schema_fingerprint_changed_for_final_control() -> None:
    """KOM-010 appended a field, so the data generation is a new one.

    ``final_control`` is a ``(81,)`` ``int64`` appended after
    ``plane_map_version``, which moves the ``sha256`` over the field list --
    and it must, because a pre-KOM-010 shard has no ownership target and the
    trainer's ownership head (KOM-005) would read a column that is not there.
    Unlike the plane-map break, this one the fingerprint CAN see: the field set
    itself changed. Pinning the old digest is what makes the assertion
    falsifiable; "the fingerprint differs from itself" would pass forever.
    """
    assert PRE_KOM010_SCHEMA.fingerprint == PRE_KOM010_SCHEMA_FINGERPRINT, (
        "the reconstruction of the pre-KOM-010 schema no longer hashes to the "
        "digest KOM-010 broke; the reconstruction is what to fix, not the "
        "literal"
    )
    assert SCHEMA.fingerprint != PRE_KOM010_SCHEMA_FINGERPRINT
    assert SCHEMA.fingerprint != PMV1_SCHEMA_FINGERPRINT
    # Appended, not inserted: every pre-existing field keeps its position, so
    # the change is the one KOM-010 pre-registered and not a reordering.
    assert SCHEMA.names[: len(PRE_KOM010_SCHEMA.names)] == PRE_KOM010_SCHEMA.names
    # KOM-014 appended ``action`` after ``final_control`` and KOM-005b appended
    # three more after that, so the tail has moved on twice -- but the property
    # KOM-010 pinned is unchanged and is what is asserted: the prefix above is
    # untouched, and ``final_control`` is still the (81,) int64 field KOM-010
    # added, at the position it has always had.
    by_name = {field.name: field for field in SCHEMA.fields}
    assert by_name["final_control"].dtype == "int64"
    assert by_name["final_control"].shape == (CONTROL_SIZE,)
    assert SCHEMA.names[len(PRE_KOM010_SCHEMA.names)] == "final_control"
    # Relative index, never an absolute position: every field KOM-010 predates
    # comes before it and every field appended since comes after, which is the
    # claim -- and it survives the next append, which ``names[-1] == "action"``
    # did not.
    assert SCHEMA.names.index("final_control") < SCHEMA.names.index("action")


# ---------------------------------------------------------------------------
# KOM-014 -- the recorded action, seen from the shard side
#
# These two nodes are §8's #24 and #25. The design sites them in a new
# ``tests/train/test_records.py``; that file cannot exist, because
# ``tests/eval/test_records.py`` already does and neither directory is a package
# -- two same-named test modules collide in pytest's prepend import mode and in
# mypy's module map. They are sited here instead, which is the file that already
# owns "what a schema change means for a shard".
# ---------------------------------------------------------------------------


def test_the_schema_appends_every_new_field_after_the_ones_that_predate_it() -> None:
    """Every append lands after every field that predates it.

    The rewrite of ``test_the_schema_ends_with_plane_map_version_then_action``,
    and the **second** time this node has been amended in the same direction.
    KOM-014 §8 #24 originally read
    ``SCHEMA.names[-2:] == ("plane_map_version", "action")``, written when
    ``plane_map_version`` was expected to be the tail; KOM-010 appended
    ``final_control`` after it, so the implementing worker amended it to
    ``names[-1] == "action"`` plus a relative-index chain. KOM-005b then
    appends three columns, which falsifies the ``[-1]`` literal by any correct
    implementation of the design's own "append LAST" rule -- and weakening the
    rule to satisfy the literal is exactly the mistake the rule exists to
    prevent.

    So the literal is dropped and the *claim* KOM-014 needs is kept, in the
    form that is stable under append: the chain of relative indices below says
    ``action`` comes after every field that predates it, and it will still say
    so after the next appender. The tail itself is owned by
    ``tests/selfplay/unit/test_staged_targets.py``'s suffix node, which is
    where a new append is expected to be edited.
    """
    names = SCHEMA.names
    assert (
        names.index("plane_map_version")
        < names.index("final_control")
        < names.index("action")
        < names.index("termination")
    )


def test_a_pre_action_schema_has_a_different_fingerprint() -> None:
    """A new field is a new data generation -- asserted as a property, not a digest.

    No hex literal is pinned: the golden canonical JSON in
    ``tests/selfplay/unit/test_worker_unit.py`` is the literal whose job that is,
    and a second one here would have to be hand-edited on every future append
    for no extra coverage. What must never be true is that a shard written before
    ``action`` existed passes as this generation's -- it has no ``action``
    column, and a reader holding it to this schema would read a column that is
    not there.

    Sliced **by name**, not by ``[:-1]``. Position-sliced, this node stopped
    being about its own subject the moment KOM-005b appended: ``fields[:-1]``
    then drops ``draft_outcome`` and still contains ``action``, so the first
    assertion fails for a reason that has nothing to do with a pre-``action``
    generation. By name it reconstructs the generation it is actually about,
    and it is stable under every future append.
    """
    pre_bump = Schema(fields=SCHEMA.fields[: SCHEMA.names.index("action")])
    assert "action" not in pre_bump.names
    assert pre_bump.fingerprint != SCHEMA.fingerprint


def test_a_pre_kom010_shard_is_refused(tmp_path: Path) -> None:
    """A shard written without the ownership target cannot join a new window.

    The same refusal ``test_a_118_plane_shard_is_refused`` pins for the
    plane-map break, on the field-set break: the raw check names both
    fingerprints, and the reader downgrades that to a skip plus a warning so
    one stale directory cannot take a campaign down.
    """
    row = {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in PRE_KOM010_SCHEMA.fields
    }
    with ShardWriter(tmp_path, PRE_KOM010_SCHEMA, samples_per_shard=1) as writer:
        writer.append(row)

    with pytest.raises(ValueError) as raised:
        _validated_arrays(_load_shard_arrays(tmp_path / "shard-000000.npz"), SCHEMA)
    message = str(raised.value)
    assert PRE_KOM010_SCHEMA_FINGERPRINT in message
    assert SCHEMA.fingerprint in message

    with pytest.warns(ShardWarning, match="fingerprint"):
        reader = ShardReader(tmp_path, schema=SCHEMA)
    assert len(reader) == 0
