"""Compressed self-play sample shards, written to survive a preemption.

A training sample is a mapping of named tensors -- ``float32`` features and
targets, ``int64`` labels -- whose names, dtypes and per-sample shapes are
fixed by a :class:`Schema` handed to the writer. Samples accumulate in memory
and are sealed into shards; a shard is the unit that a reader ever sees.

Storage format -- one ``.npz`` per shard
----------------------------------------
:func:`numpy.savez_compressed` (a zip container, DEFLATE per member) was
chosen over ``torch.save`` for three reasons that matter here and nowhere
else: the container is a documented, tool-inspectable format rather than a
pickle; every member carries its own CRC-32, so bit rot is detectable without
trusting us; and loading needs no unpickler and therefore no
``weights_only`` question. The cost is that tensors round-trip through numpy,
which for CPU ``float32``/``int64`` is a byte-exact reinterpretation, not a
conversion. Compression is zlib at numpy's default level; zstd is not in the
pinned devshell closure, and the level knob was deliberately not exposed
rather than shipping a config field nothing reads.

Fields are stored *batched*: a shard holding ``n`` samples stores each schema
field once, as an array of shape ``(n, *field.shape)``. Four reserved keys
travel with them:

``__count__``
    ``n``, as an ``int64`` scalar. The authority on how many samples the
    shard holds -- never inferred from an array's leading dimension.
``__schema__``
    The canonical JSON of the schema the writer was constructed with, so a
    reader can rebuild the schema without being told it.
``__fingerprint__``
    ``sha256`` of that canonical JSON. A shard whose fingerprint does not
    match its own embedded schema, or does not match the schema the reader
    was given, is not this dataset's shard.
``__digest__``
    ``sha256`` over the fingerprint, the count, and every field's dtype,
    shape and raw bytes, in schema order. This is the integrity check that
    does not depend on zip internals: a flipped byte that somehow survives
    DEFLATE and CRC-32 still fails here.

Crash-safety invariants
-----------------------
These are the claims a reviewer should attack; each is enforced in exactly
one place in this module.

1. **No partially written shard is ever observable.** A shard is built in a
   temporary file named ``.shard-NNNNNN.npz.tmp-<pid>``, which is both
   dot-prefixed and outside the reader's ``shard-*.npz`` glob, so it is
   invisible to a reader twice over. The file is ``fsync``-ed, then
   :func:`os.replace` publishes it under its final name -- an atomic rename
   within one directory -- and the *directory* is then ``fsync``-ed so the
   new name itself survives a power loss. Readers therefore see a shard
   either not at all or entirely.
2. **A preemption forfeits minutes of *appends*, not an iteration.** Samples
   are sealed into a shard when ``samples_per_shard`` accumulate *or* when
   ``flush_every_seconds`` have passed since the last seal, whichever comes
   first. The time-triggered seal writes a *short but complete* shard; short
   shards are ordinary shards, not a special case anywhere in the reader.

   Both triggers are checked inside :meth:`ShardWriter.append`, and there is
   no timer thread, so the bound is stated in production-time rather than
   wall-clock: **the buffer never holds more than one interval's worth of
   appends, plus whatever the current inter-append gap adds.** A writer that
   stops appending -- a stalled self-play worker, a half-hour game, a process
   parked in a debugger -- keeps its tail buffered for exactly as long as it
   stays quiet. This is not the same as "the buffer's age is bounded by
   ``flush_every_seconds``", which is the stronger claim a timer would buy
   and this module deliberately does not make. A caller that needs the
   wall-clock bound owns it: :meth:`ShardWriter.flush` is public precisely so
   it can be called from a heartbeat, and sealing an empty buffer is free.
3. **A crashed writer costs at most its buffer, never a written shard.**
   Nothing in this module opens an existing shard for writing, truncates one,
   or reuses a shard index: :func:`_next_shard_index` resumes after the
   highest index already present, so a writer restarted on a populated
   directory appends rather than overwrites. One writer per directory is
   *enforced*, not merely advised: two writers pick the same next index, so
   :meth:`ShardWriter._seal` refuses to publish over a name that already
   exists and raises ``FileExistsError`` instead of letting one writer's
   ``os.replace`` destroy the other's shard.
4. **A damaged shard costs that shard.** :class:`ShardReader` validates every
   shard at construction and skips the ones it cannot trust -- truncated
   file, broken DEFLATE stream, bad CRC, missing reserved key, foreign
   fingerprint, digest mismatch, wrong dtype or shape -- with a
   :class:`ShardWarning` naming the file and the reason. It does not raise,
   because the common instance of a damaged shard is the trailing one a kill
   interrupted, and losing a window because its last shard is stubby is the
   failure this module exists to prevent.

What is *not* claimed: that a sealed shard is durable the instant
:meth:`ShardWriter.append` returns (it is durable once sealed, and sealing is
what the flush policy schedules), and that a torn tmp file cleans itself up.
Leftover ``.tmp-`` files are left on disk deliberately -- they are evidence of
a crash, they cost one shard's disk, and they are inert to every reader.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

_TORCH_DTYPES: Final[Mapping[str, torch.dtype]] = {
    "float32": torch.float32,
    "int64": torch.int64,
}
"""The dtypes a shard may hold. Deliberately short: a training sample is
features, distributions and labels, and anything else is a schema mistake."""

COUNT_KEY: Final[str] = "__count__"
SCHEMA_KEY: Final[str] = "__schema__"
FINGERPRINT_KEY: Final[str] = "__fingerprint__"
DIGEST_KEY: Final[str] = "__digest__"

RESERVED_KEYS: Final[frozenset[str]] = frozenset(
    {COUNT_KEY, SCHEMA_KEY, FINGERPRINT_KEY, DIGEST_KEY}
)

_SAVEZ_PARAMETERS: Final[frozenset[str]] = frozenset({"file", "args", "kwds", "allow_pickle"})
"""Names :func:`numpy.savez_compressed` uses for its own parameters.

Members are passed to it as keyword arguments, so a field called ``file``
would be swallowed as the destination rather than stored. Rejected at schema
construction, where the mistake is a one-line fix, rather than discovered as a
``TypeError`` deep in a self-play worker.
"""

SHARD_GLOB: Final[str] = "shard-*.npz"
_SHARD_NAME: Final[str] = "shard-{index:06d}.npz"
_SHARD_RE: Final[re.Pattern[str]] = re.compile(r"shard-(\d{6,})\.npz")

_SHARD_MODE: Final[int] = 0o644

_CORRUPTION_ERRORS: Final[tuple[type[Exception], ...]] = (
    zipfile.BadZipFile,
    zlib.error,
    OSError,
    ValueError,
    KeyError,
    EOFError,
)
"""Everything a damaged shard can throw on the way back in.

``zipfile.BadZipFile`` covers truncation and a lost central directory,
``zlib.error`` a scrambled DEFLATE stream, ``ValueError`` numpy's own header
complaints plus our schema and digest checks, ``KeyError`` a missing reserved
key. Caught as a tuple rather than bare ``Exception`` so that a genuine bug in
this module still escapes as a traceback instead of being reported as bit rot.
"""


class ShardWarning(UserWarning):
    """A shard was skipped at read time.

    Its own category so a caller can escalate it to an error (a trainer that
    would rather stop than silently train on 90% of a window) or count it,
    without also catching unrelated warnings.
    """


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named tensor in a sample: its dtype and its per-sample shape.

    ``shape`` is the shape of a *single* sample's tensor, so a policy target
    over 4096 actions is ``shape=(4096,)``. Scalars are ``shape=()``.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject names and dtypes that would make a shard unreadable."""
        if not self.name:
            raise ValueError("field name must be non-empty")
        if self.name in RESERVED_KEYS or self.name.startswith("__"):
            raise ValueError(
                f"field name {self.name!r} is reserved: names starting with "
                "'__' belong to the shard container"
            )
        if not self.name.isidentifier() or self.name in _SAVEZ_PARAMETERS:
            raise ValueError(
                f"field name {self.name!r} is not usable as a shard member: "
                "it must be a Python identifier and must not collide with "
                f"numpy's savez parameters {sorted(_SAVEZ_PARAMETERS)}"
            )
        if self.dtype not in _TORCH_DTYPES:
            raise ValueError(
                f"field {self.name!r} has unsupported dtype {self.dtype!r}; "
                f"supported: {sorted(_TORCH_DTYPES)}"
            )
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(
                f"field {self.name!r} has non-positive dimension in {self.shape!r}"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype a tensor for this field must have."""
        return _TORCH_DTYPES[self.dtype]


@dataclass(frozen=True, slots=True)
class Schema:
    """The full set of named tensors a sample must carry, in a fixed order.

    Order is part of the schema: it fixes the digest computation and the
    canonical JSON, so two schemas with the same fields in a different order
    have different fingerprints and their shards do not mix. That is
    intentional -- a silent reordering is exactly the kind of change that
    should invalidate a window rather than be absorbed.
    """

    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        """Reject an empty schema or duplicate field names."""
        if not self.fields:
            raise ValueError("schema must declare at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"schema has duplicate field names: {names}")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the field names, in schema order."""
        return tuple(field.name for field in self.fields)

    def canonical_json(self) -> str:
        """Return the schema's canonical serialisation.

        Keys sorted and separators tight, so the text -- and therefore the
        fingerprint -- depends only on the schema's content and field order,
        never on how it was constructed.
        """
        payload = [
            {"name": field.name, "dtype": field.dtype, "shape": list(field.shape)}
            for field in self.fields
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Return ``sha256`` of :meth:`canonical_json`, as hex."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> Schema:
        """Rebuild a schema from :meth:`canonical_json` output.

        Raises ``ValueError`` for anything that is not a list of well-formed
        field objects, so a tampered ``__schema__`` member is corruption
        rather than a surprising attribute error later.
        """
        try:
            parsed: object = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"schema is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"schema must be a JSON array, got {type(parsed).__name__}")
        fields: list[FieldSpec] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                raise ValueError(f"schema entry must be an object, got {entry!r}")
            name = entry.get("name")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            if not isinstance(name, str) or not isinstance(dtype, str):
                raise ValueError(f"schema entry has a bad name/dtype: {entry!r}")
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            ):
                raise ValueError(f"schema entry has a bad shape: {entry!r}")
            fields.append(FieldSpec(name=name, dtype=dtype, shape=tuple(shape)))
        return cls(fields=tuple(fields))

    def validate(self, sample: Mapping[str, torch.Tensor]) -> None:
        """Check one sample against this schema, raising ``ValueError`` if it fails.

        Exact key equality: a missing field is a hole in the batch and an
        extra field is a name that would be silently dropped at seal time, so
        both are rejected here rather than at read time by someone else.
        """
        present = set(sample)
        expected = set(self.names)
        if present != expected:
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            raise ValueError(
                f"sample does not match schema: missing={missing} extra={extra}"
            )
        for field in self.fields:
            tensor = sample[field.name]
            if tensor.dtype is not field.torch_dtype:
                raise ValueError(
                    f"field {field.name!r} must be {field.dtype}, "
                    f"got {tensor.dtype}"
                )
            if tuple(tensor.shape) != field.shape:
                raise ValueError(
                    f"field {field.name!r} must have shape {field.shape}, "
                    f"got {tuple(tensor.shape)}"
                )


def _fsync_directory(directory: Path) -> None:
    """``fsync`` a directory, so a rename into it survives a power loss.

    Renaming is atomic with respect to readers the instant it returns, but the
    *name* is only durable once the directory's own metadata is on the
    platter. Without this, a crash can leave a shard whose bytes exist and
    whose name does not.
    """
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _payload_digest(
    schema: Schema, count: int, arrays: Mapping[str, npt.NDArray[Any]]
) -> str:
    """Return the integrity digest over a shard's payload.

    Covers the schema fingerprint, the sample count, and for each field in
    schema order its name, dtype string and exact bytes. Byte-order is part of
    the dtype string, so a shard moved between a little- and big-endian host
    fails the digest instead of loading transposed nonsense -- the honest
    outcome, and the reason this is not merely a checksum of the compressed
    container.
    """
    hasher = hashlib.sha256()
    hasher.update(schema.fingerprint.encode("ascii"))
    hasher.update(str(count).encode("ascii"))
    for field in schema.fields:
        array = np.ascontiguousarray(arrays[field.name])
        hasher.update(field.name.encode("utf-8"))
        hasher.update(str(array.dtype.str).encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _next_shard_index(directory: Path) -> int:
    """Return the first shard index not already used in `directory`.

    Resumption, not creation: a writer restarted after a kill must not reuse
    an index, because :func:`os.replace` onto an existing shard would destroy
    a shard that has already been counted as safe.
    """
    highest = -1
    for path in directory.glob(SHARD_GLOB):
        match = _SHARD_RE.fullmatch(path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


class ShardWriter:
    """Accumulates samples and seals them into atomically published shards.

    Sealing happens on two triggers, checked in :meth:`append`: the buffer
    reaching ``samples_per_shard``, or ``flush_every_seconds`` elapsing since
    the previous seal. The second is the preemption budget -- it caps how much
    unsealed work a kill can destroy, at the price of shards shorter than
    ``samples_per_shard``, which readers treat identically.

    Because both triggers live in :meth:`append`, the interval bounds the
    backlog *in appends*, not in wall-clock: a writer that goes quiet holds
    its tail until the next append or until someone calls :meth:`flush`.
    Callers that care about the wall-clock bound should call :meth:`flush`
    from whatever heartbeat they already have.

    ``clock`` is injectable purely so the flush policy is testable without
    sleeping; it must be monotonic, and defaults to :func:`time.monotonic`
    rather than wall-clock so an NTP step cannot make the interval negative.

    The writer copies every tensor it is handed, so callers may reuse a
    scratch buffer between :meth:`append` calls.

    Not thread-safe, and one writer per directory -- enforced rather than
    hoped for: two writers would pick the same next index, and the second to
    seal raises ``FileExistsError`` rather than replacing a shard the first
    has already published (see :meth:`_seal`).
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        schema: Schema,
        *,
        samples_per_shard: int = 1024,
        flush_every_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Open `directory` for writing, creating it if needed."""
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        if flush_every_seconds <= 0.0:
            raise ValueError(
                f"flush_every_seconds must be > 0, got {flush_every_seconds}"
            )
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._samples_per_shard = samples_per_shard
        self._flush_every_seconds = flush_every_seconds
        self._clock = clock
        self._buffer: list[dict[str, npt.NDArray[Any]]] = []
        self._next_index = _next_shard_index(self._directory)
        self._last_seal_at = clock()
        self._closed = False
        self._sealed: list[Path] = []

    @property
    def directory(self) -> Path:
        """Return the directory shards are published into."""
        return self._directory

    @property
    def schema(self) -> Schema:
        """Return the schema every appended sample is validated against."""
        return self._schema

    @property
    def pending(self) -> int:
        """Return how many samples are buffered but not yet sealed.

        This is exactly what a kill would forfeit.
        """
        return len(self._buffer)

    @property
    def sealed_shards(self) -> tuple[Path, ...]:
        """Return the shards this writer has published, oldest first."""
        return tuple(self._sealed)

    def append(self, sample: Mapping[str, torch.Tensor]) -> None:
        """Validate and buffer one sample, sealing a shard if a trigger fires.

        Raises ``ValueError`` if the sample does not match the schema and
        ``RuntimeError`` if the writer is closed. A rejected sample does not
        enter the buffer, so a schema violation cannot poison a shard.
        """
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        self._schema.validate(sample)
        row: dict[str, npt.NDArray[Any]] = {
            field.name: sample[field.name].detach().cpu().clone().contiguous().numpy()
            for field in self._schema.fields
        }
        self._buffer.append(row)
        now = self._clock()
        if len(self._buffer) >= self._samples_per_shard:
            self._seal(now)
        elif now - self._last_seal_at >= self._flush_every_seconds:
            self._seal(now)

    def flush(self) -> Path | None:
        """Seal the buffer now, returning the shard written, or ``None`` if empty.

        The manual form of the periodic trigger, and the only way to bound the
        backlog in wall-clock time: the timed trigger is evaluated inside
        :meth:`append`, so a writer that has stopped appending never seals on
        its own. A caller that has just finished an iteration -- or that runs
        a heartbeat and wants the tail durable regardless of self-play
        throughput -- calls this. Sealing an empty buffer is a no-op returning
        ``None``, so a heartbeat may call it as often as it likes.

        Raises ``FileExistsError`` if the next shard name is already taken,
        which means a second writer is publishing into this directory.
        """
        if not self._buffer:
            return None
        return self._seal(self._clock())

    def close(self) -> None:
        """Seal any buffered samples and refuse further appends.

        Idempotent. After this returns, every sample ever accepted by
        :meth:`append` is in a published shard.
        """
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self) -> ShardWriter:
        """Return self; the writer is already open."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer, sealing the tail even when the body raised.

        Deliberately does not suppress the exception: the samples are saved,
        the error is still the caller's problem.
        """
        self.close()

    def _seal(self, now: float) -> Path:
        """Write the buffer as one shard and publish it atomically.

        Raises ``FileExistsError`` if this writer's next shard name is already
        on disk. Only two things produce that: a second writer in the same
        directory, or a directory that grew a shard under this writer's feet.
        Both mean :func:`os.replace` below would delete somebody else's
        published, already-counted-as-safe shard, so the seal stops here --
        loudly, with the buffer intact and nothing written -- rather than
        losing data silently to satisfy a call that was invalid to begin with.
        """
        count = len(self._buffer)
        assert count > 0, "_seal called with an empty buffer"

        index = self._next_index
        final_path = self._directory / _SHARD_NAME.format(index=index)
        if final_path.exists():
            raise FileExistsError(
                f"refusing to publish over {final_path}: the name is already "
                "taken, which means another writer is publishing into "
                f"{self._directory}. One writer per directory."
            )

        arrays: dict[str, npt.NDArray[Any]] = {}
        for field in self._schema.fields:
            stacked = np.stack([row[field.name] for row in self._buffer])
            arrays[field.name] = np.ascontiguousarray(
                stacked.astype(field.dtype, copy=False)
            )

        payload: dict[str, npt.NDArray[Any]] = dict(arrays)
        payload[COUNT_KEY] = np.array(count, dtype=np.int64)
        payload[SCHEMA_KEY] = np.array(self._schema.canonical_json())
        payload[FINGERPRINT_KEY] = np.array(self._schema.fingerprint)
        payload[DIGEST_KEY] = np.array(_payload_digest(self._schema, count, arrays))

        # Dot-prefixed *and* outside SHARD_GLOB: a reader cannot see this file
        # under either rule. The pid keeps a stale tmp from a previous process
        # from being clobbered, so crash evidence survives.
        tmp_path = self._directory / f".{final_path.name}.tmp-{os.getpid()}"

        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SHARD_MODE)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                # A file object, not a path: numpy appends '.npz' to any path
                # that does not already end in it, which would rename our tmp
                # file out from under the replace below.
                np.savez_compressed(handle, **payload)  # type: ignore[arg-type]  # numpy stubs mistype savez_compressed's **kwargs
                handle.flush()
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, final_path)
        _fsync_directory(self._directory)

        self._buffer.clear()
        self._next_index = index + 1
        self._last_seal_at = now
        self._sealed.append(final_path)
        return final_path


def _load_shard_arrays(path: Path) -> dict[str, npt.NDArray[Any]]:
    """Read every member of a shard into memory.

    Eager by design: pulling each member all the way out is what forces
    DEFLATE and the per-member CRC-32 to be checked, so an unreadable shard
    fails here rather than three batches into training.
    """
    loaded = np.load(path, allow_pickle=False)
    try:
        return {str(name): loaded[name] for name in loaded.files}
    finally:
        loaded.close()


def _validated_arrays(
    arrays: Mapping[str, npt.NDArray[Any]], expected: Schema | None
) -> tuple[Schema, int, dict[str, npt.NDArray[Any]]]:
    """Check a shard's reserved keys and payload, or raise ``ValueError``.

    The checks, in the order a corruption is cheapest to detect: reserved keys
    present, embedded schema parses, fingerprint matches that schema,
    fingerprint matches the reader's schema if it has one, count is sane,
    every field has the right dtype and shape, digest matches.
    """
    for key in (COUNT_KEY, SCHEMA_KEY, FINGERPRINT_KEY, DIGEST_KEY):
        if key not in arrays:
            raise ValueError(f"shard is missing reserved key {key!r}")

    schema = Schema.from_json(str(arrays[SCHEMA_KEY].item()))
    fingerprint = str(arrays[FINGERPRINT_KEY].item())
    if schema.fingerprint != fingerprint:
        raise ValueError(
            "embedded fingerprint does not match the embedded schema "
            f"({fingerprint} != {schema.fingerprint})"
        )
    if expected is not None and fingerprint != expected.fingerprint:
        raise ValueError(
            f"shard fingerprint {fingerprint} is not the requested "
            f"{expected.fingerprint}"
        )

    count = int(arrays[COUNT_KEY].item())
    if count < 0:
        raise ValueError(f"shard declares a negative count {count}")

    fields: dict[str, npt.NDArray[Any]] = {}
    for field in schema.fields:
        if field.name not in arrays:
            raise ValueError(f"shard is missing field {field.name!r}")
        array = arrays[field.name]
        if array.dtype != np.dtype(field.dtype):
            raise ValueError(
                f"field {field.name!r} has dtype {array.dtype}, expected {field.dtype}"
            )
        if tuple(array.shape) != (count, *field.shape):
            raise ValueError(
                f"field {field.name!r} has shape {tuple(array.shape)}, "
                f"expected {(count, *field.shape)}"
            )
        fields[field.name] = array

    digest = str(arrays[DIGEST_KEY].item())
    recomputed = _payload_digest(schema, count, fields)
    if digest != recomputed:
        raise ValueError(f"payload digest mismatch ({digest} != {recomputed})")
    return schema, count, fields


def read_columns(
    path: Path, names: Sequence[str]
) -> tuple[Schema, int, dict[str, npt.NDArray[Any]]]:
    """Return a shard's schema, count, and **only** the named field columns.

    The projecting reader KOM-014 §2.4 needs, and the reason it cannot use
    :class:`ShardReader`: that class decompresses and digest-verifies every
    shard of a directory at construction, and :meth:`ShardReader.__getitem__`
    materialises *every* field of a row (~190 KB at production shapes). A book
    builder wants three ``int64`` columns out of a whole window and one
    ``policy_target`` column out of a bounded subset of shards, which is two
    orders of magnitude less work -- and a shard is a zip of independently
    deflated members, so ``np.load(path)[name]`` really does cost one member.

    Names are a *projection*, not a schema: ``names=()`` reads the reserved keys
    alone, which is how a caller asks "what generation is this shard?" without
    paying for a single row.

    **What this gives up, stated rather than hidden.** :data:`DIGEST_KEY` covers
    the whole payload in schema order, so a projection **cannot** verify it and
    this function does not try. What it *does* re-run, from
    :func:`_validated_arrays`, is every check that a projection can honestly
    make: the four reserved keys are present, the embedded ``__schema__``
    parses, the embedded fingerprint matches that schema, the count is
    non-negative, and each **named** column has the dtype and the
    ``(count, *shape)`` the embedded schema declares for it. A caller is
    therefore a *digest-unverified* reader, which KOM-014 §2.4 accepts for the
    book builder on two grounds it states there -- every shard it projects is
    digest-verified by the training window build of the same or an adjacent
    iteration, and a corrupted prefix is caught twice more downstream (the
    legality replay at build time and the load-and-replay guard in the worker).
    Nothing else in this repository should adopt it on weaker grounds.

    Read-only and stateless: no existing call site changes, and this function
    deliberately does **not** go through :func:`_load_shard_arrays`, whose whole
    contract is to pull every member out.

    Args:
        path: The shard file.
        names: The field names to materialise, in any order. Each must be a
            field of the shard's **own** embedded schema. The shard's schema is
            adopted and returned rather than checked against a caller's: the
            builder's fingerprint guard (KOM-014 §2.1) is a *decision* about
            which shards to mine, not a refusal, so it is made by the caller
            against the returned schema.

    Returns:
        ``(schema, count, columns)`` -- the shard's embedded schema, its
        authoritative sample count, and one ``(count, *shape)`` array per
        requested name.

    Raises:
        ValueError: On a missing reserved key, an unparseable or
            self-inconsistent embedded schema, a negative count, a requested
            name the shard's schema does not declare, or a column whose dtype
            or shape disagrees with that declaration.
    """
    loaded = np.load(path, allow_pickle=False)
    try:
        present = {str(name) for name in loaded.files}
        for key in (COUNT_KEY, SCHEMA_KEY, FINGERPRINT_KEY, DIGEST_KEY):
            if key not in present:
                raise ValueError(f"shard {path.name} is missing reserved key {key!r}")

        schema = Schema.from_json(str(loaded[SCHEMA_KEY].item()))
        fingerprint = str(loaded[FINGERPRINT_KEY].item())
        if schema.fingerprint != fingerprint:
            raise ValueError(
                f"shard {path.name}: embedded fingerprint does not match the "
                f"embedded schema ({fingerprint} != {schema.fingerprint})"
            )
        count = int(loaded[COUNT_KEY].item())
        if count < 0:
            raise ValueError(f"shard {path.name} declares a negative count {count}")

        by_name = {field.name: field for field in schema.fields}
        columns: dict[str, npt.NDArray[Any]] = {}
        for name in names:
            field = by_name.get(name)
            if field is None:
                raise ValueError(
                    f"shard {path.name} has no field {name!r}; its schema "
                    f"declares {list(schema.names)}"
                )
            if name not in present:
                raise ValueError(f"shard {path.name} is missing field {name!r}")
            array = loaded[name]
            if array.dtype != np.dtype(field.dtype):
                raise ValueError(
                    f"shard {path.name}: field {name!r} has dtype "
                    f"{array.dtype}, expected {field.dtype}"
                )
            if tuple(array.shape) != (count, *field.shape):
                raise ValueError(
                    f"shard {path.name}: field {name!r} has shape "
                    f"{tuple(array.shape)}, expected {(count, *field.shape)}"
                )
            columns[name] = array
        return schema, count, columns
    finally:
        loaded.close()


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """A shard that passed validation, and how many samples it holds."""

    path: Path
    count: int


@dataclass(frozen=True, slots=True)
class SkippedShard:
    """A shard that failed validation, and why."""

    path: Path
    reason: str


class ShardReader:
    """Reads samples back out of a shard directory, tolerating damage.

    Construction scans every ``shard-*.npz`` in the directory and fully
    decompresses it, which is what makes :meth:`__len__` an exact count rather
    than an estimate and what turns a damaged shard into a
    :class:`ShardWarning` at a predictable moment. Shards that fail land in
    :attr:`skipped` and are absent from every other view; the rest are usable
    in full.

    Memory, and why the default is to spend it
    ------------------------------------------
    That scan already decompresses and digest-verifies every shard, and a
    sealed shard never changes, so ``retain_arrays=True`` (the default) simply
    *keeps* the decoded arrays the scan produced: reads then slice memory, with
    no further file I/O, DEFLATE or digest work for the life of the reader.
    The price is honest -- the reader holds its whole window decompressed in
    RAM. That is NOT small: zero-heavy float32 planes compress ~200x, so a
    production campaign iteration that is ~7 MB on disk is **~1.5-2.2 GB
    decompressed** (measured, KOM-033 review F1). A retained reader is for
    data that will actually be sampled; whoever builds readers over many
    iterations must retain only the ones its window can use
    (:func:`komugi.campaign.loop.window_refs` does exactly that), or the
    resident set grows by GBs per iteration without bound.

    ``retain_arrays=False`` restores the older behaviour byte for byte:
    nothing is retained, and reads re-open and re-validate the shard behind a
    one-shard cache. That is the mode for a memory-constrained reader, and it
    is genuinely slower -- a sampler drawing uniformly across a window's shards
    misses the one-shard cache almost every time, so nearly every row costs a
    whole shard's decompression and a whole shard's digest. One more mode
    divergence to know about: a shard file deleted after construction is
    invisible to a retained reader but raises ``OSError`` from an unretained
    one on its next cache miss -- neither mode promises anything about files
    removed from under a live reader.

    Passing ``schema`` additionally requires each shard's fingerprint to match
    it. Without it, the reader adopts the first valid shard's schema and holds
    later shards to *that*, so a directory that accidentally mixes two
    generations yields one of them plus warnings, never an interleaving.

    A reader is a snapshot: shards sealed after construction are not seen.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        schema: Schema | None = None,
        retain_arrays: bool = True,
    ) -> None:
        """Scan and validate every shard in `directory`.

        With `retain_arrays` (the default) each valid shard's decoded arrays
        are kept as the scan produces them, so no shard is ever read,
        decompressed or digested twice. Validation is unchanged either way:
        it happens exactly once per shard, here.
        """
        self._directory = Path(directory)
        expected = schema
        shards: list[ShardInfo] = []
        skipped: list[SkippedShard] = []
        retained: list[dict[str, npt.NDArray[Any]]] | None = [] if retain_arrays else None

        for path in self._shard_paths():
            try:
                arrays = _load_shard_arrays(path)
                shard_schema, count, fields = _validated_arrays(arrays, expected)
            except _CORRUPTION_ERRORS as exc:
                skipped.append(SkippedShard(path=path, reason=str(exc)))
                warnings.warn(
                    f"skipping unreadable shard {path.name}: {exc}",
                    ShardWarning,
                    stacklevel=2,
                )
                continue
            if expected is None:
                expected = shard_schema
            shards.append(ShardInfo(path=path, count=count))
            if retained is not None:
                retained.append(fields)

        self._retained = None if retained is None else tuple(retained)
        self._schema = expected
        self._shards = tuple(shards)
        self._skipped = tuple(skipped)
        # Prefix sums over shard counts, so a flat sample index maps to a
        # (shard, offset) pair by bisection. Length is len(shards) + 1.
        offsets = [0]
        for info in self._shards:
            offsets.append(offsets[-1] + info.count)
        self._offsets = tuple(offsets)
        self._cache_path: Path | None = None
        self._cache_arrays: dict[str, npt.NDArray[Any]] = {}

    def _shard_paths(self) -> list[Path]:
        """Return the candidate shard files, in index order.

        Sorted by parsed index rather than by name so that a directory which
        has run past six digits (``shard-1000000.npz``) still reads in
        chronological order.
        """
        paths: list[tuple[int, Path]] = []
        for path in self._directory.glob(SHARD_GLOB):
            match = _SHARD_RE.fullmatch(path.name)
            if match is None:
                continue
            paths.append((int(match.group(1)), path))
        return [path for _, path in sorted(paths)]

    @property
    def directory(self) -> Path:
        """Return the directory being read."""
        return self._directory

    @property
    def schema(self) -> Schema | None:
        """Return the schema in force, or ``None`` if no shard was readable."""
        return self._schema

    @property
    def shards(self) -> tuple[ShardInfo, ...]:
        """Return the shards that passed validation, oldest first."""
        return self._shards

    @property
    def skipped(self) -> tuple[SkippedShard, ...]:
        """Return the shards that failed validation, with their reasons.

        Non-empty is a signal, not an error: one entry after a kill is the
        expected shape of a preemption. Several, or one on a directory nothing
        crashed on, means bit rot.
        """
        return self._skipped

    def __len__(self) -> int:
        """Return the total number of readable samples."""
        return self._offsets[-1]

    @property
    def retains_arrays(self) -> bool:
        """Return whether decoded shards are held in memory for the reader's life."""
        return self._retained is not None

    def _arrays_for(self, slot: int) -> dict[str, npt.NDArray[Any]]:
        """Return shard `slot`'s field arrays: retained, cached, or re-read.

        In retained mode this is a dictionary lookup and nothing else -- the
        arrays are the ones construction already validated. Otherwise the
        shard is re-opened and re-validated unless it is the cached one.
        """
        if self._retained is not None:
            return self._retained[slot]
        path = self._shards[slot].path
        if self._cache_path == path:
            return self._cache_arrays
        arrays = _load_shard_arrays(path)
        _, _, fields = _validated_arrays(arrays, self._schema)
        self._cache_path = path
        self._cache_arrays = fields
        return fields

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return sample `index`, counting across shards in order.

        Negative indices count from the end. In retained mode the row is a
        slice of memory: no I/O, no decompression, no digest, whatever order
        the sampler draws in. Without retention the one-shard cache makes
        sequential access cost one decompression per shard and shuffled access
        cost one per call in the worst case.
        """
        total = len(self)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"sample index {index} out of range for {total} samples")
        # Rightmost offset <= index; offsets is short (one entry per shard).
        slot = 0
        for position in range(len(self._shards)):
            if self._offsets[position] <= index < self._offsets[position + 1]:
                slot = position
                break
        arrays = self._arrays_for(slot)
        row = index - self._offsets[slot]
        return {
            name: torch.from_numpy(np.array(array[row], copy=True))
            for name, array in arrays.items()
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate every readable sample, shard by shard, in write order."""
        for slot, info in enumerate(self._shards):
            arrays = self._arrays_for(slot)
            for row in range(info.count):
                yield {
                    name: torch.from_numpy(np.array(array[row], copy=True))
                    for name, array in arrays.items()
                }


def stack_samples(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a sequence of samples into a batch of ``(n, *field.shape)`` tensors.

    Raises ``ValueError`` on an empty sequence or on samples whose key sets
    disagree -- a batch assembled from mismatched samples would train on
    whichever keys happened to be common.
    """
    if not samples:
        raise ValueError("cannot stack an empty sequence of samples")
    names = set(samples[0])
    for position, sample in enumerate(samples[1:], start=1):
        if set(sample) != names:
            raise ValueError(
                f"sample {position} has keys {sorted(sample)}, expected {sorted(names)}"
            )
    return {
        name: torch.stack([sample[name] for sample in samples]) for name in sorted(names)
    }
