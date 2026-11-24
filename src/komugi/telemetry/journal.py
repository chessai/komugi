"""Append-only JSONL journal for telemetry records.

The v1 campaign lost its metrics twice, and both times the mechanism was the
same: telemetry lived in a mutable log that something else was free to
rewrite. The journal here is append-only by construction and durable per
record --

* every record is one line, opened ``O_APPEND`` and ``fsync``-ed before the
  call returns, so a record that has been written is on disk even if the
  campaign dies in the next instruction;
* ``O_APPEND`` puts the kernel in charge of the offset, so concurrent writers
  (a trainer and several self-play workers sharing one journal) cannot land
  on top of each other's bytes;
* nothing in this module opens a journal for writing in any mode that can
  shorten it. Truncation is reachable only through
  :mod:`komugi.telemetry.archive`, which archives first.

What is *not* claimed is that a line is all-or-nothing. A crash (or a full
disk) between the first byte and the newline leaves a partial trailing line,
and no fsync discipline prevents that. So the reader tolerates it:
:func:`read_records` skips any line it cannot decode and reports how many it
skipped, rather than raising and taking the whole run's history with it. A
non-zero ``skipped`` on a journal that was not just crash-truncated is a
signal worth surfacing, which is why it is a returned number and not a
warnings-module side effect.

Wire format -- one JSON object per line, keys sorted, ``NaN``/``Infinity``
rejected by the encoder::

    {"draw_rate":0.31,...,"kind":"iteration_metrics","v":1}

The ``kind`` tag is what lets one journal hold both record types
interleaved in real time order; :func:`read_records` selects by type and
quietly passes over the other kinds (they are not corruption).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Generic, TypeVar

from komugi.telemetry.records import IterationMetrics, NetFingerprint

Record = IterationMetrics | NetFingerprint
"""Any record the journal knows how to store."""

R = TypeVar("R", IterationMetrics, NetFingerprint)

SCHEMA_VERSION: Final[int] = 1
"""Version stamped on every line, so a future format change is detectable."""

ITERATION_METRICS_KIND: Final[str] = "iteration_metrics"
NET_FINGERPRINT_KIND: Final[str] = "net_fingerprint"

_JOURNAL_MODE: Final[int] = 0o644


def kind_of(record: Record) -> str:
    """Return the wire tag `record` is stored under."""
    if isinstance(record, IterationMetrics):
        return ITERATION_METRICS_KIND
    return NET_FINGERPRINT_KIND


class JournalDecodeError(ValueError):
    """A journal line could not be turned back into a record.

    Raised by :func:`decode_line` and swallowed-and-counted by
    :func:`read_records`; a ``ValueError`` subclass because a malformed line
    is bad data, not a bug in the caller.
    """


@dataclass(frozen=True)
class JournalRead(Generic[R]):
    """The result of reading one kind of record out of a journal.

    Deliberately not a bare tuple: the count of unreadable lines has to come
    back with the records, because "17 iterations" and "17 iterations and 4
    lines we could not parse" are very different claims and the second one
    must not be reachable by ignoring a return value.
    """

    records: tuple[R, ...]
    skipped: int

    def __len__(self) -> int:
        """Return the number of records decoded."""
        return len(self.records)

    @property
    def ok(self) -> bool:
        """Return whether every line of the requested kind decoded cleanly."""
        return self.skipped == 0


def encode_record(record: Record) -> str:
    """Return the single-line JSON encoding of `record` (no trailing newline).

    ``allow_nan=False`` is load-bearing: it makes a non-finite float a
    ``ValueError`` here rather than the non-standard ``NaN`` token in a file
    that other tools then fail to parse. The records themselves already
    reject non-finite values, so this is the second of two locks.
    """
    if isinstance(record, IterationMetrics):
        fields: dict[str, object] = {
            "iteration": record.iteration,
            "games": record.games,
            "draw_rate": record.draw_rate,
            "avg_game_len": record.avg_game_len,
            "policy_entropy": record.policy_entropy,
            "extras": dict(record.extras),
        }
    else:
        fields = {
            "iteration": record.iteration,
            "role": record.role,
            "sha256": record.sha256,
            "param_count": record.param_count,
        }
    payload: dict[str, object] = {
        "kind": kind_of(record),
        "v": SCHEMA_VERSION,
        **fields,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _field(payload: Mapping[str, object], key: str) -> object:
    """Return ``payload[key]``, as a :class:`JournalDecodeError` if absent."""
    if key not in payload:
        raise JournalDecodeError(f"journal line is missing field {key!r}")
    return payload[key]


def _as_int(payload: Mapping[str, object], key: str) -> int:
    """Return an int field, rejecting bools and JSON floats."""
    value = _field(payload, key)
    # bool is an int subclass; `true` where a count belongs is corruption.
    if isinstance(value, bool) or not isinstance(value, int):
        raise JournalDecodeError(f"field {key!r} must be an integer, got {value!r}")
    return value


def _as_float(payload: Mapping[str, object], key: str) -> float:
    """Return a float field, accepting a JSON integer, rejecting bools."""
    value = _field(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JournalDecodeError(f"field {key!r} must be a number, got {value!r}")
    return float(value)


def _as_str(payload: Mapping[str, object], key: str) -> str:
    """Return a string field."""
    value = _field(payload, key)
    if not isinstance(value, str):
        raise JournalDecodeError(f"field {key!r} must be a string, got {value!r}")
    return value


def _as_extras(payload: Mapping[str, object], key: str) -> dict[str, float]:
    """Return the ``extras`` mapping, checking every key and value."""
    value = _field(payload, key)
    if not isinstance(value, dict):
        raise JournalDecodeError(f"field {key!r} must be an object, got {value!r}")
    extras: dict[str, float] = {}
    for name, entry in value.items():
        if not isinstance(name, str):
            raise JournalDecodeError(f"extras keys must be strings, got {name!r}")
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise JournalDecodeError(
                f"extras[{name!r}] must be a number, got {entry!r}"
            )
        extras[name] = float(entry)
    return extras


def _decode_iteration_metrics(payload: Mapping[str, object]) -> IterationMetrics:
    """Rebuild an :class:`IterationMetrics` from a decoded JSON object."""
    return IterationMetrics(
        iteration=_as_int(payload, "iteration"),
        games=_as_int(payload, "games"),
        draw_rate=_as_float(payload, "draw_rate"),
        avg_game_len=_as_float(payload, "avg_game_len"),
        policy_entropy=_as_float(payload, "policy_entropy"),
        extras=_as_extras(payload, "extras"),
    )


def _decode_net_fingerprint(payload: Mapping[str, object]) -> NetFingerprint:
    """Rebuild a :class:`NetFingerprint` from a decoded JSON object."""
    return NetFingerprint(
        iteration=_as_int(payload, "iteration"),
        role=_as_str(payload, "role"),
        sha256=_as_str(payload, "sha256"),
        param_count=_as_int(payload, "param_count"),
    )


_DECODERS: Final[Mapping[str, Callable[[Mapping[str, object]], Record]]] = {
    ITERATION_METRICS_KIND: _decode_iteration_metrics,
    NET_FINGERPRINT_KIND: _decode_net_fingerprint,
}


def decode_line(line: str) -> Record:
    """Decode one journal line into a record.

    Raises :class:`JournalDecodeError` for anything unusable: truncated JSON,
    an unknown ``kind``, a future schema version, a field of the wrong type,
    or a value the record's own validation rejects.
    """
    try:
        parsed: object = json.loads(line)
    except ValueError as exc:  # json.JSONDecodeError
        raise JournalDecodeError(f"line is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JournalDecodeError(f"line is not a JSON object: {parsed!r}")

    version = _as_int(parsed, "v")
    if version != SCHEMA_VERSION:
        raise JournalDecodeError(
            f"unsupported journal schema version {version}, expected {SCHEMA_VERSION}"
        )
    kind = _as_str(parsed, "kind")
    decoder = _DECODERS.get(kind)
    if decoder is None:
        raise JournalDecodeError(f"unknown record kind {kind!r}")
    try:
        return decoder(parsed)
    except ValueError as exc:
        # Covers the record's own __post_init__ as well as the field helpers:
        # a draw rate of 1.7 on disk is as unusable as a truncated line.
        raise JournalDecodeError(f"invalid {kind} record: {exc}") from exc


def write_record(path: str | os.PathLike[str], record: Record) -> None:
    """Append `record` to the journal at `path`, durably.

    Creates the file (but not its parent directory) if it does not exist.
    Returns only once the bytes are on the platter: the cost is one fsync per
    record, which is the correct trade at campaign-iteration frequency and
    the reason a lost run can still be reconstructed after a hard kill.
    """
    payload = (encode_record(record) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _JOURNAL_MODE)
    try:
        written = 0
        while written < len(payload):
            # os.write may be short; under O_APPEND the remainder still lands
            # at the (new) end of file, so the line stays contiguous.
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def read_records(
    path: str | os.PathLike[str],
    kind: type[R],
    *,
    missing_ok: bool = False,
) -> JournalRead[R]:
    """Read every record of type `kind` from the journal at `path`.

    Lines of another known kind are passed over silently -- one journal
    holding both metrics and fingerprints is the intended usage. Lines that
    cannot be decoded at all are skipped and counted in
    :attr:`JournalRead.skipped`; this call does not raise on bad data, which
    is what keeps a partially written trailing line from costing a whole
    campaign's history.

    With ``missing_ok=True`` a journal that does not exist yet reads as empty,
    for the startup case where a gate runs before the first record is written.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        if missing_ok:
            return JournalRead(records=(), skipped=0)
        raise

    records: list[R] = []
    skipped = 0
    for line in text.split("\n"):
        if not line.strip():
            # Trailing empty string after the final newline, and any blank
            # padding: absence of a record, not a damaged one.
            continue
        try:
            record = decode_line(line)
        except JournalDecodeError:
            skipped += 1
            continue
        if isinstance(record, kind):
            records.append(record)
    return JournalRead(records=tuple(records), skipped=skipped)
