"""The journal keeps what it was given, and survives what it was interrupted by.

Three properties, in order of how much a campaign would miss them:

* round-trip -- what goes in comes back out, exactly, including the mixed-kind
  case where one file holds metrics and fingerprints interleaved;
* crash tolerance -- a partially written trailing line costs that one record
  and nothing else. This is the negative test: the reader must not raise, and
  must say how many lines it could not read;
* append-only durability -- writing never rewrites earlier bytes, and each
  record is on disk before the call returns.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from komugi.telemetry.journal import (
    SCHEMA_VERSION,
    JournalDecodeError,
    decode_line,
    encode_record,
    read_records,
    write_record,
)
from komugi.telemetry.records import (
    EMA,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)

_DIGEST_A: str = "a" * 64
_DIGEST_B: str = "b" * 64


def _metrics(iteration: int) -> IterationMetrics:
    """Return a plausible metrics record for `iteration`."""
    return IterationMetrics(
        iteration=iteration,
        games=4096,
        draw_rate=0.31 + 0.01 * iteration,
        avg_game_len=118.5,
        policy_entropy=1.87,
        extras={"value_loss": 0.42, "lr": 0.01},
    )


def _fingerprint(iteration: int, role: str, digest: str) -> NetFingerprint:
    """Return a fingerprint record."""
    return NetFingerprint(
        iteration=iteration, role=role, sha256=digest, param_count=12_345_678
    )


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    """Records read back equal to the ones written, in write order."""
    path = tmp_path / "journal.jsonl"
    written = [_metrics(i) for i in range(5)]
    for record in written:
        write_record(path, record)

    read = read_records(path, IterationMetrics)

    assert read.records == tuple(written)
    assert read.skipped == 0
    assert read.ok
    assert len(read) == 5


def test_extras_survive_the_round_trip(tmp_path: Path) -> None:
    """The open-ended extras mapping is not flattened or dropped."""
    path = tmp_path / "journal.jsonl"
    record = IterationMetrics(
        iteration=7,
        games=1,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras={"resign_rate": 0.05, "batch_ms": 12.5},
    )
    write_record(path, record)

    (restored,) = read_records(path, IterationMetrics).records

    assert dict(restored.extras) == {"resign_rate": 0.05, "batch_ms": 12.5}


def test_one_journal_holds_both_kinds(tmp_path: Path) -> None:
    """Interleaved kinds are selected by type, and the other kind is not a warning."""
    path = tmp_path / "journal.jsonl"
    write_record(path, _metrics(0))
    write_record(path, _fingerprint(0, SELF_PLAY, _DIGEST_A))
    write_record(path, _fingerprint(0, TRAIN, _DIGEST_B))
    write_record(path, _metrics(1))

    metrics = read_records(path, IterationMetrics)
    fingerprints = read_records(path, NetFingerprint)

    assert [m.iteration for m in metrics.records] == [0, 1]
    assert [f.role for f in fingerprints.records] == [SELF_PLAY, TRAIN]
    assert metrics.skipped == 0
    assert fingerprints.skipped == 0


def test_corrupted_trailing_line_is_tolerated(tmp_path: Path) -> None:
    """A half-written final record costs itself and nothing else.

    The negative test for the v1 failure mode: a campaign killed mid-write
    must still be able to read back every complete iteration it recorded.
    """
    path = tmp_path / "journal.jsonl"
    for i in range(3):
        write_record(path, _metrics(i))

    # Simulate a crash between the first byte of a record and its newline.
    partial = encode_record(_metrics(3))[:37]
    assert "\n" not in partial
    with path.open("a", encoding="utf-8") as handle:
        handle.write(partial)

    read = read_records(path, IterationMetrics)

    assert [m.iteration for m in read.records] == [0, 1, 2]
    assert read.skipped == 1
    assert not read.ok


def test_the_partial_line_really_is_unparseable() -> None:
    """Guard the guard: the simulated tail must actually be broken JSON."""
    partial = encode_record(_metrics(3))[:37]
    with pytest.raises(JournalDecodeError):
        decode_line(partial)


def test_corruption_in_the_middle_is_skipped_not_fatal(tmp_path: Path) -> None:
    """Damage anywhere in the file costs only the damaged lines."""
    path = tmp_path / "journal.jsonl"
    lines = [
        encode_record(_metrics(0)),
        '{"kind":"iteration_metrics","v":1,"iteration":1,',  # truncated object
        "",  # blank padding, not a record
        "not json at all",
        encode_record(_metrics(2)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    read = read_records(path, IterationMetrics)

    assert [m.iteration for m in read.records] == [0, 2]
    assert read.skipped == 2  # the blank line is absence, not corruption


def test_a_record_with_an_impossible_value_is_rejected_on_read(tmp_path: Path) -> None:
    """On-disk values still have to satisfy the record's own invariants."""
    path = tmp_path / "journal.jsonl"
    path.write_text(
        '{"avg_game_len":1.0,"draw_rate":1.7,"extras":{},"games":1,'
        '"iteration":0,"kind":"iteration_metrics","policy_entropy":0.5,"v":1}\n',
        encoding="utf-8",
    )

    read = read_records(path, IterationMetrics)

    assert read.records == ()
    assert read.skipped == 1


def test_unknown_kind_and_future_version_are_skipped(tmp_path: Path) -> None:
    """Forward-compatibility is a skip, never a crash."""
    path = tmp_path / "journal.jsonl"
    path.write_text(
        '{"kind":"gpu_temperature","v":1,"celsius":71}\n'
        f'{{"kind":"iteration_metrics","v":{SCHEMA_VERSION + 1},"iteration":0}}\n'
        + encode_record(_metrics(9))
        + "\n",
        encoding="utf-8",
    )

    read = read_records(path, IterationMetrics)

    assert [m.iteration for m in read.records] == [9]
    assert read.skipped == 2


def test_writes_only_append(tmp_path: Path) -> None:
    """Earlier bytes are never rewritten: the file grows by exactly one line."""
    path = tmp_path / "journal.jsonl"
    write_record(path, _metrics(0))
    prefix = path.read_bytes()

    write_record(path, _fingerprint(1, EMA, _DIGEST_B))
    after = path.read_bytes()

    assert after.startswith(prefix)
    assert after[len(prefix) :] == (
        encode_record(_fingerprint(1, EMA, _DIGEST_B)) + "\n"
    ).encode("utf-8")


def test_each_record_is_one_newline_terminated_line(tmp_path: Path) -> None:
    """The format's one invariant: line count equals record count."""
    path = tmp_path / "journal.jsonl"
    for i in range(4):
        write_record(path, _metrics(i))

    raw = path.read_bytes()

    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4


def test_a_record_is_durable_as_soon_as_the_call_returns(tmp_path: Path) -> None:
    """A separate reader sees the record without any flush from the writer."""
    path = tmp_path / "journal.jsonl"
    write_record(path, _metrics(0))

    fd = os.open(path, os.O_RDONLY)
    try:
        contents = os.read(fd, 65536)
    finally:
        os.close(fd)

    assert contents.decode("utf-8").strip() == encode_record(_metrics(0))


def test_missing_journal_raises_unless_allowed(tmp_path: Path) -> None:
    """A typo'd path is an error; a not-yet-created journal can read as empty."""
    path = tmp_path / "never-written.jsonl"

    with pytest.raises(FileNotFoundError):
        read_records(path, IterationMetrics)

    read = read_records(path, IterationMetrics, missing_ok=True)
    assert read.records == ()
    assert read.ok


def test_encoder_refuses_non_finite_extras() -> None:
    """NaN cannot reach the file, because it cannot reach the record."""
    with pytest.raises(ValueError):
        IterationMetrics(
            iteration=0,
            games=1,
            draw_rate=0.5,
            avg_game_len=1.0,
            policy_entropy=1.0,
            extras={"value_loss": float("nan")},
        )
