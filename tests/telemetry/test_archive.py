"""Archiving is exact, and refusing to overwrite is the point.

Two properties carry the whole module. The archive must be *byte-identical*,
because an archive that is merely approximately the log is not evidence; and
the destination must never be silently reused, because the failure this
guards against -- a relaunch quietly replacing the previous run's history --
is exactly what a silent overwrite would reintroduce one level up.

The rest checks the naming: the suffix comes from the log's own mtime, not
from the wall clock, so archiving is deterministic, orderable and honest
about *when the data was written* rather than when someone got around to
rescuing it.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from komugi.telemetry.archive import (
    TruncationToken,
    archive_before_truncate,
    archive_name,
    guard_truncation,
)

_LOG_BYTES: bytes = (
    b'{"kind":"iteration_metrics","v":1,"iteration":0}\n'
    b"\x00\x01\x02 raw bytes and a partial trailing line, kept verbatim"
)


def _write_log(tmp_path: Path, *, mtime_ns: int = 1_700_000_000_123_456_789) -> Path:
    """Write a log with deterministic contents and a pinned mtime."""
    path = tmp_path / "campaign.log"
    path.write_bytes(_LOG_BYTES)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_archive_is_byte_identical(tmp_path: Path) -> None:
    """The copy is the log, to the byte -- including a broken trailing line."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "archive"

    archived = archive_before_truncate(log, archive_dir)

    assert archived.read_bytes() == _LOG_BYTES
    assert archived.read_bytes() == log.read_bytes()


def test_archive_is_byte_identical_for_a_large_log(tmp_path: Path) -> None:
    """Bigger than the copy buffer, so the chunked path is the one under test."""
    log = tmp_path / "big.log"
    payload = bytes(range(256)) * 20_000  # ~5 MiB, past the 1 MiB buffer
    log.write_bytes(payload)

    archived = archive_before_truncate(log, tmp_path / "archive")

    assert archived.read_bytes() == payload


def test_archive_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    """The negative test: a second archive of the same log raises, never clobbers."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "archive"
    archived = archive_before_truncate(log, archive_dir)

    with pytest.raises(FileExistsError):
        archive_before_truncate(log, archive_dir)

    assert archived.read_bytes() == _LOG_BYTES


def test_archive_refuses_a_destination_it_did_not_write(tmp_path: Path) -> None:
    """Someone else's file under our name is still not ours to overwrite."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    squatter = archive_dir / archive_name(log, log.stat().st_mtime_ns)
    squatter.write_bytes(b"precious")

    with pytest.raises(FileExistsError):
        archive_before_truncate(log, archive_dir)

    assert squatter.read_bytes() == b"precious"


def test_suffix_comes_from_the_logs_own_mtime(tmp_path: Path) -> None:
    """Not the wall clock: the name identifies the data, not the rescue."""
    log = _write_log(tmp_path, mtime_ns=1_699_999_999_000_000_001)
    on_disk_ns = log.stat().st_mtime_ns  # some filesystems coarsen the mtime

    archived = archive_before_truncate(log, tmp_path / "archive")

    assert archived.name == f"campaign.log.{on_disk_ns:020d}"
    # The falsifier for a time.time() implementation: the suffix is the log's
    # 2023 mtime, and any wall-clock suffix would be far past this bound.
    assert int(archived.name.rsplit(".", 1)[1]) < 1_700_000_000_000_000_000


def test_archive_names_sort_chronologically() -> None:
    """Fixed-width digits mean lexicographic order replays a campaign in order."""
    names = [
        archive_name("campaign.log", ns)
        for ns in (999_999_999, 1_699_999_999_000_000_000, 1_700_000_000_000_000_000)
    ]

    assert names == sorted(names)


def test_archive_preserves_mtime_so_the_name_is_stable(tmp_path: Path) -> None:
    """An archive re-derives its own name: re-archiving it is a refusal, not a copy."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "archive"

    archived = archive_before_truncate(log, archive_dir)

    assert archived.stat().st_mtime_ns == log.stat().st_mtime_ns
    assert archive_name(archived, archived.stat().st_mtime_ns) == archived.name


def test_a_second_run_archives_under_a_new_name(tmp_path: Path) -> None:
    """Different content, written later, archives alongside -- nothing is lost."""
    log = _write_log(tmp_path, mtime_ns=1_700_000_000_000_000_000)
    archive_dir = tmp_path / "archive"
    first = guard_truncation(log, archive_dir)
    first.truncate()

    log.write_bytes(b"second run\n")
    os.utime(log, ns=(1_700_000_060_000_000_000, 1_700_000_060_000_000_000))
    second = guard_truncation(log, archive_dir)

    assert first.archive_path != second.archive_path
    assert first.archive_path.read_bytes() == _LOG_BYTES
    assert second.archive_path.read_bytes() == b"second run\n"
    assert sorted(p.name for p in archive_dir.iterdir()) == [
        first.archive_path.name,
        second.archive_path.name,
    ]


def test_guard_truncation_archives_before_it_permits(tmp_path: Path) -> None:
    """The token exists only after a copy does, and truncation empties the log."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "archive"

    token = guard_truncation(log, archive_dir)

    assert isinstance(token, TruncationToken)
    assert token.archive_path.exists()
    assert token.archive_path.read_bytes() == _LOG_BYTES
    assert token.authorizes(log)
    assert not token.authorizes(tmp_path / "other.log")

    token.truncate()

    assert log.read_bytes() == b""
    assert token.archive_path.read_bytes() == _LOG_BYTES


def test_truncate_is_idempotent(tmp_path: Path) -> None:
    """Emptying an empty log is a no-op, and never touches the archive."""
    log = _write_log(tmp_path)
    token = guard_truncation(log, tmp_path / "archive")

    token.truncate()
    token.truncate()

    assert log.read_bytes() == b""
    assert token.archive_path.read_bytes() == _LOG_BYTES


def test_no_permission_without_a_log_to_archive(tmp_path: Path) -> None:
    """A run that cannot archive does not get to truncate."""
    with pytest.raises(FileNotFoundError):
        guard_truncation(tmp_path / "absent.log", tmp_path / "archive")


def test_archive_dir_is_created_on_demand(tmp_path: Path) -> None:
    """A first relaunch must not fail because nobody made the directory."""
    log = _write_log(tmp_path)
    archive_dir = tmp_path / "deeply" / "nested" / "archive"

    archived = archive_before_truncate(log, archive_dir)

    assert archived.parent == archive_dir
    assert archived.read_bytes() == _LOG_BYTES


def test_a_directory_is_not_a_log(tmp_path: Path) -> None:
    """Pointing the guard at a directory raises rather than doing something odd."""
    target = tmp_path / "not-a-log"
    target.mkdir()

    with pytest.raises(IsADirectoryError):
        archive_before_truncate(target, tmp_path / "archive")


def test_archive_name_rejects_a_negative_mtime() -> None:
    """A pre-epoch mtime would break the fixed-width ordering guarantee."""
    with pytest.raises(ValueError):
        archive_name("campaign.log", -1)


def test_token_is_frozen(tmp_path: Path) -> None:
    """The permission cannot be re-pointed at a log that was never archived."""
    log = _write_log(tmp_path)
    token = guard_truncation(log, tmp_path / "archive")

    # setattr, not direct assignment: the runtime guard is what is under test.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(token, "log_path", tmp_path / "elsewhere.log")
