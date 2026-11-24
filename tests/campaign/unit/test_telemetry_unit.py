"""Archive-on-relaunch, at the campaign's own seam (KOM-025).

:mod:`komugi.telemetry.archive` already owns the primitive -- naming by mtime,
``O_EXCL`` refusal, byte-for-byte copy -- and ``tests/telemetry/test_archive.py``
pins all of it. What is pinned *here* is the campaign's use of it: which files
a relaunch saves, where it puts them, and the two decisions that make the
difference between a history kept and v1's history destroyed twice.

Both decisions are properties a mutant can break silently, which is why each
has its own node:

* **the log is copied, never truncated.** A relaunch that archived and *then*
  spent its :class:`~komugi.telemetry.archive.TruncationToken` would look
  entirely correct -- there would be an archive, with the right bytes in it --
  and would still be a catastrophe, because a resumed campaign reads its own
  loss baseline, its pre-registered constants and its whole fingerprint series
  back out of ``journal.jsonl``. Emptying it does not lose an audit trail, it
  disarms every gate that counts across a restart.
* **an existing archive is never overwritten.** The name is derived from the
  log's own mtime, so a collision means these exact bytes are already saved;
  the second call keeps the first call's copy and reports nothing new.

Nothing here spawns a process, plays a game or imports torch: the whole seam
is two paths and a directory.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from komugi.campaign import telemetry as campaign_telemetry
from komugi.campaign.telemetry import (
    ARCHIVE_RETAINED,
    TELEMETRY_ARCHIVE_DIRNAME,
    archive_telemetry,
    selfplay_journal_path,
    telemetry_archive_dir,
)
from komugi.telemetry.archive import archive_name

FIRST_LAUNCH: bytes = b'{"kind":"iteration_metrics","v":1}\n'
"""What launch one is imagined to have left behind. Content is never parsed
here -- the claim is about *bytes*, so opaque bytes make the claim harder to
pass by accident."""

SECOND_LAUNCH: bytes = b'{"kind":"net_fingerprint","v":1}\n'


def _write(path: Path, payload: bytes) -> Path:
    """Write `payload` to `path`, creating its parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _log(root: Path, payload: bytes = FIRST_LAUNCH) -> Path:
    """A campaign log under `root`, holding `payload`."""
    return _write(root / "journal.jsonl", payload)


def test_the_archive_directory_is_under_the_campaign_root(tmp_path: Path) -> None:
    """A history kept beside the root is a history that does not travel.

    The root is the unit an operator copies off a dead box, so the archive is
    a directory *inside* it. A premise check with teeth: every assertion below
    looks in this directory, so an archive written to the CWD or to a temp
    directory would make them vacuous rather than failing.
    """
    assert telemetry_archive_dir(tmp_path) == tmp_path / TELEMETRY_ARCHIVE_DIRNAME
    assert telemetry_archive_dir(str(tmp_path)) == tmp_path / TELEMETRY_ARCHIVE_DIRNAME


def test_an_existing_log_is_copied_byte_for_byte(tmp_path: Path) -> None:
    """The copy is the whole point: the previous launch's bytes, unaltered."""
    log = _log(tmp_path)

    archived = archive_telemetry(tmp_path, (log,))

    assert len(archived) == 1, f"want one archived copy, got {archived}"
    copy = archived[0]
    assert copy.parent == telemetry_archive_dir(tmp_path), (
        f"the copy landed at {copy}, outside {telemetry_archive_dir(tmp_path)}"
    )
    assert copy.read_bytes() == FIRST_LAUNCH, (
        "the archived copy does not hold the log's bytes"
    )
    assert copy.name == archive_name(log, log.stat().st_mtime_ns), (
        f"the copy is named {copy.name!r}; the archive names by the log's own "
        f"mtime so that lexicographic order is chronological order"
    )


def test_the_log_itself_is_left_in_place_and_not_truncated(tmp_path: Path) -> None:
    """The regression that matters: archiving must not empty what it saved.

    v1 lost its telemetry twice to launch truncation, and the obvious fix --
    archive, then truncate -- is *still* wrong here, invisibly so: a resumed
    campaign recovers its loss baseline, its pre-registered constants and its
    fingerprint series from this file, so an emptied journal silently disarms
    every gate that counts across a restart while leaving a perfectly good
    archive behind to prove the archiving worked.

    Red against a mutant that spends the truncation token.
    """
    log = _log(tmp_path)

    archive_telemetry(tmp_path, (log,))

    assert log.is_file(), "the log was removed rather than copied"
    assert log.read_bytes() == FIRST_LAUNCH, (
        f"the log holds {log.read_bytes()!r} after being archived; a relaunch "
        f"must not be able to empty the history a resume reads back"
    )


def test_a_missing_log_is_skipped_rather_than_raising(tmp_path: Path) -> None:
    """A fresh root has no history to save, and that is not an error."""
    absent = tmp_path / "journal.jsonl"

    archived = archive_telemetry(tmp_path, (absent,))

    assert archived == (), f"a fresh root archived {archived}"
    assert not telemetry_archive_dir(tmp_path).exists(), (
        "an archive directory was created for a root with nothing in it"
    )


def test_a_directory_where_a_log_belongs_is_skipped(tmp_path: Path) -> None:
    """``is_file`` and not ``exists``: a directory is not a log to copy."""
    (tmp_path / "journal.jsonl").mkdir(parents=True)

    assert archive_telemetry(tmp_path, (tmp_path / "journal.jsonl",)) == ()


def test_both_of_a_roots_logs_are_archived_in_order(tmp_path: Path) -> None:
    """The journal and its self-play sibling, both, in the order named.

    The sibling file exists *because* the root journal cannot hold a third
    ``kind``; a relaunch guard that saved only one of them would leave the
    other exactly as exposed as v1's was.
    """
    journal = _log(tmp_path)
    selfplay = _write(selfplay_journal_path(tmp_path), b'{"kind":"selfplay","v":1}\n')

    archived = archive_telemetry(tmp_path, (journal, selfplay))

    assert len(archived) == 2, f"want both logs archived, got {archived}"
    assert [copy.read_bytes() for copy in archived] == [
        journal.read_bytes(),
        selfplay.read_bytes(),
    ], "the archived copies are not the two logs, in the order they were named"


def test_re_archiving_an_untouched_log_keeps_the_first_copy(tmp_path: Path) -> None:
    """Two starts against a root nothing appended to: one copy, not two.

    The destination name is derived from the log's own mtime, so a collision
    means *these exact bytes at this exact mtime* are already archived. The
    second call therefore reports nothing new -- and, crucially, does not
    overwrite: :func:`~komugi.telemetry.archive.archive_before_truncate` opens
    ``O_EXCL``, so the refusal is the kernel's rather than a check-then-write
    race, and the copy that survives is the one that was already there.
    """
    log = _log(tmp_path)
    first = archive_telemetry(tmp_path, (log,))
    assert len(first) == 1

    second = archive_telemetry(tmp_path, (log,))

    assert second == (), (
        f"the second start archived {second} for a log nothing had appended to"
    )
    copies = sorted(telemetry_archive_dir(tmp_path).iterdir())
    assert copies == [first[0]], f"want exactly the first copy, found {copies}"
    assert first[0].read_bytes() == FIRST_LAUNCH


def test_a_log_that_grew_is_archived_again_under_a_new_name(tmp_path: Path) -> None:
    """Three launches, three histories: each launch's bytes get their own copy.

    The mtime names the *data*, not the moment of the rescue, so a log that
    was appended to between two starts archives under a different name and
    both copies survive. That is what makes ``sorted(archive_dir.iterdir())``
    a replay of a root's launches, oldest first.
    """
    log = _log(tmp_path)
    stat = log.stat()
    first = archive_telemetry(tmp_path, (log,))[0]

    log.write_bytes(FIRST_LAUNCH + SECOND_LAUNCH)
    # An explicit, later mtime rather than a sleep: the copy's name is a
    # function of this number, so setting it is what makes the test's claim
    # ("a different launch, a different name") exact on a coarse-grained
    # filesystem clock.
    os.utime(log, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    second = archive_telemetry(tmp_path, (log,))[0]

    assert second != first, "a grown log re-archived under the same name"
    assert first.read_bytes() == FIRST_LAUNCH, (
        "the earlier launch's copy was overwritten by the later one"
    )
    assert second.read_bytes() == FIRST_LAUNCH + SECOND_LAUNCH
    assert sorted(telemetry_archive_dir(tmp_path).iterdir()) == [first, second], (
        "archive names must sort chronologically; they are fixed-width mtimes"
    )


# ---------------------------------------------------------------------------
# KOM-025 wire-in review F7/F9 -- the archive is bounded, and its failures loud
# ---------------------------------------------------------------------------


def _relaunch(log: Path, payload: bytes, mtime_ns: int) -> None:
    """Leave `log` holding `payload` as of `mtime_ns`, as a launch would have.

    The mtime is set explicitly rather than slept for: the copy's name is a
    function of exactly this number, so setting it is what makes "a different
    launch" exact on a filesystem whose clock is coarse.
    """
    stat = log.stat()
    log.write_bytes(payload)
    os.utime(log, ns=(stat.st_atime_ns, mtime_ns))


def test_the_archive_is_bounded_and_the_newest_copies_are_the_survivors(
    tmp_path: Path,
) -> None:
    """F7: a restart loop must not fill the disk the live journal is on.

    An INTERRUPT marker is cleared and the start continues, so a supervisor
    that keeps restarting a campaign archives the whole journal once per
    restart -- and the journal only grows. Unbounded, the copies of a root's
    own history are what it runs out of space for, and the file it then cannot
    append to is the one every gate reads back across a restart.

    Both logs are grown every launch, because the bound is **per log**: a big
    journal must not be able to evict the self-play sibling's history.
    """
    journal = _log(tmp_path, FIRST_LAUNCH)
    selfplay = _write(selfplay_journal_path(tmp_path), SECOND_LAUNCH)
    base_ns = journal.stat().st_mtime_ns
    launches = ARCHIVE_RETAINED + 3
    journal_payloads: list[bytes] = []
    selfplay_payloads: list[bytes] = []

    for launch in range(launches):
        journal_payloads.append(FIRST_LAUNCH * (launch + 1))
        selfplay_payloads.append(SECOND_LAUNCH * (launch + 1))
        stamp = base_ns + (launch + 1) * 1_000_000_000
        _relaunch(journal, journal_payloads[-1], stamp)
        _relaunch(selfplay, selfplay_payloads[-1], stamp)
        archive_telemetry(tmp_path, (journal, selfplay))

    archive_dir = telemetry_archive_dir(tmp_path)
    for log, payloads in ((journal, journal_payloads), (selfplay, selfplay_payloads)):
        copies = sorted(
            path for path in archive_dir.iterdir() if path.name.startswith(log.name)
        )
        assert len(copies) == ARCHIVE_RETAINED, (
            f"{launches} launches left {len(copies)} copies of {log.name}; the "
            f"archive is bounded at {ARCHIVE_RETAINED}"
        )
        assert [path.read_bytes() for path in copies] == payloads[-ARCHIVE_RETAINED:], (
            f"the surviving copies of {log.name} are not the newest ones; "
            f"eviction takes the oldest, never the launch that just ended"
        )


def test_a_bounded_archive_still_keeps_a_single_launchs_copy(tmp_path: Path) -> None:
    """Non-vacuity for the bound: it evicts, it does not empty.

    A retention rule that deleted every copy would satisfy "at most K files"
    and lose exactly what the archive exists for.
    """
    log = _log(tmp_path)

    archived = archive_telemetry(tmp_path, (log,))

    assert len(archived) == 1
    assert archived[0].read_bytes() == FIRST_LAUNCH


def test_an_unwritable_archive_destination_stops_the_start(tmp_path: Path) -> None:
    """F9: only a name collision is skipped; every other failure is loud.

    ``FileExistsError`` means *these exact bytes at this exact mtime are
    already saved* -- the one condition under which continuing is safe.
    Broadening the refusal to ``except OSError`` (the natural "make it work on
    a read-only root" edit) reinstates v1's failure class exactly: a relaunch
    onto a root whose archive cannot be written proceeds to append to a history
    nothing saved, silently.

    Here the archive directory exists and is not writable, so the ``O_EXCL``
    create fails with ``EACCES`` -- an ``OSError`` that is not a
    ``FileExistsError``, which is the whole distinction.
    """
    if getattr(os, "geteuid", lambda: 1)() == 0:
        pytest.skip("root ignores directory permissions, so nothing would fail")
    log = _log(tmp_path)
    archive_dir = telemetry_archive_dir(tmp_path)
    archive_dir.mkdir(parents=True)
    archive_dir.chmod(0o500)

    try:
        with pytest.raises(OSError) as raised:
            archive_telemetry(tmp_path, (log,))
    finally:
        # Restored unconditionally, or pytest's own cleanup cannot remove it.
        archive_dir.chmod(0o700)

    assert not isinstance(raised.value, FileExistsError), (
        f"an unwritable archive raised {raised.value!r}, which the collision "
        f"path skips: the start would have continued with nothing saved"
    )


def test_a_failure_to_archive_is_not_swallowed_as_a_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same claim without a permission bit: ENOSPC must escape.

    ``guard_truncation`` is the one call that can fail here, and a full disk is
    the failure this bound (:data:`~komugi.campaign.telemetry.ARCHIVE_RETAINED`)
    exists to make less likely rather than impossible. It must still stop the
    start rather than be mistaken for "already archived".
    """
    log = _log(tmp_path)

    def full_disk(path: Path, archive_dir: Path) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(campaign_telemetry, "guard_truncation", full_disk)

    with pytest.raises(OSError) as raised:
        archive_telemetry(tmp_path, (log,))

    assert raised.value.errno == errno.ENOSPC
