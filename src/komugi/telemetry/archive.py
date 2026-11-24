"""Archive-on-relaunch: making log truncation cost nothing.

The v1 campaign lost telemetry twice because a relaunch opened the run log
for writing and the previous run's history went with it. The fix is not "be
careful": it is that the only supported way to get an empty log is
:func:`guard_truncation`, which archives first and hands back a
:class:`TruncationToken`. Code that wants to overwrite has to hold the token,
and the token cannot exist unless a copy was made.

Two design choices are load-bearing:

* The archive suffix comes from the log's **own mtime**, never from
  ``time.time()``. A wall-clock suffix names the moment of the *rescue*,
  which is useless for ordering (two relaunches inside the same second
  collide) and actively misleading when a log is archived days after it was
  written. The mtime names the data, so re-archiving the same file computes
  the same destination -- which is what turns a double-archive into a loud
  :class:`FileExistsError` instead of a silent second copy.
* An existing destination is **never** overwritten. The archive is opened
  ``O_EXCL``, so the refusal is the kernel's, not a check-then-write race.

The copy is byte-for-byte and the mtime is carried across, so an archived log
is indistinguishable from the original and can itself be archived again to
the same name.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_COPY_BUFFER_BYTES: Final[int] = 1 << 20

_ARCHIVE_MODE: Final[int] = 0o644

_SUFFIX_DIGITS: Final[int] = 20
"""Zero-padding for the mtime-in-nanoseconds suffix.

Wide enough for nanosecond epochs well past any plausible run date, and fixed
width so lexicographic order over archive names *is* chronological order --
``sorted(archive_dir.iterdir())`` replays a campaign's logs oldest-first
without parsing anything.
"""

_ARCHIVE_SUFFIX: Final[re.Pattern[str]] = re.compile(rf"\.\d{{{_SUFFIX_DIGITS}}}\Z")
"""Matches a suffix this module already appended.

Stripped before a new one is added, so naming is idempotent: an archived log
re-derives its *own* name rather than growing a second suffix, which is what
makes re-archiving a copy a :class:`FileExistsError` instead of a fresh file.
"""


@dataclass(frozen=True, slots=True)
class TruncationToken:
    """Proof that `log_path` was archived, and therefore may be overwritten.

    Only :func:`guard_truncation` produces one. Passing it around is how a
    call chain carries the permission; :meth:`truncate` is the act itself.
    """

    log_path: Path
    archive_path: Path

    def truncate(self) -> None:
        """Empty the log this token was issued for, durably.

        Idempotent: truncating an already-empty log is a no-op, and the
        archived copy is untouched either way.
        """
        fd = os.open(
            self.log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _ARCHIVE_MODE
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def authorizes(self, path: str | os.PathLike[str]) -> bool:
        """Return whether this token permits overwriting `path`."""
        return Path(path).resolve() == self.log_path.resolve()


def archive_name(log_path: str | os.PathLike[str], mtime_ns: int) -> str:
    """Return the archive file name for `log_path` at modification time `mtime_ns`.

    Idempotent: an existing archive suffix is replaced rather than extended,
    so an archived copy names itself the same way its source did.

    Pure, so the naming scheme is testable without touching a filesystem.
    """
    if mtime_ns < 0:
        raise ValueError(f"mtime_ns must be non-negative, got {mtime_ns!r}")
    stem = _ARCHIVE_SUFFIX.sub("", Path(log_path).name)
    return f"{stem}.{mtime_ns:0{_SUFFIX_DIGITS}d}"


def archive_before_truncate(
    log_path: str | os.PathLike[str],
    archive_dir: str | os.PathLike[str],
) -> Path:
    """Copy `log_path` into `archive_dir` under its mtime-derived name.

    Returns the path of the archived copy. Creates `archive_dir` (and its
    parents) if needed.

    Raises ``FileNotFoundError`` if the log does not exist, ``IsADirectoryError``
    if it is not a regular file, and ``FileExistsError`` if the destination is
    already there -- the last is the important one: a name collision means
    *this exact log content* has already been archived, or that something else
    owns that name, and either way overwriting it would destroy the history
    this module exists to keep.
    """
    source = Path(log_path)
    destination_dir = Path(archive_dir)

    stat_result = source.stat()  # FileNotFoundError propagates by design
    if not source.is_file():
        raise IsADirectoryError(f"{source} is not a regular file")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / archive_name(source, stat_result.st_mtime_ns)

    # O_EXCL makes the refusal atomic: no window between "does it exist" and
    # "create it" for a second process to slip through.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _ARCHIVE_MODE)
    try:
        with open(fd, "wb", closefd=False) as sink, source.open("rb") as origin:
            shutil.copyfileobj(origin, sink, _COPY_BUFFER_BYTES)
            sink.flush()
        os.fsync(fd)
    except BaseException:
        # A half-written archive is worse than none: it would occupy the name
        # and make the next attempt refuse.
        os.close(fd)
        destination.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)

    # Carry the mtime across, so the archive re-derives its own name and the
    # copy is indistinguishable from what it replaced.
    shutil.copystat(source, destination)
    return destination


def guard_truncation(
    log_path: str | os.PathLike[str],
    archive_dir: str | os.PathLike[str],
) -> TruncationToken:
    """Archive `log_path` and return the token that permits overwriting it.

    The one supported entry point for relaunch: call it before opening a run
    log for writing, then use :meth:`TruncationToken.truncate` (or simply hold
    the token as evidence for whatever does the overwriting). Every failure
    mode of :func:`archive_before_truncate` propagates, so a run that cannot
    archive does not get permission to truncate.
    """
    source = Path(log_path)
    archived = archive_before_truncate(source, archive_dir)
    return TruncationToken(log_path=source, archive_path=archived)
