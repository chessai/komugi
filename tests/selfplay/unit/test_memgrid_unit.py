"""Unit tests for the memory grid's measurability seam (KOM-055).

The design F12 grid is only evidence if the number it reports belongs to the
run that produced it, and ``memory.peak`` is a high-water mark that has to be
*reset* for that to be true. Two of the three ways the reset can fail to make
the reading attributable were already covered -- no controller file at all
(``cgroup_memory_peak() is None``) and a cohabited cgroup
(``cgroup_is_shared()``). The third is the one KOM-055 hit, and it has two
layers:

* the file is there, readable, and nobody foreign is in the cgroup, but the
  cgroup filesystem is mounted **read-only** because the process is a sandboxed
  gate run inside a systemd service cgroup, so the write is simply refused;
* and even where the write *succeeds*, on Linux >= 6.11 it resets the watermark
  only for the **file description it was written through**. A reset written
  through one fd and a reading ``open()`` of the same path are two different
  counters, and the reader still gets the cgroup's lifetime mark. The observed
  symptom was a toy point at 5.4 GB, byte-for-byte the orchestrating daemon's
  own ``memory.peak``, on a host where the write had not been refused at all.

So what is pinned here is the seam ``measure_grid`` measures through, against
real files rather than mocks:

* a **read-only** file answers ``False`` from ``memory_peak_is_resettable`` --
  the gate's own situation, and the first half of the finding;
* a **writable** one hands back an open fd whose ``pread`` sees the reset, and
  the reading is proved to happen without reopening the path -- the second
  half, which a predicate that only probed permissions could never catch;
* ``_read_current`` answers ``None`` or a byte count, because it is the
  baseline every reported point is a growth over.

All of them patch ``_peak_path``, the single seam every reader and the writer
go through, so no test here touches the machine's real cgroup.
"""

from __future__ import annotations

import builtins
import os

from pathlib import Path
from typing import NoReturn

import pytest

from komugi.selfplay import memgrid


def test_a_read_only_peak_file_is_reported_as_not_resettable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's case: the write is refused, so the mark is not ours to read."""
    if os.geteuid() == 0:
        pytest.skip(
            "running as root, which bypasses the mode bits this test uses to "
            "stand in for a read-only cgroup filesystem -- the write would "
            "succeed and the case would not be reproduced"
        )
    peak = tmp_path / "memory.peak"
    peak.write_text("5403983872")
    peak.chmod(0o444)
    monkeypatch.setattr(memgrid, "_peak_path", lambda: peak)

    assert memgrid.memory_peak_is_resettable() is False, (
        "a memory.peak that cannot be written must report as not resettable: "
        "the peak read after a grid point would be the cgroup's lifetime "
        "high-water mark, not the point's"
    )
    assert peak.read_text() == "5403983872", (
        "the refused reset must leave the file alone; anything else means the "
        "predicate reported False after partially writing"
    )


def test_the_reset_fd_reads_back_the_reset_without_reopening_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-fd finding: the mark must be read through the fd that reset it.

    ``_open_peak_reset`` hands back the descriptor it wrote through precisely so
    that ``_read_peak`` can ``pread`` it. If ``_read_peak`` ever reopened the
    path instead it would read the cgroup's lifetime mark -- correct-looking,
    reproducibly wrong, and invisible on any host where the write succeeds. So
    ``builtins.open`` is made to explode for the duration of the read: the only
    way to pass is not to open anything.

    The seeded value is one byte because this stands in for a kernfs file with a
    regular one: ``O_RDWR`` without ``O_TRUNC`` overwrites in place, so a longer
    seed would leave its own tail behind on a tmpfile in a way the real
    controller file does not.
    """
    peak = tmp_path / "memory.peak"
    peak.write_text("7")
    monkeypatch.setattr(memgrid, "_peak_path", lambda: peak)

    fd = memgrid._open_peak_reset()
    assert fd is not None, "a writable memory.peak must yield a reset fd"
    try:
        assert peak.read_text() == "0", (
            f"the mark still reads {peak.read_text()!r}: _open_peak_reset "
            f"returned an fd without performing the reset it claims to have "
            f"performed"
        )
        expected = int(os.pread(fd, 64, 0).decode().strip())

        def no_open(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError(
                "_read_peak reopened the path: the reset is per-file-"
                "description on Linux >= 6.11, so a fresh open reads the "
                "cgroup's lifetime high-water mark and not this point's"
            )

        monkeypatch.setattr(builtins, "open", no_open)
        try:
            observed = memgrid._read_peak(fd)
        finally:
            monkeypatch.undo()
    finally:
        os.close(fd)

    assert observed == 0, (
        f"_read_peak returned {observed}, want 0: the fd that wrote the reset "
        f"must see the reset"
    )
    assert observed == expected, (
        f"_read_peak returned {observed} but a pread of the same fd gives "
        f"{expected}: the value did not come from this file description"
    )


def test_the_current_usage_baseline_is_read_when_present_and_none_when_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_read_current`` is the subtrahend, so its absence must be visible.

    Every reported point is ``peak - current-at-the-reset``. A ``_read_current``
    that answered 0 for a missing file would silently turn that growth back into
    a level -- the exact number KOM-055 was about -- so the missing case answers
    ``None`` and ``measure_grid`` raises on it.
    """
    peak = tmp_path / "memory.peak"
    peak.write_text("0")
    monkeypatch.setattr(memgrid, "_peak_path", lambda: peak)

    assert memgrid._read_current() is None, (
        "with no memory.current beside memory.peak there is no baseline, and "
        "None is the only honest answer"
    )

    (tmp_path / "memory.current").write_text("1900544\n")
    assert memgrid._read_current() == 1900544, (
        "a present memory.current must be read as a byte count, trailing "
        "newline and all"
    )
