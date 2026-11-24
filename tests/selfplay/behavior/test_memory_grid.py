"""The desktop memory grid (design F12/D9): the harness, not the fitted model.

Design D9 defers node pools -- an explicit ``DESIGN.html`` section 7.3 deviation
-- on the strength of an arithmetic model: a ``_Node`` costs about 1.08MB at the
engine's action space, a search allocates at most ``n + 1`` of them, so a full
search at ``n=256`` is around 278MB and sixteen workers are bounded by
``workers * (n_full + 1) * 1.1MB``.  The pre-registered evidence that re-opens
the pools ticket if that model is wrong is a measured grid of ``(workers,
n_full)`` against the cgroup's own ``memory.peak``.

This module tests the **harness**, at two toy points.  It cannot test the fitted
model: that is a number produced by a real desktop run and committed with it
(design section 3's "fitted model committed with the run"), and asserting a
memory figure inside CI would pin the CI machine's allocator rather than the
design's claim.

What is asserted, and why each is worth a line:

* ``cgroup_memory_peak()`` either reports a real peak or says ``None`` -- and
  ``None`` skips, loudly, with the reason.  A harness that silently reported 0
  on a machine without cgroup v2 would turn "we never measured" into "we
  measured nothing", which is how a deferral quietly loses its evidence.
* the grid echoes back the points it was asked for.  A grid whose rows do not
  say which ``(workers, n_full)`` they came from is unfittable.
* every peak is positive and **under 2 GiB**.  No ordering is asserted between
  the two points: at these toy sizes the difference between one worker and two
  is inside the noise of importing torch twice, and a suite that asserted
  monotonicity here would be asserting scheduling luck.  The ceiling is the real
  assertion -- a toy net at ``n_full=4`` that manages to touch two gigabytes has
  found a runaway allocation, which is precisely what the grid exists to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_POINTS: list[tuple[int, int]] = [(1, 4), (2, 4)]
"""``(workers, n_full)``, both tiny.  Two points, because one cannot show that
the grid varies with its input and three would double the runtime for nothing."""

_CEILING_BYTES = 2 * 1024**3
"""Two gibibytes: a sanity ceiling for a toy net, not a budget."""

_RUN_SEED = 606060


@pytest.fixture(scope="module")
def grid(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    """Measure the two toy points, or skip if this machine cannot be measured."""
    peak = sp.cgroup_memory_peak()
    if peak is None:
        pytest.skip(
            "cgroup_memory_peak() is None: this machine exposes no cgroup v2 "
            "memory.peak (no unified hierarchy, or the memory controller is not "
            "delegated to this cgroup), so design F12's grid has nothing to "
            "measure against. The grid is desktop evidence, not a CI gate."
        )
    assert isinstance(peak, int) and peak > 0, (
        f"cgroup_memory_peak() returned {peak!r}: a non-None reading is a byte "
        f"count and must be positive"
    )
    if sp.cgroup_is_shared():
        pytest.skip(
            "this process cohabits its cgroup with foreign processes (a "
            "desktop app scope), so memory.peak is the high-water mark of "
            "everything in it -- a toy grid point 'peaked' at 46 GiB of "
            "browser here (wave-9c review F7). The committed grid run must "
            "use an isolated scope: systemd-run --user --scope -- <cmd>."
        )
    # KOM-055: cohabitation is not the only way the reading stops being this
    # grid's. `memory_peak_is_resettable` is reached through the module rather
    # than the package because the package's __all__ is not this ticket's to
    # edit; it is the same function either way.
    if not sp.memgrid.memory_peak_is_resettable():
        pytest.skip(
            "KOM-055: memory.peak is not resettable here -- the cgroup "
            "filesystem is read-only, as it is for a sandboxed gate running "
            "inside a systemd service cgroup. Every point would then report "
            "the cgroup's LIFETIME high-water mark (the service manager's, "
            "since boot) instead of this grid's, which is how a toy point "
            "came back at 5.4 GB. Unattributable, so not measured."
        )

    root = Path(tmp_path_factory.mktemp("grid")) / "shards"
    root.mkdir()
    # The KOM-023 pin is positional: measure_grid(model_factory, points,
    # base_orchestrator, selfplay, inference). The original bind_kwargs guess
    # at a knob-style signature never constructed an OrchestratorConfig and
    # TypeError'd on any cgroup-v2 machine (wave-9c review F1) -- this fixture
    # had never executed until this fix.
    base = rig.orchestrator_config(root, run_seed=_RUN_SEED, games_per_worker=1)
    selfplay_config = rig.tiny_selfplay_config(workers=1)
    inference = rig.inference_config(max_batch=2)

    shm_before = rig.shm_snapshot()
    # `new_shm_survivors` is run-scoped since KOM-054, and this is the one
    # caller whose `run_selfplay` happens in a SPAWNED CHILD (measure_grid runs
    # a point per child). It still sees the child's segments because the scope
    # travels in the environment, which a spawned child inherits -- so a
    # measure_grid that ever passed an explicit `env=` to its children would
    # silently make the assertion below vacuous.

    def call() -> Any:
        return sp.measure_grid(
            rig.tiny_factory, _POINTS, base, selfplay_config, inference
        )

    outcome = rig.walled(call, rig.E2E_WALL_S, "memory-grid")
    if outcome.error is not None:
        raise AssertionError(
            f"measure_grid raised {type(outcome.error).__name__}: {outcome.error}"
        ) from outcome.error
    leaked = rig.new_shm_survivors(shm_before)
    rig.kill_stragglers()
    assert not leaked, f"the grid leaked segments: {sorted(leaked)}"
    assert outcome.summary is not None, "measure_grid returned None"
    return list(outcome.summary)


def test_the_grid_has_one_row_per_requested_point(grid: list[Any]) -> None:
    """Two points in, two ``GridPoint``s out, in the order they were asked for."""
    assert len(grid) == len(_POINTS), (
        f"asked for {_POINTS}, got {len(grid)} row(s): {grid}"
    )


def test_each_row_echoes_the_point_it_measured(grid: list[Any]) -> None:
    """``workers`` and ``n_full`` come back, so the grid can be fitted.

    The whole artefact is a regression of peak bytes on ``(workers, n_full)``;
    a row that does not carry its own coordinates is a number with no x-axis.
    """
    for point, requested in zip(grid, _POINTS):
        workers, n_full = requested
        assert point.workers == workers, (
            f"row {point} reports workers={point.workers}, requested {workers}"
        )
        assert point.n_full == n_full, (
            f"row {point} reports n_full={point.n_full}, requested {n_full}"
        )


def test_every_peak_is_a_positive_number_of_bytes(grid: list[Any]) -> None:
    """A zero or negative peak means the reading failed and was not reported."""
    for point in grid:
        assert isinstance(point.peak_bytes, int), (
            f"peak_bytes is {point.peak_bytes!r}, want an integer byte count"
        )
        assert point.peak_bytes > 0, (
            f"row {point} measured a peak of {point.peak_bytes} bytes; a run that "
            f"spawned processes and played games cannot have peaked at zero"
        )


def test_no_toy_point_comes_anywhere_near_two_gibibytes(grid: list[Any]) -> None:
    """The sanity ceiling: a runaway allocation fails loudly instead of swapping.

    At ``n_full=4`` design D9's own model predicts single-digit megabytes of node
    arrays per in-flight search, so the entire distance between that and 2 GiB is
    slack for torch's own footprint.  Crossing it is not a tight budget being
    missed, it is the allocation model being wrong -- which is exactly the
    trigger the grid was pre-registered to provide.
    """
    for point in grid:
        assert point.peak_bytes < _CEILING_BYTES, (
            f"row {point} peaked at {point.peak_bytes / 1024**3:.2f} GiB, over the "
            f"{_CEILING_BYTES / 1024**3:.0f} GiB sanity ceiling for a toy net at "
            f"n_full={point.n_full}"
        )
