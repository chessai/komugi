"""The leak detector must see this run's segments, and only this run's.

KOM-054, and it is a test about a *test*: ``/dev/shm`` is machine-global, so
"did this run leak a shared-memory segment?" is answered by diffing a directory
every process on the box writes to.  Until now
:func:`orchestrate_rig.new_shm_survivors` diffed it filtered on the bare
``psm_`` prefix -- which is the prefix of *every* POSIX segment -- and its own
docstring admitted the consequence: the suite only worked if nothing else on
the machine held a segment across its window.  KOM-053's investigation
root-caused an overnight gate flake to exactly that, so the assumption is now
retired: segments carry the run scope that created them
(:func:`komugi.selfplay.orchestrate.slot_shm_name`) and the detector filters on
it.

A filter is only worth having if it is tight in both directions, and one
direction is much easier to get accidentally right than the other.  So this
module pins both, and pins them against the *same* mechanism:

* **(a) foreign segments are invisible.**  A second, deliberately
  leaky-looking source creates ``psm_``-style segments -- one with
  ``SharedMemory``'s own default name, one shaped exactly like an orchestrator
  segment from a *different* process tree -- after the pre-run snapshot and
  holds them across a normal bounded run.  They must not appear in this run's
  survivors.  And because "nothing appeared" is also what a broken filter
  says, the same window is re-read through the OLD ``psm_`` filter, which must
  see them: that is what makes this a demonstration of the flake rather than a
  claim about it.
* **(b) real leaks are still caught.**  The non-vacuity guard, without which
  (a) is satisfied by a filter that matches nothing.  A launch fails after
  creating its segments and its cleanup *skips* one of them -- a genuine leak,
  of a segment belonging to this run -- and the detector has to name it.

(b) reuses KOM-053's half-launched harness (``test_failures_are_loud``'s
``half_launched``) with one difference that is the whole point: KOM-053's
patched ``_release`` did its job and *then* raised, because it was testing that
the sweep continues; this one does not release at all, because it is testing
that the detector notices.
"""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

_orchestrate: ModuleType = importlib.import_module("komugi.selfplay.orchestrate")
"""The implementation module: this suite patches its process targets and its
segment release, exactly as ``test_failures_are_loud`` does and for the same
reason -- ``run_selfplay`` reads both out of these globals."""

if not hasattr(_orchestrate, "shm_scope"):
    pytest.skip(
        "komugi.selfplay.orchestrate exports no shm_scope yet (KOM-054)",
        allow_module_level=True,
    )

_RUN_SEED = 54054

_FOREIGN_SCOPE: Final[str] = "psm_kzz054"
"""A scope no run of THIS process tree can ever mint.

:func:`~komugi.selfplay.orchestrate._resolve_shm_scope` spells a scope
``psm_k<pid in hex><4 hex>``, so a ``z`` in the first position after the ``k``
is unreachable by construction -- this cannot accidentally land inside our own
scope and make the test pass for the wrong reason.  :func:`_foreign_segments`
asserts that anyway, because "by construction" is how the original
serial-execution assumption was justified too.
"""

_FOREIGN_SIZE: Final[int] = 64
"""Big enough to be a real segment, small enough to cost nothing."""


def _foreign_segments() -> list[SharedMemory]:
    """Create the segments a *concurrent* orchestrator would be holding.

    In this process rather than a spawned one, deliberately: the detector
    filters on a NAME, so where the creating code runs is not a variable it can
    read, and a helper process would only add a kill-and-reap dance whose
    failure mode is a hung gate.  What has to be foreign is the name, and these
    two cover both shapes a real neighbour produces -- ``SharedMemory``'s
    default ``psm_<hex>``, and an orchestrator segment from another process
    tree (which is what the overnight flake actually was).

    The caller owns them and must unlink them; they are created *after* the
    pre-run snapshot so that they are genuinely new arrivals in the window
    under audit rather than pre-existing furniture the diff would drop anyway.
    """
    scope = _orchestrate.shm_scope()
    names = (None, f"{_FOREIGN_SCOPE}r0i0s0")
    created: list[SharedMemory] = []
    try:
        for name in names:
            created.append(
                SharedMemory(create=True, size=_FOREIGN_SIZE, name=name)
            )
    except BaseException:
        _unlink_all(created)
        raise
    for segment in created:
        assert not segment.name.startswith(scope), (
            f"the 'foreign' segment {segment.name!r} is inside this run's own "
            f"scope {scope!r}, so it proves nothing about foreign segments"
        )
        assert segment.name.startswith("psm_"), (
            f"the foreign segment {segment.name!r} is not in the psm_ family, "
            f"so the old prefix filter would not have seen it either"
        )
    return created


def _unlink_all(segments: list[SharedMemory]) -> None:
    """Close and unlink every foreign segment, whatever else happened.

    Never skipped: a test about leaked segments that leaks its own props would
    poison every module that runs after it.  ``FileNotFoundError`` is the one
    thing suppressed, because "already gone" is this function's postcondition.
    """
    for segment in segments:
        segment.close()
        with contextlib.suppress(FileNotFoundError):
            segment.unlink()


# ---------------------------------------------------------------------------
# process targets for the half-launched scenario
#
# Local copies of ``test_failures_are_loud``'s, module-level because ``spawn``
# pickles a process target by qualified name.  Copied rather than imported so
# that KOM-053's module keeps passing exactly as written -- this ticket may not
# reshape it -- and so a spawned child re-imports THIS module, whose top-level
# guards cannot skip once ``run_selfplay`` and ``shm_scope`` exist.
# ---------------------------------------------------------------------------


def idle_server_main(
    model_factory: Any, inference: Any, handles: Any, stop_event: Any
) -> None:
    """``serve_shard``'s signature, and nothing but a wait for the stop event."""
    del model_factory, inference, handles
    stop_event.wait()


def unspawnable_worker_target() -> Any:
    """Return a worker target ``spawn`` cannot pickle, so ``start()`` raises.

    ``multiprocessing`` pickles the process object in the parent, so this fails
    deterministically inside ``_launch`` -- after the segments are created and
    before the incarnation reaches ``started``, which is the window where
    ``_launch`` owns the cleanup itself.
    """

    def target(*args: Any) -> None:
        raise AssertionError("this target can never reach a child process")

    return target


# ---------------------------------------------------------------------------
# (a) a clean run beside a foreign segment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopedRun:
    """One clean run measured through both the new filter and the old one."""

    scoped: set[str]
    """Survivors under this process tree's run scope -- the real check."""

    unscoped: set[str]
    """Survivors under the bare ``psm_`` prefix the rig used before KOM-054."""

    foreign: frozenset[str]
    """The names the concurrent-neighbour stand-in held across the run."""


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory: pytest.TempPathFactory) -> ScopedRun:
    """A normal bounded run with foreign ``psm_*`` segments alive beside it."""
    root = Path(tmp_path_factory.mktemp("shm-scope-clean")) / "shards"
    root.mkdir()

    before = rig.shm_snapshot()
    foreign = _foreign_segments()
    try:
        outcome = rig.walled_run_selfplay(
            rig.tiny_factory,
            rig.orchestrator_config(
                root, run_seed=_RUN_SEED, workers=1, games_per_worker=1
            ),
            rig.tiny_selfplay_config(workers=1),
            rig.inference_config(max_batch=1),
            wall_s=rig.E2E_WALL_S,
            label="shm-scope-clean",
        )
        outcome.unwrap()
        # Both readings before any cleanup, and the scoped one first: it is the
        # assertion the ticket is about, and the unscoped one exists only to
        # show what it is protecting against.
        scoped = rig.new_shm_survivors(before)
        unscoped = rig.new_shm_survivors(before, scope="psm_")
        rig.kill_stragglers()
        return ScopedRun(
            scoped=scoped,
            unscoped=unscoped,
            foreign=frozenset(segment.name for segment in foreign),
        )
    finally:
        _unlink_all(foreign)


def test_a_foreign_segment_is_invisible_to_this_runs_leak_check(
    clean_run: ScopedRun,
) -> None:
    """The KOM-054 regression: a neighbour's segment is not this run's leak.

    Both halves matter.  The foreign segments must be absent from the scoped
    survivors -- and they must be *present* in the unscoped ones, because that
    is the reading that used to fail the gate.  Without the second assertion
    this test would also pass against a filter that had simply stopped
    detecting anything.
    """
    assert clean_run.foreign, "the fixture held no foreign segments to be blind to"
    assert clean_run.foreign & clean_run.scoped == frozenset(), (
        f"segments created by another source are being reported as this run's "
        f"leak: {sorted(clean_run.foreign & clean_run.scoped)}. That is the "
        f"KOM-053 gate flake, and the run scope exists to make it impossible"
    )
    assert clean_run.foreign <= clean_run.unscoped, (
        f"the foreign segments were not visible under the pre-KOM-054 psm_ "
        f"filter either ({sorted(clean_run.foreign - clean_run.unscoped)}), so "
        f"this test is not exercising the false-failure it claims to prevent"
    )


def test_a_clean_run_leaves_nothing_of_its_own_behind(clean_run: ScopedRun) -> None:
    """Design D2/F9 still holds, and is still what this suite asserts.

    Scoping the filter narrows *whose* segments are policed, not whether they
    are: a run that unlinked everything it created reports an empty set.
    """
    assert clean_run.scoped == set(), (
        f"the run leaked segments of its own: {sorted(clean_run.scoped)}"
    )


# ---------------------------------------------------------------------------
# (b) non-vacuity: a segment this run really did not release
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakyRun:
    """A launch that failed and genuinely dropped one of its own segments."""

    survivors: set[str]
    leaked_name: str
    error: BaseException


@pytest.fixture(scope="module")
def leaky_run(tmp_path_factory: pytest.TempPathFactory) -> LeakyRun:
    """Half-launch a run whose cleanup skips its first segment (KOM-053 shape).

    ``_release`` is replaced with one that *returns without releasing* the
    first segment it is handed.  Every other obligation of the failure path is
    untouched, so what reaches ``/dev/shm`` afterwards is one live segment of
    this run and nothing else -- the exact signature the detector exists to
    catch, manufactured rather than waited for.

    Two workers, so the run also has a segment that IS released: a detector
    that reported both would be as wrong as one that reported neither.
    """
    root = Path(tmp_path_factory.mktemp("shm-scope-leaky")) / "shards"
    root.mkdir()

    real_release = _orchestrate._release
    skipped: list[Any] = []

    def skipping_release(slots: Any) -> None:
        if not skipped:
            # Held, not merely un-unlinked: keeping the Slots reachable keeps
            # its numpy views and its SharedMemory alive, so the segment stays
            # in /dev/shm for the detector to find and stays releasable here.
            skipped.append(slots)
            return
        real_release(slots)

    before = rig.shm_snapshot()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_orchestrate, "serve_shard", idle_server_main)
            patch.setattr(_orchestrate, "worker_main", unspawnable_worker_target())
            patch.setattr(_orchestrate, "_release", skipping_release)
            outcome = rig.walled_run_selfplay(
                rig.tiny_factory,
                rig.orchestrator_config(
                    root, run_seed=_RUN_SEED, workers=2, games_per_worker=1
                ),
                rig.tiny_selfplay_config(workers=2),
                rig.inference_config(max_batch=2),
                wall_s=rig.LOUD_WALL_S,
                label="shm-scope-leaky",
            )
        error = outcome.raised()
        assert skipped, (
            "the launch never reached its segment cleanup, so no leak was "
            "manufactured and this fixture proves nothing"
        )
        survivors = rig.new_shm_survivors(before)
        rig.kill_stragglers()
        return LeakyRun(
            survivors=survivors,
            leaked_name=str(skipped[0].shm.name),
            error=error,
        )
    finally:
        # This module manufactured the leak, so this module cleans it up --
        # after the reading, never before, or the fixture would repair exactly
        # what it is measuring.
        for slots in skipped:
            real_release(slots)


def test_a_segment_this_run_failed_to_release_is_still_caught(
    leaky_run: LeakyRun,
) -> None:
    """The non-vacuity guard: scoping must not blind the detector.

    The failure this test is written against is the tempting version of
    KOM-054 -- a filter narrow enough that nothing matches it, which reports a
    clean ``/dev/shm`` forever and would have silently disarmed every leak
    assertion in this suite.  Here the leak is real, it belongs to this run,
    and the detector has to name it.
    """
    assert leaky_run.leaked_name in leaky_run.survivors, (
        f"the run kept {leaky_run.leaked_name!r} in /dev/shm and the run-scoped "
        f"leak check did not report it (it reported "
        f"{sorted(leaky_run.survivors)}). A scope that hides this run's own "
        f"leaks is worse than the machine-global filter it replaced"
    )


def test_the_leaky_run_still_fails_loudly(leaky_run: LeakyRun) -> None:
    """A skipped release does not turn a failed launch into a quiet success.

    Guards the manufacturing rather than the detector: if the patched release
    had changed the run's *outcome*, the scenario would no longer be the
    half-launched failure path KOM-053 pinned.
    """
    assert isinstance(leaky_run.error, Exception), (
        f"run_selfplay raised {leaky_run.error!r}, want a normal exception"
    )


def test_only_the_skipped_segment_survives_the_leaky_run(leaky_run: LeakyRun) -> None:
    """The sibling segment, which was released normally, is gone.

    Two workers means two segments; one was skipped and one was released by the
    real ``_release``.  A detector reporting both would be reporting the
    released one as a leak, which is the false-positive direction of the same
    bug.
    """
    assert leaky_run.survivors == {leaky_run.leaked_name}, (
        f"the run-scoped leak check reported {sorted(leaky_run.survivors)}, "
        f"want exactly the one segment whose release was skipped "
        f"({leaky_run.leaked_name!r})"
    )


# ---------------------------------------------------------------------------
# the naming contract itself
# ---------------------------------------------------------------------------


def test_segment_names_stay_in_the_psm_family_and_inside_the_name_budget() -> None:
    """``psm_`` prefix, one scope per tree, one token per run, short enough.

    All four are load-bearing and none of them is checkable from a running
    orchestrator: the family prefix is what every leak tool in this repository
    greps for, distinct tokens are what let two runs of one process be told
    apart, and the length budget is what keeps ``shm_open`` from refusing the
    name on a platform with a tight ``PSHMNAMLEN`` (31 bytes including the
    leading ``/``).
    """
    scope = _orchestrate.shm_scope()
    assert scope.startswith("psm_"), (
        f"the run scope {scope!r} leaves the psm_ family, so every existing "
        f"leak check and runbook stops seeing komugi's segments"
    )
    assert _orchestrate.shm_scope() == scope, "the scope is not stable in a process"

    first = _orchestrate.mint_run_token()
    second = _orchestrate.mint_run_token()
    assert first != second, (
        f"two runs of one process minted the same token {first!r}: their "
        f"segments would collide and neither could be audited alone"
    )
    assert not second.startswith(first) and not first.startswith(second), (
        f"run token {first!r} is a prefix of {second!r}: a token is used as a "
        f"prefix filter, so one run's audit would silently police the other's "
        f"segments too"
    )
    for token in (first, second):
        assert token.startswith(scope), (
            f"run token {token!r} is outside the process scope {scope!r}, so a "
            f"scope-filtered leak check would miss that run entirely"
        )

    names = {
        _orchestrate.slot_shm_name(token, incarnation, slot)
        for token in (first, second)
        for incarnation in range(3)
        for slot in range(8)
    }
    assert len(names) == 2 * 3 * 8, (
        "slot_shm_name collides across (run, incarnation, slot); a reused name "
        "would make a fresh segment race its predecessor's unlink"
    )
    for name in names:
        assert name.startswith(first) or name.startswith(second)
        assert len(name) + 1 <= 31, (
            f"segment name {name!r} is {len(name)} characters; with shm_open's "
            f"leading '/' that exceeds the 31-byte PSHMNAMLEN budget"
        )
