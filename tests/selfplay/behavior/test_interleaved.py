"""Two game lanes per worker process: accounting, disjointness, and a hard kill.

``SelfPlayConfig.games_in_flight`` (KOM-140) lets one worker *process* keep more
than one game in flight at a time: ``games_in_flight=2`` gives a worker two
lanes, each with its own SharedMemory segment, its own
:class:`~komugi.selfplay.inference.SlotClient` and its own
:class:`~komugi.train.shards.ShardWriter` directory, playing games from disjoint
deterministic seed lanes.  The default is ``1``, and ``--serial-debug`` forces
``1``.  The point is throughput: a lane blocked on an evaluation is a lane the
other one can overlap with, so the server sees a deeper batch from the same
number of processes.

The feature is implemented concurrently by a different worker, so this module
tests the **contract** and nothing else, and it is written to keep working under
any of the layouts that contract permits:

* **Lane directories are read off ``RunSummary.shard_dirs``, never off a name.**
  Design D10 fixes ``worker_<k>_r<i>`` for the one-lane case; how a second lane
  is spelled inside a run is the implementer's call (a flattened slot id, a
  ``_lane<j>`` suffix, anything else).  The summary is the pinned surface, so
  every assertion below starts there, and the ``_r<i>`` incarnation tag is read
  with a *tolerant* pattern (:func:`_incarnation_of`) rather than
  ``orchestrate_rig.parse_shard_dir``'s ``fullmatch``.
* **Provenance is read off the data.**  ``game_id`` packs ``(worker_id,
  incarnation, game_index)``, so whether the two lanes of one worker differ in
  the worker component (flattened ids) or in the game component (a split
  budget), "no lane replayed another lane's games" is one assertion: the
  per-directory ``game_id`` sets are pairwise disjoint.
* **Every run is real.**  Spawned processes, the real engine, the rig's toy net
  and tiny budgets -- an interleaving bug lives in the process, not in a
  unit-testable object, and a mocked lane would pin nothing.

Why the runs are shaped the way they are
----------------------------------------
Four runs, each in a module-scoped fixture because a real self-play run costs
real seconds and several tests are views of the same run:

``run``/``replica``
    ``workers=2, games_in_flight=2, games_per_worker=3``: the accounting case
    and, at the same ``run_seed`` in a second root, the determinism case.
    Three games over two lanes is deliberately *odd*, so an implementation that
    can only split an even budget fails here rather than in production.
``killed``
    The same shape with one worker process SIGKILLed mid-run.  SIGKILL, not
    ``terminate()``: it leaves both of that worker's segments in whatever state
    they were in mid-request, which is exactly what the liveness gate's own kill
    and the OOM killer do.
``single``
    ``games_in_flight=1``: the equivalence guard.  The flag's default must have
    changed *nothing* observable, so this run is held to the assertions
    ``test_e2e_loop`` already makes about a two-worker run.

Determinism, and why it is claimed here at all.  Production games batch across
workers, so :mod:`komugi.selfplay.worker` promises replay only up to evaluator
tolerance -- but :class:`orchestrate_rig.TinyNet` is elementwise and
deterministic in its input alone (no cross-row reduction anywhere), so batch
composition cannot move a single number.  Under that net two runs of one
``run_seed`` must agree game for game, and the comparison below is per-``game_id``
row counts: a lane that leaked into another lane's seed stream, or a lane
assignment decided by arrival order rather than by construction, changes which
games get played and shows up here as a different map.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import re
import signal
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final

import pytest

from komugi.config import SelfPlayConfig
from komugi.train.shards import ShardReader

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

# The feature gate.  A field probe rather than a ``hasattr`` on an instance,
# because ``SelfPlayConfig`` is a frozen slots dataclass: the field either
# exists in its schema or the whole feature is absent, and constructing one to
# find out would raise ``TypeError`` in the module body instead of skipping.
_CONFIG_FIELDS = {field.name for field in dataclass_fields(SelfPlayConfig)}

if "games_in_flight" not in _CONFIG_FIELDS:
    pytest.skip(
        "SelfPlayConfig has no games_in_flight field yet (KOM-140): per-worker "
        "game interleaving is not implemented on this base",
        allow_module_level=True,
    )

_WORKERS: Final[int] = 2
_LANES: Final[int] = 2
_GAMES_PER_WORKER: Final[int] = 3
"""Odd on purpose: three games over two lanes cannot be split evenly."""

_RUN_SEED: Final[int] = 140140
_KILL_RUN_SEED: Final[int] = 140141
_SINGLE_RUN_SEED: Final[int] = 140142
_GATE_RUN_SEED: Final[int] = 140143

_POLL_S: Final[float] = 0.05
"""How often the killer thread looks at the run.  Fast: it is racing a worker
process, and every pass is two cheap syscalls."""

_KILL_DEADLINE_S: Final[float] = rig.RESTART_WALL_S / 2.0
"""How long the killer waits for something to kill before giving up.

Half the run's wall, so a killer that never fires reports "the scenario never
ran" as a *test failure with a reason* rather than as the run's wall expiring
with no explanation.
"""

_INCARNATION_RE: Final[str] = r"_r(\d+)"
"""The incarnation tag inside a shard directory name, matched permissively.

``orchestrate_rig.SHARD_DIR_RE`` is a ``fullmatch`` on ``worker_<k>_r<i>`` and
is correct for one lane per worker; a second lane may carry a further component
in its name, which that pattern would reject.  This searches for the tag
instead, so ``worker_0_r1``, ``worker_0_r1_lane1`` and ``worker_0_lane1_r1`` all
report incarnation 1.
"""


def _incarnation_of(path: Path) -> int | None:
    """Return the incarnation tagged in a shard directory name, if any."""
    match = re.search(_INCARNATION_RE, path.name)
    return int(match.group(1)) if match is not None else None


@dataclass(frozen=True)
class Lane:
    """One writer directory, reduced to what the assertions need.

    Read once, in the fixture, because a shard row carries a ``(128, 9, 9)``
    observation and half of this module would otherwise inflate the same
    hundreds of megabytes again per test.  Every row is schema-validated during
    that read, so a lane whose rows are malformed fails at the fixture rather
    than surviving into a digest comparison that happens to agree.
    """

    directory: Path
    rows: int
    per_game_rows: dict[int, int]
    digest: str
    skipped: tuple[str, ...]

    @property
    def game_ids(self) -> frozenset[int]:
        """The distinct games whose rows landed in this directory."""
        return frozenset(self.per_game_rows)


@dataclass(frozen=True)
class Run:
    """One finished run and everything read off it, snapshotted in the fixture."""

    summary: Any
    root: Path
    elapsed: float
    lanes: tuple[Lane, ...]
    shm_leaked: set[str]
    children_leaked: dict[int, str]
    victim_exit: int | None = None
    killer_error: str | None = None

    @property
    def rows_on_disk(self) -> int:
        """Rows readable across every directory the summary reported."""
        return sum(lane.rows for lane in self.lanes)

    @property
    def per_game_rows(self) -> dict[int, int]:
        """``game_id -> rows``, merged over every lane of the run."""
        merged: dict[int, int] = {}
        for lane in self.lanes:
            for game, count in lane.per_game_rows.items():
                merged[game] = merged.get(game, 0) + count
        return merged


def _scan(directory: Path) -> Lane:
    """Read one lane directory and reduce it to a :class:`Lane`.

    The digest covers the per-row identifying columns only -- ``game_id``,
    ``ply``, ``to_play``, ``search_mode``, ``sims_used`` -- and deliberately not
    the float payloads: two lanes playing genuinely different games differ in
    these integers, while float equality would make the comparison a statement
    about the toy net's arithmetic instead of about the lanes.

    :class:`~komugi.train.shards.ShardReader` is built here rather than through
    ``orchestrate_rig.reader_for``, which asserts that *nothing* was skipped: a
    SIGKILLed worker may leave one half-written shard behind, which the
    orchestrator's own ``count_rows`` skips too, so the run whose whole subject
    is a hard kill must be able to read its directories at all.  What was
    skipped is carried on the :class:`Lane` instead, and the tests that expect a
    clean run assert it is empty.
    """
    hasher = hashlib.sha256()
    per_game: dict[int, int] = {}
    # ``retain_arrays=False`` and a streaming loop rather than ``list(reader)``:
    # a row carries a (128, 9, 9) observation and a 37341-wide policy target, so
    # a lane's worth is hundreds of megabytes held twice over. This walks the
    # shards in order, which is the access pattern the unretained reader's
    # one-shard cache is built for, and keeps only the counters.
    reader = ShardReader(directory, schema=sp.SCHEMA, retain_arrays=False)
    skipped = tuple(
        f"{entry.path.name}: {entry.reason}" for entry in reader.skipped
    )
    rows = 0
    for row in reader:
        rows += 1
        sp.SCHEMA.validate(row)
        game = rig.as_int(row, "game_id")
        per_game[game] = per_game.get(game, 0) + 1
        hasher.update(
            f"{game},{rig.as_int(row, 'ply')},{rig.as_int(row, 'to_play')},"
            f"{rig.as_int(row, 'search_mode')},{rig.as_int(row, 'sims_used')};".encode(
                "ascii"
            )
        )
    return Lane(
        directory=directory,
        rows=rows,
        per_game_rows=per_game,
        digest=hasher.hexdigest(),
        skipped=skipped,
    )


def _has_data(root: Path) -> bool:
    """Whether any worker has written a file anywhere under `root` yet."""
    try:
        return any(entry.is_file() for entry in root.rglob("*"))
    except OSError:
        return False


def _live_workers() -> list[BaseProcess]:
    """Every live child of this interpreter that is not the inference server.

    ``multiprocessing.active_children`` rather than a ``/proc`` scan, because
    the orchestrator runs on the wall's helper thread *in this process* and its
    children are therefore in this module's registry -- and because a ``/proc``
    entry's ``comm`` is the interpreter's name for every one of them, which
    cannot tell a worker from the server.  The server is excluded by name: it is
    the one child whose death is specified to fail the run loudly (design
    D2/Q4), so killing it would test something else entirely.
    """
    try:
        children = multiprocessing.active_children()
    except RuntimeError:
        # The registry was mutated by the orchestrator thread mid-iteration.
        # Nothing to do but look again on the next pass.
        return []
    return [proc for proc in children if "server" not in proc.name.lower()]


class _WorkerKiller:
    """SIGKILL one worker process, once, after the run has produced data.

    Waiting for the first file under the shard root before firing is what makes
    this a *mid-run* kill rather than a race with process startup: a worker that
    has sealed a shard has attached its segments, rung its doorbells and
    finished at least one game, so the kill lands on a live seam with lanes in
    flight.  Only one process is ever killed -- design D2 restarts the whole
    shard on any worker death, so a second kill would be testing the restart
    budget instead.
    """

    def __init__(self, deadline_s: float = _KILL_DEADLINE_S) -> None:
        self.deadline_s = deadline_s
        self.victim: BaseProcess | None = None
        self.error: str | None = None

    def __call__(self, root: Path, stop: threading.Event) -> None:
        """Run on a daemon thread for the duration of one ``run_selfplay``."""
        end = time.monotonic() + self.deadline_s
        while not stop.is_set() and time.monotonic() < end:
            if _has_data(root):
                break
            time.sleep(_POLL_S)
        while not stop.is_set() and time.monotonic() < end:
            for proc in _live_workers():
                pid = proc.pid
                if pid is None or not proc.is_alive():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    # It exited between the check and the signal; try the next.
                    self.error = f"kill of {proc.name} (pid {pid}) failed: {exc}"
                    continue
                self.error = None
                self.victim = proc
                return
            time.sleep(_POLL_S)
        if self.victim is None and self.error is None:
            self.error = (
                f"nothing was killed: no live worker process was found before the "
                f"run ended or the {self.deadline_s}s deadline expired "
                f"(stopped={stop.is_set()}). The restart this scenario is about "
                f"was never provoked, so the run below proves nothing"
            )


def _play(
    root: Path,
    *,
    lanes: int,
    games_per_worker: int,
    run_seed: int,
    wall_s: float,
    label: str,
    workers: int = _WORKERS,
    killer: _WorkerKiller | None = None,
) -> Run:
    """Run ``run_selfplay`` once under the rig's wall and read the result back.

    ``killer`` runs on its own daemon thread beside the run, because
    :func:`orchestrate_rig.walled` occupies the calling thread until the
    orchestrator returns: a kill scheduled from here would land after the run it
    was supposed to interrupt.  The thread is stopped in a ``finally`` so a run
    that raises cannot leave it hunting for a process in the *next* test's run.
    """
    root.mkdir(parents=True, exist_ok=True)
    shm_before = rig.shm_snapshot()
    children_before = set(rig.child_pids())

    stop = threading.Event()
    helper: threading.Thread | None = None
    if killer is not None:
        helper = threading.Thread(
            target=killer, args=(root, stop), name=f"killer-{label}", daemon=True
        )
        helper.start()
    try:
        outcome = rig.walled_run_selfplay(
            rig.tiny_factory,
            rig.orchestrator_config(
                root,
                run_seed=run_seed,
                workers=workers,
                games_per_worker=games_per_worker,
            ),
            rig.tiny_selfplay_config(workers=workers, games_in_flight=lanes),
            # One batch slot per lane in flight: the whole point of interleaving
            # is that all workers * lanes clients can have a request pending at
            # once, and a shallower batch would serialise them back.
            rig.inference_config(max_batch=workers * lanes),
            wall_s=wall_s,
            label=label,
        )
    finally:
        stop.set()
        if helper is not None:
            helper.join(rig.KILL_GRACE_S)
    summary = outcome.unwrap()

    # Snapshot the leftovers before anything here can tidy them by accident,
    # then tidy up for whatever runs after this module.
    leaked_shm = rig.new_shm_survivors(shm_before)
    leaked_children = {
        pid: name
        for pid, name in rig.child_pids().items()
        if pid not in children_before
    }
    rig.kill_stragglers()
    victim = killer.victim if killer is not None else None
    return Run(
        summary=summary,
        root=root,
        elapsed=outcome.elapsed,
        lanes=tuple(_scan(directory) for directory in rig.summary_dirs(summary)),
        shm_leaked=leaked_shm,
        children_leaked=leaked_children,
        victim_exit=None if victim is None else victim.exitcode,
        killer_error=killer.error if killer is not None else None,
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Two workers, two lanes each, three games per worker."""
    return _play(
        Path(tmp_path_factory.mktemp("lanes")) / "shards",
        lanes=_LANES,
        games_per_worker=_GAMES_PER_WORKER,
        run_seed=_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="interleaved",
    )


@pytest.fixture(scope="module")
def replica(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The same run again, same ``run_seed``, its own root."""
    return _play(
        Path(tmp_path_factory.mktemp("lanes-replica")) / "shards",
        lanes=_LANES,
        games_per_worker=_GAMES_PER_WORKER,
        run_seed=_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="interleaved-replica",
    )


@pytest.fixture(scope="module")
def killed(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The same shape, with one worker process SIGKILLed mid-run."""
    return _play(
        Path(tmp_path_factory.mktemp("lanes-killed")) / "shards",
        lanes=_LANES,
        games_per_worker=_GAMES_PER_WORKER,
        run_seed=_KILL_RUN_SEED,
        wall_s=rig.RESTART_WALL_S,
        label="interleaved-killed",
        killer=_WorkerKiller(),
    )


@pytest.fixture(scope="module")
def single(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """``games_in_flight=1``: the shape ``test_e2e_loop`` already pins."""
    return _play(
        Path(tmp_path_factory.mktemp("lanes-single")) / "shards",
        lanes=1,
        games_per_worker=1,
        run_seed=_SINGLE_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="single-lane",
    )


def test_the_interleaved_run_completed_exactly_the_budget_it_was_asked_for(
    run: Run,
) -> None:
    """``games_per_worker`` is the worker's budget, not each lane's.

    The failure this catches is a lane that reads the budget as its own: with
    two lanes that quietly doubles the run, which looks like success (more data,
    no error) while every downstream count -- iteration size, window recency,
    the wall-clock an operator planned for -- is wrong by a factor of
    ``games_in_flight``.  Asserted as equality rather than ``>=`` for exactly
    that reason, with ``restarts == 0`` first so a restart's documented
    over-count cannot be mistaken for it.
    """
    assert run.summary.restarts == 0, (
        f"nothing was killed or wedged, so nothing should have restarted "
        f"(restarts={run.summary.restarts}); a restart replays the whole budget "
        f"and would make the count below meaningless"
    )
    assert rig.summary_incarnations(run.summary) == [0] * _WORKERS, (
        "no restart means every worker finished on its first incarnation"
    )
    assert run.summary.games_completed == _WORKERS * _GAMES_PER_WORKER, (
        f"asked for {_WORKERS} workers x {_GAMES_PER_WORKER} games with "
        f"{_LANES} lanes each; summary reports "
        f"{run.summary.games_completed} completed"
    )
    assert len(run.per_game_rows) == _WORKERS * _GAMES_PER_WORKER, (
        f"the run's rows decode to {len(run.per_game_rows)} distinct game_ids, "
        f"want {_WORKERS * _GAMES_PER_WORKER}: the lanes must SPLIT the budget, "
        f"not each play it"
    )


def test_every_lane_wrote_rows_and_the_summary_counted_all_of_them(run: Run) -> None:
    """No silent single-lane run, and ``rows_written`` is the truth on disk.

    Two failures, one test, because either one alone passes the other's check.
    A run whose second lane never started produces a perfectly consistent
    summary over half the directories -- the interleaving did not happen and
    nothing says so -- hence the ``workers * lanes`` floor on *populated* writer
    directories.  And a ``rows_written`` computed from what the orchestrator
    expected rather than counted from the shards is how a lane's tail goes
    missing without anybody noticing, hence the comparison against rows read
    back through the schema-checked reader.
    """
    assert len(run.lanes) >= _WORKERS * _LANES, (
        f"the summary reports {len(run.lanes)} productive writer directories: "
        f"{rig.describe([lane.directory for lane in run.lanes])}. With "
        f"{_WORKERS} workers x {_LANES} lanes every lane owns its own directory, "
        f"so fewer means at least one lane never played"
    )
    empty = [lane.directory.name for lane in run.lanes if lane.rows == 0]
    assert not empty, f"lane directories reported but holding no rows: {empty}"
    damaged = {lane.directory.name: lane.skipped for lane in run.lanes if lane.skipped}
    assert not damaged, (
        f"nothing was killed, so every shard must be readable: {damaged}"
    )
    assert run.rows_on_disk == run.summary.rows_written, (
        f"summary claims {run.summary.rows_written} rows, the shards hold "
        f"{run.rows_on_disk}"
    )


def test_the_lanes_of_a_worker_played_different_games(run: Run) -> None:
    """Disjoint seed lanes, read off the data rather than off the seeding code.

    The bug is a lane whose stream is keyed on ``(run_seed, worker_id,
    incarnation, game_index)`` without the lane in it: both lanes of a worker
    then derive the same seeds, play the same games move for move, and write two
    copies of one game under one ``game_id``.  Nothing else notices -- the row
    count is right, the schema is right, the search is right -- and the trainer
    silently gets half the entropy it was promised.

    Stated as pairwise disjointness over every lane directory, which is the same
    assertion whichever way the lane enters the key: a flattened slot id moves
    the ``worker_id`` component of ``game_id``, a split budget moves the
    ``game_index`` component, and a lane that entered neither collides here.
    The digest is the belt to that braces: two directories holding genuinely
    different play cannot agree on the identifying columns of every row.
    """
    lanes = run.lanes
    assert len(lanes) >= _WORKERS * _LANES, "the run did not produce lane dirs"
    for i, first in enumerate(lanes):
        for second in lanes[i + 1 :]:
            shared = first.game_ids & second.game_ids
            assert not shared, (
                f"{first.directory.name} and {second.directory.name} both hold "
                f"game_id(s) {sorted(shared)}: two lanes played the same game, so "
                f"their seed streams are not disjoint"
            )
            assert first.digest != second.digest, (
                f"{first.directory.name} and {second.directory.name} are "
                f"identical row for row ({first.digest[:16]}...): the two lanes "
                f"replayed one game"
            )
    workers_seen = {
        rig.worker_of(game) for lane in lanes for game in lane.game_ids
    }
    assert len(workers_seen) >= _WORKERS, (
        f"game_id decodes to worker ids {sorted(workers_seen)}; want at least "
        f"{_WORKERS} distinct workers to have played"
    )


# KOM-236: this is the one node that needs the *second* whole spawned run
# (``replica``), which the gate sandbox cannot afford.
@pytest.mark.sandbox_unsafe
def test_two_runs_of_one_seed_play_the_same_games(run: Run, replica: Run) -> None:
    """Same ``run_seed``, same games, through the real spawned path.

    Determinism is not a nicety here: design D11 makes ``(run_seed, worker_id,
    incarnation, game_index)`` the whole key, and interleaving adds a lane whose
    assignment of games could easily be decided by *arrival* -- whichever lane
    finishes first takes the next index -- which is reproducible only by
    accident.  This compares the map from ``game_id`` to row count over the
    whole run, so it does not care which lane played which game, only that the
    same games got played and each ended where it ended before.

    The comparison is safe under batching because
    :class:`orchestrate_rig.TinyNet` is elementwise: no row's evaluation depends
    on what shared its batch, so two runs cannot legitimately diverge (see the
    module docstring).  A mismatch is a lane leaking into another lane's stream,
    or a seed derived from something the run does not control.
    """
    first, second = run.per_game_rows, replica.per_game_rows
    assert set(first) == set(second), (
        f"the two runs played different games: only in run A "
        f"{sorted(set(first) - set(second))}, only in run B "
        f"{sorted(set(second) - set(first))}"
    )
    differing = {
        game: (first[game], second[game])
        for game in sorted(first)
        if first[game] != second[game]
    }
    assert not differing, (
        f"same run_seed, different game lengths (game_id -> (run A, run B)): "
        f"{differing}"
    )


def test_a_sigkilled_worker_restarts_the_whole_shard_with_fresh_lanes(
    killed: Run,
) -> None:
    """One worker dies hard; both of its lanes come back on fresh everything.

    The interleaving-specific failure is a restart that rebuilds only *one*
    segment per worker: the second lane's respawned client then rings a doorbell
    in a segment the new server never scanned (or, worse, re-attaches the dead
    incarnation's), and the run either wedges until the liveness gate fires again
    or silently plays half the lanes it was configured for.  So the restarted
    incarnation is required to own a COMPLETE fresh set of ``workers * lanes``
    directories -- one per lane, none of them a predecessor's, which is design
    D10/F6's rule (a zombie writer must never be able to collide with its
    successor) applied per lane rather than per worker.
    """
    assert killed.killer_error is None, killed.killer_error
    assert killed.victim_exit == -signal.SIGKILL, (
        f"the victim exited {killed.victim_exit}, not by SIGKILL: the scenario "
        f"did not run, so nothing below was tested"
    )
    assert killed.summary.restarts >= 1, (
        "a worker was SIGKILLed and the run recorded no restart"
    )
    incarnations = rig.summary_incarnations(killed.summary)
    assert incarnations == [killed.summary.restarts] * _WORKERS, (
        f"worker_incarnations {incarnations} after {killed.summary.restarts} "
        f"restart(s): design D2 restarts the WHOLE shard, so every worker ends "
        f"on the same incarnation"
    )

    # Read off the summary rather than off ``root.iterdir()``: the summary is
    # the pinned surface and it names writer directories at whatever depth they
    # live, so this stays true of a layout that nests a lane below its worker.
    tags = {lane.directory: _incarnation_of(lane.directory) for lane in killed.lanes}
    untagged = [path for path, tag in tags.items() if tag is None]
    assert not untagged, (
        f"lane directories without an _r<incarnation> tag: "
        f"{rig.describe(untagged)}. Design D10 puts the incarnation in the NAME, "
        f"which is what keeps a zombie writer off its successor's directory"
    )
    final = max(tag for tag in tags.values() if tag is not None)
    assert final == killed.summary.restarts, (
        f"the newest incarnation on disk is _r{final} after "
        f"{killed.summary.restarts} restart(s); the two must agree or the "
        f"summary is describing a different run than the shards are"
    )
    fresh = [path for path, tag in tags.items() if tag == final]
    assert len(fresh) == _WORKERS * _LANES, (
        f"the restarted incarnation (_r{final}) produced {len(fresh)} lane "
        f"directories, want {_WORKERS} workers x {_LANES} lanes: "
        f"{rig.describe(fresh)}. A restart that rebuilt one segment per worker "
        f"instead of one per lane comes back with half its lanes"
    )


def test_the_killed_run_still_accounts_for_every_row_it_kept(killed: Run) -> None:
    """Correct final accounting across a restart, lanes included.

    A restarted shard replays its whole budget under a new incarnation, so the
    completed count is a floor rather than an equality (the orchestrator's
    documented P0a semantics) -- but it is still a *multiple* of
    ``games_per_worker``, because ``games_completed`` is exit-code bookkeeping
    over whole workers.  ``rows_written``, meanwhile, must still equal what a
    reader can actually read: the killed incarnation's sealed games are real
    data and stay, its half-written tail is skipped, and neither may be counted
    twice or dropped.
    """
    assert killed.summary.games_completed >= _WORKERS * _GAMES_PER_WORKER, (
        f"the run completed {killed.summary.games_completed} games; the restart "
        f"exists so that {_WORKERS * _GAMES_PER_WORKER} still get played"
    )
    assert killed.summary.games_completed % _GAMES_PER_WORKER == 0, (
        f"games_completed={killed.summary.games_completed} is not a multiple of "
        f"games_per_worker={_GAMES_PER_WORKER}: it counts whole workers that "
        f"exited cleanly, so a fraction means a lane was counted on its own"
    )
    damaged = {
        lane.directory.name: lane.skipped for lane in killed.lanes if lane.skipped
    }
    assert killed.rows_on_disk == killed.summary.rows_written, (
        f"summary claims {killed.summary.rows_written} rows, the shards hold "
        f"{killed.rows_on_disk} (unreadable shards skipped by both sides: "
        f"{damaged})"
    )
    for name, entries in damaged.items():
        # One is the documented residue of a SIGKILL: the shard the victim was
        # in the middle of sealing. More than one means a lane's writer was
        # being torn down while another still had the directory open.
        assert len(entries) <= 1, (
            f"{name} holds {len(entries)} unreadable shards, want at most the "
            f"one the kill interrupted: {entries}"
        )
    empty = [lane.directory.name for lane in killed.lanes if lane.rows == 0]
    assert not empty, f"lane directories reported but holding no rows: {empty}"


def test_the_killed_run_left_no_segment_and_no_process_behind(killed: Run) -> None:
    """Design D2/F9 with twice the segments to leak.

    A SIGKILLed worker is where segment ownership goes wrong: Python 3.12 has no
    ``SharedMemory(track=...)`` (bpo-38119), so the orchestrator's ``finally`` is
    the only backstop, and interleaving multiplies what it has to sweep by
    ``games_in_flight`` -- for both the dead incarnation and the live one.  A
    survivor here is pressure on ``/dev/shm`` that the next run inherits.
    """
    assert not killed.shm_leaked, (
        f"segments still in /dev/shm after the kill and restart: "
        f"{sorted(killed.shm_leaked)}"
    )
    assert not killed.children_leaked, (
        f"child processes surviving the run: {killed.children_leaked}"
    )


def test_one_lane_in_flight_is_the_run_the_old_suite_pins(single: Run) -> None:
    """The equivalence guard: at the default the run is observably unchanged.

    ``games_in_flight`` defaults to 1, so every existing consumer -- the P0a
    end-to-end loop, the restart suite, an operator's config file that predates
    the flag -- must see precisely what it saw before: one writer directory per
    worker, named the way design D10 names it, holding that worker's games.  The
    assertions here are ``test_e2e_loop``'s, restated against a run made through
    the interleaving code path, which is the only way to notice that the
    one-lane case now goes through a lane loop that is subtly not the old one.
    """
    # Read off the field's declared default rather than off an instance: this
    # module must stay importable (and type-checkable) on a base where the
    # attribute does not exist at all, which is the whole point of the skip gate
    # at the top -- a `SelfPlayConfig(...).games_in_flight` here would be a
    # static error there rather than a skip.
    default = [
        field.default
        for field in dataclass_fields(SelfPlayConfig)
        if field.name == "games_in_flight"
    ]
    assert default == [1], (
        f"games_in_flight declares default {default}, want 1: every config that "
        f"predates the flag must keep meaning exactly what it meant"
    )

    assert single.summary.restarts == 0
    assert rig.summary_incarnations(single.summary) == [0] * _WORKERS
    assert single.summary.games_completed == _WORKERS, (
        f"one game per worker, {_WORKERS} workers; summary reports "
        f"{single.summary.games_completed}"
    )

    directories = [lane.directory for lane in single.lanes]
    assert len(directories) == _WORKERS, (
        f"one lane per worker means exactly {_WORKERS} writer directories, got "
        f"{rig.describe(directories)}"
    )
    parsed = [rig.parse_shard_dir(path) for path in directories]
    assert {worker for worker, _ in parsed} == set(range(_WORKERS))
    assert all(incarnation == 0 for _, incarnation in parsed)
    for path in directories:
        assert rig.sealed_shards(path), f"{path.name} sealed no shard"

    assert single.rows_on_disk == single.summary.rows_written
    assert not [lane.skipped for lane in single.lanes if lane.skipped]
    assert len(single.per_game_rows) == _WORKERS, (
        f"want one game per worker, the rows decode to "
        f"{sorted(single.per_game_rows)}"
    )
    assert {
        rig.worker_of(game) for game in single.per_game_rows
    } == set(range(_WORKERS))
    assert not single.shm_leaked, sorted(single.shm_leaked)
    assert not single.children_leaked, single.children_leaked


_LANE_TAG_RE: Final[str] = r"_l(\d+)$"
"""The lane component of ``worker_<k>_r<i>_l<lane>`` (``worker_shard_dir``)."""


def test_the_liveness_gate_is_asked_about_every_lane_of_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's CALL SITE passes all lane dirs, not just lane 0's (F2/M8).

    The union pinned ``_silence_reason``'s union semantics directly -- stale
    lane 0 beside a fresh lane 1 reads as alive -- but that leaves the mutation
    the review actually mounted alive: passing ``directories[:1]`` at the
    caller. Semantics and call site fail differently, and the second one is the
    one an operator feels: judged on lane 0 alone, a worker whose lane 0 has
    finished its share of the budget while lane 1 plays on looks silent, and the
    gate SIGKILLs a perfectly healthy shard and replays its whole budget. The
    rig's own gates never fire on a ~15s incarnation, so nothing in a behaviour
    run reaches the reason string; what IS reachable is *what the caller asked*,
    which is what this records.

    ``_silence_reason`` runs on the orchestrator's own thread in this process
    (see :func:`orchestrate_rig.walled`), so a monkeypatched wrapper sees every
    call the real gate made. It delegates rather than stubbing: the run must
    stay a real run, and a wrapper that returned ``None`` unconditionally would
    disable the very gate whose input it is inspecting.
    """
    from komugi.selfplay import orchestrate

    real_silence_reason = orchestrate._silence_reason
    seen: list[tuple[int, tuple[Path, ...]]] = []

    def recording(
        directories: Sequence[Path],
        worker_id: int,
        since_start_s: float,
        config: Any,
    ) -> str | None:
        """Record which worker was judged on which dirs, then judge for real."""
        seen.append((worker_id, tuple(directories)))
        return real_silence_reason(directories, worker_id, since_start_s, config)

    monkeypatch.setattr(orchestrate, "_silence_reason", recording)

    outcome = _play(
        tmp_path / "shards",
        lanes=_LANES,
        games_per_worker=_GAMES_PER_WORKER,
        run_seed=_GATE_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="lane-gate",
    )
    assert outcome.summary.restarts == 0, (
        f"the gate fired during a healthy run (restarts="
        f"{outcome.summary.restarts}); the recording below describes a run that "
        f"was already going wrong"
    )
    assert seen, (
        "the liveness gate was never consulted during the run, so this test "
        "pinned nothing: every poll found the workers already exited"
    )

    for worker_id, directories in seen:
        assert len(directories) == _LANES, (
            f"the gate was asked about {len(directories)} directory(ies) "
            f"{[path.name for path in directories]}, want one per lane "
            f"({_LANES}): judged on a subset, a worker whose other lane is "
            f"still writing reads as silent and its healthy shard is killed"
        )
        tags = [re.search(_LANE_TAG_RE, path.name) for path in directories]
        untagged = [
            path.name for path, tag in zip(directories, tags, strict=True)
            if tag is None
        ]
        assert not untagged, (
            f"lane directories without an _l<lane> tag: {untagged}; the gate "
            f"cannot be shown to cover both lanes if the names do not say which"
        )
        lanes_seen = {int(tag.group(1)) for tag in tags if tag is not None}
        assert lanes_seen == set(range(_LANES)), (
            f"the gate saw lanes {sorted(lanes_seen)} of "
            f"{[path.name for path in directories]}, want every lane "
            f"{list(range(_LANES))}"
        )
        owners = {re.sub(_LANE_TAG_RE, "", path.name) for path in directories}
        assert len(owners) == 1, (
            f"one call must judge ONE worker's lanes; these span {sorted(owners)}"
        )
        owner = next(iter(owners))
        assert owner.startswith(f"worker_{worker_id}_r"), (
            f"the gate judged worker {worker_id} by {owner}'s directories: a "
            f"wedged worker whose NEIGHBOR stays fresh would never be "
            f"restarted (impl review R2 -- mutant M6, incarnation.dirs[0])"
        )


def test_a_lane_clients_diagnostics_know_the_lane_layout(tmp_path: Path) -> None:
    """The lane path passes ``lanes`` to its client (impl review R1).

    ``slot_label`` and ``SlotClient(lanes=...)`` existed but no production
    caller passed the geometry, so a real two-lane run's timeouts said
    ``slot 3`` where an operator needed ``worker 1 lane 1``. The wiring now
    lives in :func:`komugi.selfplay.worker._lane_client`; this pins that the
    client it builds labels itself with both identities. A tiny real segment
    is created because attaching one is the client's first act.
    """
    from komugi.selfplay import inference as inf
    from komugi.selfplay import worker as w

    config = inf.InferenceConfig(
        in_planes=4,
        num_actions=16,
        max_batch=4,
        batch_timeout_ms=0.0,
        request_timeout_s=1.0,
    )
    slots = inf.create_slots(config)
    try:
        spec = w.LaneSpec(
            lane=1,
            slot_id=3,
            shm_name=slots.shm.name,
            shard_dir=str(tmp_path / "worker_1_r0_l1"),
            games=1,
        )
        client = w._lane_client(
            spec, SelfPlayConfig(workers=2, process_batch_size=1, games_in_flight=2), config
        )
        try:
            assert client.label == "worker 1 lane 1 (slot 3)", (
                f"lane client labels itself {client.label!r}: the lane path "
                f"did not hand its geometry to the client, so incident "
                f"diagnostics fall back to the bare slot form"
            )
        finally:
            client.close()
    finally:
        slots.shm.close()
        slots.shm.unlink()
