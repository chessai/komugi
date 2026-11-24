"""Wave-batched self-play, end to end: does a real run still mean what it meant?

``InferenceConfig.max_wave > 1`` (KOM-040 b2) lets one ticket carry a whole
*wave* of leaves: :meth:`~komugi.selfplay.inference.SlotClient.submit_wave`
packs ``R`` rows behind one doorbell,
:func:`~komugi.selfplay.adapter.make_wave_evaluator` adapts that to the search's
batched seam, and :class:`~komugi.search.gumbel.GumbelSearch` -- given such an
evaluator -- descends every surviving root action of a halving round's visit
index to its own leaf and submits all of them in one call.  The lever is the
**count** of round trips, not their width: the b1 bench found the shared batch
already synchronises the lanes.

The transport and search halves are merged; the *worker* wiring (waves on iff
``max_wave > 1``, per lane, never in ``--serial-debug``) lands beside this
module.  So this suite tests the only thing neither half can test alone: the
end-to-end contract of a real, spawned, wave-enabled run.  Four runs, each in a
module-scoped fixture because a self-play run costs real seconds and several
tests are views of one run:

``waves``
    ``workers=2, games_per_worker=1, max_wave=2``.  The plumbing case: does a
    wave-enabled run complete, account for itself, and never restart?
``narrow``
    The same ``run_seed`` and the same shape at ``max_wave=1`` -- the pre-wave
    protocol.  The equivalence case; see below.
``laned``
    ``games_in_flight=2`` *and* ``max_wave=2``: two lanes per worker, each lane
    its own segment, its own client and its own wave.  Waves and lanes are
    orthogonal features that meet in one worker process, and "orthogonal" is a
    claim about accounting that only a real run can settle.
``killed``
    A worker SIGKILLed mid-run under waves.  A wave-sized slot is a bigger
    thing to leak and a bigger thing to rebuild, and the victim dies holding a
    ticket that describes ``R`` rows rather than one.

What is pinned, and what is deliberately not
--------------------------------------------
The equivalence test compares a ``max_wave=1`` run against a ``max_wave=2`` run
at one ``run_seed`` and asserts:

* the same games, of the same length;
* **identical integer columns** -- ``search_mode``, ``to_play``, ``ply``,
  ``game_id``, ``sims_used``.  These are the columns no evaluator tolerance
  excuses: ``search_mode`` is the seeded full/fast draw, ``to_play``/``ply``
  are the engine walking a tree, ``game_id`` is a pure function of
  ``(worker_id, incarnation, game_index)``, and ``sims_used`` is the budget
  spent exactly.  A difference in any of them means the two runs played
  different games, which is precisely what wave scheduling promises not to do:
  within a halving round the survivors' subtrees are disjoint and each action's
  own visits stay in order, so reordering them is bit-identical rather than
  merely statistically equivalent (``komugi.search.gumbel``'s module docstring,
  "Wave independence: VERIFIED" in the b2 design review, and the stub golden in
  ``tests/search/unit/test_gumbel_waves.py``).
* float fields agreeing to ``1e-5`` per element -- design D3's pre-registered
  bound, the same one ``test_serial_vs_production`` uses and for the same
  reason.

**Bit-identity across the two runs is NOT asserted**, and that is a decision
rather than an omission.  Waves change *batch composition* at the server: a
wave of ``R`` leaves arrives as ``R`` rows of one forward, where the narrow run
delivered them as ``R`` separate forwards.  Design D11 promises reproducibility
through the shared-memory path only "up to evaluator tolerance" -- the KOM-104
GEMM lesson -- so pinning bytes here would pin a property the design explicitly
declines to promise and would turn a legitimate future change (a fused kernel,
a different admission order) into a red build.  Under
:class:`orchestrate_rig.TinyNet` the two should agree far inside the bound, so a
failure of the float test is a plumbing difference -- a mis-scattered wave row,
a stale slot row, an answer read at the wrong ``slot_row`` -- and not noise.

Sizing, which is not free choice
--------------------------------
``orchestrate.validate_wave_fit`` refuses ``1 < max_wave < m`` where ``m =
max(gumbel_m_full, gumbel_m_fast)``: a wave too narrow for one root's sampled
actions would split back into the round trips waves exist to delete.  The rig's
tiny config sets both ``m`` s to 2, so :data:`_MAX_WAVE` is 2 -- the smallest
width that is legal *and* actually a wave.  ``InferenceConfig`` separately
refuses ``max_wave > max_batch``, and the server's effective ceiling is
``min(max_batch, max_wave * len(clients))``, so every run below asks for
``workers * lanes * max_wave`` batch rows: enough for every live client to have
a full wave in flight at once, which is the shape the feature is for.

The skip gate
-------------
Waves only reach a worker if the worker builds a wave evaluator, and that
wiring is a concurrent ticket.  The gate is therefore a source probe on
:mod:`komugi.selfplay.worker`: no ``wave_evaluator`` in it means the wiring has
not landed, and a run with ``max_wave=2`` would then quietly take the
single-position path -- passing every assertion below while testing nothing.
A source probe is a proxy, deliberately a cheap one; the *real* proof is the
union gate, where the wiring and this module land together and these four runs
actually exercise waves.  A behavioural probe (a throwaway run inspected for a
wave marker) would cost a whole extra spawned run per session to learn what one
``in`` costs nothing to learn.
"""

from __future__ import annotations

import importlib
import inspect
import multiprocessing
import os
import re
import signal
import threading
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from komugi.train.shards import ShardReader

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

# The transport gate.  A field probe rather than a construction attempt:
# ``InferenceConfig`` is frozen and validating, so asking for ``max_wave`` on a
# base that has no such field would raise ``TypeError`` in this module body
# instead of skipping.
if "max_wave" not in {field.name for field in dataclass_fields(sp.InferenceConfig)}:
    pytest.skip(
        "InferenceConfig has no max_wave field yet (KOM-040 b2): the wave "
        "transport is not implemented on this base",
        allow_module_level=True,
    )


def _worker_source() -> str:
    """The worker module's source, or ``""`` if it cannot be read.

    ``inspect.getsource`` needs the ``.py`` on disk; an installed-as-bytecode
    komugi would raise :class:`OSError` here, and the honest answer to "can I
    see whether the wiring landed?" is then "no", which the gate below reads as
    "do not claim to have tested it".
    """
    try:
        return inspect.getsource(importlib.import_module("komugi.selfplay.worker"))
    except OSError:
        return ""


if "wave_evaluator" not in _worker_source():
    pytest.skip(
        "komugi.selfplay.worker builds no wave evaluator yet (KOM-040 b2 "
        "worker wiring): a max_wave>1 run would silently take the "
        "single-position path, so these runs would pin nothing",
        allow_module_level=True,
    )

_WORKERS: Final[int] = 2
_MAX_WAVE: Final[int] = 2
"""``max(gumbel_m_full, gumbel_m_fast)`` of the rig's tiny config -- the
narrowest width ``validate_wave_fit`` accepts above 1 (see the module docstring)."""

_LANES: Final[int] = 2
_EQUIV_GAMES: Final[int] = 1
"""One game per worker for the equivalence pair: the comparison is per game and
row-for-row, so a second game buys nothing but wall-clock."""

_LANED_GAMES: Final[int] = 2
"""Two games per worker over two lanes: one game each, so a lane that never ran
shows up as an empty directory rather than as a slow one."""

_KILL_GAMES: Final[int] = 2
"""The kill run needs a worker that has SEALED a shard and is still playing --
the killer fires on the first file under the root, so a one-game budget would
race the run to its own end."""

_WAVE_RUN_SEED: Final[int] = 40402
_LANED_RUN_SEED: Final[int] = 40403
_KILL_RUN_SEED: Final[int] = 40404

_LOGIT_TOLERANCE: Final[float] = 1e-5
"""Design D3's pre-registered per-element bound, as ``test_serial_vs_production``
pre-registers it: what crossing the shared-memory seam in a different batch is
allowed to move."""

_POLL_S: Final[float] = 0.05
"""How often the killer thread looks at the run.  Two cheap syscalls a pass."""

_KILL_DEADLINE_S: Final[float] = rig.RESTART_WALL_S / 2.0
"""How long the killer waits for something to kill: half the run's wall, so a
killer that never fires reports "the scenario never ran" as a failure with a
reason rather than as an unexplained wall expiry."""

_INCARNATION_RE: Final[str] = r"_r(\d+)"
"""The incarnation tag inside a shard directory name, matched permissively --
``orchestrate_rig.SHARD_DIR_RE``'s ``fullmatch`` rejects a lane-tagged name, and
the laned run below has those."""

_INT_FIELDS: Final[tuple[str, ...]] = rig.int_fields()
"""``search_mode``, ``to_play``, ``ply``, ``game_id``, ``sims_used`` -- read off
the schema rather than restated, so a new provenance column joins the
equivalence claim automatically."""

_SEARCH_MODE: Final[int] = _INT_FIELDS.index("search_mode")


def _incarnation_of(path: Path) -> int | None:
    """Return the incarnation tagged in a shard directory name, if any."""
    match = re.search(_INCARNATION_RE, path.name)
    return int(match.group(1)) if match is not None else None


@dataclass(frozen=True)
class Lane:
    """One writer directory, reduced to what the assertions need.

    Read once, in the fixture: a row carries a ``(128, 9, 9)`` observation and a
    37341-wide policy target, so keeping the arrays would hold hundreds of
    megabytes per lane for the sake of five integers a row.  The integer columns
    ARE kept -- they are the equivalence claim, and they are tiny.
    """

    directory: Path
    rows: int
    per_game_rows: dict[int, int]
    per_game_ints: dict[int, list[tuple[int, ...]]]
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

    @property
    def per_game_ints(self) -> dict[int, list[tuple[int, ...]]]:
        """``game_id -> the integer columns of its rows, in write order``.

        Merged over lanes, which is safe precisely because a ``game_id`` is
        owned by one lane (``test_interleaved`` pins that disjointness); a
        collision here would concatenate two games and the equivalence test
        would report it as a length mismatch.
        """
        merged: dict[int, list[tuple[int, ...]]] = {}
        for lane in self.lanes:
            for game, rows in lane.per_game_ints.items():
                merged.setdefault(game, []).extend(rows)
        return merged

    @property
    def search_modes(self) -> set[int]:
        """The distinct ``search_mode`` values the run recorded."""
        return {
            row[_SEARCH_MODE] for rows in self.per_game_ints.values() for row in rows
        }

    def directories_by_name(self) -> dict[str, Path]:
        """``directory name -> path``, for pairing two runs of one shape."""
        return {lane.directory.name: lane.directory for lane in self.lanes}


def _scan(directory: Path) -> Lane:
    """Read one writer directory and reduce it to a :class:`Lane`.

    ``retain_arrays=False`` and a streaming loop rather than ``list(reader)``,
    for the memory reason in :class:`Lane`.

    :class:`~komugi.train.shards.ShardReader` is built here rather than through
    ``orchestrate_rig.reader_for``, which asserts that *nothing* was skipped: a
    SIGKILLed worker may leave one half-written shard behind, which the
    orchestrator's own row count skips too, so the run whose whole subject is a
    hard kill must be able to read its directories at all.  What was skipped
    rides on the :class:`Lane`, and the tests expecting a clean run assert it is
    empty.
    """
    per_game: dict[int, int] = {}
    per_game_ints: dict[int, list[tuple[int, ...]]] = {}
    reader = ShardReader(directory, schema=sp.SCHEMA, retain_arrays=False)
    skipped = tuple(f"{entry.path.name}: {entry.reason}" for entry in reader.skipped)
    rows = 0
    for row in reader:
        rows += 1
        sp.SCHEMA.validate(row)
        game = rig.as_int(row, "game_id")
        per_game[game] = per_game.get(game, 0) + 1
        per_game_ints.setdefault(game, []).append(
            tuple(rig.as_int(row, name) for name in _INT_FIELDS)
        )
    return Lane(
        directory=directory,
        rows=rows,
        per_game_rows=per_game,
        per_game_ints=per_game_ints,
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

    ``multiprocessing.active_children`` rather than a ``/proc`` scan: the
    orchestrator runs on the wall's helper thread *in this process*, so its
    children are in this module's registry, and a ``/proc`` entry's ``comm`` is
    the interpreter's name for all of them alike.  The server is excluded by
    name -- its death is specified to fail the run loudly (design D2/Q4), so
    killing it would test something else.
    """
    try:
        children = multiprocessing.active_children()
    except RuntimeError:
        # The registry was mutated by the orchestrator thread mid-iteration.
        return []
    return [proc for proc in children if "server" not in proc.name.lower()]


class _WorkerKiller:
    """SIGKILL one worker process, once, after the run has produced data.

    Waiting for the first file under the shard root is what makes this a
    *mid-run* kill rather than a race with process startup: a worker that has
    sealed a shard has attached its segments, rung its doorbell with a wave and
    finished at least one game, so the kill lands on a live seam with a ticket
    plausibly in flight.  Only one process is ever killed -- design D2 restarts
    the whole shard on any worker death, so a second kill would be testing the
    restart budget instead.
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
    max_wave: int,
    lanes: int,
    games_per_worker: int,
    run_seed: int,
    wall_s: float,
    label: str,
    workers: int = _WORKERS,
    killer: _WorkerKiller | None = None,
) -> Run:
    """Run ``run_selfplay`` once under the rig's wall and read the result back.

    ``max_batch`` is ``workers * lanes * max_wave``: the server's effective
    ceiling is ``min(max_batch, max_wave * len(clients))``, so this is exactly
    "every live client may have a full wave in flight" and nothing wider.  A
    shallower batch would serialise the waves back into the round trips the
    feature exists to delete -- and, below ``max_wave``, would be refused by
    ``InferenceConfig`` outright.

    ``killer`` runs on its own daemon thread beside the run, because
    :func:`orchestrate_rig.walled` occupies the calling thread until the
    orchestrator returns: a kill scheduled from here would land after the run it
    was supposed to interrupt.  The thread is stopped in a ``finally`` so a run
    that raises cannot leave it hunting for a process in the next test's run.
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
            rig.inference_config(
                max_batch=workers * lanes * max_wave, max_wave=max_wave
            ),
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
        pid: name for pid, name in rig.child_pids().items() if pid not in children_before
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
def waves(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Two workers, one game each, one lane, ``max_wave=2``."""
    return _play(
        Path(tmp_path_factory.mktemp("waves")) / "shards",
        max_wave=_MAX_WAVE,
        lanes=1,
        games_per_worker=_EQUIV_GAMES,
        run_seed=_WAVE_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="waves",
    )


@pytest.fixture(scope="module")
def narrow(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The same shape and the same ``run_seed`` at ``max_wave=1``."""
    return _play(
        Path(tmp_path_factory.mktemp("waves-narrow")) / "shards",
        max_wave=1,
        lanes=1,
        games_per_worker=_EQUIV_GAMES,
        run_seed=_WAVE_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="narrow",
    )


@pytest.fixture(scope="module")
def laned(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Two lanes per worker AND waves: the two features in one process."""
    return _play(
        Path(tmp_path_factory.mktemp("waves-laned")) / "shards",
        max_wave=_MAX_WAVE,
        lanes=_LANES,
        games_per_worker=_LANED_GAMES,
        run_seed=_LANED_RUN_SEED,
        wall_s=rig.E2E_WALL_S,
        label="waves-laned",
    )


@pytest.fixture(scope="module")
def killed(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """A wave-enabled run with one worker SIGKILLed mid-flight."""
    return _play(
        Path(tmp_path_factory.mktemp("waves-killed")) / "shards",
        max_wave=_MAX_WAVE,
        lanes=1,
        games_per_worker=_KILL_GAMES,
        run_seed=_KILL_RUN_SEED,
        wall_s=rig.RESTART_WALL_S,
        label="waves-killed",
        killer=_WorkerKiller(),
    )


def test_a_wave_enabled_run_completes_and_accounts_for_itself(waves: Run) -> None:
    """The plumbing case: the run finishes, and its books balance.

    Everything a wave adds is a way for a run to end wrong without ending
    loudly.  A ticket whose ``row_count`` the server misreads answers the wrong
    number of rows; a client that reads its wave at ``slot_row=0`` throughout
    gets one leaf's answer ``R`` times; a wave that never fits the batch is
    skipped at every rank until the client times out and the shard restarts.
    The first two produce finite, plausible, wrong numbers and would show up
    here only as an *accounting* difference; the third shows up as a restart.
    So this asserts restarts first, then that the games asked for are the games
    played, then that ``rows_written`` is what a reader can actually read back.

    Both ``search_mode`` values are required to appear, which is what makes this
    a test of waves at *both* budgets: the rig's ``prob_fast=0.5`` over a few
    hundred moves makes their absence a bug rather than luck, and the full and
    fast searches sample the same ``m`` but spend different ``n`` -- a wave
    scheduler that only ever ran full searches would pass every other assertion.
    """
    assert waves.summary.restarts == 0, (
        f"nothing was killed or wedged, so nothing should have restarted "
        f"(restarts={waves.summary.restarts}); a wave that cannot be admitted "
        f"ends as a client timeout and a restart, which is what this catches"
    )
    assert rig.summary_incarnations(waves.summary) == [0] * _WORKERS, (
        "no restart means every worker finished on its first incarnation"
    )
    assert waves.summary.games_completed == _WORKERS * _EQUIV_GAMES, (
        f"asked for {_WORKERS} workers x {_EQUIV_GAMES} game(s) at "
        f"max_wave={_MAX_WAVE}; the summary reports "
        f"{waves.summary.games_completed} completed"
    )
    assert len(waves.per_game_rows) == _WORKERS * _EQUIV_GAMES, (
        f"the run's rows decode to {sorted(waves.per_game_rows)}, want "
        f"{_WORKERS * _EQUIV_GAMES} distinct game_ids"
    )
    damaged = {lane.directory.name: lane.skipped for lane in waves.lanes if lane.skipped}
    assert not damaged, f"nothing was killed, so every shard must be readable: {damaged}"
    empty = [lane.directory.name for lane in waves.lanes if lane.rows == 0]
    assert not empty, f"writer directories reported but holding no rows: {empty}"
    assert waves.rows_on_disk == waves.summary.rows_written, (
        f"summary claims {waves.summary.rows_written} rows, the shards hold "
        f"{waves.rows_on_disk}"
    )
    assert waves.search_modes == {0, 1}, (
        f"the run recorded search_mode(s) {sorted(waves.search_modes)}; with "
        f"prob_fast=0.5 over {waves.rows_on_disk} moves both the full (0) and "
        f"fast (1) search must have played, or the wave path was exercised at "
        f"only one budget"
    )


def test_the_wave_run_left_no_segment_and_no_process_behind(waves: Run) -> None:
    """Design D2/F9 with a wave-sized slot to leak.

    A ``max_wave=W`` segment is ``W`` times the payload of the one this rule was
    written for, so a leak costs ``W`` times as much ``/dev/shm`` pressure for
    the next run to inherit -- and Python 3.12 has no ``SharedMemory(track=...)``
    (bpo-38119), so the orchestrator's ``finally`` is the only backstop there
    is.  Asserted after a *clean* run, where there is no excuse at all.
    """
    assert not waves.shm_leaked, (
        f"segments still in /dev/shm after a clean wave run: "
        f"{sorted(waves.shm_leaked)}"
    )
    assert not waves.children_leaked, (
        f"child processes surviving the run: {waves.children_leaked}"
    )


def test_waves_and_the_narrow_protocol_play_the_same_games(
    waves: Run, narrow: Run
) -> None:
    """One ``run_seed``, two wave widths, identical integer columns.

    The claim being pinned, in the terms the module docstring states it: wave
    scheduling reorders provably independent simulations, so it must not change
    *which* games get played, how long they run, which search mode each move
    drew, or how many simulations it spent.  A difference here is not float
    noise and no tolerance excuses it -- it means the wave path perturbed the
    tree: a leaf backed up against the wrong path, an answer aligned to the
    wrong pending row, a terminal descent that entered the wave instead of being
    backed up on the spot, or a wave whose actions were not sorted the way two
    runs must sort them.

    Compared per ``game_id`` rather than per directory, so it says nothing about
    which lane or which shard file a row landed in -- only that the same games
    were played, row for row.
    """
    left, right = waves.per_game_ints, narrow.per_game_ints
    assert set(left) == set(right), (
        f"the two runs played different games: only at max_wave={_MAX_WAVE} "
        f"{sorted(set(left) - set(right))}, only at max_wave=1 "
        f"{sorted(set(right) - set(left))}"
    )
    lengths = {
        game: (len(left[game]), len(right[game]))
        for game in sorted(left)
        if len(left[game]) != len(right[game])
    }
    assert not lengths, (
        f"same run_seed, different game lengths (game_id -> (wave, narrow)): "
        f"{lengths}"
    )
    for game in sorted(left):
        for index, (wave_row, narrow_row) in enumerate(zip(left[game], right[game])):
            assert wave_row == narrow_row, (
                f"game {game} row {index} differs between max_wave={_MAX_WAVE} "
                f"and max_wave=1: {dict(zip(_INT_FIELDS, wave_row))} vs "
                f"{dict(zip(_INT_FIELDS, narrow_row))}"
            )


def test_the_float_fields_agree_to_the_preregistered_bound(
    waves: Run, narrow: Run
) -> None:
    """Design D3's ``1e-5``, elementwise, over every float field of every row.

    This is the weaker half of the equivalence claim and it is weaker on
    purpose (see the module docstring): a wave delivers ``R`` leaves as ``R``
    rows of one forward where the narrow run delivered ``R`` forwards, and
    design D11 promises the shared-memory path only "up to evaluator tolerance".
    Bit-identity would pin an unpromised property.

    What the bound still catches is everything that is not arithmetic:
    :class:`orchestrate_rig.TinyNet` is elementwise with no cross-row reduction,
    so batch composition cannot legitimately move a single number, and a failure
    is a wave row scattered to the wrong slot, a response read at the wrong
    ``slot_row``, or a policy unpacked against a mask from a different leaf.

    Streamed a row at a time through two unretained readers rather than
    materialised, because a row carries a ``(128, 9, 9)`` observation and a
    37341-wide policy target, and this walks two whole runs of them.
    """
    left = waves.directories_by_name()
    right = narrow.directories_by_name()
    assert set(left) == set(right), (
        f"the two runs produced different writer directories: "
        f"{sorted(left)} vs {sorted(right)}. Same workers, same lanes, same "
        f"incarnation -- the names are design D10's and must not depend on "
        f"max_wave"
    )
    # Row counts first: the comparison below walks two readers in lockstep and
    # would compare a prefix in silence if one run wrote fewer rows.
    wave_rows = {lane.directory.name: lane.rows for lane in waves.lanes}
    narrow_rows = {lane.directory.name: lane.rows for lane in narrow.lanes}
    assert wave_rows == narrow_rows, (
        f"the two runs wrote different row counts per directory: "
        f"max_wave={_MAX_WAVE} {wave_rows}, max_wave=1 {narrow_rows}"
    )
    for name in sorted(left):
        wave_reader = ShardReader(left[name], schema=sp.SCHEMA, retain_arrays=False)
        narrow_reader = ShardReader(right[name], schema=sp.SCHEMA, retain_arrays=False)
        worst: dict[str, tuple[float, int]] = {
            field: (0.0, -1) for field in rig.float_fields()
        }
        for index, (wave_row, narrow_row) in enumerate(zip(wave_reader, narrow_reader)):
            for field in rig.float_fields():
                delta = float(
                    np.max(
                        np.abs(
                            wave_row[field].numpy().astype(np.float64)
                            - narrow_row[field].numpy().astype(np.float64)
                        )
                    )
                )
                if delta > worst[field][0]:
                    worst[field] = (delta, index)
        over = {
            field: value
            for field, value in worst.items()
            if value[0] > _LOGIT_TOLERANCE
        }
        assert not over, (
            f"{name}: float field(s) differ between max_wave={_MAX_WAVE} and "
            f"max_wave=1 by more than design D3's bound of {_LOGIT_TOLERANCE} "
            f"(field -> (worst delta, row)): {over}"
        )


def test_waves_and_lanes_compose(laned: Run) -> None:
    """Two lanes, each submitting its own waves, in one worker process.

    Waves and lanes are orthogonal by construction -- a lane owns a segment, a
    wave is a ticket in one segment -- and the failure this catches is an
    implementation in which they are not: a worker that builds ONE wave
    evaluator and shares it across both lanes has two threads writing one slot,
    which the doorbell protocol's single-writer discipline forbids and which
    shows up here as a lane that never produced data, a game replayed in both
    lanes, or a row count that does not match the shards.  ``games_completed``
    is asserted as equality, because a lane reading ``games_per_worker`` as its
    own budget silently doubles the run.
    """
    assert laned.summary.restarts == 0, (
        f"nothing was killed or wedged, so nothing should have restarted "
        f"(restarts={laned.summary.restarts})"
    )
    assert rig.summary_incarnations(laned.summary) == [0] * _WORKERS
    assert laned.summary.games_completed == _WORKERS * _LANED_GAMES, (
        f"asked for {_WORKERS} workers x {_LANED_GAMES} games over {_LANES} "
        f"lanes at max_wave={_MAX_WAVE}; summary reports "
        f"{laned.summary.games_completed}"
    )
    assert len(laned.lanes) >= _WORKERS * _LANES, (
        f"the summary reports {len(laned.lanes)} writer directories: "
        f"{rig.describe([lane.directory for lane in laned.lanes])}. Every lane "
        f"owns one, so fewer means a lane never played"
    )
    empty = [lane.directory.name for lane in laned.lanes if lane.rows == 0]
    assert not empty, f"lane directories reported but holding no rows: {empty}"
    damaged = {lane.directory.name: lane.skipped for lane in laned.lanes if lane.skipped}
    assert not damaged, f"nothing was killed, so every shard must be readable: {damaged}"
    assert laned.rows_on_disk == laned.summary.rows_written, (
        f"summary claims {laned.summary.rows_written} rows, the shards hold "
        f"{laned.rows_on_disk}"
    )
    assert len(laned.per_game_rows) == _WORKERS * _LANED_GAMES, (
        f"the rows decode to {len(laned.per_game_rows)} distinct game_ids, want "
        f"{_WORKERS * _LANED_GAMES}: the lanes must SPLIT the budget, not each "
        f"play it"
    )
    for i, first in enumerate(laned.lanes):
        for second in laned.lanes[i + 1 :]:
            shared = first.game_ids & second.game_ids
            assert not shared, (
                f"{first.directory.name} and {second.directory.name} both hold "
                f"game_id(s) {sorted(shared)}: two lanes played one game, so "
                f"either their seed streams or their slots are shared"
            )
    assert not laned.shm_leaked, sorted(laned.shm_leaked)
    assert not laned.children_leaked, laned.children_leaked


def test_a_sigkilled_worker_under_waves_restarts_on_fresh_segments(
    killed: Run,
) -> None:
    """A worker dies holding a wave; the shard comes back whole.

    SIGKILL, not ``terminate()``: the victim dies at whatever instruction it was
    on, which under waves may be between the rows of a ``pack_wave_request`` --
    a slot whose ``row_count`` header describes rows that were never written, or
    a doorbell rung for a ticket half in place.  The lock-free protocol says
    that costs its siblings nothing (the victim holds no lock anyone needs), and
    design D2 says the whole shard restarts on fresh segments with the
    incarnation bumped, so a zombie writer can never meet its successor.

    The wave-specific failure is a restart that rebuilds segments at the OLD
    width -- a server sized for one row scanning slots a client fills with two
    -- which does not present as a crash: it presents as the run never finishing
    (caught by the run's wall), or as the restarted incarnation writing fewer
    directories than the run has workers, which is what the count below pins.
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
    tags = {lane.directory: _incarnation_of(lane.directory) for lane in killed.lanes}
    untagged = [path for path, tag in tags.items() if tag is None]
    assert not untagged, (
        f"writer directories without an _r<incarnation> tag: "
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
    assert len(fresh) == _WORKERS, (
        f"the restarted incarnation (_r{final}) produced {len(fresh)} writer "
        f"directories, want one per worker ({_WORKERS}): {rig.describe(fresh)}"
    )


def test_the_killed_wave_run_still_accounts_for_every_row_it_kept(
    killed: Run,
) -> None:
    """Correct final accounting across a restart, with waves in flight.

    A restarted shard replays its whole budget under a new incarnation, so the
    completed count is a floor rather than an equality (the orchestrator's
    documented P0a semantics) -- but it is still a multiple of
    ``games_per_worker``, because ``games_completed`` is exit-code bookkeeping
    over whole workers.  ``rows_written`` must still equal what a reader can
    read: the killed incarnation's sealed games are real data and stay, its
    half-written tail is skipped, and neither may be counted twice or dropped.
    A wave does not change that arithmetic, which is the point -- a ticket that
    carried two leaves is still at most one interrupted shard's worth of
    residue.
    """
    assert killed.summary.games_completed >= _WORKERS * _KILL_GAMES, (
        f"the run completed {killed.summary.games_completed} games; the restart "
        f"exists so that {_WORKERS * _KILL_GAMES} still get played"
    )
    assert killed.summary.games_completed % _KILL_GAMES == 0, (
        f"games_completed={killed.summary.games_completed} is not a multiple of "
        f"games_per_worker={_KILL_GAMES}: it counts whole workers that exited "
        f"cleanly"
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
        # in the middle of sealing.
        assert len(entries) <= 1, (
            f"{name} holds {len(entries)} unreadable shards, want at most the "
            f"one the kill interrupted: {entries}"
        )
    empty = [lane.directory.name for lane in killed.lanes if lane.rows == 0]
    assert not empty, f"writer directories reported but holding no rows: {empty}"


def test_the_killed_wave_run_left_no_segment_and_no_process_behind(
    killed: Run,
) -> None:
    """The leak surface at its widest: wave-sized slots, two incarnations.

    The dead incarnation's segments and the live one's are both the
    orchestrator's to unlink in its ``finally``, and under waves each of them is
    ``max_wave`` rows wide.  A survivor here is ``/dev/shm`` pressure the next
    run inherits -- and it is the one leak nothing else in the system would ever
    notice, because a leaked segment is perfectly healthy from the outside.

    Only segments this suite's own runs created are policed
    (:func:`orchestrate_rig.new_shm_survivors` diffs against a pre-run snapshot
    and re-checks with a grace), so a concurrent gate's transient segments are
    not this module's failure.
    """
    assert not killed.shm_leaked, (
        f"segments still in /dev/shm after the kill and restart: "
        f"{sorted(killed.shm_leaked)}"
    )
    assert not killed.children_leaked, (
        f"child processes surviving the run: {killed.children_leaked}"
    )
