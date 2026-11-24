"""The whole seam in one process tree: ``worker_main`` against a real server.

Everything else in this suite runs the worker in-process with a stub
evaluator, which is the right way to test perspective and seeding and nothing
else.  This module tests the thing those cannot: that ``worker_main`` is a
real, spawnable entry point (design D2 -- a spawned process can only target
something importable by name), that it wires a ``SlotClient`` to a live
``ShardServer`` over the KOM-021b doorbell segment, plays a real game through
the real engine, writes shards a reader accepts, and exits 0.

And the failure half, which matters more: **timeout is fatal** (design D2/Q1).
A worker whose server never answers must die nonzero rather than retry, block
forever, or -- worst -- carry on with a stale response.  The orchestrator's
whole recovery story (fresh segment, fresh incarnation, whole-shard restart)
rests on that exit code, so it is asserted here against a server that was never
started at all.

Both cases are joined inside an explicit wall (:func:`worker_rig.join_all`):
``pytest-timeout`` is not in the dev shell, so a hang has to fail rather than
wedge ``ci/check.sh``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import worker_rig as rig

from komugi.engine.observation import CONTROL_SIZE

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "SelfPlayWorker"):
    pytest.skip(
        "komugi.selfplay exports no SelfPlayWorker yet (KOM-022)",
        allow_module_level=True,
    )

_RUN_SEED = 31337
_WORKER_ID = 0
_TIMEOUT_WALL_S = 240.0
"""Wall for the fatal-timeout child: it has to import torch, attach, submit
once and wait out ``request_timeout_s``.  Generous, because it bounds a hang
and not a latency."""


@contextlib.contextmanager
def segment(config: Any) -> Iterator[str]:
    """Create one slot segment as the ORCHESTRATOR does, and unlink it after.

    Design D2/F9: the orchestrator creates and unlinks every segment in a
    ``finally``; the worker and the server only ever attach, through
    ``attach_untracked``.  The parent's own views are closed immediately -- it
    is not a participant in the protocol, it is the owner of the memory.
    """
    created = sp.create_slots(config)
    name = str(created.shm.name)
    created.close()
    try:
        yield name
    finally:
        try:
            leftover = SharedMemory(name=name)
        except FileNotFoundError:
            return
        leftover.close()
        leftover.unlink()


# ---------------------------------------------------------------------------
# the record schema the shards are validated against
# ---------------------------------------------------------------------------

_EXPECTED_FIELDS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("observation", "float32", (rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W)),
    ("policy_target", "float32", (rig.NUM_ACTIONS,)),
    ("wdl_target", "float32", (3,)),
    ("mlh_target", "float32", ()),
    ("policy_kl", "float32", ()),
    ("root_value", "float32", ()),
    ("search_mode", "int64", ()),
    ("to_play", "int64", ()),
    ("ply", "int64", ()),
    ("game_id", "int64", ()),
    ("sims_used", "int64", ()),
    ("plane_map_version", "int64", ()),
    ("final_control", "int64", (CONTROL_SIZE,)),
    ("action", "int64", ()),
    ("termination", "int64", ()),
    ("short_term_value", "float32", (3,)),
    ("draft_outcome", "float32", (3,)),
)


def test_the_schema_is_exactly_design_d4s_row() -> None:
    """Design D4/F4: float32 and int64 only, and no field nobody consumes.

    Field *order* is deliberately not asserted -- it is part of the
    fingerprint, so it is the records module's to fix -- but the set is, both
    ways: a missing field is a hole in training and an extra one is a schema
    change that invalidates every shard already written.
    """
    got = {
        (field.name, field.dtype, tuple(field.shape)) for field in sp.SCHEMA.fields
    }
    assert got == set(_EXPECTED_FIELDS), (
        f"schema mismatch\n  missing: {sorted(set(_EXPECTED_FIELDS) - got)}\n"
        f"  extra:   {sorted(got - set(_EXPECTED_FIELDS))}"
    )
    assert "visit_counts" not in sp.SCHEMA.names, (
        "visit_counts was rejected in design F17/Q2: (A,) int64 is ~8x the "
        "observation and policy_kl replaced it"
    )
    assert len(sp.SCHEMA.fingerprint) == 64, "the schema must fingerprint"


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_worker_main_plays_a_game_against_a_real_shard_server(
    tmp_path: Path,
) -> None:
    """One spawned worker, one spawned server, one real game, exit 0.

    The net is a tiny module built inside the server process by a module-level
    factory -- never a pickled model across the boundary (design D2) -- and it
    is KomugiNet-*shaped* rather than a KomugiNet, so this test measures the
    seam and not KOM-005a.
    """
    config = rig.inference_config(request_timeout_s=120.0)
    selfplay_config = rig.tiny_selfplay_config()
    shard_dir = tmp_path / "worker_0"
    stop_event = rig.CTX.Event()
    failures: list[str] = []
    failures_server: list[str] = []

    with segment(config) as shm_name:
        handles = [sp.SlotHandle(shm_name=shm_name, worker_id=_WORKER_ID)]
        server = rig.spawn(
            sp.serve_shard,
            rig.tiny_factory,
            config,
            handles,
            stop_event,
            name="shard-server",
        )
        worker = rig.spawn(
            sp.worker_main,
            _WORKER_ID,
            0,
            _RUN_SEED,
            1,
            shard_dir,
            shm_name,
            selfplay_config,
            config,
            name="worker-main",
        )
        try:
            failures = rig.join_all([worker], rig.SPAWN_WALL_S)
        finally:
            stop_event.set()
            failures_server.extend(rig.join_all([server], rig.JOIN_GRACE_S * 3))

    assert not failures, f"worker: {failures}"
    assert not failures_server, f"server: {failures_server}"
    assert worker.exitcode == 0, (
        f"worker_main exited {worker.exitcode}; a worker that reaches the end of "
        "its games exits 0 (nonzero is reserved for the fatal paths)"
    )
    assert server.exitcode in (0, None), f"server exited {server.exitcode}"

    rows = rig.read_rows(shard_dir)
    assert rows, f"{shard_dir} holds no samples after a completed game"
    for row in rows:
        sp.SCHEMA.validate(row)

    # The rows describe ONE game, played by THIS worker, from ply zero.
    games = rig.by_game(rows)
    assert len(games) == 1, f"games=1 produced {len(games)} game ids: {sorted(games)}"
    plies = [rig.as_int(row, "ply") for row in rows]
    assert plies == list(range(len(plies)))
    assert {rig.as_int(row, "search_mode") for row in rows} <= {0, 1}
    assert np.isfinite([rig.as_float(row, "policy_kl") for row in rows]).all()


# ---------------------------------------------------------------------------
# the fatal path
# ---------------------------------------------------------------------------


def test_worker_main_dies_nonzero_when_no_server_ever_answers(
    tmp_path: Path,
) -> None:
    """Design D2/Q1: ``InferenceTimeout`` is fatal, and fatal means the process.

    The segment exists and is perfectly healthy -- nobody is serving it.  The
    worker must submit, wait out ``request_timeout_s``, and die.  Anything else
    is a worse outcome than a crash: a retry re-rings a doorbell the server may
    still be answering (the stale-response race the fresh-segment rule deletes),
    and a swallowed timeout produces a worker that is alive, silent and
    contributing nothing while the liveness gate waits on its mtime.
    """
    config = rig.inference_config(request_timeout_s=2.0)
    selfplay_config = rig.tiny_selfplay_config()
    shard_dir = tmp_path / "orphan"

    with segment(config) as shm_name:
        worker = rig.spawn(
            sp.worker_main,
            _WORKER_ID,
            0,
            _RUN_SEED,
            1,
            shard_dir,
            shm_name,
            selfplay_config,
            config,
            name="worker-orphan",
        )
        failures = rig.join_all([worker], _TIMEOUT_WALL_S)

    assert not failures, (
        f"the orphaned worker had to be killed at the wall: {failures}. "
        "request_timeout_s was 2s, so it should have died on its own long "
        "before -- a worker that blocks forever is the failure D2 calls fatal"
    )
    assert worker.exitcode not in (0, None), (
        f"the orphaned worker exited {worker.exitcode}; a timeout must be fatal "
        "and visible to the orchestrator as a nonzero exit (design D2/Q1)"
    )


def test_a_timed_out_client_is_poisoned_rather_than_retryable() -> None:
    """The in-process half of the same rule, cheap enough to assert directly.

    ``SlotClient`` refuses a second submit after a timeout.  It is what makes
    "timeout is fatal" a property of the seam rather than a convention the
    worker is trusted to follow.
    """
    config = rig.inference_config(request_timeout_s=0.05)
    obs = np.zeros((rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W), dtype=np.float32)
    legal = np.ones((rig.NUM_ACTIONS,), dtype=np.bool_)

    with segment(config) as shm_name:
        client = sp.SlotClient(shm_name, _WORKER_ID, config)
        try:
            with pytest.raises(sp.InferenceTimeout):
                client.submit(obs, legal)
            with pytest.raises(sp.InferenceTimeout):
                client.submit(obs, legal)
        finally:
            client.close()
