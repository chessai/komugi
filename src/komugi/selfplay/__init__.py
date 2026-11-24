"""Self-play: the loop that turns a network into training data.

KOM-021a lands the first piece, :mod:`komugi.selfplay.inference` -- the batching
inference seam of ``docs/DESIGN.html`` §7.3, designed in
``docs/design/kom-021-selfplay.md``. KOM-022 adds the three modules that turn
that seam into data: :mod:`komugi.selfplay.adapter` (the engine->search
adapter and the evaluator it feeds), :mod:`komugi.selfplay.records` (the sample
schema, the moves-left scale, KOM-005b's short-term horizons, the seed
derivation and ``ConcatSource``) and
:mod:`komugi.selfplay.worker` (the dual-mode game loop). KOM-023 closes the
loop with :mod:`komugi.selfplay.orchestrate` -- the process supervisor that
owns every segment's lifetime, restarts a whole shard when a worker dies, and
replays one game in-process for the bit-reproducibility claim of design D11 --
and :mod:`komugi.selfplay.memgrid`, the design F12 memory grid that is the
evidence the node-pool deferral rests on.

Everything the seam exposes is re-exported here, because "what crosses the
process boundary" is exactly the list a reader wants to see in one place.

**The shape-check seam arms at this import.** :mod:`komugi.shapecheck` reads
``KOMUGI_TYPECHECK`` when *it* is first imported, and the modules below import
it at their own import time, so a spawned worker that inherits the environment
variable gets the ``@jaxtyped`` decorations applied during bootstrap -- design
D7, and the same ordering ``tests/test_env.py`` pins down. Setting the variable
after importing this package does nothing.
"""

from __future__ import annotations

from komugi.selfplay.adapter import (
    EvaluableState,
    GameState,
    make_evaluator,
)
from komugi.selfplay.inference import (
    BOARD_SIZE,
    PLANE_CELLS,
    WDL_SIZE,
    BatchEvaluator,
    InferenceConfig,
    InferenceTimeout,
    InProcessEvaluator,
    NetEval,
    ShardServer,
    SlotClient,
    SlotHandle,
    Slots,
    attach_slots,
    attach_untracked,
    create_slots,
    pack_request,
    read_request,
    read_response,
    serve_once,
    serve_shard,
    slot_nbytes,
    wdl_utility,
    write_response,
)
from komugi.selfplay.memgrid import (
    GridPoint,
    cgroup_is_shared,
    cgroup_memory_peak,
    measure_grid,
)
from komugi.selfplay.orchestrate import (
    OrchestratorConfig,
    RunSummary,
    count_rows,
    reregister_and_unlink,
    require_module_level_factory,
    run_selfplay,
    serial_debug,
    worker_shard_dir,
)
from komugi.selfplay.records import (
    MLH_SCALE,
    SCHEMA,
    SHORT_TERM_HORIZONS,
    ConcatSource,
    game_seed,
    move_seed,
)
from komugi.selfplay.worker import (
    SelfPlayWorker,
    WorkerConfig,
    worker_main,
)

__all__ = [
    "BOARD_SIZE",
    "MLH_SCALE",
    "PLANE_CELLS",
    "SCHEMA",
    "SHORT_TERM_HORIZONS",
    "WDL_SIZE",
    "BatchEvaluator",
    "ConcatSource",
    "EvaluableState",
    "GameState",
    "GridPoint",
    "InProcessEvaluator",
    "InferenceConfig",
    "InferenceTimeout",
    "NetEval",
    "OrchestratorConfig",
    "RunSummary",
    "SelfPlayWorker",
    "ShardServer",
    "SlotClient",
    "SlotHandle",
    "Slots",
    "WorkerConfig",
    "attach_slots",
    "attach_untracked",
    "cgroup_is_shared",
    "cgroup_memory_peak",
    "count_rows",
    "create_slots",
    "game_seed",
    "make_evaluator",
    "measure_grid",
    "move_seed",
    "pack_request",
    "read_request",
    "read_response",
    "reregister_and_unlink",
    "require_module_level_factory",
    "run_selfplay",
    "serial_debug",
    "serve_once",
    "serve_shard",
    "slot_nbytes",
    "wdl_utility",
    "worker_main",
    "worker_shard_dir",
    "write_response",
]
