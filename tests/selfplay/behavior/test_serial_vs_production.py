"""Design D11/D3: what ``--serial-debug`` promises, and what it does not.

Two claims, of very different strength, and this module exists to keep them
apart.

**The honest replay claim (D11).**  A game played in serial-debug mode -- one
worker, in-process evaluator -- is *bit*-reproducible from ``(run_seed,
worker_id, incarnation, game_index)``.  That is a property of the seeding
(``SeedSequence`` with a spawn key, never ``hash()``; a fresh per-move Gumbel
seed) and of nothing else, so it is asserted here at the orchestrator's own API:
``serial_debug`` twice into two fresh directories, byte for byte.

**The equivalence claim (D3), which is weaker on purpose.**  The same game
played through the *production* path goes over shared memory to a server
process, and its evaluations arrive in whatever batch the server happened to
assemble.  Design D11 promises only reproducibility "up to evaluator tolerance"
there -- the KOM-104 GEMM lesson -- and D3 fixes that tolerance as the
pre-registered per-logit bound of ``1e-5``.  So the comparison below asserts:

* the same number of rows;
* *identical* integer columns -- ``search_mode``, ``to_play``, ``ply`` -- because
  those come from the seeded mode draw and from the game tree, and a difference
  there would mean the two runs played different games, which no float tolerance
  excuses;
* float fields agreeing to ``1e-5``, elementwise.

**Bit-identity across the shared-memory path is NOT asserted**, and that is a
decision rather than an omission: it would be asserting something design D11
explicitly declines to promise, and a suite that pins an unpromised property
turns a legitimate future change (a different batch shape, a fused kernel) into
a red build.

**Scope: one game, exactly.**  Everything here is a per-row comparison of a
single paired game, which is what makes the ``1e-5`` assertion sharp -- there
is nowhere for a difference to average out.  The *distributional* half of the
same D3 gate (>= 64 seeded games per mode, compared on game length, draw rate,
mean ``|root_value|`` and top-1 agreement) is deliberately not duplicated here:
it lives with the rest of D3's pre-registration in
``test_equivalence_corpus.py``, where its thresholds sit next to half (a)'s.

The stub net makes the comparison meaningful rather than vacuous: it is
elementwise in the observation with no cross-row reduction, so a row's value
does not depend on its batch-mates, and the two paths *should* agree far inside
the bound.  A failure here is therefore a plumbing difference -- a transposed
buffer, a stale slot, a different draw_score -- not float noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import orchestrate_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "run_selfplay"):
    pytest.skip(
        "komugi.selfplay exports no run_selfplay yet (KOM-023)",
        allow_module_level=True,
    )

# KOM-236: every node here plays a whole game down both the serial and the
# shared-memory production path, which the gate sandbox cannot afford.
pytestmark = pytest.mark.sandbox_unsafe

_RUN_SEED = 771177
_WORKER_ID = 0
_INCARNATION = 0
_GAME_INDEX = 0

_LOGIT_TOLERANCE = 1e-5
"""Design D3's pre-registered per-logit bound for the serial-vs-shard comparison."""

_WALL_S = 600.0
"""Both paths play one whole game; the wall bounds a hang, not a latency."""

Rows = list[dict[str, torch.Tensor]]


@dataclass(frozen=True)
class Paired:
    """One game played both ways, plus a second serial replay of it."""

    production: Rows
    serial: Rows
    serial_again: Rows
    serial_rows_reported: int


def _serial_debug(
    shard_dir: Path, label: str, selfplay_config: Any = None
) -> tuple[int, Rows]:
    """One ``serial_debug`` game into a fresh directory, walled.

    Positional arguments in the pinned order ``(model_factory, run_seed,
    worker_id, incarnation, game_index, shard_dir, selfplay, inference)``.

    `selfplay_config` defaults to the same tiny config every other run in this
    module uses; the schedule nodes below pass one that differs **only** in the
    two KOM-036 scales, so any difference between their rows and the default
    run's is attributable to the schedule and to nothing else.
    """
    shard_dir.mkdir(parents=True)
    config = (
        rig.tiny_selfplay_config(workers=1)
        if selfplay_config is None
        else selfplay_config
    )

    def call() -> Any:
        return sp.serial_debug(
            rig.tiny_factory,
            _RUN_SEED,
            _WORKER_ID,
            _INCARNATION,
            _GAME_INDEX,
            str(shard_dir),
            config,
            rig.inference_config(max_batch=1),
        )

    outcome = rig.walled(call, _WALL_S, label)
    if outcome.error is not None:
        raise AssertionError(
            f"serial_debug raised {type(outcome.error).__name__}: {outcome.error}"
        ) from outcome.error
    # WalledRun sets exactly one of summary/error; error was raised on above,
    # so summary is present -- the assert narrows the Optional for mypy.
    assert outcome.summary is not None
    return int(outcome.summary), rig.read_rows(shard_dir)


@pytest.fixture(scope="module")
def paired(tmp_path_factory: pytest.TempPathFactory) -> Paired:
    """Play game 0 through the orchestrator, then twice through serial_debug."""
    base = Path(tmp_path_factory.mktemp("serial-vs-production"))
    root = base / "shards"
    root.mkdir()

    shm_before = rig.shm_snapshot()
    outcome = rig.walled_run_selfplay(
        rig.tiny_factory,
        rig.orchestrator_config(
            root, run_seed=_RUN_SEED, workers=1, games_per_worker=1
        ),
        rig.tiny_selfplay_config(workers=1),
        rig.inference_config(max_batch=1),
        wall_s=_WALL_S,
        label="production",
    )
    summary = outcome.unwrap()
    leaked = rig.new_shm_survivors(shm_before)
    rig.kill_stragglers()
    assert not leaked, f"the production run leaked segments: {sorted(leaked)}"

    dirs = rig.summary_dirs(summary)
    assert len(dirs) == 1, (
        f"one worker x one game should report one shard dir, got "
        f"{rig.describe(dirs)}"
    )
    production = rig.read_rows(dirs[0])

    reported, serial = _serial_debug(base / "serial", "serial")
    _, serial_again = _serial_debug(base / "serial-again", "serial-again")
    return Paired(
        production=production,
        serial=serial,
        serial_again=serial_again,
        serial_rows_reported=reported,
    )


def test_serial_debug_reports_the_rows_it_wrote(paired: Paired) -> None:
    """The ``-> int`` in the contract is the row count, not a status code."""
    assert paired.serial_rows_reported == len(paired.serial) > 0, (
        f"serial_debug returned {paired.serial_rows_reported} but wrote "
        f"{len(paired.serial)} rows"
    )


def test_both_paths_played_the_same_length_of_game(paired: Paired) -> None:
    """Same seed, same net, same engine: the same number of moves.

    First place a real divergence shows.  If the two paths disagree here, every
    per-row comparison below is meaningless, so it is asserted separately and
    with the counts in the message.
    """
    assert len(paired.production) == len(paired.serial), (
        f"production wrote {len(paired.production)} rows, serial_debug wrote "
        f"{len(paired.serial)}: the two paths played different games"
    )


def test_the_integer_columns_are_identical(paired: Paired) -> None:
    """``search_mode``, ``to_play``, ``ply`` -- and the provenance around them.

    These are the columns no evaluator tolerance can excuse.  ``search_mode``
    comes from the seeded full/fast draw on the game's own stream (design
    D6/D11), ``to_play`` and ``ply`` come from the engine walking the same tree,
    and ``game_id`` is a pure function of ``(worker_id, incarnation,
    game_index)``.  ``sims_used`` rides along because the budget is fixed by
    config and the trajectory is the same one.
    """
    for name in rig.int_fields():
        produced = [rig.as_int(row, name) for row in paired.production]
        serial = [rig.as_int(row, name) for row in paired.serial]
        first = next(
            (
                index
                for index, (left, right) in enumerate(zip(produced, serial))
                if left != right
            ),
            min(len(produced), len(serial)),
        )
        assert produced == serial, (
            f"column {name!r} differs between the production and serial paths; "
            f"first difference at row {first}"
        )


def test_the_float_fields_agree_to_the_preregistered_bound(paired: Paired) -> None:
    """Design D3's ``1e-5``, elementwise, over every float field of every row.

    Batch-of-1 through the shared-memory server versus in-process, which is the
    exact comparison D3 pre-registers.  Bit-identity is deliberately not
    asserted (see this module's docstring): the shm path is promised only to
    evaluator tolerance.
    """
    for name in rig.float_fields():
        worst = 0.0
        where = -1
        for index, (left, right) in enumerate(zip(paired.production, paired.serial)):
            delta = float(
                np.max(
                    np.abs(
                        left[name].numpy().astype(np.float64)
                        - right[name].numpy().astype(np.float64)
                    )
                )
            )
            if delta > worst:
                worst, where = delta, index
        assert worst <= _LOGIT_TOLERANCE, (
            f"field {name!r} differs by {worst} at row {where}, above design "
            f"D3's per-logit bound of {_LOGIT_TOLERANCE}"
        )


def test_serial_debug_replays_bit_identically(paired: Paired) -> None:
    """Design D11's bit-reproducibility claim, at the orchestrator's API.

    Two ``serial_debug`` calls with the same ``(run_seed, worker_id,
    incarnation, game_index)`` into two fresh directories, compared as bytes.
    This is the strong claim, and it is the one that makes a self-play bug
    debuggable at all: without it, "reproduce the game that produced this row"
    is not a thing anyone can do.

    Bytes rather than ``allclose`` on purpose -- a tolerance here would hide the
    failure mode this pins, which is a seed derived from something other than
    the four integers (a clock, a ``hash()`` under a random ``PYTHONHASHSEED``,
    or a ``SeedSequence.spawn`` whose statefulness makes the second ask differ).
    """
    assert len(paired.serial) == len(paired.serial_again), (
        f"the same serial game produced {len(paired.serial)} rows and then "
        f"{len(paired.serial_again)}"
    )
    for index, (left, right) in enumerate(zip(paired.serial, paired.serial_again)):
        assert set(left) == set(right)
        for name in sorted(left):
            assert left[name].numpy().tobytes() == right[name].numpy().tobytes(), (
                f"row {index} field {name!r} is not byte-identical across two "
                f"serial_debug replays of game {_GAME_INDEX}"
            )


# ---------------------------------------------------------------------------
# The KOM-036 schedule: determinism, and the no-extra-draw rule (§3.3)
# ---------------------------------------------------------------------------

_SCHEDULE_DRAFT_SCALE = 2.0
_SCHEDULE_FAST_SCALE = 0.5
"""§2.3/§3.2's pre-registered non-default scales.

Both nodes below run at these values on purpose: nodes #23 and #24 already
cover the *defaults*, where ``_scale_for`` returns exactly ``1.0`` and no new
code path executes at all -- which is why the design says in as many words that
neither of them can see the rule these two exist for.
"""


@pytest.fixture(scope="module")
def scheduled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Rows, Rows]:
    """Game 0, twice, under a **non-default** schedule.

    Same ``(run_seed, worker_id, incarnation, game_index)`` as the ``paired``
    fixture and the same tiny config in every other field, so the default-scale
    rows this module already has are a matched control.
    """
    base = Path(tmp_path_factory.mktemp("serial-scheduled"))
    config = rig.tiny_selfplay_config(
        workers=1,
        draft_gumbel_scale=_SCHEDULE_DRAFT_SCALE,
        fast_gumbel_scale=_SCHEDULE_FAST_SCALE,
    )
    _, first = _serial_debug(base / "scheduled", "scheduled", config)
    _, second = _serial_debug(base / "scheduled-again", "scheduled-again", config)
    return first, second


def test_two_runs_of_one_game_index_agree_under_a_schedule(
    scheduled: tuple[Rows, Rows],
) -> None:
    """#25: the replay claim survives the schedule, byte for byte.

    ``move_seed(seeds, ply)`` is unchanged by KOM-036 -- only the
    ``gumbel_scale`` field of the config built around it changes -- so two runs
    of one ``game_index`` under a non-default schedule are still bit-identical.

    **Kept as a determinism node only.** Per review M6 this is *not* evidence
    for §3.3's no-extra-draw rule: an extra ``rng.random()`` per move is itself
    perfectly deterministic, both runs would draw the same *shifted* sequence,
    and this node would stay green. The node that fires is the next one.
    """
    first, second = scheduled
    assert len(first) == len(second) > 0, (
        f"the same scheduled game produced {len(first)} rows and then "
        f"{len(second)}"
    )
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        assert set(left) == set(right)
        for name in sorted(left):
            assert left[name].numpy().tobytes() == right[name].numpy().tobytes(), (
                f"row {index} field {name!r} is not byte-identical across two "
                f"scheduled replays of game {_GAME_INDEX}"
            )


def test_the_full_fast_draw_sequence_is_scale_independent(
    paired: Paired, scheduled: tuple[Rows, Rows]
) -> None:
    """#25a: the mode column is the same sequence at both scale settings.

    The full/fast draw is one ``rng.random()`` per ply off the game's own
    stream, and the scale is consumed by ``GumbelConfig`` and never by ``rng``.
    So draw ``i`` is the *same float* in both runs whatever the scales are, and
    the ``search_mode`` column indexed by ``ply`` must match exactly. Take one
    extra draw from the game's generator on the scheduled path and the two
    sequences desynchronise from the very next ply -- which is the firing test
    §3.3's rule had none of.

    **What is asserted, and what the design over-claims.** The design states
    this as an identical mode column *and* an identical ply count. The ply count
    is not scale-independent and cannot be: a different ``gumbel_scale`` samples
    a different root candidate set and therefore plays a different move, which
    is the entire purpose of the schedule -- so the two runs are different
    games and may end at different lengths. What *is* exactly equal, and what
    the design's own mechanism argument actually supports, is the mode column on
    every ply both games reached. That is asserted here, over a prefix whose
    non-triviality is asserted first so the check cannot pass vacuously.
    """
    default = paired.serial
    with_schedule, _ = scheduled
    assert default and with_schedule

    shared = min(len(default), len(with_schedule))
    assert shared >= 8, (
        f"the two runs share only {shared} plies, which is too few for the "
        f"mode sequence to say anything; at prob_fast=0.5 eight plies is where "
        f"a shifted draw sequence stops being able to hide"
    )

    left = [rig.as_int(row, "search_mode") for row in default[:shared]]
    right = [rig.as_int(row, "search_mode") for row in with_schedule[:shared]]
    first_difference = next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        None,
    )
    assert first_difference is None, (
        f"the full/fast draw sequence differs at ply {first_difference} between "
        f"scales (1.0, 1.0) and ({_SCHEDULE_DRAFT_SCALE}, {_SCHEDULE_FAST_SCALE}); "
        f"the schedule has taken a draw from the game's own generator"
    )
    # Non-vacuity: a column that is all one mode would match under any bug that
    # shifted the sequence by a constant number of draws.
    assert len(set(left)) == 2, (
        f"the shared prefix drew only mode(s) {sorted(set(left))}; with both "
        f"modes present a shifted draw sequence is visible, without them it is "
        f"not"
    )

    # The schedule really did run: at these scales the two runs are different
    # searches, so the games diverge somewhere even though the draws do not.
    assert [rig.as_int(row, "ply") for row in default[:shared]] == list(range(shared))
    assert [rig.as_int(row, "ply") for row in with_schedule[:shared]] == list(
        range(shared)
    )
