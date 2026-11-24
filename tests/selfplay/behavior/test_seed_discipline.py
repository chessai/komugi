"""Seeding, replay and the one failure mode a single interpreter cannot show.

Design D11 makes three claims, and this module is each of them made
falsifiable:

* **Derivation is by ``SeedSequence``, never ``hash()``.**  ``hash()`` of
  anything but an int is salted by ``PYTHONHASHSEED``, so a run seeded through
  it is reproducible only within one interpreter -- and it fails *silently*,
  producing perfectly plausible different games.  A test that runs in one
  process cannot see this at all, so the check here spawns **two subprocesses
  with different ``PYTHONHASHSEED`` values** and compares their output byte for
  byte.  That is design F10b, and it is the only shape of test that can catch
  it.
* **One seed per MOVE, not per game.**  Reusing a game's seed every ply replays
  the identical Gumbel noise vector at every node, which systematically
  oversamples whichever action ids drew lucky variates -- a correlated
  exploration bias that looks like a policy quirk, not like a bug (F10d).
* **``incarnation`` is in the spawn key.**  A worker respawned after a SIGKILL
  must not replay the games its predecessor already wrote (design D10's fresh
  ``worker_<k>_r<incarnation>`` subdir is the storage half of the same rule).

The replay half of the module is the honest version of D11's claim: with one
in-process deterministic evaluator, two runs of the same
``(run_seed, worker_id, incarnation, game_index)`` must produce **byte-identical
rows**, every field compared; a bumped incarnation must not.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import worker_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "SelfPlayWorker"):
    pytest.skip(
        "komugi.selfplay exports no SelfPlayWorker yet (KOM-022)",
        allow_module_level=True,
    )

Row = dict[str, torch.Tensor]

_RUN_SEED = 7
_WORKER_ID = 3
_GAME_INDEX = 5

_PROBE = """
import komugi.selfplay as sp

seq = sp.game_seed(7, 3, 0, 5)
print([sp.move_seed(seq, ply) for ply in range(4)])
"""
"""What both hash-seed children print.  Deliberately the spec's own example
tuple, and deliberately re-derived from scratch in each child: the claim is
about the *derivation*, so nothing may be carried in."""


# ---------------------------------------------------------------------------
# F10b -- PYTHONHASHSEED immunity, across two interpreters
# ---------------------------------------------------------------------------


def _probe_with_hash_seed(value: str) -> str:
    """Run the probe under ``PYTHONHASHSEED=value`` and return its stdout."""
    result = rig.run_python(_PROBE, env=rig.child_env(PYTHONHASHSEED=value))
    assert result.returncode == 0, (
        f"the seed probe failed under PYTHONHASHSEED={value} "
        f"(exit {result.returncode}):\n{result.stderr}"
    )
    return result.stdout.strip()


def test_move_seeds_are_immune_to_pythonhashseed() -> None:
    """Design F10b: two interpreters, two salts, one answer.

    If any step of the derivation goes through ``hash()`` on a tuple, a string
    or a frozenset, these two lines differ and every "reproducible" claim in
    the design is void -- including the ``--serial-debug`` bit-replay this
    module tests below, which would pass inside one interpreter regardless.
    """
    first = _probe_with_hash_seed("1")
    second = _probe_with_hash_seed("2")
    assert first == second, (
        "move seeds depend on PYTHONHASHSEED -- something in the derivation "
        f"went through hash() (design D11 forbids it):\n"
        f"  PYTHONHASHSEED=1 -> {first}\n  PYTHONHASHSEED=2 -> {second}"
    )
    assert first, "the probe printed nothing"


def test_the_subprocess_seeds_match_this_interpreters() -> None:
    """The child's answer is this process's answer, salt notwithstanding."""
    here = [
        int(sp.move_seed(sp.game_seed(_RUN_SEED, _WORKER_ID, 0, _GAME_INDEX), ply))
        for ply in range(4)
    ]
    assert _probe_with_hash_seed("1") == str(here)


def test_a_child_that_cannot_import_is_a_failure_not_a_pass() -> None:
    """Guard the guard: the probe's environment really does reach komugi.

    A ``python -c`` child that dies on ``ImportError`` also prints "the same
    thing" both times.  This asserts the probe's exit status is meaningful,
    which is what makes the comparison above evidence.
    """
    result = rig.run_python(_PROBE, env=rig.child_env(PYTHONHASHSEED="1"))
    assert result.returncode == 0, result.stderr
    values = ast.literal_eval(result.stdout.strip())
    assert isinstance(values, list) and len(values) == 4
    assert all(isinstance(value, int) for value in values)


# ---------------------------------------------------------------------------
# the derivation itself
# ---------------------------------------------------------------------------


def test_game_seed_is_a_seed_sequence_keyed_by_the_designs_tuple() -> None:
    """Design D11 spells the construction out; this is that sentence."""
    seq = sp.game_seed(_RUN_SEED, _WORKER_ID, 2, _GAME_INDEX)
    assert isinstance(seq, np.random.SeedSequence)
    assert seq.entropy == _RUN_SEED
    assert tuple(seq.spawn_key) == (_WORKER_ID, 2, _GAME_INDEX), (
        f"spawn_key {tuple(seq.spawn_key)} is not "
        f"(worker_id, incarnation, game_index) = ({_WORKER_ID}, 2, {_GAME_INDEX})"
    )


def test_move_seed_is_a_pure_function_of_its_arguments() -> None:
    """Calling it twice on one ``SeedSequence`` gives the same seed twice.

    Not pedantry: a derivation built on ``SeedSequence.spawn()`` advances the
    parent's child counter, so the second call would silently return a
    different seed and replay would depend on how many times anybody asked.
    """
    seq = sp.game_seed(_RUN_SEED, _WORKER_ID, 0, _GAME_INDEX)
    first = [int(sp.move_seed(seq, ply)) for ply in range(4)]
    second = [int(sp.move_seed(seq, ply)) for ply in range(4)]
    assert first == second, (
        f"move_seed is stateful: {first} then {second} from the same "
        "SeedSequence -- derive with an explicit spawn_key, not spawn()"
    )
    fresh = sp.game_seed(_RUN_SEED, _WORKER_ID, 0, _GAME_INDEX)
    assert [int(sp.move_seed(fresh, ply)) for ply in range(4)] == first


def test_every_ply_of_a_game_draws_its_own_seed() -> None:
    """Design F10d: one seed per game replays one Gumbel vector all game."""
    seq = sp.game_seed(_RUN_SEED, _WORKER_ID, 0, _GAME_INDEX)
    seeds = [int(sp.move_seed(seq, ply)) for ply in range(64)]
    assert len(set(seeds)) == len(seeds), (
        f"only {len(set(seeds))} distinct seeds across 64 plies -- a repeated "
        "seed replays the identical root Gumbel noise and oversamples whichever "
        "action ids it favours"
    )


@pytest.mark.parametrize(
    ("axis", "position"),
    [("worker_id", 1), ("incarnation", 2), ("game_index", 3)],
)
def test_seeds_are_distinct_along_every_axis_of_the_spawn_key(
    axis: str, position: int
) -> None:
    """Moving any one coordinate moves the seed.

    ``incarnation`` is the one with teeth: without it in the key a worker
    respawned after a liveness kill replays exactly the games it already lost,
    and the run quietly trains on duplicates (design D11/D10).
    """
    del axis  # named for the failure message the parametrisation prints
    seeds: set[int] = set()
    for value in range(8):
        key = [_RUN_SEED, _WORKER_ID, 0, _GAME_INDEX]
        key[position] = value
        seq = sp.game_seed(key[0], key[1], key[2], key[3])
        seeds.update(int(sp.move_seed(seq, ply)) for ply in range(2))
    assert len(seeds) == 16, (
        f"8 values along this axis x 2 plies produced {len(seeds)} distinct "
        "seeds instead of 16 -- the coordinate is not reaching the derivation"
    )


# ---------------------------------------------------------------------------
# replay: same key, byte-identical rows
# ---------------------------------------------------------------------------


def _play(directory: Path, *, incarnation: int) -> list[Row]:
    """One game by one worker, with a deterministic in-process evaluator."""
    rig.run_worker(
        rig.StubBatchEvaluator(seed=5),
        directory,
        worker_id=_WORKER_ID,
        incarnation=incarnation,
        run_seed=_RUN_SEED,
        games=1,
    )
    return rig.read_rows(directory)


def _differences(left: list[Row], right: list[Row]) -> list[str]:
    """Every field where two row lists disagree, as human-readable strings."""
    if len(left) != len(right):
        return [f"row counts {len(left)} != {len(right)}"]
    found: list[str] = []
    for index, (row, other) in enumerate(zip(left, right)):
        if set(row) != set(other):
            found.append(f"row {index}: fields {sorted(row)} != {sorted(other)}")
            continue
        for name in sorted(row):
            if not torch.equal(row[name], other[name]):
                found.append(f"row {index}: field {name!r} differs")
    return found


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Row], list[Row]]:
    """The same worker key, played twice into two shard directories."""
    root = Path(tmp_path_factory.mktemp("replay"))
    return (
        _play(root / "first", incarnation=0),
        _play(root / "second", incarnation=0),
    )


def test_the_same_key_replays_byte_identically(
    replayed: tuple[list[Row], list[Row]],
) -> None:
    """D11's serial-debug claim: same key + same evaluator == same rows.

    Every field, not a sampled one: the observation, the policy target, the
    outcome, the KL, the provenance ints.  A single differing field means the
    run cannot be reproduced from ``(run_seed, worker_id, incarnation,
    game_index)``, which is the only handle a debugging session has.
    """
    first, second = replayed
    assert first, "the replay fixture recorded no rows"
    differences = _differences(first, second)
    assert not differences, (
        "two runs of the same (run_seed, worker_id, incarnation, game_index) "
        f"diverged in {len(differences)} place(s): {differences[:6]}"
    )


def test_a_bumped_incarnation_plays_a_different_game(
    tmp_path: Path, replayed: tuple[list[Row], list[Row]]
) -> None:
    """Restart seed collisions are what ``incarnation`` exists to kill.

    Same run seed, same worker, same game index, one incarnation later: the
    Gumbel noise is drawn from a different stream, so the game must not be a
    copy of the one before it.
    """
    baseline, _ = replayed
    respawned = _play(tmp_path / "incarnation_one", incarnation=1)
    assert respawned, "the respawned worker recorded no rows"
    assert _differences(baseline, respawned), (
        "incarnation 0 and incarnation 1 produced byte-identical games -- the "
        "incarnation is not reaching the spawn key, so a respawned worker "
        "would re-play (and re-record) its predecessor's games (design D11)"
    )


def test_a_hung_seed_probe_fails_rather_than_hangs() -> None:
    """The wall is real: ``pytest-timeout`` is not in the dev shell.

    Cheap insurance that this module's subprocess helper actually enforces its
    timeout, so a wedged child in the tests above fails the gate in bounded
    time instead of stalling ``ci/check.sh``.
    """
    with pytest.raises(subprocess.TimeoutExpired):
        rig.run_python("import time; time.sleep(30)", wall_s=1.0)
