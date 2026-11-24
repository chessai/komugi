"""KOM-158: ``scripts/analysis/cap_game_values.py``.

The script is loaded **by path**, the way ``tests/analysis/test_game_stats_backfill.py``
loads its own: ``scripts/`` is not a package, is not on ``sys.path`` and is
outside ``[tool.mypy] files``, so a plain import would be a hard mypy failure.
Everything the loader returns is ``Any``; every test binds what it reads to an
explicitly annotated local so this module stays strict-clean.

**No campaign, no checkpoint, no GPU, and no real self-play game.** The three
seams that would need one are each replaced by the script's own parameter:

* the *net* by :class:`_StubEvaluator`, which answers one fixed mover-relative
  ``wdl`` for every position -- so the white-frame arithmetic is asserted
  against a number chosen here rather than against whatever a net happened to
  believe;
* the *start mode* by :class:`_AdvancedGame`, the ``_START_MODE`` class-attribute
  idiom ``tests/selfplay/unit/test_min_placements.py`` documents: ``Game``
  follows ``GUNGI_MODE``, and mutating that process-global for the rest of the
  suite is exactly what the script's ``game_factory`` seam exists to avoid;
* the *turn cap* by ``cap_plies``/``cap_reason``, because a synthetic game that
  is "capped" at 12 plies was not adjudicated by the engine's 256-ply cap and
  must not be asserted to have been.

**One exception, and it is the point of KOM-181.** ``main`` now resolves which
net ``--ckpt`` is against the journal's ``net_fingerprint`` records, and a guard
a stub can satisfy is not a guard -- so the tests on that path write **real**
checkpoints of the real :func:`~komugi.campaign.factory.new_net` and digest them
with the campaign's own :func:`~komugi.train.digest.canonical_digest`
(:func:`_checkpoint`). ``KOMUGI_CAMPAIGN_NET=small`` keeps that to a b2c16 net,
the same trade ``tests/telemetry/test_liveness.py`` makes; the *census* on those
runs is still the stub evaluator's.

The games below are **scripted**: played move by move through the real engine
with a seeded generator choosing among ``valid_moves()``, so every action id in
a synthetic shard is one this engine actually produced. A replay that agrees
with them is agreeing with the engine, which is the whole claim the script
makes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.campaign.factory import NET_ENV, SMALL_NET, new_net
from komugi.campaign.game_stats import (
    DRAFT_PLANE_BLACK,
    DRAFT_PLANE_WHITE,
    TURN_CAP_PLIES,
    game_stats,
)
from komugi.campaign.loop import journal_path, shard_root
from komugi.engine.game import Game, RepetitionRule
from komugi.engine.observation import PLANE_MAP_VERSION, observation_size
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import MLH_SCALE
from komugi.selfplay.worker import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.telemetry.journal import write_record
from komugi.telemetry.records import EMA, SELF_PLAY, NetFingerprint
from komugi.train.checkpoint import save_checkpoint
from komugi.train.digest import canonical_digest, parameter_keys
from komugi.train.shards import FieldSpec, Schema, ShardWriter

#: ``<repo>/scripts/analysis`` -- this file is ``<repo>/tests/analysis/…``.
ANALYSIS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "scripts" / "analysis"

#: The module name the script is cached under in ``sys.modules``.
SCRIPT_NAME: Final[str] = "cap_game_values"


def load(name: str) -> ModuleType:
    """Return ``scripts/analysis/<name>.py``, imported once per process."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = ANALYSIS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load an analysis script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script: Final[Any] = load(SCRIPT_NAME)

_OBS_SHAPE: Final[tuple[int, ...]] = observation_size()

_CENSUS_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("game_id", "int64"),
    ("ply", "int64"),
    ("mlh_target", "float32"),
    ("termination", "int64"),
    ("to_play", "int64"),
    ("action", "int64"),
    ("root_value", "float32"),
)
"""The seven small columns the script's first pass projects."""

_CENSUS_SCHEMA: Final[Schema] = Schema(
    fields=tuple(
        FieldSpec(name=name, dtype=dtype, shape=())
        for name, dtype in _CENSUS_FIELDS
    )
)
"""A projection tree: enough to census game shape, no observations at all.

A legitimate shape rather than damage -- :func:`read_observations` contributes
nothing from such a shard and :func:`plane_map_guard` refuses nothing, exactly
as ``game_stats`` treats a shard with no draft columns.
"""

_FULL_SCHEMA: Final[Schema] = Schema(
    fields=(
        *(
            FieldSpec(name=name, dtype=dtype, shape=())
            for name, dtype in _CENSUS_FIELDS
        ),
        FieldSpec(name="observation", dtype="float32", shape=_OBS_SHAPE),
        FieldSpec(name="plane_map_version", dtype="int64", shape=()),
    )
)
"""The census columns plus the two the guards read."""

_WDL_FIELD: Final[FieldSpec] = FieldSpec(
    name="wdl_target", dtype="float32", shape=(3,)
)
"""The label KOM-162's unweighted cross-entropy is scored against.

Appended to a schema rather than built into one, because the script reads it
**optionally** and both cases are under test: a tree that declares it reports a
CE, and a tree that does not reports ``null`` rather than a number taken against
nothing. The real schema's own field (``selfplay/records.py:93``), restated here
for the same reason the seven census fields are -- a test that imported the
writer's schema would be asserting that the script agrees with itself.
"""


def _schema_with_wdl(base: Schema) -> Schema:
    """Return `base` with :data:`_WDL_FIELD` appended."""
    return Schema(fields=(*base.fields, _WDL_FIELD))


_SIDE_TO_MOVE_PLANE: Final[int] = 114
"""``0.0`` white, ``1.0`` black (``observation.py:16``) -- read from the encoder
rather than from a private attribute, so a test asks the engine the same way the
script does."""


class _AdvancedGame(Game):
    """The real engine started in **advanced** mode.

    Set as a class attribute rather than through ``Game.set_start_mode``:
    ``Game.__init__`` reads ``type(self)._START_MODE``, so overriding it here
    changes this class and nothing else, while the setter would mutate the
    shared base for the rest of the process
    (``tests/selfplay/unit/test_min_placements.py``'s idiom, for its reason).
    """

    _START_MODE: str = "advanced"


class _IntroGame(Game):
    """The real engine started in **intro** mode -- a board past the draft.

    Named explicitly rather than reached through ``Game()`` wherever a test
    asserts on the *mode*: ``Game()`` follows ``GUNGI_MODE``, so a suite run
    under ``GUNGI_MODE=advanced`` would have the mode-mismatch test comparing
    advanced against advanced and passing vacuously.
    """

    _START_MODE: str = "intro"


def _advanced() -> Game:
    """Return a fresh advanced-mode game under the rule self-play plays under."""
    return _AdvancedGame(repetition_rule=RepetitionRule.FULL)


class _StubEvaluator:
    """A batch evaluator that answers one fixed mover-relative ``wdl``.

    Fixed on purpose: the quantity under test is the script's **framing** of the
    network's answer (the white-frame flip, the pooling window, the quantiles),
    and a stub whose answer varies would make an assertion about the framing
    indistinguishable from an assertion about the stub.
    """

    def __init__(self, wdl: tuple[float, float, float]) -> None:
        self._wdl: npt.NDArray[np.float32] = np.asarray(wdl, dtype=np.float32)
        self.batches: list[int] = []

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return the fixed ``wdl`` for every row of `obs`."""
        rows = int(obs.shape[0])
        assert legal.shape[0] == rows, "the seam carries one mask row per position"
        self.batches.append(rows)
        return NetEval(
            policy_logits=np.zeros((rows, 1), dtype=np.float32),
            wdl=np.asarray(
                np.repeat(self._wdl[None, :], rows, axis=0), dtype=np.float32
            ),
            moves_left=np.zeros((rows,), dtype=np.float32),
        )


def _script_game(
    plies: int, *, seed: int, factory: Any = _advanced
) -> tuple[list[Any], list[npt.NDArray[np.float32]], Game]:
    """Play `plies` legal moves and return the rows, their observations, the game.

    The moves are drawn with a seeded generator from the engine's own
    ``valid_moves()``, so the actions are this engine's and the game is the same
    game on every run. Play stops early if the position becomes terminal -- a
    scripted game is a real game and a real game can end.

    The observation is captured **before** each action, which is the position
    the row describes and the one a replay has to reproduce.
    """
    game = factory()
    rng = np.random.Generator(np.random.PCG64(seed))
    rows: list[Any] = []
    observations: list[npt.NDArray[np.float32]] = []
    for ply in range(plies):
        mask = np.asarray(game.valid_moves(), dtype=np.uint8)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        observed = np.asarray(game.observation(), dtype=np.float32)
        observations.append(observed)
        action = int(legal[rng.integers(legal.size)])
        rows.append(
            script.Row(
                ply=ply,
                to_play=int(round(float(observed[_SIDE_TO_MOVE_PLANE, 0, 0]))),
                action=action,
                root_value=0.25 if ply % 2 == 0 else -0.25,
            )
        )
        game.play_action(action)
    return rows, observations, game


def _record(
    rows: list[Any], *, game_id: int, total_plies: int, termination: int
) -> Any:
    """Return the script's ``GameRecord`` for `rows`."""
    return script.GameRecord(
        game_id=game_id,
        termination=termination,
        total_plies=total_plies,
        rows=tuple(rows),
    )


def _stored(
    game_id: int, rows: list[Any], observations: list[npt.NDArray[np.float32]]
) -> dict[tuple[int, int], npt.NDArray[np.float32]]:
    """Return the ``(game_id, ply) -> observation`` map the cross-check reads."""
    return {
        (game_id, row.ply): observation
        for row, observation in zip(rows, observations)
    }


def _write_rows(
    directory: Path,
    rows: list[Any],
    *,
    game_id: int,
    total_plies: int,
    termination: int,
    observations: list[npt.NDArray[np.float32]] | None = None,
    plane_map_version: int = PLANE_MAP_VERSION,
    wdl_target: tuple[float, float, float] | None = None,
) -> None:
    """Seal one game's rows into `directory`.

    ``mlh_target`` is written the way :func:`komugi.selfplay.worker.finalize_rows`
    writes it -- ``(total_plies - ply) * MLH_SCALE`` -- so the script is
    inverting the writer's arithmetic rather than agreeing with a number
    invented here.

    `wdl_target` is written on every row of the game when given, and the column
    is left out of the schema entirely when it is not -- which is the shape the
    script has to treat as a projection tree rather than as damage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    schema = _CENSUS_SCHEMA if observations is None else _FULL_SCHEMA
    if wdl_target is not None:
        schema = _schema_with_wdl(schema)
    with ShardWriter(directory, schema, samples_per_shard=64) as writer:
        for index, row in enumerate(rows):
            sample: dict[str, torch.Tensor] = {
                "game_id": torch.tensor(game_id, dtype=torch.int64),
                "ply": torch.tensor(row.ply, dtype=torch.int64),
                "mlh_target": torch.tensor(
                    float(total_plies - row.ply) * MLH_SCALE, dtype=torch.float32
                ),
                "termination": torch.tensor(termination, dtype=torch.int64),
                "to_play": torch.tensor(row.to_play, dtype=torch.int64),
                "action": torch.tensor(row.action, dtype=torch.int64),
                "root_value": torch.tensor(row.root_value, dtype=torch.float32),
            }
            if observations is not None:
                sample["observation"] = torch.from_numpy(
                    np.ascontiguousarray(observations[index], dtype=np.float32)
                )
                sample["plane_map_version"] = torch.tensor(
                    plane_map_version, dtype=torch.int64
                )
            if wdl_target is not None:
                sample["wdl_target"] = torch.tensor(
                    list(wdl_target), dtype=torch.float32
                )
            writer.append(sample)


def _flat_rows(game_id: int, plies: int, *, start: int = 0) -> list[Any]:
    """Return `plies` rows carrying an action id that is never replayed.

    For the halves of the census that need no engine -- the cap-game set, the
    row share, the seat split, the booked-game exclusion -- where a real action
    would be noise rather than evidence.
    """
    del game_id
    return [
        script.Row(ply=start + index, to_play=index % 2, action=0, root_value=0.5)
        for index in range(plies)
    ]


STEPS_PER_ITERATION: Final[int] = 300
"""The campaign's training steps an iteration, as the fixtures step their files.

Only the *arithmetic* matters here -- iteration N's producer is the EMA at
``N * STEPS_PER_ITERATION`` and the head trained on its rows is the one at
``(N + 1) * STEPS_PER_ITERATION`` -- which is the confusion review EP-01 found
and the thing these fixtures have to be able to spell.
"""


def _checkpoint(
    directory: Path, *, step: int, offset: float
) -> tuple[Path, str, int]:
    """Save a real b2c16 checkpoint and return its path, digest and scalar count.

    `offset` is added to every parameter before saving, which is how two
    fixtures become two *different* nets: :func:`~komugi.campaign.factory.new_net`
    is seeded, so two fresh ones are bit-identical and would digest the same --
    a fixture pair that could not fail the check it exists to test.

    The digest is :func:`~komugi.train.digest.canonical_digest` over
    :func:`~komugi.train.digest.parameter_keys`, i.e. what
    ``boundary.record_fingerprints`` writes into the journal, taken here on the
    state the file is written from. The script recomputes it from the file, so
    the two agreeing is a round trip and not a shared variable.
    """
    net = new_net()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(offset)
    state: dict[str, torch.Tensor] = {
        name: tensor.detach().clone() for name, tensor in net.state_dict().items()
    }
    digest = canonical_digest(state, parameter_keys(net))
    path: Path = save_checkpoint(directory, step, state, {}, {})
    return path, digest.sha256, digest.scalars


def _plant_fingerprint(
    root: Path, *, iteration: int, role: str, sha256: str, param_count: int
) -> None:
    """Append one ``net_fingerprint`` record to `root`'s journal."""
    write_record(
        journal_path(root),
        NetFingerprint(
            iteration=iteration, role=role, sha256=sha256, param_count=param_count
        ),
    )


def _plant_producer(root: Path, *, iteration: int) -> Path:
    """Write iteration `iteration`'s producer EMA and record it as ``self_play``.

    The minimum a ``main`` run needs since KOM-181: a checkpoint the journal
    names as the net that played the iteration's rows. Returns the path to pass
    as ``--ckpt``.
    """
    path, sha256, scalars = _checkpoint(
        root / "ema", step=iteration * STEPS_PER_ITERATION, offset=0.0
    )
    _plant_fingerprint(
        root, iteration=iteration, role=SELF_PLAY, sha256=sha256, param_count=scalars
    )
    return path


# ---------------------------------------------------------------------------
# the cap-game set: the same games game_stats counts
# ---------------------------------------------------------------------------


def test_the_cap_set_is_exactly_what_game_stats_counts(tmp_path: Path) -> None:
    """``cap_games`` and ``game_stats``'s ``turn_cap_games`` see one population.

    The acceptance condition KOM-151 §3.1 states first, and the reason the
    predicate is the reconstructed **length** rather than ``termination``: if
    the two censuses could disagree about which games hit the cap, neither could
    be checked against the other and the campaign's own telemetry would be
    describing a different set of games from this report.
    """
    shards = shard_root(tmp_path, 0) / "worker_0_r0"
    _write_rows(
        shards,
        _flat_rows(1, TURN_CAP_PLIES),
        game_id=1,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
    )
    _write_rows(
        shards,
        _flat_rows(2, 40),
        game_id=2,
        total_plies=40,
        termination=TERMINATION_DECISIVE,
    )
    _write_rows(
        shards,
        _flat_rows(3, TURN_CAP_PLIES),
        game_id=3,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
    )

    found: set[int] = script.cap_games(shards)
    counted: int = game_stats(shards).turn_cap_games

    assert found == {1, 3}
    assert len(found) == counted == 2


def test_a_booked_game_is_counted_rather_than_replayed() -> None:
    """First row past ply 0 => ``book_excluded``, never a replay and never a drop.

    A booked game records no row for its prefix (``worker.py:832-844``), so it
    is still a *cap game* -- it is counted in ``cap_games`` and in the row share
    -- and it is only the replay half it cannot join.
    """
    booked: Any = _record(
        _flat_rows(7, 9, start=3), game_id=7, total_plies=12, termination=TERMINATION_DRAW
    )
    assert booked.booked is True

    with pytest.raises(script.BookedGameError, match="ply 3"):
        script.replay_game(booked.rows)

    report: dict[str, Any] = script.run_census(
        [booked],
        iteration=4,
        evaluator=_StubEvaluator((0.4, 0.2, 0.4)),
        stored={},
        mode="advanced",
        cap_plies=12,
        cap_reason=None,
    )

    assert report["cap_games"] == 1
    assert report["book_excluded"] == 1
    assert report["replay_failures"] == 0
    assert report["crosscheck_ok"] is False, "nothing was compared, so nothing passed"
    assert report["cap_row_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------


def test_a_scripted_game_replays_to_the_same_position() -> None:
    """The action column replays to the same final state and the same planes.

    The claim the whole replay half rests on: ``win_state()`` and
    ``termination_reason()`` agree, and so do the four **history** planes
    (115 turn clock, 116/117 repetition, 120 progress clock) at every recorded
    row -- which are exactly the planes a replay that dropped or duplicated a
    ply would get wrong while still producing a plausible board.
    """
    rows, observations, played = _script_game(12, seed=20260911)
    assert len(rows) >= 8, "a 12-ply scripted advanced game should not end early"

    replayed: Game = script.replay_game(rows, game_factory=_advanced)

    assert np.array_equal(
        np.asarray(replayed.win_state()), np.asarray(played.win_state())
    )
    assert replayed.termination_reason() == played.termination_reason()
    final_replay = np.asarray(replayed.observation(), dtype=np.float64)
    final_played = np.asarray(played.observation(), dtype=np.float64)
    for plane in script.CROSSCHECK_PLANES:
        assert np.allclose(final_replay[plane], final_played[plane]), (
            f"plane {plane} disagrees at the final position"
        )

    result: Any = script.replay_with_positions(
        _record(rows, game_id=11, total_plies=len(rows), termination=TERMINATION_DRAW),
        _stored(11, rows, observations),
        game_factory=_advanced,
    )
    assert result.crosschecked == len(rows)
    assert result.mismatches == 0
    assert [position.ply for position in result.positions] == [
        row.ply for row in rows
    ]


def test_a_replay_that_diverges_fails_the_crosscheck() -> None:
    """The cross-check is not vacuous: a wrong stored plane is caught.

    Asserted by handing the row at ply ``k`` the observation stored for ply
    ``k + 1``, which is what a replay one ply out of step would compare against.
    """
    rows, observations, _played = _script_game(8, seed=20260912)
    assert len(rows) >= 4
    shifted = {
        (11, row.ply): observations[min(index + 1, len(observations) - 1)]
        for index, row in enumerate(rows)
    }

    result: Any = script.replay_with_positions(
        _record(rows, game_id=11, total_plies=len(rows), termination=TERMINATION_DRAW),
        shifted,
        game_factory=_advanced,
    )

    assert result.crosschecked == len(rows)
    assert result.mismatches > 0


def test_an_illegal_action_is_refused_by_its_ply() -> None:
    """A row this engine did not produce fails by name, at its own ply."""
    rows, _observations, _played = _script_game(6, seed=20260913)
    assert len(rows) >= 3, "an advanced game's first plies are draft placements"

    # An id the engine itself calls illegal at ply 2, taken from its own mask at
    # that position -- rather than "some action + 1", which can land on a legal
    # move and turn this into a test of nothing.
    at_ply_2 = _advanced()
    for row in rows[:2]:
        at_ply_2.play_action(row.action)
    illegal = int(np.flatnonzero(np.asarray(at_ply_2.valid_moves(), dtype=np.uint8) == 0)[0])

    broken = list(rows)
    broken[2] = script.Row(
        ply=broken[2].ply,
        to_play=broken[2].to_play,
        action=illegal,
        root_value=0.0,
    )

    with pytest.raises(script.IllegalActionError, match="ply 2"):
        script.replay_game(broken, game_factory=_advanced)


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def test_a_mode_mismatch_names_both_modes() -> None:
    """The refusal names the shard's mode class AND this process's mode (F7).

    ``engine/game.py:267`` defaults ``GUNGI_MODE`` to ``intro``, so an advanced
    shard replayed in a default process fails at ply 0 with an illegal action
    several hundred replays deep. The guard turns that into one sentence naming
    both sides of the disagreement.
    """
    advanced_obs = np.asarray(_advanced().observation(), dtype=np.float32)
    intro_obs = np.asarray(_IntroGame().observation(), dtype=np.float32)

    flat = advanced_obs.reshape(advanced_obs.shape[0], -1)
    assert np.any(flat[DRAFT_PLANE_WHITE] != 0.0) and np.any(
        flat[DRAFT_PLANE_BLACK] != 0.0
    ), "the premise: an advanced game starts with both sides drafting"

    with pytest.raises(script.ModeMismatch) as advanced_in_intro:
        script.mode_guard(advanced_obs, mode="intro")
    message = str(advanced_in_intro.value)
    assert "advanced" in message and "'intro'" in message

    with pytest.raises(script.ModeMismatch) as intro_in_advanced:
        script.mode_guard(intro_obs, mode="advanced")
    reverse = str(intro_in_advanced.value)
    assert "intro" in reverse and "'advanced'" in reverse

    assert script.mode_guard(advanced_obs, mode="advanced") == "advanced"
    assert script.mode_guard(intro_obs, mode="intro") == "intro"


def test_a_foreign_plane_map_is_refused_with_both_versions(tmp_path: Path) -> None:
    """KOM-002 §4's read-side half: the fingerprint cannot see this, so it is read.

    A shard written under another plane map has the same field names, dtypes and
    shapes, so nothing upstream refuses it -- and planes 115/116/117/120 would be
    compared against whatever that map put at those indices.
    """
    rows, observations, _played = _script_game(4, seed=20260914)
    shards = shard_root(tmp_path, 0) / "worker_0_r0"
    _write_rows(
        shards,
        rows,
        game_id=5,
        total_plies=len(rows),
        termination=TERMINATION_DECISIVE,
        observations=observations,
        plane_map_version=PLANE_MAP_VERSION + 1,
    )

    with pytest.raises(script.PlaneMapMismatch) as refused:
        script.read_games(shards)

    message = str(refused.value)
    assert str(PLANE_MAP_VERSION + 1) in message and str(PLANE_MAP_VERSION) in message


# ---------------------------------------------------------------------------
# the forward pass
# ---------------------------------------------------------------------------


def test_the_forward_pass_reports_white_frame_medians() -> None:
    """``d`` and ``w - l`` are reported in the white frame, from raw ``wdl``.

    The stub answers ``(0.7, 0.2, 0.1)`` **mover-relative** for every position,
    so in the white frame a white-to-move row reads ``w - l = +0.6`` and a
    black-to-move row reads ``-0.6``. The expected quantiles are computed here
    from the scripted game's own ``to_play`` column rather than assumed to
    alternate -- the draft interleaves, and an assertion that encoded a guess
    about it would be testing the engine's turn order, not the script's framing.

    ``d`` has no frame, which is why the pre-registration (review item 2) is
    stated on it: it comes back 0.2 whoever is to move.
    """
    rows, observations, _played = _script_game(10, seed=20260915)
    assert len(rows) >= 6
    record: Any = _record(
        rows, game_id=21, total_plies=len(rows), termination=TERMINATION_DRAW
    )
    evaluator = _StubEvaluator((0.7, 0.2, 0.1))

    report: dict[str, Any] = script.run_census(
        [record],
        iteration=14,
        evaluator=evaluator,
        stored=_stored(21, rows, observations),
        mode="advanced",
        cap_plies=len(rows),
        cap_reason=None,
        game_factory=_advanced,
    )

    signed = np.asarray(
        [0.6 if row.to_play == 0 else -0.6 for row in rows], dtype=np.float64
    )
    assert report["cap_games"] == 1
    assert report["crosscheck_ok"] is True
    assert report["cap_d_median"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_d_p10"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_d_p90"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_wl_median"] == pytest.approx(float(np.median(signed)), abs=1e-6)
    assert report["cap_wl_p10"] == pytest.approx(
        float(np.percentile(signed, 10.0)), abs=1e-6
    )
    assert report["cap_wl_p90"] == pytest.approx(
        float(np.percentile(signed, 90.0)), abs=1e-6
    )
    assert sum(evaluator.batches) == len(rows), "one forward per replayed position"
    # The stored scalar, kept as the secondary read: |±0.25| on every row.
    assert report["cap_root_value_abs_mean"] == pytest.approx(0.25, abs=1e-6)


def test_the_white_frame_flip_is_a_swap_and_leaves_the_draw_alone() -> None:
    """``white_frame`` swaps win and loss for a black-to-move row, and only that."""
    wdl = np.asarray([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]], dtype=np.float32)
    framed: npt.NDArray[np.float32] = script.white_frame(wdl, [0, 1])

    assert framed[0].tolist() == pytest.approx([0.7, 0.2, 0.1])
    assert framed[1].tolist() == pytest.approx([0.1, 0.2, 0.7])
    with pytest.raises(ValueError, match="one side per evaluated position"):
        script.white_frame(wdl, [0])


# ---------------------------------------------------------------------------
# the shuffle statistics
# ---------------------------------------------------------------------------


def test_progress_and_repetition_are_read_over_the_last_window() -> None:
    """F8's read: the clock's MAGNITUDE, and the repeated-position share.

    Built from positions directly, because the quantity under test is the
    windowing and not the engine: the first entries are outside the window and
    must not move either number.
    """
    def position(ply: int, clock: int, repetition: int) -> Any:
        """One replayed position carrying only the two numbers F8 reads."""
        return script.Position(
            ply=ply,
            to_play=ply % 2,
            progress_plies=clock,
            repetition=repetition,
            observation=np.zeros((1, 1, 1), dtype=np.float32),
        )

    positions = [position(ply, clock=0, repetition=1) for ply in range(10)]
    positions += [position(10 + index, clock=30, repetition=1) for index in range(2)]
    positions += [position(12 + index, clock=40, repetition=3) for index in range(2)]

    stats: Any = script.progress_stats(positions, window=4)

    assert stats.median_progress == pytest.approx(35.0)
    assert stats.repetition_share == pytest.approx(0.5)
    assert script.progress_stats([], window=4).median_progress == 0.0


def test_the_row_share_and_the_seat_split_count_what_they_say() -> None:
    """Review F5.2: cap games are a larger share of ROWS than of games."""
    long_game: Any = _record(
        _flat_rows(1, 200), game_id=1, total_plies=200, termination=TERMINATION_DRAW
    )
    short_a: Any = _record(
        _flat_rows(2, 50), game_id=2, total_plies=50, termination=TERMINATION_DECISIVE
    )
    short_b: Any = _record(
        _flat_rows(3, 50), game_id=3, total_plies=50, termination=TERMINATION_DECISIVE
    )

    share: float = script.row_share([long_game, short_a, short_b], {1})
    assert share == pytest.approx(200 / 300)

    split: dict[str, int] = script.seat_split([long_game, short_a, short_b])
    assert split == {"white": 3, "black": 0}, "every first row here is white to move"


# ---------------------------------------------------------------------------
# KOM-162: the ply window, the magnitudes, the unweighted loss, the tripwire
# ---------------------------------------------------------------------------


_WHITE_WL: Final[float] = 0.6
"""``w - l`` in the white frame under the ``(0.7, 0.2, 0.1)`` stub.

``0.7 - 0.1`` for a white-to-move row and its negation for a black-to-move one,
which is the whole arithmetic the white frame does -- stated once here so four
tests below assert against the same number rather than four copies of it.
"""


def _white_frame_wl(
    rows: list[Any], window: tuple[int, int] | None = None
) -> npt.NDArray[np.float64]:
    """Return the white-frame ``w - l`` the stub gives `rows`, in ply order.

    `window` is the inclusive ply range to keep, or ``None`` for every row.
    Computed from each row's own recorded ``to_play`` rather than from an
    assumed alternation, for the white-frame test above's reason: the draft
    interleaves, and an expectation that encoded a guess about it would be
    testing the engine's turn order rather than the script's framing.
    """
    return np.asarray(
        [
            _WHITE_WL if row.to_play == 0 else -_WHITE_WL
            for row in rows
            if window is None or window[0] <= row.ply <= window[1]
        ],
        dtype=np.float64,
    )


def test_the_ply_window_keeps_exactly_the_plies_in_range() -> None:
    """``ply_window`` fills a second buffer with the in-range plies, and only those.

    Review F1's instrument: a per-row draw weight moves the head most in the
    mid-game, where cap and decisive rows overlap, and least in the last 32
    plies where the turn-clock plane already separates them. So the window is a
    ply range chosen by the caller, asserted here against the rows the scripted
    game actually produced -- and the tail buffer the original census reads is
    asserted to be untouched beside it, because C3 and C3' are two claims and
    the window is not allowed to replace the first.
    """
    rows, observations, _played = _script_game(14, seed=20260917)
    assert len(rows) >= 10, "a 14-ply scripted advanced game should reach ply 9"
    record: Any = _record(
        rows, game_id=31, total_plies=len(rows), termination=TERMINATION_DRAW
    )

    result: Any = script.replay_with_positions(
        record,
        _stored(31, rows, observations),
        game_factory=_advanced,
        ply_window=(3, 8),
    )

    assert [position.ply for position in result.windowed] == [
        row.ply for row in rows if 3 <= row.ply <= 8
    ]
    assert [position.ply for position in result.positions] == [
        row.ply for row in rows
    ], "the tail buffer is unchanged by a window"
    assert result.crosschecked == len(rows)
    assert result.mismatches == 0

    plain: Any = script.replay_with_positions(record, {}, game_factory=_advanced)
    assert plain.windowed == (), "no window asked for, no window collected"


def test_the_window_reads_both_classes_at_the_same_plies() -> None:
    """The window pools cap positions against decisive ones still being played.

    The comparison C3' is stated on, and the reason the decisive side is
    :func:`window_decisive_sample` rather than the tail sample: a decisive game
    that **ended inside** the window contributes only the plies before it
    finished, and pooling the last moves of a lost game against a cap game's
    middlegame is the like-for-unlike read F1 warns about. Game 44 below ends at
    ply 7 and is excluded for exactly that; games 41-43 are still running at
    ply 8 and are kept.

    Every key that is not the window's is asserted **equal** to the same census
    run without one, which is the claim that this option adds a read rather than
    changing one.
    """
    cap_a_rows, cap_a_obs, _a = _script_game(14, seed=20260918)
    cap_b_rows, cap_b_obs, _b = _script_game(14, seed=20260919)
    long_rows, long_obs, _c = _script_game(12, seed=20260920)
    short_rows, short_obs, _d = _script_game(7, seed=20260921)
    assert min(len(cap_a_rows), len(cap_b_rows), len(long_rows)) >= 10
    assert len(short_rows) >= 5, "the excluded game must have rows inside 3:8"

    games: list[Any] = [
        _record(cap_a_rows, game_id=41, total_plies=14, termination=TERMINATION_DRAW),
        _record(cap_b_rows, game_id=42, total_plies=14, termination=TERMINATION_DRAW),
        _record(
            long_rows, game_id=43, total_plies=12, termination=TERMINATION_DECISIVE
        ),
        _record(
            short_rows, game_id=44, total_plies=7, termination=TERMINATION_DECISIVE
        ),
    ]
    stored: dict[tuple[int, int], npt.NDArray[np.float32]] = {
        **_stored(41, cap_a_rows, cap_a_obs),
        **_stored(42, cap_b_rows, cap_b_obs),
        **_stored(43, long_rows, long_obs),
        **_stored(44, short_rows, short_obs),
    }

    sampled: tuple[Any, ...] = script.window_decisive_sample(
        games, size=2, last=8, cap_plies=14
    )
    assert [game.game_id for game in sampled] == [43], (
        "game 44 ran out at ply 7, so it has nothing to say at ply 8"
    )

    def census(window: tuple[int, int] | None) -> dict[str, Any]:
        """Run the whole census over these four games, with or without a window."""
        result: dict[str, Any] = script.run_census(
            games,
            iteration=9,
            evaluator=_StubEvaluator((0.7, 0.2, 0.1)),
            stored=stored,
            mode="advanced",
            cap_plies=14,
            cap_reason=None,
            game_factory=_advanced,
            window=window,
        )
        return result

    report = census((3, 8))
    plain = census(None)

    assert report["window"] == "3:8"
    assert plain["window"] is None
    windowed_keys = [key for key in script.REPORT_KEYS if "_win_" in key]
    assert len(windowed_keys) == 8, "four keys a class, two classes"
    # KOM-159's go/no-go read is the ninth window-conditional key, and it is
    # named `mlh_mae_dec_win` rather than `mlh_win_mae_dec` so that the block
    # sorts together; it is listed by name here rather than caught by the
    # substring, which would also have to change `len(windowed_keys)` above and
    # so would hide a real new key behind a rename.
    window_conditional = [*windowed_keys, "mlh_mae_dec_win"]
    assert all(plain[key] is None for key in window_conditional)
    for key in script.REPORT_KEYS:
        if key not in window_conditional and key != "window":
            assert report[key] == plain[key], f"{key} is not the window's to change"

    cap_signed = np.concatenate(
        (_white_frame_wl(cap_a_rows, (3, 8)), _white_frame_wl(cap_b_rows, (3, 8)))
    )
    dec_signed = _white_frame_wl(long_rows, (3, 8))
    assert cap_signed.size == 12, "six plies (3..8) from each of the two cap games"
    assert dec_signed.size == 6
    assert report["cap_d_win_median"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_d_win_p10"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_d_win_p90"] == pytest.approx(0.2, abs=1e-6)
    assert report["dec_d_win_median"] == pytest.approx(0.2, abs=1e-6)
    assert report["cap_wl_win_median"] == pytest.approx(
        float(np.median(cap_signed)), abs=1e-6
    )
    assert report["dec_wl_win_median"] == pytest.approx(
        float(np.median(dec_signed)), abs=1e-6
    )


def test_the_magnitude_stats_are_reported_beside_the_signed_median() -> None:
    """Review F2: ``|w - l|`` and the spread, which the signed median hides.

    ``cap_wl_median`` is a **seat-bias** read -- a head that is confidently
    winning for whoever is to move has half its positions at ``+0.6`` and half
    at ``-0.6`` and a median near zero, which passes ``|median| < 0.3`` while
    being exactly the fabricated decisiveness C5 exists to catch. The stub here
    is that head: every position reads ``|w - l| = 0.6`` whatever the seat, so
    the magnitude is exactly ``0.6`` and the spread is the width between the
    tails, hand-computed from the game's own ``to_play`` column.
    """
    rows, observations, _played = _script_game(12, seed=20260922)
    assert len(rows) >= 8
    assert len({row.to_play for row in rows}) == 2, (
        "both seats appear -- white starts and the turn passes at ply 1 -- so "
        "the signed reads are two-valued and the magnitude is the finding"
    )
    record: Any = _record(
        rows, game_id=51, total_plies=len(rows), termination=TERMINATION_DRAW
    )

    report: dict[str, Any] = script.run_census(
        [record],
        iteration=14,
        evaluator=_StubEvaluator((0.7, 0.2, 0.1)),
        stored=_stored(51, rows, observations),
        mode="advanced",
        cap_plies=len(rows),
        cap_reason=None,
        game_factory=_advanced,
    )

    signed = _white_frame_wl(rows)
    assert report["cap_wl_abs_median"] == pytest.approx(_WHITE_WL, abs=1e-6)
    assert report["cap_wl_spread"] == pytest.approx(
        float(np.percentile(signed, 90.0) - np.percentile(signed, 10.0)), abs=1e-6
    )
    assert report["cap_wl_spread"] == pytest.approx(
        float(report["cap_wl_p90"]) - float(report["cap_wl_p10"]), abs=1e-6
    ), "the spread is the distance between the tails, not the tail to the median"
    assert abs(float(report["cap_wl_median"])) <= report["cap_wl_abs_median"], (
        "F2's point: the signed median cannot exceed the magnitude, and here it "
        "is the magnitude that carries the finding"
    )
    assert report["dec_wl_abs_median"] == 0.0, "no decisive game was sampled"
    assert report["dec_wl_spread"] == 0.0


def test_the_unweighted_cross_entropy_is_scored_against_the_stored_target(
    tmp_path: Path,
) -> None:
    """Review F8/M3: the like-for-like loss, read off the shard's own label.

    The journal's ``loss_wdl`` is a weighted mean, so a campaign that
    down-weights the rows labelled draw improves it by construction. This is the
    same loss with every row counted once. The arithmetic is hand-computed: the
    stub answers ``(0.5, 0.3, 0.2)`` for every position and every row carries the
    one-hot ``(1, 0, 0)``, so the cross-entropy is ``-log(0.5)`` and nothing
    else.

    The second half is the shape the column is optional for: a shard tree that
    never declared ``wdl_target`` reports ``null``, not a number scored against
    an absent label.
    """
    rows, observations, _played = _script_game(10, seed=20260923)
    assert len(rows) >= 6

    def census(shards: Path) -> dict[str, Any]:
        """Census one written tree with the fixed stub."""
        games: tuple[Any, ...] = script.read_games(shards)
        assert len(games) == 1
        result: dict[str, Any] = script.run_census(
            list(games),
            iteration=5,
            evaluator=_StubEvaluator((0.5, 0.3, 0.2)),
            stored=_stored(61, rows, observations),
            mode="advanced",
            cap_plies=len(rows),
            cap_reason=None,
            game_factory=_advanced,
        )
        assert result["cap_games"] == 1
        return result

    labelled = shard_root(tmp_path, 5) / "worker_0_r0"
    _write_rows(
        labelled,
        rows,
        game_id=61,
        total_plies=len(rows),
        termination=TERMINATION_DRAW,
        observations=observations,
        wdl_target=(1.0, 0.0, 0.0),
    )
    read_back: tuple[Any, ...] = script.read_games(labelled)
    assert all(row.wdl_target == (1.0, 0.0, 0.0) for row in read_back[0].rows)

    report = census(labelled)
    # abs=1e-6, not exact: the stub answers in float32, so 0.3 arrives as
    # 0.30000001 and the log of it is a few parts in 10^8 away from the log of
    # the decimal written here.
    assert report["cap_ce_unweighted"] == pytest.approx(-math.log(0.5), abs=1e-6)
    assert report["dec_ce_unweighted"] is None, "no decisive game was sampled"

    bare = shard_root(tmp_path, 6) / "worker_0_r0"
    _write_rows(
        bare,
        rows,
        game_id=61,
        total_plies=len(rows),
        termination=TERMINATION_DRAW,
        observations=observations,
    )
    unlabelled: tuple[Any, ...] = script.read_games(bare)
    assert all(row.wdl_target is None for row in unlabelled[0].rows)
    assert census(bare)["cap_ce_unweighted"] is None

    # The sum runs over all three components, not just the one-hot's, and an
    # unlabelled row is dropped rather than scored as a perfect prediction.
    mixed: float | None = script.cross_entropy(
        np.asarray([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], dtype=np.float32),
        [(0.0, 1.0, 0.0), None],
    )
    assert mixed == pytest.approx(-math.log(0.3), abs=1e-6)
    assert script.cross_entropy(np.zeros((0, 3), dtype=np.float32), []) is None


def test_noncap_draws_is_draws_minus_the_turn_cap_count(tmp_path: Path) -> None:
    """R5's tripwire (review F6), on the two counts ``game_stats`` already keeps.

    Campaign 4D down-weights every row labelled draw on the premise that a draw
    in this campaign is the turn cap. A draw the engine reached by repetition or
    stalemate is that premise failing, and it is invisible in ``draws`` alone --
    which counts the cap games too -- and in ``turn_cap_games`` alone. Game 2
    below is that draw: ``termination`` is draw and it ended at ply 40, nowhere
    near the 256-ply cap.

    Asserted against :func:`komugi.campaign.game_stats.game_stats` over the same
    tree rather than against a number written here, for
    :func:`test_the_cap_set_is_exactly_what_game_stats_counts`'s reason: the
    campaign's own telemetry is the instrument C10 is read on, and the two must
    be one number.
    """
    shards = shard_root(tmp_path, 7) / "worker_0_r0"
    _write_rows(
        shards,
        _flat_rows(1, TURN_CAP_PLIES),
        game_id=1,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
    )
    _write_rows(
        shards,
        _flat_rows(2, 40),
        game_id=2,
        total_plies=40,
        termination=TERMINATION_DRAW,
    )
    _write_rows(
        shards,
        _flat_rows(3, 60),
        game_id=3,
        total_plies=60,
        termination=TERMINATION_DECISIVE,
    )

    counted = game_stats(shards)
    assert counted.draws == 2
    assert counted.turn_cap_games == 1

    games: tuple[Any, ...] = script.read_games(shards)
    assert script.noncap_draws(games) == counted.draws - counted.turn_cap_games == 1

    # And on the report, where C10 reads it. Both games are booked -- their rows
    # start at ply 1 -- so nothing is replayed and the number is the population's
    # alone, which is what it is a count of.
    booked: list[Any] = [
        _record(
            _flat_rows(4, TURN_CAP_PLIES - 1, start=1),
            game_id=4,
            total_plies=TURN_CAP_PLIES,
            termination=TERMINATION_DRAW,
        ),
        _record(
            _flat_rows(5, 39, start=1),
            game_id=5,
            total_plies=40,
            termination=TERMINATION_DRAW,
        ),
    ]
    report: dict[str, Any] = script.run_census(
        booked,
        iteration=7,
        evaluator=_StubEvaluator((0.4, 0.2, 0.4)),
        stored={},
        mode="advanced",
    )
    assert report["cap_games"] == 1
    assert report["book_excluded"] == 1
    assert report["noncap_draws"] == 1


# ---------------------------------------------------------------------------
# KOM-159 T8: the moves-left calibration block
# ---------------------------------------------------------------------------


class _MlhStubEvaluator:
    """A batch evaluator whose ``moves_left`` is scripted row by row.

    :class:`_StubEvaluator` answers one fixed triple and a ``moves_left`` of
    zero, which is the right stub for the framing tests above and useless here:
    the whole subject is what the census does with a *varying* head. This one
    hands out a queued value per position, in the order the positions were
    submitted, so the expected mean absolute error can be computed by hand from
    the queue and the game's own ply column.

    Every scripted value is a multiple of ``1/512``, so the round trip through
    float32 and back out as plies is exact and the assertions are about the
    arithmetic rather than about rounding.
    """

    def __init__(
        self,
        moves_left: list[float],
        wdl: tuple[float, float, float] = (0.7, 0.2, 0.1),
    ) -> None:
        self._queue = list(moves_left)
        self._wdl: npt.NDArray[np.float32] = np.asarray(wdl, dtype=np.float32)
        self.served = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return the next `rows` scripted answers as one batch."""
        rows = int(obs.shape[0])
        assert legal.shape[0] == rows, "the seam carries one mask row per position"
        assert len(self._queue) >= rows, (
            f"the census asked for {rows} more positions than the script has; "
            f"{self.served} were already served"
        )
        taken = [self._queue.pop(0) for _ in range(rows)]
        self.served += rows
        return NetEval(
            policy_logits=np.zeros((rows, 1), dtype=np.float32),
            wdl=np.asarray(
                np.repeat(self._wdl[None, :], rows, axis=0), dtype=np.float32
            ),
            moves_left=np.asarray(taken, dtype=np.float32),
        )


def _decisive(rows: list[Any], *, game_id: int) -> Any:
    """Return a decisive ``GameRecord`` whose length is its own row count."""
    return _record(
        rows,
        game_id=game_id,
        total_plies=len(rows),
        termination=TERMINATION_DECISIVE,
    )


def test_the_mlh_sample_is_every_decisive_game_not_the_matched_one() -> None:
    """Review F3: the calibration cannot be read off the cap-matched sample.

    ``decisive_sample`` is sized to the cap count and sorted longest-first,
    which at campaign 5's iteration 14 meant **two** games. The go/no-go for the
    utility is a 40-ply threshold on a quantity with a ~68-ply per-position
    spread, so the sample has to be the class.
    """
    games: list[Any] = [
        _record(
            _flat_rows(1, 200), game_id=1, total_plies=200, termination=TERMINATION_DRAW
        ),
        _decisive(_flat_rows(2, 50), game_id=2),
        _decisive(_flat_rows(3, 40), game_id=3),
        _decisive(_flat_rows(4, 30), game_id=4),
    ]

    matched: tuple[Any, ...] = script.decisive_sample(games, size=1, cap_plies=200)
    everything: tuple[Any, ...] = script.all_decisive_games(games, cap_plies=200)

    assert [game.game_id for game in matched] == [2], "one cap game, one match"
    assert [game.game_id for game in everything] == [2, 3, 4]


def test_the_mlh_mae_is_the_hand_computed_error_on_four_rows() -> None:
    """T8's arithmetic, on a four-row decisive game and a constant head.

    The head answers ``4/512`` at every position, i.e. "four plies to go"; the
    game really has ``4, 3, 2, 1`` plies to go at its four rows. The signed
    errors are therefore ``0, 1, 2, 3``: a mean absolute error of **1.5** plies
    and a bias of **+1.5** (the head over-predicts, as a constant must on a
    game that is ending).

    Every row is inside the first ply band, so ``mlh_mae_dec_b0`` is the pooled
    number and the other three bands are ``null`` rather than zero -- the
    distinction the whole block is nullable for.
    """
    rows, _observations, _played = _script_game(4, seed=20260926)
    assert len(rows) == 4, "four legal moves should not end an advanced game"
    evaluator = _MlhStubEvaluator([4.0 / 512.0] * 4)

    calibration: Any = script.mlh_calibration(
        evaluator,
        [_decisive(rows, game_id=81)],
        game_factory=_advanced,
        cap_plies=12,
    )
    report: dict[str, Any] = script.mlh_report(calibration)

    assert evaluator.served == 4, "one forward per row of the decisive game"
    assert report["mlh_n_dec_games"] == 1
    assert report["mlh_n_dec_rows"] == 4
    assert report["mlh_mae_dec"] == pytest.approx(1.5, abs=1e-9)
    assert report["mlh_bias_dec"] == pytest.approx(1.5, abs=1e-9)
    assert report["mlh_mae_dec_b0"] == pytest.approx(1.5, abs=1e-9)
    assert report["mlh_bias_dec_b0"] == pytest.approx(1.5, abs=1e-9)
    assert report["mlh_mae_dec_b1"] is None
    assert report["mlh_mae_dec_b2"] is None
    assert report["mlh_mae_dec_b3"] is None
    assert report["mlh_mae_dec_win"] is None, "no window was asked for"

    windowed: dict[str, Any] = script.mlh_report(calibration, window=(1, 2))
    # Rows 1 and 2 have errors 1 and 2, so the window's MAE is 1.5 as well --
    # asserted because a window that silently fell back to the pooled sample
    # would also read 1.5, and the row count below is what tells them apart.
    assert windowed["mlh_mae_dec_win"] == pytest.approx(1.5, abs=1e-9)
    assert script.mlh_report(calibration, window=(3, 3))[
        "mlh_mae_dec_win"
    ] == pytest.approx(3.0, abs=1e-9)
    assert script.mlh_report(calibration, window=(90, 99))["mlh_mae_dec_win"] is None


def test_the_one_ply_consistency_is_the_hand_computed_median() -> None:
    """T8's second read: ``m^(t) - m^(t+1) - 1``, over one three-row game.

    The quantity ``f(m)`` actually depends on (review F3). The head answers 20,
    12 and 11 plies at three consecutive rows, so the two deltas are
    ``20 - 12 - 1 = 7`` and ``12 - 11 - 1 = 0``: a perfectly consistent head
    would answer one ply less each time and read ``0`` twice. The median of the
    magnitudes is **3.5** over **2** pairs.

    A head at 40-ply MAE whose sibling errors were independent would saturate
    ``f`` on noise at the pre-registered 32-ply slope, which is why this is a
    separate pre-registration from the MAE and not a corollary of it.
    """
    rows, _observations, _played = _script_game(3, seed=20260927)
    assert len(rows) == 3
    evaluator = _MlhStubEvaluator([20.0 / 512.0, 12.0 / 512.0, 11.0 / 512.0])

    calibration: Any = script.mlh_calibration(
        evaluator,
        [_decisive(rows, game_id=82)],
        game_factory=_advanced,
        cap_plies=12,
    )
    report: dict[str, Any] = script.mlh_report(calibration)

    assert report["mlh_oneply_n"] == 2, "three rows make two consecutive pairs"
    assert report["mlh_oneply_median_abs_delta"] == pytest.approx(3.5, abs=1e-9)
    assert list(calibration.deltas) == pytest.approx([7.0, 0.0], abs=1e-9)
    # The pairs are within one game and never across the boundary between two:
    # a second game of the same shape doubles the count and leaves the median.
    pair: Any = script.mlh_calibration(
        _MlhStubEvaluator([20.0 / 512.0, 12.0 / 512.0, 11.0 / 512.0] * 2),
        [_decisive(rows, game_id=82), _decisive(rows, game_id=83)],
        game_factory=_advanced,
        cap_plies=12,
    )
    assert script.mlh_report(pair)["mlh_oneply_n"] == 4


def test_the_bands_split_the_rows_by_ply() -> None:
    """The stratification is by the row's own ply, into four half-open bands."""
    assert [name for name, _low, _high in script.MLH_PLY_BANDS] == [
        "b0",
        "b1",
        "b2",
        "b3",
    ]
    assert [low for _name, low, _high in script.MLH_PLY_BANDS] == [0, 60, 160, 300]

    calibration: Any = script.MlhCalibration(
        games=1,
        ply=np.asarray([0, 59, 60, 159, 160, 299, 300, 500], dtype=np.int64),
        predicted=np.zeros(8, dtype=np.float64),
        actual=np.asarray([1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 9.0]),
        deltas=np.zeros(0, dtype=np.float64),
    )
    report: dict[str, Any] = script.mlh_report(calibration)

    assert report["mlh_mae_dec_b0"] == pytest.approx(1.0)
    assert report["mlh_mae_dec_b1"] == pytest.approx(3.0)
    assert report["mlh_mae_dec_b2"] == pytest.approx(5.0)
    assert report["mlh_mae_dec_b3"] == pytest.approx(8.0)
    assert report["mlh_bias_dec_b0"] == pytest.approx(-1.0), (
        "the bias is signed: this head under-predicts by a ply"
    )
    assert report["mlh_oneply_median_abs_delta"] is None
    assert report["mlh_oneply_n"] == 0


def test_a_census_run_emits_every_mlh_key_on_its_own_sample() -> None:
    """The block reaches the report, over the whole decisive class (F3).

    Two decisive games and one cap game: the belief reads above match ONE
    decisive game to the one cap game, and the ``mlh_*`` block reads both --
    which is exactly the difference review F3 asked for, asserted as a
    difference between two counts in one report.
    """
    cap_rows, cap_obs, _a = _script_game(12, seed=20260928)
    long_rows, long_obs, _b = _script_game(10, seed=20260929)
    short_rows, short_obs, _c = _script_game(6, seed=20260930)
    assert min(len(cap_rows), len(long_rows)) >= 8
    assert len(short_rows) >= 4

    games: list[Any] = [
        _record(
            cap_rows, game_id=91, total_plies=12, termination=TERMINATION_DRAW
        ),
        _decisive(long_rows, game_id=92),
        _decisive(short_rows, game_id=93),
    ]
    stored: dict[tuple[int, int], npt.NDArray[np.float32]] = {
        **_stored(91, cap_rows, cap_obs),
        **_stored(92, long_rows, long_obs),
        **_stored(93, short_rows, short_obs),
    }
    decisive_rows = len(long_rows) + len(short_rows)

    report: dict[str, Any] = script.run_census(
        games,
        iteration=14,
        evaluator=_StubEvaluator((0.7, 0.2, 0.1)),
        stored=stored,
        mode="advanced",
        cap_plies=12,
        cap_reason=None,
        game_factory=_advanced,
        window=(2, 5),
    )

    assert set(report) == set(script.REPORT_KEYS)
    assert report["decisive_sample"] == 1, "one cap game matches one decisive"
    assert report["mlh_n_dec_games"] == 2, "the calibration reads the class"
    assert report["mlh_n_dec_rows"] == decisive_rows
    assert report["mlh_oneply_n"] == decisive_rows - 2, "one pair lost per game"

    # The fixed stub answers moves_left = 0 for every position, so the predicted
    # length is 0 plies and the error is the actual plies-to-end, which is
    # `total - ply` and is always positive: MAE == -bias, hand-checkable.
    expected = [
        float(total - row.ply)
        for rows, total in ((long_rows, len(long_rows)), (short_rows, len(short_rows)))
        for row in rows
    ]
    assert report["mlh_mae_dec"] == pytest.approx(float(np.mean(expected)), abs=1e-9)
    assert report["mlh_bias_dec"] == pytest.approx(
        -float(np.mean(expected)), abs=1e-9
    )
    assert report["mlh_mae_dec_b0"] == report["mlh_mae_dec"], "every row is band 0"
    assert report["mlh_mae_dec_win"] is not None
    # A constant head is off by exactly one ply per step, so every delta is -1.
    assert report["mlh_oneply_median_abs_delta"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_writes_a_report_with_exactly_the_report_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON contract, over a real shard tree, with a stub net.

    The evaluator factory is the one seam ``main`` takes, so everything else on
    this path -- the shard read, the guards, the replay, the report -- is the
    real thing. ``--ckpt`` names a **real** checkpoint since KOM-181, and the
    journal names it as the iteration's producer: the identity check opens the
    file itself, so a run that could not be resolved would be refused before any
    of this.

    The shards are written from a bare ``Game()``, which follows this process's
    own ``GUNGI_MODE`` -- so the mode guard on this path compares the shard
    against the mode that wrote it and passes whatever the suite runs under. The
    *mismatch* is asserted on its own in
    :func:`test_a_mode_mismatch_names_both_modes`, where both sides are stated
    rather than inherited.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    rows, observations, _played = _script_game(10, seed=20260916, factory=Game)
    assert len(rows) >= 2, "ten random legal moves should not end a game"
    _write_rows(
        shard_root(tmp_path, 3) / "worker_0_r0",
        rows,
        game_id=42,
        total_plies=len(rows),
        termination=TERMINATION_DECISIVE,
        observations=observations,
    )
    producer = _plant_producer(tmp_path, iteration=3)
    out = tmp_path / "report" / "cap-values.json"

    status: int = script.main(
        [
            str(tmp_path),
            "3",
            "--ckpt",
            str(producer),
            "--threads",
            "1",
            "--out",
            str(out),
        ],
        evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == set(script.REPORT_KEYS) | set(script.IDENTITY_KEYS)
    assert report["iteration"] == 3
    assert report["games"] == 1
    assert report["cap_games"] == 0, "a 10-ply game is not a turn cap"
    assert report["mode"] == script.engine_mode()
    assert set(report["cap_seat_split"]) == {"white", "black"}

    printed = capsys.readouterr().out
    assert "verdict" in printed and "wrote" in printed


def test_the_window_option_is_parsed_and_reaches_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--window FIRST:LAST`` is refused when malformed and carried when not.

    The CLI half of KOM-162: the range is parsed once, at the argument, so a
    typo prints the usage line instead of failing several hundred shard reads
    later. The census below selects no cap game -- a ten-ply game is not a turn
    cap -- so every ``*_win_*`` key is ``null`` beside a ``window`` of ``2:5``,
    which is the distinction the nullable keys exist to draw: the window was
    asked for and nothing fell inside it.
    """
    assert script.parse_window("96:160") == (96, 160)
    for malformed in ("96", "a:b", "-1:5", "9:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            script.parse_window(malformed)

    monkeypatch.setenv(NET_ENV, SMALL_NET)
    rows, observations, _played = _script_game(10, seed=20260925, factory=Game)
    assert len(rows) >= 6
    _write_rows(
        shard_root(tmp_path, 2) / "worker_0_r0",
        rows,
        game_id=71,
        total_plies=len(rows),
        termination=TERMINATION_DRAW,
        observations=observations,
        wdl_target=(0.0, 1.0, 0.0),
    )
    producer = _plant_producer(tmp_path, iteration=2)
    out = tmp_path / "windowed.json"

    status: int = script.main(
        [
            str(tmp_path),
            "2",
            "--ckpt",
            str(producer),
            "--threads",
            "1",
            "--window",
            "2:5",
            "--out",
            str(out),
        ],
        evaluator_factory=lambda path, device: _StubEvaluator((0.25, 0.5, 0.25)),
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == set(script.REPORT_KEYS) | set(script.IDENTITY_KEYS)
    assert report["window"] == "2:5"
    assert report["cap_games"] == 0, "a 10-ply game is not a turn cap"
    assert all(
        report[key] is None for key in script.REPORT_KEYS if "_win_" in key
    )
    assert report["noncap_draws"] == 1, "one draw, no cap game"


def test_main_refuses_an_iteration_that_sealed_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty iteration is refused, not reported as a census of zero games.

    The checkpoint identity resolves first and cleanly, so the refusal under
    test is the empty tree's and not KOM-181's.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    (shard_root(tmp_path, 9) / "worker_0_r0").mkdir(parents=True)
    producer = _plant_producer(tmp_path, iteration=9)

    with pytest.raises(ValueError, match="no readable game rows"):
        script.main(
            [str(tmp_path), "9", "--ckpt", str(producer)],
            evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
        )


# ---------------------------------------------------------------------------
# which net produced the rows (KOM-181, review EP-01/EV-09)
# ---------------------------------------------------------------------------


def _census_tree(root: Path, *, iteration: int, seed: int) -> None:
    """Seal one ten-ply decisive game as `iteration`'s shard tree."""
    rows, observations, _played = _script_game(10, seed=seed, factory=Game)
    assert len(rows) >= 2, "ten random legal moves should not end a game"
    _write_rows(
        shard_root(root, iteration) / "worker_0_r0",
        rows,
        game_id=42,
        total_plies=len(rows),
        termination=TERMINATION_DECISIVE,
        observations=observations,
    )


def _all_keys(payload: dict[str, Any]) -> list[str]:
    """Return every key of `payload`, nested mappings included."""
    keys: list[str] = []
    for key, value in payload.items():
        keys.append(key)
        if isinstance(value, dict):
            keys.extend(_all_keys(value))
    return keys


def _two_heads(root: Path, *, iteration: int) -> tuple[Path, str, Path, str]:
    """Plant iteration `iteration`'s producer and its post-training head.

    The two files the campaign really has and the KOM-159 censuses confused: the
    EMA served BEFORE self-play (step ``N * 300``, journal ``self_play@N``) and
    the EMA published after that iteration's training (step ``(N + 1) * 300``,
    journal ``ema@N``). Distinct nets, so the check under test can fail.
    """
    producer, producer_sha, producer_count = _checkpoint(
        root / "ema", step=iteration * STEPS_PER_ITERATION, offset=0.0
    )
    later, later_sha, later_count = _checkpoint(
        root / "ema", step=(iteration + 1) * STEPS_PER_ITERATION, offset=0.25
    )
    assert producer_sha != later_sha, "the two fixtures are the same net"
    _plant_fingerprint(
        root,
        iteration=iteration,
        role=SELF_PLAY,
        sha256=producer_sha,
        param_count=producer_count,
    )
    _plant_fingerprint(
        root,
        iteration=iteration,
        role=EMA,
        sha256=later_sha,
        param_count=later_count,
    )
    return producer, producer_sha, later, later_sha


def test_the_served_producer_is_selected_by_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review EP-01: the producer passes, the head trained on the rows does not.

    The discrimination is by **parameter digest** against the journal, not by
    filename or step, which is the only version of this check that could have
    caught the KOM-159 confusion: both files are ``ema/checkpoint-*.pt`` of the
    same campaign, and the one that is wrong is wrong by one training iteration.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    producer, producer_sha, later, later_sha = _two_heads(tmp_path, iteration=14)
    journal = journal_path(tmp_path)

    identity: Any = script.checkpoint_identity(
        producer, iteration=14, role=script.SERVED_PRODUCER, journal=journal
    )
    assert identity.sha256 == producer_sha
    assert identity.journal_role == SELF_PLAY
    assert identity.step == 14 * STEPS_PER_ITERATION
    assert identity.path == producer.resolve()

    # The same file the census got wrong is a perfectly good post_training
    # checkpoint -- the role is the claim, and this one it satisfies.
    post: Any = script.checkpoint_identity(
        later, iteration=14, role=script.POST_TRAINING, journal=journal
    )
    assert (post.sha256, post.journal_role) == (later_sha, EMA)

    # ... and claiming it produced the rows is refused, with both digests named.
    status: int = script.main(
        [str(tmp_path), "14", "--ckpt", str(later), "--threads", "1"],
        evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
    )
    assert status == 2
    refusal = capsys.readouterr().err
    assert producer_sha in refusal, "the refusal must name the digest expected"
    assert later_sha in refusal, "the refusal must name the digest computed"
    assert script.SERVED_PRODUCER in refusal and SELF_PLAY in refusal


def test_later_head_prefixes_every_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--later-head`` runs the census and marks every key of what it wrote.

    The override exists because "what does a later head think of these rows" is
    a real question; the prefix exists because its answer must be impossible to
    pool with an on-role census by a reader (or a script) looking for
    ``cap_d_median``. Nested keys included -- ``cap_seat_split``'s ``white``
    would otherwise be the one key that survives the marking.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    _census_tree(tmp_path, iteration=14, seed=20260912)
    _producer, _producer_sha, later, later_sha = _two_heads(tmp_path, iteration=14)
    out = tmp_path / "laterhead.json"

    status: int = script.main(
        [
            str(tmp_path),
            "14",
            "--ckpt",
            str(later),
            "--threads",
            "1",
            "--later-head",
            "--out",
            str(out),
        ],
        evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    keys = _all_keys(report)
    assert keys, "an empty report would pass the assertion below vacuously"
    assert all(key.startswith(script.LATER_HEAD_PREFIX) for key in keys), (
        f"unprefixed keys in a later-head report: "
        f"{[key for key in keys if not key.startswith(script.LATER_HEAD_PREFIX)]}"
    )
    assert set(report) == {
        f"{script.LATER_HEAD_PREFIX}{key}"
        for key in (*script.REPORT_KEYS, *script.IDENTITY_KEYS)
    }
    # The identity still tells the truth about the net that was forwarded.
    assert report["laterhead_census_ckpt_sha256"] == later_sha
    assert report["laterhead_census_ckpt_role"] == script.SERVED_PRODUCER
    assert report["laterhead_census_ckpt_step"] == 15 * STEPS_PER_ITERATION


def test_iteration_zero_has_no_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No journal, and no ``self_play@0``, are both refusals rather than skips.

    Iteration 0 played on freshly initialised weights (``CHECKPOINT_ENV`` empty,
    so ``record_fingerprints`` writes no ``self_play`` record), which means no
    file is its producer and no ``--ckpt`` can be checked. Passing the census
    anyway is exactly the "nothing could tell" state KOM-181 closes, so the
    absence of evidence is the refusal -- ``--later-head`` included, since there
    is no record for it to be a later head *than*.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    ckpt, _sha, _count = _checkpoint(tmp_path / "ema", step=0, offset=0.0)

    missing_journal: int = script.main(
        [str(tmp_path), "0", "--ckpt", str(ckpt), "--threads", "1"],
        evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
    )
    assert missing_journal == 2
    assert "journal.jsonl" in capsys.readouterr().err

    # A journal that records iteration 1 still records no producer for 0.
    _plant_producer(tmp_path, iteration=1)
    both_ways: tuple[list[str], list[str]] = ([], ["--later-head"])
    for extra in both_ways:
        status: int = script.main(
            [str(tmp_path), "0", "--ckpt", str(ckpt), "--threads", "1", *extra],
            evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
        )
        assert status == 2, f"iteration 0 was censused as a producer read {extra}"
        refusal = capsys.readouterr().err
        assert "iteration 0" in refusal and SELF_PLAY in refusal

    with pytest.raises(script.MissingFingerprint, match="iteration 0"):
        script.checkpoint_identity(
            ckpt,
            iteration=0,
            role=script.SERVED_PRODUCER,
            journal=journal_path(tmp_path),
        )


def test_the_report_names_the_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """EV-09: the census JSON says which net it forwarded, and the summary too.

    The invariant the ticket states: every census names the role, path and
    parameter digest of the net it read, and that digest is the journal's for
    the declared role. Asserted against the journal record the fixture wrote,
    not against the value the script reported to itself.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    _census_tree(tmp_path, iteration=7, seed=20260913)
    producer, producer_sha, _later, _later_sha = _two_heads(tmp_path, iteration=7)
    out = tmp_path / "identity.json"

    status: int = script.main(
        [
            str(tmp_path),
            "7",
            "--ckpt",
            str(producer),
            "--threads",
            "1",
            "--out",
            str(out),
        ],
        evaluator_factory=lambda path, device: _StubEvaluator((0.4, 0.2, 0.4)),
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == set(script.REPORT_KEYS) | set(script.IDENTITY_KEYS)
    assert report["census_ckpt_role"] == script.SERVED_PRODUCER
    assert report["census_ckpt_journal_role"] == SELF_PLAY
    assert report["census_ckpt_path"] == str(producer.resolve())
    assert Path(report["census_ckpt_path"]).is_absolute()
    assert report["census_ckpt_step"] == 7 * STEPS_PER_ITERATION
    assert report["census_ckpt_sha256"] == producer_sha
    assert report["iteration"] == 7

    printed = capsys.readouterr().out
    assert producer_sha in printed, "the human summary names the digest too"
    assert script.SERVED_PRODUCER in printed
