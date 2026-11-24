"""KOM-150: ``scripts/analysis/game_stats_backfill.py``.

The script is loaded **by path**, the way ``tests/analysis/test_termination_census.py``
loads its own script: ``scripts/`` is not a package, is not on ``sys.path`` and
is outside ``[tool.mypy] files``, so a plain import would be a hard mypy
failure. Everything the loader returns is ``Any``; every test binds what it
reads to an explicitly annotated local so this module stays strict-clean.

No self-play is played here. The census is a pure function of sealed shards, so
the campaign roots below are **written**: two iterations, with games whose
lengths and terminations this module chooses, which is the only way to pin
"iteration 1 drew more than iteration 0" as a fact rather than as whatever the
random seed did that day. The rows are labelled the way
:func:`komugi.selfplay.worker.finalize_rows` labels them --
``mlh_target = (total_plies - ply) * MLH_SCALE`` -- so the script is inverting
the writer's arithmetic rather than agreeing with a number invented here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
import torch

from komugi.campaign.game_stats import TURN_CAP_PLIES
from komugi.campaign.loop import iteration_root, shard_root
from komugi.selfplay.records import MLH_SCALE
from komugi.selfplay.worker import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.train.shards import FieldSpec, Schema, ShardWriter

#: ``<repo>/scripts/analysis`` -- this file is ``<repo>/tests/analysis/…``.
ANALYSIS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "scripts" / "analysis"
)

#: The module name the script is cached under in ``sys.modules``.
SCRIPT_NAME: Final[str] = "game_stats_backfill"


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


backfill_script: Final[Any] = load(SCRIPT_NAME)

_CENSUS_SCHEMA: Final[Schema] = Schema(
    fields=tuple(
        FieldSpec(name=name, dtype=dtype, shape=())
        for name, dtype in (
            ("game_id", "int64"),
            ("ply", "int64"),
            ("mlh_target", "float32"),
            ("termination", "int64"),
        )
    )
)
"""The four columns the census reads; a shard needs no others to be censused."""


def _write_game(
    directory: Path, *, game: int, total_plies: int, termination: int
) -> None:
    """Seal one whole game's rows into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    with ShardWriter(directory, _CENSUS_SCHEMA, samples_per_shard=16) as writer:
        for ply in range(total_plies):
            writer.append(
                {
                    "game_id": torch.tensor(game, dtype=torch.int64),
                    "ply": torch.tensor(ply, dtype=torch.int64),
                    "mlh_target": torch.tensor(
                        float(total_plies - ply) * MLH_SCALE, dtype=torch.float32
                    ),
                    "termination": torch.tensor(termination, dtype=torch.int64),
                }
            )


_LABELLED_SCHEMA: Final[Schema] = Schema(
    fields=_CENSUS_SCHEMA.fields
    + (FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),)
)
"""The census four plus KOM-245's label column, for the adjudicated fixture.

The four-column schema above is deliberately kept as the *other* case: it is
what every shard written before the column existed looks like, and a backfill
runs over those roots far more often than over a labelled one.
"""


def _write_labelled_game(
    directory: Path, *, game: int, total_plies: int, termination: int, p: float
) -> None:
    """Seal one whole game's rows, with KOM-245's ``wdl_target``, into `directory`.

    The leader's rows carry ``(p, 1-p, 0)`` and the trailer's the mirror, which
    is the label :func:`komugi.selfplay.worker.finalize_rows` writes for an
    adjudicated cap game; ``p = 0.0`` writes the plain ``(0, 1, 0)`` draw an
    un-adjudicated one carries. Movers alternate by ply, which is all the
    census asks of them -- it reads ``max(w, l)`` off the game's **first** row.
    """
    directory.mkdir(parents=True, exist_ok=True)
    leader = [p, 1.0 - p, 0.0]
    trailer = [0.0, 1.0 - p, p]
    with ShardWriter(directory, _LABELLED_SCHEMA, samples_per_shard=16) as writer:
        for ply in range(total_plies):
            writer.append(
                {
                    "game_id": torch.tensor(game, dtype=torch.int64),
                    "ply": torch.tensor(ply, dtype=torch.int64),
                    "mlh_target": torch.tensor(
                        float(total_plies - ply) * MLH_SCALE, dtype=torch.float32
                    ),
                    "termination": torch.tensor(termination, dtype=torch.int64),
                    "wdl_target": torch.tensor(
                        leader if ply % 2 == 0 else trailer, dtype=torch.float32
                    ),
                }
            )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A two-iteration campaign root, written rather than played.

    Iteration 0: two decisive games, 40 and 60 plies. Iteration 1: one decisive
    60-ply game and one drawn game at the turn cap -- the spiral's first
    footprint, and the reason the two iterations must not report the same
    numbers.
    """
    _write_game(
        shard_root(tmp_path, 0) / "worker_0_r0",
        game=1,
        total_plies=40,
        termination=TERMINATION_DECISIVE,
    )
    _write_game(
        shard_root(tmp_path, 0) / "worker_1_r0",
        game=2,
        total_plies=60,
        termination=TERMINATION_DECISIVE,
    )
    _write_game(
        shard_root(tmp_path, 1) / "worker_0_r0",
        game=1_000_001,
        total_plies=60,
        termination=TERMINATION_DECISIVE,
    )
    _write_game(
        shard_root(tmp_path, 1) / "worker_1_r0",
        game=1_000_002,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
    )
    return tmp_path


@pytest.fixture
def labelled_root(tmp_path: Path) -> Path:
    """A one-iteration root whose shards carry KOM-245's ``wdl_target``.

    Iteration 0 holds two turn-capped games: one adjudicated at ``p = 0.5``
    and one plain ``(0, 1, 0)`` draw. That pair is the arithmetic the plan
    pre-registered -- one adjudicated game, a mean of 0.25 over **both** cap
    games, and half of them at or above the 0.3 threshold -- so a report that
    dropped the fields, or that took the mean over the adjudicated subset
    alone, is a different set of numbers here.
    """
    _write_labelled_game(
        shard_root(tmp_path, 0) / "worker_0_r0",
        game=1,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
        p=0.5,
    )
    _write_labelled_game(
        shard_root(tmp_path, 0) / "worker_1_r0",
        game=2,
        total_plies=TURN_CAP_PLIES,
        termination=TERMINATION_DRAW,
        p=0.0,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# discovery and the census
# ---------------------------------------------------------------------------


def test_it_finds_the_iterations_the_loop_wrote(root: Path) -> None:
    """Iterations are discovered, and only real ones are.

    A directory that merely ends in a number is not an iteration: the candidate
    is confirmed against :func:`~komugi.campaign.loop.iteration_root`, so a
    root's other furniture cannot be censused as iteration 42.
    """
    (root / "ckpt").mkdir()
    (root / "notes_42").mkdir()

    found: tuple[int, ...] = backfill_script.iterations_under(root)

    assert found == (0, 1)
    assert iteration_root(root, 0).is_dir(), "the premise: the layout is the loop's"


def test_an_empty_root_is_censused_as_nothing(tmp_path: Path) -> None:
    """A root that has not played yet reads as no iterations, not as a crash."""
    assert backfill_script.iterations_under(tmp_path) == ()
    assert backfill_script.backfill(tmp_path) == {}


def test_each_iteration_is_censused_on_its_own_shards(root: Path) -> None:
    """The two iterations report the two different shapes they hold.

    The draw is in iteration 1 alone, so a census that pooled the root's shards
    -- or that read one iteration's tree for every iteration -- reports the same
    draw rate twice and fails here.
    """
    censused: dict[int, Any] = backfill_script.backfill(root)

    assert sorted(censused) == [0, 1]
    assert censused[0].games == 2
    assert censused[0].draws == 0
    assert censused[0].mean_plies == pytest.approx(50.0)
    assert censused[0].max_plies == 60

    assert censused[1].games == 2
    assert censused[1].draws == 1
    assert censused[1].turn_cap_games == 1
    assert censused[1].draw_rate == pytest.approx(0.5)
    assert censused[1].mean_plies == pytest.approx((60 + TURN_CAP_PLIES) / 2)
    assert censused[1].max_plies == TURN_CAP_PLIES


def test_the_bounds_are_inclusive_and_select_iterations(root: Path) -> None:
    """``--first`` and ``--last`` bound the walk; both ends are included."""
    assert sorted(backfill_script.backfill(root, first=1)) == [1]
    assert sorted(backfill_script.backfill(root, last=0)) == [0]
    assert sorted(backfill_script.backfill(root, first=0, last=1)) == [0, 1]
    assert backfill_script.backfill(root, first=9) == {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_writes_a_report_keyed_by_iteration(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON contract: one object per iteration, keyed by its number.

    Keyed rather than a list, because the interesting question of a backfill is
    "what did iteration 37 look like" and a list makes that a positional guess
    on a root with holes in it.
    """
    out = root / "report" / "game-shape.json"
    status: int = backfill_script.main([str(root), "--out", str(out)])
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(report) == ["0", "1"]
    for row in report.values():
        assert set(row) == set(backfill_script.REPORT_FIELDS)
        # By name as well as by set, so a rename of one of KOM-245's keys in
        # the script cannot be "fixed" by the pin agreeing with it.
        assert "cap_adjudicated_games" in row
        assert "cap_p_mean" in row
        assert "cap_p_ge_03_share" in row

    assert report["1"]["draws"] == 1
    assert report["1"]["turn_cap_games"] == 1
    assert report["1"]["draw_rate"] == pytest.approx(0.5)
    assert report["1"]["cap_rate"] == pytest.approx(0.5)
    assert report["1"]["max_plies"] == TURN_CAP_PLIES
    assert report["0"]["draws"] == 0
    assert report["0"]["mean_plies"] == pytest.approx(50.0)
    assert report["0"]["rows"] == 100

    printed = capsys.readouterr().out
    assert "iter" in printed and "draw" in printed


def test_the_cap_label_fields_are_reported_for_an_adjudicated_iteration(
    labelled_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """KOM-245's three numbers survive the backfill's own JSON enumeration.

    One adjudicated cap game and one plain one: 1 adjudicated, a mean of 0.25
    over both, and a 0.5 share at the threshold. The script re-lists its keys
    by hand (:data:`REPORT_FIELDS`, ``as_row``), so this is the test that says
    an offline census reports what the live loop stamps.
    """
    out = labelled_root / "report" / "game-shape.json"
    assert backfill_script.main([str(labelled_root), "--out", str(out)]) == 0
    capsys.readouterr()

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    row: dict[str, Any] = report["0"]

    assert row["turn_cap_games"] == 2, "the premise: both games reached the cap"
    assert row["cap_adjudicated_games"] == 1
    assert row["cap_p_mean"] == pytest.approx(0.25)
    assert row["cap_p_ge_03_share"] == pytest.approx(0.5)


def test_a_schema_without_wdl_target_reports_zero_cap_labels(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pre-KOM-245 shard tree reports zeros rather than omitting the keys.

    Which is the case a backfill is *for*: iteration 1 of this root holds a
    turn-capped game written against the four-column census schema, so there
    is a cap denominator and no label column to fill it from. Zero adjudicated
    games is the honest answer, and it is reported under the same keys as a
    labelled root's so a reader never has to branch on the campaign's age.
    """
    out = root / "report" / "game-shape.json"
    assert backfill_script.main([str(root), "--out", str(out)]) == 0
    capsys.readouterr()

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    row: dict[str, Any] = report["1"]

    assert row["turn_cap_games"] == 1, "the premise: there is a cap game to label"
    assert row["cap_adjudicated_games"] == 0
    assert row["cap_p_mean"] == pytest.approx(0.0)
    assert row["cap_p_ge_03_share"] == pytest.approx(0.0)


def test_main_bounds_the_report_it_writes(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--first``/``--last`` reach the report, not only the in-process census."""
    out = root / "one.json"
    assert backfill_script.main([str(root), "--first", "1", "--out", str(out)]) == 0
    capsys.readouterr()

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(report) == ["1"]


def test_main_prints_the_report_when_no_file_is_named(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--out`` the JSON goes to stdout, so the script pipes."""
    assert backfill_script.main([str(root)]) == 0

    printed = capsys.readouterr().out
    payload = printed[printed.index("{") :]
    report: dict[str, Any] = json.loads(payload)
    assert sorted(report) == ["0", "1"]


def test_main_refuses_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    """A typo'd path fails loudly rather than reporting an empty campaign."""
    with pytest.raises(ValueError, match="not a directory"):
        backfill_script.main([str(tmp_path / "no-such-root")])


def test_the_table_carries_a_line_per_iteration(root: Path) -> None:
    """The pasted table is readable on its own: header plus one row each."""
    censused: dict[int, Any] = backfill_script.backfill(root)
    table: str = backfill_script.format_table(censused)

    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].split()[0] == "0" and lines[2].split()[0] == "1"
    # KOM-250's two columns are in the header a reader pastes into a ticket,
    # not only in the JSON.
    assert "p>=.3" in lines[0] and "adj" in lines[0]
    assert backfill_script.format_table({}).endswith("(no iterations found)")
