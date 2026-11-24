"""KOM-142: ``scripts/analysis/termination_census.py``.

The script is loaded **by path**: ``scripts/`` is not a package, is not on
``sys.path``, and is outside ``[tool.mypy] files``, so a plain import would be
a hard mypy failure. Everything the loader returns is ``Any``; every test
binds what it reads to an explicitly annotated local so this module stays
strict-clean.

The move lists are **played, not written**: a hand-typed list of action ids
would be a list of numbers whose legality nothing checks, and the one thing
:func:`classify_outcome` must do is replay a real game. So the fixtures here
are uniformly random legal playouts from the engine's own start position --
short, seeded, and terminal by the engine's adjudication rather than by a ply
budget of this module's invention.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import pytest

from komugi.engine.game import TERMINATION_NONE, TERMINATION_REASONS, Game
from komugi.engine.repetition import RepetitionRule
from komugi.eval.arena import GameOutcome
from komugi.eval.records import DRAW, WHITE_WIN

#: ``<repo>/scripts/analysis`` -- this file is ``<repo>/tests/analysis/…``.
ANALYSIS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "scripts" / "analysis"
)

#: The module name the script is cached under in ``sys.modules``.
SCRIPT_NAME: Final[str] = "termination_census"


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


census_script: Final[Any] = load(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# Playout fixtures
# ---------------------------------------------------------------------------


def random_playout(
    seed: int, rule: RepetitionRule = RepetitionRule.FULL
) -> tuple[int, ...]:
    """Return the action ids of one uniformly random legal game.

    Terminates by the engine's own adjudication -- the loop condition is
    ``win_state()``, exactly the arena's -- so it is bounded by the engine's
    ply turn cap and needs no budget of its own.

    *rule* is the repetition key the playout is REFEREED under (KOM-141). It
    matters here for the same reason it matters in the script: a move list is
    only replayable under the rule that produced it, because the rule decides
    which ply was the last one.
    """
    game = Game(repetition_rule=rule)
    rng = np.random.default_rng(seed)
    moves: list[int] = []
    while not bool(np.asarray(game.win_state(), dtype=np.uint8).any()):
        legal = np.nonzero(np.asarray(game.valid_moves(), dtype=np.uint8))[0]
        assert legal.size > 0, (
            f"ply {len(moves)}: no legal action in a position the engine has "
            f"not adjudicated"
        )
        action = int(rng.choice(legal))
        game.play_action(action)
        moves.append(action)
    return tuple(moves)


def outcome_of(
    seed: int,
    result: float = DRAW,
    rule: RepetitionRule = RepetitionRule.FULL,
) -> GameOutcome:
    """A :class:`GameOutcome` over a real playout, for :func:`census`.

    ``result`` is the arena's own convention -- white's perspective, so
    ``DRAW`` is ``0.5`` and not ``0.0`` -- because ``census`` counts a draw by
    comparing against :data:`komugi.eval.records.DRAW`.
    """
    return GameOutcome(result=result, moves=random_playout(seed, rule))


# ---------------------------------------------------------------------------
# parse_entrant
# ---------------------------------------------------------------------------


def test_parse_entrant_reads_the_rawmcts_budget() -> None:
    spec: Any = census_script.parse_entrant("rawmcts:64")
    assert spec.scheme == "rawmcts"
    assert spec.sims == 64
    assert spec.label == "rawmcts:64"


def test_parse_entrant_accepts_random_and_a_checkpoint_path() -> None:
    random_spec: Any = census_script.parse_entrant("random")
    assert random_spec.scheme == "random"
    assert random_spec.label == "random"

    ckpt: Any = census_script.parse_entrant("checkpoint:/tmp/iteration-0042.pt")
    assert ckpt.scheme == "checkpoint"
    assert Path(ckpt.path) == Path("/tmp/iteration-0042.pt")


@pytest.mark.parametrize(
    "spec", ["rawmcts", "rawmcts:0", "rawmcts:many", "checkpoint:", "gumbel:64"]
)
def test_parse_entrant_refuses_what_it_cannot_play(spec: str) -> None:
    """A misspelled entrant is refused, not silently defaulted."""
    with pytest.raises(ValueError):
        census_script.parse_entrant(spec)


# ---------------------------------------------------------------------------
# classify_outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classify_outcome_names_a_real_reason(seed: int) -> None:
    """A replayed game reports one of the seven reasons, never ``"none"``."""
    moves = random_playout(seed)
    reason: str = census_script.classify_outcome(moves)
    assert reason in TERMINATION_REASONS
    assert reason != TERMINATION_NONE


def test_classify_outcome_agrees_with_the_game_it_replays() -> None:
    """The reason is the ENGINE's, not an inference: play it twice, compare."""
    moves = random_playout(7)
    game = Game()
    for action in moves:
        game.play_action(action)
    expected: str = game.termination_reason()

    assert census_script.classify_outcome(moves) == expected


def test_classify_outcome_refuses_a_game_that_has_not_ended() -> None:
    """A truncated move list ends live, which the ticket calls a bug."""
    moves = random_playout(3)
    with pytest.raises(ValueError, match="has not adjudicated"):
        census_script.classify_outcome(moves[:-1])


def test_classify_outcome_refuses_an_illegal_move_list() -> None:
    """A move list this engine cannot replay fails, naming the ply.

    The illegal id is READ OFF the start position's own mask rather than
    invented, so the test cannot become vacuous by picking an id that happens
    to be legal after all.
    """
    mask = np.asarray(Game().valid_moves(), dtype=np.uint8)
    illegal = int(np.nonzero(mask == 0)[0][0])
    assert mask[illegal] == 0

    with pytest.raises(ValueError, match="ply 0"):
        census_script.classify_outcome((illegal,))


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------


def test_census_reports_every_reason_and_sums_to_games() -> None:
    """All seven reasons appear, zeros included, and they sum to ``games``."""
    outcomes = (outcome_of(11), outcome_of(12, result=WHITE_WIN), outcome_of(13))
    report: dict[str, Any] = census_script.census(
        outcomes, seed=99, entrant="random"
    )

    assert set(report) == set(census_script.REPORT_KEYS)
    by_reason: dict[str, int] = report["by_reason"]
    assert set(by_reason) == set(TERMINATION_REASONS)
    assert sum(by_reason.values()) == report["games"] == len(outcomes)
    assert report["seed"] == 99
    assert report["entrant"] == "random"
    assert report["by_reason"][TERMINATION_NONE] == 0


def test_census_reads_the_draw_share_off_the_results() -> None:
    """``draw_share`` is the arena's result, not a function of the reasons."""
    outcomes = (outcome_of(21, result=DRAW), outcome_of(22, result=WHITE_WIN))
    report: dict[str, Any] = census_script.census(outcomes)

    assert report["draw_share"] == pytest.approx(0.5)
    assert report["mean_plies"] == pytest.approx(
        sum(len(outcome.moves) for outcome in outcomes) / 2
    )


def test_census_of_no_games_divides_by_nothing() -> None:
    report: dict[str, Any] = census_script.census(())
    assert report["games"] == 0
    assert report["draw_share"] == 0.0
    assert report["fourfold_share"] == 0.0
    assert report["mean_plies"] == 0.0


def test_format_table_prints_a_row_per_reason() -> None:
    report: dict[str, Any] = census_script.census((outcome_of(31),), entrant="random")
    table: str = census_script.format_table(report)
    for reason in TERMINATION_REASONS:
        assert reason in table


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_plays_two_games_and_writes_the_report(tmp_path: Path) -> None:
    """The smoke: two random-mover games, through the arena's own loop.

    ``random`` rather than ``rawmcts:2`` on purpose -- a RawMCTS simulation is
    itself a random playout to a terminal, so even the smallest budget plays
    each game hundreds of times over and the smoke would cost tens of seconds.
    The pipeline under test is the same one either way.
    """
    out = tmp_path / "census.json"
    status: int = census_script.main(
        [
            "--games",
            "2",
            "--seed",
            "5",
            "--entrant",
            "random",
            "--out",
            str(out),
        ]
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == set(census_script.REPORT_KEYS)
    assert report["games"] == 2
    assert report["seed"] == 5
    assert report["entrant"] == "random"

    by_reason: dict[str, int] = report["by_reason"]
    assert set(by_reason) == set(TERMINATION_REASONS)
    assert sum(by_reason.values()) == 2
    assert by_reason[TERMINATION_NONE] == 0
    assert report["mean_plies"] > 0.0


def test_main_refuses_an_odd_game_count(tmp_path: Path) -> None:
    """``ArenaConfig`` refuses it before a game is paid for."""
    with pytest.raises(ValueError, match="even"):
        census_script.main(
            ["--games", "3", "--entrant", "random", "--out", str(tmp_path / "x.json")]
        )


# ---------------------------------------------------------------------------
# KOM-141 -- --rule, threaded to the games AND to their replays
# ---------------------------------------------------------------------------


def test_the_report_names_the_rule_it_was_taken_under() -> None:
    """``rule`` is part of the JSON contract, and defaults to ``full``.

    Two censuses of the same players at the same seed under the two repetition
    rules are different measurements; a report that did not say which one it is
    could not be told apart from the other afterwards.
    """
    assert "rule" in census_script.REPORT_KEYS

    default: dict[str, Any] = census_script.census((outcome_of(41),))
    assert default["rule"] == "full"

    placement: dict[str, Any] = census_script.census(
        (outcome_of(41, rule=RepetitionRule.PLACEMENT),),
        rule=RepetitionRule.PLACEMENT,
    )
    assert placement["rule"] == "placement"
    assert sum(placement["by_reason"].values()) == placement["games"] == 1


def test_the_table_prints_the_rule() -> None:
    """A pasted table has to carry the rule too, not only the JSON."""
    report: dict[str, Any] = census_script.census(
        (outcome_of(42, rule=RepetitionRule.PLACEMENT),),
        entrant="random",
        rule=RepetitionRule.PLACEMENT,
    )
    assert "placement" in census_script.format_table(report)


@pytest.mark.parametrize(
    "rule", [RepetitionRule.FULL, RepetitionRule.PLACEMENT], ids=["full", "placement"]
)
def test_classify_outcome_replays_under_the_rule_it_is_given(
    rule: RepetitionRule,
) -> None:
    """A game refereed under `rule` classifies under `rule`.

    The replay is compared against the SAME game played independently under the
    same rule, so this asserts the reason is the engine's rather than merely
    that some reason came back -- the shape ``classify_outcome`` already has
    for the default rule, extended to the one that was added.
    """
    moves = random_playout(51, rule)

    game = Game(repetition_rule=rule)
    for action in moves:
        game.play_action(action)
    expected: str = game.termination_reason()

    reason: str = census_script.classify_outcome(moves, rule)
    assert reason == expected
    assert reason in TERMINATION_REASONS
    assert reason != TERMINATION_NONE


def test_main_smoke_under_the_placement_rule(tmp_path: Path) -> None:
    """The whole pipeline at ``--rule placement``: play, replay, report.

    This is the node that would fail if ``--rule`` reached the arena but not
    :func:`classify_outcome` (or the reverse): a game refereed under one rule
    and replayed under the other ends live or goes illegal, and the script
    raises rather than reporting. So a green run here is the threading, end to
    end, and ``by_reason`` summing to ``games`` says every game was classified.
    """
    out = tmp_path / "census-placement.json"
    status: int = census_script.main(
        [
            "--games",
            "2",
            "--seed",
            "5",
            "--entrant",
            "random",
            "--rule",
            "placement",
            "--out",
            str(out),
        ]
    )
    assert status == 0

    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == set(census_script.REPORT_KEYS)
    assert report["rule"] == "placement"
    assert report["games"] == 2
    assert sum(report["by_reason"].values()) == 2
    assert report["by_reason"][TERMINATION_NONE] == 0


def test_main_refuses_a_rule_it_does_not_know(tmp_path: Path) -> None:
    """A misspelled rule is refused by argparse, not silently defaulted."""
    with pytest.raises(SystemExit):
        census_script.main(
            [
                "--games",
                "2",
                "--entrant",
                "random",
                "--rule",
                "cells",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
