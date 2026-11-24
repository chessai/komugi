"""KOM-148: ``scripts/analysis/claim1_match.py``.

The script is loaded **by path**, the way
``tests/analysis/test_termination_census.py`` loads its own: ``scripts/`` is not
a package, is not on ``sys.path``, and is outside ``[tool.mypy] files``, so a
plain import would be a hard mypy failure. Everything the loader returns is
``Any``; every test binds what it reads to an explicitly annotated local so this
module stays strict-clean.

**No game is played here.** A 64-game fast-budget match is minutes of real
search, and none of what this ticket adds is search: it is a seed domain, a
normal-approximation threshold, a set of shard requests, and a pooled verdict.
So the shard runner is replaced by one returning canned records built exactly as
``tests/campaign/behavior/test_gauntlet.py`` builds them, and the pool by a
serial stand-in -- which still goes through ``pool.map``, so the fan-out the
real path uses is exercised rather than bypassed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Final, TypeVar

import pytest

from komugi.campaign.gauntlet import gauntlet_seed
from komugi.campaign.pinned import GAUNTLET_GAMES, GAUNTLET_SHARDS
from komugi.eval.records import DRAW, WHITE_WIN, GameRecord

#: ``<repo>/scripts/analysis`` -- this file is ``<repo>/tests/analysis/…``.
ANALYSIS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "scripts" / "analysis"

#: The module name the script is cached under in ``sys.modules``.
SCRIPT_NAME: Final[str] = "claim1_match"


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


claim1: Final[Any] = load(SCRIPT_NAME)

EMA: Final[str] = "ema"
THETA: Final[str] = "theta"

_T = TypeVar("_T")
_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Fixtures: canned games and a serial pool
# ---------------------------------------------------------------------------


def _record(pair: int, seat: int, result: float) -> GameRecord:
    """One constructed game of the EMA-versus-theta match.

    Built the way ``tests/campaign/behavior/test_gauntlet.py`` builds them: the
    seat decides which id is white, and ``result`` is stated from **white's**
    perspective, which is what makes ``score_for`` more than an average.
    """
    white, black = (EMA, THETA) if seat == 0 else (THETA, EMA)
    return GameRecord(
        white=white,
        black=black,
        result=result,
        book_id="empty",
        pair_id=f"{EMA}:{THETA}:{pair:05d}",
    )


def canned_runner(
    result: float = DRAW,
) -> Callable[[Any], tuple[tuple[GameRecord, ...], tuple[int, ...]]]:
    """Return a shard runner that plays no games and returns `result` for all.

    Its records carry the shard's **global** pair indices, so a runner handed a
    request for pairs 8..10 returns four games numbered 8 and 9 -- the property
    the merge relies on, and the one a renumbering shard would break.
    """

    def run(request: Any) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
        """Return two records per pair in the request's range."""
        start: int = int(request.pairs_start)
        stop: int = int(request.pairs_stop)
        records = tuple(
            _record(pair, seat, result)
            for pair in range(start, stop)
            for seat in (0, 1)
        )
        return records, tuple(40 + index for index in range(len(records)))

    return run


class SerialPool:
    """A ``multiprocessing`` pool that runs everything in this process.

    Substituted for the ``spawn`` pool so the test can inject a *local* shard
    runner: a spawned child re-imports the module and unpickles the callable by
    name, so a closure could never reach one. What is under test here is the
    parent's arithmetic, and it runs through the same ``pool.map`` call.
    """

    def __init__(self, processes: int) -> None:
        self.processes = processes

    def __enter__(self) -> SerialPool:
        """Enter the pool's context, as ``Pool`` does."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the pool's context; there is nothing to terminate."""

    def map(self, func: Callable[[_T], _R], iterable: Sequence[_T]) -> list[_R]:
        """Apply `func` to every element, in order."""
        return [func(item) for item in iterable]


def serial_pool_factory(seen: list[int]) -> Callable[[int], SerialPool]:
    """Return a pool factory that records the process count it was asked for."""

    def build(processes: int) -> SerialPool:
        """Build one serial pool, noting its requested width."""
        seen.append(processes)
        return SerialPool(processes)

    return build


# ---------------------------------------------------------------------------
# (1) the pre-registered threshold
# ---------------------------------------------------------------------------


def test_the_threshold_at_128_games_is_the_designs_0_427() -> None:
    """§10's number, re-derived rather than copied.

    ``s + 1.645 * sqrt(0.25/128) < 0.50`` gives 0.4273. If this drifts, the
    refutation criterion has been moved, which is the one edit a pre-registered
    design exists to make visible.
    """
    threshold: float = claim1.refutation_threshold(128)
    assert threshold == pytest.approx(0.427, abs=1e-3)


def test_fewer_games_demand_a_lower_score() -> None:
    """The bound is wider at 64 games, so the score must be worse to refute."""
    at_64: float = claim1.refutation_threshold(64)
    at_128: float = claim1.refutation_threshold(128)
    assert at_64 < at_128 < 0.5


def test_a_threshold_over_no_games_is_refused() -> None:
    with pytest.raises(ValueError):
        claim1.refutation_threshold(0)


# ---------------------------------------------------------------------------
# (2) pooling two matches into a verdict
# ---------------------------------------------------------------------------


def _result(score: float, games: int = 64, decisive: int = 40) -> dict[str, Any]:
    """The subset of a match report that :func:`pool_results` reads."""
    return {"games": games, "decisive": decisive, "score_ema": score}


def test_two_matches_below_the_threshold_refute_claim_1() -> None:
    """0.40 and 0.44 over 128 games pool to 0.42, under 0.4273."""
    pooled: dict[str, Any] = claim1.pool_results(
        [_result(0.40), _result(0.44)]
    )

    assert set(pooled) == set(claim1.POOL_KEYS)
    assert pooled["n_games"] == 128
    assert pooled["n_decisive"] == 80
    assert pooled["pooled_score"] == pytest.approx(0.42)
    assert pooled["threshold"] == pytest.approx(claim1.refutation_threshold(128))
    assert pooled["upper_95"] == pytest.approx(0.42 + 1.645 * (0.25 / 128) ** 0.5)
    assert pooled["verdict"] == "refuted"


def test_a_match_at_parity_does_not_refute_claim_1() -> None:
    """0.50 and 0.52 pool to 0.51: the EMA is not measurably weaker."""
    pooled: dict[str, Any] = claim1.pool_results([_result(0.50), _result(0.52)])

    assert pooled["pooled_score"] == pytest.approx(0.51)
    assert pooled["verdict"] == "not_refuted"


def test_one_match_alone_is_underpowered_not_a_verdict() -> None:
    """§10 pre-registers the bound at 128 games, so 64 buys no verdict.

    Reporting ``not_refuted`` here would be a design the claim can pass by being
    measured half, which is exactly the failure pre-registration is for.
    """
    pooled: dict[str, Any] = claim1.pool_results([_result(0.40)])

    assert pooled["n_games"] == 64
    assert pooled["pooled_score"] == pytest.approx(0.40)
    assert pooled["verdict"] == "underpowered"


def test_the_pooled_score_weights_by_games_not_by_match() -> None:
    """Two unequal matches are one sample, not two equally-weighted ones."""
    pooled: dict[str, Any] = claim1.pool_results(
        [_result(0.30, games=32, decisive=20), _result(0.50, games=96, decisive=60)]
    )

    assert pooled["n_games"] == 128
    assert pooled["pooled_score"] == pytest.approx((0.30 * 32 + 0.50 * 96) / 128)


# ---------------------------------------------------------------------------
# (3) the shard requests: EMA challenges, theta sits in the anchor's slot
# ---------------------------------------------------------------------------


def test_the_requests_seat_the_ema_against_theta_at_the_fast_budget() -> None:
    """Sixteen requests, the campaign's m/n, and the two paths in their slots."""
    requests: tuple[Any, ...] = claim1.build_requests(
        Path("/tmp/ema/checkpoint-000012288-aaaaaaaaaaaaaaaa.pt"),
        Path("/tmp/ckpt/checkpoint-000012288-bbbbbbbbbbbbbbbb.pt"),
        games=GAUNTLET_GAMES,
        shards=GAUNTLET_SHARDS,
        seed=4242,
    )

    assert len(requests) == GAUNTLET_SHARDS
    for request in requests:
        assert request.challenger_path.endswith("aaaaaaaaaaaaaaaa.pt")
        assert request.challenger_id == claim1.EMA_ID
        assert request.anchor_path.endswith("bbbbbbbbbbbbbbbb.pt")
        assert request.anchor_id == claim1.THETA_ID
        assert request.games == GAUNTLET_GAMES
        assert request.m == 8 and request.n == 32
        assert request.seed == 4242
        assert request.device == "cpu"


def test_the_request_ranges_tile_the_pairs_exactly_once() -> None:
    """32 pairs, contiguous and index-preserving, no pair played twice.

    Index-preserving is the load-bearing half: every game's randomness is a pure
    function of the match seed and the GLOBAL pair coordinates, so a shard that
    renumbered its pairs would play a different match.
    """
    requests: tuple[Any, ...] = claim1.build_requests(
        Path("ema.pt"), Path("theta.pt"), games=64, shards=16, seed=1
    )

    ranges = sorted((int(r.pairs_start), int(r.pairs_stop)) for r in requests)
    covered = [pair for start, stop in ranges for pair in range(start, stop)]
    assert covered == list(range(32))


def test_more_shards_than_pairs_drops_the_empty_ones() -> None:
    """A process that plays nothing is pure cost, so it is never requested."""
    requests: tuple[Any, ...] = claim1.build_requests(
        Path("ema.pt"), Path("theta.pt"), games=4, shards=16, seed=1
    )

    assert len(requests) == 2
    assert all(r.pairs_stop > r.pairs_start for r in requests)


# ---------------------------------------------------------------------------
# (4) the seed domain: this match is not the gauntlet replayed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_seed", [0, 20260910])
@pytest.mark.parametrize("iteration", list(range(9)))
def test_the_match_never_replays_the_gauntlets_games(
    run_seed: int, iteration: int
) -> None:
    """A shared seed would be the same openings against a swapped opponent.

    That is a correlated sample presented as an independent one: the gauntlet
    and this match run at the same iterations, and every game's randomness is a
    pure function of the match seed and the pair coordinates.
    """
    mine: int = claim1.claim1_seed(run_seed, iteration)
    theirs: int = gauntlet_seed(run_seed, iteration)
    assert mine != theirs


def test_the_seed_is_a_function_of_the_run_seed_and_the_iteration() -> None:
    """Reproducible, and different per iteration -- ``gauntlet_seed``'s shape."""
    assert claim1.claim1_seed(7, 3) == claim1.claim1_seed(7, 3)
    assert claim1.claim1_seed(7, 3) != claim1.claim1_seed(7, 4)
    assert claim1.claim1_seed(7, 3) != claim1.claim1_seed(8, 3)


def test_the_domain_is_not_the_gauntlets() -> None:
    """Stated as a constant too: the two domains are different numbers."""
    from komugi.campaign.gauntlet import GAUNTLET_DOMAIN

    assert claim1.CLAIM1_DOMAIN != GAUNTLET_DOMAIN


# ---------------------------------------------------------------------------
# (5) play_match, end to end, with the games replaced
# ---------------------------------------------------------------------------


def test_play_match_reports_a_full_64_game_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole parent-side path: requests, fan-out, merge, score, report.

    ``monkeypatch.setenv`` first, so the ``OMP_NUM_THREADS`` this test observes
    is restored on teardown rather than leaked into every later test in the
    process -- the variable is process-global and torch reads it on import.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    widths: list[int] = []

    report: dict[str, Any] = claim1.play_match(
        tmp_path / "ema.pt",
        tmp_path / "theta.pt",
        run_seed=20260910,
        iteration=11,
        shard_runner=canned_runner(DRAW),
        pool_factory=serial_pool_factory(widths),
    )

    assert set(report) == set(claim1.REPORT_KEYS)
    assert report["games"] == GAUNTLET_GAMES
    assert report["draws"] == GAUNTLET_GAMES
    assert report["decisive"] == 0
    assert report["score_ema"] == pytest.approx(0.5)
    assert report["iteration"] == 11
    assert report["run_seed"] == 20260910
    assert report["seed"] == claim1.claim1_seed(20260910, 11)
    assert report["shards"] == GAUNTLET_SHARDS
    assert report["m"] == 8 and report["n"] == 32
    assert report["device"] == "cpu"
    assert report["threads_per_shard"] == 2
    assert report["wall_s"] >= 0.0

    assert widths == [GAUNTLET_SHARDS], (
        "the pool was not sized to the requests, or was never created: the "
        "canned runner must go through pool.map like the real one"
    )


def test_play_match_sets_the_thread_cap_before_the_pool_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """KOM-146: a child inherits the environment as it stood at spawn time.

    So the pool factory -- which stands where ``Pool(...)`` stands on the real
    path -- must already see the cap. Setting it afterwards is a no-op that
    looks like a fix, and it cost a four-minute gauntlet sixty-three minutes.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    at_pool_creation: list[str] = []

    def build(processes: int) -> SerialPool:
        """Note the cap as it stands when the pool is built."""
        at_pool_creation.append(os.environ.get("OMP_NUM_THREADS", ""))
        return SerialPool(processes)

    claim1.play_match(
        tmp_path / "ema.pt",
        tmp_path / "theta.pt",
        run_seed=1,
        iteration=0,
        threads_per_shard=3,
        shard_runner=canned_runner(DRAW),
        pool_factory=build,
    )

    assert at_pool_creation == ["3"]
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_play_match_scores_from_the_emas_side_in_both_seats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Half the EMA's games are as black; a raw result average would say 0.5.

    Every canned game is a WHITE win, so an EMA that scored its own side reads
    0.5 -- one win and one loss per seat-swapped pair -- and one that read
    ``result`` directly would read 1.0.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    report: dict[str, Any] = claim1.play_match(
        tmp_path / "ema.pt",
        tmp_path / "theta.pt",
        run_seed=5,
        iteration=7,
        shard_runner=canned_runner(WHITE_WIN),
        pool_factory=serial_pool_factory([]),
    )

    assert report["score_ema"] == pytest.approx(0.5)
    assert report["decisive"] == GAUNTLET_GAMES
    assert report["draws"] == 0


# ---------------------------------------------------------------------------
# (6) the CLI's `pool` subcommand
# ---------------------------------------------------------------------------


def test_the_cli_pools_two_reports_into_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through ``main([...])``, over JSON files, exactly as the advisor runs it."""
    first = tmp_path / "claim1-i11.json"
    second = tmp_path / "claim1-i15.json"
    first.write_text(json.dumps(_result(0.40)), encoding="utf-8")
    second.write_text(json.dumps(_result(0.44)), encoding="utf-8")
    out = tmp_path / "pooled.json"

    status: int = claim1.main(["pool", str(first), str(second), "--out", str(out)])
    assert status == 0

    printed: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert set(printed) == set(claim1.POOL_KEYS)
    assert printed["verdict"] == "refuted"

    written: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert set(written) == set(claim1.POOL_KEYS)
    assert written == printed


def test_the_cli_pool_of_one_report_says_underpowered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """And writes nothing when no ``--out`` is given."""
    only = tmp_path / "claim1-i11.json"
    only.write_text(json.dumps(_result(0.40)), encoding="utf-8")

    assert claim1.main(["pool", str(only)]) == 0

    printed: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert printed["n_games"] == 64
    assert printed["verdict"] == "underpowered"


def test_the_cli_refuses_a_subcommand_it_does_not_have() -> None:
    """A misspelled subcommand exits rather than defaulting to playing games."""
    with pytest.raises(SystemExit):
        claim1.main(["poool", "x.json"])
