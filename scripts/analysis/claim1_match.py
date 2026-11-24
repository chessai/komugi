#!/usr/bin/env python3
"""KOM-148: the pre-registered refutation of KOM-022 §10 Claim 1.

Claim 1 is *"serving the EMA does not reduce self-play strength"*, and §10
pre-registers exactly how it dies: one extra **64-game match between the served
EMA and the raw newest checkpoint** at the fast budget, at two gauntlet
iterations; pooled over the 128 games, a one-sided 95% upper bound on the EMA's
score below 0.50 -- pooled ``s < 0.427`` -- refutes it.

The campaign loop never plays this match. It plays the *anchor* gauntlet, whose
opponent is a frozen, pinned generation; θ_i is not a pinned anchor and never
will be, so :func:`~komugi.campaign.gauntlet.run_gauntlet` -- which begins by
verifying a pinfile line -- cannot be the thing that plays it. This script is
the offline counterpart the advisor runs in a pause window.

**No new play code.** Everything below the request objects is the gauntlet's:
:func:`~komugi.campaign.gauntlet.run_shard` in the child,
:func:`~komugi.campaign.gauntlet.play_pairs` inside it,
:func:`~komugi.campaign.gauntlet.merge_outcomes` and
:func:`~komugi.campaign.gauntlet.score_for` in the parent. The EMA takes the
challenger slot and θ takes the slot the anchor would have had, which is the
whole difference.

Two things are deliberately *not* shared with the gauntlet:

* **The seed domain.** :data:`CLAIM1_DOMAIN` is a new spawn-key domain over the
  same run seed. Reusing ``GAUNTLET_DOMAIN`` would make this match replay the
  gauntlet's games at the same iteration -- the same openings and the same
  Gumbel streams -- against a different opponent, which is a correlated sample
  dressed up as an independent one.
* **The threads.** ``OMP_NUM_THREADS`` is set before the pool is created
  (KOM-146): sixteen shards times torch's default sixteen threads oversubscribe
  a 32-thread box eight-fold, and a four-minute gauntlet took sixty-three
  minutes.

Usage::

    python scripts/analysis/claim1_match.py play \\
        --ema  <root>/ema/checkpoint-000012288-<digest>.pt \\
        --theta <root>/checkpoints/checkpoint-000012288-<digest>.pt \\
        --run-seed 20260910 --iteration 11 --out claim1-i11.json

    python scripts/analysis/claim1_match.py pool claim1-i11.json claim1-i15.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, ContextManager, Final, Protocol, TypeVar

import numpy as np

from komugi.campaign.factory import NET_ENV
from komugi.campaign.gauntlet import (
    ShardRequest,
    merge_outcomes,
    run_shard,
    score_for,
    shard_ranges,
)
from komugi.campaign.pinned import GAUNTLET_GAMES, GAUNTLET_SHARDS
from komugi.config import SelfPlayConfig
from komugi.eval.records import GameRecord

CLAIM1_DOMAIN: Final[int] = 0x9022_0148
"""Spawn-key domain for this match's seeds -- **not** ``GAUNTLET_DOMAIN``.

Same run-seed family, its own domain. ``gauntlet_seed(s, i)`` and
:func:`claim1_seed` at the same ``(s, i)`` must differ, or the EMA-vs-θ match
would be the gauntlet's games played again with one entrant swapped: every
game's randomness is a pure function of the match seed and the pair
coordinates, so an identical seed is an identical set of openings.
"""

EMA_ID: Final[str] = "ema"
THETA_ID: Final[str] = "theta"

_FAST: Final[SelfPlayConfig] = SelfPlayConfig(workers=1, process_batch_size=1)
"""A throwaway config, read only for its **defaults**.

The pin the loop builds takes ``m``/``n``/``draw_score`` from
:class:`~komugi.config.SelfPlayConfig`'s fast-budget fields, so reading them
back off the same dataclass is what keeps this match at the campaign's budget
rather than at three numbers copied into a script and then left behind.
"""

FAST_M: Final[int] = _FAST.gumbel_m_fast
FAST_N: Final[int] = _FAST.gumbel_n_fast
DRAW_SCORE: Final[float] = _FAST.draw_score

DEFAULT_THREADS_PER_SHARD: Final[int] = 2
"""``OMP_NUM_THREADS`` per shard process (KOM-146).

Sixteen shards on a 32-thread box. Torch's default is one thread per core *per
process*, which at sixteen processes is an eight-fold oversubscription and the
measured difference between a four-minute match and an hour-long one.
"""

POOLED_GAMES: Final[int] = 128
"""The pre-registered pool: two gauntlet iterations at 64 games each.

Below this the verdict is ``underpowered`` rather than ``not_refuted``: §10
pre-registers the bound *at 128 games*, and a single 64-game match reporting
"not refuted" would be a design the claim can pass by being measured half.
"""

Z_95: Final[float] = 1.645
"""One-sided 95% normal quantile -- the same constant ``GAUNTLET_FLOOR``'s
docstring derives its bound from."""

REPORT_KEYS: Final[tuple[str, ...]] = (
    "ema",
    "theta",
    "iteration",
    "run_seed",
    "seed",
    "games",
    "decisive",
    "draws",
    "score_ema",
    "wall_s",
    "m",
    "n",
    "shards",
    "device",
    "threads_per_shard",
)
"""Exactly the keys :func:`play_match` writes -- the JSON contract, in one place."""

POOL_KEYS: Final[tuple[str, ...]] = (
    "n_games",
    "n_decisive",
    "pooled_score",
    "upper_95",
    "threshold",
    "verdict",
)
"""Exactly the keys :func:`pool_results` writes."""

REFUTED: Final[str] = "refuted"
NOT_REFUTED: Final[str] = "not_refuted"
UNDERPOWERED: Final[str] = "underpowered"

_T = TypeVar("_T")
_R = TypeVar("_R")

ShardResult = tuple[tuple[GameRecord, ...], tuple[int, ...]]
ShardRunner = Callable[[ShardRequest], ShardResult]


class _MapPool(Protocol):
    """The one method this script uses of a ``multiprocessing`` pool."""

    def map(self, func: Callable[[_T], _R], iterable: Sequence[_T]) -> list[_R]:
        """Apply `func` to every element, in order."""


PoolFactory = Callable[[int], ContextManager[_MapPool]]


def claim1_seed(run_seed: int, iteration: int) -> int:
    """Return this iteration's Claim-1 match seed, in its own domain.

    The same shape as :func:`~komugi.campaign.gauntlet.gauntlet_seed` over the
    same run seed, differing only in the spawn-key domain -- which is the point
    (see :data:`CLAIM1_DOMAIN`).
    """
    child = np.random.SeedSequence(run_seed, spawn_key=(CLAIM1_DOMAIN, iteration))
    return int(child.generate_state(1, dtype=np.uint32)[0])


def refutation_threshold(n: int) -> float:
    """Return the pooled score below which Claim 1 is refuted at `n` games.

    ``s + 1.645 * sqrt(0.25 / n) < 0.50``, rearranged: the one-sided 95% upper
    bound on a proportion at its worst-case variance, exactly the derivation
    :data:`~komugi.campaign.pinned.GAUNTLET_FLOOR` states for the gauntlet. At
    ``n = 128`` this is 0.4273 -- §10's pre-registered 0.427.

    Raises:
        ValueError: If `n` is not positive. A threshold over no games is not a
            wide threshold, it is not a threshold.
    """
    if n < 1:
        raise ValueError(f"a refutation threshold needs at least one game, got {n!r}")
    return 0.5 - Z_95 * math.sqrt(0.25 / n)


def build_requests(
    ema_path: str | os.PathLike[str],
    theta_path: str | os.PathLike[str],
    *,
    games: int = GAUNTLET_GAMES,
    shards: int = GAUNTLET_SHARDS,
    seed: int,
    draw_score: float = DRAW_SCORE,
    m: int = FAST_M,
    n: int = FAST_N,
    device: str = "cpu",
) -> tuple[ShardRequest, ...]:
    """Build one shard request per non-empty pair range, in range order.

    The EMA is the **challenger** and θ sits in the slot the gauntlet's anchor
    would have had, so :func:`~komugi.campaign.gauntlet.score_for` at
    :data:`EMA_ID` reads the score the claim is about. Nothing here verifies a
    pinfile: θ is the newest raw checkpoint, which is by construction not a
    pinned anchor.

    ``net_env`` is read from this process's environment and carried in the
    request, exactly as ``_shard_requests`` does it: a spawned child inherits
    nothing, and one spawned without ``NET_ENV`` against a small-net root dies
    inside ``load_state_dict`` with a shape error that reads like a corrupt
    checkpoint.
    """
    ranges = shard_ranges(games // 2, shards)
    net_env = os.environ.get(NET_ENV, "")
    return tuple(
        ShardRequest(
            challenger_path=str(ema_path),
            challenger_id=EMA_ID,
            anchor_path=str(theta_path),
            anchor_id=THETA_ID,
            pairs_start=piece.start,
            pairs_stop=piece.stop,
            games=games,
            seed=seed,
            draw_score=draw_score,
            m=m,
            n=n,
            device=device,
            net_env=net_env,
        )
        for piece in ranges
        if len(piece) > 0
    )


def _spawn_pool(processes: int) -> ContextManager[_MapPool]:
    """Return a ``spawn`` pool of `processes` workers.

    ``spawn`` for the reason every other multi-process seam in this repo uses
    it, stated in ``run_gauntlet``: a forked child inherits torch's threading
    state and the parent's CUDA context.
    """
    return multiprocessing.get_context("spawn").Pool(processes=processes)


def play_match(
    ema_path: str | os.PathLike[str],
    theta_path: str | os.PathLike[str],
    *,
    run_seed: int,
    iteration: int,
    games: int = GAUNTLET_GAMES,
    shards: int = GAUNTLET_SHARDS,
    device: str = "cpu",
    threads_per_shard: int = DEFAULT_THREADS_PER_SHARD,
    shard_runner: ShardRunner = run_shard,
    pool_factory: PoolFactory = _spawn_pool,
) -> dict[str, Any]:
    """Play one EMA-vs-θ match and return its report.

    The parent process plays nothing: it builds requests, fans out, merges and
    scores -- the gauntlet's division of labour, for the gauntlet's reason
    (``pin_determinism`` sets *process-global* torch state).

    ``OMP_NUM_THREADS`` is set **before the pool exists**, not inside the child:
    a spawned child inherits the environment as it stood at spawn time, and torch
    reads the variable when it is first imported there. Setting it afterwards
    would be a no-op that looks like a fix (KOM-146).

    `shard_runner` and `pool_factory` are the test seam. The real path is the
    default pair, and substituting either does not change what the other does --
    a canned runner still goes through ``pool.map``, so the fan-out is exercised
    rather than bypassed.
    """
    os.environ["OMP_NUM_THREADS"] = str(threads_per_shard)

    seed = claim1_seed(run_seed, iteration)
    requests = build_requests(
        ema_path,
        theta_path,
        games=games,
        shards=shards,
        seed=seed,
        device=device,
    )

    started = time.monotonic()
    with pool_factory(max(len(requests), 1)) as pool:
        played = pool.map(shard_runner, requests)
    wall_s = time.monotonic() - started

    outcome = merge_outcomes(played)
    return {
        "ema": str(ema_path),
        "theta": str(theta_path),
        "iteration": iteration,
        "run_seed": run_seed,
        "seed": seed,
        "games": outcome.summary.games,
        "decisive": outcome.summary.decisive,
        "draws": outcome.summary.draws,
        "score_ema": score_for(outcome.records, EMA_ID),
        "wall_s": wall_s,
        "m": FAST_M,
        "n": FAST_N,
        "shards": len(requests),
        "device": device,
        "threads_per_shard": threads_per_shard,
    }


def pool_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the matches and return §10's verdict on Claim 1.

    The pooled score is weighted by each match's **game count**, not a mean of
    means: two matches of unequal length are one 128-game sample, and averaging
    their scores would silently reweight the shorter one upward.

    The verdict is ``underpowered`` below :data:`POOLED_GAMES` rather than
    ``not_refuted`` -- see that constant. Above it, the comparison is against
    :func:`refutation_threshold` at the games actually pooled, so pooling more
    than 128 tightens the bound rather than leaving it at the design's number.
    """
    n_games = sum(int(result["games"]) for result in results)
    n_decisive = sum(int(result["decisive"]) for result in results)
    if n_games == 0:
        pooled = 0.0
    else:
        pooled = (
            sum(float(result["score_ema"]) * int(result["games"]) for result in results)
            / n_games
        )

    if n_games == 0:
        upper = 0.0
        threshold = 0.0
    else:
        upper = pooled + Z_95 * math.sqrt(0.25 / n_games)
        threshold = refutation_threshold(n_games)

    if n_games < POOLED_GAMES:
        verdict = UNDERPOWERED
    elif pooled < threshold:
        verdict = REFUTED
    else:
        verdict = NOT_REFUTED

    return {
        "n_games": n_games,
        "n_decisive": n_decisive,
        "pooled_score": pooled,
        "upper_95": upper,
        "threshold": threshold,
        "verdict": verdict,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write `payload` as pretty, key-sorted JSON, creating the parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser: ``play`` one match, ``pool`` several."""
    parser = argparse.ArgumentParser(
        prog="claim1_match.py",
        description=(
            "KOM-022 §10 Claim 1: play one 64-game EMA-vs-theta match at the "
            "fast budget, and pool two of them into the pre-registered verdict."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="play one match and write its JSON report")
    play.add_argument("--ema", required=True, type=Path, help="the served EMA .pt")
    play.add_argument(
        "--theta", required=True, type=Path, help="the raw newest checkpoint .pt"
    )
    play.add_argument(
        "--run-seed",
        required=True,
        type=int,
        help="the campaign's run_seed; the match seed is derived from it",
    )
    play.add_argument(
        "--iteration",
        required=True,
        type=int,
        help="the iteration these two checkpoints are from",
    )
    play.add_argument("--out", required=True, type=Path, help="where the report goes")
    play.add_argument(
        "--games", type=int, default=GAUNTLET_GAMES, help="games (seat-swapped pairs)"
    )
    play.add_argument(
        "--shards", type=int, default=GAUNTLET_SHARDS, help="shard processes"
    )
    play.add_argument("--device", default="cpu", help="device for both entrants' nets")
    play.add_argument(
        "--threads-per-shard",
        type=int,
        default=DEFAULT_THREADS_PER_SHARD,
        help=(
            "OMP_NUM_THREADS in each shard; the default keeps 16 shards from "
            "oversubscribing a 32-thread box eight-fold (KOM-146)"
        ),
    )

    pool = sub.add_parser("pool", help="pool match reports into the §10 verdict")
    pool.add_argument("reports", nargs="+", type=Path, help="JSON reports from `play`")
    pool.add_argument(
        "--out", type=Path, default=None, help="where the pooled verdict goes"
    )
    return parser


def _run_play(args: argparse.Namespace) -> int:
    """Play one match, write the report, print one line."""
    report = play_match(
        args.ema,
        args.theta,
        run_seed=args.run_seed,
        iteration=args.iteration,
        games=args.games,
        shards=args.shards,
        device=args.device,
        threads_per_shard=args.threads_per_shard,
    )
    out: Path = args.out
    _write_json(out, report)
    print(
        f"iteration {report['iteration']}: {EMA_ID} scored "
        f"{report['score_ema']:.4f} over {report['games']} games "
        f"({report['decisive']} decisive, {report['draws']} drawn) in "
        f"{report['wall_s']:.1f}s -> {out}"
    )
    return 0


def _run_pool(args: argparse.Namespace) -> int:
    """Pool the named reports, print the verdict, write it when asked."""
    results: list[Mapping[str, Any]] = []
    for path in args.reports:
        loaded: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} is not a match report object")
        results.append(loaded)

    pooled = pool_results(results)
    print(json.dumps(pooled, indent=2, sort_keys=True))
    if args.out is not None:
        _write_json(args.out, pooled)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the subcommand and return an exit status."""
    args = build_parser().parse_args(argv)
    if args.command == "play":
        return _run_play(args)
    return _run_pool(args)


if __name__ == "__main__":
    # Guarded, and not decoration: `spawn` re-imports this module in every child
    # process, so an unguarded entry point would have each shard re-parse argv
    # and start a match of its own.
    raise SystemExit(main())
