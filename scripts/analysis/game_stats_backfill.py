#!/usr/bin/env python3
"""KOM-150: the draw share and game length of a campaign that already ran.

:mod:`komugi.campaign.game_stats` is wired into the loop from this ticket
forward, so every iteration a campaign plays *from now on* stamps its draw
rate, mean game length and turn-cap share onto both of its records. This script
is the other direction: the same census, run over a campaign root that was
written before the wiring existed.

That is not a hypothetical. Campaign 1's ``selfplay.jsonl`` is schema v1 from
end to end -- games, rows, restarts and wall time -- and its journal reports
``draw_rate`` and ``avg_game_len`` as literal zeros, because the orchestrator
did not measure them. The shards, however, have carried ``game_id``, ``ply``,
``mlh_target`` and ``termination`` since KOM-005b, so **the numbers were on
disk the whole time**; only the reader was missing. Running this over campaign
1's root recovers its history exactly: 840 games, 13 draws (1.5%), every one of
them a 256-ply turn cap, mean 115 plies.

What each iteration reports is :data:`REPORT_FIELDS`: the games, rows and draws
with their rates, the three length numbers, and -- since KOM-245 --
``cap_adjudicated_games``, ``cap_p_mean`` and ``cap_p_ge_03_share``, the
cap-label numbers the loop now stamps live. Those three read as zeros over a
root whose shards predate the ``wdl_target`` column, which is the true answer
for a campaign that never adjudicated a cap game.

Nothing here writes into the campaign root. It reads shards and prints, so it
is safe to run against a **live** campaign -- the per-iteration census reads
sealed shards only, and the iteration currently being played simply reports
whatever it has sealed so far.

Usage::

    python scripts/analysis/game_stats_backfill.py runs/campaign-1
    python scripts/analysis/game_stats_backfill.py runs/campaign-1 \\
        --first 5 --last 20 --out game-shape.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from komugi.campaign.game_stats import GameStats, stats_for_iteration
from komugi.campaign.loop import iteration_root

REPORT_FIELDS: Final[tuple[str, ...]] = (
    "games",
    "rows",
    "draws",
    "draw_rate",
    "turn_cap_games",
    "cap_rate",
    "mean_plies",
    "median_plies",
    "max_plies",
    "cap_adjudicated_games",
    "cap_p_mean",
    "cap_p_ge_03_share",
)
"""Exactly the keys this script writes per iteration -- the JSON contract.

KOM-245's three cap-label numbers are **appended** rather than interleaved, so
an existing reader that took these positionally keeps every position it had. A
tree whose shards carry no ``wdl_target`` -- every campaign written before the
column existed, which is most of what a backfill runs over -- reports them as
the honest zeros :func:`~komugi.campaign.game_stats._label_p` returns there.
"""

_ITERATION_NAME: Final[re.Pattern[str]] = re.compile(r".*?(\d+)$")
"""Trailing digits of a directory name, as a candidate iteration number.

A *candidate*: the number is only accepted once
:func:`~komugi.campaign.loop.iteration_root` agrees that this is where that
iteration lives (see :func:`iterations_under`). The campaign's layout stays the
loop's to state; this pattern only proposes numbers to ask it about.
"""


def iterations_under(root: Path) -> tuple[int, ...]:
    """Return every iteration number `root` holds a directory for, ascending.

    Discovered rather than assumed: a resumed campaign's root can have holes
    (an iteration whose tree was cleaned by the loop's stale-shard rmtree and
    never replayed), and a run that is still going has no total to count up to.

    Every candidate is confirmed against
    :func:`~komugi.campaign.loop.iteration_root`, so a directory that merely
    ends in a number is not mistaken for an iteration.
    """
    if not root.is_dir():
        return ()
    found: list[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = _ITERATION_NAME.fullmatch(entry.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration_root(root, iteration) == entry:
            found.append(iteration)
    return tuple(sorted(found))


def as_row(stats: GameStats) -> dict[str, Any]:
    """Return one iteration's census as the JSON object this script writes."""
    return {
        "games": stats.games,
        "rows": stats.rows,
        "draws": stats.draws,
        "draw_rate": stats.draw_rate,
        "turn_cap_games": stats.turn_cap_games,
        "cap_rate": stats.cap_rate,
        "mean_plies": stats.mean_plies,
        "median_plies": stats.median_plies,
        "max_plies": stats.max_plies,
        "cap_adjudicated_games": stats.cap_adjudicated_games,
        "cap_p_mean": stats.cap_p_mean,
        "cap_p_ge_03_share": stats.cap_p_ge_03_share,
    }


def backfill(
    root: Path, *, first: int | None = None, last: int | None = None
) -> dict[int, GameStats]:
    """Census every iteration under `root`, bounded by `first` and `last`.

    Both bounds are **inclusive** and both are optional; they exist because a
    long campaign's early iterations are usually the ones an operator has
    already read, and because a census of one iteration should not have to walk
    a hundred shard trees to reach it.
    """
    censused: dict[int, GameStats] = {}
    for iteration in iterations_under(root):
        if first is not None and iteration < first:
            continue
        if last is not None and iteration > last:
            continue
        censused[iteration] = stats_for_iteration(root, iteration)
    return censused


def format_table(censused: dict[int, GameStats]) -> str:
    """Return the human-readable table, one line per iteration, oldest first.

    Rates are printed beside their counts rather than instead of them: a draw
    rate of ``0.50`` over four games and over four hundred are different
    findings, and only the count tells them apart. The same reason puts
    ``adj`` -- the adjudicated cap-game count -- beside ``p>=.3``, K1's share.
    """
    header = (
        f"{'iter':>5}  {'games':>6}  {'draws':>6}  {'draw':>6}  "
        f"{'capped':>6}  {'cap':>6}  {'mean':>7}  {'median':>7}  {'max':>5}  "
        f"{'p>=.3':>6}  {'adj':>5}"
    )
    lines = [header]
    for iteration in sorted(censused):
        stats = censused[iteration]
        lines.append(
            f"{iteration:>5}  {stats.games:>6}  {stats.draws:>6}  "
            f"{stats.draw_rate:>6.3f}  {stats.turn_cap_games:>6}  "
            f"{stats.cap_rate:>6.3f}  {stats.mean_plies:>7.1f}  "
            f"{stats.median_plies:>7.1f}  {stats.max_plies:>5}  "
            f"{stats.cap_p_ge_03_share:>6.3f}  "
            f"{stats.cap_adjudicated_games:>5}"
        )
    if not censused:
        lines.append("(no iterations found)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="game_stats_backfill.py",
        description=(
            "Census a campaign root's sealed shards: draw share, turn-cap "
            "share and game length, per iteration."
        ),
    )
    parser.add_argument("root", type=Path, help="the campaign root to read")
    parser.add_argument(
        "--first",
        type=int,
        default=None,
        help="skip iterations below this number (inclusive bound)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="skip iterations above this number (inclusive bound)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the JSON report; stdout when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Census the root, print the table, write the report, return a status."""
    args = build_parser().parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory; there is nothing to census")

    censused = backfill(root, first=args.first, last=args.last)
    report = {str(iteration): as_row(censused[iteration]) for iteration in censused}

    print(format_table(censused))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(serialized)
    else:
        out: Path = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
