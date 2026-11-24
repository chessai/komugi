"""Cost of the solve, recorded per puzzle and bounded in aggregate.

Two separate budgets, because they measure different things. The df-pn
solver is the thing that has to stay fast enough to be called from an
experiment loop, so the friend set gets a wall-clock and a node bound. The
reference solver is a naive unmemoised search over the pure-Python rules
engine and is expected to be orders of magnitude slower; it gets its own,
generous bound, and it is never allowed to spend the solver's.

Both bounds are far above the measured numbers on purpose. A perf test that
sits near its limit fails on a loaded CI box and teaches everyone to ignore
it; these fail only if something has changed by an order of magnitude, which
is the only perf regression worth a red run.
"""

from __future__ import annotations

import time

from reference_tsume import reference_solve
from tsume_fixtures import FRIEND

from komugi.search.tsume import solve_tsume

#: df-pn over the 14 composed puzzles. Measured: ~0.4 s, ~1200 nodes total.
DFPN_SECONDS = 60.0
DFPN_NODES = 5_000_000
#: The api-backed oracle over the same puzzles. Measured: ~8 s.
REFERENCE_SECONDS = 300.0


def test_dfpn_solves_the_friend_set_within_its_budget() -> None:
    total_nodes = 0
    total_seconds = 0.0
    rows: list[str] = []
    for fixture in FRIEND:
        started = time.perf_counter()
        result = solve_tsume(fixture.fen, fixture.tsume_plies)
        elapsed = time.perf_counter() - started
        assert result.verdict == "MATE", f"{fixture.name}: {result.verdict}"
        total_nodes += result.nodes
        total_seconds += elapsed
        rate = result.nodes / elapsed if elapsed > 0 else float("inf")
        rows.append(
            f"{fixture.name:<12} plies={fixture.tsume_plies} "
            f"nodes={result.nodes:>7} {elapsed:>7.3f}s {rate:>12,.0f} nodes/s"
        )

    print("\ndf-pn tsume solver, friend set:")
    for row in rows:
        print("  " + row)
    overall = total_nodes / total_seconds if total_seconds > 0 else float("inf")
    print(
        f"  {'TOTAL':<12} nodes={total_nodes:>7} {total_seconds:>7.3f}s "
        f"{overall:>12,.0f} nodes/s"
    )

    assert total_seconds < DFPN_SECONDS, total_seconds
    assert total_nodes < DFPN_NODES, total_nodes


def test_the_reference_solver_stays_inside_its_own_budget() -> None:
    """Timed separately: the oracle's cost is not the solver's cost."""
    started = time.perf_counter()
    for fixture in FRIEND:
        assert reference_solve(fixture.fen, fixture.tsume_plies) == "MATE", fixture.name
    elapsed = time.perf_counter() - started
    print(f"\nreference (api.Gungi) friend set: {elapsed:.2f}s")
    assert elapsed < REFERENCE_SECONDS, elapsed
