"""Deterministic enumeration of the T13 tactical fixtures. Committed, not run.

"Has a unique solver-optimal root move" is a property of a payoff table, and
"a search at 64 playouts finds it" is a property of the *search*. The design
pre-registers the provenance discipline that keeps the two apart: the fixtures
are produced by a committed deterministic enumeration, frozen as literals in
the test, and the test does **not** run the enumeration. Fewer than twenty
fixtures found is a finding, not a licence to lower the count.

The enumeration below is the rule the frozen fixtures were written to, stated
as code so a reader can check the literals against it rather than against a
paragraph:

1. The best row is **constant** at ``best``, so every leaf under it backs up
   the same value and its edge variance is zero. That is what makes the LCB
   penalty on the winning move the floor term ``k_lcb * var_floor / sqrt(W)``
   and nothing else, which in turn is what makes "the search returns it" a
   consequence of the design's own constants rather than a hope about visit
   distributions.
2. **Every** entry of every other row is at most ``best - margin``. So no
   other row's *mean* can reach the best row's value however the opponent
   distributes its visits, and the unique minimax argmax is the constant row
   by construction.
3. At least one other row is a **trap**: its entries are as high as rule 2
   allows except for one much lower entry, so a search that fails to let the
   opponent answer prefers it. A table of flat rows would be won by a search
   that never looked past ply one.

Run as ``python find_tactical_fixtures.py`` to print the tables.
"""

from __future__ import annotations

from collections.abc import Iterator

#: How far below the best row every other entry must sit. Large enough that
#: the LCB floor penalty (``5 * 0.05 / sqrt(W)``, i.e. 0.25/sqrt(W)) cannot
#: close the gap at any weight the eligibility rule admits.
MARGIN = 0.4

#: The best-row values swept, in order.
BEST_VALUES = (0.9, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2)

#: The widths swept, in order.
WIDTHS = (3, 4)


def _trap_row(width: int, high: float, low: float) -> tuple[float, ...]:
    """Return a row that looks good everywhere but one reply."""
    return (*(high for _ in range(width - 1)), low)


def _flat_row(width: int, value: float) -> tuple[float, ...]:
    """Return a row with the same payoff against every reply."""
    return tuple(value for _ in range(width))


def enumerate_fixtures() -> Iterator[tuple[tuple[tuple[float, ...], ...], int]]:
    """Yield ``(payoff table, unique best root action)`` pairs, in fixed order.

    The order is the product of :data:`WIDTHS`, :data:`BEST_VALUES` and the
    position of the best row, iterated smallest first. No randomness, so two
    runs enumerate the same tables in the same order.
    """
    for width in WIDTHS:
        for best in BEST_VALUES:
            ceiling = round(best - MARGIN, 3)
            for best_row in range(width):
                rows: list[tuple[float, ...]] = []
                for index in range(width):
                    if index == best_row:
                        rows.append(_flat_row(width, best))
                    elif index % 2 == 0:
                        rows.append(
                            _trap_row(width, ceiling, round(max(-1.0, ceiling - 0.5), 3))
                        )
                    else:
                        rows.append(
                            _flat_row(width, round(max(-1.0, ceiling - 0.2), 3))
                        )
                yield tuple(rows), best_row


def row_minima(table: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Return ``min_b payoffs[a][b]`` per row -- the exact value of each move."""
    return tuple(min(row) for row in table)


def is_unique_best(table: tuple[tuple[float, ...], ...], action: int) -> bool:
    """Return whether `action` is the strictly best root move of `table`."""
    minima = row_minima(table)
    best = minima[action]
    return all(value < best for index, value in enumerate(minima) if index != action)


def main() -> None:
    """Print every fixture the enumeration finds, with its verdict."""
    found = 0
    for table, action in enumerate_fixtures():
        if not is_unique_best(table, action):
            continue
        found += 1
        print(f"{table!r},  # best={action} minima={row_minima(table)}")
    print(f"# {found} fixtures found")


if __name__ == "__main__":
    main()
