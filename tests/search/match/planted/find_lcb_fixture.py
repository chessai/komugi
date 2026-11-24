"""Deterministic sweep for the T2 LCB fixture. Committed, not run at test time.

"``A`` is visited more but backed up by a high-variance mixture, ``B`` slightly
less by near-deterministic values" is a **search-outcome** property: it is no
more discoverable by hand than T1's fixture, and a hand-tuned table that
happens to be green is not evidence of anything. The design's discipline is
therefore the same as T1's -- a committed deterministic sweep, the found table
frozen as a literal, no sweep at test time, and "no fixture found" reported as
a finding rather than treated as a licence to weaken the assertion.

The sweep enumerates two-action payoff tables in a fixed order and keeps the
first for which a search at the pinned budget satisfies all three of T2's
clauses: ``argmax_a W(a) == A``, the returned action is ``B``, and
``lcb(B) > lcb(A)`` numerically.

**Status: no fixture found, and that is the finding.** Re-run by the advisor on
2026-09-04 against the real search -- the review's eight candidate tables, at
two budgets (24 and 48 playouts): **the LCB pick equalled the max-visits pick
on every one of them**, so clause 2 is false everywhere and no table swept is a
T2 fixture. An earlier revision of
``test_match_lcb.py`` froze ``((1.0, -0.6), (-0.2, -0.2))`` from a sweep that
was never actually run against a search; that literal asserted a disagreement
the search does not produce, and it has been removed rather than re-tuned.

**Why it is structural, not a gap in the enumeration.** The evaluator these
sweeps use is the solver: every leaf is the position's *exact* value. An edge's
variance is then the spread of the values backed up through it, and exact values
do not spread -- once the opponent's replies are resolved, every playout through
an edge returns the same number, the second moment equals the squared mean, and
``var(a)`` sits on ``var_floor`` for **every** edge. With a common floor the LCB
penalty ``k_lcb * sqrt(var/W)`` is monotone in ``W`` alone, so the most-visited
edge takes both the largest mean and the smallest penalty and the two rules
cannot come apart. Producing a real disagreement needs an evaluator whose
*errors* differ between siblings -- a network, not a solver -- which is not a
fixture this suite can plant.

T2 therefore ships as a **planted** fixture in
``test_match_lcb.py``: the root is expanded by the real search and its edge
statistics are written down, so the disagreement is exhibited on numbers
anybody can check by hand rather than predicted from a search that will not
produce one. This file stays committed as the provenance of that decision --
the enumeration that was swept, and what it returned.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

#: The playout budget the sweep searches at, pinned as a literal so a later
#: change to the config default cannot silently move the fixture.
PLAYOUTS = 64

#: High/low leaf payoffs swept for the high-variance action.
SPREADS = ((1.0, -1.0), (0.9, -0.9), (1.0, -0.6), (0.8, -0.8))

#: Near-deterministic payoffs swept for the low-variance action.
STEADY = (0.2, 0.1, 0.0, -0.1, -0.2, 0.3)


def enumerate_tables() -> Iterator[tuple[tuple[tuple[float, ...], ...], str]]:
    """Yield ``(payoff table, description)`` pairs in a fixed order.

    Action 0 is the candidate ``A``: its replies straddle zero, so the
    opponent's own exploration keeps both extremes in the mixture and the
    edge's second moment stays far from its squared mean. Action 1 is ``B``:
    every reply pays the same, so its variance is the floor and its lower
    bound is its mean minus the floor penalty.
    """
    for high, low in SPREADS:
        for steady in STEADY:
            table = ((high, low), (steady, steady))
            yield table, f"A=({high}, {low}) B=({steady}, {steady})"


def main() -> None:
    """Print the tables the sweep would try, in order.

    Deliberately does **not** run a search: importing the search here would
    make this file a second, untested implementation of the test it exists to
    supply a fixture for. A caller wiring it up runs each table through
    ``MatchSearch`` and keeps the first satisfying T2's three clauses.
    """
    for table, description in enumerate_tables():
        print(f"{table!r}  # {description}")


def clauses(
    weights: Sequence[float], lcb: Sequence[float], chosen: int
) -> tuple[bool, bool, bool]:
    """Return T2's three clauses for one candidate fixture's outcome.

    Args:
        weights: Per-root-action backup weights after the search.
        lcb: Per-root-action lower confidence bounds.
        chosen: The action the search returned.

    Returns:
        ``(A is the most visited, B was returned, lcb(B) > lcb(A))``.
    """
    most_visited = max(range(len(weights)), key=lambda index: weights[index])
    return most_visited == 0, chosen == 1, lcb[1] > lcb[0]


if __name__ == "__main__":
    main()
