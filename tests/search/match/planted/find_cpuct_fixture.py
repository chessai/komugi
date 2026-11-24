"""Deterministic sweep for the T1 dynamic-cPUCT fixture. Committed, not run.

**Status.** Run by the advisor on 2026-09-04, at ``subtree_bias_factor = 0.0``
-- which is what ``test_match_cpuct.py`` pins when it replays the fixture, so
that neither arm is reading design §A2's table. The candidate frozen is the
**third** in the enumeration order below: payoffs ``((1.0, 1.0), (-1.0,
-1.0))`` with logits ``(0.0, 3.0)``, the high prior on action 1, at
:data:`PLAYOUTS`. Its measured outcome is

    dynamic: action 1, visits ``[0, 48]``  /  fixed: action 0, visits
    ``[17, 31]``

and it is frozen as ``T1_TABLE`` / ``T1_LOGITS`` / ``T1_PLAYOUTS`` in
``tests/search/match/test_match_cpuct.py``, which does not run this sweep.

**The first candidate -- the same table at logits ``(0.0, 1.0)`` -- does not
discriminate**, which is why it is no longer the frozen one: at a prior of
``e**1 : 1`` the probe found the clauses no longer separating the two arms, so
the test that carried it was asserting nothing about the feature. A prior mass
of ``3.0`` is what makes "explore the child the values do not support" a
decision the exploration constant is actually left to make.

**And clause 3 as it was originally written is unattainable, on this fixture
and structurally.** ``phi(root) > 1.2`` was to be measured on the dynamic arm
*after* the search, but the between-child spread it reads is destroyed by the
very concentration that makes the arms differ: the dynamic arm here spends its
whole budget on one edge, every backup through that edge is the same ``-1``,
so ``SS(root)`` is exactly ``0`` and ``phi`` lands near ``0.32`` rather than
above ``1.2``. A fixture on which the arms diverge *and* the root ends with
widely disagreeing children would need the dynamic arm to both commit and keep
sampling the child it has rejected. The clause is therefore replaced, here and
in the test, by ``phi != 1.0`` in the dynamic arm -- the scaling is live and
the fixed arm's is exactly ``1.0`` -- and the loss of teeth is recorded rather
than papered over: it is the mechanism clause, and what survives of it is that
the two arms are genuinely running different exploration constants.

The fixture must satisfy two conditions at the root, and each one only became
achievable when the design was corrected:

1. **The root children disagree.** Their backed-up utilities are widely spread
   *between* children -- one row minimaxing to about ``+1``, another to about
   ``-1`` -- so the **between-child** term of ``SS(root)`` is large,
   ``sigma(root)`` far exceeds ``sigma_prior``, and ``phi(root) > 1``. Under a
   within-edge-only sum of squares this condition contributed *nothing*: the
   spread among root children cancelled out, so a sweep written to that recipe
   was hunting a fixture the formula could not produce.
2. **One child carries a high prior but mediocre early backups**, so exploring
   it is a decision the exploration term has to make. That lever only exists
   because the exploration numerator is ``sqrt(1 + W_tot)``: with
   ``sqrt(W_tot)`` the first descent from every node is prior-blind in *both*
   arms, and "a high-prior child" is not something a test can pull on.

The sweep is over small payoff tables crossed with a small set of logit shapes,
in a fixed enumeration order and with no RNG. A found fixture is frozen as a
literal table plus literal logits in the test file, and the test does not run
the sweep. **If the sweep finds no such fixture, that is a finding about the
feature**, reported in ``tickets/KOM-007.md``, and not a reason to weaken T1's
assertions -- which are that the two arms return different actions, that the
dynamic arm spends more on the high-prior child, that ``phi`` is not ``1.0`` in
the dynamic arm, and that it is identically ``1.0`` in the fixed arm. The one
clause that *was* weakened is recorded above, with the structural reason and
the number measured in its place; a clause dropped silently would leave the
suite claiming a mechanism nobody checked.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

#: The playout budget the sweep searches at, pinned as a literal. Lowered from
#: 64 to the budget the discriminating candidate was measured at: a fixture
#: found at one budget is not a fixture at another, and the test pins this
#: number rather than a config default.
PLAYOUTS = 48

#: What ``phi(root)`` must **not** be in the dynamic arm. The original clause
#: was ``phi > 1.2``; see the module docstring for why no discriminating
#: fixture can satisfy it, and what is asserted instead.
FIXED_PHI = 1.0

#: Payoff pairs for the two disagreeing root children, swept in order: the
#: first row minimaxes high, the second low.
DISAGREEMENTS = (
    ((1.0, 1.0), (-1.0, -1.0)),
    ((0.9, 0.8), (-0.9, -0.8)),
    ((1.0, 0.6), (-0.6, -1.0)),
)

#: Logit masses swept onto the high-prior child, in order.
PRIOR_MASSES = (1.0, 2.0, 3.0, 4.0)


def enumerate_fixtures() -> Iterator[
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...], int]
]:
    """Yield ``(payoff table, logits, the high-prior action)``, in fixed order.

    The high-prior child is the one that minimaxes **low**: a prior pointing
    at a child the values do not support is precisely the case where how much
    exploration the node buys decides the move.
    """
    for high_row, low_row in DISAGREEMENTS:
        for mass in PRIOR_MASSES:
            table = (high_row, low_row)
            logits = (0.0, mass)
            yield table, logits, 1


def clauses(
    dynamic_action: int,
    fixed_action: int,
    dynamic_visits: Sequence[int],
    fixed_visits: Sequence[int],
    dynamic_phi: float,
    fixed_phi: float,
    high_prior_action: int,
) -> tuple[bool, bool, bool, bool]:
    """Return T1's four clauses for one candidate fixture's outcome."""
    return (
        dynamic_action != fixed_action,
        dynamic_visits[high_prior_action] > fixed_visits[high_prior_action],
        dynamic_phi != FIXED_PHI,
        fixed_phi == FIXED_PHI,
    )


def main() -> None:
    """Print the candidate fixtures in enumeration order.

    Deliberately does not run a search: a sweep script that reimplemented the
    search would be a second, untested copy of the thing under test.
    """
    for table, logits, high_prior in enumerate_fixtures():
        print(f"{table!r}, logits={logits!r}  # high-prior action {high_prior}")


if __name__ == "__main__":
    main()
