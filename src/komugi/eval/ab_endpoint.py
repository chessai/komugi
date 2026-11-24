"""The KOM-036 A/B endpoint: Wilson interval, decision rule, cross-arm pool.

This is the §1.6/§1.7a half of ``docs/design/kom-036-fast-mode-ab.md``: the
statistic the down-weight-versus-discard A/B is decided on, the three-way rule
that decides it, and the structural guards a cross-arm ladder pool needs before
its numbers mean anything. Everything here is **pre-registered** -- it exists so
that the arithmetic and the decision boundary are fixed in code before either
campaign has played a game, which is what makes the experiment a test rather
than a search.

Three pieces, in the order the experiment uses them:

* :func:`wilson_interval` -- a two-sided score interval on the head-to-head
  score rate. Wilson rather than Wald because the interval must stay inside
  ``[0, 1]`` and stay sane near a sweep. It accepts a **half-integer** success
  count, because the statistic it is fed is ``wins + 0.5 * draws`` and that is
  half-integer whenever ``draws`` is odd.
* :func:`ab_endpoint` -- the pre-registered decision: adopt A, adopt B, or
  inconclusive-and-ship-A, behind a validity guard that voids the result
  outright when fewer than half the games were decisive.
* :func:`namespaced_entrants` / :func:`pool_entrants` /
  :func:`pool_components` / :func:`fit_cross_arm_pool` -- the S1 pool. Two
  campaigns' checkpoint directories cannot simply be concatenated: their
  ``ladder_players`` ids collide at ``ckpt-<step>`` (both arms run the same
  number of steps), and two within-arm chains with no cross-arm games are a
  match graph with **two components**, whose relative offset -- the only
  quantity S1 exists to show -- is an unidentified free parameter of the joint
  likelihood. Both failures are silent; both are refused here, before the fit.

**Conservatism, stated where the code is.** The interval treats the 400 games
as independent Bernoulli trials, ignoring the seat-swapped pair structure
(which can only reduce variance), and feeds a *trinomial* score statistic to a
*binomial* variance ``p(1-p)/n`` (a draw contributes a quarter of a coin flip's
variance, so the binomial interval is too wide). Both approximations are
conservative, they compound rather than cancel, and an over-wide interval is
likelier to straddle ``0.5`` -- so both bias the experiment toward
:data:`INCONCLUSIVE`, and therefore toward the tie-break to arm A. That is a
bias in favour of the prior, and it is recorded before the data so that a
result adopted on the tie-break is read as the weaker evidence it is.

**No games are played here.** This module consumes counts and
:class:`~komugi.eval.records.MatchSet` records, so it imports neither torch nor
an engine; :class:`~komugi.eval.arena.ArenaPlayer` is referenced for typing
only and rebuilt through :func:`dataclasses.replace`, which needs no class
object. Import it by name (``from komugi.eval.ab_endpoint import ab_endpoint``)
the way :mod:`komugi.eval.arena` and :mod:`komugi.eval.ladder` are.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Final

from komugi.eval.elo import EloResult, fit_elo
from komugi.eval.records import MatchSet

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch off this import
    from komugi.eval.arena import ArenaPlayer

CONFIDENCE: Final[float] = 0.95
"""The pre-registered two-sided confidence level for the primary endpoint."""

Z_95: Final[float] = NormalDist().inv_cdf(0.5 + CONFIDENCE / 2.0)
"""The standard-normal quantile the 95% interval inverts the score test at.

Derived from :class:`statistics.NormalDist` rather than typed as a literal
``1.96``: ``scipy`` is not in this environment, the stdlib has the exact
quantile, and a rounded literal would move the decision boundary by more than
the tolerance :func:`wilson_interval`'s own acceptance node checks to.
"""

ENDPOINT_GAMES: Final[int] = 400
"""The pre-registered head-to-head match size: 200 seat-swapped pairs.

At this ``n`` the test detects a score difference of ``0.07`` -- about 49 Elo --
at 80% power. A smaller difference is pre-registered as undetectable and is not
claimed from this experiment.
"""

MIN_DECISIVE_FRACTION: Final[float] = 0.5
"""Fraction of games that must be decisive for the primary result to stand.

Below it the match has fit a score rate out of draws, and the pre-registered
outcome is :data:`VOID` -- no arm is adopted, rather than an arm adopted on a
half-drawn interval.
"""

ADOPT_A: Final[str] = "A"
"""Wilson lower bound above ``0.5``: adopt arm A, the ``0.125`` down-weight."""

ADOPT_B: Final[str] = "B"
"""Wilson upper bound below ``0.5``: adopt arm B, the PCR-style discard."""

INCONCLUSIVE: Final[str] = "inconclusive-default-A"
"""The interval straddles ``0.5``: ship arm A, and record the run as non-discriminating.

The tie-break goes to A because ``docs/DESIGN.html:229`` names the down-weight
*"the Gumbel-native choice"*, i.e. it is the prior. Spelling the tie-break into
the decision's own vocabulary -- rather than returning a bare ``"A"`` -- is what
keeps a run adopted on the tie-break distinguishable, afterwards, from one
adopted on the interval.
"""

VOID: Final[str] = "void"
"""Fewer than :data:`MIN_DECISIVE_FRACTION` of the games were decisive."""

DECISIONS: Final[frozenset[str]] = frozenset({ADOPT_A, ADOPT_B, INCONCLUSIVE, VOID})
"""Every value :attr:`ABEndpoint.decision` can take."""

ARM_A_PREFIX: Final[str] = "A/"
"""Id namespace for arm A's rungs in the S1 pool."""

ARM_B_PREFIX: Final[str] = "B/"
"""Id namespace for arm B's rungs in the S1 pool."""

REFERENCE_ID: Final[str] = "ref/soak49-15000"
"""The shared reference entrant, and the S1 pool's gauge at 0 Elo.

A third, fixed net that plays **both** arms under one identical protocol -- a
common opponent, not a curve comparison, so §1.7's soak49 trap does not apply.
Anchoring on an arm's own checkpoint is rejected: it would make the reported
inter-arm offset depend on which arm was named.
"""


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """A two-sided Wilson score interval, with the point estimate it is around.

    ``low`` and ``high`` are the two roots of ``(p_hat - p)**2 ==
    z**2 * p * (1 - p) / n`` in ``p``: the set of null score rates the observed
    ``p_hat`` does not reject at the score test's level. That definition, not
    the closed form below it, is what the interval *is*, and it is the one the
    acceptance node checks against.
    """

    point: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        """Return whether `value` lies within the closed interval."""
        return self.low <= value <= self.high


def wilson_interval(
    successes: float, trials: int, z: float = Z_95
) -> WilsonInterval:
    """Return the two-sided Wilson score interval for `successes` of `trials`.

    Args:
        successes: The score, which may be **half-integer**: the endpoint's
            statistic is ``wins + 0.5 * draws``. Must lie in ``[0, trials]``.
        trials: Number of games. Must be positive.
        z: The standard-normal quantile to invert the score test at, defaulting
            to :data:`Z_95`.

    Returns:
        The interval, whose bounds are inside ``[0, 1]`` by construction --
        which is the property Wald's ``p_hat +- z * SE`` loses at extreme
        ``p_hat`` and the reason this is the pre-registered statistic.

    Raises:
        ValueError: On a non-positive `trials`, a `successes` outside
            ``[0, trials]``, or a non-positive `z`.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials!r}")
    if not 0.0 <= successes <= float(trials):
        raise ValueError(
            f"successes must lie in [0, {trials}], got {successes!r}"
        )
    if z <= 0.0:
        raise ValueError(f"z must be positive, got {z!r}")
    count = float(trials)
    point = successes / count
    z_sq = z * z
    denominator = 1.0 + z_sq / count
    centre = (point + z_sq / (2.0 * count)) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(point * (1.0 - point) / count + z_sq / (4.0 * count * count))
    )
    # Analytically ``centre -+ spread`` already lies in [0, 1] -- that is the
    # property Wald loses and this statistic is chosen for. The clamp removes
    # rounding overshoot of order 1e-18 at the degenerate ends (``successes``
    # of 0 or ``trials``, where the exact bound *is* 0 or 1) so that "inside
    # the unit interval" is true of the returned floats and not only of the
    # algebra.
    return WilsonInterval(
        point=point,
        low=max(0.0, centre - spread),
        high=min(1.0, centre + spread),
    )


@dataclass(frozen=True, slots=True)
class ABEndpoint:
    """The primary endpoint, computed: the counts, the interval, the decision.

    Every field is kept rather than only the decision, because the validity
    guard and the two approximations documented at module level are all read
    off the counts, and a decision reported without them cannot be re-checked.
    """

    games: int
    wins_a: int
    draws: int
    losses_a: int
    decisive: int
    score: float
    interval: WilsonInterval
    decision: str


def ab_endpoint(
    wins_a: int, draws: int, games: int, *, z: float = Z_95
) -> ABEndpoint:
    """Apply the pre-registered three-way decision rule to a head-to-head match.

    The rule, fixed before any data (§1.6):

    ==============================  ==========================================
    Condition                       Decision
    ==============================  ==========================================
    ``decisive < games / 2``        :data:`VOID` -- no arm is adopted.
    Wilson lower bound ``> 0.5``    :data:`ADOPT_A`
    Wilson upper bound ``< 0.5``    :data:`ADOPT_B`
    interval contains ``0.5``       :data:`INCONCLUSIVE`
    ==============================  ==========================================

    The validity guard is checked **first**, and deliberately: a match that is
    mostly draws has an interval, and reading a decision off it would be
    reading the tie-break's bias rather than the arms.

    Args:
        wins_a: Games arm A won.
        draws: Drawn games.
        games: Total games played; :data:`ENDPOINT_GAMES` in the real run.
        z: Passed through to :func:`wilson_interval`.

    Returns:
        The :class:`ABEndpoint`, whose ``decision`` is one of
        :data:`DECISIONS`.

    Raises:
        ValueError: On a non-positive `games`, a negative count, or counts
            whose sum exceeds `games`.
    """
    if games <= 0:
        raise ValueError(f"games must be positive, got {games!r}")
    if wins_a < 0 or draws < 0:
        raise ValueError(
            f"counts must be non-negative, got wins_a={wins_a!r}, draws={draws!r}"
        )
    if wins_a + draws > games:
        raise ValueError(
            f"wins_a + draws must not exceed games, got {wins_a!r} + {draws!r} "
            f"> {games!r}"
        )
    losses_a = games - wins_a - draws
    decisive = games - draws
    interval = wilson_interval(wins_a + 0.5 * draws, games, z)
    if decisive < games * MIN_DECISIVE_FRACTION:
        decision = VOID
    elif interval.low > 0.5:
        decision = ADOPT_A
    elif interval.high < 0.5:
        decision = ADOPT_B
    else:
        decision = INCONCLUSIVE
    return ABEndpoint(
        games=games,
        wins_a=wins_a,
        draws=draws,
        losses_a=losses_a,
        decisive=decisive,
        score=interval.point,
        interval=interval,
        decision=decision,
    )


def namespaced_entrants(
    entrants: Sequence[ArenaPlayer], prefix: str
) -> tuple[ArenaPlayer, ...]:
    """Return `entrants` with `prefix` prepended to every player id.

    The id the ladder minted is **prefixed**, not rebuilt: an
    :class:`~komugi.eval.arena.ArenaPlayer` does not carry ``ref.step`` -- the
    step survives only inside the ``ckpt-<step>`` string -- so constructing a
    fresh ``Player`` would force the caller to parse the step back out of the
    id, and would drop ``family`` unless it restated it, changing the pooled
    set's family mid-fit. Prefixing preserves the family and needs no step
    recovery.

    Args:
        entrants: The rungs of one arm, as :func:`komugi.eval.arena.ladder_players`
            returns them.
        prefix: The arm's namespace, e.g. :data:`ARM_A_PREFIX`.

    Returns:
        The same entrants, in the same order, with namespaced ids.

    Raises:
        ValueError: If `prefix` is empty or blank.
    """
    if not prefix or not prefix.strip():
        raise ValueError(f"prefix must be a non-empty namespace, got {prefix!r}")
    return tuple(
        dataclasses.replace(
            entrant,
            player=dataclasses.replace(
                entrant.player, id=f"{prefix}{entrant.player.id}"
            ),
        )
        for entrant in entrants
    )


def require_distinct_ids(ids: Sequence[str]) -> None:
    """Raise if `ids` holds a repeat, naming every id that repeats.

    This reproduces :func:`komugi.eval.ladder.run_ladder`'s duplicate refusal
    rather than inheriting it: that guard lives *inside* ``run_ladder``, and a
    hand-rolled cross-arm driver gets none of it. Without this, arm A's
    ``ckpt-6000`` and arm B's ``ckpt-6000`` pool into **one** player --
    :func:`komugi.eval.elo.fit_elo` keys ratings by id and cannot tell them
    apart -- and the result is conflation rather than an error.

    Raises:
        ValueError: If any id appears more than once.
    """
    seen: set[str] = set()
    repeated: list[str] = []
    for identifier in ids:
        if identifier in seen and identifier not in repeated:
            repeated.append(identifier)
        seen.add(identifier)
    if repeated:
        raise ValueError(
            f"the pooled entrants share {len(repeated)} id(s): {repeated}. "
            f"Both arms run the same number of steps, so their ladder ids "
            f"collide at 'ckpt-<step>'; namespace each arm with "
            f"namespaced_entrants() before pooling"
        )


def pool_entrants(*groups: Sequence[ArenaPlayer]) -> tuple[ArenaPlayer, ...]:
    """Concatenate entrant groups into one pool, refusing colliding ids.

    Args:
        *groups: The per-arm entrant sequences, plus the shared reference
            entrant's group, in the order they should appear in the pool.

    Returns:
        Every entrant, in group order then within-group order.

    Raises:
        ValueError: If two entrants share a player id
            (:func:`require_distinct_ids`).
    """
    pooled = tuple(entrant for group in groups for entrant in group)
    require_distinct_ids([entrant.player.id for entrant in pooled])
    return pooled


def cross_arm_schedule(
    arm_a_ids: Sequence[str],
    arm_b_ids: Sequence[str],
    reference_id: str = REFERENCE_ID,
) -> tuple[tuple[str, str], ...]:
    """Return the matchups that make a pooled two-arm ladder identified.

    Three links per rung index ``k``, and each of them is load-bearing:

    * ``A_k`` versus ``B_k`` -- the explicit cross-arm link. This is what makes
      the pooled graph connected and the inter-arm offset identified, and it is
      both the shortest path between the two curves and the most interpretable
      rung of the report.
    * ``A_k`` versus the reference and ``B_k`` versus the reference -- the
      common opponent that pins the gauge on a net neither arm produced.

    The reference alone would also connect the graph; the direct same-iteration
    links are scheduled anyway for the reason above.

    Args:
        arm_a_ids: Arm A's namespaced rung ids, oldest first.
        arm_b_ids: Arm B's namespaced rung ids, oldest first, one per rung
            index of `arm_a_ids`.
        reference_id: The shared reference entrant's id.

    Returns:
        The matchups as ``(first, second)`` id pairs, in schedule order. Give
        each its own seed with :func:`komugi.eval.ladder.matchup_seed` and its
        own index -- a hand-rolled driver inherits ``run_ladder``'s per-matchup
        seeding no more than it inherits its duplicate refusal, and one seed
        across matchups fits the joint Elo on correlated games.

    Raises:
        ValueError: If the two arms have different rung counts, if either is
            empty, or if `reference_id` is blank.
    """
    if not reference_id.strip():
        raise ValueError("reference_id must be a non-empty player id")
    if not arm_a_ids:
        raise ValueError("a cross-arm pool needs at least one rung per arm")
    if len(arm_a_ids) != len(arm_b_ids):
        raise ValueError(
            f"the arms must be sampled at the same rung indices, got "
            f"{len(arm_a_ids)} and {len(arm_b_ids)} rungs"
        )
    schedule: list[tuple[str, str]] = []
    for first, second in zip(arm_a_ids, arm_b_ids, strict=True):
        schedule.append((first, second))
        schedule.append((first, reference_id))
        schedule.append((second, reference_id))
    return tuple(schedule)


def pool_components(match_set: MatchSet) -> tuple[frozenset[str], ...]:
    """Return the connected components of `match_set`'s comparison graph.

    Players are vertices and a game is an edge. Checked **before** fitting,
    and cheaply: a disconnected pool still converges -- each component finds
    its own gauge -- and reports ``converged=True``, so the fit itself cannot
    be used as the detector. The joint Davidson likelihood is invariant to an
    arbitrary offset between components, which is exactly the quantity a
    cross-arm pool exists to report.

    Returns:
        The components, each a frozenset of player ids, ordered by their
        smallest id so the result is deterministic.
    """
    parent: dict[str, str] = {player: player for player in match_set.players()}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for record in match_set.records:
        left, right = find(record.white), find(record.black)
        if left != right:
            parent[left] = right

    grouped: dict[str, set[str]] = {}
    for player in parent:
        grouped.setdefault(find(player), set()).add(player)
    return tuple(
        frozenset(members)
        for _, members in sorted(
            grouped.items(), key=lambda item: min(item[1])
        )
    )


def cross_arm_pool_is_reportable(result: EloResult, match_set: MatchSet) -> bool:
    """Return whether an S1 fit may be reported at all.

    The pre-registered refusal: a pool that did not converge, or whose graph
    has more than one component, is **not reported** -- not reported with a
    caveat. :class:`~komugi.eval.elo.EloResult` carries no dispersion, so there
    is no interval to express the uncertainty with, and publishing an
    unidentified offset as a figure is the failure this refusal prevents.
    """
    return result.converged and len(pool_components(match_set)) == 1


def fit_cross_arm_pool(
    match_set: MatchSet,
    *,
    anchor: str = REFERENCE_ID,
    expected_ids: Sequence[str] | None = None,
) -> EloResult:
    """Fit one Elo table to a pooled two-arm ladder, refusing what S1 refuses.

    Both guards run before :func:`komugi.eval.elo.fit_elo` is called, because
    both failures are silent afterwards: colliding ids conflate two players
    into one row, and a disconnected graph converges on two independent gauges.

    Args:
        match_set: Every game of the pool, from both arms and the reference.
        anchor: The player pinned to 0 Elo, defaulting to :data:`REFERENCE_ID`.
        expected_ids: If given, the full entrant roster -- checked for
            duplicates, and checked to have actually played. An entrant that
            appears in no game is absent from ``ratings`` entirely, which a
            caller reading the table by index would not notice.

    Returns:
        The fit. ``converged=False`` still means "not identified from this
        data" and must not be reported; see
        :func:`cross_arm_pool_is_reportable`.

    Raises:
        ValueError: On duplicate `expected_ids`, on an expected id that played
            no games, on a comparison graph with more than one component, or on
            anything :func:`komugi.eval.elo.fit_elo` refuses.
    """
    played = set(match_set.players())
    if expected_ids is not None:
        require_distinct_ids(expected_ids)
        missing = [name for name in expected_ids if name not in played]
        if missing:
            raise ValueError(
                f"{len(missing)} pooled entrant(s) played no games: {missing}; "
                f"they would be absent from the fitted ratings rather than "
                f"reported as unrated"
            )
    components = pool_components(match_set)
    if len(components) > 1:
        summary = [sorted(component) for component in components]
        raise ValueError(
            f"the pooled match graph has {len(components)} components: "
            f"{summary}. Two within-arm chains with no cross-arm games leave "
            f"the offset between the curves -- the only quantity this pool "
            f"exists to show -- an unidentified free parameter of the joint "
            f"likelihood; schedule the links cross_arm_schedule() returns"
        )
    return fit_elo(match_set, anchor=anchor)
