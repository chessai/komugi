"""§D5's in/cross divergence stop-gate, as a predicate over pool records.

``docs/DESIGN.html:260`` pre-registers this gate in one sentence -- *"sustained
divergence between them is a pre-registered stop-and-diagnose gate"*. One
sentence is not a predicate; this module is the predicate.

The statistic
=============

Fix a milestone window ``W = (A, B)``. Three fits, all over the **pool's own**
records and all sub-fits of one id space (§D5.1):

* ``G_in``  -- over komugi x komugi records **with no ``#`` suffix on either
  side**, i.e. exactly §D2.2(b)1's chain of 256-game in-family links;
* ``G_cross`` -- over every other pool record except §D8's visits legs: the
  anchor links, the bracket links, the External legs, and the ``#anchor400``
  tie-in, which is komugi x komugi by family but is a *configuration* link
  rather than a training link;
* ``G_cross_no_external`` -- the same set with the External records dropped.

``D = G_in - G_cross`` is published; **``D_no_external = G_in -
G_cross_no_external`` is the gate's statistic** (§D5.1, rev 3). A joint Davidson
MLE propagates every edge, and the External vertex is a pendant on ``#anchor400``
which is a pendant on the milestone, so a site-side change nothing in this
repository would notice (§0.2a, U4) moves ``r_cross(B)``. Reading the gate off
the without-External statistic is what keeps the primary series from being
declared broken on the strength of a >=50-game leg whose rules are unverified.
A row where the two disagree about firing is an **External-drift finding**,
reported as such, and never a silent change of statistic.

Each gain is a difference **within one fit**, so each is invariant to that
fit's own anchor -- which is the decision the whole gate turns on: ``G_in`` and
``G_cross`` are directly comparable with no offset, no shared player and no
reconciliation step. What is never done is pooling the two record sets and
subtracting (§D4.2, §6 node 24).

Sign convention, because the two directions mean opposite things. ``D > 0`` is
**in-family inflation** -- the lineage says it improved more than the outside
world agrees with, which is v1's failure verbatim (``DESIGN.html:168``: the
gauntlet read 1.000 while the engine lost every game to a novice bot). ``D < 0``
is in-family under-reading, the signature of a lineage that has stopped
exploring. Both fire the gate.

The thresholds are not parameters
=================================

:data:`THRESHOLD_SINGLE_ELO` and :data:`THRESHOLD_SUSTAINED_ELO` are frozen
before any window is measured and take no argument, keyword or attribute to
override -- the ``P4_TOLERANCE`` idiom (``r5.py:100-101``, ``:392-399``:
*"The tolerance is **not** a parameter"*). A ``threshold=`` keyword is precisely
how a pre-registration gets quietly retuned after its first failure, and
``r5.py:429-436`` already names re-running until a gate passes as selection on
the gate.

Not evaluable is not a pass
===========================

Six validity conditions (§D5.3), and a window failing any of them is reported
as ``not_evaluable`` **with the failing condition named**. "The instrument
could not see" and "the instrument saw nothing" are different states, and the
only way they stay different is if the first one has a name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from komugi.eval.ab_endpoint import pool_components
from komugi.eval.elo import fit_elo
from komugi.eval.interval import (
    MAX_DISCARD_RATE,
    REPLICATES,
    BootstrapResult,
    Statistic,
    pair_bootstrap,
)
from komugi.eval.pool import (
    KOMUGI_FAMILY,
    EXTERNAL_FAMILY,
    PoolRoster,
    base_entrant_id,
    family_of,
    is_visits_variant,
    variant_of,
)
from komugi.eval.records import WHITE_WIN, GameRecord, MatchSet

THRESHOLD_SINGLE_ELO: Final[float] = 75.0
"""§D5.3's single-epoch threshold, frozen before any window was measured.

Rev 2 replaced rev 1's 50 on arithmetic, not taste: ``SE(D)`` runs 33-48 Elo
over the design's own SE chain, so the 95% interval excludes 0 only at
``|D| >~ 66-94``, and a 50-Elo pre-registration would have been fired by the
interval condition at ~0.2 power. This gate is honestly a **large-effect
alarm**: ~80% power at ``|D| ~ 100-150``, not at 50. Never widened after
seeing data (``r5.py:100-101``).
"""

THRESHOLD_SUSTAINED_ELO: Final[float] = 90.0
"""§D5.3's threshold on the three-epoch window, which costs no extra games.

A sustained per-epoch divergence accumulates -- over ``(E-3, E)`` the true
statistic is ``~3d`` while ``SE`` grows only as ``sqrt(3)`` on the in-family
leg -- so a sustained 50 Elo/epoch fires at 0.75-0.91 power. This is also the
closest reading of ``DESIGN.html:260``'s word *"sustained"*.
"""

GATE_STATISTIC: Final[str] = "d_no_external"
"""Named in the record, never inferred (§D5.1, §D9.2)."""

PUBLISHED_STATISTIC: Final[str] = "d"
"""Reported beside the gate's statistic, and never the predicate."""

MIN_GPU_PATH_GAMES: Final[int] = 128
"""§D5.3 condition 4: ``DESIGN.html:260``'s ``n >= 128``, per matchup (§D2.1)."""

MIN_DECISIVE: Final[int] = 40
"""§D5.3 condition 4: the anchor-match protocol's own decisive floor.

An all-draws matchup *"has fit an Elo gap out of nothing"* (``arena.py:367-380``).
"""

INFORMATIVE_BAND: Final[tuple[float, float]] = (0.15, 0.85)
"""§D5.3 condition 6: the score-rate band a link is still informative in.

Where the SE chain is within ~1.5x of its ``p = 0.5`` value
(``sqrt(0.25/0.1275) = 1.40``), so inside it §D5.3's ``sigma ~ 33-48`` and its
power table are approximately the numbers pre-registered, and outside it they
are not.
"""

MIN_INFORMATIVE_LINKS: Final[int] = 2
"""§D5.3 condition 6: each endpoint must retain at least this many.

Two entrants at >=0.85 against each of three links is the regime
``DESIGN.html:168`` records v1 dying in (*"saturated at 0.97+ by iter 11 --
blind for the entire second half"*); this condition is the pool's refusal to be
blind quietly.
"""

SINGLE_WINDOW_SPAN: Final[int] = 1
"""``(E-1, E)``: the single-epoch window."""

SUSTAINED_WINDOW_SPAN: Final[int] = 3
"""``(E-3, E)``: the sustained window. No third window is ever formed."""

CONDITION_WINDOW_NOT_FORMED: Final[str] = "window not yet formed"
"""§D2.2/§D5.1 at ``E = 1``: there is no predecessor, so there is no window."""

CONDITION_CONVERGED: Final[str] = "a fit did not converge"
"""Condition 1. ``converged=False`` means the ratings are clamps (``elo.py:70-77``)."""

CONDITION_COMPONENTS: Final[str] = "comparison graph is not connected"
"""Condition 2, over ``ab_endpoint.pool_components``."""

CONDITION_BOOK_ERA: Final[str] = "book era boundary"
"""Condition 3: a gain measured across an opening-set change confounds strength
with opening set (§D3.3)."""

CONDITION_GAMES_AND_DECISIVE: Final[str] = "link below 128 games or 40 decisive"
"""Condition 4, over the endpoints' **GPU-path** links only."""

CONDITION_DISCARD_RATE: Final[str] = "bootstrap discard rate above 10%"
"""Condition 5 (§D5.2's pre-registered tolerance for the narrowing bias)."""

CONDITION_LINKS_SATURATED: Final[str] = "links saturated"
"""Condition 6: the instrument has stopped being able to see (§D5.3, rev 3)."""


class Verdict(StrEnum):
    """The three states a window can be in. ``not_evaluable`` is not a pass."""

    OK = "ok"
    FIRED = "fired"
    NOT_EVALUABLE = "not_evaluable"


class WindowKind(StrEnum):
    """The two windows, both evaluated every epoch; either firing fires the gate."""

    SINGLE = "single"
    SUSTAINED = "sustained"


def threshold_for(kind: WindowKind) -> float:
    """Return `kind`'s pre-registered threshold in Elo.

    A pure function of the window kind, decided before any window existed.
    There is deliberately no argument that changes what it returns.
    """
    if kind is WindowKind.SINGLE:
        return THRESHOLD_SINGLE_ELO
    return THRESHOLD_SUSTAINED_ELO


def window_span(kind: WindowKind) -> int:
    """Return how many pool epochs `kind`'s window spans."""
    if kind is WindowKind.SINGLE:
        return SINGLE_WINDOW_SPAN
    return SUSTAINED_WINDOW_SPAN


def window_for(
    roster: PoolRoster, *, epoch: int, kind: WindowKind
) -> tuple[str, str] | None:
    """Return the window's ``(A, B)`` milestones, or ``None`` if not yet formed.

    The windows are ``(E-1, E)`` and ``(E-3, E)``, both evaluated every epoch,
    mechanically. A window whose far endpoint has not entered the pool yet is
    **not formed**, which is a named state and not a pass (§D5.1 at ``E = 1``).
    """
    later = roster.milestone_at(epoch)
    earlier = roster.milestone_at(epoch - window_span(kind))
    if later is None or earlier is None:
        return None
    return (earlier.id, later.id)


def book_era(book_id: str) -> str:
    """Return the book **era** a ``book_id`` belongs to.

    §D3.2's entry ids are ``E<NNN>-<sha8>``, prefixed with the epoch precisely
    so a stored record set cannot silently mix two book epochs
    (``records.py:160-162``). Anything else is its own era.
    """
    prefix, separator, _ = book_id.partition("-")
    if separator and prefix.startswith("E") and prefix[1:].isdigit():
        return prefix
    return book_id


def _touches(record_players: tuple[str, str], milestone: str) -> bool:
    """Return whether a matchup touches `milestone` or a variant of it.

    Conditions 3 and 4 are stated over the endpoints' links, and the External leg
    reaches the endpoint through its ``#anchor400`` variant (§D6.1(b)) -- so
    the entrant *group* (the same weights under any configuration), not the
    bare id, is what "the endpoint's links" means.
    """
    return any(base_entrant_id(player) == milestone for player in record_players)


@dataclass(frozen=True, slots=True)
class WindowRecords:
    """§D5.1's partition of the pool's records for one window.

    The three sets are built by construction rather than by a reader's care:
    the ``#`` suffix is what separates evidence about **configuration** from
    evidence about **training**, and the visits legs are excluded from both
    because §D8.2 forbids citing them as rated ratings and because they exist
    for only one milestone in three.
    """

    in_family: MatchSet
    cross: MatchSet
    cross_no_external: MatchSet
    visits: MatchSet


def partition_records(
    match_set: MatchSet, roster_map: Mapping[str, str]
) -> WindowRecords:
    """Partition `match_set` into §D5.1's record sets.

    Family is read from `roster_map`, never parsed from an id's prefix
    (§D1.2 property 3, :func:`komugi.eval.pool.family_of`).

    Raises:
        ValueError: if a record names a player the roster map does not carry.
    """
    in_family: list[GameRecord] = []
    cross: list[GameRecord] = []
    cross_no_external: list[GameRecord] = []
    visits: list[GameRecord] = []
    for record in match_set.records:
        players = (record.white, record.black)
        families = tuple(family_of(roster_map, player) for player in players)
        if any(is_visits_variant(player) for player in players):
            visits.append(record)
            continue
        both_komugi = all(family == KOMUGI_FAMILY for family in families)
        unsuffixed = all(variant_of(player) is None for player in players)
        if both_komugi and unsuffixed:
            in_family.append(record)
            continue
        cross.append(record)
        if EXTERNAL_FAMILY not in families:
            cross_no_external.append(record)
    return WindowRecords(
        in_family=MatchSet.of(in_family),
        cross=MatchSet.of(cross),
        cross_no_external=MatchSet.of(cross_no_external),
        visits=MatchSet.of(visits),
    )


@dataclass(frozen=True, slots=True)
class WindowGains:
    """The three gains of one window, plus whether every sub-fit converged."""

    g_in: float | None
    g_cross: float | None
    g_cross_no_external: float | None
    converged: bool

    @property
    def d(self) -> float | None:
        """``G_in - G_cross``: published, and never the predicate."""
        if self.g_in is None or self.g_cross is None:
            return None
        return self.g_in - self.g_cross

    @property
    def d_no_external(self) -> float | None:
        """``G_in - G_cross_no_external``: the gate's own statistic."""
        if self.g_in is None or self.g_cross_no_external is None:
            return None
        return self.g_in - self.g_cross_no_external


def _gain(match_set: MatchSet, window: tuple[str, str]) -> tuple[float | None, bool]:
    """Return ``(r(B) - r(A), converged)`` from one fit over `match_set`.

    The sub-fit's anchor is the alphabetically first player that actually
    played, chosen deterministically and never re-derived from the ratings
    (``ladder.py:218-241``'s lesson). The returned quantity is a difference
    within this fit, so it is invariant to that choice.
    """
    earlier, later = window
    if len(match_set) == 0:
        return (None, False)
    players = match_set.players()
    if earlier not in players or later not in players:
        return (None, False)
    result = fit_elo(match_set, anchor=players[0])
    if not result.converged:
        return (None, False)
    return (result.ratings[later] - result.ratings[earlier], True)


def window_gains(records: WindowRecords, window: tuple[str, str]) -> WindowGains:
    """Return the window's three gains, from **three** separate fits.

    Exactly three ``fit_elo`` calls, over three record sets that are never
    merged: the failure §D4.2 rejects is pooling two game sets and subtracting,
    and ``D_no_external`` is *refit*, never derived by adjusting ``D``.
    """
    g_in, in_ok = _gain(records.in_family, window)
    g_cross, cross_ok = _gain(records.cross, window)
    g_cross_no_external, no_external_ok = _gain(records.cross_no_external, window)
    return WindowGains(
        g_in=g_in,
        g_cross=g_cross,
        g_cross_no_external=g_cross_no_external,
        converged=in_ok and cross_ok and no_external_ok,
    )


@dataclass(frozen=True, slots=True)
class MatchupFacts:
    """What a window's validity conditions need to know about one matchup."""

    players: tuple[str, str]
    games: int
    decisive: int
    draws: int
    book_eras: tuple[str, ...]
    gpu_path: bool
    score_rate: Mapping[str, float]

    def rate_for(self, player: str) -> float | None:
        """Return `player`'s observed score rate in this matchup."""
        return self.score_rate.get(player)


def summarise_matchups(
    match_set: MatchSet, roster_map: Mapping[str, str]
) -> tuple[MatchupFacts, ...]:
    """Return per-matchup facts, derived from the records themselves.

    Everything the validity conditions read -- game count, decisive count, book
    era, score rate, and whether the matchup was played on the GPU path -- is a
    function of the stored records plus the roster map, so no side table can
    drift out of step with the games that were actually played.
    """
    games: dict[tuple[str, str], list[GameRecord]] = {}
    for record in match_set.records:
        first, second = sorted((record.white, record.black))
        games.setdefault((first, second), []).append(record)
    facts: list[MatchupFacts] = []
    for players, records in sorted(games.items()):
        decisive = sum(1 for record in records if not record.is_draw)
        draws = len(records) - decisive
        scores = {player: 0.0 for player in players}
        for record in records:
            scores[record.white] += record.result
            scores[record.black] += WHITE_WIN - record.result
        eras = tuple(sorted({book_era(record.book_id) for record in records}))
        families = tuple(family_of(roster_map, player) for player in players)
        facts.append(
            MatchupFacts(
                players=players,
                games=len(records),
                decisive=decisive,
                draws=draws,
                book_eras=eras,
                # The External leg is played on the site's clock, not the GPU's,
                # and is the only matchup in the pool that cannot be bought
                # more games by spending GPU (§D2.2(b)4).
                gpu_path=EXTERNAL_FAMILY not in families,
                score_rate={
                    player: scores[player] / len(records) for player in players
                },
            )
        )
    return tuple(facts)


@dataclass(frozen=True, slots=True)
class DivergenceResult:
    """One window's row of §D9.1's divergence block, and the gate's verdict."""

    window_kind: WindowKind
    window: tuple[str, str] | None
    verdict: Verdict
    threshold: float
    condition: str | None = None
    g_in: float | None = None
    g_cross: float | None = None
    g_cross_no_external: float | None = None
    d: float | None = None
    d_no_external: float | None = None
    lo: float | None = None
    hi: float | None = None
    lo_no_external: float | None = None
    hi_no_external: float | None = None
    discard_rate: float | None = None
    bootstrap_wall_clock_s: float | None = None
    bracket_saturated: tuple[str, ...] = ()
    gate_statistic: str = GATE_STATISTIC
    external_drift: bool = False
    """The two statistics disagree about firing: a finding, not a rounding."""

    @property
    def sign(self) -> int | None:
        """Return the sign of the gate's statistic: ``+1`` is in-family inflation."""
        if self.d_no_external is None:
            return None
        if self.d_no_external > 0.0:
            return 1
        if self.d_no_external < 0.0:
            return -1
        return 0

    @property
    def message(self) -> str:
        """Return the verdict as §D9.1 prints it, naming the failing condition."""
        if self.verdict is Verdict.NOT_EVALUABLE:
            return f"not evaluable: {self.condition}"
        return str(self.verdict.value)


def _not_evaluable(
    kind: WindowKind,
    window: tuple[str, str] | None,
    condition: str,
    *,
    gains: WindowGains | None = None,
    bootstrap: BootstrapResult | None = None,
    bracket_saturated: tuple[str, ...] = (),
) -> DivergenceResult:
    """Return a ``not_evaluable`` row that names the condition it failed."""
    return DivergenceResult(
        window_kind=kind,
        window=window,
        verdict=Verdict.NOT_EVALUABLE,
        threshold=threshold_for(kind),
        condition=condition,
        g_in=None if gains is None else gains.g_in,
        g_cross=None if gains is None else gains.g_cross,
        g_cross_no_external=None if gains is None else gains.g_cross_no_external,
        d=None if gains is None else gains.d,
        d_no_external=None if gains is None else gains.d_no_external,
        discard_rate=None if bootstrap is None else bootstrap.discard_rate,
        bootstrap_wall_clock_s=None if bootstrap is None else bootstrap.wall_clock_s,
        bracket_saturated=bracket_saturated,
    )


def _endpoint_links(
    facts: Sequence[MatchupFacts], window: tuple[str, str]
) -> tuple[MatchupFacts, ...]:
    """Return the matchups of the two endpoint milestones' entrant groups."""
    return tuple(
        fact
        for fact in facts
        if any(_touches(fact.players, milestone) for milestone in window)
    )


def _informative_links(
    facts: Sequence[MatchupFacts], milestone: str, cross_no_external: MatchSet
) -> int:
    """Return how many of `milestone`'s GPU-path cross links are still informative.

    Counted over the links entering ``G_cross_no_external`` that touch the
    milestone **itself** -- the score rate of a link the milestone is not in
    says nothing about whether the milestone's own rating is still resolvable.

    A **configuration-variant** link -- one whose two players share a
    ``base_entrant_id``, i.e. §D2.2(b)5's ``milestone x milestone#anchor400``
    tie-in -- is excluded, however near 0.5 it sits. It is a same-weights link
    that §D2.2(b)5(i) argues is *"as near in strength as any pair in the pool"*,
    so it is permanently informative-looking whatever the **reference field**
    has done; counting it would leave condition 6's pre-registered floor of two
    external links standing at effectively one, and §D5.3's ``sigma ~ 33-48``
    would be quoted for a window whose real ``sigma`` is ~89.
    """
    entering = {
        tuple(sorted((record.white, record.black)))
        for record in cross_no_external.records
    }
    low, high = INFORMATIVE_BAND
    informative = 0
    for fact in facts:
        if fact.players not in entering or not fact.gpu_path:
            continue
        first, second = fact.players
        if base_entrant_id(first) == base_entrant_id(second):
            continue
        rate = fact.rate_for(milestone)
        if rate is None:
            continue
        if low <= rate <= high:
            informative += 1
    return informative


class _WindowFits:
    """Computes a replicate's three sub-fits once and serves both statistics.

    ``D`` and ``D_no_external`` share ``G_in``, and §D5.2 pre-registers **three**
    fits per replicate, not five. The two statistics therefore go through one
    cache keyed on the replicate object the bootstrap hands them, which is the
    same object for both statistics within a replicate.
    """

    def __init__(self, roster_map: Mapping[str, str], window: tuple[str, str]) -> None:
        self._roster_map = roster_map
        self._window = window
        self._cached_for: MatchSet | None = None
        self._cached: WindowGains | None = None

    def gains(self, replicate: MatchSet) -> WindowGains:
        """Return the replicate's gains, refitting only when it is a new one."""
        if self._cached is None or self._cached_for is not replicate:
            self._cached_for = replicate
            self._cached = window_gains(
                partition_records(replicate, self._roster_map), self._window
            )
        return self._cached


def window_statistics(
    roster_map: Mapping[str, str], window: tuple[str, str]
) -> tuple[Statistic, ...]:
    """Return the two bootstrap statistics of a window: ``d`` and ``d_no_external``.

    A replicate whose in-family, cross-family or cross-family-without-External
    fit fails to converge yields ``None`` from both, so the replicate is
    discarded whole and the two intervals are read off the same survivors
    (§D5.2 step 3).
    """
    fits = _WindowFits(roster_map, window)

    def published(replicate: MatchSet) -> float | None:
        return fits.gains(replicate).d

    def gated(replicate: MatchSet) -> float | None:
        return fits.gains(replicate).d_no_external

    return (
        Statistic(name=PUBLISHED_STATISTIC, compute=published),
        Statistic(name=GATE_STATISTIC, compute=gated),
    )


def evaluate_window(
    match_set: MatchSet,
    *,
    roster_map: Mapping[str, str],
    window: tuple[str, str] | None,
    kind: WindowKind,
    bootstrap_seed: int,
    replicates: int = REPLICATES,
) -> DivergenceResult:
    """Evaluate §D5's gate on one window and return its row.

    The conditions are checked in a fixed order -- window formed, then 1, 2, 3,
    4, 6, then the bootstrap and 5 -- so that a window failing several names the
    first one deterministically, and so that the bootstrap (the only expensive
    step) is not paid for a window already known to be unevaluable. Condition 5
    is read from the bootstrap, so it is necessarily last.

    The verdict is computed from ``D_no_external`` alone: it **fires** when
    ``|D_no_external| >= T(kind)`` *and* its 95% interval excludes 0. ``D`` and
    its interval are carried on the row beside it, and a row where the two
    disagree about firing carries ``external_drift``.

    Args:
        match_set: every pool record the two fits may draw on.
        roster_map: ``id -> family``; family is never parsed from a prefix.
        window: the window's ``(A, B)``, or ``None`` when it is not yet formed.
        kind: which of the two pre-registered windows this is.
        bootstrap_seed: seeds §D5.2's resampling; stored in the epoch record.
        replicates: ``B``; defaults to §D5.2's pre-registered 2,000.

    Returns:
        The window's :class:`DivergenceResult`.
    """
    threshold = threshold_for(kind)
    if window is None:
        return _not_evaluable(kind, None, CONDITION_WINDOW_NOT_FORMED)

    records = partition_records(match_set, roster_map)
    gains = window_gains(records, window)
    published_d = gains.d
    gated_d = gains.d_no_external
    if not gains.converged or published_d is None or gated_d is None:
        return _not_evaluable(kind, window, CONDITION_CONVERGED, gains=gains)

    for fitted in (records.in_family, records.cross, records.cross_no_external):
        if len(pool_components(fitted)) != 1:
            return _not_evaluable(kind, window, CONDITION_COMPONENTS, gains=gains)

    facts = summarise_matchups(match_set, roster_map)
    entering = {
        tuple(sorted((record.white, record.black)))
        for record in (*records.in_family.records, *records.cross.records)
    }
    endpoint_links = tuple(
        fact for fact in _endpoint_links(facts, window) if fact.players in entering
    )

    # Condition 3, over the endpoints' GPU-path links only. The External leg
    # carries a single constant book id naming the site's own start position
    # (§D6.1(1)) -- reading it into an era comparison would mark every window
    # `not evaluable` permanently, which is the defect §D3.2 diagnosed in
    # rev 1's per-epoch books and R8 diagnosed again in rev 2's wording.
    eras = {
        era
        for fact in endpoint_links
        if fact.gpu_path
        for era in fact.book_eras
    }
    if len(eras) > 1:
        return _not_evaluable(kind, window, CONDITION_BOOK_ERA, gains=gains)

    # Condition 4. The External leg is exempt by the declared deviation
    # §D2.2(b)4 registers: it is a >=50-game leg on the site's clock, and
    # reading it into this condition would mark *every* window not evaluable.
    for fact in endpoint_links:
        if not fact.gpu_path:
            continue
        if fact.games < MIN_GPU_PATH_GAMES or fact.decisive < MIN_DECISIVE:
            return _not_evaluable(
                kind, window, CONDITION_GAMES_AND_DECISIVE, gains=gains
            )

    # Condition 6: the pool's refusal to be blind quietly.
    saturated = tuple(
        milestone
        for milestone in window
        if _informative_links(facts, milestone, records.cross_no_external)
        < MIN_INFORMATIVE_LINKS
    )
    if saturated:
        return _not_evaluable(
            kind,
            window,
            CONDITION_LINKS_SATURATED,
            gains=gains,
            bracket_saturated=saturated,
        )

    bootstrap = pair_bootstrap(
        match_set,
        window_statistics(roster_map, window),
        replicates=replicates,
        seed=bootstrap_seed,
    )
    published = bootstrap.intervals[PUBLISHED_STATISTIC]
    gated = bootstrap.intervals[GATE_STATISTIC]
    if bootstrap.discard_rate > MAX_DISCARD_RATE:
        return _not_evaluable(
            kind, window, CONDITION_DISCARD_RATE, gains=gains, bootstrap=bootstrap
        )

    fires = abs(gated_d) >= threshold and gated.excludes_zero
    published_fires = abs(published_d) >= threshold and published.excludes_zero
    return DivergenceResult(
        window_kind=kind,
        window=window,
        verdict=Verdict.FIRED if fires else Verdict.OK,
        threshold=threshold,
        g_in=gains.g_in,
        g_cross=gains.g_cross,
        g_cross_no_external=gains.g_cross_no_external,
        d=gains.d,
        d_no_external=gains.d_no_external,
        lo=published.lo,
        hi=published.hi,
        lo_no_external=gated.lo,
        hi_no_external=gated.hi,
        discard_rate=bootstrap.discard_rate,
        bootstrap_wall_clock_s=bootstrap.wall_clock_s,
        external_drift=fires != published_fires,
    )


def evaluate_epoch(
    match_set: MatchSet,
    roster: PoolRoster,
    *,
    epoch: int,
    bootstrap_seed: int,
    replicates: int = REPLICATES,
) -> tuple[DivergenceResult, DivergenceResult]:
    """Evaluate **both** pre-registered windows for `epoch`, in order.

    The windows are ``(E-1, E)`` and ``(E-3, E)``, both evaluated every epoch,
    mechanically; either firing fires the gate, and no third window is ever
    formed (§D5.4).
    """
    roster_map = roster.roster_map
    rows: list[DivergenceResult] = [
        evaluate_window(
            match_set,
            roster_map=roster_map,
            window=window_for(roster, epoch=epoch, kind=kind),
            kind=kind,
            bootstrap_seed=bootstrap_seed,
            replicates=replicates,
        )
        for kind in (WindowKind.SINGLE, WindowKind.SUSTAINED)
    ]
    single, sustained = rows
    return (single, sustained)
