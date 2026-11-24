"""The cross-family pool: entrant identity, the roster, and the schedule.

No game is played by this module, and nothing here imports torch -- not at
import time and not through a deferred, function-body import either. Building
a schedule and constructing a :class:`PoolRoster` must work on a box with no
torch installed, which is what makes the identity and scheduling half of
``docs/design/kom-011d-ladder-pool.md`` reviewable by reading (design §5's
torch rule, §6 node 42). The cost of that property is paid explicitly in one
place: :func:`matchups` is a **copy** of ``komugi.eval.ladder.matchups``
(``ladder.py:339-368``) rather than an import, because ``ladder.py:58`` imports
``komugi.eval.arena``, which imports torch at ``arena.py:81``. §6 node 39 pins
the copy against the original so the duplication cannot drift silently.

Three rules from §D1/§D2 are enforced here rather than remembered:

* **Identity** (§D1.2) is ``family/member``, minted once, never containing
  ``':'`` -- ``play_match`` joins a ``pair_id`` as
  ``f"{a.player.id}:{b.player.id}:{pair:05d}"`` (``arena.py:691``), so an id
  carrying a colon makes a pair id that cannot be parsed back to its entrants.
  A frozen entrant's member carries an ``@sha8`` digest, so an id is
  self-verifying; ``external`` has nothing to pin (§0.2a, U4) and is not frozen.
* **Family is read from the roster map, never parsed from the prefix**
  (§D1.2 property 3). The prefix is for human legibility and is *checked*
  against the declared family at construction; every reader goes through
  :func:`family_of`, which consults the map and refuses to guess.
* **``frozen`` and ``reference`` are two properties** (§D1.1). ``reference`` is
  ``frozen and family != "komugi"``: a komugi milestone is frozen and is
  emphatically not a reference entrant. Reading the cross-family bracket over
  ``frozen`` selects the milestone's own predecessors and turns the pool back
  into the mirror ``docs/DESIGN.html:260`` exists to prevent, so ``reference``
  is a derived property here and never a free flag.

The schedule builder is §D2.2: a frozen sub-pool played once ever, and six
matchups per epoch (five at ``E = 1``, which has no in-family predecessor).
Connectivity is asserted rather than hoped for -- a disconnected comparison
graph still reports ``converged=True`` from ``fit_elo``, so the fit cannot be
its own detector. :func:`require_connected_pool` reuses
``komugi.eval.ab_endpoint.pool_components`` and raises in the **pool's own**
words: ``fit_cross_arm_pool``'s refusal (``ab_endpoint.py:518-527``) is
KOM-036's endpoint vocabulary and would report a pool failure in terms of two
experiment groups that do not exist here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from komugi.eval.ab_endpoint import pool_components
from komugi.eval.anchor_aliases import resolve_family
from komugi.eval.records import (
    DRAW,
    KNOWN_FAMILIES,
    GameRecord,
    MatchSet,
    Player,
    roster_families,
)

FAMILY_SEPARATOR: Final[str] = "/"
"""Separates the family prefix from the member in an entrant id (§D1.2)."""

VARIANT_SEPARATOR: Final[str] = "#"
"""Introduces a configuration variant: ``…#anchor400``, ``…#n1600`` (§D8.2)."""

PAIR_ID_SEPARATOR: Final[str] = ":"
"""The character ``arena.py:691`` joins a ``pair_id`` with, and so the one
character an entrant id may never contain."""

KOMUGI_FAMILY: Final[str] = "komugi"
"""The campaign's own lineage; frozen but never a *reference* entrant (§D1.1)."""

EXTERNAL_FAMILY: Final[str] = "external"
"""The external anchor engine: an opponent, not a frozen artefact (§0.2a, U4)."""

ANCHOR400_VARIANT: Final[str] = "anchor400"
"""The KOM-007 match-stack configuration an External leg actually rates (§D6.1)."""

MATCHUP_GAMES: Final[int] = 128
"""§D2.1: ``n >= 128`` is per matchup -- 64 seat-swapped pairs."""

IN_FAMILY_GAMES: Final[int] = 256
"""§D5.3 step 2: the in-family link carries ``G_in`` alone, so it is bought a
second 128 games; ``SE(G_in)`` drops from ~31 to ~22 Elo for +0.28 h/epoch."""

EXTERNAL_GAMES: Final[int] = 50
"""KOM-017's cadence (``tickets/KOM-017.md:11``). A **declared deviation** from
``DESIGN.html:260``'s ``n >= 128`` (§D2.2(b)4): the games are on the site's
clock and cannot be bought with GPU."""

FROZEN_SUBPOOL_ADJACENCY: Final[int] = 2
"""§D2.2(a): the chain plus the double link over the reference field."""

KIND_FROZEN_SUBPOOL: Final[str] = "frozen-subpool"
"""§D2.2(a): a reference x reference link, played once ever and re-used."""

KIND_IN_FAMILY: Final[str] = "in-family"
"""§D2.2(b)1: the milestone against its epoch predecessor, at 256 games."""

KIND_ANCHOR: Final[str] = "anchor"
"""§D2.2(b)2: every milestone is tied directly to the gauge."""

KIND_BRACKET: Final[str] = "bracket"
"""§D2.2(b)3: the nearest reference entrants above and below the prediction."""

KIND_EXTERNAL: Final[str] = "external"
"""§D2.2(b)4: the site leg, played by the ``#anchor400`` entrant (§D6.1)."""

KIND_TIE_IN: Final[str] = "anchor400-tie-in"
"""§D2.2(b)5: ``#anchor400`` x the milestone, the edge without which the
External component is not in the comparison graph at all."""

_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"@[0-9a-f]{8}$")
"""§D1.2: a frozen entrant's member ends in ``@`` plus eight lowercase hex."""

_VISITS_VARIANT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^n[0-9]+$")
"""§D8.2's ``#n<visits>`` grid points, which §D5.1 excludes from both fits."""


def split_entrant_id(entrant_id: str) -> tuple[str, str, str | None]:
    """Return ``(prefix, stem, variant)`` for `entrant_id`.

    ``komugi/ckpt-15000@a1b2c3d4#anchor400`` splits into
    ``("komugi", "ckpt-15000@a1b2c3d4", "anchor400")``. The prefix is returned
    for *validation* only -- readers that want the family call
    :func:`family_of`, which consults the roster map (§D1.2 property 3).

    Raises:
        ValueError: if `entrant_id` is not ``family/member``.
    """
    if entrant_id.count(FAMILY_SEPARATOR) != 1:
        raise ValueError(
            f"entrant id {entrant_id!r} is not '<family>/<member>': it must "
            f"contain exactly one {FAMILY_SEPARATOR!r}"
        )
    prefix, member = entrant_id.split(FAMILY_SEPARATOR)
    stem, separator, variant = member.partition(VARIANT_SEPARATOR)
    if separator and not variant:
        raise ValueError(
            f"entrant id {entrant_id!r} ends in a bare {VARIANT_SEPARATOR!r}: "
            f"a variant suffix must name the configuration it varies"
        )
    return (prefix, stem, variant or None)


def base_entrant_id(entrant_id: str) -> str:
    """Return `entrant_id` with any ``#variant`` suffix removed."""
    return entrant_id.split(VARIANT_SEPARATOR, 1)[0]


def variant_of(entrant_id: str) -> str | None:
    """Return the ``#variant`` suffix of `entrant_id`, or ``None``."""
    _, _, variant = split_entrant_id(entrant_id)
    return variant


def is_visits_variant(entrant_id: str) -> bool:
    """Return whether `entrant_id` is one of §D8's ``#n<visits>`` grid points.

    §D8.2's naming rule: ``#n256`` **is** the rated milestone row and carries
    no suffix at all, so only the off-grid budgets mint a suffixed id. That is
    what makes §D5.1's "no ``#`` suffix" rule pick out exactly the rated
    milestones, and it is why the visits legs can be dropped mechanically from
    both of the gate's record sets.
    """
    variant = variant_of(entrant_id)
    return variant is not None and _VISITS_VARIANT_PATTERN.match(variant) is not None


def _validate_entrant_id(entrant_id: str, family: str, frozen: bool) -> None:
    """Raise ``ValueError`` unless `entrant_id` satisfies §D1.2."""
    if not entrant_id or not entrant_id.strip():
        raise ValueError(f"entrant id must be a non-empty identifier, got {entrant_id!r}")
    if PAIR_ID_SEPARATOR in entrant_id:
        raise ValueError(
            f"entrant id {entrant_id!r} contains {PAIR_ID_SEPARATOR!r}, which "
            f"is the character play_match joins a pair_id with "
            f"(arena.py:691 builds it as '<a>:<b>:<pair>'); an id carrying one "
            f"makes a pair_id that cannot be parsed back to its entrants"
        )
    prefix, stem, _ = split_entrant_id(entrant_id)
    if not stem:
        raise ValueError(f"entrant id {entrant_id!r} has an empty member")
    if family not in KNOWN_FAMILIES:
        raise ValueError(
            f"entrant {entrant_id!r} declares family {family!r}, which is not "
            f"one of the pool's families {sorted(KNOWN_FAMILIES)}"
        )
    if prefix != family:
        raise ValueError(
            f"entrant {entrant_id!r} has family prefix {prefix!r} but declares "
            f"family {family!r}; the prefix is for legibility and the roster "
            f"map is what a reader consults, so the two may not disagree"
        )
    if frozen and _DIGEST_PATTERN.search(stem) is None:
        raise ValueError(
            f"frozen entrant {entrant_id!r} carries no '@sha8' digest; §D1.2 "
            f"puts the digest in the id so an entrant whose artefact does not "
            f"hash to its id is refused before a game is played"
        )


@dataclass(frozen=True, slots=True)
class PoolEntrant:
    """One pool member: an immutable identity plus what it declares about itself.

    ``frozen`` says the artefact behind the id cannot change (weights on disk,
    or a code-and-config digest); :attr:`reference` is derived from it and the
    family, never set by hand (§D1.1). ``compute`` is the row's compute **in
    its own unit** (§D7.5): ``{"evals_per_move": 257}`` for komugi and v1,
    ``{"sims_per_move": 800}`` for a RawMCTS rung, and ``None`` -- not ``0`` --
    for external, whose budget is unknown and is never inherited from ours.
    """

    id: str
    family: str
    frozen: bool
    epoch: int | None = None
    """The pool epoch a komugi milestone entered at; ``None`` for everyone else.

    Carried rather than derived: §D2.2(b)1's in-family link is the **epoch
    predecessor**, and the campaign's own curve is non-monotone
    (``tickets/KOM-011.md:98``), so "the strongest existing milestone" names a
    different player on a dip (§6 node 57).
    """

    config_digest: str | None = None
    """The digest the id's ``@sha8`` abbreviates, for the §D1.3 config block."""

    compute: Mapping[str, float] | None = None
    ceiling_exempt: bool | None = False
    """``True``: consults no network, so KOM-023's 257 unit does not apply
    (§D7.4). ``False``: ceiling-bound. ``None``: not applicable (external)."""

    eval_ceiling: int | None = None
    """The per-move evaluation ceiling when ceiling-bound; ``None`` otherwise."""

    mover_factory: Callable[[], object] | None = None
    """Builds the entrant's ``Mover`` when a driver seats it in a match.

    Typed structurally on purpose: constructing a roster must not need an
    engine, so this module never names the mover's class (design §5's torch
    rule). T4's driver is what calls it.
    """

    def __post_init__(self) -> None:
        _validate_entrant_id(self.id, self.family, self.frozen)
        if self.ceiling_exempt is False and self.eval_ceiling is not None:
            if self.eval_ceiling <= 0:
                raise ValueError(
                    f"entrant {self.id!r} declares a non-positive evaluation "
                    f"ceiling {self.eval_ceiling!r}"
                )

    @property
    def player(self) -> Player:
        """Return the :class:`~komugi.eval.records.Player` this entrant records as."""
        return Player(id=self.id, family=self.family)

    @property
    def reference(self) -> bool:
        """Return whether this is a **reference** entrant: frozen and out of family.

        Derived, never stored: §D1.1's whole point is that reading the bracket
        over ``frozen`` silently selects the milestone's own predecessors.
        """
        return self.frozen and self.family != KOMUGI_FAMILY

    @property
    def variant(self) -> str | None:
        """Return this entrant's ``#variant`` suffix, if it has one."""
        return variant_of(self.id)

    @property
    def base_id(self) -> str:
        """Return the id this entrant is a configuration variant of."""
        return base_entrant_id(self.id)

    @property
    def is_ceiling_bound(self) -> bool:
        """Return whether this entrant is bound by a declared evaluation ceiling."""
        return self.ceiling_exempt is False and self.eval_ceiling is not None


def family_of(roster_map: Mapping[str, str], player_id: str) -> str:
    """Return `player_id`'s family from `roster_map`, refusing to guess.

    §D1.2 property 3: ``GameRecord`` has no family field (``records.py:89-93``),
    so the roster map is persisted beside the ``MatchSet`` and *a report reads
    family from it, never from the prefix* (``v1bridge.py:2323-2332``). An id
    the map does not carry is an error, not an invitation to parse -- including
    a legacy prefix-free id such as ``ckpt-15000``, which the map can name and
    a prefix reader cannot.

    A value stored under an old family name (C1 backward compatibility, before
    the external-anchor rename) is mapped to its current name through
    :func:`~komugi.eval.anchor_aliases.resolve_family` before it is returned,
    so a caller never has to know whether the roster map is old or new data.

    Raises:
        ValueError: if `player_id` is absent from `roster_map`.
    """
    family = roster_map.get(player_id)
    if family is None:
        raise ValueError(
            f"player {player_id!r} is not in the roster map, so its family is "
            f"unknown; the prefix is not consulted (§D1.2 property 3). Known "
            f"players: {sorted(roster_map)}"
        )
    return resolve_family(family)


@dataclass(frozen=True, slots=True)
class PoolRoster:
    """Every entrant of one pool epoch, plus the anchor that fixes the gauge.

    Four refusals live at construction, because every one of them is silent
    afterwards: a contradictory family declaration merges two engines into one
    row (``roster_families``, ``records.py:185-201``); a duplicate id does the
    same thing more quietly; two ids over one artefact make a checkpoint appear
    twice in one fit as two independent players (§D1.1); and a non-frozen
    anchor is a gauge that can move without a diff anywhere (§0.2a, §D4.1).
    """

    entrants: tuple[PoolEntrant, ...]
    anchor_id: str

    def __post_init__(self) -> None:
        if not self.entrants:
            raise ValueError("a pool roster needs at least one entrant")
        # Reused, not re-derived: this is the check records.py already owns.
        roster_families(entrant.player for entrant in self.entrants)
        seen: set[str] = set()
        for entrant in self.entrants:
            if entrant.id in seen:
                raise ValueError(
                    f"entrant {entrant.id!r} appears twice in the roster; an id "
                    f"is minted once and names one artefact (§D1.1)"
                )
            seen.add(entrant.id)
        self._refuse_split_artefacts()
        if self.anchor_id not in seen:
            raise ValueError(
                f"anchor {self.anchor_id!r} is not in the roster; the anchor id "
                f"is carried in the record, never re-derived (ladder.py:218-241)"
            )
        anchor = self.entrant(self.anchor_id)
        if not anchor.frozen:
            raise ValueError(
                f"anchor {self.anchor_id!r} is not frozen, so it may not be the "
                f"gauge: a non-frozen entrant's artefact can change with no "
                f"diff anywhere to show it (§D1.1, §D4.1)"
            )

    def _refuse_split_artefacts(self) -> None:
        """Raise if one ``config_digest`` is claimed by two different artefacts.

        Configuration variants of one entrant (``…#anchor400``) share the
        milestone's digest by design (§D6.1: *the same weights*), so the rule
        is stated over the **base** id: two different base ids over one digest
        are one checkpoint entered twice, which §D1.1 forbids.
        """
        by_digest: dict[str, set[str]] = {}
        for entrant in self.entrants:
            if entrant.config_digest is None:
                continue
            by_digest.setdefault(entrant.config_digest, set()).add(entrant.base_id)
        for digest, bases in sorted(by_digest.items()):
            if len(bases) > 1:
                raise ValueError(
                    f"config digest {digest!r} is claimed by two entrants: "
                    f"{sorted(bases)}. One artefact enters the pool exactly "
                    f"once and never re-enters; two ids over one checkpoint "
                    f"appear in one fit as two independent players (§D1.1)"
                )

    @property
    def roster_map(self) -> dict[str, str]:
        """Return the ``id -> family`` map that is persisted beside the records."""
        return roster_families(entrant.player for entrant in self.entrants)

    @property
    def anchor(self) -> PoolEntrant:
        """Return the anchor entrant."""
        return self.entrant(self.anchor_id)

    def entrant(self, entrant_id: str) -> PoolEntrant:
        """Return the entrant with `entrant_id`.

        Raises:
            ValueError: if no such entrant is in the roster.
        """
        for entrant in self.entrants:
            if entrant.id == entrant_id:
                return entrant
        raise ValueError(
            f"{entrant_id!r} is not in the roster; entrants are "
            f"{[entrant.id for entrant in self.entrants]}"
        )

    def references(self) -> tuple[PoolEntrant, ...]:
        """Return the reference field: frozen, out of family, in id order (§D1.1)."""
        return tuple(
            sorted(
                (entrant for entrant in self.entrants if entrant.reference),
                key=lambda entrant: entrant.id,
            )
        )

    def milestones(self) -> tuple[PoolEntrant, ...]:
        """Return the rated komugi milestones, in epoch order.

        Rated means **no ``#`` suffix** (§D8.2's naming rule): the ``#anchor400``
        and ``#n<visits>`` entrants are configuration variants, not milestones,
        and §D5.1's ``G_in`` is defined over exactly the unsuffixed ones.
        """
        rated = [
            entrant
            for entrant in self.entrants
            if entrant.family == KOMUGI_FAMILY and entrant.variant is None
        ]
        return tuple(
            sorted(rated, key=lambda entrant: (entrant.epoch or 0, entrant.id))
        )

    def milestone_at(self, epoch: int) -> PoolEntrant | None:
        """Return the milestone that entered at `epoch`, or ``None``."""
        for entrant in self.milestones():
            if entrant.epoch == epoch:
                return entrant
        return None


@dataclass(frozen=True, slots=True)
class Matchup:
    """One scheduled matchup: two entrants, a game count, and its compute.

    Both sides' compute is carried on every matchup and ``unequal_compute`` is
    computed from them (§D7.5(2)), so a rating column can be read without
    cross-referencing the schedule. ``exempt`` names the sides whose ceiling
    KOM-023's 257 unit does not apply to -- a net-free opponent consults no
    network, so there is nothing to count, and the exemption is *declared*
    rather than assumed.
    """

    a: str
    b: str
    games: int
    kind: str
    compute_a: Mapping[str, float] | None = None
    compute_b: Mapping[str, float] | None = None
    unequal_compute: bool = False
    exempt: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError(f"a matchup cannot be a self-link: {self.a!r}")
        if self.games <= 0:
            raise ValueError(
                f"matchup {self.a!r} x {self.b!r} must play at least one game, "
                f"got {self.games!r}"
            )

    @property
    def players(self) -> frozenset[str]:
        """Return the unordered pair this matchup is between."""
        return frozenset({self.a, self.b})


def computes_differ(
    first: Mapping[str, float] | None, second: Mapping[str, float] | None
) -> bool:
    """Return whether two compute declarations differ in unit or in value.

    An **unknown** compute (``None``, which today is exactly ``external``) counts
    as differing: §D7.5 permits no equal-compute claim from a link whose two
    sides are not both ceiling-bound at the same ceiling, and two unknowns are
    not a demonstration of equality.
    """
    if first is None or second is None:
        return True
    return dict(first) != dict(second)


def _matchup(
    first: PoolEntrant, second: PoolEntrant, *, games: int, kind: str
) -> Matchup:
    """Build a :class:`Matchup` carrying both sides' compute and exemptions."""
    exempt = tuple(
        entrant.id for entrant in (first, second) if entrant.ceiling_exempt is True
    )
    return Matchup(
        a=first.id,
        b=second.id,
        games=games,
        kind=kind,
        compute_a=first.compute,
        compute_b=second.compute,
        unequal_compute=computes_differ(first.compute, second.compute),
        exempt=exempt,
    )


def rated_matchup(
    first: PoolEntrant, second: PoolEntrant, *, games: int, kind: str
) -> Matchup:
    """Return a matchup on the **rated cross-family path**, refusing §D7.4's case.

    Two ceiling-bound entrants at *different* ceilings is refused here rather
    than discovered when the counted evaluator raises mid-match: the pool is a
    single configuration regime with three declared exemptions, and a matchup
    whose two sides disagree about the ceiling is permitted only when one of
    them is exempt. The design's own unequal matchups -- §D2.2(b)5's tie-in and
    the §D8 visits legs -- are scheduled through
    :func:`declared_unequal_matchup`, which is the declaration §D7.5 requires.

    Raises:
        ValueError: if both sides are ceiling-bound at different ceilings.
    """
    if (
        first.is_ceiling_bound
        and second.is_ceiling_bound
        and first.eval_ceiling != second.eval_ceiling
    ):
        raise ValueError(
            f"rated matchup {first.id!r} x {second.id!r} puts two "
            f"ceiling-bound entrants at different ceilings "
            f"({first.eval_ceiling} and {second.eval_ceiling}); equal compute "
            f"is KOM-023 §D3.3's per-move evaluation ceiling, and neither side "
            f"is exempt, so this link cannot carry the claim it would be read "
            f"as making (§D7.4, §D7.5)"
        )
    return _matchup(first, second, games=games, kind=kind)


def declared_unequal_matchup(
    first: PoolEntrant, second: PoolEntrant, *, games: int, kind: str
) -> Matchup:
    """Return a matchup the design schedules at *declared* unequal compute.

    §D7.5: the exemption is not a waiver of the equal-compute standard, it is a
    declaration that the standard's claim is not being made. The two such
    matchups this design schedules on purpose are §D2.2(b)5's ``#anchor400``
    tie-in (both sides ceiling-bound, at 257 and ``SIMS+2``) and §D8's visits
    legs, and both come out carrying ``unequal_compute=True``.
    """
    matchup = _matchup(first, second, games=games, kind=kind)
    if not matchup.unequal_compute:
        raise ValueError(
            f"matchup {first.id!r} x {second.id!r} is declared unequal-compute "
            f"but both sides declare the same compute {dict(first.compute or {})}; "
            f"schedule it with rated_matchup instead of mislabelling it"
        )
    return matchup


def matchups(count: int, adjacency: int) -> tuple[tuple[int, int], ...]:
    """Return the ``(i, j)`` rung pairs of a ladder over `count` players.

    **Copied verbatim from ``komugi.eval.ladder.matchups`` (``ladder.py:339-368``),
    not imported**: ``ladder.py:58`` imports ``komugi.eval.arena``, which
    imports torch at ``arena.py:81``, so importing the original would drag the
    engine into this module and destroy the property §5 states for it. The
    duplication is the cost, and §6 node 39 is what stops it drifting -- it
    asserts the two agree for ``count = 2..12`` at ``adjacency = 2``.

    In **step-major** play order: the whole chain (``i`` vs ``i+1``) first, then
    every ``i`` vs ``i+2``, and so on. That order is what makes the property the
    seeds rest on true -- ``matchups(count, k)`` is a *prefix* of
    ``matchups(count, k + 1)``, so a matchup's index, and with it its seed, is a
    function of the rung pair alone and does not shuffle when `adjacency`
    changes the *number* of matches.

    Pairs beyond the top of the ladder are dropped rather than wrapped: a ladder
    is a line, not a ring.
    """

    def pairs() -> Iterator[tuple[int, int]]:
        for step in range(1, min(adjacency, count - 1) + 1):
            for i in range(count):
                if i + step < count:
                    yield (i, i + step)

    return tuple(pairs())


def frozen_subpool_schedule(
    roster: PoolRoster,
    order: Sequence[str],
    *,
    games: int = MATCHUP_GAMES,
) -> tuple[Matchup, ...]:
    """Return §D2.2(a)'s reference x reference schedule over `order`.

    `order` is the pre-registered strength ordering of the **reference** field,
    weakest first -- at the first epoch a recorded guess that is never edited
    afterwards, and from then on the previous epoch's ratings. A wrong guess
    costs precision, not validity: the graph is connected either way.

    Played once, ever: its records are stored and every later epoch re-uses
    them verbatim, keeping the book era they were played in (§D2.2(a)).

    Raises:
        ValueError: if `order` names a non-reference entrant, repeats one, or
            holds fewer than two.
    """
    if len(order) < 2:
        raise ValueError(
            f"the frozen sub-pool needs at least two reference entrants, got "
            f"{list(order)}"
        )
    if len(set(order)) != len(order):
        raise ValueError(f"the frozen sub-pool ordering repeats an entrant: {list(order)}")
    entrants = [roster.entrant(entrant_id) for entrant_id in order]
    for entrant in entrants:
        if not entrant.reference:
            raise ValueError(
                f"{entrant.id!r} is not a reference entrant (frozen="
                f"{entrant.frozen}, family={entrant.family!r}) and may not enter "
                f"the frozen sub-pool: §D2.2(a)'s reuse is what makes the "
                f"frozen/reference distinction cost something (§D1.1)"
            )
    return tuple(
        rated_matchup(
            entrants[i], entrants[j], games=games, kind=KIND_FROZEN_SUBPOOL
        )
        for i, j in matchups(len(entrants), FROZEN_SUBPOOL_ADJACENCY)
    )


def bracket_for(
    roster: PoolRoster,
    *,
    predicted_rating: float,
    ratings: Mapping[str, float],
    exclude: Sequence[str] = (),
) -> tuple[tuple[str, str], bool]:
    """Return §D2.2(b)3's bracket and whether the reference field is saturated.

    The two **reference** entrants whose previous-epoch ratings most nearly
    bracket `predicted_rating` -- nearest above and nearest below. Never a
    komugi milestone, however near its rating: read over ``frozen`` instead of
    ``reference`` the bracket degenerates into the milestone's own predecessors
    and ``G_cross`` loses every non-anchor edge (§D1.1).

    When the milestone has passed the **top** of the reference field the
    bracket degenerates to the two strongest entrants and ``bracket_saturated``
    is returned ``True`` -- the pool reports that it has run out of opponents
    rather than reporting a number that has stopped moving (§D2.2's
    anti-saturation argument, Q5).

    **The below-the-field branch is not symmetric with it, on purpose.** When
    the milestone is weaker than every reference entrant the bracket
    degenerates the same way -- to the two weakest, which is correct scheduling
    -- but ``bracket_saturated`` is ``False``. §D2.2's saturation signal means
    one thing only: *"as the milestone gets stronger its bracket walks up the
    reference field, and when it passes the top the pool reports that it has
    run out of opponents"*, and §D5.3 condition 6 ties the flag on a rating row
    to *"the gate can no longer see"*. A first-epoch milestone weaker than
    ``rawmcts/n100`` is not a saturated instrument; it is a young one, and its
    links are the most informative the pool will ever have. Flagging it would
    fire §Q5's pre-registered response -- extend the reference field -- at
    exactly the moment the field is at its most adequate.

    `exclude` names entrants the bracket may not choose. §D2.2 does not say
    what happens when the **anchor** is itself the nearest bracket candidate,
    and it has to say something: (b)2 already schedules the anchor link
    unconditionally, and a matchup played twice in one epoch collides on its
    ``pair_id`` and fails ``check_paired_books`` with ``GAME_COUNT`` violations
    (``arena.py:645-654``, §D2.1). :func:`build_epoch_schedule` therefore
    excludes the anchor here; the link it would have contributed is the anchor
    link the schedule pays for anyway.

    Returns:
        ``((below_or_nearest, above_or_nearest), saturated)``, ids ordered
        low-rating first.

    Raises:
        ValueError: if fewer than two eligible reference entrants carry a rating.
    """
    barred = set(exclude)
    rated = [
        (ratings[entrant.id], entrant.id)
        for entrant in roster.references()
        if entrant.id in ratings and entrant.id not in barred
    ]
    if len(rated) < 2:
        raise ValueError(
            f"the cross-family bracket needs at least two rated reference "
            f"entrants, got {sorted(name for _, name in rated)}"
        )
    rated.sort()
    below = [name for rating, name in rated if rating <= predicted_rating]
    above = [name for rating, name in rated if rating > predicted_rating]
    if not above:
        return ((rated[-2][1], rated[-1][1]), True)
    if not below:
        # The two weakest, and NOT saturated: see the docstring's asymmetry.
        return ((rated[0][1], rated[1][1]), False)
    return ((below[-1], above[0]), False)


@dataclass(frozen=True, slots=True)
class EpochSchedule:
    """§D2.2(b)'s per-epoch schedule: six matchups, or five at ``E = 1``."""

    epoch: int
    milestone: str
    matchups: tuple[Matchup, ...]
    bracket_saturated: bool

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError(f"a pool epoch is numbered from 1, got {self.epoch!r}")

    def of_kind(self, kind: str) -> tuple[Matchup, ...]:
        """Return the scheduled matchups of one §D2.2 kind, in schedule order."""
        return tuple(matchup for matchup in self.matchups if matchup.kind == kind)

    @property
    def games(self) -> int:
        """Return the total games this epoch's schedule pays for."""
        return sum(matchup.games for matchup in self.matchups)


def anchor400_id(milestone: str) -> str:
    """Return the ``#anchor400`` variant id of `milestone` (§D6.1(b))."""
    return f"{milestone}{VARIANT_SEPARATOR}{ANCHOR400_VARIANT}"


def build_epoch_schedule(
    roster: PoolRoster,
    *,
    epoch: int,
    milestone: str,
    predicted_rating: float,
    ratings: Mapping[str, float],
    external: str | None = None,
) -> EpochSchedule:
    """Return §D2.2(b)'s schedule for the milestone entering at `epoch`.

    Six matchups at ``E >= 2``: the in-family predecessor at **256** games, the
    anchor, the two bracket links, the External leg, and the ``#anchor400``
    tie-in. At ``E = 1`` the in-family link does not exist -- there is no
    predecessor -- so the schedule is the other **five**, with the bracket read
    from the same pre-registered ordering guess §D2.2(a) already records. The
    builder does not raise at ``E = 1`` and does not invent a sixth self-link;
    both §D5 windows report ``not evaluable: window not yet formed`` instead
    (§D5.1, §6 node 6).

    `predicted_rating` is the previous milestone's rating plus the previous
    epoch's observed gain. It is a **scheduling** input only and enters no
    estimate.

    Raises:
        ValueError: if `milestone` is not a rated komugi entrant of this epoch,
            if its ``#anchor400`` entrant is absent, if `epoch >= 2` and no
            milestone entered at ``epoch - 1``, or if `external` is ambiguous.
    """
    entrant = roster.entrant(milestone)
    if entrant.family != KOMUGI_FAMILY or entrant.variant is not None:
        raise ValueError(
            f"{milestone!r} is not a rated komugi milestone (family="
            f"{entrant.family!r}, variant={entrant.variant!r}); §D8.2's naming "
            f"rule reserves the unsuffixed id for the rated row"
        )
    if entrant.epoch is not None and entrant.epoch != epoch:
        raise ValueError(
            f"milestone {milestone!r} entered at epoch {entrant.epoch}, not "
            f"{epoch}; a milestone enters the pool exactly once (§D1.1)"
        )
    variant = roster.entrant(anchor400_id(milestone))
    anchor = roster.anchor
    scheduled: list[Matchup] = []

    predecessor = roster.milestone_at(epoch - 1) if epoch >= 2 else None
    if epoch >= 2 and predecessor is None:
        raise ValueError(
            f"epoch {epoch} has no milestone from epoch {epoch - 1} in the "
            f"roster, so §D2.2(b)1's in-family link has no far end; G_in is fit "
            f"over exactly this chain of links (§D5.1) and a gap in it is not a "
            f"schedule this builder may guess at"
        )
    if predecessor is not None:
        scheduled.append(
            rated_matchup(
                predecessor, entrant, games=IN_FAMILY_GAMES, kind=KIND_IN_FAMILY
            )
        )

    scheduled.append(
        rated_matchup(entrant, anchor, games=MATCHUP_GAMES, kind=KIND_ANCHOR)
    )

    (low, high), saturated = bracket_for(
        roster,
        predicted_rating=predicted_rating,
        ratings=ratings,
        exclude=(anchor.id,),
    )
    for opponent_id in (low, high):
        scheduled.append(
            rated_matchup(
                entrant,
                roster.entrant(opponent_id),
                games=MATCHUP_GAMES,
                kind=KIND_BRACKET,
            )
        )

    external_id = _resolve_external(roster, external)
    if external_id is not None:
        scheduled.append(
            _matchup(
                variant,
                roster.entrant(external_id),
                games=EXTERNAL_GAMES,
                kind=KIND_EXTERNAL,
            )
        )

    # §D2.2(b)5: without this edge {external, …#anchor400} is a two-vertex
    # component with no edge to the rest of the pool, pool_components returns 2
    # and the driver refuses on *every* epoch rather than on a mis-edited
    # schedule. Both sides are ceiling-bound at different ceilings (257 vs
    # SIMS+2), so it is permitted only as a declared-unequal link (§D7.5).
    scheduled.append(
        declared_unequal_matchup(
            variant, entrant, games=MATCHUP_GAMES, kind=KIND_TIE_IN
        )
    )

    return EpochSchedule(
        epoch=epoch,
        milestone=milestone,
        matchups=tuple(scheduled),
        bracket_saturated=saturated,
    )


def _resolve_external(roster: PoolRoster, external: str | None) -> str | None:
    """Return the external entrant to schedule, or ``None`` if there is none."""
    if external is not None:
        entrant = roster.entrant(external)
        if entrant.family != EXTERNAL_FAMILY:
            raise ValueError(
                f"{external!r} is not an external entrant (family={entrant.family!r})"
            )
        return entrant.id
    candidates = [
        entrant.id for entrant in roster.entrants if entrant.family == EXTERNAL_FAMILY
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"the roster holds {len(candidates)} external entrants {sorted(candidates)}; "
            f"one entrant per epoch date (§D1.1) -- name the one this epoch plays"
        )
    return candidates[0]


def schedule_match_set(schedule: Sequence[Matchup]) -> MatchSet:
    """Return a placeholder :class:`MatchSet` with one drawn game per matchup.

    Only the *graph* of a schedule is defined before its games are played, and
    :func:`komugi.eval.ab_endpoint.pool_components` is defined over a match set,
    so connectivity of a schedule is checked by handing it one record per
    matchup. The results are drawn placeholders and are never fitted.
    """
    return MatchSet.of(
        GameRecord(
            white=matchup.a,
            black=matchup.b,
            result=DRAW,
            book_id="schedule",
            pair_id=f"{matchup.a}{PAIR_ID_SEPARATOR}{matchup.b}"
            f"{PAIR_ID_SEPARATOR}{index:05d}",
        )
        for index, matchup in enumerate(schedule)
    )


def require_connected_pool(
    match_set: MatchSet, roster_map: Mapping[str, str]
) -> tuple[frozenset[str], ...]:
    """Return the pool's components, raising in the pool's own words if not one.

    Checked **before** the fit, never after: a disconnected comparison graph
    still converges -- each component finds its own gauge -- and reports
    ``converged=True`` (``ab_endpoint.py:426-463``), so the fit cannot be its
    own detector.

    The refusal is the pool's, not ``fit_cross_arm_pool``'s
    (``ab_endpoint.py:518-527``). That one is KOM-036's endpoint driver: its
    message is about two within-experiment chains with no games between them
    and its anchor defaults to a reference id from a different experiment, so
    it would report this pool's failure in a vocabulary that names nothing
    here. This message names the pool's own components by entrant id and
    family.

    Raises:
        ValueError: if the comparison graph has more than one component.
    """
    components = pool_components(match_set)
    if len(components) > 1:
        described = [
            ", ".join(
                f"{player} [{roster_map.get(player, 'family unknown')}]"
                for player in sorted(component)
            )
            for component in components
        ]
        listed = "; ".join(f"({group})" for group in described)
        raise ValueError(
            f"the pool's comparison graph has {len(components)} components, "
            f"not one: {listed}. The joint fit is invariant to an offset "
            f"between them, so every rating it reports across the split is an "
            f"unidentified free parameter -- and it would still report "
            f"converged=True. Schedule the links §D2.2 lists (the anchor link "
            f"for every milestone, and the #anchor400 tie-in) before fitting"
        )
    return components


def require_connected_schedule(
    schedule: Sequence[Matchup], roster_map: Mapping[str, str]
) -> tuple[frozenset[str], ...]:
    """Return the components of `schedule`, refusing a disconnected one.

    §D2.2 asserts ``pool_components == 1`` on the designed schedule directly
    rather than only refusing when it is violated, which is what §6 node 38
    checks: the tie-in is the one edge whose removal splits the graph.
    """
    return require_connected_pool(schedule_match_set(schedule), roster_map)


@dataclass(frozen=True, slots=True)
class EpochManifest:
    """The frozen description of one pool epoch, as §D9.2 stores it.

    Everything a reader needs to interpret the epoch is **copied in, not
    referenced** (§D9.2 property 3): the anchor id (carried, never re-derived
    -- ``ladder.py:218-241``'s lesson), the roster map a report reads family
    from, the book epoch the games were played on, and the reference field's
    own book epoch when it differs, because §D2.2(a)'s re-used records keep the
    era they were played in forever.
    """

    epoch: int
    anchor_id: str
    roster_map: Mapping[str, str]
    schedule: tuple[Matchup, ...]
    epoch_book_id: str
    pair_namespace: str
    bracket_saturated: bool = False
    reference_book_epoch: str | None = None

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError(f"a pool epoch is numbered from 1, got {self.epoch!r}")
        if self.anchor_id not in self.roster_map:
            raise ValueError(
                f"anchor {self.anchor_id!r} is absent from the manifest's roster "
                f"map; the anchor is carried in the record (ladder.py:218-241)"
            )
        if PAIR_ID_SEPARATOR in self.pair_namespace:
            raise ValueError(
                f"pair_namespace {self.pair_namespace!r} contains "
                f"{PAIR_ID_SEPARATOR!r}, which the pair id is joined with "
                f"(arena.py:691)"
            )

    @classmethod
    def of(
        cls,
        roster: PoolRoster,
        schedule: EpochSchedule,
        *,
        epoch_book_id: str,
        reference_book_epoch: str | None = None,
    ) -> EpochManifest:
        """Build a manifest from a roster and one epoch's schedule."""
        return cls(
            epoch=schedule.epoch,
            anchor_id=roster.anchor_id,
            roster_map=roster.roster_map,
            schedule=schedule.matchups,
            epoch_book_id=epoch_book_id,
            pair_namespace=pair_namespace_for(schedule.epoch),
            bracket_saturated=schedule.bracket_saturated,
            reference_book_epoch=reference_book_epoch,
        )


def pair_namespace_for(epoch: int) -> str:
    """Return the epoch's ``pair_namespace`` (§D2.3): ``E007`` at epoch 7.

    The ``pair_id`` carries no epoch and no seed (``arena.py:645-654``), so a
    repeat of a matchup collides and fails ``check_paired_books`` with
    ``GAME_COUNT`` violations. The namespace is what makes an append-only,
    multi-epoch record set concatenate safely.
    """
    if epoch < 1:
        raise ValueError(f"a pool epoch is numbered from 1, got {epoch!r}")
    return f"E{epoch:03d}"
