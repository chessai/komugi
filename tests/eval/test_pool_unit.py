"""§D1/§D2 unit nodes: entrant identity, the roster, and the epoch schedule.

Design: ``docs/design/kom-011d-ladder-pool.md`` (rev 3). This file carries §6
nodes 1-7, 16, 22, 38, 39, 42, 50 and 57. Every node names, in a comment, the
concrete wrong implementation it is red against -- a fixture that passes under
both the rule and its plausible mutant is a fixture that tests nothing, which
is the discipline the KOM-014 implementation review found missing.

Two of these nodes are about the *import graph* rather than about arithmetic:
node 39 pins ``pool.matchups`` against the ``ladder.matchups`` it was copied
from (this **test** may import ladder, and therefore torch; the module may
not), and node 42 spawns a fresh interpreter which builds a schedule and
constructs a roster before asserting torch never arrived.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from komugi.eval import anchor_aliases, pool
from komugi.eval.ab_endpoint import pool_components
from komugi.eval.divergence import WindowKind, window_for
from komugi.eval.elo import fit_elo
from komugi.eval.pool import (
    EpochManifest,
    PoolEntrant,
    PoolRoster,
    build_epoch_schedule,
    bracket_for,
    declared_unequal_matchup,
    family_of,
    frozen_subpool_schedule,
    rated_matchup,
    require_connected_pool,
    require_connected_schedule,
    schedule_match_set,
)
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
)

_CHILD_TIMEOUT_S: float = 180.0

_ANCHOR: str = "v1/iter-0020@7eef51d0"
_V1_0015: str = "v1/iter-0015@7eef51d1"
_MCTS_100: str = "rawmcts/n100@0f31ab70"
_MCTS_800: str = "rawmcts/n800@0f31ab77"
_MCTS_6400: str = "rawmcts/n6400@0f31ab78"
_EXTERNAL: str = "external/site-2026-09-08"

_MILESTONES: tuple[str, ...] = (
    "komugi/ckpt-3000@aaaa0001",
    "komugi/ckpt-6000@aaaa0002",
    "komugi/ckpt-9000@aaaa0003",
    "komugi/ckpt-12000@aaaa0004",
)

_REFERENCE_RATINGS: dict[str, float] = {
    _MCTS_100: -400.0,
    _MCTS_800: -200.0,
    _V1_0015: -120.0,
    _ANCHOR: 0.0,
    _MCTS_6400: 150.0,
}
"""A planted previous-epoch rating table over the reference field."""


def _reference(entrant_id: str, *, sims: float | None = None) -> PoolEntrant:
    """Return a frozen reference entrant: v1 is ceiling-bound, RawMCTS exempt."""
    if sims is not None:
        return PoolEntrant(
            id=entrant_id,
            family="rawmcts",
            frozen=True,
            config_digest=f"digest-{entrant_id}",
            compute={"sims_per_move": sims},
            ceiling_exempt=True,
        )
    return PoolEntrant(
        id=entrant_id,
        family="v1",
        frozen=True,
        config_digest=f"digest-{entrant_id}",
        compute={"evals_per_move": 257},
        ceiling_exempt=False,
        eval_ceiling=257,
    )


def _milestone(entrant_id: str, epoch: int) -> PoolEntrant:
    """Return a rated komugi milestone: frozen, and never a *reference* entrant."""
    return PoolEntrant(
        id=entrant_id,
        family="komugi",
        frozen=True,
        epoch=epoch,
        config_digest=f"digest-{entrant_id}",
        compute={"evals_per_move": 257},
        ceiling_exempt=False,
        eval_ceiling=257,
    )


def _anchor400(milestone: str) -> PoolEntrant:
    """Return the ``#anchor400`` variant: the same weights, another configuration."""
    return PoolEntrant(
        id=pool.anchor400_id(milestone),
        family="komugi",
        frozen=True,
        config_digest=f"digest-{milestone}",
        compute={"playouts_per_move": 400},
        ceiling_exempt=False,
        eval_ceiling=402,
    )


def _external() -> PoolEntrant:
    """Return the site entrant: not frozen, and its compute is unknown, not 0."""
    return PoolEntrant(
        id=_EXTERNAL,
        family="external",
        frozen=False,
        compute=None,
        ceiling_exempt=None,
    )


def _roster(*, milestones: int = 3) -> PoolRoster:
    """Return the standard fixture roster: 5 references, N milestones, external."""
    entrants: list[PoolEntrant] = [
        _reference(_ANCHOR),
        _reference(_V1_0015),
        _reference(_MCTS_100, sims=100.0),
        _reference(_MCTS_800, sims=800.0),
        _reference(_MCTS_6400, sims=6400.0),
        _external(),
    ]
    for index in range(milestones):
        entrants.append(_milestone(_MILESTONES[index], index + 1))
        entrants.append(_anchor400(_MILESTONES[index]))
    return PoolRoster(entrants=tuple(entrants), anchor_id=_ANCHOR)


def _mini_records(first: str, second: str, pairs: int) -> list[GameRecord]:
    """Return `pairs` seat-swapped, mixed-result pairs between two players.

    Mixed on purpose: an identified mini-match, so a fixture built from these
    converges and node 22's "the fit cannot be its own detector" half is a real
    demonstration rather than an artefact of a degenerate record set.
    """
    records: list[GameRecord] = []
    for index in range(pairs):
        outcome = (WHITE_WIN, DRAW, BLACK_WIN)[index % 3]
        pair_id = f"{first}:{second}:{index:05d}"
        records.append(
            GameRecord(
                white=first,
                black=second,
                result=outcome,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=second,
                black=first,
                result=WHITE_WIN - outcome,
                book_id="E001-book",
                pair_id=pair_id,
            )
        )
    return records


def _epoch_schedule(
    roster: PoolRoster, *, epoch: int, predicted: float
) -> pool.EpochSchedule:
    """Build the §D2.2(b) schedule for `epoch` over the fixture rating table."""
    return build_epoch_schedule(
        roster,
        epoch=epoch,
        milestone=_MILESTONES[epoch - 1],
        predicted_rating=predicted,
        ratings=_REFERENCE_RATINGS,
    )


# --------------------------------------------------------------------------
# §6 node 1
# --------------------------------------------------------------------------
def test_an_entrant_id_containing_a_colon_is_refused() -> None:
    """A ``:`` in an id makes a pair_id that cannot be parsed back (§D1.2)."""
    # Red against: no id validation at all, or validation that only checks
    # non-emptiness. The non-vacuity half is that `Player` -- the record-level
    # type -- *accepts* the same id, so this node is testing the pool's own
    # rule rather than restating one records.py already has.
    with pytest.raises(ValueError) as excinfo:
        PoolEntrant(id="komugi/ckpt:15000@a1b2c3d4", family="komugi", frozen=True)
    message = str(excinfo.value)
    assert "pair_id" in message
    assert "arena.py:691" in message

    accepted = Player(id="komugi/ckpt:15000@a1b2c3d4", family="komugi")
    assert accepted.id == "komugi/ckpt:15000@a1b2c3d4"


# --------------------------------------------------------------------------
# §6 node 2
# --------------------------------------------------------------------------
def test_a_frozen_entrant_without_a_digest_is_refused() -> None:
    """§D1.2 property 2: the digest is part of the id, not beside it."""
    # Red against: requiring `@sha8` of every entrant (which would refuse
    # external, whose date is all there is to pin -- U4), and against requiring
    # it of none (which lets `komugi/ckpt-15000` collide across campaigns,
    # the collision ab_endpoint.require_distinct_ids exists for).
    with pytest.raises(ValueError, match="@sha8"):
        PoolEntrant(id="komugi/ckpt-15000", family="komugi", frozen=True)

    digested = PoolEntrant(
        id="komugi/ckpt-15000@a1b2c3d4", family="komugi", frozen=True
    )
    assert digested.frozen

    site = _external()
    assert not site.frozen
    assert site.id == _EXTERNAL


# --------------------------------------------------------------------------
# §6 node 3
# --------------------------------------------------------------------------
def test_a_non_frozen_entrant_cannot_be_the_anchor_or_enter_the_frozen_subpool(
) -> None:
    """§D1.1: a gauge that can move without a diff is not a gauge."""
    # Red against: a roster that stores `anchor` as a plain string with no
    # check, which is how the site entrant -- the one member of this pool that
    # can change silently (U4) -- ends up defining 0 Elo for the campaign.
    entrants = _roster().entrants
    with pytest.raises(ValueError, match="not frozen"):
        PoolRoster(entrants=entrants, anchor_id=_EXTERNAL)

    good = PoolRoster(entrants=entrants, anchor_id=_ANCHOR)
    assert good.anchor.id == _ANCHOR

    with pytest.raises(ValueError, match="not a reference entrant"):
        frozen_subpool_schedule(good, (_ANCHOR, _V1_0015, _EXTERNAL))


# --------------------------------------------------------------------------
# §6 node 4
# --------------------------------------------------------------------------
def test_two_entrants_over_one_checkpoint_with_different_ids_are_refused() -> None:
    """§D1.1: a milestone enters the pool exactly once and never re-enters."""
    # Red against: a driver that re-mints an id per epoch (two rows, one net,
    # each holding half the evidence). The non-vacuity half is the #anchor400
    # variant, which shares the milestone's digest *by design* (§D6.1) and
    # must still construct -- so the rule is stated over the base id.
    first = _milestone(_MILESTONES[0], 1)
    duplicate = PoolEntrant(
        id="komugi/ckpt-3001@aaaa0001",
        family="komugi",
        frozen=True,
        epoch=2,
        config_digest=first.config_digest,
    )
    with pytest.raises(ValueError) as excinfo:
        PoolRoster(
            entrants=(_reference(_ANCHOR), first, duplicate), anchor_id=_ANCHOR
        )
    message = str(excinfo.value)
    assert first.id in message
    assert duplicate.id in message

    shared = PoolRoster(
        entrants=(_reference(_ANCHOR), first, _anchor400(first.id)),
        anchor_id=_ANCHOR,
    )
    assert len(shared.entrants) == 3


# --------------------------------------------------------------------------
# §6 node 5
# --------------------------------------------------------------------------
def test_family_is_read_from_the_roster_not_the_prefix() -> None:
    """§D1.2 property 3: the prefix is legibility; the map is the truth."""
    # Red against: `family = entrant_id.split("/")[0]` anywhere in a reader.
    # The prefix-free id below is what makes that mutant fail: a real ladder
    # record set holds bare `ckpt-<step>` ids (arena.py:811-818), and only the
    # roster map can say what family they belong to.
    with pytest.raises(ValueError) as excinfo:
        PoolEntrant(id="komugi/ckpt-1@aaaaaaaa", family="v1", frozen=True)
    message = str(excinfo.value)
    assert "komugi" in message
    assert "v1" in message

    roster_map: Mapping[str, str] = {"ckpt-15000": "komugi", _ANCHOR: "v1"}
    assert family_of(roster_map, "ckpt-15000") == "komugi"
    assert family_of(roster_map, _ANCHOR) == "v1"
    with pytest.raises(ValueError, match="not in the roster map"):
        family_of(roster_map, "komugi/ckpt-99999@deadbeef")

    # The roster's own map is records.roster_families' output, reused.
    roster = _roster()
    assert roster.roster_map[_EXTERNAL] == "external"
    assert roster.roster_map[_MILESTONES[0]] == "komugi"


@pytest.fixture
def _legacy_anchor_alias_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point ``KOMUGI_ANCHOR_ALIASES`` at a placeholder ``{legacy_anchor: external}``.

    `monkeypatch` undoes the env var on teardown, but the module's
    ``lru_cache`` does not know that: without a matching :func:`reset_cache`
    here, the placeholder table it cached would stay the answer every later
    test's :func:`family_of` gets, env var gone or not.
    """
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps({"legacy_anchor": "external"}), encoding="utf-8")
    monkeypatch.setenv("KOMUGI_ANCHOR_ALIASES", str(alias_path))
    anchor_aliases.reset_cache()
    yield
    anchor_aliases.reset_cache()


def test_family_of_maps_an_old_name_through_komugi_anchor_aliases(
    _legacy_anchor_alias_table: None,
) -> None:
    """C1: data stored under a pre-rename family name still reads as the new one.

    ``KOMUGI_ANCHOR_ALIASES`` names a JSON file of ``{old: new}``; the table
    here uses a placeholder old token (``legacy_anchor``) rather than any real
    pre-rename name, since this unit's rename makes the real one obsolete in
    this file's own fixtures. Red against a reader that parses the family from
    the id's prefix instead of going through :func:`family_of` -- that mutant
    would see ``"legacy_anchor"`` on the roster map's own value and never
    consult the alias table at all.
    """
    roster_map: Mapping[str, str] = {"legacy_anchor/site-2026-09-08": "legacy_anchor"}
    assert family_of(roster_map, "legacy_anchor/site-2026-09-08") == "external"


def test_family_of_names_no_old_token_when_komugi_anchor_aliases_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1: with the variable unset, `family_of` maps nothing -- the identity.

    This is the other half of the backward-compatibility contract: public
    code must never name an old family token on its own, so an unmapped value
    passes through `family_of` unchanged rather than being guessed at.
    """
    monkeypatch.delenv("KOMUGI_ANCHOR_ALIASES", raising=False)
    anchor_aliases.reset_cache()

    roster_map: Mapping[str, str] = {"legacy_anchor/site-2026-09-08": "legacy_anchor"}
    assert family_of(roster_map, "legacy_anchor/site-2026-09-08") == "legacy_anchor"


# --------------------------------------------------------------------------
# §6 node 6
# --------------------------------------------------------------------------
@pytest.mark.parametrize("epoch", [1, 3])
def test_the_schedule_has_exactly_six_matchups_for_a_new_milestone(
    epoch: int,
) -> None:
    """§D2.2(b): six matchups per epoch, and exactly five at ``E = 1``."""
    # Red against an implementation that raises at E = 1 (there is no
    # predecessor, and the first epoch still has to be played), and against one
    # that silently emits a sixth self-link `milestone x milestone` to keep the
    # count at six. Asserted by identity and game count, never by count alone.
    roster = _roster()
    schedule = _epoch_schedule(roster, epoch=epoch, predicted=50.0)
    milestone = _MILESTONES[epoch - 1]
    variant = pool.anchor400_id(milestone)

    expected: list[tuple[frozenset[str], int]] = []
    if epoch >= 2:
        expected.append(
            (frozenset({_MILESTONES[epoch - 2], milestone}), pool.IN_FAMILY_GAMES)
        )
    expected.extend(
        [
            (frozenset({milestone, _ANCHOR}), pool.MATCHUP_GAMES),
            (frozenset({milestone, _V1_0015}), pool.MATCHUP_GAMES),
            (frozenset({milestone, _MCTS_6400}), pool.MATCHUP_GAMES),
            (frozenset({variant, _EXTERNAL}), pool.EXTERNAL_GAMES),
            (frozenset({variant, milestone}), pool.MATCHUP_GAMES),
        ]
    )
    assert len(schedule.matchups) == (6 if epoch >= 2 else 5)
    assert [(m.players, m.games) for m in schedule.matchups] == expected
    assert all(m.a != m.b for m in schedule.matchups)

    if epoch == 1:
        # The other half of the E = 1 state (R7): with no predecessor there is
        # no window, so both §D5 windows report "window not yet formed" rather
        # than an `ok` computed from a gain that does not exist.
        for kind in WindowKind:
            assert window_for(roster, epoch=1, kind=kind) is None


# --------------------------------------------------------------------------
# §6 node 7
# --------------------------------------------------------------------------
def test_the_bracket_is_the_nearest_reference_entrant_above_and_below() -> None:
    """§D2.2(b)3: nearest above and nearest below, over **reference** entrants."""
    roster = _roster()

    # (a) Non-vacuity: at this prediction the nearest-above/below pair is NOT
    # the two strongest, so "always pick the two strongest" fails here while
    # passing the saturation half below.
    bracket, saturated = bracket_for(
        roster, predicted_rating=-150.0, ratings=_REFERENCE_RATINGS
    )
    assert bracket == (_MCTS_800, _V1_0015)
    assert not saturated
    strongest = (_ANCHOR, _MCTS_6400)
    assert bracket != strongest

    # Saturation: past the top of the field the bracket degenerates to the two
    # strongest and says so, rather than reporting a number that has stopped
    # moving (§D2.2's anti-saturation argument, Q5).
    top, saturated_top = bracket_for(
        roster, predicted_rating=900.0, ratings=_REFERENCE_RATINGS
    )
    assert top == strongest
    assert saturated_top

    # (b) The load-bearing fixture (F4): the milestone's own two predecessors
    # are the nearest-above and nearest-below entrants *overall*. They are
    # `frozen`, so an implementation reading §D2.2(b)3 over `frozen` instead of
    # `reference` selects them, G_cross loses every non-anchor edge, and the
    # pool becomes the mirror DESIGN.html:260 exists to prevent.
    with_milestones = dict(_REFERENCE_RATINGS)
    with_milestones[_MILESTONES[0]] = -155.0
    with_milestones[_MILESTONES[1]] = -145.0
    skipped, _ = bracket_for(
        roster, predicted_rating=-150.0, ratings=with_milestones
    )
    assert skipped == (_MCTS_800, _V1_0015)
    assert all(not name.startswith("komugi/") for name in skipped)

    # (c) Below the whole field (F9): the bracket degenerates to the two
    # *weakest*, which is correct scheduling -- and it is NOT saturated. The
    # branch was untested in both directions, so `saturated=True` (the
    # symmetric reading of the no-`above` branch above) survived every node.
    #
    # The decision, stated because the review left it to this task: `False` is
    # right. §D2.2's saturation signal means one thing -- the milestone's
    # bracket has walked *up* past the top of the reference field and the pool
    # has run out of opponents -- and §D5.3 condition 6 ties `bracket_saturated`
    # on a rating row to "the gate can no longer see". A first-epoch milestone
    # weaker than rawmcts/n100 is a young instrument, not a blind one: its
    # links are the most informative the pool will ever have. Under `True`,
    # epoch 1 publishes `bracket_saturated`, the report says the pool has run
    # out of opponents, and §Q5's pre-registered response (extend the reference
    # field) fires at exactly the moment the field is at its most adequate.
    below, saturated_below = bracket_for(
        roster, predicted_rating=-900.0, ratings=_REFERENCE_RATINGS
    )
    assert below == (_MCTS_100, _MCTS_800)
    assert not saturated_below


# --------------------------------------------------------------------------
# §6 node 16
# --------------------------------------------------------------------------
def test_a_matchup_between_two_ceiling_bound_entrants_at_different_ceilings_is_refused(
) -> None:
    """§D7.4: refused on the rated path; exempt and declared links are not."""
    # Red against a scheduler that plays any two entrants and discovers the
    # mismatch when CountedEvaluator raises mid-match (an M8 void, after the
    # games have been paid for), and against one that "fixes" the mismatch by
    # raising the lower ceiling -- which redefines a frozen anchor silently.
    milestone = _milestone(_MILESTONES[2], 3)
    variant = _anchor400(_MILESTONES[2])
    with pytest.raises(ValueError, match="different ceilings"):
        rated_matchup(milestone, variant, games=128, kind=pool.KIND_TIE_IN)

    exempt = rated_matchup(
        milestone,
        _reference(_MCTS_800, sims=800.0),
        games=128,
        kind=pool.KIND_BRACKET,
    )
    assert exempt.exempt == (_MCTS_800,)
    assert exempt.unequal_compute

    # The two matchups the design schedules at declared-unequal compute.
    tie_in = declared_unequal_matchup(
        variant, milestone, games=128, kind=pool.KIND_TIE_IN
    )
    assert tie_in.unequal_compute
    visits = PoolEntrant(
        id=f"{_MILESTONES[2]}#n1600",
        family="komugi",
        frozen=True,
        config_digest=f"digest-{_MILESTONES[2]}",
        compute={"evals_per_move": 1601},
        ceiling_exempt=False,
        eval_ceiling=1601,
    )
    visits_leg = declared_unequal_matchup(
        visits, milestone, games=32, kind="visits"
    )
    assert visits_leg.unequal_compute


# --------------------------------------------------------------------------
# §6 node 22
# --------------------------------------------------------------------------
def test_a_disconnected_pool_is_refused_before_the_fit_in_the_pools_own_words(
) -> None:
    """§D2.2: the fit cannot be its own detector, so the refusal is the pool's."""
    # Red against reusing ab_endpoint.fit_cross_arm_pool's refusal, whose
    # message is KOM-036's vocabulary (two within-experiment chains, and a
    # reference id from a different experiment) and would report this pool's
    # failure in terms of objects that do not exist here.
    records = _mini_records(_MILESTONES[0], _MILESTONES[1], 9)
    records.extend(_mini_records(_ANCHOR, _MCTS_800, 9))
    match_set = MatchSet.of(records)
    roster_map = {
        _MILESTONES[0]: "komugi",
        _MILESTONES[1]: "komugi",
        _ANCHOR: "v1",
        _MCTS_800: "rawmcts",
    }

    with pytest.raises(ValueError) as excinfo:
        require_connected_pool(match_set, roster_map)
    message = str(excinfo.value)
    assert "2 components" in message
    for player, family in roster_map.items():
        assert player in message
        assert f"[{family}]" in message
    assert "arm" not in message
    assert "REFERENCE_ID" not in message
    assert "ref/soak49-15000" not in message

    # Non-vacuity: the same records fit happily and report converged=True --
    # each component finds its own gauge, which is why the fit cannot detect
    # this and the check has to run before it.
    fitted = fit_elo(match_set, anchor=_ANCHOR)
    assert fitted.converged


# --------------------------------------------------------------------------
# §6 node 38
# --------------------------------------------------------------------------
def test_the_epoch_schedule_is_connected_including_the_external_leg() -> None:
    """§D2.2: ``pool_components == 1`` on the **designed** schedule, asserted."""
    roster = _roster()
    order = (_MCTS_100, _MCTS_800, _V1_0015, _ANCHOR, _MCTS_6400)
    full = (
        *frozen_subpool_schedule(roster, order),
        *_epoch_schedule(roster, epoch=3, predicted=50.0).matchups,
    )
    components = require_connected_schedule(full, roster.roster_map)
    assert len(components) == 1
    assert _EXTERNAL in components[0]

    # Non-vacuity: remove §D2.2(b)5's tie-in -- the schedule rev 1 had -- and
    # {external, ...#anchor400} is a two-vertex component with no edge to the
    # pool, so the driver would refuse on *every* epoch.
    without_tie_in = tuple(
        matchup for matchup in full if matchup.kind != pool.KIND_TIE_IN
    )
    split = pool_components(schedule_match_set(without_tie_in))
    assert len(split) == 2
    assert frozenset({_EXTERNAL, pool.anchor400_id(_MILESTONES[2])}) in split


# --------------------------------------------------------------------------
# §6 node 39
# --------------------------------------------------------------------------
def test_the_copied_matchup_schedule_agrees_with_the_ladders() -> None:
    """The copy §5's torch rule forces cannot drift from its original."""
    # Red against an edited copy (a rung-major loop, an off-by-one on the
    # reach) -- the whole risk of copying rather than importing. This *test*
    # imports ladder, and therefore torch; the module under test may not, which
    # is what node 42's spawned child pins.
    from komugi.eval.ladder import matchups as ladder_matchups

    for count in range(2, 13):
        assert pool.matchups(count, 2) == ladder_matchups(count, 2)
    assert pool.matchups(7, pool.FROZEN_SUBPOOL_ADJACENCY) == ladder_matchups(7, 2)
    assert len(pool.matchups(7, 2)) == 2 * 7 - 3


# --------------------------------------------------------------------------
# §6 node 42
# --------------------------------------------------------------------------
_CHILD_PROGRAM: str = '''
import sys

from komugi.eval import divergence, interval, pool, pool_report
from komugi.eval.records import GameRecord, MatchSet

anchor = pool.PoolEntrant(
    id="v1/iter-0020@7eef51d0", family="v1", frozen=True,
    compute={"evals_per_move": 257}, ceiling_exempt=False, eval_ceiling=257,
)
reference = pool.PoolEntrant(
    id="rawmcts/n800@0f31ab77", family="rawmcts", frozen=True,
    compute={"sims_per_move": 800}, ceiling_exempt=True,
)
weak = pool.PoolEntrant(
    id="rawmcts/n100@0f31ab70", family="rawmcts", frozen=True,
    compute={"sims_per_move": 100}, ceiling_exempt=True,
)
site = pool.PoolEntrant(
    id="external/site-2026-09-08", family="external", frozen=False,
    compute=None, ceiling_exempt=None,
)
entrants = [anchor, reference, weak, site]
for step, epoch in ((3000, 1), (6000, 2)):
    milestone = "komugi/ckpt-%d@aaaa000%d" % (step, epoch)
    entrants.append(pool.PoolEntrant(
        id=milestone, family="komugi", frozen=True, epoch=epoch,
        compute={"evals_per_move": 257}, ceiling_exempt=False, eval_ceiling=257,
    ))
    entrants.append(pool.PoolEntrant(
        id=pool.anchor400_id(milestone), family="komugi", frozen=True,
        compute={"playouts_per_move": 400}, ceiling_exempt=False,
        eval_ceiling=402,
    ))

roster = pool.PoolRoster(entrants=tuple(entrants), anchor_id=anchor.id)
schedule = pool.build_epoch_schedule(
    roster,
    epoch=2,
    milestone="komugi/ckpt-6000@aaaa0002",
    predicted_rating=-150.0,
    ratings={anchor.id: 0.0, reference.id: -200.0, weak.id: -400.0},
)
assert len(schedule.matchups) == 6, schedule.matchups
components = pool.require_connected_schedule(schedule.matchups, roster.roster_map)
assert len(components) == 1, components
manifest = pool.EpochManifest.of(roster, schedule, epoch_book_id="E002-3f9c1a20")
assert manifest.pair_namespace == "E002"
assert divergence.threshold_for(divergence.WindowKind.SINGLE) == 75.0
assert divergence.threshold_for(divergence.WindowKind.SUSTAINED) == 90.0
low, high = interval.percentile_interval([0.0, 1.0, 2.0, 3.0])
assert abs(low - 0.075) < 1e-9 and abs(high - 2.925) < 1e-9, (low, high)

# pool_report is the fifth torch-free module and rev 4's node 42 predates it
# (T4 review F11): its "No torch" claim had no guard at all, and the module is
# one `from komugi.eval.ladder import _table` away from being false -- which is
# precisely the temptation its own copied `_table` records. Rendered, not
# merely imported, in the stronger form rev 3's M5 amendment requires.
match_set = MatchSet.of([
    GameRecord(white=anchor.id, black=reference.id, result=1.0,
               book_id="E002-3f9c1a20", pair_id="E002/pair:00000"),
    GameRecord(white=reference.id, black=anchor.id, result=0.0,
               book_id="E002-3f9c1a20", pair_id="E002/pair:00000"),
])
rows = pool_report.rating_rows(
    ratings={anchor.id: 0.0, reference.id: -200.0},
    intervals={anchor.id: (-10.0, 10.0), reference.id: (-215.0, -185.0)},
    roster_map=roster.roster_map,
    match_set=match_set,
    anchor_id=anchor.id,
)
report = pool_report.render_epoch_report(pool_report.EpochReport(
    header=pool_report.ReportHeader(
        epoch=2,
        epoch_book_id="E002-3f9c1a20",
        anchor_id=anchor.id,
        contempt=pool_report.contempt_for(roster.roster_map.values()),
        pinned_budget={"m": 32.0, "n": 256.0},
        engine_rules_version=3,
        r5_branch="proceed",
        entrants=len(entrants),
        matchups=len(schedule.matchups),
        games=len(match_set),
        epoch_games=len(match_set),
        converged=True,
    ),
    ratings=rows,
    match_set=match_set,
))
assert "ratings (family, then strength)" in report, report
assert pool_report.CLEAN_AUDIT in report, report
assert pool_report.FLAG_ANCHOR in report, report

if "torch" in sys.modules:
    raise SystemExit("torch reached a pure module: " + repr(sorted(
        name for name in sys.modules if name.startswith("torch")
    )[:5]))
'''


def test_the_pure_modules_import_no_torch_and_stay_clean_when_used() -> None:
    """§5's torch rule, at the strength §7 sells: **used**, not merely imported.

    **Review F11** adds ``komugi.eval.pool_report`` to the child, which is the
    fifth module claiming *"No torch"* and was the only one with no node
    behind the claim: node 42's list was written before the module existed.
    Red against a renderer that reaches ``komugi.eval.ladder`` for its table
    helper -- the module's own docstring records that temptation -- because
    ``ladder.py:58`` reaches ``arena.py:81`` reaches torch.
    """
    # Red against a deferred import (`def _seat(...): from komugi.eval.arena
    # import ArenaPlayer`), which honours "no torch at import time" to the
    # letter while making every one of this file's other nodes need torch on
    # the box. The child therefore builds a schedule and constructs a roster
    # *before* it looks at sys.modules. In-process asserts cannot see this at
    # all: another test module imports torch first and the check is vacuous.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        env=environment,
    )
    assert result.returncode == 0, (
        f"the pure-module child failed:\n{result.stdout}\n{result.stderr}"
    )


# --------------------------------------------------------------------------
# §6 node 50
# --------------------------------------------------------------------------
def test_an_unequal_compute_matchup_records_both_sides_compute() -> None:
    """§D7.5(2): both computes on every matchup, and external's stays ``None``."""
    # Red against a single `compute` field on the matchup (which cannot express
    # two units), against inheriting our compute for the site's unknown one,
    # and against flagging every cross-family link unequal regardless: the
    # milestone x v1 bracket link below is equal-compute and must read false.
    roster = _roster()
    schedule = _epoch_schedule(roster, epoch=3, predicted=50.0)
    milestone = _MILESTONES[2]
    variant = pool.anchor400_id(milestone)
    by_players = {matchup.players: matchup for matchup in schedule.matchups}

    for matchup in schedule.matchups:
        for side, compute in ((matchup.a, matchup.compute_a), (matchup.b, matchup.compute_b)):
            assert compute == roster.entrant(side).compute

    equal = by_players[frozenset({milestone, _V1_0015})]
    assert equal.compute_a == {"evals_per_move": 257}
    assert equal.compute_b == {"evals_per_move": 257}
    assert not equal.unequal_compute

    different_unit = by_players[frozenset({milestone, _MCTS_6400})]
    assert different_unit.compute_b == {"sims_per_move": 6400.0}
    assert different_unit.unequal_compute

    site_leg = by_players[frozenset({variant, _EXTERNAL})]
    assert site_leg.compute_b is None
    assert site_leg.unequal_compute

    tie_in = by_players[frozenset({variant, milestone})]
    assert tie_in.compute_a == {"playouts_per_move": 400}
    assert tie_in.compute_b == {"evals_per_move": 257}
    assert tie_in.unequal_compute


# --------------------------------------------------------------------------
# §6 node 57
# --------------------------------------------------------------------------
def test_the_in_family_link_names_the_epoch_predecessor_not_the_strongest() -> None:
    """§D2.2(b)1: the in-family link is the **epoch** predecessor (M2)."""
    # Red against "the highest-rated existing milestone", which agrees with the
    # rule on a monotone curve and disagrees on a dip -- and the campaign's own
    # curve is non-monotone (tickets/KOM-011.md:98). On a dip that mutant stops
    # G_in's chain being a path over consecutive epochs, so §D5.3's sqrt(3)
    # standard error (which assumes three adjacent links) is wrong by an
    # unknown amount, silently.
    roster = _roster(milestones=4)
    dipped = dict(_REFERENCE_RATINGS)
    dipped[_MILESTONES[0]] = 100.0
    dipped[_MILESTONES[1]] = 300.0  # the strongest komugi entrant
    dipped[_MILESTONES[2]] = 200.0  # the epoch-3 milestone: a dip
    schedule = build_epoch_schedule(
        roster,
        epoch=4,
        milestone=_MILESTONES[3],
        predicted_rating=250.0,
        ratings=dipped,
    )
    in_family = schedule.of_kind(pool.KIND_IN_FAMILY)
    assert len(in_family) == 1
    assert in_family[0].players == frozenset({_MILESTONES[3], _MILESTONES[2]})
    assert in_family[0].games == pool.IN_FAMILY_GAMES
    assert _MILESTONES[1] not in in_family[0].players


def test_the_epoch_manifest_carries_the_anchor_and_the_namespace() -> None:
    """§D9.2 property 3: the anchor is copied into the record, never re-derived."""
    # Red against a manifest that stores a path to the roster instead of the
    # map, or that re-derives the anchor by scanning for a 0.0 rating -- the
    # ladder's F1, which named the wrong player on any balanced ladder.
    roster = _roster()
    schedule = _epoch_schedule(roster, epoch=3, predicted=50.0)
    manifest = EpochManifest.of(
        roster, schedule, epoch_book_id="E003-3f9c1a20", reference_book_epoch="E001-aa"
    )
    assert manifest.anchor_id == _ANCHOR
    assert manifest.pair_namespace == "E003"
    assert manifest.roster_map[_EXTERNAL] == "external"
    assert manifest.reference_book_epoch == "E001-aa"
    assert len(manifest.schedule) == 6
