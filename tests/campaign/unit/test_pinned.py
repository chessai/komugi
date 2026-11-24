"""The pre-registered constants: measured once, stamped into the root, re-checked.

KOM-022 §3b/§4/§5, acceptances #10, #16 and #40. A threshold that can be edited
without the edit being visible is not pre-registered; it is a number that will
eventually be moved to make a red campaign green. Two mechanisms were supposed
to make that real, and this file gates what is left of them:

* **the band was derived, not chosen** -- ``(d/(1-d))**2`` divided and
  multiplied by ``tolerance**2``. That derivation is **refuted** (KOM-147,
  KOM-149): measured ``r`` never came within a factor of ten of it over 184
  iterations across nine roots. The band is now pinned on a *measured* centre,
  :data:`~komugi.campaign.pinned.RATIO_CENTRE`, so the property this file can
  still gate is the arithmetic one -- the shipped bounds are exactly that
  constant spread by the declared tolerance, so moving a bound means moving a
  declared constant in a commit;
* **the constants are stamped into the root at iteration 0** and re-checked at
  every later iteration, so a mid-campaign change makes the gate refuse to run,
  naming both values. ``git merge-base --is-ancestor`` bans back-dating; it
  does not ban tuning, and this does.

With the centre measured rather than derived, the second mechanism carries the
whole argument: a number fitted to campaigns is one commit away from being
re-fitted to the next red one, and the only thing standing between those two
states is that a running root refuses the edit. Stated rather than glossed,
because the design states it: **neither ``RATIO_TOLERANCE = 2`` nor
``RATIO_CENTRE`` has a derivation.** The centre at least has a measurement, on
named roots whose held-out majority did not choose it. The mitigation is
mechanical, not evidential, and this file is that mechanism rather than an
argument for the numbers.
"""

from __future__ import annotations

from typing import Final

import pytest

from komugi.campaign.pinned import (
    BAND_PATIENCE,
    CONSTANTS_KEYS,
    DENOMINATOR_FLOOR,
    DISPLACEMENT_BAND,
    FLAT_TRAINER,
    FLAT_TRAINER_PATIENCE,
    GAUNTLET_BREACH_PATIENCE,
    GAUNTLET_EVERY,
    GAUNTLET_FLOOR,
    GAUNTLET_GAMES,
    GAUNTLET_MIN_DECISIVE,
    GAUNTLET_SHARD_DEADLINE_S,
    HASH_GATE,
    LOSS_SPIKE,
    MIN_ANCHOR_AGE,
    MUTE_DEFAULTS,
    RATIO_CENTRE,
    RATIO_TOLERANCE,
    SPIKE_FACTOR,
    TRIPWIRES,
    UNMUTABLE,
    WARMUP_ITERATIONS,
    ConstantsDrift,
    MissingConstants,
    band,
    check_against_root,
    check_first_record,
    gauntlet_iteration,
    mute_length,
    pinned_constants,
)
from komugi.telemetry.liveness import ratio_band
from komugi.train.ema import EMA_DECAY

CAMPAIGN_6_ROOT: Final[dict[str, float]] = {
    "ema_decay": 0.75,
    "ratio_tolerance": 2.0,
    "warmup_iterations": 4.0,
    "denominator_floor": 1e-06,
    "min_anchor_age": 12.0,
    "spike_factor": 3.0,
}
"""Campaign 6's iteration-0 ``extras``, verbatim -- a root stamped *before*
``ratio_centre`` existed.

A literal rather than a hand-typed approximation of one, because the property
it tests is about a record that is on disk right now: the re-pin has to land
under a live campaign without turning its resume into a ``MissingConstants`` or
a drift refusal (review F4/F9).
"""


def test_the_shipped_band_is_the_measured_centre_and_the_declared_tolerance() -> None:
    """Acceptance #10, re-pinned: ``(0.13325, 2.132)`` from 0.533 and t = 2.

    The band's *centre* is no longer derived from :data:`EMA_DECAY` -- the
    derivation is refuted (KOM-149) -- so this asserts what survives: the
    bounds are exactly the declared constants, and the refuted centre 9 now
    sits outside the band it used to be the middle of.
    """
    assert band() == ratio_band(centre=RATIO_CENTRE, tolerance=RATIO_TOLERANCE)
    assert band() == pytest.approx((0.13325, 2.132), abs=1e-12)

    low, high = band()
    assert low < RATIO_CENTRE < high
    assert high < 9.0, "the refuted derived centre is inside the re-pinned band"


def test_the_centre_is_the_measured_median_and_is_stamped() -> None:
    """The re-pin's whole anti-re-tuning guard, in two assertions.

    A measured threshold is one commit away from being re-measured on whichever
    campaign it has just failed. What stops that here is not a derivation --
    there isn't one -- it is that the centre is stamped into the root at
    iteration 0 and re-checked at every later iteration, so re-fitting it
    mid-campaign is a refusal rather than an edit nobody sees.
    """
    assert RATIO_CENTRE == 0.533, "the post-warm-up median, to three decimals"
    assert pinned_constants()["ratio_centre"] == 0.533
    assert "ratio_centre" in CONSTANTS_KEYS


def test_the_denominator_floor_is_derived_from_the_dtype() -> None:
    """§3b guard 2: one decade above float32's accumulation noise floor.

    ``1e-6`` is not a statement about this net's learning scale -- it is
    ``~1.2e-7`` (float32's relative resolution) with a decade of margin, and a
    mean over 512 positions in float64 does not reduce a *bias* of that size.
    It moves only if the dtype does, which is what makes an otherwise absolute
    number defensible in a section that argues absolute numbers cannot be
    pinned in advance.
    """
    float32_resolution = 1.2e-7
    assert DENOMINATOR_FLOOR == 1e-6
    assert 5.0 < DENOMINATOR_FLOOR / float32_resolution < 20.0


def test_the_escalation_patiences_are_the_designs() -> None:
    """Who records first and who pauses first, in one place.

    The hash gate, the cross-role invariant and the aliasing check pause on the
    **first** occurrence and are unmutable. The band pauses on the second, the
    flat-trainer verdict on the third (one sub-floor iteration is a converged
    trainer; three in a row is one that has stopped), and the gauntlet floor on
    the second consecutive breach within one anchor generation.
    """
    assert BAND_PATIENCE == 2
    assert FLAT_TRAINER_PATIENCE == 3
    assert GAUNTLET_BREACH_PATIENCE == 2
    assert WARMUP_ITERATIONS == 4, "one EMA horizon"
    assert SPIKE_FACTOR == 3.0


def test_the_gauntlet_constants_are_the_pre_registered_ones() -> None:
    """64 games, floor 0.40, and a validity condition that can actually fire."""
    assert GAUNTLET_GAMES == 64
    assert GAUNTLET_FLOOR == 0.40
    assert GAUNTLET_MIN_DECISIVE == 8
    assert MIN_ANCHOR_AGE == 12, "three gauntlets at cadence 4"
    # The floor's derivation: a one-sided 95% upper bound on s below 0.50.
    bound = 0.50 - 1.645 * (0.25 / GAUNTLET_GAMES) ** 0.5
    assert GAUNTLET_FLOOR == pytest.approx(bound, abs=0.005)


WORST_MEASURED_GAUNTLET_S: Final[float] = 384.2
"""The longest gauntlet either campaign has recorded, in seconds of wall.

``gauntlet_wall_s`` over campaign 6's 27 gauntlets: min 129.9, median 209.0,
p90 353.6, **max 384.2** (iteration 55); campaign 5's 7 gauntlets peak at 304.8.
Read off the journals by the KOM-180 implementation review, which is also where
the number this replaces was shown to be wrong: the deadline's docstring
justified itself by "about two minutes for all sixteen shards together", which
is below the observed *minimum*, and the test that was supposed to pin it
carried the same 120.0 s -- so its only substantive assertion was
``> 10 x 120``, which any pin above 1200 s satisfies. The value was restated in
a literal; the *basis* was pinned to nothing (review F2).

The whole-gauntlet number is the right comparison because the shards run
**concurrently** (``run_gauntlet`` sizes its pool at one worker per request), so
a shard's wall is approximately the gauntlet's.
"""

DEADLINE_MULTIPLE: Final[float] = 10.0
"""The multiple of :data:`WORST_MEASURED_GAUNTLET_S` the deadline claims.

Stated here as well as in the constant's docstring so that raising the pin is
an edit to a declared multiple rather than to an opaque number of seconds: the
deadline weakens the stall detector in exact proportion to how far it is raised.
"""


def test_the_gauntlet_deadline_clears_the_worst_measured_gauntlet() -> None:
    """The pin is ten times the worst gauntlet on record, and says so.

    Two failures this forbids, in the two directions:

    * **too low** -- a healthy-but-loaded gauntlet is killed, which costs a
      ``GauntletStalled``, the boundary's crash ``PAUSED``, an unmutable
      ``unfinished_boundary`` on the next startup and a manual rollback, with no
      retry anywhere in that chain. The lower bound is the real measured
      maximum, not a remembered one;
    * **too high** -- the detector finds a wedged gauntlet one multiple later,
      so silently inflating the bound is caught by the upper bound here and has
      to be a visible edit to :data:`DEADLINE_MULTIPLE`.

    The predecessor pin (1800 s) fails the lower bound, which is what makes this
    a test rather than a restatement of the constant.
    """
    assert GAUNTLET_SHARD_DEADLINE_S == 3850.0, "10 x 385 s, rounded up from 384.2"
    assert GAUNTLET_SHARD_DEADLINE_S >= DEADLINE_MULTIPLE * WORST_MEASURED_GAUNTLET_S, (
        f"the deadline is {GAUNTLET_SHARD_DEADLINE_S:g}s, under the "
        f"{DEADLINE_MULTIPLE:g}x of the worst recorded gauntlet "
        f"({WORST_MEASURED_GAUNTLET_S:g}s) its docstring claims"
    )
    assert GAUNTLET_SHARD_DEADLINE_S <= (
        DEADLINE_MULTIPLE + 1.0
    ) * WORST_MEASURED_GAUNTLET_S, (
        "the deadline is more than the multiple it declares: a stall is then "
        "detected later than the docstring says, without the docstring changing"
    )
    assert 1800.0 < DEADLINE_MULTIPLE * WORST_MEASURED_GAUNTLET_S, (
        "the pin this replaced would satisfy the bound above; the test would "
        "pin nothing"
    )

    # And it stays OUT of the stamped set, deliberately: a supervisor's bound is
    # not a measurement a result depends on, so correcting it must not make
    # `check_against_root` refuse a campaign that is already running.
    assert "gauntlet_shard_deadline_s" not in CONSTANTS_KEYS
    assert "gauntlet_shard_deadline_s" not in pinned_constants()


@pytest.mark.parametrize(
    ("every", "expected"),
    [(4, [3, 7, 11]), (2, [1, 3, 5, 7, 9, 11]), (1, list(range(12)))],
)
def test_cadence(every: int, expected: list[int]) -> None:
    """Acceptance #16: ``(i + 1) % k == 0``, and nowhere else.

    At cadence 4 the gauntlets are iterations 3, 7 and 11 -- the *last*
    iteration of each block, which is the one whose EMA is worth measuring.
    """
    fired = [i for i in range(12) if gauntlet_iteration(i, every)]
    assert fired == expected


def test_the_shipped_cadence_is_the_provisional_four() -> None:
    """§6: 4 is the conservative default until the measurement lands.

    The cost model is a derivation from self-play throughput, not a
    measurement, and every one of its errors points toward affordability -- so
    ``k`` may only ever be **lowered**, by the pre-registered rule, and raising
    it is a ticket.
    """
    assert GAUNTLET_EVERY == 4
    assert gauntlet_iteration(3)
    assert not gauntlet_iteration(2)


def test_a_cadence_below_one_is_refused() -> None:
    """Cadence 0 would run a gauntlet never, or always, depending on the modulo."""
    with pytest.raises(ValueError, match="cadence must be >= 1"):
        gauntlet_iteration(3, 0)


# ---------------------------------------------------------------------------
# the mute table
# ---------------------------------------------------------------------------


def test_the_mute_defaults_are_per_tripwire() -> None:
    """§5/F7: a single global 2 was rev 1's, and its rationale is spike-only.

    A band violation caused by non-constant displacement does not re-establish
    in two iterations, so a global mute would expire and the campaign would
    re-pause on the tripwire the operator had just acked -- the trap the ack
    mechanism exists to avoid, reintroduced for a different tripwire. And a
    gauntlet mute shorter than the gauntlet cadence mutes nothing at all.
    """
    assert MUTE_DEFAULTS[LOSS_SPIKE] == 2
    assert MUTE_DEFAULTS[DISPLACEMENT_BAND] == 8, "two EMA horizons"
    assert MUTE_DEFAULTS[FLAT_TRAINER] == 4, "one EMA horizon"
    assert MUTE_DEFAULTS["gauntlet_floor"] >= GAUNTLET_EVERY, (
        "a mute shorter than the gauntlet cadence mutes nothing"
    )


def test_an_ack_may_shorten_a_mute_and_never_lengthen_one() -> None:
    """A longer mute is a ticket, not an operator decision."""
    assert mute_length(LOSS_SPIKE) == MUTE_DEFAULTS[LOSS_SPIKE]
    assert mute_length(LOSS_SPIKE, 1) == 1
    assert mute_length(LOSS_SPIKE, 0) == 0

    with pytest.raises(ValueError, match="never lengthen"):
        mute_length(LOSS_SPIKE, MUTE_DEFAULTS[LOSS_SPIKE] + 1)
    with pytest.raises(ValueError, match=">= 0"):
        mute_length(LOSS_SPIKE, -1)


def test_the_unmutable_gates_have_no_length_at_all() -> None:
    """Acceptance #20's constant half: an unmutable gate cannot acquire a mute.

    A frozen self-play net is never a false alarm, and the aliasing check is
    assumption-free, so neither has a false-alarm mode to mute.
    """
    assert UNMUTABLE <= TRIPWIRES
    assert UNMUTABLE.isdisjoint(MUTE_DEFAULTS)
    for gate in UNMUTABLE:
        with pytest.raises(ValueError, match="not mutable at any length"):
            mute_length(gate)


def test_an_unknown_tripwire_has_no_mute() -> None:
    """A mute for a gate nobody declared would suppress an unknown thing."""
    with pytest.raises(ValueError, match="unknown tripwire"):
        mute_length("no_such_gate")


# ---------------------------------------------------------------------------
# acceptance #40 -- the constants are pinned to the root
# ---------------------------------------------------------------------------


def test_band_constants_are_pinned_to_the_root() -> None:
    """Acceptance #40: a mid-campaign re-band makes the gate refuse to run.

    Widening the band mid-campaign is then indistinguishable from tampering,
    which is the property the pre-registration was reaching for and which
    ``--is-ancestor`` alone does **not** deliver -- it bans back-dating, not
    tuning.
    """
    stamped = pinned_constants()
    assert set(stamped) == set(CONSTANTS_KEYS)
    assert all(isinstance(value, float) for value in stamped.values()), (
        "extras values must be floats or the record refuses to construct"
    )

    check_against_root(stamped)  # the healthy case: no drift, no raise.

    for key in CONSTANTS_KEYS:
        drifted = dict(stamped)
        drifted[key] = stamped[key] * 2.0 + 1.0
        with pytest.raises(ConstantsDrift) as raised:
            check_against_root(drifted)
        message = str(raised.value)
        assert key in message
        assert repr(stamped[key]) in message, "the live value is not named"
        assert repr(drifted[key]) in message, "the root's value is not named"


def test_a_live_root_stamped_before_the_re_pin_still_resumes() -> None:
    """KOM-149 F4: the re-pin lands under a running campaign, or it does not land.

    Campaign 6's root was stamped before ``ratio_centre`` was a constant, so
    its iteration-0 record carries the six older keys and not the new one. The
    band changed underneath it; none of the *stamped* values did. So the drift
    check must accept it -- a key a root never recorded is not drift -- which is
    the property that lets this ticket ship without ending a live campaign.

    What that costs, recorded rather than glossed: iterations 27+ of campaign 6
    run under a centre its root never pinned, so its tail carries no
    tamper-evidence for the band. The first *pinned* run of the new band is
    campaign 7, which is also the first out-of-sample one, and the
    pre-registered claim is made there.
    """
    check_against_root(CAMPAIGN_6_ROOT)
    check_first_record(CAMPAIGN_6_ROOT)

    # And the other direction: a root that *did* stamp the old derived centre
    # under this key is a different experiment, and is refused by name.
    re_tuned = dict(CAMPAIGN_6_ROOT)
    re_tuned["ratio_centre"] = 9.0
    with pytest.raises(ConstantsDrift) as raised:
        check_against_root(re_tuned)
    message = str(raised.value)
    assert "ratio_centre" in message
    assert repr(RATIO_CENTRE) in message, "the live centre is not named"
    assert repr(9.0) in message, "the root's centre is not named"


def test_a_constant_the_root_never_recorded_is_not_drift() -> None:
    """A root started before a constant existed cannot have stamped it.

    Refusing there would make *adding* a constant a breaking change to every
    live root, which is a different failure from the one #40 is about.
    """
    check_against_root({})
    partial = {"ema_decay": float(EMA_DECAY)}
    check_against_root(partial)


def test_a_root_whose_first_record_has_no_constants_is_refused() -> None:
    """Review F9: "no keys recorded" must not silently mean "no drift".

    The loop hands ``records[0].extras`` to the drift check, trusting the first
    campaign record in the journal to be iteration 0's -- the one that stamped
    the constants. If that record is ever lost (a rewritten journal, a root
    assembled by hand), ``records[0]`` is some later iteration's, it carries
    **none** of the keys, ``check_against_root`` finds nothing to disagree with,
    and the pre-registration quietly stops being checked for the rest of the
    campaign with nothing on disk saying so.

    A record with *some* of the keys is still not an error -- that is the
    "a root started before a constant existed" case below, and refusing there
    would make adding a constant a breaking change to every live root. A record
    with none of them is a different animal, and it is a named refusal.
    """
    stamped = pinned_constants()
    check_first_record(stamped)

    with pytest.raises(MissingConstants) as raised:
        check_first_record({"rows": 512.0, "campaign_record": 1.0})
    message = str(raised.value)
    assert "iteration 0" in message
    for key in CONSTANTS_KEYS:
        assert key in message, "the refusal does not name what was expected"

    # An empty record is the same failure, not a special case.
    with pytest.raises(MissingConstants):
        check_first_record({})

    # A partial record still passes, and still catches drift among its keys.
    check_first_record({"ema_decay": float(EMA_DECAY)})
    with pytest.raises(ConstantsDrift):
        check_first_record({"ema_decay": float(EMA_DECAY) + 0.01})


def test_every_gate_name_is_a_declared_tripwire() -> None:
    """``PAUSED`` and the mute files may only ever carry a name from this set."""
    assert HASH_GATE in TRIPWIRES
    assert LOSS_SPIKE in TRIPWIRES
    assert TRIPWIRES == UNMUTABLE | set(MUTE_DEFAULTS)
