"""The four knobs KOM-245 and KOM-246 are written against (KOM-249).

Shared plumbing, landed ahead of both consumers: ``cap_adjudication`` and its
two ramp knobs (how a game stopped by the turn cap is labelled) and
``min_placements_max`` (the high edge of a drafted threshold). Nothing reads
them yet, which is exactly why these nodes exist -- a knob with no consumer has
only two properties worth pinning, and both are the operator's:

* **the defaults are today's behaviour**, so every construction written before
  these fields existed still means what it meant;
* **a value nobody can play is refused where the run is configured**, naming the
  field, rather than at the first capped game of a campaign that has been
  running for hours.

The profile half of the same change -- the four on the record, the digest that
moves with them -- is ``tests/campaign/unit/test_search_profile.py``.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from komugi.config import SelfPlayConfig


def _selfplay(**overrides: object) -> SelfPlayConfig:
    """The smallest valid config, one field at a time varied."""
    fields: dict[str, object] = {"workers": 1, "process_batch_size": 1}
    fields.update(overrides)
    return SelfPlayConfig(**fields)  # type: ignore[arg-type]


def test_the_defaults_are_todays_behaviour_to_the_line() -> None:
    """A config written before KOM-249 constructs, and means what it meant.

    ``off`` is the cap draw campaigns 1-6 trained on and ``0`` is "one draft
    threshold, the one ``min_placements`` already names". The two ramp knobs
    carry the pre-registered values so that turning the mode on is one word
    rather than three numbers an operator has to remember.
    """
    config = _selfplay()
    assert config.cap_adjudication == "off"
    assert config.cap_adjudication_p_max == 0.8
    assert config.cap_adjudication_d_full == 8
    assert config.min_placements_max == 0


def test_the_four_are_appended_after_the_search_modes() -> None:
    """Field ORDER is the transport contract, not a style choice.

    The config travels to a spawned worker as a positional tuple
    (``_config_getstate``), so a field inserted anywhere but the end would
    silently re-assign every value past it in a payload built by a parent
    running older code. This is that rule as an assertion.
    """
    names = [f.name for f in dataclasses.fields(SelfPlayConfig)]
    assert names[-8:] == [
        "proof_mode",
        "discovery_mode",
        "cap_adjudication",
        "cap_adjudication_p_max",
        "cap_adjudication_d_full",
        "min_placements_max",
        "seed_openings_path",
        "seed_openings_prob",
    ]


def test_the_material_mode_and_its_ramp_construct() -> None:
    """The configuration KOM-245 switches on, at its pre-registered values."""
    armed = _selfplay(cap_adjudication="material")
    assert armed.cap_adjudication == "material"
    assert (armed.cap_adjudication_p_max, armed.cap_adjudication_d_full) == (0.8, 8)

    tuned = _selfplay(
        cap_adjudication="material", cap_adjudication_p_max=1.0, cap_adjudication_d_full=1
    )
    assert tuned.cap_adjudication_p_max == 1.0
    assert tuned.cap_adjudication_d_full == 1


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param({"cap_adjudication": "off_by_default"}, "cap_adjudication", id="mode"),
        pytest.param({"cap_adjudication": ""}, "cap_adjudication", id="empty-mode"),
        pytest.param(
            {"cap_adjudication_p_max": 0.0}, "cap_adjudication_p_max", id="p-zero"
        ),
        pytest.param(
            {"cap_adjudication_p_max": 1.5}, "cap_adjudication_p_max", id="p-above-one"
        ),
        pytest.param(
            {"cap_adjudication_p_max": -0.1}, "cap_adjudication_p_max", id="p-negative"
        ),
        pytest.param(
            {"cap_adjudication_p_max": float("nan")},
            "cap_adjudication_p_max",
            id="p-nan",
        ),
        pytest.param(
            {"cap_adjudication_d_full": 0}, "cap_adjudication_d_full", id="d-zero"
        ),
        pytest.param(
            {"cap_adjudication_d_full": 26}, "cap_adjudication_d_full", id="d-above-hand"
        ),
        pytest.param(
            {"min_placements": 10, "min_placements_max": 9},
            "min_placements_max",
            id="max-below-low-edge",
        ),
        # A low edge of 1 rather than the default 0, so each case is refused
        # for the high edge it names and not for the off-switch rule -- a
        # drawn range's low edge must be at least 1.
        pytest.param(
            {"min_placements": 1, "min_placements_max": 26},
            "min_placements_max",
            id="max-above-hand",
        ),
        pytest.param(
            {"min_placements": 1, "min_placements_max": -1},
            "min_placements_max",
            id="max-negative",
        ),
    ],
)
def test_a_value_nobody_can_play_is_refused_naming_the_field(
    overrides: dict[str, object], field: str
) -> None:
    """Every refusal names the field and shows the value it was given.

    ``0.0`` and ``26`` are the interesting ones. A ``p_max`` of zero is not "the
    mode off", it is a ramp that labels the side which is AHEAD as the loser;
    ``d_full`` above a hand is a ramp no game can reach, since with both
    marshals alive the material difference at the cap is at most 24. NaN is in
    the list because ``not 0 < x <= 1`` is how it is caught -- a ``<``/``>``
    spelling would let a NaN through to be discovered as a shard of NaN targets.

    The *value* is asserted beside the name (review F4), because a message that
    names only the field sends an operator back to a config file to find which
    of several places set it. As a whole word: ``cap_adjudication`` must be
    reported by the mode's own refusal, and a substring match would let
    ``cap_adjudication_p_max``'s message answer for it.
    """
    value = overrides[field]
    with pytest.raises(ValueError) as refusal:
        _selfplay(**overrides)
    message = str(refusal.value)

    assert re.search(rf"\b{field}\b", message), (
        f"{message!r} does not name {field} as a field of its own"
    )
    assert re.search(re.escape(repr(value)), message), (
        f"{message!r} does not show the {value!r} it was given"
    )
    if field == "min_placements_max":
        # The cross-field rule: the low edge is half of why the value is wrong,
        # so a message that withholds it states half a condition.
        low_edge = overrides.get("min_placements", _selfplay().min_placements)
        assert f"min_placements={low_edge!r}" in message


def test_a_max_of_zero_and_a_max_equal_to_the_low_edge_both_construct() -> None:
    """The two spellings of "one threshold" (KOM-246).

    ``0`` is the default and the compact spelling; ``min_placements_max ==
    min_placements`` is the operator who prefers to write the range out. Both
    are accepted and both describe a degenerate draw, so the worker KOM-246
    lands may implement either as the other.
    """
    assert _selfplay(min_placements=10, min_placements_max=0).min_placements_max == 0
    written_out = _selfplay(min_placements=10, min_placements_max=10)
    assert written_out.min_placements_max == written_out.min_placements

    drawn = _selfplay(min_placements=10, min_placements_max=18)
    assert (drawn.min_placements, drawn.min_placements_max) == (10, 18)
    # The hand is the ceiling, as it is for the low edge itself. Written on a
    # low edge of 1 -- the smallest a DRAWN range may have -- so the claim is
    # about the ceiling alone.
    assert _selfplay(min_placements=1, min_placements_max=25).min_placements_max == 25


def test_a_drawn_range_may_not_start_at_the_mask_s_off_switch() -> None:
    """``min_placements=0`` is the mask OFF, so it is not a low edge (F7).

    A range starting at 0 would hand a share of the seats no draft mask at all
    -- a population silently mixed between masked and unmasked games, which is
    precisely what the lever exists to prevent. Refused where the run is
    configured, naming the low edge, rather than at the first game of a
    launched run.
    """
    with pytest.raises(ValueError, match="min_placements") as refusal:
        _selfplay(min_placements=0, min_placements_max=25)
    message = str(refusal.value)
    assert "min_placements=0" in message, (
        f"{message!r} does not name the low edge that makes the range wrong"
    )
    assert "min_placements_max=25" in message


def test_the_low_edge_may_exceed_the_high_one_only_when_the_high_one_is_off() -> None:
    """``0`` means "same as ``min_placements``", not "zero placements".

    Without the special case a default config with ``min_placements=10`` would
    be refused as an inverted range -- which is every campaign that has ever run
    the mask curriculum.
    """
    assert _selfplay(min_placements=25).min_placements_max == 0
