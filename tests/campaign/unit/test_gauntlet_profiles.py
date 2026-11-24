"""Which search a gauntlet played, recorded where it can be read back.

KOM-226 (lite). Three seams, and one question each:

* the **pinfile** -- a line written before the mode keys existed is read as the
  legacy search it can only have been played by, and is marked
  ``legacy_inferred`` so a census can tell an inference from an observation. A
  line that carries the keys is a claim, validated as one;
* the **shard request** -- pickled into a spawned process, so a parent on old
  code can hand a new child a 13-field payload. That payload predates the modes
  and restores as ``off``/``off``; any other length is refused rather than
  padded, because a silently-defaulted mode is a shard playing a search its
  request does not name;
* the **profile switch** -- ``legacy`` keeps both sides on the search every
  gauntlet in the campaign's history was played with, whatever self-play is
  doing. That is the default and campaign 6 phase b keeps it.

The two mutation guards at the end are the load-bearing part: they show that
dropping a side's modes in :func:`~komugi.campaign.gauntlet.run_shard`, or
writing a pin without its mode keys, is caught here rather than discovered in a
census months later.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from komugi.campaign import gauntlet as gauntlet_module
from komugi.campaign.factory import NET_ENV
from komugi.campaign.gauntlet import (
    GAUNTLET_PROFILE_LEGACY,
    GAUNTLET_PROFILE_SELFPLAY,
    PROFILE_SOURCE_LEGACY_INFERRED,
    PROFILE_SOURCE_RECORDED,
    AnchorPin,
    ShardRequest,
    checkpoint_player,
    gauntlet_modes,
    pin_anchor,
    pinfile_path,
    read_pins,
    run_gauntlet,
    run_shard,
)
from komugi.eval.records import GameRecord

M = 8
N = 32

PROOF = "interval_v1"
DISCOVERY = "root_marshal_v1"

SELFPLAY_PAIR = (PROOF, "root_win_v1")
"""Campaign 6 phase b's actual self-play pair, as `loop` would pass it."""

_LEGACY_LINE: dict[str, Any] = {
    "gen": 0,
    "pinned_at_iteration": 3,
    "path": "/runs/c6/checkpoints/checkpoint-000000900-0123456789abcdef.pt",
    "name_digest": "0123456789abcdef",
    "weight_sha256": "a" * 64,
    "bn_sha256": "b" * 64,
    "m": M,
    "n": N,
    "games": 64,
    "floor": 0.35,
    "draw_score": 0.0,
}


def write_pinfile(root: Path, lines: list[dict[str, Any]]) -> None:
    """Write `lines` as the root's pinfile, one compact JSON object each."""
    path = pinfile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n"
            for line in lines
        ),
        encoding="utf-8",
    )


def legacy_state() -> tuple[object, ...]:
    """Return a pre-KOM-226 :class:`ShardRequest` payload: 13 field values."""
    return (
        "/runs/c6/challenger.pt",
        "ema-000000900",
        "/runs/c6/anchors/gen_0/anchor.pt",
        "anchor-g0",
        0,
        8,
        64,
        12345,
        0.0,
        M,
        N,
        "cpu",
        "",
    )


def current_request(**overrides: Any) -> ShardRequest:
    """Build a request with today's 17 fields, legacy modes unless overridden."""
    fields: dict[str, Any] = {
        "challenger_path": "/runs/c6/challenger.pt",
        "challenger_id": "ema-000000900",
        "anchor_path": "/runs/c6/anchors/gen_0/anchor.pt",
        "anchor_id": "anchor-g0",
        "pairs_start": 0,
        "pairs_stop": 8,
        "games": 64,
        "seed": 12345,
        "draw_score": 0.0,
        "m": M,
        "n": N,
        "device": "cpu",
        "net_env": "",
    }
    fields.update(overrides)
    return ShardRequest(**fields)


def restore(state: object) -> ShardRequest:
    """Restore a request from a raw pickle payload, without constructing one."""
    request = object.__new__(ShardRequest)
    request.__setstate__(state)
    return request


# ---------------------------------------------------------------------------
# T1: the pinfile
# ---------------------------------------------------------------------------


def test_legacy_pin_line_reads_as_inferred_legacy_search(tmp_path: Path) -> None:
    """A line without the mode keys is off/off, and says the pair was INFERRED.

    Every pin written before KOM-226 came from a build that could only play the
    legacy search, so the value is right -- but a census that could not tell it
    from an observed ``off`` would be reading an assumption as a measurement.
    """
    write_pinfile(tmp_path, [_LEGACY_LINE])

    (pin,) = read_pins(tmp_path)

    assert (pin.proof_mode, pin.discovery_mode) == ("off", "off")
    assert pin.profile_source == PROFILE_SOURCE_LEGACY_INFERRED
    assert pin.search_identity.decision_id == "gumbel_sh_survivor_v1"


def test_pin_line_with_modes_is_recorded(tmp_path: Path) -> None:
    """A line carrying the keys is an observation of what the gauntlet played."""
    write_pinfile(
        tmp_path,
        [{**_LEGACY_LINE, "proof_mode": PROOF, "discovery_mode": DISCOVERY}],
    )

    (pin,) = read_pins(tmp_path)

    assert (pin.proof_mode, pin.discovery_mode) == (PROOF, DISCOVERY)
    assert pin.profile_source == PROFILE_SOURCE_RECORDED
    assert pin.search_identity.decision_id == "gumbel_sh_interval_v1"


def test_pin_line_with_recorded_off_modes_is_not_inferred(tmp_path: Path) -> None:
    """``off``/``off`` written explicitly is RECORDED, not legacy_inferred."""
    write_pinfile(
        tmp_path, [{**_LEGACY_LINE, "proof_mode": "off", "discovery_mode": "off"}]
    )

    (pin,) = read_pins(tmp_path)

    assert (pin.proof_mode, pin.discovery_mode) == ("off", "off")
    assert pin.profile_source == PROFILE_SOURCE_RECORDED


@pytest.mark.parametrize(
    "modes",
    [
        {"proof_mode": "interval_v2", "discovery_mode": "off"},
        {"proof_mode": "off", "discovery_mode": "root_marshal_v1"},
        {"proof_mode": "off", "discovery_mode": "marshal_v9"},
    ],
)
def test_pin_line_with_unrunnable_modes_raises(
    tmp_path: Path, modes: dict[str, str]
) -> None:
    """A recorded pair the search cannot run is refused, not skipped.

    Deliberately NOT folded into the partial-line tolerance: a truncated line
    names a pin that never committed, but this one committed and claims a search
    that does not exist. Skipping it would drop an anchor generation out of an
    append-only history without a word.
    """
    write_pinfile(tmp_path, [{**_LEGACY_LINE, **modes}])

    with pytest.raises(ValueError):
        read_pins(tmp_path)


def test_new_pin_round_trips_through_read_pins(tmp_path: Path) -> None:
    """``as_json`` -> pinfile -> ``read_pins`` preserves the modes and the source."""
    pin = AnchorPin(
        generation=2,
        pinned_at_iteration=571,
        path=_LEGACY_LINE["path"],
        name_digest=_LEGACY_LINE["name_digest"],
        weight_sha256=_LEGACY_LINE["weight_sha256"],
        bn_sha256=_LEGACY_LINE["bn_sha256"],
        m=M,
        n=N,
        games=64,
        floor=0.35,
        draw_score=0.0,
        proof_mode=PROOF,
        discovery_mode=DISCOVERY,
    )
    write_pinfile(tmp_path, [pin.as_json()])

    (restored,) = read_pins(tmp_path)

    assert restored == pin
    assert restored.profile_source == PROFILE_SOURCE_RECORDED


def test_pin_refuses_an_unrunnable_mode_pair() -> None:
    """Discovery without the proof layer is refused where the pin is built."""
    with pytest.raises(ValueError):
        AnchorPin(
            generation=0,
            pinned_at_iteration=0,
            path=_LEGACY_LINE["path"],
            name_digest=_LEGACY_LINE["name_digest"],
            weight_sha256=_LEGACY_LINE["weight_sha256"],
            bn_sha256=_LEGACY_LINE["bn_sha256"],
            m=M,
            n=N,
            games=64,
            floor=0.35,
            draw_score=0.0,
            discovery_mode=DISCOVERY,
        )


# ---------------------------------------------------------------------------
# T2: the pickled shard request
# ---------------------------------------------------------------------------


def test_request_pickle_round_trip_preserves_both_sides() -> None:
    """Protocol 4 -- what ``spawn`` uses -- carries each side's own pair."""
    request = current_request(
        challenger_proof_mode=PROOF,
        challenger_discovery_mode=DISCOVERY,
        anchor_proof_mode="off",
        anchor_discovery_mode="off",
    )

    restored = pickle.loads(pickle.dumps(request, protocol=4))

    assert restored == request
    assert (restored.challenger_proof_mode, restored.challenger_discovery_mode) == (
        PROOF,
        DISCOVERY,
    )
    assert (restored.anchor_proof_mode, restored.anchor_discovery_mode) == ("off", "off")


@pytest.mark.parametrize("as_list", [False, True])
def test_legacy_thirteen_field_payload_restores_as_legacy_search(
    as_list: bool,
) -> None:
    """A payload from a parent on pre-KOM-226 code is off/off on both sides.

    A list as well as a tuple: the default dataclass hook on the old class sent
    a LIST, and that is exactly the payload a worker respawned on this code can
    be handed.
    """
    state = legacy_state()
    restored = restore(list(state) if as_list else state)

    assert (restored.challenger_proof_mode, restored.challenger_discovery_mode) == (
        "off",
        "off",
    )
    assert (restored.anchor_proof_mode, restored.anchor_discovery_mode) == ("off", "off")
    assert restored.challenger_id == "ema-000000900"
    assert restored.net_env == ""


@pytest.mark.parametrize("length", [14, 16, 18])
def test_payload_of_an_uninterpretable_length_is_refused(length: int) -> None:
    """Not 13 and not 17: refused rather than padded or truncated.

    A 14- or 16-field payload would pad to a request whose modes nothing chose,
    and an 18-field one comes from a build this one cannot read.
    """
    base = (*legacy_state(), "off", "off", "off", "off")
    state = base[:length] if length <= len(base) else (*base, "off")

    with pytest.raises(ValueError):
        restore(state)


@pytest.mark.parametrize(
    ("index", "value"),
    [(13, "interval_v2"), (14, "root_marshal_v1"), (16, "marshal_v9")],
)
def test_payload_with_unrunnable_modes_is_refused(index: int, value: str) -> None:
    """A 17-field payload gets the validation a constructed request gets.

    Index 13 is an unknown proof mode, 14 is discovery with the proof layer off
    on the challenger, 16 an unknown discovery mode on the anchor -- both sides
    are checked, not just the first.
    """
    values = [*legacy_state(), "off", "off", "off", "off"]
    values[index] = value

    with pytest.raises(ValueError):
        restore(tuple(values))


def test_constructing_a_request_with_unrunnable_modes_is_refused() -> None:
    """Refused in the PARENT, where it reads as a bad request.

    After ``spawn`` the same mistake arrives as a child that died without
    returning -- :class:`~komugi.campaign.gauntlet.GauntletLost` -- and sends an
    operator to ``dmesg`` for a typo'd mode string.
    """
    with pytest.raises(ValueError):
        current_request(anchor_discovery_mode=DISCOVERY)


def test_getstate_is_every_field_in_order() -> None:
    """The payload is positional, so its ORDER is the compatibility contract."""
    request = current_request(
        challenger_proof_mode=PROOF, challenger_discovery_mode=DISCOVERY
    )

    state = request.__getstate__()

    assert len(state) == 17
    assert state[:13] == legacy_state()
    assert state[13:] == (PROOF, DISCOVERY, "off", "off")


# ---------------------------------------------------------------------------
# T3: checkpoint_player forwards the modes
# ---------------------------------------------------------------------------


def test_checkpoint_player_forwards_the_modes(tmp_path: Path) -> None:
    """The entrant carries the search it was asked for, and loads nothing yet.

    The loader is inside the evaluator FACTORY (it must be: the net is built in
    the shard, once per process), so this needs no checkpoint on disk -- the
    path is never opened here.
    """
    player = checkpoint_player(
        tmp_path / "checkpoint-000000900-0123456789abcdef.pt",
        "challenger",
        m=M,
        n=N,
        proof_mode=PROOF,
        discovery_mode=DISCOVERY,
    )

    assert (player.proof_mode, player.discovery_mode) == (PROOF, DISCOVERY)
    assert player.search_identity.decision_id == "gumbel_sh_interval_v1"


def test_checkpoint_player_defaults_to_the_legacy_search(tmp_path: Path) -> None:
    """Every existing call site keeps the entrant it already had."""
    player = checkpoint_player(tmp_path / "anchor.pt", "anchor-g0", m=M, n=N)

    assert (player.proof_mode, player.discovery_mode) == ("off", "off")


# ---------------------------------------------------------------------------
# T4: run_shard builds each side with its own modes
# ---------------------------------------------------------------------------


def shard_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, str]]:
    """Record ``(player_id, proof_mode, discovery_mode)`` per built entrant.

    Both the player factory and the match are substituted: this asks what the
    shard BUILDS, and playing even one game would need a checkpoint, a net and
    a GPU that this test has no business touching.
    """
    seen: list[tuple[str, str, str]] = []

    def spy(
        path: Any,
        player_id: str,
        *,
        device: str = "cpu",
        m: int,
        n: int,
        proof_mode: str = "off",
        discovery_mode: str = "off",
    ) -> Any:
        seen.append((player_id, proof_mode, discovery_mode))
        return player_id

    def no_games(
        *args: Any, **kwargs: Any
    ) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
        return ((), ())

    monkeypatch.setattr(gauntlet_module, "checkpoint_player", spy)
    monkeypatch.setattr(gauntlet_module, "play_pairs", no_games)
    return seen


def test_run_shard_gives_each_side_its_own_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The challenger's pair does not leak onto the anchor, or the reverse.

    The asymmetric case is the one that catches a dropped argument: with both
    sides equal -- which is all the ``legacy`` profile ever produces -- a shard
    that built the anchor from the challenger's modes would look correct.
    """
    monkeypatch.setenv(NET_ENV, "test-net")
    seen = shard_spy(monkeypatch)

    run_shard(
        current_request(
            net_env="test-net",
            challenger_proof_mode=PROOF,
            challenger_discovery_mode=DISCOVERY,
            anchor_proof_mode="off",
            anchor_discovery_mode="off",
        )
    )

    assert seen == [
        ("ema-000000900", PROOF, DISCOVERY),
        ("anchor-g0", "off", "off"),
    ]


def test_run_shard_carries_a_legacy_payload_through_as_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request restored from a 13-field payload plays the legacy search."""
    monkeypatch.setenv(NET_ENV, "test-net")
    seen = shard_spy(monkeypatch)

    run_shard(restore(legacy_state()))

    assert seen == [("ema-000000900", "off", "off"), ("anchor-g0", "off", "off")]


# ---------------------------------------------------------------------------
# T5: the profile switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selfplay",
    [("off", "off"), (PROOF, "off"), (PROOF, DISCOVERY)],
)
def test_legacy_profile_is_off_off_whatever_selfplay_runs(
    selfplay: tuple[str, str],
) -> None:
    """The gauntlet's history was played legacy, and stays comparable to it."""
    assert gauntlet_modes(GAUNTLET_PROFILE_LEGACY, selfplay) == ("off", "off")


@pytest.mark.parametrize(
    "selfplay", [("off", "off"), (PROOF, "off"), (PROOF, DISCOVERY)]
)
def test_selfplay_profile_is_the_selfplay_pair(selfplay: tuple[str, str]) -> None:
    """The alternative, for a future campaign: exactly what self-play plays."""
    assert gauntlet_modes(GAUNTLET_PROFILE_SELFPLAY, selfplay) == selfplay


def test_selfplay_profile_refuses_an_unrunnable_selfplay_pair() -> None:
    """A pair that cannot be played is not silently downgraded to legacy."""
    with pytest.raises(ValueError):
        gauntlet_modes(GAUNTLET_PROFILE_SELFPLAY, ("off", DISCOVERY))


@pytest.mark.parametrize("profile", ["", "Legacy", "self-play", "default"])
def test_unknown_profile_is_refused(profile: str) -> None:
    """A typo'd profile raises rather than falling back to legacy.

    The fallback would be the worst outcome available: a gauntlet that played
    one search while its operator believed it played the other.
    """
    with pytest.raises(ValueError):
        gauntlet_modes(profile, (PROOF, DISCOVERY))


def test_gauntlet_request_construction_stays_legacy() -> None:
    """The wiring today: both sides off/off, whatever self-play is doing.

    Campaign 6 phase b turned the modes on for self-play at iteration 571 and
    deliberately did NOT turn them on for the gauntlet.
    """
    pin = AnchorPin(
        generation=0,
        pinned_at_iteration=3,
        path=_LEGACY_LINE["path"],
        name_digest=_LEGACY_LINE["name_digest"],
        weight_sha256=_LEGACY_LINE["weight_sha256"],
        bn_sha256=_LEGACY_LINE["bn_sha256"],
        m=M,
        n=N,
        games=64,
        floor=0.35,
        draw_score=0.0,
    )
    modes = gauntlet_modes(GAUNTLET_PROFILE_LEGACY, (PROOF, DISCOVERY))

    requests = gauntlet_module._shard_requests(
        "/runs/c6/challenger.pt",
        "ema-000000900",
        "/runs/c6/anchors/gen_0/anchor.pt",
        "anchor-g0",
        pin,
        [range(0, 8)],
        seed=12345,
        device="cpu",
        challenger_modes=modes,
        anchor_modes=modes,
    )

    (request,) = requests
    assert (request.challenger_proof_mode, request.challenger_discovery_mode) == (
        "off",
        "off",
    )
    assert (request.anchor_proof_mode, request.anchor_discovery_mode) == ("off", "off")


# ---------------------------------------------------------------------------
# T6: the guards are load-bearing
# ---------------------------------------------------------------------------


def test_dropping_a_sides_modes_in_the_shard_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutant: build the anchor from the CHALLENGER's modes.

    A shard written that way rates a challenger against an opponent playing a
    search the pin does not name. This runs the mutant's construction and shows
    the assertion in ``test_run_shard_gives_each_side_its_own_modes`` fails on
    it -- so that test is what stops the mutation, not decoration around it.
    """
    seen = shard_spy(monkeypatch)
    request = current_request(
        challenger_proof_mode=PROOF,
        challenger_discovery_mode=DISCOVERY,
        anchor_proof_mode="off",
        anchor_discovery_mode="off",
    )

    # The mutation, inlined: both entrants built from the challenger's pair.
    for player_id in (request.challenger_id, request.anchor_id):
        gauntlet_module.checkpoint_player(
            request.challenger_path,
            player_id,
            device=request.device,
            m=request.m,
            n=request.n,
            proof_mode=request.challenger_proof_mode,
            discovery_mode=request.challenger_discovery_mode,
        )

    assert seen != [("ema-000000900", PROOF, DISCOVERY), ("anchor-g0", "off", "off")]
    assert seen == [
        ("ema-000000900", PROOF, DISCOVERY),
        ("anchor-g0", PROOF, DISCOVERY),
    ]


def test_writing_a_pin_without_its_mode_keys_is_caught(tmp_path: Path) -> None:
    """The mutant: an ``as_json`` that forgets the modes.

    Such a pin reads back as ``legacy_inferred`` off/off, so a gauntlet played
    with the proof layer on would be recorded as one played without it -- and
    the round-trip in
    ``test_new_pin_round_trips_through_read_pins`` is what refuses it.
    """
    pin = AnchorPin(
        generation=2,
        pinned_at_iteration=571,
        path=_LEGACY_LINE["path"],
        name_digest=_LEGACY_LINE["name_digest"],
        weight_sha256=_LEGACY_LINE["weight_sha256"],
        bn_sha256=_LEGACY_LINE["bn_sha256"],
        m=M,
        n=N,
        games=64,
        floor=0.35,
        draw_score=0.0,
        proof_mode=PROOF,
        discovery_mode=DISCOVERY,
    )
    stripped = {
        key: value
        for key, value in pin.as_json().items()
        if key not in {"proof_mode", "discovery_mode", "profile_source"}
    }
    write_pinfile(tmp_path, [stripped])

    (restored,) = read_pins(tmp_path)

    assert restored != pin
    assert (restored.proof_mode, restored.discovery_mode) == ("off", "off")
    assert restored.profile_source == PROFILE_SOURCE_LEGACY_INFERRED


# ---------------------------------------------------------------------------
# T7: the pinfile parser is strict about a HALF claim (review F3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        {"proof_mode": PROOF},
        {"discovery_mode": "off"},
        {"proof_mode": "off"},
    ],
    ids=["proof-only", "discovery-only", "off-proof-only"],
)
def test_pin_line_with_one_mode_key_is_refused(
    tmp_path: Path, line: dict[str, str]
) -> None:
    """Half a pair names no search, so it is refused rather than half-defaulted.

    Reading the missing half as ``off`` would manufacture a pair the line never
    claimed and stamp it ``recorded`` -- an invented observation, which is worse
    than the inference the legacy path makes honestly.
    """
    write_pinfile(tmp_path, [{**_LEGACY_LINE, **line}])

    with pytest.raises(ValueError, match="proof_mode|discovery_mode"):
        read_pins(tmp_path)


@pytest.mark.parametrize("source", ["bogus", PROFILE_SOURCE_LEGACY_INFERRED, ""])
def test_pin_line_asserting_a_wrong_profile_source_is_refused(
    tmp_path: Path, source: str
) -> None:
    """A line that carries its modes may not claim they were inferred.

    ``legacy_inferred`` is the pointed case: a hand-edited line asserting it
    while carrying the keys would read back as an inference, which is exactly
    the confusion between assumption and measurement the source field exists to
    prevent. The source is a fact about the line's keys, not a field it asserts.
    """
    write_pinfile(
        tmp_path,
        [
            {
                **_LEGACY_LINE,
                "proof_mode": PROOF,
                "discovery_mode": DISCOVERY,
                "profile_source": source,
            }
        ],
    )

    with pytest.raises(ValueError, match="profile_source"):
        read_pins(tmp_path)


def test_pin_line_with_modes_and_a_recorded_source_is_accepted(tmp_path: Path) -> None:
    """The shape `as_json` writes: both keys plus ``recorded``, read unchanged."""
    write_pinfile(
        tmp_path,
        [
            {
                **_LEGACY_LINE,
                "proof_mode": PROOF,
                "discovery_mode": DISCOVERY,
                "profile_source": PROFILE_SOURCE_RECORDED,
            }
        ],
    )

    (pin,) = read_pins(tmp_path)

    assert (pin.proof_mode, pin.discovery_mode) == (PROOF, DISCOVERY)
    assert pin.profile_source == PROFILE_SOURCE_RECORDED


def test_legacy_line_is_still_inferred_under_the_strict_parser(tmp_path: Path) -> None:
    """Neither key: the pre-KOM-226 line keeps reading exactly as before.

    The strictness above is about *claims*; a line that makes none is the whole
    live pinfile, and tightening the parser must not reject a single one of its
    generations.
    """
    write_pinfile(tmp_path, [_LEGACY_LINE, {**_LEGACY_LINE, "gen": 1}])

    first, second = read_pins(tmp_path)

    assert (first.proof_mode, first.discovery_mode) == ("off", "off")
    assert first.profile_source == PROFILE_SOURCE_LEGACY_INFERRED
    assert second.profile_source == PROFILE_SOURCE_LEGACY_INFERRED


# ---------------------------------------------------------------------------
# T8: _shard_requests maps each side to ITS OWN fields (review F4)
# ---------------------------------------------------------------------------


def anchor_pin(**overrides: Any) -> AnchorPin:
    """Build a pin with the live campaign's shape, legacy modes by default."""
    fields: dict[str, Any] = {
        "generation": 0,
        "pinned_at_iteration": 3,
        "path": _LEGACY_LINE["path"],
        "name_digest": _LEGACY_LINE["name_digest"],
        "weight_sha256": _LEGACY_LINE["weight_sha256"],
        "bn_sha256": _LEGACY_LINE["bn_sha256"],
        "m": M,
        "n": N,
        "games": 64,
        "floor": 0.35,
        "draw_score": 0.0,
    }
    fields.update(overrides)
    return AnchorPin(**fields)


def test_shard_requests_keeps_the_two_sides_apart() -> None:
    """Asymmetric pairs: challenger's modes on the challenger fields, anchor's on the anchor's.

    With the two pairs equal -- all the ``legacy`` profile ever produces -- a
    builder that wrote one side's pair into both fields would look correct, so
    this is the only construction that can catch a swapped or copied argument.
    """
    requests = gauntlet_module._shard_requests(
        "/runs/c6/challenger.pt",
        "ema-000000900",
        "/runs/c6/anchors/gen_0/anchor.pt",
        "anchor-g0",
        anchor_pin(),
        [range(0, 8), range(8, 16)],
        seed=12345,
        device="cpu",
        challenger_modes=SELFPLAY_PAIR,
        anchor_modes=("off", "off"),
    )

    assert len(requests) == 2
    for request in requests:
        assert (
            request.challenger_proof_mode,
            request.challenger_discovery_mode,
        ) == SELFPLAY_PAIR
        assert (request.anchor_proof_mode, request.anchor_discovery_mode) == (
            "off",
            "off",
        )


# ---------------------------------------------------------------------------
# T9: pin_anchor stamps the modes it was given (review F5)
# ---------------------------------------------------------------------------


def stub_pin_anchor_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `pin_anchor` runnable with no checkpoint, no torch and no copy.

    Everything stubbed is I/O the mode forwarding does not touch: the loader,
    the two digests and the durable copy. The pinfile append is left REAL --
    it is the thing under test, and a stubbed one would assert nothing about
    what a later `read_pins` sees.
    """
    checkpoint = SimpleNamespace(model_state={"trunk.weight": 0})
    monkeypatch.setattr(gauntlet_module, "load_checkpoint_file", lambda path: checkpoint)
    monkeypatch.setattr(
        gauntlet_module,
        "canonical_digest",
        lambda state, keys: SimpleNamespace(sha256="a" * 64),
    )
    monkeypatch.setattr(
        gauntlet_module, "bn_digest", lambda state, keys: SimpleNamespace(sha256="b" * 64)
    )
    monkeypatch.setattr(gauntlet_module, "_copy_durably", lambda source, target: None)


def test_pin_anchor_records_the_modes_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair reaches the pinfile, marked as an observation.

    A `pin_anchor` that accepted the kwargs and dropped them would write an
    off/off line while the gauntlet played the proof layer -- readable months
    later only as a search the campaign never ran.
    """
    stub_pin_anchor_io(monkeypatch)
    source = tmp_path / "checkpoint-000000900-0123456789abcdef.pt"

    returned = pin_anchor(
        tmp_path,
        2,
        571,
        source,
        ["trunk.weight"],
        m=M,
        n=N,
        games=64,
        floor=0.35,
        draw_score=0.0,
        proof_mode=SELFPLAY_PAIR[0],
        discovery_mode=SELFPLAY_PAIR[1],
    )

    assert (returned.proof_mode, returned.discovery_mode) == SELFPLAY_PAIR
    (stored,) = read_pins(tmp_path)
    assert (stored.proof_mode, stored.discovery_mode) == SELFPLAY_PAIR
    assert stored.profile_source == PROFILE_SOURCE_RECORDED
    assert stored == returned


def test_pin_anchor_without_modes_records_the_legacy_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call `loop` makes today: off/off, and RECORDED rather than inferred."""
    stub_pin_anchor_io(monkeypatch)

    pin_anchor(
        tmp_path,
        0,
        3,
        tmp_path / "checkpoint-000000900-0123456789abcdef.pt",
        ["trunk.weight"],
        m=M,
        n=N,
        games=64,
        floor=0.35,
        draw_score=0.0,
    )

    (stored,) = read_pins(tmp_path)
    assert (stored.proof_mode, stored.discovery_mode) == ("off", "off")
    assert stored.profile_source == PROFILE_SOURCE_RECORDED


# ---------------------------------------------------------------------------
# T10: run_gauntlet -- the default profile, and the pin/profile guard (F1, F2, F7c)
# ---------------------------------------------------------------------------


class ShardsBuilt(Exception):
    """Raised by the spy once the requests are built: nothing may be played here."""


def gauntlet_spies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Stub the anchor verify and intercept the request build.

    `run_gauntlet` fans out into spawned processes immediately afterwards, so
    the spy raises rather than returning: these tests ask what the parent
    DECIDED, and letting them reach the pool would play real games.
    """
    seen: list[tuple[tuple[str, str], tuple[str, str]]] = []

    def verified(root: Any, pin: AnchorPin, param_keys: Any) -> Path:
        return tmp_path / "anchors" / "gen_0" / "anchor.pt"

    def spy(
        *args: Any,
        challenger_modes: tuple[str, str],
        anchor_modes: tuple[str, str],
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        seen.append((challenger_modes, anchor_modes))
        raise ShardsBuilt

    monkeypatch.setattr(gauntlet_module, "verify_anchor", verified)
    monkeypatch.setattr(gauntlet_module, "_shard_requests", spy)
    return seen


def refuse_to_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stub verify to succeed and fail the test if any request is built."""

    def verified(root: Any, pin: AnchorPin, param_keys: Any) -> Path:
        return tmp_path / "anchors" / "gen_0" / "anchor.pt"

    def never(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        pytest.fail("run_gauntlet built shard requests past a configuration error")

    monkeypatch.setattr(gauntlet_module, "verify_anchor", verified)
    monkeypatch.setattr(gauntlet_module, "_shard_requests", never)


def launch(pin: AnchorPin, tmp_path: Path, **kwargs: Any) -> Any:
    """Call `run_gauntlet` the way `loop` does, with only `kwargs` varying."""
    return run_gauntlet(
        tmp_path,
        tmp_path / "checkpoint-000000900-0123456789abcdef.pt",
        "ema-000000900",
        pin,
        ["trunk.weight"],
        seed=12345,
        min_decisive=8,
        **kwargs,
    )


def test_run_gauntlet_defaults_to_the_legacy_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NO profile kwarg and self-play running the proof layer: still off/off.

    This is the promise campaign 6 phase b is running under -- self-play turned
    the modes on at iteration 571 and the gauntlet did not -- and it lives in a
    keyword DEFAULT, which nothing else in this file pins.
    """
    seen = gauntlet_spies(monkeypatch, tmp_path)

    with pytest.raises(ShardsBuilt):
        launch(anchor_pin(), tmp_path, selfplay_modes=SELFPLAY_PAIR)

    assert seen == [(("off", "off"), ("off", "off"))]


def test_run_gauntlet_refuses_a_profile_the_pin_does_not_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping the profile without re-pinning is a configuration error.

    Deliberately a :class:`ValueError` and not a
    :class:`~komugi.campaign.gauntlet.GauntletAborted`: the campaign records the
    latter as a gauntlet that came to nothing and carries on, while this is a
    launch that must be corrected before any games are worth playing.
    """
    refuse_to_build(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="re-pin the anchor"):
        launch(
            anchor_pin(),
            tmp_path,
            gauntlet_profile=GAUNTLET_PROFILE_SELFPLAY,
            selfplay_modes=SELFPLAY_PAIR,
        )


def test_run_gauntlet_accepts_a_pin_recorded_with_the_modes_it_plays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is a match check, not a ban on the selfplay profile."""
    seen = gauntlet_spies(monkeypatch, tmp_path)
    pin = anchor_pin(proof_mode=SELFPLAY_PAIR[0], discovery_mode=SELFPLAY_PAIR[1])

    with pytest.raises(ShardsBuilt):
        launch(
            pin,
            tmp_path,
            gauntlet_profile=GAUNTLET_PROFILE_SELFPLAY,
            selfplay_modes=SELFPLAY_PAIR,
        )

    assert seen == [(SELFPLAY_PAIR, SELFPLAY_PAIR)]


def test_run_gauntlet_refuses_a_legacy_profile_against_a_recorded_selfplay_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mismatch is caught in BOTH directions, not only profile-ahead-of-pin."""
    refuse_to_build(monkeypatch, tmp_path)
    pin = anchor_pin(proof_mode=SELFPLAY_PAIR[0], discovery_mode=SELFPLAY_PAIR[1])

    with pytest.raises(ValueError, match="re-pin the anchor"):
        launch(pin, tmp_path, gauntlet_profile=GAUNTLET_PROFILE_LEGACY)


def test_selfplay_profile_without_selfplay_modes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` under ``selfplay`` does not stand in for the legacy pair.

    The old default -- ``("off", "off")`` -- made a forgotten pair play the
    legacy search under a profile whose whole point is that it does not.
    """
    refuse_to_build(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="needs selfplay_modes"):
        launch(anchor_pin(), tmp_path, gauntlet_profile=GAUNTLET_PROFILE_SELFPLAY)


def test_legacy_profile_needs_no_selfplay_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`loop`'s call today passes neither kwarg, and that stays legal."""
    seen = gauntlet_spies(monkeypatch, tmp_path)

    with pytest.raises(ShardsBuilt):
        launch(anchor_pin(), tmp_path)

    assert seen == [(("off", "off"), ("off", "off"))]
