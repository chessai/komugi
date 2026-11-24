"""The bridge's cheap half: framing, the wire form, P1, and the seating wrapper.

Everything here is a pure function or a dataclass -- no process is spawned and
no game is played, which is what keeps it cheap enough to state the boundary
conditions exhaustively. The expensive half, where a real subprocess really
misbehaves and the §D6 outcome is asserted end to end, is
``tests/eval/test_v1bridge_behavior.py``.

Three properties are worth naming because they are what this module exists to
pin, and each would otherwise be discovered only by a rated match:

* **The M4 boundary is a schema check, not a shrug.** Every field of §D2.3's
  wire form is validated for presence, type *and* range, so a version-skewed
  adapter is caught at the wire rather than by the engine several frames later,
  where §D6 would have booked it as a rules question (M3) instead of a
  version-skew question (M4).
* **P1 is stated over the two quantities that are on the wire**, and its
  forced-move clause runs in one direction only. The reverse implication is
  false *guaranteed*, at the final ply of every turn-capped game, and a test
  that pinned the ``iff`` would fire on legitimate play.
* **The evaluator seam refuses.** §D1's central finding is a negative one, and a
  negative finding needs a test or the next implementer re-discovers it by
  shipping the hybrid.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.game import Game
from komugi.eval import v1bridge
from komugi.eval.arena import (
    DEFAULT_GUMBEL_SCALE,
    DEFAULT_M,
    DEFAULT_N,
    ArenaConfig,
    ArenaPlayer,
    GameOutcome,
    pair_id_for,
)
from komugi.eval.records import DRAW, Player
from komugi.eval.v1bridge import (
    ABORTED,
    COMPLETED,
    EVAL_CEILING,
    PROTOCOL_VERSION,
    V1_FAMILY,
    VOID_RATE_GATE,
    BridgedGame,
    BridgeEntrant,
    BridgedMatch,
    BridgeVoid,
    CapabilityBlock,
    DeviceBlock,
    HelloResponse,
    M1,
    M3,
    M8,
    MoveRecord,
    Mover,
    ProtocolViolation,
    VoidBucket,
    VoidScope,
    VoidTally,
    WrongAltitude,
    assert_pinned_budget,
    bridged_book_id,
    budget_invariant_breach,
    decode_message,
    encode_message,
    legal_records,
    record_of_action,
    roster_map,
)
from komugi.selfplay.adapter import GameState

DRAW_SCORE: float = 0.0
"""§D3.4 pre-registers ``draw_score = 0.0`` for every cross-family match."""


def start_state() -> GameState:
    """Return a fresh game at the arena's start position."""
    return GameState(Game(), DRAW_SCORE)


def a_legal_record(state: GameState) -> tuple[MoveRecord, int]:
    """Return one legal ``(record, action id)`` pair from `state`."""
    index = legal_records(state)
    key = sorted(index, key=lambda item: index[item])[0]
    action = index[key]
    return record_of_action(state, action), action


# ---------------------------------------------------------------------------
# framing (§D1: line-delimited JSON, one request per line)
# ---------------------------------------------------------------------------


def test_a_message_round_trips_through_the_framing() -> None:
    """Encode then decode is the identity on a protocol message."""
    payload: dict[str, Any] = {
        "type": "pick",
        "ply": 7,
        "seed": 12345678901234,
        "nested": {"b": 1, "a": [1, 2, 3]},
    }
    line = encode_message(payload)
    assert line.endswith("\n")
    assert line.count("\n") == 1, "one message per line is the framing"
    assert decode_message(line) == payload


def test_the_framing_survives_a_payload_that_contains_a_newline() -> None:
    """A newline inside a string cannot break the one-message-per-line frame.

    JSON escapes it. Without this the far end's stderr text -- which §D6 M1 puts
    verbatim into a void record -- could split one message into two and
    desynchronise the wire at exactly the moment the harness is trying to report
    a failure.
    """
    line = encode_message({"type": "ok", "note": "line one\nline two"})
    assert line.count("\n") == 1
    assert decode_message(line)["note"] == "line one\nline two"


def test_a_message_must_declare_a_type() -> None:
    """Both directions refuse a message with no ``type``."""
    with pytest.raises(ProtocolViolation, match="type"):
        encode_message({"ply": 1})
    with pytest.raises(ProtocolViolation, match="type"):
        decode_message('{"ply": 1}')
    with pytest.raises(ProtocolViolation, match="type"):
        decode_message('{"type": 3}')


@pytest.mark.parametrize(
    "line",
    ["", "   \n", "not json at all", "[1, 2, 3]", '"a string"', "null", "3"],
)
def test_garbage_on_the_wire_is_a_protocol_violation(line: str) -> None:
    """Anything that is not one JSON object per line is M4, not a mystery."""
    with pytest.raises(ProtocolViolation):
        decode_message(line)


# ---------------------------------------------------------------------------
# §D2.3's wire form
# ---------------------------------------------------------------------------


def test_every_legal_engine_record_round_trips_through_the_wire_form() -> None:
    """Engine record -> wire -> record is the identity, over a whole position.

    The wire form is the record, so this is the claim the whole bridge rests on
    (§D2.3): if a field were dropped or narrowed in transit, the far engine
    would apply a different move from the one the near engine played, and the
    two boards would diverge with nothing detecting it until a much later ply
    produced an illegal move against a position nobody could reconstruct.
    """
    state = start_state()
    records = generate_all_moves(state.game._board)
    assert len(records) > 1, "the start position has moves"
    for engine_record in records:
        record = MoveRecord.from_engine(engine_record)
        assert MoveRecord.from_wire(record.to_wire()) == record
        assert MoveRecord.from_wire(record.to_wire()).key() == record.key()


def test_the_record_index_is_keyed_injectively_and_stays_inside_the_mask() -> None:
    """The M3 lookup is one id per record, and every id it yields is legal.

    Both halves matter. Injectivity is what makes "the record matches an entry
    of ``generate_all_moves``" answer *which* entry; containment in
    ``valid_moves()`` is what makes an accepted record safe to play on the
    arena's own state, which is the referee's mask and the only thing §D6 lets
    adjudicate legality.
    """
    state = start_state()
    index = legal_records(state)
    assert len(index) == len(generate_all_moves(state.game._board)), (
        "two distinct legal moves share a ten-field record key: the wire form "
        "cannot name them apart"
    )
    mask = state.legal_mask()
    assert index, "the start position has legal moves"
    for action in index.values():
        assert bool(mask[action]), f"record index yielded illegal action {action}"


def test_a_decoded_action_is_the_record_the_index_named() -> None:
    """``record_of_action`` and ``legal_records`` agree about the same move."""
    state = start_state()
    index = legal_records(state)
    for key, action in sorted(index.items(), key=lambda item: item[1])[:8]:
        assert record_of_action(state, action).key() == key


def test_san_is_a_cross_check_and_not_part_of_the_move_s_identity() -> None:
    """Two records differing only in ``san`` name the same move (A3).

    ``san`` is carried "as a redundant cross-check", so it must not participate
    in the identity: an adapter whose notation writer differs would otherwise
    have every one of its moves rejected as M3.
    """
    state = start_state()
    record, _ = a_legal_record(state)
    renamed = replace(record, san="totally-different")
    assert renamed.key() == record.key()
    assert renamed != record
    assert legal_records(state)[renamed.key()] == legal_records(state)[record.key()]


@pytest.mark.parametrize("field", ["from_r", "to_f", "move_type", "captured"])
def test_a_record_missing_a_structural_field_is_refused(field: str) -> None:
    """§D6 M4: a missing field is version skew, caught at the wire."""
    state = start_state()
    record, _ = a_legal_record(state)
    payload = record.to_wire()
    del payload[field]
    with pytest.raises(ProtocolViolation, match=field):
        MoveRecord.from_wire(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("to_r", 9),
        ("to_f", -1),
        ("from_r", -2),
        ("to_tier", 4),
        ("move_type", 5),
        ("piece_type_index", 14),
        ("color", 2),
    ],
)
def test_a_record_with_an_out_of_range_field_is_refused(field: str, value: int) -> None:
    """An in-JSON-range integer outside the *engine's* range is still M4.

    The failure this forbids is the quiet one: an out-of-range square is a
    perfectly good JSON number, and left unchecked it reaches the engine's
    encoder as an index.
    """
    state = start_state()
    record, _ = a_legal_record(state)
    payload = record.to_wire()
    payload[field] = value
    with pytest.raises(ProtocolViolation, match=field):
        MoveRecord.from_wire(payload)


@pytest.mark.parametrize(
    "value",
    [
        "not a list",
        [[1]],
        [[1, 2, 3]],
        [[1, 3]],
        [["a", 1]],
        [[1, True]],
    ],
)
def test_a_malformed_captured_list_is_refused(value: object) -> None:
    """``captured`` carries the conversion selector, so its shape is checked.

    ``sel_from_captured`` (``action_encoding.py:133-151``) reads ``item[1]`` as
    a tier and asserts ``1 <= tier <= 2``. A wrongly-shaped list would therefore
    either crash the encoder or, worse, silently select a different member of
    the betray family than the one the far engine played.
    """
    state = start_state()
    record, _ = a_legal_record(state)
    payload = record.to_wire()
    payload["captured"] = value
    with pytest.raises(ProtocolViolation, match="captured"):
        MoveRecord.from_wire(payload)


def test_a_bool_is_not_accepted_where_an_int_is_required() -> None:
    """``True`` is an ``int`` in Python, and must not pass as square 1."""
    state = start_state()
    record, _ = a_legal_record(state)
    payload = record.to_wire()
    payload["to_r"] = True
    with pytest.raises(ProtocolViolation, match="to_r"):
        MoveRecord.from_wire(payload)


def test_draft_finished_and_san_are_type_checked() -> None:
    """The two non-integer fields get the same treatment as the integers."""
    state = start_state()
    record, _ = a_legal_record(state)
    payload = record.to_wire()
    payload["draft_finished"] = 1
    with pytest.raises(ProtocolViolation, match="draft_finished"):
        MoveRecord.from_wire(payload)
    payload = record.to_wire()
    payload["san"] = 5
    with pytest.raises(ProtocolViolation, match="san"):
        MoveRecord.from_wire(payload)


def test_a_record_that_is_not_an_object_is_refused() -> None:
    """``pick`` with no record at all is M4, not an attribute error."""
    with pytest.raises(ProtocolViolation):
        MoveRecord.from_wire(None)
    with pytest.raises(ProtocolViolation):
        MoveRecord.from_wire([1, 2, 3])


# ---------------------------------------------------------------------------
# §D3.6 P1, as §D6 M10 enforces it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evals", "sims"),
    [
        (1, 0),
        (257, 256),
        (1, 256),
        (2, 1),
        (128, 256),
        (257, 300),
    ],
)
def test_the_legitimate_per_move_counts_are_accepted(evals: int, sims: int) -> None:
    """P1 must not fire on play the design pre-registers as expected.

    Three of these rows are ply classes §D3.4's sweep table names, and one of
    them is the reason rev 1's invariant was wrong: ``(1, 256)`` is the
    **turn-cap final ply**, where every legal move leads to a terminal position,
    so a non-forced search reports a full simulation count and exactly one
    evaluation. An invariant stated as ``evals == 1 iff sims_used == 0`` would
    fire there -- at the last ply of *every* capped game, after the game was
    paid for.
    """
    assert budget_invariant_breach(evals, sims) is None


@pytest.mark.parametrize(
    ("evals", "sims", "why"),
    [
        (0, 0, "below 1"),
        (0, 256, "below 1"),
        (1, -1, "negative"),
        (258, 300, "ceiling"),
        (5, 3, r"1 \+ sims_used=3"),
        (2, 0, r"1 \+ sims_used=0"),
    ],
)
def test_each_limb_of_p1_fires_on_its_own_breach(
    evals: int, sims: int, why: str
) -> None:
    """Every disjunct §D6 M10 can return is reachable and names what it saw.

    ``(2, 0)`` is the forced-move *shape*, and the limb that answers it is
    ``evals <= 1 + sims_used``: with ``sims_used == 0`` that bound is 1, so any
    ``evals >= 2`` is already over it and returns before
    ``budget_invariant_breach``'s trailing ``sims_used == 0 => evals == 1``
    clause is reached at all. The convention is still enforced -- ``(2, 0)`` is
    refused, which is what A7b asks (and
    ``test_the_forced_move_convention_rejects_zero_evaluations`` pins) -- but it
    is refused as RISK C4's first limb, so this row expects that limb's message
    and names the ``sims_used=0`` the limb saw.
    """
    breach = budget_invariant_breach(evals, sims)
    assert breach is not None, f"P1 accepted evals={evals}, sims={sims}"
    assert str(evals) in breach
    assert re.search(why, breach), f"{breach!r} does not explain {why!r}"


def test_the_forced_move_convention_rejects_zero_evaluations() -> None:
    """Acceptance A7b's v1-side half, as a pure function.

    "The adapter must short-circuit identically" is readable as *zero*
    evaluations on a forced move, and an implementer who reads it that way
    would void **every pair containing a v1 forced ply** into a bucket that
    counts against the 2% admissibility gate. §D3.4 pins the convention: exactly
    one root evaluation through the counted evaluator, mirroring
    ``gumbel.py:858``.
    """
    assert budget_invariant_breach(0, 0) is not None
    assert budget_invariant_breach(1, 0) is None
    assert budget_invariant_breach(2, 0) is not None


def test_the_ceiling_is_the_pre_registered_literal() -> None:
    """§D3.3's 257, and A15's assertion that the production budget still is it."""
    assert EVAL_CEILING == 257
    assert (DEFAULT_M, DEFAULT_N, DEFAULT_GUMBEL_SCALE) == (32, 256, 1.0)
    assert budget_invariant_breach(EVAL_CEILING, 256) is None
    assert budget_invariant_breach(EVAL_CEILING + 1, 300) is not None
    assert_pinned_budget()


# ---------------------------------------------------------------------------
# the handshake blocks (§D1)
# ---------------------------------------------------------------------------


def conforming_hello() -> dict[str, Any]:
    """Return a well-formed ``hello`` response payload."""
    return {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_digest": "sha256:abc",
        "action_size": 15390,
        "gungi_mode": "intro",
        "capabilities": {
            "supports_no_move": True,
            "supports_budget_breach": True,
            "memo_separable": False,
            "memo_citation": "MCTS.py:120",
        },
        "device": {
            "torch_version": "2.4.0",
            "numpy_version": "2.0.0",
            "device": "cpu",
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
        },
    }


def test_the_handshake_parses_both_declared_blocks() -> None:
    """The capability and device blocks reach the caller intact.

    They are what the anchor's configuration block (P5) and A21's device-class
    check are built from, so a silently dropped field here is an anchor whose
    identity is missing exactly the parts §D3.6 says a reader needs.
    """
    hello = HelloResponse.from_wire(conforming_hello())
    assert hello.protocol_version == PROTOCOL_VERSION
    assert hello.action_size == 15390
    assert hello.gungi_mode == "intro"
    assert hello.capabilities == CapabilityBlock(
        supports_no_move=True,
        supports_budget_breach=True,
        memo_separable=False,
        memo_citation="MCTS.py:120",
    )
    assert hello.device == DeviceBlock(
        torch_version="2.4.0",
        numpy_version="2.0.0",
        device="cpu",
        cudnn_benchmark=False,
        deterministic_algorithms=True,
    )


@pytest.mark.parametrize("field", ["supports_no_move", "supports_budget_breach", "memo_separable"])
def test_an_undeclared_capability_is_not_read_as_false(field: str) -> None:
    """A missing capability flag raises rather than defaulting.

    Defaulting it to ``False`` would be survivable; defaulting it to ``True``
    would re-create the re-review F1 collision silently. Raising is the only
    reading that cannot be wrong, and §D1 makes the refusal a startup failure.
    """
    payload = conforming_hello()
    del payload["capabilities"][field]
    with pytest.raises(ProtocolViolation, match=field):
        HelloResponse.from_wire(payload)


def test_a_hello_missing_its_integers_is_refused() -> None:
    """Version and action size are required, and are not strings."""
    payload = conforming_hello()
    payload["protocol_version"] = "1"
    with pytest.raises(ProtocolViolation, match="protocol_version"):
        HelloResponse.from_wire(payload)


# ---------------------------------------------------------------------------
# the seating wrapper (§D1's altitude finding, §D4's pairing)
# ---------------------------------------------------------------------------


class _NullMover:
    """A mover that is never asked anything -- the factory is what is tested."""

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Never called."""
        raise AssertionError("this mover plays no games")

    def observe(self, action: int, record: MoveRecord) -> None:
        """Never called."""

    def reset(self, *, pair: int, seat: int) -> None:
        """Never called."""

    def close(self) -> None:
        """Never called."""


def a_bridge_entrant(player_id: str = "v1-0015") -> BridgeEntrant:
    """Return a v1 entrant whose mover is never used."""
    return BridgeEntrant(
        player=Player(id=player_id, family=V1_FAMILY),
        mover_factory=_NullMover,
    )


def test_the_null_mover_satisfies_the_mover_protocol() -> None:
    """The test double is checked against §D1's seam, not against nothing."""
    mover: Mover = _NullMover()
    assert mover is not None


def test_a_bridged_entrant_seats_as_an_arena_player() -> None:
    """Identity, family and budget survive into the ``ArenaPlayer`` (§D4).

    §D4's pairing is structural and family-independent -- ``pair_id`` is built
    from the two ``player.id`` s and the pair index, and the seats swap by
    index, not by player type -- so a v1 entrant must be a *real*
    ``ArenaPlayer`` in everything the pairing and the roster read.
    """
    entrant = a_bridge_entrant()
    seated = entrant.arena_player()
    assert isinstance(seated, ArenaPlayer)
    assert seated.player == Player(id="v1-0015", family="v1")
    assert (seated.m, seated.n, seated.gumbel_scale) == (
        DEFAULT_M,
        DEFAULT_N,
        DEFAULT_GUMBEL_SCALE,
    )


def _instant_draw(
    white: Mover,
    black: Mover,
    config: ArenaConfig,
    seeds: np.random.SeedSequence,
    *,
    pair: int = 0,
    seat: int = 0,
) -> BridgedGame:
    """Return a completed one-ply drawn game without playing anything.

    The stand-in for :func:`~komugi.eval.v1bridge.play_bridged_game` in the
    ``pair_namespace`` test below. The ``pair_id`` is built by
    ``play_bridged_match`` itself, *before* the game is dispatched, so replacing
    the game loop removes a whole cross-engine match from the bill without
    touching the line under test -- which is what keeps this module's "no
    process is spawned and no game is played" true.
    """
    del white, black, config, seeds, pair, seat
    return BridgedGame(outcome=GameOutcome(result=DRAW, moves=(0,)))


def test_the_bridged_pair_id_honours_the_pair_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kom-011d §D2.3: one namespace rule, and the bridge is its second site.

    ``play_bridged_match`` builds ``pair_id`` at its own line
    (``v1bridge.py:2171``), so the namespace could perfectly well land on the
    arena's site alone and leave every cross-family match un-namespaced -- and
    nothing downstream would notice until an epoch's records were concatenated
    and ``check_paired_books`` reported GAME_COUNT violations on the v1 links
    only. Both halves of the arena's node 8 are therefore asserted here too:
    the default is **byte-identical** to today's ids (red against an
    unconditional prefix, which would re-key every stored bridged record), and
    a namespace prefixes exactly as :func:`komugi.eval.arena.pair_id_for` says
    it does -- compared against that function rather than against a second
    hand-written format, because "one rule, two sites" is the property.
    """
    monkeypatch.setattr(v1bridge, "play_bridged_game", _instant_draw)
    komugi = BridgeEntrant(
        player=Player(id="ckpt-15000", family="komugi"), mover_factory=_NullMover
    )

    plain = ArenaConfig(games=4, seed=3, draw_score=DRAW_SCORE)
    assert plain.pair_namespace == ""
    match = v1bridge.play_bridged_match(a_bridge_entrant(), komugi, plain)
    assert match.outcome is not None
    assert [record.pair_id for record in match.outcome.records] == [
        "v1-0015:ckpt-15000:00000",
        "v1-0015:ckpt-15000:00000",
        "v1-0015:ckpt-15000:00001",
        "v1-0015:ckpt-15000:00001",
    ]

    namespaced = ArenaConfig(
        games=2, seed=3, draw_score=DRAW_SCORE, pair_namespace="E007"
    )
    epoch = v1bridge.play_bridged_match(a_bridge_entrant(), komugi, namespaced)
    assert epoch.outcome is not None
    assert [record.pair_id for record in epoch.outcome.records] == [
        "E007/v1-0015:ckpt-15000:00000",
        "E007/v1-0015:ckpt-15000:00000",
    ]
    assert epoch.outcome.records[0].pair_id == pair_id_for(
        namespaced, "v1-0015", "ckpt-15000", 0
    ), "the two construction sites must agree by calling one rule, not by luck"


def test_the_evaluator_seam_refuses_and_says_why() -> None:
    """§D1's central finding, as a test: v1 must not mount at the network seam.

    "So mounting v1 at the ``BatchEvaluator`` seam means: **v1's weights, driven
    by Komugi's Gumbel search.** That is not the v1 engine." The refusal is the
    product here -- without it the mistake is a silent one that produces an
    anchor answering a question KOM-011 did not ask.
    """
    seated = a_bridge_entrant().arena_player()
    with pytest.raises(WrongAltitude) as caught:
        seated.evaluator_factory()
    message = str(caught.value)
    assert "v1-0015" in message
    assert "play_bridged_match" in message, "the refusal must name the way that works"


def test_a_v1_entrant_is_not_built_through_the_ladder() -> None:
    """Acceptance A6, asserted by inspecting the built entrant, not by mocking.

    ``ladder_players`` stamps ``LADDER_FAMILY`` and hardcodes the checkpoint
    evaluator (``arena.py:735-736``), so a v1 entrant built through it would be
    a Komugi-family entrant carrying v1's id -- and ``Player.family`` never
    reaches a persisted ``GameRecord``, so nothing downstream could catch it.
    """
    from komugi.eval import arena

    seated = a_bridge_entrant().arena_player()
    assert seated.player.family == V1_FAMILY
    assert seated.player.family != arena.LADDER_FAMILY
    # The factory the entrant carries is not the ladder's. ``_checkpoint_evaluator``
    # is a factory *of* factories -- ``ladder_players`` calls it as
    # ``_checkpoint_evaluator(path, device)`` (``arena.py:736``) and seats what it
    # returns -- so an ``is`` check of the seated factory against it compares a
    # built factory with its builder and could never be anything but true.
    # Calling it is the assertion with teeth: the ladder's factory builds a
    # ``BatchEvaluator``, and this one refuses to.
    with pytest.raises(WrongAltitude):
        seated.evaluator_factory()


def test_the_roster_map_rejects_one_id_in_two_families() -> None:
    """§D4 amendment 3's guard, actually called.

    ``GameRecord`` has no ``family`` field, so this map is the only thing that
    keeps a v1 anchor from being read as a Komugi rung in a report.
    """
    komugi = BridgeEntrant(
        player=Player(id="ckpt-15000", family="komugi"), mover_factory=_NullMover
    )
    assert roster_map([a_bridge_entrant(), komugi]) == {
        "v1-0015": "v1",
        "ckpt-15000": "komugi",
    }
    clash = BridgeEntrant(
        player=Player(id="v1-0015", family="komugi"), mover_factory=_NullMover
    )
    with pytest.raises(ValueError, match="two families"):
        roster_map([a_bridge_entrant(), clash])


def test_the_book_id_names_the_start_mode() -> None:
    """§D4 amendment 2: ``"empty"`` alone is a lie two processes can share."""
    assert bridged_book_id("intro") == "empty-intro"
    assert bridged_book_id("advanced") == "empty-advanced"
    with pytest.raises(ValueError):
        bridged_book_id("  ")


# ---------------------------------------------------------------------------
# §D6's void accounting
# ---------------------------------------------------------------------------


def a_void(bucket: VoidBucket, mode: str = M3) -> BridgeVoid:
    """Return a void in `bucket`, for the tally arithmetic."""
    return BridgeVoid(
        mode=mode, bucket=bucket, scope=VoidScope.PAIR, detail="synthetic"
    )


def test_the_tally_counts_by_bucket_and_by_mode() -> None:
    """Voids are counted where §D6 puts them, and nowhere near the results."""
    tally = VoidTally(
        voids=(
            a_void(VoidBucket.CRASH, M1),
            a_void(VoidBucket.CRASH, M1),
            a_void(VoidBucket.ILLEGAL_MOVE, M3),
        ),
        pairs_attempted=100,
        pairs_scored=97,
    )
    assert tally.by_bucket() == {"crash": 2, "illegal_move": 1}
    assert tally.by_mode() == {"M1": 2, "M3": 1}
    assert tally.crashes == 2


def test_the_admissibility_gate_is_two_percent_of_pairs_attempted() -> None:
    """§D6's gate, with its own denominator: pairs **attempted**, voids included."""
    assert VOID_RATE_GATE == pytest.approx(0.02)
    admissible = VoidTally(
        voids=tuple(a_void(VoidBucket.RULES_DIVERGENCE) for _ in range(2)),
        pairs_attempted=100,
        pairs_scored=98,
    )
    assert admissible.void_rate == pytest.approx(0.02)
    assert admissible.admissible

    inadmissible = VoidTally(
        voids=tuple(a_void(VoidBucket.RULES_DIVERGENCE) for _ in range(3)),
        pairs_attempted=100,
        pairs_scored=97,
    )
    assert inadmissible.void_rate == pytest.approx(0.03)
    assert not inadmissible.admissible
    assert VoidTally().void_rate == 0.0


def test_a_run_that_did_not_complete_is_inadmissible_whatever_its_rate() -> None:
    """§D6 precedence 2 and 3: an abort keeps nothing and is rated not at all.

    "Voids are non-random, so the surviving pairs of an aborted match are
    exactly the biased sample the 2% gate exists to refuse." A match aborted at
    pair 3 of 500 has a *tiny* void rate, and reading admissibility off the rate
    alone would wave it through.
    """
    tally = VoidTally(
        voids=tuple(a_void(VoidBucket.CRASH, M1) for _ in range(3)),
        pairs_attempted=500,
        pairs_scored=497,
    )
    assert tally.admissible, "by rate alone this looks fine"
    aborted = BridgedMatch(
        status=ABORTED, reason="crash", outcome=None, tally=tally, pairs_completed=497
    )
    assert not aborted.admissible
    assert aborted.outcome is None, "an aborted match emits no MatchSet"

    completed = BridgedMatch(
        status=COMPLETED, reason="", outcome=None, tally=tally, pairs_completed=497
    )
    assert completed.admissible


def test_a_match_void_carries_the_invariant_that_voided_it() -> None:
    """M8/M9 report ``voided(P2|P3, pairs_completed)``, not a bare bucket."""
    void = BridgeVoid(
        mode=M8,
        bucket=VoidBucket.BUDGET_BREACH,
        scope=VoidScope.MATCH,
        invariant="P2",
        detail="the counting evaluator refused evaluation 258",
        evals_used=258,
    )
    assert void.scope is VoidScope.MATCH
    assert void.invariant == "P2"
    assert void.evals_used == 258


def test_a_void_is_never_a_result() -> None:
    """The type system's half of §D6's opening rule.

    A :class:`BridgeVoid` has no ``result`` field and no way to grow one, so
    there is no representable path from "the process crashed" to a 0 or a 1 in
    a ``GameRecord``. This test exists so that adding one is a deliberate act
    with a red test attached.
    """
    void = a_void(VoidBucket.CRASH, M1)
    assert not hasattr(void, "result")
    assert not hasattr(void, "score")
    assert isinstance(void.mode, str) and void.mode == M1
