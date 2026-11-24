"""T1 (both directions) and T2 -- exposure and injectivity over the v2 space.

T1 asserts set equality between our generator and the frozen gungi-js dump,
encoded into the extended space:

    {encode_v2(m) for m in generate_all_moves(P)} == {encode_v2(j) for j in J(P)}

with the reference side encoded by ``helpers.v2_id`` -- this suite's own
transcription of the design's layout table -- and our side by the encoder
KOM-028 ships.  A missing partial betrayal (JS not in engine) and an invented
one (engine not in JS) both fail it, which the v1-collapsed check in
``test_reference_dumps_sanity`` structurally cannot do.

T2 asserts that the encoding is injective on the record list and that
``decode_v2`` inverts it on the canonical normal form.  Injectivity is the
property v1 provably fails at every betray-capable node -- and
``test_reference_dumps_sanity.test_v1_provably_collapses_wherever_the_reference_betrays``
pins that failure, so the two modules bracket the change.

Gated on KOM-028.
"""

from __future__ import annotations

from typing import Any

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine.action_encoding import (  # noqa: E402
    encode_cython_move,
    decode_cython_action,
    legal_action_ids,
    legal_cython_action_ids,
)
from komugi.engine.api import Gungi  # noqa: E402
from komugi.engine.cyboard import Board  # noqa: E402

from helpers import (  # noqa: E402
    MoveKey,
    board_of_fen,
    corpus_fen,
    corpus_names,
    dumped_moves,
    engine_records,
    is_draft_fen,
    kind_counts,
    normalize_dump_move,
    normalize_engine_move,
    v2_block,
    v2_id,
)

CORPUS: tuple[str, ...] = corpus_names()

#: Fixtures whose reference dump contains at least one conversion -- the ones
#: where v1 and v2 actually differ.  Kept as a literal list so a regenerated
#: corpus that quietly loses its betrayals fails rather than passes vacuously.
CONVERTING_FIXTURES: tuple[str, ...] = (
    "mixed_pair_hand_d_only",
    "mixed_pair_hand_both",
    "same_type_pair_hand_two",
    "friendly_topped_enemy_below",
    "arata_conversion",
    "intermediate_mode_bit2_dead",
    "tactician_on_tower",
    "black_seat_mirror",
    "multi_type_hand_rich",
)


def _reference_keys(name: str) -> list[MoveKey]:
    """Reference normal forms our engine is expected to offer for *name*.

    Drafting FENs drop the harness's unconditional board-move sweep; see
    ``test_reference_dumps_sanity._comparable_dump_keys`` for why.
    """
    keys = [normalize_dump_move(move) for move in dumped_moves(name)]
    if is_draft_fen(corpus_fen(name)):
        return [key for key in keys if key.kind == "arata"]
    return keys


def _board(name: str) -> Board:
    return board_of_fen(corpus_fen(name))


# ---------------------------------------------------------------------------
# T1 -- differential exposure, both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t1_engine_and_reference_expose_the_same_v2_ids(name: str) -> None:
    """The set equality T1 names, with no collapse on either side."""
    expected = {v2_id(key) for key in _reference_keys(name)}
    actual = {encode_cython_move(record) for record in engine_records(_board(name))}

    assert actual == expected, (
        f"{name}: engine-only {sorted(actual - expected)}, "
        f"reference-only {sorted(expected - actual)}"
    )


@pytest.mark.parametrize("name", CONVERTING_FIXTURES)
def test_t1_every_reference_conversion_has_its_own_id(name: str) -> None:
    """Each distinct converted-tier subset reaches a distinct action id.

    The pointed form of T1: it is not enough for the id SETS to match if the
    conversions all landed on one id and some other move made up the count.
    """
    conversions = [
        key
        for key in _reference_keys(name)
        if key.kind in ("betray", "arata") and key.sel != 0
    ]
    assert conversions, f"{name} is listed as converting but dumps no conversion"

    ids = {v2_id(key) for key in conversions}
    assert len(ids) == len(conversions), "two conversions share an id"

    engine_ids = {encode_cython_move(r) for r in engine_records(_board(name))}
    assert ids <= engine_ids

    # ...and the plain decline is a separate, also-reachable id.
    for key in conversions:
        declined = v2_id(key._replace(sel=0))
        assert declined in engine_ids, (
            "the decline-into-tsuke / plain drop must stay reachable beside "
            "every conversion (design: 'selector 0 always means convert nothing')"
        )
        assert declined != v2_id(key)


@pytest.mark.parametrize("name", CORPUS)
def test_t1_the_ids_land_in_the_blocks_their_kind_names(name: str) -> None:
    """route/capture -> A0, tsuke/betray -> A1, arata -> B or C by 終."""
    expected_block = {
        "route": "A0",
        "capture": "A0",
        "tsuke": "A1",
        "betray": "A1",
    }
    for key in _reference_keys(name):
        block = v2_block(v2_id(key))
        if key.kind == "arata":
            assert block == ("C" if key.draft_finished else "B")
        else:
            assert block == expected_block[key.kind]


@pytest.mark.parametrize("name", CORPUS)
def test_t1_the_string_encoder_agrees_with_the_cython_encoder(name: str) -> None:
    """``encode_move`` and ``encode_cython_move`` must gain the selector together.

    ``legal_action_ids`` walks the pure-Python oracle's verbose Move dicts and
    ``legal_cython_action_ids`` the record path; the design changes both
    encoders and they are the two halves of the same contract.
    """
    fen = corpus_fen(name)
    assert legal_action_ids(Gungi(fen)) == legal_cython_action_ids(board_of_fen(fen))


@pytest.mark.parametrize("name", CORPUS)
def test_t1_the_san_of_each_id_matches_the_reference(name: str) -> None:
    """Same id, same move: the ids are aligned to the reference move by move.

    Set equality alone would survive a permutation that gave two moves each
    other's ids.  SAN is reference truth in the dump and is built independently
    by ``cymovegen._build_san``, so matching it id-for-id closes that gap.
    """
    reference: dict[int, str] = {}
    for move in dumped_moves(name):
        san = move["san"]
        assert san is not None
        key = normalize_dump_move(move)
        if is_draft_fen(corpus_fen(name)) and key.kind != "arata":
            continue
        reference[v2_id(key)] = san
    assert len(reference) == len(_reference_keys(name)), (
        "two reference moves share a v2 id -- the layout cannot separate them"
    )

    ours = {
        encode_cython_move(record): str(record["san"])
        for record in engine_records(_board(name))
    }
    assert ours == reference


# ---------------------------------------------------------------------------
# T2 -- injectivity and round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t2_the_encoding_is_injective_on_the_move_list(name: str) -> None:
    """``len({encode_v2(m)}) == len(moves)`` -- no residual collapse anywhere."""
    records = engine_records(_board(name))
    ids = [encode_cython_move(record) for record in records]

    if len(set(ids)) != len(ids):
        seen: dict[int, str] = {}
        clashes: list[str] = []
        for record, action_id in zip(records, ids):
            san = str(record["san"])
            if action_id in seen:
                clashes.append(f"{action_id}: {seen[action_id]!r} vs {san!r}")
            seen[action_id] = san
        pytest.fail(f"{name}: colliding ids -- " + "; ".join(clashes))


@pytest.mark.parametrize("name", CORPUS)
def test_t2_decode_inverts_encode_on_the_normal_form(name: str) -> None:
    """``decode_v2(encode_v2(m), P)`` returns a record equal to ``m``.

    Equality is on the canonical normal form -- kind, origin, destination,
    destination tier, piece index, converted-tier mask and draft-finish -- plus
    the SAN, which carries the pieces converted and so distinguishes two
    same-mask records if the mask were ever ambiguous.
    """
    board = _board(name)
    for record in engine_records(board):
        action_id = encode_cython_move(record)
        decoded = decode_cython_action(action_id, board)
        assert normalize_engine_move(decoded) == normalize_engine_move(record), (
            f"{name}: id {action_id} decoded to {decoded['san']!r}, "
            f"expected {record['san']!r}"
        )
        assert decoded["san"] == record["san"]


@pytest.mark.parametrize("name", CONVERTING_FIXTURES)
def test_t2_each_selector_decodes_to_its_own_conversion(name: str) -> None:
    """The tsuke and every partial betrayal are separately reachable.

    This is the ceiling removal stated as a decode property: under v1 the
    ``sel = 0`` id at a betray-capable square pair returned the maximal
    betrayal and the plain tsuke had no id at all
    (``action_encoding.py:176-182``).
    """
    board = _board(name)
    for key in _reference_keys(name):
        if key.kind not in ("betray", "arata", "tsuke"):
            continue
        decoded = decode_cython_action(v2_id(key), board)
        assert normalize_engine_move(decoded) == key


def test_t2_the_three_combo_fixture_reaches_four_distinct_stack_actions() -> None:
    """``mixed_pair_hand_both``: tsuke plus three betrayals, four ids, four moves.

    The single most concrete statement of what the extension buys: v1 had one
    id here and could express only the maximal conversion.
    """
    name = "mixed_pair_hand_both"
    board = _board(name)
    stack_keys = [
        key for key in _reference_keys(name) if key.kind in ("tsuke", "betray")
    ]
    assert len(stack_keys) == 4
    assert sorted(key.sel for key in stack_keys) == [0, 1, 2, 3]

    ids = {v2_id(key) for key in stack_keys}
    assert len(ids) == 4
    assert len({decode_cython_action(i, board)["san"] for i in ids}) == 4
    assert kind_counts(name)["betray"] == 3
