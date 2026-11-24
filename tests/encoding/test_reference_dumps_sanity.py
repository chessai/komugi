"""The unconditional heart of T1 -- reference truth, and our engine against it.

Nothing in this module is gated on KOM-028.  It asserts two things that are
true TODAY and must stay true after the extension lands:

* the frozen dumps in ``goldens/betrayal_dumps.json`` still say what
  ``goldens/GENERATION.md`` recorded them as saying at generation time, and
  the two reference tables in that file (per-position dumps and the perft
  tables) agree with each other at depth 1;
* every corpus FEN loads in our engine, and our CURRENT move generator
  produces exactly the reference's move set once both are V1-COLLAPSED --
  every betrayal at a square pair and the tsuke beside it folded onto one A1
  id, every arata conversion folded into the plain drop (design "Current
  encoding").  That is the strongest differential statement the v1 layout can
  carry: it pins move GENERATION against gungi-js right now, and leaves only
  the collapse itself for the gated T1 module to remove.

The collapse is applied to the reference side too, so a failure here is a
real generation disagreement and never the known encoding ceiling.
"""

from __future__ import annotations

import pytest

from helpers import (
    KIND_BY_MT,
    MoveKey,
    board_of_fen,
    corpus_fen,
    corpus_names,
    dumped_moves,
    engine_keys,
    is_draft_fen,
    kind_counts,
    load_dumps,
    normalize_dump_move,
    perft_fixture_names,
    perft_table,
    v1_id,
)

#: Every fixture in the frozen corpus.  COVERAGE GAP, recorded rather than
#: repaired: the corpus's FENs span modes 3 (advanced), 2 (intermediate) and 1
#: (beginner) only -- mode 0 (intro) appears in no fixture, so every
#: mode-dependent claim below and in T6.4 is unchecked there.  Intro shares its
#: ``maxTier == 2`` bound with beginner, which is why the gap is narrow rather
#: than alarming; closing it properly needs an intro fixture, and the goldens
#: are read-only, so it closes only with a dated regeneration entry per
#: ``goldens/GENERATION.md``.
CORPUS: tuple[str, ...] = corpus_names()


def _comparable_dump_keys(name: str) -> list[MoveKey]:
    """Reference moves for *name* that our engine is expected to offer.

    The generation harness calls ``generateMovesForSquare`` over all 81
    squares UNCONDITIONALLY, rather than going through ``Gungi.moves()``.
    During the draft the reference's own ``moves()`` returns arata only
    (``api.Gungi.moves``: ``elif self.in_draft(): ...``), and so does
    ``cymovegen.generate_all_moves``.  So on a drafting FEN the dumped board
    moves are a harness artefact with no counterpart in either engine's legal
    move list, and only the arata subset is comparable.
    """
    keys = [normalize_dump_move(move) for move in dumped_moves(name)]
    if is_draft_fen(corpus_fen(name)):
        return [key for key in keys if key.kind == "arata"]
    return keys


# ---------------------------------------------------------------------------
# The dump file itself
# ---------------------------------------------------------------------------


def test_the_dumps_load_and_name_their_generator() -> None:
    """The goldens parse and still identify the verbatim gungi-js reference."""
    data = load_dumps()
    assert "gungi-js verbatim reference" in data["generator"]
    assert len(data["dumps"]) == 14
    assert set(data["perfts"]) <= set(data["dumps"])


@pytest.mark.parametrize("name", CORPUS)
def test_every_dumped_move_has_the_schema_the_readers_assume(name: str) -> None:
    """Each entry carries a SAN, a ``rank-file-tier`` destination and sorted tiers."""
    for move in dumped_moves(name):
        assert move["san"], f"{name}: move without SAN: {move!r}"
        assert move["kind"] in set(KIND_BY_MT.values()), f"{name}: {move['kind']!r}"
        assert len(move["to"].split("-")) == 3, f"{name}: to={move['to']!r}"
        assert move["capturedTiers"] == sorted(move["capturedTiers"])
        assert len(move["capturedTiers"]) == len(move["capturedTypes"])


def test_the_harness_lost_the_from_square_and_san_recovers_it() -> None:
    """FINDING, pinned so it cannot change unnoticed under the SAN readers.

    The reference dump generator dumps ``m.piece?.square`` / ``m.piece?.tier`` /
    ``m.piece?.type`` / ``m.piece?.color``, but the reference ``Move`` object
    has no ``piece`` member, so all four fields are ``null`` / ``'?'`` for
    every move in the corpus.  Everything the encoding needs is still in the
    SAN, and ``helpers.parse_san_origin`` / ``helpers.san_piece_kanji``
    recover it -- ``test_our_engine_matches_the_reference_when_v1_collapsed``
    below is what proves the recovery is right, since a mis-parsed origin
    would land the move on a different A1 id.

    The goldens are read-only (KOM-128 may not regenerate them), so this is
    recorded rather than repaired.  If the dumps are ever regenerated with the
    origin populated, this test is the one that should fail first.
    """
    for name in CORPUS:
        for move in dumped_moves(name):
            assert move["from"] is None
            assert move["fromTier"] is None
            assert move["pieceType"] == "?"
            assert move["color"] == "?"


# ---------------------------------------------------------------------------
# GENERATION.md's recorded kind-counts
# ---------------------------------------------------------------------------

#: "Verified kind-counts at generation time" from ``goldens/GENERATION.md``.
RECORDED_BETRAY_COUNTS: dict[str, int] = {
    "mixed_pair_hand_d_only": 1,
    "mixed_pair_hand_both": 3,
    "same_type_pair_hand_one": 0,
    "same_type_pair_hand_two": 3,
    "friendly_topped_enemy_below": 1,
    "black_seat_mirror": 3,
    "multi_type_hand_rich": 3,
}


@pytest.mark.parametrize(
    ("name", "expected"), sorted(RECORDED_BETRAY_COUNTS.items())
)
def test_generation_md_betray_counts_hold(name: str, expected: int) -> None:
    """The betray counts GENERATION.md recorded are still in the file."""
    assert kind_counts(name).get("betray", 0) == expected


def test_mixed_pair_hand_both_has_the_three_distinct_tier_subsets() -> None:
    """The 3-combo fixture: ``{1}``, ``{2}`` and ``{1,2}``, each exactly once.

    This is the case the design's "Subsets are over piece *instances*" bullet
    describes and the case T7(d) needs -- three distinct resulting towers, and
    therefore three ids the v1 layout cannot tell apart.
    """
    tiers = sorted(
        tuple(move["capturedTiers"])
        for move in dumped_moves("mixed_pair_hand_both")
        if move["kind"] == "betray"
    )
    assert tiers == [(1,), (1, 2), (2,)]


def test_same_type_pair_hand_one_generates_no_betrayal_at_all() -> None:
    """T6.1's fixture: the over-strict hand test kills even the singletons.

    Two enemy 兵 in the tower, one 兵 in hand: ``count = 2`` fails the
    ``hand >= count`` test at ``move_gen.ts:416-419``, so the type does not
    qualify and NO combination containing either 兵 is generated -- not the
    pair, and not either singleton the mover could plainly afford.
    """
    counts = kind_counts("same_type_pair_hand_one")
    assert counts.get("betray", 0) == 0
    assert counts.get("tsuke", 0) == 1, "the decline-into-tsuke survives"


def test_same_type_pair_hand_two_lifts_the_same_tower_to_three_combos() -> None:
    """One more 兵 in hand and the identical tower yields all three subsets."""
    tiers = sorted(
        tuple(move["capturedTiers"])
        for move in dumped_moves("same_type_pair_hand_two")
        if move["kind"] == "betray"
    )
    assert tiers == [(1,), (1, 2), (2,)]


def test_arata_conversion_has_exactly_one_converting_drop_at_tier_one() -> None:
    """The tactician-arata-onto-own-tower fixture (T1's arata requirement).

    A legal arata target is empty or own-coloured on top, so the only enemy it
    can hold sits at tier 1 -- which is why ``bsel`` is one bit (T6.5).
    """
    converting = [
        move
        for move in dumped_moves("arata_conversion")
        if move["kind"] == "arata" and move["capturedTiers"]
    ]
    assert len(converting) == 1
    assert converting[0]["capturedTiers"] == [1]
    assert converting[0]["to"] == "7-5-3"


def test_full_height_fixture_offers_no_stacking_move_at_all() -> None:
    """``h == 3`` in advanced mode closes the whole tsuke/betray branch."""
    counts = kind_counts("full_height_no_stack")
    assert counts.get("tsuke", 0) == 0
    assert counts.get("betray", 0) == 0


def test_intermediate_mode_never_converts_above_tier_one() -> None:
    """T6.4's fixture: ``maxTier == 2`` leaves tier 1 as the only enemy slot."""
    for move in dumped_moves("intermediate_mode_bit2_dead"):
        if move["kind"] == "betray":
            assert move["capturedTiers"] == [1]


def test_every_conversion_in_the_corpus_stays_within_two_tiers() -> None:
    """The design's hard bound, checked against reference truth directly."""
    for name in CORPUS:
        for move in dumped_moves(name):
            if move["kind"] in ("betray", "arata"):
                assert all(1 <= tier <= 2 for tier in move["capturedTiers"]), (
                    f"{name}: {move['san']!r} converts outside tiers 1-2"
                )


# ---------------------------------------------------------------------------
# The two reference tables against each other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", perft_fixture_names())
def test_depth_one_perft_equals_the_dumped_move_counts(name: str) -> None:
    """The perft table and the per-position dump are the same reference run."""
    assert perft_table(name)["1"] == kind_counts(name)


def test_the_betray_bucket_is_nonzero_in_at_least_three_fixtures() -> None:
    """T9's coverage requirement, asserted on the reference side.

    The corpus has to carry enough betrayals for the gated perft test to mean
    anything; if a future regeneration thins it out, this fails first.
    """
    with_betrays = [
        name for name in CORPUS if kind_counts(name).get("betray", 0) > 0
    ]
    assert len(with_betrays) >= 3, with_betrays


# ---------------------------------------------------------------------------
# Our engine against the reference, v1-collapsed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_every_corpus_fen_loads_in_our_engine(name: str) -> None:
    """The frozen FENs parse, with the turn and mode the FEN states."""
    fen = corpus_fen(name)
    board = board_of_fen(fen)
    fields = fen.split(" ")
    assert board.turn == (0 if fields[2] == "w" else 1)
    assert board.mode == int(fields[3])
    assert bool(board.draft_w or board.draft_b) == is_draft_fen(fen)


@pytest.mark.parametrize("name", CORPUS)
def test_our_engine_matches_the_reference_when_v1_collapsed(name: str) -> None:
    """T1, as far as the v1 layout can express it: identical collapsed id sets.

    Both sides go through the same collapse, so this isolates move GENERATION
    from the encoding ceiling.  A missing partial betrayal cannot show up here
    (it collapses onto the tsuke's id) -- that is exactly what the gated
    ``test_differential_exposure`` module adds once KOM-028 lands.  What DOES
    show up is a missing or invented square pair, drop square, piece type,
    draft-finish variant or move kind, in either direction.
    """
    fen = corpus_fen(name)
    expected = {v1_id(key) for key in _comparable_dump_keys(name)}
    actual = {v1_id(key) for key in engine_keys(board_of_fen(fen))}

    assert actual == expected, (
        f"{name}: engine-only {sorted(actual - expected)}, "
        f"reference-only {sorted(expected - actual)}"
    )


@pytest.mark.parametrize("name", CORPUS)
def test_the_collapsed_move_totals_agree(name: str) -> None:
    """The counted form of the same statement, per the T1 obligation."""
    expected = {v1_id(key) for key in _comparable_dump_keys(name)}
    actual = {v1_id(key) for key in engine_keys(board_of_fen(corpus_fen(name)))}
    assert len(actual) == len(expected)


@pytest.mark.parametrize("name", CORPUS)
def test_v1_provably_collapses_wherever_the_reference_betrays(name: str) -> None:
    """The ceiling itself, stated as a fact about today's engine.

    Wherever the reference offers a betrayal, our engine's move list is
    strictly longer than its set of v1 ids -- distinct rules-engine moves with
    no distinct action id.  This is the property T2 inverts after KOM-028.
    """
    keys = engine_keys(board_of_fen(corpus_fen(name)))
    has_betray = kind_counts(name).get("betray", 0) > 0
    collapsed = len(keys) > len({v1_id(key) for key in keys})
    if has_betray:
        assert collapsed, "a betray fixture whose v1 ids are already injective"


# ---------------------------------------------------------------------------
# T10 -- not this ticket's
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "T10 (policy-head width == ACTION_SIZE, mask length/dtype, MCTS prior "
        "renormalisation over a sel=0-only node) belongs to the network "
        "ticket: no network module exists in this tree for KOM-128 to bind to."
    )
)
def test_t10_head_and_mask_plumbing() -> None:
    """Placeholder so T10 is visibly unowned rather than silently missing."""
