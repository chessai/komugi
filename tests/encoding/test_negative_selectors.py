"""T6 -- the six negative selector cases, each rejected three ways.

Every case below must be rejected by all three surfaces the design names:
the mask bit is clear, ``decode_v2`` raises ``ValueError``, and
``Game.play_action`` refuses.  A selector that only one of the three rejects
is a live desync between the policy mask, the decoder and the environment --
precisely the class of bug ``cymovegen.pyx:826-830`` embodied in v1.

The ids are recomputed for the frozen fixtures' geometry rather than copied
from the design's worked example, which uses ``to = 4-5``; the corpus puts the
same tower on ``4-4`` (see
``test_mask_consistency.test_t5_the_worked_example_geometry_is_the_fixtures_geometry``),
shifting each id by exactly one.

Gated on KOM-028.
"""

from __future__ import annotations

from typing import Any

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine.action_encoding import decode_cython_action  # noqa: E402
from komugi.engine.cyboard import Board  # noqa: E402
from komugi.engine.game import Game  # noqa: E402

from helpers import (  # noqa: E402
    TACTICIAN_KANJI,
    V2_A1_START,
    V2_B_START,
    a1_parts,
    arata_parts,
    board_of_fen,
    corpus_fen,
    corpus_names,
    engine_records,
    legal_mask,
    masked_ids,
    piece_index,
    rank_file,
    sq_index,
    top_piece,
    tower,
    v2_block,
)

CORPUS: tuple[str, ...] = corpus_names()

TACTICIAN: int = piece_index(TACTICIAN_KANJI)

#: The square pair every board fixture in the corpus betrays across.
FROM_SQ: int = sq_index(5, 5)
TO_SQ: int = sq_index(4, 4)


def a1_id(from_sq: int, sel: int, to_sq: int) -> int:
    """The block-A1 id of ``(from_sq, sel, to_sq)``."""
    return V2_A1_START + (from_sq * 4 + sel) * 81 + to_sq


def assert_rejected(name: str, action_id: int, why: str) -> None:
    """The id is refused by the mask, the decoder and the environment."""
    fen = corpus_fen(name)
    board = board_of_fen(fen)

    assert legal_mask(board)[action_id] == 0, f"{name}: mask sets {action_id} -- {why}"

    with pytest.raises(ValueError):
        decode_cython_action(action_id, board)

    game = Game(board_of_fen(fen))
    with pytest.raises(ValueError):
        game.play_action(action_id)


def assert_accepted(name: str, action_id: int) -> None:
    """The id is offered by the mask, decodes, and can actually be played."""
    fen = corpus_fen(name)
    board = board_of_fen(fen)

    assert legal_mask(board)[action_id] == 1, f"{name}: mask clears {action_id}"
    decode_cython_action(action_id, board)

    game = Game(board_of_fen(fen))
    game.play_action(action_id)


# ---------------------------------------------------------------------------
# T6.1 -- unaffordable duplicate type
# ---------------------------------------------------------------------------


def test_t6_1_an_unaffordable_duplicate_type_kills_even_the_singletons() -> None:
    """Two enemy 兵, one 兵 in hand: ``sel in {1,2,3}`` all illegal.

    The over-strict hand test of ``move_gen.ts:416-419`` demands
    ``hand[T] >= count(T in tower)`` before ANY instance of ``T`` becomes
    convertible, so the singleton the mover could plainly pay for is refused
    too.  Design open question 6 pins this as reference behaviour, not a bug to
    round off here.
    """
    name = "same_type_pair_hand_one"
    assert_accepted(name, a1_id(FROM_SQ, 0, TO_SQ))
    for sel in (1, 2, 3):
        assert_rejected(
            name, a1_id(FROM_SQ, sel, TO_SQ), f"sel={sel:02b} with Q=0b00"
        )


def test_t6_1_one_more_copy_in_hand_opens_all_three() -> None:
    """The control: the identical tower with ``兵 x2`` in hand offers everything.

    Without this, T6.1 would also pass if the fixture's tower were simply
    unreachable.
    """
    name = "same_type_pair_hand_two"
    for sel in (0, 1, 2, 3):
        assert_accepted(name, a1_id(FROM_SQ, sel, TO_SQ))


# ---------------------------------------------------------------------------
# T6.2 -- non-qualifying member of a mixed pair (the worked example)
# ---------------------------------------------------------------------------


def test_t6_2_the_worked_example_accepts_only_the_affordable_tier() -> None:
    """``Q = 0b01``: ``sel = 0`` and ``sel = 1`` legal, ``sel = 2`` and ``3`` not.

    Design "Worked example" quotes 19552 / 19633 legal and 19714 / 19795
    rejected for ``to_sq = 31``; the fixture's ``to_sq`` is 32, so every id
    shifts by one.
    """
    name = "mixed_pair_hand_d_only"
    assert (FROM_SQ, TO_SQ) == (40, 32)
    assert [a1_id(FROM_SQ, sel, TO_SQ) for sel in range(4)] == [
        19553,
        19634,
        19715,
        19796,
    ]

    assert_accepted(name, 19553)
    assert_accepted(name, 19634)
    assert_rejected(name, 19715, "converts the unaffordable 侍 at tier 2")
    assert_rejected(name, 19796, "converts both, including the unaffordable 侍")


def test_t6_2_the_two_accepted_ids_are_genuinely_different_moves() -> None:
    """``sel = 0`` declines the betrayal; under v1 it had no id at all."""
    board = board_of_fen(corpus_fen("mixed_pair_hand_d_only"))
    declined = decode_cython_action(19553, board)
    converted = decode_cython_action(19634, board)

    assert declined["move_type"] != converted["move_type"]
    assert declined["captured"] == []
    assert [tier for _, tier in converted["captured"]] == [1]


# ---------------------------------------------------------------------------
# T6.3 -- a selector bit naming an empty or friendly tier
# ---------------------------------------------------------------------------


def test_t6_3_a_bit_naming_a_friendly_tier_is_refused() -> None:
    """``friendly_topped_enemy_below``: tier 2 is the mover's own 兵.

    Tier 1 holds a convertible enemy, so ``sel = 1`` is legal -- the rejection
    is specifically about naming the friendly tier, not about the square.
    """
    name = "friendly_topped_enemy_below"
    board = board_of_fen(corpus_fen(name))
    assert top_piece(board, 4, 4) == (piece_index("侍"), 0), "friendly on top"
    assert tower(board, 4, 4)[0][1] == 1, "an enemy underneath it"

    assert_accepted(name, a1_id(FROM_SQ, 0, TO_SQ))
    assert_accepted(name, a1_id(FROM_SQ, 1, TO_SQ))
    assert_rejected(name, a1_id(FROM_SQ, 2, TO_SQ), "tier 2 is friendly")
    assert_rejected(name, a1_id(FROM_SQ, 3, TO_SQ), "tier 2 is friendly")


def test_t6_3_a_bit_above_the_tower_height_is_refused() -> None:
    """``intermediate_mode_bit2_dead``: the target tower is one piece tall.

    Tier 2 holds nothing at all, so ``sel = 2`` names a piece that does not
    exist.  The destination tier is 2 here (the tsuke lands on top of a
    height-1 tower), which is why this case is distinct from T6.4's mode bound
    even though it uses the same fixture.
    """
    name = "intermediate_mode_bit2_dead"
    board = board_of_fen(corpus_fen(name))
    assert len(tower(board, 4, 4)) == 1, "a one-piece tower"

    assert_accepted(name, a1_id(FROM_SQ, 0, TO_SQ))
    assert_accepted(name, a1_id(FROM_SQ, 1, TO_SQ))
    assert_rejected(name, a1_id(FROM_SQ, 2, TO_SQ), "tier 2 is empty")
    assert_rejected(name, a1_id(FROM_SQ, 3, TO_SQ), "tier 2 is empty")


# ---------------------------------------------------------------------------
# T6.4 -- the mode bound
# ---------------------------------------------------------------------------


def test_t6_4_bit_one_is_dead_everywhere_in_intermediate_mode() -> None:
    """``maxTier == 2`` leaves no board square where tier 2 can be converted.

    The action space stays mode-independent by design (open question 2): the
    plane exists, it is simply never unmasked.  Asserted over the WHOLE mask,
    not just the fixture's betray square.
    """
    board = board_of_fen(corpus_fen("intermediate_mode_bit2_dead"))
    assert board.mode == 2

    for action_id in masked_ids(legal_mask(board)):
        if v2_block(action_id) != "A1":
            continue
        _, sel, _ = a1_parts(action_id)
        assert sel & 0b10 == 0, f"id {action_id} unmasks tier 2 in mode 2"


@pytest.mark.parametrize("name", CORPUS)
def test_t6_4_no_record_anywhere_converts_above_tier_two(name: str) -> None:
    """The record path agrees: the design's 2-bit bound is exactly saturated."""
    for record in engine_records(board_of_fen(corpus_fen(name))):
        if int(record["move_type"]) in (1, 3, 4):  # tsuke, betray, arata
            tiers = [tier for _, tier in record["captured"]]
            assert all(1 <= tier <= 2 for tier in tiers), record["san"]


# ---------------------------------------------------------------------------
# T6.5 -- arata conversions are tier 1 only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t6_5_no_arata_record_converts_anything_but_tier_one(name: str) -> None:
    """``bsel`` is one bit because tier 1 is the only slot the rules expose.

    Placement requires the target tower to be empty or own-coloured on top
    (``move_gen.ts:237-241``), so with height <= 2 the only possible enemy sits
    at tier 1.  A tier-2 arata conversion is not merely unreached, it is
    unreachable -- and if one ever appeared, this is where it would surface.
    """
    for record in engine_records(board_of_fen(corpus_fen(name))):
        if int(record["move_type"]) != 4:
            continue
        tiers = [tier for _, tier in record["captured"]]
        assert tiers in ([], [1]), f"{name}: arata {record['san']!r} converts {tiers}"


def test_t6_5_the_arata_conversion_fixture_reaches_exactly_two_drop_ids() -> None:
    """One 謀 drop square, two ids: the plain drop and the converting drop."""
    name = "arata_conversion"
    board = board_of_fen(corpus_fen(name))
    to_sq = sq_index(7, 5)

    drops: dict[int, int] = {}
    for action_id in masked_ids(legal_mask(board)):
        if v2_block(action_id) not in ("B", "C"):
            continue
        piece_idx, bsel, drop_sq, _ = arata_parts(action_id)
        if piece_idx == TACTICIAN and drop_sq == to_sq:
            drops[bsel] = action_id

    assert sorted(drops) == [0, 1], "the plain drop and the converting drop"
    sans = {bsel: str(decode_cython_action(i, board)["san"]) for bsel, i in drops.items()}
    assert sans == {0: "新謀(7-5-3)付", 1: "新謀(7-5-3)返兵"}


def test_t6_5_a_drop_betray_bit_over_an_empty_square_is_rejected_three_ways() -> None:
    """The B-block twin of T6.1-T6.3: ``bsel = 1`` where no enemy exists at all.

    Every other triple-rejection case above lives in block A1.  A desync
    confined to the arata blocks -- the mask filler emitting the ``bsel = 1``
    plane unconditionally, or the decoder accepting it and dropping the bit --
    would slip past all of them, so one case is carried here explicitly:
    ``arata_conversion`` offers ``新謀(7-4-1)`` onto an EMPTY square, where the
    plain drop is legal and there is nothing whatsoever for a conversion to
    name.
    """
    name = "arata_conversion"
    board = board_of_fen(corpus_fen(name))

    to_sq = sq_index(7, 4)
    assert to_sq == 59, "(rank 7, file 4) -> (7 - 1) * 9 + (9 - 4)"
    assert tower(board, 7, 4) == [], "the drop target holds no piece to convert"

    # Hand-derived from the design's "New layout", NOT via the encoder: block B
    # is the drop block for a draft that has not finished (``新謀(7-4-1)`` in
    # the frozen dump carries ``draftFinished: false`` and no ``終``), its
    # planes run piece-major with the betray bit minor, and every plane is 81
    # destinations wide:
    #
    #     id = B_START + (piece_idx * 2 + bsel) * 81 + to_sq
    #
    # with ``piece_idx`` the engine's own ``PIECE_INDEX[謀]``.
    plain = V2_B_START + (TACTICIAN * 2 + 0) * 81 + to_sq
    converting = V2_B_START + (TACTICIAN * 2 + 1) * 81 + to_sq
    assert converting - plain == 81, "bsel is minor to the piece index"
    assert arata_parts(converting) == (TACTICIAN, 1, to_sq, False)

    assert_accepted(name, plain)
    assert_rejected(name, converting, "bsel=1 over an empty drop square (Q=0)")


# ---------------------------------------------------------------------------
# T6.6 -- a selector from a non-tactician origin
# ---------------------------------------------------------------------------


def _first_non_tactician_stack(board: Board) -> tuple[int, int]:
    """``(from_sq, to_sq)`` of the lowest-id A1 bit whose mover is not a 謀."""
    for action_id in sorted(masked_ids(legal_mask(board))):
        if v2_block(action_id) != "A1":
            continue
        from_sq, sel, to_sq = a1_parts(action_id)
        if sel != 0:
            continue
        rank, file = rank_file(from_sq)
        mover = top_piece(board, rank, file)
        if mover is not None and mover[0] != TACTICIAN:
            return from_sq, to_sq
    raise AssertionError("fixture has no non-tactician stacking move")


def test_t6_6_a_non_tactician_cannot_carry_a_selector() -> None:
    """Only 謀 betrays, so every ``sel != 0`` from another piece is refused."""
    name = "beginner_start"
    board = board_of_fen(corpus_fen(name))
    from_sq, to_sq = _first_non_tactician_stack(board)

    assert_accepted(name, a1_id(from_sq, 0, to_sq))
    for sel in (1, 2, 3):
        assert_rejected(
            name,
            a1_id(from_sq, sel, to_sq),
            f"sel={sel:02b} from a non-謀 on square {from_sq}",
        )


@pytest.mark.parametrize("name", CORPUS)
def test_t6_6_every_unmasked_selector_comes_from_a_tactician(name: str) -> None:
    """The sweep form: no ``sel != 0`` bit anywhere has a non-謀 origin."""
    board = board_of_fen(corpus_fen(name))
    for action_id in masked_ids(legal_mask(board)):
        if v2_block(action_id) != "A1":
            continue
        from_sq, sel, _ = a1_parts(action_id)
        if sel == 0:
            continue
        rank, file = rank_file(from_sq)
        mover = top_piece(board, rank, file)
        assert mover is not None and mover[0] == TACTICIAN, (
            f"{name}: id {action_id} carries sel={sel:02b} from {rank}-{file}"
        )
