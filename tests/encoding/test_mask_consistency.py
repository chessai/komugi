"""T3 and T5 -- the mask filler against the record path, and against ``Q``.

T3 is the design's "single strongest migration check":

    set(np.nonzero(fill_legal_mask(P))[0]) == {encode_cython_move(m)
                                               for m in generate_all_moves(P)}

Before KOM-028 the two paths *could not* agree: ``_fill_moves_for_square`` set
one A1 bit for a whole tsuke/betray family and never called
``_get_betrayal_combos`` at all (``cymovegen.pyx:823-830``), while the record
path enumerated every combination.  The equality is the statement that the
shortcut is gone.

T5 checks the selectors the mask exposes against ``Q`` recomputed by
``helpers.qualifying_mask`` -- this suite's own transcription of
``move_gen.ts:402-423``, deliberately not the engine's.  Both directions:
nothing outside ``Q`` is offered, and nothing inside it is dropped.

Gated on KOM-028.
"""

from __future__ import annotations

from typing import Any

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine.action_encoding import encode_cython_move  # noqa: E402
from komugi.engine.cyboard import Board  # noqa: E402

from helpers import (  # noqa: E402
    TACTICIAN_KANJI,
    a1_parts,
    arata_parts,
    board_of_fen,
    corpus_fen,
    corpus_names,
    engine_records,
    legal_mask,
    masked_ids,
    piece_index,
    qualifying_mask,
    rank_file,
    submasks,
    top_piece,
    tower,
    v2_block,
)

CORPUS: tuple[str, ...] = corpus_names()

TACTICIAN: int = piece_index(TACTICIAN_KANJI)


def _board(name: str) -> Board:
    return board_of_fen(corpus_fen(name))


def _ids_by_block(board: Board, block: str) -> set[int]:
    return {i for i in masked_ids(legal_mask(board)) if v2_block(i) == block}


# ---------------------------------------------------------------------------
# T3 -- mask / record consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t3_the_mask_is_exactly_the_encoded_record_set(name: str) -> None:
    """Bit for bit: the two enumerations cannot disagree about a selector."""
    board = _board(name)
    from_mask = masked_ids(legal_mask(board))
    from_records = {encode_cython_move(record) for record in engine_records(board)}

    assert from_mask == from_records, (
        f"{name}: mask-only {sorted(from_mask - from_records)}, "
        f"records-only {sorted(from_records - from_mask)}"
    )


@pytest.mark.parametrize("name", CORPUS)
def test_t3_the_mask_never_sets_a_bit_outside_the_space(name: str) -> None:
    """A stale block constant in ``cymovegen`` shows up as an id in no block."""
    mask = legal_mask(board_of_fen(corpus_fen(name)))
    assert mask.shape == (ae.ACTION_SIZE,)
    for action_id in masked_ids(mask):
        assert v2_block(action_id) in ("A0", "A1", "B", "C")


# ---------------------------------------------------------------------------
# T5 -- selector legality against the test's own Q
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t5_every_masked_a1_selector_is_a_submask_of_q(name: str) -> None:
    """``(sel & ~Q) == 0`` for every A1 bit the mask sets with ``sel != 0``."""
    board = _board(name)
    color = int(board.turn)

    for action_id in _ids_by_block(board, "A1"):
        from_sq, sel, to_sq = a1_parts(action_id)
        if sel == 0:
            continue
        to_rank, to_file = rank_file(to_sq)
        qualifying = qualifying_mask(board, to_rank, to_file, color)
        assert sel & ~qualifying == 0, (
            f"{name}: id {action_id} offers sel={sel:02b} at "
            f"{to_rank}-{to_file} where Q={qualifying:02b}"
        )
        assert sel & ~0b11 == 0, "a selector bit above tier 2"

        from_rank, from_file = rank_file(from_sq)
        mover = top_piece(board, from_rank, from_file)
        assert mover is not None
        assert mover == (TACTICIAN, color), (
            f"{name}: a conversion offered from {from_rank}-{from_file}, "
            "whose top piece is not the mover's 謀"
        )


@pytest.mark.parametrize("name", CORPUS)
def test_t5_every_submask_of_q_is_offered(name: str) -> None:
    """The converse: no legal subset is dropped from the mask.

    At a square pair where the stacking branch fires, the legal selector set is
    exactly ``{0} ∪ non-empty subsets of Q`` when the mover is a tactician --
    which is ``submasks(Q)``, since ``submasks(0) == (0,)`` covers both the
    no-enemy and the nothing-qualifies cases -- and exactly ``{0}`` otherwise.
    """
    board = _board(name)
    color = int(board.turn)

    offered: dict[tuple[int, int], set[int]] = {}
    for action_id in _ids_by_block(board, "A1"):
        from_sq, sel, to_sq = a1_parts(action_id)
        offered.setdefault((from_sq, to_sq), set()).add(sel)

    for (from_sq, to_sq), sels in offered.items():
        from_rank, from_file = rank_file(from_sq)
        to_rank, to_file = rank_file(to_sq)
        mover = top_piece(board, from_rank, from_file)
        assert mover is not None
        if mover[0] == TACTICIAN:
            expected = set(submasks(qualifying_mask(board, to_rank, to_file, color)))
        else:
            expected = {0}
        assert sels == expected, (
            f"{name}: {from_rank}-{from_file} -> {to_rank}-{to_file} offers "
            f"{sorted(sels)}, expected {sorted(expected)}"
        )


@pytest.mark.parametrize("name", CORPUS)
def test_t5_the_arata_bit_matches_q_at_the_drop_square(name: str) -> None:
    """``bsel`` is set iff a 謀 drop can convert the tier-1 enemy underneath."""
    board = _board(name)
    color = int(board.turn)

    offered: dict[tuple[int, int, bool], set[int]] = {}
    for action_id in masked_ids(legal_mask(board)):
        if v2_block(action_id) not in ("B", "C"):
            continue
        piece_idx, bsel, to_sq, finished = arata_parts(action_id)
        offered.setdefault((piece_idx, to_sq, finished), set()).add(bsel)

    for (piece_idx, to_sq, _), bsels in offered.items():
        to_rank, to_file = rank_file(to_sq)
        qualifying = qualifying_mask(board, to_rank, to_file, color)
        assert qualifying & ~0b01 == 0, (
            "a legal arata target is empty or own-coloured on top, so tier 1 "
            f"is the only convertible slot; Q={qualifying:02b} at "
            f"{to_rank}-{to_file}"
        )
        if piece_idx == TACTICIAN and qualifying == 0b01:
            expected = {0, 1}
        else:
            expected = {0}
        assert bsels == expected, (
            f"{name}: drop of piece {piece_idx} on {to_rank}-{to_file} offers "
            f"bsel {sorted(bsels)}, expected {sorted(expected)}"
        )


def test_t5_the_worked_example_geometry_is_the_fixtures_geometry() -> None:
    """``mixed_pair_hand_d_only`` is the design's worked example, one file over.

    Design: tactician on 5-5, target 4-5 holding an enemy 兵 at tier 1 and an
    enemy 侍 at tier 2, hand ``兵 x1``, so ``Q = 0b01``.  The frozen fixture
    puts the same tower on 4-4 (the corpus uses ``|D:T|`` bases so the
    tactician's tier reaches the target).  Everything else is identical, and
    this test pins that -- ``test_negative_selectors`` then quotes the ids.
    """
    board = _board("mixed_pair_hand_d_only")
    assert top_piece(board, 5, 5) == (TACTICIAN, 0)
    assert len(tower(board, 5, 5)) == 2, "tactician on tier 2"

    target = tower(board, 4, 4)
    assert len(target) == 2
    assert [color for _, color in target] == [1, 1], "both tiers hostile"
    assert target[0][0] != target[1][0], "a mixed-type pair"
    assert qualifying_mask(board, 4, 4, 0) == 0b01
