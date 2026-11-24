"""T8 -- ``_mirror_action`` under the left-right file reflection.

The only symmetry Gungi has is the file flip, and the training loop uses it to
double every self-play sample (``game.symmetries``).  A tier subset is
invariant under a file reflection, so the widened A1 arm must mirror
``from_sq`` and ``to_sq`` and leave ``sel`` alone; blocks B/C must preserve
``bsel``.  Getting it wrong corrupts symmetry-augmented policy targets
silently -- the mirrored probability mass lands on a *different betrayal* --
which is why this is asserted as an involution and as a legality-preserving
bijection rather than spot-checked.

Gated on KOM-028.
"""

from __future__ import annotations

from typing import Any

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine.game import _mirror_action  # noqa: E402

from helpers import (  # noqa: E402
    V2_ACTION_SIZE,
    a1_parts,
    arata_parts,
    board_of_fen,
    corpus_fen,
    corpus_names,
    legal_mask,
    masked_ids,
    mirror_fen,
    mirror_sq,
    v2_block,
)

CORPUS: tuple[str, ...] = corpus_names()

#: A stride over the whole space for the board-free sweeps.  Prime, so it hits
#: every residue class mod 81 (the destination square) and mod 4 (the
#: selector) rather than sampling one plane over and over.
_STRIDE: int = 7


def test_the_file_mirror_is_an_involution_on_the_whole_space() -> None:
    """Board-free: ``_mirror_action`` squares to the identity everywhere.

    Not just on legal ids -- ``_mirror_pi`` walks whatever is nonzero in a
    policy vector, so an id that is illegal at this node still has to survive
    the round trip.
    """
    for action_id in range(0, V2_ACTION_SIZE, _STRIDE):
        mirrored = _mirror_action(action_id)
        assert 0 <= mirrored < V2_ACTION_SIZE
        assert v2_block(mirrored) == v2_block(action_id)
        assert _mirror_action(mirrored) == action_id


def test_the_mirror_moves_the_squares_and_nothing_else() -> None:
    """A1: ``from_sq`` and ``to_sq`` flip, ``sel`` does not.  B/C: same for ``bsel``."""
    for action_id in range(0, V2_ACTION_SIZE, _STRIDE):
        block = v2_block(action_id)
        mirrored = _mirror_action(action_id)
        if block == "A1":
            from_sq, sel, to_sq = a1_parts(action_id)
            m_from, m_sel, m_to = a1_parts(mirrored)
            assert (m_from, m_to) == (mirror_sq(from_sq), mirror_sq(to_sq))
            assert m_sel == sel, "a file reflection does not reorder a tower"
        elif block in ("B", "C"):
            piece_idx, bsel, to_sq, finished = arata_parts(action_id)
            m_piece, m_bsel, m_to, m_finished = arata_parts(mirrored)
            assert (m_piece, m_bsel, m_finished) == (piece_idx, bsel, finished)
            assert m_to == mirror_sq(to_sq)


@pytest.mark.parametrize("name", CORPUS)
def test_t8_every_legal_id_mirrors_to_a_legal_id_in_the_mirrored_position(
    name: str,
) -> None:
    """The obligation as stated: involution, legality, selector identity."""
    board = board_of_fen(corpus_fen(name))
    mirrored_board = board_of_fen(mirror_fen(corpus_fen(name)))

    legal = masked_ids(legal_mask(board))
    mirrored_legal = masked_ids(legal_mask(mirrored_board))
    assert legal, f"{name} has no legal move to mirror"

    for action_id in sorted(legal):
        mirrored = _mirror_action(action_id)
        assert _mirror_action(mirrored) == action_id
        assert mirrored in mirrored_legal, (
            f"{name}: {action_id} mirrors to {mirrored}, illegal in the "
            "file-mirrored position"
        )
        if v2_block(action_id) == "A1":
            assert a1_parts(mirrored)[1] == a1_parts(action_id)[1]
        elif v2_block(action_id) in ("B", "C"):
            assert arata_parts(mirrored)[1] == arata_parts(action_id)[1]


@pytest.mark.parametrize("name", CORPUS)
def test_t8_the_mirror_is_a_bijection_between_the_two_masks(name: str) -> None:
    """The stronger form: the mirrored mask is exactly the mirrored image.

    ``⊆`` is the test above; equality additionally rules out the mirrored
    position offering something the original does not, which would mean the
    augmented sample and its policy target disagree about the legal set.
    """
    board = board_of_fen(corpus_fen(name))
    mirrored_board = board_of_fen(mirror_fen(corpus_fen(name)))

    image = {_mirror_action(i) for i in masked_ids(legal_mask(board))}
    assert image == masked_ids(legal_mask(mirrored_board))


def test_the_fen_mirror_is_itself_an_involution() -> None:
    """Sanity on the fixture transform this module leans on."""
    for name in CORPUS:
        fen = corpus_fen(name)
        assert mirror_fen(mirror_fen(fen)) == fen


def test_a_selector_carrying_id_really_is_exercised_by_the_mirror() -> None:
    """The corpus reaches the widened A1 arm, so T8 is not vacuous there."""
    board = board_of_fen(corpus_fen("mixed_pair_hand_both"))
    selectors = {
        a1_parts(i)[1]
        for i in masked_ids(legal_mask(board))
        if v2_block(i) == "A1"
    }
    assert selectors == {0, 1, 2, 3}
