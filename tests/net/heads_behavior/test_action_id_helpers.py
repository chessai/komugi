"""The heads suite's own arithmetic, checked against the engine (KOM-105).

UNCONDITIONAL -- no ``komugi.net`` gate.  The heads (KOM-005a) do not exist in
this worktree; the engine does, and every number ``action_ids.py`` transcribes
belongs to the engine.  So the transcription is verifiable *now*, and the
gated modules that lean on it are only as trustworthy as this file is.

Two failure modes it exists to catch:

* a transcription slip -- the four block starts, the 461/81/37341 triple, the
  per-block plane counts, or the square convention, typed wrong here and then
  used to "prove" a head correct;
* the engine moving underneath the suite -- an ``ACTION_SIZE`` drift, a
  re-ordered block table, a changed square index -- which must surface as a
  named failure here rather than as a silent shape mismatch inside a head.

The square convention gets the sharpest treatment: ``sq_index == row * 9 + col``
is the single identity that makes a policy plane's row-major flatten land on
the engine's ids, and it is checked over all 81 squares against the engine's
own ``sq_index`` / ``sq_from_index``.
"""

from __future__ import annotations

import action_ids
from action_ids import (
    BLOCK_NAMES,
    BLOCK_PLANE_OFFSETS,
    BLOCK_PLANES,
    BLOCK_STARTS,
    BOARD,
    NUM_ACTIONS,
    PINS,
    POLICY_PLANES,
    SQUARES,
)
from komugi.engine import action_encoding as ae
from komugi.engine.game import Game
from komugi.engine.observation import observation_size

#: ``block name -> the engine's own start constant``.
_ENGINE_STARTS: dict[str, int] = {
    "A0": ae.A0_START,
    "A1": ae.A1_START,
    "B": ae.B_START,
    "C": ae.C_START,
}


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------


def test_the_transcribed_layout_is_the_engines() -> None:
    """461 planes x 81 squares == 37341 == ``ACTION_SIZE``, block for block."""
    assert sum(BLOCK_PLANES) == POLICY_PLANES
    assert POLICY_PLANES * SQUARES == NUM_ACTIONS == 37341
    assert ae.ACTION_SIZE == NUM_ACTIONS
    assert [_ENGINE_STARTS[name] for name in BLOCK_NAMES] == list(BLOCK_STARTS)
    assert BLOCK_STARTS == (0, 6561, 32805, 35073)
    assert ae.SHU_BLOCK == (BLOCK_STARTS[3], NUM_ACTIONS)


def test_the_plane_offsets_and_the_id_starts_are_one_statement() -> None:
    """A block's first plane and its first id differ by exactly the 81 stride.

    This is the whole reason a 461-plane conv can carry the action space: the
    blocks are contiguous *and* identically strided, so plane-major within a
    block and plane-major across the stack are the same ordering.
    """
    assert BLOCK_PLANE_OFFSETS == (0, 81, 405, 433)
    for offset, start in zip(BLOCK_PLANE_OFFSETS, BLOCK_STARTS):
        assert start == offset * SQUARES


def test_the_game_action_size_follows_the_encoding() -> None:
    """The width the policy head must match is the engine's, symbolically."""
    assert Game.action_size() == NUM_ACTIONS


def test_the_observation_plane_count_is_read_from_the_encoder() -> None:
    """``in_planes`` follows the encoder; 128 is pinned so a change is visible."""
    assert observation_size() == (128, BOARD, BOARD)
    assert action_ids.observation_planes() == 128


# ---------------------------------------------------------------------------
# The bijection
# ---------------------------------------------------------------------------


def test_every_id_splits_and_rejoins_through_the_block_table() -> None:
    """``plane_and_square_to_id(id_to_plane_and_square(i)) == i``, all 37341.

    And, in the same loop, the claim the head's ``flatten(1)`` depends on: the
    block-table split agrees with the flat ``divmod(i, 81)``.  A layout whose
    blocks were not contiguous, or whose strides differed, would separate the
    two here -- which is exactly the drift a future block would introduce.
    """
    for action_id in range(NUM_ACTIONS):
        plane, square = action_ids.id_to_plane_and_square(action_id)
        assert (plane, square) == divmod(action_id, SQUARES), action_id
        assert action_ids.plane_and_square_to_id(plane, square) == action_id


def test_every_plane_owns_81_contiguous_ids_and_they_partition_the_space() -> None:
    """The planes tile ``[0, 37341)`` exactly once, in order."""
    seen = 0
    for plane in range(POLICY_PLANES):
        ids = action_ids.plane_ids(plane)
        assert ids.start == seen and len(ids) == SQUARES
        seen = ids.stop
    assert seen == NUM_ACTIONS


def test_the_hand_derived_pins_land_where_the_derivation_says() -> None:
    """Each pinned id, re-derived from the ENGINE's block start and back."""
    for pin in PINS:
        assert action_ids.id_to_plane_and_square(pin.action_id) == (
            pin.plane,
            pin.square,
        ), pin
        assert action_ids.plane_and_square_to_id(pin.plane, pin.square) == pin.action_id
        assert action_ids.block_of_plane(pin.plane) == pin.block, pin

        # The independent path: the engine's own start constant plus the
        # block-local plane, which is what the docstring's
        # ``block_start + plane * 81 + to_sq`` actually says.
        block = BLOCK_NAMES.index(pin.block)
        local_plane = pin.plane - BLOCK_PLANE_OFFSETS[block]
        assert 0 <= local_plane < BLOCK_PLANES[block], pin
        expected = _ENGINE_STARTS[pin.block] + local_plane * SQUARES + pin.square
        assert expected == pin.action_id, pin

        assert ae.is_c_block(pin.action_id) == (pin.block == "C"), pin


def test_the_pins_cover_every_block_boundary() -> None:
    """First and last id of all four blocks are pinned -- no corner unpinned."""
    pinned = {pin.action_id for pin in PINS}
    for block, start in enumerate(BLOCK_STARTS):
        last = start + BLOCK_PLANES[block] * SQUARES - 1
        assert start in pinned, f"block {BLOCK_NAMES[block]} start {start} unpinned"
        assert last in pinned, f"block {BLOCK_NAMES[block]} end {last} unpinned"


def test_the_worked_example_id_is_the_engines_arithmetic() -> None:
    """19553 = tsuke 5-5 -> 4-4 with the betrayal declined (``sel = 0``).

    ``from_sq = sq_index(5-5) = 40``, ``to_sq = sq_index(4-4) = 32``, so the
    id is ``A1_START + (40 * 4 + 0) * 81 + 32`` -- plane 241, square 32.  The
    three neighbouring selectors sit one plane (81 ids) apart, which is the
    property that keeps a square pair's four selector planes adjacent.
    """
    assert ae.sq_index("5-5") == 40
    assert ae.sq_index("4-4") == 32
    assert ae.A1_START + (40 * 4 + 0) * SQUARES + 32 == 19553
    assert action_ids.id_to_plane_and_square(19553) == (241, 32)
    for selector in range(4):
        action_id = ae.A1_START + (40 * 4 + selector) * SQUARES + 32
        plane, square = action_ids.id_to_plane_and_square(action_id)
        assert (plane, square) == (241 + selector, 32)


# ---------------------------------------------------------------------------
# Squares
# ---------------------------------------------------------------------------


def test_the_square_index_is_the_row_major_flatten_of_a_plane() -> None:
    """``sq_index(rank, file) == row * 9 + col`` for every square.

    ``row = rank - 1`` and ``col = 9 - file`` are the coordinates the
    observation encoder writes to (``observation_from_cyboard``: ``j = 8 - f``
    on 0-based files), so this is the bridge between "square 32" in the action
    space and "(3, 5)" in a 9x9 feature map.  Checked against the engine's own
    ``sq_index`` and its inverse, both directions, all 81 squares.
    """
    for rank in range(1, BOARD + 1):
        for file in range(1, BOARD + 1):
            square = ae.sq_index(f"{rank}-{file}")
            assert action_ids.sq_index(rank, file) == square
            assert action_ids.rank_file(square) == (rank, file)
            assert ae.sq_from_index(square) == f"{rank}-{file}"

            row, col = action_ids.spatial_of_sq(square)
            assert (row, col) == (rank - 1, BOARD - file)
            assert action_ids.sq_of_spatial(row, col) == square


def test_the_square_indices_are_a_bijection_onto_0_80() -> None:
    """No square index is used twice and none is missing."""
    squares = {
        action_ids.sq_index(rank, file)
        for rank in range(1, BOARD + 1)
        for file in range(1, BOARD + 1)
    }
    assert squares == set(range(SQUARES))


def test_the_gated_sibling_modules_can_never_skip_silently() -> None:
    """``komugi.net`` must import and export ``KomugiNet``, unconditionally.

    Every other module in this directory skip-gates on that export so the
    suite could land red-first, before the heads existed. Post-union those
    gates are dead code with one live failure mode: a refactor that breaks
    the ``komugi.net`` import or drops the export would turn the whole armed
    suite back into silent skips under a green gate. This module runs
    unconditionally, so the regression is loud here (wave-8 review, F2).
    """
    import komugi.net

    assert hasattr(komugi.net, "KomugiNet")
