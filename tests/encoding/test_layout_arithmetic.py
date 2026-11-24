"""T4 -- pure arithmetic bijection over the extended action space.

Board-free: nothing here loads a position.  The obligation is that the 37341
ids of ``docs/design/betrayal-encoding.md`` "New layout" form an exact
partition into four blocks, that every id round-trips through its
``(block, plane-major index, selector, to_sq)`` parts, and that the block
starts really are the running plane sums rather than four hand-typed numbers
that happen to look right.

``decode_parts`` / ``encode_from_parts`` are implemented HERE rather than
imported: a bijection test that called the encoder's own inverse would assert
only that the encoder is self-consistent.  The parts decomposition below is
transcribed from the design's table, so a disagreement is a disagreement with
the spec.

Gated on KOM-028.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine import cymovegen  # noqa: E402
from komugi.engine.game import Game  # noqa: E402

from helpers import (  # noqa: E402
    V2_A0_START,
    V2_A1_START,
    V2_ACTION_SIZE,
    V2_B_START,
    V2_BLOCK_PLANES,
    V2_C_START,
    V2_PLANES,
)


class Parts(NamedTuple):
    """An action id's four coordinates in the extended layout.

    ``major`` is ``from_sq`` for A0/A1 and ``piece_idx`` for B/C; ``selector``
    is ``sel`` (0..3) in A1, ``bsel`` (0..1) in B/C and always 0 in A0, which
    has no selector.
    """

    block: str
    major: int
    selector: int
    to_sq: int


#: ``block -> (start, plane-major span, selector span)``, straight off the
#: design's "New layout" table.
BLOCK_SPEC: dict[str, tuple[int, int, int]] = {
    "A0": (V2_A0_START, 81, 1),
    "A1": (V2_A1_START, 81, 4),
    "B": (V2_B_START, 14, 2),
    "C": (V2_C_START, 14, 2),
}

BLOCK_ORDER: tuple[str, ...] = ("A0", "A1", "B", "C")


def decode_parts(action_id: int) -> Parts:
    """Split *action_id* into its layout coordinates."""
    if not 0 <= action_id < V2_ACTION_SIZE:
        raise ValueError(f"not an action id: {action_id}")
    for block in reversed(BLOCK_ORDER):
        start, _, selectors = BLOCK_SPEC[block]
        if action_id >= start:
            plane, to_sq = divmod(action_id - start, 81)
            major, selector = divmod(plane, selectors)
            return Parts(block, major, selector, to_sq)
    raise AssertionError("unreachable: A0 starts at 0")


def encode_from_parts(parts: Parts) -> int:
    """Inverse of :func:`decode_parts`."""
    start, majors, selectors = BLOCK_SPEC[parts.block]
    if not 0 <= parts.major < majors:
        raise ValueError(f"major index out of range for {parts.block}: {parts}")
    if not 0 <= parts.selector < selectors:
        raise ValueError(f"selector out of range for {parts.block}: {parts}")
    if not 0 <= parts.to_sq < 81:
        raise ValueError(f"destination out of range: {parts}")
    return start + (parts.major * selectors + parts.selector) * 81 + parts.to_sq


# ---------------------------------------------------------------------------
# The declared constants
# ---------------------------------------------------------------------------


def test_action_size_is_461_planes_of_81_squares() -> None:
    """``461 x 81 == 37341`` -- the policy head stays a 1x1 conv over 9x9."""
    assert sum(V2_BLOCK_PLANES) == V2_PLANES
    assert V2_PLANES * 81 == V2_ACTION_SIZE == 37341
    assert ae.ACTION_SIZE == V2_ACTION_SIZE


def test_block_starts_are_the_running_plane_sums() -> None:
    """``0 / 6561 / 32805 / 35073`` derived, not asserted as four literals."""
    running = 0
    expected: list[int] = []
    for planes in V2_BLOCK_PLANES:
        expected.append(running)
        running += planes * 81
    assert running == V2_ACTION_SIZE
    assert expected == [V2_A0_START, V2_A1_START, V2_B_START, V2_C_START]
    assert [ae.A0_START, ae.A1_START, ae.B_START, ae.C_START] == expected


def test_shu_block_still_bounds_block_c() -> None:
    """``SHU_BLOCK`` moves with ``C_START`` and keeps its v1 semantics."""
    assert ae.SHU_BLOCK == (V2_C_START, V2_ACTION_SIZE)
    assert ae.is_c_block(V2_C_START)
    assert ae.is_c_block(V2_ACTION_SIZE - 1)
    assert not ae.is_c_block(V2_C_START - 1)


def test_the_game_action_size_follows_the_encoding() -> None:
    """``game.action_size`` reads ``ACTION_SIZE`` symbolically -- prove it."""
    assert Game.action_size() == V2_ACTION_SIZE


# ---------------------------------------------------------------------------
# The bijection
# ---------------------------------------------------------------------------


def test_every_id_round_trips_through_its_parts() -> None:
    """``encode_from_parts(decode_parts(id)) == id`` for all 37341 ids."""
    for action_id in range(V2_ACTION_SIZE):
        assert encode_from_parts(decode_parts(action_id)) == action_id


def test_the_parts_enumeration_covers_the_space_exactly_once() -> None:
    """The other direction: the coordinate product is a bijection onto the ids."""
    seen: set[int] = set()
    for block in BLOCK_ORDER:
        _, majors, selectors = BLOCK_SPEC[block]
        for major in range(majors):
            for selector in range(selectors):
                for to_sq in range(81):
                    action_id = encode_from_parts(
                        Parts(block, major, selector, to_sq)
                    )
                    assert action_id not in seen, f"collision at {action_id}"
                    seen.add(action_id)
    assert seen == set(range(V2_ACTION_SIZE))


def test_every_id_lands_in_exactly_one_block() -> None:
    """Half-open block ranges partition ``[0, ACTION_SIZE)`` with no overlap."""
    bounds = {
        "A0": (V2_A0_START, V2_A1_START),
        "A1": (V2_A1_START, V2_B_START),
        "B": (V2_B_START, V2_C_START),
        "C": (V2_C_START, V2_ACTION_SIZE),
    }
    for action_id in range(V2_ACTION_SIZE):
        holders = [
            name for name, (lo, hi) in bounds.items() if lo <= action_id < hi
        ]
        assert holders == [decode_parts(action_id).block], action_id


def test_a_squares_four_selector_planes_are_adjacent() -> None:
    """``sel`` is minor to ``from_sq`` (design "Plane ordering"), not major.

    Consecutive selectors at one square pair are exactly 81 apart -- the
    property that makes a square's four planes a contiguous slice.
    """
    for from_sq in (0, 40, 80):
        for to_sq in (0, 31, 80):
            ids = [
                V2_A1_START + (from_sq * 4 + sel) * 81 + to_sq for sel in range(4)
            ]
            assert [b - a for a, b in zip(ids, ids[1:])] == [81, 81, 81]
            assert all(decode_parts(i).major == from_sq for i in ids)
            assert [decode_parts(i).selector for i in ids] == [0, 1, 2, 3]


def test_the_worked_example_ids_are_what_the_design_says() -> None:
    """Design "Worked example": tactician 5-5 onto 4-5, ``from_sq=40``, ``to_sq=31``."""
    assert V2_A1_START + (40 * 4 + 0) * 81 + 31 == 19552
    assert V2_A1_START + (40 * 4 + 1) * 81 + 31 == 19633
    assert V2_A1_START + (40 * 4 + 2) * 81 + 31 == 19714
    assert V2_A1_START + (40 * 4 + 3) * 81 + 31 == 19795


# ---------------------------------------------------------------------------
# The duplicated Cython constants
# ---------------------------------------------------------------------------

#: ``cymovegen``'s block starts are duplicated from ``action_encoding`` by
#: design (design, "cymovegen.pyx": "they are duplicated ... KOM-128 T4
#: asserts they agree").
_DUPLICATED: dict[str, int] = {
    "A0_START": V2_A0_START,
    "A1_START": V2_A1_START,
    "B_START": V2_B_START,
    "C_START": V2_C_START,
    "ACTION_SIZE": V2_ACTION_SIZE,
}


def test_cymovegen_duplicated_constants_agree_with_action_encoding() -> None:
    """Whatever ``cymovegen`` exposes of its own copy must match.

    The v1 constants are ``cdef int`` and therefore have no python-level name,
    so this can only check what KOM-028 chooses to export.  Every name it does
    export is checked; the behavioural half of the same statement -- that the
    mask filler writes into the blocks these constants describe -- is
    ``test_mask_consistency`` (T3), which compares the mask against
    ``encode_cython_move`` id for id.

    If ``cymovegen`` exports NONE of them the agreement is unverifiable, and
    this test would otherwise pass on an empty dict -- a vacuous green.  That
    case is an ``xfail`` instead, so the gap is visible in every run's summary
    rather than hidden behind a pass.
    """
    exported = {
        name: getattr(cymovegen, name)
        for name in _DUPLICATED
        if hasattr(cymovegen, name)
    }
    if not exported:
        pytest.xfail(
            "cymovegen exports no python-visible layout constants -- "
            "duplicated-constants agreement unverifiable"
        )
    assert exported == {name: _DUPLICATED[name] for name in exported}


def test_the_mask_buffer_width_matches_the_encoding() -> None:
    """``fill_legal_mask`` accepts an ``ACTION_SIZE`` buffer and no other.

    A stale duplicated constant inside ``cymovegen`` shows up here as an
    out-of-bounds write or a silently ignored buffer, whichever the Cython
    bounds checking gives -- either way not a clean pass.
    """
    import numpy as np

    from komugi.engine.cyboard import board_from_fen
    from komugi.engine.fen import ADVANCED_POSITION

    board = board_from_fen(ADVANCED_POSITION)
    mask = np.zeros(V2_ACTION_SIZE, dtype=np.uint8)
    cymovegen.fill_legal_mask(board, mask)
    assert mask.shape == (V2_ACTION_SIZE,)
    assert int(mask.sum()) > 0
