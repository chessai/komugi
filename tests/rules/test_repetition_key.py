"""RT-02, RT-06, RT-08, RT-11 -- the repaired repetition KEY (KOM-027).

Red-first against the v1 rule.  RT-06 and RT-08 are the two aliasing
regressions from ``docs/design/rules-audit.md``: they must FAIL on the v1
cells-only key (``cyengine.pyx:630-636``) and pass on the repaired 274-byte
key.  RT-11 is the completeness property that stops the repair from being
"fixed" in the one direction that would delete the rule entirely -- hashing the
whole FEN, move number and all.

The module skips wholesale until KOM-027 lands; the fixtures themselves are
exercised unconditionally next door in ``test_fixtures_sanity.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

if not hasattr(cyengine, "repetition_key"):
    pytest.skip("KOM-027 repair not yet implemented", allow_module_level=True)

from komugi.engine.cyboard import Board, board_from_fen  # noqa: E402

from fixtures import (  # noqa: E402
    CELLS_SLICE,
    HAND_SLICE,
    POSITION_P_FEN,
    POSITION_Q_FEN,
    REPETITION_KEY_LEN,
    SCALARS_SLICE,
    SCENARIO_A_CYCLE,
    SCENARIO_B_LOOP,
    SOLDIER_INDEX,
    WHITE_COLOR,
    board_of,
    cells_bytes,
    game_with_repeated_history,
    hand_count,
    load_position_p,
    load_position_q,
    play_sequence,
    repetition_count,
    repetition_key,
    valid_mask,
    win_state,
)


def _hand_bytes(board: Board) -> bytes:
    """``hand_arr`` narrowed to ``uint8[14][2]``, as the key stores it."""
    hand: npt.NDArray[np.uint8] = np.asarray(board.hand_arr).astype(np.uint8)
    return hand.tobytes()


# ---------------------------------------------------------------------------
# Shape of the repaired key
# ---------------------------------------------------------------------------


def test_repetition_key_is_exactly_274_bytes() -> None:
    """243 cells + 3 scalars + 28 hand slots, per the audit's Proposed key."""
    for fen in (POSITION_P_FEN, POSITION_Q_FEN):
        key = repetition_key(board_from_fen(fen))
        assert isinstance(key, bytes)
        assert len(key) == REPETITION_KEY_LEN, (
            f"{fen!r}: expected {REPETITION_KEY_LEN} bytes, got {len(key)}"
        )


def test_repetition_key_layout_is_cells_then_scalars_then_hand() -> None:
    """The composition is pinned, not just the length.

    A key of the right length but the wrong composition is the silent-failure
    mode the audit warns about under Q5: it produces wrong draws rather than
    crashing.  So the three slices are compared byte-for-byte against their
    sources.
    """
    board = board_from_fen(POSITION_Q_FEN)
    key = repetition_key(board)

    cells: npt.NDArray[np.int8] = np.asarray(board.cells_arr, dtype=np.int8)
    assert key[CELLS_SLICE] == cells.tobytes(), "leading 243 bytes are the cells"
    assert key[SCALARS_SLICE] == bytes(
        [board.turn, board.draft_w, board.draft_b]
    ), "then turn, draft_w, draft_b -- in that order"
    assert key[HAND_SLICE] == _hand_bytes(board), (
        "then the hand narrowed to uint8[14][2] -- a canonicalization, so two "
        "keys can never differ by int32 padding"
    )


def test_placement_key_is_the_243_byte_leading_block_of_the_full_key() -> None:
    """KOM-141: ``placement_key`` builds gungi.js@3.4.11's cells-only key.

    It used to be a deprecated alias that raised, which was how rules-audit
    Q13 -- "no call site silently keeps the aliased key" -- was enforced. The
    key is real now because a REFEREE has to be able to apply the site's rule;
    Q13's guard moved to ``tests/engine/test_repetition_rule_default.py``,
    which requires every ``Game(`` site in ``src/komugi`` to name its rule or
    be a listed default-FULL site.

    The identity asserted here is the load-bearing one: the placement key IS
    the leading block of the full key, byte for byte. If the two ever built
    their cells block differently, a game refereed under one rule could not be
    replayed under the other, and the parity measurement this exists for would
    be measuring the divergence of two encoders instead of two rules.
    """
    board = board_from_fen(POSITION_P_FEN)
    placement: bytes = cyengine.placement_key(board)

    assert len(placement) == 243, "cells only -- FEN field 1 and nothing else"
    assert placement == repetition_key(board)[CELLS_SLICE]

    cells: npt.NDArray[np.int8] = np.asarray(board.cells_arr, dtype=np.int8)
    assert placement == cells.tobytes(), "and it is the cells themselves"


# ---------------------------------------------------------------------------
# RT-06 -- Scenario A, key level
# ---------------------------------------------------------------------------


def test_rt06_scenario_a_same_cells_different_key() -> None:
    """RT-06: after the 5-ply cycle the cells match P but the key must not.

    ``w = 3, b = 2`` flips the side to move.  Under the v1 key -- cells only --
    the two states collapse to one, which is what makes fourfold behave as
    twofold for every parity-flipping maneuver (F3, and triangulation is the
    canonical one).  ``turn`` in the key is the minimum acceptable repair.
    """
    game = load_position_p()
    start_key = repetition_key(board_of(game))
    start_cells = cells_bytes(game)

    play_sequence(game, SCENARIO_A_CYCLE)

    assert cells_bytes(game) == start_cells, "the cells really did come back"
    assert repetition_key(board_of(game)) != start_key, (
        "same cells, opposite side to move -- different game state, so the "
        "keys must differ (FAILS on the v1 cells-only key)"
    )
    assert repetition_count(game) == 1, (
        "this position has occurred once: the start key and the current key "
        "are different keys"
    )


def test_rt06_the_difference_is_the_turn_byte_alone() -> None:
    """The two Scenario A keys differ in exactly the side-to-move byte."""
    game = load_position_p()
    start_key = repetition_key(board_of(game))

    play_sequence(game, SCENARIO_A_CYCLE)
    now_key = repetition_key(board_of(game))

    assert now_key[CELLS_SLICE] == start_key[CELLS_SLICE]
    assert now_key[HAND_SLICE] == start_key[HAND_SLICE]
    assert now_key[SCALARS_SLICE] != start_key[SCALARS_SLICE]


# ---------------------------------------------------------------------------
# RT-08 -- Scenario B, key level
# ---------------------------------------------------------------------------


def test_rt08_scenario_b_same_cells_same_turn_different_key() -> None:
    """RT-08: the 8-ply loop restores cells AND side to move -- but not hands.

    ``w = b = 4``, so unlike Scenario A nothing about the side to move gives
    the difference away.  What changed is a 兵 spent from White's hand and
    captured off the board: irreversible progress, which the v1 key counts as
    "the same position again" (F4).  Hands are monotone, so including them can
    only ever split key classes.
    """
    game = load_position_q()
    start_key = repetition_key(board_of(game))
    start_cells = cells_bytes(game)
    soldiers_before = hand_count(game, SOLDIER_INDEX, WHITE_COLOR)

    play_sequence(game, SCENARIO_B_LOOP)

    assert cells_bytes(game) == start_cells, "cells byte-identical to Q"
    assert board_of(game).turn == WHITE_COLOR, "and White to move again"
    assert hand_count(game, SOLDIER_INDEX, WHITE_COLOR) == soldiers_before - 1
    assert repetition_key(board_of(game)) != start_key, (
        "the spent 兵 is in the key (FAILS on the v1 cells-only key)"
    )


def test_rt08_the_difference_is_the_hand_slice_alone() -> None:
    """Scenario B's two keys differ in exactly the hand bytes."""
    game = load_position_q()
    start_key = repetition_key(board_of(game))

    play_sequence(game, SCENARIO_B_LOOP)
    now_key = repetition_key(board_of(game))

    assert now_key[CELLS_SLICE] == start_key[CELLS_SLICE]
    assert now_key[SCALARS_SLICE] == start_key[SCALARS_SLICE]
    assert now_key[HAND_SLICE] != start_key[HAND_SLICE]


def test_rt08_three_loops_worth_of_keys_are_all_distinct() -> None:
    """Q, Q-minus-one-兵 and Q-minus-two-兵 are three positions, not one.

    The audit's table: three occurrences on the v1 clock across three states
    that are not the same position and can never recur.
    """
    game = load_position_q()
    keys = [repetition_key(board_of(game))]
    for _ in range(2):
        play_sequence(game, SCENARIO_B_LOOP)
        keys.append(repetition_key(board_of(game)))

    assert len(set(keys)) == 3, "hands only decrease -- no key may repeat"
    assert hand_count(game, SOLDIER_INDEX, WHITE_COLOR) == 0


# ---------------------------------------------------------------------------
# RT-11 -- key completeness
# ---------------------------------------------------------------------------


def _mutated(fen: str, field: str, value: int) -> Board:
    """Load *fen* and set one scalar field on the resulting board."""
    board = board_from_fen(fen)
    setattr(board, field, value)
    return board


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("turn", 1),
        ("draft_w", 1),
        ("draft_b", 1),
    ],
)
def test_rt11_scalar_fields_are_in_the_key(field: str, value: int) -> None:
    """RT-11: boards differing in exactly one of turn/draft_w/draft_b differ.

    Each of the three changes the legal move set -- ``turn`` and the hand gate
    ``_generate_arata`` (``cymovegen.pyx:597-604``), the draft bits confine
    drops to ranks 7-9 / 1-3 (``:616-620``).  That is the same test all four
    included components pass: *equal keys => the engine cannot distinguish the
    states*.
    """
    base = board_from_fen(POSITION_Q_FEN)
    assert getattr(base, field) != value, "the mutation must actually change it"

    other = _mutated(POSITION_Q_FEN, field, value)
    assert repetition_key(base) != repetition_key(other), (
        f"boards differing only in {field} must not share a key"
    )


@pytest.mark.parametrize("piece_index", list(range(14)))
@pytest.mark.parametrize("color", [0, 1])
def test_rt11_every_hand_slot_is_in_the_key(piece_index: int, color: int) -> None:
    """RT-11: all 28 hand slots are load-bearing, not just the ones in use."""
    base = board_from_fen(POSITION_Q_FEN)
    other = board_from_fen(POSITION_Q_FEN)
    other.hand_arr[piece_index, color] += 1

    assert repetition_key(base) != repetition_key(other), (
        f"hand[{piece_index}, {color}] must be in the key"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("move_number", 7),
        ("mode", 2),
    ],
)
def test_rt11_time_and_mode_are_out_of_the_key(field: str, value: int) -> None:
    """RT-11: differing only in move_number (or mode) must give EQUAL keys.

    This is the ``hash(fen)`` trap, called out explicitly in the audit's
    Proposed key: FEN field 6 is the move number, so hashing the whole FEN
    makes every key unique and deletes the repetition rule outright.  ``mode``
    is constant for the lifetime of a game, so including it would be harmless
    but meaningless -- and this test says it is not included.
    """
    base = board_from_fen(POSITION_Q_FEN)
    assert getattr(base, field) != value

    other = _mutated(POSITION_Q_FEN, field, value)
    assert repetition_key(base) == repetition_key(other), (
        f"{field} advances (or is constant) and must NOT be in the key"
    )


# ---------------------------------------------------------------------------
# RT-02 -- the threshold is >=, not ==
# ---------------------------------------------------------------------------


def test_rt02_five_occurrences_still_draw() -> None:
    """RT-02: with the current key present 5 times the rule still fires.

    v1 asks ``max_count == 4`` (``cyengine.pyx:660``), which is only sound
    under an invariant nobody enforces -- "every ply is adjudicated".  A
    ``Game`` reconstructed mid-game from a FEN with a history it did not play
    (``gungi.py:200-208``, exactly what a bridge to an external game server does) breaks it, and
    under ``== 4`` a 5th occurrence silently switches the draw OFF.  The
    repaired predicate is monotone.
    """
    game = game_with_repeated_history(POSITION_P_FEN, 5)

    assert repetition_count(game) == 5, (
        "precondition: the history convention is v1's -- the list CONTAINS "
        "the current position's key, so the count is its multiplicity"
    )
    assert int(win_state(game)[2]) == 1, "still a draw at five, not just four"
    assert int(valid_mask(game).sum()) == 0, "and search sees a terminal node"


@pytest.mark.parametrize("occurrences", [4, 5, 6, 9])
def test_rt02_the_predicate_is_monotone_above_the_threshold(
    occurrences: int,
) -> None:
    """Every count at or above 4 draws -- there is no upper window."""
    game = game_with_repeated_history(POSITION_P_FEN, occurrences)

    assert repetition_count(game) == occurrences
    assert int(win_state(game)[2]) == 1


@pytest.mark.parametrize("occurrences", [1, 2, 3])
def test_rt02_below_the_threshold_nothing_fires(occurrences: int) -> None:
    """...and nothing below 4 draws, so the test above is not vacuous."""
    game = game_with_repeated_history(POSITION_P_FEN, occurrences)

    assert repetition_count(game) == occurrences
    assert int(win_state(game).sum()) == 0, "P is not terminal for any reason"
    assert int(valid_mask(game).sum()) > 0
