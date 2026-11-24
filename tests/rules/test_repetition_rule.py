"""RT-01/12, RT-03, RT-05, RT-07, RT-09, RT-19 -- the repaired repetition RULE.

The key tests next door say the repair splits the states v1 merged.  These say
the rule built on that key still adjudicates: a genuine cycle draws at exactly
the 4th occurrence (RT-01/RT-12 -- the repair must not delete the rule), the
two aliasing scenarios no longer draw (RT-07, RT-09 -- red on v1), search and
scoring agree at every ply (RT-05), and the F8 precedence question is pinned
rather than left to drift (RT-19).

Gated on KOM-027; the sequences themselves are exercised unconditionally in
``test_fixtures_sanity.py``.
"""

from __future__ import annotations

import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

if not hasattr(cyengine, "repetition_key"):
    pytest.skip("KOM-027 repair not yet implemented", allow_module_level=True)

from fixtures import (  # noqa: E402
    BLACK_COLOR,
    FOURFOLD,
    MATE_AND_REPETITION_FEN,
    PURE_SHUFFLE,
    SCENARIO_A_15PLY,
    SCENARIO_B_LOOP,
    WHITE_COLOR,
    board_of,
    cells_bytes,
    game_with_repeated_history,
    is_terminal,
    load_position_p,
    load_position_q,
    play_action_checked,
    play_sequence,
    repetition_count,
    valid_mask,
    win_state,
)

#: RT-01's line: the genuine 4-ply cycle, three times over.  Occurrence 1 is
#: the start position itself (seeded into the history at construction), so
#: three cycles is what it takes to reach four.
RT01_LINE: tuple[int, ...] = PURE_SHUFFLE * 3

#: RT-09's line: both Scenario B loops, then the genuine shuffle that the
#: audit says "cashes" the banked counts under the v1 rule.
RT09_LINE: tuple[int, ...] = SCENARIO_B_LOOP * 2 + PURE_SHUFFLE


# ---------------------------------------------------------------------------
# RT-01 / RT-12 -- a genuine cycle still draws, at exactly the 4th occurrence
# ---------------------------------------------------------------------------


def test_rt01_draw_fires_at_occurrence_four_and_not_before() -> None:
    """RT-01 (and RT-12): non-terminal at occurrences 1, 2, 3; drawn at 4.

    The cycle is a true repetition by every measure the repaired key records:
    same cells, same side to move, same hands, same draft flags.  So the repair
    must leave it drawing -- a repair that only ever splits key classes could
    otherwise "fix" the rule into never firing.
    """
    game = load_position_p()
    baseline = cells_bytes(game)

    assert repetition_count(game) == 1, "the start position is occurrence 1"
    assert not is_terminal(game)

    for occurrence in (2, 3):
        play_sequence(game, PURE_SHUFFLE)
        assert cells_bytes(game) == baseline
        assert repetition_count(game) == occurrence
        assert not is_terminal(game), (
            f"occurrence {occurrence} of {FOURFOLD} must not end the game"
        )

    # The fourth occurrence: play the cycle one ply at a time so the draw can
    # be located at a ply, not merely at a cycle boundary.
    for ply, action in enumerate(PURE_SHUFFLE, start=1):
        assert not is_terminal(game), f"still live entering ply {ply} of the cycle"
        play_action_checked(game, action, ply)

    assert repetition_count(game) == FOURFOLD
    assert int(win_state(game)[2]) == 1, "draw slot set at the 4th occurrence"
    assert int(win_state(game)[WHITE_COLOR]) == 0
    assert int(win_state(game)[BLACK_COLOR]) == 0


def test_rt01_the_intermediate_positions_never_reach_four() -> None:
    """The draw is keyed on the position ON THE BOARD, not on some other one.

    Across the three cycles each intermediate position occurs three times and
    only the cycle's base position reaches four, so a max-over-all-keys rule
    and a current-key rule agree here.  Asserting the intermediate counts makes
    the *reason* the draw fires at ply 12 explicit: it is this position's
    fourth occurrence, not somebody else's.
    """
    game = load_position_p()
    counts: list[int] = []
    for ply, action in enumerate(RT01_LINE, start=1):
        play_action_checked(game, action, ply)
        counts.append(repetition_count(game))

    # plies 1..11 are the three intermediate positions of the cycle, each seen
    # once per completed cycle; ply 12 is the base position's 4th occurrence.
    assert counts[-1] == FOURFOLD
    assert max(counts[:-1]) == 3
    assert counts[:4] == [1, 1, 1, 2], (
        "first cycle: three fresh positions, then the base position's 2nd"
    )


# ---------------------------------------------------------------------------
# RT-03 -- the counted multiset includes the start position
# ---------------------------------------------------------------------------


def test_rt03_the_start_position_counts_as_an_occurrence() -> None:
    """RT-03: a cycle back to the initial position needs 3 further occurrences.

    ``Game.__init__`` seeds the history with the start position's key
    (``gungi.py:208``), so the position on the board at ply 0 is already
    occurrence 1.  Three cycles draw; if the seed were dropped it would take
    four, and this test would fail at ply 12 rather than at ply 16.
    """
    game = load_position_p()
    assert repetition_count(game) == 1, "seeded at construction"

    play_sequence(game, PURE_SHUFFLE * 2)
    assert repetition_count(game) == 3
    assert not is_terminal(game), "two further occurrences is not enough"

    play_sequence(game, PURE_SHUFFLE)
    assert repetition_count(game) == FOURFOLD
    assert int(win_state(game)[2]) == 1, "the third further occurrence draws"


# ---------------------------------------------------------------------------
# RT-05 -- valid_moves() and win_state() agree at every ply
# ---------------------------------------------------------------------------


def test_rt05_empty_mask_and_draw_label_fire_together() -> None:
    """RT-05: no ply where one fires without the other.

    ``valid_moves`` gates on the cheap non-mate terminal check
    (``gungi.py:236``) and returns an all-zero mask, so search sees a terminal
    node; the label then comes from ``win_state``.  If the two disagree on the
    threshold, search and scoring disagree about the game.
    """
    game = load_position_p()

    for ply in range(len(RT01_LINE) + 1):
        mask_empty = int(valid_mask(game).sum()) == 0
        drawn = int(win_state(game)[2]) == 1
        assert mask_empty == drawn, (
            f"ply {ply}: valid_moves all-zero={mask_empty} but "
            f"win_state draw={drawn}"
        )
        if ply < len(RT01_LINE):
            assert not mask_empty, f"ply {ply} should still be playable"
            play_action_checked(game, RT01_LINE[ply], ply + 1)

    assert int(valid_mask(game).sum()) == 0
    assert int(win_state(game)[2]) == 1


# ---------------------------------------------------------------------------
# RT-07 -- Scenario A, rule level (red on v1)
# ---------------------------------------------------------------------------


def test_rt07_scenario_a_is_not_a_draw_at_ply_15() -> None:
    """RT-07: no true game state has occurred four times, so no draw.

    v1 sees ``key(P)`` at history indices 0, 5, 10 and 15 and declares a draw
    while the two states behind those four entries have each occurred exactly
    twice (F3).  Under the repaired key they are two keys of two occurrences
    each, and the game goes on.
    """
    game = load_position_p()
    baseline = cells_bytes(game)

    play_sequence(game, SCENARIO_A_15PLY)

    assert cells_bytes(game) == baseline, "cells are back at P for the 4th time"
    assert board_of(game).turn == BLACK_COLOR
    assert repetition_count(game) == 2, (
        "(P, Black to move) has occurred at plies 5 and 15 -- twice, not four "
        "times"
    )
    assert not is_terminal(game), "FAILS on v1, which draws here"
    assert int(valid_mask(game).sum()) > 0, "and the position is still playable"


def test_rt07_both_scenario_a_states_are_counted_separately() -> None:
    """Each of the two aliased states carries its own, honest count."""
    game = load_position_p()

    play_sequence(game, SCENARIO_A_15PLY[:5])
    assert repetition_count(game) == 1, "(P, Black) -- first occurrence"

    play_sequence(game, SCENARIO_A_15PLY[5:10])
    assert repetition_count(game) == 2, "(P, White) -- second occurrence"

    play_sequence(game, SCENARIO_A_15PLY[10:])
    assert repetition_count(game) == 2, "(P, Black) -- second occurrence"


# ---------------------------------------------------------------------------
# RT-09 -- Scenario B, rule level (red on v1)
# ---------------------------------------------------------------------------


def test_rt09_scenario_b_is_not_a_draw_after_both_loops_and_a_shuffle() -> None:
    """RT-09: repetition cannot be bought with irreversible progress.

    The three v1 "occurrences" of ``key(Q)`` span two soldiers spent from
    White's hand and two captures -- permanent, winning progress for Black.
    v1 lets a single genuine 4-ply shuffle cash them into a draw (F4); the
    repaired key counts the genuine shuffle as what it is, occurrence 2.
    """
    game = load_position_q()
    baseline = cells_bytes(game)

    play_sequence(game, RT09_LINE)

    assert cells_bytes(game) == baseline
    assert board_of(game).turn == WHITE_COLOR
    assert repetition_count(game) == 2, (
        "only the post-loop-2 position and the post-shuffle position share a "
        "key: the two earlier ones had 兵 x2 and 兵 x1 in hand"
    )
    assert not is_terminal(game), "FAILS on v1, which draws here"


def test_rt09_the_loop_positions_never_share_a_key() -> None:
    """Each loop ends in a state that can never recur -- hands only decrease."""
    game = load_position_q()

    for loop in (1, 2):
        play_sequence(game, SCENARIO_B_LOOP)
        assert repetition_count(game) == 1, (
            f"end of loop {loop}: a state seen for the first and only time"
        )
        assert not is_terminal(game)


# ---------------------------------------------------------------------------
# RT-19 -- F8 precedence pin
# ---------------------------------------------------------------------------


def test_rt19_checkmate_that_is_also_a_fourth_occurrence_is_a_win() -> None:
    """RT-19: pin the env's answer to U10 so changing it is deliberate.

    ``win_state`` tests checkmate first (``gungi.py:271``) while the gate it
    sits behind, ``is_game_over``, tests fourfold first
    (``cyengine.pyx:697-710``).  The env therefore scores such a position 1-0.
    gungi.js declines to assign a result at all, so nothing contradicts that
    yet -- but when U10 is answered, flipping it must be a one-line edit with
    this test failing, not silent drift.

    CONSTRUCTION NOTE.  Under per-ply adjudication this position is not
    reachable by play: the position after the mating move is the same position
    as its three earlier occurrences, so those occurrences were *also*
    checkmate, and the game would have ended at the first one.  It is reached
    the only way it can be -- a ``Game`` handed a history it did not play, the
    bridge shape from ``gungi.py:200-208``.  The mate itself is not synthetic:
    ``test_fixtures_sanity`` verifies it against the live engine.
    """
    game = game_with_repeated_history(MATE_AND_REPETITION_FEN, FOURFOLD)
    board = board_of(game)

    assert repetition_count(game) == FOURFOLD, "precondition: 4th occurrence"
    assert cyengine.is_checkmate(board), "precondition: and checkmate"
    assert board.turn == BLACK_COLOR

    result = win_state(game)
    mover = 1 - board.turn
    assert mover == WHITE_COLOR
    assert int(result[mover]) == 1, (
        "checkmate wins for the mover even when it completes the 4th "
        "occurrence (F8 / Q10 -- pinned, not endorsed)"
    )
    assert int(result[2]) == 0, "and specifically is NOT scored a draw"
    assert int(result.sum()) == 1


def test_rt19_the_same_position_without_the_repetition_still_wins() -> None:
    """The control: the pin above is about precedence, not about the mate."""
    game = game_with_repeated_history(MATE_AND_REPETITION_FEN, 1)

    assert repetition_count(game) == 1
    assert int(win_state(game)[WHITE_COLOR]) == 1
