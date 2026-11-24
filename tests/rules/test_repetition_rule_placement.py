"""KOM-141: the same two lines, refereed under BOTH repetition rules.

``RepetitionRule.PLACEMENT`` is gungi.js@3.4.11's key -- FEN field 1, the cells
and nothing else -- and it exists so a referee can apply the site's rule and the
Elo cost of the mismatch can be measured. It re-introduces the rules-audit
aliasing **by construction**, and that is exactly what these tests pin: the two
scenarios the audit built to exhibit the defect (F1-F4) draw under PLACEMENT and
do not draw under FULL, in the same test, so the contrast is one thing to read
rather than two files to compare.

The FULL halves are already pinned next door by ``test_repetition_rule.py``
RT-07 and RT-09; they are re-asserted here deliberately. A contrast test whose
control lives in another module silently stops being a contrast the day the
control moves.

What is NOT here, on purpose: any claim that PLACEMENT is a better rule, and any
default. Every game below names its rule with an explicit keyword;
``tests/engine/test_repetition_rule_default.py`` is what guarantees nothing else
in the repo reaches this key without doing the same.
"""

from __future__ import annotations

import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

from komugi.engine.game import (  # noqa: E402
    TERMINATION_FOURFOLD_REPETITION,
    Game,
)
from komugi.engine.repetition import (  # noqa: E402
    FULL_KEY_BYTES,
    PLACEMENT_KEY_BYTES,
    RepetitionRule,
)

from fixtures import (  # noqa: E402
    BLACK_COLOR,
    FOURFOLD,
    PURE_SHUFFLE,
    SCENARIO_A_15PLY,
    SCENARIO_B_LOOP,
    WHITE_COLOR,
    board_of,
    cells_bytes,
    is_terminal,
    load_position_p,
    load_position_q,
    play_action_checked,
    repetition_count,
    valid_mask,
    win_state,
)

#: RT-09's line, reused: both Scenario B loops, then the genuine shuffle the
#: audit says "cashes" the banked cells-only occurrences.
SCENARIO_B_LINE: tuple[int, ...] = SCENARIO_B_LOOP * 2 + PURE_SHUFFLE

#: The ply at which Scenario A's cells return to P for the fourth time.
SCENARIO_A_DRAW_PLY: int = 15

#: The same, for Scenario B (two 8-ply loops, then the 4-ply shuffle).
SCENARIO_B_DRAW_PLY: int = 20

#: RT-01's line: a genuine 4-ply cycle three times over. The two rules agree
#: about this one, which is the whole reason it is here.
GENUINE_LINE: tuple[int, ...] = PURE_SHUFFLE * 3


def _play_watching_for_the_draw(game: Game, line: tuple[int, ...]) -> int:
    """Play *line* into *game*; return the 1-based ply the draw first fires at.

    Asserts at every ply that the mask and the win state agree -- an empty
    legal mask iff the draw slot is set -- so a rule that ended the game in one
    of the two places and not the other is a failure here rather than a
    divergence between search and scoring later. Returns ``0`` if the game is
    still live after the last ply.
    """
    drew_at = 0
    for ply, action in enumerate(line, start=1):
        assert not is_terminal(game), f"the game ended before ply {ply}"
        play_action_checked(game, action, ply)
        mask_empty = int(valid_mask(game).sum()) == 0
        drawn = int(win_state(game)[2]) == 1
        assert mask_empty == drawn, (
            f"ply {ply}: legal mask empty={mask_empty} but draw={drawn}"
        )
        if drawn and drew_at == 0:
            drew_at = ply
    return drew_at


# ---------------------------------------------------------------------------
# (i) Scenario A -- the side-to-move alias
# ---------------------------------------------------------------------------


def test_scenario_a_draws_under_placement_and_not_under_full() -> None:
    """The audit's Scenario A, refereed both ways, in one place.

    ``w = 3, b = 2`` returns the cells to P with the side to move FLIPPED, so
    the cells-only key sees ``key(P)`` at plies 0, 5, 10 and 15 while the two
    game states behind those four entries have each occurred exactly twice
    (F3). Under PLACEMENT that is a fourfold draw -- gungi.js's answer, and the
    reason this rule is reachable at all. Under FULL it is two keys of two
    occurrences each and the game goes on (RT-07).

    The draw is located at a PLY, not merely at the end of the line: nothing
    before ply 15 may be terminal, or the aliasing would be firing somewhere
    the audit did not claim it does.
    """
    placement = load_position_p(RepetitionRule.PLACEMENT)
    baseline = cells_bytes(placement)

    drew_at = _play_watching_for_the_draw(placement, SCENARIO_A_15PLY)

    assert drew_at == SCENARIO_A_DRAW_PLY, (
        "the cells-only key reaches its fourth occurrence exactly when the "
        "cells return to P for the fourth time"
    )
    assert cells_bytes(placement) == baseline
    assert board_of(placement).turn == BLACK_COLOR
    assert repetition_count(placement) == FOURFOLD
    assert int(win_state(placement)[2]) == 1, "draw slot set at ply 15"
    assert int(win_state(placement)[WHITE_COLOR]) == 0
    assert int(win_state(placement)[BLACK_COLOR]) == 0
    assert int(valid_mask(placement).sum()) == 0, "and the mask is empty"

    # The contrast, in the same test: the identical line under Komugi's key.
    full = load_position_p()
    full_drew_at = _play_watching_for_the_draw(full, SCENARIO_A_15PLY)

    assert full_drew_at == 0, "RT-07: no true game state has occurred four times"
    assert cells_bytes(full) == baseline, "the cells really did come back"
    assert repetition_count(full) == 2, "(P, Black to move), twice"
    assert not is_terminal(full)
    assert int(valid_mask(full).sum()) > 0, "and the position is still playable"


def test_scenario_a_under_placement_reports_fourfold_repetition() -> None:
    """(v) The reason, not just the slot: the census must name this correctly."""
    game = load_position_p(RepetitionRule.PLACEMENT)
    _play_watching_for_the_draw(game, SCENARIO_A_15PLY)

    assert game.termination_reason() == TERMINATION_FOURFOLD_REPETITION


# ---------------------------------------------------------------------------
# (ii) Scenario B -- the hand-contents alias
# ---------------------------------------------------------------------------


def test_scenario_b_draws_under_placement_and_not_under_full() -> None:
    """The audit's Scenario B, refereed both ways.

    The three cells-only "occurrences" of ``Q`` span two soldiers spent from
    White's hand and two captures -- permanent, winning progress for Black --
    and a single genuine 4-ply shuffle then cashes them into a draw (F4). The
    cells cannot see any of it, so PLACEMENT draws at ply 20. FULL counts the
    genuine shuffle as what it is, occurrence 2, and plays on (RT-09).
    """
    placement = load_position_q(RepetitionRule.PLACEMENT)
    baseline = cells_bytes(placement)

    drew_at = _play_watching_for_the_draw(placement, SCENARIO_B_LINE)

    assert drew_at == SCENARIO_B_DRAW_PLY
    assert cells_bytes(placement) == baseline
    assert board_of(placement).turn == WHITE_COLOR
    assert repetition_count(placement) == FOURFOLD
    assert int(win_state(placement)[2]) == 1
    assert placement.termination_reason() == TERMINATION_FOURFOLD_REPETITION

    full = load_position_q()
    full_drew_at = _play_watching_for_the_draw(full, SCENARIO_B_LINE)

    assert full_drew_at == 0, "RT-09: repetition cannot be bought with progress"
    assert repetition_count(full) == 2
    assert not is_terminal(full)


# ---------------------------------------------------------------------------
# (iii) The genuine cycle -- where the two rules must AGREE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [RepetitionRule.FULL, RepetitionRule.PLACEMENT],
    ids=["full", "placement"],
)
def test_a_genuine_cycle_draws_at_the_fourth_occurrence_under_either_rule(
    rule: RepetitionRule,
) -> None:
    """A true repetition is a true repetition under both keys.

    ``PURE_SHUFFLE`` changes no hand, captures nothing and returns the side to
    move, so the two keys agree that the position really is coming round again.
    This is the node that stops PLACEMENT from being tested only where it is
    wrong: a rule that never fired would pass every test above and fail here,
    at the same ply as FULL, for the same reason.
    """
    game = load_position_p(rule)

    assert game.repetition_rule is rule
    assert repetition_count(game) == 1, "the start position is occurrence 1"

    drew_at = _play_watching_for_the_draw(game, GENUINE_LINE)

    assert drew_at == len(GENUINE_LINE), "ply 12 -- the base position's 4th"
    assert repetition_count(game) == FOURFOLD
    assert game.termination_reason() == TERMINATION_FOURFOLD_REPETITION


# ---------------------------------------------------------------------------
# (iv) The key a game of each rule actually stores
# ---------------------------------------------------------------------------


def test_a_placement_game_stores_243_byte_cells_keys() -> None:
    """Every history entry is the cells, and the last one is the board's.

    The length alone would be satisfied by a 243-byte blob of anything, so the
    final entry is compared against ``cells_arr`` read independently off the
    board -- the same bytes ``cyengine.placement_key`` claims to return.
    """
    game = load_position_p(RepetitionRule.PLACEMENT)
    for ply, action in enumerate(SCENARIO_A_15PLY[:10], start=1):
        play_action_checked(game, action, ply)

    history: list[bytes] = list(game._history)
    assert len(history) == 11, "the seed key plus one per ply"
    assert all(len(entry) == PLACEMENT_KEY_BYTES for entry in history)
    assert history[-1] == cells_bytes(game)
    assert game._key == cells_bytes(game)


def test_a_full_game_stores_274_byte_repaired_keys() -> None:
    """The control: the default rule is unchanged, byte length included."""
    game = load_position_p()
    for ply, action in enumerate(SCENARIO_A_15PLY[:10], start=1):
        play_action_checked(game, action, ply)

    history: list[bytes] = list(game._history)
    assert len(history) == 11
    assert all(len(entry) == FULL_KEY_BYTES for entry in history)
    assert history[-1][:PLACEMENT_KEY_BYTES] == cells_bytes(game), (
        "the full key's leading block is the placement key, byte for byte"
    )


def test_the_two_rules_disagree_about_this_position_and_that_is_the_point() -> None:
    """One position, two keys: same leading block, different lengths.

    Read straight off the engine rather than through a Game, so the identity
    the whole design rests on -- ``repetition_key(b)[:243] == placement_key(b)``
    -- is asserted where both builders can be seen at once.
    """
    board = board_of(load_position_p())
    full_key: bytes = cyengine.repetition_key(board)
    placement_key: bytes = cyengine.placement_key(board)

    assert len(full_key) == FULL_KEY_BYTES
    assert len(placement_key) == PLACEMENT_KEY_BYTES
    assert full_key[:PLACEMENT_KEY_BYTES] == placement_key
    assert full_key != placement_key
