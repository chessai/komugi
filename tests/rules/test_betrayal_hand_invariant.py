"""KOM-170: an arata-betray may never drive a hand count negative.

THE DEFECT.  ``cyengine.apply_move`` pays for an arata-betray twice out of the
same hand: once for the piece being dropped (``hand[piece_ti] -= 1``) and once
per converted enemy (``hand[cap_ti] -= 1``, gungi.js ``updateHand`` semantics,
docs/design/betrayal-encoding.md:153-167).  The legality test in
``cymovegen._get_betrayal_combos`` / ``_qualifying_tier_mask`` asked only
``hand[eti] >= ecount`` -- the hand as it stands BEFORE the drop -- so when the
dropped type IS a converted type the two debits hit one slot and the count
underflows.  With exactly one 謀 per side in the advanced piece set
(``ADVANCED_POSITION``'s ``T1``), that is *every* drop of a 謀 onto a tower
whose tier-1 enemy is a 謀: ``1 - 1 - 1 = -1``.  A self-play worker died on it
inside ``cyengine.repetition_key`` ("hand counts must fit in uint8 ... min=-1",
campaign 5b iteration 0).  The fix requires one extra copy in hand when the
dropped type is converted; with one 謀 per side that makes the conversion of an
enemy 謀 by a dropped 謀 permanently unaffordable, which is the intent.

WHAT IS RED ON MAIN.  ``test_arata_betray_of_the_same_type_needs_one_more_in_hand``
fails on main at its first assertion: ``mask[drop_action(TACTICIAN, (5, 5),
bsel=1)] == 0``.  Main sets that bit, and playing it drives
``hand_arr[謀, white]`` to -1 -- so the same test's per-action clone sweep is
red on main too, there by a ``ValueError`` out of ``repetition_key`` rather
than by an assertion.  Both are green after the cymovegen fix.

PARITY WITH gungi.js: THIS IS A DELIBERATE DIVERGENCE, NOT A PORTING ERROR.
The reference has the same legality defect.  ``getBetrayalCombos``
(``move_gen.ts:402-423``, quoted at docs/design/betrayal-encoding.md:178-192)
filters types on ``p.count >= count`` against the hand BEFORE the placement is
deducted, exactly as our port did, so the reference offers the same move.  It
does not underflow only because its hand is a LIST of ``{type, count}`` entries
whose entry is DELETED when the count reaches zero (``utils.update_hand``): the
placement removes the 謀 entry and the conversion's decrement then finds no
entry and silently does nothing.  The reference therefore lets a dropped 謀
convert an enemy 謀 for free -- a payment it never collects -- and
``tests/engine/goldens/playouts_betray.json`` records it doing precisely that
(``新謀(9-9-3)返謀``, whose ``fen_after`` shows black's hand with the 謀 entry
simply gone).  Komugi's hand is a dense ``int8`` array, where the same move is
a count of -1, and a negative hand is not a position.  So: **divergence** --
komugi refuses a move gungi.js offers and plays, whenever the dropped type
equals a converted type and the hand holds exactly the count.  **Parity** is
otherwise untouched: no other move of any kind changes, because the extra
requirement is inert unless ``eti == piece_ti`` on an arata.  For the rules
audit's U-list, the open question the reference leaves is which of the two the
site intends -- an unpayable conversion granted for free, or an illegal move.

GOLDEN SAFETY.  The four full-game corpora (``playouts_intro`` /
``_beginner`` / ``_intermediate`` / ``_advanced``) contain no ``返謀`` at all,
in a ``chosen`` or in a ``legal`` list, so every replay through cymovegen is
bit-identical.  ``playouts_betray.json`` holds exactly one such ply -- the
``新謀(9-9-3)返謀`` above -- and it is *chosen*, not merely legal.  The corpus
is NOT edited.  Two tests that replayed it through the Cython engine are
updated in place to state the refusal instead, each pointing back here:
``tests/rules/test_last_move_planes.py::test_a_betray_ply_decodes_as_a_board_move``
and
``tests/engine/test_cymovegen.py::TestDraftShuCython::test_last_piece_draft_only_shu_variants``.
The pure-Python oracle (``komugi.engine.move_gen``) is deliberately NOT changed
-- it is the v1 reference transcription, and ``tests/engine/test_move_gen.py``
still asserts the reference behaviour against the same golden.

RUNTIME.  The two playout tests run 150 seeded games each, capped at 40
(advanced) and 50 (intro) plies, which is what keeps each of them inside the
~60 s budget: an advanced position after the draft generates on the order of a
thousand move records, and ``valid_moves`` + ``play_action`` walk that list
twice per ply.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

cyengine = pytest.importorskip("komugi.engine.cyengine")

from komugi.engine.action_encoding import PIECE_INDEX  # noqa: E402
from komugi.engine.fen import ADVANCED_POSITION, INTRO_POSITION  # noqa: E402
from komugi.engine.game import Game  # noqa: E402
from komugi.engine.utils import TACTICIAN  # noqa: E402

from fixtures import (  # noqa: E402
    WHITE_COLOR,
    board_of,
    drop_action,
    hand_count,
    is_terminal,
    load_game,
    repetition_key,
    valid_mask,
)

#: ``hand_arr`` row of the 謀 (tactician) -- the only piece that ever converts.
TACTICIAN_INDEX: int = PIECE_INDEX[TACTICIAN]

#: The tower the drop lands on, 1-based ``(rank, file)``.
TOWER: tuple[int, int] = (5, 5)

#: The reproducing position, advanced mode, draft over, White to move.
#:
#: * ``5-5`` is a two-piece tower: a BLACK 謀 on tier 1 with a WHITE 兵 on top
#:   of it, so the square is a legal drop target (friendly on top, height 2 <
#:   max_tier 3) whose only convertible enemy is the 謀 underneath.
#: * White's hand is a single 謀 -- the count ``ADVANCED_POSITION`` actually
#:   deals (``T1``), and exactly the count that makes ``hand >= ecount`` true
#:   before the drop and false after it.
#: * Both 帥 are on the board, White's on 9-5, so the position is neither
#:   terminal nor insufficient material, and rank 5 is inside White's drop
#:   range (the rank walk runs 9 -> 1 and stops after the last rank with a
#:   friendly top piece).
#: FEN ranks are written file 9 -> file 1 (``fen.parse_fen``: ``9 - len(row)``).
SAME_TYPE_DROP_FEN: str = "4m4/9/9/9/4|t:D|4/9/9/9/4M4 T1/- w 3 - 1"

#: The same position with a SECOND 謀 in White's hand.  The conversion is
#: affordable there -- one 謀 for the placement, one for the flip -- so it is
#: the control that keeps the test above from passing for the trivial reason
#: that arata conversions were disabled outright.
TWO_IN_HAND_FEN: str = "4m4/9/9/9/4|t:D|4/9/9/9/4M4 T2/- w 3 - 1"

#: Games and ply cap per fuzz test; see the module docstring's RUNTIME note.
PLAYOUT_GAMES: int = 150
ADVANCED_PLY_CAP: int = 40
INTRO_PLY_CAP: int = 50

#: Seed base, so the two fuzz tests do not draw the same stream.
SEED_BASE: int = 170_000


def _hand_min(game: Game) -> int:
    """The smallest count in *game*'s hand -- the quantity that goes to -1."""
    hand: npt.NDArray[np.int64] = np.asarray(
        board_of(game).hand_arr, dtype=np.int64
    )
    return int(hand.min())


def _legal_actions(game: Game) -> npt.NDArray[np.int64]:
    """Every action id *game* offers, as a flat array."""
    return np.asarray(np.nonzero(valid_mask(game))[0], dtype=np.int64)


def _play_every_legal_action_on_a_clone(game: Game) -> None:
    """Play each legal action on its own clone; the hand may never go negative.

    The whole-mask form of the invariant: it does not matter which action is
    the offender, only that no offered action can produce one.  ``play_action``
    itself rebuilds the repetition key, so an underflow surfaces here as the
    ``ValueError`` the self-play worker died on; the explicit key call and the
    ``>= 0`` assertion make the failure legible either way.
    """
    for action in _legal_actions(game):
        clone = game.clone()
        clone.play_action(int(action))
        assert _hand_min(clone) >= 0, (
            f"action {int(action)} drove a hand count to {_hand_min(clone)}; "
            "an arata-betray must pay for the dropped piece AND the "
            "conversion out of a hand that holds both (KOM-170)"
        )
        repetition_key(board_of(clone))


def test_arata_betray_of_the_same_type_needs_one_more_in_hand() -> None:
    """Dropping the last 謀 onto an enemy 謀 is refused, not offered at -1.

    RED ON MAIN at the first assertion: main's ``_qualifying_tier_mask`` tests
    ``hand[謀] >= 1`` against the hand BEFORE the placement, sets the block-B
    ``bsel = 1`` bit, and playing that action leaves ``hand[謀, white] == -1``
    -- which ``repetition_key`` then refuses to encode.
    """
    game = load_game(SAME_TYPE_DROP_FEN)

    assert hand_count(game, TACTICIAN_INDEX, WHITE_COLOR) == 1, (
        "the fixture is only about the boundary while the hand holds exactly "
        "the count the tower demands"
    )
    assert not is_terminal(game), "the fixture position must still be playable"

    converting = drop_action(TACTICIAN, TOWER, bsel=1)
    plain = drop_action(TACTICIAN, TOWER, bsel=0)
    mask = valid_mask(game)

    assert mask[converting] == 0, (
        "the drop that converts the enemy 謀 costs two 謀 from a hand holding "
        "one; it must not be offered (KOM-170)"
    )
    assert mask[plain] == 1, (
        "the plain drop onto the same tower is unaffected -- the fix removes "
        "the conversion, not the placement"
    )

    _play_every_legal_action_on_a_clone(game)


def test_a_second_tactician_in_hand_makes_the_same_conversion_legal() -> None:
    """The control: with 謀 x2 the same conversion is affordable and offered.

    Without this, the test above would pass just as well against an engine that
    had deleted arata conversions altogether.  Here the placement spends one
    謀 and the flip the other, landing on exactly zero.
    """
    game = load_game(TWO_IN_HAND_FEN)
    assert hand_count(game, TACTICIAN_INDEX, WHITE_COLOR) == 2

    converting = drop_action(TACTICIAN, TOWER, bsel=1)
    assert valid_mask(game)[converting] == 1, (
        "hand >= count + 1 is the requirement; two 謀 meet it"
    )

    game.play_action(converting)

    assert hand_count(game, TACTICIAN_INDEX, WHITE_COLOR) == 0, (
        "both 謀 are spent: one placed, one paid for the conversion"
    )
    assert _hand_min(game) >= 0
    repetition_key(board_of(game))


def _seeded_playout_holds_the_invariant(fen: str, seed: int, ply_cap: int) -> int:
    """Play up to *ply_cap* uniformly-random legal plies from *fen*.

    Returns the number of plies actually played (a random game can end early),
    asserting after every one of them that no hand count is negative and that
    the repetition key still builds.
    """
    game = load_game(fen)
    rng = np.random.default_rng(seed)

    plies = 0
    while plies < ply_cap:
        if is_terminal(game):
            break
        legal = _legal_actions(game)
        if legal.size == 0:
            break
        action = int(rng.choice(legal))
        game.play_action(action)
        plies += 1
        assert _hand_min(game) >= 0, (
            f"seed {seed} ply {plies}: hand count {_hand_min(game)} after "
            f"action {action}"
        )
        repetition_key(board_of(game))

    return plies


def test_random_playouts_never_drive_a_hand_negative_advanced() -> None:
    """150 seeded advanced games, 40 plies each: hands stay non-negative.

    Advanced is the mode that deals a 謀 (``ADVANCED_POSITION``'s ``T1``), so
    it is the only mode whose random play can reach the defect at all.  This is
    a guard rather than a reproducer -- the constructed position above is the
    reproducer -- and its non-vacuity is asserted: a run that played almost no
    plies would hold the invariant for free.
    """
    total = 0
    for index in range(PLAYOUT_GAMES):
        total += _seeded_playout_holds_the_invariant(
            ADVANCED_POSITION, SEED_BASE + index, ADVANCED_PLY_CAP
        )

    assert total >= PLAYOUT_GAMES, (
        f"{PLAYOUT_GAMES} advanced playouts produced only {total} plies "
        "between them; the invariant was checked on almost nothing"
    )


def test_random_playouts_never_drive_a_hand_negative_intro() -> None:
    """150 seeded intro games, 50 plies each: the same invariant, other mode.

    Intro deals no 謀 at all (``INTRO_POSITION``'s hands are ``J2N2R2D1``), so
    no conversion of any kind is reachable here.  That is the point: it pins
    that the KOM-170 change costs the conversion-free mode nothing, and it
    exercises the hand arithmetic of ordinary drops and captures.
    """
    total = 0
    for index in range(PLAYOUT_GAMES):
        total += _seeded_playout_holds_the_invariant(
            INTRO_POSITION, SEED_BASE + 500_000 + index, INTRO_PLY_CAP
        )

    assert total >= PLAYOUT_GAMES, (
        f"{PLAYOUT_GAMES} intro playouts produced only {total} plies between "
        "them; the invariant was checked on almost nothing"
    )
