"""The DRAFTMIN root view: 終 withheld at a root until N placements (KOM-156).

``DESIGN.html`` pre-registers a draft-mask curriculum as a contingency, and
campaign 2 is the contingency arriving: from an empty advanced-mode board the
median side placed four pieces and ended its draft, which is v1's draft
attractor reproduced. v1's mitigation was ``draftMinPlacements`` -- a mask that
withholds the draft-finish action from the side to move until it has placed N
pieces -- and the tooling for it already lives in the engine
(:func:`komugi.engine.game.draft_root_mask`). This module is about the wiring
that finally calls it: :attr:`komugi.config.SelfPlayConfig.min_placements` and
the two keyword arguments :class:`komugi.selfplay.adapter.GameState` grew.

Every claim here is about a **view**, never a rule. The engine's own
``valid_moves`` and ``win_state`` answer exactly as they always did; what
changes is the legal set the search is shown *at a root*, which is the only
place a move is chosen and a policy target is recorded.

Advanced mode throughout, because it is the mode that has a draft at all: its
start position is an empty board with both hands full and both draft bits set.
Intro -- the process default, and what ``GUNGI_MODE`` leaves ``Game`` at unless
a test says otherwise -- starts from a laid-out board past the draft, so every
assertion below would be vacuous there. That vacuity is itself an acceptance
condition (the feature must no-op in intro), so it gets its own node.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import SHU_BLOCK
from komugi.engine.game import Game, draft_placements, draft_root_mask
from komugi.selfplay.adapter import GameState

_C_START: Final[int] = SHU_BLOCK[0]
"""First action id of block C -- "arata + draft-finish", the 終 block."""

_THRESHOLD: Final[int] = 10
"""v1's first curriculum rung (``10 -> 7 -> 4``), used as the test threshold."""


class _AdvancedGame(Game):
    """The real engine started in **advanced** mode.

    Set as a class attribute rather than through ``Game.set_start_mode``:
    ``Game.__init__`` reads ``type(self)._START_MODE``, so overriding it here
    changes this class and nothing else, while the setter would mutate the
    shared base for the rest of the process and leave this module owning a
    global nobody restores. The same idiom ``tests/selfplay/unit/
    test_staged_targets.py`` uses for the same reason.
    """

    _START_MODE: str = "advanced"


class _IntroGame(Game):
    """The real engine started in **intro** mode -- a board past the draft."""

    _START_MODE: str = "intro"


def _unmasked(game: Game) -> npt.NDArray[np.bool_]:
    """Return the engine's own legality mask, as bool, with no view applied."""
    return np.asarray(game.valid_moves(), dtype=np.bool_)


def _drafted(placements: int) -> Game:
    """Return an advanced-mode game in which BOTH sides have placed `placements`.

    Placements are made by playing the first legal **block A/B** id at every
    ply -- every id below ``C_START`` -- which is a placement that *keeps the
    draft open*. Block C is avoided precisely because playing it would end a
    draft, and a finished draft is the state this feature exists to postpone.

    Both sides advance because neither ever plays 終: the draft alternates, so
    ``2 * placements`` plies leave white to move with `placements` pieces down
    and black with the same. Which square each piece lands on is irrelevant to
    every claim here -- only the count is -- so "first legal" is a fixed choice
    rather than a meaningful one.
    """
    game = _AdvancedGame()
    for _ in range(2 * placements):
        legal = np.flatnonzero(_unmasked(game)[:_C_START])
        assert legal.size > 0, (
            "no draft-continuing placement is legal: this fixture can no longer "
            "reach the position the test is about"
        )
        game.play_action(int(legal[0]))
    return game


def _placements(game: Game) -> int:
    """Return how many pieces the side to move has placed in its draft."""
    return draft_placements(game._board, int(game.player))  # noqa: SLF001


# --------------------------------------------------------------------------
# The config knob
# --------------------------------------------------------------------------


def test_min_placements_defaults_to_off() -> None:
    """The knob is off unless a run asks for it -- DESIGN's "off by default"."""
    assert SelfPlayConfig(workers=1, process_batch_size=1).min_placements == 0


@pytest.mark.parametrize("value", [0, 1, 4, 7, 10, 25])
def test_min_placements_accepts_the_curriculum_rungs(value: int) -> None:
    """Every rung of v1's ``10 -> 7 -> 4`` curriculum is a legal setting."""
    config = SelfPlayConfig(
        workers=1, process_batch_size=1, min_placements=value
    )
    assert config.min_placements == value


@pytest.mark.parametrize("value", [-1, 26, 100])
def test_min_placements_outside_the_bound_is_refused(value: int) -> None:
    """A negative threshold is meaningless and one above a hand is unreachable.

    Refused at construction, where the operator is still reading the error,
    rather than discovered as a generation that masked nothing (a threshold
    above the hand is never met, so every root falls through to the mask's
    last-piece escape hatch) or one that masked with a negative number.
    """
    with pytest.raises(ValueError, match="min_placements"):
        SelfPlayConfig(workers=1, process_batch_size=1, min_placements=value)


# --------------------------------------------------------------------------
# The root view
# --------------------------------------------------------------------------


def test_a_root_below_the_threshold_withholds_every_shu_id() -> None:
    """#(i): at 3 placements under a threshold of 10, no 終 id is legal.

    And the rest of the legal set is untouched: the view withholds one block,
    it does not re-adjudicate the position. The unmasked mask is asserted to
    *contain* a 終 id first, so the node cannot pass by there being nothing to
    withhold.
    """
    game = _drafted(3)
    assert _placements(game) == 3
    raw = _unmasked(game)
    assert raw[_C_START:].any(), "fixture is vacuous: no 終 id is legal here"

    state = GameState(game, 0.0, min_placements=_THRESHOLD, is_root=True)
    masked = state.legal_mask()

    assert not masked[_C_START:].any()
    assert np.array_equal(masked[:_C_START], raw[:_C_START])


def test_a_root_at_the_threshold_is_the_engines_own_mask() -> None:
    """#(ii): once the mover has placed N, the root mask is the true one again.

    The view is a *delay*, not a ban: the whole curriculum is the threshold
    coming down, and at or above it the position the search sees is exactly the
    position the engine describes.
    """
    game = _drafted(_THRESHOLD)
    assert _placements(game) == _THRESHOLD
    raw = _unmasked(game)
    assert raw[_C_START:].any(), "fixture is vacuous: no 終 id is legal here"

    state = GameState(game, 0.0, min_placements=_THRESHOLD, is_root=True)
    assert np.array_equal(state.legal_mask(), raw)


def test_the_view_dies_on_the_first_move_played() -> None:
    """#(iii): an interior position sees the engine's true legal set.

    The seam is :meth:`GameState.play`, and it is the seam because of how both
    searches build an interior position: ``clone()`` and then ``play(edge)``
    (``gumbel.py:653/781``, ``match/tree.py:346-356``). The clone carries the
    root view -- the root NODE is built from a clone, and a clone that dropped
    it would leave the root node's legal set unmasked while the candidate draw
    used the masked one -- and the first edge played takes it away.

    Asserted on a clone, so the state the caller handed in is not the one
    inspected, exactly as in a search.
    """
    game = _drafted(3)
    root = GameState(game, 0.0, min_placements=_THRESHOLD, is_root=True)

    child = root.clone()
    assert child.is_root, "a clone is what the search builds its root node from"
    assert child.min_placements == (_THRESHOLD, _THRESHOLD)
    assert not child.legal_mask()[_C_START:].any()

    child.play(int(np.flatnonzero(child.legal_mask())[0]))
    assert not child.is_root
    assert np.array_equal(child.legal_mask(), _unmasked(child.game))
    assert child.legal_mask()[_C_START:].any(), (
        "the interior position must see the 終 ids the root withheld"
    )

    # And the root the clone came from is untouched by any of it.
    assert root.is_root
    assert not root.legal_mask()[_C_START:].any()

    # The same seam under a PAIR of thresholds (KOM-246): the clone carries
    # both entries, not the mover's own, because the position it is cloned
    # from is searched for both colours below it.
    pair_root = GameState(_drafted(3), 0.0, min_placements=(4, 16), is_root=True)
    pair_child = pair_root.clone()
    assert pair_child.min_placements == (4, 16)
    assert pair_child.is_root
    pair_child.play(int(np.flatnonzero(pair_child.legal_mask())[0]))
    assert not pair_child.is_root
    assert pair_child.min_placements == (4, 16), (
        "the thresholds survive the move; it is the ROOT-ness that dies"
    )
    assert np.array_equal(pair_child.legal_mask(), _unmasked(pair_child.game))


def test_the_threshold_off_returns_the_engines_own_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#(iv): a zero threshold is today's path, object for object.

    ``draft_root_mask`` hands its input straight back at a zero threshold, and
    the adapter does not even call it -- so a run that never sets the knob
    allocates nothing, copies nothing and compares equal to the engine's answer
    element for element.

    Extended for KOM-246's pair: ``(0, 18)`` is "off for white, on for black",
    and the per-MOVER guard (design review F6) is what makes white's root take
    the untouched path while black's, one ply later, is masked. A guard that
    still asked "is any threshold positive" would route white through
    ``draft_root_mask`` -- harmless here, but no longer the line it was -- so
    the helper is replaced by one that refuses to be called.
    """
    game = _drafted(3)
    raw = _unmasked(game)
    for is_root in (False, True):
        state = GameState(game, 0.0, min_placements=0, is_root=is_root)
        assert np.array_equal(state.legal_mask(), raw)

    assert draft_root_mask(game._board, raw, 0) is raw  # noqa: SLF001

    paired = _drafted(3)
    white_root = GameState(paired, 0.0, min_placements=(0, 18), is_root=True)
    assert int(paired.player) == 0, "white must be to move at this fixture"

    def _never(*args: object, **kwargs: object) -> npt.NDArray[np.bool_]:
        raise AssertionError("white's zero threshold reached draft_root_mask")

    monkeypatch.setattr("komugi.selfplay.adapter.draft_root_mask", _never)
    white_mask = white_root.legal_mask()
    monkeypatch.undo()
    assert np.array_equal(white_mask, _unmasked(paired))
    assert white_mask[_C_START:].any(), "white's 終 ids are not withheld at all"

    white_root.play(
        int(np.flatnonzero(white_mask[:_C_START])[0]), keep_root=True
    )
    assert int(paired.player) == 1, "black must be to move after white's ply"
    assert not white_root.legal_mask()[_C_START:].any(), (
        "black has placed 3 of 18: its own entry of the pair must withhold 終"
    )


# --------------------------------------------------------------------------
# The per-colour pair (KOM-246)
# --------------------------------------------------------------------------


def test_a_pair_masks_each_side_by_its_own_threshold() -> None:
    """Each colour is held to ITS OWN entry of ``(N_w, N_b)``.

    The fixture has both sides on four placements, so one position exercises
    both answers: at ``(4, 16)`` white has met its threshold and black has not,
    and at ``(16, 4)`` the verdicts swap. That is the whole of KOM-246's mask
    change -- the index is the side to move, not the game.
    """
    game = _drafted(4)
    assert int(game.player) == 0, "white must be to move at this fixture"
    raw = _unmasked(game)
    assert raw[_C_START:].any(), "fixture is vacuous: no 終 id is legal here"

    lenient_for_white = GameState(game, 0.0, min_placements=(4, 16), is_root=True)
    assert np.array_equal(lenient_for_white.legal_mask(), raw), (
        "white has placed 4 of 4: its root is the engine's own answer"
    )

    lenient_for_white.play(
        int(np.flatnonzero(raw[:_C_START])[0]), keep_root=True
    )
    assert int(game.player) == 1, "black must be to move after white's placement"
    assert not lenient_for_white.legal_mask()[_C_START:].any(), (
        "black has placed 4 of 16: every 終 id must be withheld"
    )

    reversed_game = _drafted(4)
    strict_for_white = GameState(
        reversed_game, 0.0, min_placements=(16, 4), is_root=True
    )
    assert not strict_for_white.legal_mask()[_C_START:].any(), (
        "white has placed 4 of 16: every 終 id must be withheld"
    )
    strict_for_white.play(
        int(np.flatnonzero(_unmasked(reversed_game)[:_C_START])[0]),
        keep_root=True,
    )
    assert strict_for_white.legal_mask()[_C_START:].any(), (
        "black has placed 4 of 4: its own 終 ids are not withheld"
    )


def test_blacks_shu_ends_both_drafts_and_the_mask_no_ops_for_white() -> None:
    """F1: black's 終 clears BOTH draft bits, and white's floor cannot stop it.

    The engine's asymmetry, asserted rather than assumed: with ``(16, 4)``
    white is held to sixteen placements and black to four, so black may close
    at four -- and closing ends white's draft too, at one placement. White's
    next root therefore has no draft bit at all, ``draft_root_mask``'s second
    guard fires, and the mask is the engine's own answer however large ``N_w``
    is. A census that read ``N_w`` as a floor on white's placements would be
    reading a number this position falsifies.
    """
    game = _drafted(4)
    root = GameState(game, 0.0, min_placements=(16, 4), is_root=True)
    root.play(int(np.flatnonzero(_unmasked(game)[:_C_START])[0]), keep_root=True)

    black_mask = root.legal_mask()
    shu_ids = np.flatnonzero(black_mask[_C_START:]) + _C_START
    assert shu_ids.size > 0, "black at 4 of 4 must be allowed to close"
    root.play(int(shu_ids[0]), keep_root=True)

    board = game._board  # noqa: SLF001
    assert board.draft_w == 0 and board.draft_b == 0, (
        "black's 終 must end BOTH drafts (cyengine); F1 rests on it"
    )
    assert int(game.player) == 0, "white must be to move after black's close"
    assert np.array_equal(root.legal_mask(), _unmasked(game)), (
        "white's root under N_w = 16 must be the engine's own answer: there is "
        "no draft bit left for the mask to act on"
    )


def test_a_non_root_state_is_never_masked() -> None:
    """A position that is not a root carries the threshold and ignores it.

    The default for ``is_root`` is ``False``, so every state built by code that
    has never heard of this feature -- the eval bridge, the arena, a test --
    keeps the engine's legal set whatever the threshold says.
    """
    game = _drafted(3)
    state = GameState(game, 0.0, min_placements=_THRESHOLD)
    assert not state.is_root
    assert np.array_equal(state.legal_mask(), _unmasked(game))


def test_intro_mode_is_untouched_by_any_threshold() -> None:
    """#(v): with both draft bits clear the view is a no-op at any threshold.

    Intro and beginner start from laid-out boards whose draft is already over,
    so ``draft_root_mask``'s first guard returns the input mask unchanged. A
    run of those modes configured with a threshold plays precisely the game it
    would have played without one.
    """
    game = _IntroGame()
    raw = _unmasked(game)
    state = GameState(game, 0.0, min_placements=_THRESHOLD, is_root=True)
    assert np.array_equal(state.legal_mask(), raw)


def test_the_view_never_reaches_the_engine() -> None:
    """``valid_moves`` and ``win_state`` answer as they always did.

    The masked root is a lens the search looks through; the engine behind it
    is unmoved, which is what keeps every adjudication, every training target
    and every rules claim independent of this feature.
    """
    game = _drafted(3)
    before = _unmasked(game)
    outcome = np.asarray(game.win_state(), dtype=np.uint8)

    state = GameState(game, 0.0, min_placements=_THRESHOLD, is_root=True)
    assert not state.legal_mask()[_C_START:].any()

    assert np.array_equal(_unmasked(game), before)
    assert before[_C_START:].any(), "the engine still calls 終 legal here"
    assert np.array_equal(np.asarray(game.win_state(), dtype=np.uint8), outcome)
    assert np.array_equal(state.absolute_outcome(), outcome)
