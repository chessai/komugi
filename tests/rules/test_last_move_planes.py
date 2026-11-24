"""KOM-002 §7 C -- the last-move planes, 118 (origin) and 119 (destination).

Both planes carry ``tier / 3`` at a single square and 0 everywhere else, so
what has to be pinned is *which* square and *which* tier -- neither of which a
shape check can see.  The four positions below are hand-built for exactly that,
in advanced mode (``max_tier = 3``) with the draft bits clear:

``TIER_TWO_TSUKE_FEN``
    A white tower ``|D:D|`` on 6-5 whose top soldier sits at **tier 2**, and a
    lone black soldier on 5-5.  #8's trap position.

``CAPTURE_OVER_FRIEND_FEN``
    The same white tower, and a MIXED tower ``|D:d|`` on 5-5: white underneath,
    black on top.  Capturing the black top lands the mover *above the friendly
    survivor*, at tier 2 -- not at tier 1, which is what
    ``action_encoding.py``'s module docstring still claims.  #29.

``BETRAYAL_FEN``
    A white 謀 tactician on 6-5, a lone black soldier on 5-6, and one 兵 in
    White's hand (a betrayal converts by paying from hand, so the hand piece is
    what makes the betray legal at all).  The tsuke and the betray to that
    square have the SAME tiers, which is #10's whole point.

``OFF_CENTRE_ROUTE_FEN``
    A white soldier on 6-3 with an empty 5-3 in front of it.  Both squares are
    off the centre file, so the file reflection genuinely moves them.  #11.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.game import Game, _mirror_action

from fixtures import (
    MT_ARATA,
    MT_BETRAY,
    MT_CAPTURE,
    MT_ROUTE,
    MT_TSUKE,
    PLANE_LAST_MOVE_FROM,
    PLANE_LAST_MOVE_TO,
    action_of,
    board_of,
    decoded_action,
    golden_fens,
    load_game,
    load_golden_playouts,
    move_by_san,
    move_matching,
    observation_planes,
    play_action_checked,
    sq_index,
)

TIER_TWO_TSUKE_FEN: Final[str] = "4m4/9/9/9/4d4/4|D:D|4/9/9/4M4 -/- w 3 - 1"
CAPTURE_OVER_FRIEND_FEN: Final[str] = "4m4/9/9/9/4|D:d|4/4|D:D|4/9/9/4M4 -/- w 3 - 1"
BETRAYAL_FEN: Final[str] = "4m4/9/9/9/3d5/4T4/9/9/4M4 D1/- w 3 - 1"
OFF_CENTRE_ROUTE_FEN: Final[str] = "4m3n/9/9/9/9/6D2/9/9/4M4 -/- w 3 - 1"

_TIER_TWO: Final[float] = 2.0 / 3.0
"""``to_tier / 3`` for a tier-2 landing -- the value both #8 and #29 read."""


def _play(game: Game, move: dict[str, Any]) -> int:
    """Play *move* through the legality check, returning its action id."""
    action = action_of(move)
    play_action_checked(game, action, ply=1)
    return action


def _cell(planes: npt.NDArray[np.float32], plane: int, square: int) -> float:
    """Plane *plane*'s value at the 0..80 square index *square*.

    ``s = r * 9 + (8 - f)`` is BOTH the action encoding's square index and the
    row-major flat index of an observation plane, so no translation happens
    here -- which is the property design §2 spends a section on.
    """
    return float(planes[plane].reshape(81)[square])


def test_from_tier_is_the_one_based_top_tier_read_before_the_move() -> None:
    """#8: a tsuke played FROM tier 2 records ``from_tier == 2`` -> 2/3 on 118.

    ``Board.height``'s COORDINATES are 0-based while its RETURN is the 1-based
    top tier, so the decoded record's ``from_r``/``from_f`` go in unadjusted.
    The trap is passing the 1-based coordinates ``apply_move`` converts to
    internally (``cyengine.pyx:199-200``).

    **The assertion that fires against the naive implementation** is
    ``_cell(planes, PLANE_LAST_MOVE_FROM, origin) == 2/3``.  This position is
    built so the naive read is unambiguous: the mover is on 6-5, so its
    0-based coordinates are ``(5, 4)`` and the 1-based ones are ``(6, 5)`` --
    which ``height`` would resolve to the 0-based square 6-5, i.e. rank 7 file
    6, which is EMPTY here.  A naive implementation therefore records
    ``from_tier = 0`` and leaves plane 118 identically zero, so the assertion
    fails on a value of 0.0 rather than on a subtly wrong tier.

    (Against ``main@29213f2`` this test fails at attribute level instead --
    ``_last_move`` does not exist there.  The claim above is about a naive
    implementation of THIS ticket, which is the one that matters.)
    """
    game = load_game(TIER_TWO_TSUKE_FEN)
    origin, destination = sq_index((6, 5)), sq_index((5, 5))
    move = move_matching(board_of(game), MT_TSUKE, (5, 5))
    assert int(move["to_tier"]) == 2, "the tsuke lands on tier 2 (target_tier + 1)"

    _play(game, move)

    assert game._last_move == (origin, 2, destination, 2)
    planes = observation_planes(game)
    assert _cell(planes, PLANE_LAST_MOVE_FROM, origin) == pytest.approx(_TIER_TWO)
    assert _cell(planes, PLANE_LAST_MOVE_TO, destination) == pytest.approx(_TIER_TWO)
    # At most one nonzero cell per plane -- the whole rest of the board is 0.
    assert np.count_nonzero(planes[PLANE_LAST_MOVE_FROM]) == 1
    assert np.count_nonzero(planes[PLANE_LAST_MOVE_TO]) == 1


def test_a_capture_above_a_friendly_piece_lands_above_tier_one() -> None:
    """#29: capturing an enemy stacked on a friend records ``to_tier == 2``.

    ``cymovegen.pyx:611-633`` lands a capture at *(friendly survivors in the
    target tower) + 1*, so the tower ``[white@t1, black@t2]`` gives the mover
    tier 2, not tier 1.  ``action_encoding.py:18-19``'s "a capture (lands at
    tier 1)" is a stale docstring and this test is what stops it becoming true
    by accident.

    **The assertion that fires against the naive implementation** -- one that
    hard-codes ``1/3`` for ``MT_CAPTURE``, which is what the struck rev-1
    design table would have produced -- is
    ``_cell(planes, PLANE_LAST_MOVE_TO, destination) == 2/3``: it would read
    ``1/3``.  ``decoded["to_tier"] == 2`` above it is the generator-side half
    of the same claim, asserted separately so a failure says which side moved.
    """
    game = load_game(CAPTURE_OVER_FRIEND_FEN)
    destination = sq_index((5, 5))
    move = move_matching(board_of(game), MT_CAPTURE, (5, 5))
    action = action_of(move)

    # The record's own value, and the value ``play_action`` will actually read:
    # decoded BEFORE the move, because afterwards the tower has changed.
    decoded = decoded_action(board_of(game), action)
    assert int(decoded["to_tier"]) == 2
    assert int(move["to_tier"]) == 2

    play_action_checked(game, action, ply=1)

    planes = observation_planes(game)
    assert _cell(planes, PLANE_LAST_MOVE_TO, destination) == pytest.approx(_TIER_TWO)
    assert game._last_move is not None
    assert game._last_move[3] == 2


def test_the_pair_decodes_move_class() -> None:
    """#9: (118, 119) reads "no last move" / "arata" / "board move", exactly.

    Replayed over golden games via their recorded ``chosen`` sequence, so the
    move types are the engine's own rather than a fixture's claim about them.
    Zero/nonzero only -- this criterion cannot catch a wrong ``to_tier``
    VALUE, which is what #8 and #29 are for.

    The whole advanced corpus is replayed, not a prefix, because the coverage
    floor below is what makes the node non-vacuous: it must be met by plies
    that actually occurred, and a prefix could stop meeting it after any
    regeneration of the goldens.  Betrayal is the one move type the corpus does
    not contain at all; it has its own skip-guarded node below.
    """
    games = load_golden_playouts("playouts_advanced.json")
    assert games, "the advanced playout corpus holds at least one game"

    seen: set[int] = set()
    for game_index, plies in enumerate(games):
        game = load_game(plies[0]["fen_before"])

        planes = observation_planes(game)
        assert not planes[PLANE_LAST_MOVE_FROM].any(), "ply 0 has no last move"
        assert not planes[PLANE_LAST_MOVE_TO].any(), "ply 0 has no last move"

        for ply_index, ply in enumerate(plies):
            record = move_by_san(board_of(game), ply["chosen"])
            move_type = int(record["move_type"])
            seen.add(move_type)
            game.play_action(action_of(record))

            planes = observation_planes(game)
            origin_cells = int(np.count_nonzero(planes[PLANE_LAST_MOVE_FROM]))
            destination_cells = int(np.count_nonzero(planes[PLANE_LAST_MOVE_TO]))
            where = f"game {game_index} ply {ply_index} ({ply['chosen']})"

            assert destination_cells == 1, (
                f"{where}: every move has a destination, so plane 119 carries "
                f"exactly one nonzero cell, got {destination_cells}"
            )
            expected_origin = 0 if move_type == MT_ARATA else 1
            assert origin_cells == expected_origin, (
                f"{where}: move_type={move_type} implies "
                f"{expected_origin} nonzero cell(s) on plane 118, "
                f"got {origin_cells}"
            )

    required = {MT_ROUTE, MT_TSUKE, MT_CAPTURE, MT_ARATA}
    assert required <= seen, (
        "the replayed goldens did not cover every move type this node claims "
        "to decode, so the assertions above ran on a corpus that cannot "
        f"distinguish them (missing {sorted(required - seen)}, saw "
        f"{sorted(seen)})"
    )


# ---------------------------------------------------------------------------
# The one ply of the betray corpus this engine refuses (KOM-170)
#
# A 謀 dropped from hand onto a tower whose tier-1 enemy is also a 謀 is paid
# for TWICE out of the same hand slot -- once for the placement, once for the
# conversion.  gungi.js offers it anyway and silently skips the second payment
# (its hand is a list whose entry is already deleted); komugi's dense hand array
# would go to -1, so the move is not generated.  The corpus keeps the reference
# behaviour, and the predicates below name the divergence structurally rather
# than by pasting the one SAN, so a re-recorded corpus is read, not assumed.
# ---------------------------------------------------------------------------

#: The three SAN markers this predicate reads: drop, conversion, draft-finish.
ARATA_MARKER: Final[str] = "新"
BETRAY_MARKER: Final[str] = "返"
DRAFT_FINISH_MARKER: Final[str] = "終"

#: Same-type arata-betrays in ``playouts_betray.json`` today: exactly one,
#: ``新謀(9-9-3)返謀``.  Pinned so the exemption cannot quietly widen.
KNOWN_SAME_TYPE_ARATA_BETRAYS: Final[int] = 1


def _converted_kanji(san: str) -> str:
    """The piece kanji a SAN's ``返`` suffix names, as a string of kanji."""
    marker = san.find(BETRAY_MARKER)
    if marker < 0:
        return ""
    return san[marker + 1 :].rstrip("=#").rstrip(DRAFT_FINISH_MARKER)


def _is_same_type_arata_betray(san: str) -> bool:
    """True iff *san* drops a piece from hand and converts that same type.

    ``新`` + the dropped kanji is how ``cymovegen._build_san`` spells a drop, so
    the dropped type is the second character; the converted types follow ``返``.
    """
    if not san.startswith(ARATA_MARKER) or BETRAY_MARKER not in san:
        return False
    return san[1] in _converted_kanji(san)


def _san_is_generated(game: Game, san: str) -> bool:
    """True iff the Cython generator offers *san* at *game*'s position."""
    normalized = san.rstrip("=#")
    return any(
        str(move["san"]) in (normalized, san)
        for move in generate_all_moves(board_of(game))
    )


def test_a_betray_ply_decodes_as_a_board_move() -> None:
    """N1: the one move type the advanced corpus never plays.

    ``playouts_advanced.json`` contains no betrayal at all (no ``返`` marker in
    any recorded ``chosen``), so #9's coverage floor cannot ask for one: a
    betrayal reaches the "both planes nonzero" branch only through a corpus
    that has one.  ``playouts_betray.json`` (KOM-101 G2) is that corpus.

    It is **filtered, not a playout**: only the plies where a ``返`` move was
    legal are emitted, so a ``games`` entry is a bag of unrelated positions and
    ply *i*'s ``fen_after`` is not ply *i+1*'s ``fen_before``
    (``scripts/oracle/README.md`` "Consequently a ``games`` entry ... is not a
    contiguous playout", and ``tests/engine/test_move_gen.py``'s
    ``_FULL_GAME_FILES`` comment).  Every ply is therefore loaded from its OWN
    ``fen_before`` -- chaining them, as #9 legitimately does for the four
    full-game files, walks the board off the corpus after the first ply and
    the next ply's ``chosen`` is simply not generated.
    """
    games = load_golden_playouts("playouts_betray.json")
    assert games, "the betray playout corpus holds at least one game"

    betrays = 0
    refused = 0
    for game_index, plies in enumerate(games):
        for ply_index, ply in enumerate(plies):
            game = load_game(ply["fen_before"])
            if _is_same_type_arata_betray(str(ply["chosen"])):
                # KOM-170: the reference plays this move; this engine refuses
                # it.  See the assertion below and
                # tests/rules/test_betrayal_hand_invariant.py for the whole
                # argument -- in short, gungi.js debits a LIST-shaped hand
                # whose entry is already gone, while a dense hand array would
                # go to -1.  The corpus is deliberately not re-recorded, so the
                # divergence is asserted here rather than hidden.
                refused += 1
                assert not _san_is_generated(game, str(ply["chosen"])), (
                    f"game {game_index} ply {ply_index}: "
                    f"{ply['chosen']} is a same-type arata-betray and must "
                    "not be generated (KOM-170)"
                )
                continue
            record = move_by_san(board_of(game), ply["chosen"])
            move_type = int(record["move_type"])
            if move_type != MT_BETRAY:
                continue
            betrays += 1
            # Read the generator's own destination BEFORE the move: afterwards
            # the tower has changed and the tier is no longer recoverable.
            destination = sq_index(
                (int(record["to_r"]) + 1, int(record["to_f"]) + 1)
            )
            origin = sq_index(
                (int(record["from_r"]) + 1, int(record["from_f"]) + 1)
            )
            to_tier = int(record["to_tier"])

            game.play_action(action_of(record))

            planes = observation_planes(game)
            where = f"game {game_index} ply {ply_index} ({ply['chosen']})"
            # A betrayal is a board move: it has an origin square, so BOTH
            # planes carry exactly one cell -- the same shape as a tsuke, which
            # is the loss #10 pins.
            assert np.count_nonzero(planes[PLANE_LAST_MOVE_FROM]) == 1, where
            assert np.count_nonzero(planes[PLANE_LAST_MOVE_TO]) == 1, where
            # ... and that one cell is at the generator's square, carrying the
            # generator's tier.  Zero/nonzero alone cannot see a wrong
            # ``to_tier``; this can, on every betray ply in the corpus.
            assert _cell(planes, PLANE_LAST_MOVE_FROM, origin) != 0.0, where
            assert _cell(planes, PLANE_LAST_MOVE_TO, destination) == (
                pytest.approx(to_tier / 3.0)
            ), where

    assert betrays >= 1, (
        "the betray corpus replayed without a single MT_BETRAY ply, so this "
        "node asserted nothing"
    )
    assert refused == KNOWN_SAME_TYPE_ARATA_BETRAYS, (
        f"the corpus now holds {refused} same-type arata-betrays, not "
        f"{KNOWN_SAME_TYPE_ARATA_BETRAYS}; the KOM-170 divergence from "
        "gungi.js has moved and wants re-reading, not a bumped constant"
    )


def test_a_betray_and_a_tsuke_are_indistinguishable_in_118_119() -> None:
    """#10: the two moves are EQUAL in 118/119 and UNEQUAL elsewhere.

    Design §3.2 chooses not to spend a plane on the conversion selector.  What
    survives is stated here as a tested claim rather than a silent one: the
    last-move planes cannot separate a betrayal from the tsuke to the same
    square at the same tiers, while the occupancy planes (the converted
    piece's new colour) and the hand planes (the mover's debit) can.
    """
    tsuke_game = load_game(BETRAYAL_FEN)
    betray_game = load_game(BETRAYAL_FEN)

    tsuke = move_matching(board_of(tsuke_game), MT_TSUKE, (5, 6))
    betray = move_matching(board_of(betray_game), MT_BETRAY, (5, 6))
    assert int(tsuke["to_tier"]) == int(betray["to_tier"]) == 2, (
        "precondition: the same square at the same tier, or the equality "
        "below would be trivially true for the wrong reason"
    )

    _play(tsuke_game, tsuke)
    _play(betray_game, betray)

    after_tsuke = observation_planes(tsuke_game)
    after_betray = observation_planes(betray_game)

    last_move = slice(PLANE_LAST_MOVE_FROM, PLANE_LAST_MOVE_TO + 1)
    assert np.array_equal(after_tsuke[last_move], after_betray[last_move]), (
        "planes 118/119 record squares and tiers only, so a betrayal and the "
        "tsuke to the same square must be identical in them"
    )
    assert not np.array_equal(after_tsuke[0:84], after_betray[0:84]), (
        "the converted piece's colour is visible in the occupancy planes"
    )
    assert not np.array_equal(after_tsuke[84:112], after_betray[84:112]), (
        "the mover's per-conversion hand debit is visible in the hand planes"
    )


def test_symmetries_mirror_the_last_move() -> None:
    """#11: the mirrored sample carries the MIRRORED move, not the original.

    Two assertions, and design §7 (N8) is precise about which is red where:

    * the observation assertion is red against a naive implementation of this
      ticket -- one that copies ``_last_move`` across a file reflection
      unmirrored, which would mark the original move's squares on the
      reflected board in half of every augmented batch, in range and silent;
    * the ``last_action`` assertion is red against ``main@29213f2`` itself,
      where ``game.py:480`` copies the action id across unmirrored while
      ``_mirror_action`` is applied to the policy vector five lines above it.
    """
    game = load_game(OFF_CENTRE_ROUTE_FEN)
    move = move_matching(board_of(game), MT_ROUTE, (5, 3))
    action = _play(game, move)

    pi = np.zeros(Game.action_size(), dtype=np.float32)
    pi[action] = 1.0
    pairs = game.symmetries(pi)
    assert len(pairs) == 2, "identity and the one file mirror"
    mirrored_game, _ = pairs[1]

    original = observation_planes(game)
    mirrored = observation_planes(mirrored_game)
    last_move = slice(PLANE_LAST_MOVE_FROM, PLANE_LAST_MOVE_TO + 1)

    # The encoder's file axis is ``j = 8 - f``, so a file reflection is a
    # reversal of the plane's last axis and nothing else.
    assert np.array_equal(mirrored[last_move], original[last_move][:, :, ::-1])
    # Non-vacuity: the move really is off the centre file, so the reflection
    # moved it.  A centre-file move would satisfy the assertion above whether
    # or not any mirroring happened.
    assert not np.array_equal(mirrored[last_move], original[last_move])

    # Non-vacuity for the action half, matching the plane half above: on a
    # centre-file move ``_mirror_action(a) == a``, and the assertion below
    # would then pass against a copy that mirrored nothing.  The fixture is
    # off-centre for exactly this reason, so the inequality is asserted first
    # rather than left to the fixture's docstring.
    assert _mirror_action(action) != action, (
        "the chosen move is on the centre file, so mirroring is the identity "
        "on its action id and the assertion below cannot fail"
    )
    # Read through ``Any``: ``GameState.__init__`` assigns ``last_action =
    # None`` with no annotation, so mypy infers the attribute as ``None`` and
    # a direct ``== int`` comparison trips --strict-equality.
    mirrored_action: Any = mirrored_game.last_action
    assert mirrored_action == _mirror_action(action)


def test_a_fen_loaded_game_has_no_last_move() -> None:
    """#13: planes 118 and 119 are identically zero for every golden FEN.

    A FEN has no move history -- ``fen.py``'s six fields are placement, hands,
    turn, mode, drafting and move number -- so a ``Game`` built from one
    cannot know its last move and must say so by emitting zeros, exactly as it
    already does for the repetition count.
    """
    fens = golden_fens()
    assert len(fens) >= 1500, f"expected the full golden corpus, got {len(fens)}"

    for fen in fens:
        game = load_game(fen)
        assert game._last_move is None, fen
        planes = observation_planes(game)
        if planes[PLANE_LAST_MOVE_FROM].any() or planes[PLANE_LAST_MOVE_TO].any():
            raise AssertionError(f"a FEN-loaded game claims a last move: {fen!r}")
