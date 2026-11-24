"""The self-play tsume miner: round trips, the deep case, hygiene, budgets.

KOM-260's acceptance conditions, as tests. The shape of the suite follows the
two claims the module makes that the old end-of-game mate-in-1 miner could
not:

*It finds the mates the net missed.* Every committed mined fixture is a
position with a forced mate, and every one of them is run twice -- once with a
mating key move played, once with a legal non-mating move played -- so the
same position must come back ``played_kept_mate`` True and then False. A
miner that flagged everything as a miss fails the first half; one that flagged
nothing fails the second. The vacuity guard closes the same door from the
other side: a sequence mixing positions with and without forced mates has to
produce a strict subset.

*It sees past one ply.* A composed mate-in-3 is graded 3 with a three-move
line at ``max_plies=3`` and reads as no mate at all at ``max_plies=1`` -- the
old miner's blind spot, pinned rather than described.

Every position here comes from ``tsume_fixtures``; no FEN is re-embedded. The
two synthetic boards the hygiene tests need -- one mid-draft, one with a
marshal captured -- are *derived* from a committed fixture through
``board_from_fen``/``board_to_fen`` rather than transcribed, so they cannot
drift from it.
"""

from __future__ import annotations

import inspect

import pytest
from tsume_fixtures import (
    FRIEND,
    MARSHAL_CAPTURE_ROWS,
    MINED,
    FriendFixture,
    enumeration_keys,
    forced_mate_keys,
    marshal_capturing_replies,
    no_mate_positions,
)
from tsume_fixtures import MinedPuzzle as MinedFixture

from komugi.engine import cyboard, cyengine, cymovegen
from komugi.search.tsume_mining import (
    MinedPuzzle,
    ScanResult,
    dedupe_by_position,
    mine_missed_mates,
    scan_positions,
)

#: The mined rows the solver calls MATE at depth 1: the committed rows minus
#: the three the defender refutes by capturing the attacker's marshal. Nine
#: committed rows since KOM-261 quarantined ``iter_827-g200002``, so six.
#: A3 is stated over exactly these.
CLEAN_MINED = tuple(m for m in MINED if m.name not in MARSHAL_CAPTURE_ROWS)

#: The composed mate-in-3s. A4 needs one; running all of them costs nothing.
DEEP_FRIEND = tuple(f for f in FRIEND if f.tsume_plies == 3)

#: Generous for these positions -- the whole mined set solves in under a
#: hundred nodes apiece -- and small enough to keep a pathological position
#: from running the suite long.
NODE_BUDGET = 20_000


def _key_move(fen: str) -> str:
    """A move that mates in one under the solver's semantics."""
    keys = forced_mate_keys(fen)
    assert keys, f"fixture has no forced mate in 1: {fen!r}"
    return keys[0]


def _non_mating_move(fen: str) -> str:
    """A legal move that neither mates nor ends the game by capture.

    Excluded, in order: every move ``is_checkmate`` calls mate (a superset of
    the solver's forced mates), and every move that takes the opponent's
    marshal. Without the second exclusion the move could end the game outright
    and the record would read "did not keep the mate" for the wrong reason.
    """
    board = cyboard.board_from_fen(fen)
    barred = set(enumeration_keys(fen)) | set(marshal_capturing_replies(fen))
    for move in cymovegen.generate_all_moves(board):
        san = str(move["san"])
        if san not in barred:
            return san
    raise AssertionError(f"every legal move mates or ends the game: {fen!r}")


def _mid_draft_fen(fen: str) -> str:
    """*fen* with the white draft flag set, both marshals still on the board.

    Derived rather than transcribed, and deliberately not one of the published
    start positions: those have an empty board, so they are game-over *and*
    in the draft and could not tell the two skip reasons apart.
    """
    board = cyboard.board_from_fen(fen)
    board.draft_w = 1
    return cyboard.board_to_fen(board)


def _marshal_captured_fen(fen: str) -> str:
    """*fen* with white's marshal removed: a game that is already over."""
    board = cyboard.board_from_fen(fen)
    code = cyboard.make_code(0, 0)
    board.cells_arr[board.cells_arr == code] = 0
    return cyboard.board_to_fen(board)


def _renumbered_fen(fen: str, move_number: int) -> str:
    """*fen* at a different move number: a different string, same position."""
    board = cyboard.board_from_fen(fen)
    board.move_number = move_number
    return cyboard.board_to_fen(board)


def _record(
    fen: str, distance: int, *, index: int = 0, kept: bool = True
) -> MinedPuzzle:
    """A hand-built record, for the dedupe tests (no solver involved)."""
    return MinedPuzzle(
        fen=fen,
        distance=distance,
        line=tuple(f"m{i}" for i in range(distance)),
        played_san=None,
        played_kept_mate=kept,
        index=index,
        nodes=0,
    )


def _scan(
    pairs: list[tuple[str, str | None]],
    *,
    max_plies: int = 1,
) -> ScanResult:
    """A scan at the suite's standard budget, so the tests read as intent."""
    return scan_positions(
        pairs,
        max_plies=max_plies,
        node_budget=NODE_BUDGET,
    )


# ---------------------------------------------------------------------------
# The fixture set the round trip is stated over
# ---------------------------------------------------------------------------


def test_the_clean_mined_set_is_the_rows_a3_names() -> None:
    """A3 is "the non-capture-refutable mined fixtures", not "however many".

    A3 was written over 7 rows, when the mined set had 10. KOM-261
    quarantined ``iter_827-g200002`` -- an already-in-check root, and *not*
    one of the three marshal-capture rows -- so the clean set is 6 of 9.
    The relation between the three sets is what is pinned, and derived
    rather than transcribed, so a row added to ``mined.json`` or a
    marshal-capture row added shows up in exactly one place; the floor
    underneath it is what stops the clean set quietly emptying out.
    """
    assert len(MINED) >= 9
    assert MARSHAL_CAPTURE_ROWS <= {m.name for m in MINED}
    assert len(CLEAN_MINED) == len(MINED) - len(MARSHAL_CAPTURE_ROWS)
    assert len(CLEAN_MINED) >= 6
    assert all(m.mate_in == 1 for m in CLEAN_MINED)


# ---------------------------------------------------------------------------
# A3: the round trip, both halves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", CLEAN_MINED, ids=lambda m: m.name)
def test_key_move_scans_as_a_kept_mate_in_one(fixture: MinedFixture) -> None:
    """A3, first half: MATE, distance 1, and the mover kept it."""
    fen = fixture.fen
    result = _scan([(fen, _key_move(fen))])

    assert result.skipped == 0
    assert result.scanned == 1
    assert not result.budget_exhausted
    assert len(result.puzzles) == 1
    record = result.puzzles[0]
    assert record.fen == fen
    assert record.distance == 1
    assert len(record.line) == 1
    assert record.line[0] in forced_mate_keys(fen)
    assert record.played_kept_mate is True
    assert record.index == 0
    assert 0 < record.nodes <= NODE_BUDGET
    assert mine_missed_mates(result) == ()


@pytest.mark.parametrize("fixture", CLEAN_MINED, ids=lambda m: m.name)
def test_non_mating_move_scans_as_a_missed_mate(fixture: MinedFixture) -> None:
    """A3, second half: the same position, a legal non-mating move played."""
    fen = fixture.fen
    played = _non_mating_move(fen)
    result = _scan([(fen, played)])

    assert len(result.puzzles) == 1
    record = result.puzzles[0]
    assert record.distance == 1
    assert record.played_san == played
    assert record.played_kept_mate is False
    assert mine_missed_mates(result) == (record,)


def test_missed_and_kept_differ_only_in_the_move_played() -> None:
    """The two halves of A3 over one scan, so neither can be an artefact.

    Each fixture appears twice in one sequence -- once with its key move, once
    with a non-mating move -- and the miner must split them down the middle.
    A miner that ignored ``played_san`` entirely would put all fourteen
    records on one side and fail here.
    """
    pairs: list[tuple[str, str | None]] = []
    for fixture in CLEAN_MINED:
        pairs.append((fixture.fen, _key_move(fixture.fen)))
        pairs.append((fixture.fen, _non_mating_move(fixture.fen)))

    result = _scan(pairs)

    assert len(result.puzzles) == len(pairs)
    kept = [r.index for r in result.puzzles if r.played_kept_mate]
    missed = [r.index for r in mine_missed_mates(result)]
    assert kept == list(range(0, len(pairs), 2))
    assert missed == list(range(1, len(pairs), 2))


def test_positions_without_a_forced_mate_are_a_strict_subset() -> None:
    """The vacuity guard: mates and non-mates mixed, only the mates kept.

    ``no_mate_positions()`` is the committed set where a mate would mean the
    rules are broken (the published start positions) or that the replayed
    midgame is not quiet after all. Interleaved with the mined mate-in-1s, the
    record set must be neither empty nor everything, and it must be exactly
    the mate half -- which is what makes the A3 pass above non-vacuous.
    """
    quiet = [fen for _, fen in no_mate_positions()]
    pairs: list[tuple[str, str | None]] = []
    mate_indices: list[int] = []
    for offset, fixture in enumerate(CLEAN_MINED):
        pairs.append((quiet[offset % len(quiet)], None))
        mate_indices.append(len(pairs))
        pairs.append((fixture.fen, _key_move(fixture.fen)))

    result = _scan(pairs)

    assert result.scanned == len(pairs)
    assert result.skipped == 0
    assert 0 < len(result.puzzles) < len(pairs)
    assert [r.index for r in result.puzzles] == mate_indices


# ---------------------------------------------------------------------------
# A4: the deep case, and the one-ply miner's blind spot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", DEEP_FRIEND, ids=lambda f: f.name)
def test_deep_mate_is_graded_three_with_a_three_move_line(
    fixture: FriendFixture,
) -> None:
    """A4, first half: a forced mate in 3 reported as distance 3."""
    result = _scan([(fixture.fen, None)], max_plies=3)

    assert len(result.puzzles) == 1
    record = result.puzzles[0]
    assert record.distance == 3
    assert len(record.line) == 3
    assert all(move for move in record.line)


@pytest.mark.parametrize("fixture", DEEP_FRIEND, ids=lambda f: f.name)
def test_deep_mate_is_invisible_at_one_ply(fixture: FriendFixture) -> None:
    """A4, second half: the same position, nothing found at ``max_plies=1``.

    This is the old miner's blind spot demonstrated rather than asserted in
    prose: it only ever asked the ``max_plies=1`` question, so every puzzle
    the test above finds was invisible to it.
    """
    shallow = _scan([(fixture.fen, None)], max_plies=1)

    assert shallow.scanned == 1
    assert shallow.puzzles == ()
    assert not shallow.budget_exhausted


def test_deep_key_move_keeps_the_mate_through_every_defence() -> None:
    """The kept-mate check at depth: an AND node, one re-solve per reply.

    Distance 1 never exercises that path -- a mating move leaves the defender
    no reply at all -- so the deep case is where "still holds a forced mate
    within ``max_plies - 1``" is actually computed: every defender answer to
    the key move must still be mate within one more ply.
    """
    fixture = DEEP_FRIEND[0]
    found = _scan([(fixture.fen, None)], max_plies=3).puzzles[0]
    key = found.line[0]

    kept = _scan([(fixture.fen, key)], max_plies=3).puzzles[0]
    assert kept.played_san == key
    assert kept.played_kept_mate is True
    # The re-solves cost nodes on top of the root solve, which is the whole
    # point: the flag was computed, not inherited from the root verdict.
    assert kept.nodes > found.nodes
    assert mine_missed_mates(_scan([(fixture.fen, key)], max_plies=3)) == ()


def test_played_move_that_is_not_generated_here_did_not_keep_the_mate() -> None:
    """An unplayable SAN is not a kept mate, and is not an exception either."""
    fixture = CLEAN_MINED[0]
    result = _scan([(fixture.fen, "not-a-move")])

    assert len(result.puzzles) == 1
    assert result.puzzles[0].played_kept_mate is False


def test_no_played_move_is_reported_as_not_kept() -> None:
    """``played_san=None`` records the puzzle without claiming a miss was made.

    The record still counts as a missed mate by the ``played_kept_mate``
    definition, because nothing is known to have kept it; the harness is what
    decides whether a position with no follow-up move is interesting.
    """
    fixture = CLEAN_MINED[0]
    record = _scan([(fixture.fen, None)]).puzzles[0]

    assert record.played_san is None
    assert record.played_kept_mate is False


# ---------------------------------------------------------------------------
# A5: draft and terminal hygiene
# ---------------------------------------------------------------------------


def test_draft_positions_are_skipped_not_scanned() -> None:
    """A5: in-draft positions are counted into ``skipped`` and never solved."""
    fixture = CLEAN_MINED[0]
    draft_fen = _mid_draft_fen(fixture.fen)
    board = cyboard.board_from_fen(draft_fen)
    assert int(board.draft_w) or int(board.draft_b)

    result = _scan([(draft_fen, None), (fixture.fen, _key_move(fixture.fen))])

    assert result.skipped == 1
    assert result.scanned == 1
    assert [r.index for r in result.puzzles] == [1]


def test_a_draft_root_is_skipped_and_never_raises() -> None:
    """KOM-282 D4: a mid-draft root alone is a skip, not a solver ValueError.

    ``solve_tsume`` refuses a draft root since KOM-282, so a miner that handed
    one to the solver would abort a whole mining run on it. The non-draft
    position the draft FEN is derived from still scans (the test above), so
    this is not a miner that skips everything.
    """
    draft_fen = _mid_draft_fen(CLEAN_MINED[0].fen)

    result = _scan([(draft_fen, None)])

    assert result.skipped == 1
    assert result.scanned == 0
    assert result.puzzles == ()
    assert result.nodes == 0


def test_the_miner_has_no_skip_draft_switch() -> None:
    """KOM-282 D4: draft roots are always skipped; there is no opt-out."""
    assert "skip_draft" not in inspect.signature(scan_positions).parameters
    assert "skip_draft" not in inspect.signature(mine_missed_mates).parameters


def test_game_over_position_is_skipped_rather_than_raising() -> None:
    """A5: a captured marshal is a skip here, though ``solve_tsume`` rejects it.

    KOM-258 F1 made ``solve_tsume`` raise on such a root, so a miner that
    handed one straight to the solver would abort a whole mining run on one
    position.
    """
    over_fen = _marshal_captured_fen(CLEAN_MINED[0].fen)
    board = cyboard.board_from_fen(over_fen)
    assert not int(board.draft_w) and not int(board.draft_b)

    result = _scan([(over_fen, None)])

    assert result.skipped == 1
    assert result.scanned == 0
    assert result.puzzles == ()
    assert result.nodes == 0


def test_draft_and_game_over_positions_are_both_skipped() -> None:
    """Both hygiene rules apply unconditionally, and each counts once."""
    fen = CLEAN_MINED[0].fen

    result = _scan([(_mid_draft_fen(fen), None), (_marshal_captured_fen(fen), None)])

    assert result.skipped == 2
    assert result.scanned == 0


def test_index_counts_skipped_positions() -> None:
    """Provenance survives skipping: the index is into the input sequence.

    A record has to point back at the ply of the game it came from, so the
    counter cannot be "records so far" or "positions solved so far".
    """
    fixture = CLEAN_MINED[0]
    pairs: list[tuple[str, str | None]] = [
        (_mid_draft_fen(fixture.fen), None),
        (_marshal_captured_fen(fixture.fen), None),
        (fixture.fen, _key_move(fixture.fen)),
    ]

    result = _scan(pairs)

    assert result.skipped == 2
    assert [r.index for r in result.puzzles] == [2]


# ---------------------------------------------------------------------------
# A6: budgets
# ---------------------------------------------------------------------------


def test_total_budget_below_one_solve_fabricates_nothing() -> None:
    """A6: too small a total budget stops the scan and invents no records.

    The same position with an adequate budget is checked to be a real mate in
    the same test, so this cannot pass because the position was uninteresting.
    """
    fixture = CLEAN_MINED[0]
    pairs: list[tuple[str, str | None]] = [(fixture.fen, _key_move(fixture.fen))]

    funded = scan_positions(pairs, max_plies=1, node_budget=NODE_BUDGET)
    assert len(funded.puzzles) == 1
    assert funded.nodes > 1

    starved = scan_positions(
        pairs, max_plies=1, node_budget=NODE_BUDGET, total_budget=1
    )
    assert starved.budget_exhausted is True
    assert starved.puzzles == ()
    assert starved.scanned == 0
    assert starved.nodes <= 1


def test_total_budget_stops_after_the_positions_it_can_fund() -> None:
    """A6: a partially funded scan returns what it found, flagged, under budget.

    The budget is honoured before the fact -- a solve starts only when its
    full per-call budget still fits -- so the first position gets a complete
    search and the second is not started at all. Nothing half-searched
    reaches the record set.
    """
    first, second = CLEAN_MINED[0], CLEAN_MINED[1]
    node_budget = 200
    solo = scan_positions(
        [(first.fen, _key_move(first.fen))], max_plies=1, node_budget=node_budget
    )
    assert solo.nodes > 2
    total_budget = node_budget + solo.nodes - 1

    result = scan_positions(
        [
            (first.fen, _key_move(first.fen)),
            (second.fen, _key_move(second.fen)),
        ],
        max_plies=1,
        node_budget=node_budget,
        total_budget=total_budget,
    )

    assert result.budget_exhausted is True
    assert result.scanned == 1
    assert [r.index for r in result.puzzles] == [0]
    assert result.puzzles[0].distance == 1
    assert result.nodes <= total_budget


def test_an_unconstrained_scan_is_not_flagged_exhausted() -> None:
    """``budget_exhausted`` means the scan stopped, not that a budget existed."""
    pairs: list[tuple[str, str | None]] = [
        (m.fen, _key_move(m.fen)) for m in CLEAN_MINED
    ]

    result = scan_positions(
        pairs, max_plies=1, node_budget=NODE_BUDGET, total_budget=NODE_BUDGET * 100
    )

    assert result.budget_exhausted is False
    assert result.scanned == len(pairs)
    assert len(result.puzzles) == len(pairs)


def test_even_max_plies_is_rejected_before_anything_is_scanned() -> None:
    """The ``solve_tsume`` depth rule, enforced even for an empty sequence."""
    with pytest.raises(ValueError, match="odd"):
        scan_positions([], max_plies=2, node_budget=NODE_BUDGET)
    with pytest.raises(ValueError, match="node_budget"):
        scan_positions([], max_plies=1, node_budget=0)


# ---------------------------------------------------------------------------
# A7: dedupe
# ---------------------------------------------------------------------------


def test_dedupe_keeps_the_shorter_distance() -> None:
    """A7: the same position at two grades collapses to the shorter mate."""
    fen = CLEAN_MINED[0].fen
    long_record = _record(fen, 3, index=0)
    short_record = _record(fen, 1, index=7)

    assert dedupe_by_position([long_record, short_record]) == (short_record,)
    assert dedupe_by_position([short_record, long_record]) == (short_record,)


def test_dedupe_is_keyed_by_repetition_key_not_by_fen_string() -> None:
    """A7: two FENs of one position, differing only in the move counter.

    ``repetition_key`` leaves the move number out, which is exactly why it is
    the key: the same position recurring in a later game arrives with a
    different counter and must still collapse.
    """
    fen = CLEAN_MINED[0].fen
    renumbered = _renumbered_fen(fen, 99)
    assert renumbered != fen
    assert cyengine.repetition_key(
        cyboard.board_from_fen(renumbered)
    ) == cyengine.repetition_key(cyboard.board_from_fen(fen))

    deduped = dedupe_by_position([_record(fen, 3), _record(renumbered, 1, index=1)])

    assert len(deduped) == 1
    assert deduped[0].distance == 1
    assert deduped[0].fen == renumbered


def test_dedupe_keeps_the_first_of_equal_distances_and_distinct_positions() -> None:
    """Ties keep the first record; genuinely different positions both survive."""
    first_fen = CLEAN_MINED[0].fen
    other_fen = CLEAN_MINED[1].fen
    first = _record(first_fen, 1, index=0)
    duplicate = _record(_renumbered_fen(first_fen, 55), 1, index=4)
    other = _record(other_fen, 1, index=9)

    deduped = dedupe_by_position([first, duplicate, other])

    assert deduped == (first, other)


# ---------------------------------------------------------------------------
# mine_missed_mates' two calling forms
# ---------------------------------------------------------------------------


def test_mine_missed_mates_scans_pairs_when_handed_pairs() -> None:
    """The convenience form: pairs in, the missed subset out."""
    fixture = CLEAN_MINED[0]
    missed = mine_missed_mates(
        [
            (fixture.fen, _key_move(fixture.fen)),
            (fixture.fen, _non_mating_move(fixture.fen)),
        ],
        max_plies=1,
        node_budget=NODE_BUDGET,
    )

    assert [r.index for r in missed] == [1]
    assert all(not r.played_kept_mate for r in missed)


def test_mine_missed_mates_rejects_budgets_it_cannot_honour() -> None:
    """Budget arguments against an existing ScanResult are an error, not a no-op.

    They cannot change what that scan already measured, and silently ignoring
    them would let a caller believe a deeper search had been run.
    """
    result = _scan([(CLEAN_MINED[0].fen, None)])

    with pytest.raises(ValueError, match="ScanResult"):
        mine_missed_mates(result, max_plies=3)
    with pytest.raises(ValueError, match="max_plies"):
        mine_missed_mates([(CLEAN_MINED[0].fen, None)])
