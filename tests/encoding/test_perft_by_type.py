"""T9 -- move-type-bucketed perft against the reference engine.

Set equality on a single ply (T1) cannot see a combination that is generated
correctly but *applied* wrongly: a betray that converts the wrong tier leaves
a tower the reference never produces, and the error only shows up in the
children.  Depth-1..3 bucketed counts catch it, because every subsequent ply
is generated from the position the previous move produced.

The reference tables in ``goldens/betrayal_dumps.json`` come from the same
gungi-js harness as the dumps, walking ``Gungi.moves({verbose: true})`` and
rebuilding a ``Gungi`` from each move's after-FEN.  Two shape details of that
harness are reproduced in ``helpers.perft_by_type`` and matter for reading the
numbers: the counts are CUMULATIVE over plies 1..depth, and a terminal node
contributes nothing because ``moves()`` returns ``[]`` there.

UNGATED, deliberately.  The counts depend only on move generation and
application -- ``helpers.perft_by_type`` uses nothing but
``cymovegen.generate_all_moves`` and ``cyengine.apply_move``, no encoder and no
mask -- and KOM-028 touches neither; see the design's "cymovegen.pyx" bullet on
``_get_betrayal_combos``: "unchanged ... only the *encoder* was throwing that
information away".  So these tables run against TODAY's engine, and a failure
here points at the rules engine rather than at the layout.  (Depth 1 is also
covered unconditionally by
``test_reference_dumps_sanity.test_depth_one_perft_equals_the_dumped_move_counts``,
on the reference side of the same numbers.)
"""

from __future__ import annotations

import pytest

from helpers import (
    corpus_fen,
    perft_by_type,
    perft_fixture_names,
    perft_table,
)

PERFT_FIXTURES: tuple[str, ...] = perft_fixture_names()

#: ``(fixture, depth)`` for every table the reference recorded.
PERFT_CASES: tuple[tuple[str, int], ...] = tuple(
    (name, int(depth))
    for name in PERFT_FIXTURES
    for depth in sorted(perft_table(name), key=int)
)


@pytest.mark.parametrize(("name", "depth"), PERFT_CASES)
def test_t9_bucketed_perft_matches_the_reference(name: str, depth: int) -> None:
    """Exact per-bucket equality at every recorded depth."""
    expected = perft_table(name)[str(depth)]
    actual = perft_by_type(corpus_fen(name), depth)
    assert actual == expected, f"{name} d{depth}: {actual} != {expected}"


@pytest.mark.parametrize("name", PERFT_FIXTURES)
def test_t9_the_buckets_are_monotone_in_depth(name: str) -> None:
    """Cumulative counts can only grow -- a guard on the harness's own shape.

    If ``perft_by_type`` were ever "fixed" to count only the deepest ply, this
    fails before the comparison above turns into a rewrite of the goldens.
    """
    table = perft_table(name)
    depths = sorted(table, key=int)
    for shallower, deeper in zip(depths, depths[1:]):
        for bucket, count in table[shallower].items():
            assert table[deeper].get(bucket, 0) >= count


def test_t9_the_betray_bucket_is_nonzero_in_at_least_three_fixtures() -> None:
    """T9's coverage clause, measured on OUR engine rather than the goldens.

    ``test_reference_dumps_sanity`` makes the same assertion about the
    reference side; this one fails if our generator stops producing betrayals
    even though the corpus still contains them.
    """
    betraying = [
        name
        for name in PERFT_FIXTURES
        if perft_by_type(corpus_fen(name), 1).get("betray", 0) > 0
    ]
    assert len(betraying) >= 3, betraying


def test_t9_betrayals_survive_into_the_second_ply() -> None:
    """Depth 2 keeps a betray bucket, so the applied betray really is a position.

    ``mixed_pair_hand_both`` betrays three ways at the root; at depth 2 the
    reference still counts three, because the betrayals consume the tower they
    convert and Black has none of its own -- i.e. the three root betrayals are
    the only ones in the whole two-ply tree.  Reproducing that number is a
    statement about the *resulting boards*, which one-ply set equality never
    reaches.
    """
    name = "mixed_pair_hand_both"
    assert perft_table(name)["2"]["betray"] == 3
    assert perft_by_type(corpus_fen(name), 2)["betray"] == 3
