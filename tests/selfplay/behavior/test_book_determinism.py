"""Two builds of one shard set are byte-identical (KOM-014 §8 #9, §6.6).

The invariant §2.3 argues from five facts -- deterministic shard enumeration, a
derived ``build_seed``, one named ``PCG64`` generator consumed in one documented
order, entries emitted in draw order, and canonical JSON with rounded floats --
and the fifth is the one an implementer skips. So the check is on **bytes**, not
on the parsed object: a book whose floats round-trip differently on two machines
would compare equal as objects and unequal as the artefact anyone actually
diffs.

Two *roots* rather than one root built twice, because rebuilding in place would
also pass for a builder that read its own previous output.
"""

from __future__ import annotations

from pathlib import Path

import test_book_prefixes_replay as rig

from komugi.selfplay.book import book_path_for, build_book, load_book


def _seeded_root(root: Path) -> Path:
    """Plant one identical 24-game iteration under `root`."""
    rig.plant(root, 0, rig.ordinary_games(24, seed=11))
    return root


def test_two_builds_of_one_shard_set_are_byte_identical(tmp_path: Path) -> None:
    """Same rows, same run seed, same generation -- therefore the same bytes."""
    first = _seeded_root(tmp_path / "first")
    second = _seeded_root(tmp_path / "second")

    left = build_book(first, 1, 20260905, cap=96, min_entries=32)
    right = build_book(second, 1, 20260905, cap=96, min_entries=32)
    assert left == book_path_for(first, 1)
    assert right == book_path_for(second, 1)
    assert left is not None and right is not None

    assert left.read_bytes() == right.read_bytes(), (
        "two builds over identical shard trees at one seed disagreed; the "
        "candidate order, the generator or the float serialisation is not the "
        "pure function of the tree §2.3 requires"
    )


def test_a_different_run_seed_moves_the_book(tmp_path: Path) -> None:
    """The control: determinism must not be the builder ignoring its seed.

    Without this, a builder that never touched its generator at all would pass
    the byte-comparison above -- which is the failure mode a determinism test is
    most likely to have.
    """
    root = _seeded_root(tmp_path / "one")
    baseline = build_book(root, 1, 20260905, cap=96, min_entries=32)
    assert baseline is not None
    original = baseline.read_bytes()

    moved = build_book(root, 1, 4242, cap=96, min_entries=32)
    assert moved is not None
    assert moved.read_bytes() != original, (
        "changing run_seed left the book untouched, so build_seed is not "
        "reaching the sampler"
    )


def test_entries_are_written_in_draw_order(tmp_path: Path) -> None:
    """§2.3 fact 4: **draw** order, not sorted order (impl-review F4).

    Kills the "write ``sorted(entries, key=...)`` instead of ``entries``" mutant
    (M7), which the two nodes above cannot see: both compare one implementation
    against itself, so *any* deterministic reordering passes them, and the
    order-blind nodes elsewhere (#34's ``deviation_ply`` filter, #36a's per-game
    counts) never look at position.

    The one artefact of draw order the file carries is the ``{index:06d}`` field
    of ``BookEntry.id`` (``book.py``'s ``_entry_id``), which is stamped from
    ``position=len(entries)`` at the moment the candidate was accepted. So the
    id's index and the entry's position in ``entries`` agree if and only if the
    list was never reordered, and a sort by ``actions`` over 32+ entries cannot
    coincide with draw order.

    Draw order matters for the reason ``build_book``'s docstring gives: sorting
    would be equally byte-deterministic and would *hide* a sampler that drew
    badly, and it would decouple the ``stats`` counters from the entry list they
    are supposed to be readable against.
    """
    root = _seeded_root(tmp_path / "order")
    path = build_book(root, 1, 20260905, cap=96, min_entries=32)
    assert path is not None
    loaded = load_book(path)
    assert len(loaded.entries) >= 32, (
        "too few entries for a reordering to be visible; this node needs a book "
        "whose draw order and sorted order cannot coincide"
    )
    for position, entry in enumerate(loaded.entries):
        stamped = int(entry.id.split("-")[1])
        assert stamped == position, (
            f"entry {entry.id} carries draw index {stamped} but sits at "
            f"position {position} in the manifest: the entry list was reordered "
            "after the ids were stamped, so the stats counters can no longer be "
            "read against it"
        )
