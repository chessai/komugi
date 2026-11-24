"""The builder projects columns and stays inside its shard budget (§8 #38, B5).

Rev 1 of the design claimed the builder "touches a few dozen rows"; against the
only reader in the repo that was false, and §2.4 is the correction. Two halves,
both here:

* **No whole-row reader, at all.** ``_load_shard_arrays`` and
  ``ShardReader.__init__`` are monkeypatched to fail, and a full build still
  succeeds. A builder that reached for either -- which is the obvious way to
  write it -- would decompress and digest-verify a whole window (~1.5-2.2 GB per
  production iteration) to read three ``int64`` columns.
* **The visit budget holds.** Over a tree with more candidate shards than
  :data:`~komugi.selfplay.book.TARGET_SHARD_BUDGET`, the target pass opens at
  most that many, and the games it therefore never reaches are *counted* rather
  than silently lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import test_book_prefixes_replay as rig

from komugi.selfplay.book import TARGET_SHARD_BUDGET, build_book, load_book
from komugi.train import shards

_GAMES: int = TARGET_SHARD_BUDGET + 8
"""One shard per game below, so the tree carries eight more candidate shards
than the budget -- which is what makes the budget observable at all.

Derived from the budget, not a literal: the budget is itself derived from the
engine's turn cap (``book.TARGET_SHARD_BUDGET``), so KOM-165's 256 → 512 doubled
it to 64. A fixture pinned at 40 shards would have stopped exercising the bound
entirely -- every game reachable, nothing skipped -- and this test would have
gone quietly vacuous instead of red."""

_CAP: int = 64


@pytest.fixture(scope="module")
def wide_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """:data:`_GAMES` single-game shards in one worker directory."""
    root = tmp_path_factory.mktemp("book-cost")
    rig.plant(root, 0, rig.ordinary_games(_GAMES, seed=77), samples_per_shard=rig.PLIES)
    written = sorted(
        (root / "iter_0" / "shards" / "worker_0_r0").glob("shard-*.npz")
    )
    assert len(written) == _GAMES, (
        f"the fixture was meant to plant one shard per game; it planted "
        f"{len(written)} for {_GAMES} games"
    )
    return root


def test_the_builder_reads_columns_and_stays_inside_its_shard_budget(
    wide_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``ShardReader``, no whole row, and at most the budget's shards opened."""

    def _refuse_arrays(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the book builder called _load_shard_arrays, which pulls EVERY "
            "member of a shard out; §2.4's whole point is that it projects "
            "columns instead"
        )

    def _refuse_reader(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "the book builder constructed a ShardReader, whose construction "
            "decompresses and digest-verifies every shard in the directory"
        )

    monkeypatch.setattr(shards, "_load_shard_arrays", _refuse_arrays)
    monkeypatch.setattr(shards.ShardReader, "__init__", _refuse_reader)

    path = build_book(wide_tree, 1, 20260905, cap=_CAP, min_entries=32)
    assert path is not None, "the build did not produce a book at all"
    book = load_book(path)

    read = book.stats["target_shards_read"]
    assert read <= TARGET_SHARD_BUDGET, (
        f"the target pass opened {read} shards against a budget of "
        f"{TARGET_SHARD_BUDGET}; one shard's policy_target column is ~153 MB "
        "decompressed, so the budget is the whole cost bound"
    )
    assert read >= 1
    assert book.target_shard_budget == TARGET_SHARD_BUDGET
    # The games the budget put out of reach are counted once each, in games,
    # and never as sampler rejections (§2.2, R7).
    assert book.stats["skipped_budget_source_games"] == _GAMES - TARGET_SHARD_BUDGET
