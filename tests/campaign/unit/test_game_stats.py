"""KOM-150: the shard census, on shards written here rather than played.

:func:`komugi.campaign.game_stats.game_stats` is a pure function of a directory
of sealed shards, so every case below is a shard tree this module writes: a
decisive game, a game that ran into the turn cap, a booked game whose rows
start deep in the game, an empty tree, and a rotted shard. Real self-play
cannot produce those five on demand -- and a test that waited for a turn-cap
draw to occur naturally would be a test that runs for minutes and pins nothing.

Two schemas are used deliberately:

* a **four-column schema** for the arithmetic. The census reads exactly
  ``game_id``, ``ply``, ``mlh_target`` and ``termination`` through
  :func:`komugi.train.shards.read_columns`, which adopts each shard's *own*
  embedded schema, so a shard carrying only those four is the same read the
  production one gets at a fraction of the bytes;
* the **production** :data:`komugi.selfplay.records.SCHEMA` once, on three
  rows, because "the projection works on the shards self-play actually writes"
  is the claim the cheap schema cannot make.

The lengths are built from :data:`~komugi.selfplay.records.MLH_SCALE` the way
:func:`komugi.selfplay.worker.finalize_rows` builds them --
``mlh_target = (total_plies - ply) * MLH_SCALE`` -- so the reader is inverting
the writer's arithmetic rather than agreeing with a number this file invented.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from komugi.campaign.game_stats import (
    CAP_P_THRESHOLD,
    DRAFT_COLUMNS,
    DRAFT_PLANE_BLACK,
    DRAFT_PLANE_WHITE,
    EARLY_SHU_PLACEMENTS,
    STAT_COLUMNS,
    TURN_CAP_PLIES,
    GameStats,
    game_stats,
    stats_for_iteration,
)
from komugi.campaign.loop import shard_root
from komugi.engine.action_encoding import SHU_BLOCK
from komugi.engine.game import Game
from komugi.engine.observation import MAX_TURNS
from komugi.selfplay.records import MLH_SCALE, SCHEMA
from komugi.selfplay.worker import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.train.shards import (
    COUNT_KEY,
    SHARD_GLOB,
    FieldSpec,
    Schema,
    ShardWarning,
    ShardWriter,
)

_CENSUS_SCHEMA: Final[Schema] = Schema(
    fields=tuple(
        FieldSpec(name=name, dtype=dtype, shape=())
        for name, dtype in (
            ("game_id", "int64"),
            ("ply", "int64"),
            ("mlh_target", "float32"),
            ("termination", "int64"),
        )
    )
)
"""The four columns the census reads, and nothing else.

Its field *names* are the census's contract, so it is built from the same four
names :data:`~komugi.campaign.game_stats.STAT_COLUMNS` states; a rename on
either side fails ``test_the_census_schema_is_the_columns_it_reads`` rather
than silently reading zero games.
"""

_ROWS_PER_SHARD: Final[int] = 7
"""Small enough that every multi-ply game below spans several shards.

A game whose rows all land in one shard would let a per-shard grouping bug pass
-- and per-shard grouping is exactly what a worker's shard tree makes tempting.
"""


def _rows(
    *, game: int, total_plies: int, termination: int, first_ply: int = 0
) -> list[dict[str, torch.Tensor]]:
    """Return one game's rows, labelled as ``finalize_rows`` labels them.

    ``mlh_target`` is ``(total_plies - ply) * MLH_SCALE`` on every row, which
    is the identity the census inverts. `first_ply` is the booked case: a game
    that started from a draft-book prefix records rows from the ply the book
    left off at, so its row *count* is not its length.
    """
    return [
        {
            "game_id": torch.tensor(game, dtype=torch.int64),
            "ply": torch.tensor(ply, dtype=torch.int64),
            "mlh_target": torch.tensor(
                float(total_plies - ply) * MLH_SCALE, dtype=torch.float32
            ),
            "termination": torch.tensor(termination, dtype=torch.int64),
        }
        for ply in range(first_ply, total_plies)
    ]


def _write(
    directory: Path,
    rows: list[dict[str, torch.Tensor]],
    *,
    schema: Schema = _CENSUS_SCHEMA,
) -> Path:
    """Seal `rows` into `directory` as shards, and return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with ShardWriter(directory, schema, samples_per_shard=_ROWS_PER_SHARD) as writer:
        for row in rows:
            writer.append(row)
    return directory


def _production_rows(
    *, game: int, total_plies: int, termination: int
) -> list[dict[str, torch.Tensor]]:
    """The same game under the real 17-field schema, zeros everywhere else.

    Built by walking :data:`~komugi.selfplay.records.SCHEMA` rather than by
    listing its fields, so a future field lands here as a zero instead of as a
    test that no longer writes a valid row.
    """
    census = _rows(game=game, total_plies=total_plies, termination=termination)
    produced: list[dict[str, torch.Tensor]] = []
    for row in census:
        full = {
            field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
            for field in SCHEMA.fields
        }
        full.update(row)
        produced.append(full)
    return produced


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------


def test_a_decisive_game_and_a_capped_draw_are_counted_apart(tmp_path: Path) -> None:
    """The headline census: 2 games, 1 draw, 1 of them at the turn cap.

    A 10-ply decisive game and a cap-length draw -- the two ends of the
    distribution the previous architecture died in the middle of. The mean of
    ``(10 + cap) / 2`` is the number that matters most: it is neither game's
    length, so a reader that returned the first game's length, or the row
    count, or the last game's, cannot produce it.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _rows(game=1, total_plies=10, termination=TERMINATION_DECISIVE)
        + _rows(game=2, total_plies=TURN_CAP_PLIES, termination=TERMINATION_DRAW),
    )

    stats = game_stats(directory)

    assert stats.games == 2
    assert stats.draws == 1
    assert stats.turn_cap_games == 1
    assert stats.mean_plies == pytest.approx((10 + TURN_CAP_PLIES) / 2)
    assert stats.median_plies == pytest.approx((10 + TURN_CAP_PLIES) / 2)
    assert stats.max_plies == TURN_CAP_PLIES
    assert stats.draw_rate == pytest.approx(0.5)
    assert stats.cap_rate == pytest.approx(0.5)
    assert stats.rows == 10 + TURN_CAP_PLIES


def test_a_booked_game_is_as_long_as_its_labels_say(tmp_path: Path) -> None:
    """``ply + mlh_target / MLH_SCALE``, never the row count.

    A game that started from a draft-book prefix records its first row at ply
    20 and holds 40 rows, but it is a **60-ply** game -- the book's plies were
    played, they were simply not searched. A census that counted rows would
    report 40, and would under-report every booked iteration's game length by
    exactly the book depth.
    """
    directory = tmp_path / "worker_0_r0"
    rows = _rows(
        game=7, total_plies=60, termination=TERMINATION_DECISIVE, first_ply=20
    )
    assert len(rows) == 40, "the premise: the row count is not the length"
    _write(directory, rows)

    stats = game_stats(directory)

    assert stats.games == 1
    assert stats.rows == 40
    assert stats.mean_plies == pytest.approx(60.0)
    assert stats.median_plies == pytest.approx(60.0)
    assert stats.max_plies == 60
    assert stats.turn_cap_games == 0


def test_an_empty_tree_reads_as_zeros_and_divides_by_nothing(tmp_path: Path) -> None:
    """A wedged incarnation's empty directory is a finding, not an exception.

    Both rates are read, because they are the two places a games-of-zero census
    could raise ``ZeroDivisionError`` -- on a boundary that is stamping them
    onto a journal record, mid-campaign.
    """
    empty = tmp_path / "iter_3" / "shards" / "worker_0_r0"
    empty.mkdir(parents=True)

    stats = game_stats(empty)

    assert stats == GameStats(
        games=0,
        rows=0,
        draws=0,
        turn_cap_games=0,
        mean_plies=0.0,
        median_plies=0.0,
        max_plies=0,
        median_placements_w=0.0,
        median_placements_b=0.0,
        early_shu_games=0,
    )
    assert stats.draw_rate == 0.0
    assert stats.cap_rate == 0.0


def test_a_directory_that_does_not_exist_reads_as_zeros(tmp_path: Path) -> None:
    """A backfill over a root whose last iteration never sealed anything."""
    assert game_stats(tmp_path / "iter_99").games == 0


def test_the_row_count_is_the_sum_of_the_shards_own_counts(tmp_path: Path) -> None:
    """``rows`` is the shards' ``__count__`` sidecars, added up.

    The cross-check the self-play record gets for free: ``rows`` here and
    ``RunSummary.rows_written`` there are measured by different code over the
    same tree, so they agree or one of them is wrong.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _rows(game=1, total_plies=31, termination=TERMINATION_DECISIVE)
        + _rows(game=2, total_plies=17, termination=TERMINATION_DRAW),
    )
    shards = sorted(directory.rglob(SHARD_GLOB))
    counted = 0
    for shard in shards:
        with np.load(shard, allow_pickle=False) as loaded:
            counted += int(loaded[COUNT_KEY].item())

    assert len(shards) > 1, "the premise: the games span several shards"
    assert game_stats(directory).rows == counted == 31 + 17


# ---------------------------------------------------------------------------
# the tree, not the directory
# ---------------------------------------------------------------------------


def test_a_game_is_one_game_however_its_rows_are_sharded(tmp_path: Path) -> None:
    """Grouping is by ``game_id`` across the whole tree, not per directory.

    ``game_id`` packs ``(worker, incarnation, game_index)`` and is therefore
    unique across an iteration; a per-directory census would report this
    interleaved pair of workers as four games of half the length.
    """
    root = tmp_path / "shards"
    _write(
        root / "worker_0_r0",
        _rows(game=101, total_plies=40, termination=TERMINATION_DECISIVE)[:15],
    )
    _write(
        root / "worker_1_r0",
        _rows(game=101, total_plies=40, termination=TERMINATION_DECISIVE)[15:]
        + _rows(game=202, total_plies=80, termination=TERMINATION_DRAW),
    )

    stats = game_stats(root)

    assert stats.games == 2
    assert stats.rows == 40 + 80
    assert stats.max_plies == 80
    assert stats.mean_plies == pytest.approx(60.0)
    assert stats.draws == 1


def test_an_unreadable_shard_is_skipped_and_warned_about(tmp_path: Path) -> None:
    """One rotted shard costs its own rows, not the census.

    A diagnostic that aborted an iteration because a byte flipped would be
    worse than the gap it exists to close -- and the warning is a
    :class:`~komugi.train.shards.ShardWarning`, so a caller that would rather
    stop can escalate it.
    """
    directory = tmp_path / "worker_0_r0"
    _write(directory, _rows(game=1, total_plies=20, termination=TERMINATION_DRAW))
    torn = sorted(directory.rglob(SHARD_GLOB))[0]
    intact = torn.read_bytes()
    torn.write_bytes(intact[: len(intact) // 2])

    with pytest.warns(ShardWarning, match="skipping unreadable shard"):
        stats = game_stats(directory)

    assert stats.rows == 20 - _ROWS_PER_SHARD
    assert stats.games == 1


def test_shards_of_a_foreign_schema_are_not_censused(tmp_path: Path) -> None:
    """A tree that carries no ``game_id`` reads as zero games, loudly.

    Not silence: the tree exists and holds rows, so returning zeros without a
    word would look exactly like an iteration that played nothing.
    """
    directory = tmp_path / "worker_0_r0"
    foreign = Schema(fields=(FieldSpec(name="x", dtype="float32", shape=(2,)),))
    _write(
        directory,
        [{"x": torch.zeros(2)} for _ in range(3)],
        schema=foreign,
    )

    with pytest.warns(ShardWarning):
        assert game_stats(directory).games == 0


# ---------------------------------------------------------------------------
# the columns, the cap, and the campaign layout
# ---------------------------------------------------------------------------


def test_the_census_schema_is_the_columns_it_reads() -> None:
    """The four names this module writes are the four the census projects."""
    assert set(STAT_COLUMNS) == set(_CENSUS_SCHEMA.names)
    assert set(STAT_COLUMNS).issubset(set(SCHEMA.names))


def test_the_turn_cap_is_the_scale_the_labels_are_written_at() -> None:
    """The engine's cap, derived from ``MLH_SCALE`` rather than restated.

    Pinned against the leaf (KOM-165), not against a literal: the census's cap,
    the label scale and ``Game.max_turns()`` are one number, so a shard's
    ``mlh_target`` and this reader's ``cap_plies`` cannot disagree.
    """
    assert TURN_CAP_PLIES == round(1.0 / MLH_SCALE) == MAX_TURNS
    assert TURN_CAP_PLIES == Game.max_turns()


def test_the_cap_length_is_a_stated_parameter_not_an_assumption(
    tmp_path: Path,
) -> None:
    """``cap_plies`` decides which games are capped, and ``>=`` is the test.

    A game *past* the stated cap still counts as capped: a census run against a
    tree written under a different cap must not report the long games as
    uncapped merely because they are not exactly that length.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _rows(game=1, total_plies=64, termination=TERMINATION_DRAW)
        + _rows(game=2, total_plies=30, termination=TERMINATION_DECISIVE),
    )

    assert game_stats(directory, cap_plies=64).turn_cap_games == 1
    assert game_stats(directory, cap_plies=30).turn_cap_games == 2
    assert game_stats(directory, cap_plies=65).turn_cap_games == 0
    assert game_stats(directory).turn_cap_games == 0


def test_a_draw_that_is_not_at_the_cap_is_still_a_draw(tmp_path: Path) -> None:
    """``draws`` is the ``termination`` column; ``turn_cap_games`` is a length.

    They are two different measurements and the census must not derive one from
    the other -- the whole reason the pair is reported is that "13 draws, all
    of them at the cap" and "13 draws, none of them" are different diagnoses.
    """
    directory = tmp_path / "worker_0_r0"
    _write(directory, _rows(game=5, total_plies=44, termination=TERMINATION_DRAW))

    stats = game_stats(directory)

    assert (stats.draws, stats.turn_cap_games) == (1, 0)
    assert stats.draw_rate == pytest.approx(1.0)
    assert stats.cap_rate == 0.0


def test_the_production_schema_projects_the_same_four_columns(tmp_path: Path) -> None:
    """The census reads the shards self-play actually writes.

    Three rows under the real 17-field schema: the projection has to pick the
    four columns out of a row that is almost entirely ``observation`` and
    ``policy_target``, which is the read every production census makes.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _production_rows(game=3, total_plies=3, termination=TERMINATION_DRAW),
        schema=SCHEMA,
    )

    stats = game_stats(directory)

    assert stats.games == 1
    assert stats.rows == 3
    assert stats.draws == 1
    assert stats.mean_plies == pytest.approx(3.0)


def test_stats_for_iteration_reads_the_loops_own_shard_root(tmp_path: Path) -> None:
    """The campaign layout comes from ``shard_root``, not from a second copy.

    Written through :func:`~komugi.campaign.loop.shard_root` and read through
    :func:`~komugi.campaign.game_stats.stats_for_iteration`: if the two ever
    disagree about where iteration ``i`` lives, this reports zero games.
    """
    _write(
        shard_root(tmp_path, 4) / "worker_0_r0",
        _rows(game=9, total_plies=12, termination=TERMINATION_DECISIVE),
    )

    assert stats_for_iteration(tmp_path, 4).games == 1
    assert stats_for_iteration(tmp_path, 4).mean_plies == pytest.approx(12.0)
    assert stats_for_iteration(tmp_path, 5).games == 0, (
        "an iteration that has not run yet must read as empty, not as its "
        "neighbour"
    )


# ---------------------------------------------------------------------------
# the draft (KOM-156b): placements per side, and the early-終 rate
# ---------------------------------------------------------------------------

_PLANES: Final[int] = DRAFT_PLANE_BLACK + 1
"""Just enough observation planes to carry the two drafting-rights planes.

``(114, 1, 1)`` rather than the encoder's ``(114, 9, 9)``: the census reads the
plane's *value*, not its shape, and a 1x1 board keeps these shards small. The
production shape is exercised separately below, on the real schema.
"""

_DRAFT_SCHEMA: Final[Schema] = Schema(
    fields=(
        *_CENSUS_SCHEMA.fields,
        FieldSpec(name="to_play", dtype="int64", shape=()),
        FieldSpec(name="action", dtype="int64", shape=()),
        FieldSpec(name="observation", dtype="float32", shape=(_PLANES, 1, 1)),
    )
)
"""The census schema plus the three columns the draft half projects."""

_PLACE: Final[int] = 0
"""An ordinary (block A) action id -- a placement that does not end a draft."""

_ARATA: Final[int] = SHU_BLOCK[0] + 1
"""A C-block id played by a side that is **not** drafting: a plain hand drop.

Block C is context-dependent (:data:`~komugi.engine.action_encoding.SHU_BLOCK`),
so an id in that range is a 終 only while the mover's draft bit is set. Rows
like this one are what a census that tested the *id* instead of the id **and**
the mover's plane would miscount as draft finishes.
"""

_Move = tuple[int, bool, bool, int]
"""One row: ``(to_play, white is drafting, black is drafting, action)``."""


def _alternating_draft(
    *, white: int, black: int, white_shu: bool = True, black_shu: bool = True
) -> list[_Move]:
    """Return an interleaved draft in which each side makes its placements.

    White and black move in turn. A side still drafting *places* (its last
    placement is a 終 unless its ``*_shu`` flag says the rows run out first);
    a side that has finished keeps moving, and its rows deliberately carry a
    C-block id -- the arata drop that shares the 終 ids -- while the other
    side's drafting plane is still set. Those rows are the discriminator: a
    reader that ORed the two planes, or that read the action id alone, counts
    them as the wrong side's placements.
    """
    moves: list[_Move] = []
    placed_w = placed_b = 0
    while placed_w < white or placed_b < black:
        for mover in (0, 1):
            drafting_w = placed_w < white
            drafting_b = placed_b < black
            action = _ARATA
            if mover == 0 and drafting_w:
                placed_w += 1
                action = (
                    SHU_BLOCK[0] if (placed_w == white and white_shu) else _PLACE
                )
            elif mover == 1 and drafting_b:
                placed_b += 1
                action = (
                    SHU_BLOCK[0] if (placed_b == black and black_shu) else _PLACE
                )
            moves.append((mover, drafting_w, drafting_b, action))
    return moves


def _drafted_rows(
    *,
    game: int,
    moves: Sequence[_Move],
    termination: int = TERMINATION_DECISIVE,
) -> list[dict[str, torch.Tensor]]:
    """Return one game's rows under :data:`_DRAFT_SCHEMA`, one per move.

    The four census columns are built by :func:`_rows`, so the lengths are the
    writer's own arithmetic here too; the three draft columns are laid on top.
    """
    rows = _rows(game=game, total_plies=len(moves), termination=termination)
    drafted: list[dict[str, torch.Tensor]] = []
    for row, (mover, drafting_w, drafting_b, action) in zip(
        rows, moves, strict=True
    ):
        observation = torch.zeros((_PLANES, 1, 1), dtype=torch.float32)
        if drafting_w:
            observation[DRAFT_PLANE_WHITE] = 1.0
        if drafting_b:
            observation[DRAFT_PLANE_BLACK] = 1.0
        drafted.append(
            {
                **row,
                "to_play": torch.tensor(mover, dtype=torch.int64),
                "action": torch.tensor(action, dtype=torch.int64),
                "observation": observation,
            }
        )
    return drafted


def test_placements_are_counted_per_side_and_a_short_shu_is_early(
    tmp_path: Path,
) -> None:
    """The headline draft census, on the attractor campaign 2 fell into.

    Two games. In the first, white ends its draft on **3** placements while
    black goes on to 12 -- the shape the advisor measured by hand at iteration
    14. In the second both sides place 12. So the per-side medians are 7.5 and
    12, and exactly **one** of the two games is an early 終: white's, at 3
    placements against a threshold of 10. Black's 12-placement draft ends in a
    終 as well and is not early, which is the pair that makes this a per-side
    measurement rather than a count of 終 actions.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _drafted_rows(game=1, moves=_alternating_draft(white=3, black=12))
        + _drafted_rows(game=2, moves=_alternating_draft(white=12, black=12)),
        schema=_DRAFT_SCHEMA,
    )

    stats = game_stats(directory)

    assert stats.games == 2
    assert stats.median_placements_w == pytest.approx(7.5)
    assert stats.median_placements_b == pytest.approx(12.0)
    assert stats.early_shu_games == 1
    assert stats.early_shu_rate == pytest.approx(0.5)


def test_one_game_reports_each_sides_own_placement_count(tmp_path: Path) -> None:
    """A single game, so each median **is** that side's placements: 3 and 12.

    The two-game case above averages; this one cannot, so a reader that
    swapped the sides, or that counted every draft row as both sides', is red
    here and nowhere else.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _drafted_rows(game=11, moves=_alternating_draft(white=3, black=12)),
        schema=_DRAFT_SCHEMA,
    )

    stats = game_stats(directory)

    assert stats.median_placements_w == pytest.approx(3.0)
    assert stats.median_placements_b == pytest.approx(12.0)
    assert stats.early_shu_games == 1
    assert stats.early_shu_rate == pytest.approx(1.0)


def test_the_shu_row_is_itself_a_placement(tmp_path: Path) -> None:
    """A side's placements are its draft **rows**, the 終 row included.

    Stated in :class:`~komugi.campaign.game_stats.GameStats` and pinned here,
    because "three placements and then a 終" and "four placements, the last of
    them the 終" describe the same four rows -- and a curriculum minimum is
    compared against this number.
    """
    directory = tmp_path / "worker_0_r0"
    moves = _alternating_draft(white=4, black=4)
    white_draft_rows = [
        move for move in moves if move[0] == 0 and move[1]
    ]
    assert white_draft_rows[-1][3] == SHU_BLOCK[0], (
        "the premise: white's last draft row is the 終"
    )
    _write(directory, _drafted_rows(game=1, moves=moves), schema=_DRAFT_SCHEMA)

    assert game_stats(directory).median_placements_w == pytest.approx(
        len(white_draft_rows)
    )


def test_intro_shards_without_draft_columns_read_as_zero_placements(
    tmp_path: Path,
) -> None:
    """A tree whose shards carry no draft columns is not a tree with a defect.

    The four-column census schema declares no ``observation``, so the draft
    half has nothing to read and says zero -- silently, because a projection
    tree is a legitimate shape. The game shape is still censused, which is the
    point: the two halves are independent.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _rows(game=1, total_plies=30, termination=TERMINATION_DECISIVE)
        + _rows(game=2, total_plies=40, termination=TERMINATION_DRAW),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = game_stats(directory)

    assert stats.games == 2
    assert (stats.median_placements_w, stats.median_placements_b) == (0.0, 0.0)
    assert stats.early_shu_games == 0
    assert stats.early_shu_rate == 0.0


def test_intro_mode_games_that_never_draft_read_as_zero_placements(
    tmp_path: Path,
) -> None:
    """Both planes clear on every row: intro mode, where no draft is played.

    Zeros, not a gap. An intro-mode iteration really did place nothing, and
    the ramp gate reading ``0`` placements at ``0`` early 終 is reading the
    truth about it.
    """
    directory = tmp_path / "worker_0_r0"
    moves: list[_Move] = [
        (ply % 2, False, False, _PLACE) for ply in range(12)
    ]
    _write(directory, _drafted_rows(game=4, moves=moves), schema=_DRAFT_SCHEMA)

    stats = game_stats(directory)

    assert stats.games == 1
    assert (stats.median_placements_w, stats.median_placements_b) == (0.0, 0.0)
    assert stats.early_shu_games == 0


def test_a_draft_that_never_ended_is_not_an_early_shu(tmp_path: Path) -> None:
    """An unfinished draft is not a short one.

    Both sides place three pieces and the rows stop -- a game truncated by a
    killed worker, or a book that ate the rest. The placements are counted (3
    and 3, and they are *below* the threshold), but neither side's last draft
    row is a 終, so nothing ended early and the ramp gate must not fire.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _drafted_rows(
            game=5,
            moves=_alternating_draft(
                white=3, black=3, white_shu=False, black_shu=False
            ),
        ),
        schema=_DRAFT_SCHEMA,
    )

    stats = game_stats(directory)

    assert stats.median_placements_w == pytest.approx(3.0)
    assert stats.median_placements_b == pytest.approx(3.0)
    assert stats.early_shu_games == 0, (
        "a draft that has not ended within its rows is not an early 終"
    )


def test_the_threshold_is_strict_and_is_ten(tmp_path: Path) -> None:
    """``< EARLY_SHU_PLACEMENTS``, not ``<=``: exactly 10 is not early.

    The rung v1's curriculum started at. A side that meets the minimum
    exactly has satisfied it, and an off-by-one here would report a healthy
    iteration as the attractor and stall the ramp forever.
    """
    assert EARLY_SHU_PLACEMENTS == 10
    at = tmp_path / "at"
    below = tmp_path / "below"
    _write(
        at / "worker_0_r0",
        _drafted_rows(
            game=1, moves=_alternating_draft(white=EARLY_SHU_PLACEMENTS, black=12)
        ),
        schema=_DRAFT_SCHEMA,
    )
    _write(
        below / "worker_0_r0",
        _drafted_rows(
            game=1,
            moves=_alternating_draft(white=EARLY_SHU_PLACEMENTS - 1, black=12),
        ),
        schema=_DRAFT_SCHEMA,
    )

    assert game_stats(at).early_shu_games == 0
    assert game_stats(below).early_shu_games == 1


def test_a_games_draft_is_one_draft_however_its_rows_are_sharded(
    tmp_path: Path,
) -> None:
    """The draft half groups by ``game_id`` across the tree, like the other.

    One game's rows split across two worker directories, and its 終 lands in
    the second: a per-shard or per-directory draft census would see two short
    drafts, one of them unfinished, and would report an early 終 that never
    happened.
    """
    root = tmp_path / "shards"
    moves = _drafted_rows(game=77, moves=_alternating_draft(white=12, black=12))
    _write(root / "worker_0_r0", moves[: len(moves) // 2], schema=_DRAFT_SCHEMA)
    _write(root / "worker_1_r0", moves[len(moves) // 2 :], schema=_DRAFT_SCHEMA)

    stats = game_stats(root)

    assert stats.games == 1
    assert stats.median_placements_w == pytest.approx(12.0)
    assert stats.early_shu_games == 0


def test_the_draft_columns_are_the_production_schemas_own() -> None:
    """The three names are fields of the shards self-play actually writes.

    Read off :data:`~komugi.selfplay.records.SCHEMA` rather than asserted
    against a list here: a rename of ``to_play``, ``action`` or
    ``observation`` would otherwise leave the draft census silently reading no
    shards at all.
    """
    assert set(DRAFT_COLUMNS).issubset(set(SCHEMA.names))
    assert DRAFT_PLANE_WHITE < DRAFT_PLANE_BLACK < _PLANES


def test_the_production_observation_is_read_at_its_own_shape(
    tmp_path: Path,
) -> None:
    """``(114, 9, 9)`` planes, filled the way the encoder fills them.

    The cheap ``(114, 1, 1)`` shards above cannot make this claim: a reader
    that indexed the plane axis wrongly, or that expected a flat vector, reads
    every production row as "not drafting" and reports an iteration of
    zero-placement games as though nothing drafted.
    """
    directory = tmp_path / "worker_0_r0"
    moves = _alternating_draft(white=3, black=3)
    rows = _production_rows(
        game=2, total_plies=len(moves), termination=TERMINATION_DECISIVE
    )
    for row, (mover, drafting_w, drafting_b, action) in zip(
        rows, moves, strict=True
    ):
        if drafting_w:
            row["observation"][DRAFT_PLANE_WHITE, :, :] = 1.0
        if drafting_b:
            row["observation"][DRAFT_PLANE_BLACK, :, :] = 1.0
        row["to_play"] = torch.tensor(mover, dtype=torch.int64)
        row["action"] = torch.tensor(action, dtype=torch.int64)
    _write(directory, rows, schema=SCHEMA)

    stats = game_stats(directory)

    assert stats.games == 1
    assert stats.median_placements_w == pytest.approx(3.0)
    assert stats.median_placements_b == pytest.approx(3.0)
    assert stats.early_shu_games == 1, (
        "both sides ended their drafts on three placements"
    )


# ---------------------------------------------------------------------------
# KOM-245 -- the cap labels
# ---------------------------------------------------------------------------


_LABELLED_SCHEMA: Final[Schema] = Schema(
    fields=_CENSUS_SCHEMA.fields
    + (FieldSpec(name="wdl_target", dtype="float32", shape=(3,)),)
)
"""The census four plus the one column the cap labels are read from.

Deliberately not the production schema: this half of the census reads three
float32 per row, and paying 37 KB of observation per row to write them would
make a cap-length game a megabyte for no extra claim.
"""

_SHORT_CAP: Final[int] = 20
"""The cap these games are censused against.

``game_stats(cap_plies=...)`` is a stated parameter, so a test does not have to
write 512 rows per game to have a turn-capped one.
"""


def _labelled_rows(
    *, game: int, total_plies: int, termination: int, p: float
) -> list[dict[str, torch.Tensor]]:
    """One game's rows with KOM-245's label: leader ``(p, 1-p, 0)``, else mirrored.

    Movers alternate by ply, which is enough here -- the per-row frame is
    ``finalize_rows``' business and is pinned in its own suite. ``p = 0.0``
    writes the plain ``(0, 1, 0)`` draw an un-adjudicated cap game carries.
    """
    leader = [p, 1.0 - p, 0.0]
    trailer = [0.0, 1.0 - p, p]
    rows = _rows(game=game, total_plies=total_plies, termination=termination)
    for ply, row in enumerate(rows):
        row["wdl_target"] = torch.tensor(
            leader if ply % 2 == 0 else trailer, dtype=torch.float32
        )
    return rows


def test_an_adjudicated_cap_game_and_a_plain_one_are_counted_apart(
    tmp_path: Path,
) -> None:
    """One cap game at ``p = 0.5``, one at ``(0, 1, 0)``: 1 game, mean 0.25.

    The mean is over **all** cap games -- the plain draw counts ``0`` and stays
    in the denominator -- which is the definition the plan's 0.25 is stated on
    (review F8). The share is K1's gate, and 0.5 here is one of the two games.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _labelled_rows(
            game=1, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW, p=0.5
        )
        + _labelled_rows(
            game=2, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW, p=0.0
        ),
        schema=_LABELLED_SCHEMA,
    )

    stats = game_stats(directory, cap_plies=_SHORT_CAP)

    assert stats.turn_cap_games == 2
    assert stats.cap_adjudicated_games == 1
    assert stats.cap_p_mean == pytest.approx(0.25)
    assert stats.cap_p_ge_03_share == pytest.approx(0.5)


def test_a_decisive_game_is_not_in_the_cap_denominator(tmp_path: Path) -> None:
    """The three numbers are about CAP games; a short decisive game is not one.

    Its rows carry a one-hot label, so a census that read every game's ``p``
    would report a mean of ~0.5 and a share of 1.0 on an iteration with no
    adjudication in it at all.
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _labelled_rows(
            game=1, total_plies=8, termination=TERMINATION_DECISIVE, p=1.0
        )
        + _labelled_rows(
            game=2, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW, p=0.4
        ),
        schema=_LABELLED_SCHEMA,
    )

    stats = game_stats(directory, cap_plies=_SHORT_CAP)

    assert stats.turn_cap_games == 1
    assert stats.cap_adjudicated_games == 1
    assert stats.cap_p_mean == pytest.approx(0.4)
    assert stats.cap_p_ge_03_share == pytest.approx(1.0)


_A_ROUNDING_BELOW_03: Final[float] = float(np.float32(0.3 - 4e-7))
"""The largest ``p`` the tolerance is *for*: below 0.3, by less than 1e-6.

``np.float32(0.3)`` is ``0.30000001...`` -- **above** 0.3 -- so a stored 0.3
would clear a bare ``>=`` and says nothing about the slack. What the tolerance
decides is the value a rounding *under* the threshold: a ``p`` the worker
computed as 0.3 from a ``|d|`` that divides exactly, then lost a ulp to
float32, must not drop out of K1's gate.
"""


def test_a_cap_game_a_rounding_below_the_threshold_is_still_in_the_share(
    tmp_path: Path,
) -> None:
    """The tolerance is real: 0.3 minus a rounding counts, 0.29 does not.

    Both sides of the slack, not both sides of the threshold (review F3): the
    0.3/0.2 pair this replaces passed with the tolerance deleted, because
    float32 rounds 0.3 *up*. ``_A_ROUNDING_BELOW_03`` is red under a bare
    ``>=``, and 0.29 -- a real 0.01 below -- stays out, so a tolerance widened
    into a second threshold is red too.
    """
    assert _A_ROUNDING_BELOW_03 < CAP_P_THRESHOLD, (
        "the premise: this value must really be below the threshold, or the "
        "test asserts nothing about the tolerance"
    )
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _labelled_rows(
            game=1, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW, p=0.29
        )
        + _labelled_rows(
            game=2,
            total_plies=_SHORT_CAP,
            termination=TERMINATION_DRAW,
            p=_A_ROUNDING_BELOW_03,
        ),
        schema=_LABELLED_SCHEMA,
    )

    stats = game_stats(directory, cap_plies=_SHORT_CAP)

    assert stats.cap_adjudicated_games == 2
    assert stats.cap_p_mean == pytest.approx((0.29 + _A_ROUNDING_BELOW_03) / 2)
    assert stats.cap_p_ge_03_share == pytest.approx(0.5), (
        "a p a single float32 rounding below 0.3 must count as 0.3, and 0.29 "
        "must not; a bare >= on the stored value fails the first"
    )


def test_game_stats_without_a_wdl_column_reports_zero_adjudicated(
    tmp_path: Path,
) -> None:
    """A shard schema with no label column reads as zeros, not as damage.

    The unit census schema here, and every shard written before the column
    existed: ``_draft_rows`` tolerates a projection the same way, and "no
    stored label" really is "no adjudication" (review F8).
    """
    directory = tmp_path / "worker_0_r0"
    _write(
        directory,
        _rows(game=1, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW)
        + _rows(game=2, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ShardWarning)
        stats = game_stats(directory, cap_plies=_SHORT_CAP)

    assert stats.turn_cap_games == 2
    assert stats.cap_adjudicated_games == 0
    assert stats.cap_p_mean == 0.0
    assert stats.cap_p_ge_03_share == 0.0


def test_a_tree_that_mixes_a_labelled_shard_with_a_projection_stays_aligned(
    tmp_path: Path,
) -> None:
    """One worker's shards carry labels, another's do not: no row misalignment.

    The failure this pins is a tree-level concatenation that dropped the
    unlabelled shards' rows instead of padding them -- which would shift every
    later game's label onto the wrong game.
    """
    _write(
        tmp_path / "worker_0_r0",
        _labelled_rows(
            game=1, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW, p=0.8
        ),
        schema=_LABELLED_SCHEMA,
    )
    _write(
        tmp_path / "worker_1_r0",
        _rows(game=2, total_plies=_SHORT_CAP, termination=TERMINATION_DRAW),
    )

    stats = game_stats(tmp_path, cap_plies=_SHORT_CAP)

    assert stats.games == 2
    assert stats.turn_cap_games == 2
    assert stats.cap_adjudicated_games == 1
    assert stats.cap_p_mean == pytest.approx(0.4)
    assert stats.cap_p_ge_03_share == pytest.approx(0.5)
