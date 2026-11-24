"""The self-play half of a campaign's history: record, file, and file choice.

:func:`~komugi.campaign.loop.run_campaign` appends one
:class:`~komugi.campaign.telemetry.SelfPlayTelemetry` per iteration the moment
:func:`~komugi.selfplay.orchestrate.run_selfplay` returns (KOM-042). The
behavioural suite pins that a *real* campaign's numbers come back off disk;
what is pinned here is everything that can be wrong without spawning a single
process:

* the record survives the round trip through the file with its values intact,
  including the ones a JSON encoder is happy to mangle (a whole-number wall
  time, a zero count);
* **the file choice**, which is the design decision this ticket had to make.
  ``komugi.telemetry.journal.read_records`` does *not* pass over an unknown
  ``kind``: :func:`~komugi.telemetry.journal.decode_line` raises on it and the
  reader counts it in the same ``skipped`` tally as a corrupt or
  crash-truncated line. The root-journal test below is that fact, executed
  rather than asserted in a docstring, and it is the whole reason the
  telemetry lives in a sibling file;
* **append-only**, including across a resume: the second run adds its
  iterations' lines and leaves the first run's bytes exactly where they were.
  A telemetry file that a resume rewrites is a telemetry file that a resume
  can lose;
* the reader's tolerance for a torn trailing line, which is the crash this
  whole record exists to survive.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import pytest

from komugi.campaign.loop import journal_path
from komugi.campaign.telemetry import (
    DRAFT_SINCE,
    GAME_SHAPE_SINCE,
    NOT_MEASURED,
    SELFPLAY_JOURNAL_NAME,
    SELFPLAY_KIND,
    SELFPLAY_READABLE_VERSIONS,
    SELFPLAY_SCHEMA_VERSION,
    SelfPlayTelemetry,
    append_selfplay,
    decode_selfplay_line,
    encode_selfplay,
    read_selfplay,
    selfplay_journal_path,
)
from komugi.telemetry.journal import (
    JournalDecodeError,
    encode_record,
    read_records,
    write_record,
)
from komugi.telemetry.records import IterationMetrics


def _telemetry(iteration: int = 0, **overrides: Any) -> SelfPlayTelemetry:
    """A plausible record, with fields overridable one at a time."""
    fields: dict[str, Any] = {
        "iteration": iteration,
        "games": 4,
        "rows": 611,
        "restarts": 0,
        "shard_dirs": 2,
        "wall_s": 41.75,
        # The KOM-150 game shape: one of the four games was a drawn 256-ply
        # turn cap, which is campaign 1's own ratio in miniature.
        "draws": 1,
        "turn_cap_games": 1,
        "mean_plies": 115.5,
        "max_plies": 256,
        # The KOM-156b draft shape: both sides drafting most of their hand,
        # and two of the four games ending a draft early -- campaign 2's
        # attractor caught mid-slide rather than at the bottom.
        "median_placements_w": 11.0,
        "median_placements_b": 12.0,
        "early_shu_games": 2,
    }
    fields.update(overrides)
    return SelfPlayTelemetry(**fields)


def _trainer_record(iteration: int) -> IterationMetrics:
    """What ``training_iteration`` appends to the root journal, in miniature.

    ``games=0`` is not a placeholder for this test's convenience: it is what
    the trainer really writes, because it never meets a ``RunSummary``. That
    zero is the gap KOM-042 fills.
    """
    return IterationMetrics(
        iteration=iteration,
        games=0,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras={"steps": 8.0},
    )


# ---------------------------------------------------------------------------
# the record and its wire form
# ---------------------------------------------------------------------------


def test_a_record_round_trips_through_the_file(tmp_path: Path) -> None:
    """Every field written comes back equal, through the real append + read."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    record = _telemetry(3, games=9, rows=1234, restarts=2, shard_dirs=5, wall_s=61.5)

    append_selfplay(path, record)
    read = read_selfplay(path)

    assert read.ok and len(read) == 1, f"skipped {read.skipped} line(s)"
    assert read.records[0] == record


def test_the_line_carries_the_documented_fields(tmp_path: Path) -> None:
    """One JSON object per line, tagged and versioned, counts as integers.

    The tag and the version are what make a line self-describing when it is
    grepped out of a dead campaign's directory months later; the integer types
    are what keep ``games`` from coming back as ``4.0`` and comparing unequal
    to the ``IterationResult`` it is supposed to match.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    append_selfplay(path, _telemetry(1))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])

    assert payload["kind"] == SELFPLAY_KIND
    assert payload["v"] == SELFPLAY_SCHEMA_VERSION
    assert payload["iteration"] == 1
    assert payload["games"] == 4
    assert payload["rows"] == 611
    assert payload["restarts"] == 0
    assert payload["shard_dirs"] == 2
    assert math.isclose(payload["wall_s"], 41.75)
    assert payload["draws"] == 1
    assert payload["turn_cap_games"] == 1
    assert math.isclose(payload["mean_plies"], 115.5)
    assert payload["max_plies"] == 256
    assert math.isclose(payload["median_placements_w"], 11.0)
    assert math.isclose(payload["median_placements_b"], 12.0)
    assert payload["early_shu_games"] == 2
    for name in (
        "iteration",
        "games",
        "rows",
        "restarts",
        "shard_dirs",
        "draws",
        "turn_cap_games",
        "max_plies",
        "early_shu_games",
    ):
        assert isinstance(payload[name], int), f"{name} must stay an integer"


def test_a_whole_number_wall_time_survives_as_a_number(tmp_path: Path) -> None:
    """``wall_s=3.0`` encodes and decodes as a float, not as a truthy int.

    JSON has one number type, so a whole-number duration is written ``3.0``
    and could plausibly come back as ``3``. It must still be usable as a
    duration -- the decoder accepts either and returns a float.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    append_selfplay(path, _telemetry(0, wall_s=3.0))
    (restored,) = read_selfplay(path).records
    assert isinstance(restored.wall_s, float)
    assert restored.wall_s == pytest.approx(3.0)


def test_a_zero_iteration_with_no_games_is_a_legal_record() -> None:
    """A self-play run that produced nothing is a *finding*, not an error.

    The campaign raises on an empty window one step later, and this record is
    what tells an operator which of the two halves was empty. Refusing to
    represent it would delete the evidence.
    """
    record = _telemetry(
        0,
        games=0,
        rows=0,
        shard_dirs=0,
        wall_s=0.0,
        draws=0,
        turn_cap_games=0,
        mean_plies=0.0,
        max_plies=0,
        median_placements_w=0.0,
        median_placements_b=0.0,
        early_shu_games=0,
    )
    assert decode_selfplay_line(encode_selfplay(record)) == record
    assert record.draw_rate == 0.0, "no games is no draw rate, not a crash"
    assert record.early_shu_rate == 0.0, "nor an early-終 rate"


@pytest.mark.parametrize(
    "overrides",
    [
        {"iteration": -1},
        {"games": -1},
        {"rows": -1},
        {"restarts": -1},
        {"shard_dirs": -1},
        {"wall_s": -0.5},
        {"wall_s": math.nan},
        {"wall_s": math.inf},
        {"draws": -2},
        {"turn_cap_games": -2},
        {"max_plies": -2},
        {"mean_plies": -1.0},
        {"mean_plies": math.inf},
        {"early_shu_games": -2},
        {"median_placements_w": -1.0},
        {"median_placements_b": math.inf},
    ],
)
def test_impossible_numbers_are_refused_where_they_are_built(
    overrides: dict[str, Any],
) -> None:
    """Validation at construction, in the spirit of ``komugi.telemetry.records``.

    A ``NaN`` duration is the one that matters most: it would compare ``False``
    against every threshold a future throughput gate sets, so a poisoned record
    would read as "fine" forever.
    """
    # No positional argument: one of the cases overrides ``iteration`` itself,
    # and a positional would collide with it as a TypeError before the
    # record's own validation ever ran.
    with pytest.raises(ValueError):
        _telemetry(**overrides)


# ---------------------------------------------------------------------------
# the file choice: why a sibling and not a third kind in journal.jsonl
# ---------------------------------------------------------------------------


def test_a_selfplay_line_in_the_root_journal_reads_as_corruption(
    tmp_path: Path,
) -> None:
    """The measurement that decided the design: unknown kinds are *not* tolerated.

    ``komugi.telemetry.journal``'s docstring says the reader "quietly passes
    over the other kinds", and it does -- for the kinds in its decoder table.
    A ``selfplay`` line is not one of them, so it lands in ``skipped``: the
    same counter that reports a crash-truncated line, and the one
    ``tests/train/behavior/test_checkpoint_journal.py`` requires to be zero.
    Sharing the root journal would therefore have made every healthy campaign
    report a damaged line per iteration.

    If a future change teaches :mod:`komugi.telemetry.journal` a ``selfplay``
    decoder, this test fails -- and that failure is the signal that the
    sibling file can be merged into the journal.
    """
    path = tmp_path / "journal.jsonl"
    path.write_text(
        encode_record(_trainer_record(0))
        + "\n"
        + encode_selfplay(_telemetry(0))
        + "\n",
        encoding="utf-8",
    )

    read = read_records(path, IterationMetrics)

    assert [record.iteration for record in read.records] == [0]
    assert read.skipped == 1, (
        "the root journal reader now decodes a selfplay line; the sibling "
        "file exists only because it did not"
    )


def test_the_split_leaves_both_files_readable_by_their_own_reader(
    tmp_path: Path,
) -> None:
    """The layout as run_campaign writes it: two files, each clean on its own.

    This is the positive half of the finding above. The trainer's journal
    holds only kinds its reader knows (``skipped == 0``), and the self-play
    numbers -- the ones that journal reports as ``games=0`` -- are complete in
    the sibling.
    """
    root = tmp_path / "campaign"
    root.mkdir()
    for iteration in range(2):
        append_selfplay(selfplay_journal_path(root), _telemetry(iteration))
        write_record(journal_path(root), _trainer_record(iteration))

    trainer = read_records(journal_path(root), IterationMetrics)
    self_play = read_selfplay(selfplay_journal_path(root))

    assert trainer.ok and [record.games for record in trainer.records] == [0, 0]
    assert self_play.ok
    assert [record.iteration for record in self_play.records] == [0, 1]
    assert all(record.games > 0 for record in self_play.records)


def test_the_sibling_sits_next_to_the_journal_and_is_not_it(tmp_path: Path) -> None:
    """Same directory, different file: an audit finds both by listing ``root``."""
    assert selfplay_journal_path(tmp_path) == tmp_path / SELFPLAY_JOURNAL_NAME
    assert selfplay_journal_path(tmp_path).parent == journal_path(tmp_path).parent
    assert selfplay_journal_path(tmp_path) != journal_path(tmp_path)


def test_a_foreign_kind_or_version_in_the_sibling_is_not_a_record(
    tmp_path: Path,
) -> None:
    """This file is single-kind, so a foreign line there is a wrong file.

    The mirror image of the root journal's decision, and deliberate: nothing
    else writes here, so "a line of another kind" means the path was pointed
    at something it should not have been.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    path.write_text(
        encode_record(_trainer_record(0))
        + "\n"
        + json.dumps({"kind": SELFPLAY_KIND, "v": SELFPLAY_SCHEMA_VERSION + 1})
        + "\n"
        + encode_selfplay(_telemetry(7))
        + "\n",
        encoding="utf-8",
    )

    read = read_selfplay(path)

    assert [record.iteration for record in read.records] == [7]
    assert read.skipped == 2


@pytest.mark.parametrize(
    "line",
    [
        '{"kind":"selfplay","v":1,"iteration":0,"games":4,"rows":6,"restarts":0}',
        '{"kind":"selfplay","v":1,"iteration":true,"games":4,"rows":6,'
        '"restarts":0,"shard_dirs":1,"wall_s":1.0}',
        '{"kind":"selfplay","v":1,"iteration":0,"games":4.5,"rows":6,'
        '"restarts":0,"shard_dirs":1,"wall_s":1.0}',
        '{"kind":"selfplay","v":1,"iteration":0,"games":-4,"rows":6,'
        '"restarts":0,"shard_dirs":1,"wall_s":1.0}',
        "[1, 2, 3]",
        "{not json",
    ],
)
def test_unusable_lines_raise_the_journals_own_error(line: str) -> None:
    """One error type for both files, so an auditor reading both catches once.

    A missing field, a bool where a count belongs, a float game count, a
    value the record itself rejects, a non-object and a truncation are all
    the same thing to a reader: a line that is not a record.
    """
    with pytest.raises(JournalDecodeError):
        decode_selfplay_line(line)


# ---------------------------------------------------------------------------
# append-only, across a crash and across a resume
# ---------------------------------------------------------------------------


def test_a_torn_trailing_line_costs_one_record_not_the_history(
    tmp_path: Path,
) -> None:
    """A campaign killed mid-write keeps every record it finished.

    The file is fsynced per record, but no fsync makes a line atomic: a kill
    between the first byte and the newline leaves a partial one. Raising there
    would take the whole run's self-play history with it, which is precisely
    the failure this record exists to prevent.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    append_selfplay(path, _telemetry(0))
    append_selfplay(path, _telemetry(1))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encode_selfplay(_telemetry(2))[:20])

    read = read_selfplay(path)

    assert [record.iteration for record in read.records] == [0, 1]
    assert read.skipped == 1
    assert not read.ok


def test_a_missing_file_is_empty_only_when_asked(tmp_path: Path) -> None:
    """``missing_ok`` distinguishes "died before iteration 0" from a typo'd path."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    empty = read_selfplay(path, missing_ok=True)
    assert len(empty) == 0 and empty.ok
    with pytest.raises(FileNotFoundError):
        read_selfplay(path)


def test_appending_never_disturbs_the_lines_already_written(tmp_path: Path) -> None:
    """Byte-for-byte: line ``i`` is a prefix-preserving addition, not a rewrite."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    append_selfplay(path, _telemetry(0))
    first = path.read_bytes()
    append_selfplay(path, _telemetry(1))
    second = path.read_bytes()

    assert second.startswith(first)
    assert len(second) > len(first)
    assert second.count(b"\n") == 2


def test_a_resume_appends_only_the_iterations_it_ran(tmp_path: Path) -> None:
    """The resume shape, at unit scale: no rewriting, no duplicates.

    ``run_campaign`` writes a line only inside the loop over the iterations it
    actually runs (``range(resume_point, iterations)``), so a campaign resumed
    after iteration 0 contributes iteration 1 and nothing else. The failure
    this forbids is a resumed run that re-emits the recovered iterations --
    which would double every count an audit sums, silently and plausibly.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    for iteration in range(2):  # the run that died after iteration 1
        append_selfplay(path, _telemetry(iteration))
    before = path.read_bytes()

    for iteration in range(2, 4):  # the resumed run: iterations 2 and 3 only
        append_selfplay(path, _telemetry(iteration))

    read = read_selfplay(path)
    iterations = [record.iteration for record in read.records]
    assert iterations == [0, 1, 2, 3]
    assert len(iterations) == len(set(iterations)), f"duplicate records: {iterations}"
    assert read.ok
    assert path.read_bytes().startswith(before), "the resume rewrote earlier records"


def test_a_genuinely_rerun_iteration_appends_a_second_line(tmp_path: Path) -> None:
    """The file is a log of self-play *runs*, not a table keyed by iteration.

    A campaign that dies after sealing iteration 0's shards but before
    publishing its checkpoint re-runs iteration 0 into a cleaned tree, and the
    second attempt is a second self-play run: it gets its own line. Collapsing
    it into the first (or refusing to write it) would hide the re-run, which
    is exactly the event an operator is looking for when a campaign's wall
    clock does not match its iteration count. An auditor summing rows takes
    the last line per iteration, because the earlier attempt's shards were
    deleted.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    append_selfplay(path, _telemetry(0, rows=100, wall_s=10.0))
    append_selfplay(path, _telemetry(0, rows=104, wall_s=11.0))

    read = read_selfplay(path)

    assert [record.iteration for record in read.records] == [0, 0]
    assert [record.rows for record in read.records] == [100, 104], (
        "records must come back in write order, so the surviving attempt is "
        "the last one"
    )


# ---------------------------------------------------------------------------
# KOM-150: the game shape, and the v1 history that predates it
# ---------------------------------------------------------------------------

_V1_LINE: Final[str] = (
    '{"games":8,"iteration":11,"kind":"selfplay","restarts":0,"rows":611,'
    '"shard_dirs":2,"v":1,"wall_s":41.7}'
)
"""One line of campaign 1's ``selfplay.jsonl``, in the shape it was written.

Written out literally rather than produced by an encoder this ticket controls:
the claim being pinned is about **bytes already on disk**, and an encoder that
regressed would produce a "v1 line" that agrees with itself and with nothing in
the campaign root it is supposed to keep readable.
"""


def test_a_v2_record_round_trips_with_its_game_shape(tmp_path: Path) -> None:
    """The four new fields survive the file, as numbers rather than as text."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    record = _telemetry(
        2, games=100, draws=13, turn_cap_games=13, mean_plies=115.25, max_plies=256
    )

    append_selfplay(path, record)
    (restored,) = read_selfplay(path).records

    assert restored == record
    assert (restored.draws, restored.turn_cap_games) == (13, 13)
    assert restored.mean_plies == pytest.approx(115.25)
    assert restored.max_plies == 256
    assert restored.measured
    assert restored.draw_rate == pytest.approx(0.13)


def test_a_v1_line_still_decodes_and_says_it_was_never_measured() -> None:
    """Campaign 1's history stays readable, and stays honest about the gap.

    The four game-shape fields come back as sentinels -- ``-1`` and ``NaN`` --
    rather than as zeros. A zero would read as "that campaign drew nothing and
    played zero-ply games", which is the opposite of the truth: it drew 13 of
    840 and nobody wrote it down.
    """
    record = decode_selfplay_line(_V1_LINE)

    assert (record.iteration, record.games, record.rows) == (11, 8, 611)
    assert record.shard_dirs == 2 and record.restarts == 0
    assert record.wall_s == pytest.approx(41.7)
    assert record.draws == NOT_MEASURED
    assert record.turn_cap_games == NOT_MEASURED
    assert record.max_plies == NOT_MEASURED
    assert math.isnan(record.mean_plies)
    assert not record.measured
    assert math.isnan(record.draw_rate), (
        "an unmeasured draw rate must not compare equal to a measured 0.0"
    )


def test_a_v1_file_reads_clean_through_the_reader(tmp_path: Path) -> None:
    """A whole v1 log is history, not a file of corrupt lines.

    ``read_selfplay`` counts an undecodable line in ``skipped`` -- the tally a
    healthy campaign requires to be zero -- so a version bump that could not
    read v1 would report every line of campaign 1's telemetry as damage.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    path.write_text(_V1_LINE + "\n" + _V1_LINE + "\n", encoding="utf-8")

    read = read_selfplay(path)

    assert read.ok and len(read) == 2
    assert all(not record.measured for record in read.records)


def test_the_two_versions_read_out_of_one_file(tmp_path: Path) -> None:
    """The real shape of a campaign resumed across this ticket.

    A root written by the old code and appended to by the new holds both
    versions, in order, and the reader must return both -- the resumed
    iterations measured, the earlier ones not.
    """
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    path.write_text(_V1_LINE + "\n", encoding="utf-8")
    append_selfplay(path, _telemetry(12))

    read = read_selfplay(path)

    assert read.ok and [record.iteration for record in read.records] == [11, 12]
    assert [record.measured for record in read.records] == [False, True]


def test_a_version_the_reader_does_not_know_is_still_refused() -> None:
    """v4 is an undecodable line, not a record with fields left at zero.

    The widening covers the versions that were actually written and no more. A
    future schema that this code cannot read must not be silently
    half-decoded, which is the whole reason the version is on the line.
    """
    future = json.loads(encode_selfplay(_telemetry(0)))
    future["v"] = SELFPLAY_SCHEMA_VERSION + 1

    assert SELFPLAY_SCHEMA_VERSION + 1 not in SELFPLAY_READABLE_VERSIONS
    with pytest.raises(JournalDecodeError):
        decode_selfplay_line(json.dumps(future))


def test_a_v2_line_missing_a_game_shape_field_is_not_a_record() -> None:
    """A line claiming v2 must carry v2's fields; absence is corruption.

    The mirror of the v1 tolerance: the sentinels are for lines that never
    promised the measurement, never for a v2 line that promised it and did not
    deliver.
    """
    payload = json.loads(encode_selfplay(_telemetry(0)))
    for missing in ("draws", "turn_cap_games", "mean_plies", "max_plies"):
        truncated = {key: value for key, value in payload.items() if key != missing}
        with pytest.raises(JournalDecodeError):
            decode_selfplay_line(json.dumps(truncated))


def test_a_not_measured_record_cannot_be_written_back_out() -> None:
    """Reading v1 is history; re-writing it as v2 would be a fabrication.

    ``encode_selfplay`` refuses, naming the reason. Without this a tool that
    read a v1 file and appended it elsewhere would turn "never measured" into
    "measured as NaN" -- or crash inside ``json.dumps`` with a message about a
    token, which tells the operator nothing.
    """
    record = decode_selfplay_line(_V1_LINE)
    with pytest.raises(ValueError, match="not-measured"):
        encode_selfplay(record)


def test_the_game_shape_has_no_defaults() -> None:
    """A construction site that forgets to measure fails at the call.

    A default would let a forgotten site write a plausible zero into the file
    for the rest of the campaign's life, which is precisely the failure this
    ticket exists to end. The draft three are held to the same rule as the
    game-shape four.
    """
    for missing in (
        "draws",
        "turn_cap_games",
        "mean_plies",
        "max_plies",
        "median_placements_w",
        "median_placements_b",
        "early_shu_games",
    ):
        fields: dict[str, Any] = {
            "iteration": 0,
            "games": 4,
            "rows": 611,
            "restarts": 0,
            "shard_dirs": 2,
            "wall_s": 41.75,
            "draws": 1,
            "turn_cap_games": 1,
            "mean_plies": 115.5,
            "max_plies": 256,
            "median_placements_w": 11.0,
            "median_placements_b": 12.0,
            "early_shu_games": 2,
        }
        del fields[missing]
        with pytest.raises(TypeError):
            SelfPlayTelemetry(**fields)


# ---------------------------------------------------------------------------
# KOM-156b: the draft, and the v2 history that predates it
# ---------------------------------------------------------------------------

_V2_LINE: Final[str] = (
    '{"draws":1,"games":8,"iteration":14,"kind":"selfplay","max_plies":256,'
    '"mean_plies":115.5,"restarts":0,"rows":611,"shard_dirs":2,'
    '"turn_cap_games":1,"v":2,"wall_s":41.7}'
)
"""One line of campaign 2's ``selfplay.jsonl``, in the shape it was written.

Literal, for the reason :data:`_V1_LINE` is literal: the claim is about bytes
already on disk. Iteration 14 is the one the draft attractor was diagnosed at
-- and this line is exactly what the diagnosis could *not* be made from.
"""


def test_a_v3_record_round_trips_with_its_draft(tmp_path: Path) -> None:
    """The three draft fields survive the file, as numbers rather than text."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    record = _telemetry(
        14,
        games=200,
        median_placements_w=4.0,
        median_placements_b=4.5,
        early_shu_games=44,
    )

    append_selfplay(path, record)
    (restored,) = read_selfplay(path).records

    assert restored == record
    assert restored.median_placements_w == pytest.approx(4.0)
    assert restored.median_placements_b == pytest.approx(4.5)
    assert restored.early_shu_games == 44
    assert restored.draft_measured
    assert restored.early_shu_rate == pytest.approx(0.22)


def test_a_v2_line_decodes_with_the_draft_never_measured() -> None:
    """Campaign 2's history stays readable, and stays honest about the gap.

    The three draft fields come back as sentinels, and -- the part that makes
    this more than a version bump -- the **game shape does not**. A v2 line's
    draw rate is a measurement and keeps reading as one; only the draft was
    never counted.
    """
    record = decode_selfplay_line(_V2_LINE)

    assert (record.iteration, record.games, record.rows) == (14, 8, 611)
    assert record.draws == 1 and record.turn_cap_games == 1
    assert record.max_plies == 256
    assert record.mean_plies == pytest.approx(115.5)
    assert record.measured, "a v2 line's game shape was measured"
    assert math.isnan(record.median_placements_w)
    assert math.isnan(record.median_placements_b)
    assert record.early_shu_games == NOT_MEASURED
    assert not record.draft_measured
    assert record.draw_rate == pytest.approx(0.125), (
        "the v2 draw rate must survive the bump that added the draft"
    )
    assert math.isnan(record.early_shu_rate), (
        "an unmeasured early-終 rate must not compare below a ramp threshold"
    )
    assert not (record.early_shu_rate < 0.3), "the ramp gate must not fire on NaN"


def test_all_three_versions_read_out_of_one_file(tmp_path: Path) -> None:
    """A root written across both bumps holds v1, v2 and v3 lines, in order."""
    path = tmp_path / SELFPLAY_JOURNAL_NAME
    path.write_text(_V1_LINE + "\n" + _V2_LINE + "\n", encoding="utf-8")
    append_selfplay(path, _telemetry(15))

    read = read_selfplay(path)

    assert read.ok and len(read) == 3, f"skipped {read.skipped} line(s)"
    assert [record.measured for record in read.records] == [False, True, True]
    assert [record.draft_measured for record in read.records] == [
        False,
        False,
        True,
    ]


def test_a_v3_line_missing_a_draft_field_is_not_a_record() -> None:
    """A line claiming v3 must carry v3's fields; absence is corruption.

    The mirror of the v2 tolerance, exactly as v2 mirrors v1: the sentinels
    are for lines that never promised the measurement.
    """
    payload = json.loads(encode_selfplay(_telemetry(0)))
    for missing in (
        "median_placements_w",
        "median_placements_b",
        "early_shu_games",
    ):
        truncated = {key: value for key, value in payload.items() if key != missing}
        with pytest.raises(JournalDecodeError):
            decode_selfplay_line(json.dumps(truncated))


def test_a_v2_record_cannot_be_written_back_out_as_v3() -> None:
    """Reading v2 is history; re-writing it as v3 would fabricate a draft.

    The same refusal v1 gets, one version along: a tool that read campaign 2's
    file and appended it elsewhere would turn "the draft was never counted"
    into a measurement of it.
    """
    record = decode_selfplay_line(_V2_LINE)
    with pytest.raises(ValueError, match="not-measured"):
        encode_selfplay(record)


def test_the_version_each_field_group_arrived_in_is_stated_once() -> None:
    """The decoder keys on the arrival versions, not on the head version.

    ``version >= SELFPLAY_SCHEMA_VERSION`` would have demoted every v2 line's
    game shape to a sentinel the moment the head moved to 3 -- silently
    erasing campaign 2's draw rates. The two constants are what stop that, so
    they are pinned.
    """
    assert (GAME_SHAPE_SINCE, DRAFT_SINCE) == (2, 3)
    assert DRAFT_SINCE == SELFPLAY_SCHEMA_VERSION
    assert SELFPLAY_READABLE_VERSIONS == frozenset({1, 2, 3})
