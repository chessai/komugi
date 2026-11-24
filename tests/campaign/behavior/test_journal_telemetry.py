"""A dead campaign's self-play half, read back off disk (KOM-042).

Every other module in this suite asks what ``run_campaign`` *returned*. This
one deliberately does not: it runs real campaigns and then reads their
self-play numbers out of ``<root>/selfplay.jsonl`` with nothing but the root
path, because that is the situation the record exists for. The trainer's
journal record reports ``games=0`` -- honestly, since
:func:`~komugi.train.step.training_iteration` never sees a
:class:`~komugi.selfplay.orchestrate.RunSummary` -- so before KOM-042 a
campaign that died had no persisted evidence that self-play had produced
anything at all, and the operator's only copy of the games, rows, restarts and
duration was a return value that a dead process never delivered.

Two campaigns are read here, and they pin different halves:

* **the shared two-iteration campaign** (paid for by the whole suite, not by
  this module): one record per iteration, ``games`` and ``rows`` positive,
  each equal to the ``IterationResult`` the same run returned and -- for rows
  -- to what the shard tree reads back. "Equal to the return value" is the
  claim that the file is not a plausible-looking fiction;
* **a resume** (its own campaign, one iteration then two): exactly **two**
  records across both runs. The arithmetic is the point -- run one plays
  iteration 0 and writes one line; the resumed run starts *after* iteration
  0's checkpoint, plays iteration 1 only, and writes one more. A third record
  would mean the resumed run either replayed iteration 0 or re-emitted its
  telemetry, and either way every count an audit sums over this file would be
  doubled.

Both also check the thing the split had to preserve: ``<root>/journal.jsonl``
still reads clean through :func:`komugi.telemetry.journal.read_records`, with
zero skipped lines. That is why the telemetry is a sibling file rather than a
third ``kind`` in the journal (see :mod:`komugi.campaign.telemetry`).

Every ``run_campaign`` call goes through the rig's wall, for the reason
``campaign_rig`` gives: a campaign owns spawned processes, and a hang here
must fail the test rather than wedge ``ci/check.sh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import campaign_rig as rig
from komugi.campaign.boundary import CAMPAIGN_MARKER
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics

camp = pytest.importorskip("komugi.campaign.loop")
telemetry = pytest.importorskip("komugi.campaign.telemetry")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )

if not hasattr(telemetry, "read_selfplay"):
    pytest.skip(
        "komugi.campaign.telemetry exports no read_selfplay yet (KOM-042)",
        allow_module_level=True,
    )


def _records_under(root: Path) -> list[Any]:
    """Every self-play record persisted under `root`, in write order.

    Read through the module's own reader from the path the loop documents,
    with the root as the only input -- which is exactly what an operator has
    when the process that ran the campaign is gone.
    """
    read = telemetry.read_selfplay(
        telemetry.selfplay_journal_path(root), missing_ok=True
    )
    assert read.ok, (
        f"{telemetry.selfplay_journal_path(root)} holds {read.skipped} "
        f"undecodable line(s); a healthy campaign writes none"
    )
    return list(read.records)


# ---------------------------------------------------------------------------
# the completed two-iteration campaign
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign() -> rig.SharedCampaign:
    """The shared two-iteration campaign (run at most once per interpreter).

    Stragglers are reaped afterwards. This module is the first in the suite to
    ask for the shared campaign, so it is the first to own spawned children;
    a child that outlived its run would go on creating and unlinking
    ``/dev/shm`` segments during *later* modules' before/after snapshots, and
    a self-play suite's leak assertions would blame the run they bracket.
    """
    built = rig.shared_campaign()
    rig.kill_stragglers()
    return built


@pytest.fixture(scope="module")
def results(campaign: rig.SharedCampaign) -> list[Any]:
    """Its ``IterationResult`` list, or the failure that stopped it."""
    return campaign.results()


@pytest.fixture(scope="module")
def persisted(campaign: rig.SharedCampaign, results: list[Any]) -> list[Any]:
    """Its self-play records, read back from disk after the run finished."""
    return _records_under(campaign.root)


def test_the_telemetry_file_exists_under_the_campaign_root(
    campaign: rig.SharedCampaign, results: list[Any]
) -> None:
    """An audit that lists the root finds it, next to the journal.

    A premise check with teeth: every assertion below reads this one path, so
    telemetry written somewhere else (a default directory, the CWD) would make
    them vacuous rather than failing.
    """
    path = telemetry.selfplay_journal_path(campaign.root)
    assert path.is_file(), (
        f"{path} does not exist; {campaign.root} holds "
        f"{rig.describe(sorted(campaign.root.iterdir()))}"
    )
    assert path.stat().st_size > 0, f"{path} is empty"


def test_one_selfplay_record_per_iteration(
    persisted: list[Any], results: list[Any]
) -> None:
    """Two iterations ran; two records are on disk, numbered 0 and 1, in order."""
    assert [record.iteration for record in persisted] == list(range(rig.ITERATIONS)), (
        f"want one record per iteration 0..{rig.ITERATIONS - 1} in write order, "
        f"got iterations {[record.iteration for record in persisted]}"
    )
    assert len(persisted) == len(results)


def test_every_persisted_iteration_played_games_and_wrote_rows(
    persisted: list[Any],
) -> None:
    """The heart of it: non-zero self-play numbers survive the process.

    Positive rather than exactly ``workers * games_per_worker``, for the reason
    ``test_two_iterations`` gives: design D10 lets a restarted shard replay its
    budget, so pinning the exact number would make a legitimate restart a test
    failure. Zero is the number the trainer's journal record reports, and
    distinguishing "self-play produced nothing" from "nobody wrote it down" is
    the entire ticket.
    """
    for record in persisted:
        assert record.games >= 1, f"iteration {record.iteration} recorded no games"
        assert record.rows >= 1, f"iteration {record.iteration} recorded no rows"
        assert record.restarts >= 0
        assert record.shard_dirs >= 1, (
            f"iteration {record.iteration} recorded no shard directory"
        )
        assert record.wall_s > 0.0, (
            f"iteration {record.iteration} claims self-play took "
            f"{record.wall_s}s; real games take measurable time"
        )


def test_the_persisted_numbers_are_the_returned_numbers(
    persisted: list[Any], results: list[Any]
) -> None:
    """Disk and return value agree, iteration by iteration.

    This is what makes the file an audit trail rather than a plausible
    fiction: the in-memory ``IterationResult`` is what a live operator sees,
    and the record is what is left when the operator is reading a directory
    instead of a return value.
    """
    by_iteration = {record.iteration: record for record in persisted}
    for result in results:
        record = by_iteration.get(int(result.iteration))
        assert record is not None, (
            f"iteration {result.iteration} returned {result} and persisted "
            f"nothing; disk holds iterations {sorted(by_iteration)}"
        )
        assert record.games == int(result.games), (
            f"iteration {result.iteration}: disk says {record.games} games, "
            f"the run returned {result.games}"
        )
        assert record.rows == int(result.rows), (
            f"iteration {result.iteration}: disk says {record.rows} rows, "
            f"the run returned {result.rows}"
        )


def test_the_persisted_rows_are_the_rows_on_disk(
    campaign: rig.SharedCampaign, persisted: list[Any]
) -> None:
    """The recorded row count is what the shard tree actually reads back.

    Read through :class:`~komugi.train.shards.ShardReader` under the real
    schema, the same adjudication ``test_two_iterations`` makes: the number
    worth recording is the number of samples a trainer could draw, not the
    number a header claimed.
    """
    for record in persisted:
        directory = rig.iteration_dir(campaign.root, record.iteration)
        source = rig.concat_source(rig.shard_dirs_under(directory))
        assert len(source) == record.rows, (
            f"iteration {record.iteration} recorded {record.rows} rows, but "
            f"{directory} reads back {len(source)}"
        )
        assert record.shard_dirs <= len(rig.shard_dirs_under(directory)), (
            f"iteration {record.iteration} recorded {record.shard_dirs} shard "
            f"directories, more than exist under {directory}"
        )


def test_the_trainers_journal_still_reads_clean(
    campaign: rig.SharedCampaign, results: list[Any]
) -> None:
    """The sibling-file decision, checked on a real campaign's journal.

    ``read_records`` counts an unknown ``kind`` as a skipped line -- the same
    tally as a corrupt or crash-truncated one -- so self-play telemetry inside
    ``journal.jsonl`` would make every healthy campaign report damage. Zero
    skipped lines here is that decision holding at campaign scale.
    """
    path = camp.journal_path(campaign.root)
    assert path.is_file(), f"no journal at {path}"
    read = read_records(path, IterationMetrics)
    assert read.ok, (
        f"{path} holds {read.skipped} line(s) its own reader cannot decode"
    )
    assert len(read) >= rig.ITERATIONS, (
        f"{path} holds {len(read)} training record(s) for {rig.ITERATIONS} "
        f"iterations"
    )


def test_the_game_shape_is_measured_and_agrees_across_both_files(
    campaign: rig.SharedCampaign, persisted: list[Any], results: list[Any]
) -> None:
    """KOM-150: draws, turn caps and game length, on a real campaign's disk.

    The gap this closes is not a missing field, it is a **missing
    measurement**: the campaign record used to write ``draw_rate`` and
    ``avg_game_len`` as literal zeros ("the orchestrator does not measure
    them") and the self-play line carried games, rows, restarts and wall time,
    so the draw spiral that killed the previous architecture had no on-disk
    signature at all until an operator went and read the shards by hand.

    Two claims, on the real two-iteration campaign:

    * every self-play record carries a **plausible** shape -- a non-negative
      draw count, no more turn-capped games than games, and a positive mean
      length, which is what a census that silently read zero rows would fail;
    * the two files **agree** iteration by iteration. The boundary and the
      self-play line are written at different moments by different code, and
      both are stamped from one census of one sealed shard tree, so
      ``avg_game_len`` is the record's ``mean_plies`` exactly and ``draw_rate``
      is its ``draws / games``. A second, independent census would make this a
      test of two readers agreeing; one census makes it a test of the wiring.
    """
    read = read_records(camp.journal_path(campaign.root), IterationMetrics)
    assert read.ok, f"{read.skipped} undecodable journal line(s)"
    by_iteration = {
        record.iteration: record
        for record in read.records
        if record.extras.get(CAMPAIGN_MARKER, 0.0) == 1.0
    }
    assert sorted(by_iteration) == list(range(rig.ITERATIONS)), (
        f"want one campaign record per iteration, got {sorted(by_iteration)}"
    )

    for record in persisted:
        assert record.measured, (
            f"iteration {record.iteration}'s self-play line was written "
            f"without a game census; the whole point is that it is measured"
        )
        assert record.draws >= 0
        assert 0 <= record.turn_cap_games <= record.games, (
            f"iteration {record.iteration} reports {record.turn_cap_games} "
            f"turn-capped games out of {record.games}"
        )
        assert record.mean_plies > 0.0, (
            f"iteration {record.iteration} reports a mean game length of "
            f"{record.mean_plies}; real games have plies"
        )
        assert record.max_plies >= record.mean_plies

        metrics = by_iteration[record.iteration]
        assert metrics.avg_game_len == pytest.approx(record.mean_plies), (
            f"iteration {record.iteration}: the journal says the mean game "
            f"was {metrics.avg_game_len} plies, the self-play line says "
            f"{record.mean_plies}"
        )
        assert metrics.draw_rate == pytest.approx(record.draws / record.games), (
            f"iteration {record.iteration}: the journal's draw rate "
            f"{metrics.draw_rate} is not {record.draws}/{record.games}"
        )
        assert metrics.extras["draws"] == pytest.approx(float(record.draws))
        assert metrics.extras["turn_cap_games"] == pytest.approx(
            float(record.turn_cap_games)
        )
        assert metrics.extras["max_plies"] == pytest.approx(float(record.max_plies))
        assert metrics.extras["cap_rate"] == pytest.approx(
            record.turn_cap_games / record.games
        )


# ---------------------------------------------------------------------------
# the resume: append-only, once per iteration actually run
# ---------------------------------------------------------------------------


class _Resumed:
    """One campaign asked for one iteration, then asked for two, on one root.

    The bytes of the telemetry file are stamped between the two runs, because
    "the resumed run did not rewrite the first run's record" is a claim about
    what was on disk *before* it started, and a stamp taken afterwards would
    be a tautology.
    """

    def __init__(self, root: Path) -> None:
        """Run, stamp the telemetry file, resume."""
        self.root = root
        self.first = rig.walled_run_campaign(
            root, iterations=1, wall_s=rig.RESUME_WALL_S, label="telemetry-first"
        ).unwrap()
        self.after_first = _records_under(root)
        self.bytes_after_first = telemetry.selfplay_journal_path(root).read_bytes()
        self.second = rig.walled_run_campaign(
            root, iterations=2, wall_s=rig.RESUME_WALL_S, label="telemetry-resumed"
        ).unwrap()
        self.after_second = _records_under(root)
        # Same reason as the `campaign` fixture: these two runs own spawned
        # children, and a survivor would still be touching /dev/shm while a
        # later module snapshots it.
        rig.kill_stragglers()


@pytest.fixture(scope="module")
def resumed(tmp_path_factory: pytest.TempPathFactory) -> _Resumed:
    """The two-call round trip, on its own root.

    Not the shared campaign: this scenario needs a campaign that *stopped*,
    which by definition cannot be the one that ran to completion.
    """
    return _Resumed(Path(tmp_path_factory.mktemp("telemetry-resume")))


def test_the_first_run_persisted_its_one_iteration(resumed: _Resumed) -> None:
    """The premise: one iteration ran, one record, before anything is resumed."""
    assert [result.iteration for result in resumed.first] == [0]
    assert [record.iteration for record in resumed.after_first] == [0]
    assert resumed.after_first[0].games >= 1
    assert resumed.after_first[0].rows >= 1


def test_the_resumed_run_added_exactly_one_record(resumed: _Resumed) -> None:
    """Two records across both runs: 1 written + 1 written, 0 rewritten.

    The arithmetic, spelled out because it is the assertion: the first run was
    asked for one iteration and played iteration 0 (one record). The second
    was asked for two, resumed after iteration 0's checkpoint, and played
    iteration 1 alone (one more). Two records, iterations ``[0, 1]``, each
    appearing once. Three would mean iteration 0's telemetry was written twice
    -- by a replay, or by a resume that re-emits recovered iterations -- and
    would double every total an audit computes from this file.
    """
    iterations = [record.iteration for record in resumed.after_second]
    assert iterations == [0, 1], (
        f"want records for iterations [0, 1] in write order, got {iterations}. "
        f"The resumed run returned {[r.iteration for r in resumed.second]}"
    )
    assert len(iterations) == len(set(iterations)), f"duplicates: {iterations}"
    assert [result.iteration for result in resumed.second] == [1], (
        "the resumed run must play iteration 1 only; it returned "
        f"{[result.iteration for result in resumed.second]}"
    )


def test_the_resume_appended_and_rewrote_nothing(resumed: _Resumed) -> None:
    """Iteration 0's line is the same bytes, in the same place, afterwards.

    Byte-level rather than value-level: a resume that rewrote the file with
    identical-looking numbers would still have taken a file that is only
    append-only by construction and made it something a crash can truncate to
    nothing.
    """
    after = telemetry.selfplay_journal_path(resumed.root).read_bytes()
    lines = after.count(b"\n")
    assert after.startswith(resumed.bytes_after_first), (
        "the resumed run did not leave the first run's telemetry untouched"
    )
    assert len(after) > len(resumed.bytes_after_first), (
        "the resumed run appended nothing"
    )
    assert lines == 2, f"want one line per iteration run, got {lines}"


def test_the_resumed_records_match_what_each_run_returned(
    resumed: _Resumed,
) -> None:
    """Both runs' return values are reconstructible from the one file.

    The claim an audit actually needs: the file is not merely append-only and
    the right length, it holds the numbers the two processes reported -- which
    is what makes it usable when neither process is around to ask.
    """
    by_iteration = {record.iteration: record for record in resumed.after_second}
    for result in list(resumed.first) + list(resumed.second):
        record = by_iteration[int(result.iteration)]
        assert (record.games, record.rows) == (int(result.games), int(result.rows)), (
            f"iteration {result.iteration}: disk says "
            f"{(record.games, record.rows)}, the run returned "
            f"{(result.games, result.rows)}"
        )
        assert record.wall_s > 0.0


def test_the_resumed_campaigns_journal_still_reads_clean(resumed: _Resumed) -> None:
    """A resumed campaign's journal is undamaged by the telemetry beside it."""
    read = read_records(camp.journal_path(resumed.root), IterationMetrics)
    assert read.ok, f"{read.skipped} undecodable journal line(s) after a resume"
    assert len(read) >= 2, (
        f"want a training record for each of the two iterations, got {len(read)}"
    )


# ---------------------------------------------------------------------------
# the wiring: fields nothing else can cross-check (fanout-1 union review F4)
# ---------------------------------------------------------------------------


def test_the_record_echoes_the_summary_fields_nothing_else_can_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``restarts`` and ``shard_dirs`` cross the wiring unaltered.

    The equality tests above pin ``games`` and ``rows`` against the
    ``IterationResult`` the run returned -- but ``IterationResult`` carries
    neither ``restarts`` nor ``shard_dirs``, so a wiring bug in those two (an
    off-by-one, a swapped field) is invisible to every other assertion in this
    suite. The telemetry docstring calls the record the only place the restart
    count survives the process, which is precisely why nothing else can
    cross-check it (fanout-1 union review F4 -- a ``restarts + 1`` mutant
    survived the original suite). So: doctor the ``RunSummary`` the loop sees
    to hold values a healthy tiny run never produces, and assert the persisted
    line echoes them exactly.
    """
    real = camp.run_selfplay
    doctored: list[Any] = []

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        summary = real(*args, **kwargs)
        fake = summary._replace(
            restarts=7, shard_dirs=summary.shard_dirs + (summary.shard_dirs[-1],)
        )
        doctored.append(fake)
        return fake

    monkeypatch.setattr(camp, "run_selfplay", wrapper)
    rig.walled_run_campaign(
        tmp_path, iterations=1, wall_s=rig.RESUME_WALL_S, label="telemetry-wiring"
    ).unwrap()
    rig.kill_stragglers()

    records = _records_under(tmp_path)
    assert len(doctored) == 1, "the doctored self-play ran more than once"
    assert len(records) == 1, f"one iteration, {len(records)} records"
    record, fake = records[0], doctored[0]
    assert record.restarts == 7, (
        f"summary.restarts=7 persisted as {record.restarts}: the restart "
        f"count exists nowhere else, so this wiring must be exact"
    )
    assert record.shard_dirs == len(fake.shard_dirs), (
        f"summary carried {len(fake.shard_dirs)} shard dirs, the record says "
        f"{record.shard_dirs}"
    )
    assert record.games == fake.games_completed
    assert record.rows == fake.rows_written
    assert record.wall_s > 0.0
