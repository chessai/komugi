"""The loop's telemetry wire-in, on a campaign that actually ran (KOM-025).

The 025a modules landed as a library -- journal, records, liveness, archive --
with unit suites of their own and no caller. This module is about the two
halves of that wire-in that only a real campaign can show:

* **the trainer's ``extras`` cross the boundary.** ``training_iteration``
  computes KOM-005b's staging statistic (``loss_<name>_first_decile``,
  ``_last_decile``, ``_step_std``) over the step history that only it holds,
  and journals it on *its* record. Every campaign-level reader filters on
  :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`
  (:func:`~komugi.campaign.boundary.campaign_records`), so until the loop
  carried those keys onto its own record the statistic every stage-1-to-3 gate
  reads was on a record those readers skip. The assertion is **equality**, not
  presence: two numbers under one name would be a second implementation of a
  decile mean, and the test that both exist would pass on the day they
  disagreed.
* **a relaunch cannot destroy the previous launch's history.** v1 lost its
  telemetry twice by opening the run log for writing at launch. The proof here
  is byte-level and end-to-end: a completed campaign's root is relaunched, and
  afterwards launch one's exact journal bytes are readable *both* from the
  archive and from the live log.

The relaunches run against **copies** of the shared campaign's root, because
several other modules read that root and a suite that mutated it would make
their results depend on collection order. There are two, and the second is
what makes the first's claim falsifiable:

* onto a **completed** root, which runs no iteration -- start-up alone, for
  free, which is where v1's truncation lived;
* onto an **incomplete** one (the same root, configured for one iteration
  more), which does run the loop body. On a root that appends nothing,
  "archived before this launch appended a byte" and "archived on the way out"
  leave byte-identical archives, so only this one can tell them apart -- the
  siting of the ``archive_telemetry`` call, rather than merely its existence
  (KOM-025 wire-in review, F1).

Every ``run_campaign`` call goes through the rig's wall, for the reason
``campaign_rig`` gives.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import campaign_rig as rig
from komugi.campaign.boundary import (
    BAND_VIOLATION,
    BN_DIGEST_ADVANCED,
    BN_DIGEST_LO,
    CAMPAIGN_MARKER,
    FLAT_TRAINER_FLAG,
    GAUNTLET_BREACH,
    GAUNTLET_RAN,
    LOSS_TAIL_MEDIAN,
    MUTED_FLAG,
    SPIKE_FLAG,
)
from komugi.campaign.pinned import EMA_PUBLISH_STALLED
from komugi.telemetry.journal import read_records
from komugi.telemetry.records import IterationMetrics

camp = pytest.importorskip("komugi.campaign.loop")
telemetry = pytest.importorskip("komugi.campaign.telemetry")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )

if not hasattr(telemetry, "archive_telemetry"):
    pytest.skip(
        "komugi.campaign.telemetry exports no archive_telemetry yet (KOM-025)",
        allow_module_level=True,
    )

RELAUNCH_WALL_S: float = 240.0
"""Wall for a relaunch onto a completed root.

A correct implementation runs no iteration at all -- ``resume_point`` lands
past the last one -- so this is generous by orders of magnitude. It exists so
that an implementation which *did* start playing games fails this test rather
than wedging ``ci/check.sh``.
"""

DECILE_SUFFIXES: tuple[str, ...] = ("_first_decile", "_last_decile", "_step_std")
"""The three whole-iteration summaries KOM-005b adds per loss scalar.

Spelled here rather than imported: these suffixes are the interface between
the trainer that computes the statistic and the stage gate that reads it, and
a rename that broke it would otherwise be invisible -- the campaign record
would simply stop carrying a key nobody names twice.
"""


def _journal_records(root: Path) -> list[IterationMetrics]:
    """Every :class:`IterationMetrics` in ``<root>/journal.jsonl``, in write order.

    The documented path, not a search: a journal written somewhere else is a
    failure this module should report rather than route around.
    """
    read = read_records(camp.journal_path(root), IterationMetrics, missing_ok=True)
    assert read.ok, (
        f"{camp.journal_path(root)} holds {read.skipped} undecodable line(s); "
        f"a healthy campaign writes none"
    )
    return list(read.records)


Split = tuple[list[IterationMetrics], list[IterationMetrics]]
"""One iteration's records, the trainer's first and the campaign's second."""


def _split(records: list[IterationMetrics], iteration: int) -> Split:
    """Return (trainer records, campaign records) for one iteration.

    Split on the explicit marker rather than on ``games == 0``: an iteration
    whose self-play produced nothing still trains on the older window and still
    writes a campaign record, so a ``games``-based split would misclassify
    exactly during a self-play outage.
    """
    mine = [record for record in records if record.iteration == iteration]
    campaign = [r for r in mine if r.extras.get(CAMPAIGN_MARKER, 0.0) == 1.0]
    trainer = [r for r in mine if r.extras.get(CAMPAIGN_MARKER, 0.0) != 1.0]
    return trainer, campaign


BOUNDARY_OWNED: frozenset[str] = frozenset(
    {
        CAMPAIGN_MARKER,
        LOSS_TAIL_MEDIAN,
        SPIKE_FLAG,
        BAND_VIOLATION,
        FLAT_TRAINER_FLAG,
        MUTED_FLAG,
        BN_DIGEST_LO,
        BN_DIGEST_ADVANCED,
        GAUNTLET_BREACH,
        *GAUNTLET_RAN,
        EMA_PUBLISH_STALLED,
        camp.EMA_PUBLISHED_STEP,
        "rows",
        "mute_applied",
        "mute_until_iteration",
        "probe_sha256_lo",
    }
)
"""The keys the **boundary** writes, and therefore the ones the merge may
override.

Everything else on the trainer's record must reach the campaign's untouched:
the merge's documented rule is "the trainer's extras go in first, the
boundary's measurements override them", so the campaign record is a *superset*
of the trainer's extras outside this set. Asserting the superset rather than
only the decile keys is what makes the loss of ``loss_total``, ``steps`` or
``batch_size`` visible (KOM-025 wire-in review, F3).
"""


def _decile_keys(extras: Mapping[str, float]) -> set[str]:
    """Return the KOM-005b summary keys present in `extras`."""
    return {
        key
        for key in extras
        if any(key.endswith(suffix) for suffix in DECILE_SUFFIXES)
    }


# ---------------------------------------------------------------------------
# the shared two-iteration campaign
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign() -> rig.SharedCampaign:
    """The shared two-iteration campaign (run at most once per interpreter)."""
    built = rig.shared_campaign()
    rig.kill_stragglers()
    return built


@pytest.fixture(scope="module")
def records(campaign: rig.SharedCampaign) -> list[IterationMetrics]:
    """Its journal records, after the run finished."""
    campaign.results()
    return _journal_records(campaign.root)


def test_each_iteration_wrote_both_records(
    records: list[IterationMetrics],
) -> None:
    """The premise: a trainer record *and* a campaign record, per iteration.

    Both shapes are honest and the trainer's docstring invites the second, but
    every assertion below compares one against the other -- so an iteration
    missing either would make them vacuous rather than failing.
    """
    for iteration in range(rig.ITERATIONS):
        trainer, campaign_written = _split(records, iteration)
        assert trainer, (
            f"iteration {iteration} has no trainer record; the journal holds "
            f"{[(r.iteration, sorted(r.extras)[:3]) for r in records]}"
        )
        assert campaign_written, (
            f"iteration {iteration} has no record carrying {CAMPAIGN_MARKER!r}: "
            f"the boundary never wrote the campaign's own telemetry"
        )


def test_the_trainer_recorded_the_staging_statistic(
    records: list[IterationMetrics],
) -> None:
    """Non-vacuity for the flow-through below: the keys exist at their source.

    If KOM-005b's decile block were removed from ``training_iteration`` this
    node fails here, rather than the flow-through node passing on an empty
    intersection.
    """
    for iteration in range(rig.ITERATIONS):
        trainer, _ = _split(records, iteration)
        keys = _decile_keys(trainer[-1].extras)
        assert keys, (
            f"iteration {iteration}'s trainer record carries no "
            f"{DECILE_SUFFIXES} key; it holds {sorted(trainer[-1].extras)}"
        )
        assert "loss_total_first_decile" in keys, (
            f"the total loss has no first-decile mean: {sorted(keys)}"
        )


def test_the_campaign_record_carries_the_trainers_decile_keys(
    records: list[IterationMetrics],
) -> None:
    """KOM-025: the statistic reaches the record campaign-level readers keep.

    Equality against the trainer's own numbers, key by key. Presence alone
    would be satisfied by a loop that recomputed the deciles from the step
    history it is still holding -- which is the failure this pins: two
    implementations of one statistic, under one name, agreeing until a slice
    or a rounding differs, with no test able to say which one the campaign
    means.
    """
    for iteration in range(rig.ITERATIONS):
        trainer, campaign_written = _split(records, iteration)
        source = trainer[-1].extras
        written = campaign_written[-1].extras
        for key in sorted(_decile_keys(source)):
            assert key in written, (
                f"iteration {iteration}: the campaign record does not carry "
                f"{key!r}. It holds {sorted(written)}"
            )
            assert written[key] == source[key], (
                f"iteration {iteration}: {key} is {written[key]!r} on the "
                f"campaign record and {source[key]!r} on the trainer's. The "
                f"campaign must carry the trainer's number across, not compute "
                f"a second one"
            )


def test_the_campaign_record_is_a_superset_of_the_trainers_extras(
    records: list[IterationMetrics],
) -> None:
    """F3: the *whole* trainer record crosses, not only the decile keys.

    ``loss_total``, ``steps`` and ``batch_size`` are carried by the same call
    and read beside the deciles by the staging ladder, and a loop that
    recomputed the deciles locally would drop them silently -- the equality
    node above cannot see that, because it iterates only the keys it found on
    the trainer's record *and* a faithful re-derivation satisfies it bit for
    bit.

    The excluded set is exactly what the boundary measures itself
    (:data:`BOUNDARY_OWNED`); everything else must arrive unchanged, so the
    only way to pass is to have read the trainer's record off the journal.
    """
    for iteration in range(rig.ITERATIONS):
        trainer, campaign_written = _split(records, iteration)
        source = trainer[-1].extras
        written = campaign_written[-1].extras
        carried = {
            key: value
            for key, value in source.items()
            if key not in BOUNDARY_OWNED
        }
        assert carried, (
            f"iteration {iteration}'s trainer record carries nothing outside "
            f"the boundary's own keys; this assertion would be vacuous"
        )
        missing = sorted(key for key in carried if key not in written)
        assert not missing, (
            f"iteration {iteration}: the campaign record dropped "
            f"{missing} from the trainer's extras. It holds {sorted(written)}"
        )
        differing = {
            key: (written[key], value)
            for key, value in carried.items()
            if written[key] != value
        }
        assert not differing, (
            f"iteration {iteration}: {differing} differ between the campaign "
            f"record (first) and the trainer's (second); the campaign carries "
            f"the trainer's numbers across, it does not compute its own"
        )
        assert "loss_total" in carried and "steps" in carried, (
            f"iteration {iteration}'s trainer record is missing the keys this "
            f"node exists to protect: {sorted(carried)}"
        )


def test_every_campaign_record_names_the_step_its_ema_was_published_at(
    records: list[IterationMetrics],
) -> None:
    """F6: the EMA advance gate leaves evidence on the healthy path too.

    ``BoundaryOutcome.verdicts`` is in-memory and has no reader anywhere in
    ``src/``, so a *passing* EMA gate used to leave nothing behind: no journal
    reader could count stalls, and no test could tell a gate that ran from one
    that had been deleted. The stamp is the published file's own step, and
    KOM-022 §2 files ``EMA_i`` at ``theta_i``'s step -- which the loop numbers
    ``(i + 1) * steps_per_iteration``.
    """
    for iteration in range(rig.ITERATIONS):
        _, campaign_written = _split(records, iteration)
        written = campaign_written[-1].extras
        expected = float((iteration + 1) * rig.STEPS_PER_ITERATION)
        assert written.get(camp.EMA_PUBLISHED_STEP) == expected, (
            f"iteration {iteration}'s campaign record carries "
            f"{written.get(camp.EMA_PUBLISHED_STEP)!r} for the published EMA "
            f"step; the EMA this boundary published is at {expected}"
        )
        assert EMA_PUBLISH_STALLED not in written, (
            f"iteration {iteration} flagged a stalled EMA publish on a "
            f"campaign that completed"
        )


def test_the_boundarys_own_keys_survive_the_merge(
    records: list[IterationMetrics],
) -> None:
    """The trainer's extras ride along; they do not displace the boundary's.

    The campaign record's *subject* is what the boundary measured -- the
    marker every reader filters on, the self-play row count, the BatchNorm
    digest series threaded from iteration to iteration. A merge that let the
    trainer's keys win would leave the record looking complete and quietly
    change what several gates read.
    """
    for iteration in range(rig.ITERATIONS):
        _, campaign_written = _split(records, iteration)
        written = campaign_written[-1].extras
        assert written[CAMPAIGN_MARKER] == 1.0
        assert "rows" in written, f"iteration {iteration} recorded no row count"
        assert "bn_digest_lo" in written, (
            f"iteration {iteration} recorded no BatchNorm digest: the series a "
            f"restart compares against starts here"
        )


def test_iteration_zero_still_stamped_the_pinned_constants(
    records: list[IterationMetrics],
) -> None:
    """The merge did not cost the root its pre-registration.

    KOM-022 stamps the band, the anchor age and the spike factor into the
    first campaign record of a root, and ``check_first_record`` refuses a root
    whose first record carries none of them -- so a boundary that overwrote
    them with the trainer's extras would make every later iteration refuse to
    start.
    """
    _, campaign_written = _split(records, 0)
    assert "ema_decay" in campaign_written[0].extras, (
        f"iteration 0's campaign record stamped no constants; it holds "
        f"{sorted(campaign_written[0].extras)}"
    )


# ---------------------------------------------------------------------------
# archive-on-relaunch (v1 destroyed its telemetry twice)
# ---------------------------------------------------------------------------


class _Relaunched:
    """A completed campaign's root, copied, then started a second time.

    The bytes of both logs are stamped **before** the second launch, because
    "launch two preserved launch one's history" is a claim about what was on
    disk before it started; a stamp taken afterwards would be a tautology.
    """

    def __init__(self, source: Path, root: Path) -> None:
        """Copy `source` to `root`, stamp its logs, and relaunch onto it."""
        shutil.copytree(source, root, dirs_exist_ok=True)
        self.root = root
        self.journal = camp.journal_path(root)
        self.selfplay = telemetry.selfplay_journal_path(root)
        self.journal_bytes = self.journal.read_bytes()
        self.selfplay_bytes = self.selfplay.read_bytes()

        def call() -> Any:
            with rig.small_net_env():
                return rig.campaign_loop().run_campaign(
                    rig.campaign_config(root, rig.ITERATIONS),
                    rig.tiny_selfplay_config(),
                    orchestrator_overrides=rig.orchestrator_overrides(),
                    inference=rig.inference_config(),
                )

        self.run = rig.walled(call, RELAUNCH_WALL_S, "relaunch")
        rig.kill_stragglers()


@pytest.fixture(scope="module")
def relaunched(
    campaign: rig.SharedCampaign, tmp_path_factory: pytest.TempPathFactory
) -> _Relaunched:
    """The completed campaign, copied aside and started again."""
    campaign.results()
    return _Relaunched(
        campaign.root, Path(tmp_path_factory.mktemp("relaunch")) / "root"
    )


def test_the_relaunch_ran_no_iteration(relaunched: _Relaunched) -> None:
    """The premise, and the reason this costs nothing.

    ``resume_point`` lands past the last completed iteration, so the loop body
    never executes and the call returns an empty list. Everything asserted
    below is therefore a property of *start-up* alone -- which is exactly where
    v1's truncation lived.
    """
    assert relaunched.run.unwrap() == [], (
        "the relaunch replayed iterations a completed root had already run"
    )


def test_the_second_launch_did_not_truncate_the_first_launchs_journal(
    relaunched: _Relaunched,
) -> None:
    """The regression, byte for byte: launch one's journal is still there.

    v1's telemetry died twice this way -- a relaunch opened the run log for
    writing -- and the failure is silent, because the campaign that follows
    fills the file back up with plausible lines. Byte-level rather than
    record-level: a launch that rewrote the file with identical-looking
    records would still have taken a file that is only append-only by
    construction and made it something a start can shorten.
    """
    after = relaunched.journal.read_bytes()
    assert after == relaunched.journal_bytes, (
        f"the journal was {len(relaunched.journal_bytes)} bytes before the "
        f"relaunch and {len(after)} after; a launch that ran no iteration must "
        f"leave it exactly as it found it"
    )
    selfplay_after = relaunched.selfplay.read_bytes()
    assert selfplay_after == relaunched.selfplay_bytes, (
        "the self-play telemetry was rewritten by a launch that played nothing"
    )


def test_the_second_launch_archived_the_first_launchs_bytes(
    relaunched: _Relaunched,
) -> None:
    """And the copy exists, holding exactly what launch one wrote.

    The archive is what makes the claim survive a *future* defect: even if
    something later does truncate the live log, the previous launch's history
    is already a separate file under the root, byte-identical, named by the
    log's own mtime so that its position in the campaign's sequence is
    recoverable.
    """
    archive_dir = telemetry.telemetry_archive_dir(relaunched.root)
    assert archive_dir.is_dir(), (
        f"the relaunch created no archive under {archive_dir}; "
        f"{relaunched.root} holds "
        f"{rig.describe(sorted(relaunched.root.iterdir()))}"
    )
    copies = {path.read_bytes() for path in archive_dir.iterdir() if path.is_file()}
    assert relaunched.journal_bytes in copies, (
        f"no archived copy holds the first launch's journal bytes; "
        f"{archive_dir} holds "
        f"{rig.describe(sorted(archive_dir.iterdir()))}"
    )
    assert relaunched.selfplay_bytes in copies, (
        "the self-play telemetry was not archived; the sibling file is as "
        "exposed to a relaunch as the journal is"
    )


class _RelaunchedIncomplete:
    """A completed campaign's root, copied, then given **one more** iteration.

    The sibling of :class:`_Relaunched`, and the reason it exists: on a root
    that runs no iteration, "archived before this process appends a byte" and
    "archived on the way out" are byte-identical outcomes, so the claim that
    motivates the archive call's *siting* is unobservable there (KOM-025
    wire-in review, F1 / mutant M9). Here the relaunch has work to do -- the
    same root, one iteration further -- so the two sitings produce different
    archives: one holding the previous launch's history exactly, the other
    holding this launch's records as well.

    The stamp is taken before the launch, as it must be.
    """

    def __init__(self, source: Path, root: Path) -> None:
        """Copy `source` to `root`, stamp its journal, and run one more iteration."""
        shutil.copytree(source, root, dirs_exist_ok=True)
        self.root = root
        self.journal = camp.journal_path(root)
        self.journal_bytes = self.journal.read_bytes()

        def call() -> Any:
            with rig.small_net_env():
                return rig.campaign_loop().run_campaign(
                    rig.campaign_config(root, rig.ITERATIONS + 1),
                    rig.tiny_selfplay_config(),
                    orchestrator_overrides=rig.orchestrator_overrides(),
                    inference=rig.inference_config(),
                )

        self.run = rig.walled(call, rig.RESUME_WALL_S, "relaunch-incomplete")
        rig.kill_stragglers()


@pytest.fixture(scope="module")
def relaunched_incomplete(
    campaign: rig.SharedCampaign, tmp_path_factory: pytest.TempPathFactory
) -> _RelaunchedIncomplete:
    """The completed campaign, copied aside and continued for one iteration."""
    campaign.results()
    return _RelaunchedIncomplete(
        campaign.root, Path(tmp_path_factory.mktemp("relaunch-incomplete")) / "root"
    )


def test_the_incomplete_relaunch_actually_ran_an_iteration(
    relaunched_incomplete: _RelaunchedIncomplete,
) -> None:
    """The premise the two nodes below rest on, and the whole point of the root.

    ``resume_point`` lands on the iteration after the newest checkpoint, and
    the config gives it one more than the shared campaign ran -- so the loop
    body executes, self-play plays, the trainer trains and the boundary
    appends. Without this the archive assertions would be about a root nothing
    wrote to, which is precisely the case that cannot discriminate.
    """
    played = relaunched_incomplete.run.unwrap()
    assert played, (
        "the relaunch ran no iteration, so 'archived before a byte was "
        "appended' is again indistinguishable from 'archived on the way out'"
    )


def test_the_archived_copy_is_the_journal_as_the_launch_found_it(
    relaunched_incomplete: _RelaunchedIncomplete,
) -> None:
    """F1: the archive marks a launch boundary, not a moment near the exit.

    Exactly one copy of the journal, holding exactly the bytes that were there
    before this launch started. A copy taken on the way out would hold this
    launch's records too -- an archive that no longer says what the *previous*
    launch left behind, and one that a process which died mid-campaign would
    never have written at all.
    """
    archive_dir = telemetry.telemetry_archive_dir(relaunched_incomplete.root)
    assert archive_dir.is_dir(), (
        f"the relaunch created no archive under {archive_dir}; "
        f"{relaunched_incomplete.root} holds "
        f"{rig.describe(sorted(relaunched_incomplete.root.iterdir()))}"
    )
    copies = sorted(
        path
        for path in archive_dir.iterdir()
        if path.is_file() and path.name.startswith(relaunched_incomplete.journal.name)
    )
    assert len(copies) == 1, (
        f"want one archived journal for one previous launch, found "
        f"{rig.describe(copies)}"
    )
    archived = copies[0].read_bytes()
    assert archived == relaunched_incomplete.journal_bytes, (
        f"the archived journal is {len(archived)} bytes and the journal held "
        f"{len(relaunched_incomplete.journal_bytes)} before this launch "
        f"started. The copy is taken before a single byte is appended, so it "
        f"is the previous launch's history and nothing else"
    )


def test_the_live_journal_grew_from_exactly_the_archived_bytes(
    relaunched_incomplete: _RelaunchedIncomplete,
) -> None:
    """And the live log is the same history, continued -- never restarted.

    Strictly longer, and prefixed by exactly what was archived: together with
    the node above that is the whole claim -- the previous launch's bytes are
    both preserved in the archive *and* still at the head of the log every gate
    reads its baselines back out of.
    """
    stamp = relaunched_incomplete.journal_bytes
    after = relaunched_incomplete.journal.read_bytes()
    assert len(after) > len(stamp), (
        f"the journal is {len(after)} bytes after a launch that played "
        f"{len(relaunched_incomplete.run.unwrap())} iteration(s) and was "
        f"{len(stamp)} before; the boundary appends to it"
    )
    assert after.startswith(stamp), (
        "the relaunch rewrote the journal rather than appending to it: the "
        "first launch's bytes are no longer its prefix"
    )


def test_the_archive_is_named_for_the_data_not_the_rescue(
    relaunched: _Relaunched,
) -> None:
    """Each copy's name carries its source's mtime, so the order is the history.

    A wall-clock suffix would name the moment of the rescue: two relaunches
    inside one second would collide, and a log archived days after it was
    written would sort as if it were new. The mtime names the data, which is
    also what turns a second archive of an untouched log into a refusal rather
    than a silent duplicate.
    """
    archive_dir = telemetry.telemetry_archive_dir(relaunched.root)
    names = sorted(path.name for path in archive_dir.iterdir() if path.is_file())
    assert names, f"{archive_dir} is empty"
    for name in names:
        stem, _, suffix = name.rpartition(".")
        assert stem in {
            relaunched.journal.name,
            relaunched.selfplay.name,
        }, f"{name!r} does not name one of this root's logs"
        assert suffix.isdigit(), (
            f"{name!r} is not suffixed with a numeric mtime; lexicographic "
            f"order over the archive must be chronological order"
        )
