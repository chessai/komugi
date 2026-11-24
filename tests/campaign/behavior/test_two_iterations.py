"""The loop proof: two iterations happen, and the second does not eat the first.

Everything else in this suite reads the campaign this module runs. What is
pinned here is the shape of a finished two-iteration campaign -- the part that
no component test can see, because each component only ever sees one iteration:

* **two** ``IterationResult`` s, numbered 0 and 1, each having played games and
  written rows;
* **two checkpoint files coexisting**, at strictly increasing steps. This is
  the campaign-level form of the wave-10 review's F5 obligation ("global
  checkpoint step counter -- same-step supersede currently annihilates prior
  iterations"): ``training_iteration`` names its checkpoint after the steps
  *that call* ran, so two iterations of eight steps each would both be
  ``checkpoint-000000008-*``, and ``save_checkpoint`` unlinks the superseded
  one by design. The observable consequence is a campaign that keeps exactly
  one checkpoint no matter how long it runs, and the only place it is visible
  is here, after the second iteration;
* a journal holding at least one record per iteration, since a campaign whose
  history cannot be reconstructed off disk is a campaign that cannot be
  audited after it dies;
* both iterations' shard trees on disk under ``root/iter_<n>``, which is what
  the recency window is later asked to index (see
  ``test_window_spans_iterations``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

import campaign_rig as rig

camp = pytest.importorskip("komugi.campaign.loop")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def campaign() -> rig.SharedCampaign:
    """The shared two-iteration campaign (run at most once per interpreter)."""
    return rig.shared_campaign()


@pytest.fixture(scope="module")
def results(campaign: rig.SharedCampaign) -> list[Any]:
    """Its ``IterationResult`` list, or the failure that stopped it."""
    return campaign.results()


def test_the_campaign_returns_one_result_per_iteration(results: list[Any]) -> None:
    """Two iterations were asked for; two are reported, numbered 0 and 1."""
    assert len(results) == rig.ITERATIONS, (
        f"run_campaign(iterations={rig.ITERATIONS}) returned {len(results)} "
        f"IterationResult(s): {results}"
    )
    assert [result.iteration for result in results] == list(range(rig.ITERATIONS)), (
        "iterations must be reported in order, numbered from zero, got "
        f"{[result.iteration for result in results]}"
    )


def test_every_iteration_played_games_and_wrote_rows(results: list[Any]) -> None:
    """A campaign that reports an iteration in which nothing happened is lying.

    The counts are asserted as *positive*, not as exactly
    ``workers * games_per_worker``: design D10 lets a restarted shard replay its
    whole budget under a new incarnation, so the orchestrator's own contract
    reports more games than were asked for when a restart happened. Pinning the
    exact number would make a legitimate restart a test failure.
    """
    for result in results:
        assert result.games >= 1, f"iteration {result.iteration} played no games"
        assert result.rows >= 1, f"iteration {result.iteration} wrote no rows"


def test_the_losses_are_real_numbers(results: list[Any]) -> None:
    """Each iteration reports a finite first and last loss.

    Not "the loss went down": eight steps at batch 16 on a few hundred rows is
    a smoke run, and a suite that asserted descent there would be asserting
    noise. What must hold is that the numbers were *measured* -- a NaN here is
    a diverged step that the campaign carried on from, and it would poison
    every iteration after it.
    """
    for result in results:
        for name in ("first_loss", "last_loss"):
            value = float(getattr(result, name))
            assert math.isfinite(value), (
                f"iteration {result.iteration} reports {name}={value}"
            )


def test_two_checkpoints_coexist_at_strictly_increasing_steps(
    campaign: rig.SharedCampaign, results: list[Any]
) -> None:
    """The F5 regression, at campaign level: iteration 1 must not supersede 0.

    Both halves matter and they are different failures. Strictly increasing
    ``checkpoint_step`` is the *global* step counter existing at all; two files
    surviving on disk is that counter reaching :func:`save_checkpoint`, whose
    same-step supersede would otherwise have deleted iteration 0's file after
    iteration 1 published an identically named one.
    """
    steps = [int(result.checkpoint_step) for result in results]
    assert steps == sorted(set(steps)) and len(set(steps)) == len(steps), (
        f"checkpoint_step must increase strictly across iterations, got {steps}"
    )

    # The *trainer's* checkpoints. KOM-022 §2 (decision W1) publishes a second
    # series into the same root -- the EMA self-play serves, an ordinary
    # checkpoint under `<root>/ema` -- so "one surviving checkpoint per
    # iteration" is now a claim about one of two series, and the one this test
    # is about is the one `save_checkpoint`'s same-step supersede would have
    # annihilated (wave-10 F5). The EMA series' own step invariant is
    # KOM-022 acceptance #5.
    refs = rig.train_checkpoint_refs(campaign.root)
    assert len(refs) == rig.ITERATIONS, (
        f"want one surviving checkpoint per iteration under {campaign.root}, "
        f"found {len(refs)}: {[ref.path.name for ref in refs]}. One file for "
        f"two iterations is the same-step supersede (wave-10 F5)"
    )
    assert [ref.step for ref in refs] == steps, (
        f"the checkpoints on disk are at steps {[ref.step for ref in refs]}, "
        f"but the campaign reported {steps}"
    )
    for ref in refs:
        assert ref.path.is_file(), f"{ref.path} vanished between listing and reading"

    # The gap, not the absolute value: "the counter starts at the first
    # iteration's steps" and "it starts at zero" are both defensible readings
    # of `training_iteration(..., global_step_base=)`, and they differ by a
    # constant. What the global counter *claims* is that one iteration of
    # `steps_per_iteration` steps advances the count by exactly that much --
    # which is the same statement under either reading.
    gaps = [second - first for first, second in zip(steps, steps[1:])]
    assert gaps == [rig.STEPS_PER_ITERATION] * (len(steps) - 1), (
        f"consecutive checkpoint steps {steps} advance by {gaps}, but each "
        f"iteration ran {rig.STEPS_PER_ITERATION} steps"
    )


def test_one_ema_file_per_iteration_too(
    campaign: rig.SharedCampaign, results: list[Any]
) -> None:
    """Review F8: the *other* series has the same shape, and nobody asserted it.

    KOM-022 §2 publishes a second checkpoint series into the same root, and the
    v6 fix to this module narrowed every "one file per iteration" claim to the
    trainer's half (``train_checkpoint_refs``). That is correct -- the EMA
    carries ``optim_state={}`` and would answer the optimiser questions with a
    file that never held an optimiser -- but it left the EMA series' own count
    asserted nowhere at campaign level: acceptance #5's node is over a planted
    directory pair, and the live check is a single equality of newest steps.

    So: one EMA file per iteration, at the trainer's own steps (the EMA is
    published at its source's step, which is what makes "EMA ahead of the
    trainer" an impossible state rather than an ambiguous one), and each of them
    identified by what the file *says it is* rather than by which directory it
    sits in.
    """
    refs = [
        ref
        for ref in rig.checkpoint_refs(campaign.root)
        if rig.is_ema_checkpoint(ref)
    ]
    steps = [int(result.checkpoint_step) for result in results]

    assert len(refs) == rig.ITERATIONS, (
        f"want one EMA checkpoint per iteration under {campaign.root}, found "
        f"{[ref.path.name for ref in refs]}. One file for two iterations is "
        f"the same-step supersede arriving through the second series"
    )
    assert [ref.step for ref in refs] == steps, (
        f"the EMA series is at steps {[ref.step for ref in refs]}, but the "
        f"trainer published at {steps}: the EMA is published at its source's "
        f"step, so these cannot differ"
    )
    assert [ref.step for ref in refs] == sorted(set(ref.step for ref in refs)), (
        "the EMA steps are not strictly increasing"
    )
    trained = {ref.path for ref in rig.train_checkpoint_refs(campaign.root)}
    assert trained.isdisjoint({ref.path for ref in refs}), (
        "a file is counted in both series: the two are told apart by "
        "meta['role'], and one file cannot be both"
    )


def test_the_journal_holds_a_record_for_each_iteration(
    campaign: rig.SharedCampaign,
) -> None:
    """At least one telemetry record per iteration survives on disk.

    "At least" rather than "exactly": ``training_iteration`` writes its own
    record with the loss extras and zeros for the game counts it never saw, and
    the campaign is explicitly invited (by that function's docstring) to write a
    second one for the same iteration carrying the real self-play numbers. Both
    shapes are honest; a journal that cannot account for both iterations is
    not.
    """
    records = rig.journal_records(campaign.root)
    assert len(records) >= rig.ITERATIONS, (
        f"want >= {rig.ITERATIONS} journal records under {campaign.root}, found "
        f"{len(records)} in {rig.describe(rig.journal_candidates(campaign.root))}"
    )
    covered = {record.iteration for record in records}
    assert covered >= set(range(rig.ITERATIONS)), (
        f"journal records cover iterations {sorted(covered)}, want at least "
        f"{list(range(rig.ITERATIONS))}"
    )


def test_both_iterations_left_a_shard_tree_on_disk(
    campaign: rig.SharedCampaign,
) -> None:
    """``root/iter_0`` and ``root/iter_1`` both hold real, readable shards.

    The layout is the contract (``root/iter_<n>``) and the contents are the
    point: an iteration directory that exists but holds no sealed shard is an
    iteration whose data the window cannot index, which would make the recency
    schedule a no-op wearing a schedule's clothes.
    """
    for iteration in range(rig.ITERATIONS):
        directory = rig.iteration_dir(campaign.root, iteration)
        assert directory.is_dir(), (
            f"{directory} does not exist; {campaign.root} holds "
            f"{rig.describe(sorted(campaign.root.iterdir()))}"
        )
        worker_dirs = rig.shard_dirs_under(directory)
        assert worker_dirs, (
            f"{directory} holds no worker_<k>_r<i> shard directory (design D10)"
        )
        shards = rig.sealed_shards(directory)
        assert shards, f"{directory} holds no sealed shard file"


def test_the_reported_rows_are_the_rows_on_disk(
    campaign: rig.SharedCampaign, results: list[Any]
) -> None:
    """Every row an iteration claims is readable back through the real schema.

    Read with :class:`~komugi.train.shards.ShardReader` under
    ``komugi.selfplay.records.SCHEMA`` rather than counted from the ``npz``
    headers, because the number that matters to the next iteration is the
    number of samples the *trainer* can draw -- a shard that fails its schema
    fingerprint is a row the campaign counted and the window cannot see.
    """
    for result in results:
        directory = rig.iteration_dir(campaign.root, result.iteration)
        source = rig.concat_source(rig.shard_dirs_under(directory))
        assert len(source) == int(result.rows), (
            f"iteration {result.iteration} reported {result.rows} rows, but "
            f"{directory} reads back {len(source)}"
        )


def test_the_campaign_root_is_the_one_it_was_given(
    campaign: rig.SharedCampaign,
) -> None:
    """Nothing was written outside ``CampaignConfig.root``.

    A cheap premise check with real teeth: every other assertion in this suite
    looks *under the root*, so a loop that quietly wrote its checkpoints or its
    journal somewhere else (a default directory, the CWD) would make several of
    them vacuous rather than failing.
    """
    root: Path = campaign.root
    assert rig.checkpoint_refs(root), f"no checkpoint anywhere under {root}"
    assert rig.journal_records(root), f"no journal record anywhere under {root}"
