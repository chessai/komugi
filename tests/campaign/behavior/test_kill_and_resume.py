"""A crash costs at most one iteration -- and never the ones already paid for.

This is the L15-shaped gap the KOM-020 wave-10 note recorded as a campaign-loop
obligation: "resume round-trip-and-continue test". The claim a training
campaign makes to whoever is paying for the machine is that a preemption at
hour nine does not restart at hour zero, and the only way to falsify it is to
stop a campaign and start another one over the same directory.

The stop here is a *clean* one -- a campaign asked for one iteration, which
returned -- rather than a SIGKILL mid-iteration, and that is deliberate. A
kill leaves a torn shard, a possibly-absent checkpoint and a partially written
journal line all at once; each is already owned by the suite of the component
that must tolerate it (``komugi.train.shards``' single skipped shard,
``load_checkpoint``'s digest walk, ``read_records``' skipped-line count). What
none of them can see, and what this module tests, is the *loop's* decision
after those pieces have been read: iteration 0 is on disk, therefore iteration
0 does not happen again, therefore the run continues at iteration 1.

What must hold, whatever shape the resumed call reports
-------------------------------------------------------
The contract fixes ``run_campaign(...) -> list[IterationResult]`` but not what
a resumed run puts in that list: reporting only the iterations it *ran*, and
reporting the recovered one alongside them, are both defensible. Both are
accepted (and the accepted shape is asserted precisely, so a third shape is
still a failure). What is *not* negotiable is on disk:

* iteration 0's shard tree is byte-for-byte and mtime-for-mtime what it was --
  no replay, not even a deterministic one that would produce the same rows;
* iteration 0's checkpoint is the same file with the same bytes;
* exactly one new iteration exists afterwards, at a strictly higher checkpoint
  step;
* the final checkpoint carries optimiser state, because a resume that dropped
  Adam's moments would look exactly like this test's success while quietly
  throwing away the second-moment estimates that make the next iteration's
  first steps sane.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from komugi.train.checkpoint import (
    Checkpoint,
    list_checkpoints,
    load_checkpoint,
    load_checkpoint_file,
)
from komugi.train.digest import canonical_digest, parameter_keys
from komugi.train.ema import EMA_DECAY, reconcile_ema
from komugi.train.losses import LossAssembler
from komugi.train.step import StepConfig, TrainStep, training_iteration
from komugi.train.window import RecencyWindow, WindowConfig
from komugi.train.wiring import P0A_RECIPE

import campaign_rig as rig

camp = pytest.importorskip("komugi.campaign.loop")
factory = pytest.importorskip("komugi.campaign.factory")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )

_ADAM_MOMENTS: tuple[str, ...] = ("exp_avg", "exp_avg_sq")
"""What AdamW keeps per parameter, and therefore what a resume must carry."""


class _Resumed:
    """One campaign stopped after iteration 0, then asked for two iterations.

    A class rather than a tuple because six things have to travel together and
    one of them is a before/after pair whose halves are meaningless apart;
    naming them is what keeps the assertions below readable as claims.

    The order in the constructor is the experiment: run, stamp, run again,
    stamp. Both stamps are taken *here* rather than in the tests that use them,
    because a "before" recorded after the second run would be a tautology.
    """

    def __init__(self, root: Path) -> None:
        """Run the first campaign, stamp the disk, then run the resumed one."""
        self.root = root
        self.first = rig.walled_run_campaign(
            root, iterations=1, wall_s=rig.RESUME_WALL_S, label="resume-first"
        ).unwrap()
        self.before_tree = rig.stamp_tree(rig.iteration_dir(root, 0))
        self.before_checkpoints = {
            ref.path.name: rig.digest_of(ref.path)
            for ref in rig.checkpoint_refs(root)
        }
        self.second = rig.walled_run_campaign(
            root, iterations=2, wall_s=rig.RESUME_WALL_S, label="resume-second"
        ).unwrap()
        self.after_tree = rig.stamp_tree(rig.iteration_dir(root, 0))


@pytest.fixture(scope="module")
def resumed(tmp_path_factory: pytest.TempPathFactory) -> _Resumed:
    """The two-call round trip.

    Its own campaign, and the only test in this suite that pays for a second
    one: the scenario needs a campaign that *stopped*, which by definition
    cannot be the completed run every other module shares.
    """
    return _Resumed(Path(tmp_path_factory.mktemp("resume")))


@pytest.fixture(scope="module")
def final_checkpoint(resumed: _Resumed) -> Checkpoint:
    """The newest **trainer** checkpoint after the resumed run, decoded.

    The trainer's, specifically: KOM-022 §2 (decision W1) publishes a second
    checkpoint series into the same root -- the EMA self-play serves -- and
    that one carries ``optim_state={}`` by design, so "the newest checkpoint
    anywhere under the root" would answer the optimiser questions below with a
    file that never held an optimiser.
    """
    refs = rig.train_checkpoint_refs(resumed.root)
    assert refs, f"no train checkpoint under {resumed.root} after the resumed run"
    checkpoint = load_checkpoint(refs[-1].path.parent)
    assert checkpoint is not None, f"cannot load the newest checkpoint in {refs[-1].path.parent}"
    return checkpoint


def test_the_first_campaign_really_ran_one_iteration(resumed: _Resumed) -> None:
    """The premise: one iteration, on disk, before anything is resumed.

    Asserted rather than assumed, because every claim below is of the form
    "this did not change" -- and a thing that was never there does not change
    either.
    """
    assert [result.iteration for result in resumed.first] == [0], (
        f"run_campaign(iterations=1) reported "
        f"{[result.iteration for result in resumed.first]}"
    )
    assert resumed.before_tree, (
        f"{rig.iteration_dir(resumed.root, 0)} held no files after iteration 0"
    )
    assert resumed.before_checkpoints, (
        f"iteration 0 wrote no checkpoint under {resumed.root}"
    )


def test_exactly_one_further_iteration_ran(resumed: _Resumed) -> None:
    """Asked for two, one was already done, so precisely one more happened.

    Counted from the directories rather than from the returned list, because
    the returned list is the thing whose shape the contract leaves open. Two
    iteration directories and no third is the claim; a campaign that replayed
    iteration 0 into ``iter_2`` (or appended a third) fails here even if its
    return value looked perfect.
    """
    for iteration in (0, 1):
        directory = rig.iteration_dir(resumed.root, iteration)
        assert directory.is_dir(), f"{directory} does not exist after the resume"
        assert rig.sealed_shards(directory), f"{directory} holds no sealed shard"
    stray = rig.iteration_dir(resumed.root, 2)
    assert not stray.exists(), (
        f"{stray} exists: the resumed run ran a third iteration for a "
        f"two-iteration campaign"
    )


def test_the_resumed_run_reports_a_documented_shape(resumed: _Resumed) -> None:
    """The returned list is either "what I ran" or "the whole campaign".

    Both are legitimate and the difference is a documentation choice, so both
    are accepted -- precisely. The one thing common to them, and the thing that
    makes the return value usable at all, is that the *last* entry is iteration
    1 and its checkpoint step is the newest on disk.
    """
    numbers = [result.iteration for result in resumed.second]
    assert numbers in ([1], [0, 1]), (
        f"a resumed two-iteration campaign reported iterations {numbers}; the "
        f"documented shapes are [1] (only what ran) or [0, 1] (the whole "
        f"campaign, iteration 0 recovered)"
    )
    refs = rig.checkpoint_refs(resumed.root)
    assert refs, f"no checkpoint under {resumed.root} after the resumed run"
    assert int(resumed.second[-1].checkpoint_step) == refs[-1].step, (
        f"the last IterationResult claims checkpoint_step "
        f"{resumed.second[-1].checkpoint_step}, but the newest checkpoint on "
        f"disk is at step {refs[-1].step}"
    )
    if numbers == [0, 1]:
        first_step = int(resumed.first[0].checkpoint_step)
        assert int(resumed.second[0].checkpoint_step) == first_step, (
            f"the resumed run reports iteration 0 at checkpoint_step "
            f"{resumed.second[0].checkpoint_step}, but the iteration 0 that "
            f"actually ran is at {first_step}: it is reporting a replay, not a "
            f"recovery"
        )


def test_iteration_zero_was_not_replayed(resumed: _Resumed) -> None:
    """Its shard tree is untouched: same files, same bytes, same mtimes.

    The mtime half is the load-bearing one. Self-play is seeded from
    ``(run_seed, worker_id, incarnation, game_index)`` (design D11), so a
    campaign that blindly replayed iteration 0 would write *the same rows
    again* -- a digest-only comparison would call that success. What it would
    actually be is every preempted run paying twice for the same games.
    """
    assert set(resumed.after_tree) == set(resumed.before_tree), (
        "iteration 0's shard tree gained or lost files during the resumed run: "
        f"added={sorted(set(resumed.after_tree) - set(resumed.before_tree))}, "
        f"removed={sorted(set(resumed.before_tree) - set(resumed.after_tree))}"
    )
    changed = [
        name
        for name, stamp in resumed.before_tree.items()
        if resumed.after_tree[name] != stamp
    ]
    assert not changed, (
        f"{len(changed)} file(s) under {rig.iteration_dir(resumed.root, 0)} were "
        f"rewritten by the resumed run: {changed[:8]}"
    )


def test_iteration_zeros_checkpoint_is_the_same_bytes(resumed: _Resumed) -> None:
    """It survives the resume unmodified and unsuperseded.

    The failure this rules out is not corruption but arithmetic: if the second
    run numbered its checkpoint by the steps *it* ran rather than by a global
    counter, it would publish a file with iteration 0's name, and
    :func:`~komugi.train.checkpoint.save_checkpoint` would unlink the original
    as superseded -- the wave-10 F5 finding, arriving through the resume path
    instead of the loop path.
    """
    after = {
        ref.path.name: rig.digest_of(ref.path) for ref in rig.checkpoint_refs(resumed.root)
    }
    for name, digest in resumed.before_checkpoints.items():
        assert name in after, (
            f"checkpoint {name} was deleted by the resumed run; the run now "
            f"holds {sorted(after)}"
        )
        assert after[name] == digest, f"checkpoint {name} was rewritten in place"


def test_the_checkpoint_steps_stayed_monotone(resumed: _Resumed) -> None:
    """Across the crash, steps strictly increase and both files coexist.

    Counted over the trainer's series alone: KOM-022 §2 (W1) adds the EMA's
    series under the same root, and one file per iteration is a claim about
    each series separately, not about their union.
    """
    refs = rig.train_checkpoint_refs(resumed.root)
    steps = [ref.step for ref in refs]
    assert len(refs) == 2, (
        f"want one train checkpoint per iteration after the resume, found "
        f"{[ref.path.name for ref in refs]}"
    )
    assert steps[0] < steps[1], (
        f"checkpoint steps {steps} are not strictly increasing across the "
        f"resume: the second iteration reused the first's step number"
    )


def test_adam_moments_survived_the_crash(final_checkpoint: Checkpoint) -> None:
    """The final checkpoint carries a populated optimiser state.

    An empty ``state`` is what a checkpoint looks like when the optimiser was
    rebuilt and never stepped, or when the loop saved
    ``AdamW(...).state_dict()`` from the wrong object. It is invisible to every
    other assertion here -- the weights would still be trained, the steps would
    still be monotone -- and it costs the next iteration its second-moment
    estimates, which is a learning-rate bug that shows up as instability hours
    later.
    """
    optim = final_checkpoint.optim_state
    assert optim, f"{final_checkpoint.path.name} carries no optimiser state at all"
    per_param = optim.get("state")
    assert isinstance(per_param, dict) and per_param, (
        f"{final_checkpoint.path.name}'s optimiser state has an empty 'state' "
        f"map (keys present: {sorted(optim)})"
    )
    entries: list[Any] = [value for value in per_param.values() if isinstance(value, dict)]
    assert entries, "no per-parameter optimiser entries to inspect"
    with_moments = [
        entry for entry in entries if all(name in entry for name in _ADAM_MOMENTS)
    ]
    assert with_moments, (
        f"no per-parameter entry carries AdamW's {list(_ADAM_MOMENTS)}; found "
        f"keys {sorted({key for entry in entries for key in entry})}"
    )


def test_the_optimiser_was_the_campaigns_own(final_checkpoint: Checkpoint) -> None:
    """Its hyperparameters are the campaign's, not a default.

    Cheap, and it pins the one thing the assertion above cannot: that the
    optimiser state travelling in the checkpoint belongs to the run that wrote
    it. A saved ``state_dict`` from some other optimiser would satisfy "has
    moments" perfectly well.
    """
    groups = final_checkpoint.optim_state.get("param_groups")
    assert isinstance(groups, list) and groups, (
        f"{final_checkpoint.path.name} has no optimiser param_groups"
    )
    learning_rates = {float(group["lr"]) for group in groups if "lr" in group}
    assert learning_rates == {rig.LR}, (
        f"checkpointed optimiser runs at lr {sorted(learning_rates)}, but the "
        f"campaign was configured with lr={rig.LR}"
    )


def test_the_optimiser_stepped_through_both_iterations(
    final_checkpoint: Checkpoint,
) -> None:
    """Adam's per-parameter step counters count the whole campaign, not one run.

    The assertion :func:`test_adam_moments_survived_the_crash` cannot make: a
    cold AdamW rebuilt at iteration 1 populates ``exp_avg``/``exp_avg_sq``
    within its first step, so "moments are present" stays green with the
    loop's ``optimizer.load_state_dict`` deleted -- the wave-11 union review
    ran exactly that mutant and all eight tests here passed (finding F1). The
    step counter is not satisfiable cold: a restored optimiser resumes
    counting at ``STEPS_PER_ITERATION`` and ends at twice that; a cold one
    ends at once. The two outcomes share no value.
    """
    per_param = final_checkpoint.optim_state.get("state")
    assert isinstance(per_param, dict) and per_param, (
        f"{final_checkpoint.path.name} has no per-parameter optimiser state"
    )
    counters = {
        int(entry["step"])
        for entry in per_param.values()
        if isinstance(entry, dict) and "step" in entry
    }
    assert counters, "no per-parameter entry carries a step counter"
    assert counters == {2 * rig.STEPS_PER_ITERATION}, (
        f"per-parameter optimiser step counters are {sorted(counters)}; a "
        f"resume that restored iteration 0's moments ends at "
        f"{2 * rig.STEPS_PER_ITERATION}, a cold rebuild at "
        f"{rig.STEPS_PER_ITERATION}"
    )


def test_iteration_one_trained_from_iteration_zeros_checkpoint(
    resumed: _Resumed, final_checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """Bit-replay: checkpoint 0 plus eight documented steps *is* checkpoint 1.

    The training half of the resume seam, pinned through what
    :func:`run_campaign` wrote rather than through a reconstruction of its
    internals' return values: the wave-11 union review deleted the loop's
    ``net.load_state_dict`` -- a campaign that re-initialises its trainer
    every iteration and accumulates nothing -- and the entire suite stayed
    green (finding F2). This test is red under that mutant.

    :func:`~komugi.train.step.training_iteration` documents that two calls
    with the same ``rng_seed`` over the same source draw the same batches in
    the same order, and the loop documents its wiring (one window over
    iterations ``0..i``, seeded per iteration, ``StepConfig`` seeded with the
    run seed). So training is a pure function of (starting state, shards on
    disk, seed) -- and replaying it from iteration 0's checkpoint must land
    bit-for-bit on iteration 1's. If the loop's trainer had started anywhere
    other than checkpoint 0 -- fresh init, wrong file, weights without
    moments -- the replay and the artifact diverge.
    """
    # The trainer's series, not the EMA's: this replay starts from what the
    # trainer resumed from, and KOM-022 §2 (W1) puts the EMA's own checkpoints
    # under the same root.
    refs = rig.train_checkpoint_refs(resumed.root)
    assert len(refs) == 2, (
        f"want two train checkpoints, found {[r.path.name for r in refs]}"
    )
    ckpt0 = load_checkpoint_file(refs[0].path)
    assert ckpt0 is not None, f"{refs[0].path} does not verify"
    assert ckpt0.step == rig.STEPS_PER_ITERATION

    with rig.small_net_env():
        net = factory.new_net()
    net.load_state_dict(ckpt0.model_state)
    step = TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(batch_size=rig.BATCH_SIZE, lr=rig.LR, seed=rig.RUN_SEED),
    )
    step.optimizer.load_state_dict(ckpt0.optim_state)

    seed = camp.iteration_seed(rig.RUN_SEED, 1)
    source = camp.WindowSource(
        RecencyWindow(camp.window_refs(resumed.root, 1), WindowConfig(), seed=seed)
    )
    training_iteration(
        net,
        source,
        step,
        steps=rig.STEPS_PER_ITERATION,
        rng_seed=seed,
        checkpoint_dir=tmp_path / "replay-ckpt",
        journal_path=tmp_path / "replay-journal.jsonl",
        iteration=1,
        global_step_base=rig.STEPS_PER_ITERATION,
    )
    assert rig.state_dicts_match(dict(net.state_dict()), final_checkpoint.model_state), (
        "replaying iteration 1 from iteration 0's checkpoint does not reproduce "
        "the campaign's final weights: the loop's trainer did not start from "
        "the checkpoint it claims to resume from"
    )


def test_the_resumed_ema_is_the_uninterrupted_ema(
    resumed: _Resumed, tmp_path: Path
) -> None:
    """Acceptance #3, after a real campaign stop rather than at file level.

    The EMA is **recomputable, not atomic with the trainer**: whatever a crash
    cost is rebuilt at the next start, in ascending step order, before a game is
    played. The suite asserted that as arithmetic over a planted directory pair
    (``test_ema_unit``'s replay) and never over a campaign that actually
    stopped (review F7), which leaves the interesting half untested: the
    restart's own reconcile is the code that can double-apply an update, skip
    one, or publish from the wrong source step, and none of those is visible in
    the trainer's series.

    So: replay the whole EMA series from the campaign's own train checkpoints
    into a fresh directory, one publish-then-reconcile per iteration, exactly
    as an **uninterrupted** run would have produced it -- and the campaign's
    published series must be that series, tensor for tensor.
    """
    train_refs = rig.train_checkpoint_refs(resumed.root)
    published = tuple(reversed(list_checkpoints(camp.ema_dir(resumed.root))))
    assert len(train_refs) == 2, (
        f"want two train checkpoints across the resume, found "
        f"{[ref.path.name for ref in train_refs]}"
    )
    assert [ref.step for ref in published] == [ref.step for ref in train_refs], (
        f"the EMA series is at steps {[ref.step for ref in published]} and the "
        f"trainer's at {[ref.step for ref in train_refs]}: the EMA is published "
        f"at its source's step, so a resume that skipped or repeated one shows "
        f"up here"
    )

    with rig.small_net_env():
        keys = parameter_keys(factory.new_net())

    # The uninterrupted run, reconstructed: one train checkpoint at a time,
    # reconciling after each, which is what the loop does and what §2's resume
    # rule requires (an empty EMA directory beside two checkpoints is refused).
    replay_ckpt = tmp_path / "ckpt"
    replay_ema = tmp_path / "ema"
    replay_ckpt.mkdir(parents=True)
    for ref in train_refs:
        shutil.copy2(ref.path, replay_ckpt / ref.path.name)
        reconcile_ema(
            replay_ckpt,
            replay_ema,
            keys,
            decay=EMA_DECAY,
            steps_per_iteration=rig.STEPS_PER_ITERATION,
        )

    replayed = tuple(reversed(list_checkpoints(replay_ema)))
    assert [ref.step for ref in replayed] == [ref.step for ref in published]

    for interrupted_ref, uninterrupted_ref in zip(published, replayed):
        interrupted = load_checkpoint_file(interrupted_ref.path)
        uninterrupted = load_checkpoint_file(uninterrupted_ref.path)
        assert interrupted is not None and uninterrupted is not None
        assert rig.state_dicts_match(
            dict(interrupted.model_state), dict(uninterrupted.model_state)
        ), (
            f"the EMA published at step {interrupted_ref.step} by the "
            f"interrupted campaign is not the one an uninterrupted run over "
            f"the same checkpoints produces: "
            f"{canonical_digest(interrupted.model_state, keys).sha256[:16]} vs "
            f"{canonical_digest(uninterrupted.model_state, keys).sha256[:16]}"
        )
        assert interrupted.optim_state == {}, (
            "an EMA checkpoint carries optimiser state; nothing may resume "
            "from it"
        )

    # Non-vacuity: smoothing has begun, so the two EMA files are different nets
    # and "they match the replay" is not a statement about one constant series.
    digests: set[str] = set()
    for ref in published:
        checkpoint = load_checkpoint_file(ref.path)
        assert checkpoint is not None
        digests.add(canonical_digest(checkpoint.model_state, keys).sha256)
    assert len(digests) == len(published), (
        "the published EMA series is constant, so this comparison could not "
        "tell a correct replay from a frozen one"
    )


def test_a_mid_iteration_crash_does_not_double_the_iterations_data(
    tmp_path: Path,
) -> None:
    """Crash after self-play, before the checkpoint: the re-run replaces rows.

    The crash :func:`test_iteration_zero_was_not_replayed` deliberately does
    not stage. A machine that dies between sealing iteration 0's shards and
    publishing its checkpoint leaves a full shard tree and no checkpoint; the
    resume must re-run iteration 0 (its training never happened) -- but into a
    *clean* tree. The wave-11 union review staged exactly this and the
    crashed attempt's rows survived alongside the same-seed replay: 279 rows
    became 558 with 280 duplicate observations, all of them counted by the
    window and reported by ``IterationResult.rows`` (finding F3). Self-play
    is seeded, so the honest outcome is the same rows once, not twice.
    """
    root = tmp_path
    first = rig.walled_run_campaign(
        root, iterations=1, wall_s=rig.RESUME_WALL_S, label="crash-first"
    ).unwrap()
    assert [result.iteration for result in first] == [0]
    rows_before = len(
        rig.concat_source(rig.shard_dirs_under(rig.iteration_dir(root, 0)))
    )
    assert rows_before > 0, "the premise run produced no rows"

    for directory in {ref.path.parent for ref in rig.checkpoint_refs(root)}:
        shutil.rmtree(directory)

    second = rig.walled_run_campaign(
        root, iterations=1, wall_s=rig.RESUME_WALL_S, label="crash-second"
    ).unwrap()
    assert [result.iteration for result in second] == [0], (
        f"a campaign with shards but no checkpoint reported iterations "
        f"{[result.iteration for result in second]}; iteration 0 never "
        f"trained, so it must run again"
    )
    rows_after = len(
        rig.concat_source(rig.shard_dirs_under(rig.iteration_dir(root, 0)))
    )
    assert rows_after == rows_before, (
        f"iteration 0 held {rows_before} rows before the staged crash and "
        f"{rows_after} after the resume: the crashed attempt's partial shards "
        f"were kept alongside the replay instead of being cleared"
    )
    assert int(second[0].rows) == rows_after, (
        f"IterationResult.rows says {second[0].rows} but iteration 0's tree "
        f"holds {rows_after} readable rows"
    )
